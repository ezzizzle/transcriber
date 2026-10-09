"""HTTP server: Whisper-compatible API, job API for the web UI, and the UI itself.

Standard library only (http.server); see multipart.py for upload parsing.
"""

import json
import re
import secrets
import socket
import sys
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer
from urllib.parse import parse_qs, quote, urlsplit

from . import formats, multipart
from .config import Config, parse_bool
from .pipeline import Engine, Job, Options

STATIC_DIR = Path(__file__).parent / "static"
SESSION_COOKIE = "transcriber_session"
# Set by an authenticating reverse proxy. Trusted as-is: see README, "Privacy".
USER_HEADER = "X-Forwarded-User"

# Model ids advertised on /v1/models. Clients that can't send extra form
# fields can switch options on by picking a model name instead: any name
# containing "diariz", "clean" or "summar" turns that option on.
MODEL_IDS = [
    "parakeet", "parakeet-diarize", "parakeet-clean", "parakeet-diarize-clean",
    "parakeet-diarize-summary", "parakeet-diarize-clean-summary", "whisper-1",
]  # fmt: skip

DOWNLOADS = {
    "txt": ("text/plain; charset=utf-8", lambda r, name: formats.to_text_with_summary(r) + "\n"),
    "md": ("text/markdown; charset=utf-8", lambda r, name: formats.to_markdown(r, name)),
    "srt": ("application/x-subrip; charset=utf-8", lambda r, name: formats.to_srt(r)),
    "vtt": ("text/vtt; charset=utf-8", lambda r, name: formats.to_vtt(r)),
    "json": ("application/json", lambda r, name: formats.to_full_json(r)),
}


class HTTPError(Exception):
    def __init__(self, status: int, message: str, kind: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.message, self.kind = status, message, kind


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "transcriber"

    @property
    def engine(self) -> Engine:
        return self.server.engine

    @property
    def config(self) -> Config:
        return self.server.config

    # ------------------------------------------------------------- plumbing

    def log_message(self, fmt, *args):
        if self.command == "GET" and self.path.startswith("/api/jobs"):
            return  # the UI polls these every second
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    def _send(self, status: int, body: bytes | str, content_type: str, headers: dict | None = None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        if getattr(self, "_new_session", None):
            cookie = f"{SESSION_COOKIE}={self._new_session}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Strict"
            self.send_header("Set-Cookie", cookie)
            self._new_session = None
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload):
        self._send(status, json.dumps(payload, ensure_ascii=False), "application/json")

    def _error(self, err: HTTPError):
        # OpenAI's error envelope, so SDK clients surface the message.
        self._json(err.status, {"error": {"message": err.message, "type": err.kind, "param": None, "code": None}})

    def _session(self) -> str:
        """Who is asking; jobs are only ever listed or served to their uploader.

        Behind an authenticating reverse proxy, that is the user it names in
        X-Forwarded-User. Otherwise there are no accounts, and each browser
        gets a random cookie on first contact.
        """
        user = self._forwarded_user()
        if user:
            return f"user:{user}"  # the colon keeps these apart from cookie tokens
        if getattr(self, "_new_session", None):
            return self._new_session
        try:
            morsel = SimpleCookie(self.headers.get("Cookie", "")).get(SESSION_COOKIE)
        except Exception:  # noqa: BLE001 - malformed Cookie header
            morsel = None
        if morsel and re.fullmatch(r"[\w-]{32,64}", morsel.value):
            return morsel.value
        self._new_session = secrets.token_urlsafe(32)
        return self._new_session

    def _forwarded_user(self) -> str | None:
        return (self.headers.get(USER_HEADER) or "").strip() or None

    def _dispatch(self, routes):
        self._new_session = None  # handler instances are reused across keep-alive requests
        path = urlsplit(self.path).path.rstrip("/") or "/"
        try:
            for pattern, handler in routes:
                m = re.fullmatch(pattern, path)
                if m:
                    return handler(*m.groups())
            raise HTTPError(404, f"No route for {self.command} {path}")
        except HTTPError as err:
            self._error(err)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def do_GET(self):
        self._dispatch([
            (r"/", self.get_index),
            (r"/health", self.get_health),
            (r"/v1/models", self.get_models),
            (r"/api/jobs", self.get_jobs),
            (r"/api/jobs/(\w+)", self.get_job),
            (r"/api/jobs/(\w+)/download", self.get_download),
        ])  # fmt: skip

    do_HEAD = do_GET

    def do_POST(self):
        self._dispatch([
            (r"/v1/audio/transcriptions", self.post_transcription),
            (r"/api/jobs", self.post_job),
        ])  # fmt: skip

    def do_DELETE(self):
        self._dispatch([(r"/api/jobs/(\w+)", self.delete_job)])

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _read_upload(self):
        """Parse the multipart body; returns (fields, Upload for the `file` part)."""
        length = self.headers.get("Content-Length")
        if length is None or not length.isdigit():
            self.close_connection = True
            raise HTTPError(411, "Content-Length is required")
        # Fields and multipart framing ride along with the file, hence the slack.
        if int(length) > self.config.max_upload_bytes + 1024 * 1024:
            self.close_connection = True
            raise HTTPError(413, f"Upload exceeds the {self.config.max_upload_mb} MB limit")
        try:
            fields, files = multipart.parse(
                self.rfile,
                self.headers.get("Content-Type", ""),
                int(length),
                self.config.work_dir,
                self.config.max_upload_bytes,
            )
        except multipart.MultipartError as exc:
            self.close_connection = True
            raise HTTPError(400, f"Bad upload: {exc}") from exc
        upload = files.pop("file", None)
        for extra in files.values():
            extra.path.unlink(missing_ok=True)
        if upload is None or upload.size == 0:
            if upload:
                upload.path.unlink(missing_ok=True)
            raise HTTPError(400, "Missing `file` form field")
        return fields, upload

    def _job_or_404(self, job_id: str) -> Job:
        job = self.engine.get(job_id, self._session())
        if job is None:  # also the answer for someone else's job
            raise HTTPError(404, "No such job")
        return job

    # --------------------------------------------------------------- routes

    def get_index(self):
        self._session()
        self._send(200, (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")

    def get_health(self):
        engine = self.engine
        status = "error" if engine.load_error else "ok" if engine.ready.is_set() else "loading"
        self._json(200 if status != "error" else 503, {
            "status": status,
            "error": engine.load_error,
            "asr_model": self.config.asr_model,
            "cleanup_model": self.config.cleanup_model,
            **engine.stats(),
        })  # fmt: skip

    def get_models(self):
        self._json(200, {
            "object": "list",
            "data": [{"id": m, "object": "model", "created": 0, "owned_by": "local"} for m in MODEL_IDS],
        })  # fmt: skip

    def post_transcription(self):
        """OpenAI-compatible POST /v1/audio/transcriptions."""
        fields, upload = self._read_upload()

        def field(name, default=None):
            return fields.get(name, [default])[-1]

        model = (field("model") or "").lower()
        response_format = (field("response_format") or "json").lower()
        if response_format not in {"json", "text", "srt", "vtt", "verbose_json", "diarized_json"}:
            upload.path.unlink(missing_ok=True)
            raise HTTPError(400, f"Unsupported response_format: {response_format}")
        options = Options(
            diarize=parse_bool(field("diarize"), self.config.default_diarize or "diariz" in model)
            or response_format == "diarized_json",
            cleanup=parse_bool(field("cleanup"), self.config.default_cleanup or "clean" in model),
            summary=parse_bool(field("summary"), self.config.default_summary or "summar" in model),
            language=field("language") or None,
        )
        granularities = fields.get("timestamp_granularities[]", []) + fields.get("timestamp_granularities", [])
        granularities = {g.strip() for value in granularities for g in value.split(",")}

        job = self.engine.submit(upload.filename, upload.path, options)
        job.done.wait()
        if job.status == "error":
            if job.user_error:
                raise HTTPError(400, job.error)
            raise HTTPError(500, job.error, "server_error")

        result = job.result
        if response_format == "json":
            summary = {"summary": result["summary"]} if result["summary"] else {}
            self._json(200, {"text": formats.to_text(result), **summary})
        elif response_format == "verbose_json":
            payload = formats.to_verbose_json(
                result,
                words="word" in granularities,
                segments="segment" in granularities or "word" not in granularities,
            )
            self._json(200, payload)
        elif response_format == "diarized_json":
            self._json(200, formats.to_diarized_json(result))
        elif response_format == "text":
            self._send(200, formats.to_text_with_summary(result) + "\n", "text/plain; charset=utf-8")
        else:
            content_type, render = DOWNLOADS[response_format]
            self._send(200, render(result, job.filename), content_type)

    def post_job(self):
        """Web UI upload: returns immediately, the UI polls GET /api/jobs/{id}."""
        fields, upload = self._read_upload()
        options = Options(
            diarize=parse_bool(fields.get("diarize", [None])[-1]),
            cleanup=parse_bool(fields.get("cleanup", [None])[-1]),
            summary=parse_bool(fields.get("summary", [None])[-1]),
        )
        job = self.engine.submit(upload.filename, upload.path, options, owner=self._session())
        self._json(202, self.engine.summary(job))

    def get_jobs(self):
        jobs = [self.engine.summary(job) for job in self.engine.list_jobs(self._session())]
        self._json(200, {"user": self._forwarded_user(), "jobs": jobs})

    def get_job(self, job_id):
        job = self._job_or_404(job_id)
        payload = self.engine.summary(job)
        if job.result:
            payload["result"] = {key: job.result[key] for key in ("diarized", "cleaned", "speakers", "turns", "summary")}
        self._json(200, payload)

    def get_download(self, job_id):
        job = self._job_or_404(job_id)
        if job.result is None:
            raise HTTPError(409, "Job has not finished")
        fmt = parse_qs(urlsplit(self.path).query).get("format", ["txt"])[-1]
        if fmt not in DOWNLOADS:
            raise HTTPError(400, f"Unknown format: {fmt}. Choose from {', '.join(DOWNLOADS)}")
        content_type, render = DOWNLOADS[fmt]
        stem = Path(job.filename).stem or "transcript"
        disposition = f"attachment; filename*=UTF-8''{quote(f'{stem}.{fmt}')}"
        self._send(200, render(job.result, stem), content_type, {"Content-Disposition": disposition})

    def delete_job(self, job_id):
        self._job_or_404(job_id)
        if not self.engine.delete(job_id, self._session()):
            raise HTTPError(409, "Job is still running")
        self._json(200, {"deleted": job_id})


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, config: Config, engine: Engine):
        self.config, self.engine = config, engine
        if ":" in config.host:  # e.g. "::" for every interface, IPv4 and IPv6
            self.address_family = socket.AF_INET6
        super().__init__((config.host, config.port), Handler)

    def server_bind(self):
        # Deliberately not HTTPServer.server_bind: it resolves the bind address
        # with getfqdn(), which can stall startup for a long time where DNS is
        # restricted, and nothing here uses the name.
        if self.address_family == socket.AF_INET6:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]
