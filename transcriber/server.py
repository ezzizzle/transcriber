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

from . import compare, formats, multipart
from .config import Config, parse_bool
from .engine import Engine, NoRoom
from .jobs import Job, Options

STATIC_DIR = Path(__file__).parent / "static"
SESSION_COOKIE = "transcriber_session"
# What senko runs on macOS; it bundles these rather than taking a model name.
DIARIZATION_MODEL = "senko (pyannote segmentation 3.0 + CAM++)"
# Set by an authenticating reverse proxy. Only believed when the
# TRANSCRIBER_TRUST_USER_HEADER setting is on: see README, "Privacy and users".
USER_HEADER = "X-Forwarded-User"

# A client that sends nothing for this long in the middle of an upload is dropped.
UPLOAD_STALL_SECONDS = 60

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
        if self.path.startswith("/v1/"):  # the Whisper-compatible API may be called from other sites' pages
            self.send_header("Access-Control-Allow-Origin", "*")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        if self._new_cookie:
            cookie = f"{SESSION_COOKIE}={self._new_cookie}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Strict"
            self.send_header("Set-Cookie", cookie)
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

    def _identity(self) -> str:
        """Who is asking. Jobs are only ever listed or served to whoever uploaded them.

        Behind an authenticating reverse proxy (and with the setting that says
        to believe it), that is the user named in X-Forwarded-User. Otherwise
        there are no accounts, and each browser gets a random cookie on first
        contact.
        """
        user = self._forwarded_user()
        if user:
            return f"user:{user}"  # the colon keeps these apart from cookie values
        if self._new_cookie:
            return self._new_cookie
        try:
            morsel = SimpleCookie(self.headers.get("Cookie", "")).get(SESSION_COOKIE)
        except Exception:  # noqa: BLE001 - malformed Cookie header
            morsel = None
        if morsel and re.fullmatch(r"[\w-]{32,64}", morsel.value):
            return morsel.value
        self._new_cookie = secrets.token_urlsafe(32)  # sent with the response; see _send
        return self._new_cookie

    def _forwarded_user(self) -> str | None:
        if not self.config.trust_user_header:
            return None
        return (self.headers.get(USER_HEADER) or "").strip() or None

    def _dispatch(self, routes):
        self._new_cookie = None  # one handler object serves every request on a kept-alive connection
        path = urlsplit(self.path).path.rstrip("/") or "/"
        try:
            for pattern, handler in routes:
                m = re.fullmatch(pattern, path)
                if m:
                    return handler(*m.groups())
            if self.command == "POST":
                self.close_connection = True  # its body was never read, so the connection can't be reused
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
            (r"/api/jobs/(\w+)/comparison", self.get_comparison),
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
        """Browsers ask this before a cross-site call to the Whisper-compatible API."""
        self.send_response(204)
        if self.path.startswith("/v1/"):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
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
        size = int(length)
        try:
            self.engine.reserve(size)  # before reading, so being turned away costs no disk space
        except NoRoom as exc:
            self.close_connection = True
            raise HTTPError(503, str(exc), "server_error") from None
        try:
            self.connection.settimeout(UPLOAD_STALL_SECONDS)
            fields, files = multipart.parse(
                self.rfile, self.headers.get("Content-Type", ""), size, self.engine.files_dir, self.config.max_upload_bytes
            )
        except multipart.MultipartError as exc:
            self.close_connection = True
            raise HTTPError(400, f"Bad upload: {exc}") from exc
        finally:
            # Also reached when the client stalls or hangs up, which http.server then deals with.
            self.connection.settimeout(None)
            self.engine.release(size)
        upload = files.pop("file", None)
        for extra in files.values():
            extra.path.unlink(missing_ok=True)
        if upload is None or upload.size == 0:
            if upload:
                upload.path.unlink(missing_ok=True)
            raise HTTPError(400, "Missing `file` form field")
        return fields, upload

    def _submit(self, upload, options: Options, owner: str | None = None) -> Job:
        try:
            return self.engine.submit(upload.filename, upload.path, options, owner)
        except NoRoom as exc:
            upload.path.unlink(missing_ok=True)
            raise HTTPError(503, str(exc), "server_error") from None

    def _job_or_404(self, job_id: str) -> Job:
        job = self.engine.get(job_id, self._identity())
        if job is None:  # also the answer for someone else's job
            raise HTTPError(404, "No such job")
        return job

    # --------------------------------------------------------------- routes

    def get_index(self):
        self._identity()  # so a first-time visitor gets their cookie with the page
        self._send(200, (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")

    def get_health(self):
        engine = self.engine
        status = "error" if engine.load_error else "ok" if engine.ready.is_set() else "loading"
        self._json(200 if status != "error" else 503, {
            "status": status,
            "error": engine.load_error,
            "asr_model": self.config.asr_model,
            "diarization_model": DIARIZATION_MODEL,
            "llm_model": self.config.llm_model,
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

        job = self._submit(upload, options)
        job.done.wait()
        if job.status == "error":
            if job.user_error:
                raise HTTPError(400, job.error)
            raise HTTPError(500, job.error, "server_error")

        result = job.result
        if response_format == "json":
            self._json(200, {"text": formats.to_text(result), **formats.extras(result)})
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
        job = self._submit(upload, options, owner=self._identity())
        self._json(202, self.engine.job_status(job))

    def get_jobs(self):
        jobs = [self.engine.job_status(job) for job in self.engine.list_jobs(self._identity())]
        self._json(200, {"user": self._forwarded_user(), "jobs": jobs})

    def get_job(self, job_id):
        job = self._job_or_404(job_id)
        payload = self.engine.job_status(job)
        if job.result:  # what the page shows; timestamps and words are only in the downloads
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

    def get_comparison(self, job_id):
        """What clean-up changed in each turn, for the page's "Show changes"."""
        job = self._job_or_404(job_id)
        if job.result is None:
            raise HTTPError(409, "Job has not finished")
        if not job.result["cleaned"]:
            raise HTTPError(400, "This transcript was not cleaned up, so there is nothing to compare.")
        turns = [
            {"speaker": t["speaker"], "start": t["start"], "changes": compare.changes(t["raw_text"], t["text"])}
            for t in job.result["turns"]
        ]
        self._json(200, {"turns": turns})

    def delete_job(self, job_id):
        self._job_or_404(job_id)
        if not self.engine.delete(job_id, self._identity()):
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
