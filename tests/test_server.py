"""The HTTP routes, run against a real server with the fake processes in fakes.py."""

import http.client
import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from tests import fakes
from transcriber.config import Config
from transcriber.engine import Engine
from transcriber import server as server_module
from transcriber.server import Handler, Server


class Browser:
    """A client that keeps its cookie between requests, as a browser would."""

    def __init__(self, port: int, headers: dict | None = None):
        self.port, self.headers, self.cookie = port, headers or {}, None

    def request(self, method: str, path: str, body: bytes = b"", content_type: str | None = None):
        headers = dict(self.headers)
        if self.cookie:
            headers["Cookie"] = self.cookie
        if content_type:
            headers["Content-Type"] = content_type
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        if response.getheader("Set-Cookie"):
            self.cookie = response.getheader("Set-Cookie").split(";")[0]
        if response.getheader("Content-Type") == "application/json":
            data = json.loads(data)
        return response.status, data

    def get(self, path: str):
        return self.request("GET", path)

    def upload(self, path: str, recording: str, **fields):
        boundary = "test-boundary"
        parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n' for k, v in fields.items()]
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="talk.mp3"\r\n\r\n{recording}\r\n')
        body = ("".join(parts) + f"--{boundary}--\r\n").encode()
        return self.request("POST", path, body, f"multipart/form-data; boundary={boundary}")

    def upload_and_wait(self, recording: str, **fields) -> dict:
        status, job = self.upload("/api/jobs", recording, **fields)
        assert status == 202, (status, job)
        return self.wait(job)

    def wait(self, job: dict) -> dict:
        for _ in range(400):
            status, job = self.get(f"/api/jobs/{job['id']}")
            if job["status"] in ("done", "error"):
                return job
            time.sleep(0.05)
        raise AssertionError("job did not finish")


class ServerTest(unittest.TestCase):
    def serve(self, worker=fakes.worker, **settings) -> int:
        """Start a server on a free port and return the port."""
        work_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.object(Handler, "log_message", lambda *args: None))  # keep the test output quiet
        config = Config(host="127.0.0.1", port=0, work_dir=work_dir, max_parallel=1, **settings)
        engine = self.engine = Engine(config, worker_main=worker, llm_main=fakes.llm)
        server = Server(config, engine)
        engine.start()
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.assertTrue(engine.ready.wait(20))
        return server.server_address[1]

    def test_upload_view_download_and_remove(self):
        browser = Browser(self.serve())
        job = browser.upload_and_wait("hello there", summary="true")
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["result"]["turns"][0]["text"], "hello there")
        self.assertEqual(job["result"]["summary"]["executive_summary"], "Summary of: hello there")
        self.assertEqual([j["id"] for j in browser.get("/api/jobs")[1]["jobs"]], [job["id"]])

        status, text = browser.get(f"/api/jobs/{job['id']}/download?format=txt")
        self.assertEqual(status, 200)
        self.assertTrue(text.decode().startswith("hello there\n"))
        self.assertEqual(browser.get(f"/api/jobs/{job['id']}/download?format=docx")[0], 400)

        self.assertEqual(browser.request("DELETE", f"/api/jobs/{job['id']}")[0], 200)
        self.assertEqual(browser.get("/api/jobs")[1]["jobs"], [])

    def test_one_browser_cannot_see_anothers_transcripts(self):
        port = self.serve()
        alice, bob = Browser(port), Browser(port)
        job = alice.upload_and_wait("alice's private meeting")
        bob.get("/")  # bob gets a cookie of his own
        self.assertNotEqual(alice.cookie, bob.cookie)

        self.assertEqual(bob.get("/api/jobs")[1]["jobs"], [])
        self.assertEqual(bob.get(f"/api/jobs/{job['id']}")[0], 404)
        self.assertEqual(bob.get(f"/api/jobs/{job['id']}/download?format=txt")[0], 404)
        self.assertEqual(bob.request("DELETE", f"/api/jobs/{job['id']}")[0], 404)
        self.assertEqual(alice.get(f"/api/jobs/{job['id']}")[0], 200)  # and bob's attempt removed nothing

    def test_the_user_header_is_ignored_unless_trusted(self):
        port = self.serve()
        job = Browser(port, {"X-Forwarded-User": "alice"}).upload_and_wait("alice's private meeting")
        impostor = Browser(port, {"X-Forwarded-User": "alice"})
        self.assertEqual(impostor.get("/api/jobs")[1], {"user": None, "jobs": []})
        self.assertEqual(impostor.get(f"/api/jobs/{job['id']}")[0], 404)

    def test_a_trusted_user_header_identifies_people(self):
        port = self.serve(trust_user_header=True)
        job = Browser(port, {"X-Forwarded-User": "alice"}).upload_and_wait("alice's private meeting")
        alice_elsewhere, bob = Browser(port, {"X-Forwarded-User": "alice"}), Browser(port, {"X-Forwarded-User": "bob"})
        self.assertEqual(alice_elsewhere.get(f"/api/jobs/{job['id']}")[0], 200)  # same person, another browser
        self.assertEqual(alice_elsewhere.get("/api/jobs")[1]["user"], "alice")
        self.assertEqual(bob.get(f"/api/jobs/{job['id']}")[0], 404)
        self.assertEqual(bob.get("/api/jobs")[1]["jobs"], [])

    def test_whisper_compatible_api(self):
        api = Browser(self.serve())
        self.assertEqual(api.upload("/v1/audio/transcriptions", "hello there"), (200, {"text": "hello there"}))
        status, verbose = api.upload("/v1/audio/transcriptions", "hello there", response_format="verbose_json")
        self.assertEqual((status, verbose["segments"][0]["text"], verbose["duration"]), (200, "hello there", 1.0))
        self.assertEqual(api.get("/api/jobs")[1]["jobs"], [])  # API calls are not kept

        status, error = api.upload("/v1/audio/transcriptions", "bad")
        self.assertEqual((status, error["error"]["type"]), (400, "invalid_request_error"))
        status, error = api.upload("/v1/audio/transcriptions", "crash")
        self.assertEqual((status, error["error"]["type"]), (500, "server_error"))
        self.assertEqual(api.upload("/v1/audio/transcriptions", "x", response_format="docx")[0], 400)

    def test_requests_that_make_no_sense(self):
        browser = Browser(self.serve())
        self.assertEqual(browser.get("/nowhere")[0], 404)
        self.assertEqual(browser.upload("/nowhere", "hello")[0], 404)
        self.assertEqual(browser.request("POST", "/api/jobs", b"not a form", "text/plain")[0], 400)
        boundary = "multipart/form-data; boundary=b"
        self.assertEqual(browser.request("POST", "/api/jobs", b"--b--\r\n", boundary)[0], 400)  # no file

    def test_a_full_queue_turns_uploads_away(self):
        browser = Browser(self.serve(max_queue=1))
        _, first = browser.upload("/api/jobs", "sleep 1 first")
        time.sleep(0.3)  # the first is now running, so the queue is empty again
        status, second = browser.upload("/api/jobs", "second")
        self.assertEqual(status, 202)
        self.assertEqual(browser.upload("/api/jobs", "third")[0], 503)
        self.assertEqual([browser.wait(job)["status"] for job in (first, second)], ["done", "done"])

    def test_an_upload_that_stalls_is_dropped_and_frees_its_place(self):
        with mock.patch.object(server_module, "UPLOAD_STALL_SECONDS", 0.5):
            port = self.serve(max_queue=1)
            stalled = socket.create_connection(("127.0.0.1", port), timeout=10)
            stalled.sendall(
                b"POST /api/jobs HTTP/1.1\r\nHost: x\r\nContent-Type: multipart/form-data; boundary=b\r\n"
                b"Content-Length: 100000\r\n\r\n"
                b'--b\r\nContent-Disposition: form-data; name="file"; filename="talk.mp3"\r\n\r\nthe first few bytes'
            )
            time.sleep(0.2)
            self.assertEqual(Browser(port).upload("/api/jobs", "hello")[0], 503)  # the only place is taken
            self.assertEqual(stalled.recv(1000), b"")  # the server gives up and hangs up
            stalled.close()
        self.assertEqual(list(self.engine.files_dir.iterdir()), [])  # nothing of the partial upload is kept
        self.assertEqual(Browser(port).upload_and_wait("hello")["status"], "done")  # and its place is free again

    def test_health(self):
        status, health = Browser(self.serve()).get("/health")
        self.assertEqual((status, health["status"], health["queued"]), (200, "ok", 0))

    def test_health_recovers_once_a_job_has_worked(self):
        browser = Browser(self.serve(worker=fakes.worker_that_cannot_load_its_model))
        status, health = browser.get("/health")
        self.assertEqual((status, health["status"]), (503, "error"))
        self.assertIn("the network is down", health["error"])
        browser.upload_and_wait("hello there")
        self.assertEqual(browser.get("/health")[1]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
