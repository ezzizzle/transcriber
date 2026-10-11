"""The streaming upload parser."""

import io
import tempfile
import unittest
from pathlib import Path

from transcriber import multipart


def body(boundary: str, *parts: tuple[str, bytes]) -> bytes:
    out = b""
    for headers, payload in parts:
        out += f"--{boundary}\r\n{headers}\r\n\r\n".encode() + payload + b"\r\n"
    return out + f"--{boundary}--\r\n".encode()


class MultipartTest(unittest.TestCase):
    def parse(self, raw: bytes, max_file=10**9):
        self.dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        return multipart.parse(io.BytesIO(raw), "multipart/form-data; boundary=XyZ", len(raw), self.dir, max_file)

    def test_fields_and_file(self):
        # Larger than one read, and containing bytes that look like a boundary.
        payload = (b"\r\n--Xy" + bytes(range(256))) * 3000
        raw = body(
            "XyZ",
            ('Content-Disposition: form-data; name="diarize"', b"true"),
            ('Content-Disposition: form-data; name="timestamp_granularities[]"', b"word"),
            ('Content-Disposition: form-data; name="timestamp_granularities[]"', b"segment"),
            ('Content-Disposition: form-data; name="file"; filename="my talk.WebM"\r\nContent-Type: video/webm', payload),
        )
        fields, files = self.parse(raw)
        self.assertEqual(fields["diarize"], ["true"])
        self.assertEqual(fields["timestamp_granularities[]"], ["word", "segment"])
        upload = files["file"]
        self.assertEqual((upload.filename, upload.size, upload.path.suffix), ("my talk.WebM", len(payload), ".WebM"))
        self.assertEqual(upload.path.read_bytes(), payload)

    def test_oversized_file_is_rejected_and_removed(self):
        raw = body("XyZ", ('Content-Disposition: form-data; name="file"; filename="a.mp3"', b"x" * 5000))
        with self.assertRaises(multipart.MultipartError):
            self.parse(raw, max_file=1000)
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_truncated_body(self):
        raw = body("XyZ", ('Content-Disposition: form-data; name="file"; filename="a.mp3"', b"abc"))[:-12]
        with self.assertRaises(multipart.MultipartError):
            self.parse(raw)

    def test_text_fields_share_one_budget(self):
        # Each field is under the limit on its own; together they are not.
        field = ('Content-Disposition: form-data; name="prompt"', b"x" * 400_000)
        with self.assertRaises(multipart.MultipartError):
            self.parse(body("XyZ", field, field, field))
        self.parse(body("XyZ", field, field))


if __name__ == "__main__":
    unittest.main()
