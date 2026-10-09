"""A small streaming multipart/form-data parser.

The stdlib lost `cgi` in 3.13, and media uploads are too big to buffer in
memory, so file parts are streamed straight to disk.
"""

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

_CHUNK = 256 * 1024
_MAX_FIELD_BYTES = 1024 * 1024
_MAX_HEADER_BYTES = 16 * 1024


class MultipartError(ValueError):
    pass


@dataclass
class Upload:
    filename: str
    path: Path
    size: int


class _Reader:
    """Reads at most `length` bytes from `rfile`, with delimiter search."""

    def __init__(self, rfile, length: int):
        self.rfile = rfile
        self.remaining = length
        self.buf = b""

    def _fill(self) -> bool:
        if self.remaining <= 0:
            return False
        data = self.rfile.read(min(_CHUNK, self.remaining))
        if not data:
            self.remaining = 0
            return False
        self.remaining -= len(data)
        self.buf += data
        return True

    def read_until(self, delim: bytes, sink=None, limit: int | None = None) -> bytes:
        """Consume through `delim`. Data before it goes to `sink`, or is returned."""
        collected = [] if sink is None else None
        total = 0

        def emit(data: bytes):
            nonlocal total
            total += len(data)
            if limit is not None and total > limit:
                raise MultipartError("part too large")
            if sink is None:
                collected.append(data)
            else:
                sink.write(data)

        while True:
            idx = self.buf.find(delim)
            if idx >= 0:
                emit(self.buf[:idx])
                self.buf = self.buf[idx + len(delim):]
                return b"".join(collected) if collected is not None else b""
            # Keep a tail that could be the start of a delimiter split across reads.
            keep = len(delim) - 1
            if len(self.buf) > keep:
                emit(self.buf[:-keep])
                self.buf = self.buf[-keep:]
            if not self._fill():
                raise MultipartError("unexpected end of request body")

    def read_exact(self, n: int) -> bytes:
        while len(self.buf) < n:
            if not self._fill():
                raise MultipartError("unexpected end of request body")
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def drain(self):
        self.buf = b""
        while self._fill():
            self.buf = b""


def _disposition_param(header: str, name: str) -> str | None:
    m = re.search(rf'[;\s]{name}="((?:[^"\\]|\\.)*)"', header) or re.search(
        rf"[;\s]{name}=([^;\s]+)", header
    )
    return re.sub(r"\\(.)", r"\1", m.group(1)) if m else None


def parse(rfile, content_type: str, content_length: int, upload_dir: Path, max_file_bytes: int):
    """Parse a multipart body.

    Returns (fields, files): fields maps name -> list of string values, files
    maps name -> Upload. The caller owns (and must delete) the uploaded files.
    """
    m = re.search(r'boundary="?([^";]+)"?', content_type or "", re.I)
    if not m or "multipart/form-data" not in content_type.lower():
        raise MultipartError("expected multipart/form-data with a boundary")
    delim = b"--" + m.group(1).strip().encode("latin-1")

    reader = _Reader(rfile, content_length)
    fields: dict[str, list[str]] = {}
    files: dict[str, Upload] = {}
    try:
        reader.read_until(delim, limit=_MAX_FIELD_BYTES)  # preamble
        while True:
            tail = reader.read_exact(2)
            if tail == b"--":
                break
            if tail != b"\r\n":
                raise MultipartError("malformed boundary")
            raw_headers = reader.read_until(b"\r\n\r\n", limit=_MAX_HEADER_BYTES)
            disposition = ""
            for line in raw_headers.decode("utf-8", "replace").split("\r\n"):
                if line.lower().startswith("content-disposition:"):
                    disposition = line
            name = _disposition_param(disposition, "name")
            if name is None:
                raise MultipartError("part without a name")
            filename = _disposition_param(disposition, "filename")

            if filename is None:
                value = reader.read_until(b"\r\n" + delim, limit=_MAX_FIELD_BYTES)
                fields.setdefault(name, []).append(value.decode("utf-8", "replace"))
                continue

            suffix = Path(filename).suffix[:16]
            if not re.fullmatch(r"\.[A-Za-z0-9]+", suffix):
                suffix = ""
            tmp = tempfile.NamedTemporaryFile(dir=upload_dir, suffix=suffix, delete=False)
            path = Path(tmp.name)
            if name in files:  # only the last file with a given name is kept
                files.pop(name).path.unlink(missing_ok=True)
            files[name] = Upload(Path(filename).name or "upload", path, 0)
            with tmp:
                reader.read_until(b"\r\n" + delim, sink=tmp, limit=max_file_bytes)
                files[name].size = tmp.tell()
        reader.drain()
    except BaseException:
        for upload in files.values():
            upload.path.unlink(missing_ok=True)
        raise
    return fields, files
