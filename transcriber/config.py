"""Runtime settings, read from TRANSCRIBER_* environment variables."""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"TRANSCRIBER_{name}", default)


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def _default_parallel() -> int:
    """Jobs that fit in RAM: 1 on a 16 GB Mac, 2 on 24 GB, 3 on 32 GB, capped at 4.

    Sets aside 8 GB for the shared language model and macOS, then allows one
    worker per 7 GB (a worker peaks at 6.5 to 8 GB transcribing and diarizing).
    """
    try:
        ram_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (ValueError, OSError):
        return 1
    return max(1, min(4, int((ram_gb - 8) // 7)))


@dataclass
class Config:
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8000")))

    asr_model: str = field(
        default_factory=lambda: _env("ASR_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
    )
    cleanup_model: str = field(
        default_factory=lambda: _env("CLEANUP_MODEL", "mlx-community/Qwen3.5-9B-MLX-4bit")
    )

    # Defaults for the Whisper-compatible endpoint when the request doesn't say.
    default_diarize: bool = field(default_factory=lambda: parse_bool(_env("DEFAULT_DIARIZE", "false")))
    default_cleanup: bool = field(default_factory=lambda: parse_bool(_env("DEFAULT_CLEANUP", "false")))
    default_summary: bool = field(default_factory=lambda: parse_bool(_env("DEFAULT_SUMMARY", "false")))

    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "4096")))
    # Long audio is transcribed in overlapping windows of this many seconds.
    chunk_seconds: float = field(default_factory=lambda: float(_env("CHUNK_SECONDS", "120")))
    # Jobs processed at once. Each parallel job has its own worker process and
    # its own speech models (about 6 GB at peak); the language model is shared.
    max_parallel: int = field(default_factory=lambda: max(1, int(_env("MAX_PARALLEL", str(_default_parallel())))))
    # Extra workers (beyond the first) exit after this long with nothing to do.
    idle_unload_seconds: float = field(default_factory=lambda: float(_env("IDLE_UNLOAD_SECONDS", "600")))
    # Finished jobs kept in memory for the web UI.
    max_jobs: int = field(default_factory=lambda: int(_env("MAX_JOBS", "50")))
    work_dir: Path = field(
        default_factory=lambda: Path(_env("WORK_DIR", str(Path(tempfile.gettempdir()) / "transcriber")))
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
