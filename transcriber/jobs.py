"""What a job is: the options it was submitted with and its state as it runs."""

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# Keys of a job's per-stage timings, in the order the stages run. "speakers" is
# the diarizer's own running time, which mostly overlaps with "transcribe".
STAGE_KEYS = ("queue", "convert", "speech_load", "transcribe", "speakers", "llm_load", "cleanup", "summary")


@dataclass
class Options:
    diarize: bool = False
    cleanup: bool = False
    summary: bool = False
    language: str | None = None


@dataclass
class Job:
    filename: str
    path: Path  # the uploaded file
    options: Options
    owner: str | None = None  # who uploaded it (see server.Handler._identity); None for API calls
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "queued"  # queued | processing | done | error
    stage: str = "Queued"  # what to show while it is processing
    progress: float | None = None  # 0..1 within the current stage, when known
    error: str | None = None
    user_error: bool = False  # the file was the problem, not the server
    result: dict | None = None  # see formats.py for its shape
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    done: threading.Event = field(default_factory=threading.Event)
    llm_load_seconds: float = 0.0  # time this job spent waiting for the language model to load

    def timings(self) -> dict | None:
        """Seconds per stage for a finished job, in the order the stages ran."""
        if not self.result:
            return None
        spent = dict(self.result.get("timings") or {})
        if self.llm_load_seconds:
            # The worker can't see the load; it happened inside its first language-model stage.
            first = "cleanup" if "cleanup" in spent else "summary"
            spent[first] = round(max(0.0, spent.get(first, 0.0) - self.llm_load_seconds), 2)
            spent["llm_load"] = round(self.llm_load_seconds, 2)
        spent["queue"] = round(self.started - self.created, 2)
        return {key: spent[key] for key in STAGE_KEYS if key in spent}

    def as_dict(self, queue_position: int | None = None) -> dict:
        """The job's state as the web UI sees it (the transcript itself is not included)."""
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "queue_position": queue_position,  # 1 = next to start; None once running
            "error": self.error,
            "diarize": self.options.diarize,
            "cleanup": self.options.cleanup,
            "summarize": self.options.summary,
            "cleaned": bool(self.result and self.result["cleaned"]),  # clean-up was asked for and worked
            "created": self.created,
            "elapsed": round((self.finished or time.time()) - self.started, 1) if self.started else None,
            "duration": self.result["duration"] if self.result else None,
            "timings": self.timings(),
            "warnings": self.result["warnings"] if self.result else [],
        }
