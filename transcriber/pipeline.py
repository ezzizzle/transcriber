"""The transcription pipeline and the job queue that runs it.

    any media file --ffmpeg--> 16 kHz mono WAV --+--> parakeet-mlx (text + timestamps)
                                                 +--> senko (who spoke when)   [optional]
                                     merge by time overlap --> local LLM cleanup [optional]

Jobs run in worker processes, up to `max_parallel` at once; the rest wait in a
FIFO queue. Each worker owns its own speech and diarization models: MLX state
is per-thread and not safe to share, and a process boundary also means a crash
on one file can't take the server or anyone else's job down with it.

The language model is the exception. It is the largest model by far, so one
extra process holds a single copy and the workers take turns with it, relayed
through the server process (worker -> Engine -> SharedLLM -> llm_main).
"""

import multiprocessing
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import cleanup
from .config import Config

SAMPLE_RATE = 16000


class PipelineError(Exception):
    """A failure the user can do something about (bad file, missing ffmpeg...)."""


@dataclass
class Options:
    diarize: bool = False
    cleanup: bool = False
    summary: bool = False
    language: str | None = None


@dataclass
class Job:
    filename: str
    path: Path
    options: Options
    owner: str | None = None  # browser session that uploaded it; None for API calls
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "queued"  # queued | processing | done | error
    stage: str = "Queued"
    progress: float | None = None  # 0..1 within the current stage, when known
    error: str | None = None
    user_error: bool = False
    result: dict | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    done: threading.Event = field(default_factory=threading.Event)

    def summary(self, queue_position: int | None = None) -> dict:
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
            "created": self.created,
            "elapsed": round((self.finished or time.time()) - self.started, 1) if self.started else None,
            "duration": self.result["duration"] if self.result else None,
        }


# ---------------------------------------------------------------- pure helpers


def convert_to_wav(src: Path, dst: Path):
    """Decode the first audio stream of anything ffmpeg can read to 16 kHz mono PCM."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise PipelineError("ffmpeg is not installed. Install it with: brew install ffmpeg")
    cmd = [
        ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(src),
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst),
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0 or not dst.exists():
        detail = (proc.stderr.strip().splitlines() or ["unknown error"])[-1]
        raise PipelineError(f"ffmpeg could not read audio from this file: {detail}")


def wav_duration(path: Path) -> float:
    return max(0, path.stat().st_size - 44) / (SAMPLE_RATE * 2)


def tokens_to_words(tokens) -> list[dict]:
    """Join parakeet's sub-word tokens (word starts carry a leading space)."""
    words: list[dict] = []
    for tok in tokens:
        if words and not tok.text.startswith(" "):
            words[-1]["word"] += tok.text
            words[-1]["end"] = round(tok.end, 3)
        elif tok.text.strip():
            words.append({"word": tok.text.strip(), "start": round(tok.start, 3), "end": round(tok.end, 3)})
    return words


def assign_speakers(segments: list[dict], diarization: list[dict]) -> None:
    """Label each segment with the diarization speaker it overlaps most.

    Segments that overlap nobody (diarizers drop very short utterances) take
    the nearest speaker in time. Speakers are renamed "Speaker 1", "Speaker 2",
    ... in order of first appearance.
    """
    if not diarization:
        for seg in segments:
            seg["speaker"] = "Speaker 1"
        return
    diarization = sorted(diarization, key=lambda d: d["start"])
    names: dict[str, str] = {}
    first = 0
    for seg in segments:
        while first < len(diarization) - 1 and diarization[first]["end"] <= seg["start"]:
            first += 1
        best, best_overlap = None, 0.0
        for d in diarization[first:]:
            if d["start"] >= seg["end"]:
                break
            overlap = min(seg["end"], d["end"]) - max(seg["start"], d["start"])
            if overlap > best_overlap:
                best, best_overlap = d, overlap
        if best is None:
            mid = (seg["start"] + seg["end"]) / 2
            nearby = diarization[max(0, first - 1): first + 2]
            best = min(nearby, key=lambda d: max(d["start"] - mid, mid - d["end"], 0))
        seg["speaker"] = names.setdefault(best["speaker"], f"Speaker {len(names) + 1}")


def build_turns(segments: list[dict]) -> list[dict]:
    """Group consecutive segments by the same speaker."""
    turns: list[dict] = []
    for seg in segments:
        if turns and turns[-1]["speaker"] == seg.get("speaker"):
            turns[-1]["end"] = seg["end"]
            turns[-1]["sentences"].append(seg["text"])
        else:
            turns.append({
                "speaker": seg.get("speaker"),
                "start": seg["start"],
                "end": seg["end"],
                "sentences": [seg["text"]],
            })  # fmt: skip
    for turn in turns:
        turn["text"] = " ".join(turn["sentences"])
    return turns


# ------------------------------------------------------------ worker process


class Pipeline:
    """Owns the speech models and runs one file at a time. Lives in a worker process.

    `ask(system, user, max_tokens) -> str` is how it reaches the shared language model.
    """

    def __init__(self, config: Config, ask):
        self.config = config
        self._editor = cleanup.Editor(ask)
        self._diar_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diarize")
        self._asr = None
        self._diarizer = None

    def load_asr(self):
        if self._asr is None:
            from parakeet_mlx import from_pretrained

            self._asr = from_pretrained(self.config.asr_model)

    def run(self, job_id: str, src: Path, options: Options, report) -> dict:
        """Transcribe `src`. `report(stage, progress=None)` receives status updates."""
        wav = self.config.work_dir / f"{job_id}.wav"
        diar_future = None
        try:
            report("Converting audio")
            convert_to_wav(src, wav)
            duration = wav_duration(wav)
            if duration < 0.1:
                raise PipelineError("This file contains no audio.")

            # Diarization runs on CoreML, so it can overlap with transcription.
            if options.diarize:
                diar_future = self._diar_pool.submit(self._diarize, wav)

            segments, words = self._transcribe(wav, report)
            self._free_gpu_cache()

            if diar_future is not None:
                report("Identifying speakers")
                diarization = diar_future.result()
                assign_speakers(segments, diarization)
                by_start = iter(words)
                for seg in segments:  # words inherit their segment's speaker
                    for _ in range(seg.pop("_n_words")):
                        next(by_start)["speaker"] = seg["speaker"]
            for seg in segments:
                seg.pop("_n_words", None)

            turns = build_turns(segments)
            cleaned = False
            if options.cleanup and turns:
                turns = self._cleanup(turns, report)
                cleaned = True
            for turn in turns:
                del turn["sentences"]

            summary = None
            if options.summary and turns:
                summary = self._summarize(turns, options.diarize, report)

            return {
                "duration": round(duration, 3),
                "language": options.language or "unknown",
                "model": self.config.asr_model,
                "diarized": options.diarize,
                "cleaned": cleaned,
                "cleanup_model": self.config.cleanup_model if cleaned else None,
                # {"key_points": [...], "action_items": [...]}; None if not requested or unusable.
                "summary": summary,
                "speakers": sorted({t["speaker"] for t in turns if t["speaker"]}, key=lambda s: int(s.split()[-1])),
                "turns": turns,
                "segments": segments,
                "words": words,
            }
        finally:
            self._free_gpu_cache()
            if diar_future is not None:  # don't pull the file out from under senko
                diar_future.cancel() or diar_future.exception()
            wav.unlink(missing_ok=True)

    @staticmethod
    def _free_gpu_cache():
        """Hand MLX's buffer cache back between stages; it matters on 16 GB Macs."""
        import mlx.core as mx

        mx.clear_cache()

    def _transcribe(self, wav: Path, report) -> tuple[list[dict], list[dict]]:
        from parakeet_mlx import DecodingConfig, SentenceConfig

        if self._asr is None:
            report("Loading speech model")
            self.load_asr()
        report("Transcribing", 0.0)

        def on_chunk(position, total):
            # Called before each window is decoded, so report the previous one.
            step = (self.config.chunk_seconds - 15.0) * SAMPLE_RATE
            report("Transcribing", max(0.0, min(1.0, (position - step) / total)))

        try:
            result = self._asr.transcribe(
                str(wav),
                chunk_duration=self.config.chunk_seconds,
                overlap_duration=15.0,
                chunk_callback=on_chunk,
                # Sentences normally end at punctuation; these stop run-ons from
                # becoming one enormous subtitle when the speaker never pauses.
                decoding_config=DecodingConfig(sentence=SentenceConfig(silence_gap=3.0, max_duration=30.0)),
            )
        except ValueError as exc:  # audio shorter than one frame
            raise PipelineError(str(exc)) from exc

        segments, words = [], []
        for sentence in result.sentences:
            text = sentence.text.strip()
            if not text:
                continue
            sentence_words = tokens_to_words(sentence.tokens)
            words.extend(sentence_words)
            segments.append({
                "id": len(segments),
                "start": round(sentence.start, 3),
                "end": round(sentence.end, 3),
                "text": text,
                "speaker": None,
                "confidence": round(sentence.confidence, 4),
                "_n_words": len(sentence_words),
            })  # fmt: skip
        return segments, words

    def _diarize(self, wav: Path) -> list[dict]:
        if self._diarizer is None:
            import senko

            self._diarizer = senko.Diarizer(device="auto", warmup=True, quiet=True)
        result = self._diarizer.diarize(str(wav), generate_colors=False)
        return list(result["merged_segments"]) if result else []

    def _summarize(self, turns: list[dict], diarized: bool, report) -> dict | None:
        report("Summarizing", 0.0)
        blocks = [f"{t['speaker']}: {t['text']}" if diarized else t["text"] for t in turns]
        return self._editor.summarize(blocks, diarized, lambda fraction: report("Summarizing", fraction))

    def _cleanup(self, turns: list[dict], report) -> list[dict]:
        report("Cleaning up text", 0.0)
        total = sum(len(t["text"].split()) for t in turns) or 1
        seen = 0
        kept = []
        for turn in turns:
            parts = []
            for chunk in cleanup.chunk_sentences(turn["sentences"]):
                parts.append(self._editor.clean(chunk))
                seen += len(chunk.split())
                report("Cleaning up text", seen / total)
            text = "\n\n".join(p for p in parts if p)
            if not text:  # the whole turn was filler ("Um.")
                continue
            if kept and kept[-1]["speaker"] == turn["speaker"]:
                kept[-1]["text"] += "\n\n" + text
                kept[-1]["end"] = turn["end"]
            else:
                kept.append({**turn, "text": text})
        return kept


def worker_main(conn, config: Config, preload: bool):
    """Worker process entry point.

    Receives (job_id, path, options) over `conn`; replies with any number of
    ("stage", name, progress) messages, then ("done", result) or
    ("error", message, is_user_error). To use the language model it sends
    ("llm", system, user, max_tokens) and waits for ("llm", text_or_None, error).
    """

    def ask(system: str, user: str, max_tokens: int) -> str:
        conn.send(("llm", system, user, max_tokens))
        _, text, error = conn.recv()
        if error:
            raise RuntimeError(error)
        return text

    pipeline = Pipeline(config, ask)
    try:
        if preload:
            error = None
            try:
                print(f"Loading speech model {config.asr_model} ...", flush=True)
                pipeline.load_asr()
                print("Speech model ready.", flush=True)
            except Exception as exc:  # noqa: BLE001 - reported through /health
                traceback.print_exc()
                error = f"Could not load speech model: {exc}"
            conn.send(("ready", error))

        while True:
            job_id, path, options = conn.recv()
            last = None

            def report(stage, progress=None):
                nonlocal last
                update = (stage, None if progress is None else round(progress, 2))
                if update != last:
                    last = update
                    conn.send(("stage", *update))

            try:
                conn.send(("done", pipeline.run(job_id, Path(path), options, report)))
            except PipelineError as exc:
                conn.send(("error", str(exc), True))
            except Exception as exc:  # noqa: BLE001 - one bad file must not kill the worker
                traceback.print_exc()
                conn.send(("error", f"{type(exc).__name__}: {exc}", False))
    except (EOFError, KeyboardInterrupt):
        pass  # server is shutting down


# ---------------------------------------------------------------------- engine


class SharedLLM:
    """The single language-model process, started on first use and then kept."""

    def __init__(self, mp_context, model_id: str):
        self._mp, self._model_id = mp_context, model_id
        self._lock = threading.Lock()  # one request at a time; jobs interleave chunk by chunk
        self._proc = self._conn = None

    def ask(self, job: Job, system: str, user: str, max_tokens: int) -> str:
        with self._lock:
            try:
                if self._proc is None or not self._proc.is_alive():
                    self._start(job)
                self._conn.send((system, user, max_tokens))
                status, value = self._conn.recv()
            except (EOFError, OSError) as exc:
                self._stop()
                raise RuntimeError("The language model process crashed.") from exc
        if status == "error":
            raise RuntimeError(value)
        return value

    def _start(self, job: Job):
        shown = job.stage, job.progress
        job.stage, job.progress = "Loading language model", None  # may include the download
        parent_conn, child_conn = self._mp.Pipe()
        self._proc = self._mp.Process(
            target=cleanup.llm_main, args=(child_conn, self._model_id), name="transcriber-llm", daemon=True
        )
        self._proc.start()
        child_conn.close()
        self._conn = parent_conn
        error = self._conn.recv()[1]
        job.stage, job.progress = shown
        if error:
            self._stop()
            raise RuntimeError(error)

    def _stop(self):
        if self._proc is not None:
            self._proc.kill()
        self._proc = self._conn = None


class Engine:
    """Job queue plus a pool of worker processes. Lives in the server process."""

    def __init__(self, config: Config):
        self.config = config
        self.jobs: OrderedDict[str, Job] = OrderedDict()  # jobs kept for the web UI
        self._cond = threading.Condition()  # guards everything below, and self.jobs
        self._pending: deque[Job] = deque()
        self._idle: set[int] = set()
        self._running = 0
        self._mp = multiprocessing.get_context("spawn")
        self._llm = SharedLLM(self._mp, config.cleanup_model)
        self.ready = threading.Event()
        self.load_error: str | None = None
        config.work_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        for slot in range(self.config.max_parallel):
            threading.Thread(target=self._slot, args=(slot,), name=f"slot-{slot}", daemon=True).start()

    # -- job bookkeeping

    def submit(self, filename: str, path: Path, options: Options, owner: str | None = None) -> Job:
        """Queue a job. Jobs without an owner (API calls) are not kept for the web UI."""
        job = Job(filename=filename, path=path, options=options, owner=owner)
        with self._cond:
            if owner is not None:
                self.jobs[job.id] = job
                finished = [j for j in self.jobs.values() if j.done.is_set()]
                for old in finished[: max(0, len(self.jobs) - self.config.max_jobs)]:
                    del self.jobs[old.id]
            self._pending.append(job)
            self._cond.notify_all()
        return job

    def summary(self, job: Job) -> dict:
        with self._cond:
            position = self._pending.index(job) + 1 if job in self._pending else None
        return job.summary(position)

    def stats(self) -> dict:
        with self._cond:
            return {"max_parallel": self.config.max_parallel, "running": self._running, "queued": len(self._pending)}

    def get(self, job_id: str, owner: str) -> Job | None:
        with self._cond:
            job = self.jobs.get(job_id)
            return job if job and job.owner == owner else None

    def list_jobs(self, owner: str) -> list[Job]:
        with self._cond:
            return [job for job in reversed(self.jobs.values()) if job.owner == owner]

    def delete(self, job_id: str, owner: str) -> bool:
        """Forget a finished job. Queued/running jobs can't be cancelled."""
        with self._cond:
            job = self.jobs.get(job_id)
            if job is None or job.owner != owner or not job.done.is_set():
                return False
            del self.jobs[job_id]
            return True

    # -- worker slots

    def _spawn(self, slot: int, preload: bool = False):
        parent_conn, child_conn = self._mp.Pipe()
        proc = self._mp.Process(
            target=worker_main, args=(child_conn, self.config, preload), name=f"transcriber-worker-{slot}", daemon=True
        )
        proc.start()
        child_conn.close()
        return proc, parent_conn

    def _slot(self, slot: int):
        """One thread per worker process: feeds it jobs and relays its progress."""
        proc = conn = None
        if slot == 0:
            # The first worker stays warm for the life of the server and loads
            # (downloading if need be) the speech model before the first upload.
            proc, conn = self._spawn(slot, preload=True)
            try:
                self.load_error = conn.recv()[1]
            except EOFError:
                self.load_error = "The speech model worker exited during startup."
            self.ready.set()

        while True:
            with self._cond:
                self._idle.add(slot)
                idle_since = time.monotonic()
                # Lowest idle slot takes the next job, so warm workers are preferred.
                while not (self._pending and min(self._idle) == slot):
                    timeout = None
                    if proc is not None and slot > 0:
                        # Extra workers hold a few GB of models each; let go when idle.
                        timeout = self.config.idle_unload_seconds - (time.monotonic() - idle_since)
                        if timeout <= 0:
                            proc.terminate()
                            proc = conn = None
                            continue
                    self._cond.wait(timeout)
                job = self._pending.popleft()
                self._idle.discard(slot)
                self._running += 1
                self._cond.notify_all()

            job.status, job.started, job.stage = "processing", time.time(), "Starting"
            try:
                if proc is None or not proc.is_alive():
                    proc, conn = self._spawn(slot)
                conn.send((job.id, str(job.path), job.options))
                while True:
                    kind, *payload = conn.recv()
                    if kind == "stage":
                        job.stage, job.progress = payload
                    elif kind == "llm":
                        try:
                            conn.send(("llm", self._llm.ask(job, *payload), None))
                        except RuntimeError as exc:
                            conn.send(("llm", None, str(exc)))
                    elif kind == "done":
                        job.result = payload[0]
                        job.status, job.stage, job.progress = "done", "Done", None
                        break
                    else:
                        job.status, job.error, job.user_error = "error", payload[0], payload[1]
                        break
            except (EOFError, OSError):
                job.status, job.error = "error", "The transcription worker crashed while processing this file."
                if proc is not None:
                    proc.kill()
                proc = conn = None
            finally:
                job.finished = time.time()
                job.path.unlink(missing_ok=True)
                with self._cond:
                    self._running -= 1
                job.done.set()
