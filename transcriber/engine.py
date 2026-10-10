"""The job queue and the worker processes that drain it. Lives in the server process.

Up to `max_parallel` jobs run at once, each in a worker process of its own
(see worker.py); the rest wait in line. One thread per worker, a "slot",
hands it jobs and relays what it says back:

    HTTP request --> Engine.submit --> queue --> slot thread <--pipe--> worker process
                                                     |
                                                     +--> SharedLLM, when the worker asks for the language model
"""

import atexit
import multiprocessing
import os
import shutil
import threading
import time
from collections import deque
from pathlib import Path

from .config import Config
from .jobs import Job, Options
from .llm import LanguageModelError, SharedLLM, llm_main
from .worker import worker_main


class QueueFull(Exception):
    """Too many uploads are already waiting."""


class Worker:
    """A worker process and the pipe to it."""

    def __init__(self, mp_context, main, config: Config, slot: int, preload: bool = False):
        self.conn, child_conn = mp_context.Pipe()
        self.proc = mp_context.Process(
            target=main, args=(child_conn, config, preload), name=f"transcriber-worker-{slot}", daemon=True
        )
        self.proc.start()
        child_conn.close()

    def stop(self):
        self.proc.kill()


class Engine:
    def __init__(self, config: Config, worker_main=worker_main, llm_main=llm_main):
        self.config = config
        self._worker_main = worker_main
        self._mp = multiprocessing.get_context("spawn")
        self._llm = SharedLLM(self._mp, config.llm_model, config.stall_seconds, config.idle_unload_seconds, llm_main)

        self._cond = threading.Condition()  # guards the four attributes below
        self._jobs: dict[str, Job] = {}  # web UI jobs, oldest first; API jobs are never kept
        self._pending: deque[Job] = deque()
        self._idle_slots: set[int] = set()
        self._running = 0

        self.ready = threading.Event()  # set once the first worker has loaded the speech model
        self.load_error: str | None = None
        self.files_dir = _private_dir(config.work_dir)  # uploads and decoded audio, while their job is alive

    def start(self):
        for slot in range(self.config.max_parallel):
            threading.Thread(target=self._slot, args=(slot,), name=f"slot-{slot}", daemon=True).start()

    # ------------------------------------------------------------ for the server

    def queue_full(self) -> bool:
        with self._cond:
            return len(self._pending) >= self.config.max_queue

    def submit(self, filename: str, path: Path, options: Options, owner: str | None = None) -> Job:
        """Queue a job. Jobs without an owner (API calls) are not kept for the web UI."""
        job = Job(filename=filename, path=path, options=options, owner=owner)
        with self._cond:
            if len(self._pending) >= self.config.max_queue:
                raise QueueFull
            if owner is not None:
                self._jobs[job.id] = job
            self._pending.append(job)
            self._cond.notify_all()
        return job

    def job_status(self, job: Job) -> dict:
        with self._cond:
            position = self._pending.index(job) + 1 if job in self._pending else None
        return job.as_dict(position)

    def stats(self) -> dict:
        with self._cond:
            return {"max_parallel": self.config.max_parallel, "running": self._running, "queued": len(self._pending)}

    # Jobs are only ever handed to the person who uploaded them, so these all take the owner.

    def get(self, job_id: str, owner: str) -> Job | None:
        with self._cond:
            job = self._jobs.get(job_id)
            return job if job and job.owner == owner else None

    def list_jobs(self, owner: str) -> list[Job]:
        """The owner's jobs, newest first."""
        with self._cond:
            return [job for job in reversed(self._jobs.values()) if job.owner == owner]

    def delete(self, job_id: str, owner: str) -> bool:
        """Forget a finished job. Queued and running jobs can't be cancelled."""
        with self._cond:
            job = self._jobs.get(job_id)
            if job is None or job.owner != owner or not job.done.is_set():
                return False
            del self._jobs[job_id]
            return True

    # ------------------------------------------------------------------- slots

    def _slot(self, slot: int):
        """One thread per worker process: start it, feed it jobs, replace it if it breaks."""
        worker = None
        if slot == 0:
            # The first worker stays for the life of the server, and loads
            # (downloading if need be) the speech model before the first upload.
            worker = Worker(self._mp, self._worker_main, self.config, slot, preload=True)
            try:
                _, self.load_error = worker.conn.recv()
            except EOFError:
                self.load_error = "The speech model worker exited during startup."
            self.ready.set()

        while True:
            job = self._next_job(slot, may_give_up=worker is not None and slot > 0)
            if job is None:
                # Nothing to do for a long time. An extra worker holds a few GB of models; let it go.
                worker.stop()
                worker = None
                continue
            if worker is None or not worker.proc.is_alive():
                worker = Worker(self._mp, self._worker_main, self.config, slot)
            if not self._run(job, worker):
                worker.stop()
                worker = None
            self._finish(job)

    def _next_job(self, slot: int, may_give_up: bool) -> Job | None:
        """Wait for a job. With `may_give_up`, returns None after `idle_unload_seconds` without one."""
        give_up_at = time.monotonic() + self.config.idle_unload_seconds
        with self._cond:
            self._idle_slots.add(slot)
            try:
                # The lowest idle slot takes the job, so a worker with its
                # models already loaded is preferred over starting a new one.
                while not (self._pending and min(self._idle_slots) == slot):
                    remaining = give_up_at - time.monotonic() if may_give_up else None
                    if remaining is not None and remaining <= 0:
                        return None
                    self._cond.wait(remaining)
                self._running += 1
                return self._pending.popleft()
            finally:
                self._idle_slots.discard(slot)
                self._cond.notify_all()  # the next idle slot may now be the lowest

    def _run(self, job: Job, worker: Worker) -> bool:
        """Run a job on a worker, relaying its messages. False if the worker must be replaced."""
        job.status, job.started, job.stage = "processing", time.time(), "Starting"
        wav = self.files_dir / f"{job.id}.wav"
        try:
            worker.conn.send((str(job.path), str(wav), job.options))
            while True:
                if not worker.conn.poll(self.config.stall_seconds):
                    job.status, job.error = "error", "The transcription worker stopped responding and was restarted."
                    return False
                kind, *payload = worker.conn.recv()  # the messages are listed in worker.worker_main
                if kind == "stage":
                    job.stage, job.progress = payload
                elif kind == "llm":
                    worker.conn.send(("llm", *self._ask_llm(job, payload[0])))
                elif kind == "done":
                    job.result = payload[0]
                    job.status, job.stage, job.progress = "done", "Done", None
                    return True
                else:
                    job.status, job.error, job.user_error = "error", *payload
                    return True
        except (EOFError, OSError):
            job.status, job.error = "error", "The transcription worker crashed while processing this file."
            return False
        finally:
            wav.unlink(missing_ok=True)  # here rather than in the worker, so a crash can't leave it behind

    def _ask_llm(self, job: Job, requests: list) -> tuple[list[str] | None, str | None]:
        """Returns (answers, None), or (None, why not)."""
        shown = job.stage, job.progress

        def loading():  # may include the download, so say what the wait is for
            job.stage, job.progress = "Loading language model", None

        try:
            answers, load_seconds = self._llm.ask(requests, loading)
        except LanguageModelError as exc:
            return None, str(exc)
        finally:
            job.stage, job.progress = shown
        job.llm_load_seconds += load_seconds
        return answers, None

    def _finish(self, job: Job):
        job.finished = time.time()
        job.path.unlink(missing_ok=True)
        with self._cond:
            self._running -= 1
            # Keep each person's newest `max_jobs` finished transcripts.
            finished = [j for j in self._jobs.values() if j.owner == job.owner and (j.done.is_set() or j is job)]
            for old in finished[: max(0, len(finished) - self.config.max_jobs)]:
                del self._jobs[old.id]
        job.done.set()


def _private_dir(work_dir: Path) -> Path:
    """A directory for this server's files, removed again when the server exits.

    It is named after the process, so that a server that was killed without the
    chance to tidy up has its leftover recordings removed by the next one to start.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    for leftover in work_dir.glob("run-*"):
        pid = leftover.name.removeprefix("run-")
        if pid.isdigit() and not _is_running(int(pid)):
            shutil.rmtree(leftover, ignore_errors=True)
    mine = work_dir / f"run-{os.getpid()}"
    mine.mkdir(exist_ok=True)
    atexit.register(shutil.rmtree, mine, ignore_errors=True)
    return mine


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # signal 0 checks that the process exists without touching it
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists, and belongs to someone else
    return True
