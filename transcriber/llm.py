"""The language model, which lives in a process of its own.

It is the largest model by far, so there is exactly one copy and every worker
shares it:

    worker  --ask-->  server process (SharedLLM)  --ask-->  llm_main (LocalModel)

A request is (system prompt, user message, max_tokens); what comes back is an Answer.
SharedLLM is the server's handle on the process. LocalModel and llm_main run
inside it, and are the only code that loads the model.
"""

import threading
import time
from typing import NamedTuple

# Loading can include a multi-gigabyte download the first time.
_LOAD_TIMEOUT = 3600.0


class Answer(NamedTuple):
    text: str
    cut_off: bool = False  # the model was still writing when it reached max_tokens, so the end is missing


class LanguageModelError(Exception):
    """The language model could not answer. Clean-up and summaries are optional, so jobs survive this."""


# ------------------------------------------------------- in the model's process


class LocalModel:
    """The model itself. Must be created and used on a single thread (MLX streams are thread-local)."""

    def __init__(self, model_id: str):
        from mlx_lm import load

        self.model, self.tokenizer = load(model_id)

    def ask_many(self, requests: list[tuple[str, str, int]]) -> list[Answer]:
        """Answer each (system, user, max_tokens) request; several are generated as one batch."""
        import mlx.core as mx
        from mlx_lm import batch_generate, stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompts = []
        for system, user, _ in requests:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            try:
                # Reasoning models would otherwise spend the token budget thinking out loud.
                prompts.append(self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False))
            except TypeError:
                prompts.append(self.tokenizer.apply_chat_template(messages, add_generation_prompt=True))
        limits = [max_tokens for _, _, max_tokens in requests]
        sampler = make_sampler(temp=0.0)
        try:
            if len(prompts) == 1:
                text, finished_because = "", None
                for piece in stream_generate(self.model, self.tokenizer, prompts[0], max_tokens=limits[0], sampler=sampler):
                    text += piece.text
                    finished_because = piece.finish_reason  # "stop" at a natural end, "length" at the limit
                return [Answer(text, cut_off=finished_because == "length")]
            batch = batch_generate(
                self.model, self.tokenizer, prompts, max_tokens=limits, sampler=sampler, return_token_ids=True
            )
            # A batch doesn't say why each answer ended, but one that used its whole allowance was cut off.
            return [
                Answer(text, cut_off=len(tokens) >= limit)
                for text, tokens, limit in zip(batch.texts, batch.token_ids, limits)
            ]
        finally:
            if sum(len(prompt) for prompt in prompts) > 2000:  # long inputs leave gigabytes of buffers cached
                mx.clear_cache()


def llm_main(conn, model_id: str):
    """Entry point of the model's process.

    Sends ("ready", error_or_None) once loaded. Then, for each list of requests
    it receives, replies ("ok", [Answer, ...]) or ("error", message).
    """
    try:
        try:
            print(f"Loading language model {model_id} ...", flush=True)
            model = LocalModel(model_id)
            print("Language model ready.", flush=True)
        except Exception as exc:  # noqa: BLE001 - reported to the job that needed it
            conn.send(("ready", f"Could not load the language model {model_id}: {exc}"))
            return
        conn.send(("ready", None))
        while True:
            requests = conn.recv()
            try:
                conn.send(("ok", model.ask_many(requests)))
            except Exception as exc:  # noqa: BLE001 - one bad request must not kill the model
                conn.send(("error", f"{type(exc).__name__}: {exc}"))
    except (EOFError, KeyboardInterrupt):
        pass  # server is shutting down


# ----------------------------------------------------- in the server's process


class SharedLLM:
    """Starts the model's process on first use, and stops it after a while with nothing to do."""

    def __init__(self, mp_context, model_id: str, answer_timeout: float, idle_seconds: float, main=llm_main):
        self._mp, self._model_id, self._main = mp_context, model_id, main
        self._answer_timeout, self._idle_seconds = answer_timeout, idle_seconds
        self._lock = threading.Lock()  # one batch of requests at a time; jobs take turns
        self._proc = self._conn = None
        self._last_used = time.monotonic()
        threading.Thread(target=self._stop_when_idle, name="llm-idle", daemon=True).start()

    def ask(self, requests: list[tuple[str, str, int]], on_loading=lambda: None) -> tuple[list[Answer], float]:
        """Returns (answers, seconds spent loading the model first, usually 0).

        `on_loading` is called if the model has to be loaded, which takes a while.
        """
        with self._lock:
            load_seconds = 0.0
            try:
                if self._proc is None or not self._proc.is_alive():
                    on_loading()
                    load_seconds = self._start()
                self._conn.send(requests)
                status, value = self._receive(self._answer_timeout)
            except (EOFError, OSError) as exc:
                self._stop()
                raise LanguageModelError("The language model process crashed.") from exc
            finally:
                self._last_used = time.monotonic()
        if status == "error":
            raise LanguageModelError(value)
        return value, load_seconds

    def _start(self) -> float:
        started = time.monotonic()
        self._conn, child_conn = self._mp.Pipe()
        self._proc = self._mp.Process(target=self._main, args=(child_conn, self._model_id), name="transcriber-llm", daemon=True)
        self._proc.start()
        child_conn.close()
        _, error = self._receive(_LOAD_TIMEOUT)
        if error:
            self._stop()
            raise LanguageModelError(error)
        return time.monotonic() - started

    def _receive(self, timeout: float):
        if not self._conn.poll(timeout):
            self._stop()  # stuck: a fresh process will be started for the next request
            raise LanguageModelError(f"The language model did not answer within {timeout / 60:.0f} minutes.")
        return self._conn.recv()

    def _stop(self):
        if self._proc is not None:
            self._proc.kill()
        self._proc = self._conn = None

    def _stop_when_idle(self):
        while True:
            time.sleep(min(30.0, self._idle_seconds))
            with self._lock:  # so never in the middle of a request
                if self._proc is not None and time.monotonic() - self._last_used > self._idle_seconds:
                    self._stop()
