"""A worker process: takes one recording at a time through the whole pipeline.

    any media file --ffmpeg--> 16 kHz mono WAV --+--> parakeet-mlx (text + timestamps)
                                                 +--> senko (who spoke when)      [optional]
                                  merge by time overlap --> clean-up, summary      [optional]

Each worker owns its own speech and speaker models: MLX state belongs to one
thread and can't be shared, and a process of its own means a crash on one
file can't take the server or anyone else's job down with it. The language
model is the exception; see llm.py.
"""

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from . import audio
from .cleanup import clean_turns
from .config import Config
from .jobs import Options
from .llm import LanguageModelError
from .summary import summarize
from .transcript import Sentence, assign_speakers, build_turns, to_result, tokens_to_words

# Long audio is transcribed in windows that overlap by this many seconds.
_OVERLAP_SECONDS = 15.0


class Pipeline:
    """Owns the speech and speaker models, and runs one recording at a time.

    `ask_many(requests) -> list[str]` is how it reaches the shared language model.
    """

    def __init__(self, config: Config, ask_many):
        self.config = config
        self._ask_many = ask_many
        self._speech_model = None
        self._diarizer = None
        # Speaker identification runs on CoreML rather than MLX, so it can have
        # a thread of its own and overlap with transcription.
        self._diarize_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diarize")

    def load_speech_model(self):
        if self._speech_model is None:
            from parakeet_mlx import from_pretrained

            self._speech_model = from_pretrained(self.config.asr_model)

    def run(self, src: Path, wav: Path, options: Options, report) -> dict:
        """Transcribe `src`, using `wav` as scratch space. `report(stage, progress=None)` receives status updates.

        Returns the result described in formats.py. Clean-up and the summary
        are optional extras: if one fails, the transcript is still returned,
        with a line in its "warnings" saying what is missing.
        """
        timings: dict[str, float] = {}  # seconds per stage, keyed as in jobs.STAGE_KEYS
        warnings: list[str] = []
        speakers_pending = None

        @contextmanager
        def timed(stage: str):
            start = time.monotonic()
            try:
                yield
            finally:
                timings[stage] = round(time.monotonic() - start, 2)

        try:
            report("Converting audio")
            with timed("convert"):
                audio.convert_to_wav(src, wav, self.config.stall_seconds)
                silences = audio.detect_silences(wav, self.config.stall_seconds) if options.cleanup else []
            duration = audio.wav_duration(wav)
            if duration < 0.1:
                raise audio.AudioError("This file contains no audio.")

            if options.diarize:
                speakers_pending = self._diarize_thread.submit(self._diarize, wav)

            if self._speech_model is None:
                report("Loading speech model")
                with timed("speech_load"):
                    self.load_speech_model()
            with timed("transcribe"):
                sentences = self._transcribe(wav, report)
            self._free_gpu_cache()

            if speakers_pending is not None:
                report("Identifying speakers")
                # Its own running time, most of which overlapped with transcription.
                diarization, timings["speakers"] = speakers_pending.result()
                assign_speakers(sentences, diarization)
            turns = build_turns(sentences)

            cleaned = False
            if options.cleanup and turns:
                report("Cleaning up text", 0.0)
                try:
                    with timed("cleanup"):
                        turns = clean_turns(
                            self._ask_many, turns, silences, self.config.cleanup_batch,
                            lambda fraction: report("Cleaning up text", fraction),
                        )  # fmt: skip
                    cleaned = True
                except Exception as exc:  # noqa: BLE001 - an optional extra must not cost the transcript
                    warnings.append(f"Clean-up was skipped, so this is the raw transcript. {_reason(exc)}")

            summary = None
            if options.summary and turns:
                report("Summarizing", 0.0)
                blocks = [f"{turn.speaker}: {turn.text}" if options.diarize else turn.text for turn in turns]
                try:
                    with timed("summary"):
                        summary, problems = summarize(
                            self._ask_many, blocks, options.diarize, lambda fraction: report("Summarizing", fraction)
                        )
                    warnings += problems
                except Exception as exc:  # noqa: BLE001 - as above
                    warnings.append(f"The summary was skipped. {_reason(exc)}")

            return to_result(
                sentences,
                turns,
                duration=round(duration, 3),
                language=options.language or "unknown",
                model=self.config.asr_model,
                diarized=options.diarize,
                cleaned=cleaned,
                llm_model=self.config.llm_model if cleaned or summary else None,
                summary=summary,
                warnings=warnings,
                timings=timings,
            )
        finally:
            self._free_gpu_cache()
            if speakers_pending is not None:  # don't let the WAV be deleted under senko
                speakers_pending.cancel() or speakers_pending.exception()

    @staticmethod
    def _free_gpu_cache():
        """Hand MLX's buffer cache back between stages; it matters on 16 GB Macs."""
        import mlx.core as mx

        mx.clear_cache()

    def _transcribe(self, wav: Path, report) -> list[Sentence]:
        from parakeet_mlx import DecodingConfig, SentenceConfig

        report("Transcribing", 0.0)

        def on_window(position, total):
            # Called before each window is decoded, so report the one before it.
            step = (self.config.chunk_seconds - _OVERLAP_SECONDS) * audio.SAMPLE_RATE
            report("Transcribing", max(0.0, min(1.0, (position - step) / total)))

        try:
            result = self._speech_model.transcribe(
                str(wav),
                chunk_duration=self.config.chunk_seconds,
                overlap_duration=_OVERLAP_SECONDS,
                chunk_callback=on_window,
                # Sentences normally end at punctuation; these stop run-ons from
                # becoming one enormous subtitle when the speaker never pauses.
                decoding_config=DecodingConfig(sentence=SentenceConfig(silence_gap=3.0, max_duration=30.0)),
            )
        except ValueError as exc:  # audio shorter than one frame
            raise audio.AudioError(str(exc)) from exc

        return [
            Sentence(
                start=round(s.start, 3),
                end=round(s.end, 3),
                text=s.text.strip(),
                confidence=round(s.confidence, 4),
                words=tokens_to_words(s.tokens),
            )
            for s in result.sentences
            if s.text.strip()
        ]

    def _diarize(self, wav: Path) -> tuple[list[dict], float]:
        """Returns (who spoke when, seconds taken)."""
        start = time.monotonic()
        if self._diarizer is None:
            import senko

            self._diarizer = senko.Diarizer(device="auto", warmup=True, quiet=True)
        result = self._diarizer.diarize(str(wav), generate_colors=False)
        return (list(result["merged_segments"]) if result else []), round(time.monotonic() - start, 2)


def _reason(exc: Exception) -> str:
    """Why an optional stage failed, for the warning shown with the transcript."""
    if isinstance(exc, LanguageModelError):
        return str(exc)
    traceback.print_exc()  # a bug of ours rather than the model's; leave the detail in the server log
    return f"({type(exc).__name__}: {exc})"


def worker_main(conn, config: Config, preload: bool):
    """Entry point of a worker process. `conn` is its pipe to the server process (see engine.py).

    Receives:  (src, wav, options)                 a job to run
    Sends:     ("stage", name, progress)           any number of times
               ("llm", [request, ...])             and waits for ("llm", answers_or_None, error)
               ("done", result)  or  ("error", message, is_user_error)

    With `preload`, it loads the speech model first and sends ("ready", error_or_None).
    """

    def ask_many(requests: list[tuple[str, str, int]]) -> list[str]:
        conn.send(("llm", requests))
        _, answers, error = conn.recv()
        if error:
            raise LanguageModelError(error)
        return answers

    pipeline = Pipeline(config, ask_many)
    try:
        if preload:
            error = None
            try:
                print(f"Loading speech model {config.asr_model} ...", flush=True)
                pipeline.load_speech_model()
                print("Speech model ready.", flush=True)
            except Exception as exc:  # noqa: BLE001 - reported through /health
                traceback.print_exc()
                error = f"Could not load speech model: {exc}"
            conn.send(("ready", error))

        while True:
            src, wav, options = conn.recv()
            last = None

            def report(stage, progress=None):
                nonlocal last
                update = (stage, None if progress is None else round(progress, 2))
                if update != last:  # the transcriber reports far more often than anyone needs
                    last = update
                    conn.send(("stage", *update))

            try:
                conn.send(("done", pipeline.run(Path(src), Path(wav), options, report)))
            except audio.AudioError as exc:
                conn.send(("error", str(exc), True))
            except Exception as exc:  # noqa: BLE001 - one bad file must not kill the worker
                traceback.print_exc()
                conn.send(("error", f"{type(exc).__name__}: {exc}", False))
    except (EOFError, KeyboardInterrupt):
        pass  # server is shutting down
