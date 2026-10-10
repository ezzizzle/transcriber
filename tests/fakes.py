"""Stand-ins for the worker and language-model processes, so the queue and the
HTTP routes can be tested in seconds and without any models.

The "recording" a fake worker is given is a text file saying what to do:

    hello there         -> a transcript with that text
    bad                 -> the error a real worker sends for a file that isn't audio
    crash               -> the process dies
    hang                -> the process stops responding
    sleep 0.5 some text -> takes that long, then a transcript of "some text"
"""

import os
import time
from pathlib import Path

from transcriber.transcript import Sentence, build_turns, to_result


def worker(conn, config, preload):
    if preload:
        conn.send(("ready", None))
    while True:
        src, wav, options = conn.recv()
        text = Path(src).read_text()
        if text == "bad":
            conn.send(("error", "ffmpeg could not read audio from this file", True))
            continue
        if text == "crash":
            os._exit(1)
        if text == "hang":
            time.sleep(3600)
        if text.startswith("sleep "):
            _, seconds, text = text.split(" ", 2)
            time.sleep(float(seconds))

        Path(wav).write_text("decoded audio")
        conn.send(("stage", "Transcribing", 0.5))
        summary, warnings = None, []
        if options.summary:
            conn.send(("llm", [("system", text, 10)]))
            _, answers, error = conn.recv()
            if error:
                warnings.append(f"The summary was skipped. {error}")
            else:
                summary = {"executive_summary": answers[0], "key_points": [], "action_items": []}

        sentences = [Sentence(start=0.0, end=1.0, text=text, confidence=1.0, words=[])]
        conn.send(("done", to_result(
            sentences, build_turns(sentences), duration=1.0, language="unknown", model="fake",
            diarized=False, cleaned=False, summary=summary, warnings=warnings, timings={"transcribe": 0.01},
        )))  # fmt: skip


def llm(conn, model_id):
    conn.send(("ready", None))
    while True:
        requests = conn.recv()
        conn.send(("ok", [f"Summary of: {user}" for _, user, _ in requests]))


def llm_that_cannot_load(conn, model_id):
    conn.send(("ready", "Could not load the language model."))
