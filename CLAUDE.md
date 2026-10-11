# Transcriber

Local transcription + speaker diarization server for Apple Silicon Macs: a
Whisper-compatible HTTP API and a single-page web UI. See README.md for usage,
configuration and the user-facing behavior.

## Commands

- Run: `uv run python -m transcriber` (serves http://127.0.0.1:8000/)
- Tests: `uv run python -m unittest` (about ten seconds, no models needed)
- Model checks: `uv run python -m evals.run` (loads the real language model)
- There is no linter or build step. `.claude/launch.json` defines the dev server.

Python code changes need a server restart. `static/index.html` is read from
disk on every request, so UI edits only need a page reload.

## Guiding principle

Simple enough for a person to read top to bottom. Prefer a plain loop to a
clever one, a small function with a name to a comment, and one more module to
a module that does two things. Comments say why, not what.

## Constraints

- **Dependencies:** only `parakeet-mlx`, `senko` and `mlx-lm`. Everything else
  is the standard library, including the HTTP server (`http.server`) and the
  multipart parser (`multipart.py`). Do not add a web framework or other
  packages without asking.
- **Self-contained:** no cloud services, and the web UI must load nothing from
  the network (no CDNs, web fonts or external scripts). The target network has
  access to Homebrew, PyPI, GitHub and Hugging Face only.
- **Python 3.13 only:** senko requires `<3.14`.
- **macOS on Apple Silicon only:** MLX and CoreML.

## Architecture

Three kinds of process:

- **The server** (`server.py`, `engine.py`): HTTP routes, who is asking, the
  job queue. It never loads a model.
- **Workers** (`worker.py`), one per parallel job: each owns its own speech and
  speaker models and takes one recording at a time through the pipeline.
- **The language model** (`llm.py`): one process, shared by every worker to
  save memory, started on first use and stopped when idle.

`Engine` runs one thread per worker (a "slot") that hands it jobs over a pipe
and relays what comes back. The messages are listed in the docstring of
`worker.worker_main`. When a worker wants the language model it asks its slot,
which asks `SharedLLM`; workers never talk to the model directly.

The text handling is ordinary functions with no model or file access, so it
is all unit-tested: `transcript.py` (sentences, speakers, turns),
`cleanup.py`, `summary.py`, `formats.py`. Clean-up and summarizing take an
`ask_many(requests) -> answers` function instead of a model, which is how
tests and `evals/` drive them.

A finished job's result is a plain dict, documented at the top of
`formats.py` and built in `transcript.to_result`.

## Things that are easy to break

- **Do not load the language model in workers.** One copy per job is what made
  two parallel jobs not fit in 24 GB. Check `footprint` on every child process
  after changing model loading.
- **MLX is not shareable across threads.** A model must be loaded and used on
  the same thread. That is why workers are processes and why the server never
  touches MLX. Speaker identification (CoreML) is the one thing allowed on a
  second thread.
- **Privacy is enforced in `Engine`.** `get`, `list_jobs` and `delete` all take
  an `owner` and must keep doing so; another user's job is a 404, not a 403.
  The owner comes from `Handler._identity()`. Any new job route must go through
  `_job_or_404`. `tests/test_server.py` checks this over real HTTP.
- **`X-Forwarded-User` is only believed when `trust_user_header` is on.** It is
  off by default because anyone who can reach the port can send the header.
- **API jobs are never stored.** `/v1/audio/transcriptions` submits with no
  owner, so the result exists only in the response.
- **Optional stages must not cost the transcript.** If clean-up or the summary
  fails, `worker.Pipeline.run` records a warning and still returns the
  transcript. Keep new optional stages inside the same kind of `try`.
- **The server owns the files.** Uploads and decoded audio live in
  `engine.files_dir` and are deleted by the slot thread, not the worker, so a
  crashed or killed worker can't leave a recording behind.
- **Everything that waits has a deadline.** Worker replies, language-model
  replies and ffmpeg all give up after `stall_seconds`. Don't add a wait
  without one.
- **Timestamps carry raw text.** Clean-up rewrites the text of turns only.
  Segments, words, SRT and VTT stay as transcribed, because cleaned text no
  longer lines up with the timings.
- **Transcription timestamps cannot show pauses.** Parakeet's tokens run edge
  to edge, so the gap between sentences is always zero. Pauses come from
  `audio.detect_silences` (ffmpeg) instead; `cleanup.chunk_sentences` uses them
  to decide where a long turn is cut, and every cut becomes a paragraph break.
- **Clean-up must fail safe.** `cleanup.plausible()` discards model output that
  is much shorter or longer than its input and keeps the original passage.
  Keep that guard when changing the prompt or the model.
- **The summary never goes into JSON `text`.** It is appended to plain-text and
  Markdown output only (`formats.to_text_with_summary`); JSON formats carry it
  as a separate `summary` field so `text` stays the transcript.
- **Prompts are sensitive.** Small wording changes have made the model drop
  bullet markers, echo the format line, invent owners or borrow deadlines.
  After editing `prompts.py`, run `uv run python -m evals.run` and read the
  output. `summary.parse_summary` is deliberately lenient about format.
- **Undiarized summaries must not attribute.** Without speaker labels the model
  guesses owners from names said in passing and gets them wrong, so
  `summary_prompt(False)` asks for tasks with no owner. Don't add owners back.
- **Uploads stream to disk.** Never read a request body into memory; files can
  be gigabytes.
- **Render user content with `textContent`** in the UI (filenames, transcripts),
  never `innerHTML`.

## Verifying changes

Unit tests cover the text handling, the queue, the HTTP routes and failure
recovery, all without models. For anything touching the real pipeline, also
run the server and exercise it:

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions -F file=@some.m4a -F diarize=true -F cleanup=true -F summary=true -F response_format=text
```

A two-voice test clip can be made with macOS `say -v Samantha` / `say -v Daniel`
and joined with ffmpeg. Check memory with `footprint -p <pid>` on each child
process when changing anything about model loading or parallelism.
