# Transcriber

Local transcription + speaker diarization server for Apple Silicon Macs: a
Whisper-compatible HTTP API and a single-page web UI. See README.md for usage,
configuration and the user-facing behavior.

## Commands

- Run: `uv run python -m transcriber` (serves http://127.0.0.1:8000/)
- Tests: `uv run python -m unittest` (fast, no models needed)
- There is no linter or build step. `.claude/launch.json` defines the dev server.

Python code changes need a server restart. `static/index.html` is read from
disk on every request, so UI edits only need a page reload.

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

- `server.py` is the parent process: HTTP routes, uploads, user identity. It
  never imports a model.
- `pipeline.py` holds both halves of the job system. `Engine` (parent) owns the
  queue and one thread per worker slot. `Pipeline` and `worker_main` run in
  spawned worker processes, one per parallel job, each with its own speech and
  diarization models. They talk over a pipe: `("stage", name, progress)`, then
  `("done", result)` or `("error", message, is_user_error)`.
- The language model lives in one extra process (`cleanup.llm_main`), shared by
  all workers to save memory. A worker sends `("llm", [(system, user,
  max_tokens), ...])` up its pipe; the parent's `SharedLLM` forwards the batch
  under a lock and relays the answers. Prompt building and output checking
  (`cleanup.Editor`) stay in the worker; only raw `ask_many` calls cross
  processes. Clean-up sends several passages per call because batched
  generation is about twice as fast.
- `cleanup.py` holds both LLM passes (clean-up and the key points / action
  items summary); `formats.py` renders a result dict (shape
  documented at the top of that file) into text, Markdown, SRT, VTT and JSON.

## Things that are easy to break

- **Do not load the language model in workers.** One copy per job is what made
  two parallel jobs not fit in 24 GB. Check `footprint` on every child process
  after changing model loading.
- **MLX is not shareable across threads.** A model must be loaded and used on
  the same thread. That is why workers are processes and why the parent never
  touches MLX. Diarization (CoreML) is the one thing allowed on a second thread.
- **Privacy is enforced in `Engine`.** `get`, `list_jobs` and `delete` all take
  an `owner` and must keep doing so; another user's job is a 404, not a 403.
  The owner comes from `Handler._session()`: the `X-Forwarded-User` header if
  present, else the session cookie. Any new job route must go through
  `_job_or_404`.
- **API jobs are never stored.** `/v1/audio/transcriptions` submits with no
  owner, so the result exists only in the response.
- **Timestamps carry raw text.** Clean-up rewrites `turns` only. Segments,
  words, SRT and VTT stay as transcribed, because cleaned text no longer lines
  up with the timings.
- **Transcription timestamps cannot show pauses.** Parakeet's tokens run edge
  to edge, so the gap between sentences is always zero. Pauses come from
  `detect_silences` (ffmpeg) instead; `chunk_sentences` uses them to decide
  where a long turn is cut, and every cut becomes a paragraph break.
- **Clean-up must fail safe.** `cleanup.plausible()` discards LLM output that
  is much shorter or longer than its input and keeps the original passage.
  Keep that guard when changing the prompt or the model.
- **The summary never goes into JSON `text`.** It is appended to plain-text and
  Markdown output only (`formats.to_text_with_summary`); JSON formats carry it
  as a separate `summary` field so `text` stays the transcript.
- **Summary prompts are sensitive.** Small wording changes have made the model
  drop bullet markers, echo the format line, invent owners or borrow deadlines.
  After editing `SUMMARY_PROMPT`, re-run a diarized and an undiarized transcript
  and read the output. `parse_summary` is deliberately lenient about format.
- **Undiarized summaries must not attribute.** Without speaker labels the model
  guesses owners from names said in passing and gets them wrong, so
  `summary_prompt(False)` asks for tasks with no owner. Don't add owners back.
- **Uploads stream to disk.** Never read a request body into memory; files can
  be gigabytes.
- **Render user content with `textContent`** in the UI (filenames, transcripts),
  never `innerHTML`.
- A method named `list` on a class breaks `list[...]` annotations later in the
  class body; that is why it is `list_jobs`.

## Verifying changes

Unit tests cover only the model-free parts. For anything touching the pipeline
or routes, run the server and exercise it for real:

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions -F file=@some.m4a -F diarize=true -F cleanup=true -F response_format=text
```

A two-voice test clip can be made with macOS `say -v Samantha` / `say -v Daniel`
and joined with ffmpeg. Check memory with `footprint -p <worker pid>` when
changing anything about model loading or parallelism.
