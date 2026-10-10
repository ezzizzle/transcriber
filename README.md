# Transcriber

Local speech-to-text with speaker diarization for Apple Silicon Macs. Upload any
audio or video file ffmpeg can read and get back a transcript, optionally with
speaker labels, an LLM clean-up pass, and a summary of key points and action items. Nothing leaves the machine.

This project was developed by Claude, not me.

- **Web UI** at `/`: drag in files, pick options, read the transcript, download TXT / Markdown / SRT / VTT / JSON.
- **Whisper-compatible API** at `POST /v1/audio/transcriptions`, for anything that speaks the OpenAI audio API.

```
any media file ──ffmpeg──▶ 16 kHz mono WAV ─┬─▶ parakeet-mlx   (text + timestamps)
                                            └─▶ senko          (who spoke when, optional)
                              merge by time overlap ──▶ local LLM: clean-up, summary (optional)
```

## Run it

Requires macOS on Apple Silicon, [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
brew install uv ffmpeg
```

```bash
uv run python -m transcriber
```

Then open http://127.0.0.1:8000/. Models download from Hugging Face on first use
(speech model at startup, ~2.5 GB; the language model the first time clean-up or
a summary is requested, ~5.6 GB) and are cached in `~/.cache/huggingface`.

The server binds to localhost only. To serve other machines on your network:

```bash
uv run python -m transcriber --host 0.0.0.0 --port 8000
```

Use `--host ::` instead to listen on IPv6 as well as IPv4. If other machines
can't connect (the request hangs rather than being refused), the macOS firewall
on the server is the usual cause: uv's Python is not signed by a known
developer, so incoming connections are blocked until it is allowed. Check and
allow it with:

```bash
/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate
```

```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add "$(readlink -f .venv/bin/python)" --unblockapp "$(readlink -f .venv/bin/python)"
```

There is no login of its own. See [Privacy and users](#privacy-and-users) before exposing it.

## Restricted networks

Setup needs Homebrew, PyPI, GitHub and Hugging Face, and nothing else. Running
needs no network at all once the models are cached; the web UI loads no
external assets.

- **Xcode Command Line Tools** must be present, because senko has no prebuilt
  package for Python 3.13 and compiles a small C++ library on install. Homebrew
  already requires them, but they come from Apple, not from any of the four
  sources above.
- **Python 3.13**: uv downloads it from GitHub releases. If that is blocked,
  `brew install python@3.13` and uv will use it.
- **Models** otherwise download on first use, which for the clean-up model is
  the first clean-up request. Fetch both up front instead:

```bash
hf download mlx-community/parakeet-tdt-0.6b-v3
```

```bash
hf download mlx-community/Qwen3.5-9B-MLX-4bit
```

  Then start the server with `HF_HUB_OFFLINE=1` so it never tries to reach
  Hugging Face. The diarization models ship inside the senko package.

**Custom root certificates** (a TLS-inspecting proxy): Python ignores the macOS
keychain, so model downloads fail with `CERTIFICATE_VERIFY_FAILED`. Export the
system roots to a file and point `SSL_CERT_FILE` at it, for both `hf download`
and the server:

```bash
security find-certificate -a -p /Library/Keychains/System.keychain /System/Library/Keychains/SystemRootCertificates.keychain > ~/system-roots.pem
```

```bash
SSL_CERT_FILE=~/system-roots.pem uv run python -m transcriber
```

`REQUESTS_CA_BUNDLE` is not honored. If large files still fail, add
`HF_HUB_DISABLE_XET=1`. For `uv sync` itself, set `UV_SYSTEM_CERTS=1`.

Allowlists by hostname need the CDN hosts as well as the front doors:
`formulae.brew.sh`, `ghcr.io` and `pkg-containers.githubusercontent.com`
(Homebrew); `pypi.org` and `files.pythonhosted.org` (PyPI); `github.com` and
`release-assets.githubusercontent.com` (uv's Python); `huggingface.co` and
`*.hf.co` (models).

Each finished transcript in the web UI has a "Times" toggle next to its total that opens a table of how long every stage took (convert,
transcribe, identify speakers, clean up, summary, and any model loading or
queueing). Speaker identification runs alongside transcription, so the stages
can add up to more than the total. The same figures are in `GET /api/jobs` as
`timings` and in the JSON download.

## Concurrency

Up to `TRANSCRIBER_MAX_PARALLEL` jobs are processed at once; the rest wait in a
first-come-first-served queue, and the web UI shows each waiting job's place in
line. API requests share the same queue and simply block until their turn.

Each parallel job runs in its own worker process with its own speech and
diarization models, so every extra slot costs memory (see
[Resource use](#resource-use)). The language model is loaded once, in a process
of its own, and shared: transcription and speaker identification run fully in
parallel, while clean-up and summary requests from different jobs take turns.
The default is 1 slot on a 16 GB Mac, 2 on 24 GB, 3 on 32 GB, at most 4. The first worker stays
loaded; extra workers exit after 10 idle minutes to give their memory back.
All workers share one GPU, so parallel jobs each run slower: this is about
nobody waiting behind a long file, not about total throughput.

## API

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@meeting.m4a \
  -F diarize=true \
  -F cleanup=true \
  -F summary=true \
  -F response_format=text
```

| Field | Values | Notes |
|---|---|---|
| `file` | any media file | required |
| `response_format` | `json` (default), `text`, `srt`, `vtt`, `verbose_json`, `diarized_json` | `diarized_json` is OpenAI's diarization response shape and implies `diarize=true` |
| `diarize` | `true` / `false` | extension; adds `Speaker N:` to the text and a `speaker` key to segments/words |
| `cleanup` | `true` / `false` | extension; removes disfluencies and adds paragraphs |
| `summary` | `true` / `false` | extension; adds key points and action items (see below) |
| `timestamp_granularities[]` | `segment`, `word` | with `verbose_json` |
| `model` | anything | see below |
| `language` | e.g. `en` | echoed back; the model detects the language itself |

Clients that can't send extra form fields can select options through the model
name instead: `parakeet-diarize`, `parakeet-clean`, `parakeet-diarize-clean-summary`
and so on (any name containing `diariz`, `clean` or `summar`). Any other name, including `whisper-1`,
uses the server defaults. `prompt` and `temperature` are accepted and ignored.

With the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
with open("meeting.m4a", "rb") as f:
    print(client.audio.transcriptions.create(model="parakeet-diarize", file=f).text)
```

Also: `GET /v1/models`, `GET /health`.

Segments, words and subtitles always carry the raw transcription, because they
are tied to timestamps. Clean-up applies to the text (`text`, TXT, Markdown, and
the turns shown in the UI).

### Clean-up

Clean-up rewrites the whole transcript, so it is the slowest step. To keep the
model accurate, each speaker's turn is handled in passages of up to about 250
words, and a few passages are generated at once (1.6 to 2.1 times faster than
one after another in testing).

A cut between passages becomes a paragraph break, so long turns are cut where
the speaker paused longest rather than at a fixed word count; that is usually a
change of subject. Pauses are measured from the audio itself with ffmpeg.
Within a passage the model chooses the paragraph breaks.

### Key points and action items

With `summary=true` the same local model that does clean-up reads the finished
transcript and writes two short lists. They are appended below the transcript
in `text` responses and in the TXT and Markdown downloads, and returned as a
separate field in every JSON format (where `text` stays transcript-only):

```json
{"text": "...", "summary": {"key_points": ["..."], "action_items": ["Speaker 3: Send the breakdown (Thursday)"]}}
```

Action items name an owner only when speakers are identified ("Speaker 2", or
"Speaker 2 (Priya)" when the name is clear from the conversation). Without
diarization the model cannot tell who said what, so tasks are listed without an
owner and the summary avoids attributing statements to anyone. Transcripts longer than
about 50 minutes of speech are summarized in parts and then merged. Treat the
result as a draft: see [Known limits](#known-limits).

## Configuration

Environment variables, all optional:

| Variable | Default | |
|---|---|---|
| `TRANSCRIBER_HOST` / `TRANSCRIBER_PORT` | `127.0.0.1` / `8000` | |
| `TRANSCRIBER_ASR_MODEL` | `mlx-community/parakeet-tdt-0.6b-v3` | any parakeet-mlx model |
| `TRANSCRIBER_CLEANUP_MODEL` | `mlx-community/Qwen3.5-9B-MLX-4bit` | any mlx-lm chat model; used for clean-up and summaries |
| `TRANSCRIBER_DEFAULT_DIARIZE` / `_CLEANUP` / `_SUMMARY` | `false` | API defaults when the request doesn't say |
| `TRANSCRIBER_CLEANUP_BATCH` | `4` | clean-up passages sent through the language model at once; `1` disables batching |
| `TRANSCRIBER_MAX_PARALLEL` | 1 at 16 GB, 2 at 24 GB, max 4 | jobs processed at once |
| `TRANSCRIBER_IDLE_UNLOAD_SECONDS` | `600` | idle time before extra workers exit |
| `TRANSCRIBER_CHUNK_SECONDS` | `120` | transcription window; lower it to reduce peak memory |
| `TRANSCRIBER_MAX_UPLOAD_MB` | `4096` | |
| `TRANSCRIBER_MAX_JOBS` | `50` | finished web UI transcripts kept in memory |
| `TRANSCRIBER_WORK_DIR` | `$TMPDIR/transcriber` | uploads live here only while being processed |

## Privacy and users

Web UI transcripts are listed and served only to whoever uploaded them. Who
that is gets decided per request:

1. If the request has an `X-Forwarded-User` header, that value is the user.
   This is for running behind a reverse proxy that authenticates people
   (oauth2-proxy, Authelia, Caddy `forward_auth`, ...). Users see their
   transcripts from any browser.
2. Otherwise a random session cookie identifies the browser. There are no
   accounts in this mode, so clearing cookies or switching browsers loses access.

The header is trusted as-is. If you rely on it, the proxy must be the only way
to reach the server (keep the default `127.0.0.1` bind, or firewall the port)
and must overwrite any `X-Forwarded-User` sent by the client. Anyone who can
reach the port directly can otherwise claim to be any user.

Also worth knowing:

- Transcripts live in memory only and are gone when the server restarts.
  Uploaded media is deleted as soon as its job finishes.
- API calls return their result in the response and are not kept at all.
- The server does no authentication itself and speaks plain HTTP; without a
  proxy in front, anyone who can reach the port can submit audio.

## Resource use

Measured on an M4 Pro, limited to two parallel jobs, with the default models:

| Process | At rest | Peak |
|---|---|---|
| Each worker (speech + speaker models) | 2–3 GB | 6.5–8.1 GB while transcribing |
| Language model (one, shared) | 5.5 GB | 8.6 GB summarizing an hour-long transcript |

A job's worker and the language model don't peak together, but two workers
can. Two jobs transcribing at once with the language model loaded therefore
needs up to about 22 GB at the worst moment, and about 10 GB once idle. On a
24 GB Mac that leaves little to spare, and no room for a larger language model.
`TRANSCRIBER_CHUNK_SECONDS` does not help here: workers peaked at 7–8 GB with
60 and 30 second windows as well.

Timings from the same runs:

| Two jobs at once | Options | Wall time for both |
|---|---|---|
| 51-minute recordings | speakers + summary | 2.5 min |
| 8.5-minute recordings | speakers + clean-up + summary | 80 s |

Clean-up is the slow step, since it rewrites the whole transcript; transcription
alone handled a 51-minute file in 39 s. A smaller language model
(`mlx-community/Qwen3-4B-Instruct-2507-4bit`, 2.1 GB loaded against 5.5 GB) is
the alternative if speed or memory matters more than accuracy. In testing it
cleaned up text about twice as fast (44 against 23 words per second) and removed
fillers at least as well, but its summaries contained factual slips that the
default model did not make: a cost read as a travel time, a request attributed
to the wrong person, an action item nobody had agreed to. Select it with
`TRANSCRIBER_CLEANUP_MODEL`.

## Layout

```
transcriber/
  __main__.py     entry point
  server.py       HTTP routes (stdlib http.server)
  multipart.py    streaming upload parser
  pipeline.py     ffmpeg → parakeet → senko → merge; job queue and worker processes
  cleanup.py      LLM clean-up pass
  formats.py      text / Markdown / SRT / VTT / JSON renderers
  static/index.html
tests/            uv run python -m unittest
```

The only Python dependencies are `parakeet-mlx`, `senko` and `mlx-lm`.

## Known limits

- Speakers are assigned per sentence, by whichever diarized speaker overlaps it
  most. A very short reply right at a speaker change ("Got it.") can land on the
  wrong side.
- Diarization can over-split one voice into several speakers on some audio.
- The language model is small and instructed to change as little as possible; if
  its output looks wrong for a passage (much shorter or longer than the
  original), the original text is kept for that passage.
- Summaries come from a small model and contain mistakes. In testing the key
  points were dependable, while action items sometimes had the wrong owner or
  deadline, missed a task, or listed something nobody committed to. This got
  worse on transcripts long enough to be summarized in parts.
- In a long stretch by one speaker, clean-up starts a new paragraph at least
  every 250 words or so. The break is placed at a pause, but a speaker who
  changes subject without pausing will get a break slightly off the topic change.
- Queued or running jobs can't be cancelled.

## License

[MIT](LICENSE). The models it downloads and ffmpeg carry their own licenses.
