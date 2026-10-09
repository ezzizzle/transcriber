"""Render a transcription result as text, Markdown, subtitles, or Whisper-style JSON.

A result is a plain dict:

    {
      "duration": float, "language": str, "model": str,
      "diarized": bool, "cleaned": bool, "speakers": [str],
      "segments": [{"id", "start", "end", "text", "speaker"|None, "confidence"}],
      "words":    [{"word", "start", "end", "speaker"|None}],
      "turns":    [{"speaker"|None, "start", "end", "text"}],
      "summary":  {"key_points": [str], "action_items": [str]} | None,
    }

Segments and words always carry the raw transcription (they have timestamps).
Turns are consecutive segments by one speaker; their text is what the cleanup
pass rewrites, with paragraphs separated by blank lines.
"""

import json
import math


def timestamp(seconds: float, ms_sep: str = ".") -> str:
    ms = max(0, round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{ms_sep}{ms:03d}"


def clock(seconds: float) -> str:
    s = max(0, int(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def to_text(result: dict) -> str:
    """Plain text. Speaker turns are prefixed with "Speaker N:" when diarized."""
    if not result["diarized"]:
        return "\n\n".join(t["text"] for t in result["turns"])
    return "\n\n".join(f"{t['speaker']}: {t['text']}" for t in result["turns"])


SUMMARY_NOTE = "Machine-generated summary. Check it against the transcript before relying on it."


def summary_sections(result: dict) -> list[tuple[str, list[str]]]:
    """[(heading, bullets)] for the summary, or [] when there isn't one."""
    summary = result.get("summary")
    if not summary:
        return []
    return [
        ("Key points", summary["key_points"] or ["None"]),
        ("Action items", summary["action_items"] or ["None"]),
    ]


def to_text_with_summary(result: dict) -> str:
    """The transcript, then key points and action items if they were requested."""
    out = [to_text(result)]
    for heading, bullets in summary_sections(result):
        out.append(f"{heading.upper()}\n" + "\n".join(f"- {b}" for b in bullets))
    if len(out) > 1:
        out.insert(1, "-" * 40)
        out.append(f"({SUMMARY_NOTE})")
    return "\n\n".join(out)


def to_markdown(result: dict, title: str = "Transcript") -> str:
    out = [f"# {title}", ""]
    for t in result["turns"]:
        if result["diarized"]:
            out += [f"**{t['speaker']}** · {clock(t['start'])}", ""]
        out += [t["text"], ""]
    for heading, bullets in summary_sections(result):
        out += [f"## {heading}", "", *(f"- {b}" for b in bullets), ""]
    if result.get("summary"):
        out += [f"*{SUMMARY_NOTE}*", ""]
    return "\n".join(out).rstrip() + "\n"


def to_srt(result: dict) -> str:
    blocks = []
    for i, s in enumerate(result["segments"], 1):
        text = f"[{s['speaker']}] {s['text']}" if s.get("speaker") else s["text"]
        blocks.append(f"{i}\n{timestamp(s['start'], ',')} --> {timestamp(s['end'], ',')}\n{text}\n")
    return "\n".join(blocks)


def to_vtt(result: dict) -> str:
    blocks = ["WEBVTT\n"]
    for s in result["segments"]:
        text = f"<v {s['speaker']}>{s['text']}" if s.get("speaker") else s["text"]
        blocks.append(f"{timestamp(s['start'])} --> {timestamp(s['end'])}\n{text}\n")
    return "\n".join(blocks)


def to_verbose_json(result: dict, words: bool = False, segments: bool = True) -> dict:
    """OpenAI `verbose_json`, plus a `speaker` key on segments/words when diarized."""
    out = {
        "task": "transcribe",
        "language": result["language"],
        "duration": result["duration"],
        "text": to_text(result),
    }
    if result.get("summary"):
        out["summary"] = result["summary"]
    if segments:
        out["segments"] = [
            {
                "id": s["id"],
                "seek": 0,
                "start": s["start"],
                "end": s["end"],
                "text": s["text"],
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": round(math.log(max(s["confidence"], 1e-10)), 4),
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
                **({"speaker": s["speaker"]} if s.get("speaker") else {}),
            }
            for s in result["segments"]
        ]
    if words:
        out["words"] = [
            {k: v for k, v in w.items() if v is not None} for w in result["words"]
        ]
    return out


def to_diarized_json(result: dict) -> dict:
    """OpenAI `diarized_json` (the gpt-4o-transcribe-diarize response shape)."""
    return {
        "task": "transcribe",
        "duration": result["duration"],
        "text": to_text(result),
        **({"summary": result["summary"]} if result.get("summary") else {}),
        "segments": [
            {
                "type": "transcript.text.segment",
                "id": f"seg_{s['id']}",
                "start": s["start"],
                "end": s["end"],
                "text": s["text"],
                "speaker": s.get("speaker") or "Speaker 1",
            }
            for s in result["segments"]
        ],
    }


def to_full_json(result: dict) -> str:
    """Everything we know, for the web UI's JSON download."""
    return json.dumps({"text": to_text(result), **result}, indent=2, ensure_ascii=False) + "\n"
