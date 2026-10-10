"""Everything done to the audio itself, all of it with ffmpeg."""

import re
import shutil
import subprocess
from pathlib import Path

SAMPLE_RATE = 16000


class AudioError(Exception):
    """The file could not be used. The message is shown to whoever uploaded it."""


def _ffmpeg(args: list[str], timeout: float) -> subprocess.CompletedProcess:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioError("ffmpeg is not installed. Install it with: brew install ffmpeg")
    try:
        return subprocess.run(
            [ffmpeg, "-nostdin", "-hide_banner", *args], capture_output=True, text=True, errors="replace", timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioError("ffmpeg took too long to read this file.") from exc


def convert_to_wav(src: Path, dst: Path, timeout: float):
    """Decode the first audio stream of anything ffmpeg can read to 16 kHz mono PCM."""
    proc = _ffmpeg(
        ["-v", "error", "-y", "-i", str(src), "-map", "0:a:0", "-vn",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)],
        timeout,
    )  # fmt: skip
    if proc.returncode != 0 or not dst.exists():
        detail = (proc.stderr.strip().splitlines() or ["unknown error"])[-1]
        raise AudioError(f"ffmpeg could not read audio from this file: {detail}")


def wav_duration(path: Path) -> float:
    """Seconds of audio in a WAV written by convert_to_wav (44-byte header, 16-bit mono)."""
    return max(0, path.stat().st_size - 44) / (SAMPLE_RATE * 2)


def detect_silences(wav: Path, timeout: float) -> list[tuple[float, float]]:
    """(start, end) of every pause of a quarter second or more.

    The transcription's own timestamps can't provide this: its tokens run
    edge to edge, so sentences never show a gap. "Silent" is judged against
    the recording's average level, so quiet and loud recordings both work.
    Returns [] if anything goes wrong; pauses only steer where clean-up cuts.
    """
    try:
        levels = _ffmpeg(["-i", str(wav), "-af", "volumedetect", "-f", "null", "-"], timeout).stderr
        mean = float(re.search(r"mean_volume: (-?[\d.]+) dB", levels).group(1))
        threshold = max(-60.0, min(-30.0, mean - 17.0))
        log = _ffmpeg(["-i", str(wav), "-af", f"silencedetect=noise={threshold:.1f}dB:d=0.25", "-f", "null", "-"], timeout).stderr
    except (AudioError, AttributeError, ValueError):
        return []
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: (-?[\d.]+)", log)]
    return list(zip(starts, ends))  # a silence still open when the file ends has no end, and is dropped


def pause_before(silences: list[tuple[float, float]], start: float) -> float:
    """Length of the longest silence around a sentence's start time, in seconds."""
    earliest, latest = start - 0.6, start + 0.3  # sentence starts are only accurate to a few tenths
    return max((end - begin for begin, end in silences if begin <= latest and end >= earliest), default=0.0)
