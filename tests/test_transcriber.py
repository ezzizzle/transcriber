"""Fast tests for the parts that don't need a model. Run: uv run python -m unittest"""

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from transcriber import cleanup, formats, multipart
from transcriber.pipeline import assign_speakers, build_turns, tokens_to_words


def body(boundary: str, *parts: tuple[str, bytes]) -> bytes:
    out = b""
    for headers, payload in parts:
        out += f"--{boundary}\r\n{headers}\r\n\r\n".encode() + payload + b"\r\n"
    return out + f"--{boundary}--\r\n".encode()


class MultipartTest(unittest.TestCase):
    def parse(self, raw: bytes, max_file=10**9):
        self.dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        return multipart.parse(io.BytesIO(raw), "multipart/form-data; boundary=XyZ", len(raw), self.dir, max_file)

    def test_fields_and_file(self):
        # Larger than one read, and containing bytes that look like a boundary.
        payload = (b"\r\n--Xy" + bytes(range(256))) * 3000
        raw = body(
            "XyZ",
            ('Content-Disposition: form-data; name="diarize"', b"true"),
            ('Content-Disposition: form-data; name="timestamp_granularities[]"', b"word"),
            ('Content-Disposition: form-data; name="timestamp_granularities[]"', b"segment"),
            ('Content-Disposition: form-data; name="file"; filename="my talk.WebM"\r\nContent-Type: video/webm', payload),
        )
        fields, files = self.parse(raw)
        self.assertEqual(fields["diarize"], ["true"])
        self.assertEqual(fields["timestamp_granularities[]"], ["word", "segment"])
        upload = files["file"]
        self.assertEqual((upload.filename, upload.size, upload.path.suffix), ("my talk.WebM", len(payload), ".WebM"))
        self.assertEqual(upload.path.read_bytes(), payload)

    def test_oversized_file_is_rejected_and_removed(self):
        raw = body("XyZ", ('Content-Disposition: form-data; name="file"; filename="a.mp3"', b"x" * 5000))
        with self.assertRaises(multipart.MultipartError):
            self.parse(raw, max_file=1000)
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_truncated_body(self):
        raw = body("XyZ", ('Content-Disposition: form-data; name="file"; filename="a.mp3"', b"abc"))[:-12]
        with self.assertRaises(multipart.MultipartError):
            self.parse(raw)


class PipelineHelpersTest(unittest.TestCase):
    def test_tokens_to_words(self):
        tokens = [SimpleNamespace(text=t, start=i, end=i + 1) for i, t in enumerate([" Hel", "lo", ",", " world"])]
        self.assertEqual(
            tokens_to_words(tokens),
            [{"word": "Hello,", "start": 0, "end": 3}, {"word": "world", "start": 3, "end": 4}],
        )

    def test_assign_speakers_and_turns(self):
        segments = [
            {"start": 0.0, "end": 4.0, "text": "Hi."},
            {"start": 4.0, "end": 6.0, "text": "How are you?"},
            {"start": 6.2, "end": 6.5, "text": "Good."},  # falls in a gap: nearest speaker wins
            {"start": 9.0, "end": 12.0, "text": "Great."},
        ]
        diarization = [
            {"start": 6.6, "end": 8.0, "speaker": "SPEAKER_07"},
            {"start": 0.0, "end": 6.0, "speaker": "SPEAKER_03"},
            {"start": 8.5, "end": 12.0, "speaker": "SPEAKER_03"},
        ]
        assign_speakers(segments, diarization)
        self.assertEqual([s["speaker"] for s in segments], ["Speaker 1", "Speaker 1", "Speaker 2", "Speaker 1"])
        turns = build_turns(segments)
        self.assertEqual([(t["speaker"], t["text"]) for t in turns],
                         [("Speaker 1", "Hi. How are you?"), ("Speaker 2", "Good."), ("Speaker 1", "Great.")])


class FormatsTest(unittest.TestCase):
    result = {
        "duration": 3725.5, "language": "en", "diarized": True,
        "segments": [{"id": 0, "start": 3723.004, "end": 3725.5, "text": "Bye.", "speaker": "Speaker 1", "confidence": 0.9}],
        "words": [{"word": "Bye.", "start": 3723.004, "end": 3725.5, "speaker": "Speaker 1"}],
        "turns": [{"speaker": "Speaker 1", "start": 3723.004, "end": 3725.5, "text": "Bye."}],
    }  # fmt: skip

    def test_subtitles(self):
        self.assertEqual(formats.to_srt(self.result), "1\n01:02:03,004 --> 01:02:05,500\n[Speaker 1] Bye.\n")
        self.assertEqual(formats.to_vtt(self.result), "WEBVTT\n\n01:02:03.004 --> 01:02:05.500\n<v Speaker 1>Bye.\n")

    def test_text_and_verbose_json(self):
        self.assertEqual(formats.to_text(self.result), "Speaker 1: Bye.")
        verbose = formats.to_verbose_json(self.result, words=True)
        self.assertEqual(verbose["segments"][0]["speaker"], "Speaker 1")
        self.assertEqual(verbose["words"][0]["word"], "Bye.")


class CleanupHelpersTest(unittest.TestCase):
    def test_strip_fillers(self):
        self.assertEqual(cleanup.strip_fillers("Um, yeah, uh, sure."), "Yeah, sure.")
        self.assertEqual(cleanup.strip_fillers("Umm."), "")
        self.assertEqual(cleanup.strip_fillers("The summer hummed."), "The summer hummed.")

    def test_chunking_and_guard(self):
        self.assertEqual(cleanup.chunk_sentences(["a b c", "d e", "f"], max_words=4), ["a b c", "d e f"])
        original = "word " * 100
        self.assertTrue(cleanup.plausible(original, "word " * 80))
        self.assertFalse(cleanup.plausible(original, "A short summary."))
        self.assertFalse(cleanup.plausible(original, "word " * 200))
        self.assertEqual(cleanup.normalize_paragraphs("<think>x</think>One.\nTwo.\n\n\nThree."), "One.\n\nTwo.\n\nThree.")


class SummaryTest(unittest.TestCase):
    def test_parse_summary(self):
        raw = "<think>hm</think>## Key points\n- **Release** moved to March.\n* Budget is over.\n\n**Action items**\n1. Speaker 2: tell the team (no deadline specified)\nSpeaker 3: send it (Thursday)\n- None\n"
        self.assertEqual(
            cleanup.parse_summary(raw),
            {"key_points": ["Release moved to March.", "Budget is over."], "action_items": ["Speaker 2: tell the team", "Speaker 3: send it (Thursday)"]},
        )
        self.assertEqual(cleanup.parse_summary("## Key points\n- A\n## Action items\n- None"),
                         {"key_points": ["A"], "action_items": []})
        self.assertIsNone(cleanup.parse_summary("Sure! Here is a summary of the meeting."))

    def test_split_for_summary(self):
        self.assertEqual(cleanup.split_for_summary(["a b", "c d", "e"], max_words=4), ["a b\n\nc d", "e"])
        self.assertEqual(cleanup.split_for_summary(["a b c d e"], max_words=2), ["a b", "c d", "e"])

    def test_rendering(self):
        result = {**FormatsTest.result, "summary": {"key_points": ["Said bye."], "action_items": []}}
        text = formats.to_text_with_summary(result)
        self.assertTrue(text.startswith("Speaker 1: Bye.\n\n"))
        self.assertIn("KEY POINTS\n- Said bye.\n\nACTION ITEMS\n- None", text)
        self.assertIn("## Action items\n\n- None", formats.to_markdown(result))
        self.assertEqual(formats.to_text(result), "Speaker 1: Bye.")  # JSON `text` stays transcript-only
        self.assertEqual(formats.to_verbose_json(result)["summary"]["key_points"], ["Said bye."])
        self.assertEqual(formats.to_text_with_summary(FormatsTest.result), "Speaker 1: Bye.")
        self.assertNotIn("summary", formats.to_verbose_json(FormatsTest.result))


if __name__ == "__main__":
    unittest.main()
