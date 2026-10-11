"""The transcript while it is being built: sentences, who said them, and speaker turns.

Nothing here touches a model or a file, so all of it is easy to test.
"""

from dataclasses import dataclass


@dataclass
class Sentence:
    """One sentence as transcribed. These become the timestamped `segments` of a result."""

    start: float
    end: float
    text: str
    confidence: float
    words: list[dict]  # [{"word", "start", "end"}]
    speaker: str | None = None  # "Speaker 1", ... once speakers have been assigned


@dataclass
class Turn:
    """Consecutive sentences by one speaker. Without speaker labels the whole recording is one turn."""

    speaker: str | None
    sentences: list[Sentence]
    text: str = ""  # what is shown; clean-up replaces it, with paragraphs separated by blank lines

    @property
    def start(self) -> float:
        return self.sentences[0].start

    @property
    def end(self) -> float:
        return self.sentences[-1].end


def tokens_to_words(tokens) -> list[dict]:
    """Join parakeet's sub-word tokens (word starts carry a leading space)."""
    words: list[dict] = []
    for tok in tokens:
        if words and not tok.text.startswith(" "):
            words[-1]["word"] += tok.text
            words[-1]["end"] = round(tok.end, 3)
        elif tok.text.strip():
            words.append({"word": tok.text.strip(), "start": round(tok.start, 3), "end": round(tok.end, 3)})
    return words


def assign_speakers(sentences: list[Sentence], diarization: list[dict]) -> None:
    """Label each sentence with the diarization speaker it overlaps most.

    `diarization` is senko's output: [{"start", "end", "speaker"}]. Sentences
    that overlap nobody (diarizers drop very short utterances) take the nearest
    speaker in time. Speakers are renamed "Speaker 1", "Speaker 2", ... in
    order of first appearance.
    """
    names: dict[str, str] = {}
    for sentence in sentences:
        if not diarization:
            sentence.speaker = "Speaker 1"
            continue

        def overlap(d: dict) -> float:
            return min(sentence.end, d["end"]) - max(sentence.start, d["start"])

        best = max(diarization, key=overlap)
        if overlap(best) <= 0:
            middle = (sentence.start + sentence.end) / 2
            best = min(diarization, key=lambda d: max(d["start"] - middle, middle - d["end"], 0))
        sentence.speaker = names.setdefault(best["speaker"], f"Speaker {len(names) + 1}")


def build_turns(sentences: list[Sentence]) -> list[Turn]:
    """Group consecutive sentences by the same speaker."""
    turns: list[Turn] = []
    for sentence in sentences:
        if turns and turns[-1].speaker == sentence.speaker:
            turns[-1].sentences.append(sentence)
        else:
            turns.append(Turn(sentence.speaker, [sentence]))
    for turn in turns:
        turn.text = " ".join(sentence.text for sentence in turn.sentences)
    return turns


def to_result(sentences: list[Sentence], turns: list[Turn], **details) -> dict:
    """The finished job as plain data, ready to be sent as JSON. formats.py documents the shape."""
    speakers = []
    for turn in turns:
        if turn.speaker and turn.speaker not in speakers:
            speakers.append(turn.speaker)
    return {
        **details,
        "speakers": speakers,
        "turns": [
            {"speaker": turn.speaker, "start": turn.start, "end": turn.end, "text": turn.text} for turn in turns
        ],
        "segments": [
            {"id": i, "start": s.start, "end": s.end, "text": s.text, "speaker": s.speaker, "confidence": s.confidence}
            for i, s in enumerate(sentences)
        ],
        "words": [{**word, "speaker": s.speaker} for s in sentences for word in s.words],
    }
