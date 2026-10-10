"""Clean-up: remove filler words and false starts, fix punctuation, add paragraphs.

The language model does the rewriting. Small models drift on long inputs, so
each speaker turn goes through in passages of up to about 250 words, and any
answer that looks like the model went off-script (far too short, far too
long) is discarded in favour of the original text.
"""

import re

from .audio import pause_before
from .prompts import CLEANUP_PROMPT
from .transcript import Turn

# Filler sounds as whole words. The hyphen guards keep "uh-huh" and "mm-hmm", which mean something.
_FILLER = r"(?<![\w-])(?:u+h+m*|u+m+|e+r+m*|a+h+|h+m+)(?![\w-])"
_MIN_LLM_WORDS = 12


def strip_fillers(text: str) -> str:
    """Remove "um", "uh", "er", "ah" and "hmm", tidying the punctuation they leave behind.

    Plain pattern matching, no model. It is the whole clean-up for passages too
    short to be worth a model call, and a final sweep over the model's output,
    because the model lets some fillers through.
    """

    def one(paragraph: str) -> str:
        flags = re.I
        p = paragraph
        # A filler that is a whole sentence: "Right. Um. Okay." -> "Right. Okay."
        p = re.sub(rf"(^|[.?!…]\s+){_FILLER}[.?!…]+\s*", r"\1", p, flags=flags)
        # At the start of a sentence, the next word takes the capital: "Um, the plan" -> "The plan"
        p = re.sub(rf"(^|[.?!…]\s+){_FILLER}[,…]*\s+(\w)", lambda m: m.group(1) + m.group(2).upper(), p, flags=flags)
        # After a joining word the commas go too: "and, uh, nobody" -> "and nobody"
        p = re.sub(rf"\b(and|but|or|so|because|that|then),\s+{_FILLER},\s+", r"\1 ", p, flags=flags)
        # At the end of a sentence: "it was, um." -> "it was."
        p = re.sub(rf",?\s*{_FILLER}(?=\s*[.?!…])", "", p, flags=flags)
        # Anywhere else, with the comma that follows: "Anyway, um, the plan" -> "Anyway, the plan"
        p = re.sub(rf"{_FILLER}[,…]*[ \t]*", "", p, flags=flags)
        p = re.sub(r"[ \t]+([,.?!])", r"\1", p)
        return re.sub(r"[ \t]{2,}", " ", p).strip()

    return "\n\n".join(filter(None, (one(paragraph) for paragraph in text.split("\n\n"))))


def chunk_sentences(sentences: list[str], pauses: list[float] | None = None, max_words: int = 250) -> list[str]:
    """Split a run of sentences into chunks of at most `max_words` for the LLM.

    `pauses[i]` is the silence before sentence i, in seconds. A chunk boundary
    becomes a paragraph break, so when a cut is needed it goes where the
    speaker paused longest in the second half of the chunk, which is usually a
    change of subject. Without pauses the chunk is simply filled.
    """
    pauses = pauses or [0.0] * len(sentences)
    counts = [len(sentence.split()) for sentence in sentences]
    chunks, start = [], 0
    while start < len(sentences):
        end, words = start, 0
        while end < len(sentences) and (end == start or words + counts[end] <= max_words):
            words += counts[end]
            end += 1
        if end < len(sentences):
            running, candidates = 0, []
            for cut in range(start + 1, end + 1):  # cut = index of the first sentence of the next chunk
                running += counts[cut - 1]
                if running >= max_words / 2:
                    candidates.append(cut)
            if candidates:
                end = max(candidates, key=lambda cut: (pauses[cut], cut))
        chunks.append(" ".join(sentences[start:end]))
        start = end
    return chunks


def normalize_paragraphs(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"</?transcript>", "", text)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n|\n", text)]
    return "\n\n".join(p for p in paragraphs if p)


def plausible(original: str, cleaned: str) -> bool:
    """A cleaned passage should be a bit shorter than the original, never wildly off."""
    before, after = len(original.split()), len(cleaned.split())
    return bool(cleaned) and 0.4 * before <= after <= 1.15 * before + 5


def clean_passages(ask_many, texts: list[str], batch_size: int = 4, progress=lambda fraction: None) -> list[str]:
    """Clean each passage on its own, sending a few at a time to the model.

    `ask_many` takes a list of (system, user, max_tokens) requests and returns
    the model's answers. Batching is only for speed: generating for several
    passages at once is about twice as fast as one after another.
    """
    out = [strip_fillers(text) if len(text.split()) < _MIN_LLM_WORDS else None for text in texts]
    todo = [i for i, cleaned in enumerate(out) if cleaned is None]
    total = sum(len(texts[i].split()) for i in todo) or 1
    done = 0
    for at in range(0, len(todo), batch_size):
        batch = todo[at : at + batch_size]
        answers = ask_many([
            (CLEANUP_PROMPT, f"<transcript>\n{texts[i]}\n</transcript>", len(texts[i].split()) * 3 + 200)
            for i in batch
        ])  # fmt: skip
        for i, raw in zip(batch, answers):
            cleaned = normalize_paragraphs(raw)
            # The model lets some fillers through, and its rejected answers leave them all in.
            out[i] = strip_fillers(cleaned if plausible(texts[i], cleaned) else texts[i])
            done += len(texts[i].split())
        progress(done / total)
    return out


def clean_turns(
    ask_many, turns: list[Turn], silences: list[tuple[float, float]], batch_size: int = 4, progress=lambda fraction: None
) -> list[Turn]:
    """Clean every turn's text. Returns the turns that still have something in them."""
    # Long turns are cut into passages at the speaker's pauses. The passages of
    # all turns go to the model as one list, and `owner` remembers whose is whose.
    passages, owner = [], []
    for turn in turns:
        pauses = [pause_before(silences, sentence.start) for sentence in turn.sentences]
        for passage in chunk_sentences([sentence.text for sentence in turn.sentences], pauses):
            passages.append(passage)
            owner.append(turn)
    cleaned = clean_passages(ask_many, passages, batch_size, progress)

    kept: list[Turn] = []
    for turn in turns:
        parts = [text for text, whose in zip(cleaned, owner) if whose is turn and text]
        if not parts:  # the whole turn was filler ("Um.")
            continue
        text = "\n\n".join(parts)  # a cut between passages becomes a paragraph break
        if kept and kept[-1].speaker == turn.speaker:  # the turn between them was dropped
            kept[-1] = Turn(turn.speaker, kept[-1].sentences + turn.sentences, kept[-1].text + "\n\n" + text)
        else:
            kept.append(Turn(turn.speaker, turn.sentences, text))
    return kept
