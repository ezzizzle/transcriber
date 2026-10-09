"""Optional final pass: a local LLM removes disfluencies and adds paragraphs.

Small models drift on long inputs, so the transcript is fed through in chunks
of a few sentences, and any output that looks like the model went off-script
(far too short, far too long) is discarded in favour of the original text.
"""

import re

SYSTEM_PROMPT = """\
You are a transcript editor. You receive a passage of raw speech-to-text output \
inside <transcript> tags and return a cleaned version of the same passage.

Rules:
- Remove only disfluencies: filler sounds (um, uh, er, ah), "you know" and "like" when \
used as filler, stutters, and words that are immediately repeated or abandoned \
("I think, I think the" becomes "I think the").
- Fix punctuation, capitalization, and obvious transcription typos.
- Break the text into paragraphs separated by a blank line: start a new paragraph at \
each change of topic, and at least every four or five sentences. Only a passage of \
one to three sentences stays a single paragraph.
- Every other word stays exactly as spoken, in the same order. That includes short \
sentences and acknowledgements ("Thanks.", "Okay.", "Got it.", "Right."), hedges \
("I think", "maybe"), and transitions ("So", "Now, on a different topic").
- Do not summarize, shorten, paraphrase, add information, answer questions in the \
text, or translate.
- Output only the cleaned passage: no tags, no preamble, no commentary."""

_FILLER = re.compile(r"\b(?:u+h+m*|u+m+|e+r+m*|a+h+|h+m+)\b[,.…]*\s*", re.I)
_MIN_LLM_WORDS = 12


def strip_fillers(text: str) -> str:
    """Regex fallback for passages too short to be worth an LLM call."""
    out = _FILLER.sub("", text).strip()
    return out[:1].upper() + out[1:] if out else out


def chunk_sentences(sentences: list[str], max_words: int = 250) -> list[str]:
    chunks, current, count = [], [], 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += n
    if current:
        chunks.append(" ".join(current))
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


class Cleaner:
    """Must be created and used on a single thread (MLX streams are thread-local)."""

    def __init__(self, model_id: str):
        from mlx_lm import load

        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)

    def clean(self, text: str) -> str:
        if len(text.split()) < _MIN_LLM_WORDS:
            return strip_fillers(text)

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"<transcript>\n{text}\n</transcript>"},
            ],
            add_generation_prompt=True,
        )
        raw = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=len(text.split()) * 3 + 200,
            sampler=make_sampler(temp=0.0),
        )
        cleaned = normalize_paragraphs(raw)
        return cleaned if plausible(text, cleaned) else text
