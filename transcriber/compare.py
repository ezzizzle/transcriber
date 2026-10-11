"""What clean-up changed, word by word, so a person can check its work.

Clean-up is done by a language model, and the only automatic check on it is
that the result is about the right length (cleanup.plausible). That cannot
notice a changed name, a changed number or a dropped "not". Showing exactly
what was taken out and put in lets a reader notice instead.
"""

import difflib
import re

from .cleanup import is_filler


def changes(raw: str, cleaned: str) -> list[list[str]]:
    """Compare a turn as transcribed with the same turn after clean-up.

    Returns the cleaned text in reading order as [kind, text] pieces:

        "same"     words found in both
        "filler"   "um", "uh" and the like, which clean-up is meant to take out
        "removed"  other words clean-up took out
        "added"    words clean-up put in, often in place of removed ones
        "break"    a paragraph break in the cleaned text; its text is empty

    Fillers and removed words are not part of the cleaned text. Capital
    letters and punctuation are ignored when comparing, because clean-up
    changes those all the time, on purpose. Punctuation inside a number is
    not ignored: see _comparable.
    """
    raw_words = raw.split()
    cleaned_words, paragraph_starts = [], set()
    for paragraph in cleaned.split("\n\n"):
        paragraph_starts.add(len(cleaned_words))
        cleaned_words += paragraph.split()
    paragraph_starts.discard(0)

    pieces: list[list[str]] = []

    def add(kind: str, word: str):
        if pieces and pieces[-1][0] == kind:
            pieces[-1][1] += " " + word
        else:
            pieces.append([kind, word])

    def add_cleaned(kind: str, start: int, end: int):
        for i in range(start, end):
            if i in paragraph_starts:
                pieces.append(["break", ""])
            # A dash or other stray punctuation on its own is not a change worth pointing out.
            add(kind if _comparable(cleaned_words[i]) else "same", cleaned_words[i])

    matcher = difflib.SequenceMatcher(None, [_comparable(w) for w in raw_words], [_comparable(w) for w in cleaned_words])
    for operation, raw_start, raw_end, cleaned_start, cleaned_end in matcher.get_opcodes():
        if operation == "equal":
            add_cleaned("same", cleaned_start, cleaned_end)
        else:  # words replaced, taken out or put in
            for word in raw_words[raw_start:raw_end]:
                if _comparable(word):
                    add("filler" if is_filler(word) else "removed", word)
            add_cleaned("added", cleaned_start, cleaned_end)
    return pieces


def as_markdown(raw: str, cleaned: str) -> str:
    """The cleaned text of a turn as Markdown, marked with what clean-up changed.

    Words it took out are ~~struck through~~ and words it put in are **bold**,
    so the result reads sensibly as plain text and shows up properly anywhere
    that displays Markdown.
    """
    paragraphs: list[list[str]] = [[]]
    for kind, text in changes(raw, cleaned):
        text = re.sub(r"([*~])", r"\\\1", text)  # so a * or ~ that was actually said isn't taken for a mark
        if kind == "break":
            paragraphs.append([])
        elif kind == "same":
            paragraphs[-1].append(text)
        elif kind == "added":
            paragraphs[-1].append(f"**{text}**")
        else:  # a filler or another removed word
            paragraphs[-1].append(f"~~{text}~~")
    return "\n\n".join(" ".join(paragraph) for paragraph in paragraphs)


# Punctuation that wraps a word or ends a sentence, rather than being part of a number.
_OPENING = "\"'“‘([{"
_CLOSING = "\"'”’)]}.,;:!?…"


def _comparable(word: str) -> str:
    """A word as it is compared: lower case and without punctuation, so "So," matches "so".

    A number keeps the punctuation inside it, because there it carries
    meaning: "1.5%" is not "15%", "-5" is not "5" and "1/2" is not "12". Only
    what wraps a number or ends the sentence is ignored, so "14th." still
    matches "14th".
    """
    word = word.lower()
    if any(character.isdigit() for character in word):
        return word.lstrip(_OPENING).rstrip(_CLOSING)
    return re.sub(r"\W", "", word)
