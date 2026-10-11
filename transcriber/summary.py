"""The summary: an executive summary, key points and action items.

The model reads the whole transcript in one go when it fits. A longer one
is summarized in parts, and the parts' lists are joined end to end. The
model's answers are Markdown, which is taken apart here into plain data.
"""

import re

from .llm import LanguageModelError
from .prompts import EXECUTIVE_PROMPT, summary_prompt

# Roughly 50 minutes of speech; comfortably inside the model's context window.
_SUMMARY_MAX_WORDS = 9000

def split_for_summary(blocks: list[str], max_words: int = _SUMMARY_MAX_WORDS) -> list[str]:
    """Join transcript blocks (speaker turns) into as few parts as fit the model."""
    pieces = []
    for block in blocks:
        words = block.split()
        if len(words) <= max_words:
            pieces.append(block)
            continue
        # One speaker talking for an hour is still a single block. Cut it up,
        # and say who is speaking at the start of every piece, not just the first.
        label = re.match(r"Speaker \d+: ", block)
        label = label.group() if label else ""
        words = block[len(label) :].split()
        pieces += [label + " ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]

    parts, current, count = [], [], 0
    for piece in pieces:
        n = len(piece.split())
        if current and count + n > max_words:
            parts.append("\n\n".join(current))
            current, count = [], 0
        current.append(piece)
        count += n
    if current:
        parts.append("\n\n".join(current))
    return parts


# The model is told to leave the brackets out when a task has no deadline, and
# sometimes fills them with a way of saying so instead: "(No deadline specified)",
# "(not stated)", "(Deadline: none)", "(N/A)". Real deadlines such as "(not before
# Friday)" or "(to be decided next week)" don't match and are kept.
_NO_DEADLINE = re.compile(
    r"""\s*\(\s*(?:
        (?:no|none|not|n/?a|unspecified|unknown)
        (?:\s+(?:deadline|due\s+date|date|time\s?frame|timeline))?
        (?:\s+(?:was\s+|were\s+)?(?:specified|given|stated|mentioned|said|set|provided|applicable|discussed))?
      |
        (?:deadline|due\s+date|due)\s*(?:is|was|:|-)?\s*
        (?:none|n/?a|unspecified|unknown|not\s+(?:specified|given|stated|mentioned|said|set|provided|discussed))
    )\s*\.?\s*\)""",
    re.I | re.X,
)


def parse_summary(markdown: str) -> dict | None:
    """Take the model's answer apart; None if it has none of the expected sections.

    Returns {"executive_summary": str, "key_points": [str], "action_items": [str] or None}.
    Action items are None when the answer has no such section at all, which is
    different from a section that says there are none.
    """
    markdown = re.sub(r"<think>.*?</think>", "", markdown, flags=re.S)
    names = {"executive summary": "executive_summary", "key points": "key_points", "action items": "action_items"}
    sections: dict[str, list[str]] = {}
    current = None
    for line in markdown.splitlines():
        line = line.strip()
        heading = re.sub(r"[^a-z ]", "", line.lower()).strip()
        if line.startswith("#") or heading in names:
            current = names.get(heading)
            if current:
                sections.setdefault(current, [])
        elif current and line:
            item = _tidy_item(line)
            if item and item.lower().rstrip(".") != "none":
                sections[current].append(item)
    if not sections:
        return None
    return {
        "executive_summary": " ".join(sections.get("executive_summary", [])),  # prose, not a list
        "key_points": sections.get("key_points", []),
        "action_items": sections.get("action_items"),
    }


def _tidy_item(line: str) -> str:
    """One line of the model's answer with its formatting quirks removed."""
    item = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", line)  # the bullet marker, which it sometimes forgets
    item = re.sub(r"^(?:OWNER|TASK):\s*", "", item)  # the format line taken literally
    item = re.sub(r"\*\*(.+?)\*\*", r"\1", item)  # bold
    item = _NO_DEADLINE.sub("", item)
    return re.sub(r"\s+([.,;])", r"\1", item).strip()


def unassign_unknown_speaker(item: str) -> str:
    """Without diarization nobody is "Speaker 2"; the model sometimes says so anyway."""
    return re.sub(r"^(?:the )?speaker(?: \d+)?\s*:", "Unassigned:", item, flags=re.I)


def summarize(ask_many, blocks: list[str], diarized: bool, progress=lambda fraction: None) -> tuple[dict | None, list[str]]:
    """Summarize a transcript given as speaker turns.

    `ask_many` takes a list of (system, user, max_tokens) requests and returns
    the model's answers (see llm.Answer). `diarized` says whether the blocks
    start with speaker labels.

    Returns (summary, problems). The summary is {"executive_summary": str,
    "key_points": [str], "action_items": [str]}, or None if nothing usable
    came back. `problems` are things the reader should be told: an answer
    that was cut short and may be missing items, or a step the language model
    failed at. A failed step costs only that step; whatever else was written
    is kept.
    """

    def ask(system: str, user: str, max_tokens: int):
        return ask_many([(system, user, max_tokens)])[0]

    parts = split_for_summary(blocks)
    key_points, action_items, problems = [], [], []
    for number, part in enumerate(parts, 1):
        where = f" for part {number} of {len(parts)} of the recording" if len(parts) > 1 else ""
        try:
            answer = ask(summary_prompt(diarized), f"<transcript>\n{part}\n</transcript>", 2000)
        except LanguageModelError as exc:
            # Carry on: the other parts are still worth having.
            problems.append(f"The summary{where} is missing. {exc}")
            continue
        finally:
            progress(number / (len(parts) + 1))  # the last step is the executive summary
        found = parse_summary(answer.text)
        if found is None:
            problems.append(f"The summary{where} could not be read from the language model's answer.")
            continue
        if answer.cut_off or found["action_items"] is None:
            problems.append(f"The summary{where} was cut short, so key points or action items may be missing.")
        key_points += found["key_points"]
        action_items += [item for item in found["action_items"] or [] if item not in action_items]

    if not key_points and not action_items:
        return None, problems or ["The summary came back empty."]
    if not diarized:
        action_items = [unassign_unknown_speaker(item) for item in action_items]
    summary = {"executive_summary": "", "key_points": key_points, "action_items": action_items}

    # The executive summary is written last, from the lists above. Those are
    # finished, and worth having whatever happens to this final step.
    try:
        answer = ask(EXECUTIVE_PROMPT, f"<notes>\n{_notes(summary)}\n</notes>", 300)
    except LanguageModelError as exc:
        problems.append(f"The executive summary is missing. {exc}")
    else:
        summary["executive_summary"] = _prose(answer.text)
        if answer.cut_off:
            problems.append("The executive summary was cut short.")
    return summary, problems


def _notes(summary: dict) -> str:
    """The key points and action items, as the executive summary is written from them."""
    notes = "Key points:\n" + "\n".join(f"- {point}" for point in summary["key_points"])
    if summary["action_items"]:
        notes += "\n\nAction items:\n" + "\n".join(f"- {item}" for item in summary["action_items"])
    return notes


def _prose(answer: str) -> str:
    """The model's few sentences as one paragraph, without any headings, bullets or bold it added."""
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S)
    lines = [re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", line.strip()) for line in answer.splitlines()]
    return " ".join(line for line in lines if line and not line.startswith("#")).replace("**", "")
