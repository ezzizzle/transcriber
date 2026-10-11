"""The summary: an executive summary, key points and action items.

The model reads the whole transcript in one go when it fits, and in parts
that are then merged when it doesn't. Its answer is Markdown, which is taken
apart here into plain data.
"""

import re

from .prompts import EXECUTIVE_PROMPT, MERGE_PROMPT, summary_prompt

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
    the model's answers. `diarized` says whether the blocks start with speaker
    labels.

    Returns (summary, problems). The summary is {"executive_summary": str,
    "key_points": [str], "action_items": [str]}, or None if the model's answer
    could not be read. `problems` are things the reader should be told.
    """

    def ask(system: str, user: str, max_tokens: int) -> str:
        return ask_many([(system, user, max_tokens)])[0]

    parts = split_for_summary(blocks)
    steps = len(parts) + (len(parts) > 1) + 1  # each part, the merge if there is one, the executive summary
    answers = []
    for part in parts:
        answers.append(ask(summary_prompt(diarized), f"<transcript>\n{part}\n</transcript>", 2000))
        progress(len(answers) / steps)
    answer = answers[0]
    if len(answers) > 1:
        numbered = "\n\n".join(f"Part {i} of {len(answers)}:\n{a.strip()}" for i, a in enumerate(answers, 1))
        answer = ask(MERGE_PROMPT, f"<parts>\n{numbered}\n</parts>", 2500)

    summary = parse_summary(answer)
    if summary is None:
        return None, ["The summary could not be read from the language model's answer."]
    problems = []
    if summary["action_items"] is None:
        # Usually the answer was cut off before it got that far.
        problems.append("The summary may be incomplete: the language model's answer had no action items section.")
        summary["action_items"] = []
    if not diarized:
        summary["action_items"] = [unassign_unknown_speaker(item) for item in summary["action_items"]]
    if not summary["executive_summary"]:
        summary["executive_summary"] = executive_summary(ask, summary)
    return summary, problems


def executive_summary(ask, summary: dict) -> str:
    """A few sentences written from the finished key points and action items."""
    notes = "Key points:\n" + "\n".join(f"- {point}" for point in summary["key_points"])
    if summary["action_items"]:
        notes += "\n\nAction items:\n" + "\n".join(f"- {item}" for item in summary["action_items"])
    answer = ask(EXECUTIVE_PROMPT, f"<notes>\n{notes}\n</notes>", 300)
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S)
    lines = [re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", line.strip()) for line in answer.splitlines()]
    return " ".join(line for line in lines if line and not line.startswith("#")).replace("**", "")
