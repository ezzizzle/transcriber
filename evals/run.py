"""Check what the language model does with the prompts, on transcripts with known answers.

    uv run python -m evals.run                  # the configured model
    uv run python -m evals.run some/other-model

Run this after changing a prompt or the model. It loads the real model, so it
takes a minute; it is not part of the unit tests. The checks are deliberately
few and concrete. Read the printed output as well: a summary can pass every
check and still be poor.

meeting.txt is a made-up three-person meeting with eleven things people agreed
to do, and three traps: a project that is dropped, a question nobody answers,
and a task one person offers to do and then hands to someone else.
"""

import re
import sys
from pathlib import Path

from transcriber import cleanup, summary
from transcriber.config import Config
from transcriber.llm import LocalModel

HERE = Path(__file__).parent

# (who should own it, a word the task must contain)
EXPECTED_ACTIONS = [
    ("Speaker 2", "login"), ("Speaker 2", "crash"), ("Speaker 3", "pricing"), ("Speaker 3", "lab"),
    ("Speaker 1", "legal"), ("Speaker 2", "post-mortem"), ("Speaker 2", "alert"), ("Speaker 3", "finance"),
    ("Speaker 1", "certificate"), ("Speaker 3", "interview"), ("Speaker 1", "Friday"),
]  # fmt: skip
NOT_ACTIONS = ["dashboard", "contractor", "document the deploy"]  # dropped, undecided, only wished for

failures = 0


def check(description: str, ok: bool, detail: str = ""):
    global failures
    failures += not ok
    print(f"  {'pass' if ok else 'FAIL'}  {description}{'  ' + detail if detail else ''}")


def check_summary(ask_many, blocks: list[str]):
    print("\nSummary, with speakers")
    result, problems = summary.summarize(ask_many, blocks, diarized=True)
    for item in result["action_items"]:
        print(f"      - {item}")
    found = [(owner, word) for owner, word in EXPECTED_ACTIONS
             if any(item.startswith(owner) and word.lower() in item.lower() for item in result["action_items"])]
    missing = [pair for pair in EXPECTED_ACTIONS if pair not in found]
    check(f"action items with the right owner: {len(found)} of {len(EXPECTED_ACTIONS)}", len(found) >= 9, f"missing {missing}" if missing else "")
    wrong = [item for item in result["action_items"] if any(word in item.lower() for word in NOT_ACTIONS)]
    check("nothing listed that nobody agreed to do", not wrong, str(wrong) if wrong else "")
    check("no empty deadline notes", not any(re.search(r"\((?:no|not|none)\b", item, re.I) for item in result["action_items"]))
    check("key points mention the corrected date (April), not only March", any("April" in point for point in result["key_points"]))
    words = len(result["executive_summary"].split())
    print(f"      {result['executive_summary']}")
    check(f"executive summary is short: {words} words", 15 <= words <= 100)
    check("no problems reported", not problems, str(problems) if problems else "")

    print("\nSummary, without speakers")
    plain = [re.sub(r"^Speaker \d+: ", "", block) for block in blocks]
    result, _ = summary.summarize(ask_many, plain, diarized=False)
    for item in result["action_items"]:
        print(f"      - {item}")
    owned = [item for item in result["action_items"] if re.match(r"(?:Speaker|Priya|Tom|Dana)\b[^:]{0,20}:", item)]
    check("no owners are guessed", not owned, str(owned) if owned else "")
    check(f"most tasks still found: {len(result['action_items'])}", len(result["action_items"]) >= 8)


def check_cleanup(ask_many, passages: list[str]):
    print("\nClean-up")
    cleaned = cleanup.clean_passages(ask_many, passages)
    fillers = re.compile(r"\b(?:um+|uh+|erm*)\b", re.I)
    left = sum(len(fillers.findall(text)) for text in cleaned)
    check(f"filler words left: {left} of {sum(len(fillers.findall(text)) for text in passages)}", left == 0)
    before, after = sum(len(t.split()) for t in passages), sum(len(t.split()) for t in cleaned)
    check(f"length is about the same: {before} words in, {after} out", 0.75 * before <= after <= before)
    whole = cleanup.clean_passages(ask_many, [" ".join(passages)])[0]
    check(f"a long passage is broken into paragraphs: {whole.count(chr(10) * 2) + 1}", "\n\n" in whole)
    print(f"      {cleaned[0]}")


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else Config().llm_model
    print(f"Model: {model_id}")
    model = LocalModel(model_id)
    check_summary(model.ask_many, (HERE / "meeting.txt").read_text().strip().split("\n\n"))
    check_cleanup(model.ask_many, (HERE / "fillers.txt").read_text().strip().split("\n\n"))
    print(f"\n{failures} check(s) failed." if failures else "\nAll checks passed.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
