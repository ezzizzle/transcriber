"""Optional LLM passes over a finished transcript, run by a small local model.

Clean-up removes disfluencies and adds paragraphs. Small models drift on long
inputs, so the transcript is fed through in chunks of a few sentences, and any
output that looks like the model went off-script (far too short, far too long)
is discarded in favour of the original text.

Summarizing produces key points and action items. It reads the whole
transcript in one go when it fits, and in parts that are then merged when it
doesn't.
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

SUMMARY_PROMPT = """\
You write up recordings of meetings and conversations. You receive the transcript of \
the audio inside <transcript> tags. It was transcribed by machine, so expect filler \
words, repetition and the occasional misheard word.

Generate a dot point summary of what was said, then record any action items. Use only \
what is in the transcript: never guess or add names, numbers, dates or tasks.

Lay the answer out exactly like this, with every point on its own line starting with "- ":

## Key points
- (the dot point summary)

## Action items
- ITEM_FORMAT

For the summary:
- Cover every topic that was discussed, in order, with one point per thing worth \
knowing. Use as many points as the conversation needs and skip greetings and small talk.
- Be specific. Include the figures, names, dates, decisions and reasons that were \
given ("the login bug affects about 4% of Android sessions", not "a bug was discussed").
- When something was corrected or changed during the conversation, report the final version.

For the action items:
- List everything someone said they would do or was asked to do, one task per line.
- OWNER_RULE
- DEADLINE is when that task is due, in the speaker's own words ("today", "by \
Thursday"). Leave the brackets out if no deadline was said for that task.
- Things that were decided, dropped or only wished for are not action items.
- If there are no action items, write "- None"."""

# Who owns a task can only be read off speaker labels, which exist only when
# diarized. Without them the model guesses from names in passing ("Thanks, Tom")
# and gets it wrong, so it is told not to attribute anything at all.
_ITEM_FORMATS = {True: "OWNER: TASK (DEADLINE)", False: "TASK (DEADLINE)"}
_OWNER_RULES = {
    True: 'OWNER is the label of the speaker who will do the task, exactly as it appears at the '
    'start of their lines, followed by their name in brackets if the transcript makes it clear '
    '(such as "Speaker 2 (Priya)"). If a task was raised but nobody took it, OWNER is '
    '"Unassigned".',
    False: 'This transcript does not show who is speaking, so you cannot tell who said what. Do '
    'not say who will do a task, and in the summary do not attribute statements or plans to '
    'anyone by name: write what was said and what needs doing ("the pricing copy will be '
    'drafted by the end of the week").',
}


def summary_prompt(diarized: bool) -> str:
    return SUMMARY_PROMPT.replace("ITEM_FORMAT", _ITEM_FORMATS[diarized]).replace("OWNER_RULE", _OWNER_RULES[diarized])


MERGE_PROMPT = """\
You receive summaries of consecutive parts of one long recording, inside <parts> tags. \
Merge them into a single summary. Use only what the part summaries say.

Lay the answer out exactly like this, with every point on its own line starting with "- ":

## Key points
- (every key point from the parts, in order, keeping the figures, names and dates; \
merge points that say the same thing)

## Action items
- (every distinct action item from the parts, copied unchanged; drop exact \
duplicates; if there are none, write "- None")"""

# Asked separately, from the finished key points. Folding it into SUMMARY_PROMPT
# as a third section made the model misattribute action items it had been
# getting right, and write five lines where two were wanted.
EXECUTIVE_PROMPT = """\
You receive the key points and action items from a recording, inside <notes> tags. \
Write an executive summary of it for someone who will read nothing else.

Write between two to five sentences of plain prose, no more than 80 words in total: what the \
recording was about, then its most important outcomes. Use only what is in the notes. \
Output only the sentences: no heading, no dot points, no preamble."""

# Roughly 50 minutes of speech; comfortably inside the model's context window.
_SUMMARY_MAX_WORDS = 9000

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


def split_for_summary(blocks: list[str], max_words: int = _SUMMARY_MAX_WORDS) -> list[str]:
    """Join transcript blocks (speaker turns) into as few parts as fit the model."""
    pieces = []
    for block in blocks:  # one speaker talking for an hour is still a single block
        words = block.split()
        if len(words) <= max_words:
            pieces.append(block)
        else:
            pieces += [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]
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
    """Pull the model's three sections apart; None if it produced none of them.

    Returns {"executive_summary": str, "key_points": [str], "action_items": [str]}.
    """
    markdown = re.sub(r"<think>.*?</think>", "", markdown, flags=re.S)
    names = {"executive summary": "executive_summary", "key points": "key_points", "action items": "action_items"}
    sections = {"executive_summary": [], "key_points": [], "action_items": []}
    current = None
    for line in markdown.splitlines():
        line = line.strip()
        heading = re.sub(r"[^a-z ]", "", line.lower()).strip()
        if line.startswith("#") or heading in names:
            current = names.get(heading)
            continue
        if current and line:  # the model sometimes forgets the bullet marker
            item = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", line)
            item = re.sub(r"^(?:OWNER|TASK):\s*", "", item)  # the format line taken literally
            item = re.sub(r"\*\*(.+?)\*\*", r"\1", item)
            item = _NO_DEADLINE.sub("", item)
            item = re.sub(r"\s+([.,;])", r"\1", item).strip()
            if item and item.lower().rstrip(".") != "none":
                sections[current].append(item)
    if not any(sections.values()):
        return None
    return {**sections, "executive_summary": " ".join(sections["executive_summary"])}  # prose, not a list


def unassign_unknown_speaker(item: str) -> str:
    """Without diarization nobody is "Speaker 2"; the model sometimes says so anyway."""
    return re.sub(r"^(?:the )?speaker(?: \d+)?\s*:", "Unassigned:", item, flags=re.I)


class Editor:
    """The clean-up and summary passes, on top of any `ask_many(requests) -> list[str]`.

    A request is (system, user, max_tokens). Everything here is plain text
    handling, so it runs wherever the transcript is; only `ask_many` reaches
    the model (see LocalModel and llm_main).
    """

    def __init__(self, ask_many, batch_size: int = 4):
        self._ask_many = ask_many
        self._batch_size = max(1, batch_size)

    def _ask(self, system: str, user: str, max_tokens: int) -> str:
        return self._ask_many([(system, user, max_tokens)])[0]

    def clean_many(self, texts: list[str], progress=lambda fraction: None) -> list[str]:
        """Clean each passage independently, several per model call.

        Batching is only for speed: the model generates for a few passages at
        once, about twice as fast as one after another.
        """
        out = [strip_fillers(text) if len(text.split()) < _MIN_LLM_WORDS else None for text in texts]
        todo = [i for i, cleaned in enumerate(out) if cleaned is None]
        total = sum(len(texts[i].split()) for i in todo) or 1
        done = 0
        for at in range(0, len(todo), self._batch_size):
            batch = todo[at : at + self._batch_size]
            answers = self._ask_many([
                (SYSTEM_PROMPT, f"<transcript>\n{texts[i]}\n</transcript>", len(texts[i].split()) * 3 + 200)
                for i in batch
            ])  # fmt: skip
            for i, raw in zip(batch, answers):
                cleaned = normalize_paragraphs(raw)
                # The model lets some fillers through, and its rejected answers leave them all in.
                out[i] = strip_fillers(cleaned if plausible(texts[i], cleaned) else texts[i])
                done += len(texts[i].split())
            progress(done / total)
        return out

    def clean(self, text: str) -> str:
        return self.clean_many([text])[0]

    def summarize(self, blocks: list[str], diarized: bool, progress=lambda fraction: None) -> dict | None:
        """Key points and action items for a transcript given as speaker turns.

        `diarized` says whether the blocks start with speaker labels.

        Returns {"key_points": [...], "action_items": [...]}, or None if the
        model's answer could not be read.
        """
        system = summary_prompt(diarized)
        parts = split_for_summary(blocks)
        answers = []
        for i, part in enumerate(parts):
            answers.append(self._ask(system, f"<transcript>\n{part}\n</transcript>", 900))
            progress((i + 1) / (len(parts) + 1 + (len(parts) > 1)))
        if len(answers) > 1:
            numbered = "\n\n".join(f"Part {i} of {len(answers)}:\n{a.strip()}" for i, a in enumerate(answers, 1))
            answers = [self._ask(MERGE_PROMPT, f"<parts>\n{numbered}\n</parts>", 1200)]
        summary = parse_summary(answers[0])
        if summary is None:
            return None
        if not diarized:
            summary["action_items"] = [unassign_unknown_speaker(item) for item in summary["action_items"]]
        if not summary["executive_summary"]:
            summary["executive_summary"] = self._executive_summary(summary)
        return summary

    def _executive_summary(self, summary: dict) -> str:
        notes = "Key points:\n" + "\n".join(f"- {point}" for point in summary["key_points"])
        if summary["action_items"]:
            notes += "\n\nAction items:\n" + "\n".join(f"- {item}" for item in summary["action_items"])
        answer = self._ask(EXECUTIVE_PROMPT, f"<notes>\n{notes}\n</notes>", 200)
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S)
        lines = [re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", line.strip()) for line in answer.splitlines()]
        return " ".join(line for line in lines if line and not line.startswith("#")).replace("**", "")


class LocalModel:
    """The LLM itself. Must be created and used on a single thread (MLX streams are thread-local)."""

    def __init__(self, model_id: str):
        from mlx_lm import load

        self.model, self.tokenizer = load(model_id)

    def ask_many(self, requests: list[tuple[str, str, int]]) -> list[str]:
        """Answer each (system, user, max_tokens) request; several are generated as one batch."""
        import mlx.core as mx
        from mlx_lm import batch_generate, generate
        from mlx_lm.sample_utils import make_sampler

        prompts = []
        for system, user, _ in requests:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            try:
                # Reasoning models would otherwise spend the token budget thinking out loud.
                prompts.append(self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False))
            except TypeError:
                prompts.append(self.tokenizer.apply_chat_template(messages, add_generation_prompt=True))
        limits = [max_tokens for _, _, max_tokens in requests]
        sampler = make_sampler(temp=0.0)
        try:
            if len(prompts) == 1:
                return [generate(self.model, self.tokenizer, prompt=prompts[0], max_tokens=limits[0], sampler=sampler)]
            return batch_generate(self.model, self.tokenizer, prompts, max_tokens=limits, sampler=sampler).texts
        finally:
            if sum(len(prompt) for prompt in prompts) > 2000:  # long inputs leave gigabytes of buffers cached
                mx.clear_cache()

    def ask(self, system: str, user: str, max_tokens: int) -> str:
        return self.ask_many([(system, user, max_tokens)])[0]


def llm_main(conn, model_id: str):
    """Entry point of the one process that holds the language model.

    The model is several GB, so it is loaded once and shared by every worker
    rather than once per parallel job. Receives a list of (system, user,
    max_tokens) requests; replies ("ok", [text, ...]) or ("error", message).
    The first message sent is ("ready", error_or_None).
    """
    try:
        try:
            print(f"Loading language model {model_id} ...", flush=True)
            model = LocalModel(model_id)
            print("Language model ready.", flush=True)
        except Exception as exc:  # noqa: BLE001 - reported to the job that needed it
            conn.send(("ready", f"Could not load the language model {model_id}: {exc}"))
            return
        conn.send(("ready", None))
        while True:
            requests = conn.recv()
            try:
                conn.send(("ok", model.ask_many(requests)))
            except Exception as exc:  # noqa: BLE001 - one bad request must not kill the model
                conn.send(("error", f"{type(exc).__name__}: {exc}"))
    except (EOFError, KeyboardInterrupt):
        pass  # server is shutting down
