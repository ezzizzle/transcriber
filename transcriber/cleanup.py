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
You summarize transcripts of meetings and conversations. You receive a transcript \
inside <transcript> tags. Use only what is stated in it: never invent names, dates, \
tasks or decisions.

Output exactly these two Markdown sections and nothing else. Every item is a line \
starting with "- ".

## Key points
- 3 to 8 bullets covering the main topics, facts and decisions, in the order discussed.

## Action items
- One bullet per task that someone said they will do ("I will...", "I can...") or \
was asked to do, in the form: OWNER: TASK (DEADLINE)

Rules for action items:
- OWNER_RULE
- DEADLINE is when that particular task is due, in the words the speaker used for it \
("today", "by Thursday", "the 20th"). Do not add a month, year or weekday that was \
not spoken, and do not borrow a deadline from a different task. Leave the \
parentheses out when no deadline was given.
- A decision is a key point, not an action item, unless someone has to do something.
- If there are no action items, write "- None"."""

# Who owns a task can only be read off speaker labels, which exist only when diarized.
_OWNER_RULES = {
    True: 'OWNER is the label of the speaker who will do the task, exactly as it appears at the '
    'start of their lines (such as "Speaker 1"). If a task was raised but nobody took it, '
    'OWNER is "Unassigned".',
    False: 'This transcript does not say who is speaking. OWNER is a person\'s name only if that '
    'name is spoken in the transcript as the one doing the task; otherwise OWNER is '
    '"Unassigned". Never write "Speaker" or a job title as OWNER.',
}

MERGE_PROMPT = """\
You receive summaries of consecutive parts of one long transcript, inside <parts> tags. \
Merge them into a single summary. Use only what the part summaries say.

Output exactly these two Markdown sections and nothing else:

## Key points
- 3 to 10 bullets covering the most important topics, facts and decisions, in order.

## Action items
- Every distinct action item from the parts, copied unchanged, each on a line \
starting with "- ". Drop exact duplicates. If there are none, write "- None"."""

# Roughly 50 minutes of speech; comfortably inside the model's context window.
_SUMMARY_MAX_WORDS = 9000

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


def parse_summary(markdown: str) -> dict | None:
    """Pull the bullets out of the model's two sections; None if it produced neither."""
    markdown = re.sub(r"<think>.*?</think>", "", markdown, flags=re.S)
    sections = {"key_points": [], "action_items": []}
    current = None
    for line in markdown.splitlines():
        line = line.strip()
        heading = re.sub(r"[^a-z ]", "", line.lower()).strip()
        if line.startswith("#") or heading in {"key points", "action items"}:
            current = {"key points": "key_points", "action items": "action_items"}.get(heading)
            continue
        if current and line:  # the model sometimes forgets the bullet marker
            item = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", line)
            item = re.sub(r"\*\*(.+?)\*\*", r"\1", item)
            item = re.sub(r"\s*\((?:no|none|not)\b[^)]*\)\s*$", "", item, flags=re.I).strip()  # "(no deadline given)"
            if item and item.lower().rstrip(".") != "none":
                sections[current].append(item)
    return sections if sections["key_points"] or sections["action_items"] else None


def unassign_unknown_speaker(item: str) -> str:
    """Without diarization nobody is "Speaker 2"; the model sometimes says so anyway."""
    return re.sub(r"^(?:the )?speaker(?: \d+)?\s*:", "Unassigned:", item, flags=re.I)


class Editor:
    """The clean-up and summary passes, on top of any `ask(system, user, max_tokens) -> str`.

    Everything here is plain text handling, so it runs wherever the transcript
    is; only `ask` reaches the model (see LocalModel and llm_main).
    """

    def __init__(self, ask):
        self._ask = ask

    def clean(self, text: str) -> str:
        if len(text.split()) < _MIN_LLM_WORDS:
            return strip_fillers(text)

        raw = self._ask(SYSTEM_PROMPT, f"<transcript>\n{text}\n</transcript>", len(text.split()) * 3 + 200)
        cleaned = normalize_paragraphs(raw)
        return cleaned if plausible(text, cleaned) else text

    def summarize(self, blocks: list[str], diarized: bool, progress=lambda fraction: None) -> dict | None:
        """Key points and action items for a transcript given as speaker turns.

        `diarized` says whether the blocks start with speaker labels.

        Returns {"key_points": [...], "action_items": [...]}, or None if the
        model's answer could not be read.
        """
        system = SUMMARY_PROMPT.replace("OWNER_RULE", _OWNER_RULES[diarized])
        parts = split_for_summary(blocks)
        answers = []
        for i, part in enumerate(parts):
            answers.append(self._ask(system, f"<transcript>\n{part}\n</transcript>", 900))
            progress((i + 1) / (len(parts) + (len(parts) > 1)))
        if len(answers) > 1:
            numbered = "\n\n".join(f"Part {i} of {len(answers)}:\n{a.strip()}" for i, a in enumerate(answers, 1))
            answers = [self._ask(MERGE_PROMPT, f"<parts>\n{numbered}\n</parts>", 1200)]
        summary = parse_summary(answers[0])
        if summary and not diarized:
            summary["action_items"] = [unassign_unknown_speaker(item) for item in summary["action_items"]]
        return summary


class LocalModel:
    """The LLM itself. Must be created and used on a single thread (MLX streams are thread-local)."""

    def __init__(self, model_id: str):
        from mlx_lm import load

        self.model, self.tokenizer = load(model_id)

    def ask(self, system: str, user: str, max_tokens: int) -> str:
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            # Reasoning models would otherwise spend the token budget thinking out loud.
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        try:
            return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=make_sampler(temp=0.0))
        finally:
            if len(prompt) > 2000:  # a long transcript leaves gigabytes of buffers cached
                mx.clear_cache()


def llm_main(conn, model_id: str):
    """Entry point of the one process that holds the language model.

    The model is several GB, so it is loaded once and shared by every worker
    rather than once per parallel job. Receives (system, user, max_tokens);
    replies ("ok", text) or ("error", message). The first message sent is
    ("ready", error_or_None).
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
            request = conn.recv()
            try:
                conn.send(("ok", model.ask(*request)))
            except Exception as exc:  # noqa: BLE001 - one bad request must not kill the model
                conn.send(("error", f"{type(exc).__name__}: {exc}"))
    except (EOFError, KeyboardInterrupt):
        pass  # server is shutting down
