"""Everything that only handles text: building the transcript, clean-up, the summary, and the output formats."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from transcriber import cleanup, compare, formats, summary
from transcriber.audio import pause_before
from transcriber.jobs import Job, Options
from transcriber.llm import Answer, LanguageModelError
from transcriber.transcript import Sentence, assign_speakers, build_turns, to_result, tokens_to_words


def model(reply, cut_off: bool = False):
    """A stand-in for the language model: `reply(user_message)` is the text of each answer."""
    return lambda requests: [Answer(reply(user), cut_off) for _, user, _ in requests]


def passage(user_message: str) -> str:
    """The passage inside a clean-up request, which a stand-in model can hand straight back."""
    return user_message.split("\n")[1]


def sentence(start: float, end: float, text: str, speaker: str | None = None) -> Sentence:
    return Sentence(start, end, text, confidence=0.9, words=[{"word": text, "start": start, "end": end}], speaker=speaker)


class TranscriptTest(unittest.TestCase):
    def test_tokens_to_words(self):
        tokens = [SimpleNamespace(text=t, start=i, end=i + 1) for i, t in enumerate([" Hel", "lo", ",", " world"])]
        self.assertEqual(
            tokens_to_words(tokens),
            [{"word": "Hello,", "start": 0, "end": 3}, {"word": "world", "start": 3, "end": 4}],
        )

    def test_speakers_turns_and_result(self):
        sentences = [
            sentence(0.0, 4.0, "Hi."),
            sentence(4.0, 6.0, "How are you?"),
            sentence(6.2, 6.5, "Good."),  # falls in a gap: nearest speaker wins
            sentence(9.0, 12.0, "Great."),
        ]
        diarization = [
            {"start": 6.6, "end": 8.0, "speaker": "SPEAKER_07"},
            {"start": 0.0, "end": 6.0, "speaker": "SPEAKER_03"},
            {"start": 8.5, "end": 12.0, "speaker": "SPEAKER_03"},
        ]
        assign_speakers(sentences, diarization)
        self.assertEqual([s.speaker for s in sentences], ["Speaker 1", "Speaker 1", "Speaker 2", "Speaker 1"])

        turns = build_turns(sentences)
        self.assertEqual([(t.speaker, t.text) for t in turns],
                         [("Speaker 1", "Hi. How are you?"), ("Speaker 2", "Good."), ("Speaker 1", "Great.")])
        self.assertEqual((turns[0].start, turns[0].end), (0.0, 6.0))

        result = to_result(sentences, turns, cleaned=False, duration=12.0)
        self.assertEqual((result["duration"], result["cleaned"]), (12.0, False))
        self.assertEqual(result["speakers"], ["Speaker 1", "Speaker 2"])
        self.assertEqual(result["turns"][0], {"speaker": "Speaker 1", "start": 0.0, "end": 6.0, "text": "Hi. How are you?"})
        self.assertEqual(result["segments"][2],
                         {"id": 2, "start": 6.2, "end": 6.5, "text": "Good.", "speaker": "Speaker 2", "confidence": 0.9})
        self.assertEqual(result["words"][2], {"word": "Good.", "start": 6.2, "end": 6.5, "speaker": "Speaker 2"})

    def test_without_diarization_everyone_is_one_speaker(self):
        sentences = [sentence(0.0, 1.0, "Hi.")]
        assign_speakers(sentences, [])
        self.assertEqual(sentences[0].speaker, "Speaker 1")


class TimingsTest(unittest.TestCase):
    def test_job_timings(self):
        job = Job(filename="a.mp3", path=Path("a.mp3"), options=Options(cleanup=True, summary=True))
        self.assertIsNone(job.timings())
        job.created, job.started = 100.0, 103.5
        job.llm_load_seconds = 8.0
        job.result = {"timings": {"summary": 4.0, "cleanup": 20.0, "transcribe": 3.0, "convert": 0.4}}
        self.assertEqual(
            list(job.timings().items()),  # in the order the stages ran, with the model load split out of clean-up
            [("queue", 3.5), ("convert", 0.4), ("transcribe", 3.0), ("llm_load", 8.0), ("cleanup", 12.0), ("summary", 4.0)],
        )


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


class CleanupTest(unittest.TestCase):
    def test_strip_fillers(self):
        self.assertEqual(cleanup.strip_fillers("Um, yeah, uh, sure."), "Yeah, sure.")
        self.assertEqual(cleanup.strip_fillers("Umm."), "")
        self.assertEqual(cleanup.strip_fillers("The summer hummed."), "The summer hummed.")
        self.assertEqual([w for w in "Um, uh-huh. Er hmm, the summer".split() if cleanup.is_filler(w)], ["Um,", "Er", "hmm,"])
        for before, after in [
            ("Anyway, um, the database migration: are we still saying March?", "Anyway, the database migration: are we still saying March?"),
            ("It started at ten past two, and, uh, nobody noticed.", "It started at ten past two, and nobody noticed."),
            ("Great. Okay, um, I think that's it. Thanks.", "Great. Okay, I think that's it. Thanks."),
            ("We fixed it. Um, the other thing is cost.", "We fixed it. The other thing is cost."),
            ("Right. Um. Okay, next.", "Right. Okay, next."),
            ("I think it was, uh. Never mind.", "I think it was. Never mind."),
            ("Uh-huh, mm-hmm, that works.", "Uh-huh, mm-hmm, that works."),  # these mean "yes"
            ("One, um, two.\n\nUh, three.\n\nUm.", "One, two.\n\nThree."),  # paragraphs survive
            ("The number is, um, 42?", "The number is, 42?"),
        ]:
            self.assertEqual(cleanup.strip_fillers(before), after)

    def test_chunking_and_guard(self):
        self.assertEqual(cleanup.chunk_sentences(["a b c", "d e", "f"], max_words=4), ["a b c", "d e f"])
        original = "word " * 100
        self.assertTrue(cleanup.plausible(original, "word " * 80))
        self.assertFalse(cleanup.plausible(original, "A short summary."))
        self.assertFalse(cleanup.plausible(original, "word " * 200))
        self.assertEqual(cleanup.normalize_paragraphs("<think>x</think>One.\nTwo.\n\n\nThree."), "One.\n\nTwo.\n\nThree.")


    def test_cut_at_longest_pause(self):
        sentences = ["a b", "c d", "e f", "g h", "i j", "k l"]
        # No pauses known: chunks are simply filled.
        self.assertEqual(cleanup.chunk_sentences(sentences, max_words=8), ["a b c d e f g h", "i j k l"])
        # A long pause before "e f": cut there, even though more would fit.
        pauses = [0, 0.3, 1.2, 0.3, 0.3, 0.3]
        self.assertEqual(cleanup.chunk_sentences(sentences, pauses, max_words=8), ["a b c d", "e f g h i j k l"])
        # A pause in the first half of the chunk is ignored, so chunks don't become tiny.
        pauses = [0, 2.0, 0.3, 0.3, 0.3, 0.3]
        self.assertEqual(cleanup.chunk_sentences(sentences, pauses, max_words=8), ["a b c d e f g h", "i j k l"])
        # One sentence longer than the limit still gets through, alone.
        self.assertEqual(cleanup.chunk_sentences(["a b c d e", "f"], max_words=3), ["a b c d e", "f"])

    def test_pause_before(self):
        silences = [(4.6, 4.85), (12.2, 13.05), (25.8, 26.6)]
        self.assertAlmostEqual(pause_before(silences, 12.88), 0.85)  # sentence start falls inside the pause
        self.assertAlmostEqual(pause_before(silences, 4.8), 0.25)
        self.assertEqual(pause_before(silences, 20.0), 0.0)
        self.assertEqual(pause_before([], 5.0), 0.0)

    def test_clean_many_batches_and_guards(self):
        calls = []

        def ask_many(requests):
            calls.append(len(requests))
            # Hand each passage back, except for one answer that goes off the rails.
            return [Answer("Sure!" if "bad" in user else passage(user)) for _, user, _ in requests]

        long = lambda word: " ".join([word] * 20)
        texts = [long("one"), "Um, yeah.", long("two"), long("bad"), long("three"), long("four")]
        out = cleanup.clean_passages(ask_many, texts, batch_size=2)
        self.assertEqual(calls, [2, 2, 1])  # the short passage never reaches the model
        self.assertEqual(out, [long("one"), "Yeah.", long("two"), long("bad"), long("three"), long("four")])

    def test_fillers_the_model_left_are_swept(self):
        left_in = "So the queue backed up, and, uh, nobody noticed until support got a ticket about it that afternoon."
        rejected = "Um, " + " ".join(["word"] * 20)
        ask_many = model(lambda user: left_in if "queue" in user else "No.")
        out = cleanup.clean_passages(ask_many, ["So, um, the queue backed up, and, uh, nobody noticed until support got a ticket about it that afternoon.", rejected])
        self.assertEqual(out[0], "So the queue backed up, and nobody noticed until support got a ticket about it that afternoon.")
        self.assertEqual(out[1], " ".join(["Word"] + ["word"] * 19))  # the model's answer was rejected; the original is still swept

    def test_an_answer_that_ran_to_the_limit_is_not_trusted(self):
        original = "Um, " + " ".join(["word"] * 20)
        rewritten = " ".join(["other"] * 20)
        self.assertEqual(cleanup.clean_passages(model(lambda user: rewritten), [original]), [rewritten])
        self.assertEqual(cleanup.clean_passages(model(lambda user: rewritten, cut_off=True), [original]),
                         [" ".join(["Word"] + ["word"] * 19)])


    def test_clean_turns(self):
        long = " ".join(["word"] * 20)
        turns = build_turns([
            sentence(0, 5, f"Um, {long}.", "Speaker 1"),
            sentence(5, 6, "Um.", "Speaker 2"),  # nothing left once the filler is gone
            sentence(6, 9, "Okay, uh, thanks.", "Speaker 1"),
        ])
        cleaned = cleanup.clean_turns(model(passage), turns, silences=[])
        # Speaker 2's turn is dropped, which leaves Speaker 1's two turns side by side: they become one.
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0].speaker, "Speaker 1")
        self.assertEqual(cleaned[0].text, f"Word {long[5:]}.\n\nOkay, thanks.")
        self.assertEqual((cleaned[0].start, cleaned[0].end), (0, 9))
        self.assertEqual(turns[0].text, f"Um, {long}.")  # the originals are left alone

        # The result carries both versions of every turn, so they can be compared.
        result = to_result([s for turn in turns for s in turn.sentences], cleaned, cleaned=True)
        self.assertEqual(result["turns"][0]["text"], f"Word {long[5:]}.\n\nOkay, thanks.")
        self.assertEqual(result["turns"][0]["raw_text"], f"Um, {long}. Okay, uh, thanks.")
        # The whole transcript as transcribed still has the turn that clean-up dropped.
        self.assertEqual(formats.raw_text({**result, "diarized": True}),
                         f"Speaker 1: Um, {long}.\n\nSpeaker 2: Um.\n\nSpeaker 1: Okay, uh, thanks.")
        self.assertEqual(formats.raw_text({**result, "diarized": False}), f"Um, {long}.\n\nUm.\n\nOkay, uh, thanks.")


class CompareTest(unittest.TestCase):
    def test_what_clean_up_changed(self):
        raw = "um so we we shipped it on the fourteenth. it does not work on android"
        cleaned = "So, we shipped it on the 14th.\n\nIt does work on Android."
        self.assertEqual(compare.changes(raw, cleaned), [
            ["filler", "um"],
            ["same", "So,"],  # capitals and punctuation are not differences
            ["removed", "we"],  # said twice
            ["same", "we shipped it on the"],
            ["removed", "fourteenth."],
            ["added", "14th."],  # a number written differently: worth a glance
            ["break", ""],
            ["same", "It does"],
            ["removed", "not"],  # and this is the kind of thing the comparison is for
            ["same", "work on Android."],
        ])  # fmt: skip

    def test_changes_as_markdown(self):
        raw = "um so we we shipped it on the fourteenth. it does not work on android"
        cleaned = "So, we shipped it on the 14th.\n\nIt does work on Android."
        self.assertEqual(
            compare.as_markdown(raw, cleaned),
            "~~um~~ So, ~~we~~ we shipped it on the ~~fourteenth.~~ **14th.**\n\nIt does ~~not~~ work on Android.",
        )
        self.assertEqual(compare.as_markdown("Thanks, all.", "Thanks, all."), "Thanks, all.")
        # A star that was really in the text is not mistaken for a mark.
        self.assertEqual(compare.as_markdown("um press * twice", "Press * twice."), "~~um~~ Press \\* twice.")

    def test_the_json_download_carries_all_three_versions(self):
        turn = {"speaker": "Speaker 1", "start": 0.0, "end": 2.0, "text": "So, hello.", "raw_text": "um so hello"}
        segments = [  # as transcribed: Speaker 2 only said "um", so clean-up left no turn for them
            {"speaker": "Speaker 1", "text": "um so hello"},
            {"speaker": "Speaker 2", "text": "um"},
        ]
        result = {"diarized": True, "cleaned": True, "turns": [turn], "segments": segments}
        whole = json.loads(formats.to_full_json(result))
        self.assertEqual(whole["text"], "Speaker 1: So, hello.")
        self.assertEqual(whole["raw_text"], "Speaker 1: um so hello\n\nSpeaker 2: um")  # everything that was said
        self.assertEqual(whole["changes"], "Speaker 1: ~~um~~ So, hello.")
        self.assertEqual(whole["turns"][0]["changes"], "~~um~~ So, hello.")
        self.assertNotIn("changes", turn)  # worked out for the download, not stored with the job

        plain = json.loads(formats.to_full_json({"diarized": False, "cleaned": False, "turns": [{"speaker": None, "text": "um so hello"}]}))
        self.assertEqual(sorted(plain), ["cleaned", "diarized", "text", "turns"])

    def test_a_changed_number_is_never_hidden(self):
        # Punctuation inside a number is part of the number.
        for before, after in [("1.5%", "15%"), ("-5", "5"), ("1/2", "12"), (".5", "5"), ("$5", "5"), ("3:30", "330")]:
            self.assertEqual(
                compare.changes(f"it was {before} overall", f"It was {after} overall."),
                [["same", "It was"], ["removed", before], ["added", after], ["same", "overall."]],
            )
        # Punctuation around one is not: the end of a sentence, a quotation, brackets.
        for before, after in [("14th.", "14th"), ("42,000,", "42,000"), ("(2025)", "2025"), ("1.5%.", '"1.5%"')]:
            self.assertEqual(compare.changes(f"on the {before} then", f"On the {after} then"), [["same", f"On the {after} then"]])

    def test_nothing_changed(self):
        self.assertEqual(compare.changes("Thanks, all.", "Thanks, all."), [["same", "Thanks, all."]])
        self.assertEqual(compare.changes("okay — right", "Okay, right."), [["same", "Okay, right."]])  # a lone dash is not a word


class SummaryTest(unittest.TestCase):
    def test_parse_summary(self):
        raw = "<think>hm</think>## Key points\n- **Release** moved to March.\n* Budget is over.\n\n**Action items**\n1. Speaker 2: tell the team (no deadline specified)\nSpeaker 3: send it (Thursday)\n- None\n"
        self.assertEqual(
            summary.parse_summary(raw),
            {"executive_summary": "", "key_points": ["Release moved to March.", "Budget is over."], "action_items": ["Speaker 2: tell the team", "Speaker 3: send it (Thursday)"]},
        )
        self.assertEqual(summary.parse_summary("## Key points\n- A\n## Action items\n- None"),
                         {"executive_summary": "", "key_points": ["A"], "action_items": []})
        three = "## Executive Summary\nThe team met.\nThe release moved.\n\n## Key points\n- A\n\n## Action items\n- None"
        self.assertEqual(summary.parse_summary(three),
                         {"executive_summary": "The team met. The release moved.", "key_points": ["A"], "action_items": []})
        self.assertIsNone(summary.parse_summary("Sure! Here is a summary of the meeting."))
        self.assertIsNone(summary.parse_summary("## Key points\n- A")["action_items"])  # no such section

    def test_summarize(self):
        answer = "## Key points\n- A\n## Action items\n- Speaker: email legal (today)\n- Priya: fix the bug (by Wednesday)"
        ask_many = model(lambda user: "They met. Things happened." if "<notes>" in user else answer)

        result, problems = summary.summarize(ask_many, ["x"], diarized=True)
        self.assertEqual(problems, [])
        self.assertEqual(result["action_items"], ["Speaker: email legal (today)", "Priya: fix the bug (by Wednesday)"])
        self.assertEqual(result["executive_summary"], "They met. Things happened.")  # asked for separately

        # Without speaker labels nobody is "Speaker"; the model sometimes says so anyway.
        result, _ = summary.summarize(ask_many, ["x"], diarized=False)
        self.assertEqual(result["action_items"], ["Unassigned: email legal (today)", "Priya: fix the bug (by Wednesday)"])

    def test_a_long_recording_is_summarized_in_parts(self):
        def reply(user):
            if "<notes>" in user:
                return "A long meeting."
            if "first" in user:
                return "## Key points\n- Opening\n## Action items\n- Speaker 1: book the room\n- Speaker 2: send the notes"
            return "## Key points\n- Closing\n## Action items\n- Speaker 2: send the notes\n- Speaker 1: pay the invoice"

        blocks = ["Speaker 1: first " + "word " * 6000, "Speaker 2: second " + "word " * 6000]
        result, problems = summary.summarize(model(reply), blocks, diarized=True)
        self.assertEqual(problems, [])
        self.assertEqual(result["key_points"], ["Opening", "Closing"])  # every part's points, in order
        self.assertEqual(result["action_items"],  # and its tasks, without the one that came up twice
                         ["Speaker 1: book the room", "Speaker 2: send the notes", "Speaker 1: pay the invoice"])
        self.assertEqual(result["executive_summary"], "A long meeting.")

    def test_an_answer_cut_short_is_reported(self):
        complete = "## Key points\n- A\n## Action items\n- Speaker 1: one"
        reply = lambda text: (lambda user: "Fine." if "<notes>" in user else text)

        # The model ran out of room. What it wrote is kept, and the reader is told it may not be all.
        ran_out = lambda requests: [Answer(reply(complete)(user), cut_off="<notes>" not in user) for _, user, _ in requests]
        result, problems = summary.summarize(ran_out, ["x"], diarized=True)
        self.assertEqual(result["action_items"], ["Speaker 1: one"])
        self.assertEqual(problems, ["The summary was cut short, so key points or action items may be missing."])

        # An answer with no action items section at all never got that far.
        result, problems = summary.summarize(model(reply("## Key points\n- A\n- B")), ["x"], diarized=True)
        self.assertEqual((result["key_points"], result["action_items"]), (["A", "B"], []))
        self.assertEqual(len(problems), 1)

        # Saying there are none is different, and nothing to report.
        none = model(reply("## Key points\n- A\n## Action items\n- None"))
        self.assertEqual(summary.summarize(none, ["x"], diarized=True)[1], [])

    def test_a_part_cut_short_is_reported_even_when_the_others_are_fine(self):
        def ask_many(requests):
            (_, user, _), = requests
            if "<notes>" in user:
                return [Answer("Fine.")]
            return [Answer("## Key points\n- A\n## Action items\n- Speaker 1: one", cut_off="second" in user)]

        blocks = ["Speaker 1: first " + "word " * 6000, "Speaker 2: second " + "word " * 6000]
        result, problems = summary.summarize(ask_many, blocks, diarized=True)
        self.assertEqual(problems, ["The summary for part 2 of 2 of the recording was cut short, so key points or action items may be missing."])
        self.assertEqual(result["key_points"], ["A", "A"])

    def test_trouble_with_the_executive_summary_costs_only_the_executive_summary(self):
        lists = "## Key points\n- A\n## Action items\n- Speaker 1: one"

        # Cut short: shown as far as it got, with a note.
        cut_short = lambda requests: [Answer("They met and", cut_off=True) if "<notes>" in user else Answer(lists) for _, user, _ in requests]
        result, problems = summary.summarize(cut_short, ["x"], diarized=True)
        self.assertEqual(result["executive_summary"], "They met and")
        self.assertEqual(problems, ["The executive summary was cut short."])

        # The model fails at this last step: the lists it had already written are kept.
        def fails_last(requests):
            if "<notes>" in requests[0][1]:
                raise LanguageModelError("The language model process crashed.")
            return [Answer(lists)]

        result, problems = summary.summarize(fails_last, ["x"], diarized=True)
        self.assertEqual((result["key_points"], result["action_items"]), (["A"], ["Speaker 1: one"]))
        self.assertEqual(result["executive_summary"], "")
        self.assertEqual(problems, ["The executive summary is missing. The language model process crashed."])

    def test_a_part_the_model_fails_on_costs_only_that_part(self):
        def ask_many(requests):
            (_, user, _), = requests
            if "second" in user and "<notes>" not in user:
                raise LanguageModelError("The language model did not answer within 15 minutes.")
            return [Answer("Fine." if "<notes>" in user else "## Key points\n- Opening\n## Action items\n- Speaker 1: one")]

        blocks = ["Speaker 1: first " + "word " * 6000, "Speaker 2: second " + "word " * 6000]
        result, problems = summary.summarize(ask_many, blocks, diarized=True)
        self.assertEqual((result["key_points"], result["executive_summary"]), (["Opening"], "Fine."))
        self.assertEqual(problems, ["The summary for part 2 of 2 of the recording is missing. The language model did not answer within 15 minutes."])

        # With nothing else to fall back on, there is no summary, and the reason is passed on.
        def always_fails(requests):
            raise LanguageModelError("The language model process crashed.")

        self.assertEqual(summary.summarize(always_fails, ["x"], diarized=True),
                         (None, ["The summary is missing. The language model process crashed."]))

    def test_an_unreadable_answer_gives_no_summary(self):
        result, problems = summary.summarize(model(lambda user: "I'm sorry, I can't help with that."), ["x"], diarized=True)
        self.assertIsNone(result)
        self.assertEqual(len(problems), 1)

    def test_empty_deadlines_are_dropped(self):
        def items(*lines):
            return summary.parse_summary("## Action items\n" + "\n".join(f"- {line}" for line in lines))["action_items"]

        self.assertEqual(
            items(
                "Speaker 1: Email legal (No deadline specified)",
                "Speaker 2: Fix the bug (No deadline specified).",
                "Speaker 3 (Tom): Book the lab (not stated)",
                "Speaker 1: Renew the certificate (Deadline: none)",
                "Unassigned: Cover Friday (N/A)",
                "Speaker 2: Write it up (no deadline) and circulate it",
            ),
            [
                "Speaker 1: Email legal",
                "Speaker 2: Fix the bug.",
                "Speaker 3 (Tom): Book the lab",
                "Speaker 1: Renew the certificate",
                "Unassigned: Cover Friday",
                "Speaker 2: Write it up and circulate it",
            ],
        )
        kept = ["Speaker 1: Send it (by Thursday)", "Speaker 2: Ship it (not before Friday)",
                "Unassigned: Prepare the data (to be decided next week)", "Speaker 3 (Tom): Call them (today)"]
        self.assertEqual(items(*kept), kept)

    def test_split_for_summary(self):
        self.assertEqual(summary.split_for_summary(["a b", "c d", "e"], max_words=4), ["a b\n\nc d", "e"])
        self.assertEqual(summary.split_for_summary(["a b c d e"], max_words=2), ["a b", "c d", "e"])
        # Every piece of one long turn says who is speaking, not just the first.
        self.assertEqual(summary.split_for_summary(["Speaker 2: a b c d e"], max_words=2),
                         ["Speaker 2: a b", "Speaker 2: c d", "Speaker 2: e"])

    def test_rendering(self):
        result = {**FormatsTest.result, "summary": {"key_points": ["Said bye."], "action_items": []}}
        text = formats.to_text_with_summary(result)
        self.assertTrue(text.startswith("Speaker 1: Bye.\n\n"))
        self.assertIn("KEY POINTS\n- Said bye.\n\nACTION ITEMS\n- None", text)
        self.assertNotIn("EXECUTIVE", text)  # nothing to show when the model gave none
        self.assertIn("## Action Items\n\n- None", formats.to_markdown(result))
        result["summary"]["executive_summary"] = "They said goodbye."
        self.assertIn("EXECUTIVE SUMMARY\nThey said goodbye.\n\nKEY POINTS\n- Said bye.", formats.to_text_with_summary(result))
        self.assertIn("## Executive Summary\n\nThey said goodbye.\n\n## Key Points\n\n- Said bye.", formats.to_markdown(result))
        self.assertEqual(formats.to_text(result), "Speaker 1: Bye.")  # JSON `text` stays transcript-only
        self.assertEqual(formats.to_verbose_json(result)["summary"]["key_points"], ["Said bye."])
        self.assertEqual(formats.to_text_with_summary(FormatsTest.result), "Speaker 1: Bye.")
        self.assertNotIn("summary", formats.to_verbose_json(FormatsTest.result))

    def test_warnings_travel_with_the_transcript(self):
        result = {**FormatsTest.result, "warnings": ["Clean-up was skipped."]}
        self.assertTrue(formats.to_text_with_summary(result).endswith("(Note: Clean-up was skipped.)"))
        self.assertIn("*Note: Clean-up was skipped.*", formats.to_markdown(result))
        self.assertEqual(formats.to_verbose_json(result)["warnings"], ["Clean-up was skipped."])
        self.assertNotIn("warnings", formats.to_verbose_json(FormatsTest.result))



if __name__ == "__main__":
    unittest.main()
