"""The job queue and worker handling, driven with the fake processes in fakes.py."""

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from tests import fakes
from transcriber import engine as engine_module
from transcriber.config import Config
from transcriber.engine import Engine, QueueFull, _private_dir
from transcriber.jobs import Job, Options


class EngineTest(unittest.TestCase):
    def engine(self, worker=fakes.worker, llm=fakes.llm, **settings) -> Engine:
        work_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        settings = {"max_parallel": 1, "stall_seconds": 30, **settings}
        engine = Engine(Config(work_dir=work_dir, **settings), worker_main=worker, llm_main=llm)
        engine.start()
        self.assertTrue(engine.ready.wait(20))
        return engine

    def submit(self, engine: Engine, recording: str, owner: str | None = "alice", **options) -> Job:
        path = engine.files_dir / f"upload-{time.monotonic_ns()}"
        path.write_text(recording)
        return engine.submit("talk.mp3", path, Options(**options), owner)

    def finish(self, job: Job) -> Job:
        self.assertTrue(job.done.wait(20), "job did not finish")
        return job

    def test_a_job_runs_and_its_files_are_removed(self):
        engine = self.engine()
        job = self.finish(self.submit(engine, "hello there"))
        self.assertEqual((job.status, job.result["turns"][0]["text"]), ("done", "hello there"))
        self.assertEqual(list(engine.files_dir.iterdir()), [])  # neither the upload nor the decoded audio
        self.assertEqual(job.timings()["transcribe"], 0.01)

    def test_jobs_wait_their_turn_and_the_queue_has_a_limit(self):
        engine = self.engine(max_queue=2)
        first = self.submit(engine, "sleep 1 one")
        while first.status == "queued":
            time.sleep(0.01)
        second, third = self.submit(engine, "two"), self.submit(engine, "three")
        self.assertEqual(engine.job_status(first)["queue_position"], None)
        self.assertEqual(engine.job_status(second)["queue_position"], 1)
        self.assertEqual(engine.job_status(third)["queue_position"], 2)
        self.assertTrue(engine.queue_full())
        with self.assertRaises(QueueFull):
            self.submit(engine, "four")
        self.assertEqual([self.finish(j).status for j in (first, second, third)], ["done"] * 3)
        self.assertLess(second.started, third.started)  # first come, first served

    def test_two_jobs_run_at_once_when_allowed(self):
        engine = self.engine(max_parallel=2)
        a, b = self.submit(engine, "sleep 1 a"), self.submit(engine, "sleep 1 b")
        self.finish(a), self.finish(b)
        self.assertLess(abs(a.started - b.started), 0.9)

    def test_a_bad_file_is_the_users_error(self):
        engine = self.engine()
        job = self.finish(self.submit(engine, "bad"))
        self.assertEqual((job.status, job.user_error), ("error", True))
        self.assertIn("could not read audio", job.error)

    def test_a_crashed_worker_fails_its_job_and_is_replaced(self):
        engine = self.engine()
        crashed = self.finish(self.submit(engine, "crash"))
        self.assertEqual((crashed.status, crashed.user_error), ("error", False))
        self.assertIn("crashed", crashed.error)
        self.assertEqual(self.finish(self.submit(engine, "still working")).status, "done")
        self.assertEqual(list(engine.files_dir.iterdir()), [])

    def test_a_stuck_worker_is_killed_and_replaced(self):
        engine = self.engine(stall_seconds=1)
        stuck = self.finish(self.submit(engine, "hang"))
        self.assertIn("stopped responding", stuck.error)
        self.assertEqual(self.finish(self.submit(engine, "still working")).status, "done")

    def test_a_job_the_server_cannot_start_fails_without_taking_the_slot_with_it(self):
        engine = self.engine()
        self.finish(self.submit(engine, "crash"))  # so that the next job needs a new worker

        start_worker, failures = engine_module.Worker, [OSError("cannot start a process")]

        def flaky(*args, **kwargs):
            if failures:
                raise failures.pop()
            return start_worker(*args, **kwargs)

        with mock.patch.object(engine_module, "Worker", flaky), mock.patch("traceback.print_exc"):
            failed = self.finish(self.submit(engine, "unlucky"))
            self.assertEqual(failed.status, "error")
            self.assertIn("cannot start a process", failed.error)
            self.assertEqual(self.finish(self.submit(engine, "still working")).status, "done")
        self.assertEqual(engine.stats()["running"], 0)

    def test_a_file_that_cannot_be_deleted_does_not_stop_the_job_finishing(self):
        engine = self.engine()
        with mock.patch.object(engine_module.Path, "unlink", side_effect=PermissionError("locked")), \
                mock.patch.object(engine_module.sys, "stderr"):
            job = self.finish(self.submit(engine, "hello there"))
        self.assertEqual(job.status, "done")
        self.assertEqual(engine.stats()["running"], 0)
        self.assertEqual(self.finish(self.submit(engine, "still working")).status, "done")

    def test_a_speech_model_that_fails_to_load_is_reported_until_a_job_works(self):
        engine = self.engine(worker=fakes.worker_that_cannot_load_its_model)
        self.assertIn("the network is down", engine.load_error)
        self.assertEqual(self.finish(self.submit(engine, "hello there")).status, "done")
        self.assertIsNone(engine.load_error)

    def test_a_speech_model_that_never_loads_does_not_hold_up_the_queue(self):
        with mock.patch.object(engine_module, "_SPEECH_LOAD_TIMEOUT", 1):
            engine = self.engine(worker=fakes.worker_that_hangs_at_startup)
        self.assertIn("did not finish loading", engine.load_error)
        self.assertEqual(self.finish(self.submit(engine, "hello there")).status, "done")
        self.assertIsNone(engine.load_error)

    def test_the_language_model_is_reached_through_the_engine(self):
        engine = self.engine()
        job = self.finish(self.submit(engine, "the meeting", summary=True))
        self.assertEqual(job.result["summary"]["executive_summary"], "Summary of: the meeting")
        self.assertIn("llm_load", job.timings())  # this job was the one that waited for it to load

    def test_without_the_language_model_the_transcript_survives(self):
        engine = self.engine(llm=fakes.llm_that_cannot_load)
        job = self.finish(self.submit(engine, "the meeting", summary=True))
        self.assertEqual((job.status, job.result["turns"][0]["text"]), ("done", "the meeting"))
        self.assertEqual(job.as_dict()["warnings"], ["The summary was skipped. Could not load the language model."])

    def test_the_language_model_is_let_go_when_idle_and_comes_back(self):
        engine = self.engine(idle_unload_seconds=1)
        self.finish(self.submit(engine, "first meeting", summary=True))
        self.assertIsNotNone(engine._llm._proc)
        time.sleep(3)
        self.assertIsNone(engine._llm._proc)
        again = self.finish(self.submit(engine, "second meeting", summary=True))
        self.assertEqual(again.result["summary"]["executive_summary"], "Summary of: second meeting")

    def test_jobs_belong_to_whoever_submitted_them(self):
        engine = self.engine()
        job = self.finish(self.submit(engine, "private", owner="alice"))
        self.assertIs(engine.get(job.id, "alice"), job)
        self.assertEqual(engine.list_jobs("alice"), [job])
        self.assertIsNone(engine.get(job.id, "bob"))
        self.assertEqual(engine.list_jobs("bob"), [])
        self.assertFalse(engine.delete(job.id, "bob"))
        self.assertTrue(engine.delete(job.id, "alice"))
        self.assertEqual(engine.list_jobs("alice"), [])

    def test_api_jobs_are_not_kept(self):
        engine = self.engine()
        job = self.finish(self.submit(engine, "from the api", owner=None))
        self.assertEqual(job.status, "done")
        self.assertEqual(engine._jobs, {})

    def test_each_person_keeps_their_newest_transcripts(self):
        engine = self.engine(max_jobs=2)
        bobs = self.finish(self.submit(engine, "bob's", owner="bob"))
        alices = [self.submit(engine, f"number {n}", owner="alice") for n in range(4)]
        for job in alices:
            self.finish(job)
        self.assertEqual(engine.list_jobs("alice"), [alices[3], alices[2]])
        self.assertEqual(engine.list_jobs("bob"), [bobs])  # alice's uploads don't push bob's out


class PrivateDirTest(unittest.TestCase):
    def test_leftovers_of_a_dead_server_are_removed(self):
        work_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        dead = work_dir / "run-999999"  # no such process
        dead.mkdir()
        (dead / "someones-recording.wav").write_text("private")
        unrelated = work_dir / "notes.txt"
        unrelated.write_text("not ours")

        mine = _private_dir(work_dir)
        self.assertEqual(mine, work_dir / f"run-{os.getpid()}")
        self.assertFalse(dead.exists())
        self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main()
