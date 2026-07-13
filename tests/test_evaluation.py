"""Tests for the additive Q1 quality measurement harness."""

from __future__ import annotations

import json
import gc
import os
import tempfile
import unittest
from pathlib import Path

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.cli.main import build_parser
from tarjomeh.jobs.database import ChunkStatus, JobDatabase, JobStatus
from tarjomeh.quality.evaluation import (
    QualityEvaluator,
    evaluation_csv_report,
    evaluation_json_report,
    evaluation_text_report,
    inspect_chunk,
)


class TestDeterministicChecks(unittest.TestCase):
    def test_persian_digits_satisfy_number_preservation(self) -> None:
        result = inspect_chunk(
            "In 1973 the value reached 40% [1].",
            "در سال ۱۹۷۳ مقدار به ۴۰% رسید [1].",
            ChunkStatus.COMPLETED,
            [],
        )
        failed = {item["id"] for item in result["checks"] if not item["passed"]}
        self.assertNotIn("numbers_preserved", failed)
        self.assertNotIn("note_markers_preserved", failed)

    def test_superscript_is_equivalent_to_bracketed_note_number(self) -> None:
        result = inspect_chunk(
            "A claim [12].", "یک ادعا ¹².", ChunkStatus.COMPLETED, [],
        )
        failed = {item["id"] for item in result["checks"] if not item["passed"]}
        self.assertNotIn("note_markers_preserved", failed)

    def test_missing_number_and_note_are_blocking(self) -> None:
        result = inspect_chunk(
            "In 1973 the value reached 40% [1].",
            "مقدار افزایش یافت.",
            ChunkStatus.COMPLETED,
            [],
        )
        failed = {item["id"] for item in result["checks"] if not item["passed"]}
        self.assertIn("numbers_preserved", failed)
        self.assertIn("note_markers_preserved", failed)
        self.assertGreaterEqual(result["blocking_failures"], 2)


class TestQualityEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.db = JobDatabase(root / "jobs.db")
        self.baseline_output = root / "baseline.txt"
        self.candidate_output = root / "candidate.txt"
        self.baseline_output.write_text("baseline", encoding="utf-8")
        self.candidate_output.write_text("candidate", encoding="utf-8")
        source = "Introduction\n\nIn 1973 the value reached 40% [1]."
        for job_id, output in (
            ("baseline", self.baseline_output), ("candidate", self.candidate_output)
        ):
            self.db.create_job(job_id, root / "source.txt", {"translation": {"mode": "academic"}})
            self.db.save_chunks(job_id, [Chunk(0, source, "", "")])
            self.db.update_job_status(job_id, JobStatus.COMPLETED, output_path=output)
        self.db.update_chunk(
            "baseline", 0, ChunkStatus.COMPLETED,
            "مقدمه\n\nدر سال ۱۹۷۳ مقدار به ۴۰% رسید [1].",
        )
        self.db.update_chunk(
            "candidate", 0, ChunkStatus.COMPLETED,
            "مقدمه\n\nمقدار افزایش یافت.",
        )

    def tearDown(self) -> None:
        gc.collect()
        self.tmp.cleanup()

    def test_evaluation_persists_regressions_and_reports(self) -> None:
        result = QualityEvaluator(self.db).evaluate("baseline", "candidate")
        self.assertTrue(result["summary"]["release_blocked"])
        self.assertIn("numbers_preserved", result["chunks"][0]["regressions"])
        self.assertIn("note_markers_preserved", result["chunks"][0]["regressions"])
        self.assertEqual(self.db.get_evaluation(result["id"])["id"], result["id"])
        self.assertIn("Quality Regression Report", evaluation_text_report(result))
        self.assertIn("evaluation_id,chunk_index", evaluation_csv_report(result))
        self.assertEqual(json.loads(evaluation_json_report(result))["id"], result["id"])

    def test_human_preference_and_private_benchmark(self) -> None:
        result = QualityEvaluator(self.db).evaluate("baseline", "candidate")
        self.db.save_evaluation_preference(result["id"], 0, "a", notes="preferred")
        chunk = result["chunks"][0]
        self.db.save_approved_benchmark(
            chunk["source_hash"], chunk["source"], "ترجمه تاییدشده",
            {"evaluation_id": result["id"]},
        )
        loaded = self.db.get_evaluation(result["id"])
        self.assertEqual(loaded["chunks"][0]["preference"]["preference"], "a")
        self.assertEqual(len(self.db.list_approved_benchmark()), 1)

    def test_cli_parser_exposes_eval_command(self) -> None:
        args = build_parser().parse_args(["eval", "old", "new", "--format", "json"])
        self.assertEqual(args.command, "eval")
        self.assertEqual(args.format, "json")


class TestEvaluationWebAPI(unittest.TestCase):
    def test_blind_response_and_approved_preference(self) -> None:
        try:
            from tarjomeh.web.app import create_app
        except ImportError:
            self.skipTest("Flask is not installed")
        old_cwd = Path.cwd()
        old_token = os.environ.get("UI_SECRET_TOKEN")
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            os.environ["UI_SECRET_TOKEN"] = "test-token"
            try:
                db = JobDatabase()
                source = "The total was 12%."
                for job_id, translation in (("old", "مجموع ۱۲% بود."), ("new", "مجموع دوازده درصد بود.")):
                    output = Path(tmp) / f"{job_id}.txt"
                    output.write_text(translation, encoding="utf-8")
                    db.create_job(job_id, Path(tmp) / "source.txt", {})
                    db.save_chunks(job_id, [Chunk(0, source, "", "")])
                    db.update_chunk(job_id, 0, ChunkStatus.COMPLETED, translation)
                    db.update_job_status(job_id, JobStatus.COMPLETED, output_path=output)
                client = create_app().test_client()
                headers = {"Authorization": "Bearer test-token"}
                created = client.post(
                    "/api/evaluations", headers=headers,
                    json={"baseline_job_id": "old", "candidate_job_id": "new"},
                )
                self.assertEqual(created.status_code, 201)
                evaluation_id = created.get_json()["evaluation_id"]
                blind = client.get(f"/api/evaluations/{evaluation_id}", headers=headers)
                chunk = blind.get_json()["chunks"][0]
                self.assertIn("translation_a", chunk)
                self.assertNotIn("candidate_translation", chunk)
                self.assertNotIn("a_is_candidate", chunk)
                saved = client.post(
                    f"/api/evaluations/{evaluation_id}/chunks/0/preference",
                    headers=headers,
                    json={"preference": "a", "save_to_benchmark": True},
                )
                self.assertEqual(saved.status_code, 200)
                self.assertTrue(saved.get_json()["saved_to_benchmark"])
                self.assertEqual(len(db.list_approved_benchmark()), 1)
            finally:
                os.chdir(old_cwd)
                if old_token is None:
                    os.environ.pop("UI_SECRET_TOKEN", None)
                else:
                    os.environ["UI_SECRET_TOKEN"] = old_token
                gc.collect()


if __name__ == "__main__":
    unittest.main()
