"""Unit tests for Phase 6: Web UI and REST API."""

from __future__ import annotations

import io
import inspect
import json
import os
import queue
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import flask
    from tarjomeh.web.app import create_app
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


@unittest.skipUnless(HAS_FLASK, "Flask is not installed")
class TestWebUI(unittest.TestCase):
    """Test Flask Web UI and REST API routes."""

    def setUp(self) -> None:
        # Enforce UI_SECRET_TOKEN for testing authentication
        os.environ["UI_SECRET_TOKEN"] = "test-token"
        
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        if "UI_SECRET_TOKEN" in os.environ:
            del os.environ["UI_SECRET_TOKEN"]

    def test_unauthenticated_access(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 401)
        self.assertIn("Authentication required", response.get_data(as_text=True))

    def test_query_token_auth(self) -> None:
        response = self.client.get("/?token=test-token")
        self.assertEqual(response.status_code, 200)

    def test_ui_exposes_phase_6_7_and_quality_controls(self) -> None:
        response = self.client.get("/?token=test-token")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        for control_id in (
            "cfgTermNotes",
            "cfgBookResearch",
            "cfgAutoExtraction",
            "cfgEnforceAutoTerms",
            "cfgCritique",
            "cfgBackTranslation",
            "cfgWebContext",
            "cfgRefineIterations",
            "cfgCritiqueThreshold",
            "cfgBackSample",
            "cfgSearchProvider",
            "cfgResearchQueries",
            "cfgChunkQueries",
            "cfgBookQueryBudget",
            "cfgChapterMode",
            "cfgStopAfterChapter",
            "chapterList",
        ):
            self.assertIn(f'id="{control_id}"', html)

    def test_tracking_restores_visible_job_snapshot_before_streaming(self) -> None:
        script_path = (
            Path(__file__).parents[1] / "src/tarjomeh/web/static/app.js"
        )
        script = script_path.read_text(encoding="utf-8")

        self.assertIn("async function trackJobProgress(jobId)", script)
        self.assertIn(
            'document.getElementById("progressSection").hidden = false',
            script,
        )
        self.assertIn("const response = await fetch(detailUrl)", script)
        self.assertLess(
            script.index("const response = await fetch(detailUrl)"),
            script.index("currentEventSource = new EventSource(streamUrl)"),
        )

    def test_bearer_header_auth(self) -> None:
        headers = {"Authorization": "Bearer test-token"}
        response = self.client.get("/api/jobs", headers=headers)
        self.assertEqual(response.status_code, 200)

    def test_chapter_inspection_returns_detected_manifest(self) -> None:
        response = self.client.post(
            "/api/chapters",
            data={
                "file": (
                    io.BytesIO(
                        b"Chapter 1: Opening\nFirst text.\n\n"
                        b"Chapter 2: Argument\nSecond text."
                    ),
                    "book.txt",
                )
            },
            headers={"Authorization": "Bearer test-token"},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload["chapters"]), 2)
        self.assertEqual(payload["chapters"][1]["position"], 2)
        self.assertEqual(payload["chapters"][1]["number"], 2)
        self.assertEqual(payload["chapters"][1]["title"], "Argument")

    @patch("tarjomeh.jobs.database.JobDatabase")
    def test_get_jobs_api(self, mock_db_cls: MagicMock) -> None:
        # The route imports JobDatabase lazily from tarjomeh.jobs.database and
        # calls list_jobs(); it returns {"jobs": [...]}.
        mock_db = mock_db_cls.return_value
        mock_db.list_jobs.return_value = [
            {
                "id": "job-123",
                "filename": "book.pdf",
                "status": "completed",
                "mode": "academic",
                "pct": 1.0
            }
        ]

        headers = {"Authorization": "Bearer test-token"}
        response = self.client.get("/api/jobs", headers=headers)
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.get_data(as_text=True))
        self.assertIn("jobs", data)
        self.assertEqual(len(data["jobs"]), 1)
        self.assertEqual(data["jobs"][0]["id"], "job-123")

    @patch("tarjomeh.jobs.database.JobDatabase")
    def test_get_job_events_api(self, mock_db_cls: MagicMock) -> None:
        mock_db = mock_db_cls.return_value
        mock_db.get_job.return_value = {"id": "job-123", "config": {}, "input_path": "book.pdf"}
        mock_db.get_chunk_events.return_value = [
            {
                "job_id": "job-123",
                "chunk_index": 0,
                "event_type": "critique_completed",
                "payload": {"scores": {"average": 8}},
            }
        ]

        headers = {"Authorization": "Bearer test-token"}
        response = self.client.get("/api/jobs/job-123/events?chunk=0", headers=headers)
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data["events"][0]["event_type"], "critique_completed")
        mock_db.get_chunk_events.assert_called_once_with("job-123", 0)

    @patch("tarjomeh.jobs.database.JobDatabase")
    def test_job_review_flags_blocking_critique_issue(self, mock_db_cls: MagicMock) -> None:
        mock_db = mock_db_cls.return_value
        mock_db.get_job.return_value = {"id": "job-123", "config": {}, "input_path": "book.pdf"}
        mock_db.get_chunks.return_value = [
            {
                "chunk_index": 0,
                "status": "completed",
                "text": "prospectively constructed",
                "translation": "inductively constructed",
            }
        ]
        mock_db.get_chunk_events.return_value = [
            {
                "job_id": "job-123",
                "chunk_index": 0,
                "event_type": "critique_completed",
                "payload": {
                    "scores": {"average": 8},
                    "force_refinement": True,
                    "blocking_issues": [
                        '[MAJOR/accuracy] source: "prospective" | current: "inductive"',
                    ],
                },
            }
        ]

        headers = {"Authorization": "Bearer test-token"}
        response = self.client.get("/api/jobs/job-123/review", headers=headers)
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.get_data(as_text=True))
        chunk = data["chunks"][0]
        self.assertTrue(chunk["flagged"])
        self.assertEqual(len(chunk["blocking_critique_issues"]), 1)

    @patch("tarjomeh.jobs.database.JobDatabase")
    def test_job_review_uses_configured_critique_threshold(self, mock_db_cls: MagicMock) -> None:
        mock_db = mock_db_cls.return_value
        mock_db.get_job.return_value = {
            "id": "job-123",
            "config": {"translation": {"critique_threshold": 9.0}},
            "input_path": "book.pdf",
        }
        mock_db.get_chunks.return_value = [
            {
                "chunk_index": 0,
                "status": "completed",
                "text": "source",
                "translation": "translation",
            }
        ]
        mock_db.get_chunk_events.return_value = [
            {
                "job_id": "job-123",
                "chunk_index": 0,
                "event_type": "critique_completed",
                "payload": {"scores": {"average": 8}},
            }
        ]

        headers = {"Authorization": "Bearer test-token"}
        response = self.client.get("/api/jobs/job-123/review", headers=headers)
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data["critique_threshold"], 9.0)
        self.assertTrue(data["chunks"][0]["flagged"])

    @patch("tarjomeh.jobs.database.JobDatabase")
    def test_qa_report_exposes_config_and_missing_entities(self, mock_db_cls: MagicMock) -> None:
        mock_db = mock_db_cls.return_value
        mock_db.get_job.return_value = {
            "id": "job-123",
            "input_path": "book.pdf",
            "status": "completed",
            "config": {
                "translation": {
                    "mode": "academic",
                    "enable_critique": True,
                    "critique_threshold": 9.0,
                    "max_refine_iterations": 2,
                    "enable_integrity_gate": True,
                    "enable_back_translation": True,
                    "back_translation_sample_pct": 100,
                    "enable_book_research": True,
                    "enable_web_context": True,
                },
                "glossary": {
                    "enable_compliance_check": True,
                    "enable_auto_correction": True,
                    "enforce_auto_extracted_terms": False,
                },
                "output": {"format": "docx", "term_notes": "inline"},
                "llm": {
                    "critic": {
                        "recovery_model": "critic-fallback",
                        "recovery_max_attempts": 4,
                        "recovery_max_tokens": 16384,
                    }
                },
                "web_search": {
                    "provider": "tavily",
                    "phase7_max_queries": 8,
                    "max_queries_per_chunk": 2,
                    "max_queries_per_book": 100,
                },
            },
        }
        mock_db.get_job_artifact.return_value = None
        mock_db.get_chunks.return_value = [
            {"chunk_index": 0, "status": "needs_review"}
        ]
        mock_db.get_chunk_events.return_value = [
            {
                "chunk_index": 0,
                "event_type": "glossary_compliance_final",
                "payload": {
                    "compliant": False,
                    "violation_count": 1,
                    "citation_exemptions": ["Caceres"],
                    "violations": [
                        {
                            "term": "algorithms",
                            "expected": "\u0627\u0644\u06af\u0648\u0631\u06cc\u062a\u0645\u200c\u0647\u0627",
                            "status": "missing",
                        }
                    ],
                },
            },
            {
                "chunk_index": 0,
                "event_type": "back_translation_completed",
                "payload": {
                    "similarity_score": 0.65,
                    "flagged": True,
                    "diagnostics": {
                        "risk_flags": ["named_entities_missing"],
                        "missing_entities": ["The Politics of Operations"],
                    },
                },
            },
            {
                "chunk_index": 0,
                "event_type": "integrity_final_failed",
                "payload": {
                    "blocking_count": 1,
                    "findings": [{"check_id": "numbers_missing"}],
                },
            },
            {
                "chunk_index": 0,
                "event_type": "translation_recovery_part",
                "payload": {
                    "segment_id": "c0.p0",
                    "validation_attempt": 1,
                    "strict_target_only": False,
                    "source_chars": 100,
                    "output_chars": 90,
                    "size_ratio": 0.9,
                    "valid": True,
                    "errors": [],
                },
            },
        ]

        response = self.client.get(
            "/api/jobs/job-123/qa-report",
            headers={"Authorization": "Bearer test-token"},
        )
        report = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/plain; charset=utf-8")
        self.assertTrue(response.get_data().startswith(b"\xef\xbb\xbf"))
        self.assertIn("Job Configuration:", report)
        self.assertIn("QA Verdict:", report)
        self.assertIn("status=quality_fail", report)
        self.assertIn("classification=QUALITY FAIL", report)
        self.assertIn("integrity_final_failed", report)
        self.assertIn("Recovery part c0.p0", report)
        self.assertIn("source=100 output=90", report)
        self.assertIn("reasons=chunk_needs_review", report)
        self.assertIn("threshold=9.0 refinements=2", report)
        self.assertIn("auto_terms=advisory", report)
        self.assertIn(
            "critic_recovery_attempts=4 fallback=critic-fallback final_tokens=16384",
            report,
        )
        self.assertIn("search_provider=tavily", report)
        self.assertIn("Missing entities: The Politics of Operations", report)
        self.assertIn(
            "algorithms -> \u0627\u0644\u06af\u0648\u0631\u06cc\u062a\u0645\u200c\u0647\u0627 status=missing",
            report,
        )
        self.assertIn("Citation preserved (not terminology): Caceres", report)

    @patch("tarjomeh.jobs.database.JobDatabase")
    def test_qa_report_hides_obsolete_chunk_generation_details(
        self, mock_db_cls: MagicMock
    ) -> None:
        mock_db = mock_db_cls.return_value
        mock_db.get_job.return_value = {
            "id": "job-resumed",
            "input_path": "book.pdf",
            "status": "completed",
            "config": {},
        }
        mock_db.get_job_artifact.return_value = None
        mock_db.get_chunks.return_value = [{
            "chunk_index": 0,
            "status": "completed",
            "translation": "ترجمهٔ نهایی",
        }]
        mock_db.get_chunk_events.return_value = [
            {"chunk_index": 0, "event_type": "chunk_started", "payload": {}},
            {
                "chunk_index": 0,
                "event_type": "integrity_final_failed",
                "payload": {"blocking_count": 1, "findings": []},
            },
            {"chunk_index": 0, "event_type": "chunk_started", "payload": {}},
            {
                "chunk_index": 0,
                "event_type": "critique_completed",
                "payload": {
                    "valid": True,
                    "scores": {"average": 10},
                    "issue_details": [],
                },
            },
        ]

        response = self.client.get(
            "/api/jobs/job-resumed/qa-report",
            headers={"Authorization": "Bearer test-token"},
        )
        report = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("INTEGRITY FINAL:", report)
        self.assertIn("Critique: valid=True", report)

    @patch("tarjomeh.jobs.database.JobDatabase")
    def test_qa_report_lists_missing_original_targets_as_review_signals(
        self, mock_db_cls: MagicMock
    ) -> None:
        mock_db = mock_db_cls.return_value
        mock_db.get_job.return_value = {
            "id": "job-anchor",
            "input_path": "book.pdf",
            "status": "completed",
            "config": {},
        }

        def artifact(_job_id: str, name: str):
            if name == "english_original_anchor_audit":
                return {
                    "anchored_count": 1,
                    "missing_target_count": 1,
                    "missing_targets": [{
                        "paragraph_index": 3,
                        "source": "operative surface",
                        "target": "سطح عملیاتی",
                        "category": "term",
                        "reason": "known_target_not_found",
                    }],
                }
            return None

        mock_db.get_job_artifact.side_effect = artifact
        mock_db.get_chunks.return_value = [{
            "chunk_index": 0,
            "status": "completed",
            "translation": "ترجمه کامل",
        }]
        mock_db.get_chunk_events.return_value = []

        response = self.client.get(
            "/api/jobs/job-anchor/qa-report",
            headers={"Authorization": "Bearer test-token"},
        )
        report = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("status=review_required", report)
        self.assertIn("first_occurrence_target_missing", report)
        self.assertIn("MISSING TARGET: paragraph=3", report)
        self.assertIn("source='operative surface'", report)
        self.assertIn("target='سطح عملیاتی'", report)

    @patch("tarjomeh.jobs.database.JobDatabase")
    def test_qa_report_distinguishes_missing_output_from_quality_failure(
        self, mock_db_cls: MagicMock
    ) -> None:
        mock_db = mock_db_cls.return_value
        mock_db.get_job.return_value = {
            "id": "job-missing",
            "input_path": "book.pdf",
            "status": "completed",
            "config": {},
        }
        mock_db.get_job_artifact.return_value = None
        mock_db.get_chunks.return_value = [{
            "chunk_index": 0,
            "status": "completed",
            "translation": "",
        }]
        mock_db.get_chunk_events.return_value = []

        response = self.client.get(
            "/api/jobs/job-missing/qa-report",
            headers={"Authorization": "Bearer test-token"},
        )
        report = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("status=content_fail", report)
        self.assertIn("classification=CONTENT FAIL", report)
        self.assertIn("missing_output", report)

    @patch("tarjomeh.jobs.database.JobDatabase")
    def test_qa_report_classifies_intentional_checkpoint_as_partial(
        self, mock_db_cls: MagicMock
    ) -> None:
        mock_db = mock_db_cls.return_value
        mock_db.get_job.return_value = {
            "id": "job-checkpoint",
            "input_path": "book.pdf",
            "status": "paused",
            "config": {"translation": {"stop_after_chapter": 1}},
        }

        def artifact(_job_id: str, name: str):
            if name == "chapter_checkpoints":
                return {"reached_positions": [1]}
            if name == "chapter_manifest":
                return {"chapters": [{"position": 1}, {"position": 2}]}
            return None

        mock_db.get_job_artifact.side_effect = artifact
        mock_db.get_chunks.return_value = [
            {
                "chunk_index": 0,
                "status": "completed",
                "translation": "\u062a\u0631\u062c\u0645\u0647 \u0641\u0635\u0644 \u0627\u0648\u0644",
                "metadata": {"chapter_position": 1},
            },
            {
                "chunk_index": 1,
                "status": "pending",
                "translation": None,
                "metadata": {"chapter_position": 2},
            },
        ]
        mock_db.get_chunk_events.return_value = []

        response = self.client.get(
            "/api/jobs/job-checkpoint/qa-report",
            headers={"Authorization": "Bearer test-token"},
        )
        report = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("status=partial_checkpoint", report)
        self.assertIn("classification=PARTIAL CHECKPOINT", report)
        self.assertNotIn("incomplete_chunk", report)
        self.assertNotIn("classification=CONTENT FAIL", report)

    def test_safe_glossary_upload_path_sanitizes_filename(self) -> None:
        from tarjomeh.web.app import _safe_glossary_upload_path

        path = _safe_glossary_upload_path("../../evil.csv")

        self.assertEqual(path.name, "evil.csv")
        self.assertEqual(path.parent.name, "glossary")

    def test_request_log_filter_redacts_query_token(self) -> None:
        import logging
        from tarjomeh.web.app import _QuerySecretLogFilter

        record = logging.LogRecord(
            "werkzeug",
            logging.INFO,
            "",
            0,
            "%s %s",
            ("GET", "/api/jobs/1/stream?token=secret-value&mode=full"),
            None,
        )

        self.assertTrue(_QuerySecretLogFilter().filter(record))
        rendered = record.getMessage()
        self.assertNotIn("secret-value", rendered)
        self.assertIn("token=[REDACTED]", rendered)

    def test_progress_queue_cleanup_drops_finished_queue(self) -> None:
        from tarjomeh.web.app import _progress_queues, _schedule_progress_queue_cleanup

        _progress_queues["cleanup-test"] = queue.Queue()
        try:
            _schedule_progress_queue_cleanup("cleanup-test", delay=0.01)
            time.sleep(0.05)
            self.assertNotIn("cleanup-test", _progress_queues)
        finally:
            _progress_queues.pop("cleanup-test", None)

    @patch("tarjomeh.web.app._executor.submit")
    def test_translate_api_upload(self, mock_submit: MagicMock) -> None:
        # The route returns 202 Accepted with a server-generated job_id and
        # a stream URL, then runs the job in the background executor.
        headers = {"Authorization": "Bearer test-token"}

        data = {
            "file": (open("pyproject.toml", "rb"), "book.pdf"),
            "mode": "fast",
            "format": "txt",
            "bilingual_mode": "target_only",
            "enforce_auto_extracted_terms": "true",
            "stop_after_chapter": "7",
            "pause_after_each_chapter": "false",
        }

        response = self.client.post("/api/translate", data=data, headers=headers)
        self.assertEqual(response.status_code, 202)

        resp_data = json.loads(response.get_data(as_text=True))
        self.assertEqual(resp_data["status"], "submitted")
        self.assertTrue(resp_data["job_id"])
        self.assertIn("stream_url", resp_data)
        mock_submit.assert_called_once()
        submitted_job = mock_submit.call_args.args[0]
        overrides = inspect.getclosurevars(submitted_job).nonlocals[
            "config_overrides"
        ]
        self.assertTrue(overrides["glossary.enforce_auto_extracted_terms"])
        self.assertEqual(overrides["translation.stop_after_chapter"], 7)
        self.assertFalse(overrides["translation.pause_after_each_chapter"])

    @patch("tarjomeh.web.app._executor.submit")
    @patch("tarjomeh.jobs.database.JobDatabase")
    def test_resume_rejects_when_existing_worker_is_alive(
        self,
        mock_db_cls: MagicMock,
        mock_submit: MagicMock,
    ) -> None:
        from tarjomeh.web.app import _active_jobs

        class RunningFuture:
            def done(self) -> bool:
                return False

        mock_db = mock_db_cls.return_value
        mock_db.get_job.return_value = {
            "id": "job-123",
            "input_path": "jobs/uploads/job-123.pdf",
            "config": {},
        }
        _active_jobs["job-123"] = RunningFuture()

        try:
            headers = {"Authorization": "Bearer test-token"}
            response = self.client.post("/api/jobs/job-123/resume", headers=headers)

            self.assertEqual(response.status_code, 409)
            data = json.loads(response.get_data(as_text=True))
            self.assertEqual(data["status"], "pausing")
            mock_submit.assert_not_called()
            mock_db.log_event.assert_called_with(
                "job-123",
                "WARNING",
                "Resume requested while an existing worker was still running.",
            )
        finally:
            _active_jobs.pop("job-123", None)

    @patch("tarjomeh.jobs.database.JobDatabase")
    def test_research_approval_never_overwrites_curated_term(
        self,
        mock_db_cls: MagicMock,
    ) -> None:
        from tarjomeh.glossary.manager import GlossaryManager

        artifact = {
            "status": "completed",
            "terms": [{
                "source": "capital",
                "target": "پایتخت",
                "status": "suggested",
            }],
        }
        mock_db = mock_db_cls.return_value
        mock_db.get_job_artifact.return_value = artifact

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "working.csv"
            glossary = GlossaryManager()
            glossary.add_term("capital", "سرمایه", is_auto=False)
            glossary.save(path)
            with patch(
                "tarjomeh.web.app._working_glossary_path",
                return_value=path,
            ):
                response = self.client.post(
                    "/api/jobs/job-123/research/terms/0/approve",
                    headers={"Authorization": "Bearer test-token"},
                )
            reloaded = GlossaryManager()
            reloaded.load(path)

        self.assertEqual(response.status_code, 409)
        self.assertEqual(reloaded.entries[0].target, "سرمایه")
        mock_db.save_job_artifact.assert_not_called()


if __name__ == "__main__":
    unittest.main()
