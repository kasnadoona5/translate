"""Unit tests for Phase 6: Web UI and REST API."""

from __future__ import annotations

import json
import queue
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import os

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
        ):
            self.assertIn(f'id="{control_id}"', html)

    def test_bearer_header_auth(self) -> None:
        headers = {"Authorization": "Bearer test-token"}
        response = self.client.get("/api/jobs", headers=headers)
        self.assertEqual(response.status_code, 200)

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
            }
        ]

        response = self.client.get(
            "/api/jobs/job-123/qa-report",
            headers={"Authorization": "Bearer test-token"},
        )
        report = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Job Configuration:", report)
        self.assertIn("threshold=9.0 refinements=2", report)
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
            "bilingual_mode": "target_only"
        }

        response = self.client.post("/api/translate", data=data, headers=headers)
        self.assertEqual(response.status_code, 202)

        resp_data = json.loads(response.get_data(as_text=True))
        self.assertEqual(resp_data["status"], "submitted")
        self.assertTrue(resp_data["job_id"])
        self.assertIn("stream_url", resp_data)
        mock_submit.assert_called_once()

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
