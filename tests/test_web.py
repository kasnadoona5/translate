"""Unit tests for Phase 6: Web UI and REST API."""

from __future__ import annotations

import json
import unittest
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


if __name__ == "__main__":
    unittest.main()
