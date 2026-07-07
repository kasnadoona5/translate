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


if __name__ == "__main__":
    unittest.main()
