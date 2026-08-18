"""Regression tests for the Stage 1 remediation fixes.

These lock in behaviour that is security-relevant and therefore easy to
regress silently: the web UI must not authorise anonymous requests, and API
credentials must never reach the job database or the HTTP API.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from flask import Flask, jsonify

from tarjomeh.core.config import TarjomehConfig, redact_config_secrets
from tarjomeh.jobs.database import JobDatabase, _scrub_config_secrets
from tarjomeh.web.app import _require_auth, _tokens_match

TOKEN = "s3cr3t-کلید"  # deliberately non-ASCII


@pytest.fixture()
def guarded_client():
    """A minimal app with a single ``@_require_auth`` route."""
    app = Flask(__name__)
    app.secret_key = "test-only"

    @app.route("/guarded")
    @_require_auth
    def guarded() -> Any:
        return jsonify({"ok": True})

    return app


# ---------------------------------------------------------------------------
# Fix 1.1 — the UI must fail closed
# ---------------------------------------------------------------------------

def test_missing_token_is_refused_not_allowed(guarded_client, monkeypatch) -> None:
    """An unset UI_SECRET_TOKEN used to authorise *every* request."""
    monkeypatch.delenv("UI_SECRET_TOKEN", raising=False)
    monkeypatch.delenv("TARJOMEH_ALLOW_INSECURE_UI", raising=False)
    response = guarded_client.test_client().get("/guarded")
    assert response.status_code == 503
    # The operator needs to be told how to proceed deliberately.
    assert "TARJOMEH_ALLOW_INSECURE_UI" in response.get_json()["error"]


def test_whitespace_only_token_counts_as_unset(guarded_client, monkeypatch) -> None:
    monkeypatch.setenv("UI_SECRET_TOKEN", "   ")
    monkeypatch.delenv("TARJOMEH_ALLOW_INSECURE_UI", raising=False)
    assert guarded_client.test_client().get("/guarded").status_code == 503


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_explicit_opt_out_restores_open_access(guarded_client, monkeypatch, value) -> None:
    monkeypatch.delenv("UI_SECRET_TOKEN", raising=False)
    monkeypatch.setenv("TARJOMEH_ALLOW_INSECURE_UI", value)
    assert guarded_client.test_client().get("/guarded").status_code == 200


@pytest.mark.parametrize("value", ["", "no", "false", "0", "maybe"])
def test_non_truthy_opt_out_still_refuses(guarded_client, monkeypatch, value) -> None:
    monkeypatch.delenv("UI_SECRET_TOKEN", raising=False)
    monkeypatch.setenv("TARJOMEH_ALLOW_INSECURE_UI", value)
    assert guarded_client.test_client().get("/guarded").status_code == 503


def test_correct_query_token_and_bearer_header_are_accepted(
    guarded_client, monkeypatch
) -> None:
    monkeypatch.setenv("UI_SECRET_TOKEN", TOKEN)
    monkeypatch.delenv("TARJOMEH_ALLOW_INSECURE_UI", raising=False)
    assert guarded_client.test_client().get(f"/guarded?token={TOKEN}").status_code == 200
    header = {"Authorization": f"Bearer {TOKEN}"}
    assert guarded_client.test_client().get("/guarded", headers=header).status_code == 200


@pytest.mark.parametrize(
    "query,header",
    [
        ("", None),
        ("?token=", None),
        ("?token=wrong", None),
        ("", "Bearer wrong"),
        ("", TOKEN),  # raw token in the header, without the Bearer scheme
    ],
)
def test_absent_or_wrong_credentials_are_rejected(
    guarded_client, monkeypatch, query, header
) -> None:
    monkeypatch.setenv("UI_SECRET_TOKEN", TOKEN)
    monkeypatch.delenv("TARJOMEH_ALLOW_INSECURE_UI", raising=False)
    headers = {"Authorization": header} if header else {}
    response = guarded_client.test_client().get(f"/guarded{query}", headers=headers)
    assert response.status_code == 401


def test_tokens_match_handles_non_ascii_without_raising() -> None:
    """secrets.compare_digest rejects non-ASCII str; we compare bytes."""
    assert _tokens_match(TOKEN, TOKEN)
    assert not _tokens_match(TOKEN, TOKEN + "x")
    assert not _tokens_match("", TOKEN)


# ---------------------------------------------------------------------------
# Fix 1.2 — credentials must not be persisted or served
# ---------------------------------------------------------------------------

@pytest.fixture()
def keyed_config() -> TarjomehConfig:
    config = TarjomehConfig()
    config.llm.openrouter.api_keys = ["sk-live-PRIMARY"]
    config.llm.critic.api_keys = ["sk-live-CRITIC"]
    return config


def test_to_dict_redacts_only_on_request(keyed_config) -> None:
    assert keyed_config.to_dict()["llm"]["openrouter"]["api_keys"] == ["sk-live-PRIMARY"]
    redacted = keyed_config.to_dict(redact_secrets=True)
    assert redacted["llm"]["openrouter"]["api_keys"] == []
    assert redacted["llm"]["critic"]["api_keys"] == []
    assert "sk-live" not in json.dumps(redacted)
    # The live object must be untouched, or the running job loses its key.
    assert keyed_config.llm.openrouter.api_keys == ["sk-live-PRIMARY"]


def test_from_dict_rehydrates_a_redacted_config(keyed_config, monkeypatch) -> None:
    monkeypatch.setenv("TRANSLATOR_API_KEY", "sk-from-env")
    monkeypatch.delenv("CRITIC_API_KEY", raising=False)
    restored = TarjomehConfig.from_dict(keyed_config.to_dict(redact_secrets=True))
    assert restored.llm.openrouter.api_keys == ["sk-from-env"]
    # Empty critic keys already mean "inherit the OpenRouter keys".
    assert restored.llm.critic.api_keys == []


def test_rehydrate_does_not_clobber_a_present_key(keyed_config, monkeypatch) -> None:
    monkeypatch.delenv("TRANSLATOR_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    restored = TarjomehConfig.from_dict(keyed_config.to_dict())
    assert restored.llm.openrouter.api_keys == ["sk-live-PRIMARY"]


@pytest.mark.parametrize(
    "payload",
    [{}, {"llm": None}, {"llm": 5}, {"llm": {"openrouter": None}}, {"llm": {"openrouter": {}}}],
)
def test_redactors_tolerate_malformed_configs(payload) -> None:
    """A partial config must not turn redaction into a crash."""
    redact_config_secrets(json.loads(json.dumps(payload)))
    _scrub_config_secrets(json.loads(json.dumps(payload)))


def _stored_config(db_path: Path, job_id: str) -> str:
    with sqlite3.connect(db_path) as raw:
        return raw.execute("SELECT config FROM jobs WHERE id = ?", (job_id,)).fetchone()[0]


def _write_raw_config(db_path: Path, job_id: str, config: dict[str, Any]) -> None:
    with sqlite3.connect(db_path) as raw:
        raw.execute("UPDATE jobs SET config = ? WHERE id = ?", (json.dumps(config), job_id))
        raw.commit()


def test_legacy_plaintext_keys_are_migrated_away(tmp_path, keyed_config) -> None:
    db_path = tmp_path / "jobs.db"
    database = JobDatabase(db_path)
    database.create_job("legacy-job", "book.pdf", keyed_config.to_dict(redact_secrets=True))

    # Simulate a row written by a release that stored the key verbatim.
    _write_raw_config(db_path, "legacy-job", keyed_config.to_dict())
    assert "sk-live-PRIMARY" in _stored_config(db_path, "legacy-job")

    JobDatabase(db_path)  # re-opening runs the one-time migration
    migrated = _stored_config(db_path, "legacy-job")
    assert "sk-live" not in migrated
    # Nothing except the credentials may change.
    assert json.loads(migrated) == keyed_config.to_dict(redact_secrets=True)


def test_reads_scrub_a_row_that_slipped_through(tmp_path, keyed_config) -> None:
    db_path = tmp_path / "jobs.db"
    database = JobDatabase(db_path)
    database.create_job("dirty-job", "book.pdf", keyed_config.to_dict(redact_secrets=True))
    _write_raw_config(db_path, "dirty-job", keyed_config.to_dict())

    assert "sk-live" not in json.dumps(database.get_job("dirty-job"))
    assert "sk-live" not in json.dumps(database.list_jobs())


# ---------------------------------------------------------------------------
# Fix 7.9 — job ids reach the filesystem, so constrain them
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("job_id", ["", "../escape", "a/b", "a*b", "with space", "x" * 65])
def test_unsafe_job_ids_are_rejected(tmp_path, job_id) -> None:
    database = JobDatabase(tmp_path / "jobs.db")
    with pytest.raises(ValueError):
        database.create_job(job_id, "book.pdf", {})


@pytest.mark.parametrize("job_id", ["abc123", "job-1_2", "A" * 64])
def test_safe_job_ids_are_accepted(tmp_path, job_id) -> None:
    database = JobDatabase(tmp_path / "jobs.db")
    database.create_job(job_id, "book.pdf", {})
    assert database.get_job(job_id) is not None
