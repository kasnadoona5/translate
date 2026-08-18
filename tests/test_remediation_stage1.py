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


# ---------------------------------------------------------------------------
# Fix 2.1 — jobs actually reach RUNNING and FAILED
# ---------------------------------------------------------------------------

def test_pipeline_marks_a_fresh_job_running(tmp_path, monkeypatch) -> None:
    """A new job used to sit at PENDING for its whole life."""
    from unittest.mock import patch as mock_patch

    from tarjomeh.core.pipeline import TranslationPipeline
    from tarjomeh.jobs.database import JobStatus

    monkeypatch.chdir(tmp_path)
    source = tmp_path / "chapter.txt"
    source.write_text("Chapter 1: The State\nA paragraph about theory.", encoding="utf-8")

    config = TarjomehConfig()
    config.translation.mode = "fast"
    config.translation.enable_critique = False
    config.translation.enable_web_context = False
    config.translation.enable_back_translation = False
    config.glossary.enable_auto_extraction = False
    config.output.format = "txt"

    with mock_patch("tarjomeh.core.pipeline.JobDatabase") as db_cls, \
            mock_patch("tarjomeh.core.pipeline.LLMClient") as llm_cls:
        db = db_cls.return_value
        db.get_job.return_value = None
        db.get_chunk_summary.return_value = {
            "total": 0, "completed": 0, "pending": 0, "errors": 0
        }
        llm = llm_cls.return_value
        llm.count_tokens.return_value = 10
        llm.complete.return_value = "ترجمه تست"

        pipeline = TranslationPipeline(config)
        pipeline.llm_client = llm
        pipeline.db = db
        pipeline.run(input_path=source, output_path=tmp_path / "out.txt")

        statuses = [
            call.args[1]
            for call in db.update_job_status.call_args_list
            if len(call.args) >= 2
        ]
        assert JobStatus.RUNNING in statuses, f"never marked RUNNING (saw {statuses})"


def test_record_job_failure_writes_failed(tmp_path, monkeypatch) -> None:
    from tarjomeh.jobs.database import JobStatus
    from tarjomeh.web.app import _record_job_failure

    monkeypatch.chdir(tmp_path)
    db = JobDatabase()
    db.create_job("crashed", "book.pdf", {})
    _record_job_failure("crashed", "parser exploded")

    job = JobDatabase().get_job("crashed")
    assert job["raw_status"] == JobStatus.FAILED
    assert "parser exploded" in (job.get("error_message") or "")


@pytest.mark.parametrize("already", ["completed", "paused", "paused_error", "failed"])
def test_record_job_failure_never_overwrites_a_terminal_status(
    tmp_path, monkeypatch, already
) -> None:
    """PAUSED/COMPLETED are deliberate states; a late crash must not mask them."""
    from tarjomeh.web.app import _record_job_failure

    monkeypatch.chdir(tmp_path)
    db = JobDatabase()
    db.create_job("settled", "book.pdf", {})
    db.update_job_status("settled", already)
    _record_job_failure("settled", "late traceback")
    assert JobDatabase().get_job("settled")["raw_status"] == already


# ---------------------------------------------------------------------------
# Fix 2.2 — exactly one worker per job
# ---------------------------------------------------------------------------

def test_claim_is_exclusive_and_releasable() -> None:
    from tarjomeh.web.app import _release_job_claim, _try_claim_job

    assert _try_claim_job("solo")
    assert not _try_claim_job("solo"), "a second claim must lose"
    _release_job_claim("solo")
    assert _try_claim_job("solo"), "claim must be reusable after release"
    _release_job_claim("solo")


def test_concurrent_claims_produce_exactly_one_winner() -> None:
    """The old check-then-submit sequence let two racing requests both win."""
    import threading

    from tarjomeh.web.app import _release_job_claim, _try_claim_job

    start = threading.Barrier(8)
    wins: list[bool] = []
    lock = threading.Lock()

    def contend() -> None:
        start.wait()
        won = _try_claim_job("contended")
        with lock:
            wins.append(won)

    threads = [threading.Thread(target=contend) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    _release_job_claim("contended")
    assert sum(wins) == 1, f"expected exactly 1 winner, got {sum(wins)}"


def test_release_only_drops_the_matching_claim() -> None:
    from tarjomeh.web.app import _active_jobs, _release_job_claim, _try_claim_job

    assert _try_claim_job("owned")
    stale = object()
    _release_job_claim("owned", expected=stale)
    assert "owned" in _active_jobs, "a stale releaser must not drop a live claim"
    _release_job_claim("owned")
    assert "owned" not in _active_jobs


def test_retranslate_is_refused_while_a_worker_owns_the_job(tmp_path, monkeypatch) -> None:
    from tarjomeh.web.app import _release_job_claim, _try_claim_job, create_app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UI_SECRET_TOKEN", TOKEN)
    JobDatabase().create_job("busy-job", "book.pdf", {})

    app = create_app()
    app.config["TESTING"] = True

    assert _try_claim_job("busy-job")
    try:
        response = app.test_client().post(
            "/api/jobs/busy-job/chunks/0/retranslate",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert response.status_code == 409
        assert "busy" in response.get_json()["error"].lower()
    finally:
        _release_job_claim("busy-job")


# ---------------------------------------------------------------------------
# Fix 2.3 — a rejected upload is deleted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "extra",
    [
        {"config": "{not json"},
        {"max_refine_iterations": "abc"},
        {"selected_chapters": "[0]"},
        {"selected_chapters": "not-json"},
    ],
)
def test_rejected_uploads_do_not_linger(tmp_path, monkeypatch, extra) -> None:
    """Validation runs before the job row exists, so nothing else can clean up."""
    import io

    from tarjomeh.web.app import create_app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UI_SECRET_TOKEN", TOKEN)
    app = create_app()
    app.config["TESTING"] = True
    upload_dir = app.config["UPLOAD_FOLDER"]

    before = {p.name for p in upload_dir.glob("*")}
    response = app.test_client().post(
        "/api/translate",
        data={"file": (io.BytesIO(b"hello"), "book.txt"), **extra},
        content_type="multipart/form-data",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    after = {p.name for p in upload_dir.glob("*")}

    assert response.status_code == 400
    assert after == before, f"leaked upload(s): {after - before}"


# ---------------------------------------------------------------------------
# Fix 2.4 — a bad glossary upload must not destroy the good one
# ---------------------------------------------------------------------------

GOOD_CSV = "source,target,tgt_lng\nstate,دولت,fa\n".encode()


def _post_glossary(app, payload: bytes, filename: str = "philosophy.csv"):
    import io

    return app.test_client().post(
        "/api/glossary/upload",
        data={"file": (io.BytesIO(payload), filename)},
        content_type="multipart/form-data",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )


def test_invalid_glossary_upload_preserves_the_existing_file(tmp_path, monkeypatch) -> None:
    from tarjomeh.web.app import create_app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UI_SECRET_TOKEN", TOKEN)
    app = create_app()
    app.config["TESTING"] = True

    assert _post_glossary(app, GOOD_CSV).status_code == 200
    target = tmp_path / "glossary" / "philosophy.csv"
    assert target.read_bytes() == GOOD_CSV

    # Same filename => same destination. This used to overwrite, then unlink.
    response = _post_glossary(app, b"\xff\xfe\x00not,utf,8\xff")
    assert response.status_code == 400
    assert target.exists(), "the valid glossary was destroyed by a bad upload"
    assert target.read_bytes() == GOOD_CSV, "the valid glossary was corrupted"
    leftovers = list((tmp_path / "glossary").glob("*.part"))
    assert not leftovers, f"staging files left behind: {leftovers}"


def test_valid_glossary_upload_replaces_and_leaves_no_staging_file(
    tmp_path, monkeypatch
) -> None:
    from tarjomeh.web.app import create_app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UI_SECRET_TOKEN", TOKEN)
    app = create_app()
    app.config["TESTING"] = True

    _post_glossary(app, GOOD_CSV)
    replacement = GOOD_CSV + "power,قدرت,fa\n".encode()
    response = _post_glossary(app, replacement)

    assert response.status_code == 200
    assert response.get_json()["terms"] == 2
    target = tmp_path / "glossary" / "philosophy.csv"
    assert target.read_bytes() == replacement
    assert not list((tmp_path / "glossary").glob("*.part"))


# ---------------------------------------------------------------------------
# Fix 4.1 — resume refuses a changed source
# ---------------------------------------------------------------------------

def _chunk(index: int, text: str):
    from tarjomeh.chunking.chunker import Chunk

    return Chunk(index=index, text=text, chapter_title="Ch", section_title="")


def _pipeline_with_db(db):
    """A pipeline shell for testing _verify_resume_alignment in isolation.

    The verifier touches nothing but self.db, so a full __init__ (which builds
    an LLM client) would only add coupling.
    """
    from tarjomeh.core.pipeline import TranslationPipeline

    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.db = db
    return pipeline


def test_chunk_fingerprint_ignores_whitespace_only_differences() -> None:
    from tarjomeh.core.pipeline import TranslationPipeline

    fingerprint = TranslationPipeline._chunk_fingerprint
    assert fingerprint("the state") == fingerprint("the   state")
    assert fingerprint("the state") == fingerprint("  the\nstate\t")
    assert fingerprint("the state") != fingerprint("the estate")
    # A missing chunk row must not collide with a genuinely empty one.
    assert fingerprint(None) == fingerprint("")


def test_resume_alignment_accepts_an_unchanged_source(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    db = JobDatabase()
    db.create_job("aligned", "book.pdf", {})
    chunks = [_chunk(0, "The state persists."), _chunk(1, "Power is relational.")]
    db.save_chunks("aligned", chunks)

    # Reformatted whitespace is not a content change.
    reparsed = [_chunk(0, "The  state   persists."), _chunk(1, "Power is relational.")]
    _pipeline_with_db(db)._verify_resume_alignment("aligned", reparsed)


def test_resume_alignment_allows_a_job_with_no_saved_chunks(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    db = JobDatabase()
    db.create_job("fresh", "book.pdf", {})
    _pipeline_with_db(db)._verify_resume_alignment("fresh", [_chunk(0, "anything")])


@pytest.mark.parametrize(
    "reparsed_texts,label",
    [
        (["The state persists.", "Power is coercive."], "edited chunk text"),
        (["The state persists."], "a chunk disappeared"),
        (["The state persists.", "Power is relational.", "New tail."], "a chunk was added"),
        (["Power is relational.", "The state persists."], "chunks were reordered"),
    ],
)
def test_resume_alignment_refuses_a_changed_source(
    tmp_path, monkeypatch, reparsed_texts, label
) -> None:
    """Reuse is by chunk index alone, so a source edit must not pass silently."""
    from tarjomeh.core.pipeline import ResumeSourceMismatchError
    from tarjomeh.jobs.database import JobStatus

    monkeypatch.chdir(tmp_path)
    db = JobDatabase()
    db.create_job("drifted", "book.pdf", {})
    db.save_chunks(
        "drifted",
        [_chunk(0, "The state persists."), _chunk(1, "Power is relational.")],
    )

    reparsed = [_chunk(i, text) for i, text in enumerate(reparsed_texts)]
    with pytest.raises(ResumeSourceMismatchError):
        _pipeline_with_db(db)._verify_resume_alignment("drifted", reparsed)

    # The job must be parked for a human, not left looking runnable.
    assert JobDatabase().get_job("drifted")["raw_status"] == JobStatus.PAUSED_ERROR, label


def test_resume_mismatch_message_names_the_counts(tmp_path, monkeypatch) -> None:
    from tarjomeh.core.pipeline import ResumeSourceMismatchError

    monkeypatch.chdir(tmp_path)
    db = JobDatabase()
    db.create_job("counted", "book.pdf", {})
    db.save_chunks("counted", [_chunk(0, "alpha"), _chunk(1, "beta")])

    with pytest.raises(ResumeSourceMismatchError) as excinfo:
        _pipeline_with_db(db)._verify_resume_alignment("counted", [_chunk(0, "alpha")])
    message = str(excinfo.value)
    assert "saved=2" in message and "current=1" in message
    assert "dropped=[1]" in message


# ---------------------------------------------------------------------------
# Fix 4.2 — chunk completion and memory commit atomically
# ---------------------------------------------------------------------------

def test_checkpoint_commits_chunk_memory_and_search_together(tmp_path, monkeypatch) -> None:
    from tarjomeh.jobs.database import ChunkStatus

    monkeypatch.chdir(tmp_path)
    db = JobDatabase()
    db.create_job("atomic", "book.pdf", {})
    db.save_chunks("atomic", [_chunk(0, "The state persists.")])

    db.commit_chunk_checkpoint(
        "atomic",
        0,
        ChunkStatus.COMPLETED,
        "دولت پایدار است.",
        {"short_term": ["one"]},
        search_state={"queries_used": 3},
    )

    fresh = JobDatabase()
    record = fresh.get_chunks("atomic")[0]
    assert record["status"] == ChunkStatus.COMPLETED
    assert record["translation"] == "دولت پایدار است."
    assert fresh.get_memory_state("atomic") == {"short_term": ["one"]}
    assert fresh.get_job_artifact("atomic", "web_search_state") == {"queries_used": 3}


def test_checkpoint_without_translation_keeps_the_previous_one(tmp_path, monkeypatch) -> None:
    """Mirrors update_chunk: translation=None means status-only."""
    from tarjomeh.jobs.database import ChunkStatus

    monkeypatch.chdir(tmp_path)
    db = JobDatabase()
    db.create_job("status-only", "book.pdf", {})
    db.save_chunks("status-only", [_chunk(0, "source")])
    db.update_chunk("status-only", 0, ChunkStatus.TRANSLATED, "prior translation")

    db.commit_chunk_checkpoint(
        "status-only", 0, ChunkStatus.NEEDS_REVIEW, None, {"short_term": []}
    )

    record = JobDatabase().get_chunks("status-only")[0]
    assert record["status"] == ChunkStatus.NEEDS_REVIEW
    assert record["translation"] == "prior translation"


def test_checkpoint_without_search_state_leaves_the_artifact_untouched(
    tmp_path, monkeypatch
) -> None:
    from tarjomeh.jobs.database import ChunkStatus

    monkeypatch.chdir(tmp_path)
    db = JobDatabase()
    db.create_job("no-search", "book.pdf", {})
    db.save_chunks("no-search", [_chunk(0, "source")])
    db.save_job_artifact("no-search", "web_search_state", {"queries_used": 9})

    db.commit_chunk_checkpoint(
        "no-search", 0, ChunkStatus.COMPLETED, "ترجمه", {"short_term": []}
    )
    assert JobDatabase().get_job_artifact("no-search", "web_search_state") == {
        "queries_used": 9
    }


def test_a_failed_checkpoint_rolls_back_the_chunk_update(tmp_path, monkeypatch) -> None:
    """The whole point of the fix: no half-written checkpoint survives.

    Previously update_chunk committed on its own, so a crash before the memory
    write left a COMPLETED chunk whose memory contribution was missing.
    """
    from tarjomeh.jobs.database import ChunkStatus

    monkeypatch.chdir(tmp_path)
    db = JobDatabase()
    db.create_job("rollback", "book.pdf", {})
    db.save_chunks("rollback", [_chunk(0, "source")])
    db.update_chunk("rollback", 0, ChunkStatus.TRANSLATED, "prior translation")

    unserialisable = {"memory": object()}
    with pytest.raises(TypeError):
        db.commit_chunk_checkpoint(
            "rollback", 0, ChunkStatus.COMPLETED, "new translation", unserialisable
        )

    record = JobDatabase().get_chunks("rollback")[0]
    assert record["status"] == ChunkStatus.TRANSLATED, "chunk status was not rolled back"
    assert record["translation"] == "prior translation", "translation was not rolled back"
