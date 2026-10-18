import sqlite3
from copy import deepcopy

import pytest

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.jobs.database import JobDatabase
from tarjomeh.jobs.retranslation import staged_retranslation
from tarjomeh.memory.manager import MemoryManager


def _db(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("Prior source.", encoding="utf-8")
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("job", source, {})
    db.save_chunks(
        "job", [Chunk(index=0, text="Prior source.", chapter_title="Chapter", section_title="")]
    )
    db.commit_chunk_checkpoint("job", 0, "completed", "prior target", {"previous": True})
    db.log_chunk_event("job", 0, "chunk_started", {"old": True})
    return db


@pytest.mark.parametrize("failure", [False, True])
def test_unadmitted_or_exception_keeps_old_canonical_and_cost(tmp_path, failure):
    db = _db(tmp_path)
    before = deepcopy(db.get_chunk("job", 0))
    try:
        with staged_retranslation(db, "job", 0) as attempt:
            attempt.update_chunk("job", 0, "translated", "unadmitted target")
            attempt.log_chunk_event("job", 0, "chunk_started", {"candidate": True})
            attempt.log_chunk_event(
                "job", 0, "llm_call_attempt", {"success": True, "completion_tokens": 10}
            )
            assert db.get_chunk("job", 0) == before
            if failure:
                raise ValueError("simulate rejected outer admission")
    except ValueError:
        assert failure
    assert db.get_chunk("job", 0) == before
    assert db.get_memory_state("job") == {"previous": True}
    events = db.get_chunk_events("job", 0)
    assert len([event for event in events if event["event_type"] == "llm_call_attempt"]) == 1
    assert len([event for event in events if event["event_type"] == "chunk_started"]) == 1
    assert any(event["event_type"] == "aborted_retranslation/chunk_started" for event in events)
    assert events[-1]["payload"]["admitted"] is False
    assert not list(tmp_path.glob("retranslation-*"))


def test_admitted_attempt_publishes_one_atomic_checkpoint(tmp_path):
    db = _db(tmp_path)
    with staged_retranslation(db, "job", 0) as attempt:
        attempt.log_chunk_event("job", 0, "chunk_started", {"candidate": True})
        attempt.log_chunk_event("job", 0, "llm_call_attempt", {"completion_tokens": 10})
        attempt.commit_chunk_checkpoint("job", 0, "needs_review", "new target", {"new": True})
        attempt.admitted = True
        assert db.get_memory_state("job") == {"previous": True}
    assert db.get_chunk("job", 0)["translation"] == "new target"
    assert db.get_memory_state("job") == {"new": True}
    events = db.get_chunk_events("job", 0)
    assert len([event for event in events if event["event_type"] == "llm_call_attempt"]) == 1
    assert events[-1]["payload"]["admitted"] is True


def test_changed_canonical_state_prevents_publication(tmp_path):
    db = _db(tmp_path)
    with (
        pytest.raises(RuntimeError, match="changed during retranslation"),
        staged_retranslation(db, "job", 0) as attempt,
    ):
        attempt.update_chunk("job", 0, "completed", "candidate")
        attempt.admitted = True
        db.update_chunk("job", 0, "completed", "external change")
    assert db.get_chunk("job", 0)["translation"] == "external change"
    assert db.get_memory_state("job") == {"previous": True}


def test_promotion_sql_error_rolls_back_and_closes_temporary_database(tmp_path):
    db = _db(tmp_path)
    before = deepcopy(db.get_chunk("job", 0))
    with (
        pytest.raises(sqlite3.IntegrityError, match="blocked promotion"),
        staged_retranslation(db, "job", 0) as attempt,
    ):
        attempt.update_chunk("job", 0, "completed", "candidate")
        attempt.log_chunk_event("job", 0, "llm_call_attempt", {"completion_tokens": 10})
        attempt.admitted = True
        with db._get_connection() as conn:
            conn.execute(
                "CREATE TRIGGER reject_publication BEFORE DELETE ON chunks "
                "BEGIN SELECT RAISE(ABORT, 'blocked promotion'); END"
            )
    assert db.get_chunk("job", 0) == before
    assert db.get_memory_state("job") == {"previous": True}
    assert any(
        event["event_type"] == "llm_call_attempt" for event in db.get_chunk_events("job", 0)
    )
    assert not list(tmp_path.glob("retranslation-*"))


def test_prior_view_excludes_own_later_and_unproven_memory():
    manager = MemoryManager(TarjomehConfig())
    manager.quality_policy_version = 1
    rows = []
    for index in range(3):
        source = f"Source claim {index}."
        target = f"Target claim {index}."
        manager.update_after_translation(
            Chunk(index=index, text=source, chapter_title="C", section_title=""),
            target,
            style_approved=False,
        )
        rows.append(dict(chunk_index=index, text=source, translation=target, status="completed"))
    manager.proper_nouns.add_noun(
        "Ada Vail",
        "\u0622\u062f\u0627 \u0648\u06cc\u0644",
        category="person",
        provenance="observed_translation",
    )
    before = deepcopy(manager.to_dict())
    view, report = manager.prior_context_view(1, rows)
    assert [pair["chunk_index"] for pair in view.long_term.serialize()] == [0]
    assert [pair.chunk_index for pair in view.short_term.get_entries()] == [0]
    assert view.bilingual_summary.get_context() == ""
    assert "Ada Vail" not in view.proper_nouns.all_nouns()
    assert report["deferred_mappings"]
    assert manager.to_dict() == before


def test_rejection_never_refunds_extra_final_or_search_spending(tmp_path):
    import hashlib

    db = _db(tmp_path)
    assert db.claim_worker("job", "owner")["acquired"]
    reservation = dict(
        source_sha256=hashlib.sha256(b"Prior source.").hexdigest(),
        candidate_sha256="1" * 64,
        issue_ids=["issue"],
        worker_id="owner",
    )
    with staged_retranslation(db, "job", 0) as attempt:
        assert attempt.reserve_extra_final_refinement("job", 0, **reservation)
        attempt.save_job_artifact("job", "web_search_state", {"spent_queries": 3})
    assert db.get_job_artifact("job", "extra_final_refinement_v1")["entries"]["0"]["consumed"] == 1
    assert db.get_job_artifact("job", "web_search_state") == {"spent_queries": 3}
    with staged_retranslation(db, "job", 0) as attempt:
        assert not attempt.reserve_extra_final_refinement("job", 0, **reservation)


def test_prior_view_keeps_proven_layer1_and_earlier_summary_but_not_future_aliases():
    import hashlib

    manager = MemoryManager(TarjomehConfig())
    manager.quality_policy_version = 1
    source = "Lena Vale acts."
    target = "\u0644\u0646\u0627 \u0648\u06cc\u0644 (Lena Vale)."
    manager.proper_nouns.add_noun(
        "Lena Vale",
        "\u0644\u0646\u0627 \u0648\u06cc\u0644",
        category="person",
        provenance="observed_translation",
        context_independent=True,
        evidence_key="anchor:" + hashlib.sha256(source.encode()).hexdigest()[:16],
    )
    manager.update_after_translation(
        Chunk(index=0, text=source, chapter_title="C", section_title=""),
        target,
        style_approved=False,
    )
    saved = manager.to_dict()
    assert saved["mapping_evidence"]["Lena Vale"]
    prior_summary = {
        "english_summary": "Earlier context.",
        "persian_summary": "\u062e\u0644\u0627\u0635\u0647 \u067e\u06cc\u0634\u06cc\u0646",
    }
    manager.summary_history = [
        {"summary": prior_summary, "coverage": list(manager.summary_pending)}
    ]
    manager.proper_nouns.add_alias("Lena Vale", "\u0644\u06cc\u0646\u0627 \u0648\u06cc\u0644")
    manager.bilingual_summary.deserialize(
        {
            "english_summary": "Future context.",
            "persian_summary": "\u062e\u0644\u0627\u0635\u0647 \u0622\u06cc\u0646\u062f\u0647",
        }
    )
    view, report = manager.prior_context_view(
        1, [dict(chunk_index=0, text=source, translation=target, status="completed")]
    )
    assert "Lena Vale" in view.proper_nouns.all_nouns()
    assert not view.proper_nouns.aliases_for("Lena Vale")
    assert report["summary_available"]
    assert "Earlier context." in view.bilingual_summary.get_context()
    assert "Future context." not in view.bilingual_summary.get_context()
