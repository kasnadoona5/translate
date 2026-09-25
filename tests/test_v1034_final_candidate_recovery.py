from __future__ import annotations

import hashlib
import inspect
import re
from pathlib import Path

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.pipeline import (
    TranslationPipeline,
    _cached_source_obligation_candidate,
    _candidate_text_hash,
    _canonical_chunk_paragraph_identity,
    _final_candidate_selection_payload,
    _persist_source_obligation_candidate,
    _source_obligation_resolution_payload,
)
from tarjomeh.core.term_notes import normalize_citation_house_style_text
from tarjomeh.jobs.database import ChunkStatus, JobDatabase
from tarjomeh.persian.orthography import apply_safe_persian_orthography
from tarjomeh.runtime import runtime_capabilities


def test_final_candidate_identity_precedes_exact_quality_review() -> None:
    source = inspect.getsource(TranslationPipeline._translate_single_chunk)

    identity = source.index("_canonical_chunk_paragraph_identity(chunk, translation)")
    candidate_hash = source.index("canonical_candidate_hash = _candidate_text_hash")
    exact_review = source.index('candidate_stage="final_retained_candidate_validation"')

    assert identity < candidate_hash < exact_review


def test_manual_retranslation_cannot_rewrite_reviewed_candidate() -> None:
    source = inspect.getsource(TranslationPipeline.retranslate_chunk)

    assert "_canonicalize_final_translation" not in source
    assert "checkpoint_translation != translation" in source
    assert "self.db.commit_chunk_checkpoint(" in source
    assert "final_quality_admission=memory_admission" in source


def test_checkpoint_identity_cannot_rewrite_reviewed_candidate() -> None:
    chunk = Chunk(
        0,
        "First.\n\nSecond.",
        "",
        "",
        metadata={"paragraph_indices": [0, 1]},
    )
    reviewed, identity = _canonical_chunk_paragraph_identity(
        chunk, "جملهٔ نخست. جملهٔ دوم."
    )
    checkpoint, checkpoint_identity = _canonical_chunk_paragraph_identity(
        chunk, reviewed
    )

    assert checkpoint == reviewed
    assert checkpoint_identity["canonical_target_hash"] == _candidate_text_hash(
        reviewed
    )
    assert identity["target_count"] == 2


def test_final_quality_is_in_same_checkpoint_transaction(tmp_path) -> None:
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("job", tmp_path / "book.pdf", {})
    db.save_chunks("job", [Chunk(0, "source", "", "")])
    translation = "متن نهایی"
    digest = hashlib.sha256(translation.encode("utf-8")).hexdigest()
    identity = {
        "version": 1,
        "target_count": 1,
        "canonical_target_hash": digest,
    }
    quality = {
        "candidate_target_hash": digest,
        "critique_candidate_match": True,
        "durable_authority": True,
    }

    db.commit_chunk_checkpoint(
        "job",
        0,
        ChunkStatus.COMPLETED,
        translation,
        {"short_term": []},
        paragraph_identity=identity,
        candidate_selection=_final_candidate_selection_payload(
            translation, identity
        ),
        canonical_admission={"canonical_target_hash": digest},
        final_quality_admission=quality,
    )

    events = db.get_chunk_events("job", 0)
    by_type = {event["event_type"]: event["payload"] for event in events}
    assert by_type["final_quality_admission"] == quality
    assert by_type["final_candidate_selection"]["canonical_target_hash"] == digest
    assert db.get_chunks("job")[0]["translation"] == translation
    final_event_times = {
        event["timestamp"]
        for event in events
        if event["event_type"] in {
            "final_candidate_selection",
            "final_canonical_admission",
            "final_quality_admission",
        }
    }
    assert len(final_event_times) == 1


def test_source_obligation_resume_candidate_is_hash_and_source_bound(tmp_path) -> None:
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("job", tmp_path / "book.pdf", {})
    source = "The account advances three claims."
    candidate = "این روایت سه ادعا را مطرح می‌کند."
    db.save_chunks("job", [Chunk(9, source, "", "")])

    entry = _persist_source_obligation_candidate(
        db,
        "job",
        9,
        source,
        candidate,
        [{"classification": "translation_structure_mismatch"}],
    )
    cached, evidence = _cached_source_obligation_candidate(
        db, "job", 9, source
    )

    assert cached == candidate
    assert evidence["authority"] == "review_only_resume_input"
    assert entry["candidate_sha256"] == _candidate_text_hash(candidate)
    assert _cached_source_obligation_candidate(
        db, "job", 9, "Changed source."
    ) == ("", {})

    db.commit_chunk_checkpoint(
        "job",
        9,
        ChunkStatus.COMPLETED,
        candidate,
        {"short_term": []},
        source_obligation_resolution=_source_obligation_resolution_payload(
            source, candidate
        ),
    )
    assert _cached_source_obligation_candidate(db, "job", 9, source) == ("", {})
    stored = db.get_job_artifact("job", "source_obligation_recovery_v1")
    assert stored["entries"]["9"]["status"] == "resolved"
    assert "candidate" not in stored["entries"]["9"]
    events = db.get_chunk_events("job", 9)
    assert events[-1]["event_type"] == "source_obligation_recovery_resolved"


def test_standard_transition_spacing_is_conservative() -> None:
    corrected, edits = apply_safe_persian_orthography(
        "بااین‌حال، استدلال همچنان معتبر است."
    )

    assert corrected == "با این حال، استدلال همچنان معتبر است."
    assert edits[0]["rule_id"] == "standard_ba_in_hal_spacing"


def test_year_only_citation_restores_latin_separators() -> None:
    corrected, changes = normalize_citation_house_style_text(
        "(1986، 1996، 2012a و 2012b)"
    )

    assert corrected == "(1986, 1996, 2012a, 2012b)"
    assert changes


def test_citation_rule_does_not_rewrite_persian_prose_list() -> None:
    text = "در سال‌های 1986، 1996 و 2012 این بحث ادامه یافت."

    assert normalize_citation_house_style_text(text)[0] == text


def test_v1034_release_scripts_and_runtime_contract_are_complete() -> None:
    manifest = runtime_capabilities()
    assert manifest["release"] == "v10.38.0"
    for capability in (
        "identity_before_final_quality",
        "atomic_final_quality_checkpoint",
        "resumable_source_obligation_repair",
    ):
        assert manifest["capabilities"][capability] is True

    deploy = Path("scripts/deploy_tarjomeh_v1034.sh").read_text(encoding="utf-8")
    assert 'TAG="v10.34.0"' in deploy
    assert "RUNNING_V1034_CONFIRMED" in deploy
    assert "VERIFY RUNNING V10.34" in deploy
    assert "NINE_CONTAINER_ID" in deploy
    assert "historical_id" in deploy
    assert "docker system prune" not in deploy
    assert "docker volume prune" not in deploy
    assert 'docker rm -f 9router' not in deploy

    for path in (
        Path("scripts/audit_tarjomeh_v1034_reports.sh"),
        Path("scripts/audit_tarjomeh_v1034_companion.sh"),
    ):
        script = path.read_text(encoding="utf-8")
        assert "non_atomic_final_quality" in script
        assert "pending_finished_source_obligations" in script
        blocks = re.findall(
            r"<<'PY'[^\r\n]*\r?\n(.*?)\r?\nPY",
            script,
            flags=re.DOTALL,
        )
        assert blocks
        for block in blocks:
            compile(block, str(path), "exec")
