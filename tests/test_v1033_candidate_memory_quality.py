from __future__ import annotations

import hashlib
from types import SimpleNamespace

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import (
    _candidate_text_hash,
    _canonical_final_quality_record,
    _critique_survives_canonicalization,
    _critique_for_event,
    _final_candidate_selection_payload,
    _readability_review_eligible,
    _readability_review_text,
    _required_anchor_repair_prompt,
    _source_foreign_expression_inventory,
)
from tarjomeh.jobs.database import ChunkStatus, JobDatabase
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.memory.proper_nouns import (
    automatic_terminology_risk_reasons,
    low_authority_mapping_category,
)
from tarjomeh.quality.integrity import restore_source_note_markers


class _Critique(SimpleNamespace):
    def passes_threshold(self, threshold: float) -> bool:
        return self.average >= threshold


def _critique(*, issue_id: str = "", quote: str = "") -> _Critique:
    details = []
    if issue_id:
        details.append({
            "issue_id": issue_id,
            "category": "fluency",
            "severity": "major",
            "confidence": 0.95,
            "source_segment_id": "p1:s1",
            "source_quote": "The relation remains difficult to describe.",
            "current_persian_quote": quote,
            "suggested_correction": "clear target wording",
            "rationale": "The modifier dependency creates broken grammar.",
        })
    return _Critique(
        accuracy=9.5,
        fluency=9.2,
        terminology=9.4,
        register=9.3,
        average=9.35,
        valid=True,
        attempts=1,
        validation_errors=[],
        issue_details=details,
        issues=["issue"] if details else [],
        raw_response="{}",
        coverage_complete=True,
        coverage_checked_segment_ids=["p1:s1"],
        uncovered_source_segment_ids=[],
    )


class _EventDB:
    def __init__(self, events: list[dict]) -> None:
        self.events = events

    def get_chunk_events(self, _job_id: str, _chunk_index: int) -> list[dict]:
        return self.events


def test_critique_event_is_bound_to_candidate_and_stable_issue_fingerprint() -> None:
    payload = _critique_for_event(
        _critique(issue_id="provider-id", quote="opaque target"),
        9.0,
        2,
        candidate_text="retained candidate",
        candidate_stage="final_retained_candidate_validation",
    )

    assert payload["candidate_target_hash"] == _candidate_text_hash(
        "retained candidate"
    )
    assert payload["candidate_stage"] == "final_retained_candidate_validation"
    assert len(payload["issue_details"][0]["issue_fingerprint"]) == 16


def test_critique_rebind_allows_only_lexically_identical_canonicalization() -> None:
    assert _critique_survives_canonicalization(
        "در سال 1973، مقدار 40% بود.",
        "در سال ۱۹۷۳، مقدار ۴۰ ٪ بود.",
    )
    assert not _critique_survives_canonicalization(
        "یک گزاره اول.",
        "یک گزاره دیگر.",
    )


def test_final_quality_ignores_stale_critique_for_different_candidate() -> None:
    stale = _critique_for_event(
        _critique(issue_id="stale", quote="old target"),
        9.0,
        0,
        candidate_text="old candidate",
    )
    retained = _critique_for_event(
        _critique(), 9.0, 1, candidate_text="retained candidate"
    )
    db = _EventDB([
        {"event_type": "chunk_started", "payload": {}},
        {"event_type": "critique_completed", "payload": stale},
        {"event_type": "critique_completed", "payload": retained},
    ])

    final = _canonical_final_quality_record(
        db, "job", 0, candidate_text="retained candidate"
    )

    assert final["critique_candidate_match"] is True
    assert final["durable_authority"] is True
    assert final["issues"] == []


def test_candidate_hash_does_not_disable_intentionally_noncritic_workflow() -> None:
    db = _EventDB([{"event_type": "chunk_started", "payload": {}}])

    final = _canonical_final_quality_record(
        db, "job", 0, candidate_text="canonical candidate"
    )

    assert final["critique_present"] is False
    assert final["critique_candidate_match"] is True
    assert final["durable_authority"] is True


def test_refiner_rejection_survives_provider_issue_id_change_by_fingerprint() -> None:
    first = _critique_for_event(
        _critique(issue_id="first-id", quote="same opaque target"),
        9.0,
        0,
        candidate_text="retained candidate",
    )
    final = _critique_for_event(
        _critique(issue_id="new-id", quote="same opaque target"),
        9.0,
        1,
        candidate_text="retained candidate",
    )
    db = _EventDB([
        {"event_type": "chunk_started", "payload": {}},
        {"event_type": "critique_completed", "payload": first},
        {
            "event_type": "refinement_completed",
            "payload": {"issue_decisions": [{
                "issue_id": "first-id",
                "decision": "rejected",
                "commit_status": "not_committed_refiner_rejected",
            }]},
        },
        {"event_type": "critique_completed", "payload": final},
    ])

    quality = _canonical_final_quality_record(
        db, "job", 0, candidate_text="retained candidate"
    )

    assert quality["durable_authority"] is True
    assert quality["issue_status_counts"] == {
        "rejected_by_source_aware_refiner": 1
    }


def test_candidate_selection_and_canonical_admission_commit_atomically(tmp_path) -> None:
    db = JobDatabase(tmp_path / "jobs.db")
    chunk = Chunk(0, "source", "", "")
    db.create_job("job", tmp_path / "book.pdf", {})
    db.save_chunks("job", [chunk])
    translation = "canonical target"
    digest = hashlib.sha256(translation.encode("utf-8")).hexdigest()
    identity = {
        "version": 1,
        "target_count": 1,
        "canonical_target_hash": digest,
    }
    selection = _final_candidate_selection_payload(translation, identity)

    db.commit_chunk_checkpoint(
        "job",
        0,
        ChunkStatus.COMPLETED,
        translation,
        {"short_term": []},
        paragraph_identity=identity,
        candidate_selection=selection,
        canonical_admission={"canonical_target_hash": digest},
    )

    events = db.get_chunk_events("job", 0)
    by_type = {event["event_type"]: event["payload"] for event in events}
    assert db.get_chunks("job")[0]["translation"] == translation
    assert by_type["final_candidate_selection"]["canonical_target_hash"] == digest
    assert by_type["final_canonical_admission"]["canonical_target_hash"] == digest


def test_readability_review_receives_only_aligned_body_paragraphs() -> None:
    chunk = Chunk(
        0,
        "Heading\n\nBody source.",
        "",
        "",
        metadata={"structural_roles": ["heading", "body"]},
    )
    candidate = "Target heading\n\nTarget body prose."

    assert _readability_review_text(chunk, candidate) == "Target body prose."
    assert _readability_review_eligible(
        chunk,
        _critique(),
        9.0,
        final_candidate=True,
        candidate_text=candidate,
    )
    assert _readability_review_text(chunk, "Merged target paragraph.") == ""


def test_automatic_memory_rejects_partial_compound_and_foreign_paraphrase() -> None:
    assert "target_omits_source_lexical_member" in automatic_terminology_risk_reasons(
        "state apparatus", "\u0622\u067e\u0627\u0631\u0627\u062a\u0648\u0633"
    )
    assert "target_omits_source_lexical_member" not in automatic_terminology_risk_reasons(
        "theory of state", "\u062a\u0626\u0648\u0631\u06cc \u062f\u0648\u0644\u062a"
    )
    assert "foreign_expression_requires_review" in automatic_terminology_risk_reasons(
        "longue dur\u00e9e", "\u0628\u0633\u06cc\u0627\u0631 \u0637\u0648\u0644\u0627\u0646\u06cc"
    )
    assert "foreign_expression_requires_review" not in automatic_terminology_risk_reasons(
        "strategic\u2013relational",
        "\u0631\u0627\u0647\u0628\u0631\u062f\u06cc \u0631\u0627\u0628\u0637\u0647\u200c\u0627\u06cc",
    )
    assert "foreign_expression_requires_review" not in automatic_terminology_risk_reasons(
        "author\u2019s argument",
        "\u0627\u0633\u062a\u062f\u0644\u0627\u0644 \u0646\u0648\u06cc\u0633\u0646\u062f\u0647",
    )


def test_curly_apostrophe_foreign_expression_is_required_without_normalization() -> None:
    expression = "raison d\u2019\u00e9tat"

    assert _source_foreign_expression_inventory(
        f"The chapter invokes {expression} in its historical argument."
    ) == [expression]
    prompt = _required_anchor_repair_prompt(
        f"The chapter invokes {expression}.",
        "accepted target",
        [expression],
    )
    assert f"({expression})" in prompt
    assert "do not translate" in prompt


def test_low_authority_person_role_is_recovered_without_book_specific_names() -> None:
    assert low_authority_mapping_category(
        "Emmerich de Vattel",
        "\u0627\u0645\u0631\u06cc\u0634 \u062f\u0648 \u0648\u0627\u062a\u0644",
        "technical_loanword",
    ) == "person"
    assert low_authority_mapping_category(
        "Economic and Social Research Council",
        (
            "\u0634\u0648\u0631\u0627\u06cc \u067e\u0698\u0648\u0647\u0634 "
            "\u0627\u0642\u062a\u0635\u0627\u062f\u06cc \u0648 \u0627\u062c\u062a\u0645\u0627\u0639\u06cc"
        ),
        "technical_loanword",
    ) != "person"
    assert low_authority_mapping_category(
        "New York", "\u0646\u06cc\u0648\u06cc\u0648\u0631\u06a9", "technical_loanword"
    ) != "person"
    assert low_authority_mapping_category(
        "Bank of England",
        "\u0628\u0627\u0646\u06a9 \u0627\u0646\u06af\u0644\u0633\u062a\u0627\u0646",
        "technical_loanword",
    ) != "person"
    assert low_authority_mapping_category(
        "State of New York",
        "\u0627\u06cc\u0627\u0644\u062a \u0646\u06cc\u0648\u06cc\u0648\u0631\u06a9",
        "technical_loanword",
    ) != "person"


def test_moved_unique_note_marker_returns_to_source_aligned_sentence() -> None:
    source = "First source sentence.2 Second source sentence."
    target = "First target sentence. Second target sentence.2"

    repaired, report = restore_source_note_markers(source, target)

    assert repaired == "First target sentence.\u00b2 Second target sentence."
    assert report["repairs"] == [{
        "type": "relocated_aligned_sentence_terminal_note_marker",
        "marker": "2",
        "paragraph_index": 0,
        "sentence_index": 0,
        "rendered": "\u00b2",
    }]
    assert report["unresolved"] == []


def test_warming_style_profile_labels_fallback_as_non_authoritative() -> None:
    manager = MemoryManager(TarjomehConfig())
    representative = (
        "\u0627\u06cc\u0646 \u062a\u062d\u0644\u06cc\u0644 \u0631\u0627\u0628\u0637\u0647 \u0645\u06cc\u0627\u0646 "
        "\u0646\u0647\u0627\u062f\u0647\u0627 \u0631\u0627 \u0628\u0631\u0631\u0633\u06cc \u0645\u06cc\u200c\u06a9\u0646\u062f."
    )
    fallback = (
        "\u0627\u06cc\u0646 \u0646\u0645\u0648\u0646\u0647 \u0641\u0642\u0637 \u0628\u0631\u0627\u06cc \u062a\u062f\u0627\u0648\u0645 "
        "\u0632\u0645\u06cc\u0646\u0647 \u062f\u0631 \u062d\u0627\u0641\u0638\u0647 "
        "\u0646\u06af\u0647\u062f\u0627\u0631\u06cc \u0634\u062f\u0647 \u0627\u0633\u062a."
    )
    manager.style_samples = [representative, fallback]
    manager.style_sample_records = [
        {
            "text": representative,
            "text_hash": "representative",
            "representative": True,
            "fallback": False,
            "quality_score": 100.0,
            "book_genre": "academic",
        },
        {
            "text": fallback,
            "text_hash": "fallback",
            "representative": False,
            "fallback": True,
            "quality_score": 100.0,
            "book_genre": "academic",
        },
    ]

    profile = manager._render_style_profile()

    assert "[representative]" in profile
    assert "[fallback continuity only; do not imitate defects]" in profile
