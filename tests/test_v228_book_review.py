from copy import deepcopy

import pytest

from test_v2085_book_review import review_api
from tarjomeh.glossary.book_review import (
    book_term_review_evidence, book_term_scope_failure, partition_book_term_proposals,
)


@pytest.mark.parametrize("scope", ["evidence_paragraph", "evidence_required_elsewhere_recommended"])
def test_approval_cannot_exclude_its_evidence_chapter(review_api, scope):
    db, post = review_api
    before = db.get_job_artifact("terms", "book_term_review_v1")
    # Use the actual public constant, not a second spelling of its wire value.
    from tarjomeh.glossary.book_review import BOOK_WIDE_RECOMMENDED
    if scope != "evidence_paragraph":
        scope = BOOK_WIDE_RECOMMENDED
    response = post(dict(action="decide", confirm_bulk=True, decisions=[dict(
        index=0, status="approved", scope_mode=scope, chapter_positions=[2],
    )]))
    assert response.status_code == 409
    assert response.json["reason"] == "scope_excludes_evidence"
    assert db.get_job_artifact("terms", "book_term_review_v1") == before


def test_approval_chapter_accepts_evidence_and_keeps_default(review_api):
    _, post = review_api
    from tarjomeh.glossary.book_review import BOOK_WIDE_RECOMMENDED
    response = post(dict(action="decide", confirm_bulk=True, decisions=[dict(
        index=0, status="approved", chapter_positions=[1],
    )]))
    assert response.status_code == 200
    assert response.json["review"]["proposals"][0]["scope_mode"] == BOOK_WIDE_RECOMMENDED


def test_legacy_missing_withheld_collection_is_unknown_not_zero():
    result = book_term_review_evidence({}, [], [])
    assert result["withheld_proposal_count"] is None
    assert result["withheld_recording_status"] == "not_recorded"


def test_withheld_proposals_have_no_approval_or_scope_authority():
    term = dict(source="unseen", target="حدس", status="withheld",
                evidence_unavailable_reason="no_selected_body_occurrence", source_count=0)
    review = dict(proposals=[], withheld_proposals=[term])
    before = deepcopy(review)
    result = book_term_review_evidence(review, [], [])
    assert result["withheld_proposals"] == [term]
    assert result["withheld_proposal_count"] == 1
    assert result["approved_proposal_count"] == result["scoped_paragraph_count"] == 0
    assert review == before


def test_partition_keeps_source_present_malformed_option_actionable():
    proposals = [
        dict(source="found", source_count=1, target="", target_withheld_reason="malformed"),
        dict(source="absent", source_count=0, evidence_unavailable_reason="no_selected_body_occurrence"),
    ]
    before = deepcopy(proposals)
    actionable, withheld = partition_book_term_proposals(proposals)
    assert len(actionable) == len(withheld) == 1
    assert actionable[0]["source"] == "found"
    assert withheld[0]["status"] == "withheld"
    assert proposals == before


def test_ticked_scope_cannot_silently_drop_an_approved_occurrence():
    ticks = [dict(chapter_position=1), dict(chapter_position=2)]
    failure = book_term_scope_failure("reviewed_occurrences", [1], "hash", [], ticks)
    assert failure["reason"] == "scope_excludes_reviewed_occurrence"
    assert failure["available_chapter_positions"] == [1, 2]
    assert book_term_scope_failure("reviewed_occurrences", [1, 2], "hash", [], ticks) is None
    assert book_term_scope_failure("evidence_paragraph", [1], "hash", None, [])
    assert book_term_scope_failure("evidence_paragraph", [], "hash", None, []) is None


def test_invalid_chapter_positions_do_not_mutate_saved_review(review_api):
    db, post = review_api
    before = db.get_job_artifact("terms", "book_term_review_v1")
    response = post(dict(action="decide", confirm_bulk=True, decisions=[dict(
        index=0, status="approved", chapter_positions=[True],
    )]))
    assert response.status_code == 400
    assert db.get_job_artifact("terms", "book_term_review_v1") == before


def test_later_invalid_scope_rejects_entire_decision_batch(review_api):
    db, post = review_api
    review = db.get_job_artifact("terms", "book_term_review_v1")
    review["proposals"].append(deepcopy(review["proposals"][0]))
    db.save_job_artifact("terms", "book_term_review_v1", review)
    before = deepcopy(review)
    response = post(dict(action="decide", confirm_bulk=True, decisions=[
        dict(index=0, status="approved", chapter_positions=[1]),
        dict(index=1, status="approved", chapter_positions=[2]),
    ]))
    assert response.status_code == 409 and response.json["index"] == 1
    assert db.get_job_artifact("terms", "book_term_review_v1") == before


def test_qa_report_lists_withheld_proposals_without_approval_authority(review_api):
    from unittest.mock import patch
    from tarjomeh.core.config import TarjomehConfig
    from tarjomeh.web.app import create_app

    db, _ = review_api
    review = db.get_job_artifact("terms", "book_term_review_v1")
    review["withheld_proposals"] = [dict(source="unseen", source_count=0, status="withheld",
        evidence_unavailable_reason="no_selected_body_occurrence")]
    db.save_job_artifact("terms", "book_term_review_v1", review)
    with patch("tarjomeh.jobs.database.JobDatabase", return_value=db):
        response = create_app(TarjomehConfig()).test_client().get(
            "/api/jobs/terms/qa-report", headers={"Authorization": "Bearer test"},
        )
    assert response.status_code == 200
    text = response.data.decode("utf-8")
    assert "Withheld Book-Term Proposals" in text
    assert "no_selected_body_occurrence" in text and '"unseen"' in text


def test_new_report_fields_are_diagnostic_and_cover_extra_final_records():
    from tarjomeh.core.evidence_audit import narrow_release_reporting

    proof = dict(original_span="old", resulting_span="new", before_sha256="before", candidate_sha256="after")
    events = [
        dict(chunk_index=0, event_type="extra_final_refinement_completed", payload=dict(
            result=dict(decisions=[dict(issue_id="one", contextual_contraction=proof, commit_reason="retained")]))),
        dict(chunk_index=0, event_type="post_rollback_final_validation", payload=dict(
            candidate_target_hash="after", review_required=True, review_details=["merged"],
            merged_source_review=dict(force_refinement=True))),
    ]
    before = deepcopy(events)
    report = narrow_release_reporting(events, [])
    assert report["contextual_contractions"][0]["proof"] == proof
    assert report["post_rollback_dispositions"][0]["availability"] == "recorded"
    assert report["physical_attempt_records"] == 0
    assert events == before
