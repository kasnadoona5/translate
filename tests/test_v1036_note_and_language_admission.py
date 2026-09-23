"""Regression boundaries for source-confirmed notes and Persian prose."""

from collections import Counter

from tarjomeh.core.pipeline import (
    _candidate_text_hash,
    _chunk_memory_admission,
    _chunk_style_policy,
    audit_translation_language,
)
from tarjomeh.quality.integrity import (
    PostEditIntegrityGate,
    available_note_markers,
    extract_note_markers,
    restore_source_note_markers,
)


def test_spaced_final_note_is_one_physical_marker() -> None:
    source = "See (Cerny 2010.)4"
    target = "\u0628\u0646\u06af\u0631\u06cc\u062f (Cerny 2010.) \u06f4"
    required = Counter(extract_note_markers(source))

    assert required == Counter({"4": 1})
    assert available_note_markers(target, required) == required
    assert restore_source_note_markers(source, target)[0] == target
    result = PostEditIntegrityGate().evaluate(source, target, previous=target)
    assert not any(
        finding.check_id == "note_markers_surplus"
        for finding in result.findings
    )


def test_distinct_rich_and_plain_notes_remain_two_occurrences() -> None:
    source = "A claim.4"
    target = "\u06cc\u06a9 \u0627\u062f\u0639\u0627.\u2074 \u06f4"
    required = Counter(extract_note_markers(source))

    assert available_note_markers(target, required) == Counter({"4": 2})
    repaired, report = restore_source_note_markers(source, target)
    assert repaired == "\u06cc\u06a9 \u0627\u062f\u0639\u0627.\u2074"
    assert report["repairs"][-1]["type"] == (
        "unique_plain_duplicate_note_marker_removed"
    )


def test_two_ambiguous_rich_notes_are_not_removed() -> None:
    source = "A claim.4"
    target = "\u06cc\u06a9 \u0627\u062f\u0639\u0627.\u2074 [\u06f4]"

    repaired, report = restore_source_note_markers(source, target)
    assert repaired == target
    assert report["surplus"] == ["4"]


def test_localized_bracketed_note_matches_source_digit() -> None:
    assert extract_note_markers("Claim [4]") == ["4"]
    assert extract_note_markers("\u0627\u062f\u0639\u0627 [\u06f4]") == ["4"]


def test_unmatched_em_dash_before_object_marker_requires_review() -> None:
    source = (
        "They make history - their own and that of others - in institutions."
    )
    target = (
        "\u062a\u0627\u0631\u06cc\u062e \u062e\u0648\u062f \u0648 "
        "\u062f\u06cc\u06af\u0631\u0627\u0646\u2014\u0631\u0627 "
        "\u0645\u06cc\u200c\u0633\u0627\u0632\u0646\u062f."
    )

    report = audit_translation_language(source, target)
    assert report["review_required"]
    assert any(
        item["reason"] == "persian_object_marker_after_unmatched_dash"
        for item in report["unbalanced_explanatory_dash_artifacts"]
    )


def test_paired_aside_before_object_marker_is_not_new_artifact() -> None:
    source = "They make history - their own - in institutions."
    target = (
        "\u062a\u0627\u0631\u06cc\u062e\u2014\u06cc\u0639\u0646\u06cc "
        "\u062a\u0627\u0631\u06cc\u062e \u062e\u0648\u062f\u2014\u0631\u0627 "
        "\u062f\u0631 \u0646\u0647\u0627\u062f\u0647\u0627 "
        "\u0645\u06cc\u200c\u0633\u0627\u0632\u0646\u062f."
    )

    report = audit_translation_language(source, target)
    assert not any(
        item["reason"] == "persian_object_marker_after_unmatched_dash"
        for item in report["unbalanced_explanatory_dash_artifacts"]
    )


def test_non_body_dashes_do_not_trigger_prose_repair() -> None:
    report = audit_translation_language(
        "A title - with an aside - here",
        "\u0639\u0646\u0648\u0627\u0646\u2014\u0631\u0627",
        structural_role="heading",
    )
    assert not report["unbalanced_explanatory_dash_artifacts"]


class _EventDB:
    def __init__(self, events: list[dict]) -> None:
        self.events = events

    def get_chunk_events(self, _job_id: str, _chunk_index: int) -> list[dict]:
        return self.events


def test_scoped_dash_review_withholds_reliable_memory_but_not_clean_style() -> None:
    candidate = "\u0639\u0628\u0627\u0631\u062a \u0645\u0639\u06cc\u0648\u0628.\n\n\u0639\u0628\u0627\u0631\u062a \u0633\u0627\u0644\u0645."
    candidate_hash = _candidate_text_hash(candidate)
    events = [
        {"event_type": "chunk_started", "payload": {}},
        {
            "event_type": "critique_completed",
            "payload": {
                "candidate_target_hash": candidate_hash,
                "valid": True,
                "scores": {
                    "accuracy": 9.5,
                    "fluency": 9.5,
                    "terminology": 9.5,
                    "register": 9.5,
                    "average": 9.5,
                },
                "issue_details": [],
                "high_confidence_minor_refinement_issue_ids": [],
            },
        },
        {
            "event_type": "language_quality_review",
            "payload": {
                "candidate_target_hash": candidate_hash,
                "review_reason": "objective_final_language_artifact",
                "unbalanced_explanatory_dash_count": 1,
                "unbalanced_explanatory_dash_artifacts": [
                    {"paragraph_index": 0, "reason": "persian_object_marker_after_unmatched_dash"}
                ],
            },
        },
    ]
    db = _EventDB(events)

    memory = _chunk_memory_admission(db, "job", 0, candidate_text=candidate)
    style = _chunk_style_policy(db, "job", 0, candidate_text=candidate)
    assert not memory["long_term_reliable"]
    assert memory["continuity_retained"]
    assert style["approved"]
    assert style["excluded_paragraphs"] == [0]

    events[-1]["payload"].pop("candidate_target_hash")
    assert not _chunk_style_policy(
        db, "job", 0, candidate_text=candidate
    )["approved"]
