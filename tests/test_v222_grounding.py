"""Unique quote spans may cross sentence splits, not ambiguity or paragraphs."""

import hashlib
import json
from copy import deepcopy

import pytest

from tarjomeh.quality.critique import CritiqueResult, TranslationCritique
from tarjomeh.quality.grounding import resolve_source_segment, source_segments

SOURCE = "Smith et al. (2012) describe a measured trend."


def review(source=SOURCE, quote=SOURCE, current="describes a trend", **extra):
    issue = dict(
        category="accuracy",
        severity="major",
        confidence=0.95,
        source_quote=quote,
        current_persian_quote=current,
        suggested_correction="describes the measured trend",
        rationale="Source comparison",
    )
    issue.update(extra)
    return TranslationCritique._parse_response(
        json.dumps(
            {
                "scores": dict.fromkeys(("accuracy", "fluency", "terminology", "register"), 9),
                "issues": [issue],
            }
        ),
        source,
        "The report describes a trend.",
    )


def test_exact_supplied_cross_split_quote_is_grounded_without_resegmentation():
    before = deepcopy(source_segments(SOURCE))
    assert len(before) == 2
    assert resolve_source_segment(SOURCE, SOURCE) == ("p1:s1", [])
    result = review()
    assert result.valid and len(result.issue_details) == 1
    detail = result.issue_details[0]
    proof = detail["source_quote_span"]
    assert proof["segment_ids"] == ["p1:s1", "p1:s2"]
    assert SOURCE[proof["start"] : proof["end"]] == SOURCE
    assert proof["source_sha256"] == hashlib.sha256(SOURCE.encode()).hexdigest()
    assert source_segments(SOURCE) == before
    assert CritiqueResult.from_dict(result.to_dict()).to_dict() == result.to_dict()


@pytest.mark.parametrize(
    "source,quote",
    [
        (SOURCE + "\n\n" + SOURCE, SOURCE),
        ("Smith et al.\n\n(2012) describe a measured trend.", SOURCE),
        ("Smith et al.\n \n(2012) describe a measured trend.", SOURCE),
        (SOURCE, "Smith et al. invented text."),
        (SOURCE, "Smith et al. describe a measured trend."),
    ],
)
def test_ambiguous_absent_cross_paragraph_or_noncontiguous_spans_stay_invalid(source, quote):
    result = review(source, quote)
    assert not result.issue_details
    assert not result.valid


def test_cross_span_does_not_waive_target_quote_validation():
    assert not review(current="not in the delivered text").valid


def test_cross_span_does_not_forge_reviewer_coverage():
    data = dict(
        scores=dict.fromkeys(("accuracy", "fluency", "terminology", "register"), 9),
        issues=[
            dict(
                category="accuracy",
                severity="major",
                source_quote=SOURCE,
                current_persian_quote="describes a trend",
                suggested_correction="describes the measured trend",
                rationale="test",
            )
        ],
        source_coverage=dict(
            checked_source_segment_ids=["p1:s1"], uncovered_source_segment_ids=[], complete=True
        ),
    )
    result = TranslationCritique._parse_response(
        json.dumps(data), SOURCE, "The report describes a trend.", require_coverage=True
    )
    assert not result.valid and "coverage_checked_ids_incomplete" in result.validation_errors


def test_existing_single_segment_behavior_is_unchanged():
    assert resolve_source_segment(SOURCE, "describe a measured trend.") == ("p1:s2", [])
    assert resolve_source_segment(SOURCE, "describe a measured trend.", "p1:s1") == (
        "p1:s2",
        ["issue_source_segment_id_corrected"],
    )


def test_forged_span_metadata_is_not_trusted():
    result = review(source_quote_span={"start": 999, "segment_ids": ["p99:s1"]})
    assert result.issue_details[0]["source_quote_span"]["start"] == 0


def test_whitespace_and_case_matching_preserves_actual_character_offsets():
    source = "Intro.\nSmith et al.   (2012) describe a measured trend."
    quote = "smith et al. (2012) describe a measured trend."
    result = review(source, quote)
    proof = result.issue_details[0]["source_quote_span"]
    assert source[proof["start"] : proof["end"]] == SOURCE.replace(". (2012)", ".   (2012)")
    assert proof["segment_ids"] == ["p1:s2", "p1:s3"]


def test_duplicate_cross_segment_quote_inside_one_paragraph_is_not_guessed():
    assert not review(SOURCE + " " + SOURCE, SOURCE).issue_details


def test_overlapping_repeated_quotes_are_also_ambiguous():
    assert not review("A. A. A.", "A. A.").issue_details
