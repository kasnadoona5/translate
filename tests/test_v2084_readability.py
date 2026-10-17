import json

from tarjomeh.core.pipeline import _critique_requires_refinement, _merge_readability_evidence
from tarjomeh.quality.critique import CritiqueResult, TranslationCritique


def _minor(source_quote="A complete claim.", quote="عبارت ناقص است"):
    return {
        "issue_id": "a",
        "category": "fluency",
        "severity": "minor",
        "confidence": 0.5,
        "source_segment_id": "p1:s1",
        "source_quote": source_quote,
        "current_persian_quote": quote,
        "suggested_correction": "عبارت کامل است",
    }


def _advice(quote="عبارت ناقص است", correction="عبارت کامل است"):
    return {
        "severity": "major",
        "current_persian_quote": quote,
        "suggested_correction": correction,
        "rationale": "Broken grammar and missing predicate.",
    }


def test_stronger_objective_advice_routes_existing_issue_without_duplicate():
    detail = _minor()
    critique = CritiqueResult(average=9, issue_details=[detail])
    matched, unmatched = _merge_readability_evidence(
        critique, [_advice()], source_text="A complete claim.", translation="عبارت ناقص است."
    )
    assert matched == 1 and not unmatched and len(critique.issue_details) == 1
    assert detail["severity"] == "minor" and detail["confidence"] == 0.5
    assert detail["readability_advisory"]["routing_eligible"]
    assert _critique_requires_refinement(critique, 9)


def test_source_count_conflict_cannot_gain_routing_authority():
    detail = _minor("This chapter addresses two issues.", "دو مسئله")
    critique = CritiqueResult(average=9, issue_details=[detail])
    _merge_readability_evidence(
        critique,
        [_advice("دو مسئله", "سه مسئله")],
        source_text="This chapter addresses two issues.",
        translation="این فصل به دو مسئله می‌پردازد.",
    )
    assert not detail["readability_advisory"]["routing_eligible"]
    assert not _critique_requires_refinement(critique, 9)


def test_partial_invalid_response_retains_exact_span_sibling_for_merge():
    result = TranslationCritique._parse_readability_response(
        json.dumps({"issues": [_advice(), {"severity": "major"}]}), "عبارت ناقص است."
    )
    assert not result.valid and result.validation_errors and len(result.issues) == 1
    critique = CritiqueResult(average=9, issue_details=[_minor()])
    _merge_readability_evidence(
        critique, result.issues, source_text="A complete claim.", translation="عبارت ناقص است."
    )
    assert _critique_requires_refinement(critique, 9)


def test_unlocated_advice_never_acquires_routing_authority():
    detail = _minor()
    detail["source_segment_id"] = ""
    critique = CritiqueResult(average=9, issue_details=[detail])
    _merge_readability_evidence(
        critique, [_advice()], source_text="A complete claim.", translation="عبارت ناقص است."
    )
    assert not _critique_requires_refinement(critique, 9)
