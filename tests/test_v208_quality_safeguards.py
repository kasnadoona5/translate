"""Source-first name, independent-edit and advisory reporting controls."""

from copy import deepcopy

import pytest

from tarjomeh.core.pipeline import (
    _partition_unchanged_quality_count,
    _unresolved_grounded_memory_issues,
    audit_document_final_text,
    audit_translation_language,
)
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.memory.proper_nouns import (
    ProperNouns,
    is_bounded_person_name_target,
    observed_bilingual_target,
)

NAME = "Nora-Lynn Vale"
CLEAN = "\u0646\u0648\u0631\u0627-\u0644\u06cc\u0646 \u0648\u06cc\u0644"
BAD = "\u062f\u06cc\u06af\u0631\u0627\u0646 \u0646\u06cc\u0633\u062a\u2014" + CLEAN


@pytest.mark.parametrize("dash", ["\u2012", "\u2013", "\u2014", "\u2015", " - "])
def test_clause_dash_recovers_only_a_complete_person_anchor(dash):
    target = BAD.replace("\u2014", dash) + " (" + NAME + ")"
    assert observed_bilingual_target(target, NAME, "person") == CLEAN
    assert not is_bounded_person_name_target(NAME, BAD.replace("\u2014", dash))


@pytest.mark.parametrize("join", ["-", "\u2010", "\u2011", "\u2013", "\u2014"])
def test_typographic_compound_name_is_not_categorically_invalid(join):
    assert is_bounded_person_name_target(NAME, CLEAN.replace("-", join))


def test_ambiguous_dash_within_an_unapproved_name_cannot_clip_its_first_part():
    target = CLEAN.replace("-", "\u2013") + " (" + NAME + ")"
    assert observed_bilingual_target(target, NAME, "person") == ""


def test_shared_technical_compound_tokenization_is_unchanged():
    from tarjomeh.memory.proper_nouns import _PERSIAN_WORD_RE

    term = (
        "\u0631\u0627\u0647\u0628\u0631\u062f\u06cc\u2013"
        "\u0631\u0627\u0628\u0637\u0647\u200c\u0627\u06cc"
    )
    assert _PERSIAN_WORD_RE.findall(term) == [term]


@pytest.mark.parametrize(
    "origin",
    [
        "observed_translation",
        "auto_extraction",
        "incremental_extraction",
        "accepted_correction",
        "legacy",
    ],
)
def test_legacy_bad_name_is_retained_but_not_used_by_prompt_or_inline_paths(origin):
    nouns = ProperNouns()
    nouns.deserialize(
        {
            "nouns": {NAME: BAD},
            "categories": {NAME: "person"},
            "provenance": {NAME: {"origin": origin, "alignment_status": "exact_local"}},
        }
    )
    assert nouns.all_nouns()[NAME] == BAD
    assert nouns.is_context_deferred(NAME)
    assert nouns.context_deferral_reason(NAME) == "person_name_contains_context"
    assert NAME not in nouns.get_context(source_text=NAME)
    assert NAME not in nouns.inline_eligible_nouns()
    assert NAME not in nouns.pending_inline_originals(NAME)


def test_curated_name_is_not_rewritten_or_silently_demoted():
    nouns = ProperNouns()
    nouns.deserialize(
        {
            "nouns": {NAME: BAD},
            "categories": {NAME: "person"},
            "provenance": {NAME: {"origin": "curated_glossary"}},
        }
    )
    assert nouns.all_nouns()[NAME] == BAD and not nouns.is_context_deferred(NAME)


def old_issue():
    return {
        "issue_id": "old",
        "category": "terminology",
        "severity": "minor",
        "confidence": 0.6,
        "source_segment_id": "p2:s1",
        "source_quote": "civic institution",
        "current_persian_quote": "\u0646\u0647\u0627\u062f \u0645\u062f\u0646\u06cc",
        "suggested_correction": "\u0645\u0648\u0633\u0633\u0647 \u0645\u062f\u0646\u06cc",
        "rationale": "The lexical choice needs source-aware review.",
    }


SOURCE = "They act.\n\nA civic institution meets."
BEFORE = (
    "\u0622\u0646\u0647\u0627 \u0645\u06cc\u200c\u0631\u0648\u0646\u062f.\n\n"
    "\u0646\u0647\u0627\u062f \u0645\u062f\u0646\u06cc "
    "\u062f\u06cc\u062f\u0627\u0631 \u0645\u06cc\u200c\u06a9\u0646\u062f."
)
AFTER = BEFORE.replace(
    "\u0645\u06cc\u200c\u0631\u0648\u0646\u062f",
    "\u06a9\u0646\u0634 \u0645\u06cc\u200c\u06a9\u0646\u0646\u062f",
)


def test_old_minor_is_excluded_from_edit_count_not_from_final_quality():
    issue = old_issue()
    record = {"issue_details": [issue]}
    raw = _unresolved_grounded_memory_issues(record)
    remaining, carried = _partition_unchanged_quality_count(
        raw,
        source_text=SOURCE,
        previous_text=BEFORE,
        candidate_text=AFTER,
    )
    assert not remaining and carried == [issue]
    assert _unresolved_grounded_memory_issues(record) == raw


@pytest.mark.parametrize(
    "case",
    [
        "whitespace",
        "changed",
        "unlocated",
        "neighbour",
        "duplicate_target",
        "duplicate_source",
        "alignment",
    ],
)
def test_uncertain_attribution_cannot_be_excluded_from_improvement_count(case):
    issue = old_issue()
    source, before, after = SOURCE, BEFORE, AFTER
    if case == "whitespace":
        after += " "
    elif case == "changed":
        after += " \u062f\u06cc\u06af\u0631"
    elif case == "unlocated":
        issue.pop("source_segment_id")
    elif case == "neighbour":
        issue["rationale"] = "Its antecedent in the previous paragraph changed."
    elif case == "duplicate_target":
        before += " " + issue["current_persian_quote"]
        after += " " + issue["current_persian_quote"]
    elif case == "duplicate_source":
        source += " civic institution"
    elif case == "alignment":
        after = after.replace("\n\n", " ")
    snapshot = deepcopy(issue)
    remaining, carried = _partition_unchanged_quality_count(
        [issue],
        source_text=source,
        previous_text=before,
        candidate_text=after,
    )
    assert remaining == [issue] and not carried and issue == snapshot


@pytest.mark.parametrize(
    "suffix", ["\u0647\u0627", "\u0647\u0627\u06cc", "\u0647\u0627\u06cc\u06cc"]
)
def test_quoted_suffix_is_advisory_not_a_new_trust_or_repair_gate(suffix):
    target = "\u00ab\u0645\u0641\u0647\u0648\u0645\u00bb " + suffix
    audit = audit_translation_language("A quoted concept.", target)
    assert audit["quoted_suffix_review"] and not audit["review_required"]
    document = TranslatedDocument(
        paragraphs=[TranslatedParagraph(index=3, source_text="A concept.", translated_text=target)]
    )
    final = audit_document_final_text(document)
    assert document.paragraphs[0].translated_text == target
    assert not final["paragraph_findings"]
    assert any(
        item["check_id"] == "detached_quoted_suffix_unproven" and item["paragraph_index"] == 3
        for item in final["report_only_findings"]
    )


def test_joined_suffix_and_an_unrelated_longer_word_do_not_match():
    for target in (
        "\u00ab\u0645\u0641\u0647\u0648\u0645\u00bb\u0647\u0627",
        "\u00ab\u0645\u0641\u0647\u0648\u0645\u00bb \u0647\u0627\u06cc\u0645\u0627\u0646",
    ):
        assert not audit_translation_language("A concept.", target)["quoted_suffix_review"]


def run_extra_case(tmp_path, source, before, after, baseline_details, candidate_review, decisions):
    """Exercise the real bounded admission method with recorded, no-call tools."""
    import asyncio
    from types import SimpleNamespace

    from tarjomeh.chunking.chunker import Chunk
    from tarjomeh.core.config import TarjomehConfig
    from tarjomeh.core.pipeline import (
        TranslationPipeline,
        _canonical_final_quality_record,
        _critique_for_event,
    )
    from tarjomeh.jobs.database import JobDatabase
    from tarjomeh.quality.critique import CritiqueResult
    from tarjomeh.quality.integrity import PostEditIntegrityGate
    from tarjomeh.quality.refiner import RefinementResult

    db = JobDatabase(tmp_path / "isolated.db")
    db.create_job("isolated", tmp_path / "source.txt", {})
    chunk = Chunk(index=0, text=source, chapter_title="Analysis", section_title="", metadata={})
    db.save_chunks("isolated", [chunk])
    db.claim_worker("isolated", "owner")
    baseline = CritiqueResult(
        accuracy=9, fluency=9, terminology=9, register=9, average=9, issue_details=baseline_details
    )
    db.log_chunk_event("isolated", 0, "chunk_started", {})
    db.log_chunk_event(
        "isolated",
        0,
        "critique_completed",
        _critique_for_event(baseline, 9, -1, candidate_text=before),
    )
    calls = []

    async def refine(*args, **kwargs):
        calls.append("recorded_refiner")
        return RefinementResult(
            translation=after,
            decision="revised",
            rationale="Saved evidence.",
            issue_decisions=decisions,
        )

    async def critique(*args, **kwargs):
        calls.append("recorded_critic")
        assert args[1] == after
        return candidate_review

    pipeline = object.__new__(TranslationPipeline)
    pipeline.config = TarjomehConfig()
    pipeline.config.translation.extra_final_refine_attempts = 1
    pipeline.db, pipeline.worker_id = db, "owner"
    pipeline._run_async = asyncio.run
    # Inputs are already canonical and hash-verified; no historical text rewriting.
    pipeline._canonicalize_final_translation = lambda job, idx, source, text: text
    retained = pipeline._extra_final_quality_repair(
        "isolated",
        chunk,
        before,
        critique_tool=SimpleNamespace(critique=critique),
        refiner_tool=SimpleNamespace(refine_with_decision=refine),
        integrity_gate=PostEditIntegrityGate(),
        terminology="",
        review_context="",
        enforced_entries=[],
        enforce_auto=False,
        protected_terms=[],
        protect_inline_english=False,
        allowed_inline_originals=[],
        reviewed_matches=[],
        threshold=9,
    )
    events = db.get_chunk_events("isolated", 0)
    report = next(
        e["payload"]
        for e in reversed(events)
        if e["event_type"] == "extra_final_refinement_completed"
    )
    final = _canonical_final_quality_record(db, "isolated", 0, retained)
    return retained, report, final, calls


def test_full_extra_admission_keeps_independent_edit_and_withholds_memory(tmp_path):
    from tarjomeh.quality.critique import CritiqueResult

    issue = {
        **old_issue(),
        "issue_id": "changed",
        "category": "accuracy",
        "confidence": 0.8,
        "source_segment_id": "p1:s1",
        "source_quote": "They act",
        "current_persian_quote": BEFORE.split("\n\n")[0],
        "suggested_correction": AFTER.split("\n\n")[0],
    }
    decisions = [
        {
            "issue_id": "changed",
            "decision": "accepted",
            "rationale": "Source checked.",
            "original_span": issue["current_persian_quote"],
            "resulting_span": issue["suggested_correction"],
        }
    ]
    review = CritiqueResult(
        accuracy=9, fluency=9, terminology=9, register=9, average=9, issue_details=[old_issue()]
    )
    retained, report, final, calls = run_extra_case(
        tmp_path, SOURCE, BEFORE, AFTER, [issue], review, decisions
    )
    assert retained == AFTER and report["accepted"]
    assert report["after_actionable_count_raw"] == 1 and report["after_actionable_count"] == 0
    assert report["count_excluded_unchanged_issue_ids"] == ["old"]
    assert not final["durable_authority"]
    assert final["unresolved_grounded_issue_ids"] == ["old"]
    assert calls == ["recorded_refiner", "recorded_critic"]


def test_evidence_report_is_pure_and_legacy_counts_are_unavailable():
    from tarjomeh.core.evidence_audit import targeted_quality_evidence

    proper = {
        "nouns": {NAME: BAD},
        "categories": {NAME: "person"},
        "provenance": {NAME: {"origin": "observed_translation"}},
    }
    snapshot = deepcopy(proper)
    event = {
        "event_type": "extra_final_refinement_completed",
        "chunk_index": 2,
        "payload": {"after_actionable_count": 1},
    }
    report = targeted_quality_evidence(proper, [event], [])
    assert proper == snapshot
    assert report["deferred_name_mappings"][0]["reason"] == "person_name_contains_context"
    assert report["extra_final_counts"][0]["after_actionable_count_raw"] is None
    assert not report["extra_final_counts"][0]["count_evidence_available"]
