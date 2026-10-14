"""Corrective-release controls without providers or production databases."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_v208_quality_safeguards import AFTER, BEFORE, SOURCE, old_issue, run_extra_case

from tarjomeh.core.evidence_audit import targeted_quality_evidence
from tarjomeh.core.pipeline import (
    _candidate_regression_details,
    _partition_unchanged_quality_count,
    _proven_unchanged_issue,
    audit_document_final_text,
    quoted_suffix_review_findings,
)
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.memory.proper_nouns import (
    ProperNouns,
    is_bounded_person_name_target,
    observed_bilingual_target,
)
from tarjomeh.quality.critique import CritiqueResult


@pytest.mark.parametrize("dash", ["\u2012", "\u2013", "\u2014", "\u2015", " - "])
@pytest.mark.parametrize(
    "source,target,prefix",
    [
        ("Hegel", "\u0647\u06af\u0644", "\u0628\u0647 "),
        (
            "Max Weber",
            "\u0645\u0627\u06a9\u0633 \u0648\u0628\u0631",
            "\u0628\u0647 \u062a\u0639\u0628\u06cc\u0631 ",
        ),
        (
            "Jean-Jacques Rousseau",
            "\u0698\u0627\u0646-\u0698\u0627\u06a9 \u0631\u0648\u0633\u0648",
            "\u0646\u0632\u062f ",
        ),
    ],
)
def test_earlier_clause_dash_does_not_suppress_complete_name(dash, source, target, prefix):
    earlier = "\u06a9\u0647 \u067e\u06cc\u0634\u200c\u062a\u0631 \u0622\u0645\u062f"
    text = "\u0686\u06cc\u0632\u06cc" + dash + earlier + dash + prefix + target
    assert observed_bilingual_target(text + " (" + source + ")", source, "person") == target


@pytest.mark.parametrize(
    "source,target",
    [
        ("MacIntyre", "\u0645\u06a9\u200c\u0627\u06cc\u0646\u062a\u0627\u06cc\u0631"),
        ("McCarthy", "\u0645\u06a9\u200c\u06a9\u0627\u0631\u062a\u06cc"),
    ],
)
@pytest.mark.parametrize("prefix", ["", "\u0686\u06cc\u0632\u06cc\u2014\u0628\u0647 "])
def test_single_component_zwnj_names_are_preserved(source, target, prefix):
    assert (
        observed_bilingual_target(prefix + target + " (" + source + ")", source, "person") == target
    )
    assert is_bounded_person_name_target(source, target)


@pytest.mark.parametrize(
    "source,target,reason",
    [
        (
            "de Gaulle",
            "\u0633\u06cc\u0627\u0633\u062a \u062f\u0648\u06af\u0644",
            "automatic_particle_name_unverified",
        ),
        (
            "von Hayek",
            "\u0641\u0648\u0646 \u0647\u0627\u06cc\u06a9",
            "automatic_particle_name_unverified",
        ),
        (
            "van Example",
            "\u0648\u0627\u0646 \u0627\u06af\u0632\u0627\u0645\u067e\u0644",
            "automatic_particle_name_unverified",
        ),
        (
            "Saint-Simon",
            "\u0627\u0646\u062f\u06cc\u0634\u0647 \u0633\u0646\u200c\u0633\u06cc\u0645\u0648\u0646",
            "ambiguous_name_component_alignment",
        ),
        (
            "Saint-Simon",
            "\u0633\u0646\u200c\u0633\u06cc\u0645\u0648\u0646",
            "ambiguous_name_component_alignment",
        ),
        (
            "Max Weber",
            "\u0645\u0627\u06a9\u0633\u200c\u0648\u0628\u0631",
            "ambiguous_name_component_alignment",
        ),
    ],
)
def test_ambiguous_names_are_withheld_from_new_and_historical_paths(source, target, reason):
    assert not observed_bilingual_target(target + " (" + source + ")", source, "person")
    nouns = ProperNouns()
    stored = {
        "nouns": {source: target},
        "categories": {source: "person"},
        "provenance": {source: {"origin": "observed_translation"}},
    }
    nouns.deserialize(stored)
    assert nouns.serialize()["nouns"] == stored["nouns"]
    assert nouns.context_deferral_reason(source) == reason
    assert source not in nouns.get_context(source_text=source)
    assert source not in nouns.inline_eligible_nouns()
    assert source not in nouns.pending_inline_originals(source)


@pytest.mark.parametrize("origin", ["curated_glossary", "approved_research"])
def test_explicit_authority_exceptions_do_not_rewrite_or_demote_names(origin):
    source, target = "von Example", "\u0641\u0648\u0646 \u0627\u06af\u0632\u0627\u0645\u067e\u0644"
    nouns = ProperNouns()
    nouns.deserialize(
        {
            "nouns": {source: target},
            "categories": {source: "person"},
            "provenance": {source: {"origin": origin}},
        }
    )
    assert not nouns.is_context_deferred(source)
    assert nouns.all_nouns()[source] == target
    assert source in nouns.inline_eligible_nouns()


@pytest.mark.parametrize(
    "source,target",
    [
        (
            "Jean-Jacques Rousseau",
            "\u0646\u0632\u062f \u0698\u0627\u0646-\u0698\u0627\u06a9 \u0631\u0648\u0633\u0648",
        ),
        ("Ada Vale", "\u062f\u06a9\u062a\u0631 \u0648\u06cc\u0644"),
        ("Ada Vale", "\u067e\u0631\u0648\u0641\u0633\u0648\u0631 \u0648\u06cc\u0644"),
        ("Ada Vale", "\u0698\u0646\u0631\u0627\u0644 \u0648\u06cc\u0644"),
        ("Ada Vale", "\u0641\u06cc\u0644\u0633\u0648\u0641 \u0648\u06cc\u0644"),
        ("Ada Vale", "\u0646\u0648\u06cc\u0633\u0646\u062f\u0647 \u0648\u06cc\u0644"),
        ("Ada Vale", "\u0627\u0633\u062a\u0627\u062f \u0648\u06cc\u0644"),
    ],
)
def test_context_and_titles_cannot_fill_a_missing_name_component(source, target):
    observed = observed_bilingual_target(target + " (" + source + ")", source, "person")
    expected = (
        target.removeprefix("\u0646\u0632\u062f ") if source == "Jean-Jacques Rousseau" else ""
    )
    assert observed == expected
    assert not is_bounded_person_name_target(source, target)
    nouns = ProperNouns()
    nouns.deserialize({"nouns": {source: target}, "categories": {source: "person"}})
    assert nouns.context_deferral_reason(source) == "person_name_contains_context"


def test_source_authorized_title_and_existing_initial_allowance_are_preserved():
    target = "\u062f\u06a9\u062a\u0631 \u0648\u06cc\u0644"
    assert observed_bilingual_target(target + " (Dr Vale)", "Dr Vale", "person") == target
    assert is_bounded_person_name_target("Ben West", "\u0628. \u0648\u0633\u062a")


def test_existing_typographic_compound_join_requires_the_source_hyphen_boundary():
    target = "\u0645\u0631\u06cc \u0640 \u0622\u0646 \u0627\u0633\u0645\u06cc\u062a"
    assert (
        observed_bilingual_target(target + " (Mary-Anne Smith)", "Mary-Anne Smith", "person")
        == target
    )
    assert not observed_bilingual_target(target + " (Mary Anne Smith)", "Mary Anne Smith", "person")
    wrong_boundary = "\u0645\u0631\u06cc \u0622\u0646 \u0640 \u0627\u0633\u0645\u06cc\u062a"
    assert not observed_bilingual_target(
        wrong_boundary + " (Mary-Anne Smith)",
        "Mary-Anne Smith",
        "person",
    )


def test_non_person_categories_do_not_acquire_the_new_particle_restriction():
    source, target = (
        "St Ives PLC",
        "\u0634\u0631\u06a9\u062a \u0633\u0646\u062a \u0622\u06cc\u0648\u0632",
    )
    nouns = ProperNouns()
    nouns.deserialize(
        {
            "nouns": {source: target, "policy narratives": "\u0631\u0648\u0627\u06cc\u062a"},
            "categories": {source: "organization", "policy narratives": "term"},
        }
    )
    assert not nouns.is_context_deferred(source)
    assert not nouns.is_context_deferred("policy narratives")


def _changed_issue():
    return {
        **old_issue(),
        "issue_id": "changed",
        "category": "accuracy",
        "confidence": 0.8,
        "source_segment_id": "p1:s1",
        "source_quote": "They act",
        "current_persian_quote": BEFORE.split("\n\n")[0],
        "suggested_correction": AFTER.split("\n\n")[0],
    }


def _decisions(issue):
    return [
        {
            "issue_id": issue["issue_id"],
            "decision": "accepted",
            "rationale": "Source checked.",
            "original_span": issue["current_persian_quote"],
            "resulting_span": issue["suggested_correction"],
        }
    ]


def test_actual_admission_restores_r109_exclusion_with_repeated_source_quote(tmp_path):
    source = SOURCE.replace("They act.", "They act at a civic institution.")
    old = {**old_issue(), "severity": "major", "confidence": 0.9}
    snapshot = deepcopy(old)
    strict_remaining, strict_carried = _partition_unchanged_quality_count(
        [old],
        source_text=source,
        previous_text=BEFORE,
        candidate_text=AFTER,
    )
    assert strict_remaining == [old] and not strict_carried
    assert _proven_unchanged_issue(
        old, source_text=source, previous_text=BEFORE, candidate_text=AFTER
    )
    review = CritiqueResult(
        accuracy=9, fluency=9, terminology=9, register=9, average=9, issue_details=[old]
    )
    changed = _changed_issue()
    retained, report, final, calls = run_extra_case(
        tmp_path,
        source,
        BEFORE,
        AFTER,
        [changed],
        review,
        _decisions(changed),
    )
    assert retained == AFTER and report["accepted"]
    assert report["after_actionable_count_raw"] == 1 and report["after_actionable_count"] == 0
    assert report["count_excluded_unchanged_issue_ids"] == ["old"]
    assert final["unresolved_grounded_issue_ids"] == ["old"] and not final["durable_authority"]
    assert old == snapshot and calls == ["recorded_refiner", "recorded_critic"]


def test_minor_with_repeated_source_quote_does_not_gain_the_legacy_exclusion(tmp_path):
    source = SOURCE.replace("They act.", "They act at a civic institution.")
    old = old_issue()
    changed = _changed_issue()
    review = CritiqueResult(
        accuracy=9, fluency=9, terminology=9, register=9, average=9, issue_details=[old]
    )
    retained, report, _final, _calls = run_extra_case(
        tmp_path,
        source,
        BEFORE,
        AFTER,
        [changed],
        review,
        _decisions(changed),
    )
    assert retained == BEFORE and not report["accepted"]
    assert (
        report["after_actionable_count"] == 1 and not report["count_excluded_unchanged_issue_ids"]
    )


def test_both_exclusion_paths_deduplicate_by_fingerprint_without_mutating_advice():
    old = {**old_issue(), "severity": "major", "confidence": 0.9}
    duplicate = {**old, "issue_id": "provider-alias"}
    advice = [old, duplicate]
    snapshot = deepcopy(advice)
    remaining, carried = _partition_unchanged_quality_count(
        advice,
        source_text=SOURCE,
        previous_text=BEFORE,
        candidate_text=AFTER,
        r109_unchanged=[old],
    )
    assert not remaining and carried == [old] and advice == snapshot


def test_actual_report_includes_both_paths_and_provider_aliases(tmp_path):
    source = SOURCE.replace("They act.", "They act at a civic institution.")
    old = {**old_issue(), "severity": "major", "confidence": 0.9}
    alias = {**old, "issue_id": "provider-alias"}
    minor = {
        **old_issue(),
        "issue_id": "minor",
        "source_quote": "meets",
        "current_persian_quote": "\u062f\u06cc\u062f\u0627\u0631",
    }
    review = CritiqueResult(
        accuracy=9,
        fluency=9,
        terminology=9,
        register=9,
        average=9,
        issue_details=[old, alias, minor],
    )
    changed = _changed_issue()
    retained, report, final, _calls = run_extra_case(
        tmp_path,
        source,
        BEFORE,
        AFTER,
        [changed],
        review,
        _decisions(changed),
    )
    assert retained == AFTER and report["accepted"]
    assert report["after_actionable_count_raw"] == 3
    assert report["after_actionable_count"] == 0
    assert report["count_excluded_unchanged_issue_ids"] == ["old", "provider-alias", "minor"]
    assert len(report["newly_observed_unchanged_issues"]) == 2
    assert set(final["unresolved_grounded_issue_ids"]) == {"old", "provider-alias", "minor"}
    assert not final["durable_authority"]


def test_legacy_classification_cannot_bypass_changed_or_unlocated_evidence():
    old = {**old_issue(), "severity": "major", "confidence": 0.9}
    old["rationale"] = "Its antecedent in the previous paragraph changed."
    carried = []
    regressions = _candidate_regression_details(
        SimpleNamespace(issue_details=[old]),
        SimpleNamespace(issue_details=[]),
        [],
        source_text=SOURCE,
        previous_text=BEFORE,
        candidate_text=AFTER,
        newly_observed_unchanged=carried,
    )
    assert regressions == [old] and not carried
    remaining, excluded = _partition_unchanged_quality_count(
        [old],
        source_text=SOURCE,
        previous_text=BEFORE,
        candidate_text=AFTER,
        r109_unchanged=[old],
    )
    assert remaining == [old] and not excluded


def test_suffix_context_scope_never_claims_alignment_from_equal_counts():
    target = "\u00ab\u0645\u0641\u0647\u0648\u0645\u00bb \u0647\u0627"
    direct = quoted_suffix_review_findings("First.\n\nSecond.", target)[0]
    assert direct["source_context_scope"] == "provided_source_prefix"
    report = targeted_quality_evidence(
        {}, [], [{"text": "First.\n\nSecond.", "translation": "\u0627\u0648\u0644.\n\n" + target}]
    )
    finding = report["quoted_suffix_reports"][0]
    assert finding["source_context_scope"] == "unverified_same_index_paragraph_prefix"
    assert not finding["paragraph_alignment_verified"] and finding["report_only"]
    document = TranslatedDocument(
        paragraphs=[
            TranslatedParagraph(index=1, source_text="Second.", translated_text=target),
        ]
    )
    finding = audit_document_final_text(document)["report_only_findings"][0]
    assert finding["source_context_scope"] == "source_paragraph_prefix"
    assert document.paragraphs[0].translated_text == target
