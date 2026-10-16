from types import SimpleNamespace

import pytest

from tarjomeh.core.pipeline import _reconcile_committed_terminology
from tarjomeh.memory.proper_nouns import ProperNouns, lexical_annotation_deferral


@pytest.mark.parametrize("source,target", [
    ("United Kingdom", "UK پادشاهی متحده"), ("Canada", "CA کانادا"),
])
def test_annotation_mapping_deferred_without_changing_prior_or_raw_history(source, target):
    nouns = ProperNouns()
    nouns.add_noun(source, "نام پیشین", category="term")
    result = nouns.add_noun(source, target, category="term", provenance="accepted_correction",
                            alignment_status="exact_local", context_independent=True)
    assert result["action"] == "context_deferred"
    assert nouns.all_nouns()[source] == "نام پیشین"
    historical = ProperNouns()
    historical.deserialize({"nouns": {source: target}, "categories": {source: "term"},
                            "provenance": {source: {"origin": "accepted_correction",
                                "alignment_status": "exact_local"}}})
    assert historical.all_nouns()[source] == target
    assert historical.context_deferral_reason(source) == "unrelated_leading_latin_display_label"
    assert not historical.get_context(source_text=source)
    assert source not in historical.inline_eligible_nouns()
    assert source not in historical.pending_inline_originals(source)


def test_scientific_member_and_explicit_authority_are_preserved():
    assert not lexical_annotation_deferral("p53 protein", "p53 پروتئین")
    assert not lexical_annotation_deferral("Canada", "کانادا")
    nouns = ProperNouns()
    assert nouns.add_noun("Canada", "CA کانادا", category="term",
                          provenance="curated_glossary")["action"] == "added"
    assert not nouns.context_deferral_reason("Canada")


@pytest.mark.parametrize("target,action", [("CA کانادا", "context_deferred"), ("کانادا", "added")])
def test_reconciliation_reports_mutation_and_deferral(target, action):
    db = SimpleNamespace(
        get_chunk_events=lambda *args: [],
        get_qa_issues=lambda *args: [{"issue_id": "row", "category": "terminology",
            "source_quote": "Canada", "current_persian_quote": "کاندا"}],
        get_issue_decisions=lambda *args: [{"issue_id": "row", "resulting_span": target,
            "payload": {"commit_status": "committed_local"}}],
    )
    nouns = ProperNouns()
    result = _reconcile_committed_terminology(
        db, "test", 0, SimpleNamespace(proper_nouns=nouns), "CA Canada", target,
    )
    entries = result["context_deferred" if action == "context_deferred" else "reconciled"]
    assert len(entries) == 1 and entries[0]["action"] == action
    assert entries[0]["issue_id"] == "row"
    assert ("Canada" in nouns.all_nouns()) == (action == "added")
