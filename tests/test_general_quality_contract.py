from __future__ import annotations

from types import SimpleNamespace

from tarjomeh.core.pipeline import _filter_critique_policy_conflicts
from tarjomeh.core.prompts import (
    CRITIQUE_PROMPT,
    GENERAL_EDITORIAL_CONTRACT,
    REFINE_PROMPT,
    TRANSLATE_SYSTEM_PROMPT,
)
from tarjomeh.quality.back_translator import BackTranslator
from tarjomeh.quality.grounding import concept_risks
from tarjomeh.quality.integrity import (
    PostEditIntegrityGate,
    protected_source_apparatus,
)


def test_general_editorial_contract_is_shared_without_term_mappings() -> None:
    assert "Preserve semantic roles and relations" in GENERAL_EDITORIAL_CONTRACT
    assert "never force an isolated dictionary equivalent" in TRANSLATE_SYSTEM_PROMPT
    assert "never force an isolated dictionary equivalent" in CRITIQUE_PROMPT
    assert "never force an isolated dictionary equivalent" in REFINE_PROMPT
    assert "->" not in GENERAL_EDITORIAL_CONTRACT
    assert len(GENERAL_EDITORIAL_CONTRACT.split()) < 150


def test_relation_risks_are_derived_from_grammar_not_vocabulary_lists() -> None:
    risks = concept_risks({
        "source_quote": "the council's authority may not be withdrawn",
        "severity": "major",
        "category": "accuracy",
    })
    reasons = {risk["reason"] for risk in risks}
    assert {
        "possession_or_attribution",
        "modality",
        "negation",
        "major_semantic_or_terminology_disagreement",
    } <= reasons


def test_source_authored_apparatus_is_structurally_protected() -> None:
    source = 'The label reads “Ars longa” (original-language motto).'
    previous = 'برچسب «Ars longa» (original-language motto) درج شده است.'
    candidate = "برچسب درج شده است."

    assert protected_source_apparatus(source, previous) == [
        "Ars longa",
        "original-language motto",
    ]
    result = PostEditIntegrityGate().evaluate(
        source, candidate, previous=previous, stage="refinement"
    )
    finding = next(
        item for item in result.blocking
        if item.check_id == "source_apparatus_removed"
    )
    assert finding.details["missing"] == [
        "Ars longa",
        "original-language motto",
    ]


def test_critic_cannot_remove_source_authored_apparatus() -> None:
    detail = {
        "formatted": "remove the original-language label",
        "current_translation": "برچسب Ars longa درج شده است",
        "suggested_fix": "برچسب درج شده است",
    }
    critique = SimpleNamespace(
        issue_details=[detail],
        issues=[detail["formatted"]],
    )
    conflicts = _filter_critique_policy_conflicts(
        critique,
        'The label reads “Ars longa”.',
        [],
    )
    assert len(conflicts) == 1
    assert conflicts[0]["removed_source_apparatus"] == ["Ars longa"]
    assert critique.issue_details == []


def test_inline_source_entity_prevents_false_back_translation_flag() -> None:
    result = BackTranslator(SimpleNamespace()).compare(
        "Aurora Research Collective published the report.",
        "The organization published the report.",
        "گزارش را گروه پژوهشی (Aurora Research Collective) منتشر کرد.",
    )
    assert result.diagnostics["missing_entities"] == []
    assert result.diagnostics["entities_preserved_inline"] == [
        "Aurora Research Collective"
    ]
    assert "named_entities_missing" not in result.diagnostics["risk_flags"]


def test_refiner_independence_contract_remains_explicit() -> None:
    assert "it is not automatically authoritative" in REFINE_PROMPT
    assert "preserve it" in REFINE_PROMPT
    assert "suggested wording is never mandatory" in REFINE_PROMPT
