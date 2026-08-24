from __future__ import annotations

from pathlib import Path

from tarjomeh.core.pipeline import (
    _language_repair_is_local,
    _language_quality_strictly_improves,
    _targeted_language_repair_prompt,
    audit_translation_language,
)
from tarjomeh.core.prompts import CRITIQUE_PROMPT, REFINE_PROMPT, TRANSLATE_CHUNK_PROMPT
from tarjomeh.memory.manager import _clean_style_sample
from tarjomeh.quality.integrity import (
    detached_ezafe_artifacts,
    foreign_script_artifacts,
    markup_wrapper_artifacts,
    parenthesis_artifacts,
    repair_source_grounded_language_artifacts,
    source_unjustified_repeated_word_artifacts,
)


def test_duplicate_repair_uses_aligned_source_paragraph_evidence() -> None:
    source = "It was very very difficult.\n\nThey make history of themselves and others."
    target = (
        "\u0627\u06cc\u0646 \u06a9\u0627\u0631 \u0628\u0633\u06cc\u0627\u0631 \u0628\u0633\u06cc\u0627\u0631 \u062f\u0634\u0648\u0627\u0631 \u0628\u0648\u062f.\n\n"
        "\u0622\u0646\u200c\u0647\u0627 \u062a\u0627\u0631\u06cc\u062e \u062a\u0627\u0631\u06cc\u062e \u062e\u0648\u062f \u0648 \u062f\u06cc\u06af\u0631\u0627\u0646 \u0631\u0627 \u0645\u06cc\u200c\u0633\u0627\u0632\u0646\u062f."
    )

    repaired, report = repair_source_grounded_language_artifacts(source, target)

    assert "\u0628\u0633\u06cc\u0627\u0631 \u0628\u0633\u06cc\u0627\u0631" in repaired
    assert "\u062a\u0627\u0631\u06cc\u062e \u062a\u0627\u0631\u06cc\u062e" not in repaired
    assert any(item["type"] == "adjacent_duplicate" for item in report["repairs"])
    assert source_unjustified_repeated_word_artifacts(source, repaired) == []


def test_source_authored_repetition_is_not_a_language_finding() -> None:
    report = audit_translation_language(
        "This is very very important.",
        "\u0627\u06cc\u0646 \u0645\u0648\u0636\u0648\u0639 \u0628\u0633\u06cc\u0627\u0631 \u0628\u0633\u06cc\u0627\u0631 \u0645\u0647\u0645 \u0627\u0633\u062a.",
    )
    assert report["repeated_word_count"] == 0


def test_general_language_artifacts_are_source_grounded() -> None:
    source = "The state develops through institutions."
    target = (
        "* (raison d'etat) * \u062f\u0631 \u062a\u5d4c\u06cc\u062f\u06af\u06cc \u0646\u0647\u0627\u062f\u06cc "
        "(\u0627\u0631\u0648\u067e\u0627\u0645\u062d\u0648\u0631 (Eurocentric) \u0627\u0633\u062a \u0648 \u062d\u0627\u0644 (\u0622\u06cc\u0646\u062f\u0647\u200c\u0647\u0627) \u06cc \u062f\u0648\u0644\u062a."
    )

    assert foreign_script_artifacts(source, target)[0]["script"] == "han"
    assert markup_wrapper_artifacts(source, target)
    assert parenthesis_artifacts(source, target)
    assert detached_ezafe_artifacts(target)


def test_markdown_wrapper_repair_does_not_touch_source_authored_markup() -> None:
    generated, generated_report = repair_source_grounded_language_artifacts(
        "The expression raison d'etat is discussed.",
        "* (raison d'etat) * \u0628\u0631\u0631\u0633\u06cc \u0645\u06cc\u200c\u0634\u0648\u062f.",
    )
    authored, authored_report = repair_source_grounded_language_artifacts(
        "The *emphasized* expression is discussed.",
        "* \u0639\u0628\u0627\u0631\u062a \u0628\u0631\u062c\u0633\u062a\u0647 * \u0628\u0631\u0631\u0633\u06cc \u0645\u06cc\u200c\u0634\u0648\u062f.",
    )

    assert generated.startswith("(raison d'etat)")
    assert any(
        item["type"] == "markdown_emphasis_wrapper"
        for item in generated_report["repairs"]
    )
    assert authored.startswith("*")
    assert not any(
        item["type"] == "markdown_emphasis_wrapper"
        for item in authored_report["repairs"]
    )


def test_local_repair_admission_requires_monotonic_improvement() -> None:
    before = {"unexpected_latin_count": 1, "foreign_script_count": 1}
    improved = {"unexpected_latin_count": 0, "foreign_script_count": 1}
    traded = {"unexpected_latin_count": 0, "foreign_script_count": 2}

    assert _language_quality_strictly_improves(before, improved)
    assert not _language_quality_strictly_improves(before, traded)
    assert not _language_quality_strictly_improves(before, before)


def test_final_language_repair_rejects_broad_body_rewrite() -> None:
    before = "\u0627\u06cc\u0646 \u0628\u0646\u062f \u062f\u0631\u0628\u0627\u0631\u0647 March 2015 \u0648 \u062a\u062d\u0648\u0644\u0627\u062a \u0646\u0647\u0627\u062f\u06cc \u0627\u0633\u062a."
    local = "\u0627\u06cc\u0646 \u0628\u0646\u062f \u062f\u0631\u0628\u0627\u0631\u0647 \u0645\u0627\u0631\u0633 2015 \u0648 \u062a\u062d\u0648\u0644\u0627\u062a \u0646\u0647\u0627\u062f\u06cc \u0627\u0633\u062a."
    rewrite = "\u062f\u0631 \u0633\u0627\u0644 2015 \u062a\u062d\u0648\u0644\u0627\u062a \u0645\u0647\u0645\u06cc \u062f\u0631 \u0646\u0647\u0627\u062f\u0647\u0627 \u0631\u062e \u062f\u0627\u062f."

    assert _language_repair_is_local(before, local, structural_role="body")
    assert not _language_repair_is_local(before, rewrite, structural_role="body")


def test_prompts_require_fluent_persian_without_weakening_refiner_veto() -> None:
    assert "rebuild clause order" in TRANSLATE_CHUNK_PROMPT or "split sentences" in TRANSLATE_CHUNK_PROMPT
    assert "opaque modifier stacks" in CRITIQUE_PROMPT
    assert "preserve it" in REFINE_PROMPT
    assert "suggested wording is never mandatory" in REFINE_PROMPT


def test_targeted_repair_prompt_is_paragraph_local_and_meaning_preserving() -> None:
    prompt = _targeted_language_repair_prompt(
        "A source sentence.",
        "\u06cc\u06a9 \u062c\u0645\u0644\u0647.",
        {"foreign_script_artifacts": [{"script": "han", "text": "\u5d4c"}]},
        structural_role="body",
    )
    assert "exactly one complete Persian paragraph" in prompt
    assert "Preserve every proposition" in prompt
    assert "Do not choose new terminology" in prompt


def test_objective_artifacts_cannot_become_style_examples() -> None:
    clean = (
        "\u0627\u06cc\u0646 \u0628\u0646\u062f \u0628\u0627 \u0646\u062b\u0631\u06cc \u0631\u0648\u0634\u0646 \u0648 \u062f\u0642\u06cc\u0642\u060c \u0627\u0633\u062a\u062f\u0644\u0627\u0644 \u0646\u0638\u0631\u06cc \u0631\u0627 \u062a\u0648\u0636\u06cc\u062d \u0645\u06cc\u200c\u062f\u0647\u062f."
    )
    corrupt = clean.replace("\u0631\u0648\u0634\u0646", "\u0631\u0648\u5d4c\u0634\u0646")
    assert _clean_style_sample(clean)
    assert _clean_style_sample(corrupt) == ""


def test_anchor_repair_runs_after_inline_canonicalization() -> None:
    source = Path("src/tarjomeh/core/pipeline.py").read_text(encoding="utf-8")
    process_start = source.index("# Canonicalize the accepted text before measuring")
    canonicalize = source.index("_canonicalize_chunk_inline_originals(", process_start)
    missing_required = source.index("missing_required_anchors = [", process_start)
    assert canonicalize < missing_required
