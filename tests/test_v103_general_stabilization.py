from __future__ import annotations

from unittest.mock import MagicMock

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import EmptyCompletionError
from tarjomeh.core.pipeline import (
    TranslationPipeline,
    _explicit_chunk_review_payload,
    _high_confidence_person_candidates,
    _source_entity_categories,
    _source_entity_inventory,
)
from tarjomeh.core.term_notes import reconcile_redundant_original_fragments
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.memory.proper_nouns import (
    ProperNouns,
    is_safe_automatic_entity_mapping,
)
from tarjomeh.persian.typography import PersianTypographer
from tarjomeh.quality.back_translator import BackTranslator
from tarjomeh.quality.integrity import (
    extract_labeled_identifier_surfaces,
    restore_source_identifiers,
)


def test_entity_inventory_stays_on_line_and_requires_person_evidence() -> None:
    source = (
        "Polity Press\n65 Bridge Street Cambridge CB\n"
        "G. W. F. Hegel argued. Economic and Social Science Research "
        "Council funded it. Jupp Esser, who emphasized evidence, replied."
    )

    candidates = _source_entity_inventory(source)
    categories = _source_entity_categories(source, candidates)

    assert "Polity Press 65 Bridge Street" not in candidates
    assert "Economic and Social Science Research Council" in candidates
    assert categories["G. W. F. Hegel"] == "person"
    assert categories["Jupp Esser"] == "person"
    assert "Bridge Street Cambridge CB" not in candidates
    assert _high_confidence_person_candidates(candidates, source) == [
        "G. W. F. Hegel",
        "Jupp Esser",
    ]


def test_low_authority_entity_admission_rejects_noise_and_truncated_names() -> None:
    source = (
        "The Economic and Social Science Research Council supported Jupp "
        "Esser, who led the study. European debates followed."
    )

    assert not is_safe_automatic_entity_mapping(
        "Social Science Research Council",
        "شورای پژوهش علوم اجتماعی",
        "organization",
        source,
    )
    assert not is_safe_automatic_entity_mapping(
        "European", "اروپایی", "place", source
    )
    assert is_safe_automatic_entity_mapping(
        "Jupp Esser", "یوپ اسر", "person", source
    )


def test_first_occurrence_state_requires_visible_accepted_rendering() -> None:
    nouns = ProperNouns()
    nouns.add_noun(
        "Jupp Esser", "یوپ اسر", category="person", provenance="curated_glossary"
    )

    nouns.mark_introduced_from_translation(
        "Jupp Esser developed the argument.", "این استدلال بسط یافت."
    )
    assert not nouns.is_introduced("Jupp Esser")

    nouns.mark_introduced_from_translation(
        "Jupp Esser developed the argument.",
        "یوپ اسر (Jupp Esser) این استدلال را بسط داد.",
    )
    assert nouns.is_introduced("Jupp Esser")


def test_identifier_label_and_payload_survive_typography_and_repair() -> None:
    source = "ISBN-13: 978-0-7456-3304-6"
    assert PersianTypographer().process(source) == source

    damaged = "ISBN- ۱۳: ۹۷۸ - ۰ - ۷۴۵۶ - ۳۳۰۴ - ۶"
    repaired, report = restore_source_identifiers(source, damaged)
    assert repaired == source
    assert report["repair_count"] == 1
    assert extract_labeled_identifier_surfaces(repaired) == {
        source.casefold(): 1
    }


def test_redundant_partial_original_is_removed_but_citation_is_preserved() -> None:
    full = "UK Copyright, Designs and Patents Act 1988"
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(
        index=0,
        source_text=full + "; see Smith (1990).",
        translated_text=(
            f"متن ({full}) (UK, 1988) و ارجاع (Smith, 1990)."
        ),
    )])

    report = reconcile_redundant_original_fragments(document, {full: full})

    assert report["removed_count"] == 1
    assert "(UK, 1988)" not in document.paragraphs[0].translated_text
    assert "(Smith, 1990)" in document.paragraphs[0].translated_text


def test_direct_review_payload_is_canonical() -> None:
    payload = _explicit_chunk_review_payload(
        "missing_first_occurrence_entity_original",
        detail="source_entity_coverage",
        missing=["Jupp Esser"],
    )
    assert payload["reason_codes"] == [
        "missing_first_occurrence_entity_original"
    ]
    assert payload["reasons"][0]["detail"] == "source_entity_coverage"
    assert payload["human_review_required"] is True


def test_low_lexical_overlap_remains_advisory_without_structured_risk() -> None:
    checker = BackTranslator(MagicMock(), similarity_threshold=0.95)
    result = checker.compare("state power", "governmental authority")

    assert result.flagged is False
    assert result.diagnostics["below_similarity_threshold"] is True
    assert result.diagnostics["similarity_policy"] == "advisory_only"


def test_empty_whole_chunk_uses_existing_validated_split_recovery() -> None:
    config = TarjomehConfig()
    config.translation.enable_web_context = False
    config.translation.enable_back_translation = False
    config.translation.enable_critique = False
    config.translation.enable_integrity_gate = False
    config.glossary.enable_compliance_check = False
    config.glossary.enable_auto_extraction = False

    pipeline = object.__new__(TranslationPipeline)
    pipeline.config = config
    pipeline.db = MagicMock()
    pipeline.db.get_job.return_value = {"status": "running"}
    pipeline.llm_client = MagicMock()
    source = "A" * 1000 + ". " + "B" * 1000 + "."
    pipeline.llm_client.complete.side_effect = [
        EmptyCompletionError("provider returned no visible completion"),
        "<<<TRANSLATION c0.p0.s0>>>\n" + "ترجمه دقیق " * 45
        + "\n<<<END c0.p0.s0>>>",
        "<<<TRANSLATION c0.p0.s1>>>\n" + "متن دانشگاهی " * 45
        + "\n<<<END c0.p0.s1>>>",
    ]

    memory_context = MagicMock()
    memory_context.style_profile = ""
    memory_context.proper_nouns = ""
    memory_context.long_term = ""
    memory_context.short_term = ""
    memory_context.bilingual_summary = ""
    memory_context.format.return_value = ""
    memory_manager = MagicMock()
    memory_manager.get_context_for_chunk.return_value = memory_context
    memory_manager.proper_nouns.pending_inline_originals.return_value = []
    glossary = MagicMock()
    glossary.find_terms.return_value = []
    glossary.format_for_prompt.return_value = ""

    result = pipeline._translate_single_chunk(
        idx=0,
        chunk=Chunk(0, source, "", ""),
        memory_manager=memory_manager,
        web_searcher=MagicMock(),
        glossary_manager=glossary,
        compliance_checker=MagicMock(),
        critique_tool=MagicMock(),
        refiner_tool=MagicMock(),
        back_translator=MagicMock(),
        translations={},
        job_id="job-empty-split",
    )

    assert "ترجمه دقیق" in result
    operations = [
        call.kwargs.get("_operation")
        for call in pipeline.llm_client.complete.call_args_list
    ]
    assert operations == [
        "translation",
        "translation_split_recovery",
        "translation_split_recovery",
    ]
    adaptive = [
        call.args[3]
        for call in pipeline.db.log_chunk_event.call_args_list
        if call.args[2] == "translation_adaptive_split"
    ]
    assert adaptive[0]["reason"] == "repeated_empty_completion"
