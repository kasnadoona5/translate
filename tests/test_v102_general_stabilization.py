from __future__ import annotations

from unittest.mock import MagicMock

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import (
    TranslationPipeline,
    _high_confidence_person_candidates,
    _observed_anchor_target,
    _source_entity_inventory,
)
from tarjomeh.core.term_notes import audit_inline_english_originals
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.memory.proper_nouns import ProperNouns
from tarjomeh.quality.integrity import (
    source_abbreviation_expansions,
    unexpected_latin_prose,
)


def test_possessive_citation_name_uses_canonical_entity_identity() -> None:
    source = "Shmuel Eisenstadt's (1963) study changed the debate."

    candidates = _source_entity_inventory(source)

    assert candidates == ["Shmuel Eisenstadt"]
    assert _high_confidence_person_candidates(candidates, source) == [
        "Shmuel Eisenstadt"
    ]
    assert unexpected_latin_prose(
        source,
        "\u0634\u0645\u0648\u0626\u0644 \u0622\u06cc\u0632\u0646\u0634\u062a\u0627\u062a "
        "(Shmuel Eisenstadt) (1963) \u0645\u0647\u0645 \u0627\u0633\u062a.",
    ) == []


def test_observed_anchor_accepts_local_citation_but_rejects_context_capture() -> None:
    assert _observed_anchor_target(
        "\u0627\u062b\u0631 \u0634\u0645\u0648\u0626\u0644 "
        "\u0622\u06cc\u0632\u0646\u0634\u062a\u0627\u062a "
        "(Shmuel Eisenstadt, 1963) \u0645\u0647\u0645 \u0627\u0633\u062a.",
        "Shmuel Eisenstadt",
        "person",
    ) == "\u0634\u0645\u0648\u0626\u0644 \u0622\u06cc\u0632\u0646\u0634\u062a\u0627\u062a"
    assert _observed_anchor_target(
        "\u0627\u06cc\u0646 \u0628\u062d\u062b \u0645\u0647\u0645 \u0627\u0633\u062a.\n"
        "\u0627\u062b\u0631 (Shmuel Eisenstadt) \u0645\u0647\u0645 \u0627\u0633\u062a.",
        "Shmuel Eisenstadt",
        "person",
    ) == ""


def test_observed_memory_rejects_partial_or_sentence_sized_targets() -> None:
    nouns = ProperNouns()

    partial = nouns.add_noun(
        "Pascal Porcheron",
        "\u067e\u0648\u0631\u0634\u0631\u0646",
        category="person",
        provenance="observed_translation",
    )
    contextual = nouns.add_noun(
        "Den Haag",
        "\u0645\u06cc\u200c\u06a9\u0646\u0645. \u062f\u0646 \u0647\u0627\u062e",
        category="place",
        provenance="observed_translation",
    )
    curated = nouns.add_noun(
        "Pascal Porcheron",
        "\u067e\u0627\u0633\u06a9\u0627\u0644 \u067e\u0648\u0631\u0634\u0631\u0648\u0646",
        category="person",
        provenance="curated_glossary",
    )

    assert partial["action"] == "ignored"
    assert contextual["action"] == "ignored"
    assert curated["action"] == "added"


def test_source_defined_acronym_expansion_survives_inline_audit() -> None:
    expansion = (
        "Uniting and Strengthening America by Providing Appropriate Tools "
        "Required to Intercept and Obstruct Terrorism"
    )
    source = f"USA PATRIOT Act {expansion} (2001)"
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(
        index=0,
        source_text=source,
        translated_text=(
            "\u0642\u0627\u0646\u0648\u0646 "
            "\u0645\u06cc\u0647\u0646\u200c\u062f\u0648\u0633\u062a\u06cc "
            f"({expansion}) (2001)"
        ),
    )])

    assert source_abbreviation_expansions(source) == [expansion]
    report = audit_inline_english_originals(document, {})

    assert expansion in document.paragraphs[0].translated_text
    assert report["removed_unauthorized_count"] == 0


def test_invalid_initial_draft_gets_bounded_full_source_repair() -> None:
    config = TarjomehConfig()
    config.translation.enable_web_context = False
    config.translation.enable_back_translation = False
    config.translation.enable_critique = False
    config.translation.enable_integrity_gate = True
    config.glossary.enable_compliance_check = False
    config.glossary.enable_auto_extraction = False

    pipeline = object.__new__(TranslationPipeline)
    pipeline.config = config
    pipeline.db = MagicMock()
    pipeline.db.get_job.return_value = {"status": "running"}
    pipeline.llm_client = MagicMock()
    pipeline.llm_client.complete.side_effect = [
        "\u0645\u0642\u062f\u0627\u0631 \u062b\u0628\u062a \u0634\u062f.",
        (
            "\u0645\u0642\u062f\u0627\u0631 \u062f\u0631 \u0633\u0627\u0644 2004 "
            "\u062b\u0628\u062a \u0634\u062f."
        ),
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
        chunk=Chunk(0, "The value was recorded in 2004.", "", ""),
        memory_manager=memory_manager,
        web_searcher=MagicMock(),
        glossary_manager=glossary,
        compliance_checker=MagicMock(),
        critique_tool=MagicMock(),
        refiner_tool=MagicMock(),
        back_translator=MagicMock(),
        translations={},
        job_id="job-integrity-repair",
    )

    assert "2004" in result
    operations = [
        call.kwargs.get("_operation")
        for call in pipeline.llm_client.complete.call_args_list
    ]
    assert operations == ["translation", "translation_integrity_repair"]
    pipeline.llm_client.limit_next_call_attempts.assert_called_once_with(2)
    event_types = [
        call.args[2] for call in pipeline.db.log_chunk_event.call_args_list
    ]
    assert "translation_baseline_quarantined" in event_types
    assert "translation_integrity_repair" in event_types
    assert "translation_baseline_repaired" in event_types
    assert "integrity_final_failed" not in event_types


def test_academic_contract_prefers_transparent_established_persian() -> None:
    from tarjomeh.core.prompts import ACADEMIC_REGISTER_MODIFIER

    assert "Prefer established, transparent Persian academic equivalents" in (
        ACADEMIC_REGISTER_MODIFIER
    )
    assert "no precise Persian equivalent exists" in ACADEMIC_REGISTER_MODIFIER
