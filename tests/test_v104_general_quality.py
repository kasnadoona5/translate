from __future__ import annotations

from unittest.mock import MagicMock

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import EmptyCompletionError
from tarjomeh.core.pipeline import (
    TranslationPipeline,
    _high_confidence_instrument_candidates,
    _reconcile_current_entity_anchors,
    _source_entity_categories,
    _source_entity_inventory,
)
from tarjomeh.memory.proper_nouns import (
    ProperNouns,
    is_safe_automatic_entity_mapping,
    source_term_present,
)
from tarjomeh.quality.integrity import (
    restore_source_identifiers,
    unexpected_latin_prose,
)
from tarjomeh.quality.structure_audit import audit_structure


def test_source_term_matching_tolerates_layout_and_diacritic_variants() -> None:
    cases = (
        ("The State: Past, Present, Future", "The State - Past, Present, Future"),
        ("UK Copyright, Designs and Patents Act 1988", "UK Copyright Designs and Patents Act 1988"),
        ("Alex Demirovi\u0107", "Alex Demirovic"),
        ("G. W. F. Hegel", "G W F Hegel"),
        ("strategic-relational perspective", "strategic\u2013relational perspective"),
    )
    assert all(source_term_present(source, term) for term, source in cases)


def test_low_authority_memory_requires_clean_or_observed_entity_evidence() -> None:
    coordinated_target = (
        "\u0627\u0642\u06cc\u0627\u0646\u0648\u0633 \u0627\u0637\u0644\u0633 "
        "\u0634\u0645\u0627\u0644\u06cc \u0648 \u0645\u0646\u0637\u0642\u0647 "
        "\u06cc\u0648\u0631\u0648"
    )
    assert not is_safe_automatic_entity_mapping(
        "Profes-sorial Research Fellowship",
        "\u0628\u0648\u0631\u0633 \u067e\u0698\u0648\u0647\u0634\u06cc",
        "organization",
        "Profes-sorial Research Fellowship",
    )
    assert not is_safe_automatic_entity_mapping(
        "North Atlantic and Eurozone",
        coordinated_target,
        "place",
        "North Atlantic and Eurozone",
    )
    anchored = coordinated_target + " (North Atlantic and Eurozone)"
    assert is_safe_automatic_entity_mapping(
        "North Atlantic and Eurozone",
        coordinated_target,
        "place",
        "North Atlantic and Eurozone",
        translation=anchored,
        require_observed_anchor=True,
    )


def test_observed_name_anchor_survives_source_diacritic_difference() -> None:
    memory = MagicMock()
    memory.proper_nouns = ProperNouns()
    source = "Alex Demirovi\u0107 developed this argument."
    translation = (
        "\u0627\u0644\u06a9\u0633 \u062f\u0645\u06cc\u0631\u0648\u0648\u06cc\u0686 "
        "(Alex Demirovic) \u0627\u06cc\u0646 \u0627\u0633\u062a\u062f\u0644\u0627\u0644 "
        "\u0631\u0627 \u0628\u0633\u0637 \u062f\u0627\u062f."
    )

    result = _reconcile_current_entity_anchors(
        memory,
        ["Alex Demirovi\u0107"],
        translation,
        {"Alex Demirovi\u0107": "source_entity_candidate"},
        source,
    )

    assert result["observed_count"] == 1
    assert memory.proper_nouns.all_nouns()["Alex Demirovi\u0107"] == (
        "\u0627\u0644\u06a9\u0633 \u062f\u0645\u06cc\u0631\u0648\u0648\u06cc\u0686"
    )


def test_named_instrument_is_one_required_source_entity() -> None:
    source = (
        "USA PATRIOT Act Uniting and Strengthening America by Providing "
        "Appropriate Tools Required to Intercept and Obstruct Terrorism (2001)"
    )
    candidates = _source_entity_inventory(source)
    categories = _source_entity_categories(source, candidates)

    assert "USA PATRIOT Act" in candidates
    assert categories["USA PATRIOT Act"] == "publication"
    assert _high_confidence_instrument_candidates(candidates) == ["USA PATRIOT Act"]


def test_structure_audit_does_not_mix_unrelated_lists() -> None:
    source = (
        "There are two objections. First it is circular; second it is vague.\n\n"
        "First Alice was thanked; second Bob; third Carol."
    )
    candidate = (
        "\u062f\u0648 \u0627\u06cc\u0631\u0627\u062f \u0648\u062c\u0648\u062f "
        "\u062f\u0627\u0631\u062f. "
        "\u0646\u062e\u0633\u062a \u062f\u0648\u0631\u06cc \u0627\u0633\u062a\u061b "
        "\u062f\u0648\u0645 \u0645\u0628\u0647\u0645 \u0627\u0633\u062a.\n\n"
        "\u0646\u062e\u0633\u062a \u0627\u0644\u06cc\u0633\u061b "
        "\u062f\u0648\u0645 \u0628\u0627\u0628\u061b "
        "\u0633\u0648\u0645 \u06a9\u0627\u0631\u0648\u0644."
    )
    assert audit_structure(source, candidate) == []


def test_structure_audit_binds_announcement_to_next_list_paragraph() -> None:
    source = "There are two objections.\n\nFirst it is circular; second it is vague."
    candidate = (
        "\u062f\u0648 \u0627\u06cc\u0631\u0627\u062f \u0648\u062c\u0648\u062f "
        "\u062f\u0627\u0631\u062f.\n\n"
        "\u0646\u062e\u0633\u062a \u062f\u0648\u0631\u06cc \u0627\u0633\u062a\u061b "
        "\u062f\u0648\u0645 \u0645\u0628\u0647\u0645 \u0627\u0633\u062a."
    )
    assert audit_structure(source, candidate) == []


def test_source_grounded_foreign_phrase_allows_apostrophe_variant() -> None:
    assert (
        unexpected_latin_prose(
            "The concept is raison d\u2019\u00e9tat.",
            "\u0627\u06cc\u0646 \u0645\u0641\u0647\u0648\u0645 "
            "raison d'\u00e9tat \u0627\u0633\u062a.",
        )
        == []
    )


def test_identifier_repair_removes_duplicated_localized_label() -> None:
    source = "ISBN-13: 978-0-7456-3304-6"
    candidate = "\u0634\u0627\u0628\u06a9-\u06f1\u06f3: ISBN-13: 978-0-7456-3304-6"

    repaired, report = restore_source_identifiers(source, candidate)

    assert repaired == source
    assert report["repair_count"] == 1


def test_adaptive_recovery_records_final_paragraph_protocol_evidence() -> None:
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
    pipeline.llm_client.complete.side_effect = [
        EmptyCompletionError("empty whole chunk"),
        "<<<TRANSLATION c0.p0>>>\n"
        "\u062a\u0631\u062c\u0645\u0647 \u0628\u0646\u062f \u0627\u0648\u0644.\n"
        "<<<END c0.p0>>>",
        "<<<TRANSLATION c0.p1>>>\n"
        "\u062a\u0631\u062c\u0645\u0647 \u0628\u0646\u062f \u062f\u0648\u0645.\n"
        "<<<END c0.p1>>>",
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
        chunk=Chunk(
            0,
            "First source paragraph.\n\nSecond source paragraph.",
            "",
            "",
            metadata={
                "paragraph_indices": [0, 1],
                "paragraph_protocol_version": 1,
            },
        ),
        memory_manager=memory_manager,
        web_searcher=MagicMock(),
        glossary_manager=glossary,
        compliance_checker=MagicMock(),
        critique_tool=MagicMock(),
        refiner_tool=MagicMock(),
        back_translator=MagicMock(),
        translations={},
        job_id="job-v104-recovery",
    )

    assert result.count("\n\n") == 1
    evidence = [
        call.args[3]
        for call in pipeline.db.log_chunk_event.call_args_list
        if call.args[2] == "paragraph_protocol_checked" and call.args[3].get("recovery_assembly")
    ]
    assert evidence == [
        {
            "stage": "adaptive_recovery_assembly",
            "valid": True,
            "expected_paragraphs": 2,
            "candidate_paragraphs": 2,
            "recovery_assembly": True,
        }
    ]
