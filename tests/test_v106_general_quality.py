"""General quality regressions for v10.6.

Persian literals are escaped so the tests remain portable across Windows and
Linux shells with different console encodings.
"""
# ruff: noqa: E501

from __future__ import annotations

import asyncio
import json

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import (
    _chunk_style_approved,
    _source_entity_inventory,
    audit_translation_language,
)
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.memory.proper_nouns import (
    has_exact_bilingual_anchor,
    is_safe_automatic_entity_mapping,
    is_safe_automatic_source_span,
)
from tarjomeh.persian.typography import PersianTypographer
from tarjomeh.quality.integrity import (
    repeated_persian_word_artifacts,
    restore_source_identifiers,
    unexpected_latin_prose,
)
from tarjomeh.quality.structure_audit import audit_structure


class _EventDatabase:
    def __init__(self, events: list[dict]) -> None:
        self.events = events

    def get_chunk_events(self, _job_id: str, _chunk_index: int) -> list[dict]:
        return self.events


class _StructuredLLM:
    def __init__(self, payload: list[dict]) -> None:
        self.payload = payload

    def set_operation(self, _operation: str) -> None:
        return None

    async def chat(self, _prompt: str) -> str:
        return json.dumps(self.payload, ensure_ascii=False)


def _scores() -> dict[str, float]:
    return {
        "accuracy": 9,
        "fluency": 9,
        "terminology": 9,
        "register": 9,
        "average": 9,
    }


def test_source_entity_inventory_rejects_cross_line_fragments() -> None:
    source = (
        "Toppan Best-set Premedia Limited Printed in the United Kingdom. "
        "Neither the ESRC nor the authors approved it. "
        "Bridge Street Cambridge CB2 1UR United Kingdom. "
        "Profes-sorial Research Fellowship supported the work."
    )

    inventory = _source_entity_inventory(source)

    assert "Toppan Best-set Premedia Limited" in inventory
    assert "Toppan Best-set Premedia Limited Printed" not in inventory
    assert all(not item.startswith("Neither") for item in inventory)
    assert all("Bridge Street" not in item for item in inventory)
    assert all("Profes-sorial" not in item for item in inventory)


def test_source_span_guard_is_category_aware_and_general() -> None:
    assert not is_safe_automatic_source_span(
        "Neither the Research Council", "organization"
    )
    assert not is_safe_automatic_source_span(
        "Main Street Cambridge", "place"
    )
    assert is_safe_automatic_source_span(
        "Toppan Best-set Premedia Limited", "organization"
    )


def test_observed_organization_anchor_rejects_truncated_persian_mapping() -> None:
    english = "Economic and Social Science Research Council"
    partial = (
        "\u067e\u0698\u0648\u0647\u0634 \u0627\u0642\u062a\u0635\u0627\u062f\u06cc \u0648 "
        "\u0639\u0644\u0648\u0645 \u0627\u062c\u062a\u0645\u0627\u0639\u06cc"
    )
    complete = "\u0634\u0648\u0631\u0627\u06cc " + partial
    translation = f"{complete} ({english})"

    assert not has_exact_bilingual_anchor(
        translation, english, partial, "organization"
    )
    assert has_exact_bilingual_anchor(
        translation, english, complete, "organization"
    )


def test_incremental_memory_uses_exact_complete_anchor() -> None:
    english = "Economic and Social Science Research Council"
    partial = (
        "\u067e\u0698\u0648\u0647\u0634 \u0627\u0642\u062a\u0635\u0627\u062f\u06cc \u0648 "
        "\u0639\u0644\u0648\u0645 \u0627\u062c\u062a\u0645\u0627\u0639\u06cc"
    )
    complete = "\u0634\u0648\u0631\u0627\u06cc " + partial
    manager = MemoryManager(TarjomehConfig())
    llm = _StructuredLLM([{
        "term": english,
        "suggested_persian": partial,
        "category": "organization",
    }])

    result = asyncio.run(manager.update_proper_nouns(
        llm,
        f"The {english} supported the project.",
        f"{complete} ({english}) \u0627\u0632 \u067e\u0631\u0648\u0698\u0647 \u062d\u0645\u0627\u06cc\u062a \u06a9\u0631\u062f.",
    ))

    assert result["accepted_count"] == 0
    assert english not in manager.proper_nouns.all_nouns()


def test_ordinary_terms_are_not_misclassified_as_unsafe_entities() -> None:
    assert is_safe_automatic_entity_mapping(
        "statehood",
        "\u062f\u0648\u0644\u062a\u200c\u0645\u0646\u062f\u06cc",
        "term",
        "The problem of statehood matters.",
    )


def test_minor_only_high_quality_critique_can_seed_style() -> None:
    db = _EventDatabase([
        {"event_type": "chunk_started", "payload": {}},
        {
            "event_type": "critique_completed",
            "payload": {
                "valid": True,
                "blocking_issue_count": 0,
                "issue_count": 1,
                "issue_details": [{
                    "severity": "minor",
                    "category": "fluency",
                }],
                "scores": _scores(),
            },
        },
    ])

    assert _chunk_style_approved(db, "job", 0)


def test_major_issue_still_cannot_seed_style() -> None:
    db = _EventDatabase([
        {"event_type": "chunk_started", "payload": {}},
        {
            "event_type": "critique_completed",
            "payload": {
                "valid": True,
                "blocking_issue_count": 0,
                "issue_count": 1,
                "issue_details": [{
                    "severity": "major",
                    "category": "accuracy",
                }],
                "scores": _scores(),
            },
        },
    ])

    assert not _chunk_style_approved(db, "job", 0)


def test_identifier_repair_preserves_local_source_label_identity() -> None:
    payload = "978-0-7456-3304-6"
    source = f"ISBN-13: {payload}. Later record: ISBN {payload}."
    target = f"ISBN {payload}"

    repaired, report = restore_source_identifiers(source, target)

    assert repaired == target
    assert report["repair_count"] == 0


def test_latin_catalog_line_keeps_latin_punctuation_and_digits() -> None:
    typographer = PersianTypographer({
        "normalize_zwnj": False,
        "convert_numerals": True,
        "fix_punctuation": True,
    })

    assert typographer.process("1. State, The. I. Title.") == (
        "1. State, The. I. Title."
    )
    assert typographer.process("\u06f1. State\u060c The. I. Title.") == (
        "1. State, The. I. Title."
    )


def test_persian_prose_reference_uses_persian_digit() -> None:
    typographer = PersianTypographer({
        "normalize_zwnj": False,
        "convert_numerals": True,
        "fix_punctuation": False,
    })

    assert "\u0641\u0635\u0644 \u06f3" in typographer.process(
        "\u0631.\u06a9. \u0641\u0635\u0644 3"
    )


def test_untranslated_foreign_phrase_is_reviewed_but_apostrophe_phrase_is_not() -> None:
    source = "The account studies the very longue duree and raison d'etat."
    target = (
        "\u0627\u06cc\u0646 \u062a\u0628\u06cc\u06cc\u0646 very longue duree \u0648 "
        "raison d'etat \u0631\u0627 \u0628\u0631\u0631\u0633\u06cc \u0645\u06cc\u200c\u06a9\u0646\u062f."
    )

    findings = unexpected_latin_prose(source, target)

    assert {item["token"] for item in findings} >= {"very", "longue", "duree"}
    assert not {"raison", "d", "etat"}.intersection(
        item["token"] for item in findings
    )


def test_adjacent_repeated_persian_word_is_report_only_review_evidence() -> None:
    target = (
        "\u0627\u06cc\u0646 \u0631\u0648\u0627\u06cc\u062a \u062a\u0627\u0631\u06cc\u062e \u062a\u0627\u0631\u06cc\u062e "
        "\u062f\u0648\u0644\u062a \u0631\u0627 \u0628\u0631\u0631\u0633\u06cc \u0645\u06cc\u200c\u06a9\u0646\u062f."
    )

    findings = repeated_persian_word_artifacts(target)
    report = audit_translation_language("This studies history.", target)

    assert [item["word"] for item in findings] == ["\u062a\u0627\u0631\u06cc\u062e"]
    assert report["review_required"] is True
    assert report["repeated_word_count"] == 1


def test_persian_enumeration_direction_is_not_a_false_mismatch() -> None:
    source = (
        "The approach differs in three ways. First, scope; second, method; "
        "third, time."
    )
    target = (
        "\u0627\u06cc\u0646 \u0631\u0648\u06cc\u06a9\u0631\u062f \u0627\u0632 \u0633\u0647 \u062c\u0647\u062a \u062a\u0641\u0627\u0648\u062a \u062f\u0627\u0631\u062f. "
        "\u0646\u062e\u0633\u062a\u060c \u062f\u0627\u0645\u0646\u0647\u061b \u062f\u0648\u0645\u060c \u0631\u0648\u0634\u061b "
        "\u0633\u0648\u0645\u060c \u0632\u0645\u0627\u0646."
    )

    assert audit_structure(source, target) == []
