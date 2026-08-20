from __future__ import annotations

from tarjomeh.core.pipeline import (
    _high_confidence_person_candidates,
    _observed_anchor_target,
    _source_entity_inventory,
)
from tarjomeh.memory.manager import _clean_style_sample
from tarjomeh.memory.proper_nouns import (
    ProperNouns,
    looks_like_transliterated_loanword,
)
from tarjomeh.parsers.base import Paragraph
from tarjomeh.parsers.pdf_parser import _merge_table_interrupted_continuations
from tarjomeh.quality.integrity import mixed_script_artifacts


def test_current_source_entity_inventory_is_bounded_and_semantic() -> None:
    source = (
        "Jupp Esser, who insisted on evidence, worked with Nicos Poulantzas "
        "and Joachim Hirsch. This Approach differs from Western Europe."
    )

    candidates = _source_entity_inventory(source)

    assert "Jupp Esser" in candidates
    assert "Nicos Poulantzas" in candidates
    assert "Joachim Hirsch" in candidates
    assert "Western Europe" in candidates
    assert "This Approach" not in candidates
    assert _high_confidence_person_candidates(candidates) == [
        "Jupp Esser", "Nicos Poulantzas", "Joachim Hirsch"
    ]


def test_observed_entity_anchor_uses_only_adjacent_persian_name() -> None:
    assert _observed_anchor_target(
        "مطالعات نیکوس پولانتزاس (Nicos Poulantzas) مهم است.",
        "Nicos Poulantzas",
    ) == "نیکوس پولانتزاس"
    assert _observed_anchor_target(
        "نیکوس پولانتزاس در این بحث مهم است.",
        "Nicos Poulantzas",
    ) == ""


def test_lowercase_brand_mapping_does_not_override_ordinary_prose() -> None:
    nouns = ProperNouns()
    nouns.add_noun(
        "polity",
        "پالیتی",
        category="organization",
        provenance="auto_extraction",
    )

    assert "polity ->" not in nouns.get_context(
        source_text="The polity, politics, and policy are distinct concepts."
    )
    assert "polity -> پالیتی" in nouns.get_context(source_text="polity")


def test_transliterated_specialist_labels_are_inline_eligible() -> None:
    assert looks_like_transliterated_loanword("dispositif", "دیسپوزیتیف")
    assert looks_like_transliterated_loanword("assemblage", "آسامبلاژ")
    assert not looks_like_transliterated_loanword("state", "دولت")

    nouns = ProperNouns()
    nouns.add_noun("assemblage", "آسامبلاژ", category="term")
    assert "assemblage" in nouns.pending_inline_originals(
        "The state is an assemblage of institutions."
    )


def test_unusable_mappings_never_enter_or_resume_memory() -> None:
    nouns = ProperNouns()
    outcome = nouns.add_noun(
        "politybooks.com",
        "politybooks.com",
        category="product",
        provenance="auto_extraction",
    )
    assert outcome["action"] == "ignored"
    assert not nouns.all_nouns()

    nouns.deserialize({
        "nouns": {
            "politybooks.com": "politybooks.com",
            "Jupp Esser": "یوپ اسر",
        },
        "categories": {
            "politybooks.com": "product",
            "Jupp Esser": "person",
        },
    })
    assert nouns.all_nouns() == {"Jupp Esser": "یوپ اسر"}


def test_table_interrupted_sentence_is_rejoined_without_losing_table() -> None:
    left = Paragraph(
        "These studies consider the scope they give for various",
        metadata={"page": 8, "structure_role": "body", "source_fragment_ids": ["a"]},
    )
    table = Paragraph(
        "Table 1 | Perspective | Focus",
        metadata={"page": 9, "is_table": True, "structure_role": "table"},
    )
    right = Paragraph(
        "kinds of individual and collective agent to make a difference.",
        metadata={"page": 10, "structure_role": "body", "source_fragment_ids": ["b"]},
    )

    merged = _merge_table_interrupted_continuations([left, table, right])

    assert len(merged) == 2
    assert merged[0].text.endswith("collective agent to make a difference.")
    assert merged[0].metadata["cross_table_join"] is True
    assert merged[1].text.startswith("Table 1")
    assert merged[1].metadata["relocated_after_continuation"] is True


def test_parenthetical_suffix_corruption_cannot_teach_style() -> None:
    malformed = "این بحث درباره اروپا (Europe)ی غربی ادامه می‌یابد."

    assert "(Europe)ی" in mixed_script_artifacts(malformed)
    assert _clean_style_sample(malformed) == ""
    assert _clean_style_sample("… این فقط نیمه‌ای از یک بند است.") == ""
