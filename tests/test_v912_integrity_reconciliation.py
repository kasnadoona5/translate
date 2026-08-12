from __future__ import annotations

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import _advisory_terminology_consistency
from tarjomeh.core.term_notes import ensure_inline_proper_noun_originals
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.persian.typography import PersianTypographer
from tarjomeh.quality.back_translator import BackTranslator
from tarjomeh.quality.integrity import (
    PostEditIntegrityGate,
    mixed_script_artifacts,
    restore_source_identifiers,
)


FA_NEO = "\u0646\u0626\u0648\u067e\u0644\u0648\u0631\u0627\u0644\u06cc\u0633\u0645"


def test_source_identifier_repair_preserves_persian_label() -> None:
    source = "ISBN-13: 978-0-7456-3304-6 (pb)"
    translated = (
        "\u0634\u0627\u0628\u06a9-\u06f1\u06f3: "
        "\u06f9\u06f7\u06f8 - \u06f0 - \u06f7\u06f4\u06f5\u06f6 - "
        "\u06f3\u06f3\u06f0\u06f4 - \u06f6 (pb)"
    )

    repaired, report = restore_source_identifiers(source, translated)

    assert report["repair_count"] == 1
    assert "\u0634\u0627\u0628\u06a9-\u06f1\u06f3:" in repaired
    assert "978-0-7456-3304-6" in repaired
    assert PostEditIntegrityGate().evaluate(source, repaired).accepted


def test_source_bibliographic_marker_is_not_unauthorized_original() -> None:
    source = "ISBN 978-0-7456-3304-6 (pb)"
    candidate = "\u0634\u0627\u0628\u06a9 978-0-7456-3304-6 (pb)"
    result = PostEditIntegrityGate().evaluate(
        source,
        candidate,
        protect_inline_english=True,
        allowed_inline_originals=[],
    )
    assert "unauthorized_english_original_present" not in {
        finding.check_id for finding in result.findings
    }


def test_mixed_script_corruption_is_blocking_and_style_is_not_learned() -> None:
    broken = (
        "\u0627\u06cc\u0646 \u0646\u0647\u0627\u062f sanks"
        "\u0633\u06cc\u0648\u0646 \u0645\u06cc\u200c\u0634\u0648\u062f."
    )
    result = PostEditIntegrityGate().evaluate(
        "This institution is sanctioned and maintained by norms.", broken
    )
    assert mixed_script_artifacts(broken) == ["sanks\u0633\u06cc\u0648\u0646"]
    assert "mixed_script_corruption" in {
        finding.check_id for finding in result.blocking
    }

    manager = MemoryManager(TarjomehConfig())
    chunk = Chunk(
        index=0,
        text="This institution is maintained by norms.",
        chapter_title="Chapter",
        section_title="",
    )
    policy = manager.update_after_translation(chunk, broken)
    assert not policy["style_sample_added"]
    assert manager.style_samples == []


def test_inline_original_follows_productive_persian_suffix() -> None:
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(
        index=0,
        source_text="Neopluralism offers one account.",
        translated_text=f"\u062a\u0628\u06cc\u06cc\u0646 {FA_NEO} (neopluralism)\u06cc \u0627\u0633\u062a.",
    )])
    report = ensure_inline_proper_noun_originals(
        document,
        {"neopluralism": FA_NEO},
        PersianTypographer({"convert_numerals": False}),
        return_report=True,
    )
    assert report["repositioned_count"] == 1
    assert f"{FA_NEO}\u06cc (neopluralism)" in document.paragraphs[0].translated_text
    assert "(neopluralism)\u06cc" not in document.paragraphs[0].translated_text


def test_citation_only_name_does_not_create_missing_target() -> None:
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(
        index=0,
        source_text="The argument is established (Tilly 1975).",
        translated_text="\u0627\u06cc\u0646 \u0627\u0633\u062a\u062f\u0644\u0627\u0644 \u062a\u062b\u0628\u06cc\u062a \u0634\u062f\u0647 \u0627\u0633\u062a (Tilly 1975).",
    )])
    report = ensure_inline_proper_noun_originals(
        document,
        {"Tilly": "\u062a\u06cc\u0644\u06cc"},
        PersianTypographer({"convert_numerals": False}),
        return_report=True,
    )
    assert report["missing_target_count"] == 0
    assert report["citation_only_count"] == 1


def test_back_translation_reconciles_established_persian_entity() -> None:
    result = BackTranslator(object(), sample_pct=100).compare(
        "Michael Mann developed the account.",
        "A scholar developed the account.",
        "\u0645\u0627\u06cc\u06a9\u0644 \u0645\u0646 \u0627\u06cc\u0646 \u062a\u0628\u06cc\u06cc\u0646 \u0631\u0627 \u0628\u0633\u0637 \u062f\u0627\u062f.",
        entity_aliases={"Michael Mann": ["\u0645\u0627\u06cc\u06a9\u0644 \u0645\u0646"]},
    )
    assert "named_entities_missing" not in result.diagnostics["risk_flags"]
    assert result.diagnostics["entities_reconciled_by_memory"] == [{
        "source": "Michael Mann",
        "target": "\u0645\u0627\u06cc\u06a9\u0644 \u0645\u0646",
    }]


def test_consistency_check_is_advisory_and_does_not_rewrite() -> None:
    manager = MemoryManager(TarjomehConfig())
    manager.proper_nouns.add_noun(
        "state system",
        "\u0646\u0638\u0627\u0645 \u062f\u0648\u0644\u062a\u06cc",
        category="term",
        provenance="auto_extraction",
    )
    translation = "\u0633\u0627\u0645\u0627\u0646 \u062f\u0648\u0644\u062a\u200c\u0647\u0627 \u0628\u0631\u0631\u0633\u06cc \u0645\u06cc\u200c\u0634\u0648\u062f."
    report = _advisory_terminology_consistency(
        manager, "The state system is examined.", translation
    )
    assert report["inconsistent_count"] == 1
    assert translation == "\u0633\u0627\u0645\u0627\u0646 \u062f\u0648\u0644\u062a\u200c\u0647\u0627 \u0628\u0631\u0631\u0633\u06cc \u0645\u06cc\u200c\u0634\u0648\u062f."
