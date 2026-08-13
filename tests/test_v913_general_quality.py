from __future__ import annotations

from types import SimpleNamespace

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.context.search_providers import SearchResult, rank_search_results
from tarjomeh.core.pipeline import _reconcile_committed_terminology
from tarjomeh.core.term_notes import (
    ensure_inline_proper_noun_originals,
    normalize_adjacent_original_citations,
)
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.exporters.docx_exporter import directional_target_parts
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.persian.typography import PersianTypographer
from tarjomeh.quality.back_translator import BackTranslator
from tarjomeh.quality.integrity import PostEditIntegrityGate, mixed_script_artifacts


def _document(source: str, target: str) -> TranslatedDocument:
    return TranslatedDocument(paragraphs=[TranslatedParagraph(
        index=0,
        source_text=source,
        translated_text=target,
    )])


def test_mixed_script_detection_requires_letters_from_both_scripts() -> None:
    assert mixed_script_artifacts("Preface viii\u061b Tables x\u061b") == []
    assert mixed_script_artifacts("sanks\u0633\u06cc\u0648\u0646") == [
        "sanks\u0633\u06cc\u0648\u0646"
    ]


def test_translated_citation_wrapper_preserves_semantic_integrity() -> None:
    gate = PostEditIntegrityGate()
    source = (
        "The argument is developed elsewhere (see chapter 4) and "
        "documented by Marx (Marx, 1973)."
    )
    previous = (
        "\u0627\u06cc\u0646 \u0627\u0633\u062a\u062f\u0644\u0627\u0644 \u062f\u0631 \u062c\u0627\u06cc \u062f\u06cc\u06af\u0631\u06cc \u0628\u0633\u0637 \u06cc\u0627\u0641\u062a\u0647 \u0627\u0633\u062a "
        "(see chapter 4) \u0648 \u0645\u0627\u0631\u06a9\u0633 \u0622\u0646 \u0631\u0627 \u062b\u0628\u062a \u06a9\u0631\u062f\u0647 \u0627\u0633\u062a (Marx, 1973)."
    )
    candidate = previous.replace("(see chapter 4)", "(\u0631.\u06a9. \u0641\u0635\u0644 \u06f4)")
    result = gate.evaluate(
        source,
        candidate,
        previous=previous,
        protect_inline_english=True,
        allowed_inline_originals=[],
    )
    blocking = {finding.check_id for finding in result.blocking}
    assert "source_citation_removed" not in blocking
    assert "numbers_missing" not in blocking


def test_first_occurrence_anchor_allows_bounded_persian_modifier() -> None:
    document = _document(
        "A traditional three-element theory is introduced.",
        "\u062f\u0631 \u0627\u06cc\u0646\u062c\u0627 \u0646\u0638\u0631\u06cc\u0647 \u0633\u0646\u062a\u06cc \u0633\u0647\u200c\u0639\u0646\u0635\u0631\u06cc \u0645\u0639\u0631\u0641\u06cc \u0645\u06cc\u200c\u0634\u0648\u062f.",
    )
    report = ensure_inline_proper_noun_originals(
        document,
        {"three-element theory": "\u0646\u0638\u0631\u06cc\u0647 \u0633\u0647\u200c\u0639\u0646\u0635\u0631\u06cc"},
        PersianTypographer({}),
        {"three-element theory": "theory"},
        return_report=True,
    )
    assert "\u0646\u0638\u0631\u06cc\u0647 \u0633\u0646\u062a\u06cc \u0633\u0647\u200c\u0639\u0646\u0635\u0631\u06cc (three-element theory)" in document.paragraphs[0].translated_text
    assert report["missing_target_count"] == 0


def test_name_year_and_citation_house_style_are_source_grounded() -> None:
    document = _document(
        "Shmuel Eisenstadt's (1963) work is discussed (see chapter 4).",
        "\u0627\u062b\u0631 \u0634\u0645\u0648\u0626\u0644 \u0622\u06cc\u0632\u0646\u0634\u062a\u0627\u062a (Shmuel Eisenstadt) "
        "(Eisenstadt 1963) \u0628\u0631\u0631\u0633\u06cc \u0645\u06cc\u200c\u0634\u0648\u062f (see chapter 4). "
        "\u0631. \u06a9. \u062c\u062f\u0648\u0644 \u06f1 \u066b \u06f1 \u0648 \u0645\u0642\u062f\u0627\u0631 \u06f1\u066b\u06f5.",
    )
    report = normalize_adjacent_original_citations(
        document,
        {"Shmuel Eisenstadt": "\u0634\u0645\u0648\u0626\u0644 \u0622\u06cc\u0632\u0646\u0634\u062a\u0627\u062a"},
    )
    text = document.paragraphs[0].translated_text
    assert "(Shmuel Eisenstadt, 1963)" in text
    assert "(\u0631.\u06a9. \u0641\u0635\u0644 4)" in text
    assert "\u0631.\u06a9. \u062c\u062f\u0648\u0644 \u06f1.\u06f1" in text
    assert "\u06f1\u066b\u06f5" in text
    assert report["normalized_count"] >= 3


def test_docx_direction_parts_keep_mixed_citation_content_intact() -> None:
    value = "\u0645\u062a\u0646 (\u062f\u0631\u0628\u0627\u0631\u0647 \u0645\u0646\u0638\u0631\u0647\u0627\u060c \u0631.\u06a9. Lukacs 1971\u061b Althusser 2006)."
    parts = directional_target_parts(value)
    assert "".join(part for part, _rtl in parts) == value
    assert any(rtl for _part, rtl in parts)
    assert any(not rtl for _part, rtl in parts)


def test_back_translation_reconciles_localized_structural_number() -> None:
    result = BackTranslator(object()).compare(
        "Finer produced a 3-volume study of government.",
        "Finer produced a major study of government.",
        "\u0641\u0627\u06cc\u0646\u0631 \u0645\u0637\u0627\u0644\u0639\u0647\u200c\u0627\u06cc \u0633\u0647\u200c\u062c\u0644\u062f\u06cc \u062f\u0631\u0628\u0627\u0631\u0647 \u062d\u06a9\u0648\u0645\u062a \u0627\u0646\u062c\u0627\u0645 \u062f\u0627\u062f.",
    )
    assert "numbers_missing_or_changed" not in result.diagnostics["risk_flags"]
    assert result.diagnostics["numbers_reconciled_by_translation"] == ["3"]


def test_front_matter_does_not_create_named_entity_or_proposition_risks() -> None:
    source = (
        "In Memoriam\nCopyright 2020 Polity Press\nAll rights reserved\n"
        "ISBN 978-1-23456-789-0\nwww.example.com"
    )
    result = BackTranslator(object()).compare(
        source,
        "Copyright 2020. All rights reserved. ISBN 978-1-23456-789-0.",
    )
    assert result.diagnostics["qa_risk_skipped_for_non_prose_front_matter"]
    assert "named_entities_missing" not in result.diagnostics["risk_flags"]
    assert "possible_proposition_omission" not in result.diagnostics["risk_flags"]


def test_committed_subterm_defers_only_automatic_container_memory() -> None:
    class FakeDB:
        def get_chunk_events(self, _job: str, _chunk: int):
            return [{"event_type": "chunk_started", "payload": {}}]

        def get_qa_issues(self, _job: str, _chunk: int):
            return [{
                "issue_id": "term-1",
                "category": "terminology",
                "source_quote": "Patents",
                "suggested_correction": "\u062d\u0642 \u0627\u062e\u062a\u0631\u0627\u0639",
            }]

        def get_issue_decisions(self, _job: str, _chunk: int):
            return [{
                "issue_id": "term-1",
                "payload": {"commit_status": "committed_full_candidate"},
                "resulting_span": "\u062d\u0642 \u0627\u062e\u062a\u0631\u0627\u0639",
            }]

    memory = MemoryManager(TarjomehConfig())
    container = "UK Copyright, Designs and Patents Act 1988"
    memory.proper_nouns.add_noun(
        container,
        "\u0642\u0627\u0646\u0648\u0646 \u062d\u0642 \u0627\u0645\u062a\u06cc\u0627\u0632",
        category="term",
        provenance="auto_extraction",
    )
    report = _reconcile_committed_terminology(
        FakeDB(), "job", 0, memory,
        f"The {container} protects Patents.",
        "\u0627\u06cc\u0646 \u0642\u0627\u0646\u0648\u0646 \u0627\u0632 \u062d\u0642 \u0627\u062e\u062a\u0631\u0627\u0639 \u062d\u0645\u0627\u06cc\u062a \u0645\u06cc\u200c\u06a9\u0646\u062f.",
    )
    assert memory.proper_nouns.all_nouns()["Patents"] == "\u062d\u0642 \u0627\u062e\u062a\u0631\u0627\u0639"
    assert report["context_deferred"]
    assert container not in memory.proper_nouns.get_context()


def test_research_identity_requires_ordered_title_or_corroboration() -> None:
    title = "The State Past Present Future"
    author = "Bob Jessop"
    results = [
        SearchResult(
            "Present State of Future Markets",
            "https://example.com/unrelated",
            "A market report about past performance.",
        ),
        SearchResult(
            "The State: Past, Present, Future by Bob Jessop",
            "https://politybooks.com/book/state",
            "The publisher page for Bob Jessop's book.",
        ),
    ]
    accepted, audit = rank_search_results(
        f'"{title}" {author}',
        results,
        identity=title,
        title=title,
        author=author,
        strict_identity=True,
    )
    assert accepted == [results[1]]
    assert "title_identity_mismatch" in audit[0]["reasons"]
    assert "author_identity_mismatch" in audit[0]["reasons"]
    assert audit[1]["strong_title_match"]
