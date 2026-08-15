from __future__ import annotations

from tarjomeh.core.pipeline import (
    _chunk_style_approved,
    _critique_requires_refinement,
    audit_translation_language,
)
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.parsers.pdf_parser import _join_block_lines, _join_span_texts
from tarjomeh.quality.critique import CritiqueResult
from tarjomeh.quality.integrity import (
    extract_identifiers,
    reconcile_numbers,
    restore_source_identifiers,
    unexpected_latin_prose,
)


def test_identifier_repair_accepts_localized_decimal_separators() -> None:
    source = "Classification 320.1-dc23."
    target = "رده بندي ۳۲۰ ٫ ۱ -dc ۲۳ است."

    repaired, report = restore_source_identifiers(source, target)

    assert "320.1-dc23" in repaired
    assert report["repair_count"] == 1
    assert extract_identifiers(source) == extract_identifiers(repaired)


def test_language_audit_understands_unicode_names_and_citation_pairs() -> None:
    source = "Compare Lukacs (1971) and Hall and Taylor (1996)."
    target = "ر.ک. Lukacs (1971) و Hall and Taylor (1996)."

    # Use the accented source spelling to ensure the tokenizer does not emit
    # an ASCII suffix such as ``cs`` as a separate suspicious word.
    source = source.replace("Lukacs", "Lukács")
    target = target.replace("Lukacs", "Lukács")

    assert unexpected_latin_prose(source, target) == []


def test_language_audit_allows_source_grounded_parenthetical_expression() -> None:
    source = "The institution develops its own modus operandi over time."
    target = "نهاد به تدريج شيوه عمل (modus operandi) خاص خود را پديد مي آورد."

    assert unexpected_latin_prose(source, target) == []


def test_language_audit_does_not_join_unrelated_source_words_into_exception() -> None:
    source = "The method and institutional practice changed."
    target = "روش (method institutional) تغییر کرد."

    findings = unexpected_latin_prose(source, target)

    assert {finding["token"] for finding in findings} == {"method", "institutional"}


def test_language_audit_uses_structural_provenance_but_still_flags_leaks() -> None:
    source = "Q: quaderno (notebook)"
    target = "Q: quaderno (دفترچه)"

    assert audit_translation_language(
        source,
        target,
        structural_role="body",
        chapter_title="Abbreviations",
    )["review_required"] is False

    leak = audit_translation_language(
        "The institution changed.",
        "نهاد تغيير کرد. The answer has been revised.",
        structural_role="body",
        chapter_title="Chapter 1",
    )
    assert leak["review_required"] is True


class _CleanHighQualityStyleDB:
    def get_chunk_events(self, _job: str, _chunk: int):
        return [
            {"event_type": "chunk_started", "payload": {}},
            {
                "event_type": "critique_completed",
                "payload": {
                    "valid": True,
                    "blocking_issue_count": 0,
                    "scores": {
                        "accuracy": 9,
                        "fluency": 9,
                        "terminology": 9,
                        "register": 9,
                        "average": 9,
                    },
                },
            },
        ]


def test_only_clean_high_quality_prose_can_seed_style_memory() -> None:
    assert _chunk_style_approved(_CleanHighQualityStyleDB(), "job", 0)


def test_accepted_term_reconciles_stale_summary_without_becoming_curated() -> None:
    manager = MemoryManager(TarjomehConfig())
    manager.bilingual_summary.persian_summary = (
        "اين فصل علل فروپاشي دولت را بررسي مي کند."
    )
    manager.proper_nouns.add_noun(
        "state failure",
        "فروپاشي دولت",
        category="term",
        provenance="auto_extraction",
    )
    manager.proper_nouns.add_noun(
        "state failure",
        "شکست دولت",
        category="term",
        provenance="accepted_correction",
    )

    report = manager.reconcile_bilingual_summary()

    assert manager.bilingual_summary.persian_summary == (
        "اين فصل علل شکست دولت را بررسي مي کند."
    )
    assert report["replacement_count"] == 1
    assert manager.proper_nouns.provenance_for("state failure")["authority"] < 100


def test_objective_short_fluency_defect_requests_bounded_refinement() -> None:
    critique = CritiqueResult(
        accuracy=9,
        fluency=8,
        terminology=9,
        register=9,
        average=8.75,
        issues=["[MINOR/fluency] broken coordination"],
        issue_details=[{
            "issue_id": "mqm-objective-grammar",
            "category": "fluency",
            "severity": "minor",
            "confidence": 0.80,
            "source_quote": "I add a fourth element",
            "current_persian_quote": "و که عنصر چهارم افزوده مي شود",
            "suggested_correction": "و اينکه عنصر چهارم افزوده مي شود",
            "rationale": "The current phrase has broken grammar and coordination.",
        }],
    )

    assert _critique_requires_refinement(critique, 9.0)


def test_subjective_short_fluency_preference_remains_advisory() -> None:
    critique = CritiqueResult(
        accuracy=9,
        fluency=8,
        terminology=9,
        register=9,
        average=8.75,
        issues=["[MINOR/fluency] stylistic preference"],
        issue_details=[{
            "issue_id": "mqm-preference",
            "category": "fluency",
            "severity": "minor",
            "confidence": 0.99,
            "source_quote": "pay attention",
            "current_persian_quote": "توجه کنيد",
            "suggested_correction": "دقت کنيد",
            "rationale": "The alternative may sound slightly more elegant.",
        }],
    )

    assert not _critique_requires_refinement(critique, 9.0)


def test_ascii_line_break_hyphen_uses_repeated_source_evidence_only() -> None:
    evidence: list[dict[str, str]] = []
    assert _join_block_lines(
        ["Several insti-", "tutional arrangements changed."],
        known_words={"institutional"},
        dehyphenation_evidence=evidence,
    ).startswith("Several institutional")
    assert evidence[0]["joined"] == "institutional"

    assert _join_block_lines(
        ["A well-", "defined arrangement."],
        known_words={"institutional"},
    ) == "A well-defined arrangement."


def test_span_joining_uses_geometry_to_restore_missing_word_space() -> None:
    spans = [
        {"text": "(chapter 3)", "bbox": (0, 0, 50, 10), "size": 10},
        {"text": "and", "bbox": (53, 0, 68, 10), "size": 10},
        {"text": " continuation", "bbox": (69, 0, 110, 10), "size": 10},
    ]

    assert _join_span_texts(spans) == "(chapter 3) and continuation"


def test_large_localized_year_range_is_reconciled() -> None:
    report = reconcile_numbers(
        "The process unfolded over 400-500 years.",
        "اين فرايند طي چهارصد تا پانصد سال رخ داد.",
    )

    assert report["missing"] == []
    assert len(report["localized_equivalents"]) == 2
