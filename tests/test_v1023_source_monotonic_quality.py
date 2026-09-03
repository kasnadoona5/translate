from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import (
    _best_source_faithful_version,
    _chunk_needs_review,
    _introduced_structure_conflicts,
)
from tarjomeh.core.term_notes import ensure_inline_proper_noun_originals
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.persian.typography import PersianTypographer
from tarjomeh.quality.integrity import unexpected_latin_prose

SOURCE_ANOMALY = (
    "This chapter addresses two issues. First, it defines the field; "
    "second, it compares approaches; third, it states the implications."
)
FA_PRESERVED = (
    "این فصل به دو مسئله می‌پردازد. نخست، میدان را تعریف می‌کند؛ "
    "دوم، رویکردها را مقایسه می‌کند؛ سوم، پیامدها را بیان می‌کند."
)
FA_SILENTLY_CORRECTED = FA_PRESERVED.replace("دو مسئله", "سه مسئله")


def _critique(score: float = 9.0) -> SimpleNamespace:
    return SimpleNamespace(
        issue_details=[],
        issues=[],
        accuracy=score,
        terminology=score,
        fluency=score,
        register=score,
        average=score,
        valid=True,
    )


def test_structure_admission_allows_restoring_explicit_source_wording() -> None:
    assert _introduced_structure_conflicts(
        SOURCE_ANOMALY, FA_SILENTLY_CORRECTED, FA_PRESERVED
    ) == []


def test_structure_admission_blocks_silently_reconciling_source_anomaly() -> None:
    findings = _introduced_structure_conflicts(
        SOURCE_ANOMALY, FA_PRESERVED, FA_SILENTLY_CORRECTED
    )
    assert [item["classification"] for item in findings] == [
        "unauthorized_source_correction"
    ]


def test_version_selection_prefers_source_valid_wording_over_higher_score() -> None:
    selected = _best_source_faithful_version(
        [
            (FA_PRESERVED, _critique(8.7)),
            (FA_SILENTLY_CORRECTED, _critique(9.8)),
        ],
        source=SOURCE_ANOMALY,
    )
    assert selected is not None
    assert selected[1] == FA_PRESERVED


class _EventsDB:
    def get_chunk_events(self, _job: str, _chunk: int):
        return [
            {"event_type": "chunk_started", "payload": {}},
            {
                "event_type": "structure_audit",
                "payload": {"classifications": ["translation_structure_mismatch"]},
            },
            {"event_type": "structure_audit", "payload": {"classifications": []}},
        ]


def test_latest_structure_audit_supersedes_repaired_predecessor() -> None:
    assert _chunk_needs_review(_EventsDB(), "job", 0) is False


def test_latin_title_comma_is_not_converted_to_persian_punctuation() -> None:
    text = "عنوان State, Power and Society در متن آمده است."
    assert PersianTypographer().process(text) == text


def test_source_grounded_measurement_unit_is_not_a_language_leak() -> None:
    source = "The text was set in 11 pt type."
    target = "متن با حروف ۱۱ pt حروف‌چینی شده است."
    assert unexpected_latin_prose(source, target) == []


def test_ordinary_english_next_to_a_year_is_not_mistaken_for_a_unit() -> None:
    source = "The institution was founded in 2001."
    target = "این نهاد in 2001 تأسیس شد."
    assert [item["token"] for item in unexpected_latin_prose(source, target)] == [
        "in"
    ]


def test_first_occurrence_anchor_is_deferred_out_of_parentheses() -> None:
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(
        index=0,
        source_text="The account uses pluralism in a qualified sense.",
        translated_text="این روایت (برداشتی از کثرت‌گرایی) را به کار می‌گیرد.",
    )])
    report = ensure_inline_proper_noun_originals(
        document,
        {"pluralism": "کثرت‌گرایی"},
        PersianTypographer(),
        return_report=True,
    )
    assert isinstance(report, dict)
    assert report["parenthetical_deferred_count"] == 1
    assert "((" not in document.paragraphs[0].translated_text


def test_style_policy_reports_original_paragraph_index() -> None:
    manager = MemoryManager(TarjomehConfig())
    chunk = Chunk(
        0,
        "Heading\n\n" + "A complete academic body sentence. " * 18,
        "Chapter",
        "",
        metadata={
            "style_eligible": True,
            "style_body_paragraphs": [1],
            "structural_roles": ["heading", "body"],
        },
    )
    target = "عنوان\n\n" + (
        "این بند تحلیلی، استدلال را با زبانی روشن و دانشگاهی توضیح می‌دهد. " * 12
    )
    policy = manager.update_after_translation(
        chunk,
        target,
        quality_approved=True,
        style_approved=True,
        long_term_reliable=True,
    )
    selected = policy["style_sample_policy"]
    assert selected.get("source_paragraph_index") == 1


def test_low_space_deployment_is_scoped_to_tarjomeh() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "deploy_tarjomeh_v1023.sh"
    ).read_text(encoding="utf-8")
    assert "docker image save translate_tarjomeh:latest" in script
    assert 'docker rm -f "$CONTAINER"' in script
    assert 'docker inspect -f \'{{json .Mounts}}\' 9router' in script
    assert "docker stop 9router" not in script
    assert "docker rm 9router" not in script
    assert "docker image rm decolua/9router" not in script
    assert "/opt/translate/9router-data" not in script
