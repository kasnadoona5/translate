from __future__ import annotations

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import _reconcile_committed_terminology
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.memory.proper_nouns import (
    ProperNouns,
    is_reusable_terminology_mapping,
)
from tarjomeh.persian.orthography import apply_safe_persian_orthography
from tarjomeh.quality.back_translator import BackTranslator
from tarjomeh.quality.integrity import (
    PostEditIntegrityGate,
    extract_identifiers,
    restore_source_identifiers,
)


class _CommittedCorrectionDB:
    def __init__(self, source: str, target: str) -> None:
        self.source = source
        self.target = target

    def get_chunk_events(self, _job: str, _chunk: int):
        return [{"event_type": "chunk_started", "payload": {}}]

    def get_qa_issues(self, _job: str, _chunk: int):
        return [{
            "issue_id": "issue-1",
            "category": "accuracy",
            "source_quote": self.source,
            "suggested_correction": self.target,
        }]

    def get_issue_decisions(self, _job: str, _chunk: int):
        return [{
            "issue_id": "issue-1",
            "payload": {"commit_status": "committed_full_candidate"},
            "resulting_span": self.target,
        }]


def test_contextual_refinement_is_not_promoted_to_global_terminology() -> None:
    source = "Constitution"
    target = "ساختار دولت به‌مثابه شکلی از"
    assert not is_reusable_terminology_mapping(source, target)

    memory = MemoryManager(TarjomehConfig())
    report = _reconcile_committed_terminology(
        _CommittedCorrectionDB(source, target),
        "job",
        0,
        memory,
        "The Constitution of this polity changed.",
        f"{target} این نظام سیاسی تغییر کرد.",
    )

    assert source not in memory.proper_nouns.all_nouns()
    assert report["context_deferred"][0]["reason"] == (
        "contextual_correction_not_reusable_as_global_term"
    )


def test_existing_contextual_correction_is_quarantined_without_deletion() -> None:
    nouns = ProperNouns()
    nouns.deserialize({
        "nouns": {
            "Constitution": "ساختار دولت به‌مثابه شکلی از",
            "state system": "نظام دولتی",
        },
        "categories": {"Constitution": "term", "state system": "term"},
        "provenance": {
            "Constitution": {
                "origin": "accepted_correction", "authority": 80,
                "observations": 1, "superseded": [],
            },
            "state system": {
                "origin": "accepted_correction", "authority": 80,
                "observations": 1, "superseded": [],
            },
        },
        "introduced": [],
    })

    assert nouns.all_nouns()["Constitution"] == "ساختار دولت به‌مثابه شکلی از"
    assert nouns.is_context_deferred("Constitution")
    assert "Constitution" not in nouns.get_context()
    assert "state system" in nouns.get_context()


def test_layer_one_prompt_contains_only_terms_in_current_source() -> None:
    manager = MemoryManager(TarjomehConfig())
    manager.proper_nouns.add_noun(
        "state system", "نظام دولتی", category="term",
        provenance="curated_glossary",
    )
    manager.proper_nouns.add_noun(
        "civil society", "جامعه مدنی", category="term",
        provenance="curated_glossary",
    )
    chunk = Chunk(
        index=0,
        text="The state-\nsystem changes over time.",
        chapter_title="Chapter",
        section_title="",
    )

    context = manager.get_context_for_chunk(chunk).proper_nouns
    assert "state system" in context
    assert "civil society" not in context


def test_style_memory_omits_citation_and_note_marker_sentences() -> None:
    manager = MemoryManager(TarjomehConfig())
    chunk = Chunk(
        index=0,
        text="Substantive body prose. " * 30,
        chapter_title="Chapter",
        section_title="",
    )
    translation = (
        "این جمله نمونه‌ای روشن از نثر دانشگاهی پیوسته و دقیق است. "
        "این ادعا در پژوهش‌های پیشین بررسی شده است (Tilly 1975; Spruyt 1993). "
        "2 و این یادداشت توضیحی نباید الگوی سبک باشد."
    )

    policy = manager.update_after_translation(chunk, translation)
    assert policy["style_sample_added"]
    sample = manager.style_samples[0]
    assert "Tilly" not in sample
    assert "2 و" not in sample
    assert "نثر دانشگاهی" in sample


def test_machine_identifiers_are_restored_exactly() -> None:
    source = (
        "ISBN-13: 978-0-7456-3304-6; politybooks.com; "
        "class JC11.J47; code 320.1–dc23; Cambridge CB2 1UR."
    )
    translated = (
        "شابک-۱۳: ۹۷۸ - ۰ - ۷۴۵۶ - ۳۳۰۴ - ۶؛ politybooks. com؛ "
        "رده JC۱۱. J۴۷؛ کد ۳۲۰. ۱–dc۲۳؛ کمبریج CB۲ ۱UR."
    )

    repaired, report = restore_source_identifiers(source, translated)

    for value in (
        "978-0-7456-3304-6", "politybooks.com", "JC11.J47",
        "320.1–dc23", "CB2 1UR",
    ):
        assert value in repaired
    assert report["repair_count"] >= 5
    assert extract_identifiers(source) == extract_identifiers(repaired)
    assert PostEditIntegrityGate().evaluate(source, repaired).accepted


def test_back_translation_reconciles_translated_structure_and_grounded_alias() -> None:
    structural = BackTranslator(object()).compare(
        "Part II develops the framework.",
        "The framework is developed.",
        "بخش دوم چارچوب را بسط می‌دهد.",
    )
    assert "named_entities_missing" not in structural.diagnostics["risk_flags"]
    assert structural.diagnostics["entities_reconciled_by_structure"]

    full_heading = BackTranslator(object()).compare(
        "Part II: On Territory, Apparatus, and Population",
        "A major section of the book.",
        "بخش دوم: درباره قلمرو، دستگاه و جمعیت",
    )
    assert "named_entities_missing" not in full_heading.diagnostics["risk_flags"]
    assert len(
        full_heading.diagnostics["entities_reconciled_by_structure"]
    ) >= 2

    person = BackTranslator(object()).compare(
        "Emmerich de Vattel developed this account.",
        "A jurist developed this account.",
        "امریش دو واتل این تبیین را بسط داد.",
        entity_aliases={"Emmerich de Vattel": ["امریش دو واتل"]},
    )
    assert "named_entities_missing" not in person.diagnostics["risk_flags"]
    assert person.diagnostics["entities_reconciled_by_memory"]


def test_compound_axis_zwnj_is_conservative() -> None:
    fixed, edits = apply_safe_persian_orthography(
        "این رویکرد عاملمحور و دولت محور است، اما در محور دیگری قرار دارد."
    )

    assert "عامل‌محور" in fixed
    assert "دولت‌محور" in fixed
    assert "در محور" in fixed
    assert any(edit["rule_id"] == "compound_axis_zwnj" for edit in edits)
