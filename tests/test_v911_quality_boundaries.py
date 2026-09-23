from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from tarjomeh.context.book_researcher import BookResearcher
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import (
    _chunk_style_approved,
    _reconcile_committed_terminology,
)
from tarjomeh.core.term_notes import ensure_inline_proper_noun_originals
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.glossary.manager import GlossaryManager
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.memory.proper_nouns import ProperNouns
from tarjomeh.persian.typography import PersianTypographer
from tarjomeh.quality.integrity import PostEditIntegrityGate, extract_identifiers


def test_single_contextual_glossary_entry_requires_local_evidence() -> None:
    glossary = GlossaryManager()
    glossary.add_term(
        "archaeology",
        "باستان‌شناسی دانش",
        context="Foucault analysis of discursive formations",
        domain="philosophy",
    )

    assert glossary.find_terms(
        "Archaeology and sociology are disciplines.", domain="philosophy"
    ) == []
    assert [entry.source for entry in glossary.last_contextual_advisories] == [
        "archaeology"
    ]
    assert glossary.find_terms(
        "Foucault develops an archaeology of discursive formations.",
        domain="philosophy",
    )[0].target == "باستان‌شناسی دانش"


def test_unqualified_glossary_entry_remains_mandatory() -> None:
    glossary = GlossaryManager()
    glossary.add_term("state", "دولت")
    assert glossary.find_terms("The state acts.")[0].target == "دولت"
    assert glossary.last_contextual_advisories == []


def test_observed_alias_survives_round_trip_and_anchors_original() -> None:
    nouns = ProperNouns()
    nouns.add_noun(
        "Nicos Poulantzas", "نیکوس پولانزاس", category="person",
        provenance="incremental_extraction",
    )
    nouns.add_alias("Nicos Poulantzas", "نیکوس پولانتزاس")
    restored = ProperNouns()
    restored.deserialize(nouns.serialize())

    document = TranslatedDocument(title="Test", paragraphs=[TranslatedParagraph(
        index=0,
        source_text="Nicos Poulantzas developed the argument.",
        translated_text="نیکوس پولانتزاس این استدلال را بسط داد.",
    )])
    report = ensure_inline_proper_noun_originals(
        document,
        restored.inline_eligible_nouns(),
        PersianTypographer({"convert_numerals": False}),
        restored.serialize()["categories"],
        aliases=restored.inline_eligible_aliases(),
        return_report=True,
    )
    assert report["anchored_count"] == 1
    assert "نیکوس پولانتزاس (Nicos Poulantzas)" in document.paragraphs[0].translated_text


def test_incremental_extraction_prefers_rendering_visible_in_translation() -> None:
    manager = MemoryManager(TarjomehConfig())
    manager.proper_nouns.add_noun(
        "Nicos Poulantzas", "نیکوس پولانزاس", category="person",
        provenance="incremental_extraction",
    )
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=json.dumps([{
        "term": "Nicos Poulantzas",
        "category": "person",
        "suggested_persian": "نیکوس پولانتزاس",
    }], ensure_ascii=False))

    report = asyncio.run(
        manager.update_proper_nouns(
            llm,
            "Nicos Poulantzas developed the argument.",
            "نیکوس پولانتزاس (Nicos Poulantzas) این استدلال را بسط داد.",
        )
    )
    state = manager.proper_nouns.serialize()
    assert report["observed_translation_count"] == 1
    assert state["nouns"]["Nicos Poulantzas"] == "نیکوس پولانتزاس"
    assert "نیکوس پولانزاس" in state["aliases"]["Nicos Poulantzas"]


def test_identifiers_are_format_sensitive_but_prose_numbers_are_not() -> None:
    source = "Grant RES-051-27-0303 has ISBN 978-1-4028-9462-6. Chapter 3 follows."
    good = "گرنت RES-051-27-0303 دارای ISBN 978-1-4028-9462-6 است. فصل سوم می‌آید."
    bad = "گرنت RES-۰۵۱-۲۷-۰۳۰۳ دارای ISBN ۹۷۸-۱-۴۰۲۸-۹۴۶۲-۶ است. فصل سوم می‌آید."
    gate = PostEditIntegrityGate()

    assert extract_identifiers(source)
    assert gate.evaluate(source, good).accepted
    result = gate.evaluate(source, bad)
    assert not result.accepted
    assert "source_identifiers_changed" in {
        finding.check_id for finding in result.blocking
    }


def test_style_gate_rejects_average_eight_without_changing_continuity() -> None:
    class FakeDB:
        def get_chunk_events(self, _job: str, _chunk: int):
            return [
                {"event_type": "chunk_started", "payload": {}},
                {"event_type": "critique_completed", "payload": {
                    "valid": True,
                    "blocking_issue_count": 0,
                    "scores": {
                        "accuracy": 9, "fluency": 8, "terminology": 9,
                        "register": 9, "average": 8.75,
                    },
                }},
            ]

    assert not _chunk_style_approved(FakeDB(), "job", 0)


def test_committed_term_reconciliation_cannot_override_curated_authority() -> None:
    class FakeDB:
        def get_chunk_events(self, _job: str, _chunk: int):
            return [{"event_type": "chunk_started", "payload": {}}]

        def get_qa_issues(self, _job: str, _chunk: int):
                return [{
                    "issue_id": "term-1", "category": "terminology",
                    "source_quote": "strategic-relational approach",
                    "current_persian_quote": "رویکرد راهبردی-رابطه‌ای",
                    "suggested_correction": "رویکرد راهبردی-رابطه‌ای",
                }]

        def get_issue_decisions(self, _job: str, _chunk: int):
            return [{
                "issue_id": "term-1",
                "payload": {"commit_status": "committed_full_candidate"},
                "resulting_span": "رویکرد راهبردی-رابطه‌ای",
            }]

    memory = MemoryManager(TarjomehConfig())
    memory.proper_nouns.add_noun(
        "strategic-relational approach", "رویکرد استراتژیک-رابطه‌ای",
        category="theory", provenance="incremental_extraction",
    )
    report = _reconcile_committed_terminology(
        FakeDB(), "job", 0, memory,
        "The strategic-relational approach is used.",
        "رویکرد راهبردی-رابطه‌ای به کار می‌رود.",
    )
    assert report["reconciled"]
    assert memory.proper_nouns.inline_eligible_nouns()[
        "strategic-relational approach"
    ] == "رویکرد راهبردی-رابطه‌ای"

    curated = MemoryManager(TarjomehConfig())
    curated.proper_nouns.add_noun(
        "strategic-relational approach", "رویکرد استراتژیک-رابطه‌ای",
        category="theory", provenance="curated_glossary",
    )
    report = _reconcile_committed_terminology(
        FakeDB(), "job", 0, curated,
        "The strategic-relational approach is used.",
        "رویکرد راهبردی-رابطه‌ای به کار می‌رود.",
    )
    assert report["skipped"][0]["reason"] == "curated_authority_preserved"
    assert curated.proper_nouns.inline_eligible_nouns()[
        "strategic-relational approach"
    ] == "رویکرد استراتژیک-رابطه‌ای"


def test_retail_only_prior_translation_claim_is_removed() -> None:
    sources = [{
        "title": "Buy this book",
        "snippet": "Persian edition for sale",
        "query": "book Persian translation",
        "url": "https://www.amazon.com/example",
        "source_authority": "commercial",
    }]
    context, warnings = BookResearcher._guard_research_context(
        "The book studies the state. A Persian translation was published in Iran.",
        sources,
    )
    assert "studies the state" in context
    assert "Persian translation" not in context
    assert warnings
    assert BookResearcher._source_authority(sources[0]["url"]) == "commercial"
