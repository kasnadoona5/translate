from __future__ import annotations

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import (
    _chapter_summary_memory_admission,
    _chunk_memory_admission,
    _critique_requires_refinement,
)
from tarjomeh.core.term_notes import (
    ensure_inline_proper_noun_originals,
    normalize_adjacent_original_citations,
)
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.memory.bilingual_summary import BilingualSummary
from tarjomeh.memory.long_term import LongTermMemory
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.persian.typography import PersianTypographer
from tarjomeh.quality.critique import CritiqueResult


class _EventDB:
    def __init__(self, events: dict[int, list[dict]]) -> None:
        self.events = events

    def get_chunk_events(self, _job_id: str, chunk_index: int):
        return self.events.get(chunk_index, [])


def _critique_event(average: float, accuracy: float, terminology: float) -> dict:
    return {
        "event_type": "critique_completed",
        "payload": {
            "valid": True,
            "blocking_issue_count": 0,
            "scores": {
                "average": average,
                "accuracy": accuracy,
                "fluency": average,
                "terminology": terminology,
                "register": average,
            },
        },
    }


def test_advisory_long_term_translation_remains_retrievable() -> None:
    memory = LongTermMemory(retrieval_k=2)
    memory.add(
        "The strategic relational state remains contested.",
        "ترجمه‌ای که هنوز نیازمند بازبینی است.",
        reliable=False,
    )

    relevant = memory.get_relevant("strategic relational state")

    assert len(relevant) == 1
    assert relevant[0]["reliable"] is False
    assert "advisory continuity" in memory.get_context(
        "strategic relational state"
    )


def test_memory_trust_separates_continuity_from_authority() -> None:
    manager = MemoryManager(TarjomehConfig())
    chunk = Chunk(
        index=0,
        text="Institutional analysis carries the argument into the next passage.",
        chapter_title="Chapter 1",
        section_title="",
        metadata={"style_eligible": True, "structural_roles": ["body"]},
    )

    policy = manager.update_after_translation(
        chunk,
        "تحلیل نهادی استدلال را به بند بعدی پیوند می‌دهد.",
        quality_approved=True,
        long_term_reliable=False,
        short_term_trust="advisory_review",
        reliability_reasons=["deferred_mqm_advice"],
    )
    context = manager.get_context_for_chunk(chunk)

    assert policy["continuity_retained"] is True
    assert policy["long_term_reliable"] is False
    assert policy["style_sample_added"] is False
    assert len(manager.long_term.serialize()) == 1
    assert "advisory continuity" in context.long_term
    assert "needs review" in context.short_term


def test_clean_below_threshold_chunk_remains_reliable_with_advisory_reasons() -> None:
    db = _EventDB({
        0: [
            {"event_type": "chunk_started", "payload": {}},
            _critique_event(8.0, 9.0, 9.0),
            {"event_type": "mqm_minor_only_deferred", "payload": {}},
        ],
        1: [
            {"event_type": "chunk_started", "payload": {}},
            _critique_event(9.0, 9.0, 9.0),
        ],
    })

    advisory = _chunk_memory_admission(db, "job", 0)
    trusted = _chunk_memory_admission(db, "job", 1)

    assert advisory["quality_approved"] is True
    assert advisory["continuity_retained"] is True
    assert advisory["long_term_reliable"] is True
    assert advisory["short_term_trust"] == "trusted"
    assert "deferred_mqm_advice" in advisory["reliability_reasons"]
    assert advisory["disqualifying_reliability_reasons"] == []
    assert trusted["long_term_reliable"] is True
    assert trusted["short_term_trust"] == "trusted"


def test_chapter_summary_keeps_context_but_records_advisory_inputs() -> None:
    chunks = [
        Chunk(
            index=index,
            text=f"Source {index}",
            chapter_title="Chapter 1",
            section_title="",
            metadata={"chapter_position": 1},
        )
        for index in range(2)
    ]
    db = _EventDB({
        0: [{
            "event_type": "memory_update_policy",
            "payload": {"long_term_reliable": True},
        }],
        1: [{
            "event_type": "memory_update_policy",
            "payload": {
                "long_term_reliable": False,
                "reliability_reasons": ["deferred_mqm_advice"],
            },
        }],
    })

    admission = _chapter_summary_memory_admission(db, "job", chunks, 1)

    assert admission["input_trust"] == "advisory_inputs"
    assert admission["context_retained"] is True
    assert admission["contributing_chunks"] == 2


def test_summary_trust_round_trips_and_never_claims_terminology_authority() -> None:
    summary = BilingualSummary()
    summary.english_summary = "The chapter develops an institutional argument."
    summary.persian_summary = "فصل استدلالی نهادی را بسط می‌دهد."
    summary.set_input_trust("advisory_inputs", ["deferred_mqm_advice"])

    restored = BilingualSummary()
    restored.deserialize(summary.serialize())
    context = restored.get_context()

    assert restored.input_trust == "advisory_inputs"
    assert restored.trust_reasons == ["deferred_mqm_advice"]
    assert "argument orientation only" in context
    assert "Never copy its wording" in context
    assert "فصل استدلالی نهادی" in context


def test_objective_grammar_evidence_routes_without_a_verbose_rationale() -> None:
    critique = CritiqueResult(
        accuracy=9,
        fluency=8,
        terminology=9,
        register=9,
        average=8.75,
        issues=["[MINOR/fluency] malformed coordination"],
        issue_details=[{
            "issue_id": "mqm-syntax-coordination",
            "category": "fluency",
            "severity": "minor",
            "confidence": 0.95,
            "source_quote": "The fourth task considers historical semantics.",
            "current_persian_quote": "و که وظیفه چهارم معناشناسی تاریخی را بررسی می‌کند",
            "suggested_correction": "و اینکه وظیفه چهارم معناشناسی تاریخی را بررسی می‌کند",
        }],
    )

    assert _critique_requires_refinement(critique, 9.0)


def test_subjective_high_confidence_preference_is_still_not_forced() -> None:
    critique = CritiqueResult(
        accuracy=9,
        fluency=8,
        terminology=9,
        register=9,
        average=8.75,
        issues=["[MINOR/fluency] preference"],
        issue_details=[{
            "issue_id": "mqm-style-preference",
            "category": "fluency",
            "severity": "minor",
            "confidence": 0.99,
            "source_quote": "pay attention",
            "current_persian_quote": "توجه کنید",
            "suggested_correction": "دقت کنید",
            "rationale": "The alternative may sound slightly more elegant.",
        }],
    )

    assert not _critique_requires_refinement(critique, 9.0)


def test_combined_author_year_parenthetical_satisfies_inline_anchor() -> None:
    paragraph = TranslatedParagraph(
        index=0,
        source_text="Shmuel Eisenstadt's (1963) work examines empires.",
        translated_text=(
            "اثر شموئل آیزنشتات (Shmuel Eisenstadt, 1963) "
            "امپراتوری‌ها را بررسی می‌کند."
        ),
    )
    document = TranslatedDocument(paragraphs=[paragraph])

    report = ensure_inline_proper_noun_originals(
        document,
        {"Shmuel Eisenstadt": "شموئل آیزنشتات"},
        PersianTypographer(),
        {"Shmuel Eisenstadt": "person"},
        return_report=True,
    )

    assert report["inserted_count"] == 0
    assert paragraph.translated_text.count("Shmuel Eisenstadt") == 1


def test_duplicate_author_and_combined_citation_are_reconciled_generally() -> None:
    paragraph = TranslatedParagraph(
        index=0,
        source_text="Shmuel Eisenstadt's (1963) work examines empires.",
        translated_text=(
            "اثر شموئل آیزنشتات (Shmuel Eisenstadt) "
            "(Shmuel Eisenstadt, 1963) امپراتوری‌ها را بررسی می‌کند."
        ),
    )
    document = TranslatedDocument(paragraphs=[paragraph])

    report = normalize_adjacent_original_citations(
        document,
        {"Shmuel Eisenstadt": "شموئل آیزنشتات"},
    )

    assert report["normalized_count"] == 1
    assert paragraph.translated_text.count("Shmuel Eisenstadt") == 1
    assert "(Shmuel Eisenstadt, 1963)" in paragraph.translated_text
