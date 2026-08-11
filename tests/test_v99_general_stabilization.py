"""General regressions for prompt trust and deterministic QA stabilization."""

from __future__ import annotations

from types import SimpleNamespace

from tarjomeh.context.book_researcher import BookResearcher
from tarjomeh.context.search_providers import SearchResult, rank_search_results
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import (
    _chunk_needs_review,
    _ensure_chunk_review_reason,
)
from tarjomeh.glossary.compliance import target_present
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.memory.proper_nouns import ProperNouns
from tarjomeh.quality.back_translator import BackTranslator
from tarjomeh.quality.integrity import (
    PostEditIntegrityGate,
    protected_source_apparatus,
)


def test_structural_number_words_do_not_weaken_factual_number_protection() -> None:
    gate = PostEditIntegrityGate()
    chapter = gate.evaluate(
        "Chapter 12 explains the argument.",
        "\u0641\u0635\u0644 \u062f\u0648\u0627\u0632\u062f\u0647\u0645 \u0627\u0633\u062a\u062f\u0644\u0627\u0644 \u0631\u0627 \u062a\u0648\u0636\u06cc\u062d \u0645\u06cc\u200c\u062f\u0647\u062f.",
    )
    date = gate.evaluate(
        "The study was published in 2016.",
        "\u0627\u06cc\u0646 \u067e\u0698\u0648\u0647\u0634 \u0645\u0646\u062a\u0634\u0631 \u0634\u062f.",
    )
    decimal = gate.evaluate(
        "See Table 1.1 for details.",
        "\u0628\u0631\u0627\u06cc \u062c\u0632\u0626\u06cc\u0627\u062a \u0628\u0647 \u062c\u062f\u0648\u0644 \u06f1 \u066b \u06f1 \u0631\u062c\u0648\u0639 \u06a9\u0646\u06cc\u062f.",
    )
    chapters = gate.evaluate(
        "Chapters 2, 3, and 4 develop the framework.",
        "\u0641\u0635\u0644\u200c\u0647\u0627\u06cc \u062f\u0648\u0645\u060c \u0633\u0648\u0645 \u0648 \u0686\u0647\u0627\u0631\u0645 \u0686\u0627\u0631\u0686\u0648\u0628 \u0631\u0627 \u0628\u0633\u0637 \u0645\u06cc\u200c\u062f\u0647\u0646\u062f.",
    )
    assert chapter.accepted
    assert decimal.accepted
    assert chapters.accepted
    assert any(item.check_id == "numbers_missing" for item in date.blocking)


def test_glossary_accepts_only_conservative_heh_derivation() -> None:
    assert target_present(
        "\u0633\u0648\u0698\u0647",
        "\u0645\u0648\u0642\u0639\u06cc\u062a\u200c\u0647\u0627\u06cc \u0633\u0648\u0698\u06af\u06cc",
    )
    assert not target_present(
        "\u0633\u0648\u0698\u0647",
        "\u062c\u0627\u0645\u0639\u0647\u200c\u0634\u0646\u0627\u0633\u06cc",
    )


def test_citation_connective_prose_is_not_protected_as_one_english_span() -> None:
    source = "The claim is established (for discussion, see Morgan 2004)."
    previous = (
        "\u0627\u06cc\u0646 \u0627\u062f\u0639\u0627 \u062a\u062b\u0628\u06cc\u062a \u0634\u062f\u0647 \u0627\u0633\u062a "
        "(for discussion, see Morgan 2004)."
    )
    candidate = (
        "\u0627\u06cc\u0646 \u0627\u062f\u0639\u0627 \u062a\u062b\u06cc\u062a \u0634\u062f\u0647 \u0627\u0633\u062a "
        "(\u0628\u0631\u0627\u06cc \u0628\u062d\u062b \u0628\u06cc\u0634\u062a\u0631\u060c \u0631. \u06a9. Morgan 2004)."
    )
    assert protected_source_apparatus(source, previous) == ["Morgan 2004"]
    assert PostEditIntegrityGate().evaluate(
        source, candidate, previous=previous
    ).accepted


def test_prompt_memory_filters_placeholders_and_marks_terms_advisory() -> None:
    nouns = ProperNouns()
    nouns.add_noun("Example University", "\u062f\u0627\u0646\u0634\u06af\u0627\u0647 \u0646\u0645\u0648\u0646\u0647", "organization")
    nouns.add_noun("relational approach", "\u0631\u0648\u06cc\u06a9\u0631\u062f_relation", "term")
    nouns.add_noun("historical method", "\u0631\u0648\u0634 \u062a\u0627\u0631\u06cc\u062e\u06cc", "term")
    context = nouns.get_context()
    assert "Example University" in context
    assert "relational approach" not in context
    assert "historical method" in context
    assert "advisory terminology only" in context


def test_style_projection_rejects_contaminated_samples_without_mutating_state() -> None:
    manager = MemoryManager(TarjomehConfig())
    contaminated = (
        "\u0627\u06cc\u0646 \u0645\u062a\u0646 \u06cc\u06a9 \u0628\u062d\u062b \u062f\u0627\u0646\u0634\u06af\u0627\u0647\u06cc \u0637\u0648\u0644\u0627\u0646\u06cc \u0631\u0627 \u062f\u0646\u0628\u0627\u0644 \u0645\u06cc\u200c\u06a9\u0646\u062f "
        "(see Example 2001 for discussion)."
    )
    clean = (
        "\u0627\u06cc\u0646 \u067e\u0698\u0648\u0647\u0634 \u0628\u0627 \u0646\u062b\u0631\u06cc \u0631\u0633\u0645\u06cc \u0648 \u062f\u0642\u06cc\u0642\u060c \u0645\u0646\u0627\u0633\u0628\u0627\u062a \u0646\u0647\u0627\u062f\u06cc \u0631\u0627 \u062f\u0631 \u0628\u0633\u062a\u0631 \u062a\u0627\u0631\u06cc\u062e\u06cc \u0622\u0646\u200c\u0647\u0627 \u0628\u0631\u0631\u0633\u06cc \u0645\u06cc\u200c\u06a9\u0646\u062f."
    )
    manager.style_samples = [contaminated, clean]
    projected = manager.get_context_for_chunk(SimpleNamespace(text="state")).style_profile
    assert "see Example" not in projected
    assert clean in projected
    assert "not terminology authority" in projected
    assert manager.style_samples == [contaminated, clean]


def test_research_queries_require_identity_and_followups_are_anchored() -> None:
    result = SearchResult(
        title="Generic scholarship funding",
        url="https://example.test/funding",
        snippet="Applications for student support.",
    )
    ranked, diagnostics = rank_search_results(
        "concepts terminology scholarship", [result], identity="", strict_identity=True
    )
    assert ranked == []
    assert "missing_identity_anchor" in diagnostics[0]["reasons"]

    followups = BookResearcher._normalise_follow_ups(
        ["meaning of a central concept"],
        existing=[],
        limit=2,
        title="A General Academic Book",
        author="A. Scholar",
    )
    assert followups[0].startswith('"A General Academic Book" A. Scholar')


def test_unsupported_research_target_is_context_only() -> None:
    researcher = BookResearcher.__new__(BookResearcher)
    researcher.config = TarjomehConfig()
    terms = researcher._normalise_terms(
        [{
            "source": "analytical category",
            "target": "No Persian translation supported by provided sources",
            "context": "A recurring concept.",
        }],
        [],
    )
    assert terms[0]["status"] == "context_only"
    assert terms[0]["target"] == ""


def test_structural_listing_does_not_create_named_entity_false_positives() -> None:
    source = "Contents\nIntroduction 1\nGovernment and Society 23\nReferences 88"
    back = "Contents\nIntroduction 1\nGovernment and Society 23\nReferences 88"
    result = BackTranslator(SimpleNamespace()).compare(source, back)
    assert result.diagnostics["entity_check_skipped_for_structural_listing"] is True
    assert result.diagnostics["missing_entities"] == []
    assert not result.flagged


class _EventDatabase:
    def __init__(self) -> None:
        self.events = [
            {"event_type": "chunk_started", "payload": {}},
            {
                "event_type": "integrity_edit_rejected",
                "payload": {"stage": "refinement"},
            },
        ]

    def get_chunk_events(self, _job_id: str, _chunk_index: int):
        return list(self.events)

    def log_chunk_event(self, _job_id: str, _chunk_index: int, event_type: str, payload):
        self.events.append({"event_type": event_type, "payload": payload})


def test_review_status_has_one_persisted_canonical_reason() -> None:
    db = _EventDatabase()
    payload = _ensure_chunk_review_reason(db, "job", 0)
    _ensure_chunk_review_reason(db, "job", 0)
    summaries = [event for event in db.events if event["event_type"] == "chunk_review_required"]
    assert payload is not None
    assert payload["reason_codes"] == ["automatic_edit_rejected"]
    assert len(summaries) == 1
    assert _chunk_needs_review(db, "job", 0)
