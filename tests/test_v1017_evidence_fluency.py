from __future__ import annotations

from tarjomeh.context.book_researcher import BookResearcher
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import (
    _research_context_for_memory,
    _salvage_local_refinement_edits,
    audit_translation_language,
)
from tarjomeh.core.prompts import CRITIQUE_PROMPT, REFINE_PROMPT
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.memory.proper_nouns import ProperNouns
from tarjomeh.quality.integrity import (
    newly_source_unjustified_repeated_governed_spans,
    source_unjustified_repeated_governed_span_artifacts,
)


class _AcceptedIntegrity:
    accepted = True

    def to_dict(self) -> dict[str, object]:
        return {"accepted": True, "blocking_count": 0, "findings": []}


class _AcceptingGate:
    def evaluate(self, source: str, candidate: str, **kwargs: object):
        return _AcceptedIntegrity()


def _salvage(
    source: str,
    previous: str,
    current_span: str,
    resulting_span: str,
):
    proposed = previous.replace(current_span, resulting_span, 1)
    return _salvage_local_refinement_edits(
        source=source,
        previous=previous,
        proposed=proposed,
        issue_details=[{
            "issue_id": "fluency",
            "source_quote": source,
            "current_persian_quote": current_span,
        }],
        issue_decisions=[{
            "issue_id": "fluency",
            "decision": "accepted",
            "resulting_span": resulting_span,
        }],
        integrity_gate=_AcceptingGate(),  # type: ignore[arg-type]
        protected_terms=[],
        protect_inline_english=False,
        allowed_inline_originals=[],
    )


def test_governed_phrase_repetition_is_source_relative() -> None:
    target = (
        "این بند دربارهٔ زبان بحث می‌کند و دربارهٔ زبان توضیح می‌دهد."
    )
    unsupported = source_unjustified_repeated_governed_span_artifacts(
        "This paragraph discusses language and explains its role.", target
    )
    supported = source_unjustified_repeated_governed_span_artifacts(
        "It speaks about language and says more about language.", target
    )

    assert unsupported
    assert unsupported[0]["normalized_phrase"] == "درباره زبان"
    assert supported == []


def test_local_salvage_rejects_new_governed_phrase_duplication() -> None:
    previous = "این بند دربارهٔ زبان و نقش آن بحث می‌کند."
    current = "دربارهٔ زبان و نقش آن"
    replacement = "دربارهٔ زبان سخن می‌گوید و دربارهٔ زبان"

    final, decisions, report = _salvage(
        "This paragraph discusses language and its role.",
        previous,
        current,
        replacement,
    )

    assert final == previous
    assert report["committed_count"] == 0
    assert decisions[0]["commit_reason"] == "new_governed_phrase_repetition"
    assert newly_source_unjustified_repeated_governed_spans(
        "This paragraph discusses language and its role.",
        previous,
        previous.replace(current, replacement, 1),
    )


def test_local_salvage_accepts_natural_predicate_restructuring() -> None:
    previous = (
        "آن‌ها مسائل را حل نمی‌کنند و از این روند دل‌زده می‌شوند."
    )
    replacement = (
        "آن‌ها بیش از آنکه مسائل را حل کنند، از این روند دل‌زده می‌شوند."
    )

    final, decisions, report = _salvage(
        "Rather than solving the problems, they become disenchanted with them.",
        previous,
        previous,
        replacement,
    )

    assert final == replacement
    assert report["committed_count"] == 1
    assert decisions[0]["commit_status"] == "committed_local"


def test_language_audit_exposes_governed_phrase_damage() -> None:
    report = audit_translation_language(
        "The argument concerns language and its historical role.",
        "استدلال دربارهٔ زبان بحث می‌کند و دربارهٔ زبان توضیح می‌دهد.",
    )

    assert report["review_required"] is True
    assert report["repeated_governed_span_count"] == 1


def test_automatic_terminology_requires_independent_recurrence() -> None:
    memory = ProperNouns()
    first = memory.add_noun(
        "historical semantics",
        "معناشناسی تاریخی",
        category="term",
        provenance="incremental_extraction",
        evidence_key="chunk:one",
        context_independent=True,
    )
    second = memory.add_noun(
        "historical semantics",
        "معناشناسی تاریخی",
        category="term",
        provenance="incremental_extraction",
        evidence_key="chunk:two",
        context_independent=True,
    )

    assert first["authority_class"] == "contextual_advisory"
    assert second["authority_class"] == "recurring_advisory"
    context = memory.get_context(source_text="historical semantics matters")
    assert "authority=recurring_advisory" in context
    assert "curated glossary and source context override it" in context


def test_reviewed_correction_requires_two_distinct_source_passages() -> None:
    memory = ProperNouns()
    first = memory.add_noun(
        "state project",
        "پروژه دولت",
        category="term",
        provenance="accepted_correction",
        evidence_key="review:one",
        context_independent=True,
    )
    repeated_same_evidence = memory.add_noun(
        "state project",
        "پروژه دولت",
        category="term",
        provenance="accepted_correction",
        evidence_key="review:one",
        context_independent=True,
    )
    second = memory.add_noun(
        "state project",
        "پروژه دولت",
        category="term",
        provenance="accepted_correction",
        evidence_key="review:two",
        context_independent=True,
    )

    assert first["authority_class"] == "reviewed_advisory"
    assert repeated_same_evidence["authority_class"] == "reviewed_advisory"
    assert second["authority_class"] == "canonical_reviewed"


def test_one_reviewed_correction_repairs_stale_summary_but_stays_advisory() -> None:
    manager = MemoryManager(TarjomehConfig())
    manager.bilingual_summary.persian_summary = (
        "این فصل فروپاشی دولت را بررسی می‌کند."
    )
    manager.proper_nouns.add_noun(
        "state failure",
        "فروپاشی دولت",
        category="term",
        provenance="incremental_extraction",
        evidence_key="chunk:one",
        context_independent=True,
    )
    correction = manager.proper_nouns.add_noun(
        "state failure",
        "شکست دولت",
        category="term",
        provenance="accepted_correction",
        evidence_key="review:one",
        context_independent=True,
    )

    report = manager.reconcile_bilingual_summary()

    assert correction["authority_class"] == "reviewed_advisory"
    assert manager.bilingual_summary.persian_summary == (
        "این فصل شکست دولت را بررسی می‌کند."
    )
    assert report["replacement_count"] == 1


def test_dense_style_evidence_cannot_replace_a_clear_anchor() -> None:
    manager = MemoryManager(TarjomehConfig())
    clear = (
        "این پژوهش مناسبات نهادی را در زمینه تاریخی آن‌ها بررسی می‌کند."
    )
    dense = " ".join(["این تحلیل نهادی"] * 35) + "."
    manager.style_samples = [clear]

    report = manager._update_style_profile(dense)

    assert report["accepted"] is False
    assert report["reason"] == "no_fluent_complete_paragraph"
    assert "sentence_too_dense_for_style_anchor" in report["rejections"][0][
        "reasons"
    ]
    assert manager.style_samples == [clear]


def test_research_terms_keep_identity_evidence_and_advisory_authority() -> None:
    researcher = BookResearcher.__new__(BookResearcher)
    researcher.config = TarjomehConfig()
    url = "https://doi.org/10.1000/example"
    terms = researcher._normalise_terms(
        [{
            "source": "strategic-relational approach",
            "target": "رویکرد راهبردی-رابطه‌ای",
            "source_urls": [url],
            "confidence": "high",
        }],
        [{
            "url": url,
            "title": "The identified book",
            "snippet": "The author defines the strategic-relational approach.",
            "source_authority": "scholarly",
            "identity_evidence": {
                "strong_title_match": True,
                "author_support": True,
            },
        }],
    )

    assert terms[0]["identity_supported"] is True
    assert terms[0]["evidence_type"] == "source_supported"
    assert terms[0]["authority"] == "advisory_context_only"
    assert terms[0]["supporting_excerpts"][0]["url"] == url
    prompt_context = _research_context_for_memory({
        "status": "completed",
        "book_context": "Relevant context.",
        "terms": terms,
    })
    assert "book_identity=supported" in prompt_context
    assert "never override the current source" in prompt_context


def test_prompts_preserve_coordinated_source_propositions() -> None:
    assert "matrix action, coordinated actions" in CRITIQUE_PROMPT
    assert "verify every coordinated source member separately" in REFINE_PROMPT
