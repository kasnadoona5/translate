"""Focused regression tests for Q2 integrity and QA reliability."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import TruncatedCompletionError
from tarjomeh.core.pipeline import TranslationPipeline, _filter_critique_policy_conflicts
from tarjomeh.quality.back_translator import BackTranslator
from tarjomeh.quality.critique import CritiqueResult, TranslationCritique
from tarjomeh.quality.integrity import PostEditIntegrityGate
from tarjomeh.quality.refiner import RefinementResult, TranslationRefiner


def test_integrity_gate_rejects_loss_and_preserves_equivalent_percent() -> None:
    gate = PostEditIntegrityGate()
    source = "Introduction\n\nIn 1973 the result was 40% [12]."
    previous = "مقدمه\n\nدر سال ۱۹۷۳ نتیجه ۴۰ درصد بود [12]."
    safe = "مقدمه\n\nنتیجه در سال ۱۹۷۳ برابر با ۴۰ درصد بود ¹²."
    unsafe = "مقدمه"

    assert gate.evaluate(source, safe, previous=previous, stage="refinement").accepted
    rejected = gate.evaluate(source, unsafe, previous=previous, stage="refinement")
    assert not rejected.accepted
    checks = {finding.check_id for finding in rejected.blocking}
    assert "numbers_missing" in checks
    assert "note_markers_missing" in checks
    assert "edit_content_loss" in checks


def test_integrity_gate_protects_terms_and_inline_originals() -> None:
    gate = PostEditIntegrityGate()
    source = "Caceres discusses aggregate capital."
    previous = "کاسرس (Caceres) سرمایه کل را بررسی می‌کند."
    candidate = "این نویسنده سرمایه را بررسی می‌کند."
    result = gate.evaluate(
        source,
        candidate,
        previous=previous,
        protected_terms=["سرمایه کل"],
        protect_inline_english=True,
    )
    checks = {finding.check_id for finding in result.blocking}
    assert "protected_terminology_removed" in checks
    assert "english_original_removed" in checks


def test_critic_repairs_malformed_json_and_preserves_decimal_scores() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.responses = iter([
                "not json",
                json.dumps({
                    "scores": {
                        "accuracy": 8.7,
                        "fluency": 9.2,
                        "terminology": 7.8,
                        "register": 9.1,
                    },
                    "overall": 8.65,
                    "issues": [],
                }),
            ])

        async def chat(self, prompt: str) -> str:
            return next(self.responses)

    result = asyncio.run(TranslationCritique(FakeLLM()).critique("source", "ترجمه"))
    assert result.valid
    assert result.attempts == 2
    assert result.accuracy == 8.7
    assert result.average == 8.65
    assert any("invalid_json" in error for error in result.validation_errors)


def test_persistently_malformed_critic_is_never_a_valid_low_score() -> None:
    class FakeLLM:
        async def chat(self, prompt: str) -> str:
            return "still not json"

    result = asyncio.run(TranslationCritique(FakeLLM()).critique("source", "ترجمه"))
    assert not result.valid
    assert result.attempts == 2
    assert not result.passes_threshold(1)


def test_refiner_repairs_json_or_retains_prior_translation() -> None:
    critique = CritiqueResult(average=5, issues=["[MAJOR/accuracy] issue"])

    class RepairingLLM:
        def __init__(self) -> None:
            self.responses = iter([
                "plain text",
                json.dumps({
                    "translation": "ترجمه اصلاح‌شده",
                    "decision": "revised",
                    "rationale": "The accuracy concern was valid.",
                }, ensure_ascii=False),
            ])

        async def chat(self, prompt: str) -> str:
            return next(self.responses)

    repaired = asyncio.run(
        TranslationRefiner(RepairingLLM()).refine_with_decision(
            "source", "ترجمه قبلی", critique
        )
    )
    assert repaired.valid
    assert repaired.attempts == 2
    assert repaired.translation == "ترجمه اصلاح‌شده"

    class BrokenLLM:
        async def chat(self, prompt: str) -> str:
            return "plain text"

    retained = asyncio.run(
        TranslationRefiner(BrokenLLM()).refine_with_decision(
            "source", "ترجمه قبلی", critique
        )
    )
    assert not retained.valid
    assert retained.translation == "ترجمه قبلی"
    assert retained.decision == "preserved"


def test_back_translation_uses_structured_risks_not_overlap_alone() -> None:
    checker = BackTranslator(MagicMock())
    paraphrase = checker.compare(
        "Workers organize collective action across difficult conditions.",
        "Laborers build shared resistance under harsh circumstances.",
    )
    assert paraphrase.similarity_score < 0.5
    assert not paraphrase.flagged
    assert paraphrase.diagnostics["risk_flags"] == []

    drift = checker.compare(
        "The value was not 40 percent in 1973.",
        "The value was 50 percent.",
    )
    assert drift.flagged
    assert "numbers_missing_or_changed" in drift.diagnostics["risk_flags"]
    assert "numbers_added_or_changed" in drift.diagnostics["risk_flags"]
    assert "negation_mismatch" in drift.diagnostics["risk_flags"]


def test_back_translation_entity_matching_allows_harmless_title_descriptor() -> None:
    checker = BackTranslator(MagicMock())

    preserved = checker.compare(
        "The Politics of Operations examines contemporary capitalism.",
        "The book Politics of Operations examines contemporary capitalism.",
    )
    assert preserved.diagnostics["missing_entities"] == []
    assert "named_entities_missing" not in preserved.diagnostics["risk_flags"]

    missing_meaningful_token = checker.compare(
        "The Politics of Operations examines contemporary capitalism.",
        "The book Politics examines contemporary capitalism.",
    )
    assert missing_meaningful_token.diagnostics["missing_entities"] == [
        "The Politics of Operations"
    ]
    assert (
        "named_entities_missing"
        in missing_meaningful_token.diagnostics["risk_flags"]
    )


def test_back_translation_entity_matching_allows_geographic_demonym() -> None:
    checker = BackTranslator(MagicMock())

    preserved = checker.compare(
        "The study covers Latin American rural areas.",
        "The study covers rural areas of Latin America.",
    )
    assert preserved.diagnostics["missing_entities"] == []

    missing_region = checker.compare(
        "The study covers Latin American rural areas.",
        "The study covers Latin rural areas.",
    )
    assert missing_region.diagnostics["missing_entities"] == ["Latin American"]

def test_integrity_uses_authorized_original_allowlist() -> None:
    gate = PostEditIntegrityGate()
    source = "capital Monsanto field"
    previous = "sarmaye (capital), monsanto (Monsanto), meydan."
    cleaned = "sarmaye, monsanto (Monsanto), meydan."

    accepted = gate.evaluate(
        source,
        cleaned,
        previous=previous,
        protect_inline_english=True,
        allowed_inline_originals=["Monsanto"],
    )
    assert accepted.accepted

    missing_name = gate.evaluate(
        source,
        "sarmaye, monsanto, meydan.",
        previous=previous,
        protect_inline_english=True,
        allowed_inline_originals=["Monsanto"],
    )
    assert "english_original_removed" in {
        finding.check_id for finding in missing_name.blocking
    }

    unauthorized = gate.evaluate(
        source,
        "sarmaye, monsanto (Monsanto), meydan (field).",
        previous=cleaned,
        protect_inline_english=True,
        allowed_inline_originals=["Monsanto"],
    )
    assert "unauthorized_english_original_added" in {
        finding.check_id for finding in unauthorized.blocking
    }


def test_source_citations_are_separate_from_inline_original_policy() -> None:
    gate = PostEditIntegrityGate()
    source = "Marx discusses this point (Marx, 1973)."
    previous = "Marx says so (Marx, 1973)."

    accepted = gate.evaluate(
        source,
        previous,
        protect_inline_english=True,
        allowed_inline_originals=[],
    )
    assert accepted.accepted
    assert not {
        finding.check_id for finding in accepted.findings
    } & {"unauthorized_english_original_present"}

    removed = gate.evaluate(
        source,
        "Marx says so.",
        previous=previous,
        protect_inline_english=True,
        allowed_inline_originals=[],
    )
    assert "source_citation_removed" in {
        finding.check_id for finding in removed.blocking
    }


def test_policy_filter_withholds_only_impossible_critic_instructions() -> None:
    issues = [
        {
            "formatted": "add capital",
            "current_translation": "sarmaye",
            "suggested_fix": "sarmaye (capital)",
        },
        {
            "formatted": "remove Monsanto",
            "current_translation": "monsanto (Monsanto)",
            "suggested_fix": "monsanto",
        },
        {
            "formatted": "accuracy fix",
            "current_translation": "wrong rendering",
            "suggested_fix": "better rendering",
        },
    ]
    critique = CritiqueResult(
        accuracy=8,
        fluency=9,
        terminology=7,
        register=9,
        average=8.25,
        issues=[item["formatted"] for item in issues],
        issue_details=issues,
    )

    conflicts = _filter_critique_policy_conflicts(
        critique, "capital Monsanto", ["Monsanto"]
    )

    assert len(conflicts) == 2
    assert critique.issues == ["accuracy fix"]
    assert critique.accuracy == 8
    assert critique.average == 8.25


def test_q2_config_bounds_are_validated() -> None:
    config = TarjomehConfig()
    config.translation.qa_json_retries = 3
    try:
        config.validate()
    except ValueError as exc:
        assert "qa_json_retries" in str(exc)
    else:
        raise AssertionError("Invalid QA retry count was accepted")


def test_pipeline_rejects_lossy_refinement_and_keeps_prior_text() -> None:
    config = TarjomehConfig()
    config.translation.enable_web_context = False
    config.translation.enable_back_translation = False
    config.translation.enable_critique = True
    config.translation.max_refine_iterations = 1
    config.glossary.enable_compliance_check = False

    pipeline = object.__new__(TranslationPipeline)
    pipeline.config = config
    pipeline.db = MagicMock()
    pipeline.db.get_job.return_value = {"status": "running"}
    pipeline.llm_client = MagicMock()
    initial = "مقدمه\n\nدر سال ۱۹۷۳ نتیجه ۴۰ درصد بود [1]."
    pipeline.llm_client.complete.return_value = initial

    memory_context = MagicMock()
    memory_context.style_profile = ""
    memory_context.proper_nouns = ""
    memory_context.long_term = ""
    memory_context.short_term = ""
    memory_context.bilingual_summary = ""
    memory_context.format.return_value = ""
    memory_manager = MagicMock()
    memory_manager.get_context_for_chunk.return_value = memory_context

    glossary = MagicMock()
    glossary.find_terms.return_value = []
    glossary.format_for_prompt.return_value = ""

    class Critic:
        async def critique(self, *args, **kwargs):
            return CritiqueResult(
                accuracy=5, fluency=8, terminology=8, register=8, average=7.25,
                issues=['[MAJOR/accuracy] source: "1973 and 40%"'],
            )

    class Refiner:
        async def refine_with_decision(self, *args, **kwargs):
            return RefinementResult(
                translation="مقدمه", decision="revised", rationale="Applied critique."
            )

    chunk = Chunk(
        0,
        "Introduction\n\nIn 1973 the result was 40% [1].",
        "Introduction",
        "",
        metadata={"paragraph_indices": [0, 1]},
    )
    result = pipeline._translate_single_chunk(
        idx=0,
        chunk=chunk,
        memory_manager=memory_manager,
        web_searcher=MagicMock(),
        glossary_manager=glossary,
        compliance_checker=MagicMock(),
        critique_tool=Critic(),
        refiner_tool=Refiner(),
        back_translator=MagicMock(),
        translations={},
        job_id="job-1",
    )

    assert result == initial
    event_types = [call.args[2] for call in pipeline.db.log_chunk_event.call_args_list]
    assert "integrity_edit_rejected" in event_types
    assert "critique_needs_review" in event_types


def test_pipeline_keeps_translation_when_critic_provider_exhausts_length() -> None:
    config = TarjomehConfig()
    config.translation.enable_web_context = False
    config.translation.enable_back_translation = False
    config.translation.enable_critique = True
    config.glossary.enable_compliance_check = False

    pipeline = object.__new__(TranslationPipeline)
    pipeline.config = config
    pipeline.db = MagicMock()
    pipeline.db.get_job.return_value = {"status": "running"}
    pipeline.llm_client = MagicMock()
    initial = "\u0645\u0642\u062f\u0645\u0647\n\n\u062a\u0631\u062c\u0645\u0647 \u06a9\u0627\u0645\u0644 \u0648 \u0645\u0639\u062a\u0628\u0631 \u0627\u0633\u062a."
    pipeline.llm_client.complete.return_value = initial
    pipeline.llm_client.last_call_attempt_count.return_value = 4

    memory_context = MagicMock()
    memory_context.style_profile = ""
    memory_context.proper_nouns = ""
    memory_context.long_term = ""
    memory_context.short_term = ""
    memory_context.bilingual_summary = ""
    memory_context.format.return_value = ""
    memory_manager = MagicMock()
    memory_manager.get_context_for_chunk.return_value = memory_context

    glossary = MagicMock()
    glossary.find_terms.return_value = []
    glossary.format_for_prompt.return_value = ""

    class Critic:
        async def critique(self, *args, **kwargs):
            raise TruncatedCompletionError("judge exhausted output budget")

    chunk = Chunk(0, "Introduction\n\nComplete source text.", "Introduction", "")
    result = pipeline._translate_single_chunk(
        idx=0,
        chunk=chunk,
        memory_manager=memory_manager,
        web_searcher=MagicMock(),
        glossary_manager=glossary,
        compliance_checker=MagicMock(),
        critique_tool=Critic(),
        refiner_tool=MagicMock(),
        back_translator=MagicMock(),
        translations={},
        job_id="job-critic-length",
    )

    assert result == initial
    events = {
        call.args[2]: call.args[3]
        for call in pipeline.db.log_chunk_event.call_args_list
    }
    assert events["qa_unavailable"]["component"] == "critic"
    assert events["qa_unavailable"]["attempts"] == 4
    assert events["qa_unavailable"]["failure_type"] == "TruncatedCompletionError"
    assert "chunk_completed" in events
