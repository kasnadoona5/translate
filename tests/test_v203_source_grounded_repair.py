"""General positive and over-correction controls for the bounded final pass."""

import asyncio
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.context.book_researcher import BookResearcher
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import (
    _QUALITY_ATTEMPT_CONTEXT,
    TranslationPipeline,
    _canonical_final_quality_record,
    _critique_for_event,
    _refinement_span_contract,
)
from tarjomeh.jobs.database import JobDatabase
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.quality.critique import CritiqueResult
from tarjomeh.quality.integrity import PostEditIntegrityGate
from tarjomeh.quality.refiner import RefinementResult, TranslationRefiner


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.mark.parametrize("mode,budget", [("academic", 1), ("quality", 0), ("fast", 0)])
def test_fresh_defaults_and_legacy_saved_jobs(mode, budget):
    raw = {"translation": {"mode": mode}, "llm": {"provider": "ollama"}}
    fresh = TarjomehConfig._from_raw(raw)
    fresh._apply_mode_preset(raw)
    fresh.validate()
    assert fresh.translation.extra_final_refine_attempts == budget
    assert TarjomehConfig.from_dict(raw).translation.extra_final_refine_attempts == 0
    raw["translation"]["extra_final_refine_attempts"] = 1
    assert TarjomehConfig.from_dict(raw).translation.extra_final_refine_attempts == 1


@pytest.mark.parametrize("budget", [-1, 2, True, 1.0, "1"])
def test_budget_cannot_be_unbounded(budget):
    config = TarjomehConfig()
    config.llm.provider = "ollama"
    config.translation.extra_final_refine_attempts = budget
    with pytest.raises(ValueError, match="extra_final_refine_attempts"):
        config.validate()


@pytest.mark.parametrize("date", ["2015/2016", "2015-2016", "2015 or 2016", "2016"])
def test_bibliography_date_before_publisher_is_disclosed_without_guessing(date):
    text = f"A general book ({date}, Example Press) examines institutions."
    evidence = [{"kind": "first_publication", "year": 2016}]
    result, conflicts = BookResearcher._reconcile_publication_context(text, evidence)
    assert date not in result and "examines institutions" in result
    assert conflicts[0]["kind"] == "unspecified_publication"


def test_typed_alternative_date_is_not_proven_by_its_first_year():
    result, conflicts = BookResearcher._reconcile_publication_context(
        "First published in 2016/2015. Context survives.",
        [{"kind": "first_publication", "year": 2016}],
    )
    assert result == "Context survives."
    assert conflicts[0]["years"] == [2015, 2016]


def test_historical_years_are_not_publication_metadata():
    text = "The crisis (2015/2016) transformed institutions. Legal reforms occurred in 2016."
    assert BookResearcher._reconcile_publication_context(text, []) == (text, [])


def test_span_contract_preserves_a_changed_head_without_copying_it_twice():
    before = "این رویکرد مبهم است."
    after = "این نظریه روشن است."
    assert not _refinement_span_contract(
        "This theory is clear.", before, after, "p1:s1", "مبهم",
        "رویکرد مبهم", "نظریه روشن",
    )
    assert _refinement_span_contract(
        "This theory is clear.", before, after, "p1:s1", "مبهم", "", "نظریه روشن",
    ) == "local_span_disagrees_with_candidate_diff"


def test_span_contract_never_borrows_a_quote_from_another_paragraph():
    assert _refinement_span_contract(
        "First.\n\nSecond.", "متن مبهم است.\n\nمتن روشن است.",
        "متن مبهم است.\n\nمتن روشن است.", "p1:s1", "مبهم", "", "روشن",
    ) == "resulting_span_not_unique_in_candidate_paragraph"


def test_original_span_must_cover_exact_quote_and_have_proven_paragraph_scope():
    args = ("A source.", "متن مبهم است.", "متن روشن است.")
    assert _refinement_span_contract(*args, "p1:s1", "مبهم", "متن", "متن روشن")
    assert _refinement_span_contract(*args, "", "مبهم", "مبهم", "روشن")
    assert _refinement_span_contract(
        "A source.", "متن مبهم است.\n\nمتن دوم.", "متن روشن است.\n\nمتن دوم.",
        "p1:s1", "مبهم", "مبهم", "روشن",
    ) == "uncertain_candidate_paragraph_alignment"


def test_refiner_preserves_span_evidence_without_truncating_or_breaking_legacy():
    span = "x" * 360
    body = {"translation": span, "decision": "revised", "rationale": "Source supported.",
            "issue_decisions": [{"issue_id": "a", "decision": "accepted",
                                 "resulting_span": span, "original_span": "y" * 300,
                                 "rationale": "Evidence."}]}
    parsed = TranslationRefiner._parse_response(json.dumps(body), expected_issue_ids=["a"])
    assert parsed.valid and parsed.issue_decisions[0]["resulting_span"] == span
    assert parsed.issue_decisions[0]["original_span"] == "y" * 300
    del body["issue_decisions"][0]["original_span"]
    assert TranslationRefiner._parse_response(json.dumps(body), expected_issue_ids=["a"]).valid


SOURCE = "The report raises at most two issues."
BEFORE = "این گزارش در نهایت فقط دو مسئله را مطرح می‌کند."
AFTER = "این گزارش حداکثر دو مسئله را مطرح می‌کند."


def _setup(tmp_path):
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("control", tmp_path / "source.txt", {})
    chunk = Chunk(index=0, text=SOURCE, chapter_title="Analysis", section_title="",
                  metadata={"paragraph_indices": [0], "structural_roles": ["body"]})
    db.save_chunks("control", [chunk])
    db.claim_worker("control", "owner")
    return db, chunk


def _reservation(db, worker="owner", candidate=BEFORE):
    return db.reserve_extra_final_refinement(
        "control", 0, source_sha256=_hash(SOURCE), candidate_sha256=_hash(candidate),
        issue_ids=["bound"], worker_id=worker,
    )


def test_allowance_survives_restart_candidate_change_and_double_resume(tmp_path):
    db, _ = _setup(tmp_path)
    assert _reservation(db)
    assert not _reservation(JobDatabase(db.db_path), candidate=AFTER)
    entry = db.get_job_artifact("control", "extra_final_refinement_v1")["entries"]["0"]
    assert entry["consumed"] == 1


def test_reservation_is_atomic_and_checks_worker_and_source(tmp_path):
    db, _ = _setup(tmp_path)
    assert not _reservation(db, worker="other")
    assert not db.reserve_extra_final_refinement(
        "control", 0, source_sha256="wrong", candidate_sha256=_hash(BEFORE),
        issue_ids=["bound"], worker_id="owner",
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(lambda _: _reservation(db), range(2))) == 1


def test_corrupt_allowance_fails_closed(tmp_path):
    db, _ = _setup(tmp_path)
    db.save_job_artifact("control", "extra_final_refinement_v1", {"entries": []})
    assert not _reservation(db)


def _review(details):
    return CritiqueResult(
        accuracy=9, fluency=9, terminology=9, register=9, average=9,
        issue_details=details, issues=["Source bound."] if details else [],
        coverage_complete=True, coverage_checked_segment_ids=["p1:s1"],
    )


def _run_extra(tmp_path, *, reject=False, unavailable=False, regress=False, budget=1,
               stale=False, coverage=True, severity="minor", confidence=0.8):
    db, chunk = _setup(tmp_path)
    issue = {"issue_id": "bound", "category": "accuracy", "severity": "minor",
             "confidence": 0.8, "source_segment_id": "p1:s1",
             "source_quote": "at most two issues", "current_persian_quote": "در نهایت فقط دو",
             "suggested_correction": "حداکثر دو", "rationale": "Preserve the upper bound."}
    db.log_chunk_event("control", 0, "chunk_started", {})
    event = _critique_for_event(
        _review([issue]), 9, -1, candidate_text=BEFORE,
        candidate_stage="final_retained_candidate_validation",
    )
    event["coverage_complete"] = coverage
    event["issue_details"][0].update(severity=severity, confidence=confidence)
    if stale:
        event["candidate_target_hash"] = _hash("other text")
    db.log_chunk_event("control", 0, "critique_completed", event)
    calls = []

    async def refine(*args, **kwargs):
        calls.append("refiner")
        if unavailable:
            raise httpx.ReadTimeout("Unknown transport outcome")
        return RefinementResult(
            translation=BEFORE if reject else AFTER, decision="preserved" if reject else "revised",
            rationale="Source checked.", issue_decisions=[{
                "issue_id": "bound", "decision": "rejected" if reject else "accepted",
                "original_span": "در نهایت فقط دو",
                "resulting_span": "در نهایت فقط دو" if reject else "حداکثر دو",
                "rationale": "Source evidence.",
            }],
        )

    async def critique(*args, **kwargs):
        calls.append("critic")
        return _review([{**issue, "issue_id": "new", "current_persian_quote": "این گزارش",
                         "source_quote": "The report", "severity": "major"}] if regress else [])

    pipeline = object.__new__(TranslationPipeline)
    pipeline.config = TarjomehConfig()
    pipeline.config.translation.extra_final_refine_attempts = budget
    pipeline.db = db
    pipeline.worker_id = "owner"
    pipeline._run_async = asyncio.run
    pipeline._canonicalize_final_translation = lambda job, idx, source, text: text
    kwargs = dict(critique_tool=SimpleNamespace(critique=critique),
                  refiner_tool=SimpleNamespace(refine_with_decision=refine),
                  integrity_gate=PostEditIntegrityGate(), terminology="", review_context="",
                  enforced_entries=[], enforce_auto=False, protected_terms=[],
                  protect_inline_english=False, allowed_inline_originals=[], reviewed_matches=[],
                  threshold=9)
    result = pipeline._extra_final_quality_repair("control", chunk, BEFORE, **kwargs)
    return pipeline, chunk, db, result, calls, kwargs


def test_extra_repairs_then_rechecks_without_committing_early(tmp_path):
    pipeline, chunk, db, result, calls, kwargs = _run_extra(tmp_path)
    assert result == AFTER and calls == ["refiner", "critic"]
    assert not db.get_chunk("control", 0)["translation"]
    assert _canonical_final_quality_record(db, "control", 0, result)["durable_authority"]
    events = db.get_chunk_events("control", 0)
    decision_position = next(i for i, event in enumerate(events)
                             if event["event_type"] == "refinement_completed")
    final_position = max(i for i, event in enumerate(events)
                         if event["event_type"] == "critique_completed")
    assert decision_position < final_position
    pipeline._extra_final_quality_repair("control", chunk, BEFORE, **kwargs)
    assert calls == ["refiner", "critic"]


def test_rejection_is_a_veto_not_a_memory_block_or_second_critique(tmp_path):
    _, _, db, result, calls, _ = _run_extra(tmp_path, reject=True)
    assert result == BEFORE and calls == ["refiner"]
    quality = _canonical_final_quality_record(db, "control", 0, result)
    assert quality["issues"][0]["status"] == "rejected_by_source_aware_refiner"
    assert quality["durable_authority"]


@pytest.mark.parametrize("unavailable,regress", [(True, False), (False, True)])
def test_failed_attempt_keeps_baseline_and_consumed_allowance(tmp_path, unavailable, regress):
    pipeline, chunk, db, result, calls, kwargs = _run_extra(
        tmp_path, unavailable=unavailable, regress=regress
    )
    assert result == BEFORE
    assert not _canonical_final_quality_record(db, "control", 0, result)["durable_authority"]
    previous_calls = list(calls)
    pipeline._extra_final_quality_repair("control", chunk, BEFORE, **kwargs)
    assert calls == previous_calls
    assert not any(event["event_type"] == "qa_unavailable"
                   for event in db.get_chunk_events("control", 0))


def test_disabled_attempt_adds_no_calls_or_reservation(tmp_path):
    _, _, db, result, calls, _ = _run_extra(tmp_path, budget=0)
    assert result == BEFORE and not calls
    assert db.get_job_artifact("control", "extra_final_refinement_v1") is None


@pytest.mark.parametrize("options", [
    {"stale": True}, {"coverage": False}, {"confidence": 0.1},
])
def test_extra_requires_exact_complete_grounded_critique(tmp_path, options):
    _, _, db, result, calls, _ = _run_extra(tmp_path, **options)
    assert result == BEFORE and not calls
    assert db.get_job_artifact("control", "extra_final_refinement_v1") is None


def test_extra_accounting_context_isolated_across_workers_and_owned_event_loop():
    pipeline = object.__new__(TranslationPipeline)

    async def observe():
        await asyncio.sleep(0.01)
        return _QUALITY_ATTEMPT_CONTEXT.get()

    def work(index):
        token = _QUALITY_ATTEMPT_CONTEXT.set({"chunk": index})
        try:
            return pipeline._run_async(observe())
        finally:
            _QUALITY_ATTEMPT_CONTEXT.reset(token)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(work, [1, 2])) == [{"chunk": 1}, {"chunk": 2}]
        assert _QUALITY_ATTEMPT_CONTEXT.get() is None
    finally:
        pipeline.close()


def test_repeated_year_is_still_an_unverified_alternative():
    result, conflicts = BookResearcher._reconcile_publication_context(
        "First published in 2016/2016. Context survives.",
        [{"kind": "first_publication", "year": 2016}],
    )
    assert result == "Context survives." and conflicts


def test_full_rejected_summary_is_auditable_but_never_prompt_authority():
    class SummaryLLM:
        async def chat(self, prompt):
            return ("## English Summary\n"
                    "The theoretical epistem epistemological argument continues."
                    "\n## خلاصه فارسی\nبحث نظری ادامه می‌یابد.")

    manager = MemoryManager(TarjomehConfig())
    report = asyncio.run(manager.update_bilingual_summary(
        SummaryLLM(), "The theoretical and epistemological argument continues.",
        "بحث نظری و معرفت‌شناختی ادامه می‌یابد.",
    ))
    assert not report["candidate_committed"]
    record = manager.to_dict()["last_rejected_summary"]
    assert "epistem epistemological" in record["english_candidate"]["text"]
    assert not record["english_candidate"]["truncated"]
    assert "epistem epistemological" not in manager.bilingual_summary.get_context()
    restored = MemoryManager(TarjomehConfig())
    restored.from_dict(manager.to_dict())
    assert restored.to_dict()["last_rejected_summary"] == record
