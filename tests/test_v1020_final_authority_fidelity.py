from __future__ import annotations

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import LLMClient
from tarjomeh.core.pipeline import (
    TranslationPipeline,
    _canonical_final_quality_record,
    _chunk_memory_admission,
    _chunk_style_policy,
    _language_quality_does_not_regress,
    _targeted_language_repair_prompt,
)
from tarjomeh.core.prompts import PERSIAN_READABILITY_REVIEW_PROMPT
from tarjomeh.quality.integrity import (
    repair_source_grounded_language_artifacts,
    source_unjustified_repeated_adjacent_span_artifacts,
)


class _EventDB:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.events = events

    def get_chunk_events(self, _job_id: str, _chunk_index: int):
        return self.events


def _major_omission() -> dict[str, object]:
    return {
        "issue_id": "missing-perspective-count",
        "category": "omission",
        "severity": "major",
        "confidence": 0.97,
        "source_segment_id": "p1:s2",
        "source_quote": "analysed from at least six perspectives",
        "current_persian_quote": "قابل تحلیل است",
        "suggested_correction": "از دست‌کم شش منظر قابل تحلیل است",
        "rationale": "The explicit lower-bound quantity and perspective relation are omitted.",
    }


def _critique_event(issue: dict[str, object]) -> dict[str, object]:
    return {
        "event_type": "critique_completed",
        "payload": {
            "valid": True,
            "blocking_issue_count": 1,
            "scores": {
                "accuracy": 9.0,
                "fluency": 9.0,
                "terminology": 9.0,
                "register": 9.0,
                "average": 9.0,
            },
            "issue_details": [issue],
        },
    }


def test_unresolved_issue_withholds_both_durable_memory_and_style() -> None:
    db = _EventDB([
        {"event_type": "chunk_started", "payload": {}},
        _critique_event(_major_omission()),
    ])

    final = _canonical_final_quality_record(db, "job", 0)
    memory = _chunk_memory_admission(db, "job", 0)
    style = _chunk_style_policy(db, "job", 0)

    assert final["durable_authority"] is False
    assert final["unresolved_blocking_issue_ids"] == [
        "missing-perspective-count"
    ]
    assert memory["long_term_reliable"] is False
    assert memory["continuity_retained"] is True
    assert style["approved"] is False


def test_refiner_veto_resolves_advice_for_both_authority_layers() -> None:
    issue = _major_omission()
    db = _EventDB([
        {"event_type": "chunk_started", "payload": {}},
        _critique_event(issue),
        {
            "event_type": "refinement_completed",
            "payload": {
                "issue_decisions": [{
                    "issue_id": "missing-perspective-count",
                    "decision": "rejected",
                    "commit_status": "not_committed_refiner_rejected",
                }],
            },
        },
    ])

    final = _canonical_final_quality_record(db, "job", 0)
    memory = _chunk_memory_admission(db, "job", 0)
    style = _chunk_style_policy(db, "job", 0)

    assert final["durable_authority"] is True
    assert final["issue_status_counts"] == {
        "rejected_by_source_aware_refiner": 1
    }
    assert memory["long_term_reliable"] is True
    assert style["approved"] is True


def test_source_validated_repair_resolves_only_named_grounded_issue() -> None:
    issue = _major_omission()
    db = _EventDB([
        {"event_type": "chunk_started", "payload": {}},
        _critique_event(issue),
        {
            "event_type": "targeted_language_repair",
            "payload": {
                "accepted_count": 1,
                "resolved_source_issue_ids": ["missing-perspective-count"],
            },
        },
    ])

    final = _canonical_final_quality_record(db, "job", 0)

    assert final["durable_authority"] is True
    assert final["resolved_source_issue_ids"] == [
        "missing-perspective-count"
    ]
    assert final["issue_status_counts"] == {
        "resolved_by_source_validated_repair": 1
    }


def test_source_repair_prompt_is_exactly_grounded_and_accuracy_first() -> None:
    prompt = _targeted_language_repair_prompt(
        "State formation can be analysed from at least six perspectives.",
        "شکل‌گیری دولت قابل تحلیل است.",
        {},
        structural_role="body",
        source_fidelity_findings=[_major_omission()],
    )

    assert "analysed from at least six perspectives" in prompt
    assert "از دست‌کم شش منظر قابل تحلیل است" in prompt
    assert "restore exactly that obligation" in prompt
    assert "do not rewrite unrelated correct wording" in prompt


def test_language_repair_may_fix_source_issue_but_not_add_surface_defects() -> None:
    before = {
        "repeated_adjacent_span_count": 0,
        "parenthesis_artifact_count": 0,
    }
    safe = {
        "repeated_adjacent_span_count": 0,
        "parenthesis_artifact_count": 0,
    }
    unsafe = {
        "repeated_adjacent_span_count": 1,
        "parenthesis_artifact_count": 0,
    }

    assert _language_quality_does_not_regress(before, safe)
    assert not _language_quality_does_not_regress(before, unsafe)


def test_duplicate_phrase_with_optional_diacritic_is_repaired_once() -> None:
    source = "It shows how a claim to legitimacy is made."
    target = (
        "این امر نشان می‌دهد و چگونه مدعیِ چگونه مدعی نوعی "
        "مشروعیت‌بخشی می‌شود."
    )

    findings = source_unjustified_repeated_adjacent_span_artifacts(source, target)
    repaired, report = repair_source_grounded_language_artifacts(source, target)

    assert findings[0]["normalized_phrase"] == "چگونه مدعی"
    assert "چگونه مدعیِ چگونه مدعی" not in repaired
    assert "چگونه مدعیِ نوعی" in repaired
    assert report["repairs"][0]["type"] == "adjacent_duplicate_phrase"


def test_source_authored_repetition_is_not_removed() -> None:
    source = "It asks how a claim how a claim becomes legitimate."
    target = "می‌پرسد چگونه یک ادعا چگونه یک ادعا مشروع می‌شود."

    repaired, report = repair_source_grounded_language_artifacts(source, target)

    assert repaired == target
    assert not any(
        item["type"] == "adjacent_duplicate_phrase"
        for item in report["repairs"]
    )


def test_duplicate_finding_offsets_remain_absolute_after_first_paragraph() -> None:
    source = "First paragraph.\n\nIt describes an institutional relation."
    target = (
        "بند نخست.\n\n"
        "این متن یک رابطه نهادی یک رابطه نهادی را توصیف می‌کند."
    )

    findings = source_unjustified_repeated_adjacent_span_artifacts(source, target)

    assert target[findings[0]["offset"]:findings[0]["end_offset"]] == (
        "یک رابطه نهادی یک رابطه نهادی"
    )
    assert target[findings[0]["second_offset"]:].startswith("یک رابطه نهادی")


def test_llm_start_is_visible_without_counting_as_completed_attempt() -> None:
    class _DB:
        def __init__(self) -> None:
            self.events: list[tuple[int, str, dict[str, object]]] = []
            self.heartbeats: list[tuple[str, int | None]] = []

        def heartbeat_worker(
            self, _job_id: str, _worker_id: str, *, stage: str,
            chunk_index: int | None,
        ) -> bool:
            self.heartbeats.append((stage, chunk_index))
            return True

        def log_chunk_event(
            self, _job_id: str, chunk_index: int, event_type: str,
            payload: dict[str, object],
        ) -> None:
            self.events.append((chunk_index, event_type, payload))

    db = _DB()
    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.db = db
    pipeline.current_job_id = "job"
    pipeline.worker_id = "worker"
    pipeline._worker_claimed = True

    pipeline._record_llm_attempt({
        "phase": "started",
        "operation": "critique",
        "attempt": 1,
        "max_attempts": 2,
        "chunk_index": 11,
    })

    assert db.events[0][1] == "llm_call_started"
    assert all(event_type != "llm_call_attempt" for _, event_type, _ in db.events)
    assert db.heartbeats[-1] == ("llm:critique:attempt-1", 11)


def test_llm_start_observer_does_not_change_attempt_observer_contract() -> None:
    client = LLMClient(TarjomehConfig())
    starts: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    client.set_start_observer(starts.append)
    client.set_attempt_observer(attempts.append)
    try:
        client._emit_start({
            "phase": "started",
            "operation": "translation",
            "attempt": 1,
        })
    finally:
        client.close()

    assert starts[0]["operation"] == "translation"
    assert attempts == []


def test_readability_prompt_covers_malformed_words_without_policing_terms() -> None:
    assert "malformed word" in PERSIAN_READABILITY_REVIEW_PROMPT
    assert "construction" in PERSIAN_READABILITY_REVIEW_PROMPT
    assert "uncommon technical vocabulary" in PERSIAN_READABILITY_REVIEW_PROMPT
