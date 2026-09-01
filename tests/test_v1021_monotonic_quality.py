from __future__ import annotations

import asyncio
from types import SimpleNamespace

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import (
    _best_source_faithful_version,
    _candidate_regression_details,
)
from tarjomeh.jobs.database import JobDatabase
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.quality.integrity import (
    repair_source_grounded_language_artifacts,
)


def _critique(
    issues: list[dict[str, object]] | None = None,
    *,
    accuracy: float = 9.0,
    terminology: float = 9.0,
    fluency: float = 9.0,
    register: float = 9.0,
) -> SimpleNamespace:
    scores = (accuracy, terminology, fluency, register)
    return SimpleNamespace(
        issue_details=issues or [],
        issues=[],
        accuracy=accuracy,
        terminology=terminology,
        fluency=fluency,
        register=register,
        average=sum(scores) / len(scores),
        valid=True,
    )


def _major_omission() -> dict[str, object]:
    return {
        "issue_id": "missing-accumulation",
        "category": "omission",
        "severity": "major",
        "confidence": 0.95,
        "source_segment_id": "p1:s2",
        "source_quote": "profit-oriented, market-mediated accumulation",
        "current_persian_quote": "سودمحور و با میانجی‌گری بازار",
        "suggested_correction": "انباشت سودمحور با میانجی‌گری بازار",
        "rationale": "The governing concept accumulation is omitted.",
    }


def test_source_fidelity_outranks_fewer_unrelated_blockers() -> None:
    style_blockers = [
        {
            "issue_id": f"term-{index}",
            "category": "terminology",
            "severity": "major",
            "confidence": 0.8,
            "source_quote": "",
            "current_persian_quote": "",
        }
        for index in range(2)
    ]
    complete = _critique(style_blockers, fluency=8.0)
    smoother_but_incomplete = _critique([_major_omission()], accuracy=8.0)

    selected = _best_source_faithful_version([
        ("ترجمه کامل اما نیازمند ویرایش", complete),
        ("ترجمه روان‌تر اما ناقص", smoother_but_incomplete),
    ])

    assert selected is not None
    assert selected[0] == 0
    assert selected[1] == "ترجمه کامل اما نیازمند ویرایش"


def test_new_source_obligation_regression_does_not_need_target_overlap() -> None:
    regressions = _candidate_regression_details(
        _critique([_major_omission()]),
        _critique(),
        ["بخشی کاملا متفاوت از نامزد ویرایش شده"],
    )

    assert [item["issue_id"] for item in regressions] == [
        "missing-accumulation"
    ]


def test_existing_source_obligation_is_not_misattributed_to_new_candidate() -> None:
    issue = _major_omission()
    regressions = _candidate_regression_details(
        _critique([issue]),
        _critique([dict(issue)]),
        ["بخشی کاملا متفاوت از نامزد ویرایش شده"],
    )

    assert regressions == []


def test_short_adjacent_persian_duplicate_is_repaired_source_relatively() -> None:
    repaired, report = repair_source_grounded_language_artifacts(
        "The relation changes over time.",
        "این رابطه که که در طول زمان تغییر می‌کند.",
    )

    assert "که که" not in repaired
    assert any(
        item["type"] == "adjacent_duplicate"
        for item in report["repairs"]
    )


def test_deliberate_source_repetition_is_not_removed() -> None:
    target = "این همان چیزی است که که باید توضیح داده شود."
    repaired, report = repair_source_grounded_language_artifacts(
        "This is what that that must explain.",
        target,
    )

    assert repaired == target
    assert not any(
        item["type"] in {"adjacent_duplicate", "adjacent_duplicate_phrase"}
        for item in report["repairs"]
    )


class _SummaryClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def set_operation(self, _operation: str) -> None:
        return None

    async def chat(self, _prompt: str) -> str:
        return self.response


def test_incomplete_summary_candidate_cannot_replace_previous_memory() -> None:
    manager = MemoryManager(TarjomehConfig())
    manager.bilingual_summary.english_summary = (
        "The chapter explains the institutional argument in historical context."
    )
    manager.bilingual_summary.persian_summary = (
        "این فصل استدلال نهادی را در زمینه تاریخی آن توضیح می‌دهد."
    )
    before = manager.bilingual_summary.serialize()

    report = asyncio.run(manager.update_bilingual_summary(
        _SummaryClient("## English Summary\nOnly English was returned."),
        "new source",
        "ترجمه جدید",
    ))

    assert report["candidate_committed"] is False
    assert manager.bilingual_summary.serialize() == before


def test_complete_bilingual_summary_candidate_is_committed_as_advisory() -> None:
    manager = MemoryManager(TarjomehConfig())
    response = (
        "## English Summary\n"
        "The chapter develops a careful account of institutions and social power.\n"
        "## خلاصه فارسی\n"
        "این فصل روایتی دقیق از نهادها و قدرت اجتماعی عرضه می‌کند."
    )

    report = asyncio.run(manager.update_bilingual_summary(
        _SummaryClient(response),
        "new source",
        "ترجمه جدید",
        input_trust="advisory_inputs",
        trust_reasons=["test_review"],
    ))

    assert report["candidate_committed"] is True
    assert report["authority"] == "argument_orientation_only"
    assert manager.bilingual_summary.input_trust == "advisory_inputs"


def test_first_worker_release_reason_is_not_overwritten(tmp_path) -> None:
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("job", tmp_path / "book.pdf", {})
    assert db.claim_worker("job", "worker-one")["acquired"] is True

    assert db.release_worker("job", "worker-one", reason="paused") is True
    assert db.release_worker(
        "job", "worker-one", reason="web_worker_finished"
    ) is False

    lease = db.get_worker_lease("job")
    assert lease is not None
    assert lease["release_reason"] == "paused"
    lifecycle = [
        event for event in db.get_chunk_events("job")
        if event["event_type"].startswith("worker_lease_")
    ]
    assert [event["event_type"] for event in lifecycle] == [
        "worker_lease_acquired",
        "worker_lease_released",
    ]
