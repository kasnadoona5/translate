from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from tarjomeh.core.pipeline import (
    TranslationPipeline,
    _recover_non_regressed_local_edits,
    audit_translation_language,
)
from tarjomeh.core.term_notes import audit_inline_english_originals
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.memory.bilingual_summary import BilingualSummary
from tarjomeh.memory.proper_nouns import (
    ProperNouns,
    automatic_terminology_risk_reasons,
)


class _AcceptingGate:
    def evaluate(self, _source: str, _candidate: str, **_kwargs: object):
        return SimpleNamespace(
            accepted=True,
            to_dict=lambda: {"accepted": True},
        )


def test_second_stage_rollback_retains_only_unimplicated_local_edit() -> None:
    baseline = "این اصطلاح نادقیق است و این ساخت روشن است."
    proposed = "این اصطلاح دقیق است و این ساخت خراب است."
    final, decisions, report, excluded = _recover_non_regressed_local_edits(
        source="This term is precise and this construction is clear.",
        baseline=baseline,
        proposed=proposed,
        issue_details=[
            {
                "issue_id": "term",
                "source_quote": "term is precise",
                "current_persian_quote": "اصطلاح نادقیق",
            },
            {
                "issue_id": "grammar",
                "source_quote": "construction is clear",
                "current_persian_quote": "ساخت روشن",
            },
        ],
        issue_decisions=[
            {
                "issue_id": "term",
                "decision": "accepted",
                "resulting_span": "اصطلاح دقیق",
            },
            {
                "issue_id": "grammar",
                "decision": "accepted",
                "resulting_span": "ساخت خراب",
            },
        ],
        regressions=[{
            "issue_id": "late-grammar",
            "current_persian_quote": "ساخت خراب",
        }],
        integrity_gate=_AcceptingGate(),  # type: ignore[arg-type]
        protected_terms=[],
        protect_inline_english=False,
        allowed_inline_originals=[],
    )

    assert final == "این اصطلاح دقیق است و این ساخت روشن است."
    assert excluded == ["grammar"]
    assert [item["issue_id"] for item in decisions] == ["term"]
    assert report["committed_count"] == 1


def test_ambiguous_late_regression_restores_exact_baseline() -> None:
    baseline = "ترجمهٔ معتبر."
    final, decisions, report, excluded = _recover_non_regressed_local_edits(
        source="A valid translation.",
        baseline=baseline,
        proposed="ترجمهٔ بهتر.",
        issue_details=[],
        issue_decisions=[],
        regressions=[{"current_persian_quote": "عبارت نامرتبط"}],
        integrity_gate=_AcceptingGate(),  # type: ignore[arg-type]
        protected_terms=[],
        protect_inline_english=False,
        allowed_inline_originals=[],
    )

    assert final == baseline
    assert decisions == []
    assert excluded == []
    assert report["policy"] == "ambiguous_regression_restored_exact_baseline"


def test_dash_audit_distinguishes_relational_en_dash_from_em_dash_aside() -> None:
    source = "Power - a mediated condensation - reflects changing forces."
    target = (
        "در رویکرد راهبردی – رابطه‌ای، قدرت — یعنی تراکمی میانجی‌شده — "
        "توازن متغیر نیروها را بازمی‌تاباند."
    )

    report = audit_translation_language(source, target)

    assert report["unbalanced_explanatory_dash_count"] == 0


def test_dash_audit_detects_object_marker_detached_from_governor() -> None:
    source = "People make their own history - and that of others."
    target = "مردم تاریخ خود و تاریخ دیگران را — می‌سازند."

    report = audit_translation_language(source, target)

    assert report["unbalanced_explanatory_dash_count"] == 1
    assert report["unbalanced_explanatory_dash_artifacts"][0]["reason"] == (
        "persian_object_marker_detached_by_dash"
    )


def test_single_word_brand_mapping_does_not_capture_lowercase_lexical_sense() -> None:
    nouns = ProperNouns()
    nouns.add_noun(
        "Polity",
        "پالیتی",
        "publication",
        provenance="observed_translation",
        evidence_key="front-matter",
        context_independent=True,
        source_surface="Polity",
        semantic_role="publisher",
    )

    assert nouns.applies_to_source("Polity", "Published by Polity") is True
    assert nouns.applies_to_source("Polity", "the forms of polity and policy") is False


def test_automatic_term_number_drift_stays_contextual() -> None:
    assert "source_target_number_scope_mismatch" in (
        automatic_terminology_risk_reasons("state failure", "شکست دولت‌ها")
    )
    assert "source_target_number_scope_mismatch" in (
        automatic_terminology_risk_reasons("policy paradigms", "پارادایم سیاستی")
    )


def test_english_summary_uses_ascii_digits_without_changing_persian_half() -> None:
    summary = BilingualSummary()
    summary.update(
        "## English Summary\nChapter ۱۲ develops ٣ claims.\n"
        "## خلاصه فارسی\nفصل ۱۲ سه ادعا را بسط می‌دهد."
    )

    assert summary.english_summary == "Chapter 12 develops 3 claims."
    assert "۱۲" in summary.persian_summary


def test_globally_known_original_still_requires_local_source_grounding() -> None:
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(
        index=0,
        source_text="The statute applies in the UK.",
        translated_text="این قانون در بریتانیا (United Kingdom) اجرا می‌شود.",
    )])

    report = audit_inline_english_originals(
        document,
        {"United Kingdom": "بریتانیا"},
    )

    assert report["kept_authorized"] == 0
    assert report["unapproved_ungrounded_count"] == 1
    assert report["unapproved_ungrounded"][0]["reason"] == (
        "authorized_original_not_grounded_in_source_paragraph"
    )


class _HeartbeatDB:
    def __init__(self) -> None:
        self.events: list[tuple[int, str, dict[str, object]]] = []

    def heartbeat_worker(self, *_args: object, **_kwargs: object) -> None:
        return None

    def log_chunk_event(
        self,
        _job_id: str,
        chunk_index: int,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        self.events.append((chunk_index, event_type, payload))


def test_long_provider_call_emits_progress_without_retrying() -> None:
    pipeline = object.__new__(TranslationPipeline)
    pipeline.current_job_id = "job"
    pipeline.worker_id = "worker"
    pipeline._worker_claimed = True
    pipeline._worker_stage = "llm:critique:attempt-1"
    pipeline._worker_chunk_index = 0
    pipeline._lease_heartbeat_stop = threading.Event()
    pipeline._lease_heartbeat_thread = None
    pipeline._llm_observation_lock = threading.Lock()
    started = time.monotonic() - 121.0
    pipeline._active_llm_calls = {
        ("job", 0, "critique", 1): {
            "job_id": "job",
            "chunk_index": 0,
            "operation": "critique",
            "attempt": 1,
            "started_monotonic": started,
            "last_progress_monotonic": started,
        }
    }
    pipeline.db = _HeartbeatDB()

    pipeline._start_lease_heartbeat(interval_seconds=0.01)
    time.sleep(0.04)
    pipeline._stop_lease_heartbeat()

    progress = [
        event for event in pipeline.db.events
        if event[1] == "llm_call_in_progress"
    ]
    assert len(progress) == 1
    assert progress[0][0] == 0
    assert progress[0][2]["operation"] == "critique"
    assert float(progress[0][2]["elapsed_seconds"]) >= 120.0


def test_v1026_release_scripts_are_current_and_preserve_9router() -> None:
    root = Path(__file__).parents[1]
    scripts = {
        "deploy": (root / "scripts" / "deploy_tarjomeh_v1026.sh").read_text(
            encoding="utf-8"
        ),
        "reports": (
            root / "scripts" / "audit_tarjomeh_v1026_reports.sh"
        ).read_text(encoding="utf-8"),
        "companion": (
            root / "scripts" / "audit_tarjomeh_v1026_companion.sh"
        ).read_text(encoding="utf-8"),
    }
    deploy = scripts["deploy"]
    companion = scripts["companion"]

    assert deploy.startswith("#!/usr/bin/env bash\n")
    assert 'TAG="v10.26.0"' in deploy
    assert 'git diff --quiet v10.24.1 "$TAG"' not in deploy
    assert 'git diff --quiet v10.25.0 "$TAG"' in deploy
    assert "RUNNING_V1026_CONFIRMED" in deploy
    assert "docker builder prune" not in deploy
    assert "docker rm -f 9router" not in deploy
    assert "second_stage_transactional_replay" in deploy
    assert "source_scoped_originals" in deploy
    assert "long_llm_call_progress" in deploy
    assert "llm_call_in_progress" in companion
    assert "lifetime_failure_rate" in companion

    for name, script in scripts.items():
        assert "${1:-LATEST}" in script or name == "deploy"
        blocks = re.findall(
            r"<<'PY'[^\n]*\n(.*?)\nPY(?=\n|$)",
            script,
            flags=re.DOTALL,
        )
        assert blocks, f"no embedded Python found in {name} script"
        for index, block in enumerate(blocks):
            compile(block, f"{name}-embedded-{index}.py", "exec")
