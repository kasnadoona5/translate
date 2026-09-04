from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import _salvage_local_refinement_edits
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.memory.proper_nouns import ProperNouns
from tarjomeh.persian.typography import PersianTypographer
from tarjomeh.quality.integrity import (
    PostEditIntegrityGate,
    restore_source_note_markers,
)


def test_public_packages_import_in_fresh_process_without_cycle() -> None:
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import tarjomeh.quality; import tarjomeh.glossary; "
                "import tarjomeh.memory; "
                "from tarjomeh.memory import MemoryManager; "
                "assert MemoryManager.__name__ == 'MemoryManager'"
            ),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_unique_source_parenthetical_restores_dropped_note_marker() -> None:
    source = (
        "The idea concerned many writers (think of Jean Bodin, John Locke, "
        "and Jean-Jacques Rousseau)1 - reflections that shaped the state."
    )
    target = (
        "این ایده نزد نویسندگان بسیاری مطرح بود؛ از جمله ژان بودن (Jean Bodin)، "
        "جان لاک (John Locke) و ژان-ژاک روسو (Jean-Jacques Rousseau)؛ "
        "تأملاتی که دولت را شکل دادند."
    )
    repaired, report = restore_source_note_markers(source, target)

    assert "(Jean-Jacques Rousseau)¹" in repaired
    assert report["repair_count"] == 1
    assert report["unresolved"] == []
    assert PostEditIntegrityGate().evaluate(
        source, repaired, stage="v1024_probe"
    ).accepted


def test_ambiguous_note_anchor_is_not_guessed() -> None:
    source = "Compare (Smith)1 with the later account."
    target = "روایت (Smith) را با تفسیر دیگری از (Smith) مقایسه کنید."
    repaired, report = restore_source_note_markers(source, target)

    assert repaired == target
    assert report["repair_count"] == 0
    assert report["unresolved"] == ["1"]


class _SelectiveGate:
    def evaluate(self, _source: str, candidate: str, **_kwargs: object):
        accepted = "نامناسب" not in candidate
        return SimpleNamespace(
            accepted=accepted,
            to_dict=lambda: {"accepted": accepted},
        )


def test_failed_coherent_salvage_keeps_independent_valid_edit() -> None:
    previous = "این تعبیر پرسش‌برانگیز و این ساخت روشن است."
    proposed = "این تعبیر پرسش را مفروض می‌گیرد و این ساخت نامناسب است."
    final, decisions, report = _salvage_local_refinement_edits(
        source="This expression begs the question and this construction is clear.",
        previous=previous,
        proposed=proposed,
        issue_details=[
            {
                "issue_id": "semantic",
                "source_quote": "begs the question",
                "current_persian_quote": "پرسش‌برانگیز",
            },
            {
                "issue_id": "bad",
                "source_quote": "clear",
                "current_persian_quote": "روشن",
            },
        ],
        issue_decisions=[
            {
                "issue_id": "semantic",
                "decision": "accepted",
                "resulting_span": "پرسش را مفروض می‌گیرد",
            },
            {
                "issue_id": "bad",
                "decision": "accepted",
                "resulting_span": "نامناسب",
            },
        ],
        integrity_gate=_SelectiveGate(),  # type: ignore[arg-type]
        protected_terms=[],
        protect_inline_english=False,
        allowed_inline_originals=[],
    )

    assert "پرسش را مفروض می‌گیرد" in final
    assert "این ساخت روشن است" in final
    assert report["committed_count"] == 1
    assert [item["commit_status"] for item in decisions] == [
        "committed_local",
        "not_committed",
    ]


def test_latin_legal_title_keeps_its_internal_comma_and_year() -> None:
    text = "طبق UK Copyright, Designs and Patents Act 1988 عمل شد."
    assert PersianTypographer().process(text) == text


def test_observed_semantic_entity_needs_corroboration_for_authority() -> None:
    nouns = ProperNouns()
    nouns.add_noun(
        "Economic and Social Research Council",
        "شورای پژوهش اقتصادی و اجتماعی",
        "organization",
        provenance="observed_translation",
        evidence_key="chunk:1",
        context_independent=True,
    )
    assert nouns.authority_class_for(
        "Economic and Social Research Council"
    ) == "observed_entity_advisory"
    assert "Persian wording is advisory" in nouns.get_context()

    nouns.add_noun(
        "Economic and Social Research Council",
        "شورای پژوهش اقتصادی و اجتماعی",
        "organization",
        provenance="observed_translation",
        evidence_key="chunk:2",
        context_independent=True,
    )
    assert nouns.authority_class_for(
        "Economic and Social Research Council"
    ) == "source_observed_entity"


def test_dedication_paragraph_cannot_become_style_authority() -> None:
    manager = MemoryManager(TarjomehConfig())
    source_body = "A sustained analytical paragraph develops the argument. " * 12
    target_body = (
        "این بند تحلیلی استدلال را با نثری روشن، دقیق و دانشگاهی بسط می‌دهد. " * 10
    )
    chunk = Chunk(
        0,
        "In memoriam, a valued colleague.\n\n" + source_body,
        "Front matter",
        "",
        metadata={
            "style_eligible": True,
            "style_body_paragraphs": [0, 1],
            "structural_roles": ["body", "body"],
        },
    )
    policy = manager.update_after_translation(
        chunk,
        "به یاد همکاری ارجمند.\n\n" + target_body,
        quality_approved=True,
        style_approved=True,
        long_term_reliable=True,
    )

    assert policy["style_source_genre_excluded_paragraphs"] == [0]
    assert policy["style_sample_added"] is True
    assert "به یاد همکاری" not in manager.style_samples[0]


class _SummaryClient:
    def __init__(self, response: str) -> None:
        self.response = response

    async def chat(self, _prompt: str) -> str:
        return self.response

    def set_operation(self, _operation: str) -> None:
        return None


def test_bilingual_summary_rejects_explicit_count_contradiction() -> None:
    manager = MemoryManager(TarjomehConfig())
    response = (
        "## English Summary\nThe chapter addresses two distinct issues in detail.\n"
        "## خلاصه فارسی\nاین فصل سه مسئله متمایز را با جزئیات بررسی می‌کند."
    )
    report = asyncio.run(manager.update_bilingual_summary(
        _SummaryClient(response),
        "The chapter addresses two issues.",
        "این فصل دو مسئله را بررسی می‌کند.",
    ))

    assert report["candidate_committed"] is False
    assert "bilingual_summary_structure_mismatch" in report[
        "candidate_quality"
    ]["reasons"]


def test_bilingual_summary_rejects_one_letter_drift_from_accepted_term() -> None:
    manager = MemoryManager(TarjomehConfig())
    manager.bilingual_summary.persian_summary = (
        "این بحث نظریه‌ای فراتاریخی را نقد می‌کند."
    )
    response = (
        "## English Summary\nThe discussion rejects a transhistorical theory.\n"
        "## خلاصه فارسی\nاین بحث نظریه‌ای فراجاریخی را نقد می‌کند."
    )
    report = asyncio.run(manager.update_bilingual_summary(
        _SummaryClient(response),
        "The discussion rejects a transhistorical theory.",
        "این بحث نظریه‌ای فراتاریخی را نقد می‌کند.",
    ))

    assert report["candidate_committed"] is False
    assert report["candidate_quality"]["lexical_near_misses"] == [{
        "candidate": "فراجاریخی",
        "established": "فراتاریخی",
        "reason": "one_letter_drift_from_accepted_persian",
    }]


def test_v1024_release_scripts_are_current_and_audit_canonical_admission() -> None:
    root = Path(__file__).parents[1]
    deploy = (root / "scripts" / "deploy_tarjomeh_v1024.sh").read_text(
        encoding="utf-8"
    )
    reports = (root / "scripts" / "audit_tarjomeh_v1024_reports.sh").read_text(
        encoding="utf-8"
    )
    companion = (
        root / "scripts" / "audit_tarjomeh_v1024_companion.sh"
    ).read_text(encoding="utf-8")

    assert deploy.startswith("#!/usr/bin/env bash\n")
    assert 'TAG="v10.24.1"' in deploy
    assert '"coherent unit" in' not in deploy
    assert '"_salvage_local_refinement_edits" in' in deploy
    assert '"coherent_local_edits_committed" in' in deploy
    assert "VERIFY 9ROUTER UNCHANGED" in deploy
    assert 'docker rm -f "$CONTAINER"' in deploy
    assert "docker rm -f 9router" not in deploy
    assert "RUNNING_V1024_CONFIRMED" in deploy
    for audit in (reports, companion):
        assert '${1:-LATEST}' in audit
        assert "source_bound_artifact_recovery" in audit
        assert "missing_final_canonical_admission" in audit
