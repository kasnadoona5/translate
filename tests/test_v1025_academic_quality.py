from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import (
    TranslationPipeline,
    _canonical_final_quality_record,
    _grounded_objective_language_issues,
    _salvage_local_refinement_edits,
    _targeted_language_repair_prompt,
)
from tarjomeh.core.prompts import ACADEMIC_EXEMPLARS, GENERAL_EDITORIAL_CONTRACT
from tarjomeh.core.term_notes import normalize_citation_house_style_text
from tarjomeh.memory.manager import (
    _NONREPRESENTATIVE_STYLE_SOURCE_RE,
    MemoryManager,
)
from tarjomeh.quality.integrity import (
    PostEditIntegrityGate,
    extract_note_markers,
    restore_source_note_markers,
)


class _AcceptingGate:
    def evaluate(self, _source: str, _candidate: str, **_kwargs: object):
        return SimpleNamespace(
            accepted=True,
            to_dict=lambda: {"accepted": True},
        )


class _EventDB:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.events = events

    def get_chunk_events(self, _job_id: str, _chunk_index: int):
        return self.events

    def log_chunk_event(
        self,
        _job_id: str,
        _chunk_index: int,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        self.events.append({"event_type": event_type, "payload": payload})


def _salvage(
    *,
    source: str,
    previous: str,
    proposed: str,
    current_span: str,
    resulting_span: str,
):
    return _salvage_local_refinement_edits(
        source=source,
        previous=previous,
        proposed=proposed,
        issue_details=[{
            "issue_id": "issue-1",
            "source_quote": source,
            "current_persian_quote": current_span,
        }],
        issue_decisions=[{
            "issue_id": "issue-1",
            "decision": "accepted",
            "resulting_span": resulting_span,
        }],
        integrity_gate=_AcceptingGate(),  # type: ignore[arg-type]
        protected_terms=[],
        protect_inline_english=False,
        allowed_inline_originals=[],
    )


def test_unique_aligned_sentence_terminal_note_marker_is_restored() -> None:
    source = (
        "Recognition depends on acceptance by other states.2 "
        "And this relationship remains contested."
    )
    target = (
        "به‌رسمیت‌شناسی به پذیرش از سوی دیگر دولت‌ها بستگی دارد. "
        "و این رابطه همچنان محل مناقشه است."
    )

    repaired, report = restore_source_note_markers(source, target)

    assert "دارد.² و این" in repaired
    assert report["repair_count"] == 1
    assert report["repairs"][0]["type"] == (
        "aligned_sentence_terminal_note_marker"
    )
    assert report["unresolved"] == []
    assert PostEditIntegrityGate().evaluate(
        source, repaired, stage="v1025_sentence_note_probe"
    ).accepted


def test_sentence_note_recovery_refuses_split_or_merged_alignment() -> None:
    source = "The first claim ends here.2 A second claim follows."
    merged_target = "ادعای نخست در همین‌جا پایان می‌یابد و ادعای دوم پس از آن می‌آید."

    repaired, report = restore_source_note_markers(source, merged_target)

    assert repaired == merged_target
    assert report["repair_count"] == 0
    assert report["unresolved"] == ["2"]


def test_sentence_note_recovery_refuses_reused_marker_number() -> None:
    source = "First statement.2 Second statement.2 Third statement."
    target = "گزارهٔ نخست. گزارهٔ دوم. گزارهٔ سوم."

    repaired, report = restore_source_note_markers(source, target)

    assert repaired == target
    assert report["repair_count"] == 0
    assert report["unresolved"] == ["2", "2"]


def test_sentence_note_recovery_handles_terminal_closing_quote() -> None:
    source = 'The author calls it "a social relation."2 Another claim follows.'
    target = "نویسنده آن را «رابطه‌ای اجتماعی» می‌نامد. ادعای دیگری در پی می‌آید."

    repaired, report = restore_source_note_markers(source, target)

    assert "می‌نامد.² ادعای" in repaired
    assert report["repair_count"] == 1
    assert report["unresolved"] == []
    assert extract_note_markers('The board measured 5"2 inches.') == []


def test_local_salvage_deduplicates_only_boundary_function_word() -> None:
    previous = "این ادعایی است که در بدو امر پرسش‌برانگیز است."
    resulting = "که در آغاز پاسخ خود را پیشاپیش مفروض می‌گیرد"
    proposed = previous.replace("در بدو امر پرسش‌برانگیز است", resulting)

    final, decisions, report = _salvage(
        source="This claim initially begs the question.",
        previous=previous,
        proposed=proposed,
        current_span="در بدو امر پرسش‌برانگیز است",
        resulting_span=resulting,
    )

    assert "که که" not in final
    assert "که در آغاز پاسخ خود را پیشاپیش مفروض می‌گیرد" in final
    assert report["committed_count"] == 1
    assert decisions[0]["boundary_deduplication"]["word"] == "که"


def test_predicate_guard_recognizes_prefixed_verbs_with_persian_stem_letters() -> None:
    previous = "این ادعا پرسش را پیشاپیش مفروض می‌داند."
    resulting = "پاسخ را پیشاپیش مفروض می‌گیرد"
    proposed = previous.replace("پرسش را پیشاپیش مفروض می‌داند", resulting)

    final, _decisions, report = _salvage(
        source="This claim begs the question.",
        previous=previous,
        proposed=proposed,
        current_span="پرسش را پیشاپیش مفروض می‌داند",
        resulting_span=resulting,
    )

    assert resulting in final
    assert report["committed_count"] == 1


def test_local_salvage_does_not_delete_repeated_content_word() -> None:
    previous = "این دولت در بررسی حاضر اهمیت دارد."
    resulting = "دولت در این بررسی اهمیت محوری دارد"
    proposed = previous.replace("در بررسی حاضر اهمیت دارد", resulting)

    final, decisions, report = _salvage(
        source="This state matters in the present analysis.",
        previous=previous,
        proposed=proposed,
        current_span="در بررسی حاضر اهمیت دارد",
        resulting_span=resulting,
    )

    assert final == previous
    assert report["committed_count"] == 0
    assert decisions[0]["commit_reason"] == "new_adjacent_word_repetition"


def test_local_salvage_does_not_delete_source_authored_repetition() -> None:
    previous = "این همان که است که در متن آمده است."
    resulting = "که در متن به‌صراحت آمده است"
    proposed = previous.replace("در متن آمده است", resulting)

    final, decisions, report = _salvage(
        source="This is the that that appears explicitly in the text.",
        previous=previous,
        proposed=proposed,
        current_span="در متن آمده است",
        resulting_span=resulting,
    )

    assert "که که" in final
    assert report["committed_count"] == 1
    assert decisions[0]["boundary_deduplication"] is None


def test_objective_minor_calque_is_routed_but_vague_style_advice_is_not() -> None:
    actionable = {
        "issue_id": "calque-1",
        "category": "fluency",
        "severity": "minor",
        "current_persian_quote": "در اصطلاحات این رویکرد",
        "suggested_correction": "از منظر این رویکرد",
        "rationale": "This is an opaque prepositional calque with unclear attachment.",
    }
    vague = {
        "issue_id": "style-1",
        "category": "fluency",
        "severity": "minor",
        "current_persian_quote": "این رویکرد",
        "suggested_correction": "چنین رویکردی",
        "rationale": "The alternative may read more elegantly.",
    }

    assert _grounded_objective_language_issues(
        SimpleNamespace(issue_details=[actionable])
    ) == [actionable]
    assert _grounded_objective_language_issues(
        SimpleNamespace(issue_details=[vague])
    ) == []


def test_strictly_repaired_objective_issue_can_regain_memory_authority() -> None:
    issue = {
        "issue_id": "attachment-1",
        "category": "fluency",
        "severity": "major",
        "confidence": 0.95,
        "source_quote": "can be studied from several perspectives",
        "current_persian_quote": "می‌تواند چند منظر مطالعه شود",
        "suggested_correction": "می‌توان آن را از چند منظر مطالعه کرد",
        "rationale": "The governor and predicate attachment are ungrammatical.",
    }
    db = _EventDB([
        {"event_type": "chunk_started", "payload": {}},
        {
            "event_type": "critique_completed",
            "payload": {
                "valid": True,
                "blocking_issue_count": 0,
                "issue_details": [issue],
            },
        },
        {
            "event_type": "targeted_language_repair",
            "payload": {
                "accepted_count": 1,
                "resolved_objective_issue_ids": ["attachment-1"],
            },
        },
    ])

    final = _canonical_final_quality_record(db, "job", 0)

    assert final["durable_authority"] is True
    assert final["resolved_objective_issue_ids"] == ["attachment-1"]
    assert final["issue_status_counts"] == {
        "resolved_by_strict_language_repair": 1
    }


def test_targeted_prompt_keeps_objective_repair_source_bound() -> None:
    issue = {
        "issue_id": "scope-1",
        "category": "readability",
        "severity": "major",
        "current_persian_quote": "عبارت نارسا",
        "suggested_correction": "عبارت روشن",
        "rationale": "The parenthetical scope breaks the governing relation.",
    }
    prompt = _targeted_language_repair_prompt(
        "The source proposition remains complete.",
        "گزارهٔ مبدأ کامل باقی می‌ماند و عبارت نارساست.",
        {},
        structural_role="body",
        objective_language_findings=[issue],
    )

    assert "parenthetical scope" in prompt
    assert "never simplify, merge, omit, or reinterpret" in prompt
    assert "Return exactly one complete Persian paragraph" in prompt


def test_academic_contract_demonstrates_ezafe_without_blanket_insertion() -> None:
    assert ACADEMIC_EXEMPLARS.count("هٔ") >= 2
    assert "ِ" in ACADEMIC_EXEMPLARS
    assert "never add blanket diacritics" in GENERAL_EDITORIAL_CONTRACT


def test_citation_house_style_is_canonical_before_memory_admission() -> None:
    source = "For the argument, see Jessop 1990, 2002."
    target = "برای این استدلال، (see Jessop 1990, 2002) را ببینید."
    expected, changes = normalize_citation_house_style_text(target)
    assert changes
    assert "(ر.ک. Jessop 1990, 2002)" in expected

    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.config = TarjomehConfig()
    pipeline.db = _EventDB([])
    chunk = Chunk(7, source, "Chapter", "")
    canonical = pipeline._canonicalize_final_translation(
        "job", 7, chunk, target
    )
    event = pipeline.db.events[-1]["payload"]

    assert canonical == expected
    assert event["citation_house_style_changes"]
    manager = MemoryManager(TarjomehConfig())
    policy = manager.update_after_translation(chunk, canonical)
    entry = manager.long_term.serialize()[-1]
    assert entry["translation"] == canonical
    assert entry["chunk_index"] == 7
    assert entry["canonical_target_hash"] == policy["canonical_target_hash"]


def test_first_person_dedication_is_not_style_authority() -> None:
    assert _NONREPRESENTATIVE_STYLE_SOURCE_RE.search(
        "I dedicate this book to the memory of my colleague."
    )
    assert _NONREPRESENTATIVE_STYLE_SOURCE_RE.search(
        "We dedicate this volume to our teachers."
    )
    assert not _NONREPRESENTATIVE_STYLE_SOURCE_RE.search(
        "The analysis is dedicated to explaining institutional change."
    )
    assert not _NONREPRESENTATIVE_STYLE_SOURCE_RE.search(
        "Thanks to this distinction, the argument becomes clearer."
    )


def test_v1025_release_scripts_are_current_and_space_safe() -> None:
    root = Path(__file__).parents[1]
    deploy = (root / "scripts" / "deploy_tarjomeh_v1025.sh").read_text(
        encoding="utf-8"
    )
    reports = (
        root / "scripts" / "audit_tarjomeh_v1025_reports.sh"
    ).read_text(encoding="utf-8")
    companion = (
        root / "scripts" / "audit_tarjomeh_v1025_companion.sh"
    ).read_text(encoding="utf-8")

    assert deploy.startswith("#!/usr/bin/env bash\n")
    assert 'TAG="v10.25.0"' in deploy
    assert "BUILD SHARED-LAYER TARJOMEH CANDIDATE" in deploy
    assert 'git diff --quiet v10.24.1 "$TAG" -- Dockerfile pyproject.toml' in deploy
    assert "docker builder prune" not in deploy
    assert "docker rm -f 9router" not in deploy
    assert 'docker inspect -f \'{{.Id}}\' 9router' in deploy
    assert "RUNNING_V1025_CONFIRMED" in deploy
    for audit in (reports, companion):
        assert '${1:-LATEST}' in audit
        assert "canonical_event_hash_mismatches" in audit
        assert "canonical_memory_mismatches" in audit
    assert "lifetime_failure_rate" in companion
    assert '"coherent unit" in pipeline_source' not in companion
    assert "restore_source_note_markers(" in companion
    assert "normalize_citation_house_style_text(" in companion

    for name, script in {
        "deploy": deploy,
        "reports": reports,
        "companion": companion,
    }.items():
        blocks = re.findall(
            r"<<'PY'[^\n]*\n(.*?)\nPY(?=\n|$)",
            script,
            flags=re.DOTALL,
        )
        assert blocks, f"no embedded Python found in {name} script"
        for index, block in enumerate(blocks):
            compile(block, f"{name}-embedded-{index}.py", "exec")
