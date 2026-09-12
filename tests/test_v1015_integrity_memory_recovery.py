from __future__ import annotations

from collections import Counter
from unittest.mock import MagicMock

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import TruncatedCompletionError
from tarjomeh.core.pipeline import (
    TranslationPipeline,
    _chunk_memory_admission,
    _chunk_style_policy,
    _critique_requires_refinement,
    _promote_objective_readability_issues,
    _readability_advisory_decision_summary,
    _table_recovery_groups,
)
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.persian.typography import PersianTypographer
from tarjomeh.quality.critique import CritiqueResult
from tarjomeh.quality.integrity import (
    PostEditIntegrityGate,
    extract_note_markers,
    extract_numbers,
    parenthesis_artifacts,
    repair_source_grounded_language_artifacts,
)


class _StyleDB:
    def get_chunk_events(self, _job_id: str, _chunk_index: int):
        return [
            {"event_type": "chunk_started", "payload": {}},
            {
                "event_type": "critique_completed",
                "payload": {
                    "valid": True,
                    "blocking_issue_count": 0,
                    "scores": {
                        "accuracy": 9,
                        "fluency": 9,
                        "terminology": 9,
                        "register": 9,
                        "average": 9,
                    },
                    "issue_details": [{
                        "issue_id": "local-fluency",
                        "severity": "minor",
                        "category": "fluency",
                        "source_segment_id": "p1:s1",
                    }],
                    "high_confidence_minor_refinement_issue_ids": [
                        "local-fluency"
                    ],
                },
            },
        ]


class _MemoryDB:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.events = events

    def get_chunk_events(self, _job_id: str, _chunk_index: int):
        return self.events


def _critique_event(
    *,
    average: float = 8.8,
    accuracy: float = 9.0,
    fluency: float = 9.0,
    terminology: float = 9.0,
    register: float = 9.0,
) -> dict[str, object]:
    return {
        "event_type": "critique_completed",
        "payload": {
            "valid": True,
            "blocking_issue_count": 0,
            "scores": {
                "average": average,
                "accuracy": accuracy,
                "fluency": fluency,
                "terminology": terminology,
                "register": register,
            },
        },
    }


def test_pdf_flattened_superscript_is_a_note_not_a_prose_number() -> None:
    source = "Jean-Jacques Rousseau)1 - reflections continued."
    target = "تأملات روسو (Jean-Jacques Rousseau) ۱؛ ادامه یافت."
    assert extract_numbers(source) == Counter()
    assert extract_note_markers(source) == ["1"]
    assert PostEditIntegrityGate().evaluate(source, target).accepted


def test_plain_spaced_prose_number_is_not_inferred_to_be_a_source_note() -> None:
    text = "سه دسته، ۳ حالت متفاوت را در بر می‌گیرند."
    assert extract_note_markers(text) == []
    assert extract_numbers(text) == Counter({"3": 1})


def test_only_orphaned_close_before_source_note_is_repaired() -> None:
    source = "innovation (think of Rousseau)1 - reflections followed."
    broken = "این نوآوری؛ می‌توان به روسو اندیشید) ۱؛ سپس تأملات ادامه یافت."
    balanced = "این نوآوری (برای نمونه روسو) ۱؛ سپس تأملات ادامه یافت."
    repaired, report = repair_source_grounded_language_artifacts(source, broken)
    unchanged, unchanged_report = repair_source_grounded_language_artifacts(
        source, balanced
    )
    assert not parenthesis_artifacts(source, repaired)
    assert "اندیشید ۱" in repaired
    assert any(
        item["type"] == "orphaned_note_parenthesis"
        for item in report["repairs"]
    )
    assert unchanged == balanced
    assert not any(
        item["type"] == "orphaned_note_parenthesis"
        for item in unchanged_report["repairs"]
    )


def test_table_recovery_groups_rows_but_leaves_prose_recovery_unchanged() -> None:
    rows = [f"row {index}" for index in range(82)]
    groups = _table_recovery_groups(rows, table_like=True, max_rows=12)
    assert [len(group) for group in groups] == [12, 12, 12, 12, 12, 12, 10]
    assert [row for group in groups for row in group] == rows
    assert _table_recovery_groups(rows[:3], table_like=False) == [
        ["row 0"], ["row 1"], ["row 2"]
    ]


def test_table_recovery_uses_one_marked_group_with_existing_quality_profile() -> None:
    config = TarjomehConfig()
    config.translation.enable_web_context = False
    config.translation.enable_back_translation = False
    config.translation.enable_critique = False
    config.translation.enable_integrity_gate = False
    config.glossary.enable_compliance_check = False
    config.glossary.enable_auto_extraction = False

    pipeline = object.__new__(TranslationPipeline)
    pipeline.config = config
    pipeline.db = MagicMock()
    pipeline.db.get_job.return_value = {"status": "running"}
    pipeline.llm_client = MagicMock()
    grouped = "\n\n".join(
        f"[[P{index:04d}]]\nردیف {index}"
        for index in range(1, 5)
    )
    pipeline.llm_client.complete.side_effect = [
        TruncatedCompletionError("whole table exhausted output budget"),
        grouped,
    ]

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

    source_rows = ["item one", "item two", "item three", "item four"]
    chunk = Chunk(
        index=0,
        text="\n\n".join(source_rows),
        chapter_title="Table",
        section_title="",
        metadata={
            "paragraph_protocol_version": 1,
            "paragraph_indices": [0, 1, 2, 3],
            "structural_roles": ["table"] * 4,
        },
    )
    result = pipeline._translate_single_chunk(
        idx=0,
        chunk=chunk,
        memory_manager=memory_manager,
        web_searcher=MagicMock(),
        glossary_manager=glossary,
        compliance_checker=MagicMock(),
        critique_tool=MagicMock(),
        refiner_tool=MagicMock(),
        back_translator=MagicMock(),
        translations={},
        job_id="job-table-group",
    )

    typographer = PersianTypographer(config.to_dict().get("persian"))
    assert result.split("\n\n") == [
        typographer.process(f"ردیف {index}") for index in range(1, 5)
    ]
    assert [
        call.kwargs.get("_operation")
        for call in pipeline.llm_client.complete.call_args_list
    ] == ["translation", "translation_split_recovery"]
    events = [
        call.args[2] for call in pipeline.db.log_chunk_event.call_args_list
    ]
    assert events.count("translation_table_group_recovery") == 1
    assert "translation_recovery_part" not in events


def test_clean_paragraph_can_seed_style_without_becoming_durable_memory() -> None:
    policy = _chunk_style_policy(_StyleDB(), "job", 0)
    assert {key: policy[key] for key in (
        "approved", "excluded_paragraphs", "reason"
    )} == {
        "approved": True,
        "excluded_paragraphs": [0],
        "reason": "clean_final_critique",
    }
    assert policy["final_scores"] == {
        "accuracy": 9.0,
        "fluency": 9.0,
        "terminology": 9.0,
        "register": 9.0,
    }
    assert policy["unresolved_issue_paragraphs"] == {}
    manager = MemoryManager(TarjomehConfig())
    chunk = Chunk(
        index=0,
        text=(
            "The first paragraph has a local issue that must not teach style.\n\n"
            "The second paragraph is complete, accurate academic prose with enough "
            "material to provide a clean register sample for later chapters. It "
            "contains enough source material for the structural style gate while "
            "remaining separate from durable terminology authority."
        ),
        chapter_title="Chapter",
        section_title="",
        metadata={
            "style_eligible": True,
            "style_body_paragraphs": [0, 1],
            "structural_roles": ["body", "body"],
        },
    )
    first = "این بند مسئله‌ای محلی دارد و نباید نمونهٔ سبکی باشد."
    second = (
        "این بند نثر دانشگاهیِ روشن، دقیق و پیوسته‌ای دارد و می‌تواند بدون "
        "تحمیل اصطلاحات خود بر فصل‌های بعد، صرفاً شاهدی برای لحن کتاب باشد."
    )
    result = manager.update_after_translation(
        chunk,
        f"{first}\n\n{second}",
        quality_approved=False,
        style_approved=True,
        long_term_reliable=False,
        style_excluded_paragraphs=[0],
    )
    assert result["long_term_reliable"] is False
    assert result["style_sample_added"] is True
    assert manager.style_samples == [second]


def test_major_objective_readability_is_routed_without_source_authority() -> None:
    translation = "سلطه‌ای است که در این نظام حک شده‌اند."
    critique = CritiqueResult(
        accuracy=9.5,
        fluency=9.0,
        terminology=9.5,
        register=9.0,
        average=9.25,
        issues=[],
        issue_details=[],
    )
    promoted = _promote_objective_readability_issues(
        critique,
        [{
            "severity": "major",
            "current_persian_quote": "سلطه‌ای است که در این نظام حک شده‌اند",
            "suggested_correction": "سلطه‌ای است که در این نظام حک شده است",
            "rationale": "The singular subject has a plural predicate agreement.",
        }],
        translation,
    )

    assert len(promoted) == 1
    assert promoted[0]["category"] == "readability"
    assert "source_quote" not in promoted[0]
    assert promoted[0]["readability_advisory"]["authority"] == (
        "target_only_advisory"
    )
    assert critique.issues
    assert _critique_requires_refinement(critique, 9.0)


def test_minor_or_ungrounded_readability_is_not_promoted() -> None:
    critique = CritiqueResult(
        accuracy=9,
        fluency=9,
        terminology=9,
        register=9,
        average=9,
        issues=[],
        issue_details=[],
    )
    promoted = _promote_objective_readability_issues(
        critique,
        [
            {
                "severity": "minor",
                "current_persian_quote": "عبارت موجود",
                "suggested_correction": "عبارت جایگزین",
                "rationale": "A minor punctuation preference.",
            },
            {
                "severity": "major",
                "current_persian_quote": "عبارت غایب",
                "suggested_correction": "عبارت جایگزین",
                "rationale": "The sentence has broken grammar.",
            },
        ],
        "عبارت موجود در این بند آمده است.",
    )

    assert promoted == []
    assert critique.issues == []
    assert critique.issue_details == []


def test_readability_decisions_distinguish_refiner_veto_from_uncommitted_edit() -> None:
    details = [{
        "issue_id": "readability-a",
        "category": "readability",
        "readability_advisory": {"authority": "target_only_advisory"},
    }, {
        "issue_id": "readability-b",
        "category": "readability",
        "readability_advisory": {"authority": "target_only_advisory"},
    }]
    summary = _readability_advisory_decision_summary(
        details,
        [{
            "issue_id": "readability-a",
            "decision": "rejected",
            "rationale": "The suggestion changes source scope.",
        }, {
            "issue_id": "readability-b",
            "decision": "accepted",
            "commit_status": "rejected_integrity",
            "rationale": "Grammar repair proposed.",
        }],
    )

    assert summary["resolved_count"] == 1
    assert summary["unresolved_count"] == 1
    assert summary["decisions"][0]["resolution"] == (
        "rejected_by_source_aware_refiner"
    )


def test_only_aspirational_memory_reasons_are_advisory() -> None:
    events = [
        {"event_type": "chunk_started", "payload": {}},
        _critique_event(),
        {"event_type": "mqm_minor_only_deferred", "payload": {}},
    ]
    policy = _chunk_memory_admission(_MemoryDB(events), "job", 0, 9.0)

    assert policy["long_term_reliable"] is True
    assert policy["short_term_trust"] == "trusted"
    assert policy["disqualifying_reliability_reasons"] == []
    assert set(policy["advisory_reliability_reasons"]) == {
        "final_critique_below_configured_threshold",
        "deferred_mqm_advice",
    }


def test_low_semantic_or_persian_prose_dimension_stays_untrusted() -> None:
    semantic = _chunk_memory_admission(
        _MemoryDB([
            {"event_type": "chunk_started", "payload": {}},
            _critique_event(accuracy=7.9),
        ]),
        "job",
        0,
        9.0,
    )
    prose = _chunk_memory_admission(
        _MemoryDB([
            {"event_type": "chunk_started", "payload": {}},
            _critique_event(fluency=7.9),
        ]),
        "job",
        0,
        9.0,
    )

    assert semantic["long_term_reliable"] is False
    assert "semantic_dimension_below_memory_floor" in (
        semantic["disqualifying_reliability_reasons"]
    )
    assert prose["long_term_reliable"] is False
    assert "persian_prose_dimension_below_memory_floor" in (
        prose["disqualifying_reliability_reasons"]
    )


def test_false_mi_repair_uses_derived_nonverb_prefix_but_preserves_verbs() -> None:
    typographer = PersianTypographer()
    normalizer = MagicMock()
    normalizer.words = {
        "میانجی": (67073, ("N", "AJ")),
        "میانجی‌گری": (103, ("N",)),
    }
    typographer._get_hazm_normalizer = MagicMock(return_value=normalizer)
    text = "می‌انجی‌گری‌شده، می‌رود و نمی‌تواند بماند."

    repaired = typographer._repair_false_mi_splits(text)

    assert "میانجی‌گری‌شده" in repaired
    assert "می‌رود" in repaired
    assert "نمی‌تواند" in repaired
