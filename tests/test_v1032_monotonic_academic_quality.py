from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import _best_source_faithful_version, _chunk_style_policy
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.memory.proper_nouns import (
    automatic_terminology_risk_reasons,
    low_authority_mapping_category,
)
from tarjomeh.runtime import runtime_behavior_probes, runtime_capabilities


def _critique(
    *,
    accuracy: float = 9.0,
    terminology: float = 9.0,
    fluency: float = 9.0,
    register: float = 9.0,
    issues: list[dict] | None = None,
) -> SimpleNamespace:
    values = (accuracy, terminology, fluency, register)
    return SimpleNamespace(
        accuracy=accuracy,
        terminology=terminology,
        fluency=fluency,
        register=register,
        average=sum(values) / len(values),
        valid=True,
        issue_details=list(issues or []),
    )


def test_candidate_portfolio_prefers_resolved_objective_persian() -> None:
    baseline_issue = {
        "issue_id": "grammar-1",
        "category": "fluency",
        "severity": "major",
        "confidence": 0.9,
        "source_quote": "can be, and have been, analysed",
        "current_persian_quote": "bad target",
        "suggested_correction": "clear target",
        "rationale": "The matrix predicate is malformed.",
    }
    baseline = _critique(
        accuracy=10,
        terminology=10,
        fluency=8,
        register=9,
        issues=[baseline_issue],
    )
    repaired = _critique(accuracy=9, terminology=9, fluency=9, register=9)

    selected = _best_source_faithful_version(
        [("baseline", baseline), ("repaired", repaired)],
        source="The claim can be, and has been, analysed.",
    )

    assert selected is not None
    assert selected[1] == "repaired"


def test_candidate_portfolio_keeps_source_fidelity_ahead_of_fluency() -> None:
    source_issue = {
        "issue_id": "number-1",
        "category": "number",
        "severity": "major",
        "confidence": 1.0,
        "source_quote": "two issues",
        "current_persian_quote": "three issues",
        "suggested_correction": "two issues",
        "rationale": "The explicit source quantity changed.",
        "source_segment_id": "p1:s1",
    }
    faithful = _critique(fluency=8.0)
    fluent_but_wrong = _critique(fluency=10.0, issues=[source_issue])

    selected = _best_source_faithful_version(
        [("faithful", faithful), ("wrong", fluent_but_wrong)],
        source="It addresses two issues.",
    )

    assert selected is not None
    assert selected[1] == "faithful"


def test_sentence_bound_double_adjectival_suffix_is_not_reusable() -> None:
    target = (
        "\u067e\u0627\u0631\u0627\u062f\u0627\u06cc\u0645\u200c\u0647\u0627\u06cc "
        "\u0633\u06cc\u0627\u0633\u062a\u200c\u06af\u0630\u0627\u0631\u06cc\u200c\u0627\u06cc"
    )

    assert "contextual_productive_suffix" in automatic_terminology_risk_reasons(
        "policy paradigms", target
    )


def test_transliterated_term_is_classified_as_technical_loanword() -> None:
    target = "\u0622\u0633\u0627\u0645\u0628\u0644\u0627\u0698"

    assert low_authority_mapping_category(
        "assemblage", target, "term"
    ) == "technical_loanword"


def test_established_style_profile_excludes_fallback_evidence() -> None:
    config = TarjomehConfig()
    config.memory.style_min_representative_samples = 3
    manager = MemoryManager(config)
    samples = [
        "\u0627\u06cc\u0646 \u062a\u062d\u0644\u06cc\u0644 \u0631\u0627\u0628\u0637\u0647 \u0645\u06cc\u0627\u0646 \u0646\u0647\u0627\u062f\u0647\u0627 \u0631\u0627 \u0628\u0631\u0631\u0633\u06cc \u0645\u06cc\u200c\u06a9\u0646\u062f.",
        "\u0627\u0633\u062a\u062f\u0644\u0627\u0644 \u062f\u0648\u0645 \u0628\u0631 \u067e\u06cc\u0648\u0646\u062f \u0642\u062f\u0631\u062a \u0648 \u0646\u0647\u0627\u062f \u062a\u0623\u06a9\u06cc\u062f \u062f\u0627\u0631\u062f.",
        "\u0646\u0648\u06cc\u0633\u0646\u062f\u0647 \u0633\u067e\u0633 \u067e\u06cc\u0627\u0645\u062f\u0647\u0627\u06cc \u0646\u0638\u0631\u06cc \u0627\u06cc\u0646 \u062f\u06cc\u062f\u06af\u0627\u0647 \u0631\u0627 \u0634\u0631\u062d \u0645\u06cc\u200c\u062f\u0647\u062f.",
        "\u0646\u0645\u0648\u0646\u0647 \u0642\u062f\u06cc\u0645\u06cc \u0648 \u0641\u0642\u0637 \u0628\u0631\u0627\u06cc \u0627\u062f\u0627\u0645\u0647 \u0632\u0645\u06cc\u0646\u0647 \u0646\u06af\u0647\u062f\u0627\u0631\u06cc \u0634\u062f\u0647 \u0627\u0633\u062a.",
    ]
    manager.style_samples = list(samples)
    manager.style_sample_records = [
        {
            "text": sample,
            "text_hash": str(index),
            "paragraph_role": "academic_argument",
            "book_genre": "academic",
            "representative": index < 3,
            "fallback": index == 3,
            "quality_score": 100.0,
        }
        for index, sample in enumerate(samples)
    ]

    profile = manager._render_style_profile()

    assert manager._style_profile_status() == "established"
    assert samples[0] in profile
    assert samples[3] not in profile


def test_unresolved_final_paragraph_cannot_enter_style_memory() -> None:
    manager = MemoryManager(TarjomehConfig())
    sample = (
        "\u0627\u06cc\u0646 \u062a\u062d\u0644\u06cc\u0644 \u0631\u0627\u0628\u0637\u0647 \u0645\u06cc\u0627\u0646 \u0646\u0647\u0627\u062f\u0647\u0627 \u0631\u0627 "
        "\u0628\u0627 \u062f\u0642\u062a \u0628\u0631\u0631\u0633\u06cc \u0645\u06cc\u200c\u06a9\u0646\u062f \u0648 \u067e\u06cc\u0627\u0645\u062f\u0647\u0627\u06cc \u0646\u0638\u0631\u06cc \u0622\u0646 \u0631\u0627 "
        "\u062f\u0631 \u0628\u0633\u062a\u0631 \u062a\u0627\u0631\u06cc\u062e\u06cc \u0645\u0648\u0631\u062f \u062a\u0648\u0636\u06cc\u062d \u0642\u0631\u0627\u0631 \u0645\u06cc\u200c\u062f\u0647\u062f."
    )

    report = manager._update_style_profile(
        sample,
        source_paragraph_indices=[4],
        paragraph_role="academic_argument",
        book_genre="academic",
        unresolved_issue_paragraphs={"grammar-4": 4},
    )

    assert report["accepted"] is False
    assert manager.style_samples == []
    assert report["rejections"][0]["unresolved_issue_ids"] == ["grammar-4"]


def test_malformed_final_score_withholds_style_without_crashing() -> None:
    class _DB:
        @staticmethod
        def get_chunk_events(_job_id: str, _chunk_index: int) -> list[dict]:
            return [
                {"event_type": "chunk_started", "payload": {}},
                {
                    "event_type": "critique_completed",
                    "payload": {
                        "valid": True,
                        "scores": {
                            "accuracy": "not-a-score",
                            "fluency": 9,
                            "terminology": 9,
                            "register": 9,
                            "average": 9,
                        },
                        "issue_details": [],
                    },
                },
            ]

    policy = _chunk_style_policy(_DB(), "job", 0)

    assert policy["approved"] is False
    assert policy["final_scores"]["accuracy"] == 0.0

    manager = MemoryManager(TarjomehConfig())
    manager.style_samples = ["\u0645\u062a\u0646 \u0646\u0645\u0648\u0646\u0647 \u0628\u0631\u0627\u06cc \u0633\u0628\u06a9 \u062f\u0627\u0646\u0634\u06af\u0627\u0647\u06cc."]
    manager.style_sample_records = [{
        "text": manager.style_samples[0],
        "text_hash": "legacy",
        "representative": True,
        "quality_score": "not-a-score",
    }]
    assert manager._style_profile_status() == "warming_up"


def test_v1033_runtime_and_release_scripts_are_complete() -> None:
    manifest = runtime_capabilities()
    probes = runtime_behavior_probes()

    assert manifest["release"] == "v10.33.0"
    assert manifest["capabilities"]["post_rollback_final_evidence"] is True
    assert manifest["capabilities"]["objective_candidate_ranking"] is True
    assert manifest["capabilities"]["contextual_morphology_quarantine"] is True
    assert probes["contextual_morphology_is_quarantined"] is True
    assert probes["transliterated_terms_are_source_anchorable"] is True
    assert probes["candidate_selection_matches_canonical"] is True
    assert probes["mixed_role_readability_is_body_only"] is True
    assert probes["partial_compound_memory_is_quarantined"] is True
    assert probes["moved_note_marker_is_source_aligned"] is True

    for relative in (
        "scripts/audit_tarjomeh_v1033_reports.sh",
        "scripts/audit_tarjomeh_v1033_companion.sh",
    ):
        source = Path(relative).read_text(encoding="utf-8")
        blocks = re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY(?:\r?\n|$)", source, re.S)
        assert blocks, relative
        for block in blocks:
            ast.parse(block, filename=relative)
        assert "final_candidate_selection" in source
        assert "missing_final_candidate_selection" in source

    deployment = Path("scripts/deploy_tarjomeh_v1033.sh").read_text(
        encoding="utf-8"
    )
    assert 'TAG="v10.33.0"' in deployment
    assert "RUNNING_V1033_CONFIRMED" in deployment
    assert "VERIFY 9ROUTER UNCHANGED" in deployment
    assert "NINE_MOUNTS" in deployment
    for unsafe in (
        "docker system prune",
        "docker image prune",
        "docker builder prune",
        "docker rm -f 9router",
        'docker image rm "$NINE_IMAGE"',
    ):
        assert unsafe not in deployment
