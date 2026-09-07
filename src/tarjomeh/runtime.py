"""Stable runtime capability reporting for deployment and audit tooling."""

from __future__ import annotations

from typing import Any

RUNTIME_RELEASE = "v10.28.0"
RUNTIME_REVISION = 1


def runtime_capabilities() -> dict[str, Any]:
    """Return explicit policy capabilities loaded by this Python runtime.

    Capability IDs are an audit contract, not source-code marker guesses. New
    policies receive new IDs; existing IDs remain stable across refactors.
    Critical deployment audits combine this manifest with behavioral probes.
    """
    return {
        "release": RUNTIME_RELEASE,
        "revision": RUNTIME_REVISION,
        "capabilities": {
            "typed_structure_evidence": True,
            "reviewable_uncertain_structure": True,
            "source_aware_fluency_admission": True,
            "refiner_issue_veto": True,
            "canonical_final_text_audit": True,
            "layer1_evidence_safe_admission": True,
            "genre_aware_style_records": True,
            "evidence_bound_research": True,
            "source_bound_foreign_expressions": True,
            "four_layer_memory": True,
            "offline_model_benchmark": True,
            "durable_worker_lease": True,
            "checkpoint_preview_atomic_publish": True,
        },
        "policy_versions": {
            "structure_evidence": 2,
            "canonical_text": 2,
            "layer1_admission": 4,
            "style_evidence": 2,
            "benchmark_schema": 1,
            "checkpoint_export": 2,
        },
    }


def runtime_behavior_probes() -> dict[str, bool]:
    """Exercise release-critical policies without network or persistent state."""
    from pathlib import Path

    from tarjomeh.chunking.chunker import Chunk
    from tarjomeh.core.config import TarjomehConfig
    from tarjomeh.core.pipeline import (
        _blocking_structure_findings,
        _source_foreign_expression_inventory,
        audit_canonical_document_identity,
    )
    from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
    from tarjomeh.memory.manager import MemoryManager
    from tarjomeh.memory.proper_nouns import automatic_terminology_risk_reasons
    from tarjomeh.quality.model_benchmark import load_benchmark_suite
    from tarjomeh.quality.structure_audit import (
        announced_count_evidence,
        audit_payload,
    )

    manifest = runtime_capabilities()
    three_issues = (
        "\u0627\u06cc\u0646 \u0641\u0635\u0644 \u0628\u0647 \u0633\u0647 "
        "\u0645\u0633\u0626\u0644\u0647 "
        "\u0645\u06cc\u200c\u067e\u0631\u062f\u0627\u0632\u062f."
    )
    canonical_text = "\u0645\u062a\u0646 \u0646\u0647\u0627\u06cc\u06cc."
    legacy_style_sample = (
        "\u0627\u06cc\u0646 \u0645\u062a\u0646 \u0646\u0645\u0648\u0646\u0647 "
        "\u0635\u0631\u0641\u0627 \u0628\u0631\u0627\u06cc \u0628\u0631\u0631\u0633\u06cc "
        "\u0645\u0647\u0627\u062c\u0631\u062a \u0634\u0648\u0627\u0647\u062f "
        "\u0633\u0628\u06a9 "
        "\u0646\u0648\u0634\u062a\u0627\u0631 \u0627\u0633\u062a."
    )
    incomplete_target = (
        "\u0645\u06a9\u0645\u0644\u200c\u0628\u0648\u062f\u0646 "
        "\u0646\u0647\u0627\u062f\u06cc"
    )
    evidence = announced_count_evidence(
        "The account uses eight sources, including chapter 5."
    )
    mismatch = audit_payload(
        "This chapter addresses two issues.",
        three_issues,
    )["findings"]
    canonical = audit_canonical_document_identity(
        TranslatedDocument(paragraphs=[
            TranslatedParagraph(0, "Source", f"  {canonical_text}  ")
        ]),
        [Chunk(index=0, text="Source", chapter_title="", section_title="")],
        {0: canonical_text},
    )
    manager = MemoryManager(TarjomehConfig())
    manager.from_dict({"style_samples": [legacy_style_sample]})
    suite = load_benchmark_suite(
        Path("/__installed__/benchmarks/suites/academic-core.json")
    )
    return {
        "manifest_enabled": bool(
            manifest["release"] == RUNTIME_RELEASE
            and all(manifest["capabilities"].values())
        ),
        "typed_structure_evidence": bool(
            [(item.value, item.semantic_category) for item in evidence]
            == [(8, "source")]
        ),
        "exact_structure_mismatch_blocks": bool(
            _blocking_structure_findings(mismatch)
        ),
        "coordinated_memory_is_quarantined": bool(
            "coordinated_source_target_incomplete"
            in automatic_terminology_risk_reasons(
                "institutional isomorphism or complementarity",
                incomplete_target,
            )
        ),
        "foreign_expression_is_source_bound": bool(
            _source_foreign_expression_inventory(
                "The account adopts a longue dur\u00e9e perspective."
            ) == ["longue dur\u00e9e"]
        ),
        "canonical_identity_is_lexical": bool(canonical["lexically_identical"]),
        "legacy_style_is_fallback": bool(
            manager.style_sample_records
            and manager.style_sample_records[0].get("fallback") is True
            and manager._style_profile_status() == "warming_up"
        ),
        "default_benchmark_is_installed": bool(
            suite["id"] == "academic-core-v1" and len(suite["passages"]) == 4
        ),
    }
