"""Stable runtime capability reporting for deployment and audit tooling."""

from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any

RUNTIME_RELEASE = "v10.37.0"
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
            "durable_chapter_checkpoint_recovery": True,
            "canonical_paragraph_identity": True,
            "reconstructed_identity_review_only": True,
            "verified_source_coverage": True,
            "final_identifier_admission": True,
            "paragraph_scoped_refiner_salvage": True,
            "canonical_export_is_lexically_pure": True,
            "final_canonical_admission_event": True,
            "resumable_split_recovery_segments": True,
            "paragraph_scoped_language_repair_roles": True,
            "conservative_summary_stutter_rejection": True,
            "post_rollback_final_evidence": True,
            "objective_candidate_ranking": True,
            "contextual_morphology_quarantine": True,
            "representative_only_established_style": True,
            "candidate_bound_final_quality": True,
            "atomic_candidate_selection": True,
            "paragraph_scoped_readability": True,
            "lexical_scope_memory_quarantine": True,
            "source_aligned_note_marker_relocation": True,
            "lexically_safe_critique_rebind": True,
            "identity_before_final_quality": True,
            "atomic_final_quality_checkpoint": True,
            "resumable_source_obligation_repair": True,
            "conditional_exact_final_quality_repair": True,
            "equivalent_issue_deduplication": True,
            "exact_local_correction_alignment": True,
            "authoritative_style_dimension_floor": True,
            "source_cardinality_note_repair": True,
            "evidence_bound_research_prompt": True,
            "distinct_note_marker_occurrences": True,
            "unpaired_object_marker_dash_review": True,
            "paragraph_scoped_objective_style_review": True,
            "source_validated_language_repair": True,
            "persian_argument_announcement_coverage": True,
            "bounded_source_obligation_fresh_generation": True,
        },
        "policy_versions": {
            "structure_evidence": 2,
            "canonical_text": 3,
            "layer1_admission": 7,
            "style_evidence": 6,
            "benchmark_schema": 1,
            "checkpoint_export": 3,
            "paragraph_identity": 1,
            "critic_source_coverage": 1,
            "final_identifier_admission": 1,
            "recovery_segments": 1,
            "summary_admission": 2,
            "final_candidate_selection": 2,
            "final_quality_authority": 1,
            "note_marker_recovery": 4,
            "critique_canonical_rebind": 1,
            "final_quality_checkpoint": 3,
            "source_obligation_recovery": 2,
            "local_refiner_salvage": 2,
            "research_prompt_admission": 2,
            "targeted_language_repair": 2,
        },
    }


def runtime_behavior_probes() -> dict[str, bool]:
    """Exercise release-critical policies without network or persistent state."""
    from pathlib import Path

    from tarjomeh.chunking.chunker import Chunk
    from tarjomeh.core.config import TarjomehConfig
    from tarjomeh.core.pipeline import (
        audit_translation_language,
        _source_obligation_resume_action,
        _blocking_structure_findings,
        _canonical_chunk_paragraph_identity,
        _critique_survives_canonicalization,
        _chunk_needs_review,
        _final_canonical_admission_payload,
        _final_candidate_selection_payload,
        _paragraph_structural_roles,
        _readability_review_text,
        _research_context_for_memory,
        _record_reconstructed_paragraph_identity_review,
        _role_aware_recovery_groups,
        _source_foreign_expression_inventory,
        _target_units_from_identity,
        audit_canonical_document_identity,
    )
    from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
    from tarjomeh.memory.manager import (
        MemoryManager,
        _style_record_is_authoritative,
    )
    from tarjomeh.memory.proper_nouns import (
        ProperNouns,
        automatic_terminology_risk_reasons,
        low_authority_mapping_category,
    )
    from tarjomeh.quality.critique import TranslationCritique
    from tarjomeh.quality.integrity import (
        available_note_markers,
        extract_note_markers,
        restore_source_identifiers,
        restore_source_note_markers,
    )
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
    claim_source = (
        "The argument makes three claims. First, one; second, two; third, three."
    )
    claim_target = (
        "این استدلال سه مدعای اصلی دارد. نخست، یک؛ دوم، دو؛ سوم، سه."
    )
    wrong_claim_target = claim_target.replace("سه مدعای", "دو مدعای")
    wrong_claim_findings = audit_payload(
        claim_source, wrong_claim_target
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
    contextual_target = (
        "\u067e\u0627\u0631\u0627\u062f\u0627\u06cc\u0645\u200c\u0647\u0627\u06cc "
        "\u0633\u06cc\u0627\u0633\u062a\u200c\u06af\u0630\u0627\u0631\u06cc\u200c\u0627\u06cc"
    )
    suite = load_benchmark_suite(
        Path("/__installed__/benchmarks/suites/academic-core.json")
    )
    identity_chunk = Chunk(
        index=0,
        text="First.\n\nSecond.",
        chapter_title="",
        section_title="",
        metadata={
            "paragraph_indices": [0, 1],
            "paragraph_protocol_version": 1,
        },
    )
    canonical_units, paragraph_identity = _canonical_chunk_paragraph_identity(
        identity_chunk,
        "اول. دوم.",
    )
    role_chunk = Chunk(
        index=1,
        text="Prose.\n\nRow one\n\nRow two",
        chapter_title="",
        section_title="",
        metadata={"structural_roles": ["body", "table", "table"]},
    )
    role_units = role_chunk.text.split("\n\n")
    role_values = _paragraph_structural_roles(role_chunk, len(role_units))
    role_groups = _role_aware_recovery_groups(role_units, role_values)
    admission = _final_canonical_admission_payload(
        canonical_units, paragraph_identity
    )
    selection = _final_candidate_selection_payload(
        canonical_units, paragraph_identity
    )
    readability_text = _readability_review_text(
        role_chunk,
        "\u0645\u062a\u0646 \u0627\u0633\u062a\u062f\u0644\u0627\u0644\u06cc.\n\n"
        "\u0631\u062f\u06cc\u0641 \u06cc\u06a9\n\n\u0631\u062f\u06cc\u0641 \u062f\u0648",
    )
    restored_note, note_report = restore_source_note_markers(
        "First sentence.2 Second sentence.",
        (
            "\u062c\u0645\u0644\u0647 \u0646\u062e\u0633\u062a. "
            "\u062c\u0645\u0644\u0647 \u062f\u0648\u0645.2"
        ),
    )
    deduplicated_note, duplicate_note_report = restore_source_note_markers(
        "Source sentence.1",
        "\u062c\u0645\u0644\u0647\u0654 \u062a\u0631\u062c\u0645\u0647.\u00b9 \u06f1",
    )
    legacy_correction = ProperNouns()
    legacy_correction.deserialize({
        "nouns": {"Clays Ltd": "\u0628\u0631\u06cc\u062a\u0627\u0646\u06cc\u0627"},
        "categories": {"Clays Ltd": "organization"},
        "provenance": {
            "Clays Ltd": {
                "origin": "accepted_correction",
                "authority": 80,
                "context_independent": True,
            }
        },
    })
    weak_style_record = {
        "text": (
            "\u0627\u06cc\u0646 \u062a\u062d\u0644\u06cc\u0644 \u0631\u0627\u0628\u0637\u0647\u0654 \u0645\u06cc\u0627\u0646 "
            "\u0646\u0647\u0627\u062f\u0647\u0627 \u0631\u0627 \u0628\u0631\u0631\u0633\u06cc \u0645\u06cc\u200c\u06a9\u0646\u062f."
        ),
        "representative": True,
        "quality_score": 95.0,
        "final_scores": {
            "accuracy": 9.2,
            "fluency": 8.9,
            "terminology": 9.1,
            "register": 9.3,
        },
    }
    research_context = _research_context_for_memory({
        "status": "completed",
        "terms": [
            {
                "source": "safe term",
                "target": "\u0627\u0635\u0637\u0644\u0627\u062d \u0627\u0645\u0646",
                "status": "suggested",
                "identity_supported": True,
                "term_supported": True,
            },
            {
                "source": "weak term",
                "target": "\u062d\u062f\u0633 \u0636\u0639\u06cc\u0641",
                "status": "suggested",
                "identity_supported": True,
                "term_supported": False,
            },
        ],
    })
    class _ProbeDB:
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def log_chunk_event(
            self,
            _job_id: str,
            _chunk_index: int,
            event_type: str,
            payload: dict[str, Any],
        ) -> None:
            self.events.append({"event_type": event_type, "payload": payload})

        def get_chunk_events(
            self, _job_id: str, _chunk_index: int
        ) -> list[dict[str, Any]]:
            return self.events

    probe_db = _ProbeDB()
    probe_db.log_chunk_event("probe", 0, "chunk_started", {})
    _record_reconstructed_paragraph_identity_review(
        probe_db, "probe", 0, paragraph_identity
    )
    coverage = TranslationCritique._parse_response(
        '{"scores":{"accuracy":9,"fluency":9,"terminology":9,'
        '"register":9},"source_coverage":{"checked_source_segment_ids":'
        '["p1:s1","p1:s2"],"uncovered_source_segment_ids":[],'
        '"complete":true},"issues":[]}',
        "First. Second.",
        "اول. دوم.",
        require_coverage=True,
    )
    restored_identifier, _identifier_report = restore_source_identifiers(
        "Write to example.org, AB1 2CD.",
        "به example. org، AB1 ۲CD بنویسید.",
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
        "persian_argument_announcement_is_typed": bool(
            not audit_payload(claim_source, claim_target)["findings"]
            and _blocking_structure_findings(
                wrong_claim_findings
            )
        ),
        "cached_obligation_is_rechecked_before_regeneration": bool(
            _source_obligation_resume_action(
                claim_source, claim_target, {"failure_count": 4}
            ) == "recheck"
            and _source_obligation_resume_action(
                claim_source, wrong_claim_target,
                {"same_candidate_failures": 2, "findings": wrong_claim_findings},
            ) == "regenerate"
            and _source_obligation_resume_action(
                claim_source, wrong_claim_target,
                {"same_candidate_failures": 3, "fresh_generation_count": 1,
                 "findings": wrong_claim_findings},
            ) == "stop"
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
        "canonical_paragraph_identity_round_trips": bool(
            len(_target_units_from_identity(
                canonical_units, paragraph_identity
            ) or []) == 2
        ),
        "reconstructed_identity_is_review_only": bool(
            paragraph_identity.get("reconstructed")
            and _chunk_needs_review(probe_db, "probe", 0)
        ),
        "critic_source_coverage_is_verified": bool(
            coverage.valid and coverage.coverage_complete
        ),
        "source_identifiers_are_restored": bool(
            "example.org" in restored_identifier
            and "AB1 2CD" in restored_identifier
        ),
        "canonical_admission_matches_final_text": bool(
            admission["canonical_target_hash"]
            == hashlib.sha256(canonical_units.encode("utf-8")).hexdigest()
        ),
        "mixed_roles_are_paragraph_scoped": bool(
            role_groups == [
                (0, ["Prose."], "body"),
                (1, ["Row one", "Row two"], "table"),
            ]
        ),
        "contextual_morphology_is_quarantined": bool(
            "contextual_productive_suffix"
            in automatic_terminology_risk_reasons(
                "policy paradigms", contextual_target
            )
        ),
        "transliterated_terms_are_source_anchorable": bool(
            low_authority_mapping_category(
                "assemblage", "\u0622\u0633\u0627\u0645\u0628\u0644\u0627\u0698", "term"
            ) == "technical_loanword"
        ),
        "candidate_selection_matches_canonical": bool(
            selection["policy_version"] == 2
            and selection["canonical_target_hash"]
            == admission["canonical_target_hash"]
        ),
        "mixed_role_readability_is_body_only": bool(
            readability_text
            == "\u0645\u062a\u0646 \u0627\u0633\u062a\u062f\u0644\u0627\u0644\u06cc."
        ),
        "partial_compound_memory_is_quarantined": bool(
            "target_omits_source_lexical_member"
            in automatic_terminology_risk_reasons(
                "state apparatus", "\u0622\u067e\u0627\u0631\u0627\u062a\u0648\u0633"
            )
        ),
        "moved_note_marker_is_source_aligned": bool(
            restored_note
            == (
                "\u062c\u0645\u0644\u0647 \u0646\u062e\u0633\u062a.\u00b2 "
                "\u062c\u0645\u0644\u0647 \u062f\u0648\u0645."
            )
            and note_report.get("repairs", [{}])[0].get("type")
            == "relocated_aligned_sentence_terminal_note_marker"
        ),
        "canonical_typography_preserves_critique": bool(
            _critique_survives_canonicalization(
                "در سال 1973، مقدار 40% بود.",
                "در سال ۱۹۷۳، مقدار ۴۰ ٪ بود.",
            )
            and not _critique_survives_canonicalization(
                "The first target proposition.",
                "The second target proposition.",
            )
        ),
        "legacy_corrections_without_alignment_are_deferred": bool(
            legacy_correction.is_context_deferred("Clays Ltd")
        ),
        "style_requires_all_dimensions_at_nine": bool(
            not _style_record_is_authoritative(weak_style_record)
        ),
        "duplicate_note_marker_is_removed_by_source_cardinality": bool(
            deduplicated_note.endswith("\u00b9")
            and not duplicate_note_report.get("surplus")
            and any(
                item.get("type")
                == "unique_plain_duplicate_note_marker_removed"
                for item in duplicate_note_report.get("repairs", [])
            )
        ),
        "research_prompt_requires_term_evidence": bool(
            "safe term" in research_context and "weak term" not in research_context
        ),
        "spaced_note_is_one_occurrence": bool(
            available_note_markers(
                "(Cerny 2010.) \u06f4",
                Counter(extract_note_markers("(Cerny 2010.)4")),
            )["4"] == 1
        ),
        "unpaired_object_marker_dash_is_reviewed": bool(
            audit_translation_language(
                "They make history - their own - in context.",
                "\u062a\u0627\u0631\u06cc\u062e\u2014\u0631\u0627 \u0645\u06cc\u200c\u0633\u0627\u0632\u0646\u062f.",
            )["unbalanced_explanatory_dash_count"] == 1
        ),
    }
