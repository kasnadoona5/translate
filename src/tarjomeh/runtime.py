"""Stable runtime capability reporting for deployment and audit tooling."""

from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any

RUNTIME_RELEASE = "v10.40.3"
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
            "uncertain_post_edit_issue_blocks": True,
            "unchanged_grounded_issue_is_review_only": True,
            "partial_readability_evidence_survives_invalid_sibling": True,
            "source_scoped_optional_plural_spacing": True,
            "focused_attachment_trial_off_by_default": True,
            "book_scoped_term_approval": True,
            "source_family_candidates_review_only": True,
            "final_optional_plural_admission": True,
            "contextual_person_name_quarantine": True,
            "orphan_object_marker_dash_repair": True,
            "opt_in_book_term_review": True,
            "paragraph_scoped_approved_terms": True,
            "source_only_body_term_inventory": True,
            "source_confirmed_note_superscripts": True,
            "source_proven_surface_repair": True,
            "context_anchored_note_markers": True,
            "aligned_italic_runs": True,
            "paratext_excluded_term_sampling": True,
            "role_independent_zwnj_repair": True,
            "idempotent_inline_original_wrappers": True,
            "stranded_mark_relocation": True,
            "source_proven_optional_prefix_spacing": True,
            "sentence_scoped_orphan_dash_repair": True,
            "us_postal_code_identifier": True,
            "determiner_phrase_memory_quarantine": True,
            "name_shaped_entity_guard": True,
            "acknowledgement_style_exclusion": True,
            "advisory_critic_terminology_label": True,
            "complete_style_pairs": True,
            "per_paragraph_precanonical_affix_repair": True,
            "render_identity_audit": True,
            "reviewed_book_term_occurrences": True,
            "job_scoped_book_term_add": True,
            "imprint_name_grounding": True,
            "malformed_persian_option_screening": True,
            "prompt_duplication_measurement": True,
        },
        "policy_versions": {
            "structure_evidence": 2,
            "canonical_text": 3,
            "layer1_admission": 9,
            "style_evidence": 8,
            "benchmark_schema": 1,
            "checkpoint_export": 3,
            "paragraph_identity": 1,
            "critic_source_coverage": 1,
            "final_identifier_admission": 1,
            "recovery_segments": 1,
            "summary_admission": 2,
            "final_candidate_selection": 2,
            "final_quality_authority": 1,
            "note_marker_recovery": 5,
            "critique_canonical_rebind": 1,
            "final_quality_checkpoint": 3,
            "source_obligation_recovery": 2,
            "local_refiner_salvage": 2,
            "research_prompt_admission": 2,
            "targeted_language_repair": 2,
            "post_edit_issue_attribution": 1,
            "attachment_trial": 1,
            "book_term_scope": 3,
            "final_language_admission": 5,
            "book_term_review": 2,
            "source_term_inventory": 3,
            "render_identity": 1,
        },
    }


def runtime_behavior_probes() -> dict[str, bool]:
    """Exercise release-critical policies without network or persistent state."""
    from pathlib import Path

    from tarjomeh.chunking.chunker import Chunk
    from tarjomeh.core.config import TarjomehConfig
    from tarjomeh.glossary.book_review import resolve_reviewed_book_terms
    from tarjomeh.core.pipeline import (
        audit_translation_language,
        _language_quality_strictly_improves,
        _source_obligation_resume_action,
        _candidate_regression_details,
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
    from tarjomeh.quality.critique import CritiqueResult, TranslationCritique
    from tarjomeh.quality.integrity import repair_source_grounded_language_artifacts
    from tarjomeh.memory.manager import (
        MemoryManager,
        _style_record_is_authoritative,
    )
    from tarjomeh.memory.proper_nouns import (
        ProperNouns,
        is_bounded_person_name_target,
        automatic_terminology_risk_reasons,
        low_authority_mapping_category,
    )
    from tarjomeh.quality.integrity import (
        available_note_markers,
        extract_note_markers,
        repair_proven_surface_artifacts,
        restore_source_identifiers,
        restore_source_note_markers,
    )
    from tarjomeh.quality.model_benchmark import load_benchmark_suite
    from tarjomeh.quality.structure_audit import (
        announced_count_evidence,
        audit_payload,
    )

    manifest = runtime_capabilities()
    probe_source = "The polity matters."
    probe_decision = [{
        "status": "approved", "source": "polity", "target": "ساختار سیاسی",
        "scope_mode": "all_body", "chapter_positions": [1],
    }]
    body_matches, _ = resolve_reviewed_book_terms(
        probe_source, {"structural_roles": ["body"]}, 1, probe_decision,
    )
    heading_matches, heading_review = resolve_reviewed_book_terms(
        probe_source, {"structural_roles": ["heading"]}, 1, probe_decision,
    )
    surface_repaired, surface_changes = repair_proven_surface_artifacts(
        "The institutions and society matter.",
        "نهادها و و جامعه مهم‌اند.",
    )
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
    scope_source = "They act.\n\nOnly one institution has this name."
    scope_previous = "\u0622\u0646\u200c\u0647\u0627 \u2014 \u0645\u06cc\u200c\u06a9\u0646\u0646\u062f.\n\n\u0646\u0647\u0627\u062f \u0646\u0627\u0645 \u062f\u0627\u0631\u062f."
    scope_candidate = scope_previous.replace(" \u2014 ", " ")
    scope_issue = {
        "issue_id": "scope", "category": "accuracy", "severity": "major",
        "confidence": 0.8, "source_segment_id": "p2:s1",
        "source_quote": "Only one institution",
        "current_persian_quote": "\u0646\u0647\u0627\u062f \u0646\u0627\u0645",
    }
    old_scope: list[dict[str, Any]] = []
    old_regression = _candidate_regression_details(
        CritiqueResult(issue_details=[scope_issue]), CritiqueResult(), [],
        source_text=scope_source, previous_text=scope_previous,
        candidate_text=scope_candidate, newly_observed_unchanged=old_scope,
    )
    uncertain_regression = _candidate_regression_details(
        CritiqueResult(issue_details=[{**scope_issue, "source_segment_id": ""}]),
        CritiqueResult(), [], source_text=scope_source,
        previous_text=scope_previous, candidate_text=scope_candidate,
    )
    partial_readability = TranslationCritique._parse_readability_response(
        '{"issues":[{"severity":"minor","current_persian_quote":'
        '"\u0646\u0647\u0627\u062f","suggested_correction":"\u0646\u0647\u0627\u062f\u06cc",'
        '"rationale":"Predicate attachment."},{"severity":"minor",'
        '"current_persian_quote":"missing","suggested_correction":"x",'
        '"rationale":"Invalid sibling."}]}',
        "\u0646\u0647\u0627\u062f \u0646\u0627\u0645 \u062f\u0627\u0631\u062f.",
    )
    optional_repaired, _ = repair_source_grounded_language_artifacts(
        "The discourse(s) matter.",
        "\u06af\u0641\u062a\u0645\u0627\u0646 (\u0647\u0627) \u0645\u0647\u0645\u200c\u0627\u0646\u062f.",
    )
    optional_unrelated, _ = repair_source_grounded_language_artifacts(
        "The discourse(s) matter.\nOther institutions.",
        "\u06af\u0641\u062a\u0645\u0627\u0646 \u0645\u0647\u0645\u200c\u0627\u0646\u062f.\n\u0646\u0647\u0627\u062f (\u0647\u0627)",
    )
    optional_before = audit_translation_language(
        "The discourse(s) matter.", "\u06af\u0641\u062a\u0645\u0627\u0646 (\u0647\u0627) \u0645\u0647\u0645\u200c\u0627\u0646\u062f."
    )
    optional_after = audit_translation_language(
        "The discourse(s) matter.", optional_repaired
    )
    dash_repaired, _ = repair_source_grounded_language_artifacts(
        "They make history - their own and others' - in context.",
        "\u062a\u0627\u0631\u06cc\u062e \u062e\u0648\u062f \u0648 \u062f\u06cc\u06af\u0631\u0627\u0646 \u2014 \u0631\u0627 \u0645\u06cc\u200c\u0633\u0627\u0632\u0646\u062f.",
    )
    return {
        "opt_in_book_term_review_is_off_by_default": bool(
            not TarjomehConfig().translation.review_book_terms_before_translating
        ),
        "reviewed_terms_are_paragraph_scoped": bool(
            len(body_matches) == 1 and not heading_matches and heading_review
        ),
        "source_proven_surface_repair_is_bounded": bool(
            surface_changes and "و و" not in surface_repaired
        ),
        "final_optional_plural_admission": bool(
            _language_quality_strictly_improves(optional_before, optional_after)
        ),
        "contextual_person_name_quarantine": bool(
            is_bounded_person_name_target(
                "Manuela Tecusan", "\u0645\u0627\u0646\u0648\u0626\u0644\u0627 \u062a\u06a9\u0648\u0634\u0627\u0646"
            )
            and not is_bounded_person_name_target(
                "Manuela Tecusan", "\u0648\u06cc\u0631\u0627\u0633\u062a\u0627\u0631\u06cc \u0639\u0627\u0644\u0645\u0627\u0646\u0647 \u0648 \u06a9\u0627\u0645\u0644\u0627 \u062a\u062e\u0635\u0635\u06cc \u0645\u0627\u0646\u0648\u0626\u0644\u0627 \u062a\u06a9\u0648\u0634\u0627\u0646"
            )
        ),
        "orphan_object_marker_dash_repair": "\u2014 \u0631\u0627" not in dash_repaired,
        "uncertain_post_edit_issue_blocks": bool(uncertain_regression),
        "unchanged_grounded_issue_is_review_only": bool(
            not old_regression and len(old_scope) == 1
        ),
        "partial_readability_evidence_survives_invalid_sibling": bool(
            not partial_readability.valid and len(partial_readability.issues) == 1
            and partial_readability.validation_errors
        ),
        "source_scoped_optional_plural_spacing": bool(
            "\u06af\u0641\u062a\u0645\u0627\u0646(\u0647\u0627)" in optional_repaired
            and "\u0646\u0647\u0627\u062f (\u0647\u0627)" in optional_unrelated
        ),
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
        **_v10401_behavior_probes(),
        **_v10402_behavior_probes(),
        **_v10403_behavior_probes(),
    }


def _v10401_behavior_probes() -> dict[str, bool]:
    """Pure v10.40.1 probes: no network, LLM, database or job state."""
    from tarjomeh.context.book_term_candidates import body_term_paragraphs
    from tarjomeh.core.term_notes import _extend_persian_anchor_end
    from tarjomeh.exporters.docx_exporter import source_superscript_spans
    from tarjomeh.memory.manager import _is_paratext_style_source
    from tarjomeh.memory.proper_nouns import (
        is_bounded_person_name_target,
        is_reusable_terminology_mapping,
    )
    from tarjomeh.parsers.base import Chapter, Document, Paragraph, Section
    from tarjomeh.parsers.pdf_parser import _italic_runs, locate_source_marker
    from tarjomeh.quality.integrity import (
        extract_identifiers,
        repair_proven_surface_artifacts,
        repair_source_grounded_language_artifacts,
    )

    note_source = "In the 1920s (3) writers followed,3 historical accounts (chapter 3)."
    note_record = {"text": "3", "context_before": "iters followed,", "context_after": " historical"}
    note_target = "در دههٔ ۱۹۲۰ (۳) نویسندگان پیروی کردند، ۳ گزارش‌های تاریخی (فصل ۳)."
    note_spans = source_superscript_spans(
        note_target, {"superscript_markers": [note_record]}, note_source,
    )
    italic = _italic_runs([
        {"spans": [{"text": "Ideo­", "flags": 2, "font": "Italic"}]},
        {"spans": [{"text": "logiekritik", "flags": 2, "font": "Italic"}]},
    ])
    sampling_document = Document(title="T", chapters=[
        Chapter(title="T", sections=[Section(title="", level=2, paragraphs=[
            Paragraph("Copyright notice and publisher details."),
        ])]),
        Chapter(title="Contents", sections=[]),
        Chapter(title="1 Argument", sections=[Section(title="", level=2, paragraphs=[
            Paragraph("The argument concerns state power."),
        ])]),
        Chapter(title="Subject Index", sections=[Section(title="", level=2, paragraphs=[
            Paragraph("state power 3, 17, 22"),
        ])]),
    ])
    sampled = [paragraph.text for paragraph in body_term_paragraphs(sampling_document)]
    zwnj_fixed, _ = repair_proven_surface_artifacts(
        "SRA strategic–relational approach",
        "SRA رویکرد راهبردی‌‌رابطه‌ای",
        structural_role="mixed", paragraph_roles=["table"],
    )
    wrapper_source = "It acquires its own political rationale (raison d’état)."
    wrapper_once, _ = repair_source_grounded_language_artifacts(
        wrapper_source, "عقلانیت سیاسی (مصلحت دولت [(raison d’état)]) را",
    )
    wrapper_twice, _ = repair_source_grounded_language_artifacts(
        wrapper_source, wrapper_once,
    )
    kasra_fixed, _ = repair_source_grounded_language_artifacts(
        "It examines the elective affinities between them.",
        "خویشاوندی‌های انتخابی (elective affinities)ِ میان آن‌ها را بررسی می‌کند.",
    )
    return {
        "note_marker_is_context_anchored": bool(
            locate_source_marker(note_source, note_record)
            == note_source.index("followed,3") + len("followed,")
            and len(note_spans) == 1
            and note_target[:note_spans[0][0]].rstrip().endswith("،")
        ),
        "italic_run_crosses_soft_hyphen": italic == ["Ideologiekritik"],
        "paratext_excluded_from_term_sampling": sampled == [
            "The argument concerns state power."
        ],
        "zwnj_repaired_in_table_rows": "‌‌" not in zwnj_fixed,
        "inline_original_wrapper_is_idempotent": bool(
            "(مصلحت دولت [raison d’état])" in wrapper_once
            and "[(" not in wrapper_once
            and wrapper_once == wrapper_twice
        ),
        "stranded_kasra_returns_to_word": bool(
            "انتخابیِ (elective affinities)" in kasra_fixed
            and _extend_persian_anchor_end("انتخابیِ", 7) == 8
        ),
        "us_postal_code_is_identifier": bool(
            extract_identifiers("Malden, MA 02148") == {"ma 02148": 1}
            and not extract_identifiers("in 1988")
        ),
        "determiner_phrase_is_not_reusable": bool(
            not is_reusable_terminology_mapping(
                "some broad macro-trends", "کلان‌روندهای گسترده"
            )
            and is_reusable_terminology_mapping("state building", "دولت‌سازی")
        ),
        "name_guard_rejects_surrounding_prose": bool(
            not is_bounded_person_name_target(
                "Manuela Tecusan", "تخصصیِ مانوئلا تکوشان"
            )
            and not is_bounded_person_name_target("Hegel", "فیلسوف هگل")
            and is_bounded_person_name_target("Hegel", "گ. و. ف. هگل")
        ),
        "acknowledgement_is_not_style_evidence": bool(
            _is_paratext_style_source("Special thanks are also due to the editors.")
            and not _is_paratext_style_source(
                "Chapter 4 is dedicated to the analysis of class power."
            )
        ),
    }


def _v10402_behavior_probes() -> dict[str, bool]:
    """Pure identifier evidence checks; no network, LLM, or job state."""
    from tarjomeh.quality.integrity import restore_source_identifiers

    payload = "1234-5678"
    source = f"ISBN-13: {payload}\n\nISBN {payload}"
    target = f"شابک {payload}\n\nشابک-۱۳: {payload}"
    repaired, _ = restore_source_identifiers(source, target)
    bare_target = f"شابک {payload}\n\nشابک {payload}"
    bare_repaired, _ = restore_source_identifiers(source, bare_target)
    mixed_source = f"ISBN {payload}\n\nISSN {payload}"
    mixed_target = f"ISBN {payload}\n\nشابک {payload}"
    mixed_repaired, _ = restore_source_identifiers(mixed_source, mixed_target)
    return {
        "repeated_isbn_labels_use_target_evidence": repaired == (
            f"ISBN {payload}\n\nISBN-13: {payload}"
        ),
        "unproven_repeated_isbn_labels_remain_unresolved": (
            bare_repaired == bare_target
        ),
        "persian_book_number_never_becomes_issn": (
            mixed_repaired == mixed_target
        ),
    }


def _v10403_behavior_probes() -> dict[str, bool]:
    """Pure style, affix, render-identity and book-term checks; no LLM or DB."""
    import hashlib
    import re

    from tarjomeh.core.config import TarjomehConfig
    from tarjomeh.core.pipeline import repair_source_grounded_paragraphs
    from tarjomeh.core.render_identity import RenderChangeLedger
    from tarjomeh.core.term_notes import normalize_adjacent_original_citations
    from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
    from tarjomeh.glossary.book_review import (
        check_reviewed_book_terms,
        persian_option_defect,
        resolve_reviewed_book_terms,
    )
    from tarjomeh.memory.manager import MemoryManager, _style_record_is_authoritative
    from tarjomeh.quality.integrity import (
        spaced_optional_prefix_artifacts,
        unexpected_latin_prose,
    )

    def sha(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    scores = dict.fromkeys(("accuracy", "fluency", "terminology", "register"), 9.5)
    cited = (
        "این جمله نمونه‌ای روشن از نثر دانشگاهی پیوسته و دقیق است. "
        "این ادعا در پژوهش‌های پیشین بررسی شده است (Author 1990)."
    )
    manager = MemoryManager(TarjomehConfig())
    manager._update_style_profile(
        cited, source_paragraphs=["A clear sentence. The claim (Author 1990)."],
        source_paragraph_indices=[0], source_alignment_proven=True,
        final_scores=scores,
    )
    stored = manager.style_sample_records[-1] if manager.style_sample_records else {}

    legacy = MemoryManager(TarjomehConfig())
    legacy.from_dict({
        "style_samples": ["استدلال دانشگاهی."],
        "style_sample_records": [{
            "text": "استدلال دانشگاهی.", "text_hash": sha("استدلال دانشگاهی."),
            "source_text": "The academic argument continues.",
            "alignment_status": "exact_paragraph", "representative": True,
            "quality_score": 100.0, "final_scores": scores, "source_chunk_index": 0,
        }],
        "past_translations": [{
            "entry_id": 0, "source": "The academic argument continues.",
            "translation": "استدلال دانشگاهی ادامه می‌یابد.", "reliable": True,
            "chunk_index": 0,
        }],
    })

    mixed, _ = repair_source_grounded_paragraphs(
        "1 Introduction\n\nIt concerns (inter)national relations.",
        "۱ مقدمه\n\nروابط (بینا) ملی.",
        ["heading", "body"], require_monotonic=True,
    )

    def document(target: str) -> TranslatedDocument:
        return TranslatedDocument(title="t", author="a", paragraphs=[
            TranslatedParagraph(
                index=0,
                source_text="Exemplary is Jane Author's (1990) work.",
                translated_text=target,
                metadata={"structure_role": "body"},
            ),
        ], metadata={})

    merged = document("اثر جین آتور (Jane Author) (1990).")
    merge_ledger = RenderChangeLedger(merged)
    merge_ledger.step(
        "citation_merge", normalize_adjacent_original_citations,
        {"Jane Author": "جین آتور"},
    )
    unknown = document("اثر جین آتور.")
    unknown_ledger = RenderChangeLedger(unknown)
    unknown.paragraphs[0].translated_text = "متنی دیگر."

    paragraph = "The institution acts. Later the institution changes."
    metadata = {
        "paragraph_indices": [0], "source_paragraph_spans": [[0, len(paragraph)]],
        "source_paragraph_hashes": [sha(paragraph)], "structural_roles": ["body"],
    }
    spans = [(m.start(), m.end()) for m in re.finditer("institution", paragraph)]
    ticks = [
        {"paragraph_index": 0, "paragraph_sha256": sha(paragraph), "start": start,
         "end": end, "matched": paragraph[start:end]}
        for start, end in spans
    ]
    term = {"source": "institution", "target": "نهاد", "status": "approved",
            "scope_mode": "reviewed_occurrences"}
    partial, partial_review = resolve_reviewed_book_terms(
        paragraph, metadata, 1, [{**term, "approved_occurrences": ticks[:1]}])
    both, _ = resolve_reviewed_book_terms(
        paragraph, metadata, 1, [{**term, "approved_occurrences": ticks}])
    both_report, both_uncertain = check_reviewed_book_terms(
        "نهاد عمل کرد و سپس تغییر کرد.", paragraph, metadata, both)

    imprint = "Typeset in Serif by Example Typesetting Limited"
    return {
        "complete_style_pair_is_stored_whole": bool(
            stored.get("text") == cited
            and stored.get("sample_scope") == "complete_paragraph"
            and _style_record_is_authoritative(stored)
        ),
        "clipped_legacy_style_is_quarantined": (
            legacy.style_sample_records[0].get("legacy_style_status")
            == "unverifiable_quarantined"
            and not _style_record_is_authoritative(legacy.style_sample_records[0])
        ),
        "prefix_repaired_per_paragraph_before_canonical": (
            mixed == "۱ مقدمه\n\nروابط (بینا)ملی."
        ),
        "joined_plural_is_not_a_spaced_prefix": spaced_optional_prefix_artifacts(
            "policy(s) shape (inter)national claims.",
            "سیاست(ها) به مدعیات (بینا)ملی شکل دادند.",
        ) == [],
        "render_citation_merge_is_replay_proven": bool(
            merge_ledger.audit()["passed"]
            and "(Jane Author, 1990)" in merged.paragraphs[0].translated_text
        ),
        "render_change_without_producer_blocks": (
            unknown_ledger.audit()["passed"] is False
        ),
        "partial_occurrence_ticks_are_not_prompted": bool(
            not partial
            and partial_review
            and partial_review[0]["reason"] == "partial_occurrence_approval"
        ),
        "repeated_term_occurrences_stay_review": bool(
            both
            and both_report.total_checked == 0
            and both_uncertain
            and "rendering_count_short" in both_uncertain[0]["reasons"]
        ),
        "imprint_names_are_source_grounded": unexpected_latin_prose(
            imprint, "توسط شرکت نمونه (Example Typesetting Limited).",
        ) == [],
        "malformed_persian_option_is_withheld": (
            persian_option_defect("مفهوم\u200cـ\u200cنمونه")
            == "tatweel_in_persian_option"
        ),
    }
