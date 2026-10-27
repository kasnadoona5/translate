"""Stable runtime capability reporting for deployment and audit tooling."""

from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any

RUNTIME_RELEASE = "v22.5"
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
            "numeral_boundary_preservation": True,
            "source_metalinguistic_token_protection": True,
            "published_output_coverage": True,
            "reviewed_summary_revision": True,
            "cumulative_summary_input_trust": True,
            "cross_segment_quote_grounding": True,
            "offline_grouped_integer_evaluation": True,
            "compound_chapter_count_evidence": True,
            "arabic_indic_chapter_references": True,
            "empty_memory_pair_retention": True,
            "docx_heading_font_inheritance": True,
            "localized_number_evaluation": True,
            "score_grounding_diagnostics_only": True,
            "source_scoped_admission_reporting": True,
            "atomic_summary_commit": True,
            "numeric_announcement_boundaries": True,
            "final_entity_local_issue_scope": True,
            "final_candidate_entity_memory": True,
            "first_occurrence_evidence_classification": True,
            "shared_original_keeper_eligibility": True,
            "document_scoped_duplicate_proof": True,
            "closing_parenthesis_list_evidence": True,
            "paired_object_marker_dash_classification": True,
            "inflected_multiword_anchor_proof": True,
            "organization_lexical_boundary": True,
            "book_identity_date_guard": True,
            "source_eligible_style_memory": True,
            "historical_style_source_quarantine": True,
            **{
                name: True
                for name in (
                    "order_independent_readability_evidence",
                    "identical_term_proposal_dedup",
                    "fresh_term_scope_default",
                    "persian_plus_scoped_original",
                    "pending_only_term_skip",
                    "curated_revision_authority",
                    "prior_only_retranslation_context",
                    "active_memory_version_replacement",
                    "pending_canonical_summary_coverage",
                    "paragraph_role_style_evidence",
                    "raw_token_counter_diagnostics",
                    "actual_ignored_policy_diagnostics",
                )
            },
            "persisted_job_policy_isolation": True,
            "memory_category_and_conflict_authority": True,
            "accepted_entity_not_lexical_plural": True,
            "accepted_only_surface_salvage": True,
            "shared_raw_paragraph_admission": True,
            "canonical_quote_diff_binding": True,
            "matched_readability_advisory_routing": True,
            "selected_body_proposal_evidence": True,
            "paragraph_term_candidate_admission": True,
            "summary_exact_section_headers": True,
            "shared_source_entity_inventory_contract": True,
            "lexical_annotation_memory_deferral": True,
            "complete_terminology_reconciliation_events": True,
            "optional_book_term_recommendations": True,
            "advisory_term_observations_nonblocking": True,
            "report_only_book_term_presence": True,
            "publisher_before_date_research_guard": True,
            "explicit_hyphen_name_group_alignment": True,
            "final_heh_ezafe_name_edges": True,
            "source_local_annotation_syntax": True,
            "r109_serious_unchanged_count_preserved": True,
            "bounded_name_suffix_components": True,
            "conservative_automatic_name_ambiguity": True,
            "quoted_suffix_source_scope": True,
            "dash_clause_bounded_person_names": True,
            "extra_final_unchanged_minor_not_counted": True,
            "quoted_suffix_review_reporting": True,
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
            "physical_source_note_cue_inventory": True,
            "docx_visible_note_cardinality": True,
            "unchanged_scoped_term_review_is_not_edit_veto": True,
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
            "selected_chapter_term_index": True,
            "bilingual_docx_identity": True,
            "funding_disclaimer_style_exclusion": True,
            "quantified_object_term_quarantine": True,
            "exact_previous_context_reference": True,
            "source_edition_metadata_guard": True,
            "precanonical_duplicate_comma_repair": True,
            "machine_json_separate_logging": True,
            "bounded_extra_final_refinement": True,
            "candidate_diff_bound_refinement_spans": True,
            "rejected_summary_full_evidence": True,
            "current_book_term_review_audit": True,
            "native_contents_unit_audit": True,
            "all_approved_scopes_count_occurrences": True,
            "infinitival_passage_memory_guard": True,
            "shared_objective_style_guard": True,
            "source_authorized_annotation_nesting": True,
            "sentence_proven_dash_review": True,
            "research_evidence_kind_reporting": True,
            "bounded_exact_prompt_component_evidence": True,
            "reference_scoped_count_evidence": True,
            "shared_object_marker_dash_style_evidence": True,
            "bounded_prompt_component_export": True,
            "requested_and_returned_model_reporting": True,
            "report_only_semantic_evidence_labels": True,
        },
        "policy_versions": {
            "fresh_job_memory": 1,
            "manual_retranslation_admission": 1,
            "structure_evidence": 3,
            "canonical_text": 3,
            "layer1_admission": 13,
            "style_evidence": 11,
            "benchmark_schema": 1,
            "checkpoint_export": 3,
            "paragraph_identity": 1,
            "critic_source_coverage": 1,
            "final_identifier_admission": 1,
            "recovery_segments": 1,
            "summary_admission": 3,
            "final_candidate_selection": 2,
            "final_quality_authority": 1,
            "note_marker_recovery": 5,
            "critique_canonical_rebind": 1,
            "final_quality_checkpoint": 3,
            "source_obligation_recovery": 2,
            "local_refiner_salvage": 3,
            "research_prompt_admission": 2,
            "targeted_language_repair": 2,
            "post_edit_issue_attribution": 1,
            "attachment_trial": 1,
            "book_term_scope": 6,
            "final_language_admission": 7,
            "book_term_review": 4,
            "source_term_inventory": 3,
            "render_identity": 3,
            "previous_context_identity": 1,
            "research_edition_metadata": 4,
            "extra_final_refinement": 1,
            "comma_surface_admission": 1,
            "prompt_component_evidence": 1,
            "research_evidence_reporting": 1,
            "prompt_component_export": 1,
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
                "Ada North", "\u0622\u062f\u0627 \u0646\u0648\u0631\u062b"
            )
            and not is_bounded_person_name_target(
                "Ada North", "\u0648\u06cc\u0631\u0627\u0633\u062a\u0627\u0631\u06cc \u0639\u0627\u0644\u0645\u0627\u0646\u0647 \u0648 \u06a9\u0627\u0645\u0644\u0627 \u062a\u062e\u0635\u0635\u06cc \u0622\u062f\u0627 \u0646\u0648\u0631\u062b"
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
            and not manager._render_style_profile()
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
        **_v20_behavior_probes(),
        **_v202_behavior_probes(),
        **_v203_behavior_probes(),
        **_v204_behavior_probes(),
        **_v205_behavior_probes(),
        **_v206_behavior_probes(),
        **_v207_behavior_probes(),
        **_v208_behavior_probes(),
        **_v2081_behavior_probes(),
        **_v2082_behavior_probes(),
        **_v2083_behavior_probes(),
        **_v2084_behavior_probes(),
        **_v2085_behavior_probes(),
        **_v2086_behavior_probes(),
        **_v2087_behavior_probes(),
        **_v2088_behavior_probes(),
        **_v2089_behavior_probes(),
        **_v20810_behavior_probes(),
        **_v20811_behavior_probes(),
        **_v20812_behavior_probes(),
        **_v21_behavior_probes(),
        **_v22_behavior_probes(),
        **_v221_behavior_probes(),
        **_v222_behavior_probes(),
        **_v223_behavior_probes(),
        **_v224_behavior_probes(),
        **_v225_behavior_probes(),
    }


def _v225_behavior_probes() -> dict[str, bool]:
    """Synthetic behavior only: no job files, providers or configuration reads."""
    from tarjomeh.jobs.output_coverage import output_coverage
    from tarjomeh.persian.typography import PersianTypographer
    from tarjomeh.quality.integrity import unexpected_latin_prose

    spaced = "\u0633\u0647 \u0631\u0627\u0647"
    joined = "\u0633\u0647\u200c\u0631\u0627\u0647"
    p = PersianTypographer()
    rows = [dict(chunk_index=i, status="completed", translation=f"text {i}") for i in range(3)]
    identity = dict(persisted_chunk_count=2, lexically_identical=True,
                    expected_hash=hashlib.sha256(b"text 0 text 1").hexdigest())
    publication = dict(boundary_chunk_index=1, output_sha256="a" * 64)
    coverage = output_coverage(rows, identity, publication, "a" * 64)
    return {
        "numeral_boundary_preservation": p.process(spaced) == spaced and p.process(joined) == joined,
        "source_metalinguistic_token_protection": (
            not unexpected_latin_prose("The term derives from Latin civitas.", "civitas")
            and bool(unexpected_latin_prose("The discussion continues.", "civitas"))
            and bool(unexpected_latin_prose("The term derives from Recent events.", "events"))
            and bool(unexpected_latin_prose("The term derives from Latin civitas.", "civitas-like"))
        ),
        "published_output_coverage": (
            coverage["status"] == "partial" and coverage["published_finished_count"] == 2
            and output_coverage(rows, {}, {}, None)["status"] == "unknown"
        ),
    }


def _v224_behavior_probes() -> dict[str, bool]:
    """Reviewed context is hash-bound and cannot restore superseded history."""
    from copy import deepcopy

    from tarjomeh.memory.summary_revision import document_hash, history_key, select_summary

    old = {"english_summary": "The earlier claim has a different subject.",
           "persian_summary": (
               "\u0627\u062f\u0639\u0627 \u0628\u0647 \u06cc\u06a9 \u0634\u0631\u0637 "
               "\u0648\u0627\u0628\u0633\u062a\u0647 \u0627\u0633\u062a."
           ),
           "input_trust": "advisory_inputs", "trust_reasons": ["earlier_review"]}
    new = dict(old, english_summary="The enzyme carries a conditional signal.")
    refs = [{"chunk_index": 0, "source_sha256": "a" * 64, "target_sha256": "b" * 64}]
    prior = {"summary": old, "coverage": refs}
    manifest = dict(schema=1, job_id="pure_probe", document_sha256="c" * 64,
                    expected_memory_sha256="d" * 64, previous_summary=old,
                    candidate_summary=new, coverage=refs, supersedes=[list(history_key(prior))],
                    evidence=[dict(chunk_index=0, source_sha256="a" * 64, start=0, end=3,
                                   quote="The", claim="subject")])
    digest = document_hash(manifest)
    entry = {"summary": new, "coverage": refs, "reviewed_revision": {
        "schema": 1, "revision_id": digest, "approval_sha256": digest,
        "reviewer": "synthetic reviewer", "manifest": manifest}}
    accepted = select_summary([prior, entry], lambda ref: True) == entry
    excluded = not select_summary([prior, entry], lambda ref: False)
    corrupt = deepcopy(entry)
    corrupt["reviewed_revision"]["manifest"]["candidate_summary"]["input_trust"] = "reviewed_inputs"
    rejected = select_summary([prior, corrupt], lambda ref: True) == prior
    revoked = {"summary": old, "coverage": refs,
               "summary_revision_rollback": {"revision_id": digest}}
    return {"reviewed_summary_revision": bool(
        accepted and excluded and rejected
        and select_summary([prior, entry, revoked], lambda ref: True) == revoked
    )}


def _v223_behavior_probes() -> dict[str, bool]:
    """Cumulative input provenance is conservative, bounded and context-local."""
    from tarjomeh.memory.bilingual_summary import BilingualSummary

    previous = BilingualSummary()
    previous.english_summary = "An earlier conditional claim."
    previous.set_input_trust("advisory_inputs", ["earlier_review"])
    before = previous.serialize()
    candidate = BilingualSummary()
    lineage = candidate.set_cumulative_input_trust("reviewed_inputs", [], previous)
    mixed = (candidate.input_trust == "advisory_inputs"
             and "earlier_review" in candidate.trust_reasons
             and "prior_summary_advisory_inputs" in candidate.trust_reasons
             and lineage["inherited_advisory"] is True and previous.serialize() == before)
    previous.set_input_trust("reviewed_inputs", [])
    candidate.set_cumulative_input_trust("reviewed_inputs", [], previous)
    reviewed = candidate.input_trust == "reviewed_inputs"
    previous.english_summary = ""
    previous.set_input_trust("advisory_inputs", ["inactive"])
    candidate.set_cumulative_input_trust("reviewed_inputs", [], previous)
    empty = candidate.input_trust == "reviewed_inputs" and not candidate.trust_reasons
    candidate.set_cumulative_input_trust("unknown", [], previous)
    return {"cumulative_summary_input_trust": bool(
        mixed and reviewed and empty and candidate.input_trust == "advisory_inputs"
    )}


def _v222_behavior_probes() -> dict[str, bool]:
    """Location evidence and offline grouping never waive semantic admission."""
    from tarjomeh.quality.evaluation import _offline_numeric_evidence
    from tarjomeh.quality.grounding import resolve_source_segment, source_quote_span
    from tarjomeh.quality.integrity import reconcile_numbers

    source = "Vale et al. (2018) report a measured trend."
    proof = source_quote_span(source, source)
    grouped = "There are 1,000 records."
    plain = "There are 1000 records."
    return {
        "cross_segment_quote_grounding": bool(
            proof and proof["segment_ids"] == ["p1:s1", "p1:s2"]
            and resolve_source_segment(source, source) == ("p1:s1", [])
            and source_quote_span(source + "\n\n" + source, source) is None
            and source_quote_span(source.replace(". (", ".\n\n("), source) is None
            and source_quote_span("A. A. A.", "A. A.") is None
        ),
        "offline_grouped_integer_evaluation": bool(
            not _offline_numeric_evidence(grouped, plain)["missing"]
            and _offline_numeric_evidence(grouped, "There are 2000 records.")["missing"]
            and _offline_numeric_evidence("Code: 1,000", "Code: 1000")["missing"]
            and _offline_numeric_evidence(grouped, "Account 1000")["missing"]
            and _offline_numeric_evidence(grouped, "Reference 1000")["missing"]
            and _offline_numeric_evidence(
                grouped, "\u0634\u0645\u0627\u0631\u0647 \u062d\u0633\u0627\u0628 1000",
            )["missing"]
            and not _offline_numeric_evidence(
                grouped, "\u062a\u0639\u062f\u0627\u062f "
                "\u0631\u06a9\u0648\u0631\u062f\u0647\u0627 1000",
            )["missing"]
            and reconcile_numbers(grouped, plain)["missing"]
        ),
    }


def _v221_behavior_probes() -> dict[str, bool]:
    """Pure chapter evidence checks, including a real mismatch in each branch."""
    from tarjomeh.quality.structure_audit import (
        announced_count_evidence,
        audit_payload,
        chapter_marker_count,
    )

    source = "This ten-chapter report covers " + ", ".join(
        f"Chapter {index}" for index in range(1, 11)
    ) + "."
    target = (
        "\u0627\u06cc\u0646 \u06af\u0632\u0627\u0631\u0634 "
        "\u062f\u0647\u200c\u0641\u0635\u0644\u06cc \u0634\u0627\u0645\u0644 "
    ) + ", ".join(
        "\u0641\u0635\u0644 " + str(index) for index in range(1, 11)
    ) + "."

    def mismatches(candidate):
        return (
            "translation_structure_mismatch" in audit_payload(source, candidate)["classifications"]
        )

    wrong = target.replace(
        "\u062f\u0647\u200c\u0641\u0635\u0644\u06cc",
        "\u062f\u0648\u0627\u0632\u062f\u0647\u200c\u0641\u0635\u0644\u06cc",
    )
    arabic = target.translate(str.maketrans(
        "0123456789", "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669",
    ))
    return {
        "compound_chapter_count_evidence": bool(
            any(item.value == 10 and item.semantic_category == "chapter"
                for item in announced_count_evidence(target))
            and not announced_count_evidence("\u0646\u0647 \u0641\u0635\u0644\u06cc")
            and not mismatches(target) and mismatches(wrong)
        ),
        "arabic_indic_chapter_references": bool(
            chapter_marker_count(arabic) == 10 and not mismatches(arabic)
            and mismatches(arabic.replace("\u0641\u0635\u0644 \u0661\u0660", ""))
        ),
    }


def _v22_behavior_probes() -> dict[str, bool]:
    """Pure checks for the narrow release; no provider, DB, files or semantic judge."""
    import json
    from copy import deepcopy

    from tarjomeh.chunking.chunker import Chunk
    from tarjomeh.core.config import TarjomehConfig
    from tarjomeh.core.pipeline import _reviewed_term_candidate_admission
    from tarjomeh.exporters.docx_exporter import target_run_font_size
    from tarjomeh.memory.manager import MemoryManager
    from tarjomeh.quality.critique import TranslationCritique
    from tarjomeh.quality.evaluation import inspect_chunk

    retained = True
    for policy in (0, 1):
        for seeded in (False, True):
            manager = MemoryManager(TarjomehConfig())
            manager.quality_policy_version = policy
            if seeded:
                manager.update_after_translation(
                    Chunk(0, "An earlier claim.", "Earlier", ""),
                    "\u0627\u062f\u0639\u0627\u06cc \u067e\u06cc\u0634\u06cc\u0646.",
                    style_approved=False,
                )
            before = deepcopy(manager.to_dict())
            for source, target in (("New source.", " \n"), (" \n", "New target.")):
                result = manager.update_after_translation(
                    Chunk(0, source, "Other", ""), target, style_approved=False,
                )
                retained &= bool(result.get("memory_updated") is False
                                 and manager.to_dict() == before)

    def number_passes(target):
        result = inspect_chunk("Chapter 3", target, "completed", [])
        return next(item["passed"] for item in result["checks"]
                    if item["id"] == "numbers_preserved")

    data = {"scores": dict(accuracy=2, fluency=8, terminology=8, register=8),
            "overall": 9, "issues": []}
    critique = TranslationCritique._parse_response(json.dumps(data))
    diagnostic = critique.diagnostic_evidence()
    invalid = TranslationCritique._parse_response(json.dumps({**data, "issues": [
        {"category": "accuracy", "severity": "critical", "confidence": 0.99,
         "source_quote": "not in source", "current_persian_quote": "target",
         "suggested_correction": "other", "rationale": "synthetic evidence"},
    ]}), "A source.", "A target.").diagnostic_evidence()

    source = "A concept and a meta-concept differ."
    digest = hashlib.sha256(source.encode()).hexdigest()
    target = "\u0645\u0641\u0647\u0648\u0645"
    matches = [
        {"paragraph_index": 0, "source": term, "target": rendering,
         "source_paragraph_hash": digest, "source_occurrence_count": count}
        for term, rendering, count in (
            ("concept", target, 2), ("meta-concept", "\u0641\u0631\u0627" + target, 1),
        )
    ]
    previous = target + " \u0641\u0631\u0627" + target + " differ."
    candidate = previous.replace("differ", "differ clearly")
    blocked = _reviewed_term_candidate_admission(source, previous, candidate, {}, matches)
    carried = _reviewed_term_candidate_admission(source, previous, previous, {}, matches)
    return {
        "empty_memory_pair_retention": retained,
        "docx_heading_font_inheritance": all(
            target_run_font_size(f"Heading {level}") is None for level in (1, 2, 3)
        ) and target_run_font_size("Normal") == 11,
        "localized_number_evaluation": bool(
            number_passes("\u0641\u0635\u0644 \u0633\u0648\u0645")
            and not number_passes("\u0641\u0635\u0644 \u0686\u0647\u0627\u0631\u0645")
        ),
        "score_grounding_diagnostics_only": bool(
            critique.average == 9 and critique.passes_threshold(8)
            and diagnostic["dimension_mean"] == 6.5 and diagnostic["model_overall"] == 9
            and invalid["ignored_severe_issue_count"] == 1 and invalid["grounding_incomplete"]
        ),
        "source_scoped_admission_reporting": bool(
            not blocked["accepted"] and blocked["violation_count"] == 0
            and blocked["blocking_review_count"] == 1 and blocked["nested_source_matches"]
            and blocked["candidate_target_sha256"] == hashlib.sha256(candidate.encode()).hexdigest()
            and carried["accepted"] and carried["unchanged_reviews"]
        ),
    }


def _v21_behavior_probes() -> dict[str, bool]:
    """Exercise actual summary commit preparation without I/O or async providers."""
    from copy import deepcopy

    from tarjomeh.core.config import TarjomehConfig
    from tarjomeh.memory.bilingual_summary import BilingualSummary
    from tarjomeh.memory.manager import MemoryManager

    manager = MemoryManager(TarjomehConfig())
    manager.quality_policy_version = 1
    manager.bilingual_summary.english_summary = "The earlier claim is conditional."
    manager.bilingual_summary.persian_summary = (
        "\u0627\u062f\u0639\u0627\u06cc \u067e\u06cc\u0634\u06cc\u0646 "
        "\u0645\u0634\u0631\u0648\u0637 \u0627\u0633\u062a."
    )
    ref = {"chunk_index": 0, "source_sha256": "source", "target_sha256": "target"}
    manager.summary_pending = [ref]

    def state():
        return deepcopy((manager.bilingual_summary.serialize(), manager.summary_pending,
                         manager.summary_coverage, manager.summary_history))

    class FailingCandidate(BilingualSummary):
        def serialize(self):
            raise RuntimeError("synthetic history preparation failure")

    before = state()
    retained = False
    try:
        manager._commit_bilingual_summary(
            FailingCandidate(), coverage_refs=[ref], candidate_quality={"accepted": True},
        )
    except RuntimeError:
        retained = state() == before
    candidate = BilingualSummary()
    candidate.english_summary = "The later claim is conditional."
    candidate.persian_summary = (
        "\u0627\u062f\u0639\u0627\u06cc \u0628\u0639\u062f\u06cc "
        "\u0645\u0634\u0631\u0648\u0637 \u0627\u0633\u062a."
    )
    result = manager._commit_bilingual_summary(
        candidate, coverage_refs=[ref], candidate_quality={"accepted": True},
    )
    committed = bool(
        result.get("candidate_committed")
        and manager.bilingual_summary is candidate
        and manager.summary_pending == []
        and manager.summary_coverage == [ref]
        and manager.summary_history == [{"summary": candidate.serialize(), "coverage": [ref]}]
    )
    return {"atomic_summary_commit": retained and committed}


def _v20812_behavior_probes() -> dict[str, bool]:
    from tarjomeh.quality.structure_audit import announced_count_evidence, audit_structure

    source = "2.2 Three elements"
    good = "2.2 \u0633\u0647 \u0639\u0646\u0635\u0631"
    bad = "2.2 \u062f\u0648 \u0639\u0646\u0635\u0631"
    return {"numeric_announcement_boundaries": bool(
        not announced_count_evidence("2.2 elements")
        and not announced_count_evidence("\u06f2\u066b\u06f2 \u0639\u0646\u0635\u0631")
        and [item.value for item in announced_count_evidence(source)] == [3]
        and [item.value for item in announced_count_evidence(good)] == [3]
        and not audit_structure(source, good)
        and any(item.details.get("admission") == "blocking"
                for item in audit_structure(source, bad))
    )}


def _v20811_behavior_probes() -> dict[str, bool]:
    from tarjomeh.core.quality_policy_probes import final_entity_local_scope_probe

    return {"final_entity_local_issue_scope": final_entity_local_scope_probe()}


def _v20810_behavior_probes() -> dict[str, bool]:
    from tarjomeh.core.quality_policy_probes import final_entity_probes

    return final_entity_probes()


def _v2089_behavior_probes() -> dict[str, bool]:
    from tarjomeh.core.quality_policy_probes import original_keeper_probes

    return original_keeper_probes()


def _v2088_behavior_probes() -> dict[str, bool]:
    from tarjomeh.core.quality_policy_probes import checkpoint_classification_probes

    return checkpoint_classification_probes()


def _v2087_behavior_probes() -> dict[str, bool]:
    from tarjomeh.core.quality_policy_probes import export_memory_research_probes

    return export_memory_research_probes()


def _v2086_behavior_probes() -> dict[str, bool]:
    from tarjomeh.core.quality_policy_probes import style_source_probes

    return style_source_probes()


def _v2085_behavior_probes() -> dict[str, bool]:
    from tarjomeh.core.quality_policy_probes import quality_policy_probes

    return quality_policy_probes()


def _v2084_behavior_probes() -> dict[str, bool]:
    """Synthetic behavior only: no files, database, providers or environment writes."""
    from types import SimpleNamespace

    from tarjomeh.core.config import TarjomehConfig
    from tarjomeh.core.pipeline import (
        _classify_candidate_quality_count,
        _critique_requires_refinement,
        _merge_readability_evidence,
        _rebind_canonical_critique,
        _reviewed_term_candidate_admission,
        _salvage_local_refinement_edits,
    )
    from tarjomeh.glossary.book_review import _sha, attest_book_term_proposals
    from tarjomeh.memory.bilingual_summary import BilingualSummary
    from tarjomeh.memory.proper_nouns import ProperNouns
    from tarjomeh.quality.critique import CritiqueResult

    config = TarjomehConfig()
    config.llm.model = "saved"
    diagnostics: list[dict[str, Any]] = []
    config._apply_env_overrides(
        persisted_job=True,
        load_diagnostics=diagnostics,
        environment={
            "TRANSLATOR_MODEL": "other",
            "TRANSLATOR_API_BASE": "http://router.invalid/v1",
        },
    )
    nouns = ProperNouns()
    nouns.add_noun(
        "Ava Morris",
        "آوا موریس",
        category="person",
        provenance="accepted_correction",
        alignment_status="exact_local",
    )
    entity_ok = not nouns.context_deferral_reason("Ava Morris")
    nouns.add_noun("civic order", "نظم مدنی", category="term", provenance="curated_glossary")
    nouns.add_noun(
        "civic order", "نظم مدنی", category="organization", provenance="observed_translation"
    )
    nouns.add_noun("state power", "قدرت دولتی", category="term", provenance="observed_translation")
    nouns.mark_context_conflict("state power", corrected_source="power", corrected_target="قدرت")
    nouns.add_noun("state power", "قدرت دولتی", category="term", provenance="observed_translation")
    summary = BilingualSummary()
    summary.update(
        "## English Summary\nFirst claim.\n### Detail\nSecond claim.\n"
        "## خلاصه فارسی\nادعای نخست.\n### جزئیات\nادعای بعدی."
    )
    source = "They act.\n\nA civic institution meets."
    before = "آنها می‌روند.\n\nنهاد مدنی دیدار می‌کند."
    after = before.replace("می‌روند", "کنش می‌کنند")
    old = {
        "issue_id": "old",
        "severity": "major",
        "category": "accuracy",
        "confidence": 0.60,
        "source_segment_id": "p2:s1",
        "source_quote": "civic institution",
        "current_persian_quote": "نهاد مدنی",
    }
    remaining, carried, _ = _classify_candidate_quality_count(
        [old], source_text=source, previous_text=before, candidate_text=after
    )
    padded, _, _ = _classify_candidate_quality_count(
        [old], source_text=source, previous_text=before, candidate_text=after + " "
    )
    binding = _rebind_canonical_critique(
        {
            "issue_details": [
                {
                    "source_segment_id": "p1:s1",
                    "current_persian_quote": "12 مورد",
                }
            ]
        },
        "12 items.",
        "12 مورد است.",
        "۱۲ مورد است.",
    )
    issue = {
        "issue_id": "minor",
        "category": "fluency",
        "severity": "minor",
        "confidence": 0.5,
        "source_segment_id": "p1:s1",
        "source_quote": "A claim.",
        "current_persian_quote": "عبارت ناقص است",
    }
    critique = CritiqueResult(average=9, issue_details=[issue])
    _merge_readability_evidence(
        critique,
        [
            {
                "severity": "major",
                "current_persian_quote": "عبارت ناقص است",
                "suggested_correction": "عبارت کامل است",
                "rationale": "Broken grammar and missing predicate.",
            }
        ],
        source_text="A claim.",
        translation="عبارت ناقص است.",
    )
    paragraph = "Institutions act."
    metadata = {
        "paragraph_indices": [0],
        "structural_roles": ["body"],
        "source_paragraph_spans": [[0, len(paragraph)]],
        "source_paragraph_hashes": [_sha(paragraph)],
    }
    index = [
        {
            "chunk_index": 0,
            "local_paragraph": 0,
            "paragraph_index": 0,
            "paragraph_sha256": _sha(paragraph),
            "chapter_position": 2,
        }
    ]
    proposals = attest_book_term_proposals(
        [
            {"source": "institution", "source_variants": ["Institutions"]},
        ],
        index,
        {0: (paragraph, metadata)},
    )
    term = {
        "source": "Institutions",
        "target": "نهادها",
        "paragraph_index": 0,
        "source_paragraph_hash": _sha(paragraph),
        "source_occurrence_count": 1,
    }
    term_admission = _reviewed_term_candidate_admission(
        paragraph, "نهادها عمل می‌کنند.", "سامانه‌ها عمل می‌کنند.", metadata, [term]
    )

    class Gate:
        def evaluate(self, *args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(accepted=True, to_dict=lambda: {"accepted": True})

    salvaged, decisions, _ = _salvage_local_refinement_edits(
        source="It happens.",
        previous="این امر انجام می شود.",
        proposed="این امر انجام می‌شود.",
        issue_details=[
            {
                "issue_id": "space",
                "source_segment_id": "p1:s1",
                "current_persian_quote": "می شود",
                "category": "typography",
                "rationale": "Incorrect spacing.",
            }
        ],
        issue_decisions=[
            {"issue_id": "space", "decision": "accepted", "resulting_span": "می‌شود"},
            {
                "issue_id": "veto",
                "decision": "rejected",
                "original_span": "می شود",
                "resulting_span": "می‌شود",
            },
        ],
        integrity_gate=Gate(),
        protected_terms=[],
        protect_inline_english=False,
        allowed_inline_originals=[],
    )
    return {
        "persisted_job_policy_isolation": config.llm.model == "saved"
        and bool(diagnostics)
        and config.llm.openrouter.api_base == "http://router.invalid/v1",
        "memory_category_and_conflict_authority": nouns.category_for("civic order") == "term"
        and nouns.is_context_deferred("state power"),
        "accepted_entity_not_lexical_plural": entity_ok,
        "accepted_only_surface_salvage": salvaged == "این امر انجام می‌شود."
        and decisions[1]["commit_status"] == "not_committed",
        "shared_raw_paragraph_admission": not remaining and carried == [old] and padded == [old],
        "canonical_quote_diff_binding": binding["issue_details"][0]["current_persian_quote"]
        == "۱۲ مورد",
        "matched_readability_advisory_routing": _critique_requires_refinement(critique, 9)
        and issue["severity"] == "minor"
        and len(critique.issue_details) == 1,
        "selected_body_proposal_evidence": proposals[0]["source"] == "Institutions"
        and proposals[0]["source_evidence_sha256"] == _sha(paragraph),
        "paragraph_term_candidate_admission": not term_admission["accepted"],
        "summary_exact_section_headers": "Second claim." in summary.english_summary
        and "ادعای بعدی." in summary.persian_summary,
    }


def _v2083_behavior_probes() -> dict[str, bool]:
    """General pure probes; no configuration, database, provider or job I/O."""
    import json
    from types import SimpleNamespace

    from tarjomeh.context.book_researcher import BookResearcher
    from tarjomeh.context.source_entities import _source_entity_inventory
    from tarjomeh.core import pipeline as core_pipeline
    from tarjomeh.core.pipeline import (
        _canonical_chunk_paragraph_identity,
        _reconcile_committed_terminology,
    )
    from tarjomeh.glossary.book_recommendations import (
        recommendation_presence,
        resolve_book_recommendations,
    )
    from tarjomeh.glossary.book_review import _sha, resolve_reviewed_book_terms
    from tarjomeh.memory.proper_nouns import ProperNouns
    from tarjomeh.quality.critique import TranslationCritique

    nouns = ProperNouns()
    unsafe = nouns.add_noun("Canada", "CA \u06a9\u0627\u0646\u0627\u062f\u0627", category="term")
    db = SimpleNamespace(
        get_chunk_events=lambda *args: [],
        get_qa_issues=lambda *args: [{"issue_id": "probe", "category": "terminology",
                                    "source_quote": "Canada"}],
        get_issue_decisions=lambda *args: [{"issue_id": "probe",
            "resulting_span": "\u06a9\u0627\u0646\u0627\u062f\u0627",
            "payload": {"commit_status": "committed_local"}}],
    )
    reconciliation = _reconcile_committed_terminology(
        db, "synthetic", 0, SimpleNamespace(proper_nouns=nouns), "Canada",
        "\u06a9\u0627\u0646\u0627\u062f\u0627",
    )
    source = "The apparatus acts."
    metadata = {"structural_roles": ["body"], "paragraph_indices": [17],
                "source_paragraph_spans": [[0, len(source)]],
                "source_paragraph_hashes": [_sha(source)]}
    decision = {
        "source": "apparatus", "target": "\u062f\u0633\u062a\u06af\u0627\u0647",
        "status": "approved",
        "scope_mode": "book_wide_recommended", "source_evidence_sha256": _sha("Other paragraph."),
    }
    matches, _ = resolve_reviewed_book_terms(source, metadata, 1, [decision])
    advice = resolve_book_recommendations(source, metadata, 0, {
        "review_version": 3, "phase": "started", "source_sha256": "synthetic",
        "proposals": [decision],
    }, {"source_sha256": "synthetic", "entries": [{
        "chunk_index": 0, "local_paragraph": 0, "paragraph_index": 17,
        "paragraph_sha256": _sha(source), "chapter_position": 1,
    }]})["recommendations"]
    optional = TranslationCritique._parse_response(json.dumps({
        "scores": {name: 9 for name in ("accuracy", "fluency", "terminology", "register")},
        "issues": [], "advisory_term_observations": "invalid optional data",
    }))
    target, identity = _canonical_chunk_paragraph_identity(
        SimpleNamespace(text=source, metadata=metadata),
        "\u0633\u0627\u0632\u0648\u06a9\u0627\u0631",
    )
    presence = recommendation_presence(source, target, metadata, advice, identity)
    context, conflicts = BookResearcher._reconcile_publication_context(
        "A study (published by Example, 2004/2005).", [],
    )
    return {
        "shared_source_entity_inventory_contract": (
            core_pipeline._source_entity_inventory is _source_entity_inventory
            and not _source_entity_inventory("Nora\nVale")
        ) and _source_entity_inventory(
            "Nora Vale spoke. Omar Reed replied."
        ) == ["Nora Vale", "Omar Reed"],
        "lexical_annotation_memory_deferral": unsafe.get("action") == "context_deferred",
        "complete_terminology_reconciliation_events": bool(reconciliation["reconciled"])
        and reconciliation["reconciled"][0]["action"] == "added",
        "optional_book_term_recommendations": not matches and len(advice) == 1,
        "advisory_term_observations_nonblocking": optional.valid and not optional.issue_details
        and bool(optional.advisory_term_observation_errors),
        "report_only_book_term_presence": presence["authority"] == "report_only"
        and presence["occurrence_counts"]["preferred_wording_not_found"] == 1,
        "publisher_before_date_research_guard": bool(conflicts) and "2004/2005" not in context,
    }


def _v2082_behavior_probes() -> dict[str, bool]:
    """Synthetic boundary/syntax checks, without providers, jobs or file I/O."""
    from tarjomeh.memory.proper_nouns import (
        is_bounded_person_name_target,
        observed_bilingual_target,
    )
    from tarjomeh.quality.integrity import parenthesis_artifacts

    source = "Researchers (Ada North, Emmerich de Vale, G. W. F. Hale) studied the network."
    target = (
        "پژوهشگران (آدا نورث (Ada North)، امریش دو ویل (Emmerich de Vale)، "
        "گ. و. ف. هیل (G. W. F. Hale)) شبکه را بررسی کردند."
    )
    return {
        "explicit_hyphen_name_group_alignment": bool(
            not observed_bilingual_target(
                "اندیشه آنماری ویل (Anne-Marie Vale)", "Anne-Marie Vale", "person",
            )
            and is_bounded_person_name_target("Anne-Marie Vale", "آن-ماری ویل")
            and not is_bounded_person_name_target("Anne-Marie Vale", "آن ماری ویل")
        ),
        "final_heh_ezafe_name_edges": bool(
            not is_bounded_person_name_target("Ada Vail", "گفتهٔ آداویل")
            and not is_bounded_person_name_target("Ada Vail", "گفتۀ آداویل")
        ),
        "source_local_annotation_syntax": bool(
            not parenthesis_artifacts(source, target)
            and parenthesis_artifacts(source, target + ")")
            and parenthesis_artifacts(source, target.replace("(Ada North)", "[(Ada North)]"))
            and parenthesis_artifacts(
                source, target.replace("(Ada North)", "(Ada North) (Ada North)"),
            )
        ),
    }


def _v2081_behavior_probes() -> dict[str, bool]:
    """Pure corrective controls; no config, provider, job or file access."""
    from types import SimpleNamespace

    from tarjomeh.core.pipeline import (
        _candidate_regression_details,
        _partition_unchanged_quality_count,
        quoted_suffix_review_findings,
    )
    from tarjomeh.memory.proper_nouns import ProperNouns, observed_bilingual_target

    source = "They act at a civic institution.\n\nA civic institution meets."
    before = (
        "\u0622\u0646\u0647\u0627 \u0645\u06cc\u200c\u0631\u0648\u0646\u062f.\n\n"
        "\u0646\u0647\u0627\u062f \u0645\u062f\u0646\u06cc."
    )
    after = before.replace(
        "\u0645\u06cc\u200c\u0631\u0648\u0646\u062f",
        "\u06a9\u0646\u0634 \u0645\u06cc\u200c\u06a9\u0646\u0646\u062f",
    )
    issue = {"issue_id": "old", "severity": "major", "confidence": 0.9,
             "category": "terminology", "source_segment_id": "p2:s1",
             "source_quote": "civic institution",
             "current_persian_quote": "\u0646\u0647\u0627\u062f \u0645\u062f\u0646\u06cc"}
    old: list[dict[str, Any]] = []
    regressions = _candidate_regression_details(
        SimpleNamespace(issue_details=[issue]), SimpleNamespace(issue_details=[]), [],
        source_text=source, previous_text=before, candidate_text=after,
        newly_observed_unchanged=old,
    )
    remaining, carried = _partition_unchanged_quality_count(
        [issue], source_text=source, previous_text=before, candidate_text=after,
        r109_unchanged=old,
    )
    name = "Nora-Lynn Vale"
    target = "\u0646\u0648\u0631\u0627-\u0644\u06cc\u0646 \u0648\u06cc\u0644"
    prefix = "\u0686\u06cc\u0632\u06cc\u2014\u0628\u0647 "
    single = "McExample"
    joined = "\u0645\u06a9\u200c\u0627\u06af\u0632\u0627\u0645\u067e\u0644"
    particle = "Ada de Example"
    particle_target = "\u0622\u062f\u0627 \u062f\u0648 \u0627\u06af\u0632\u0627\u0645\u067e\u0644"
    contracted = target.replace("-", "\u200c")
    nouns = ProperNouns()
    nouns.deserialize({"nouns": {particle: particle_target, name: contracted},
                       "categories": {particle: "person", name: "person"},
                       "provenance": {particle: {"origin": "observed_translation"},
                                      name: {"origin": "observed_translation"}}})
    suffix = quoted_suffix_review_findings(
        "First.\n\nSecond.", "\u00ab\u0645\u0641\u0647\u0648\u0645\u00bb \u0647\u0627",
    )
    return {
        "r109_serious_unchanged_count_preserved": (
            not regressions and old == [issue] and not remaining and carried == [issue]
        ),
        "bounded_name_suffix_components": (
            observed_bilingual_target(prefix + target + " (" + name + ")", name, "person") == target
            and observed_bilingual_target(joined + " (" + single + ")", single, "person") == joined
            and observed_bilingual_target(
                "\u0628\u0647 \u0627\u06af\u0632\u0627\u0645\u067e\u0644 (O'Example)",
                "O'Example", "person",
            ) == "\u0627\u06af\u0632\u0627\u0645\u067e\u0644"
        ),
        "conservative_automatic_name_ambiguity": (
            len(nouns.all_nouns()) == 2 and not nouns.inline_eligible_nouns()
            and nouns.context_deferral_reason(particle) == "automatic_particle_name_unverified"
            and nouns.context_deferral_reason(name) == "ambiguous_name_component_alignment"
            and not observed_bilingual_target(
                particle_target + " (" + particle + ")", particle, "person",
            )
            and not observed_bilingual_target(contracted + " (" + name + ")", name, "person")
        ),
        "quoted_suffix_source_scope": bool(suffix)
        and suffix[0]["source_context_scope"] == "provided_source_prefix"
        and suffix[0]["report_only"],
    }


def _v208_behavior_probes() -> dict[str, bool]:
    """Name, attribution and reporting controls; no providers or persistent I/O."""
    from tarjomeh.core.pipeline import (
        _partition_unchanged_quality_count,
        audit_translation_language,
    )
    from tarjomeh.memory.proper_nouns import (
        ProperNouns,
        is_bounded_person_name_target,
        observed_bilingual_target,
    )

    name = "Nora-Lynn Vale"
    clean = "\u0646\u0648\u0631\u0627-\u0644\u06cc\u0646 \u0648\u06cc\u0644"
    bad = "\u062f\u06cc\u06af\u0631\u0627\u0646 \u0646\u06cc\u0633\u062a\u2014" + clean
    nouns = ProperNouns()
    nouns.deserialize({"nouns": {name: bad}, "categories": {name: "person"},
                       "provenance": {name: {"origin": "observed_translation"}}})
    source = "They act.\n\nA civic institution meets."
    before = (
        "\u0622\u0646\u0647\u0627 \u0645\u06cc\u200c\u0631\u0648\u0646\u062f.\n\n"
        "\u0646\u0647\u0627\u062f \u0645\u062f\u0646\u06cc "
        "\u062f\u06cc\u062f\u0627\u0631 \u0645\u06cc\u200c\u06a9\u0646\u062f."
    )
    after = before.replace(
        "\u0645\u06cc\u200c\u0631\u0648\u0646\u062f",
        "\u06a9\u0646\u0634 \u0645\u06cc\u200c\u06a9\u0646\u0646\u062f",
    )
    issue = {"issue_id": "old", "severity": "minor", "confidence": 0.6,
             "category": "terminology", "source_segment_id": "p2:s1",
             "source_quote": "civic institution",
             "current_persian_quote": "\u0646\u0647\u0627\u062f \u0645\u062f\u0646\u06cc"}
    remaining, carried = _partition_unchanged_quality_count(
        [issue], source_text=source, previous_text=before, candidate_text=after,
    )
    blocked, _ = _partition_unchanged_quality_count(
        [issue], source_text=source, previous_text=before, candidate_text=after + " ",
    )
    suffix = audit_translation_language(
        "A quoted concept.", "\u00ab\u0645\u0641\u0647\u0648\u0645\u00bb \u0647\u0627\u06cc",
    )
    return {
        "dash_clause_bounded_person_names": bool(
            observed_bilingual_target(bad + " (" + name + ")", name, "person") == clean
            and not is_bounded_person_name_target(name, bad)
            and is_bounded_person_name_target(name, clean.replace("-", "\u2013"))
            and nouns.all_nouns().get(name) == bad and nouns.is_context_deferred(name)
            and not nouns.inline_eligible_nouns()
        ),
        "extra_final_unchanged_minor_not_counted": (
            not remaining and carried == [issue] and bool(blocked)
        ),
        "quoted_suffix_review_reporting": bool(suffix["quoted_suffix_review"])
        and not suffix["review_required"],
    }


def _v207_behavior_probes() -> dict[str, bool]:
    """General synthetic note and scope checks; no I/O or LLM calls."""
    from tarjomeh.core.pipeline import _partition_unchanged_book_term_reviews
    from tarjomeh.glossary.book_review import check_reviewed_book_terms
    from tarjomeh.quality.integrity import restore_source_note_markers, source_note_cue_inventory

    source = "Consider (Example Author)1 - the claim follows."
    target = "(Example Author)\u00b9 \u0628\u062e\u0648\u0627\u0646\u06cc\u062f \u06f1 \u2014 \u0627\u062f\u0639\u0627."
    fixed, report = restore_source_note_markers(source, target)
    cue = fixed.index("\u00b9")
    source_term = "A civic institution meets another civic institution."
    target_term = "\u0646\u0647\u0627\u062f \u0645\u062f\u0646\u06cc \u0628\u0627 \u0646\u0647\u0627\u062f \u0645\u062f\u0646\u06cc \u062f\u06cc\u062f\u0627\u0631 \u0645\u06cc\u200c\u06a9\u0646\u062f."
    matches = [
        {
            "paragraph_index": 0,
            "source": "civic institution",
            "target": "\u0646\u0647\u0627\u062f \u0645\u062f\u0646\u06cc",
            "source_paragraph_hash": hashlib.sha256(source_term.encode()).hexdigest(),
            "source_occurrence_count": 2,
        }
    ]
    _compliance, reviews = check_reviewed_book_terms(target_term, source_term, {}, matches)
    blocked, carried = _partition_unchanged_book_term_reviews(
        source_term,
        target_term,
        target_term,
        {},
        matches,
        reviews,
    )
    return {
        "word_dash_duplicate_note_is_source_proven": report["repair_count"] == 1
        and "\u06f1" not in fixed,
        "unicode_and_docx_style_count_one_physical_cue": (
            source_note_cue_inventory(source, fixed, styled_spans=[(cue, cue + 1)])["available"]
            == {"1": 1}
        ),
        "ordinary_source_number_is_not_deleted_as_note": restore_source_note_markers(
            source + " It has 1 member.",
            target,
        )[0]
        == target,
        "unchanged_term_review_is_carried_not_cleared": not blocked
        and carried == reviews
        and bool(reviews),
        "changed_term_paragraph_still_blocks": bool(
            _partition_unchanged_book_term_reviews(
                source_term,
                target_term,
                target_term + " ",
                {},
                matches,
                reviews,
            )[0]
        ),
    }


def _v206_behavior_probes() -> dict[str, bool]:
    """Synthetic report checks; no model, file, configuration or database access."""
    from tarjomeh.core.evidence_audit import model_identity_counts

    result = model_identity_counts([
        {"model": "requested-route", "response_model": "", "success": False},
        {"model": "requested-route", "response_model": "returned-model", "success": True},
    ])
    return {
        "missing_returned_model_is_unknown": result["served_models"] == {
            "unknown": 1, "returned-model": 1,
        },
        "requested_route_is_separate": result["requested_models"] == {"requested-route": 2},
        "identity_reporting_keeps_attempt_denominator": result["physical_attempts"] == 2
            and result["failed_attempts"] == 1 and result["failure_rate"] == 0.5,
    }


def _v205_behavior_probes() -> dict[str, bool]:
    """Pure invented controls: no files, configuration, model calls or DB access."""
    from tarjomeh.core.evidence_audit import _bounded_component_export
    from tarjomeh.memory.manager import _style_record_is_authoritative, _style_record_is_prompt_safe
    from tarjomeh.quality.integrity import _sentence_scoped_orphan_dash
    from tarjomeh.quality.structure_audit import announced_count_evidence, audit_payload

    source = "Teams inspect samples - their own and those of others - in laboratories."
    bad = "گروه‌ها نمونه‌های خود—را در آزمایشگاه بررسی می‌کنند."
    good = "گروه‌ها نمونه‌های خود — و نمونه‌های دیگران — را در آزمایشگاه بررسی می‌کنند."
    def digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    def record(target: str) -> dict[str, Any]:
        return {"source_text": source, "text": target, "source_text_hash": digest(source),
                "text_hash": digest(target), "alignment_status": "exact_paragraph",
                "sample_scope": "complete_paragraph", "representative": True, "quality_score": 95,
                "final_scores": dict.fromkeys(
                    ("accuracy", "fluency", "terminology", "register"), 9.5)}
    text = "Exact bounded context."
    key = f"book_context:{digest(text)}"
    entry = {"component": "book_context", "text": text, "sha256": digest(text),
             "chars": len(text), "complete": True}
    exported, invalid = _bounded_component_export({key: entry})
    secret = "token=synthetic-placeholder"
    hidden_key = f"book_context:{digest(secret)}"
    hidden, _ = _bounded_component_export({hidden_key: {
        **entry, "text": secret, "sha256": digest(secret), "chars": len(secret),
    }})
    count = announced_count_evidence("شش رهیافت تحلیلی. (فصل‌های ۲ تا ۴) به رویکرد می‌پردازد.")
    wrong = audit_payload("Six approaches explain the result.", "پنج رهیافت نتیجه را توضیح می‌دهند.")
    return {
        "chapter_ranges_do_not_invent_approach_counts": [item.value for item in count] == [6],
        "real_approach_count_mismatch_still_blocks": any(
            item["classification"] == "translation_structure_mismatch" for item in wrong["findings"]
        ),
        "glued_orphan_has_local_proof": _sentence_scoped_orphan_dash(source, bad) is not None,
        "style_paths_reject_glued_orphan": not _style_record_is_prompt_safe(record(bad))
            and not _style_record_is_authoritative(record(bad)),
        "valid_closed_aside_remains_style_eligible": _style_record_is_prompt_safe(record(good))
            and _style_record_is_authoritative(record(good)),
        "bounded_context_export_is_exact": not invalid and exported["entries"][key]["text"] == text,
        "secret_bearing_context_is_not_exported": hidden_key not in hidden["entries"]
            and hidden["withheld"][hidden_key]["reason"] == "possible_secret_bearing_text",
    }


def _v204_behavior_probes() -> dict[str, bool]:
    """Pure synthetic inputs, with no configuration, network or database I/O."""
    from tarjomeh.context.book_researcher import BookResearcher
    from tarjomeh.core.config import TarjomehConfig
    from tarjomeh.core.pipeline import audit_translation_language
    from tarjomeh.core.prompt_evidence import component_evidence
    from tarjomeh.glossary.book_review import check_reviewed_book_terms, resolve_reviewed_book_terms
    from tarjomeh.memory.manager import _style_record_is_authoritative, _style_record_is_prompt_safe
    from tarjomeh.memory.proper_nouns import is_reusable_terminology_mapping
    from tarjomeh.quality.integrity import parenthesis_artifacts

    source = "A control network connects to a second control network."
    target = "\u0634\u0628\u06a9\u0647\u0654 \u06a9\u0646\u062a\u0631\u0644"
    def digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    metadata = {"structural_roles": ["body"]}
    matches, _ = resolve_reviewed_book_terms(source, metadata, 0, [{
        "source": "control network", "target": target, "status": "approved",
        "scope_mode": "evidence_paragraph", "source_evidence_sha256": digest(source),
    }])
    compliance, reviews = check_reviewed_book_terms(target, source, metadata, matches)
    dash_review = audit_translation_language(
        "One change - its cause is clear. Another - its effect is uncertain.",
        "\u06cc\u06a9 \u062a\u063a\u06cc\u06cc\u0631 \u2014 \u0639\u0644\u062a\u0634 \u0631\u0648\u0634\u0646 \u0627\u0633\u062a. \u0627\u062b\u0631 \u062f\u06cc\u06af\u0631\u06cc \u2014 \u0646\u0627\u0645\u0639\u0644\u0648\u0645 \u2014 \u0627\u0633\u062a.",
    )
    original_source = "Researchers (Ada North) studied the network."
    original_target = "\u067e\u0698\u0648\u0647\u0634\u06af\u0631\u0627\u0646 (\u0622\u062f\u0627 \u0646\u0648\u0631\u062b (Ada North)) \u0634\u0628\u06a9\u0647 \u0631\u0627 \u0628\u0631\u0631\u0633\u06cc \u06a9\u0631\u062f\u0646\u062f."
    anchor = {"authorized": {"Ada North": "\u0622\u062f\u0627 \u0646\u0648\u0631\u062b"}}
    style_source = "This (meta)theoretical account preserves the distinctions."
    style_target = "\u0627\u06cc\u0646 \u062a\u062d\u0644\u06cc\u0644 (\u0641\u0631\u0627) \u0646\u0638\u0631\u06cc \u062a\u0645\u0627\u06cc\u0632\u0647\u0627 \u0631\u0627 \u062d\u0641\u0638 \u0645\u06cc\u200c\u06a9\u0646\u062f."
    record = {"source_text": style_source, "text": style_target,
              "source_text_hash": digest(style_source), "text_hash": digest(style_target),
              "sample_scope": "complete_paragraph", "alignment_status": "exact_paragraph",
              "quality_score": 95, "representative": True,
              "final_scores": dict.fromkeys(("accuracy", "fluency", "terminology", "register"), 9.5)}
    researcher = object.__new__(BookResearcher)
    researcher.config = TarjomehConfig()
    research = researcher._normalise_terms([
        {"source": "control network", "target": target, "source_urls": ["https://example.invalid/book"]}
    ], [{"url": "https://example.invalid/book", "snippet": source}])[0]
    snapshots, refs = component_evidence({"book_context": "Exact context."}, {})
    return {
        "legacy_scope_repeats_are_review_only": bool(
            matches[0]["source_occurrence_count"] == 2 and compliance.total_checked == 0 and reviews
        ),
        "bare_infinitive_is_passage_evidence": bool(
            not is_reusable_terminology_mapping("to reduce uncertainty", target)
            and is_reusable_terminology_mapping("right to information", target)
        ),
        "fallback_and_authority_share_prefix_guard": bool(
            not _style_record_is_prompt_safe(record) and not _style_record_is_authoritative(record)
        ),
        "independent_single_dashes_are_not_a_pair": not dash_review["unbalanced_explanatory_dash_artifacts"],
        "authorized_annotations_preserve_source_parentheses": bool(
            not parenthesis_artifacts(original_source, original_target, anchor_context=anchor)
            and not parenthesis_artifacts(original_source, original_target)
            and parenthesis_artifacts(
                original_source, original_target.replace("(Ada North)", "[(Ada North)]"),
            )
        ),
        "english_attestation_does_not_verify_persian": bool(
            research["english_term_attested"] and not research["persian_option_quote_present"]
            and research["persian_semantic_accuracy"] == "unverified"
        ),
        "prompt_evidence_is_complete_and_deduplicated": bool(
            refs["book_context"]["available"]
            and not component_evidence({"book_context": "Exact context."}, snapshots)[0]
        ),
    }


def _v203_behavior_probes() -> dict[str, bool]:
    """Synthetic policy checks: no LLM, file I/O, DB writes or job changes."""
    from tarjomeh.context.book_researcher import BookResearcher
    from tarjomeh.core.config import TarjomehConfig
    from tarjomeh.core.pipeline import _refinement_span_contract
    from tarjomeh.glossary.book_review import book_term_review_evidence

    config = TarjomehConfig.from_dict(
        {"llm": {"provider": "ollama"}}, credential_source=TarjomehConfig()
    )
    context, conflicts = BookResearcher._reconcile_publication_context(
        "A general book (2015/2016, Example Press) examines institutions.",
        [{"kind": "first_publication", "year": 2016}],
    )
    terms = book_term_review_evidence({"proposals": [{"status": "approved"}]}, [], [])
    return {
        "legacy_job_has_no_extra_allowance": config.translation.extra_final_refine_attempts == 0,
        "fresh_academic_has_one_extra_allowance": (
            TarjomehConfig().translation.extra_final_refine_attempts == 1
        ),
        "year_before_publisher_is_disclosed": bool(conflicts and "2015/2016" not in context),
        "changed_head_requires_original_span": bool(
            not _refinement_span_contract(
                "This theory is clear.", "این رویکرد مبهم است.", "این نظریه روشن است.",
                "p1:s1", "مبهم", "رویکرد مبهم", "نظریه روشن",
            ) and _refinement_span_contract(
                "This theory is clear.", "این رویکرد مبهم است.", "این نظریه روشن است.",
                "p1:s1", "مبهم", "", "نظریه روشن",
            )
        ),
        "approval_is_not_occurrence_or_semantic_verification": (
            terms["approved_proposal_count"] == 1
            and terms["ticked_occurrence_count"] == terms["scoped_paragraph_count"] == 0
            and terms["lexically_present_paragraph_term_count"] == 0
        ),
    }


def _v202_behavior_probes() -> dict[str, bool]:
    """Synthetic in-memory controls; no LLM, network, database or job writes."""
    from tarjomeh.context.book_researcher import BookResearcher
    from tarjomeh.core.pipeline import _previous_context_reference
    from tarjomeh.memory.manager import _is_paratext_style_source
    from tarjomeh.memory.proper_nouns import is_reusable_terminology_mapping
    from tarjomeh.quality.integrity import repair_proven_surface_artifacts

    source = "Earlier prose."
    target = "متن پیشین."

    def sha(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    row = {"chunk_index": 0, "status": "completed", "text": source, "translation": target}
    policy = {"canonical_target_hash": sha(target), "short_term_trust": "trusted"}
    refs = {"short_term_identities": [{
        "entry": 1, "chunk_index": 0, "source_sha256": sha(source),
        "target_sha256": sha(target), "trust": "trusted",
    }]}
    valid_reference = _previous_context_reference(target, 0, row, policy, refs)
    refs["short_term_identities"][0]["trust"] = "advisory_review"
    unsafe_reference = _previous_context_reference(target, 0, row, policy, refs)
    comma_target = "نهادهای محلی،، روابط اجتماعی."
    repaired, edits = repair_proven_surface_artifacts(
        "Local institutions, social relations.", comma_target,
    )
    quoted = "او گفت «نهادهای محلی،، روابط اجتماعی»."
    context, conflicts = BookResearcher._reconcile_publication_context(
        "First published in 2011. It concerns ecology.",
        [{"kind": "first_publication", "year": 2012, "quote": "First published in 2012"}],
    )
    return {
        "funding_paratext_never_teaches_style": (
            _is_paratext_style_source(
                "This book was written with support from a research fellowship."
            )
            and not _is_paratext_style_source(
                "Research fellowships affect scientific institutions."
            )
        ),
        "quantified_fragment_is_not_lexical_authority": (
            not is_reusable_terminology_mapping(
                "record every observed value", "ثبت مقادیر مشاهده‌شده"
            )
            and is_reusable_terminology_mapping("control theory", "نظریهٔ کنترل")
        ),
        "previous_context_reference_requires_matching_trust": (
            bool(valid_reference) and not unsafe_reference
        ),
        "source_edition_conflict_is_disclosed": (
            context == "It concerns ecology." and bool(conflicts)
        ),
        "duplicate_comma_repair_protects_quotes": (
            repaired == comma_target.replace("،،", "،") and bool(edits)
            and repair_proven_surface_artifacts("A quotation.", quoted)[0] == quoted
        ),
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
                "Ada North", "تخصصیِ آدا نورث"
            )
            and not is_bounded_person_name_target("Ben West", "فیلسوف بن وست")
            and is_bounded_person_name_target("Ben West", "ب. وست")
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


def _v20_behavior_probes() -> dict[str, bool]:
    """Pure selected-chapter and bilingual-layout checks; no I/O or LLM calls."""
    from tarjomeh.chunking.chunker import SemanticChunker
    from tarjomeh.core.pipeline import apply_chapter_selection, build_chapter_manifest
    from tarjomeh.core.render_identity import expected_delivered_units
    from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
    from tarjomeh.glossary.book_review import body_paragraph_index
    from tarjomeh.parsers.base import Chapter, Document, Paragraph, Section

    front = Chapter("Copyright", sections=[Section("", 2, [Paragraph("Copyright 2026.")])])
    body = Chapter("Chapter 1", sections=[Section("", 2, [
        Paragraph("The institution mediates the argument.")
    ])])
    document = Document("Example", chapters=[front, body])
    build_chapter_manifest(document)
    selected = apply_chapter_selection(document, [2])
    chunks = SemanticChunker(
        max_tokens=1000, overlap_sentences=0,
        token_counter=lambda text: len(text.split()),
    ).chunk(selected)
    index = body_paragraph_index(selected, chunks)
    rendered = TranslatedDocument(title="Example", author="", paragraphs=[
        TranslatedParagraph(index=0, source_text="A source.", translated_text="متن.")
    ])
    inline = expected_delivered_units(rendered, "inline")
    side_by_side = expected_delivered_units(rendered, "side_by_side")
    return {
        "selected_chapter_body_terms_keep_local_indices": bool(
            len(index) == 1 and index[0]["paragraph_index"] == 0
            and index[0]["chapter_position"] == 2
        ),
        "bilingual_docx_layouts_have_expected_text_units": bool(
            inline and [item["text"] for item in inline] == ["A source.", "متن."]
            and side_by_side and side_by_side[0]["rows"] == [["A source.", "متن."]]
        ),
    }
