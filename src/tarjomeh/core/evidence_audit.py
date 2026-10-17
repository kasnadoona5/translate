"""Report-only accounting; these helpers grant no translation or memory authority."""

import hashlib
import json
import re
from collections import Counter
from typing import Any

from tarjomeh.core.prompt_evidence import CONTEXT_COMPONENTS

_POSSIBLE_SECRET_RE = re.compile(
    r"authorization\s*:|\bbearer\s+\S+|"
    r"\b(?:api[_-]?key|token|password|secret)\s*[=:]\s*\S+|"
    r"https?://[^\s/]+:[^\s/@]+@",
    re.IGNORECASE,
)


def targeted_quality_evidence(
    proper: dict[str, Any], events: list[dict[str, Any]], chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Expose effective deferral and current report-only checks without authority."""
    from tarjomeh.core.pipeline import quoted_suffix_review_findings
    from tarjomeh.memory.proper_nouns import ProperNouns, requires_bounded_name_target
    from tarjomeh.quality.integrity import _source_annotation_spans

    memory = ProperNouns()
    memory.deserialize(proper)
    deferred = [
        {"source": source, "target": target,
         "origin": memory.provenance_for(source).get("origin", "legacy"),
         "reason": memory.context_deferral_reason(source), "active": False}
        for source, target in memory.all_nouns().items()
        if memory.is_context_deferred(source)
        and requires_bounded_name_target(memory.category_for(source), source)
    ]
    counts = []
    for event in events:
        if event.get("event_type") not in {
            "extra_final_refinement_completed",
            "exact_final_quality_repair",
        }:
            continue
        payload = event.get("payload", {}) or {}
        counts.append(
            {
                "stage": event.get("event_type"),
                "chunk_index": event.get("chunk_index"),
                "timestamp": event.get("timestamp"),
                "candidate_target_hash": payload.get("candidate_target_hash"),
                "accepted": payload.get("accepted"),
                "before_actionable_count": payload.get("before_actionable_count"),
                "after_actionable_count": payload.get("after_actionable_count"),
                "after_actionable_count_raw": payload.get("after_actionable_count_raw"),
                "count_excluded_unchanged_issue_ids": payload.get(
                    "count_excluded_unchanged_issue_ids"
                ),
                "count_excluded_refiner_veto_issue_ids": payload.get(
                    "count_excluded_refiner_veto_issue_ids"
                ),
                "count_exclusion_policy": payload.get("count_exclusion_policy"),
                "count_evidence_available": "after_actionable_count_raw" in payload,
            }
        )
    suffixes = []
    annotation_syntax = []
    for chunk in chunks:
        source_text = str(chunk.get("text") or "")
        target_text = str(chunk.get("translation") or "")
        spans = _source_annotation_spans(source_text, target_text)
        if spans:
            annotation_syntax.append({
                "chunk_index": chunk.get("chunk_index"),
                "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
                "target_sha256": hashlib.sha256(target_text.encode("utf-8")).hexdigest(),
                "spans": [{"start": start, "end": end, "text": target_text[start:end]}
                          for start, end in spans],
                "proof_scope": "unique_ordered_current_source_parenthetical_entities",
                "authority": "syntax_only_no_mapping_or_render_approval",
            })
        source = re.split(r"\n\s*\n", str(chunk.get("text") or ""))
        target = re.split(r"\n\s*\n", str(chunk.get("translation") or ""))
        aligned = len(source) == len(target)
        for index, paragraph in enumerate(target):
            for finding in quoted_suffix_review_findings(
                source[index] if aligned else str(chunk.get("text") or ""), paragraph,
                source_context_scope=(
                    "unverified_same_index_paragraph_prefix" if aligned else "chunk_source_prefix"
                ),
            ):
                suffixes.append({**finding, "chunk_index": chunk.get("chunk_index"),
                                 "local_paragraph_index": index,
                                 "paragraph_alignment_verified": False,
                                 "paragraph_count_matches": aligned})
    return {
        "version": 2,
        "policy": "report_only_no_authority_or_text_change",
        "deferred_name_mappings": deferred,
        "extra_final_counts": [
            item for item in counts if item["stage"] == "extra_final_refinement_completed"
        ],
        "final_repair_counts": counts,
        "canonical_quote_rebindings": [
            {
                "chunk_index": event.get("chunk_index"),
                "candidate_target_hash": (event.get("payload") or {}).get("candidate_target_hash"),
                "unmatched_issue_ids": (event.get("payload") or {}).get(
                    "canonical_quote_unmatched_issue_ids"
                ),
                "issues": [
                    {"issue_id": item.get("issue_id"), "binding": item["canonical_quote_binding"]}
                    for item in (event.get("payload") or {}).get("issue_details", []) or []
                    if isinstance(item, dict) and item.get("canonical_quote_binding")
                ],
            }
            for event in events
            if event.get("event_type") == "critique_completed"
            and "canonical_quote_unmatched_issue_ids" in (event.get("payload") or {})
        ],
        "deferred_terminology_mappings": [
            {
                "source": source,
                "target": target,
                "reason": memory.context_deferral_reason(source),
                "active": False,
            }
            for source, target in memory.all_nouns().items()
            if memory.is_context_deferred(source)
            and not requires_bounded_name_target(memory.category_for(source), source)
        ],
        "accepted_terminology_reconciliation": [
            {"chunk_index": event.get("chunk_index"), **(event.get("payload") or {})}
            for event in events
            if event.get("event_type") == "accepted_terminology_reconciled"
        ],
        "readability_call_accounting": readability_call_accounting(events),
        "quoted_suffix_reports": suffixes,
        "source_annotation_syntax": annotation_syntax,
    }


def readability_call_accounting(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Count physical attempts separately from the two existing logical routes."""
    attempts = [event.get("payload") or {} for event in events
                if event.get("event_type") == "llm_call_attempt"
                and (event.get("payload") or {}).get("operation") == "persian_readability_review"]
    ordinary = sum(event.get("event_type") == "persian_readability_review" for event in events)
    rollback = sum(event.get("event_type") == "post_rollback_final_validation"
                   and bool((event.get("payload") or {}).get("readability_review_attempted"))
                   for event in events)
    return {"physical_attempts": len(attempts), "ordinary_result_batches": ordinary,
            "post_rollback_logical_attempts": rollback,
            "failed_physical_attempts": sum(item.get("success") is False for item in attempts),
            "policy": (
                "logical route counts are not transport retries; "
                "physical attempts retain their own denominator"
            )}


def model_identity_counts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep requested routes separate from model identities actually returned."""
    served: Counter[str] = Counter()
    requested: Counter[str] = Counter()
    successful: Counter[str] = Counter()
    failed = 0
    for attempt in attempts:
        returned = str(attempt.get("response_model") or "").strip() or "unknown"
        route = str(attempt.get("model") or "").strip() or "unknown"
        served[returned] += 1
        requested[route] += 1
        failed += attempt.get("success") is False
        if attempt.get("success") is True:
            successful[returned] += 1
    return {
        "physical_attempts": len(attempts),
        "failed_attempts": failed,
        "failure_rate": failed / len(attempts) if attempts else None,
        "served_models": dict(served),
        "successful_served_models": dict(successful),
        "requested_models": dict(requested),
        "identity_rule": "returned_response_model_only_missing_is_unknown",
    }


def _bounded_component_export(entries: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    exported: dict[str, Any] = {}
    withheld: dict[str, Any] = {}
    invalid: list[str] = []
    if len(entries) > 128:
        invalid.append("component_entry_storage_bound_exceeded")
    for position, (key, entry) in enumerate(entries.items()):
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
            invalid.append(str(key))
            withheld[str(key)] = {"reason": "malformed_component_entry"}
            continue
        text = entry["text"]
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if (
            entry.get("component") not in CONTEXT_COMPONENTS
            or len(text) > 12000
            or entry.get("sha256") != digest
            or type(entry.get("chars")) is not int
            or entry["chars"] != len(text)
            or entry.get("complete") is not True
            or key != f"{entry.get('component')}:{digest}"
        ):
            invalid.append(str(key))
            withheld[str(key)] = {"reason": "invalid_component_contract"}
        elif position >= 128:
            withheld[key] = {"reason": "component_entry_storage_bound"}
        elif _POSSIBLE_SECRET_RE.search(text):
            withheld[key] = {
                "reason": "possible_secret_bearing_text",
                "sha256": digest,
                "chars": len(text),
                "component": entry["component"],
            }
        else:
            exported[key] = {
                name: entry[name]
                for name in (
                    "component",
                    "text",
                    "sha256",
                    "chars",
                    "complete",
                )
            }
    return {"version": 1, "entries": exported, "withheld": withheld}, invalid


def extra_final_refinement_evidence(
    reservation: dict[str, Any],
    events: list[dict[str, Any]],
    configured: int,
) -> dict[str, Any]:
    entries = reservation.get("entries", {})
    violations = []
    if not isinstance(entries, dict):
        violations.append("malformed_extra_refinement_accounting")
        entries = {}
    for index, entry in entries.items():
        if not isinstance(entry, dict) or entry.get("consumed") != 1:
            violations.append(f"invalid_consumed_budget:{index}")
    unfinished = [
        index
        for index, entry in entries.items()
        if isinstance(entry, dict) and entry.get("state") != "finished"
    ]
    starts = {}
    for position, event in enumerate(events):
        if event.get("event_type") == "chunk_started":
            starts[event.get("chunk_index")] = position

    def totals(attempts: list[dict[str, Any]]) -> dict[str, Any]:
        failures = sum(item.get("success") is False for item in attempts)
        return {
            "physical_attempts": len(attempts),
            "failures": failures,
            "completion_tokens": sum(int(item.get("completion_tokens") or 0) for item in attempts),
            "llm_seconds": round(
                sum(float(item.get("duration_seconds") or 0) for item in attempts), 3
            ),
            "served_models": dict(
                Counter(item.get("response_model") or "unknown" for item in attempts)
            ),
            "operations": dict(Counter(item.get("operation") or "unknown" for item in attempts)),
            "model_identity_accounting": model_identity_counts(attempts),
        }

    lifetime = []
    active = []
    for position, event in enumerate(events):
        payload = event.get("payload") or {}
        if event.get("event_type") != "llm_call_attempt" or (
            payload.get("quality_attempt_context", {}).get("stage") != "extra_final_refinement"
        ):
            continue
        lifetime.append(payload)
        if position >= starts.get(event.get("chunk_index"), 0):
            active.append(payload)
    return {
        "configured_extra_attempts": configured,
        "consumed_chunk_count": len(entries),
        "reservations": entries,
        "budget_violations": violations,
        "unfinished_reservations": unfinished,
        "status": "FAIL" if violations else "REVIEW" if unfinished else "PASS",
        "active": totals(active),
        "lifetime": totals(lifetime),
        "outcomes": [
            event
            for event in events
            if event.get("event_type")
            in {
                "extra_final_refinement_reserved",
                "extra_final_refinement_completed",
                "extra_final_refinement_skipped",
                "extra_final_candidate_review",
            }
        ],
        "policy": "one extra refiner per chunk; recovery attempts included in physical cost",
    }


def quality_safeguard_evidence(
    research: dict[str, Any],
    state: dict[str, Any],
    review: dict[str, Any],
    chunks: list[dict[str, Any]],
    events: list[dict[str, Any]],
    snapshots: dict[str, Any],
    extraction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-evaluate saved evidence without rewriting historical quality decisions."""
    from tarjomeh.core.pipeline import audit_translation_language
    from tarjomeh.glossary.book_review import (
        check_reviewed_book_terms,
        persian_option_defect,
        resolve_reviewed_book_terms,
    )
    from tarjomeh.memory.manager import _style_record_is_authoritative, _style_record_is_prompt_safe
    from tarjomeh.memory.proper_nouns import ProperNouns
    from tarjomeh.quality.integrity import source_note_cue_inventory

    nouns = ProperNouns()
    nouns.deserialize(state.get("proper_nouns", {}) or {})
    decisions = [item for item in review.get("proposals", []) if isinstance(item, dict)]
    language = []
    occurrence_reviews = []
    note_cues = []
    for chunk in chunks:
        source, target = chunk.get("text") or "", chunk.get("translation") or ""
        if not source or not target or chunk.get("status") not in {"completed", "needs_review"}:
            continue
        metadata = chunk.get("metadata") or {}
        note_cues.append({
            "chunk_index": chunk["chunk_index"],
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
            **source_note_cue_inventory(source, target),
            "authority": "current_policy_evaluation_not_historical_rewrite",
        })
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        context = {
            "authorized": {
                name: value
                for name, value in nouns.inline_eligible_nouns().items()
                if nouns.applies_to_source(name, source)
            },
            "aliases": nouns.inline_eligible_aliases(),
        }
        roles = set(metadata.get("structural_roles", []) or [])
        role = next(iter(roles)) if len(roles) == 1 else "mixed"
        report = audit_translation_language(
            source,
            target,
            structural_role=role,
            chapter_title=str(metadata.get("chapter_title", "")),
            anchor_context=context,
        )
        language.append(
            {
                "chunk_index": chunk["chunk_index"],
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
                "current_policy_findings": report,
                "authority": "report_only_not_a_retroactive_memory_decision",
                "scope": "chunk_report_only_no_new_paragraph_alignment",
            }
        )
        matches, scope_reviews = resolve_reviewed_book_terms(
            source,
            metadata,
            int(metadata.get("chapter_position", 0) or 0),
            decisions,
        )
        compliance, compliance_reviews = check_reviewed_book_terms(
            target, source, metadata, matches
        )
        occurrence_reviews.append(
            {
                "chunk_index": chunk["chunk_index"],
                "matches": matches,
                "scope_review": scope_reviews,
                "compliance_review": compliance_reviews,
                "lexically_checked": compliance.total_checked,
                "violations": len(compliance.violations),
                "semantic_accuracy": "unverified",
            }
        )
    style = [
        {
            "source_sha256": item.get("source_text_hash"),
            "target_sha256": item.get("text_hash"),
            "stored_representative": bool(item.get("representative")),
            "current_prompt_eligible": _style_record_is_prompt_safe(item),
            "current_authoritative": _style_record_is_authoritative(item),
            "alignment_evidence": "mechanical_source_target_pair_not_semantic_proof",
            "quality_evidence": "stored_model_scores_not_human_verification",
            "stored_final_scores": item.get("final_scores"),
            "independent_semantic_adjudication": "not_recorded",
        }
        for item in state.get("style_sample_records", [])
        if isinstance(item, dict)
    ]
    evidence_entries = snapshots.get("entries", {})
    malformed_entries = not isinstance(evidence_entries, dict)
    if not isinstance(evidence_entries, dict):
        evidence_entries = {}
    component_export, invalid_snapshots = _bounded_component_export(evidence_entries)
    if malformed_entries:
        invalid_snapshots.append("malformed_component_entries")
    missing_context = []
    references = []
    for event in events:
        if event.get("event_type") != "translation_prompt_composition":
            continue
        refs = (event.get("payload") or {}).get("component_evidence", {})
        safe_refs = {}
        if not isinstance(refs, dict):
            invalid_snapshots.append("malformed_component_references")
            refs = {}
        if not refs:
            missing_context.append(event.get("chunk_index"))
        for name, ref in refs.items():
            if not isinstance(ref, dict):
                invalid_snapshots.append(f"malformed_component_reference:{name}")
                continue
            safe_refs[name] = {
                key: ref[key]
                for key in (
                    "sha256",
                    "chars",
                    "available",
                    "reason",
                    "artifact_entry",
                )
                if key in ref
            }
            if ref.get("available"):
                entry_key = ref.get("artifact_entry")
                entry = evidence_entries.get(entry_key, {}) if isinstance(entry_key, str) else {}
                if (
                    not isinstance(entry, dict)
                    or entry.get("sha256") != ref.get("sha256")
                    or entry.get("component") != name
                    or type(ref.get("chars")) is not int
                    or entry.get("chars") != ref.get("chars")
                ):
                    invalid_snapshots.append(str(entry_key))
        references.append({"chunk_index": event.get("chunk_index"), "components": safe_refs})
    research_quotes = []
    for term in research.get("terms", []):
        if not isinstance(term, dict):
            continue
        target = str(term.get("target") or "")
        excerpts = term.get("term_supporting_excerpts", []) or []
        quoted = bool(
            target
            and re.search(r"[\u0621-\u06ff]", target)
            and not persian_option_defect(target)
            and any(target in str(item.get("snippet", "")) for item in excerpts)
        )
        research_quotes.append(
            {
                "source": term.get("source"),
                "english_term_attested_recorded": bool(term.get("term_supported")),
                "persian_option_quote_present_rechecked": quoted,
                "persian_semantic_accuracy": "unverified",
            }
        )
    extraction = extraction or {}
    sample = extraction.get("text")
    sample_available = bool(
        isinstance(sample, str)
        and sample
        and extraction.get("sha256") == hashlib.sha256(sample.encode("utf-8")).hexdigest()
    )
    invalid_extraction = bool(isinstance(sample, str) and sample and not sample_available)
    coverage = []
    seen = set()
    for item in [*(research.get("terms", []) or []), *decisions]:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        if not source or source.casefold() in seen:
            continue
        seen.add(source.casefold())
        if len(coverage) >= 200:
            continue
        coverage.append(
            {
                "source": source,
                "lexically_present_in_saved_sample": bool(
                    sample_available
                    and re.search(r"(?<!\w)" + re.escape(source) + r"(?!\w)", sample, re.IGNORECASE)
                )
                if sample_available
                else None,
                "persian_option_available": bool(
                    re.search(r"[\u0621-\u06ff]", str(item.get("target") or ""))
                    and not persian_option_defect(str(item.get("target") or ""))
                ),
                "approval_evidence": "consult_separate_review_records_and_occurrence_scopes",
                "english_support_recorded": bool(item.get("term_supported")),
                "persian_semantic_accuracy": "unverified",
                "authority": "coverage_report_only_not_sense_or_approval_evidence",
            }
        )
    return {
        "current_language_evaluation": language,
        "current_occurrence_evaluation": occurrence_reviews,
        "current_note_cue_evaluation": note_cues,
        "current_style_eligibility": style,
        "research_evidence_kinds": research_quotes,
        "prompt_component_references": references,
        "prompt_component_entry_count": len(evidence_entries),
        "prompt_component_evidence": component_export,
        "term_sample_coverage": {
            "sample_available": sample_available,
            "sample_sha256": extraction.get("sha256") if sample_available else None,
            "candidate_count": len(seen),
            "reported_count": len(coverage),
            "sample_unavailable_reason": None
            if sample_available
            else "missing_or_invalid_saved_sample",
            "matching": "case_insensitive_exact_source_expression_not_family_or_sense_matching",
            "entries": coverage,
        },
        "invalid_auto_extraction_evidence": invalid_extraction,
        "invalid_prompt_component_evidence": sorted(set(invalid_snapshots)),
        "historical_context_unavailable_chunks": sorted(set(missing_context)),
        "production_prompts_changed": False,
        "focused_attachment_reviewer": "OFF",
        "extra_final_refinement_budget_changed": False,
        "semantic_accuracy": "requires_source_based_human_review",
        "evidence_labels_change_authority": False,
        "memory_evidence_semantics": {
            "reliable_pair": "existing_policy_admission_not_independent_semantic_verification",
            "summary": "advisory_context_not_source_authority",
            "term_approval": "approved_meaning_and_proven_occurrence_scope_not_global_replacement",
        },
    }
