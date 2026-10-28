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


def quote_and_numeric_reporting(
    events: list[dict[str, Any]], chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Report recorded quote proofs and current offline grouping, not authority."""
    from tarjomeh.quality.evaluation import _offline_numeric_evidence
    from tarjomeh.quality.grounding import source_quote_span

    sources = {chunk.get("chunk_index"): str(chunk.get("text") or "") for chunk in chunks}
    proofs = []

    def visit(value, event):
        if isinstance(value, list):
            for item in value:
                visit(item, event)
        elif isinstance(value, dict):
            if "source_quote_span" in value:
                source = sources.get(event.get("chunk_index"), "")
                expected = source_quote_span(source, str(value.get("source_quote") or ""))
                proofs.append({
                    "chunk_index": event.get("chunk_index"),
                    "event_type": event.get("event_type"),
                    "timestamp": event.get("timestamp"),
                    "issue_id": value.get("issue_id"),
                    "proof": value["source_quote_span"],
                    "matches_saved_source": expected is not None
                    and expected == value["source_quote_span"],
                })
            for key, item in value.items():
                if key != "source_quote_span":
                    visit(item, event)

    for event in events:
        visit(event.get("payload") or {}, event)
    grouping = []
    for chunk in chunks:
        target = str(chunk.get("translation") or "")
        if not target:
            continue
        evidence = _offline_numeric_evidence(str(chunk.get("text") or ""), target)
        if evidence["grouped_prose_equivalents"]:
            grouping.append({"chunk_index": chunk.get("chunk_index"), "evidence": evidence})
    return {
        "authority": "report_only_no_coverage_routing_memory_or_export_permission",
        "event_scope": "supplied_active_events_no_historical_proof_inference",
        "recorded_quote_span_count": len(proofs),
        "recorded_quote_spans": proofs,
        "historical_absence": "not_recorded_not_a_failure_or_reconstructed_proof",
        "current_offline_grouped_equivalences": grouping,
        "production_numeric_policy_changed": False,
    }


def narrow_release_reporting(
    events: list[dict[str, Any]], chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Expose existing evidence only; missing historical diagnostics are not failures."""
    from tarjomeh.glossary.book_review import book_term_review_evidence

    critiques = []
    local_admissions = []
    for event in events:
        payload = event.get("payload") or {}
        if event.get("event_type") == "critique_completed":
            diagnostic = payload.get("diagnostic_evidence")
            critiques.append({
                "chunk_index": event.get("chunk_index"),
                "iteration": payload.get("iteration"),
                "candidate_target_hash": payload.get("candidate_target_hash"),
                "availability": "recorded" if diagnostic is not None else "unavailable",
                "diagnostic": diagnostic,
            })
        if event.get("event_type") == "refinement_completed":
            for decision in payload.get("issue_decisions", []) or []:
                evidence = {key: decision.get(key) for key in (
                    "coherent_term_admission", "term_admission",
                )}
                local_admissions.append({
                    "chunk_index": event.get("chunk_index"),
                    "issue_id": decision.get("issue_id"),
                    "commit_reason": decision.get("commit_reason"),
                    "availability": "recorded" if any(
                        value is not None for value in evidence.values()
                    ) else "unavailable_or_not_reached",
                    **evidence,
                })
    attempts = [event for event in events if event.get("event_type") == "llm_call_attempt"]
    term_evidence = book_term_review_evidence({}, events, chunks)
    return {
        "authority": "report_only_no_gate_score_trust_or_call_change",
        "event_scope": "supplied_events; do_not_add_review_records_to_physical_attempts",
        "physical_attempt_records": len(attempts),
        "critique_result_records": len(critiques),
        "critique_diagnostics": critiques,
        "local_term_admissions": local_admissions,
        "nested_source_matches": term_evidence["nested_source_matches"],
        "term_authority": "mandatory_only_in_proven_paragraph; optional_elsewhere_has_no_penalty",
        "extra_final_counts": "subset_of_physical_attempts_not_additional",
        "semantic_accuracy": "requires_source_based_human_review",
    }


def original_annotation_reporting(
    audit: dict[str, Any],
    chunks: list[dict[str, Any]],
    identities: dict[str, Any],
) -> dict[str, Any]:
    """Deduplicate display locations without granting source or memory authority."""
    from tarjomeh.memory.proper_nouns import source_term_present

    sources: dict[int, str] = {}
    for chunk in chunks:
        identity = identities.get(str(chunk.get("chunk_index")), {})
        parts = re.split(r"\n\s*\n", str(chunk.get("text") or ""))
        indices, hashes = identity.get("paragraph_indices", []), identity.get("source_hashes", [])
        if len(parts) == len(indices) == len(hashes) and all(
            hashlib.sha256(part.encode()).hexdigest() == digest
            for part, digest in zip(parts, hashes, strict=True)
        ):
            sources.update(zip(indices, parts, strict=True))
    raw = audit.get("unapproved_ungrounded", [])
    locations: dict[tuple[Any, str], dict[str, Any]] = {}
    for record in raw:
        original, index = str(record.get("original", "")), record.get("paragraph_index")
        key = (index, original)
        if key not in locations:
            source = sources.get(index, "")
            candidate = re.sub(r"(?<=[A-Za-z])-(?=[a-z])", "", source)
            uncertain_join = bool(
                source
                and source != candidate
                and not source_term_present(source, original)
                and source_term_present(candidate, original)
            )
            locations[key] = {
                "paragraph_index": index,
                "original": original,
                "pass_record_count": 0,
                "classification": "possible_pdf_word_break_not_verified"
                if uncertain_join
                else "source_original_evidence_unresolved",
                "source_evidence_available": bool(source),
                "disposition": "REVIEW",
            }
        locations[key]["pass_record_count"] += 1
    return {
        "raw_pass_record_count": len(raw),
        "unique_location_count": len(locations),
        "locations": list(locations.values()),
        "authority": "report_only_no_matching_or_retention_permission",
    }


def style_source_evidence(state: dict[str, Any]) -> dict[str, Any]:
    """Distinguish retained style history from current usable evidence."""
    from tarjomeh.memory.manager import (
        _style_record_is_authoritative,
        _style_record_is_prompt_safe,
        _style_record_source_eligibility,
    )

    records = []
    for index, record in enumerate(state.get("style_sample_records", []) or []):
        if not isinstance(record, dict):
            continue
        source = _style_record_source_eligibility(record)
        records.append({
            "record_index": index, "source_chunk_index": record.get("source_chunk_index"),
            "source_paragraph_index": record.get("source_paragraph_index"),
            "stored_active": record.get("active", True),
            "stored_representative": bool(record.get("representative")),
            "target_sha256": hashlib.sha256(str(record.get("text", "")).encode()).hexdigest(),
            "source_eligibility": source,
            "current_prompt_eligible": _style_record_is_prompt_safe(record),
            "current_authoritative": _style_record_is_authoritative(record),
        })
    return {
        "records": records,
        "stored_record_count": len(records),
        "effective_prompt_count": sum(row["current_prompt_eligible"] for row in records),
        "effective_authoritative_count": sum(row["current_authoritative"] for row in records),
        "source_excluded_count": sum(
            row["stored_active"] and not row["source_eligibility"]["eligible"] for row in records
        ),
        "historical_prompt_exposure": "inspect_saved_components_not_current_eligibility",
        "semantic_quality": "requires_source_based_human_review",
        "authority": "report_only_no_saved_state_rewrite",
    }


def checkpoint_classification_evidence(
    state: dict[str, Any], render: dict[str, Any], chunks: list[dict[str, Any]],
    *, original_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose proof/classification only; never readmit history or change trust."""
    from tarjomeh.quality.integrity import object_marker_dash_classification
    from tarjomeh.quality.structure_audit import audit_payload

    rejected = state.get("last_rejected_summary", {}) or {}
    cleanup = original_audit or {}
    deferred_available = "deferred_deduplication" in cleanup
    english = (rejected.get("english_candidate", {}) or {}).get("text", "")
    persian = (rejected.get("persian_candidate", {}) or {}).get("text", "")
    classifications = []
    for chunk in chunks:
        source, target = str(chunk.get("text") or ""), str(chunk.get("translation") or "")
        if not target:
            continue
        result = object_marker_dash_classification(source, target)
        if result["pairs"] or result["findings"]:
            classifications.append({
                "chunk_index": chunk.get("chunk_index"),
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "target_sha256": hashlib.sha256(target.encode()).hexdigest(), **result,
            })
    return {
        "deferred_deduplication": {
            "available": deferred_available,
            "record_count": len(cleanup["deferred_deduplication"])
            if deferred_available else None,
            "records": list(cleanup.get("deferred_deduplication", [])),
            "count_kind": "cleanup_pass_records_not_unique_occurrences",
            "authority": "report_only_no_memory_or_quality_penalty",
        },
        "duplicate_removal_evidence": [
            evidence for change in render.get("render", {}).get("changes", [])
            for evidence in change.get("cleanup_evidence", [])
        ],
        "rejected_summary_current_structure": audit_payload(english, persian)
        if english and persian else None,
        "historical_summary_readmitted": False,
        "dash_classifications": classifications,
        "authority": "report_only_no_memory_or_quality_penalty",
        "semantic_accuracy": "requires_source_based_human_review",
    }


def export_memory_research_evidence(
    state: dict[str, Any], render: dict[str, Any], projection: dict[str, Any],
) -> dict[str, Any]:
    """Keep current effective deferral separate from historical raw records."""
    from tarjomeh.memory.proper_nouns import ProperNouns

    memory = ProperNouns()
    memory.deserialize(state.get("proper_nouns", {}))
    return {
        "deferred_organization_mappings": [
            {"source": source, "target": target, "reason": memory.context_deferral_reason(source),
             "raw_record_retained": True, "prompt_eligible": False}
            for source, target in memory.all_nouns().items()
            if memory.context_deferral_reason(source) == "organization_lexical_start_unproven"
        ],
        "render_identity": render,
        "research_context_projection": projection,
        "summary_validation_scope": "protocol_structure_and_coverage_not_semantic_entailment",
        "semantic_accuracy": "requires_source_based_human_review",
        "authority": "report_only_no_memory_or_quality_penalty",
    }


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
                "before_actionable_count_raw": payload.get("before_actionable_count_raw"),
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
                "count_comparison_policy": payload.get("count_comparison_policy"),
                "before_count_excluded_unchanged_issue_ids": payload.get(
                    "before_count_excluded_unchanged_issue_ids"
                ),
                "before_count_excluded_refiner_veto_issue_ids": payload.get(
                    "before_count_excluded_refiner_veto_issue_ids"
                ),
                "symmetric_count_evidence_available": (
                    payload.get("count_comparison_policy") == "symmetric_before_after_unchanged_scope"
                    and "before_actionable_count_raw" in payload
                ),
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
        "settings_and_token_diagnostics": settings_and_token_diagnostics(events),
    }


def settings_and_token_diagnostics(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose diagnostics without changing billing, job policy or quality."""
    differences = []
    inconsistent = []
    for event in events:
        payload = event.get("payload") or {}
        if event.get("event_type") == "settings_env_difference":
            differences.append({
                "ignored_environment_variables": payload.get("ignored_environment_variables", []),
                "policy": "persisted_job_settings_preserved",
            })
        from tarjomeh.core.llm_client import token_counter_diagnostics

        counters = token_counter_diagnostics(payload) if payload.get("usage_reported") else {}
        if (event.get("event_type") == "llm_call_attempt"
                and counters.get("token_counter_consistency") == "inconsistent"):
            inconsistent.append({
                "chunk_index": event.get("chunk_index"),
                **{key: payload.get(key) for key in (
                    "operation", "attempt", "prompt_tokens", "completion_tokens",
                    "total_tokens",
                )},
                "derived_component_token_sum": counters["derived_component_token_sum"],
            })
    return {"settings_differences": differences, "inconsistent_token_counters": inconsistent,
            "policy": "raw_provider_counters_retained_derived_sum_is_not_billing"}


def fresh_memory_policy_evidence(
    state: dict[str, Any], events: list[dict[str, Any]], chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Separate inactive history, active canonical references and unexercised policy."""
    def digest(value: Any) -> str:
        return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()

    canonical = {row["chunk_index"]: row for row in chunks
                 if row.get("status") in {"completed", "needs_review"} and row.get("translation")}
    def proven(ref: dict[str, Any]) -> bool:
        row = canonical.get(ref.get("chunk_index"))
        return bool(row and ref.get("source_sha256") == digest(row.get("text"))
                    and ref.get("target_sha256") == digest(row.get("translation")))

    pairs = [pair for pair in state.get("past_translations", []) if isinstance(pair, dict)]
    active = [pair for pair in pairs if pair.get("active", True)]
    refs = [{"chunk_index": pair.get("chunk_index"), "source_sha256": digest(pair.get("source")),
             "target_sha256": digest(pair.get("translation"))} for pair in active]
    pending = state.get("summary_pending", [])
    coverage = state.get("summary_coverage", [])
    return {
        "policy_version": state.get("quality_policy_version", 0),
        "legacy_not_migrated": state.get("quality_policy_version", 0) == 0,
        "layer3_stored_versions": len(pairs),
        "layer3_active_versions": len(active),
        "layer3_inactive_versions": len(pairs) - len(active),
        "active_pair_canonical_identity": [
            {**ref, "canonical_hashes_match": proven(ref)} for ref in refs
        ],
        "duplicate_active_indices": [
            index
            for index, count in Counter(ref["chunk_index"] for ref in refs).items()
            if index is not None and count > 1
        ],
        "pending_summary_coverage": [
            {**ref, "canonical_hashes_match": proven(ref)} for ref in pending
        ],
        "accepted_summary_coverage": [
            {**ref, "canonical_hashes_match": proven(ref)} for ref in coverage
        ],
        "historical_summary_versions": len(state.get("summary_history", [])),
        "summary_version_evidence": state.get("summary_history", []),
        "mapping_provenance_references": state.get("mapping_evidence", {}),
        "inactive_layer3_history": [pair for pair in pairs if not pair.get("active", True)],
        "inactive_style_history": [
            record
            for record in state.get("style_sample_records", [])
            if record.get("active") is False
        ],
        "inactive_style_records": sum(
            record.get("active") is False for record in state.get("style_sample_records", [])
        ),
        "curated_revision_events": [
            event
            for event in events
            if event.get("event_type")
            in {
                "curated_revision_reconciled",
                "curated_revision_conflict",
            }
        ],
        "manual_retranslation_results": [
            event for event in events if event.get("event_type") == "manual_retranslation_result"
        ],
        "retranslation_context_events": [
            event
            for event in events
            if str(event.get("event_type", "")).endswith("retranslation_context_isolation")
        ],
        "settings_and_token_diagnostics": settings_and_token_diagnostics(events),
        "authority": "report_only_no_semantic_certification_or_legacy_coverage_inference",
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
    from tarjomeh.memory.manager import (
        _style_record_is_authoritative,
        _style_record_is_prompt_safe,
        _style_record_source_eligibility,
    )
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
            "source_eligibility": _style_record_source_eligibility(item),
            "stored_active": item.get("active", True),
            "historical_prompt_exposure": "consult_saved_prompt_components_not_current_eligibility",
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
