"""Report-only accounting; these helpers grant no translation or memory authority."""

from collections import Counter
import hashlib
import json
import re
from typing import Any


def extra_final_refinement_evidence(
    reservation: dict[str, Any], events: list[dict[str, Any]], configured: int,
) -> dict[str, Any]:
    entries = reservation.get("entries", {})
    violations = []
    if not isinstance(entries, dict):
        violations.append("malformed_extra_refinement_accounting")
        entries = {}
    for index, entry in entries.items():
        if not isinstance(entry, dict) or entry.get("consumed") != 1:
            violations.append(f"invalid_consumed_budget:{index}")
    unfinished = [index for index, entry in entries.items()
                  if isinstance(entry, dict) and entry.get("state") != "finished"]
    starts = {}
    for position, event in enumerate(events):
        if event.get("event_type") == "chunk_started":
            starts[event.get("chunk_index")] = position

    def totals(attempts: list[dict[str, Any]]) -> dict[str, Any]:
        failures = sum(item.get("success") is False for item in attempts)
        return {
            "physical_attempts": len(attempts), "failures": failures,
            "completion_tokens": sum(int(item.get("completion_tokens") or 0) for item in attempts),
            "llm_seconds": round(sum(float(item.get("duration_seconds") or 0)
                                     for item in attempts), 3),
            "served_models": dict(Counter(item.get("response_model") or "unknown"
                                          for item in attempts)),
            "operations": dict(Counter(item.get("operation") or "unknown" for item in attempts)),
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
        "configured_extra_attempts": configured, "consumed_chunk_count": len(entries),
        "reservations": entries, "budget_violations": violations,
        "unfinished_reservations": unfinished,
        "status": "FAIL" if violations else "REVIEW" if unfinished else "PASS",
        "active": totals(active), "lifetime": totals(lifetime),
        "outcomes": [event for event in events if event.get("event_type") in {
            "extra_final_refinement_reserved", "extra_final_refinement_completed",
            "extra_final_refinement_skipped", "extra_final_candidate_review",
        }],
        "policy": "one extra refiner per chunk; recovery attempts included in physical cost",
    }


def quality_safeguard_evidence(
    research: dict[str, Any], state: dict[str, Any], review: dict[str, Any],
    chunks: list[dict[str, Any]], events: list[dict[str, Any]],
    snapshots: dict[str, Any],
) -> dict[str, Any]:
    """Re-evaluate saved evidence without rewriting historical quality decisions."""
    from tarjomeh.core.pipeline import audit_translation_language
    from tarjomeh.glossary.book_review import (
        check_reviewed_book_terms, persian_option_defect, resolve_reviewed_book_terms,
    )
    from tarjomeh.memory.manager import _style_record_is_authoritative, _style_record_is_prompt_safe
    from tarjomeh.memory.proper_nouns import ProperNouns

    nouns = ProperNouns()
    nouns.deserialize(state.get("proper_nouns", {}) or {})
    decisions = [item for item in review.get("proposals", []) if isinstance(item, dict)]
    language = []
    occurrence_reviews = []
    for chunk in chunks:
        source, target = chunk.get("text") or "", chunk.get("translation") or ""
        if not source or not target or chunk.get("status") not in {"completed", "needs_review"}:
            continue
        metadata = chunk.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        context = {"authorized": {
            name: value for name, value in nouns.inline_eligible_nouns().items()
            if nouns.applies_to_source(name, source)
        }, "aliases": nouns.inline_eligible_aliases()}
        roles = set(metadata.get("structural_roles", []) or [])
        role = next(iter(roles)) if len(roles) == 1 else "mixed"
        report = audit_translation_language(
            source, target, structural_role=role,
            chapter_title=str(metadata.get("chapter_title", "")), anchor_context=context,
        )
        language.append({
            "chunk_index": chunk["chunk_index"],
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
            "current_policy_findings": report,
            "authority": "report_only_not_a_retroactive_memory_decision",
            "scope": "chunk_report_only_no_new_paragraph_alignment",
        })
        matches, scope_reviews = resolve_reviewed_book_terms(
            source, metadata, int(metadata.get("chapter_position", 0) or 0), decisions,
        )
        compliance, compliance_reviews = check_reviewed_book_terms(target, source, metadata, matches)
        occurrence_reviews.append({
            "chunk_index": chunk["chunk_index"], "matches": matches,
            "scope_review": scope_reviews, "compliance_review": compliance_reviews,
            "lexically_checked": compliance.total_checked,
            "violations": len(compliance.violations), "semantic_accuracy": "unverified",
        })
    style = [{
        "source_sha256": item.get("source_text_hash"), "target_sha256": item.get("text_hash"),
        "stored_representative": bool(item.get("representative")),
        "current_prompt_eligible": _style_record_is_prompt_safe(item),
        "current_authoritative": _style_record_is_authoritative(item),
    } for item in state.get("style_sample_records", []) if isinstance(item, dict)]
    evidence_entries = snapshots.get("entries", {}) or {}
    if not isinstance(evidence_entries, dict):
        evidence_entries = {}
    invalid_snapshots = []
    for key, entry in evidence_entries.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
            invalid_snapshots.append(key)
            continue
        digest = hashlib.sha256(entry["text"].encode("utf-8")).hexdigest()
        if (entry.get("sha256") != digest or entry.get("chars") != len(entry["text"])
                or entry.get("complete") is not True or key != f"{entry.get('component')}:{digest}"):
            invalid_snapshots.append(key)
    missing_context = []
    references = []
    for event in events:
        if event.get("event_type") != "translation_prompt_composition":
            continue
        refs = (event.get("payload") or {}).get("component_evidence", {})
        references.append({"chunk_index": event.get("chunk_index"), "components": refs})
        if not refs:
            missing_context.append(event.get("chunk_index"))
        for name, ref in refs.items():
            if ref.get("available"):
                entry = evidence_entries.get(ref.get("artifact_entry"), {})
                if entry.get("sha256") != ref.get("sha256") or entry.get("component") != name:
                    invalid_snapshots.append(str(ref.get("artifact_entry")))
    research_quotes = []
    for term in research.get("terms", []):
        if not isinstance(term, dict):
            continue
        target = str(term.get("target") or "")
        excerpts = term.get("term_supporting_excerpts", []) or []
        quoted = bool(
            target and re.search(r"[\u0621-\u06ff]", target) and not persian_option_defect(target)
            and any(target in str(item.get("snippet", "")) for item in excerpts)
        )
        research_quotes.append({
            "source": term.get("source"),
            "english_term_attested_recorded": bool(term.get("term_supported")),
            "persian_option_quote_present_rechecked": quoted,
            "persian_semantic_accuracy": "unverified",
        })
    return {
        "current_language_evaluation": language, "current_occurrence_evaluation": occurrence_reviews,
        "current_style_eligibility": style, "research_evidence_kinds": research_quotes,
        "prompt_component_references": references,
        "prompt_component_entry_count": len(evidence_entries),
        "invalid_prompt_component_evidence": sorted(set(invalid_snapshots)),
        "historical_context_unavailable_chunks": sorted(set(missing_context)),
        "production_prompts_changed": False, "focused_attachment_reviewer": "OFF",
        "extra_final_refinement_budget_changed": False,
        "semantic_accuracy": "requires_source_based_human_review",
    }
