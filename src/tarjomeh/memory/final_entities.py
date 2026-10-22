"""Hash-bound final entity observations; no translation or quality authority."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from tarjomeh.memory.proper_nouns import (
    ProperNouns,
    has_exact_observed_anchor,
    is_automatic_entity_category,
    is_safe_automatic_entity_mapping,
    observed_bilingual_target,
    source_term_pattern,
    source_term_present,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_entity_target(value: str) -> str:
    value = re.sub(r"[\u064b-\u065f\u0670]", "", value or "")
    return re.sub(r"[\s\u200c\u0640\-\u2010-\u2015]+", " ", value).strip().casefold()


def first_occurrence_evidence(
    source: str, translation: str, english: str, target: str
) -> dict[str, Any]:
    """Separate actual first-use evidence from the stored target's spelling."""
    grounded = source_term_present(source, english)
    needle = normalized_entity_target(target)
    target_present = bool(
        needle
        and re.search(
            rf"(?<!\w){re.escape(needle)}(?!\w)",
            normalized_entity_target(translation),
        )
    )
    anchor_present = grounded and has_exact_observed_anchor(translation, english)
    return {
        "source_present": grounded,
        "stored_target_present": grounded and target_present,
        "source_bound_original_present": anchor_present,
        "first_occurrence_present": grounded and (target_present or anchor_present),
        "semantic_accuracy": "unverified",
    }


def _paragraphs(
    source: str, target: str, identity: dict[str, Any]
) -> tuple[list[str], list[str]] | None:
    if identity.get("reconstructed") is not False or identity.get("canonical_target_hash") != _sha(
        target
    ):
        return None
    sources = re.split(r"\n\s*\n", source)
    offsets = identity.get("target_offsets", [])
    indices = identity.get("paragraph_indices", [])
    if not all(
        isinstance(identity.get(key), list)
        for key in (
            "target_offsets",
            "paragraph_indices",
            "source_hashes",
            "target_hashes",
        )
    ) or not all(type(value) is int for value in indices):
        return None
    if not (
        len(sources)
        == len(offsets)
        == len(indices)
        == len(identity.get("source_hashes", []))
        == len(identity.get("target_hashes", []))
    ):
        return None
    targets = []
    for index, span in enumerate(offsets):
        if not (
            isinstance(span, (list, tuple))
            and len(span) == 2
            and all(type(value) is int for value in span)
            and 0 <= span[0] < span[1] <= len(target)
        ):
            return None
        part = target[span[0] : span[1]]
        if (
            _sha(sources[index]) != identity["source_hashes"][index]
            or _sha(part) != identity["target_hashes"][index]
        ):
            return None
        targets.append(part)
    if "\n\n".join(targets) != target or len(set(indices)) != len(indices):
        return None
    return sources, targets


def _accepted_edit_proofs(events: list[dict[str, Any]], target: str) -> list[dict[str, Any]]:
    last_start = max(
        (i for i, e in enumerate(events) if e.get("event_type") == "chunk_started"), default=0
    )
    proofs = []
    for event in events[last_start:]:
        payload = event.get("payload", {}) or {}
        if event.get("event_type") not in {
            "extra_final_refinement_completed",
            "exact_final_quality_repair",
            "refinement_completed",
        }:
            continue
        if not (
            payload.get("accepted") is True
            and payload.get("candidate_target_hash") == _sha(target)
            and payload.get("retained_target_hash", _sha(target)) == _sha(target)
        ):
            continue
        decisions = payload.get("decisions", payload.get("issue_decisions", []))
        if not isinstance(decisions, list):
            continue
        committed = [
            d
            for d in decisions
            if isinstance(d, dict)
            and d.get("decision") == "accepted"
            and str(d.get("commit_status", "")).startswith("committed")
        ]
        before = target
        valid = bool(committed)
        for decision in reversed(committed):
            original, result = decision.get("original_span"), decision.get("resulting_span")
            if not (
                isinstance(original, str)
                and isinstance(result, str)
                and original
                and result
                and "\n" not in original
                and "\n" not in result
                and before.count(result) == 1
            ):
                valid = False
                break
            before = before.replace(result, original, 1)
        if valid and _sha(before) == payload.get("baseline_target_hash"):
            proofs.append(
                {"before": before, "decisions": committed, "baseline_target_hash": _sha(before)}
            )
    return proofs


def reconcile_final_entities(
    nouns: ProperNouns,
    source: str,
    target: str,
    *,
    chunk_index: int,
    paragraph_identity: dict[str, Any],
    quality: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reconcile only contradicted automatic entities, never an entire chunk."""
    report: dict[str, Any] = {
        "source_sha256": _sha(source),
        "candidate_target_hash": _sha(target),
        "chunk_index": chunk_index,
        "records": [],
    }
    aligned = _paragraphs(source, target, paragraph_identity)
    proofs = _accepted_edit_proofs(events, target)
    source_keys = {prefix + _sha(source)[:16] for prefix in ("anchor:", "chunk:", "review:")}
    for english, old in nouns.all_nouns().items():
        provenance = nouns.provenance_for(english)
        category = nouns.category_for(english)
        if (
            provenance.get("origin")
            not in {
                "auto_extraction",
                "incremental_extraction",
                "research_suggestion",
                "observed_translation",
            }
            or not is_automatic_entity_category(category)
            or nouns.is_context_deferred(english)
            or not source_term_present(source, english)
        ):
            continue
        new = observed_bilingual_target(target, english, category)
        if not new or normalized_entity_target(new) == normalized_entity_target(old):
            continue
        reason = "final_entity_alignment_unproven"
        evidence: dict[str, Any] | None = None
        local_paragraph: int | None = None
        global_paragraph: int | None = None
        pattern = source_term_pattern(english)
        if aligned and pattern and len(list(pattern.finditer(source))) == 1:
            sources, targets = aligned
            local = next(
                (i for i, value in enumerate(sources) if source_term_present(value, english)), -1
            )
            if local >= 0:
                paragraph = paragraph_identity["paragraph_indices"][local]
                local_paragraph, global_paragraph = local, paragraph
                reason = "final_entity_correction_proof_unavailable"
                quality_bound = (
                    quality.get("candidate_target_hash") == _sha(target)
                    and quality.get("critique_valid") is True
                    and quality.get("critique_candidate_match") is True
                    and quality.get("candidate_matched_critique_present") is True
                )
                issues = quality.get("issues", [])
                issues_valid = isinstance(issues, list) and all(isinstance(i, dict) for i in issues)
                unresolved = (
                    [i for i in issues if str(i.get("status", "")).startswith("unresolved")]
                    if issues_valid
                    else []
                )
                # Quality pN:sM identifiers are chunk-local, unlike provenance indices.
                issue_scope_proven = issues_valid and all(
                    type(i.get("source_paragraph_index")) is int
                    and 0 <= i["source_paragraph_index"] < len(sources)
                    for i in unresolved
                )
                clean = (
                    not quality.get("unresolved_language_event_count", 0)
                    and issue_scope_proven
                    and all(i["source_paragraph_index"] != local for i in unresolved)
                )
                matching = []
                for proof in proofs:
                    before_parts = re.split(r"\n\s*\n", proof["before"])
                    if len(before_parts) != len(sources):
                        continue
                    for decision in proof["decisions"]:
                        a, b = decision["original_span"], decision["resulting_span"]
                        if (
                            decision.get("issue_id")
                            and a in before_parts[local]
                            and b in targets[local]
                            and normalized_entity_target(
                                observed_bilingual_target(a, english, category)
                            )
                            == normalized_entity_target(old)
                            and normalized_entity_target(
                                observed_bilingual_target(b, english, category)
                            )
                            == normalized_entity_target(new)
                        ):
                            matching.append((proof, decision))
                if not issue_scope_proven:
                    reason = "final_entity_issue_scope_unproven"
                elif not clean:
                    reason = "final_entity_paragraph_has_unresolved_evidence"
                elif not quality_bound:
                    reason = "final_entity_quality_not_candidate_bound"
                elif not set(provenance.get("evidence_keys", [])).issubset(source_keys):
                    reason = "final_entity_has_competing_source_observations"
                elif (
                    len(matching) == 1
                    and is_safe_automatic_entity_mapping(
                        english,
                        new,
                        category,
                        sources[local],
                        translation=targets[local],
                        require_observed_anchor=True,
                    )
                    and len(re.findall(r"\(" + re.escape(english) + r"\)", targets[local], re.I))
                    == 1
                ):
                    proof, decision = matching[0]
                    evidence = {
                        "source_sha256": _sha(source),
                        "candidate_target_hash": _sha(target),
                        "source_paragraph_hash": _sha(sources[local]),
                        "paragraph_index": paragraph,
                        "source_local_paragraph_index": local,
                        "chunk_index": chunk_index,
                        "baseline_target_hash": proof["baseline_target_hash"],
                        "decision_id": decision.get("issue_id", ""),
                        "old_target": old,
                    }
        if evidence:
            outcome = nouns.reconcile_final_observation(english, new, evidence=evidence)
            if outcome.get("action") == "unchanged":
                outcome = nouns.defer_final_observation(
                    english,
                    new,
                    reason="final_observation_proof_rejected",
                    source_sha256=_sha(source),
                    target_sha256=_sha(target),
                )
        else:
            outcome = nouns.defer_final_observation(
                english, new, reason=reason, source_sha256=_sha(source), target_sha256=_sha(target)
            )
        report["records"].append(
            {
                "source": english,
                "old_target": old,
                "observed_final_target": new,
                "source_local_paragraph_index": local_paragraph,
                "source_global_paragraph_index": global_paragraph,
                **outcome,
            }
        )
    return report


def audit_first_occurrences(proper: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Read-only raw/effective classifications; never silently repair a job."""
    nouns = ProperNouns()
    nouns.deserialize(proper)
    records, missing, stale = [], [], []
    for english in proper.get("introduced", []):
        target = nouns.all_nouns().get(english, "")
        if not target:
            continue
        evidence = [
            {
                "chunk_index": c.get("chunk_index"),
                **first_occurrence_evidence(
                    str(c.get("text", "")),
                    str(c.get("translation", "")),
                    english,
                    target,
                ),
            }
            for c in chunks
            if c.get("translation")
        ]
        present = any(e["first_occurrence_present"] for e in evidence)
        rendered = any(e["stored_target_present"] for e in evidence)
        deferred = nouns.is_context_deferred(english)
        classification = (
            "missing_first_occurrence_evidence"
            if not present
            else "deferred_historical_mapping_conflict"
            if deferred and not rendered
            else "active_mapping_not_final_bound"
            if not rendered
            else "first_occurrence_present"
        )
        if not present:
            missing.append(english)
        elif not rendered and not deferred:
            stale.append(english)
        records.append(
            {
                "source": english,
                "stored_target": target,
                "classification": classification,
                "effective_deferred": deferred,
                "evidence": [e for e in evidence if e["source_present"]],
            }
        )
    return {
        "records": records,
        "missing_first_occurrences": missing,
        "active_stale_mappings": stale,
        "policy": "raw_history_visible_effective_deferral_is_not_a_job_rewrite",
    }
