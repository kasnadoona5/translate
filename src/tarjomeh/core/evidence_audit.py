"""Report-only accounting; these helpers grant no translation or memory authority."""

from collections import Counter
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
