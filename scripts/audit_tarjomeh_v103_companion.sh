audit_tarjomeh_v103_companion() {
    cd /opt/translate || {
        echo "ERROR: /opt/translate not found"
        return 1
    }

    JOB="${1:-LATEST}"

    echo "========== HOST / CONTAINERS =========="
    git rev-parse HEAD
    git tag --points-at HEAD
    git status --short
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
    df -h /

    echo
    echo "========== V10.3 COMPANION AUDIT =========="
    docker exec -i translate_tarjomeh_1 env PYTHONIOENCODING=utf-8 \
      python - "$JOB" <<'PY' | tee /root/tarjomeh_v103_companion_audit.txt
import collections
import json
import re
import sqlite3
import sys
import zipfile
from pathlib import Path

import tarjomeh
from tarjomeh.memory.proper_nouns import (
    is_safe_automatic_entity_mapping,
    is_usable_memory_mapping,
    is_usable_observed_mapping,
)


def decode(value, default):
    try:
        return json.loads(value or "")
    except Exception:
        return default


hint = sys.argv[1]
db = sqlite3.connect("/app/jobs/jobs.db")
db.row_factory = sqlite3.Row
job_row = db.execute(
    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT 1"
    if hint == "LATEST" else "SELECT * FROM jobs WHERE id=?",
    () if hint == "LATEST" else (hint,),
).fetchone()
if not job_row:
    raise SystemExit("JOB NOT FOUND")
job = dict(job_row)
job_id = job["id"]
config = decode(job.get("config"), {})

chunks = [dict(row) for row in db.execute(
    "SELECT chunk_index,status,text,translation,metadata FROM chunks "
    "WHERE job_id=? ORDER BY chunk_index",
    (job_id,),
)]
for chunk in chunks:
    chunk["metadata"] = decode(chunk.get("metadata"), {})

events = [dict(row) for row in db.execute(
    "SELECT rowid,timestamp,chunk_index,event_type,payload FROM chunk_events "
    "WHERE job_id=? ORDER BY timestamp,rowid",
    (job_id,),
)]
for event in events:
    event["payload"] = decode(event.get("payload"), {})
counts = collections.Counter(event["event_type"] for event in events)


def artifact(key):
    row = db.execute(
        "SELECT payload FROM job_artifacts WHERE job_id=? AND artifact_key=?",
        (job_id, key),
    ).fetchone()
    return decode(row["payload"], {}) if row else {}


state_row = db.execute(
    "SELECT state_data FROM memory_state WHERE job_id=?", (job_id,)
).fetchone()
state = decode(state_row["state_data"], {}) if state_row else {}

# Ignore obsolete failures from an earlier run of a chunk after resume/retry.
latest_start = {}
for event in events:
    if event["event_type"] == "chunk_started" and event["chunk_index"] >= 0:
        latest_start[event["chunk_index"]] = event["rowid"]
active_events = [
    event for event in events
    if event["chunk_index"] < 0
    or event["rowid"] >= latest_start.get(event["chunk_index"], 0)
]
active_counts = collections.Counter(event["event_type"] for event in active_events)
all_attempts = [
    event for event in events if event["event_type"] == "llm_call_attempt"
]
active_attempt_events = [
    event for event in active_events if event["event_type"] == "llm_call_attempt"
]
active_attempt_rowids = {event["rowid"] for event in active_attempt_events}
obsolete_attempts = [
    event["payload"] for event in all_attempts
    if event["rowid"] not in active_attempt_rowids
]
obsolete_successes = [
    item for item in obsolete_attempts if item.get("success") is True
]
obsolete_failures = [
    item for item in obsolete_attempts if item.get("success") is False
]
active_attempts = [event["payload"] for event in active_attempt_events]
failed = [item for item in active_attempts if item.get("success") is False]
lifetime_attempts = [event["payload"] for event in all_attempts]
lifetime_failed = [
    item for item in lifetime_attempts if item.get("success") is False
]
zero_visible_success = [
    item for item in active_attempts
    if item.get("success") is True
    and not bool(item.get("visible_output_present"))
    and int(item.get("completion_tokens") or 0) == 0
]
failure_types = collections.Counter(
    (
        str(item.get("operation", "unknown")),
        str(item.get("failure_reason") or item.get("failure_type") or "unknown"),
    )
    for item in failed
)
transport_failure_names = {
    "connect_timeout", "empty_completion", "empty_response", "empty_stream",
    "incomplete_completion", "incomplete_stream", "malformed_json",
    "malformed_sse", "provider_error", "read_timeout", "transport_error",
}
transport_failed = [
    item for item in failed
    if str(item.get("failure_reason") or item.get("failure_type") or "")
    in transport_failure_names
]
content_failed = [item for item in failed if item not in transport_failed]
operation_stats = {}
for item in active_attempts:
    operation = str(item.get("operation", "unknown"))
    stats = operation_stats.setdefault(operation, {
        "attempts": 0,
        "failed": 0,
        "successful": 0,
        "completion_tokens": 0,
        "duration_seconds": 0.0,
        "at_85000": 0,
        "served_models": collections.Counter(),
    })
    stats["attempts"] += 1
    stats["failed"] += item.get("success") is False
    stats["successful"] += item.get("success") is True
    stats["completion_tokens"] += int(item.get("completion_tokens") or 0)
    stats["duration_seconds"] += float(item.get("duration_seconds") or 0)
    stats["at_85000"] += int(item.get("max_tokens") or 0) >= 85000
    served = str(item.get("response_model") or item.get("model") or "unknown")
    stats["served_models"][served] += 1
for stats in operation_stats.values():
    stats["duration_seconds"] = round(stats["duration_seconds"], 3)
    stats["served_models"] = dict(stats["served_models"])

# Verify the v10 retry invariants from persisted request hashes.
transport_failures = {
    "empty_stream", "empty_response", "incomplete_stream", "malformed_sse",
    "malformed_json", "provider_error", "read_timeout", "connect_timeout",
    "transport_error", "empty_completion", "incomplete_completion",
}
quality_operations = {
    "translation", "translation_integrity_repair",
    "translation_split_recovery", "critique", "refinement",
}
last_attempt = {}
exact_replay_violations = []
quality_contract_violations = []
expected_helper_recovery = []
retry_pairs = []
for event in active_attempt_events:
    item = event["payload"]
    key = (event["chunk_index"], str(item.get("operation", "unknown")))
    number = int(item.get("attempt") or 1)
    if number == 1:
        last_attempt[key] = item
        continue
    previous = last_attempt.get(key)
    if previous:
        pair = {
            "chunk_index": event["chunk_index"],
            "operation": key[1],
            "attempt": number,
            "previous_failure": previous.get("failure_reason"),
            "same_contract": (
                previous.get("request_contract_sha256")
                == item.get("request_contract_sha256")
            ),
            "same_payload": (
                previous.get("request_payload_sha256")
                == item.get("request_payload_sha256")
            ),
            "previous_max_tokens": previous.get("max_tokens"),
            "max_tokens": item.get("max_tokens"),
        }
        retry_pairs.append(pair)
        if (
            number == 2
            and str(previous.get("failure_reason", "")) in transport_failures
            and key[1] in quality_operations
            and not pair["same_payload"]
        ):
            exact_replay_violations.append(pair)
        elif (
            number == 2
            and str(previous.get("failure_reason", "")) in transport_failures
            and key[1] not in quality_operations
            and not pair["same_payload"]
        ):
            expected_helper_recovery.append(pair)
        if (
            number == 2
            and previous.get("failure_reason") == "length"
            and key[1] in quality_operations
            and not pair["same_contract"]
        ):
            quality_contract_violations.append(pair)
    last_attempt[key] = item

finished = [
    chunk for chunk in chunks
    if chunk["status"] in {"completed", "needs_review"}
]
empty_finished = [
    chunk["chunk_index"] for chunk in finished
    if not str(chunk.get("translation") or "").strip()
]
needs_review = [
    chunk["chunk_index"] for chunk in chunks if chunk["status"] == "needs_review"
]

required_flow = {
    "chunk_started",
    "translation_completed",
    "integrity_check_completed",
    "terminology_context",
    "inline_original_policy",
    "chunk_completed",
}
if config.get("memory", {}).get("enable_4layer", True):
    required_flow.update({"memory_context", "memory_update_policy"})
if config.get("glossary", {}).get("enable_compliance_check", True):
    required_flow.add("glossary_compliance_final")
missing_flow = {}
for chunk in finished:
    index = chunk["chunk_index"]
    present = {
        event["event_type"] for event in active_events
        if event["chunk_index"] == index
    }
    expected = set(required_flow)
    paragraph_count = len([
        value for value in str(chunk.get("text") or "").split("\n\n")
        if value.strip()
    ])
    if (
        paragraph_count > 1
        and int(chunk["metadata"].get("paragraph_protocol_version", 0)) >= 1
    ):
        expected.add("paragraph_protocol_checked")
    missing = sorted(expected - present)
    if (
        config.get("translation", {}).get("enable_critique", True)
        and "critique_completed" not in present
        and "qa_unavailable" not in present
    ):
        missing.append("critique_completed_or_qa_unavailable")
    if missing:
        missing_flow[index] = missing

quarantined = {
    event["chunk_index"] for event in active_events
    if event["event_type"] == "translation_baseline_quarantined"
}
repaired = {
    event["chunk_index"] for event in active_events
    if event["event_type"] == "translation_baseline_repaired"
}
unrepaired = sorted(quarantined - repaired)
memory_chunks = {
    event["chunk_index"] for event in active_events
    if event["event_type"] == "memory_update_policy"
}
unsafe_admission = sorted(set(unrepaired) & memory_chunks)
finished_unrepaired = sorted(set(unrepaired) & {
    chunk["chunk_index"] for chunk in finished
})

proper = state.get("proper_nouns", {})
nouns = proper.get("nouns", proper) if isinstance(proper, dict) else {}
categories = proper.get("categories", {}) if isinstance(proper, dict) else {}
provenance = proper.get("provenance", {}) if isinstance(proper, dict) else {}
introduced = proper.get("introduced", []) if isinstance(proper, dict) else []
invalid_nouns = [
    source for source, target in nouns.items()
    if not is_usable_memory_mapping(source, target)
    or (
        isinstance(provenance.get(source), dict)
        and provenance[source].get("origin") == "observed_translation"
        and not is_usable_observed_mapping(
            source, target, categories.get(source, "proper_noun")
        )
    )
]
source_corpus = "\n".join(str(chunk.get("text") or "") for chunk in chunks)
unsafe_automatic_nouns = [
    source for source, target in nouns.items()
    if isinstance(provenance.get(source), dict)
    and provenance[source].get("origin") in {
        "auto_extraction", "incremental_extraction", "observed_translation",
        "research_suggestion",
    }
    and not is_safe_automatic_entity_mapping(
        source,
        target,
        categories.get(source, "proper_noun"),
        source_corpus,
    )
]
unsubstantiated_introduced = []
for source in introduced:
    target = str(nouns.get(source, "")).strip()
    rendered = any(
        source.casefold() in str(chunk.get("text") or "").casefold()
        and re.sub(r"[\s\u200c]+", " ", target).casefold()
        in re.sub(
            r"[\s\u200c]+", " ", str(chunk.get("translation") or "")
        ).casefold()
        for chunk in finished
    )
    if target and not rendered:
        unsubstantiated_introduced.append(source)
long_term = state.get("past_translations", [])
short_term = state.get("short_term_context", [])
summary = state.get("bilingual_summary", {})
short_trust = collections.Counter(
    str(item.get("trust", "legacy"))
    for item in short_term if isinstance(item, dict)
)
long_trust = collections.Counter(
    "reliable" if item.get("reliable", True) else "advisory"
    for item in long_term if isinstance(item, dict)
)

research = artifact("book_research")
checkpoints = artifact("chapter_checkpoints")
protocol = artifact("protocol_integrity_audit")
final_text = artifact("final_text_quality_audit")
final_identifiers = artifact("final_identifier_reconciliation")
fragment_reconciliation = artifact("original_fragment_reconciliation")
structure = artifact("document_structure_version")
entity_coverage = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"] == "source_entity_coverage"
]
missing_person_anchors = [
    {
        "chunk_index": item.get("chunk_index"),
        "missing": item.get("required_missing", []),
    }
    for item in entity_coverage
    if item.get("required_missing")
]
review_events_without_reason = [
    {"chunk_index": event["chunk_index"], "payload": event["payload"]}
    for event in active_events
    if event["event_type"] == "chunk_review_required"
    and not (
        event["payload"].get("reason_codes")
        or event["payload"].get("reason")
    )
]
cross_table_joins = [
    {
        "chunk_index": chunk["chunk_index"],
        "intervening_table_blocks": chunk["metadata"].get(
            "intervening_table_blocks", 0
        ),
    }
    for chunk in chunks if chunk["metadata"].get("cross_table_join")
]
style_samples = state.get("style_samples", [])
malformed_style_samples = [
    index for index, sample in enumerate(style_samples)
    if isinstance(sample, str)
    and (
        sample.lstrip().startswith(("...", "…"))
        or sample.rstrip().endswith(("...", "…"))
        or re.search(
            r"\([A-Za-z][A-Za-z .'-]{1,80}\)"
            r"[\u0621-\u063a\u0641-\u064a\u066e-\u06d3\u06fa-\u06ff]",
            sample,
        )
    )
]
output = Path(job.get("output_path") or "")
if not output.is_absolute():
    output = Path("/app") / output
output_valid = bool(
    output.exists()
    and output.stat().st_size > 0
    and (
        output.suffix.lower() != ".docx"
        or zipfile.is_zipfile(output)
    )
)

# Confirm the running container, not merely Git, contains v10.3 behavior.
root = Path(tarjomeh.__file__).resolve().parent
pipeline_source = (root / "core" / "pipeline.py").read_text(encoding="utf-8")
client_source = (root / "core" / "llm_client.py").read_text(encoding="utf-8")
structure_source = (root / "quality" / "structure_audit.py").read_text(
    encoding="utf-8"
)
parser_source = (root / "parsers" / "pdf_parser.py").read_text(encoding="utf-8")
proper_source = (root / "memory" / "proper_nouns.py").read_text(encoding="utf-8")
integrity_source = (root / "quality" / "integrity.py").read_text(encoding="utf-8")
runtime_markers = {
    "persistent_async_loop": "_async_loop_ready" in pipeline_source,
    "baseline_quarantine": "translation_baseline_quarantined" in pipeline_source,
    "repair_admission": "translation_baseline_repaired" in pipeline_source,
    "route_budget_history": "_route_served_model" in client_source,
    "request_lineage_hash": "request_payload_sha256" in client_source,
    "incremental_sse_evidence": "terminal_received" in client_source,
    "source_grounded_structure": "SOURCE_ANOMALY_PRESERVED" in structure_source,
    "current_chunk_entity_coverage": "source_entity_coverage" in pipeline_source,
    "observed_entity_reconciliation": (
        "_reconcile_current_entity_anchors" in pipeline_source
    ),
    "technical_loanword_policy": (
        "looks_like_transliterated_loanword" in proper_source
    ),
    "table_interrupted_continuity": (
        "_merge_table_interrupted_continuations" in parser_source
    ),
    "parenthetical_suffix_guard": (
        "_PARENTHETICAL_PERSIAN_SUFFIX_RE" in integrity_source
    ),
    "targeted_integrity_repair": (
        "translation_integrity_repair" in pipeline_source
    ),
    "canonical_possessive_entities": (
        "_canonical_source_entity" in pipeline_source
    ),
    "strict_observed_memory": (
        "is_usable_observed_mapping" in proper_source
    ),
    "source_abbreviation_expansions": (
        "source_abbreviation_expansions" in integrity_source
    ),
    "same_line_entity_inventory": "_LATIN_ENTITY_CONNECTOR" in pipeline_source,
    "grounded_entity_admission": (
        "is_safe_automatic_entity_mapping" in proper_source
    ),
    "translation_grounded_introduction": (
        "mark_introduced_from_translation" in proper_source
    ),
    "empty_completion_split_recovery": (
        "repeated_empty_completion" in pipeline_source
    ),
    "canonical_direct_review": "_explicit_chunk_review_payload" in pipeline_source,
    "labeled_identifier_audit": (
        "extract_labeled_identifier_surfaces" in integrity_source
    ),
}

hard = []
review = []
if not all(runtime_markers.values()):
    hard.append("running_container_is_not_complete_v103")
if empty_finished:
    hard.append("empty_finished_chunk")
if missing_flow:
    hard.append("missing_finished_chunk_flow")
if unsafe_admission:
    hard.append("quarantined_translation_entered_memory")
if finished_unrepaired:
    hard.append("unrepaired_quarantine_finished")
if exact_replay_violations:
    hard.append("transport_replay_changed_payload")
if quality_contract_violations:
    hard.append("quality_attempt_2_changed_contract")
if zero_visible_success:
    hard.append("successful_attempt_without_visible_output")
if invalid_nouns:
    hard.append("invalid_proper_noun_mapping")
if unsafe_automatic_nouns:
    hard.append("unsafe_automatic_entity_mapping")
if unsubstantiated_introduced:
    hard.append("source_only_first_occurrence_state")
if review_events_without_reason:
    hard.append("review_event_without_reason")
if malformed_style_samples:
    hard.append("malformed_text_entered_style_memory")
if finished and not output_valid:
    hard.append("output_missing_or_invalid")
if int(protocol.get("remaining_artifact_count", 0) or 0):
    hard.append("protocol_artifact_in_export")
if int(final_text.get("unresolved_identifier_count", 0) or 0):
    hard.append("unresolved_identifier_in_export")
if int(final_identifiers.get("unresolved_count", 0) or 0):
    hard.append("unresolved_identifier_reconciliation")

if needs_review:
    review.append("chunk_needs_review")
if active_counts["qa_unavailable"]:
    review.append("qa_unavailable")
if active_counts["integrity_final_failed"]:
    review.append("integrity_final_failed")
if final_text.get("review_required"):
    review.append("final_text_review_required")
if job.get("error_message"):
    review.append("job_error_message")
if missing_person_anchors:
    review.append("missing_first_occurrence_person_anchor")
verdict = "FAIL" if hard else "REVIEW" if review else "PASS"
failure_rate = round(100 * len(failed) / max(1, len(active_attempts)), 2)
lifetime_failure_rate = round(
    100 * len(lifetime_failed) / max(1, len(lifetime_attempts)), 2
)

summary_payload = {
    "job": {
        "id": job_id,
        "status": job.get("status"),
        "error_message": job.get("error_message"),
        "chunks": dict(collections.Counter(chunk["status"] for chunk in chunks)),
        "finished": len(finished),
        "needs_review": needs_review,
    },
    "runtime_markers": runtime_markers,
    "pipeline": {
        "event_counts": dict(counts),
        "active_event_counts": dict(active_counts),
        "missing_finished_chunk_flow": missing_flow,
        "quarantined": sorted(quarantined),
        "repaired": sorted(repaired),
        "unrepaired": unrepaired,
        "unsafe_memory_admission": unsafe_admission,
        "source_entity_coverage": entity_coverage,
        "missing_first_occurrence_person_anchors": missing_person_anchors,
        "cross_table_continuation_joins": cross_table_joins,
    },
    "llm": {
        "all_attempts": len(all_attempts),
        "active_attempts": len(active_attempts),
        "obsolete_attempts": len(obsolete_attempts),
        "obsolete_successes": len(obsolete_successes),
        "obsolete_failures": len(obsolete_failures),
        "active_failures": len(failed),
        "transport_failures": len(transport_failed),
        "content_failures": len(content_failed),
        "active_failure_rate_pct": failure_rate,
        "lifetime_failures": len(lifetime_failed),
        "lifetime_failure_rate_pct": lifetime_failure_rate,
        "zero_visible_success": zero_visible_success,
        "failure_types": {str(key): value for key, value in failure_types.items()},
        "operation_stats": operation_stats,
        "retry_pairs": retry_pairs,
        "exact_replay_violations": exact_replay_violations,
        "quality_contract_violations": quality_contract_violations,
        "expected_helper_recovery": expected_helper_recovery,
    },
    "memory": {
        "proper_nouns": len(nouns),
        "invalid_proper_nouns": invalid_nouns,
        "unsafe_automatic_entities": unsafe_automatic_nouns,
        "unsubstantiated_introduced": unsubstantiated_introduced,
        "bilingual_summary": bool(
            isinstance(summary, dict)
            and summary.get("english_summary")
            and summary.get("persian_summary")
        ),
        "long_term": len(long_term),
        "long_term_trust": dict(long_trust),
        "short_term": len(short_term),
        "short_term_trust": dict(short_trust),
        "style_samples": len(state.get("style_samples", [])),
        "malformed_style_samples": malformed_style_samples,
        "style_profile": bool(state.get("style_profile")),
        "book_context": bool(state.get("book_context")),
    },
    "research": {
        "status": research.get("status"),
        "sources": research.get("source_count", len(research.get("sources", []))),
        "terms": len(research.get("terms", [])),
    },
    "checkpoint": checkpoints,
    "structure": structure,
    "protocol_audit": protocol,
    "final_text_audit": final_text,
    "final_identifier_reconciliation": final_identifiers,
    "original_fragment_reconciliation": fragment_reconciliation,
    "review_events_without_reason": review_events_without_reason,
    "output": {
        "path": str(output),
        "exists": output.exists(),
        "valid": output_valid,
        "size": output.stat().st_size if output.exists() else 0,
    },
    "verdict": verdict,
    "hard_failures": hard,
    "review_signals": review,
}
json_path = Path(f"/app/jobs/uploads/{job_id}_v103_companion_audit.json")
json_path.write_text(
    json.dumps(
        {
            "summary": summary_payload,
            "active_failures": failed,
            "active_attempts": active_attempts,
            "obsolete_attempts": obsolete_attempts,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
txt_path = Path(f"/app/jobs/uploads/{job_id}_v103_companion_audit.txt")

lines = [
    "Tarjomeh v10.3 Independent Companion Audit",
    f"Job: {job_id}",
    f"Status: {job.get('status')}",
    "",
    "RUNTIME",
    f"Markers: {runtime_markers}",
    "",
    "PIPELINE",
    f"Chunk statuses: {summary_payload['job']['chunks']}",
    f"Finished: {len(finished)} / {len(chunks)}",
    f"Missing flow: {missing_flow or 'none'}",
    f"Quarantined: {sorted(quarantined) or 'none'}",
    f"Repaired: {sorted(repaired) or 'none'}",
    f"Unsafe memory admission: {unsafe_admission or 'none'}",
    f"Source-entity coverage events: {len(entity_coverage)}",
    f"Missing person anchors: {missing_person_anchors or 'none'}",
    f"Cross-table continuation joins: {cross_table_joins or 'none'}",
    "",
    "LLM",
    f"All attempts: {len(all_attempts)}",
    f"Current-generation attempts: {len(active_attempts)}",
    f"Obsolete attempts: {len(obsolete_attempts)} "
    f"(successful={len(obsolete_successes)}, failed={len(obsolete_failures)})",
    f"Current-generation failures: {len(failed)} ({failure_rate}%)",
    f"Lifetime failures: {len(lifetime_failed)} ({lifetime_failure_rate}%)",
    f"Transport failures: {len(transport_failed)}",
    f"Content/quality failures: {len(content_failed)}",
    f"Zero-visible-output successes: {len(zero_visible_success)}",
    f"Failure types: {dict(failure_types)}",
    f"Exact replay violations: {exact_replay_violations or 'none'}",
    f"Quality Attempt-2 contract violations: "
    f"{quality_contract_violations or 'none'}",
    f"Expected helper recovery changes: {expected_helper_recovery or 'none'}",
]
for operation, stats in sorted(operation_stats.items()):
    lines.append(f"  {operation}: {stats}")
lines.extend([
    "",
    "FOUR-LAYER MEMORY",
    f"Proper nouns: {len(nouns)}",
    f"Invalid proper nouns: {invalid_nouns or 'none'}",
    f"Unsafe automatic mappings: {unsafe_automatic_nouns or 'none'}",
    f"Unsubstantiated introduced state: {unsubstantiated_introduced or 'none'}",
    f"Bilingual summary complete: {summary_payload['memory']['bilingual_summary']}",
    f"Long term: {len(long_term)} {dict(long_trust)}",
    f"Short term: {len(short_term)} {dict(short_trust)}",
    f"Style samples: {len(state.get('style_samples', []))}",
    f"Malformed style samples: {malformed_style_samples or 'none'}",
    f"Style profile: {bool(state.get('style_profile'))}",
    f"Book context: {bool(state.get('book_context'))}",
    "",
    "RESEARCH / EXPORT",
    f"Research: {summary_payload['research']}",
    f"Checkpoint: {checkpoints}",
    f"Protocol remaining: {protocol.get('remaining_artifact_count', 0)}",
    f"Final identifiers unresolved: "
    f"{final_text.get('unresolved_identifier_count', 0)}",
    f"Identifier reconciliation unresolved: "
    f"{final_identifiers.get('unresolved_count', 0)}",
    f"Redundant original fragments removed: "
    f"{fragment_reconciliation.get('removed_count', 0)}",
    f"Review events without reason: {review_events_without_reason or 'none'}",
    f"Final text review required: {final_text.get('review_required', False)}",
    f"Output: {summary_payload['output']}",
    "",
    "VERDICT",
    f"Hard failures: {hard or 'none'}",
    f"Review signals: {review or 'none'}",
    f"VERDICT: {verdict}",
    f"JSON: {json_path}",
])
txt_path.write_text("\n".join(lines), encoding="utf-8-sig")
print("\n".join(lines))
print("TXT:", txt_path)
PY

    RESULT=$?
    if [ "$RESULT" -ne 0 ]; then
        echo "ERROR: v10.3 companion audit failed"
        return "$RESULT"
    fi

    echo
    echo "========== GENERATED FILES =========="
    find jobs/uploads -maxdepth 1 -type f \
      \( -name '*_v103_companion_audit.txt' \
         -o -name '*_v103_companion_audit.json' \) \
      -printf '%TY-%Tm-%Td %TH:%TM %10s %p\n' | sort | tail -4
    echo "Root console copy: /root/tarjomeh_v103_companion_audit.txt"
    echo "Read-only v10.3 companion audit completed. Nothing was changed."
}

audit_tarjomeh_v103_companion "${JOB:-LATEST}"
unset -f audit_tarjomeh_v103_companion
