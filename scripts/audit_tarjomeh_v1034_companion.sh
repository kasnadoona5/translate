audit_tarjomeh_v1034_companion() {
    cd /opt/translate || {
        echo "ERROR: /opt/translate not found"
        return 1
    }

    local JOB="${1:-LATEST}"

    echo "========== HOST / CONTAINERS =========="
    git rev-parse HEAD
    git tag --points-at HEAD
    git status --short
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
    df -h /

    echo
    echo "========== V10.34 COMPANION AUDIT =========="
    docker exec -i translate_tarjomeh_1 \
      tarjomeh capabilities --verify --json \
      | tee /root/tarjomeh_v1034_companion_audit.txt
    [ "${PIPESTATUS[0]}" -eq 0 ] || return 1
    docker exec -i translate_tarjomeh_1 env PYTHONIOENCODING=utf-8 \
      python - "$JOB" <<'PY' | tee -a /root/tarjomeh_v1034_companion_audit.txt
import collections
import hashlib
import json
import os
import re
import sqlite3
import sys
import zipfile
from pathlib import Path

import tarjomeh
from docx import Document
from tarjomeh.core.term_notes import normalize_citation_house_style_text
from tarjomeh.memory.manager import (
    _NONREPRESENTATIVE_STYLE_SOURCE_RE,
    _style_sample_quality,
)
from tarjomeh.memory.proper_nouns import (
    automatic_terminology_risk_reasons,
    is_automatic_entity_category,
    is_safe_automatic_entity_mapping,
    is_safe_automatic_source_span,
    is_safe_low_authority_mapping,
    is_reusable_terminology_mapping,
    is_usable_memory_mapping,
    low_authority_mapping_category,
    source_term_present,
)
from tarjomeh.quality.integrity import (
    repeated_persian_clause_artifacts,
    restore_source_note_markers,
)
from tarjomeh.quality.grounding import source_segments
from tarjomeh.runtime import runtime_behavior_probes, runtime_capabilities


def decode(value, default):
    try:
        return json.loads(value or "")
    except Exception:
        return default


def normalized_target_evidence(value):
    """Compare lexical target evidence while ignoring separator typography."""
    value = re.sub(r"[\u064b-\u065f\u0670]", "", str(value or ""))
    return re.sub(
        r"[\s\u200c\u0640\-\u2010-\u2015]+", " ", value
    ).strip().casefold()


hint = sys.argv[1]
runtime_manifest = runtime_capabilities()
runtime_probes = runtime_behavior_probes()
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
worker_table = db.execute(
    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='job_workers'"
).fetchone()
worker_lease_row = db.execute(
    "SELECT * FROM job_workers WHERE job_id=?", (job_id,)
).fetchone() if worker_table else None
worker_lease = dict(worker_lease_row) if worker_lease_row else {}

chunks = [dict(row) for row in db.execute(
    "SELECT chunk_index,status,text,translation,metadata FROM chunks "
    "WHERE job_id=? ORDER BY chunk_index",
    (job_id,),
)]
for chunk in chunks:
    chunk["metadata"] = decode(chunk.get("metadata"), {})
chunks_by_index = {int(item["chunk_index"]): item for item in chunks}

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
worker_events = [
    event for event in events
    if event["event_type"] in {
        "worker_claimed", "worker_claim_rejected",
        "worker_lease_acquired", "worker_lease_renewed",
        "worker_lease_reclaimed", "worker_pause_requested",
        "worker_pause_persisted", "worker_pause_acknowledged",
        "worker_lease_released",
    }
]
atomic_source_rollbacks = [
    event for event in active_events
    if event["event_type"] == "refinement_atomic_recovery_rolled_back"
]
summary_policy_events = [
    event for event in active_events
    if event["event_type"] == "bilingual_summary_memory_policy"
]
rejected_summary_candidates = [
    event for event in summary_policy_events
    if event["payload"].get("candidate_committed") is False
]
all_attempts = [
    event for event in events if event["event_type"] == "llm_call_attempt"
]
active_attempt_events = [
    event for event in active_events if event["event_type"] == "llm_call_attempt"
]
active_start_events = [
    event for event in active_events if event["event_type"] == "llm_call_started"
]
active_progress_events = [
    event for event in active_events
    if event["event_type"] == "llm_call_in_progress"
]
pending_starts = collections.defaultdict(list)
for event in active_events:
    if event["event_type"] not in {"llm_call_started", "llm_call_attempt"}:
        continue
    payload = event["payload"]
    key = (
        event["chunk_index"],
        str(payload.get("operation", "unknown")),
        int(payload.get("attempt") or 1),
    )
    if event["event_type"] == "llm_call_started":
        pending_starts[key].append(event)
    elif pending_starts[key]:
        pending_starts[key].pop(0)
unmatched_llm_starts = [
    {
        "timestamp": event["timestamp"],
        "chunk_index": event["chunk_index"],
        "operation": event["payload"].get("operation"),
        "attempt": event["payload"].get("attempt"),
        "model": event["payload"].get("model"),
    }
    for queued in pending_starts.values() for event in queued
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
long_llm_attempts = [
    {
        "chunk_index": event["chunk_index"],
        "operation": event["payload"].get("operation"),
        "attempt": event["payload"].get("attempt"),
        "duration_seconds": event["payload"].get("duration_seconds"),
        "success": event["payload"].get("success"),
        "model": event["payload"].get("response_model")
            or event["payload"].get("model"),
    }
    for event in active_attempt_events
    if float(event["payload"].get("duration_seconds") or 0) >= 600
]
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
finished_prose = [
    chunk for chunk in finished
    if bool(chunk.get("metadata", {}).get("style_eligible"))
    or "body" in {
        str(role).strip().casefold()
        for role in chunk.get("metadata", {}).get("structural_roles", [])
    }
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
    required_flow.update({
        "memory_context", "final_quality_admission", "memory_update_policy",
    })
required_flow.add("memory_export_consistency")
required_flow.add("source_bound_artifact_recovery")
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
memory_policy_events = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"] == "memory_update_policy"
]
final_quality_events = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in active_events
    if event["event_type"] == "final_quality_admission"
]
final_quality_by_chunk = {
    int(item["chunk_index"]): item for item in final_quality_events
}
style_sample_records = state.get("style_sample_records", [])
active_style_hashes = {
    str(record.get("text_hash", ""))
    for record in style_sample_records
    if isinstance(record, dict) and str(record.get("text_hash", ""))
}


def style_event_is_active(item):
    selected = (item.get("style_sample_policy", {}) or {}).get("selected", {}) or {}
    sample = str(selected.get("sample", ""))
    return bool(
        sample
        and hashlib.sha256(sample.encode("utf-8")).hexdigest()
        in active_style_hashes
    )


def unresolved_style_issue_ids(item, quality):
    selected_index = (item.get("style_sample_policy", {}) or {}).get(
        "source_paragraph_index"
    )
    if not isinstance(selected_index, int):
        return list(quality.get("unresolved_grounded_issue_ids", []) or [])
    return [
        str(issue.get("issue_id"))
        for issue in list(quality.get("issues", []) or [])
        if str(issue.get("status", "")).startswith("unresolved_")
        and issue.get("source_paragraph_index") == selected_index
        and issue.get("issue_id")
    ]
source_bound_recovery_events = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"] == "source_bound_artifact_recovery"
]
final_canonical_chunks = {
    int(event["chunk_index"])
    for event in active_events
    if event["event_type"] == "final_canonical_admission"
}
final_candidate_events = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in active_events
    if event["event_type"] == "final_candidate_selection"
]
final_candidate_by_chunk = {
    int(item["chunk_index"]): item for item in final_candidate_events
}
missing_final_canonical_admission = sorted(
    {int(chunk["chunk_index"]) for chunk in finished}
    - final_canonical_chunks
)
missing_final_candidate_selection = sorted(
    {int(chunk["chunk_index"]) for chunk in finished}
    - set(final_candidate_by_chunk)
)
final_candidate_hash_mismatches = [
    int(chunk["chunk_index"])
    for chunk in finished
    if int(chunk["chunk_index"]) in final_candidate_by_chunk
    and str(final_candidate_by_chunk[int(chunk["chunk_index"])].get(
        "canonical_target_hash", ""
    )) != hashlib.sha256(
        str(chunk.get("translation", "")).encode("utf-8")
    ).hexdigest()
]
final_candidate_policy_mismatches = sorted(
    int(chunk["chunk_index"])
    for chunk in finished
    if int(chunk["chunk_index"]) in final_candidate_by_chunk
    and int(final_candidate_by_chunk[int(chunk["chunk_index"])].get(
        "policy_version", 0
    ) or 0) != 2
)
candidate_quality_hash_mismatches = sorted(
    int(chunk["chunk_index"])
    for chunk in finished
    if (
        (quality := final_quality_by_chunk.get(int(chunk["chunk_index"]), {}))
        and quality.get("critique_present") is True
        and (
            quality.get("critique_candidate_match") is not True
            or str(quality.get("candidate_target_hash", ""))
            != hashlib.sha256(
                str(chunk.get("translation", "")).encode("utf-8")
            ).hexdigest()
        )
    )
)
checkpoint_event_timestamps = collections.defaultdict(dict)
for event in active_events:
    if event["event_type"] in {
        "final_candidate_selection",
        "final_canonical_admission",
        "final_quality_admission",
    }:
        checkpoint_event_timestamps[int(event["chunk_index"])][
            event["event_type"]
        ] = event["timestamp"]
non_atomic_final_quality = sorted(
    int(chunk["chunk_index"])
    for chunk in finished
    if len(set(checkpoint_event_timestamps.get(
        int(chunk["chunk_index"]), {}
    ).values())) > 1
)
authority_violations = []
for item in memory_policy_events:
    chunk_index = int(item["chunk_index"])
    quality = final_quality_by_chunk.get(chunk_index, {})
    authority = quality.get("durable_authority")
    style_issue_ids = unresolved_style_issue_ids(item, quality)
    if (
        authority is False and item.get("long_term_reliable") is True
    ) or (
        item.get("style_sample_added") is True
        and style_event_is_active(item)
        and style_issue_ids
    ):
        authority_violations.append({
            "chunk_index": chunk_index,
            "durable_authority": authority,
            "long_term_reliable": item.get("long_term_reliable"),
            "style_sample_added": item.get("style_sample_added"),
            "unresolved_issue_ids": style_issue_ids,
        })
source_repair_events = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"] == "targeted_language_repair"
    and (
        int(event["payload"].get("source_fidelity_finding_count", 0) or 0)
        or event["payload"].get("resolved_source_issue_ids")
    )
]
unsafe_grounded_memory = [
    {
        "chunk_index": item.get("chunk_index"),
        "issue_ids": item.get("grounded_unresolved_issue_ids", []),
    }
    for item in memory_policy_events
    if item.get("grounded_unresolved_issue_ids")
    and item.get("long_term_reliable") is True
]
unsafe_admission = sorted(set(unrepaired) & memory_chunks)
finished_unrepaired = sorted(set(unrepaired) & {
    chunk["chunk_index"] for chunk in finished
})

proper = state.get("proper_nouns", {})
nouns = proper.get("nouns", proper) if isinstance(proper, dict) else {}
categories = proper.get("categories", {}) if isinstance(proper, dict) else {}
provenance = proper.get("provenance", {}) if isinstance(proper, dict) else {}
introduced = proper.get("introduced", []) if isinstance(proper, dict) else []
authority_classes = collections.Counter(
    str(item.get("authority_class", "legacy_unclassified"))
    for item in provenance.values() if isinstance(item, dict)
)
invalid_authority_promotions = [
    source for source, target in nouns.items()
    if isinstance(provenance.get(source), dict)
    and provenance[source].get("authority_class") == "canonical_reviewed"
    and (
        len(set(provenance[source].get("evidence_keys", []) or [])) < 2
        or not provenance[source].get("context_independent")
        or not is_reusable_terminology_mapping(source, target)
    )
]
low_authority_origins = {
    "auto_extraction", "incremental_extraction",
    "observed_translation", "research_suggestion",
}
invalid_nouns = [
    source for source, target in nouns.items()
    if not is_usable_memory_mapping(source, target)
]
deferred_automatic_nouns = [
    source for source in nouns
    if isinstance(provenance.get(source), dict)
    and provenance[source].get("origin") in low_authority_origins
    and provenance[source].get("context_deferred")
]
unsafe_low_authority_nouns = [
    source for source, target in nouns.items()
    if isinstance(provenance.get(source), dict)
    and provenance[source].get("origin") in low_authority_origins
    and not provenance[source].get("context_deferred")
    and not is_safe_low_authority_mapping(
        source, target, categories.get(source, "proper_noun")
    )
]
unsafe_automatic_terms = {
    source: automatic_terminology_risk_reasons(source, target)
    for source, target in nouns.items()
    if isinstance(provenance.get(source), dict)
    and provenance[source].get("origin") in low_authority_origins
    and not provenance[source].get("context_deferred")
    and low_authority_mapping_category(
        source, target, categories.get(source, "proper_noun")
    ) == "term"
    and automatic_terminology_risk_reasons(source, target)
}
source_corpus = "\n".join(str(chunk.get("text") or "") for chunk in chunks)
target_corpus = "\n".join(
    str(chunk.get("translation") or "") for chunk in finished
)
unsafe_automatic_nouns = [
    source for source, target in nouns.items()
    if isinstance(provenance.get(source), dict)
    and provenance[source].get("origin") in {
        "auto_extraction", "incremental_extraction", "observed_translation",
        "research_suggestion",
    }
    and is_automatic_entity_category(categories.get(source, "proper_noun"))
    and not is_safe_automatic_entity_mapping(
        source,
        target,
        categories.get(source, "proper_noun"),
        source_corpus,
        translation=target_corpus,
    )
]
malformed_source_entity_nouns = [
    source for source in nouns
    if isinstance(provenance.get(source), dict)
    and provenance[source].get("origin") in {
        "auto_extraction", "incremental_extraction", "observed_translation",
        "research_suggestion",
    }
    and is_automatic_entity_category(categories.get(source, "proper_noun"))
    and not is_safe_automatic_source_span(
        source, categories.get(source, "proper_noun")
    )
]
unsubstantiated_introduced = []
for source in introduced:
    target = str(nouns.get(source, "")).strip()
    rendered = any(
        source_term_present(str(chunk.get("text") or ""), source)
        and normalized_target_evidence(target)
        in normalized_target_evidence(chunk.get("translation"))
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
research_identity_gaps = [
    str(item.get("url", ""))
    for item in research.get("sources", []) or []
    if isinstance(item, dict) and not item.get("identity_evidence")
]
research_term_evidence_gaps = [
    str(item.get("source", ""))
    for item in research.get("terms", []) or []
    if isinstance(item, dict)
    and item.get("status") == "suggested"
    and (
        item.get("authority") != "advisory_context_only"
        or not item.get("intended_use")
        or "identity_supported" not in item
        or "term_supported" not in item
        or "term_supporting_excerpts" not in item
    )
]
checkpoints = artifact("chapter_checkpoints")
checkpoint_positions = list((checkpoints or {}).get("reached_positions", []) or [])
checkpoint_pending = (checkpoints or {}).get("pending")
if not isinstance(checkpoint_pending, dict):
    checkpoint_pending = None
checkpoint_publications = [
    item for item in ((checkpoints or {}).get("publications", []) or [])
    if isinstance(item, dict)
]
latest_checkpoint_publication = (checkpoints or {}).get("latest_publication")
if not isinstance(latest_checkpoint_publication, dict):
    latest_checkpoint_publication = None
checkpoint_output_path = str(job.get("output_path") or "")
checkpoint_preview_missing = bool(
    job.get("status") == "paused"
    and checkpoint_positions
    and (
        not checkpoint_output_path
        or not os.path.isfile(checkpoint_output_path)
    )
)
checkpoint_publication_error = ""
if job.get("status") == "paused" and latest_checkpoint_publication:
    published_path = str(latest_checkpoint_publication.get("output_path") or "")
    expected_bytes = int(latest_checkpoint_publication.get("output_bytes") or 0)
    expected_sha256 = str(
        latest_checkpoint_publication.get("output_sha256") or ""
    )
    if not published_path or published_path != checkpoint_output_path:
        checkpoint_publication_error = "published_path_differs_from_job_output"
    elif not os.path.isfile(published_path):
        checkpoint_publication_error = "published_preview_missing"
    else:
        actual_bytes = os.path.getsize(published_path)
        actual_sha256 = hashlib.sha256(Path(published_path).read_bytes()).hexdigest()
        if expected_bytes <= 0 or actual_bytes != expected_bytes:
            checkpoint_publication_error = "published_preview_size_mismatch"
        elif not expected_sha256 or actual_sha256 != expected_sha256:
            checkpoint_publication_error = "published_preview_hash_mismatch"

finished_statuses = {"completed", "needs_review"}
reached_checkpoint_set = {
    int(value) for value in checkpoint_positions
    if str(value).lstrip("-").isdigit()
}
translation_config = config.get("translation", {}) or {}
try:
    configured_stop_after = int(
        translation_config.get("stop_after_chapter", 0) or 0
    )
except (TypeError, ValueError):
    configured_stop_after = 0
configured_pause_each = bool(
    translation_config.get("pause_after_each_chapter", False)
)
try:
    pending_position = int(checkpoint_pending.get("chapter_position", 0)) \
        if checkpoint_pending else 0
except (TypeError, ValueError):
    pending_position = 0
missed_requested_boundaries = []
for position_index, current_chunk in enumerate(chunks[:-1]):
    current_position = int(
        current_chunk["metadata"].get("chapter_position", 1) or 1
    )
    following_position = int(
        chunks[position_index + 1]["metadata"].get("chapter_position", 1) or 1
    )
    if current_position == following_position:
        continue
    if current_chunk.get("status") not in finished_statuses:
        continue
    requested = configured_pause_each or configured_stop_after == current_position
    if (
        requested
        and current_position not in reached_checkpoint_set
        and current_position != pending_position
    ):
        missed_requested_boundaries.append({
            "chapter_position": current_position,
            "boundary_chunk_index": int(current_chunk["chunk_index"]),
        })

completed_after_pending = []
if checkpoint_pending:
    try:
        pending_boundary = int(
            checkpoint_pending.get("boundary_chunk_index", -1)
        )
    except (TypeError, ValueError):
        pending_boundary = -1
    completed_after_pending = [
        int(chunk["chunk_index"])
        for chunk in chunks
        if int(chunk["chunk_index"]) > pending_boundary
        and chunk.get("status") in finished_statuses
    ]
protocol = artifact("protocol_integrity_audit")
final_text = artifact("final_text_quality_audit")
final_identifiers = artifact("final_identifier_reconciliation")
fragment_reconciliation = artifact("original_fragment_reconciliation")
original_audit = artifact("english_original_audit")
final_rendered_repair = artifact("final_rendered_language_repair")
canonical_document_identity = artifact("canonical_document_identity")
paragraph_identity = artifact("canonical_chunk_paragraphs_v1")
paragraph_identity_chunks = paragraph_identity.get("chunks", {}) \
    if isinstance(paragraph_identity, dict) else {}
if not isinstance(paragraph_identity_chunks, dict):
    paragraph_identity_chunks = {}
finished_indices = {str(int(item["chunk_index"])) for item in finished}
missing_paragraph_identity = sorted(
    finished_indices - set(paragraph_identity_chunks), key=int
)
invalid_paragraph_identity = []
reconstructed_paragraph_identity = []
for key in sorted(finished_indices & set(paragraph_identity_chunks), key=int):
    identity = paragraph_identity_chunks.get(key, {})
    chunk = chunks_by_index.get(int(key), {})
    translation = str(chunk.get("translation", "") or "")
    if (
        not isinstance(identity, dict)
        or identity.get("canonical_target_hash")
        != hashlib.sha256(translation.encode("utf-8")).hexdigest()
        or int(identity.get("target_count", 0) or 0)
        != len(identity.get("target_offsets", []) or [])
    ):
        invalid_paragraph_identity.append(int(key))
    if isinstance(identity, dict) and identity.get("reconstructed"):
        reconstructed_paragraph_identity.append(int(key))
structure = artifact("document_structure_version")
terminal_failures = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in events if event["event_type"] == "chunk_terminal_failure"
]
source_obligation_reuse_events = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in events
    if event["event_type"] == "source_obligation_recovery_reused"
]
source_obligation_state = artifact("source_obligation_recovery_v1") or {}
source_obligation_entries = source_obligation_state.get("entries", {})
if not isinstance(source_obligation_entries, dict):
    source_obligation_entries = {}
pending_finished_source_obligations = sorted(
    int(key)
    for key, value in source_obligation_entries.items()
    if isinstance(value, dict)
    and value.get("status") == "pending"
    and str(key) in finished_indices
)
entity_coverage = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"] == "source_entity_coverage"
]
missing_entity_anchors = [
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
style_sample_records = state.get("style_sample_records", [])
memory_export_events = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"] == "memory_export_consistency"
]
canonical_admission_events = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"] == "final_canonical_admission"
]
finished_indices = {chunk["chunk_index"] for chunk in finished}
missing_canonical_admission_events = sorted(
    finished_indices
    - {item["chunk_index"] for item in canonical_admission_events}
)
canonical_event_by_chunk = {
    int(item["chunk_index"]): item for item in canonical_admission_events
}
canonical_event_hash_mismatches = []
for chunk in finished:
    chunk_index = int(chunk["chunk_index"])
    expected_hash = hashlib.sha256(
        str(chunk.get("translation") or "").strip().encode("utf-8")
    ).hexdigest()
    recorded_hash = str(
        canonical_event_by_chunk.get(chunk_index, {}).get(
            "canonical_target_hash", ""
        )
    )
    if recorded_hash != expected_hash:
        canonical_event_hash_mismatches.append({
            "chunk_index": chunk_index,
            "recorded": recorded_hash,
            "expected": expected_hash,
        })
canonical_memory_mismatches = []
for index, item in enumerate(long_term):
    if not isinstance(item, dict):
        continue
    chunk_index = item.get("chunk_index")
    translation = str(item.get("translation") or "").strip()
    expected_hash = hashlib.sha256(translation.encode("utf-8")).hexdigest()
    stored_hash = str(item.get("canonical_target_hash") or "")
    saved = next(
        (
            chunk for chunk in finished
            if int(chunk["chunk_index"]) == chunk_index
        ),
        None,
    )
    if (
        not isinstance(chunk_index, int)
        or stored_hash != expected_hash
        or saved is None
        or str(saved.get("translation") or "").strip() != translation
    ):
        canonical_memory_mismatches.append({
            "entry_index": index,
            "chunk_index": chunk_index,
            "stored_hash": stored_hash,
            "expected_hash": expected_hash,
            "matches_saved_chunk": bool(
                saved
                and str(saved.get("translation") or "").strip() == translation
            ),
        })
candidate_rollbacks = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"] == "refinement_candidate_rolled_back"
]
post_rollback_validations = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"] == "post_rollback_final_validation"
]
table_row_recoveries = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"] == "table_row_identity_recovered"
]
table_placeholder_suppressions = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"] == "table_alignment_placeholders_suppressed"
]
readability_reviews = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"] == "persian_readability_review"
]
readability_decisions = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"] == "readability_advisory_decisions"
]
readability_source_suppressions = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"]
    == "readability_advisory_suppressed_source_conflict"
]
minor_objective_readability_promotions = []
unrouted_objective_readability = []
for item in readability_reviews:
    promoted_ids = {
        str(issue_id) for issue_id in item.get("promoted_issue_ids", [])
        if issue_id
    }
    for issue in item.get("unmatched_advisories", []) or []:
        if (
            isinstance(issue, dict)
            and str(issue.get("issue_id", "")) in promoted_ids
            and str(issue.get("severity", "")).casefold() == "minor"
        ):
            minor_objective_readability_promotions.append({
                "chunk_index": item.get("chunk_index"),
                "issue": issue,
            })
    if int(item.get("unrouted_objective_count", 0) or 0):
        unrouted_objective_readability.append({
            "chunk_index": item.get("chunk_index"),
            "count": int(item.get("unrouted_objective_count", 0) or 0),
        })
language_quality_reviews = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"] == "language_quality_review"
]
table_group_recoveries = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"] == "translation_table_group_recovery"
]
invalid_memory_export_consistency = [
    item for item in memory_export_events if not item.get("valid", True)
]
unsafe_accepted_terminology = [
    source for source, target in nouns.items()
    if isinstance(provenance.get(source), dict)
    and provenance[source].get("origin") == "accepted_correction"
    and not is_reusable_terminology_mapping(source, target)
]
latest_critiques = {}
for event in active_events:
    if event["event_type"] == "critique_completed":
        latest_critiques[event["chunk_index"]] = event["payload"]
critic_coverage_violations = []
for chunk_index, payload in latest_critiques.items():
    if not payload.get("valid", True):
        continue
    chunk = chunks_by_index.get(int(chunk_index), {})
    expected = {
        str(item["segment_id"])
        for item in source_segments(str(chunk.get("text", "")))
    }
    checked = {
        str(value)
        for value in payload.get("coverage_checked_segment_ids", []) or []
    }
    uncovered = {
        str(value)
        for value in payload.get("uncovered_source_segment_ids", []) or []
    }
    if checked != expected or bool(payload.get("coverage_complete")) == bool(uncovered):
        critic_coverage_violations.append({
            "chunk_index": chunk_index,
            "expected": sorted(expected),
            "checked": sorted(checked),
            "uncovered": sorted(uncovered),
            "coverage_complete": payload.get("coverage_complete"),
        })
memory_policy_events = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"] == "memory_update_policy"
]
style_sample_policies = [
    {
        "chunk_index": item.get("chunk_index"),
        **dict(item.get("style_sample_policy") or {}),
    }
    for item in memory_policy_events if item.get("style_sample_policy")
]
style_floor = float(
    (config.get("memory", {}) or {}).get("style_min_score", 75.0) or 75.0
)
style_floor_violations = [
    {
        "chunk_index": item.get("chunk_index"),
        "score": float(
            (item.get("selected", {}) or {}).get("score", 0.0) or 0.0
        ),
    }
    for item in style_sample_policies
    if item.get("accepted")
    and float((item.get("selected", {}) or {}).get("score", 0.0) or 0.0)
    < style_floor
]
stored_style_scores = [
    {
        "index": index,
        "score": float(_style_sample_quality(sample).get("score", 0.0)),
        "active": float(_style_sample_quality(sample).get("score", 0.0))
        >= style_floor,
    }
    for index, sample in enumerate(style_samples)
    if isinstance(sample, str)
]
automatic_term_events = [
    {"chunk_index": event["chunk_index"], "event_type": event["event_type"],
     **event["payload"]}
    for event in active_events
    if event["event_type"] in {
        "auto_extraction_completed", "proper_noun_extraction",
    }
]
automatic_term_rejections = [
    {
        "chunk_index": item.get("chunk_index"),
        "event_type": item.get("event_type"),
        "rejected_count": int(
            item.get("rejected_terms", item.get("rejected_count", 0)) or 0
        ),
        "details": item.get("rejected_details", []) or [],
    }
    for item in automatic_term_events
    if int(item.get("rejected_terms", item.get("rejected_count", 0)) or 0)
]
style_paragraph_exclusions = [
    {
        "chunk_index": item.get("chunk_index"),
        "excluded_paragraphs": item.get("style_excluded_paragraphs", []),
        "reason": item.get("style_policy_reason", "legacy"),
        "sample_added": bool(item.get("style_sample_added")),
    }
    for item in memory_policy_events
    if item.get("style_excluded_paragraphs")
]
style_source_genre_exclusions = [
    {
        "chunk_index": item.get("chunk_index"),
        "excluded_paragraphs": item.get(
            "style_source_genre_excluded_paragraphs", []
        ),
        "sample_added": bool(item.get("style_sample_added")),
    }
    for item in memory_policy_events
    if item.get("style_source_genre_excluded_paragraphs")
]
unsafe_style_admission = [
    {
        "chunk_index": item.get("chunk_index"),
        "source_paragraph_index": (
            (item.get("style_sample_policy", {}) or {}).get(
                "source_paragraph_index"
            )
        ),
        "unresolved_issue_ids": unresolved_style_issue_ids(
            item,
            final_quality_by_chunk.get(int(item.get("chunk_index", -1)), {}),
        ),
    }
    for item in memory_policy_events
    if item.get("style_sample_added")
    and style_event_is_active(item)
    and unresolved_style_issue_ids(
        item,
        final_quality_by_chunk.get(int(item.get("chunk_index", -1)), {}),
    )
]
malformed_style_samples = [
    index for index, sample in enumerate(style_samples)
    if isinstance(sample, str)
    and (
        sample.lstrip().startswith(("...", "â€¦"))
        or sample.rstrip().endswith(("...", "â€¦"))
        or re.search(
            r"\([A-Za-z][A-Za-z .'-]{1,80}\)"
            r"[\u0621-\u063a\u0641-\u064a\u066e-\u06d3\u06fa-\u06ff]",
            sample,
        )
        or re.search(r"(?:(?<=\s)|^)\u0640{2,}(?=\s|$)", sample)
        or re.search(
            r"(?P<word>[\u0621-\u063a\u0641-\u064a\u066e-\u06d3"
            r"\u06fa-\u06ff]{4,})\s+(?P=word)",
            sample,
        )
        or repeated_persian_clause_artifacts(sample)
    )
]
english_summary = summary.get("english_summary", "") if isinstance(summary, dict) else ""
persian_summary = summary.get("persian_summary", "") if isinstance(summary, dict) else ""
english_summary_non_ascii_digits = sorted(set(re.findall(
    r"[\u0660-\u0669\u06f0-\u06f9]", english_summary
)))
directional_control_re = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
summary_directional_controls = {
    "english": sorted(set(directional_control_re.findall(english_summary))),
    "persian": sorted(set(directional_control_re.findall(persian_summary))),
}
output_value = str(job.get("output_path") or "").strip()
output = Path(output_value) if output_value else None
if output is not None and not output.is_absolute():
    output = Path("/app") / output
output_valid = bool(
    output is not None
    and output.is_file()
    and output.stat().st_size > 0
    and (
        output.suffix.lower() != ".docx"
        or zipfile.is_zipfile(output)
    )
)
docx_superscript_runs = []
docx_native_rtl_contents = None
docx_contents_layout = None
rendered_text = ""
if output_valid and output.suffix.lower() == ".docx":
    rendered = Document(output)
    rendered_text = "\n".join(
        paragraph.text for paragraph in rendered.paragraphs
    )
    docx_superscript_runs = [
        run.text
        for paragraph in rendered.paragraphs
        for run in paragraph.runs
        if run.font.superscript and run.text
    ]
    has_contents_rows = any(
        chunk["metadata"].get("structure_role") == "contents_entry"
        for chunk in chunks
    )
    with zipfile.ZipFile(output) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
    docx_native_rtl_contents = (
        "<w:bidiVisual" in document_xml if has_contents_rows else None
    )
    if has_contents_rows:
        rtl_tables = [
            match.group(0)
            for match in re.finditer(
                r"<w:tbl>.*?</w:tbl>", document_xml, re.DOTALL
            )
            if "<w:bidiVisual" in match.group(0)
        ]
        table_xml = rtl_tables[0] if rtl_tables else ""
        table_width_match = re.search(
            r'<w:tblW[^>]*w:type="dxa"[^>]*w:w="(\d+)"', table_xml
        ) or re.search(
            r'<w:tblW[^>]*w:w="(\d+)"[^>]*w:type="dxa"', table_xml
        )
        grid_widths = [
            int(value) for value in re.findall(
                r'<w:gridCol[^>]*w:w="(\d+)"', table_xml
            )
        ]
        table_width = int(table_width_match.group(1)) if table_width_match else 0
        docx_contents_layout = {
            "rtl": bool(table_xml and "<w:bidiVisual" in table_xml),
            "right_aligned": '<w:jc w:val="right"' in table_xml,
            "fixed_layout": '<w:tblLayout w:type="fixed"' in table_xml,
            "table_width_twips": table_width,
            "grid_widths_twips": grid_widths,
            "full_width": bool(
                table_width >= 9000
                and len(grid_widths) == 2
                and sum(grid_widths) == table_width
                and grid_widths[0] > grid_widths[1] * 4
            ),
        }
malformed_latin_scholarly_abbreviations = sorted(set(
    re.findall(
        r"(?i)(?:\b(?:e\.\s+g\.|i\.\s+e\.|c\.\s+f\.)"
        r"|\b(?:e\.g\.|i\.e\.|cf\.|ibid\.|viz\.)ØŒ)",
        rendered_text,
    )
))

# Confirm the running container, not merely Git, contains v10.34 behavior.
root = Path(tarjomeh.__file__).resolve().parent
pipeline_source = (root / "core" / "pipeline.py").read_text(encoding="utf-8")
client_source = (root / "core" / "llm_client.py").read_text(encoding="utf-8")
structure_source = (root / "quality" / "structure_audit.py").read_text(
    encoding="utf-8"
)
parser_source = (root / "parsers" / "pdf_parser.py").read_text(encoding="utf-8")
proper_source = (root / "memory" / "proper_nouns.py").read_text(encoding="utf-8")
manager_source = (root / "memory" / "manager.py").read_text(encoding="utf-8")
config_source = (root / "core" / "config.py").read_text(encoding="utf-8")
summary_source = (root / "memory" / "bilingual_summary.py").read_text(
    encoding="utf-8"
)
prompts_source = (root / "core" / "prompts.py").read_text(encoding="utf-8")
integrity_source = (root / "quality" / "integrity.py").read_text(encoding="utf-8")
exporter_source = (root / "exporters" / "docx_exporter.py").read_text(
    encoding="utf-8"
)
typography_source = (root / "persian" / "typography.py").read_text(
    encoding="utf-8"
)
critique_source = (root / "quality" / "critique.py").read_text(
    encoding="utf-8"
)
web_source = (root / "web" / "app.py").read_text(encoding="utf-8")
research_source = (root / "context" / "book_researcher.py").read_text(
    encoding="utf-8"
)
database_source = (root / "jobs" / "database.py").read_text(encoding="utf-8")
structure_version_match = re.search(
    r"structure_version\s*:\s*int\s*=\s*(\d+)", parser_source
)
note_probe_text, note_probe_report = restore_source_note_markers(
    "First source sentence.2 Second source sentence.",
    "نخستین جملهٔ مقصد. دومین جملهٔ مقصد.",
)
citation_probe_text, citation_probe_changes = normalize_citation_house_style_text(
    "(see Jessop 1990, 2002)"
)
runtime_markers = {
    "runtime_release_contract": runtime_manifest.get("release") == "v10.34.0",
    "runtime_behavior_contract": all(runtime_probes.values()),
    "canonical_paragraph_identity": (
        "canonical_paragraph_identity" in runtime_manifest.get("capabilities", {})
        and "canonical_chunk_paragraphs_v1" in pipeline_source
    ),
    "reconstructed_identity_review_only": (
        "reconstructed_identity_review_only"
        in runtime_manifest.get("capabilities", {})
        and "_record_reconstructed_paragraph_identity_review" in pipeline_source
    ),
    "verified_source_coverage": (
        "verified_source_coverage" in runtime_manifest.get("capabilities", {})
        and "coverage_checked_segment_ids" in critique_source
    ),
    "final_identifier_admission": (
        "final_identifier_admission" in runtime_manifest.get("capabilities", {})
        and "Final source-identifier admission failed" in pipeline_source
    ),
    "paragraph_scoped_refiner_salvage": (
        "paragraph_scoped_refiner_salvage"
        in runtime_manifest.get("capabilities", {})
        and "_paragraph_scoped_span_start" in pipeline_source
    ),
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
    "unicode_source_matching": "source_term_pattern" in proper_source,
    "localized_structure_episodes": "_structural_episodes" in structure_source,
    "identifier_label_deduplication": (
        "_LOCALIZED_IDENTIFIER_LABEL_SUFFIX_RE" in integrity_source
    ),
    "grounded_phrase_typography_variants": (
        "typography variants" in integrity_source
    ),
    "recovery_protocol_evidence": (
        '"recovery_assembly": True' in pipeline_source
    ),
    "named_instrument_coverage": (
        "_high_confidence_instrument_candidates" in pipeline_source
    ),
    "pre_memory_canonicalization": (
        "_canonicalize_chunk_inline_originals" in pipeline_source
    ),
    "memory_export_consistency": (
        "memory_export_consistency" in pipeline_source
    ),
    "source_grounded_inline_categories": (
        "source_grounded_entity" in proper_source
        and "legal_instrument" in proper_source
    ),
    "source_confirmed_superscripts": (
        "_line_superscript_markers" in parser_source
        and "source_superscript_spans" in exporter_source
    ),
    "catalog_language_policy": "_CATALOG_NAME_LINE_RE" in integrity_source,
    "structural_evidence_excerpts": '"source_excerpt"' in structure_source,
    "bounded_source_entities": "_bounded_source_entity" in pipeline_source,
    "exact_bilingual_entity_target": "observed_bilingual_target" in proper_source,
    "repeated_word_review": "repeated_persian_word_artifacts" in integrity_source,
    "catalog_punctuation_preserved": "_LATIN_CATALOG_LINE_RE" in typography_source,
    "identifier_label_identity": "_identifier_label_identity" in integrity_source,
    "persian_announcement_inflection": "_is_announcement_noun" in structure_source,
    "structured_refiner_failure": "structured_response_invalid" in pipeline_source,
    "contents_row_parser": "_split_contents_entry_lines" in parser_source,
    "contents_docx_layout": "add_contents_table" in exporter_source,
    "native_rtl_contents_table": "w:bidiVisual" in exporter_source,
    "exact_anchor_only_repair": "_anchor_only_repair_is_valid" in pipeline_source,
    "source_grounded_language_repair": (
        "repair_source_grounded_language_artifacts" in integrity_source
    ),
    "final_rendered_language_repair": (
        "repair_document_source_grounded_language_artifacts" in pipeline_source
        and "final_rendered_language_repair" in pipeline_source
    ),
    "tatweel_separator_audit": (
        "tatweel_separator_artifacts" in integrity_source
    ),
    "structure_version_current": bool(
        structure_version_match and int(structure_version_match.group(1)) >= 4
    ),
    "fluent_academic_contract": (
        runtime_manifest.get("capabilities", {}).get(
            "source_aware_fluency_admission"
        ) is True
        and runtime_manifest.get("capabilities", {}).get(
            "verified_source_coverage"
        ) is True
    ),
    "paragraph_local_duplicate_evidence": (
        "source_unjustified_repeated_word_artifacts" in integrity_source
    ),
    "targeted_language_repair": "translation_language_repair" in pipeline_source,
    "final_anchor_ordering": (
        pipeline_source.find("# Canonicalize the accepted text before measuring") >= 0
        and pipeline_source.find("missing_required_anchors = [") >= 0
        and pipeline_source.find("# Canonicalize the accepted text before measuring")
        < pipeline_source.find("missing_required_anchors = [")
    ),
    "foreign_script_audit": "foreign_script_artifacts" in integrity_source,
    "markup_parenthesis_ezafe_audit": (
        "markup_wrapper_artifacts" in integrity_source
        and "parenthesis_artifacts" in integrity_source
        and "detached_ezafe_artifacts" in integrity_source
    ),
    "grounded_objective_fluency_routing": (
        "_grounded_objective_language_issues" in pipeline_source
        and "grounded_objective_defect" in pipeline_source
    ),
    "repeated_clause_review": (
        "source_unjustified_repeated_clause_artifacts" in integrity_source
        and '"repeated_clause_count"' in pipeline_source
    ),
    "configured_memory_threshold": (
        "final_critique_below_configured_threshold" in pipeline_source
    ),
    "safe_automatic_terminology": (
        "is_safe_low_authority_mapping" in proper_source
        and "noncanonical_automatic_mapping" in proper_source
    ),
    "post_fragment_anchor_reconciliation": (
        "post_fragment_inserted_count" in pipeline_source
        and '"after_fragment_cleanup"' in pipeline_source
    ),
    "source_family_dehyphenation": (
        "morphological_family_observed_elsewhere_in_source" in parser_source
    ),
    "full_width_rtl_contents": (
        'w:tblLayout' in exporter_source
        and 'total_twips' in exporter_source
        and 'w:gridCol' in exporter_source
    ),
    "minimal_automatic_term_evidence": (
        "has_minimal_automatic_term_evidence" in proper_source
        and "insufficient_minimal_lexical_evidence" in pipeline_source
    ),
    "paragraph_level_style_admission": (
        "_chunk_style_policy" in pipeline_source
        and "style_excluded_paragraphs" in manager_source
    ),
    "summary_directional_control_filter": (
        "_DIRECTIONAL_CONTROL_RE" in summary_source
    ),
    "accuracy_outranks_fluency": (
        "Accuracy and completeness outrank fluency" in prompts_source
    ),
    "atomic_local_salvage": (
        "_salvage_local_refinement_edits" in pipeline_source
        and "local_edit_committed" in pipeline_source
    ),
    "source_critic_salvage_rollback": (
        "refinement_salvage_rolled_back" in pipeline_source
    ),
    "bounded_persian_readability": (
        "review_persian_readability" in critique_source
        and "target_only_advisory" in pipeline_source
    ),
    "promoted_persian_readability": (
        "_promote_objective_readability_issues" in pipeline_source
        and "readability_advisory_decisions" in pipeline_source
    ),
    "objective_minor_readability": (
        "_OBJECTIVE_GRAMMAR_RATIONALE_RE" in pipeline_source
        and "objective_minor_grammar" in pipeline_source
    ),
    "final_body_readability": (
        "review_persian_readability" in critique_source
        and "readability_reviewed" in pipeline_source
    ),
    "conservative_memory_reliability": (
        "_DISQUALIFYING_RELIABILITY_REASONS" in pipeline_source
        and "disqualifying_reliability_reasons" in pipeline_source
    ),
    "derived_false_mi_repair": "segments = joined.split" in typography_source,
    "flattened_note_markers": "available_note_markers" in integrity_source,
    "grouped_table_recovery": "_table_recovery_groups" in pipeline_source,
    "latest_generation_qa": "current_events_by_chunk" in web_source,
    "context_bound_memory_rejected": (
        "_context_bound_persian_target" in proper_source
    ),
    "authored_diacritics_preserved": (
        "_PERSIAN_COMBINING_MARK_RE" in typography_source
    ),
    "hazm_false_mi_repaired": (
        "_repair_false_mi_splits" in typography_source
    ),
    "paragraph_safe_anchor_repair": (
        "before_paragraphs" in pipeline_source
        and "stripped_paragraphs" in pipeline_source
    ),
    "backcheck_threshold_reported_as_advisory": (
        "below_similarity_threshold=" in web_source
        and "policy=" in web_source
    ),
    "full_candidate_monotonic_rollback": (
        "refinement_candidate_rolled_back" in pipeline_source
    ),
    "late_candidate_readability": "readability_reviewed" in pipeline_source,
    "source_paired_dash_audit": (
        "unbalanced_explanatory_dash_count" in pipeline_source
    ),
    "canonical_pre_memory_typography": (
        "final_canonical_admission" in pipeline_source
    ),
    "citation_tail_protection": "_AUTHOR_YEAR_CITATION_RE" in typography_source,
    "latin_scholarly_abbreviations": (
        "_LATIN_SCHOLARLY_ABBREVIATION_RE" in typography_source
    ),
    "semantic_tatweel_dash": (
        "_SEMANTIC_TATWEEL_SEPARATOR_RE" in typography_source
    ),
    "contextual_source_term_gate": "_SOURCE_NONTERM_TRAILERS" in proper_source,
    "table_line_identity": "_target_paragraphs_for_alignment" in pipeline_source,
    "table_placeholder_suppression": (
        "suppress_empty_target_export" in pipeline_source
        and "suppress_empty_target_export" in exporter_source
    ),
    "atomic_regression_salvage": (
        "_decisions_without_regressed_edits" in pipeline_source
    ),
    "second_stage_transactional_replay": (
        "_recover_non_regressed_local_edits" in pipeline_source
        and "partial_non_regressed_replay" in pipeline_source
    ),
    "typed_dash_attachment_audit": (
        "_SPACED_EM_DASH_RE" in pipeline_source
        and "persian_object_marker_detached_by_dash" in pipeline_source
    ),
    "source_scoped_originals": (
        "source_applicable_originals" in pipeline_source
        and "authorized_original_not_grounded_in_source_paragraph"
        in (root / "core" / "term_notes.py").read_text(encoding="utf-8")
    ),
    "role_scoped_entity_memory": (
        "semantic_role" in proper_source
        and "source_target_number_scope_mismatch" in proper_source
    ),
    "normalized_english_summary_digits": (
        "_normalize_english_summary_digits" in summary_source
    ),
    "long_llm_call_progress": "llm_call_in_progress" in pipeline_source,
    "research_role_evidence": (
        "semantic_role" in research_source
        and "exact_evidence_quote" in research_source
    ),
    "post_rollback_validation": (
        "post_rollback_final_validation" in pipeline_source
    ),
    "source_announcement_fidelity": (
        "announced_count_lexical_mismatch" in structure_source
        and "source-authored announced quantity" in prompts_source
    ),
    "short_span_duplicate_guard": (
        "newly_source_unjustified_repeated_adjacent_spans"
        in integrity_source
        and '"repeated_adjacent_span_count"' in pipeline_source
    ),
    "local_predicate_guard": (
        "local_predicate_evidence_removed" in pipeline_source
        and "predicate_regressions" in pipeline_source
    ),
    "source_relative_governed_repetition": (
        "newly_source_unjustified_repeated_governed_spans"
        in integrity_source
        and '"repeated_governed_span_count"' in pipeline_source
    ),
    "coordinated_proposition_review": (
        "matrix action, coordinated actions" in prompts_source
        and "verify every coordinated source member separately" in prompts_source
    ),
    "terminology_authority_classes": (
        "_mapping_authority_class" in proper_source
        and "recurring_advisory" in proper_source
        and "canonical_reviewed" in proper_source
    ),
    "quality_ranked_style_memory": (
        "_style_sample_quality" in manager_source
        and "weaker_style_sample_replaced" in manager_source
    ),
    "style_memory_floor": (
        "style_min_score" in config_source
        and "style_score_below_floor" in manager_source
    ),
    "conceptual_family_contract": (
        "recurring coordinated conceptual series" in prompts_source
    ),
    "attributable_advisory_research": (
        "identity_supported" in research_source
        and "supporting_excerpts" in research_source
        and "advisory_context_only" in research_source
    ),
    "readability_source_authority": (
        "readability_advisory_suppressed_source_conflict"
        in pipeline_source
    ),
    "structural_reference_localization": (
        "source_structural_reference" in integrity_source
    ),
    "matrix_predicate_prompt": (
        "matrix predicate" in prompts_source
        and "relative clause" in prompts_source
    ),
    "source_faithful_version_recovery": (
        "refinement_best_source_version_restored" in pipeline_source
        and "_best_source_faithful_version" in pipeline_source
    ),
    "grounded_memory_quarantine": (
        "unresolved_grounded_quality_issue" in pipeline_source
    ),
    "sense_scoped_entity_memory": (
        "_strip_redundant_entity_original" in proper_source
        and "_mapping_applies_to_source" in proper_source
    ),
    "term_specific_research_evidence": (
        "term_supporting_excerpts" in research_source
        and "term_supported" in research_source
    ),
    "durable_worker_lease": (
        "CREATE TABLE IF NOT EXISTS job_workers" in database_source
        and "worker_pause_acknowledged" in pipeline_source
        and "_start_lease_heartbeat" in pipeline_source
        and "request_job_pause" in web_source
    ),
    "checkpoint_preview_atomic_publish": (
        "_publish_chapter_checkpoint" in pipeline_source
        and "partial_path.replace(output_path)" in pipeline_source
        and "persist_job_output=False" in pipeline_source
        and "complete_chapter_checkpoint" in pipeline_source
    ),
    "durable_chapter_checkpoint_recovery": (
        "chapter_checkpoint=chapter_checkpoint" in pipeline_source
        and "_recover_pending_chapter_checkpoint" in pipeline_source
        and "save_pending_chapter_checkpoint" in database_source
        and "fail_chapter_checkpoint" in database_source
        and "pending_export" in database_source
    ),
    "canonical_final_quality_authority": (
        "_canonical_final_quality_record" in pipeline_source
        and "final_quality_admission" in pipeline_source
        and "final_quality_authority" in pipeline_source
    ),
    "bounded_source_obligation_repair": (
        "source_fidelity_findings" in pipeline_source
        and "resolved_source_issue_ids" in pipeline_source
        and "_language_quality_does_not_regress" in pipeline_source
    ),
    "diacritic_aware_duplicate_repair": (
        "_repeat_token_key" in integrity_source
        and "adjacent_duplicate_phrase" in integrity_source
    ),
    "observable_llm_stage": (
        '"phase": "started"' in client_source
        and "llm_call_started" in pipeline_source
        and "get_worker_lease(job_id)" in web_source
    ),
    "monotonic_source_obligations": (
        "_source_obligation_identity" in pipeline_source
        and "refinement_atomic_recovery_rolled_back" in pipeline_source
    ),
    "short_token_duplicate_repair": (
        "lexical_length < 2" in integrity_source
        and "adjacent_duplicate_phrase" in integrity_source
    ),
    "transactional_summary_admission": (
        "candidate_quality" in summary_source
        and "candidate_committed" in manager_source
    ),
    "complete_worker_lifecycle": (
        "worker_lease_released" in database_source
        and "worker_pause_requested" in database_source
        and "state!='released'" in database_source
    ),
    "citation_bound_governed_repetition": "contiguous_phrase" in integrity_source,
    "source_grounded_original_preservation": (
        "preserved_source_grounded" in (
            root / "core" / "term_notes.py"
        ).read_text(encoding="utf-8")
        and "merge_inline_english_original_audits" in pipeline_source
    ),
    "representative_style_source_gate": (
        "source_genre_not_representative_of_body_voice" in manager_source
    ),
    "durable_chunk_terminal_failure": (
        "chunk_terminal_failure" in pipeline_source
    ),
    "source_monotonic_structure_admission": (
        "_actionable_structure_findings" in pipeline_source
        and "refinement_source_structure_admission" in pipeline_source
        and "final_source_structure_admission" in pipeline_source
    ),
    "paragraph_scoped_style_evidence": (
        "source_paragraph_index" in manager_source
    ),
    "source_grounded_units": "_MEASUREMENT_UNIT_TOKENS" in integrity_source,
    "latin_title_punctuation": "_LATIN_PUNCTUATED_RUN_RE" in typography_source,
    "unique_note_marker_recovery": (
        "restore_source_note_markers" in integrity_source
        and "note_marker_repair_count" in pipeline_source
    ),
    "final_canonical_admission": (
        "source_bound_artifact_recovery" in pipeline_source
        and "final_canonical_admission" in pipeline_source
    ),
    "independent_refiner_salvage": (
        "coherent_local_edits_committed" in pipeline_source
        and "local_edit_committed" in pipeline_source
    ),
    "two_stage_observed_entity_authority": (
        "_SEMANTICALLY_TRANSLATED_ENTITY_CATEGORIES" in proper_source
        and "observed_entity_advisory" in proper_source
    ),
    "bilingual_summary_source_guard": (
        "_summary_lexical_near_misses" in manager_source
        and "bilingual_summary_structure_mismatch" in manager_source
    ),
    "source_genre_style_filter": (
        "style_source_genre_excluded_paragraphs" in manager_source
    ),
    "aligned_sentence_note_recovery": (
        "²" in note_probe_text
        and note_probe_report.get("repair_count") == 1
        and note_probe_report.get("repairs", [{}])[0].get("type")
        == "aligned_sentence_terminal_note_marker"
    ),
    "boundary_safe_local_salvage": (
        "_replace_local_span_with_boundary_guard" in pipeline_source
        and "boundary_deduplication" in pipeline_source
    ),
    "grounded_objective_language_repair": (
        "_grounded_objective_language_issues" in pipeline_source
        and "resolved_objective_issue_ids" in pipeline_source
    ),
    "canonical_citation_memory_text": (
        citation_probe_text == "(ر.ک. Jessop 1990, 2002)"
        and bool(citation_probe_changes)
        and "canonical_target_hash" in pipeline_source
        and "canonical_target_hash" in manager_source
    ),
    "first_person_dedication_style_filter": bool(
        _NONREPRESENTATIVE_STYLE_SOURCE_RE.search(
            "I dedicate this book to the memory of a colleague."
        )
        and not _NONREPRESENTATIVE_STYLE_SOURCE_RE.search(
            "The analysis is dedicated to explaining institutional change."
        )
    ),
    "academic_ezafe_examples": (
        prompts_source.count("هٔ") >= 2
        and "never add blanket diacritics" in prompts_source
    ),
}

hard = []
review = []
if not all(runtime_markers.values()):
    hard.append("running_container_is_not_complete_v1034")
if checkpoint_preview_missing:
    hard.append("paused_checkpoint_preview_missing")
if job.get("status") == "paused" and checkpoint_pending:
    hard.append("paused_checkpoint_has_unpublished_pending_intent")
if checkpoint_publication_error:
    hard.append(checkpoint_publication_error)
if missed_requested_boundaries:
    hard.append("completed_requested_boundary_was_not_checkpointed")
if checkpoint_pending and completed_after_pending:
    review.append("late_checkpoint_recovery_preserved_later_chunks")
if canonical_document_identity and not canonical_document_identity.get(
    "lexically_identical", False
):
    hard.append("assembled_text_differs_from_canonical_translation")
if missing_paragraph_identity:
    hard.append("missing_canonical_paragraph_identity")
if invalid_paragraph_identity:
    hard.append("invalid_canonical_paragraph_identity")
if critic_coverage_violations:
    hard.append("critic_source_coverage_not_verified")
if job.get("status") == "completed" and not canonical_document_identity:
    hard.append("missing_canonical_document_identity")
if empty_finished:
    hard.append("empty_finished_chunk")
if missing_flow:
    hard.append("missing_finished_chunk_flow")
if missing_final_canonical_admission:
    hard.append("missing_final_canonical_admission")
if missing_final_candidate_selection:
    hard.append("missing_final_candidate_selection")
if final_candidate_hash_mismatches:
    hard.append("final_candidate_selection_hash_mismatch")
if final_candidate_policy_mismatches:
    hard.append("final_candidate_selection_not_atomic_policy_v2")
if candidate_quality_hash_mismatches:
    hard.append("final_quality_not_bound_to_canonical_candidate")
if non_atomic_final_quality:
    hard.append("final_quality_not_committed_with_canonical_checkpoint")
if pending_finished_source_obligations:
    hard.append("finished_chunk_has_pending_source_obligation_recovery")
if unsafe_admission:
    hard.append("quarantined_translation_entered_memory")
if unsafe_grounded_memory:
    hard.append("grounded_quality_issue_entered_reliable_memory")
if authority_violations:
    hard.append("final_quality_memory_style_authority_disagreement")
if finished_unrepaired:
    hard.append("unrepaired_quarantine_finished")
if exact_replay_violations:
    hard.append("transport_replay_changed_payload")
if quality_contract_violations:
    hard.append("quality_attempt_2_changed_contract")
if zero_visible_success:
    hard.append("successful_attempt_without_visible_output")
if unmatched_llm_starts and str(job.get("status", "")) not in {
    "pending", "running", "processing", "pausing",
}:
    hard.append("terminal_job_has_unmatched_llm_start")
if invalid_nouns:
    hard.append("invalid_proper_noun_mapping")
if unsafe_low_authority_nouns:
    hard.append("unsafe_authoritative_automatic_mapping")
if unsafe_automatic_terms:
    hard.append("context_expanded_automatic_terminology")
if unsafe_automatic_nouns:
    hard.append("unsafe_automatic_entity_mapping")
if malformed_source_entity_nouns:
    hard.append("malformed_source_entity_span")
if unsubstantiated_introduced:
    hard.append("source_only_first_occurrence_state")
if review_events_without_reason:
    hard.append("review_event_without_reason")
if malformed_style_samples:
    hard.append("malformed_text_entered_style_memory")
if unsafe_style_admission:
    hard.append("unresolved_critique_entered_style_memory")
if style_floor_violations:
    hard.append("below_floor_style_sample_admitted")
if any(summary_directional_controls.values()):
    hard.append("directional_control_entered_summary_memory")
if english_summary_non_ascii_digits:
    hard.append("non_ascii_digit_entered_english_summary")
if invalid_memory_export_consistency:
    hard.append("invalid_memory_export_reconciliation")
if missing_canonical_admission_events:
    hard.append("missing_atomic_final_canonical_admission")
if canonical_event_hash_mismatches:
    hard.append("canonical_event_text_hash_mismatch")
if canonical_memory_mismatches:
    hard.append("canonical_layer3_text_mismatch")
if unsafe_accepted_terminology:
    hard.append("contextual_accepted_correction_entered_global_memory")
if invalid_authority_promotions:
    hard.append("terminology_promoted_without_independent_evidence")
if finished and not output_valid:
    hard.append("output_missing_or_invalid")
if int(protocol.get("remaining_artifact_count", 0) or 0):
    hard.append("protocol_artifact_in_export")
if int(final_text.get("unresolved_identifier_count", 0) or 0):
    hard.append("unresolved_identifier_in_export")
if int(final_identifiers.get("unresolved_count", 0) or 0):
    hard.append("unresolved_identifier_reconciliation")
if docx_native_rtl_contents is False:
    hard.append("contents_table_not_native_rtl")
if docx_contents_layout is not None and not all(
    docx_contents_layout.get(key)
    for key in ("rtl", "right_aligned", "fixed_layout", "full_width")
):
    hard.append("contents_table_not_full_width_rtl")

if needs_review:
    review.append("chunk_needs_review")
if reconstructed_paragraph_identity:
    review.append("paragraph_identity_reconstructed")
if active_counts["qa_unavailable"]:
    review.append("qa_unavailable")
if active_counts["integrity_final_failed"]:
    review.append("integrity_final_failed")
if final_text.get("review_required"):
    review.append("final_text_review_required")
if job.get("error_message"):
    review.append("job_error_message")
if missing_entity_anchors:
    review.append("missing_first_occurrence_entity_anchor")
if int(final_rendered_repair.get("rejected_paragraph_count", 0) or 0):
    review.append("final_rendered_repair_rejected")
if language_quality_reviews:
    review.append("objective_final_language_artifact")
if rejected_summary_candidates:
    review.append("bilingual_summary_candidate_rejected")
if finished_prose and not style_samples:
    review.append("style_samples_empty_despite_finished_prose")
if research_identity_gaps or research_term_evidence_gaps:
    review.append("research_evidence_metadata_incomplete")
if unrouted_objective_readability:
    review.append("objective_readability_finding_not_routed")
if malformed_latin_scholarly_abbreviations:
    review.append("malformed_latin_scholarly_abbreviation")
if int(original_audit.get("unapproved_ungrounded_count", 0) or 0):
    review.append("unapproved_ungrounded_english_original")
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
        "worker_lease": worker_lease,
        "worker_lifecycle_events": worker_events,
    },
    "runtime_markers": runtime_markers,
    "runtime_manifest": runtime_manifest,
    "runtime_behavior_probes": runtime_probes,
    "pipeline": {
        "event_counts": dict(counts),
        "active_event_counts": dict(active_counts),
        "missing_finished_chunk_flow": missing_flow,
        "source_bound_recovery_events": source_bound_recovery_events,
        "chapter_checkpoint_preview": {
            "state_version": (checkpoints or {}).get("version", 1),
            "reached_positions": checkpoint_positions,
            "pending": checkpoint_pending,
            "publications": checkpoint_publications,
            "latest_publication": latest_checkpoint_publication,
            "output_path": checkpoint_output_path,
            "output_exists": bool(
                checkpoint_output_path
                and os.path.isfile(checkpoint_output_path)
            ),
            "missing_while_paused": checkpoint_preview_missing,
            "publication_error": checkpoint_publication_error,
            "missed_requested_boundaries": missed_requested_boundaries,
            "completed_after_pending": completed_after_pending,
        },
        "missing_final_canonical_admission": missing_final_canonical_admission,
        "final_candidate_selections": final_candidate_events,
        "missing_final_candidate_selection": missing_final_candidate_selection,
        "final_candidate_hash_mismatches": final_candidate_hash_mismatches,
        "final_candidate_policy_mismatches": final_candidate_policy_mismatches,
        "candidate_quality_hash_mismatches": candidate_quality_hash_mismatches,
        "non_atomic_final_quality": non_atomic_final_quality,
        "source_obligation_reuse_events": source_obligation_reuse_events,
        "pending_finished_source_obligations": (
            pending_finished_source_obligations
        ),
        "canonical_paragraph_identity": paragraph_identity,
        "missing_paragraph_identity": missing_paragraph_identity,
        "invalid_paragraph_identity": invalid_paragraph_identity,
        "reconstructed_paragraph_identity": reconstructed_paragraph_identity,
        "critic_coverage_violations": critic_coverage_violations,
        "quarantined": sorted(quarantined),
        "repaired": sorted(repaired),
        "unrepaired": unrepaired,
        "unsafe_memory_admission": unsafe_admission,
        "unsafe_grounded_reliable_memory": unsafe_grounded_memory,
        "final_quality_admissions": final_quality_events,
        "authority_violations": authority_violations,
        "source_obligation_repairs": source_repair_events,
        "atomic_source_rollbacks": atomic_source_rollbacks,
        "summary_policy_events": summary_policy_events,
        "rejected_summary_candidates": rejected_summary_candidates,
        "source_entity_coverage": entity_coverage,
        "missing_first_occurrence_entity_anchors": missing_entity_anchors,
        "memory_export_consistency": memory_export_events,
        "invalid_memory_export_consistency": invalid_memory_export_consistency,
        "canonical_admission_events": canonical_admission_events,
        "missing_canonical_admission_events": missing_canonical_admission_events,
        "canonical_event_hash_mismatches": canonical_event_hash_mismatches,
        "canonical_layer3_mismatches": canonical_memory_mismatches,
        "full_candidate_rollbacks": candidate_rollbacks,
        "post_rollback_validations": post_rollback_validations,
        "table_row_identity_recoveries": table_row_recoveries,
        "table_placeholder_suppressions": table_placeholder_suppressions,
        "persian_readability_reviews": readability_reviews,
        "readability_advisory_decisions": readability_decisions,
        "readability_source_conflict_suppressions": (
            readability_source_suppressions
        ),
        "minor_objective_readability_promotions": (
            minor_objective_readability_promotions
        ),
        "unrouted_objective_readability": unrouted_objective_readability,
        "language_quality_reviews": language_quality_reviews,
        "chunk_terminal_failures": terminal_failures,
        "table_group_recoveries": table_group_recoveries,
        "unsafe_style_admission": unsafe_style_admission,
        "style_paragraph_exclusions": style_paragraph_exclusions,
        "style_source_genre_exclusions": style_source_genre_exclusions,
        "automatic_term_rejections": automatic_term_rejections,
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
        "in_progress_events": [
            {
                "timestamp": event["timestamp"],
                "chunk_index": event["chunk_index"],
                **event["payload"],
            }
            for event in active_progress_events
        ],
        "retry_pairs": retry_pairs,
        "exact_replay_violations": exact_replay_violations,
        "quality_contract_violations": quality_contract_violations,
        "expected_helper_recovery": expected_helper_recovery,
        "started_events": len(active_start_events),
        "unmatched_started_calls": unmatched_llm_starts,
        "long_attempts_10m_or_more": long_llm_attempts,
    },
    "memory": {
        "proper_nouns": len(nouns),
        "invalid_proper_nouns": invalid_nouns,
        "unsafe_authoritative_automatic_mappings": unsafe_low_authority_nouns,
        "unsafe_automatic_terminology": unsafe_automatic_terms,
        "unsafe_accepted_terminology": unsafe_accepted_terminology,
        "unsafe_grounded_reliable_memory": unsafe_grounded_memory,
        "authority_classes": dict(authority_classes),
        "invalid_authority_promotions": invalid_authority_promotions,
        "deferred_automatic_mappings": deferred_automatic_nouns,
        "unsafe_automatic_entities": unsafe_automatic_nouns,
        "malformed_source_entity_spans": malformed_source_entity_nouns,
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
        "style_sample_records": len(style_sample_records),
        "representative_style_records": sum(
            bool(item.get("representative"))
            for item in style_sample_records if isinstance(item, dict)
        ),
        "style_sample_policies": style_sample_policies,
        "style_minimum_score": style_floor,
        "style_floor_violations": style_floor_violations,
        "stored_style_scores": stored_style_scores,
        "malformed_style_samples": malformed_style_samples,
        "style_profile": bool(state.get("style_profile")),
        "summary_directional_controls": summary_directional_controls,
        "english_summary_non_ascii_digits": english_summary_non_ascii_digits,
        "book_context": bool(state.get("book_context")),
    },
    "research": {
        "status": research.get("status"),
        "sources": research.get("source_count", len(research.get("sources", []))),
        "terms": len(research.get("terms", [])),
        "identity_metadata_gaps": research_identity_gaps,
        "term_evidence_gaps": research_term_evidence_gaps,
    },
    "checkpoint": checkpoints,
    "structure": structure,
    "canonical_document_identity": canonical_document_identity,
    "protocol_audit": protocol,
    "final_text_audit": final_text,
    "final_rendered_language_repair": final_rendered_repair,
    "final_identifier_reconciliation": final_identifiers,
        "original_fragment_reconciliation": fragment_reconciliation,
        "english_original_audit": original_audit,
    "review_events_without_reason": review_events_without_reason,
    "output": {
        "path": str(output) if output is not None else None,
        "exists": bool(output is not None and output.exists()),
        "valid": output_valid,
        "size": (
            output.stat().st_size
            if output is not None and output.exists()
            else 0
        ),
        "superscript_runs": docx_superscript_runs,
        "native_rtl_contents": docx_native_rtl_contents,
        "contents_layout": docx_contents_layout,
        "malformed_latin_scholarly_abbreviations": (
            malformed_latin_scholarly_abbreviations
        ),
    },
    "verdict": verdict,
    "hard_failures": hard,
    "review_signals": review,
}
json_path = Path(f"/app/jobs/uploads/{job_id}_v1034_companion_audit.json")
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
txt_path = Path(f"/app/jobs/uploads/{job_id}_v1034_companion_audit.txt")

lines = [
    "Tarjomeh v10.34 Independent Companion Audit",
    f"Job: {job_id}",
    f"Status: {job.get('status')}",
    "",
    "RUNTIME",
    f"Markers: {runtime_markers}",
    f"Manifest: {runtime_manifest}",
    f"Behavior probes: {runtime_probes}",
    f"Canonical document identity: {canonical_document_identity or 'pending'}",
    "",
    "PIPELINE",
    f"Chunk statuses: {summary_payload['job']['chunks']}",
    f"Finished: {len(finished)} / {len(chunks)}",
    f"Missing flow: {missing_flow or 'none'}",
    f"Source-bound recovery events: {len(source_bound_recovery_events)}",
    "Missing final canonical admission: "
    f"{missing_final_canonical_admission or 'none'}",
    "Missing final candidate selections: "
    f"{missing_final_candidate_selection or 'none'}",
    "Final candidate hash mismatches: "
    f"{final_candidate_hash_mismatches or 'none'}",
    "Final candidate policy mismatches: "
    f"{final_candidate_policy_mismatches or 'none'}",
    "Final quality/candidate hash mismatches: "
    f"{candidate_quality_hash_mismatches or 'none'}",
    "Non-atomic final-quality checkpoints: "
    f"{non_atomic_final_quality or 'none'}",
    "Resumed source-obligation candidates: "
    f"{len(source_obligation_reuse_events)}",
    "Finished chunks with pending source-obligation recovery: "
    f"{pending_finished_source_obligations or 'none'}",
    f"Canonical event/hash mismatches: "
    f"{canonical_event_hash_mismatches or 'none'}",
    f"Canonical Layer-3 mismatches: {canonical_memory_mismatches or 'none'}",
    f"Quarantined: {sorted(quarantined) or 'none'}",
    f"Repaired: {sorted(repaired) or 'none'}",
    f"Unsafe memory admission: {unsafe_admission or 'none'}",
    f"Grounded issues admitted as reliable: "
    f"{unsafe_grounded_memory or 'none'}",
    f"Final quality admissions: {len(final_quality_events)}",
    f"Memory/style authority disagreements: {authority_violations or 'none'}",
    f"Source-obligation repair batches: {source_repair_events or 'none'}",
    f"Worker lease: {worker_lease or 'none'}",
    f"Worker lifecycle events: {worker_events or 'none'}",
    f"Atomic source-regression rollbacks: "
    f"{atomic_source_rollbacks or 'none'}",
    f"Rejected bilingual-summary candidates: "
    f"{rejected_summary_candidates or 'none'}",
    f"Source-entity coverage events: {len(entity_coverage)}",
    f"Missing entity anchors: {missing_entity_anchors or 'none'}",
    f"Memory/export consistency events: {len(memory_export_events)}",
    f"Invalid memory/export reconciliation: "
    f"{invalid_memory_export_consistency or 'none'}",
    f"Style samples admitted with unresolved critique: "
    f"{unsafe_style_admission or 'none'}",
    f"Paragraph-level style exclusions: {style_paragraph_exclusions or 'none'}",
    f"Source-genre style exclusions: {style_source_genre_exclusions or 'none'}",
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
    f"LLM start events: {len(active_start_events)}",
    f"LLM in-progress heartbeats: {len(active_progress_events)}",
    f"Unmatched active LLM starts: {unmatched_llm_starts or 'none'}",
    f"LLM attempts lasting at least 10 minutes: {long_llm_attempts or 'none'}",
]
for operation, stats in sorted(operation_stats.items()):
    lines.append(f"  {operation}: {stats}")
lines.extend([
    "",
    "FOUR-LAYER MEMORY",
    f"Proper nouns: {len(nouns)}",
    f"Authority classes: {dict(authority_classes)}",
    f"Invalid authority promotions: {invalid_authority_promotions or 'none'}",
    f"Invalid proper nouns: {invalid_nouns or 'none'}",
    f"Unsafe authoritative automatic mappings: "
    f"{unsafe_low_authority_nouns or 'none'}",
    f"Unsafe automatic terminology: {unsafe_automatic_terms or 'none'}",
    f"Automatic term rejections: {automatic_term_rejections or 'none'}",
    f"Deferred automatic mappings: {deferred_automatic_nouns or 'none'}",
    f"Unsafe automatic mappings: {unsafe_automatic_nouns or 'none'}",
    f"Malformed source entity spans: {malformed_source_entity_nouns or 'none'}",
    f"Unsubstantiated introduced state: {unsubstantiated_introduced or 'none'}",
    f"Bilingual summary complete: {summary_payload['memory']['bilingual_summary']}",
    f"Long term: {len(long_term)} {dict(long_trust)}",
    f"Short term: {len(short_term)} {dict(short_trust)}",
    f"Style samples: {len(state.get('style_samples', []))}",
    f"Style evidence records: {len(style_sample_records)}",
    "Representative style records: "
    f"{sum(bool(item.get('representative')) for item in style_sample_records if isinstance(item, dict))}",
    f"Style admission decisions: {style_sample_policies or 'none'}",
    f"Style minimum score: {style_floor}",
    f"Style floor violations: {style_floor_violations or 'none'}",
    f"Stored style scores: {stored_style_scores or 'none'}",
    f"Malformed style samples: {malformed_style_samples or 'none'}",
    f"Style profile: {bool(state.get('style_profile'))}",
    f"Atomic final canonical admission events: "
    f"{len(canonical_admission_events)}",
    f"Missing atomic final canonical admission: "
    f"{missing_canonical_admission_events or 'none'}",
    f"Full refiner candidates rolled back: {len(candidate_rollbacks)}",
    f"Post-rollback final validations: {len(post_rollback_validations)}",
    f"Exact table-row recoveries: {len(table_row_recoveries)}",
    f"Degraded table placeholder suppressions: "
    f"{len(table_placeholder_suppressions)}",
    f"Bounded Persian readability reviews: {len(readability_reviews)}",
    f"Promoted readability decision batches: {len(readability_decisions)}",
    f"Minor objective readability promotions: "
    f"{len(minor_objective_readability_promotions)}",
    f"Unrouted objective readability findings: "
    f"{unrouted_objective_readability or 'none'}",
    f"Readability suggestions suppressed by source authority: "
    f"{len(readability_source_suppressions)}",
    f"Final language review events: {len(language_quality_reviews)}",
    f"Adjacent-span findings: "
    f"{sum(int(item.get('repeated_adjacent_span_count', 0) or 0) for item in language_quality_reviews)}",
    f"Governed-phrase findings: "
    f"{sum(int(item.get('repeated_governed_span_count', 0) or 0) for item in language_quality_reviews)}",
    f"Historical terminal chunk failures: {terminal_failures or 'none'}",
    f"Grouped table recovery batches: {len(table_group_recoveries)}",
    f"Unsafe accepted terminology: "
    f"{unsafe_accepted_terminology or 'none'}",
    f"Summary directional controls: {summary_directional_controls}",
    f"English-summary non-ASCII digits: "
    f"{english_summary_non_ascii_digits or 'none'}",
    f"Book context: {bool(state.get('book_context'))}",
    "",
    "RESEARCH / EXPORT",
    f"Research: {summary_payload['research']}",
    f"Checkpoint: {checkpoints}",
    f"Checkpoint publication error: {checkpoint_publication_error or 'none'}",
    f"Missed requested boundaries: {missed_requested_boundaries or 'none'}",
    f"Completed chunks after pending boundary: {completed_after_pending or 'none'}",
    f"Protocol remaining: {protocol.get('remaining_artifact_count', 0)}",
    f"Canonical paragraph identity missing: {missing_paragraph_identity or 'none'}",
    f"Canonical paragraph identity invalid: {invalid_paragraph_identity or 'none'}",
    f"Canonical paragraph identity reconstructed: "
    f"{reconstructed_paragraph_identity or 'none'}",
    f"Critic source-coverage violations: {critic_coverage_violations or 'none'}",
    f"Final identifiers unresolved: "
    f"{final_text.get('unresolved_identifier_count', 0)}",
    f"Identifier reconciliation unresolved: "
    f"{final_identifiers.get('unresolved_count', 0)}",
    f"Redundant original fragments removed: "
    f"{fragment_reconciliation.get('removed_count', 0)}",
    f"Source-grounded English originals preserved: "
    f"{original_audit.get('preserved_source_grounded_count', 0)}",
    f"Ungrounded English originals requiring review: "
    f"{original_audit.get('unapproved_ungrounded_count', 0)}",
    f"English-original audit passes: "
    f"{original_audit.get('pass_count', 0)}",
    f"Review events without reason: {review_events_without_reason or 'none'}",
    f"Final text review required: {final_text.get('review_required', False)}",
    f"Final rendered repairs accepted: "
    f"{final_rendered_repair.get('accepted_repair_count', 0)}",
    f"Final rendered paragraphs rejected: "
    f"{final_rendered_repair.get('rejected_paragraph_count', 0)}",
    f"Native RTL contents table: {docx_native_rtl_contents}",
    f"RTL contents geometry: {docx_contents_layout}",
    f"Malformed Latin scholarly abbreviations: "
    f"{malformed_latin_scholarly_abbreviations or 'none'}",
    f"Output: {summary_payload['output']}",
    f"DOCX superscript runs: {docx_superscript_runs or 'none'}",
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
        echo "ERROR: v10.34 companion audit failed"
        return "$RESULT"
    fi

    echo
    echo "========== GENERATED FILES =========="
    find jobs/uploads -maxdepth 1 -type f \
      \( -name '*_v1034_companion_audit.txt' \
         -o -name '*_v1034_companion_audit.json' \) \
      -printf '%TY-%Tm-%Td %TH:%TM %10s %p\n' | sort | tail -4
    echo "Root console copy: /root/tarjomeh_v1034_companion_audit.txt"
    echo "Read-only v10.34 companion audit completed. Nothing was changed."
}

audit_tarjomeh_v1034_companion "${1:-LATEST}"
unset -f audit_tarjomeh_v1034_companion
