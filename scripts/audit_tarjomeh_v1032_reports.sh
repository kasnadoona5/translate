collect_tarjomeh_v1032_reports() {
    cd /opt/translate || {
        echo "ERROR: /opt/translate not found"
        return 1
    }

    local JOB="${1:-LATEST}"

    echo "========== V10.32 RUNTIME CONTRACT =========="
    docker exec -i translate_tarjomeh_1 \
      tarjomeh capabilities --verify --json || return 1

    docker exec -i translate_tarjomeh_1 \
      env PYTHONIOENCODING=utf-8 python - "$JOB" <<'PY'
import collections
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import httpx

from tarjomeh.memory.manager import _style_sample_quality
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
from tarjomeh.quality.integrity import repeated_persian_clause_artifacts
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
db = sqlite3.connect("/app/jobs/jobs.db")
db.row_factory = sqlite3.Row
row = db.execute(
    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT 1"
    if hint == "LATEST" else "SELECT * FROM jobs WHERE id=?",
    () if hint == "LATEST" else (hint,),
).fetchone()
if not row:
    raise SystemExit("JOB NOT FOUND")
job = dict(row)
job_id = job["id"]
config = decode(job.get("config"), {})
upload_dir = Path("/app/jobs/uploads")


def artifact(key):
    item = db.execute(
        "SELECT payload FROM job_artifacts WHERE job_id=? AND artifact_key=?",
        (job_id, key),
    ).fetchone()
    return decode(item["payload"], {}) if item else {}


chunks = [dict(item) for item in db.execute(
    "SELECT chunk_index,status,text,translation,metadata FROM chunks "
    "WHERE job_id=? ORDER BY chunk_index",
    (job_id,),
)]
for chunk in chunks:
    chunk["metadata"] = decode(chunk.get("metadata"), {})
chunks_by_index = {int(item["chunk_index"]): item for item in chunks}
events = [dict(item) for item in db.execute(
    "SELECT rowid,timestamp,chunk_index,event_type,payload FROM chunk_events "
    "WHERE job_id=? ORDER BY timestamp,rowid",
    (job_id,),
)]
for event in events:
    event["payload"] = decode(event["payload"], {})
latest_start = {}
for event in events:
    if event["event_type"] == "chunk_started" and event["chunk_index"] >= 0:
        latest_start[event["chunk_index"]] = event["rowid"]
active_events = [
    event for event in events
    if event["chunk_index"] < 0
    or event["rowid"] >= latest_start.get(event["chunk_index"], 0)
]
state_row = db.execute(
    "SELECT state_data FROM memory_state WHERE job_id=?", (job_id,)
).fetchone()
state = decode(state_row["state_data"], {}) if state_row else {}
runtime_manifest = runtime_capabilities()
runtime_probes = runtime_behavior_probes()
worker_table = db.execute(
    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='job_workers'"
).fetchone()
worker_lease_row = db.execute(
    "SELECT * FROM job_workers WHERE job_id=?", (job_id,)
).fetchone() if worker_table else None
worker_lease = dict(worker_lease_row) if worker_lease_row else {}

# Generate the application's normal QA report even when the job is paused.
token = os.getenv("UI_SECRET_TOKEN", "")
response = httpx.get(
    f"http://127.0.0.1:8080/api/jobs/{job_id}/qa-report",
    headers={"Authorization": f"Bearer {token}"} if token else {},
    params={"token": token} if token else {},
    timeout=180,
)
response.raise_for_status()
qa_path = upload_dir / f"{job_id}_qa_report.txt"
qa_path.write_bytes(response.content)

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
saved_pairs = {
    (chunk.get("text") or "", chunk.get("translation") or "")
    for chunk in finished
}
proper = state.get("proper_nouns", {})
nouns = proper.get("nouns", proper) if isinstance(proper, dict) else {}
categories = proper.get("categories", {}) if isinstance(proper, dict) else {}
provenance = proper.get("provenance", {}) if isinstance(proper, dict) else {}
aliases = proper.get("aliases", {}) if isinstance(proper, dict) else {}
introduced = proper.get("introduced", []) if isinstance(proper, dict) else []
summary = state.get("bilingual_summary", {})
long_term = state.get("past_translations", [])
short_term = state.get("short_term_context", [])
style_samples = state.get("style_samples", [])
style_sample_records = state.get("style_sample_records", [])
style_profile = state.get("style_profile", "")
book_context = state.get("book_context", "")
english_summary = summary.get("english_summary", "") if isinstance(summary, dict) else ""
persian_summary = summary.get("persian_summary", "") if isinstance(summary, dict) else ""
english_summary_non_ascii_digits = sorted(set(re.findall(
    r"[\u0660-\u0669\u06f0-\u06f9]", english_summary
)))

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
invalid_long = [
    index for index, item in enumerate(long_term)
    if not isinstance(item, dict)
    or not str(item.get("source", "")).strip()
    or not str(item.get("translation", "")).strip()
]
invalid_short = [
    index for index, item in enumerate(short_term)
    if not isinstance(item, dict)
    or not str(item.get("source", "")).strip()
    or not str(item.get("translation", "")).strip()
]
short_exact = [
    bool(
        isinstance(item, dict)
        and (item.get("source", ""), item.get("translation", "")) in saved_pairs
    )
    for item in short_term
]
protocol_re = re.compile(
    r"<<<(?:TRANSLATION|END)|"
    r'"(?:translation|decision|rationale)"\s*:|'
    r"</?(?:analysis|answer|assistant|tool)>|`{3}",
    re.IGNORECASE,
)
style_leaks = [
    index for index, sample in enumerate(style_samples)
    if isinstance(sample, str) and protocol_re.search(sample)
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
entity_coverage_events = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in active_events
    if event["event_type"] == "source_entity_coverage"
]
missing_entity_anchors = [
    {
        "chunk_index": item.get("chunk_index"),
        "missing": item.get("required_missing", []),
    }
    for item in entity_coverage_events
    if item.get("required_missing")
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
review_events_without_reason = [
    {"chunk_index": event["chunk_index"], "payload": event["payload"]}
    for event in active_events
    if event["event_type"] == "chunk_review_required"
    and not (
        event["payload"].get("reason_codes")
        or event["payload"].get("reason")
    )
]
final_identifier_audit = artifact("final_identifier_reconciliation")
fragment_audit = artifact("original_fragment_reconciliation")
original_audit = artifact("english_original_audit")
final_text_audit = artifact("final_text_quality_audit")
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
terminal_failures = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in events if event["event_type"] == "chunk_terminal_failure"
]
structure_admissions = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in active_events
    if event["payload"].get("stage") == "final_source_structure_admission"
]
structure_review_only = [
    item for item in structure_admissions
    if item.get("review_reason") == "uncertain_source_structure_evidence"
]
structure_blocking = [
    item for item in structure_admissions if item.get("accepted") is False
]
foreign_expression_policies = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"] == "inline_original_policy"
    and event["payload"].get("required_source_foreign_expressions")
]


def duplicate_lines(text):
    values = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return [
        line for line, count in collections.Counter(values).items() if count > 1
    ]


memory_events = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in active_events if event["event_type"] == "memory_update_policy"
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
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
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
missing_final_quality = sorted(
    {int(chunk["chunk_index"]) for chunk in finished}
    - set(final_quality_by_chunk)
)
authority_violations = []
for item in memory_events:
    chunk_index = int(item["chunk_index"])
    quality = final_quality_by_chunk.get(chunk_index, {})
    style_issue_ids = unresolved_style_issue_ids(item, quality)
    if (
        quality.get("durable_authority") is False
        and item.get("long_term_reliable") is True
    ) or (
        item.get("style_sample_added") is True
        and style_event_is_active(item)
        and style_issue_ids
    ):
        authority_violations.append({
            "chunk_index": chunk_index,
            "long_term_reliable": item.get("long_term_reliable"),
            "style_sample_added": item.get("style_sample_added"),
            "unresolved_issue_ids": style_issue_ids,
        })
source_repair_events = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in active_events
    if event["event_type"] == "targeted_language_repair"
    and (
        int(event["payload"].get("source_fidelity_finding_count", 0) or 0)
        or event["payload"].get("resolved_source_issue_ids")
    )
]
worker_events = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     "event_type": event["event_type"], **event["payload"]}
    for event in events
    if event["event_type"] in {
        "worker_claimed", "worker_claim_rejected",
        "worker_lease_acquired", "worker_lease_renewed",
        "worker_lease_reclaimed", "worker_pause_requested",
        "worker_pause_persisted", "worker_pause_acknowledged",
        "worker_lease_released",
    }
]
atomic_source_rollbacks = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in active_events
    if event["event_type"] == "refinement_atomic_recovery_rolled_back"
]
unsafe_grounded_memory = [
    {
        "chunk_index": item.get("chunk_index"),
        "issue_ids": item.get("grounded_unresolved_issue_ids", []),
    }
    for item in memory_events
    if item.get("grounded_unresolved_issue_ids")
    and item.get("long_term_reliable") is True
]
automatic_term_events = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     "event_type": event["event_type"], **event["payload"]}
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
    for item in memory_events
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
    for item in memory_events
    if item.get("style_source_genre_excluded_paragraphs")
]
memory_export_events = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in active_events
    if event["event_type"] == "memory_export_consistency"
]
finished_indices = {chunk["chunk_index"] for chunk in finished}
canonical_admission_events = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in active_events
    if event["event_type"] == "final_canonical_admission"
]
canonical_admission_chunks = {
    item["chunk_index"] for item in canonical_admission_events
}
missing_canonical_admission_events = sorted(
    finished_indices - canonical_admission_chunks
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
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in active_events
    if event["event_type"] == "refinement_candidate_rolled_back"
]
post_rollback_validations = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in active_events
    if event["event_type"] == "post_rollback_final_validation"
]
table_row_recoveries = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in active_events
    if event["event_type"] == "table_row_identity_recovered"
]
table_placeholder_suppressions = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in active_events
    if event["event_type"] == "table_alignment_placeholders_suppressed"
]
readability_reviews = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in active_events
    if event["event_type"] == "persian_readability_review"
]
readability_source_suppressions = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
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
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in active_events
    if event["event_type"] == "language_quality_review"
]
memory_export_chunks = {item["chunk_index"] for item in memory_export_events}
missing_memory_export_consistency = sorted(
    finished_indices - memory_export_chunks
)
invalid_memory_export_consistency = [
    item for item in memory_export_events if not item.get("valid", True)
]
unsafe_accepted_terminology = [
    source for source, target in nouns.items()
    if isinstance(provenance.get(source), dict)
    and provenance[source].get("origin") == "accepted_correction"
    and not is_reusable_terminology_mapping(source, target)
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
    for item in memory_events
    if item.get("style_sample_added")
    and style_event_is_active(item)
    and unresolved_style_issue_ids(
        item,
        final_quality_by_chunk.get(int(item.get("chunk_index", -1)), {}),
    )
]
summary_events = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in active_events
    if event["event_type"] == "bilingual_summary_memory_policy"
]
rejected_summary_candidates = [
    item for item in summary_events
    if item.get("candidate_committed") is False
]
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
long_trust = collections.Counter(
    "reliable" if item.get("reliable", True) else "advisory"
    for item in long_term if isinstance(item, dict)
)
short_trust = collections.Counter(
    str(item.get("trust", "legacy"))
    for item in short_term if isinstance(item, dict)
)
origins = collections.Counter(
    str(item.get("origin", "unknown"))
    for item in provenance.values() if isinstance(item, dict)
)
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
style_sample_policies = [
    {
        "chunk_index": item.get("chunk_index"),
        **dict(item.get("style_sample_policy") or {}),
    }
    for item in memory_events if item.get("style_sample_policy")
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
directional_control_re = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
summary_directional_controls = {
    "english": sorted(set(directional_control_re.findall(english_summary))),
    "persian": sorted(set(directional_control_re.findall(persian_summary))),
}
chapter_checkpoints = artifact("chapter_checkpoints")
checkpoint_pending = chapter_checkpoints.get("pending") \
    if isinstance(chapter_checkpoints, dict) else None
if not isinstance(checkpoint_pending, dict):
    checkpoint_pending = None
latest_checkpoint_publication = chapter_checkpoints.get("latest_publication") \
    if isinstance(chapter_checkpoints, dict) else None
if not isinstance(latest_checkpoint_publication, dict):
    latest_checkpoint_publication = None
checkpoint_publication_error = ""
if job.get("status") == "paused" and latest_checkpoint_publication:
    published_path = str(latest_checkpoint_publication.get("output_path") or "")
    expected_bytes = int(latest_checkpoint_publication.get("output_bytes") or 0)
    expected_sha256 = str(
        latest_checkpoint_publication.get("output_sha256") or ""
    )
    if not published_path or published_path != str(job.get("output_path") or ""):
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

hard = []
review = []
if not all(runtime_probes.values()):
    hard.append("runtime_behavior_probe_failed")
if job.get("status") == "paused" and checkpoint_pending:
    hard.append("paused_checkpoint_has_unpublished_pending_intent")
if checkpoint_publication_error:
    hard.append(checkpoint_publication_error)
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
if invalid_long:
    hard.append("malformed_long_term_entry")
if invalid_short or (short_term and not all(short_exact)):
    hard.append("short_term_not_grounded_in_saved_translation")
if style_leaks:
    hard.append("style_protocol_leak")
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
if missing_memory_export_consistency:
    hard.append("missing_memory_export_consistency_event")
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
if unsafe_admission:
    hard.append("quarantined_translation_entered_memory")
if unsafe_grounded_memory:
    hard.append("grounded_quality_issue_entered_reliable_memory")
if authority_violations:
    hard.append("final_quality_memory_style_authority_disagreement")
if missing_final_quality:
    hard.append("missing_final_quality_admission")
if missing_final_canonical_admission:
    hard.append("missing_final_canonical_admission")
if missing_final_candidate_selection:
    hard.append("missing_final_candidate_selection")
if final_candidate_hash_mismatches:
    hard.append("final_candidate_selection_hash_mismatch")
if review_events_without_reason:
    hard.append("review_event_without_reason")
if int(final_identifier_audit.get("unresolved_count", 0) or 0):
    hard.append("unresolved_labeled_or_payload_identifier")
if finished and not long_term:
    review.append("long_term_memory_empty")
if reconstructed_paragraph_identity:
    review.append("paragraph_identity_reconstructed")
if finished and not short_term:
    review.append("short_term_memory_empty")
if len(finished) >= 2 and (not english_summary or not persian_summary):
    review.append("bilingual_summary_incomplete")
if rejected_summary_candidates:
    review.append("bilingual_summary_candidate_rejected")
reliable_prose = sum(
    bool(item.get("reliable", True))
    for item in long_term if isinstance(item, dict)
)
if finished_prose and not style_samples:
    review.append("style_samples_empty_despite_finished_prose")
if unrepaired:
    review.append("integrity_baseline_still_quarantined")
if missing_entity_anchors:
    review.append("missing_first_occurrence_entity_anchor")
if int(final_rendered_repair.get("rejected_paragraph_count", 0) or 0):
    review.append("final_rendered_repair_rejected")
if language_quality_reviews:
    review.append("objective_final_language_artifact")
if research_identity_gaps or research_term_evidence_gaps:
    review.append("research_evidence_metadata_incomplete")
if unrouted_objective_readability:
    review.append("objective_readability_finding_not_routed")
if int(original_audit.get("unapproved_ungrounded_count", 0) or 0):
    review.append("unapproved_ungrounded_english_original")
verdict = "FAIL" if hard else "REVIEW" if review else "PASS"

bundle = {
    "job": {
        "id": job_id,
        "status": job.get("status"),
        "error_message": job.get("error_message"),
        "input_path": job.get("input_path"),
        "output_path": job.get("output_path"),
        "finished_chunks": len(finished),
        "total_chunks": len(chunks),
    },
    "checks": {
        "invalid_proper_nouns": invalid_nouns,
        "unsafe_authoritative_automatic_mappings": unsafe_low_authority_nouns,
        "unsafe_automatic_terminology": unsafe_automatic_terms,
        "deferred_automatic_mappings": deferred_automatic_nouns,
        "unsafe_automatic_entities": unsafe_automatic_nouns,
        "malformed_source_entity_spans": malformed_source_entity_nouns,
        "unsubstantiated_introduced": unsubstantiated_introduced,
        "invalid_long_term_entries": invalid_long,
        "invalid_short_term_entries": invalid_short,
        "short_term_exact_saved_chunk": short_exact,
        "style_protocol_leaks": style_leaks,
        "malformed_style_samples": malformed_style_samples,
        "duplicate_english_summary_lines": duplicate_lines(english_summary),
        "duplicate_persian_summary_lines": duplicate_lines(persian_summary),
        "quarantined_chunks": sorted(quarantined),
        "repaired_chunks": sorted(repaired),
        "unrepaired_quarantine": unrepaired,
        "unsafe_memory_admission": unsafe_admission,
        "final_quality_admissions": final_quality_events,
        "missing_final_quality_admissions": missing_final_quality,
        "source_bound_recovery_events": source_bound_recovery_events,
        "missing_final_canonical_admission": missing_final_canonical_admission,
        "final_candidate_selections": final_candidate_events,
        "missing_final_candidate_selection": missing_final_candidate_selection,
        "final_candidate_hash_mismatches": final_candidate_hash_mismatches,
        "authority_violations": authority_violations,
        "source_obligation_repairs": source_repair_events,
        "atomic_source_rollbacks": atomic_source_rollbacks,
        "missing_first_occurrence_entity_anchors": missing_entity_anchors,
        "memory_export_consistency": memory_export_events,
        "missing_memory_export_consistency": missing_memory_export_consistency,
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
        "readability_source_conflict_suppressions": (
            readability_source_suppressions
        ),
        "minor_objective_readability_promotions": (
            minor_objective_readability_promotions
        ),
        "unrouted_objective_readability": unrouted_objective_readability,
        "language_quality_reviews": language_quality_reviews,
        "unsafe_accepted_terminology": unsafe_accepted_terminology,
        "invalid_authority_promotions": invalid_authority_promotions,
        "authority_classes": dict(authority_classes),
        "unsafe_style_admission": unsafe_style_admission,
        "style_paragraph_exclusions": style_paragraph_exclusions,
        "style_source_genre_exclusions": style_source_genre_exclusions,
        "style_sample_policies": style_sample_policies,
        "style_minimum_score": style_floor,
        "style_floor_violations": style_floor_violations,
        "stored_style_scores": stored_style_scores,
        "automatic_term_rejections": automatic_term_rejections,
        "summary_directional_controls": summary_directional_controls,
        "english_summary_non_ascii_digits": english_summary_non_ascii_digits,
        "cross_table_continuation_joins": cross_table_joins,
        "review_events_without_reason": review_events_without_reason,
        "final_identifier_reconciliation": final_identifier_audit,
        "original_fragment_reconciliation": fragment_audit,
        "english_original_audit": original_audit,
        "final_text_quality_audit": final_text_audit,
        "final_rendered_language_repair": final_rendered_repair,
        "chunk_terminal_failures": terminal_failures,
        "runtime_manifest": runtime_manifest,
        "runtime_behavior_probes": runtime_probes,
        "canonical_document_identity": canonical_document_identity,
        "canonical_paragraph_identity": paragraph_identity,
        "missing_paragraph_identity": missing_paragraph_identity,
        "invalid_paragraph_identity": invalid_paragraph_identity,
        "reconstructed_paragraph_identity": reconstructed_paragraph_identity,
        "critic_coverage_violations": critic_coverage_violations,
        "typed_structure_admissions": structure_admissions,
        "review_only_structure_findings": structure_review_only,
        "blocking_structure_findings": structure_blocking,
        "source_foreign_expression_policies": foreign_expression_policies,
    },
    "layer_1_proper_nouns": proper,
    "layer_2_bilingual_summary": summary,
    "layer_3_long_term": long_term,
    "layer_4_short_term": short_term,
    "style_profile": style_profile,
    "style_samples": style_samples,
    "style_sample_records": style_sample_records,
    "book_context": book_context,
    "book_research": research,
    "research_identity_gaps": research_identity_gaps,
    "research_term_evidence_gaps": research_term_evidence_gaps,
    "chapter_checkpoints": chapter_checkpoints,
    "checkpoint_publication_error": checkpoint_publication_error,
    "worker_lease": worker_lease,
    "worker_lifecycle_events": worker_events,
    "memory_update_events": memory_events,
    "automatic_term_events": automatic_term_events,
    "summary_memory_events": summary_events,
    "rejected_summary_candidates": rejected_summary_candidates,
    "source_entity_coverage_events": entity_coverage_events,
    "verdict": verdict,
    "hard_failures": hard,
    "review_signals": review,
}
json_path = upload_dir / f"{job_id}_v1032_memory_audit.json"
txt_path = upload_dir / f"{job_id}_v1032_memory_audit.txt"
json_path.write_text(
    json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
)

lines = [
    "Tarjomeh v10.32 Memory, Style, and QA Companion Report",
    f"Job: {job_id}",
    f"Status: {job.get('status')}",
    f"Finished chunks: {len(finished)} / {len(chunks)}",
    f"Runtime release: {runtime_manifest.get('release')}",
    f"Runtime behavior probes: {runtime_probes}",
    f"Canonical document identity: {canonical_document_identity or 'pending'}",
    f"Canonical paragraph identity missing: {missing_paragraph_identity or 'none'}",
    f"Canonical paragraph identity invalid: {invalid_paragraph_identity or 'none'}",
    f"Canonical paragraph identity reconstructed: "
    f"{reconstructed_paragraph_identity or 'none'}",
    f"Critic source-coverage violations: {critic_coverage_violations or 'none'}",
    f"Typed structure admissions: {len(structure_admissions)}; "
    f"review-only={len(structure_review_only)}; blocking={len(structure_blocking)}",
    f"Source foreign-expression policies: {foreign_expression_policies or 'none'}",
    "",
    "LAYER 1 - PROPER NOUNS AND TERMINOLOGY",
    f"Mappings: {len(nouns)}",
    f"Origins: {dict(origins)}",
    f"Authority classes: {dict(authority_classes)}",
    f"Invalid authority promotions: {invalid_authority_promotions or 'none'}",
    f"Aliases: {len(aliases)}",
    f"Invalid mappings: {invalid_nouns or 'none'}",
    f"Unsafe authoritative automatic mappings: "
    f"{unsafe_low_authority_nouns or 'none'}",
    f"Deferred automatic mappings: {deferred_automatic_nouns or 'none'}",
    f"Unsafe automatic mappings: {unsafe_automatic_nouns or 'none'}",
    f"Malformed source entity spans: {malformed_source_entity_nouns or 'none'}",
    f"Unsubstantiated introduced state: {unsubstantiated_introduced or 'none'}",
    "",
    "LAYER 2 - BILINGUAL RUNNING SUMMARY",
    f"English characters: {len(english_summary)}",
    english_summary or "(empty)",
    "",
    f"Persian characters: {len(persian_summary)}",
    persian_summary or "(empty)",
    "",
    "LAYER 3 - LONG-TERM RETRIEVAL MEMORY",
    f"Entries: {len(long_term)}; trust: {dict(long_trust)}",
]
for index, item in enumerate(long_term):
    if isinstance(item, dict):
        lines.extend([
            f"  Entry {index}: reliable={item.get('reliable', True)} "
            f"chapter={item.get('chapter_title', '')}",
            f"    Source: {item.get('source', '')}",
            f"    Target: {item.get('translation', '')}",
        ])
lines.extend([
    "",
    "LAYER 4 - SHORT-TERM CONTINUITY MEMORY",
    f"Entries: {len(short_term)}; trust: {dict(short_trust)}",
])
for index, item in enumerate(short_term):
    if isinstance(item, dict):
        exact = short_exact[index] if index < len(short_exact) else False
        lines.extend([
            f"  Entry {index}: trust={item.get('trust', 'legacy')} "
            f"role={item.get('structural_role', 'body')} "
            f"chapter={item.get('chapter_title', '')} exact_saved={exact}",
            f"    Source: {item.get('source', '')}",
            f"    Target: {item.get('translation', '')}",
        ])
lines.extend([
    "",
    "STYLE PROFILE",
    f"Samples: {len(style_samples)}",
    f"Evidence records: {len(style_sample_records)}",
    "Representative evidence records: "
    f"{sum(bool(item.get('representative')) for item in style_sample_records if isinstance(item, dict))}",
    f"Protocol leaks: {style_leaks or 'none'}",
    f"Malformed samples: {malformed_style_samples or 'none'}",
    f"Style admission decisions: {style_sample_policies or 'none'}",
    f"Style minimum score: {style_floor}",
    f"Style floor violations: {style_floor_violations or 'none'}",
    f"Stored style scores: {stored_style_scores or 'none'}",
    str(style_profile or "(empty)"),
    "",
    "STYLE SAMPLES",
])
for index, sample in enumerate(style_samples):
    lines.extend([f"  Sample {index}:", str(sample), ""])
lines.extend([
    "BOOK CONTEXT",
    str(book_context or "(empty)"),
    "",
    "V10 MEMORY ADMISSION",
    f"Quarantined chunks: {sorted(quarantined) or 'none'}",
    f"Repaired chunks: {sorted(repaired) or 'none'}",
    f"Unrepaired quarantine: {unrepaired or 'none'}",
    f"Unsafe memory admission: {unsafe_admission or 'none'}",
    f"Grounded issues admitted as reliable: "
    f"{unsafe_grounded_memory or 'none'}",
    f"Worker lease: {worker_lease or 'none'}",
    f"Worker lifecycle events: {worker_events or 'none'}",
    f"Chapter checkpoint state: {chapter_checkpoints or 'none'}",
    f"Chapter checkpoint publication error: "
    f"{checkpoint_publication_error or 'none'}",
    f"Memory policy events: {len(memory_events)}",
    f"Summary policy events: {len(summary_events)}",
    f"Source-entity coverage events: {len(entity_coverage_events)}",
    f"Missing entity anchors: {missing_entity_anchors or 'none'}",
    f"Memory/export consistency events: {len(memory_export_events)}",
    f"Final quality admissions: {len(final_quality_events)}",
    f"Missing final quality admissions: {missing_final_quality or 'none'}",
    f"Source-bound recovery events: {len(source_bound_recovery_events)}",
    "Missing final canonical admission: "
    f"{missing_final_canonical_admission or 'none'}",
    "Missing final candidate selections: "
    f"{missing_final_candidate_selection or 'none'}",
    "Final candidate hash mismatches: "
    f"{final_candidate_hash_mismatches or 'none'}",
    f"Canonical event/hash mismatches: "
    f"{canonical_event_hash_mismatches or 'none'}",
    f"Canonical Layer-3 mismatches: {canonical_memory_mismatches or 'none'}",
    f"Memory/style authority disagreements: {authority_violations or 'none'}",
    f"Source-obligation repair batches: {source_repair_events or 'none'}",
    f"Atomic source-regression rollbacks: "
    f"{atomic_source_rollbacks or 'none'}",
    f"Rejected bilingual-summary candidates: "
    f"{rejected_summary_candidates or 'none'}",
    f"Missing memory/export events: "
    f"{missing_memory_export_consistency or 'none'}",
    f"Invalid memory/export reconciliation: "
    f"{invalid_memory_export_consistency or 'none'}",
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
    f"Unsafe accepted terminology: "
    f"{unsafe_accepted_terminology or 'none'}",
    f"Style samples admitted with unresolved critique: "
    f"{unsafe_style_admission or 'none'}",
    f"Paragraph-level style exclusions: "
    f"{style_paragraph_exclusions or 'none'}",
    f"Source-genre style exclusions: "
    f"{style_source_genre_exclusions or 'none'}",
    f"Unsafe automatic terminology: {unsafe_automatic_terms or 'none'}",
    f"Automatic term rejections: {automatic_term_rejections or 'none'}",
    f"Summary directional controls: {summary_directional_controls}",
    f"English-summary non-ASCII digits: "
    f"{english_summary_non_ascii_digits or 'none'}",
    f"Cross-table continuation joins: {cross_table_joins or 'none'}",
    f"Review events without reason: {review_events_without_reason or 'none'}",
    f"Final unresolved identifiers: "
    f"{final_identifier_audit.get('unresolved_count', 0)}",
    f"Redundant original fragments removed: "
    f"{fragment_audit.get('removed_count', 0)}",
    f"Source-grounded English originals preserved: "
    f"{original_audit.get('preserved_source_grounded_count', 0)}",
    f"Ungrounded English originals requiring review: "
    f"{original_audit.get('unapproved_ungrounded_count', 0)}",
    f"English-original audit passes: "
    f"{original_audit.get('pass_count', 0)}",
    f"Final rendered repairs accepted: "
    f"{final_rendered_repair.get('accepted_repair_count', 0)}",
    f"Final rendered paragraphs rejected: "
    f"{final_rendered_repair.get('rejected_paragraph_count', 0)}",
    f"Final text language findings: "
    f"{final_text_audit.get('language_finding_count', 0)}",
    f"Research identity metadata gaps: {research_identity_gaps or 'none'}",
    f"Research term evidence gaps: {research_term_evidence_gaps or 'none'}",
])
for item in memory_events:
    lines.append(
        "  chunk={chunk} short={short} trust={trust} continuity={continuity} "
        "long={long_trust} style={style} style_reason={style_reason} "
        "excluded={excluded} quality={quality} reasons={reasons} "
        "disqualifying={hard} advisory={advisory}".format(
            chunk=item.get("chunk_index"),
            short=item.get("short_term_added"),
            trust=item.get("short_term_trust"),
            continuity=item.get("continuity_retained"),
            long_trust=item.get("long_term_trust"),
            style=item.get("style_sample_added"),
            style_reason=item.get("style_policy_reason", "legacy"),
            excluded=item.get("style_excluded_paragraphs", []),
            quality=item.get("quality_approved"),
            reasons=item.get("reliability_reasons", []),
            hard=item.get("disqualifying_reliability_reasons", []),
            advisory=item.get("advisory_reliability_reasons", []),
        )
    )
lines.extend([
    "",
    "VERDICT",
    f"Hard failures: {hard or 'none'}",
    f"Review signals: {review or 'none'}",
    f"VERDICT: {verdict}",
])
txt_path.write_text("\n".join(lines), encoding="utf-8-sig")

print("========== V10 REPORT BUNDLE ==========")
print("Job:", job_id, job.get("status"))
print("QA:", qa_path, qa_path.stat().st_size, "bytes")
print("Memory TXT:", txt_path)
print("Memory JSON:", json_path)
print("Layers:", {
    "proper_nouns": len(nouns),
    "summary_en_chars": len(english_summary),
    "summary_fa_chars": len(persian_summary),
    "long_term": len(long_term),
    "short_term": len(short_term),
    "style_samples": len(style_samples),
})
print("Quarantine:", {
    "quarantined": sorted(quarantined),
    "repaired": sorted(repaired),
    "unsafe_memory_admission": unsafe_admission,
})
print("VERDICT:", verdict, "hard=", hard, "review=", review)
print("\n========== QA PREVIEW ==========")
print(response.content.decode("utf-8-sig", errors="replace")[:5000])
PY

    RESULT=$?
    if [ "$RESULT" -ne 0 ]; then
        echo "ERROR: v10.32 report collection failed"
        return "$RESULT"
    fi

    echo
    echo "========== NEWEST GENERATED FILES =========="
    find jobs/uploads -maxdepth 1 -type f \
      \( -name '*_qa_report.txt' -o -name '*_v1032_memory_audit.txt' \
         -o -name '*_v1032_memory_audit.json' \) \
      -printf '%TY-%Tm-%Td %TH:%TM %10s %p\n' | sort | tail -6
    echo "Read-only report collection completed. The job was not changed."
}

collect_tarjomeh_v1032_reports "${1:-LATEST}"
unset -f collect_tarjomeh_v1032_reports
