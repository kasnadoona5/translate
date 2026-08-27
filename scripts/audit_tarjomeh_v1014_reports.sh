collect_tarjomeh_v1014_reports() {
    cd /opt/translate || {
        echo "ERROR: /opt/translate not found"
        return 1
    }

    JOB="${1:-LATEST}"

    docker exec -i translate_tarjomeh_1 \
      env PYTHONIOENCODING=utf-8 python - "$JOB" <<'PY'
import collections
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import httpx

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


def decode(value, default):
    try:
        return json.loads(value or "")
    except Exception:
        return default


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
style_profile = state.get("style_profile", "")
book_context = state.get("book_context", "")
english_summary = summary.get("english_summary", "") if isinstance(summary, dict) else ""
persian_summary = summary.get("persian_summary", "") if isinstance(summary, dict) else ""

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
        and re.sub(r"[\s\u200c]+", " ", target).casefold()
        in re.sub(
            r"[\s\u200c]+", " ", str(chunk.get("translation") or "")
        ).casefold()
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
        sample.lstrip().startswith(("...", "…"))
        or sample.rstrip().endswith(("...", "…"))
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
final_text_audit = artifact("final_text_quality_audit")
final_rendered_repair = artifact("final_rendered_language_repair")


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
memory_export_events = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in active_events
    if event["event_type"] == "memory_export_consistency"
]
finished_indices = {chunk["chunk_index"] for chunk in finished}
canonical_typography_events = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in active_events
    if event["event_type"] == "final_candidate_typography"
]
canonical_typography_chunks = {
    item["chunk_index"] for item in canonical_typography_events
}
missing_canonical_typography = sorted(
    finished_indices - canonical_typography_chunks
)
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
latest_critiques = {}
for event in active_events:
    if event["event_type"] == "critique_completed":
        latest_critiques[event["chunk_index"]] = event["payload"]
def critique_disqualifies_style(payload):
    if int(payload.get("blocking_issue_count", 0) or 0):
        return True
    return any(
        isinstance(detail, dict)
        and str(detail.get("severity", "")).casefold() in {"major", "critical"}
        for detail in payload.get("issue_details", []) or []
    )


unsafe_style_admission = [
    item.get("chunk_index") for item in memory_events
    if item.get("style_sample_added")
    and critique_disqualifies_style(
        latest_critiques.get(item.get("chunk_index"), {})
    )
]
summary_events = [
    {"timestamp": event["timestamp"], "chunk_index": event["chunk_index"],
     **event["payload"]}
    for event in active_events
    if event["event_type"] == "bilingual_summary_memory_policy"
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
directional_control_re = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
summary_directional_controls = {
    "english": sorted(set(directional_control_re.findall(english_summary))),
    "persian": sorted(set(directional_control_re.findall(persian_summary))),
}

hard = []
review = []
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
if any(summary_directional_controls.values()):
    hard.append("directional_control_entered_summary_memory")
if missing_memory_export_consistency:
    hard.append("missing_memory_export_consistency_event")
if invalid_memory_export_consistency:
    hard.append("invalid_memory_export_reconciliation")
if missing_canonical_typography:
    hard.append("missing_pre_memory_canonical_typography")
if unsafe_accepted_terminology:
    hard.append("contextual_accepted_correction_entered_global_memory")
if unsafe_admission:
    hard.append("quarantined_translation_entered_memory")
if review_events_without_reason:
    hard.append("review_event_without_reason")
if int(final_identifier_audit.get("unresolved_count", 0) or 0):
    hard.append("unresolved_labeled_or_payload_identifier")
if finished and not long_term:
    review.append("long_term_memory_empty")
if finished and not short_term:
    review.append("short_term_memory_empty")
if len(finished) >= 2 and (not english_summary or not persian_summary):
    review.append("bilingual_summary_incomplete")
reliable_prose = sum(
    bool(item.get("reliable", True))
    for item in long_term if isinstance(item, dict)
)
if reliable_prose and not style_samples:
    review.append("style_samples_empty_despite_reliable_prose")
if unrepaired:
    review.append("integrity_baseline_still_quarantined")
if missing_entity_anchors:
    review.append("missing_first_occurrence_entity_anchor")
if int(final_rendered_repair.get("rejected_paragraph_count", 0) or 0):
    review.append("final_rendered_repair_rejected")
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
        "missing_first_occurrence_entity_anchors": missing_entity_anchors,
        "memory_export_consistency": memory_export_events,
        "missing_memory_export_consistency": missing_memory_export_consistency,
        "invalid_memory_export_consistency": invalid_memory_export_consistency,
        "canonical_typography_events": canonical_typography_events,
        "missing_canonical_typography": missing_canonical_typography,
        "full_candidate_rollbacks": candidate_rollbacks,
        "post_rollback_validations": post_rollback_validations,
        "table_row_identity_recoveries": table_row_recoveries,
        "table_placeholder_suppressions": table_placeholder_suppressions,
        "persian_readability_reviews": readability_reviews,
        "unsafe_accepted_terminology": unsafe_accepted_terminology,
        "unsafe_style_admission": unsafe_style_admission,
        "style_paragraph_exclusions": style_paragraph_exclusions,
        "automatic_term_rejections": automatic_term_rejections,
        "summary_directional_controls": summary_directional_controls,
        "cross_table_continuation_joins": cross_table_joins,
        "review_events_without_reason": review_events_without_reason,
        "final_identifier_reconciliation": final_identifier_audit,
        "original_fragment_reconciliation": fragment_audit,
        "final_text_quality_audit": final_text_audit,
        "final_rendered_language_repair": final_rendered_repair,
    },
    "layer_1_proper_nouns": proper,
    "layer_2_bilingual_summary": summary,
    "layer_3_long_term": long_term,
    "layer_4_short_term": short_term,
    "style_profile": style_profile,
    "style_samples": style_samples,
    "book_context": book_context,
    "book_research": artifact("book_research"),
    "chapter_checkpoints": artifact("chapter_checkpoints"),
    "memory_update_events": memory_events,
    "automatic_term_events": automatic_term_events,
    "summary_memory_events": summary_events,
    "source_entity_coverage_events": entity_coverage_events,
    "verdict": verdict,
    "hard_failures": hard,
    "review_signals": review,
}
json_path = upload_dir / f"{job_id}_v1014_memory_audit.json"
txt_path = upload_dir / f"{job_id}_v1014_memory_audit.txt"
json_path.write_text(
    json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
)

lines = [
    "Tarjomeh v10.14 Memory, Style, and QA Companion Report",
    f"Job: {job_id}",
    f"Status: {job.get('status')}",
    f"Finished chunks: {len(finished)} / {len(chunks)}",
    "",
    "LAYER 1 - PROPER NOUNS AND TERMINOLOGY",
    f"Mappings: {len(nouns)}",
    f"Origins: {dict(origins)}",
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
    f"Protocol leaks: {style_leaks or 'none'}",
    f"Malformed samples: {malformed_style_samples or 'none'}",
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
    f"Memory policy events: {len(memory_events)}",
    f"Summary policy events: {len(summary_events)}",
    f"Source-entity coverage events: {len(entity_coverage_events)}",
    f"Missing entity anchors: {missing_entity_anchors or 'none'}",
    f"Memory/export consistency events: {len(memory_export_events)}",
    f"Missing memory/export events: "
    f"{missing_memory_export_consistency or 'none'}",
    f"Invalid memory/export reconciliation: "
    f"{invalid_memory_export_consistency or 'none'}",
    f"Pre-memory canonical typography events: "
    f"{len(canonical_typography_events)}",
    f"Missing pre-memory canonical typography: "
    f"{missing_canonical_typography or 'none'}",
    f"Full refiner candidates rolled back: {len(candidate_rollbacks)}",
    f"Post-rollback final validations: {len(post_rollback_validations)}",
    f"Exact table-row recoveries: {len(table_row_recoveries)}",
    f"Degraded table placeholder suppressions: "
    f"{len(table_placeholder_suppressions)}",
    f"Bounded Persian readability reviews: {len(readability_reviews)}",
    f"Unsafe accepted terminology: "
    f"{unsafe_accepted_terminology or 'none'}",
    f"Style samples admitted with unresolved critique: "
    f"{unsafe_style_admission or 'none'}",
    f"Paragraph-level style exclusions: "
    f"{style_paragraph_exclusions or 'none'}",
    f"Unsafe automatic terminology: {unsafe_automatic_terms or 'none'}",
    f"Automatic term rejections: {automatic_term_rejections or 'none'}",
    f"Summary directional controls: {summary_directional_controls}",
    f"Cross-table continuation joins: {cross_table_joins or 'none'}",
    f"Review events without reason: {review_events_without_reason or 'none'}",
    f"Final unresolved identifiers: "
    f"{final_identifier_audit.get('unresolved_count', 0)}",
    f"Redundant original fragments removed: "
    f"{fragment_audit.get('removed_count', 0)}",
    f"Final rendered repairs accepted: "
    f"{final_rendered_repair.get('accepted_repair_count', 0)}",
    f"Final rendered paragraphs rejected: "
    f"{final_rendered_repair.get('rejected_paragraph_count', 0)}",
    f"Final text language findings: "
    f"{final_text_audit.get('language_finding_count', 0)}",
])
for item in memory_events:
    lines.append(
        "  chunk={chunk} short={short} trust={trust} continuity={continuity} "
        "long={long_trust} style={style} style_reason={style_reason} "
        "excluded={excluded} quality={quality} reasons={reasons}".format(
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
        echo "ERROR: v10.14 report collection failed"
        return "$RESULT"
    fi

    echo
    echo "========== NEWEST GENERATED FILES =========="
    find jobs/uploads -maxdepth 1 -type f \
      \( -name '*_qa_report.txt' -o -name '*_v1014_memory_audit.txt' \
         -o -name '*_v1014_memory_audit.json' \) \
      -printf '%TY-%Tm-%Td %TH:%TM %10s %p\n' | sort | tail -6
    echo "Read-only report collection completed. The job was not changed."
}

collect_tarjomeh_v1014_reports "${1:-${JOB:-LATEST}}"
unset -f collect_tarjomeh_v1014_reports
