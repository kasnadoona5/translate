audit_tarjomeh_v1015_companion() {
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
    echo "========== V10.15 COMPANION AUDIT =========="
    docker exec -i translate_tarjomeh_1 env PYTHONIOENCODING=utf-8 \
      python - "$JOB" <<'PY' | tee /root/tarjomeh_v1015_companion_audit.txt
import collections
import json
import re
import sqlite3
import sys
import zipfile
from pathlib import Path

import tarjomeh
from docx import Document
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
    required_flow.update({"memory_context", "memory_update_policy"})
required_flow.add("memory_export_consistency")
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
final_rendered_repair = artifact("final_rendered_language_repair")
structure = artifact("document_structure_version")
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
memory_export_events = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"] == "memory_export_consistency"
]
canonical_typography_events = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"] == "final_candidate_typography"
]
finished_indices = {chunk["chunk_index"] for chunk in finished}
missing_canonical_typography = sorted(
    finished_indices
    - {item["chunk_index"] for item in canonical_typography_events}
)
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
memory_policy_events = [
    {"chunk_index": event["chunk_index"], **event["payload"]}
    for event in active_events
    if event["event_type"] == "memory_update_policy"
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
def critique_disqualifies_style(payload):
    if int(payload.get("blocking_issue_count", 0) or 0):
        return True
    return any(
        isinstance(detail, dict)
        and str(detail.get("severity", "")).casefold() in {"major", "critical"}
        for detail in payload.get("issue_details", []) or []
    )


unsafe_style_admission = [
    item.get("chunk_index") for item in memory_policy_events
    if item.get("style_sample_added")
    and critique_disqualifies_style(
        latest_critiques.get(item.get("chunk_index"), {})
    )
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
english_summary = summary.get("english_summary", "") if isinstance(summary, dict) else ""
persian_summary = summary.get("persian_summary", "") if isinstance(summary, dict) else ""
directional_control_re = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
summary_directional_controls = {
    "english": sorted(set(directional_control_re.findall(english_summary))),
    "persian": sorted(set(directional_control_re.findall(persian_summary))),
}
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
docx_superscript_runs = []
docx_native_rtl_contents = None
docx_contents_layout = None
if output_valid and output.suffix.lower() == ".docx":
    rendered = Document(output)
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

# Confirm the running container, not merely Git, contains v10.15 behavior.
root = Path(tarjomeh.__file__).resolve().parent
pipeline_source = (root / "core" / "pipeline.py").read_text(encoding="utf-8")
client_source = (root / "core" / "llm_client.py").read_text(encoding="utf-8")
structure_source = (root / "quality" / "structure_audit.py").read_text(
    encoding="utf-8"
)
parser_source = (root / "parsers" / "pdf_parser.py").read_text(encoding="utf-8")
proper_source = (root / "memory" / "proper_nouns.py").read_text(encoding="utf-8")
manager_source = (root / "memory" / "manager.py").read_text(encoding="utf-8")
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
    "structure_version_four": "structure_version: int = 4" in parser_source,
    "fluent_academic_contract": (
        "split sentences inside a paragraph"
        in (root / "core" / "prompts.py").read_text(
            encoding="utf-8"
        )
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
        "_OBJECTIVE_FLUENCY_MINOR_THRESHOLD = 0.60" in pipeline_source
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
    "atomic_local_salvage": "coherent unit" in pipeline_source,
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
        "final_candidate_typography" in pipeline_source
    ),
    "citation_tail_protection": "_AUTHOR_YEAR_CITATION_RE" in typography_source,
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
    "post_rollback_validation": (
        "post_rollback_final_validation" in pipeline_source
    ),
    "source_announcement_fidelity": (
        "announced_count_lexical_mismatch" in structure_source
        and "source-authored announced quantity" in prompts_source
    ),
}

hard = []
review = []
if not all(runtime_markers.values()):
    hard.append("running_container_is_not_complete_v1015")
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
if any(summary_directional_controls.values()):
    hard.append("directional_control_entered_summary_memory")
if invalid_memory_export_consistency:
    hard.append("invalid_memory_export_reconciliation")
if missing_canonical_typography:
    hard.append("missing_pre_memory_canonical_typography")
if unsafe_accepted_terminology:
    hard.append("contextual_accepted_correction_entered_global_memory")
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
if finished_prose and not style_samples:
    review.append("style_samples_empty_despite_finished_prose")
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
        "missing_first_occurrence_entity_anchors": missing_entity_anchors,
        "memory_export_consistency": memory_export_events,
        "invalid_memory_export_consistency": invalid_memory_export_consistency,
        "canonical_typography_events": canonical_typography_events,
        "missing_canonical_typography": missing_canonical_typography,
        "full_candidate_rollbacks": candidate_rollbacks,
        "post_rollback_validations": post_rollback_validations,
        "table_row_identity_recoveries": table_row_recoveries,
        "table_placeholder_suppressions": table_placeholder_suppressions,
        "persian_readability_reviews": readability_reviews,
        "readability_advisory_decisions": readability_decisions,
        "table_group_recoveries": table_group_recoveries,
        "unsafe_style_admission": unsafe_style_admission,
        "style_paragraph_exclusions": style_paragraph_exclusions,
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
        "retry_pairs": retry_pairs,
        "exact_replay_violations": exact_replay_violations,
        "quality_contract_violations": quality_contract_violations,
        "expected_helper_recovery": expected_helper_recovery,
    },
    "memory": {
        "proper_nouns": len(nouns),
        "invalid_proper_nouns": invalid_nouns,
        "unsafe_authoritative_automatic_mappings": unsafe_low_authority_nouns,
        "unsafe_automatic_terminology": unsafe_automatic_terms,
        "unsafe_accepted_terminology": unsafe_accepted_terminology,
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
        "malformed_style_samples": malformed_style_samples,
        "style_profile": bool(state.get("style_profile")),
        "summary_directional_controls": summary_directional_controls,
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
    "final_rendered_language_repair": final_rendered_repair,
    "final_identifier_reconciliation": final_identifiers,
    "original_fragment_reconciliation": fragment_reconciliation,
    "review_events_without_reason": review_events_without_reason,
    "output": {
        "path": str(output),
        "exists": output.exists(),
        "valid": output_valid,
        "size": output.stat().st_size if output.exists() else 0,
        "superscript_runs": docx_superscript_runs,
        "native_rtl_contents": docx_native_rtl_contents,
        "contents_layout": docx_contents_layout,
    },
    "verdict": verdict,
    "hard_failures": hard,
    "review_signals": review,
}
json_path = Path(f"/app/jobs/uploads/{job_id}_v1015_companion_audit.json")
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
txt_path = Path(f"/app/jobs/uploads/{job_id}_v1015_companion_audit.txt")

lines = [
    "Tarjomeh v10.15 Independent Companion Audit",
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
    f"Missing entity anchors: {missing_entity_anchors or 'none'}",
    f"Memory/export consistency events: {len(memory_export_events)}",
    f"Invalid memory/export reconciliation: "
    f"{invalid_memory_export_consistency or 'none'}",
    f"Style samples admitted with unresolved critique: "
    f"{unsafe_style_admission or 'none'}",
    f"Paragraph-level style exclusions: {style_paragraph_exclusions or 'none'}",
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
    f"Malformed style samples: {malformed_style_samples or 'none'}",
    f"Style profile: {bool(state.get('style_profile'))}",
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
    f"Promoted readability decision batches: {len(readability_decisions)}",
    f"Grouped table recovery batches: {len(table_group_recoveries)}",
    f"Unsafe accepted terminology: "
    f"{unsafe_accepted_terminology or 'none'}",
    f"Summary directional controls: {summary_directional_controls}",
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
    f"Final rendered repairs accepted: "
    f"{final_rendered_repair.get('accepted_repair_count', 0)}",
    f"Final rendered paragraphs rejected: "
    f"{final_rendered_repair.get('rejected_paragraph_count', 0)}",
    f"Native RTL contents table: {docx_native_rtl_contents}",
    f"RTL contents geometry: {docx_contents_layout}",
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
        echo "ERROR: v10.15 companion audit failed"
        return "$RESULT"
    fi

    echo
    echo "========== GENERATED FILES =========="
    find jobs/uploads -maxdepth 1 -type f \
      \( -name '*_v1015_companion_audit.txt' \
         -o -name '*_v1015_companion_audit.json' \) \
      -printf '%TY-%Tm-%Td %TH:%TM %10s %p\n' | sort | tail -4
    echo "Root console copy: /root/tarjomeh_v1015_companion_audit.txt"
    echo "Read-only v10.15 companion audit completed. Nothing was changed."
}

audit_tarjomeh_v1015_companion "${1:-${JOB:-LATEST}}"
unset -f audit_tarjomeh_v1015_companion
