#!/usr/bin/env bash
set -Eeuo pipefail

cd /opt/translate

TAG="v10.24.0"
CONTAINER="translate_tarjomeh_1"
SERVICE="tarjomeh"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_ROOT="/root/tarjomeh-backups"
BACKUP="$BACKUP_ROOT/v1024-$STAMP"
ROLLBACK="translate_tarjomeh:rollback-v1024-$STAMP"
ROLLBACK_ARCHIVE="$BACKUP/translate_tarjomeh_previous_image.tar.gz"

if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
else
    COMPOSE=(docker-compose)
fi

REPLACED=0
rollback_tarjomeh_on_error() {
    exit_code=$?
    if [ "$REPLACED" -eq 1 ]; then
        echo "========== AUTOMATIC TARJOMEH ROLLBACK =========="
        docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
        if docker image inspect "$ROLLBACK" >/dev/null 2>&1; then
            docker image tag "$ROLLBACK" translate_tarjomeh:latest
        elif [ -s "$ROLLBACK_ARCHIVE" ]; then
            gzip -dc "$ROLLBACK_ARCHIVE" | docker image load
        else
            echo "ERROR: no Tarjomeh rollback image is available."
            exit "$exit_code"
        fi
        "${COMPOSE[@]}" up -d --no-deps "$SERVICE" || true
        echo "The previous Tarjomeh image was restored. 9router was not changed."
    fi
    exit "$exit_code"
}
trap rollback_tarjomeh_on_error ERR

echo "========== PRECHECK =========="
git rev-parse HEAD
git status --short
df -h /
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"

OTHER_TRACKED="$(
    git status --porcelain --untracked-files=no |
    grep -v '^ M glossary/academic_political_theory.csv$' || true
)"
if [ -n "$OTHER_TRACKED" ]; then
    echo "STOPPED: tracked files other than the local glossary are modified."
    printf '%s\n' "$OTHER_TRACKED"
    exit 1
fi

ACTIVE_JOBS="$(
    docker exec -i "$CONTAINER" python - <<'PY' 2>/dev/null || true
import sqlite3
db = sqlite3.connect("/app/jobs/jobs.db")
rows = db.execute(
    "SELECT id,status FROM jobs WHERE status IN ('pending','running','pausing')"
).fetchall()
for job_id, status in rows:
    print(job_id, status)
PY
)"
if [ -n "$ACTIVE_JOBS" ]; then
    echo "STOPPED: an active job must finish or pause before deployment."
    printf '%s\n' "$ACTIVE_JOBS"
    exit 1
fi

NINE_IMAGE="$(docker inspect -f '{{.Image}}' 9router)"
NINE_STARTED="$(docker inspect -f '{{.State.StartedAt}}' 9router)"
NINE_CONTAINER_ID="$(docker inspect -f '{{.Id}}' 9router)"
NINE_MOUNTS="$(docker inspect -f '{{json .Mounts}}' 9router)"
OLD_IMAGE="$(docker inspect -f '{{.Image}}' "$CONTAINER")"
docker image tag "$OLD_IMAGE" "$ROLLBACK"

mkdir -p "$BACKUP"
gzip -c jobs/jobs.db > "$BACKUP/jobs.db.gz"
cp -a .env config.toml "$BACKUP/"
GLOSSARY_COPY=""
if ! git diff --quiet -- glossary/academic_political_theory.csv; then
    GLOSSARY_COPY="/tmp/tarjomeh-glossary-$STAMP.csv"
    cp -a glossary/academic_political_theory.csv "$GLOSSARY_COPY"
fi

echo "========== UPDATE SOURCE =========="
git fetch origin main --tags
git checkout main
git merge --ff-only origin/main
[ "$(git rev-parse HEAD)" = "$(git rev-list -n 1 "$TAG")" ] || {
    echo "STOPPED: origin/main is not $TAG"
    exit 1
}
if [ -n "$GLOSSARY_COPY" ]; then
    cp -a "$GLOSSARY_COPY" glossary/academic_political_theory.csv
    rm -f -- "$GLOSSARY_COPY"
    echo "Local glossary restored byte-for-byte."
fi

echo "========== SAFE SPACE CLEANUP =========="
apt-get clean
journalctl --vacuum-time=3d || true
journalctl --vacuum-size=80M || true
docker builder prune -af

# Remove only stopped containers from this Compose project, never a running
# service and never the 9router container.
while IFS= read -r stale_container; do
    [ -z "$stale_container" ] && continue
    [ "$stale_container" = "$(docker inspect -f '{{.Id}}' 9router)" ] && continue
    [ "$stale_container" = "$(docker inspect -f '{{.Id}}' "$CONTAINER")" ] && continue
    docker rm "$stale_container" >/dev/null 2>&1 || true
done < <(docker ps -aq \
    --filter "label=com.docker.compose.project=translate" \
    --filter "status=exited")

# Remove stale Tarjomeh rollback tags from older successful deployments. The
# current running image is retained under $ROLLBACK until this deployment is
# healthy. No 9router image name or container is ever passed to Docker removal.
while IFS= read -r stale_rollback; do
    [ -z "$stale_rollback" ] && continue
    [ "$stale_rollback" = "$ROLLBACK" ] && continue
    docker image rm "$stale_rollback" >/dev/null 2>&1 || true
done < <(docker images translate_tarjomeh --format '{{.Repository}}:{{.Tag}}' |
    grep '^translate_tarjomeh:rollback-' || true)

# Keep the fresh rollback backup and remove older Tarjomeh release backups.
# This is the largest safe source of space on the small VPS. The path guard
# prevents deletion outside BACKUP_ROOT; no 9router path is inspected or used.
mapfile -t OLD_BACKUPS < <(find "$BACKUP_ROOT" -mindepth 1 -maxdepth 1 \
    -type d ! -path "$BACKUP" -printf '%p\n')
for old in "${OLD_BACKUPS[@]}"; do
    case "$old" in
        "$BACKUP_ROOT"/*) rm -rf -- "$old" ;;
        *) echo "Refusing unexpected backup path: $old"; exit 1 ;;
    esac
done

# Keep the two newest legacy Tarjomeh database backups and remove only older
# files from Tarjomeh's own backup directory. Current jobs and uploads remain.
JOB_BACKUP_ROOT="/opt/translate/jobs/backups"
if [ -d "$JOB_BACKUP_ROOT" ]; then
    mapfile -t JOB_BACKUPS < <(find "$JOB_BACKUP_ROOT" -maxdepth 1 -type f \
        -printf '%T@ %p\n' | sort -nr | cut -d' ' -f2-)
    for ((index=2; index<${#JOB_BACKUPS[@]}; index++)); do
        old_job_backup="${JOB_BACKUPS[$index]}"
        case "$old_job_backup" in
            "$JOB_BACKUP_ROOT"/*) rm -f -- "$old_job_backup" ;;
            *) echo "Refusing unexpected job-backup path: $old_job_backup"; exit 1 ;;
        esac
    done
fi

# Remove only dangling images whose configured command identifies Tarjomeh.
# An unrelated dangling image, including any historical 9router image, is left
# untouched even when the VPS is short on space.
for image in $(docker images --filter dangling=true -q | sort -u); do
    [ "$image" = "$NINE_IMAGE" ] && continue
    [ "$image" = "$OLD_IMAGE" ] && continue
    image_command="$(
        docker image inspect -f '{{json .Config.Cmd}}' "$image" 2>/dev/null \
        || true
    )"
    case "$image_command" in
        *tarjomeh*) ;;
        *) continue ;;
    esac
    docker image rm "$image" >/dev/null 2>&1 || true
done

AVAILABLE_KB="$(df -Pk / | awk 'NR==2 {print $4}')"
df -h /
if [ "$AVAILABLE_KB" -lt 1450000 ]; then
    echo "========== GUARDED LOW-SPACE TARJOMEH FALLBACK =========="
    echo "Archiving and removing only the inactive Tarjomeh runtime image."
    docker image save translate_tarjomeh:latest | gzip -1 > "$ROLLBACK_ARCHIVE"
    [ -s "$ROLLBACK_ARCHIVE" ]
    REPLACED=1
    docker rm -f "$CONTAINER"
    docker image rm "$ROLLBACK" translate_tarjomeh:latest
    docker builder prune -af
    AVAILABLE_KB="$(df -Pk / | awk 'NR==2 {print $4}')"
    df -h /
    if [ "$AVAILABLE_KB" -lt 1450000 ]; then
        echo "STOPPED: less than 1.45 GB remains after the guarded fallback."
        false
    fi
fi

echo "========== BUILD TARJOMEH =========="
"${COMPOSE[@]}" build "$SERVICE"

echo "========== VERIFY IMAGE =========="
docker run --rm --entrypoint python translate_tarjomeh:latest -c '
from pathlib import Path
import tarjomeh
import tarjomeh.glossary
import tarjomeh.memory
import tarjomeh.quality
from tarjomeh.memory import MemoryManager
r = Path(tarjomeh.__file__).resolve().parent
sources = {
    "pipeline": (r / "core" / "pipeline.py").read_text(encoding="utf-8"),
    "client": (r / "core" / "llm_client.py").read_text(encoding="utf-8"),
    "config": (r / "core" / "config.py").read_text(encoding="utf-8"),
    "prompts": (r / "core" / "prompts.py").read_text(encoding="utf-8"),
    "manager": (r / "memory" / "manager.py").read_text(encoding="utf-8"),
    "proper": (r / "memory" / "proper_nouns.py").read_text(encoding="utf-8"),
    "research": (r / "context" / "book_researcher.py").read_text(encoding="utf-8"),
    "summary": (r / "memory" / "bilingual_summary.py").read_text(encoding="utf-8"),
    "typography": (r / "persian" / "typography.py").read_text(encoding="utf-8"),
    "critique": (r / "quality" / "critique.py").read_text(encoding="utf-8"),
    "integrity": (r / "quality" / "integrity.py").read_text(encoding="utf-8"),
    "structure": (r / "quality" / "structure_audit.py").read_text(encoding="utf-8"),
    "exporter": (r / "exporters" / "docx_exporter.py").read_text(encoding="utf-8"),
    "web": (r / "web" / "app.py").read_text(encoding="utf-8"),
    "database": (r / "jobs" / "database.py").read_text(encoding="utf-8"),
    "term_notes": (r / "core" / "term_notes.py").read_text(encoding="utf-8"),
}
checks = {
    "atomic_local_salvage": "coherent unit" in sources["pipeline"],
    "salvage_rollback": "refinement_salvage_rolled_back" in sources["pipeline"],
    "bounded_readability": "review_persian_readability" in sources["critique"],
    "readability_advisory": "target_only_advisory" in sources["pipeline"],
    "promoted_readability": (
        "_promote_objective_readability_issues" in sources["pipeline"]
        and "readability_advisory_decisions" in sources["pipeline"]
    ),
    "objective_minor_readability": (
        "_OBJECTIVE_GRAMMAR_RATIONALE_RE" in sources["pipeline"]
        and "objective_minor_grammar" in sources["pipeline"]
    ),
    "final_body_readability": (
        "candidate_changed or final_candidate" in sources["pipeline"]
    ),
    "conservative_memory_reliability": (
        "_DISQUALIFYING_RELIABILITY_REASONS" in sources["pipeline"]
        and "disqualifying_reliability_reasons" in sources["pipeline"]
    ),
    "derived_false_mi_repair": "segments = joined.split" in sources["typography"],
    "flattened_note_markers": "available_note_markers" in sources["integrity"],
    "grouped_table_recovery": "_table_recovery_groups" in sources["pipeline"],
    "latest_generation_qa": "current_events_by_chunk" in sources["web"],
    "context_memory_gate": "_context_bound_persian_target" in sources["proper"],
    "diacritic_preservation": "_PERSIAN_COMBINING_MARK_RE" in sources["typography"],
    "hazm_false_mi_repair": "_repair_false_mi_splits" in sources["typography"],
    "paragraph_anchor_guard": "before_paragraphs" in sources["pipeline"],
    "backcheck_advisory_report": "below_similarity_threshold=" in sources["web"],
    "four_layer_memory": "style_excluded_paragraphs" in sources["manager"],
    "summary_memory": "_DIRECTIONAL_CONTROL_RE" in sources["summary"],
    "accuracy_before_fluency": "Accuracy and completeness outrank fluency" in sources["prompts"],
    "full_candidate_rollback": "refinement_candidate_rolled_back" in sources["pipeline"],
    "late_readability_review": "readability_reviewed" in sources["pipeline"],
    "balanced_source_dash_audit": "unbalanced_explanatory_dash_count" in sources["pipeline"],
    "canonical_memory_typography": "final_candidate_typography" in sources["pipeline"],
    "citation_tail_protection": "_AUTHOR_YEAR_CITATION_RE" in sources["typography"],
    "latin_scholarly_abbreviations": (
        "_LATIN_SCHOLARLY_ABBREVIATION_RE" in sources["typography"]
    ),
    "semantic_tatweel_dash": "_SEMANTIC_TATWEEL_SEPARATOR_RE" in sources["typography"],
    "contextual_source_term_gate": "_SOURCE_NONTERM_TRAILERS" in sources["proper"],
    "table_line_identity": "_target_paragraphs_for_alignment" in sources["pipeline"],
    "table_placeholder_suppression": (
        "suppress_empty_target_export" in sources["pipeline"]
        and "suppress_empty_target_export" in sources["exporter"]
    ),
    "atomic_regression_salvage": (
        "_decisions_without_regressed_edits" in sources["pipeline"]
    ),
    "post_rollback_validation": (
        "post_rollback_final_validation" in sources["pipeline"]
    ),
    "source_announcement_fidelity": (
        "announced_count_lexical_mismatch" in sources["structure"]
        and "source-authored announced quantity" in sources["prompts"]
    ),
    "short_span_duplicate_guard": (
        "newly_source_unjustified_repeated_adjacent_spans"
        in sources["integrity"]
        and "repeated_adjacent_span_count" in sources["pipeline"]
    ),
    "local_predicate_guard": (
        "local_predicate_evidence_removed" in sources["pipeline"]
        and "predicate_regressions" in sources["pipeline"]
    ),
    "source_relative_governed_repetition": (
        "newly_source_unjustified_repeated_governed_spans"
        in sources["integrity"]
        and "repeated_governed_span_count" in sources["pipeline"]
    ),
    "coordinated_proposition_review": (
        "matrix action, coordinated actions" in sources["prompts"]
        and "verify every coordinated source member separately"
        in sources["prompts"]
    ),
    "terminology_authority_classes": (
        "_mapping_authority_class" in sources["proper"]
        and "recurring_advisory" in sources["proper"]
        and "canonical_reviewed" in sources["proper"]
    ),
    "quality_ranked_style_memory": (
        "_style_sample_quality" in sources["manager"]
        and "weaker_style_sample_replaced" in sources["manager"]
    ),
    "style_memory_floor": (
        "style_min_score" in sources["config"]
        and "style_score_below_floor" in sources["manager"]
    ),
    "conceptual_family_contract": (
        "recurring coordinated conceptual series" in sources["prompts"]
    ),
    "attributable_advisory_research": (
        "identity_supported" in sources["research"]
        and "supporting_excerpts" in sources["research"]
        and "advisory_context_only" in sources["research"]
    ),
    "readability_source_authority": (
        "readability_advisory_suppressed_source_conflict"
        in sources["pipeline"]
    ),
    "structural_reference_localization": (
        "source_structural_reference" in sources["integrity"]
    ),
    "matrix_predicate_prompt": (
        "matrix predicate" in sources["prompts"]
        and "relative clause" in sources["prompts"]
    ),
    "source_faithful_version_recovery": (
        "refinement_best_source_version_restored" in sources["pipeline"]
        and "_best_source_faithful_version" in sources["pipeline"]
    ),
    "grounded_memory_quarantine": (
        "unresolved_grounded_quality_issue" in sources["pipeline"]
    ),
    "sense_scoped_entity_memory": (
        "_strip_redundant_entity_original" in sources["proper"]
        and "_mapping_applies_to_source" in sources["proper"]
    ),
    "term_specific_research_evidence": (
        "term_supporting_excerpts" in sources["research"]
        and "term_supported" in sources["research"]
    ),
    "durable_worker_lease": (
        "CREATE TABLE IF NOT EXISTS job_workers" in sources["database"]
        and "worker_pause_acknowledged" in sources["pipeline"]
        and "_start_lease_heartbeat" in sources["pipeline"]
        and "request_job_pause" in sources["web"]
    ),
    "canonical_final_quality_authority": (
        "_canonical_final_quality_record" in sources["pipeline"]
        and "final_quality_admission" in sources["pipeline"]
        and "final_quality_authority" in sources["pipeline"]
    ),
    "bounded_source_obligation_repair": (
        "source_fidelity_findings" in sources["pipeline"]
        and "resolved_source_issue_ids" in sources["pipeline"]
        and "_language_quality_does_not_regress" in sources["pipeline"]
    ),
    "diacritic_aware_duplicate_repair": (
        "_repeat_token_key" in sources["integrity"]
        and "adjacent_duplicate_phrase" in sources["integrity"]
    ),
    "observable_llm_stage": (
        "\"phase\": \"started\"" in sources["client"]
        and "llm_call_started" in sources["pipeline"]
        and "get_worker_lease(job_id)" in sources["web"]
    ),
    "monotonic_source_obligations": (
        "_source_obligation_identity" in sources["pipeline"]
        and "refinement_atomic_recovery_rolled_back" in sources["pipeline"]
    ),
    "short_token_duplicate_repair": (
        "lexical_length < 2" in sources["integrity"]
        and "adjacent_duplicate_phrase" in sources["integrity"]
    ),
    "transactional_summary_admission": (
        "candidate_quality" in sources["summary"]
        and "candidate_committed" in sources["manager"]
    ),
    "complete_worker_lifecycle": (
        "worker_lease_released" in sources["database"]
        and "worker_pause_requested" in sources["database"]
        and "state!=\x27released\x27" in sources["database"]
    ),
    "citation_bound_governed_repetition": (
        "contiguous_phrase" in sources["integrity"]
    ),
    "source_grounded_original_preservation": (
        "preserved_source_grounded" in sources["term_notes"]
        and "merge_inline_english_original_audits" in sources["pipeline"]
    ),
    "representative_style_source_gate": (
        "source_genre_not_representative_of_body_voice" in sources["manager"]
    ),
    "durable_chunk_terminal_failure": (
        "chunk_terminal_failure" in sources["pipeline"]
    ),
    "source_monotonic_structure_admission": (
        "_actionable_structure_findings" in sources["pipeline"]
        and "refinement_source_structure_admission" in sources["pipeline"]
        and "final_source_structure_admission" in sources["pipeline"]
    ),
    "paragraph_scoped_style_evidence": (
        "source_paragraph_index" in sources["manager"]
    ),
    "parenthetical_anchor_deferral": (
        "parenthetical_deferred" in sources["term_notes"]
    ),
    "source_grounded_units": "_MEASUREMENT_UNIT_TOKENS" in sources["integrity"],
    "latin_title_punctuation": "_LATIN_PUNCTUATED_RUN_RE" in sources["typography"],
    "lazy_public_memory_import": MemoryManager.__name__ == "MemoryManager",
    "unique_note_marker_recovery": (
        "restore_source_note_markers" in sources["integrity"]
        and "note_marker_repair_count" in sources["pipeline"]
    ),
    "final_canonical_admission": (
        "source_bound_artifact_recovery" in sources["pipeline"]
        and "final_canonical_admission" in sources["pipeline"]
    ),
    "independent_refiner_salvage": (
        "coherent_local_edits_committed" in sources["pipeline"]
        and "local_edit_committed" in sources["pipeline"]
    ),
    "two_stage_observed_entity_authority": (
        "_SEMANTICALLY_TRANSLATED_ENTITY_CATEGORIES" in sources["proper"]
        and "observed_entity_advisory" in sources["proper"]
    ),
    "bilingual_summary_source_guard": (
        "_summary_lexical_near_misses" in sources["manager"]
        and "bilingual_summary_structure_mismatch" in sources["manager"]
    ),
    "source_genre_style_filter": (
        "style_source_genre_excluded_paragraphs" in sources["manager"]
    ),
}
print(checks)
assert all(checks.values()), checks
print("IMAGE_V1024_CONFIRMED")
'

echo "========== RECREATE TARJOMEH ONLY =========="
REPLACED=1
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
"${COMPOSE[@]}" up -d --no-deps "$SERVICE"

echo "========== WAIT FOR HEALTH =========="
for attempt in $(seq 1 30); do
    status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$CONTAINER")"
    echo "Attempt $attempt: $status"
    [ "$status" = "healthy" ] && break
    sleep 5
done
[ "$(docker inspect -f '{{.State.Health.Status}}' "$CONTAINER")" = "healthy" ]

echo "========== VERIFY RUNNING V10.24 =========="
docker exec -i "$CONTAINER" python -c '
from pathlib import Path
import tarjomeh
import tarjomeh.glossary
import tarjomeh.memory
import tarjomeh.quality
from tarjomeh.memory import MemoryManager
r = Path(tarjomeh.__file__).resolve().parent
p = (r / "core" / "pipeline.py").read_text(encoding="utf-8")
cfg = (r / "core" / "config.py").read_text(encoding="utf-8")
m = (r / "memory" / "manager.py").read_text(encoding="utf-8")
n = (r / "memory" / "proper_nouns.py").read_text(encoding="utf-8")
rsearch = (r / "context" / "book_researcher.py").read_text(encoding="utf-8")
s = (r / "memory" / "bilingual_summary.py").read_text(encoding="utf-8")
t = (r / "persian" / "typography.py").read_text(encoding="utf-8")
c = (r / "quality" / "critique.py").read_text(encoding="utf-8")
i = (r / "quality" / "integrity.py").read_text(encoding="utf-8")
x = (r / "exporters" / "docx_exporter.py").read_text(encoding="utf-8")
a = (r / "quality" / "structure_audit.py").read_text(encoding="utf-8")
q = (r / "core" / "prompts.py").read_text(encoding="utf-8")
w = (r / "web" / "app.py").read_text(encoding="utf-8")
d = (r / "jobs" / "database.py").read_text(encoding="utf-8")
l = (r / "core" / "llm_client.py").read_text(encoding="utf-8")
o = (r / "core" / "term_notes.py").read_text(encoding="utf-8")
checks = {
    "atomic_local_salvage": "coherent unit" in p,
    "salvage_rollback": "refinement_salvage_rolled_back" in p,
    "bounded_readability": "review_persian_readability" in c,
    "promoted_readability": (
        "_promote_objective_readability_issues" in p
        and "readability_advisory_decisions" in p
    ),
    "objective_minor_readability": (
        "_OBJECTIVE_GRAMMAR_RATIONALE_RE" in p
        and "objective_minor_grammar" in p
    ),
    "final_body_readability": "candidate_changed or final_candidate" in p,
    "conservative_memory_reliability": (
        "_DISQUALIFYING_RELIABILITY_REASONS" in p
        and "disqualifying_reliability_reasons" in p
    ),
    "derived_false_mi_repair": "segments = joined.split" in t,
    "flattened_note_markers": "available_note_markers" in i,
    "grouped_table_recovery": "_table_recovery_groups" in p,
    "latest_generation_qa": "current_events_by_chunk" in w,
    "context_memory_gate": "_context_bound_persian_target" in n,
    "diacritic_preservation": "_PERSIAN_COMBINING_MARK_RE" in t,
    "hazm_false_mi_repair": "_repair_false_mi_splits" in t,
    "paragraph_anchor_guard": "before_paragraphs" in p,
    "backcheck_advisory_report": "below_similarity_threshold=" in w,
    "four_layer_memory": "style_excluded_paragraphs" in m,
    "summary_memory": "_DIRECTIONAL_CONTROL_RE" in s,
    "full_candidate_rollback": "refinement_candidate_rolled_back" in p,
    "late_readability_review": "readability_reviewed" in p,
    "balanced_source_dash_audit": "unbalanced_explanatory_dash_count" in p,
    "canonical_memory_typography": "final_candidate_typography" in p,
    "citation_tail_protection": "_AUTHOR_YEAR_CITATION_RE" in t,
    "latin_scholarly_abbreviations": "_LATIN_SCHOLARLY_ABBREVIATION_RE" in t,
    "semantic_tatweel_dash": "_SEMANTIC_TATWEEL_SEPARATOR_RE" in t,
    "contextual_source_term_gate": "_SOURCE_NONTERM_TRAILERS" in n,
    "table_line_identity": "_target_paragraphs_for_alignment" in p,
    "table_placeholder_suppression": (
        "suppress_empty_target_export" in p
        and "suppress_empty_target_export" in x
    ),
    "atomic_regression_salvage": "_decisions_without_regressed_edits" in p,
    "post_rollback_validation": "post_rollback_final_validation" in p,
    "source_announcement_fidelity": (
        "announced_count_lexical_mismatch" in a
        and "source-authored announced quantity" in q
    ),
    "short_span_duplicate_guard": (
        "newly_source_unjustified_repeated_adjacent_spans" in i
        and "repeated_adjacent_span_count" in p
    ),
    "local_predicate_guard": (
        "local_predicate_evidence_removed" in p
        and "predicate_regressions" in p
    ),
    "source_relative_governed_repetition": (
        "newly_source_unjustified_repeated_governed_spans" in i
        and "repeated_governed_span_count" in p
    ),
    "coordinated_proposition_review": (
        "matrix action, coordinated actions" in q
        and "verify every coordinated source member separately" in q
    ),
    "terminology_authority_classes": (
        "_mapping_authority_class" in n
        and "recurring_advisory" in n
        and "canonical_reviewed" in n
    ),
    "quality_ranked_style_memory": (
        "_style_sample_quality" in m
        and "weaker_style_sample_replaced" in m
    ),
    "style_memory_floor": (
        "style_min_score" in cfg and "style_score_below_floor" in m
    ),
    "conceptual_family_contract": (
        "recurring coordinated conceptual series" in q
    ),
    "attributable_advisory_research": (
        "identity_supported" in rsearch
        and "supporting_excerpts" in rsearch
        and "advisory_context_only" in rsearch
    ),
    "readability_source_authority": (
        "readability_advisory_suppressed_source_conflict" in p
    ),
    "structural_reference_localization": (
        "source_structural_reference" in i
    ),
    "matrix_predicate_prompt": (
        "matrix predicate" in q and "relative clause" in q
    ),
    "source_faithful_version_recovery": (
        "refinement_best_source_version_restored" in p
        and "_best_source_faithful_version" in p
    ),
    "grounded_memory_quarantine": "unresolved_grounded_quality_issue" in p,
    "sense_scoped_entity_memory": "_strip_redundant_entity_original" in n,
    "term_specific_research_evidence": (
        "term_supporting_excerpts" in rsearch and "term_supported" in rsearch
    ),
    "durable_worker_lease": (
        "CREATE TABLE IF NOT EXISTS job_workers" in d
        and "worker_pause_acknowledged" in p
        and "_start_lease_heartbeat" in p
        and "request_job_pause" in w
    ),
    "canonical_final_quality_authority": (
        "_canonical_final_quality_record" in p
        and "final_quality_admission" in p
        and "final_quality_authority" in p
    ),
    "bounded_source_obligation_repair": (
        "source_fidelity_findings" in p
        and "resolved_source_issue_ids" in p
        and "_language_quality_does_not_regress" in p
    ),
    "diacritic_aware_duplicate_repair": (
        "_repeat_token_key" in i and "adjacent_duplicate_phrase" in i
    ),
    "observable_llm_stage": (
        "\"phase\": \"started\"" in l
        and "llm_call_started" in p
        and "get_worker_lease(job_id)" in w
    ),
    "monotonic_source_obligations": (
        "_source_obligation_identity" in p
        and "refinement_atomic_recovery_rolled_back" in p
    ),
    "short_token_duplicate_repair": (
        "lexical_length < 2" in i and "adjacent_duplicate_phrase" in i
    ),
    "transactional_summary_admission": (
        "candidate_quality" in s and "candidate_committed" in m
    ),
    "complete_worker_lifecycle": (
        "worker_lease_released" in d
        and "worker_pause_requested" in d
        and "state!=\x27released\x27" in d
    ),
    "citation_bound_governed_repetition": "contiguous_phrase" in i,
    "source_grounded_original_preservation": (
        "preserved_source_grounded" in o
        and "merge_inline_english_original_audits" in p
    ),
    "representative_style_source_gate": (
        "source_genre_not_representative_of_body_voice" in m
    ),
    "durable_chunk_terminal_failure": "chunk_terminal_failure" in p,
    "source_monotonic_structure_admission": (
        "_actionable_structure_findings" in p
        and "refinement_source_structure_admission" in p
        and "final_source_structure_admission" in p
    ),
    "paragraph_scoped_style_evidence": "source_paragraph_index" in m,
    "parenthetical_anchor_deferral": "parenthetical_deferred" in o,
    "source_grounded_units": "_MEASUREMENT_UNIT_TOKENS" in i,
    "latin_title_punctuation": "_LATIN_PUNCTUATED_RUN_RE" in t,
    "lazy_public_memory_import": MemoryManager.__name__ == "MemoryManager",
    "unique_note_marker_recovery": (
        "restore_source_note_markers" in i
        and "note_marker_repair_count" in p
    ),
    "final_canonical_admission": (
        "source_bound_artifact_recovery" in p
        and "final_canonical_admission" in p
    ),
    "independent_refiner_salvage": (
        "coherent_local_edits_committed" in p
        and "local_edit_committed" in p
    ),
    "two_stage_observed_entity_authority": (
        "_SEMANTICALLY_TRANSLATED_ENTITY_CATEGORIES" in n
        and "observed_entity_advisory" in n
    ),
    "bilingual_summary_source_guard": (
        "_summary_lexical_near_misses" in m
        and "bilingual_summary_structure_mismatch" in m
    ),
    "source_genre_style_filter": (
        "style_source_genre_excluded_paragraphs" in m
    ),
}
print(checks)
assert all(checks.values()), checks
print("RUNNING_V1024_CONFIRMED")
'

echo "========== VERIFY 9ROUTER UNCHANGED =========="
[ "$(docker inspect -f '{{.Image}}' 9router)" = "$NINE_IMAGE" ]
[ "$(docker inspect -f '{{.State.StartedAt}}' 9router)" = "$NINE_STARTED" ]
[ "$(docker inspect -f '{{.Id}}' 9router)" = "$NINE_CONTAINER_ID" ]
[ "$(docker inspect -f '{{json .Mounts}}' 9router)" = "$NINE_MOUNTS" ]
REPLACED=0

echo "========== TARJOMEH-ONLY FINAL CLEANUP =========="
docker image rm "$ROLLBACK" >/dev/null 2>&1 || true
rm -f -- "$ROLLBACK_ARCHIVE"
docker builder prune -af

echo "========== FINAL STATE =========="
git rev-parse HEAD
git tag --points-at HEAD
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
df -h /
docker system df
echo "Deployment $TAG completed. Running 9router was unchanged."
