#!/usr/bin/env bash
set -Eeuo pipefail

cd /opt/translate

TAG="v10.18.0"
CONTAINER="translate_tarjomeh_1"
SERVICE="tarjomeh"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_ROOT="/root/tarjomeh-backups"
BACKUP="$BACKUP_ROOT/v1018-$STAMP"
ROLLBACK="translate_tarjomeh:rollback-v1018-$STAMP"

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
        docker image tag "$ROLLBACK" translate_tarjomeh:latest
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
    "SELECT id,status FROM jobs WHERE status IN ('processing','pausing')"
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
if [ "$AVAILABLE_KB" -lt 1900000 ]; then
    echo "STOPPED: less than 1.90 GB is available after safe cleanup."
    echo "Source and backup are ready; no container was replaced."
    exit 1
fi

echo "========== BUILD TARJOMEH =========="
"${COMPOSE[@]}" build --no-cache "$SERVICE"

echo "========== VERIFY IMAGE =========="
docker run --rm --entrypoint python translate_tarjomeh:latest -c '
from pathlib import Path
import tarjomeh
r = Path(tarjomeh.__file__).resolve().parent
sources = {
    "pipeline": (r / "core" / "pipeline.py").read_text(encoding="utf-8"),
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
}
print(checks)
assert all(checks.values()), checks
print("IMAGE_V1018_CONFIRMED")
'

echo "========== RECREATE TARJOMEH ONLY =========="
REPLACED=1
docker rm -f "$CONTAINER"
"${COMPOSE[@]}" up -d --no-deps "$SERVICE"

echo "========== WAIT FOR HEALTH =========="
for attempt in $(seq 1 30); do
    status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$CONTAINER")"
    echo "Attempt $attempt: $status"
    [ "$status" = "healthy" ] && break
    sleep 5
done
[ "$(docker inspect -f '{{.State.Health.Status}}' "$CONTAINER")" = "healthy" ]

echo "========== VERIFY RUNNING V10.18 =========="
docker exec -i "$CONTAINER" python -c '
from pathlib import Path
import tarjomeh
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
}
print(checks)
assert all(checks.values()), checks
print("RUNNING_V1018_CONFIRMED")
'

echo "========== VERIFY 9ROUTER UNCHANGED =========="
[ "$(docker inspect -f '{{.Image}}' 9router)" = "$NINE_IMAGE" ]
[ "$(docker inspect -f '{{.State.StartedAt}}' 9router)" = "$NINE_STARTED" ]
REPLACED=0

echo "========== TARJOMEH-ONLY FINAL CLEANUP =========="
docker image rm "$ROLLBACK" >/dev/null 2>&1 || true
docker builder prune -af

echo "========== FINAL STATE =========="
git rev-parse HEAD
git tag --points-at HEAD
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
df -h /
docker system df
echo "Deployment $TAG completed. Running 9router was unchanged."
