#!/usr/bin/env bash
# Explicit administrative route only; ordinary translation never invokes it.
set -Eeuo pipefail
umask 077
cd /opt/translate
ACTION="${1:?Pass prepare, check, apply or rollback}"
FILE="${2:?Pass a regular JSON file under /opt/translate/jobs/uploads}"
shift 2
case "$ACTION" in prepare|check|apply|rollback) ;; *) exit 2 ;; esac
test "$(id -u)" = 0
command -v flock >/dev/null
exec 9>/root/tarjomeh-deploy.lock
flock -n 9 || { echo 'STOPPED: deployment/trial in progress.' >&2; exit 1; }
exec 8>/run/lock/9router-updater.lock
flock -n 8 || { echo 'STOPPED: router update in progress.' >&2; exit 1; }
test ! -L "$FILE"
test -f "$FILE"
FILE="$(realpath -e -- "$FILE")"
case "$FILE" in /opt/translate/jobs/uploads/*.json) ;; *) exit 2 ;; esac
C=translate_tarjomeh_1
test "$(docker inspect -f '{{.State.Health.Status}}' "$C")" = healthy
JOBS_BIND="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/app/jobs"}}{{.Source}}{{end}}{{end}}' "$C")"
test -n "$JOBS_BIND"
test "$(realpath -e -- "$JOBS_BIND")" = "$(realpath -e -- /opt/translate/jobs)"
test "$(docker inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$C")" = "$(git rev-parse 'v22.4^{commit}')"
docker exec "$C" python -c '
from tarjomeh.runtime import runtime_capabilities, runtime_behavior_probes
m, p = runtime_capabilities(), runtime_behavior_probes()
assert m["release"] == "v22.4" and m["capabilities"]["reviewed_summary_revision"]
assert p and all(p.values())
' >&2
ARGS=()
if [[ "$ACTION" == apply || "$ACTION" == rollback ]]; then
    test "$#" = 2 || { echo 'Writes require exact approval hash and reviewer as arguments.' >&2; exit 2; }
    BACKUP="$(mktemp /opt/translate/jobs/summary-revision-XXXXXX.sqlite)"
    # The writer requires exclusive creation. Reserve the random name, not a file.
    rm -- "$BACKUP"
    ARGS=(--approve-sha256 "$1" --reviewer "$2" --operator-locks-held
          --backup "/app/jobs/${BACKUP##*/}")
else
    test "$#" = 0
fi
docker exec "$C" python -m tarjomeh.jobs.summary_revision "$ACTION" \
    --db /app/jobs/jobs.db --file "/app/jobs/uploads/${FILE#/opt/translate/jobs/uploads/}" "${ARGS[@]}"
