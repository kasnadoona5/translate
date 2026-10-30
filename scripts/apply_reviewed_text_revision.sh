#!/usr/bin/env bash
# Explicit maintenance only; deployment and normal translation never invoke this.
set -Eeuo pipefail
umask 077
cd /opt/translate
test "$(id -u)" = 0
MANIFEST="${1:?Pass the reviewed manifest under /opt/translate/jobs/uploads}"
APPROVAL="${2:?Pass the separately approved manifest SHA256}"
REVIEWER="${3:?Pass the explicit reviewer identity}"
[[ "$APPROVAL" =~ ^[a-f0-9]{64}$ ]]
test ! -L "$MANIFEST"
test -f "$MANIFEST"
MANIFEST="$(realpath -e -- "$MANIFEST")"
case "$MANIFEST" in /opt/translate/jobs/uploads/*) ;; *) echo 'STOPPED: manifest must be in jobs/uploads.'; exit 1 ;; esac
exec 9>/root/tarjomeh-deploy.lock
flock -n 9 || { echo 'STOPPED: Tarjomeh maintenance/deployment is running.'; exit 1; }
exec 8>/run/lock/9router-updater.lock
flock -n 8 || { echo 'STOPPED: router updater is running.'; exit 1; }
C=translate_tarjomeh_1
test "$(docker inspect -f '{{.State.Health.Status}}' "$C")" = healthy
IMAGE="$(docker inspect -f '{{.Image}}' "$C")"
REVISION="$(git rev-parse 'v22.10^{commit}')"
test "$(git rev-parse HEAD)" = "$REVISION"
test "$(docker inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$C")" = "$REVISION"
test "$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/app/jobs"}}{{.Source}}{{end}}{{end}}' "$C")" = /opt/translate/jobs
fingerprint() {
    docker inspect 9router | python3 -c '
import json,sys
d=json.load(sys.stdin)[0]
assert d["State"]["Running"]
print(json.dumps(dict(Id=d["Id"],Image=d["Image"],StartedAt=d["State"]["StartedAt"],
    Mounts=sorted(d["Mounts"],key=lambda m: json.dumps(m,sort_keys=True))),sort_keys=True))
'
}
ROUTER_BEFORE="$(fingerprint)"
INFILE="/app/jobs/${MANIFEST#/opt/translate/jobs/}"
docker exec "$C" tarjomeh capabilities --verify --json
docker exec -i "$C" python - <<'PY'
import shutil,sqlite3
with sqlite3.connect('file:/app/jobs/jobs.db?mode=ro',uri=True) as db:
    size=db.execute('PRAGMA page_count').fetchone()[0]*db.execute('PRAGMA page_size').fetchone()[0]
assert shutil.disk_usage('/app/jobs').free >= size+64*1024*1024, 'STOPPED: insufficient backup space'
PY
docker exec "$C" python -m tarjomeh.jobs.text_revision check --db /app/jobs/jobs.db --file "$INFILE"
docker exec -i "$C" python - "$INFILE" "$APPROVAL" <<'PY'
import json,sys
from pathlib import Path
from tarjomeh.memory.summary_revision import document_hash
assert document_hash(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))) == sys.argv[2], 'STOPPED: approval hash differs'
PY
DIR="$(mktemp -d /opt/translate/jobs/reviewed-text-revision-XXXXXX)"
RESTART_NEEDED=0
finish() {
    rc=$?
    trap - EXIT
    if [ "$RESTART_NEEDED" -eq 1 ]; then
        docker start "$C" >/dev/null || rc=1
    fi
    if ! test "$(fingerprint)" = "$ROUTER_BEFORE"; then rc=1; fi
    if [ "$rc" -ne 0 ]; then
        echo "STOPPED: keep the job paused and send the full log and $DIR. Do not retry or restore blindly."
    fi
    exit "$rc"
}
trap finish EXIT
RESTART_NEEDED=1
docker stop --time 60 "$C" >/dev/null
test "$(docker inspect -f '{{.State.Running}}' "$C")" = false
docker run --rm --network none --volumes-from "$C" --workdir /app --entrypoint python "$IMAGE" \
  -m tarjomeh.jobs.text_revision apply --db /app/jobs/jobs.db --file "$INFILE" \
  --approve-sha256 "$APPROVAL" --reviewer "$REVIEWER" --operator-locks-held \
  --backup "/app/jobs/${DIR##*/}/jobs.sqlite" > "$DIR/application-receipt.json"
docker start "$C" >/dev/null
RESTART_NEEDED=0
for attempt in $(seq 1 60); do
    if test "$(docker inspect -f '{{.State.Health.Status}}' "$C")" = healthy; then break; fi
    sleep 2
done
test "$(docker inspect -f '{{.State.Health.Status}}' "$C")" = healthy
docker exec "$C" tarjomeh capabilities --verify --json
docker exec "$C" python -m tarjomeh.jobs.text_revision check --db /app/jobs/jobs.db --file "$INFILE"
test "$(fingerprint)" = "$ROUTER_BEFORE"
echo "APPLIED: only the exact approved canonical revision. Job remains paused; 9router unchanged."
echo "Receipt and verified SQLite backup: $DIR"
echo 'A separate saved-prefix REVIEW export is still required; the previous DOCX was retained.'
