#!/usr/bin/env bash
# Update the known single-container 9router installation on the Tarjomeh VPS.
set -Eeuo pipefail
umask 077

DATA=/opt/translate/9router-data
BACKUP_DIR=/root/9router-updater
IMAGE=decolua/9router:latest
STAMP="$(date -u +%Y%m%d-%H%M%S)"
OLD_NAME="9router-before-updater-$STAMP"
OLD_TAG="decolua/9router:rollback-updater-$STAMP"
BACKUP="$BACKUP_DIR/9router-data-before-$STAMP.tar.gz"
PARTIAL="$BACKUP.partial"
RECEIPT="$BACKUP_DIR/success-$STAMP"
FAILED_DATA="$DATA.failed-updater-$STAMP"
STOP_ATTEMPTED=0
NEW_ATTEMPTED=0
BACKUP_READY=0
COMMITTED=0

say() { printf '%s\n' "$*"; }
die() { say "STOPPED: $*" >&2; exit 1; }
free_kb() { df -Pk / | awk 'NR==2 {print $4}'; }

health_ok() {
    local response
    response="$(curl -fsS --max-time 5 http://127.0.0.1:20128/api/health 2>/dev/null)" || return 1
    [[ "$response" =~ \"ok\"[[:space:]]*:[[:space:]]*true ]]
}

no_active_tarjomeh_work() {
    docker exec -i translate_tarjomeh_1 python - <<'PY'
import sqlite3
import sys

db = sqlite3.connect('file:/app/jobs/jobs.db?mode=ro', uri=True, timeout=10)
jobs = db.execute(
    "SELECT id, status FROM jobs WHERE status IN "
    "('pending', 'running', 'processing', 'pausing', 'resuming')"
).fetchall()
workers = db.execute(
    "SELECT job_id, state FROM job_workers WHERE state IN ('active', 'pausing')"
).fetchall()
if jobs or workers:
    print('Active jobs:', jobs)
    print('Active workers:', workers)
    sys.exit(1)
print('Tarjomeh has no active jobs or worker leases.')
PY
}

rollback() {
    local original_rc=$?
    trap - EXIT INT TERM
    (( COMMITTED == 0 && original_rc != 0 )) || return 0
    if (( STOP_ATTEMPTED == 0 )); then
        say "Update stopped before touching the running 9router."
        return 0
    fi
    say "Update failed; recovering the previous 9router."
    if (( BACKUP_READY == 0 )); then
        rm -f -- "$PARTIAL"
    fi

    if (( NEW_ATTEMPTED )); then
        # The new version may have migrated SQLite. Never restart the old
        # binary on the new version's possibly changed data directory.
        if docker container inspect 9router >/dev/null 2>&1; then
            if [[ "$(docker inspect -f '{{.Image}}' 9router)" != "$NEW_IMAGE" ]]; then
                say "URGENT: another container owns the 9router name; manual recovery required." >&2
                return 0
            fi
            docker rm -f 9router >/dev/null || {
                say "URGENT: could not remove the failed new container." >&2
                return 0
            }
        fi
        if (( BACKUP_READY == 0 )); then
            say "URGENT: no verified data snapshot is available; manual recovery required." >&2
            return 0
        fi
        if [[ -e "$FAILED_DATA" ]]; then
            say "URGENT: failed-data path already exists: $FAILED_DATA" >&2
            return 0
        fi
        mv -- "$DATA" "$FAILED_DATA" || {
            say "URGENT: could not preserve the new version's data." >&2
            return 0
        }
        tar -C /opt/translate -xzf "$BACKUP" || {
            say "URGENT: snapshot restore failed; old 9router remains stopped." >&2
            say "Snapshot: $BACKUP; failed data: $FAILED_DATA" >&2
            return 0
        }
    fi

    if docker container inspect "$OLD_NAME" >/dev/null 2>&1; then
        docker rename "$OLD_NAME" 9router || {
            say "URGENT: could not restore the old container name." >&2
            return 0
        }
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' 9router 2>/dev/null || true)" != true ]]; then
        docker start 9router >/dev/null || {
            say "URGENT: could not restart the old 9router." >&2
            return 0
        }
    fi
    say "Previous 9router restarted. Snapshot: $BACKUP"
    (( NEW_ATTEMPTED == 0 )) || say "Preserved failed data: $FAILED_DATA"
}
trap rollback EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ $# -eq 0 ]] || die 'This updater takes no arguments.'
[[ $EUID -eq 0 ]] || die 'Run this command as root.'
for command in docker curl tar df awk flock date stat sed cut sort du mv rm sleep; do
    command -v "$command" >/dev/null || die "Missing required command: $command"
done
exec 9>/run/lock/9router-updater.lock
flock -n 9 || die 'Another 9router update is already running.'
mkdir -p -m 700 "$BACKUP_DIR"
[[ ! -e "$BACKUP" && ! -e "$PARTIAL" && ! -e "$FAILED_DATA" ]] || die 'Timestamp collision in backup paths.'
! docker container inspect "$OLD_NAME" >/dev/null 2>&1 || die 'Timestamp collision in rollback container name.'

[[ "$(docker inspect -f '{{.State.Running}}' 9router)" == true ]] || die '9router is not running.'
[[ "$(docker inspect -f '{{len .Mounts}}' 9router)" == 1 ]] || die 'Unexpected 9router mounts; update manually.'
[[ "$(docker inspect -f '{{range .Mounts}}{{.Source}}:{{.Destination}}{{end}}' 9router)" == "$DATA:/app/data" ]] || die 'Unexpected data mount.'
[[ "$(docker inspect -f '{{.HostConfig.NetworkMode}}' 9router)" == bridge ]] || die 'Unexpected network mode.'
[[ "$(docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' 9router)" == no ]] || die 'Unexpected restart policy.'
[[ "$(docker inspect -f '{{json .HostConfig.PortBindings}}' 9router)" == '{"20128/tcp":[{"HostIp":"","HostPort":"20128"}]}' ]] || die 'Unexpected port binding.'
[[ -s "$DATA/db/data.sqlite" ]] || die '9router SQLite database is absent or empty.'
health_ok || die 'Current 9router failed /api/health.'
no_active_tarjomeh_work || die 'Tarjomeh is busy or its job state could not be read.'

# This VPS has only the known runtime environment. Abort if the installation has
# acquired additional settings, rather than silently dropping them.
ENV_NAMES="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' 9router | cut -d= -f1 | sort)"
EXPECTED_ENV="$(printf '%s\n' DATA_DIR HOSTNAME NEXT_TELEMETRY_DISABLED NODE_ENV NODE_VERSION PATH PORT YARN_VERSION | sort)"
[[ "$ENV_NAMES" == "$EXPECTED_ENV" ]] || die '9router environment has changed; update manually to preserve it.'
[[ "$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' 9router | sed -n 's/^DATA_DIR=//p')" == /app/data ]] || die 'Unexpected DATA_DIR.'
[[ "$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' 9router | sed -n 's/^PORT=//p')" == 20128 ]] || die 'Unexpected PORT.'
[[ "$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' 9router | sed -n 's/^HOSTNAME=//p')" == 0.0.0.0 ]] || die 'Unexpected HOSTNAME.'

OLD_IMAGE="$(docker inspect -f '{{.Image}}' 9router)"
OLD_CONTAINER="$(docker inspect -f '{{.Id}}' 9router)"
OLD_STARTED="$(docker inspect -f '{{.State.StartedAt}}' 9router)"
TARJOMEH_CONTAINER="$(docker inspect -f '{{.Id}}' translate_tarjomeh_1)"
TARJOMEH_IMAGE="$(docker inspect -f '{{.Image}}' translate_tarjomeh_1)"
TARJOMEH_STARTED="$(docker inspect -f '{{.State.StartedAt}}' translate_tarjomeh_1)"
[[ "$(docker inspect -f '{{.State.Health.Status}}' translate_tarjomeh_1)" == healthy ]] || die 'Tarjomeh is not healthy.'
say "Current image: $OLD_IMAGE"
say "Disk before update: $(free_kb) KB free"

# Reclaim only rollback artifacts made by an earlier successful run of this
# updater, after at least 24 hours of healthy operation. User-made containers,
# image tags, and every Tarjomeh artifact are outside this cleanup scope.
for stale in $(docker ps -a --format '{{.Names}}'); do
    [[ "$stale" =~ ^9router-before-updater-([0-9]{8}-[0-9]{6})$ ]] || continue
    stale_stamp="${BASH_REMATCH[1]}"
    stale_backup="$BACKUP_DIR/9router-data-before-$stale_stamp.tar.gz"
    stale_receipt="$BACKUP_DIR/success-$stale_stamp"
    stale_tag="decolua/9router:rollback-updater-$stale_stamp"
    [[ -s "$stale_backup" && -s "$stale_receipt" ]] || continue
    (( $(date +%s) - $(stat -c %Y "$stale_receipt") >= 86400 )) || continue
    [[ "$(docker inspect -f '{{.State.Running}}' "$stale")" == false ]] || continue
    stale_image="$(docker inspect -f '{{.Image}}' "$stale")"
    [[ "$stale_image" != "$OLD_IMAGE" ]] || continue
    tar -tzf "$stale_backup" >/dev/null || continue
    docker rm "$stale" >/dev/null
    if [[ "$(docker image inspect -f '{{.Id}}' "$stale_tag" 2>/dev/null || true)" == "$stale_image" ]]; then
        docker image rm "$stale_tag" >/dev/null
    fi
    say "Retired an older updater rollback: $stale"
done

CACHED="$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || true)"
if (( $(free_kb) >= 1500000 )); then
    docker pull "$IMAGE"
elif [[ -n "$CACHED" && "$CACHED" != "$OLD_IMAGE" ]]; then
    say "Only $(free_kb) KB free; using already downloaded candidate $CACHED."
    say 'The registry was not queried again. This candidate may not be the current remote latest.'
else
    die 'Need 1.5 GB free to pull a new image; nothing was stopped.'
fi
NEW_IMAGE="$(docker image inspect -f '{{.Id}}' "$IMAGE")"
if [[ "$NEW_IMAGE" == "$OLD_IMAGE" ]]; then
    say 'Already running the image tagged latest. No restart or backup was needed.'
    exit 0
fi

DATA_KB="$(du -sk "$DATA" | awk '{print $1}')"
(( $(free_kb) >= DATA_KB + 250000 )) || die 'Insufficient space for a consistent data snapshot and rollback.'
[[ "$(docker inspect -f '{{.Id}}' 9router)" == "$OLD_CONTAINER" ]] || die '9router changed during preflight.'
[[ "$(docker inspect -f '{{.State.StartedAt}}' 9router)" == "$OLD_STARTED" ]] || die '9router restarted during preflight.'
no_active_tarjomeh_work || die 'Tarjomeh became busy; nothing was stopped.'
docker image tag "$OLD_IMAGE" "$OLD_TAG"

say 'Do not start Tarjomeh jobs until the update finishes.'
say 'Stopping 9router briefly to snapshot its SQLite data.'
STOP_ATTEMPTED=1
docker stop -t 30 9router >/dev/null
tar -C /opt/translate -czf "$PARTIAL" 9router-data
tar -tzf "$PARTIAL" >/dev/null
[[ -s "$PARTIAL" ]] || die 'The data archive is empty.'
mv -- "$PARTIAL" "$BACKUP"
BACKUP_READY=1
(( $(free_kb) >= DATA_KB + 100000 )) || die 'Insufficient space to restore the snapshot if the new version fails.'

docker rename 9router "$OLD_NAME"
NEW_ATTEMPTED=1
docker run -d --name 9router --network bridge -p 20128:20128 \
    -e DATA_DIR=/app/data -e PORT=20128 -e HOSTNAME=0.0.0.0 \
    -v "$DATA:/app/data" "$NEW_IMAGE" >/dev/null

READY=0
for (( attempt=1; attempt<=30; attempt++ )); do
    if [[ "$(docker inspect -f '{{.State.Running}}' 9router 2>/dev/null || true)" == true ]] && health_ok; then
        READY=1
        break
    fi
    sleep 2
done
(( READY == 1 )) || die 'New 9router failed /api/health.'
sleep 10
[[ "$(docker inspect -f '{{.State.Running}}' 9router)" == true ]] || die 'New 9router exited after initial health check.'
health_ok || die 'New 9router failed its second health check.'
[[ "$(docker inspect -f '{{.Image}}' 9router)" == "$NEW_IMAGE" ]] || die 'New container uses an unexpected image.'
[[ "$(docker inspect -f '{{range .Mounts}}{{.Source}}:{{.Destination}}{{end}}' 9router)" == "$DATA:/app/data" ]] || die 'New container lost its data mount.'
docker exec 9router sh -c 'test -s /app/data/db/data.sqlite' || die 'New container cannot read its SQLite database.'
[[ "$(docker inspect -f '{{.Id}}' translate_tarjomeh_1)" == "$TARJOMEH_CONTAINER" ]] || die 'Tarjomeh container changed during the update.'
[[ "$(docker inspect -f '{{.Image}}' translate_tarjomeh_1)" == "$TARJOMEH_IMAGE" ]] || die 'Tarjomeh image changed during the update.'
[[ "$(docker inspect -f '{{.State.StartedAt}}' translate_tarjomeh_1)" == "$TARJOMEH_STARTED" ]] || die 'Tarjomeh restarted during the update.'
[[ "$(docker inspect -f '{{.State.Health.Status}}' translate_tarjomeh_1)" == healthy ]] || die 'Tarjomeh became unhealthy.'
docker exec translate_tarjomeh_1 python -c 'import json, urllib.request; response=json.load(urllib.request.urlopen("http://172.17.0.1:20128/api/health", timeout=10)); assert response.get("ok") is True' || die 'Tarjomeh cannot reach the new 9router.'
health_ok || die 'Final 9router health check failed.'

printf 'old_image=%s\nnew_image=%s\nbackup=%s\n' "$OLD_IMAGE" "$NEW_IMAGE" "$BACKUP" > "$RECEIPT"
COMMITTED=1
say "UPDATED: 9router is running $NEW_IMAGE"
say "Rollback container retained: $OLD_NAME"
say "Consistent data snapshot: $BACKUP"
say 'Check the 9router UI and a Tarjomeh model request before relying on the update.'
docker ps --format 'table {{.Names}}\t{{.Status}}'
df -h /
