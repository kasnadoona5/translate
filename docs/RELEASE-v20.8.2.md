# v20.8.2: Scoped Safeguards

Locally validated code is ready for a fresh checkpoint test, not proven
publication-quality Persian or whole-book production readiness.

## Changes and Limits

- Automatic compound-name mappings need matching explicit hyphen groups.
  Final-heh ezafe at a selected edge is also withheld. Correct but ambiguous
  contracted/spaced names can be deferred; translated text is not changed.
- Source-local annotation syntax no longer needs an approved reusable name.
  Recognition does not approve its Persian spelling or authorize insertion.
  Duplicates, unbalanced delimiters and unexplained wrappers remain findings.
- The existing summary prompt preserves subjects and relations more explicitly.
  This is prevention guidance, not a deterministic entailment test. It is a
  separate removable commit, fbbc861, requiring fresh source-based validation.
- No new stage, mandatory call, model, threshold or budget. All four layers,
  style evidence, refiner veto, research and approval/checkpoint workflow remain.
  No additional paid trial or experimental reviewer is enabled.

## Deploy on the VPS

Wait for jobs and worker leases to finish or pause. Do not start jobs during
deployment. Run this complete block in the VPS Bash terminal:

```bash
(
set -Eeuo pipefail
cd /opt/translate
git fetch origin main --tags
SCRIPT="$(mktemp /root/deploy_tarjomeh_v2082.XXXXXX.sh)"
git show v20.8.2:scripts/deploy_tarjomeh_v2082.sh > "$SCRIPT"
bash -n "$SCRIPT"
chmod 700 "$SCRIPT"
LOG="/root/deploy_tarjomeh_v2082_$(date +%Y%m%d-%H%M%S).log"
: > "$LOG"
nohup bash "$SCRIPT" > "$LOG" 2>&1 < /dev/null &
echo "Deployment PID: $!"
echo "Keep this complete log: $LOG"
tail --retry -F "$LOG"
)
```

Ctrl-C stops following the log, not the background deployment. Success requires
the final completion message, not just a running container or a release label.
The script verifies revision/image labels, health, all pure probes and installed
behavior; makes and restore-checks a WAL-aware SQLite backup; preserves the local
glossary; frees only verified unused Tarjomeh resources; and compares 9router ID,
image, start time and mounts. If safe disk space is insufficient it stops. Never
delete 9router, jobs/uploads or active images to bypass that stop. Previous
Tarjomeh image is retained for rollback; data backups are not automatically
reimported over newer job data. A UI health check is not a real model test.

## Audit After a Fresh Checkpoint

Keep the prior models and checkpoint selection. New terminology approvals need a
fresh opt-in job, reviewed source occurrences and explicit keep-original choices.
Do not enable pause_each unless that is your separate workflow choice.

```bash
cd /opt/translate || exit 1
bash scripts/run_tarjomeh_v2082_audits.sh YOUR_NEW_JOB_ID
```

This collects QA plus both JSON/TXT audit sets, the server-wide companion,
active/lifetime calls, retries/schema failures, served models, memory/style,
source/canonical/DOCX identity, research and new safeguard evidence. It calls no
LLM and does not change the job. A hard failure returns nonzero, not PASS.
Send the six audit artifacts, the DOCX, and the full deployment log. Human
source-vs-DOCX and summary-snapshot review remains necessary. Unsupported subject
transfer or invented opposition fails the summary-guidance experiment; missing
evidence is UNKNOWN. Do not repair canonical memory by importing an edited DOCX.
