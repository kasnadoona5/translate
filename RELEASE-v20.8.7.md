# Tarjomeh v20.8.7

## Changes And Limits

- Original insertion and render proof share contiguous lexical spans. The bounded
  fallback accepts internal indefinite yeh, not skipped words, reordered targets,
  crossed punctuation or ambiguous inflected occurrences. The Persian is preserved;
  authorization, deterministic replay, occurrence limits and DOCX read-back remain.
- Automatic organization mappings with unproven words before their semantic head
  cannot enter reusable context. Historical raw records are retained and reported,
  not silently shortened or deleted. Brand/legal suffixes and explicit approvals
  retain their existing treatment. Valid source originals already in text survive.
- Uploaded-title-adjacent ambiguous dates are removed from usable research context,
  with exact claim/source evidence retained. Cached research is rechecked without
  a new query or LLM call. Other works, historical dates and citations are protected.
- Audits expose render-span proof, deferred organization reasons, cached-context
  projection, and the fact that summary protocol/coverage validation is not semantic
  entailment. Existing full call/retry/helper/failure/model reports remain available.

No translator/critic/refiner/summary prompt, model, threshold, attempt allowance,
unconditional LLM stage, memory layer, style floor or term choice is changed.
Required here, suggested elsewhere stays the editable default. The previous style
source fix remains. Refiner veto, source-first admission and canonical checkpoints
remain mandatory. No experimental reviewer, paid trial, table reconstruction,
prose-year conversion, book-specific replacements or 9router operation is added.

The export correction can recover already finished chunks without regenerating
them. The existing pending-checkpoint path retries export before translating a
later chapter. A local saved-output replay is not a live recovery or a guarantee
of semantic accuracy: remaining aspect, attribution, attachment, summary drift and
terminology decisions still require source-based review.

## Safe Deployment

Run only when all workers/trials are idle; do not start a job during deployment.
The script verifies the host jobs bind, creates a consistent SQLite/WAL snapshot,
restores and checks it in isolation, preserves the glossary/configuration, retains
rollback, verifies installed Git/image/runtime/health and compares the running
9router ID/image/start time/mounts before and after. Only verified unused Tarjomeh
resources are eligible for cleanup. There is no global prune; insufficient safe
space means STOP, not deletion of router resources or job data.

```bash
(
set -Eeuo pipefail
umask 077
cd /opt/translate
git fetch origin main --tags
SCRIPT="$(mktemp /root/deploy-tarjomeh-v2087.XXXXXX.sh)"
git show v20.8.7:scripts/deploy_tarjomeh_v2087.sh > "$SCRIPT"
bash -n "$SCRIPT"
LOG="/root/deploy_tarjomeh_v2087_$(date +%Y%m%d-%H%M%S).log"
nohup bash "$SCRIPT" > "$LOG" 2>&1 < /dev/null &
echo "Deployment PID: $!  Full log: $LOG"
tail --retry -F "$LOG"
)
```

Ctrl+C stops log following, not deployment. Require RUNNING_V2087_CONFIRMED and
the successful unchanged-router check; retain the whole log.

## Verify And Recover

```bash
(
set -Eeuo pipefail
cd /opt/translate
C=translate_tarjomeh_1
EXPECTED="$(git rev-parse 'v20.8.7^{commit}')"
test "$(docker inspect -f '{{.State.Running}}' "$C")" = true
test "$(docker inspect -f '{{.State.Health.Status}}' "$C")" = healthy
test "$(docker inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$C")" = "$EXPECTED"
docker exec "$C" python -c '
from tarjomeh.runtime import runtime_capabilities, runtime_behavior_probes
m, p = runtime_capabilities(), runtime_behavior_probes()
assert m["release"] == "v20.8.7"
for name in ("inflected_multiword_anchor_proof", "organization_lexical_boundary", "book_identity_date_guard"):
    assert m["capabilities"].get(name) and p.get(name), name
assert p and all(p.values()), [k for k, v in p.items() if not v]
print("PASS: v20.8.7, exact revision, and", len(p), "runtime probes.")
'
)
```

For the saved export-blocked job, use Resume once in the existing UI. Do not use
whole-chunk retranslate, disable integrity, edit database text or start a new job
to work around this error. It should retry the pending checkpoint and pause with
a downloadable DOCX; stop and send the evidence if export still fails. A successful
recovery does not retrospectively change earlier prompt exposure or raw memory.

After recovery, run the combined collectors, entering the actual ID at the prompt:

```bash
(
set -Eeuo pipefail
umask 077
cd /opt/translate
read -r -p 'Job ID: ' JOB
[[ "$JOB" =~ ^[a-f0-9]{12}$ ]] || { echo 'STOPPED: invalid job ID.'; exit 1; }
bash scripts/run_tarjomeh_v2087_audits.sh "$JOB" 2>&1 |
  tee "/root/${JOB}_v2087_audit_$(date +%Y%m%d-%H%M%S).log"
)
```

Send the complete deploy and collector logs, actual DOCX, QA report, both job
JSON/TXT sets and root companion TXT. Nonzero audit verdicts are evidence, not
permission to suppress checks. The audit cannot certify every feature or every
semantic claim from one checkpoint. A subsequent declared-settings run is needed
to observe future memory/research behavior. No new paid experiment is implied.

Local ledger/tracker/run journal, editorial copies and saved-job replays are not
release files and must not be staged or pushed.
