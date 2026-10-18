# Tarjomeh v20.8.6

## Change And Limits

R190: catalogue names/titles and publishing notices cannot teach style merely
because the parser labels them body prose. One source eligibility decision covers admission,
restored records, representative/fallback prompting, effective sample capacity
and reporting. Records retain their original text, scores and history; exclusion
is not a translation error, glossary demotion or Layer-3/4 penalty.

Clean prose in mixed chunks, analytic prefaces and explicit dialogue remain
eligible under the existing alignment, score, density and representative floors.
There is no new minimum paragraph length. Legacy records without source proof
remain stored but cannot supply a profile. A legacy source-unproven set may
remain warming_up with zero effective samples; it cannot use a cached fallback
profile. The legacy runtime probe checks retained history and absent prompting.

No new LLM calls, prompt template, model, threshold, attempt allowance, stage,
glossary authority or memory-layer change. Existing conditional call counts can
vary. The experimental reviewer stays OFF. No global term replacements,
scholarly-table reconstruction, prose-year conversion or 9router operation.

Saved v20.8.5 replay excludes exactly its catalogue record, retains the other
three complete pairs and leaves source files unchanged. This is local evidence,
not a live v20.8.6 run or semantic certification of the remaining samples.
Aspect, modality, attribution, attachment and terminology decisions still need
source-aware editorial review. Do not import an edited DOCX as canonical memory.

Book-term review keeps the editable Required here, suggested elsewhere default.
Elsewhere advice remains nonblocking for translator/critic/refiner. Exact-match
consistency reports are not plural-family or semantic verification. A blank
Persian option can be edited manually; no extra extraction call is purchased.

## Safe Deployment

After publication, wait until all jobs, worker leases and trials are idle.
Do not start jobs during deployment. The script verifies host mounts, makes a
consistent SQLite/WAL backup, tests its compressed restore, preserves config
and glossary, retains rollback, verifies installed Git/image/runtime/health and
compares 9router ID, image, start time and mounts. Cleanup is limited to verified
unused Tarjomeh images. No global prune; insufficient safe space means STOP.

Run on the VPS:

```bash
(
set -Eeuo pipefail
umask 077
cd /opt/translate
git fetch origin main --tags
SCRIPT="$(mktemp /root/deploy-tarjomeh-v2086.XXXXXX.sh)"
git show v20.8.6:scripts/deploy_tarjomeh_v2086.sh > "$SCRIPT"
bash -n "$SCRIPT"
LOG="/root/deploy_tarjomeh_v2086_$(date +%Y%m%d-%H%M%S).log"
nohup bash "$SCRIPT" > "$LOG" 2>&1 < /dev/null &
echo "Deployment PID: $!  Full log: $LOG"
tail --retry -F "$LOG"
)
```

Ctrl+C stops log following, not deployment. Keep the entire log. The completion
message must include RUNNING_V2086_CONFIRMED and the unchanged-router check.
No live deployment is performed by local verification or publication.

## Fresh Checkpoint And Full Audit

Keep the existing job as comparison evidence. Use a fresh matching-source
checkpoint with declared settings to test the corrected style-learning sequence.
Term review and chapter pauses remain user choices, not newly forced settings.

After it pauses with a downloadable DOCX, run on the VPS:

```bash
(
set -Eeuo pipefail
cd /opt/translate
read -r -p 'New job ID: ' JOB
[[ "$JOB" =~ ^[a-f0-9]{12}$ ]] || { echo 'STOPPED: invalid job ID.'; exit 1; }
bash scripts/run_tarjomeh_v2086_audits.sh "$JOB" 2>&1 |
  tee "/root/${JOB}_v2086_audit_$(date +%Y%m%d-%H%M%S).log"
)
```

Send the full deploy log, combined audit log, translated DOCX, QA report,
memory JSON/TXT, job companion JSON/TXT and root companion TXT. Reports include
style-source reasons/effective counts, all four memories, summary coverage,
research, term scopes, call/retry/helper/failure/model accounting, raw counter
inconsistencies, canonical/export identity and checkpoint evidence. Historical
prompt exposure is not erased by current exclusion. Not-exercised is not PASS;
mechanical eligibility and high scores do not certify academic Persian quality.

Private ledger/tracker/run journal/editorial sheets are not release files.
