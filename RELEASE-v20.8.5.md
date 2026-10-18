# Tarjomeh v20.8.5

Source-first evidence and approved fresh-job policy corrections. Local software
verification is not proof of Persian accuracy, academic fluency or production
readiness. The source remains authoritative, including counts, scope, negation,
modality, citations and every claim. No simplifying or reinterpreting for fluency.

## Scope

- Keep validated readability alternatives without arrival-order routing changes.
  Preserve original critic severity/confidence, source checks and refiner veto.
- New term reviews default to **Required here, suggested elsewhere**, editable
  per term. Merge identical source/evidence/target/meaning proposals and preserve
  their origins; actual disagreements still require an explicit decision.
- New reviews keep approved Persian plus the first proven paragraph-scoped
  foreign original. Skip skips pending proposals, not earlier approvals. Neither
  an italic span nor a pending/rejected/skipped term gains prompt authority.
- Curated rows can be explicitly approved in the existing opt-in review. Rejecting
  a proposal does not revoke independent existing curated glossary matching.
- New-policy jobs reconcile same-meaning curated revisions on resume. Removed or
  ambiguous mappings remain stored but deferred. A saved-approval conflict stops
  before further calls; restore the glossary or create a new reviewed job.
- New-policy manual retranslation uses only hash-proven pre-chunk derived context
  plus current valid human/source evidence. A disposable copy of this job, not
  the entire jobs database, isolates the attempt. Publish admitted canonical text,
  QA and memory together; retain physical attempt costs after rejection/failure.
- Preserve historical retrieval/style records while replacing the active chunk
  version and rebuilding Layer 4 in source order. Quarantine stale Layer-2
  dependencies; hash-bound pending coverage reaches the next scheduled update.
- Admit complete body style pairs from mixed heading/notes chunks only with the
  actual body role and all existing alignment/density/dimension/genre safeguards.
- Report inconsistent raw provider counters without rewriting them into billing.
  Report real ignored environment differences by name, never credentials/values.

New review/memory policies apply to **fresh jobs only**. Saved older choices and
records are not migrated automatically. All four memory layers, existing stages,
style floors, models, thresholds, attempt budgets, chapter checkpoints and export
remain. No new unconditional LLM call, prompt cap, book-specific production rule,
paid trial repeat or experimental reviewer activation. Conditional calls and
prompt sizes can vary. Summary semantic drift remains open; table reconstruction
and prose-year changes remain deferred. A manually edited DOCX stays editorial
and must not be imported as canonical text or trusted memory.

## Deploy

Wait until all jobs, worker leases and trials are idle. Do not start jobs during
deployment. This procedure uses a consistent SQLite backup with integrity/restore
checks; removes only verified unused Tarjomeh space; retains rollback; checks the
installed Git revision, image, runtime capabilities, behavior probes and health;
and compares 9router container ID, image, start time and mounts before/after.
If space is insufficient, it stops. No global Docker prune or router cleanup.

After the immutable tag is published, run this on the VPS:

```bash
(
set -Eeuo pipefail
umask 077
cd /opt/translate
git fetch origin main --tags
SCRIPT="$(mktemp /root/deploy-tarjomeh-v2085.XXXXXX.sh)"
git show v20.8.5:scripts/deploy_tarjomeh_v2085.sh > "$SCRIPT"
bash -n "$SCRIPT"
LOG="/root/deploy_tarjomeh_v2085_$(date +%Y%m%d-%H%M%S).log"
nohup bash "$SCRIPT" > "$LOG" 2>&1 < /dev/null &
echo "Deployment PID: $!  Full log: $LOG"
tail --retry -F "$LOG"
)
```

Ctrl+C stops log following, not deployment. Keep the entire log. A successful
image build alone does not prove the new runtime is installed.

## Fresh Run And Audit

Use a fresh matching-source checkpoint job with declared comparison settings.
Review terms before Start when opted in. Choose chapter pauses yourself; the
review checkbox remains off by default. Approve terminology by source meaning
and academic Persian, not frequency alone. Coordinated families stay editable
whole-phrase proposals, never book-specific built-in replacements.

After the downloadable checkpoint/completion, run:

```bash
(
set -Eeuo pipefail
cd /opt/translate
JOB='REPLACE_WITH_NEW_JOB_ID'
bash scripts/run_tarjomeh_v2085_audits.sh "$JOB" 2>&1 |
  tee "/root/${JOB}_v2085_audit.log"
)
```

Send the full deploy log, audit log, DOCX, QA report, companion JSON/text and
memory JSON/text. Reports include active/lifetime attempts, retries, served
models/failures, raw counter inconsistencies, optional-term non-use locations,
memory/style/summary coverage, historical versus active versions, research,
canonical/export identity and checkpoints. A not-exercised policy is not a live
PASS. These reports cannot certify arbitrary meaning; compare source and DOCX.

Private ledger, tracker and run journal are not release files. No live VPS,
9router operation or paid model call is part of local verification.
