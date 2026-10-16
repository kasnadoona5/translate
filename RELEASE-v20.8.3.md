# Tarjomeh v20.8.3

This is a scoped engineering release, not a certification of Persian translation
quality. English remains authoritative; academic fluency must not omit, simplify,
reinterpret, weaken, or change source claims. A fresh source-versus-DOCX audit is
still required. The known running-summary semantic limitation remains open.

## Changes

- Defer automatic lexical mappings containing an unrelated leading display label.
  Preserve accepted translation text, historical mappings and all continuity;
  exclude unsafe reusable mappings from prompts and inline-original insertion.
- Report added and deferred terminology reconciliation outcomes and their exact
  source/candidate identities. Keep person-name and lexical deferral reasons distinct.
- Reconcile publisher-before-date research metadata with uploaded-edition evidence.
  Do not substitute guessed dates or modify the translated copyright page.
- Withdraw only the failed additional summary instructions in a separate commit.
  Keep the summary operation, existing validation and all four memory layers.
- Add a per-term **Required here, suggested elsewhere** scope in fresh opt-in term
  reviews. Existing defaults and individual occurrence approval remain unchanged.
  Translator, critic and refiner see optional recommendations through existing calls.
  Preference-only observations and lexical non-use are report-only, not MQM issues,
  score deductions, REVIEW reasons, trust/style penalties or repair triggers.
  Genuine source-meaning errors keep the existing checks and refiner veto.
- Add final canonical, hash-bound QA consistency locations. Lexical presence is
  not semantic verification; unproved alignment is reported as unknown.
- Correct stale collector assertions by testing the shared entity behavior, not
  its file location. Distinguish physical readability calls from ordinary result
  batches and existing post-rollback logical attempts. Preserve schema failures,
  retries, active/lifetime accounting and real hard-failure exit codes.

No new LLM stage or attempt budget, focused reviewer, model/threshold change,
hard character cap, style-floor reduction, memory-layer removal or 9router change.
Prompt sizes and stochastic physical-call totals can vary; these are not promises
of fewer calls or better Persian. Keep-original decisions remain exact-source and
paragraph scoped; broader original suggestions are withheld when insertion policy
is not proven. Old saved scopes remain intact; new all-body mandates are refused.

## Deploy

Run on the VPS when no translation job, worker lease, trial or deployment is active.
Do not start a job until deployment finishes. The script preserves rollback,
backs up SQLite consistently including WAL, verifies the jobs bind, frees only
verified unused Tarjomeh image space, and verifies image/revision/health and pure
runtime probes. It stops if safe space is insufficient. It does not prune Docker
globally or remove/restart any 9router resource; before/after fingerprints must match.

```bash
(
set -Eeuo pipefail
umask 077
cd /opt/translate
git fetch origin main --tags
SCRIPT="$(mktemp /root/deploy-tarjomeh-v2083.XXXXXX.sh)"
git show v20.8.3:scripts/deploy_tarjomeh_v2083.sh > "$SCRIPT"
bash -n "$SCRIPT"
LOG="/root/deploy_tarjomeh_v2083_$(date +%Y%m%d-%H%M%S).log"
nohup bash "$SCRIPT" > "$LOG" 2>&1 < /dev/null &
echo "Deployment PID: $!  Full log: $LOG"
tail --retry -F "$LOG"
)
```

Ctrl+C stops following the log, not the detached deployment. Success requires
the final completion message, exact installed revision/release verification,
healthy Tarjomeh and the unchanged-router check, not just a successful image build.
No local test can certify the VPS state before this command is run there.

## Fresh Checkpoint and Audit

Use the same source and declared comparison settings. For the new scope, create
a fresh job with **Review book terms before translating** checked. Review/edit
terms and choose their scope before Start. Do not interpret optional elsewhere
locations as approved meanings. Chapter-pausing and research settings remain yours.

After a downloadable checkpoint or completion, replace the placeholder below
with that job's exact ID. Do not audit a still-running job as a final checkpoint.

```bash
(
set -Eeuo pipefail
cd /opt/translate
JOB='REPLACE_WITH_NEW_JOB_ID'
bash scripts/run_tarjomeh_v2083_audits.sh "$JOB" 2>&1 |
  tee "/root/${JOB}_v2083_audit.log"
)
```

Download the QA report and DOCX plus both companion/memory JSON and text reports.
Include the complete deployment log, not only its final lines. The collectors
cover runtime, stage/call/model failures, checkpoint/canonical/export identity,
all four memories/style, research evidence and term scopes/optional non-use.
They cannot certify arbitrary meaning or academic fluency: compare the real source
and DOCX, and retain every unresolved issue in the private regression records.
