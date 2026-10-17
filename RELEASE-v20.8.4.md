# Tarjomeh v20.8.4

Scoped correctness release. English remains authoritative. Academic fluency may
not omit, simplify, reinterpret, weaken, or alter source claims. Local software
tests are not a certification of Persian quality or production readiness.

## Changes

- Resume/export/replay retain saved model, reasoning and quality budgets. Current
  credentials, endpoints and transport settings may still be supplied safely.
- Keep entity corrections separate from lexical plural checks. Low-authority
  sightings cannot reclassify stronger mappings or erase recorded conflicts.
- Only accepted refiner decisions participate in salvage overlap/deduplication.
  Compare the same safely normalized response/spans; permit evidenced surface
  fixes without weakening sentence, predicate, citation or integrity protection.
- Share raw-paragraph/source-scope attribution across rollback and final repairs.
  Count exclusions never resolve old findings or grant memory/style authority.
  Veto exclusions require matching findings and unchanged proven scope.
- Bind canonical critique quotes through the actual complete paragraph diff.
  Preserve historical records and suggestions; ambiguous quotes stay unresolved.
- Retain individually validated readability advice in partially invalid responses.
  Stronger objective advice may route its existing grounded issue, without a
  duplicate instruction or a fabricated critic severity/confidence.
- Protect only approved paragraph-term matches during candidate admission.
  Show approved sense context; bind new proposals to attested selected-body
  surfaces and full evidence paragraphs. Missing/colliding evidence blocks approval.
- Initial automatic terms need boundary evidence in the frozen extraction sample.
  They remain proposals, not observed renderings or mandatory terminology.
- Parse exact summary section headers without truncating level-three subsections.
  This is not a fix for semantic summary drift, which remains open.
- Add runtime behavior probes and audit fields for both final-repair count paths,
  refiner veto exclusions and canonical quote evidence.

Stage order, four memory layers, style floors, refiner veto, model settings and
attempt budgets remain. No new unconditional LLM stage, focused reviewer, prompt
cap, book-specific production rule or router change. Existing conditional calls
may run differently after corrected routing/admission; actual physical calls,
retries, failures and cost must be measured, not assumed constant.

Deferred: keep-original policy, blanket glossary mandates, Skip semantics,
curated revision/deletion synchronization, retranslation memory redesign,
mixed-role style authority and rejected-summary coverage carry-forward.

## Deploy

Run only while jobs, worker leases and trials are idle. Do not start jobs during
deployment. The script verifies a WAL-consistent SQLite backup and restore,
preserves rollback, frees only verified unused Tarjomeh space, and stops when
safe space is insufficient. No global Docker prune or 9router cleanup/restart.
Installed revision, image, health, capabilities and behavior probes must pass;
9router container ID/image/start/mount fingerprints must remain identical.

```bash
(
set -Eeuo pipefail
umask 077
cd /opt/translate
git fetch origin main --tags
SCRIPT="$(mktemp /root/deploy-tarjomeh-v2084.XXXXXX.sh)"
git show v20.8.4:scripts/deploy_tarjomeh_v2084.sh > "$SCRIPT"
bash -n "$SCRIPT"
LOG="/root/deploy_tarjomeh_v2084_$(date +%Y%m%d-%H%M%S).log"
nohup bash "$SCRIPT" > "$LOG" 2>&1 < /dev/null &
echo "Deployment PID: $!  Full log: $LOG"
tail --retry -F "$LOG"
)
```

Ctrl+C only stops log following. Keep the entire deployment log. A successful
build alone does not prove that the new container is installed and healthy.
The command requires the published immutable tag; local code alone is insufficient.

## Fresh Checkpoint and Audit

Use a fresh matching-source job and the same declared comparison settings.
Choose term review/chapter pauses yourself. Make term decisions before Start;
optional elsewhere recommendations are not approved meanings or repair mandates.
After a downloadable checkpoint or completion, run:

```bash
(
set -Eeuo pipefail
cd /opt/translate
JOB='REPLACE_WITH_NEW_JOB_ID'
bash scripts/run_tarjomeh_v2084_audits.sh "$JOB" 2>&1 |
  tee "/root/${JOB}_v2084_audit.log"
)
```

Send the complete deploy log, DOCX, QA report, companion JSON/text and memory
JSON/text reports. Collectors cover active/lifetime calls, retries, served models,
failures, canonical/export/checkpoint identity, all four memories/style, research
and approved/advisory term scopes. They cannot certify arbitrary meaning or
academic fluency; a real source-versus-DOCX review remains necessary.

The private ledger, tracker and run journal are excluded from release commits.
No paid model trial, live deployment or 9router operation is part of local gates.
