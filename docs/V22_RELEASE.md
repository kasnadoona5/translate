# v22: deterministic safeguards and evidence

## Scope

- Skip empty source/translation pairs before memory mutation. Preserve the last
  valid pair and its trust, role and coverage. Normal valid updates are unchanged.
- Let DOCX heading runs inherit heading sizes, including complex-script sizes;
  body text remains 11 pt. No translated words or canonical hashes are changed.
- Use the existing conservative number reconciler in offline evaluation for
  localized ordinals. Production numeric enforcement is unchanged.
- Report model overall, dimension mean, score of record, ignored invalid severe
  findings and grounding gaps without changing scoring or routing.
- Retain already computed coherent/local term admissions, their raw hashes and
  nested literal evidence. Scope remains source-location-specific. Suggestions
  elsewhere remain nonbinding and cannot create memory penalties.
- Distinguish physical LLM attempt records from cached/re-logged critique records
  and identify unavailable historical diagnostics honestly.

No prompt, model, threshold, call budget, memory priority, style eligibility floor,
source-structure gate or canonical-to-DOCX identity gate is relaxed. All four
memory layers and existing features remain. There is no book-specific production
rule and no additional LLM call.

## Deployment

Use `scripts/deploy_tarjomeh_v22.sh` from the immutable tag, after checking its
peeled revision against the published release revision. The script refuses active
jobs/worker leases and concurrent deployment/router updates. It keeps consistent
SQLite backup/integrity/restore checks, current-image rollback, installed revision,
health, capability and behavior checks, and guarded Tarjomeh-only image cleanup.
It does not prune Docker globally or change 9router resources.

Router evidence compares ID, image, start time and all mount fields, with stable
ordering of mount records/JSON keys. Reordering alone is not a runtime change;
real field changes still stop deployment. Before/after evidence is logged and
saved in the deployment backup directory. The previous trial warning's exact
cause remains unproven because its before/after fingerprints were not retained.

## Validation And Limits

Run `scripts/run_tarjomeh_v22_audits.sh EXACT_JOB_ID` after deployment. This collects
both existing full audit sets plus the new evidence; it does not resume a job or
send provider requests. Historical jobs may lack the newly added diagnostics.
Keep the deploy log, audit log, both JSON/TXT sets, QA report and checkpoint DOCX.

Do not resume/retranslate completed work merely to gather these reports. A fresh
matching checkpoint job is needed to assess newly produced diagnostics and
translation quality; do not interrupt an active job.

The offline summary experiment is not included. Its candidate failed fidelity
checks and increased completion tokens and recorded LLM time. Production summary
instructions remain unchanged, and summary semantic accuracy remains an open
monitoring item. No paid trial rerun or broad critic/refiner experiment is needed
for this release. Runtime probes and offline export checks are not independent
certification of accurate, fluent academic Persian. Live validation remains
pending until deployment and source-vs-DOCX evidence are reviewed.
