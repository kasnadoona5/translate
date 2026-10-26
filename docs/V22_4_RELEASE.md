# v22.4: Explicit Reviewed Summary Revisions

This release adds an administrative, paused-job correction route for Layer 2.
It is not an automatic semantic reviewer or another translation pass. It does
not change existing DOCX wording, prompts, models, thresholds, call budgets,
term approvals, the other memory layers, style rules or export admission.
Additional LLM requests: zero. A new full translation is not required.

## Approval And Provenance

The operator first reviews the **complete bilingual candidate** against the
covered source, not just changed words. A proposal contains `job_id`,
`document_sha256`, `previous_summary`, `candidate_summary`, `supersedes` and
`evidence`. Summary objects retain the four serialized fields, including input
trust and reasons. Superseded identities are the JSON hashes returned by
`tarjomeh.memory.summary_revision.history_key`. Each evidence record contains
`chunk_index`, `source_sha256`, exact character `start`/`end`, `quote` and `claim`.
Only canonical, already covered chunks can support it. Quotes/hashes establish
location and provenance; they do not prove meaning or constitute user consent.
This operation requires the existing hash-bound canonical memory policy; it
does not migrate legacy jobs lacking canonical coverage.

Preparation binds the proposal to the complete current memory-state hash.
Check defaults to read-only and requires no provider. Missing proof, stale state,
non-paused jobs or any active worker lease refuse only this operation. They do
not create new export blockers or REVIEW penalties.

## Maintenance Procedure

Use `scripts/revise_tarjomeh_summary.sh` on the VPS. It holds the actual host
deployment/trial and 9router-updater locks, verifies health, exact tagged revision
and runtime probes, then invokes the container module. No router operation,
restart, Resume, cleanup or model request occurs. The module's
`--operator-locks-held` is a privileged operator assertion, not independent lock
verification; do not call the write module directly without those host locks.

Put the reviewed proposal in the jobs uploads bind. Run **prepare**, capture its
JSON to a new private manifest, then **check** that manifest:

```bash
umask 077
bash scripts/revise_tarjomeh_summary.sh prepare \
  /opt/translate/jobs/uploads/reviewed-proposal.json \
  > /opt/translate/jobs/uploads/reviewed-manifest.json
bash scripts/revise_tarjomeh_summary.sh check \
  /opt/translate/jobs/uploads/reviewed-manifest.json
```

Do not apply until the user has approved the exact candidate/manifest printed
by check. A stale manifest needs preparation and approval again. The explicit
write command accepts that JSON-content hash and a named reviewer. Keep stdout
as the private receipt, using a new filename, not the input manifest:

```bash
bash scripts/revise_tarjomeh_summary.sh apply \
  /opt/translate/jobs/uploads/reviewed-manifest.json \
  APPROVED_MANIFEST_SHA256 'Reviewer name' \
  > /opt/translate/jobs/uploads/reviewed-receipt.json
```

The writer exclusively creates a private SQLite backup in `/app/jobs`, checks
integrity and expected-state read-back, and rechecks ownership, source, coverage
and state inside `BEGIN IMMEDIATE`. Memory, receipt and event commit together.
No schema migration is performed. Partial backup failures retain the diagnostic
file and do not modify job memory. Successful repeats are idempotent.

Rollback likewise needs explicit approval of the receipt's `after_sha256` and
the same idle checks/backup. It refuses intervening work rather than overwriting
it. It appends a revocation; it does not erase the original generated summaries,
review or events:

```bash
bash scripts/revise_tarjomeh_summary.sh rollback \
  /opt/translate/jobs/uploads/reviewed-receipt.json \
  APPROVED_AFTER_STATE_SHA256 'Reviewer name'
```

## History, Audits And Limits

Restore and pre-chunk context selection respect exact superseded identities.
If a canonical dependency changes, only an eligible non-superseded earlier
summary can be reused; normal pending-coverage rules remain. Corrupt revision
metadata cannot authorize its candidate. Jobs without revisions retain their
previous behavior. Original history is never rewritten.

The versioned audits add active revision IDs, invalid-record indices and exact
identity counts as diagnostics only. Existing calls/failures/helpers, research,
glossary, style, canonical/render checks and router fingerprints remain.

Locally verified is not live validation or semantic certification. A reviewed
correction can repair a known stored instance; future model summaries can still
be wrong. Body editorial proposals must remain separate, noncanonical copies.
Do not import them as trusted memory. Keep a job paused until its continuation
and saved pause settings are separately agreed.
