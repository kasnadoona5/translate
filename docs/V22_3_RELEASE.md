# v22.3: Cumulative Summary Provenance And Offline Numeric Evidence

## Scope

- Cumulative Layer-2 summaries retain the active previous summary's advisory
  input provenance when a later input batch is reviewed. Reasons are bounded
  and deduplicated. Empty prior context does not taint new reviewed input;
  canonical dependency rollback uses only the restored active summary.
- Offline grouped-integer equivalence requires positive target quantity context.
  An ambiguous account/reference label cannot consume a source quantity merely
  because it is not a recognized identifier. Literal and production numeric
  checks are unchanged. Unsupported equivalences remain unresolved.
- A pure cumulative-summary capability/probe and versioned collectors expose
  effective trust and available lineage evidence. Missing historical lineage is
  reported as unavailable, never reconstructed or retroactively approved.

No additional LLM request, model, prompt template, threshold, retry or call-budget
change. Affected dynamic summary trust labels intentionally remain advisory.
No canonical translation, research scope, optional-term penalty, style threshold,
memory authority, export proof or pipeline-order change. No new export blocker.

## Validation And Limits

Required verification includes direct/restored mixed-provenance regressions,
empty/all-reviewed/legacy/rollback controls, bounded reasons, atomic failures and
cancellation, offline numeric negatives, old runtime contracts, historical export
replays and all three DOCX layouts with tamper rejection. The release is locally
verified; deployed behavior requires the matching deployment and job audit logs.

Input provenance is not semantic verification. Known summary subject/scope errors
(R16) are not corrected by this patch. The failed 32-request summary experiment
is not adopted; no further paid trial or automatic historical-memory rewrite.
Do not resume a paused job automatically. Saved pause settings apply to Resume,
not the settings displayed for a new upload.

## Operations

Deploy: `scripts/deploy_tarjomeh_v223.sh` from the immutable tag.
Audit: `bash scripts/run_tarjomeh_v223_audits.sh EXACT_JOB_ID`.
Existing versioned scripts/tags remain untouched. Consistent SQLite backup and
restore checks, rollback, exact image revision, health/runtime probes, guarded
Tarjomeh-only cleanup and before/after 9router identity checks are retained.
Insufficient disk space must stop deployment; never prune unrelated resources
or delete job data to bypass the guard. Private audit records are not shipped.
