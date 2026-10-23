# v21: Atomic Summary Commit

This release fixes a reproduced failure-path inconsistency in Layer-2 memory.
Previously, a candidate summary could become active before terminology
reconciliation and coverage/history preparation completed. A later exception
reported failure while leaving the new summary paired with old records.

The candidate and its dependent records are now prepared separately and committed
together. Failed preparation retains the previous summary, trust, coverage,
pending inputs and history. Cancellation before commit also leaves them intact.
Failure events explicitly report candidate_committed=false, the failure stage
and failed_retained_previous_summary status. Audits show these diagnostics
separately from invalid-candidate rejection; they add no new publication gate.

The defect was reproduced with controlled failure injection, not observed in the
supplied fresh v20.8.12 checkpoint. This fix does not correct a well-formed but
semantically inaccurate LLM summary and does not claim improved translation
fluency. Source-based editorial review and summary semantic evaluation remain
necessary. Source-linked summary redesign is not installed or enabled.

Successful summary prompts, reconciliation rules, input coverage policy,
translation stages, all four memory layers, style safeguards, research authority,
models, thresholds, call budgets and canonical-to-DOCX proof remain unchanged.
There are no new LLM calls or automatic retries and no production DB migration.

Deployment uses scripts/deploy_tarjomeh_v21.sh. It requires an idle worker state,
checks a consistent SQLite backup and restore, retains rollback, verifies exact
installed revision and runtime probes, and performs only guarded Tarjomeh cleanup.
It never prunes shared Docker resources or modifies 9router. Insufficient safe
space stops deployment rather than deleting protected data.

Collect both audit sets with scripts/run_tarjomeh_v21_audits.sh and the exact job
ID. Send its DOCX, QA, both JSON/TXT audit sets, audit log and full deploy log.
Do not resume a deliberately paused checkpoint merely to test installation.
