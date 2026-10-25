# v22.1: Bounded Chapter Evidence

This release fixes two older structure-detector gaps. It changes no translation
text, prompts, models, thresholds, pipeline ordering, memory authority or call
allowance. It adds no unconditional LLM request.

- Explicit joined Persian nine/ten-chapter adjectives provide exact typed
  count evidence. Bare ambiguous words remain outside the cardinal dictionary.
- Chapter references recognize ASCII, Persian and Arabic-Indic digits through
  the existing digit convention. Text is not renumbered or normalized.
- Genuine structure mismatches, refiner veto, unchanged-issue attribution,
  atomic memory/checkpoints and canonical-to-DOCX identity remain enforced.
- Existing v22 memory, heading, numeric-evaluation and diagnostic protections
  remain present. New pure probes verify both corrections and negative cases.

## Deployment

Use `scripts/deploy_tarjomeh_v221.sh` from immutable tag `v22.1`, after verifying
the tag's commit and the downloaded script. The script requires idle jobs and
leases, takes both deployment/updater locks and preserves the local glossary.
It performs guarded Tarjomeh-only cleanup, a WAL-consistent SQLite snapshot,
isolated compressed-backup restore/integrity verification, rollback tagging,
exact installed revision and health/capability/probe checks. It records and
compares canonical before/after 9router metadata without changing 9router.

Insufficient disk space stops deployment. Do not remove router data, use broad
Docker pruning, delete the only backup or bypass the reserve to force a build.
Historical release scripts and tags are not changed.

## Installed Version And Audits

From `/opt/translate`, compare the container's
`org.opencontainers.image.revision` label with `git rev-parse 'v22.1^{commit}'`.
Verify running/healthy state, runtime release `v22.1`, both new capability/probe
IDs and all other runtime probes. The new IDs are:

- `compound_chapter_count_evidence`
- `arabic_indic_chapter_references`

Run the combined read-only collectors with an explicit job ID:

```bash
bash scripts/run_tarjomeh_v221_audits.sh JOB_ID
```

Both collectors run even if one fails; a hard failure retains a nonzero exit
status. Upload artifacts use `_v221_memory_audit` and `_v221_companion_audit`.
Include the QA report, downloaded checkpoint DOCX, combined audit log and full
deployment log. The root console copy is `tarjomeh_v221_companion_audit.txt`.
Auditing a saved job does not resume it or make paid provider requests.

## Validation Limits

These are detector corrections, not a semantic quality certificate. The v22
checkpoint already exported; no fresh whole-book run is required for diagnosis.
Do not automatically resume or retranslate completed chunks. Validate the next
relevant checkpoint under the normal user-selected workflow.

All four memories, complete qualified style guidance and editable
Required-here/suggested-elsewhere terminology remain. Optional suggestions do
not gain enforcement or memory penalties. The failed summary experiment is not
adopted; summary subject transfer and source-linked editorial issues stay open.
New trigger behavior is locally verified until actual live evidence is supplied.
