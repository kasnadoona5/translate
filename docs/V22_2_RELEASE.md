# v22.2: Exact Quote Spans and Offline Numeric Evidence

- A unique contiguous quote may span adjacent source segments in one paragraph.
  Existing sentence boundaries/IDs and single-segment behavior are unchanged.
  New critique records retain source hash, offsets and contributing segment IDs.
  Location evidence does not certify meaning, approve an edit or grant coverage.
- Offline evaluation recognizes conservative grouped/ungrouped prose integers
  with one-use occurrence accounting. Identifiers, structural/citation numbers,
  decimals, ranges, malformed grouping and ambiguous labels remain protected.
  Equivalence requires an explicit English count construction (for example,
  "There are 1,000 records"); unproven cases stay unresolved.
  Production integrity rules and delivered text are unchanged.
- Production prompts, models, thresholds, retry/extra-final budgets, four memory
  layers, style floors and editable scoped terminology remain unchanged. There
  is no new unconditional LLM call. Better grounded findings may use existing
  bounded critique/refinement paths; total calls are not fixed.

## Deployment and Audits

Use the pinned immutable `v22.2` deployment script and exact revision checks.
`scripts/deploy_tarjomeh_v222.sh` preserves idle/worker checks, both locks,
SQLite backup/isolated restore, local glossary, rollback, exact Docker revision,
health/probes and canonical before/after router fingerprints. Cleanup remains
guarded and Tarjomeh-only; never bypass disk reserves or prune Docker globally.
The router updater and live 9router resources are outside this release's scope.

New pure capability/probe IDs: `cross_segment_quote_grounding` and
`offline_grouped_integer_evaluation`. Both include negative controls.

Run both read-only collectors without resuming or retranslating a job:

```bash
bash scripts/run_tarjomeh_v222_audits.sh JOB_ID
```

Send the versioned memory/companion JSON and TXT, QA report, checkpoint DOCX,
combined audit log and full deploy log. REVIEW is not semantic certification.
New grounding records include server-derived `source_quote_span`; old events
have no such proof and are not rewritten or reconstructed as if they had it.
Offline grouped equivalence is diagnostic evidence, not a production waiver.

## Quality Limits

R16 summary subject substitution remains open. The failed summary experiment
was not adopted or repeated. Source-linked academic-Persian editorial findings
remain review tasks, not a shipped replacement dictionary. Numerical/location
tests cannot certify fluent, accurate academic Persian. No automatic Resume,
new whole-book translation or paid experiment is needed for these local checks.
Live trigger validation remains pending until actual new events are collected.
