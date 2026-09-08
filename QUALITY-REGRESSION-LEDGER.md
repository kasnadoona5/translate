# Tarjomeh Quality and Regression Ledger

This file is the durable quality record for Tarjomeh. Update it after every
translation test and before every release. A behavior is not `FIXED` merely
because a source marker or unit test exists: it needs source/output evidence
and a regression test. Never remove an old row; change its status and append
new evidence so later patches cannot silently revive an earlier defect.

## Status Legend

- `PROTECTED`: confirmed in source, translated output, and tests.
- `PARTIAL`: protection exists but a real counterexample remains.
- `OPEN`: confirmed defect requiring a general fix.
- `MONITOR`: currently acceptable but needs evidence from future runs.
- `DEFERRED`: deliberately postponed by the user; must not be mistaken for fixed.

## Non-Negotiable Invariants

1. The English source is authoritative. Never correct an apparent source error,
   omit a proposition, weaken modality, alter negation, or change a count merely
   to make the Persian read better.
2. Persian must be accurate, fluent, natural, and appropriately academic.
   Fluency permits reordering or sentence splitting, but never simplification,
   reinterpretation, addition, or loss of meaning.
3. The translator/refiner keeps per-issue judgment and may reject critic advice.
   No judge suggestion is applied blindly.
4. The existing translation, critique, refinement, integrity, glossary,
   back-translation, research, checkpoint, export, and recovery stages remain.
5. All four memory layers and style memory remain available. Lower trust changes
   authority and retrieval use; it does not erase useful continuity evidence.
6. Research is contextual evidence, never automatic terminology authority.
7. Incomplete or invalid text must not enter trusted memory or final export.
8. Fixes must be general across books, domains, models, and compatible providers.
   Real examples belong in tests, not in production hard-coded word lists.

## Current Baseline

- Deployed evidence baseline: `v10.27.0`, commit `b45c3f2`
- Test job: `e622b56eec7c`
- Source: *The State: Past, Present, Future* (Bob Jessop, 2016)
- Runtime result: the configured checkpoint finished 16 of 222 chunks (10 completed,
  6 needs review). All 112 recorded LLM calls completed successfully, but critique and
  readability dominated latency and token use.
- Translation verdict: worker ownership, refiner veto, four-layer trust separation,
  advisory research authority, and earlier mechanical typography protections held.
  The run remains `REVIEW`: one material postwar-revival proposition was omitted;
  several sentences remained opaque or semantically weak; exact source identifiers
  drifted in canonical DB text; style had only one warming-up sample; and normal
  checkpoint DOCX export failed because chunk 10 independently reconstructed 81 target
  blocks for 82 source paragraph identities.

## v10.26 Live Result

- Code/runtime: v10.26 reached the configured checkpoint with 11 completed chunks,
  5 needs-review chunks, and 206 pending chunks.
- Stop root cause: the durable terminal event is on database chunk 3/UI chunk 4, not
  chunk 12. The structure detector combined the quantity `eight sources` with the
  unrelated reference `chapter 5` and treated the result as a blocking enumeration
  mismatch. This was a deterministic evidence-classification defect, not an LLM or
  9router transport failure.
- LLM accounting: 126 lifetime attempts, 120 active attempts, 6 successful obsolete
  attempts, and one active incomplete stream that recovered. No successful attempt had
  an empty visible answer.
- Memory/style: all four layers remained available and trust-separated, but the style
  evidence record was not role/genre explicit and admitted a nonrepresentative sample.
  Layer 1 also retained examples whose Persian span included local syntax or omitted one
  member of a coordinated source concept. Such records remain continuity evidence but
  are unsafe as reusable lexical authority.
- Source/output: the explicit `two issues` obligation remained protected, but the
  source-authored `(longue durée)` expression was absent in this run. The central
  strategic-relational sentence and one coordinated conceptual family remained
  academically dense or inconsistent rather than naturally fluent Persian.
- Research remained attributable and advisory-only. Prior identifier, citation,
  abbreviation, note-marker, duplication, worker-lease, and refiner-veto protections
  did not regress. Full scholarly-table reconstruction remains deferred.

## v10.24 Live Result

- Code/runtime: v10.24.1 completed the configured chapter checkpoint with 10
  completed chunks, 6 needs-review chunks, and 206 pending chunks.
- Stop root cause: the durable terminal event records `note_markers_missing` for
  marker `2` on database chunk 11 while the worker was in Persian readability
  review. This was an integrity failure after successful model calls, not an LLM
  transport failure. The marker occurs at an internal sentence boundary, outside
  v10.24's unique parenthetical/final-paragraph recovery cases.
- Source fidelity: the output correctly retains `این فصل به دو مسئله می‌پردازد`
  even though the source later enumerates three items. Source-grounded originals,
  citation punctuation, identifiers, and the previously omitted state/civilization
  dynamics also remain present.
- Persian quality: the strategic-relational sentence is more complete and less
  broken than earlier runs, but `در اصطلاحات راهبردی – رابطه‌ای` and its stacked
  condensation modifiers remain an English-shaped, unnecessarily opaque frame.
- Memory/style: all four layers are populated (16 Layer-3 entries: 4 reliable and
  12 advisory; 4 Layer-4 entries). Reliability separation held, but the source line
  `I dedicate this book...` was not recognized by the source-genre filter and its
  translation became one of four active style samples.
- Research: 11 sources and 30 term proposals remained attributable, evidence-marked,
  and advisory; none gained curated terminology authority.
- Audit accuracy: the companion's sole hard failure came from a stale prose-marker
  assertion for atomic salvage. Future runtime checks must exercise behavior or use
  paired capability/event contracts instead of searching for comment wording.

## v10.22 Live Result

- Code status: deployed and exercised through the configured checkpoint. Eleven
  chunks completed, five need review (`9, 10, 11, 13, 14`), and 206 remain pending.
- Transport and worker lifecycle: all 108 calls completed without a recorded
  provider/content failure. There was one lease acquisition and one authoritative
  checkpoint release. Database chunks 11 and 13 took about 23.8 and 18.8 minutes;
  the supplied evidence does not contain the reported manual resume or an actual
  chunk-12/13 failure. Raw `job_log` evidence would be required to prove a transient
  UI/SSE error that was not persisted in the audit bundle.
- Source fidelity: the structure helper correctly knows that an explicit source
  announcement must be preserved, but the live pipeline uses its final result as
  report-only evidence. A source-faithful local count repair was later lost when
  candidate rollback restored an older `three issues` baseline. This disproves R01
  and shows that structure evidence must participate in every admission/rollback.
- Persian prose: the strategic-relational matrix sentence again separates the
  condensation head from its balance-of-forces complement; another paragraph has
  a dangling dependency, and one explanatory dash is unbalanced. Academic
  vocabulary alone did not produce fluent or propositionally transparent Persian.
- Memory and style: all four layers remained populated and trust-separated;
  unresolved chunks stayed out of reliable/style authority. Four body-prose style
  samples were active. The companion hard failure for chunk 12 is false: its raw
  major critique issue was resolved by source-validated repair and canonical final
  authority is true. The audit must consume final disposition and paragraph scope.
- Research: twelve sources and thirty proposals remained attributable and
  advisory-only; no research proposal gained automatic terminology authority.
- Export: native RTL contents and intended heading page breaks remained intact.
  Full scholarly-table reconstruction remains deferred. `(longue duree)` survived,
  but an English legal title received a Persian comma and one body-prose citation
  parenthetical gained a nested English annotation.

## v10.21 Live Result

- Code status: deployed and exercised through the configured checkpoint.
- Job state: 16 of 222 chunks finished; seven completed and nine need review.
- Transport: no recorded active or lifetime LLM failure. The repeated stop was not
  caused by the provider. Both failures are `ValueError: No integrity-valid
  translation remained after bounded critique, refinement, and glossary repair.`
- Root cause: `governed_span_repetition_introduced` reduced citation-bearing spans
  around `Connolly 1969` to adjacent Persian tokens, creating a phrase that does
  not exist in the rendered text. A third run passed only because model wording
  changed, so resume was a stochastic workaround.
- Export: the accepted stored translation contained the source-grounded expression
  `(longue durée)`, but the first English-original audit removed it. A second audit
  overwrote that evidence and reported zero removals, leaving both the document
  and report wrong.
- Memory/style: all four layers remained present and trust-separated. Automatic
  terminology authority stayed advisory. Several multiword title-cased names were
  nevertheless categorized as `technical_loanword`, and the only active style
  sample was dedication-like front matter rather than representative body prose.
  Research remained attributable and advisory-only.
- Previously protected identifier, citation typography, duplicate-span, summary
  transaction, worker-lease, and refiner-veto behavior did not regress. General
  scholarly-table reconstruction remains deferred by user decision.

## v10.20 Live Result

- Code status: deployed and tested on the VPS at the chapter-7 checkpoint.
- Job state: 16 of 222 chunks finished; 12 completed, 4 need review, and the
  partial DOCX/checkpoint artifacts are valid.
- Source fidelity: the initial and earlier refinement versions retained
  `accumulation`, but a later full candidate removed it. Final critique reported
  the omission at major/0.95. The version selector nevertheless preferred the
  smoother incomplete candidate because total blocker count ranked ahead of
  source obligations.
- Persian prose: the strategic-relational sentence still closes its copula before
  the `از` complement, and `که که` survived because one-word duplicate detection
  ignored lexical tokens shorter than four letters.
- Memory/style: all four layers remain populated and trust-separated. The final
  needs-review chunks stayed out of reliable retrieval and active style. Layer 2
  remained advisory but still needed transactional candidate validation so a
  malformed or one-language response could not replace the prior summary.
- Research: book identity and term-level evidence remained advisory and correctly
  separated; no research proposal became curated or mandatory terminology.
- Worker lifecycle: the apparent stop was not database chunk 12. The persisted
  attempts show failures on chunks 9 and 14, both recovered. A first worker made
  10 successful calls before a second generation continued, but the pipeline's
  meaningful release reason was overwritten by generic web cleanup, preventing a
  conclusive historical root-cause statement.
- Export/regressions: previously protected typography, identifiers, citations,
  entity anchors, blank-page behavior, and native RTL contents remained fixed.
  General scholarly-table reconstruction remains explicitly deferred.

## v10.18 Live Result

- Code status: deployed and tested on the VPS at the chapter-7 checkpoint.
- Job state: 16 of 222 chunks finished; 13 completed, 3 need review, and 206
  remain pending. The checkpoint artifact and partial DOCX are valid.
- Transport: 114 active-generation attempts, one recovered incomplete SSE
  failure (`0.88%`), no zero-visible-output success. Nine additional successful
  attempts belong to a superseded worker generation and are reported separately.
- Quality: the v10.18 strategic-relational repair is materially clearer and the
  duplicated appositive is gone. Latin scholarly abbreviations remain intact.
  However, the translation of the Elias sentence omits `dynamics of state and
  civilization`; this is a major source-faithfulness failure.
- Memory: all four layers remain populated and trust-aware. The active style
  floor works, but `at most -> به‌نهایت` was admitted as reliable despite a known
  final fluency defect. The brand sense `polity -> پولیتی` was also reused for
  the political-concept sense, proving that provenance without sense scope is
  insufficient.
- Research: five book-matched sources and thirty advisory suggestions were
  retained without authority escalation. Several excerpts are generic or do not
  support the exact proposed term, so proposal-level evidence remains partial.
- Export: the DOCX contains no empty body paragraphs or explicit manual blank
  pages. Six heading page breaks are intentional. The contents table is native
  RTL in OOXML; the companion report under-reports that fact.
- Performance: critique and readability calls account for most completion tokens
  and elapsed call time. This is an optimization opportunity, not permission to
  remove a quality stage or lower its acceptance standard.
- Apparent UI chunk-12 stop: the evidence shows an old worker generation made
  nine successful calls for database chunk 11, was superseded before committing
  the chunk, and the replacement generation processed it again. It was not an
  LLM-call failure. Existing reports do not retain the pause/restart/lease reason,
  so the initiating action cannot be proven after the fact.

## v10.17 Live Result

- Code status: deployed and tested on the VPS at the chapter-7 checkpoint.
- Verification: `804 passed`; source compilation passed; deployment and audit
  Bash syntax passed; embedded audit Python compiled successfully.
- Static-analysis comparison: Ruff remains at the known 244-error baseline;
  mypy reports 87 errors versus the previously recorded 89. No unrelated
  baseline cleanup is included in this release.
- Scope: source-relative grammatical-duplication detection, local predicate
  admission, coordinated-proposition review, terminology authority classes,
  quality-ranked style evidence, and attributable advisory research.
- Preserved behavior: pipeline stages, bounded retries, refiner per-issue veto,
  integrity gates, chapter checkpoints, all four memory layers, style memory,
  and research remain present.
- Live evidence: source PDF, DOCX, QA report, memory report, and companion report
  were compared. Authority separation, rollback safety, research isolation, and
  checkpoint/export behavior worked. Fluency admission and style evidence remain
  partial, so the longer translation must wait for another same-checkpoint test.

## Regression Matrix

| ID | Behavior | Current status | Evidence / required behavior |
|---|---|---|---|
| R01 | Source-authored inconsistent enumeration is preserved | PARTIAL | v10.24 live output preserved the source's explicit `two issues`, but a later run regressed despite tests. v10.29 adds verifiable per-sentence critic coverage while retaining the deterministic final announced-count blocker. Fresh live evidence is required before restoring `PROTECTED`. |
| R02 | Refiner may reject critic suggestions | PROTECTED | Full candidates and local edits are independently gated; rejected edits do not become memory. |
| R03 | Page headers do not split or contaminate prose | PROTECTED | “Third” and “Fourth” paragraphs remain separate despite page boundaries. |
| R04 | Exact adjacent duplicated phrases are blocked | PROTECTED | Earlier repetitions such as duplicated “complementarity” remain absent. |
| R05 | Non-adjacent grammatical duplication is blocked | PROTECTED | The duplicated fourth-element appositive is absent in the v10.18 DOCX, and the paired regression tests remain in the historical suite. Continue source-relative protection so deliberate repetition is retained. |
| R06 | Valid restructuring is not rejected by predicate-count heuristics | MONITOR | v10.17 replaces the paragraph-wide count with an edit-local predicate-evidence guard; valid restructuring tests pass, live confirmation pending. |
| R07 | Coordinated source meanings survive and recurring conceptual families remain consistent | PARTIAL | Source actions remained present, but `polity / politics / policy` drifted away from the previously accepted related Persian family. This is not an omission, but it is a project terminology-consistency regression. Phrase-level curated choices may guide the book; production logic must not hard-code this example. |
| R08 | Matrix predicate and attachment remain grammatical | OPEN | v10.22 again separates the condensation head from its balance-of-forces complement and permits ambiguous predicate attachment. The fix must accept multiple fluent Persian realizations while rejecting the structural defect. |
| R09 | Academic Persian is readable without semantic weakening | PARTIAL | Review exists, but opaque calques and broken matrix clauses still pass or valid repairs are rolled back. |
| R10 | Singular/plural agreement is correct | PROTECTED | The v10.17 output uses `تاریخ‌های دولت‌ها و نظام‌های دولت‌ها ... دارند`; the prior agreement defect did not return. |
| R11 | First-occurrence English anchors are complete and non-duplicated | PARTIAL | `methodological individualism` is now anchored, but final QA still reports a missing `apparatus` anchor in structural contents material. Reconciliation must distinguish prose, tables, contents, citations, and abbreviations without duplicating English originals. |
| R12 | Citations, ISBNs, superscripts, and source identifiers survive | PARTIAL | v10.27 canonical DB text damaged `CB2 1UR`, `JC11.J47`, `politybooks.com`, and `RES-051-27-0303` even though the deterministic restorer could repair them. v10.29 performs and verifies exact source-bound restoration at final canonical admission. Live output is required. |
| R13 | Persian typography mechanism does not corrupt citations or model-authored diacritics | PROTECTED | Hazm `می` splitting, authored ezafe/tanwin preservation, and Latin citation protection remain covered. This row protects preservation mechanics; R59 separately tracks whether model output uses enough academic ezafe marking. |
| R14 | Contextual phrases cannot become canonical terminology | PARTIAL | Authority classes remain present, but v10.26 retained automatic mappings whose observed Persian span included local syntax or failed to represent every coordinated source member. v10.27 strips only recognized context syntax and rejects incomplete coordination before reusable admission. |
| R15 | Accepted terminology cannot add source-external scope | PARTIAL | Reviewed/curated authority separation remains, but context-expanded observed targets show that lexical boundary evidence still needs live v10.27 confirmation. |
| R16 | Layer 2 summary stays advisory and source-consistent | PARTIAL | Trust remains advisory and the earlier malformed construction is gone, but compressed or awkward summary wording can still bias later continuity. Summary admission needs proposition coverage and readability checks without turning the summary into a glossary. |
| R17 | Layer 3 preserves retrieval while exposing reliability | PROTECTED | Reliable and advisory entries coexist; low-trust entries must not be forced as terminology. |
| R18 | Layer 4 preserves immediate continuity without granting authority | PROTECTED | Trusted, advisory-review, and structural-only entries remain available with explicit trust. |
| R19 | Style samples are accurate, fluent, representative prose | PARTIAL | Dedication filtering works, but v10.26 selected nonrepresentative/malformed evidence. v10.27 persists paragraph role, broad genre, quality score, source indices, representative/fallback status, and requires three representative samples before the profile is established. |
| R20 | Research cannot override source, glossary, or accepted translation | PROTECTED | Suggestions are unapproved/advisory; continue testing source identity and evidence quality. |
| R21 | Research evidence is attributable and book/author matched | PARTIAL | Five live sources are book/author matched and all thirty proposals remain advisory. Several proposals still rely on generic excerpts or evidence that does not contain the proposed term or alias; such evidence may provide book context but not terminology support. |
| R22 | Broken chunks cannot contaminate trusted/style memory | PROTECTED | Needs-review language defects were excluded from trusted/style admission. |
| R23 | Intentional major-section page breaks do not create blank pages | PROTECTED | Current DOCX has no blank paragraphs/pages from the prior regression. |
| R24 | Contents table is RTL and readable | PROTECTED | The v10.18 DOCX contains a native 20x2 RTL contents table with `bidiVisual` and right-aligned cells. The companion audit must report this final-export evidence instead of `unknown`. |
| R25 | General scholarly tables preserve usable RTL layout | DEFERRED | Table 1.1 remains line-oriented; user explicitly deferred full table reconstruction. |
| R26 | LLM transport/recovery is observable and bounded | PROTECTED | One active incomplete-SSE critique replayed the same request contract successfully; active failure rate was 0.88%, and no successful attempt had zero visible output. Obsolete-generation calls remain separate from transport failures. |
| R27 | Structured helper success is distinguished from usable structured output | OPEN | Database chunk 10 still became `qa_unavailable` after invalid refiner JSON and bounded repair. Transport success, schema validity, repair outcome, retained baseline, and review reason must remain separately visible. |
| R28 | Final QA and companion reports agree on anchor state | OPEN | Final QA reports the remaining structural `apparatus` anchor while companion summaries can still hide final-export scope. Report per-chunk evidence and final-export reconciliation as separate named checks. |
| R29 | Research excerpts support the exact proposed terminology | OPEN | A term may be called `source_supported` only when its normalized term or a recorded alias occurs in the cited excerpt; otherwise it is book-context evidence only. |
| R30 | Structural table content does not gain duplicate rows | DEFERRED | Table 1.1 repeats `ویژگی‌های اساسی آپاراتوس دولت`. Full scholarly table reconstruction remains deferred by the user; retain source-relative duplicate evidence for the later table phase. |
| R31 | Latin scholarly abbreviations survive Persian typography | PROTECTED | Fresh v10.18 DOCX evidence preserves bounded forms such as `e.g.`, `i.e.`, `cf.`, `ibid.`, and `viz.` together with their citation tails; paired tests still protect ordinary mixed Latin punctuation. |
| R32 | Objective readability defects can reach bounded repair regardless of reviewer severity labels | MONITOR | The route is deployed, but this run produced zero qualifying live minor promotions. Keep the bounded exact-span route and refiner veto; require a future positive live event before marking it protected. |
| R33 | Low-quality prose cannot become active style authority merely because it is complete | PARTIAL | Below-floor and unresolved-objective samples remain inactive, but v10.26 shows that quality alone is insufficient when role/genre provenance is implicit. v10.27 adds representative/fallback evidence without removing any sample from continuity memory. |
| R34 | Every material source proposition survives translation and final admission | OPEN | v10.27 paragraph 96 omitted the source's postwar western revival and European reconstruction proposition. v10.29 makes the existing critic return a complete stable source-segment inventory and requires every uncovered segment to produce a grounded accuracy/omission issue; no extra unconditional LLM stage is added. |
| R35 | Terminology memory is scoped by sense and structural role | OPEN | Front-matter publisher `polity` was correctly rendered as the brand `پولیتی`, then reused for conceptual `polity` in `polity / politics / policy`. Brand, title, person, citation, and lexical-concept senses must not share automatic authority merely because their normalized source surface matches. |
| R36 | Reliable retrieval excludes known unresolved objective defects | OPEN | `at most -> به‌نهایت` entered a reliable long-term entry even though the final fluency score was below threshold and the defect remained unresolved. A chunk may remain useful as advisory continuity, but cannot become reliable/style authority while a grounded objective defect is unresolved. |
| R37 | Worker ownership, pause acknowledgement, and supersession are persistent and auditable | OPEN | UI chunk 12 accumulated nine successful obsolete calls and was then fully reprocessed. Record a persistent worker generation/lease, pause-request and pause-acknowledgement states, heartbeats, stage boundaries, and a supersession reason; never expose an acknowledged `paused` state while the worker is still silently advancing. |
| R38 | Source text stored in retrieval memory is de-hyphenated consistently | OPEN | Delivered Persian no longer exposes `Profes-sorial`, but the source side of long-term memory still stores the PDF line-break artifact. Normalize source retrieval text using parser provenance while preserving real lexical hyphens. |
| R39 | Reusable entity memory separates the lexical target from display annotation | MONITOR | `Manuela Tecusan` is stored with an embedded English parenthetical in the target. Keep the reusable Persian name separate from first-occurrence English rendering metadata so later uses cannot duplicate the original. |
| R40 | Audit conclusions use final-export evidence and expose unknowns honestly | OPEN | v10.18 companion output reported native RTL as unknown despite valid OOXML and could not explain the obsolete worker generation. Reports must distinguish `pass`, `fail`, `not applicable`, and `unknown`, and must not infer a root cause without a persisted event. |
| R41 | A grounded source omission shared by every accepted version receives bounded repair | OPEN | v10.19 database chunk 11 omitted `analysed from at least six perspectives` from every retained candidate, so earlier-version recovery had nothing complete to restore. v10.20 adds one paragraph-local, source-grounded repair with an independent source-aware critic and monotonic integrity/language admission; live confirmation is pending. |
| R42 | Reliable retrieval and active style use one reconciled final-quality authority | OPEN | v10.19 chunk 13 admitted reliable long-term wording while style rejected the same final grounded attachment defect. v10.20 emits one canonical disposition record and fails audits if unresolved wording enters either durable authority; live confirmation is pending. |
| R43 | Duplicate Persian phrases differing only by an optional diacritic are detected source-relatively | OPEN | v10.19 exported `چگونه مدعیِ چگونه مدعی`. v10.20 compares Persian lexical tokens without optional combining marks and removes only exact adjacent multiword duplication absent from the aligned source; deliberate source repetition remains protected. |
| R44 | Parenthetical originals and structural references remain non-nested and localized | MONITOR | v10.23 defers an English first-occurrence anchor when its Persian target occurs only inside parenthetical apparatus, records that deferral, and leaves later eligible prose available. A generalized non-nesting test passes; live output remains required. |
| R45 | A long in-flight quality call is visible without inflating failure accounting | OPEN | v10.19 database chunks 11-12 spent roughly 18-25 minutes in successful quality work while completed-chunk progress stayed unchanged. v10.20 persists a separate `llm_call_started` event and worker stage, exposes it in the job snapshot/UI, and reports unmatched and >=10-minute calls without counting starts as attempts. |
| R46 | Visibly malformed Persian word construction cannot teach durable wording or style | OPEN | v10.19 contained recurring `معناب‌شناسی`. v10.20 explicitly routes malformed morpheme/ZWNJ construction as an objective target-side defect while uncommon technical vocabulary remains allowed; live correction evidence is pending. |
| R47 | Source completeness outranks aggregate polish when selecting an earlier version | OPEN | v10.20 retained a smoother candidate that omitted `accumulation` because generic blocker count preceded grounded source-fidelity count. v10.21 ranks serious source obligations first and tests the exact adverse ranking case without requiring book-specific wording. |
| R48 | Atomic recovery cannot reintroduce a source omission after a full-candidate rollback | OPEN | v10.20 could retain independently salvaged edits after candidate rollback without restoring the exact baseline when post-validation found a new source obligation. v10.21 preserves independently valid edits only when the one bounded post-review is source-monotonic; otherwise it restores the exact pre-refinement baseline and records the rollback. |
| R49 | Short adjacent lexical duplication is repaired only when unsupported by the aligned source | OPEN | `که که` survived v10.20 because one-token detection required four letters. v10.21 accepts two-letter Persian lexical tokens, retains diacritic-aware matching, and preserves deliberate adjacent repetition when the aligned English source also repeats. |
| R50 | A malformed or incomplete bilingual-summary response cannot replace prior Layer-2 memory | OPEN | v10.20 parsed summary updates directly into live state. v10.21 parses into a temporary candidate, requires both language sections and protocol-clean bounded content, then commits atomically while retaining advisory-only authority. |
| R51 | First authoritative worker release reason survives generic web cleanup | OPEN | v10.20's pipeline release could be overwritten by `web_worker_finished`, obscuring pause/supersession evidence. v10.21 makes release idempotent and persists acquire, renew, reject, stale reclaim, pause request/acknowledgement, and first release events. |
| R52 | Governed-phrase detection cannot skip citations or punctuation inside a phrase | PROTECTED | v10.22 completed the formerly failing region without a governed-span false positive, retry, or terminal failure; paired lexical-contiguity tests remain green. |
| R53 | Final cleanup never silently deletes a source-grounded lexical expression | PARTIAL | v10.22 retained `(longue duree)`, but v10.26 omitted the source-authored `(longue durée)` form. v10.27 detects compact lowercase source expressions from non-English orthographic evidence and routes them through the existing required first-occurrence anchor and integrity path; fresh live confirmation is required. |
| R54 | Multiword source names are not downgraded to technical loanwords by phonetic similarity | PROTECTED | v10.22 correctly categorized the observed multiword people in the live memory state while preserving accepted targets and provenance. |
| R55 | Every terminal chunk failure has durable stage and integrity evidence | MONITOR | Implementation and paired tests exist, but v10.22 had no terminal failure, so the durable live event path was not exercised. |
| R56 | Audit scripts default to the latest job independently of stale shell state | FIXED LOCALLY | The released v10.22 launchers used `${1:-${JOB:-LATEST}}`, so a previously assigned shell `JOB` could silently select an older run. The corrected launchers use a function-local job variable and `${1:-LATEST}`; an explicit job ID remains supported. Bash syntax and byte-for-byte companion-file checks pass. |
| R57 | Final structure evidence participates in candidate admission and rollback | MONITOR | v10.23 rejects newly introduced structure conflicts in full and local candidates, ranks source-valid rollback versions first, runs bounded source-aware repair, and blocks memory/export if actionable evidence remains. Tests cover admission, restoration, and version ranking; live adverse evidence remains required. |
| R58 | Audit hard failures consume canonical final dispositions and valid runtime evidence | PARTIAL | Final-quality disposition logic is present, but the v10.24 companion falsely failed because `atomic_local_salvage` searched for removed prose wording. v10.25 replaces the known fragile assertions with behavioral probes or paired capability/event checks; remaining marker contracts continue to be reviewed. |
| R59 | Academic ezafe convention remains stable without over-marking | MONITOR | v10.25 keeps typography preservation-only, retains selective kasra, and demonstrates final-heh ezafe in two general academic exemplars. No deterministic insertion or raw-count quality gate was added; live under/over-marking evidence remains required. |
| R60 | Book-scoped conceptual families remain coherent without global forcing | PARTIAL | `polity / politics / policy` drifted to a non-parallel Persian series. A user-curated book phrase may guide the set, but neither production code nor the shipped global glossary may require one book's exact renderings. |
| R61 | Source-grounded Latin titles retain internal punctuation | MONITOR | v10.23 protects uppercase-leading multiword Latin title/name runs with internal commas while preserving the existing localization of ordinary lowercase English prose. Positive and over-protection regression tests pass; live output remains required. |
| R62 | Style audit and admission are paragraph-scoped and resolution-aware | OPEN | Chunk 12's selected clean paragraph entered style correctly, but the companion audit treated a resolved issue elsewhere in the chunk as unresolved style contamination. Store and audit paragraph identity plus canonical issue disposition. |
| R63 | Long active quality calls identify their chunk and remain distinguishable from stops | PARTIAL | v10.22 had one uninterrupted worker but database chunks 11 and 13 took about 23.8 and 18.8 minutes. Persist and report database/UI chunk number, operation, stage, worker generation, and elapsed time without counting a long call as a failure. |
| R64 | Runtime audit markers test capabilities rather than incidental prose | MONITOR | v10.24 produced a false hard failure from a stale comment-text assertion. v10.25 directly probes aligned note recovery and citation normalization and uses paired function/event contracts for new repair, memory, and style behavior; deployment must fail only on a real missing capability. |
| R65 | Unique internal sentence-terminal note markers recover without guessing | MONITOR | v10.24 stopped on one source marker at an internal sentence boundary. v10.25 restores it only when source/target paragraph and sentence counts align and the marker occurrence and ordinal boundary are unique. Reused, split, merged, or ambiguous cases remain blocking. |
| R66 | Layer-3 text is the exact canonical text admitted before export | OPEN | v10.27 correctly blocked export when independently inferred paragraph boundaries changed canonical assembly, but it left no downloadable checkpoint DOCX. v10.29 persists source and target paragraph identity with each chunk checkpoint and consumes that exact mapping in normal and re-export assembly; uncertain legacy reconstruction is review-only and cannot enter trusted memory. |
| R67 | First-person dedication prose cannot become style authority | MONITOR | v10.24 admitted `I dedicate this book...` as an active sample. v10.25 excludes source-anchored `I/we dedicate [this book/volume/work] to...` paragraphs from style only; ordinary analytical uses of `dedicated to` and `thanks to` remain eligible. |
| R68 | Structure evidence is typed and cannot bind unrelated quantities | MONITOR | v10.26 falsely paired `eight sources` with `chapter 5`. v10.27 binds only count-before-noun announcements to matching semantic categories; exact typed mismatches block, while ambiguous episode evidence is retained for review and cannot stop sequential progress. |
| R69 | Automatic Layer-1 mappings preserve clean lexical boundaries and coordinated meaning | MONITOR | v10.27 removes only independently recognized leading context syntax before organization heads, lets source-derived entity roles override an LLM mislabel, and rejects a target that omits a source `and/or` member. Context remains available in Layers 3/4 even when Layer-1 authority is denied. |
| R70 | Style memory records representative genre dimensions without becoming terminology authority | MONITOR | v10.27 migrates old samples as fallback evidence and records role, genre, source indices, quality, and representative status. Academic guidance preserves argument structure; literary guidance preserves voice/POV. Three representative samples establish a profile; no extra LLM call is added. |
| R71 | Model comparisons are isolated, transparent, and source-disqualification aware | MONITOR | v10.27 adds an offline frozen-suite CLI benchmark that records requested and 9router-served models, weighted quality, latency, tokens, failures, and deterministic source disqualifications. It writes standalone reports and never changes jobs, memory, or runtime model configuration. |
| R72 | Deployment audits consume a stable runtime capability contract | MONITOR | v10.27 exposes a versioned capability manifest and pairs critical declarations with behavioral probes. Comment wording and incidental source strings are no longer release identity. |
| R73 | Every finished chunk has one verifiable canonical paragraph map | MONITOR | v10.29 records source hashes, target offsets/hashes, paragraph indices, and canonical target hash in the same transaction as chunk and memory checkpointing. Missing or invalid maps hard-fail the audit; legacy reconstruction is explicitly reported. |
| R74 | Critic completeness is machine-verifiable rather than prompt-only | MONITOR | v10.29 requires every stable source sentence ID exactly once in the existing critic response and requires every uncovered ID to have a grounded omission/accuracy issue. Invalid inventories receive only the existing bounded JSON repair. |
| R75 | Final canonical text cannot retain a recoverable damaged source identifier | MONITOR | v10.29 runs source-bound identifier/note reconciliation after final typography/citation normalization and blocks final admission if the source identifier counters or labeled surfaces remain incomplete. |
| R76 | Repeated Persian wording does not hide a grounded local repair | MONITOR | v10.29 resolves a critic quote inside its source-corresponding target paragraph. Identical wording elsewhere no longer defeats salvage, while ambiguity inside that paragraph, overlap, predicate loss, source-structure drift, and integrity failure still reject the edit. |
| R77 | Audit output and job selection are truthful | MONITOR | v10.29 wrappers use function-local `${1:-LATEST}`, runtime v10.29 scripts, exact file validation (`is_file`), paragraph-map hashes, critic coverage, memory/style/research authority, and active/lifetime LLM failure accounting. |

## Validated v10.17 External Audit Notes

- Confirmed: the old typography, citation, identifier, entity-gloss, and source
  de-hyphenation regressions did not return in the inspected output. Exact count
  comparisons from the external audit are informative but are not a release
  criterion by themselves.
- Confirmed with qualification: memory trust and population recovered and the
  result is no longer a vacuous pass. Layer-2 summary prose and individual style
  samples can still carry awkward language, so memory content is not yet fully
  recovered.
- Confirmed: the strategic-relational sentence is worse than its v10.11 and
  v10.14 counterparts. This is both model drift and an admission blind spot; a
  glossary alone cannot repair grammatical attachment.
- Confirmed with qualification: the conceptual triad drifted. The current terms
  are not inherently mistranslations, but they violate the book's previously
  accepted coordinated terminology. Preserve this preference through curated
  phrase-level authority, not a global production-code word list.
- Rejected as too broad: do not promote every `minor` readability finding and do
  not treat every punctuation or repetition complaint as objective. Promotion
  requires an exact grounded span, an objective grammar/spacing class or
  deterministic source-relative evidence, a meaningful candidate change, the
  refiner's veto, integrity validation, and final source-aware validation.
- Rejected as too brittle: do not regression-test one exact model-authored
  Persian sentence. Test source proposition coverage, grammatical attachment,
  modifier boundaries, punctuation, terminology authority, and known forbidden
  malformed structures.

## Implemented v10.18 Candidate Design

- Objective minor grammar findings with exact changed Persian spans can now enter
  the existing source-aware refiner path. Minor style preferences, punctuation
  preferences, and ungrounded repetition remain advisory and cannot trigger this
  route. The refiner still decides each issue, and all integrity and final
  source-aware admission checks remain in force.
- One already-existing Persian readability review is now guaranteed for an
  eligible final body-prose candidate when no earlier candidate triggered it.
  Front matter, catalog data, contents rows, tables, notes, and other structural
  material remain excluded. This adds no pipeline stage and does not alter the
  translator prompt, model, provider, retry policy, or memory authority.
- Latin scholarly abbreviations and their citation/page tails are protected
  through Persian typography normalization. The matcher is bounded to known
  scholarly forms; ordinary Latin punctuation remains localizable.
- Style memory retains all four memory layers and stored historical samples, but
  only quality-approved samples at or above `memory.style_min_score` (default
  `75`) may enter the active style profile. This prevents weak prose from gaining
  prescriptive authority without erasing continuity evidence.
- The translation/refinement contract now asks models to preserve distinctions
  and parallelism in recurring coordinated conceptual series when trustworthy
  prior context supports that relationship. Advisory memory remains revisable;
  no book-specific wording or exact Persian sentence is hard-coded.
- Deployment verifies these behaviors in both the built image and running
  container. Companion reports expose minor objective promotions, unrouted
  objective findings, style-floor violations, stored-sample activity, and
  malformed Latin scholarly abbreviations in the rendered DOCX.

Live status: the historical suite passed before release and the v10.18 run now
protects R05, R08, and R31 with delivered-output evidence. R32 remains `MONITOR`
because no live minor-objective promotion occurred. R09, R16, R19, and R33 remain
`PARTIAL`; R34-R38 and R40 are newly confirmed open defects.

Local release verification: `813 passed`; source/test compilation passed; Bash
syntax and embedded audit Python compilation passed. Ruff reports 242 known
baseline findings (previously 244), and mypy remains at the known 87 errors in
17 files. No unrelated lint or typing cleanup was included.

## Implemented v10.17 Patch Design

### 1. Transactional Candidate Admission

Evaluate the saved baseline, the refiner's complete candidate, and every local
repair independently. Apply accepted local repairs to a temporary candidate,
then validate the assembled result before commit. The last accepted translation
remains untouched until the complete transaction passes.

Before: a globally smoother candidate can be rejected because one crude count
changes, or individually safe edits can combine into broken Persian.

After: each repair must improve the targeted source-backed defect, and the final
combination must remain at least as accurate and structurally sound as baseline.

### 2. Source-Aware Proposition and Predicate Validation

Replace the global finite-predicate monotonic rule with sentence-local alignment
of source propositions: matrix predicates, coordinated actions, negation,
modality, explicit counts, entities, and citations. A Persian rewrite may merge
or split clauses when every source relation has a grammatical equivalent.

Before:

`مسائل را چندان حل نمی‌کنند که از آن‌ها ملول می‌شوند`

A valid rewrite was rejected because it contained fewer finite verbs.

After:

`بیش از آنکه مسائل را حل کنند، از آن‌ها دل‌زده می‌شوند.`

This passes because the source contrast and both semantic actions remain.

### 3. Non-Adjacent Grammatical Duplication Guard

Add a sentence-level detector for duplicated governed phrases, complement frames,
and competing matrix predicates, using normalized Persian tokens and structural
cues. It must compare against the source and must not reject deliberate lexical
repetition. A deterministic finding triggers bounded repair; it does not perform
an unconditional string deletion.

Before:

`فرصتی برای تأملی انتقادی دربارهٔ زبانی فراهم می‌کند دربارهٔ زبانی است که...`

After:

`این وظیفه همچنین ما را به تأملی انتقادی دربارهٔ زبانی فرامی‌خواند که...`

### 4. Accuracy-First Fluent Academic Review

Give the reviewer two explicit responsibilities: first preserve a compact set of
semantic invariants from the source; then realize them in idiomatic academic
Persian. Detect missing matrix predicates, bad attachment, subject-verb mismatch,
opaque nominal stacks, literal comparison frames, and unreadable parenthetical
insertion. Sentence splitting is allowed only after invariants are rechecked.

Before:

`قدرت دولتی تراکمی نهادی و گفتمانیِ میانجی‌گری‌شده (بازتاب و انکساری) ...`

After:

`قدرت دولتی، چگالشِ موازنه‌ای متغیر از نیروهاست که به‌واسطهٔ نهادها و گفتمان‌ها میانجی‌گری می‌شود؛ یعنی هم بازتاب آن موازنه است و هم انکسار آن.`

No proposition is removed; only the Persian syntax is made explicit.

### 5. Coordinated-Meaning Completeness

Track coordinated verbs/adjectives and asymmetric pairs separately. The final
candidate must represent each source member unless the Persian language encodes
the same meaning in one demonstrably equivalent expression.

Before: `extended and qualified` became only `تعدیل کرد`.

After: `می‌توان آن را بسط داد و با قیود لازم تعدیل کرد`.

### 6. Four-Layer Memory Authority

Classify extracted mappings as curated exact terms, recurring technical terms,
named entities, contextual phrases, or generic phrases.

- Curated glossary entries remain protected.
- Named entities require source-observed identity and a valid target.
- A technical term becomes canonical only after repeated, independent,
  quality-approved evidence or explicit curation.
- Contextual/generic mappings remain searchable evidence tied to their chunk;
  they are never forced on later contexts.
- Corrections supersede canonical terms only with stronger provenance; old
  evidence remains auditable.

Before: a structurally valid mapping such as `historical semantics` can acquire
context-specific words and later be treated as universal.

After: the expanded rendering remains advisory evidence for that passage; only a
context-independent, repeatedly confirmed rendering may become canonical.

### 7. Summary Reconciliation

Keep the bilingual running summary as semantic continuity, not a glossary. When
a canonical term is superseded, reconcile only exact attributable terminology
claims; do not rewrite narrative context or erase historical evidence. Reject a
summary update that introduces unsupported scope, duplicates propositions, or
loses the English/Persian correspondence.

### 8. Better Style-Memory Admission

Admit paragraph-level style samples only when they pass accuracy, terminology,
register, fluency, readability, and artifact checks. Preserve a diverse set of
representative prose rather than heavy or highly specialized sentences. Replace
a sample only when the candidate is demonstrably better; a weak run cannot erase
previously clean anchors. Rejected style samples do not remove continuity memory.

Before: safe but over-nominalized prose can become the voice model.

After: natural, source-faithful academic paragraphs anchor later wording, while
complex but awkward passages remain available only as lower-trust context.

### 9. Research Evidence Boundaries

Store research facts, style observations, and terminology proposals separately.
Every item should expose source identity, book/author match, supporting excerpt,
confidence, and intended use. Research may disambiguate a concept or identify a
publication, but cannot override the current source, curated glossary, or an
accepted translation without explicit corroboration.

### 10. First-Occurrence Anchor Reconciliation

At final assembly, compare source entities/technical terms with the export and
insert only genuinely missing required English originals. Protect citations and
avoid duplicate nested annotations. Missing anchors become review evidence.

### 11. QA and Observability

Report, per chunk, which baseline/candidate/repair won and why; proposition
coverage; memory admission class; style admission reasons; research provenance;
and any bounded readability call. Broken grammar, missing source propositions,
unsafe canonical mappings, or missing required anchors produce `REVIEW REQUIRED`.

### 12. Regression Tests and Release Gate

Use generalized fixtures derived from every real failure above. Each repair needs
a positive test and an over-correction test. Run the full historical suite and
compare pipeline-stage counts, memory layers, critic/refiner veto behavior,
research authority, checkpoint/resume, DOCX integrity, and LLM failure rates.
No release is allowed if any `PROTECTED` row regresses.

## Planned Post-v10.18 Patch

This plan is intentionally general. Real phrases below are regression fixtures,
not production substitutions or a book-specific word list.

### 1. Final Proposition-Preserving Admission

Extend the existing source-aware admission record so every accepted full
candidate and assembled local-repair candidate is checked against source
propositions: matrix action, coordinated actions, arguments, negation, modality,
explicit quantities, named entities, and citations. A grounded unresolved
omission blocks that candidate. The last complete accepted candidate is retained;
an incomplete repair cannot enter export or memory.

Live failure fixture: `the ... dynamics of state and civilization` must not be
reduced to a phrase meaning only `the very long duration`. Any fluent Persian
wording is allowed if the dynamics, their domain, and the integration/
disintegration phases all remain.

### 2. Sense-Scoped Terminology and Entity Memory

Store the normalized source surface separately from a sense/role scope derived
from document structure and observed use: lexical concept, named entity,
publisher/brand, title, citation, abbreviation, or structural label. Curated
glossary authority remains strongest. Automatic and reviewed-advisory evidence
is retrieved only for a compatible role; incompatible evidence remains visible
for audit but is not placed in the prescriptive terminology context.

Live failure fixture: front-matter publisher `polity -> پولیتی` remains valid for
the brand, but cannot control conceptual `polity` in body prose. Likewise, store
the Persian lexical form of a person's name separately from first-occurrence
English annotation metadata.

### 3. Reliability, Summary, and Style Admission

A long-term entry is `reliable` only when the final accepted text has no
unresolved grounded accuracy, grammar, terminology, integrity, or QA failure and
meets the configured quality floor. Otherwise it remains advisory continuity;
nothing is deleted. Layer 2 remains an advisory bilingual semantic summary and
must preserve source propositions without becoming a glossary. Active style
samples additionally require representative body prose, fluent Persian, and no
unresolved objective finding; diversity and replacement quality are considered
after the hard safety gates.

Live failure fixture: a chunk containing `at most -> به‌نهایت` remains available
as advisory context but cannot become reliable retrieval or active style
authority until corrected and revalidated.

### 4. Evidence-Bound Research Retrieval

Separate book-context evidence from term evidence. A proposal is term-supported
only when the cited excerpt contains the normalized term or a recorded alias and
the source identity matches. Generic abstracts remain useful book context but do
not enter the terminology prompt as support for a particular Persian rendering.
Research never overrides the source, curated glossary, or a stronger accepted
translation.

### 5. Accuracy-First Fluent Academic Refinement

Strengthen the existing translator, critic, readability reviewer, and refiner
contracts around grammatical dependency, natural Persian clause order,
coordinated negation, opaque nominal stacks, and concept-family parallelism.
Accuracy is the first admission dimension; fluency can reorder or split a
sentence only after all source propositions are preserved. The refiner keeps its
per-issue veto, and no critic wording is applied automatically.

Examples of acceptable direction, not hard-coded output:

- `نه سوژه یا چیزی` becomes a complete coordinated Persian construction such as
  `نه سوژه است و نه شیء` when that is what the source states.
- A literal value construction such as `تحلیل ... ارزش فکری دارد` may become
  `تحلیل ... از حیث معرفتی ارزشمند است`, provided every object and theoretical
  qualification remains.
- Related source concepts such as `polity / politics / policy` should retain
  parallel Persian morphology when book-scoped evidence supports it; a publisher
  brand with the same spelling must not influence the series.

No unconditional LLM call is added. Existing reviews become more precise;
additional refinement remains bounded and defect-triggered.

### 6. Persistent Worker Lease and Truthful Pause Semantics

Replace process-local ownership as the sole authority with a database-backed
worker generation/lease, heartbeat, stage name, and supersession reason. A pause
request becomes `pausing` until the active worker reaches a safe boundary and
acknowledges `paused`; the UI must not claim the job is fully paused while work is
still advancing. Resume is rejected while a live lease owns the job. A stale
lease after process death may be reclaimed exactly once and is reported as such.

The current in-flight provider call is allowed to finish; it is not abruptly
cancelled. No partial candidate enters memory or export. Normally the worker
finishes the current chunk transaction, commits it once, then pauses. If the
process dies, the uncommitted chunk may be repeated safely, but the audit records
the exact reason instead of presenting unexplained obsolete calls.

### 7. Source and Export Hygiene

Apply provenance-aware PDF de-hyphenation to source text stored in retrieval
memory while preserving genuine lexical hyphens. Keep final English-anchor
reconciliation role-aware. Correct companion reporting for native RTL contents
tables and distinguish `unknown` from `pass`; full reconstruction of scholarly
tables remains deferred by the user.

### 8. Observability and Regression Gate

Reports will show active and obsolete generations separately; pause request,
acknowledgement, lease expiry, and supersession reasons; stage/token/time totals;
structured-output failures; final proposition omissions; memory admission and
downgrade reasons; style activity; research evidence class; and final-export
anchor/RTL results. Add paired positive and over-correction tests for R34-R40 and
rerun every historical protected regression. No release may revive a protected
row or turn an unknown check into a pass.

## Acceptance Criteria for the Next Release

1. Final admission blocks a candidate with a grounded missing source proposition
   while preserving the last complete accepted candidate.
2. Publisher/brand, entity, title, citation, structural, and lexical-concept
   senses cannot automatically share terminology authority.
3. A known unresolved objective defect cannot enter reliable retrieval or active
   style authority; the evidence remains available as advisory memory.
4. Layer 2 preserves source meaning as advisory continuity, and research labels
   term support only when its cited evidence supports that exact term or alias.
5. Academic Persian is grammatically natural and readable without simplifying,
   omitting, reinterpreting, or weakening the source.
6. Pause/resume has persistent single-worker ownership, truthful `pausing` versus
   `paused` states, and an auditable reason for every superseded generation.
7. Source-side memory removes line-break hyphenation without altering genuine
   lexical hyphens; entity targets remain separate from display annotations.
8. Refiner veto, integrity protection, bounded retries, checkpoint behavior, all
   pipeline stages, all four memory layers, style memory, and research remain.
9. The full historical suite and paired R34-R40 regression tests pass, including
   over-correction tests for source-authored inconsistency and deliberate
   repetition.
10. Manual source-versus-output review finds no omission, addition, malformed
    Persian sentence, missing required anchor, false audit pass, or revived
    protected regression. Full scholarly-table reconstruction remains deferred.

## v10.19 Implementation Status (Local Release Gate)

Implemented on the v10.18 baseline without adding an unconditional LLM call or
changing the translation/critic/refiner/back-translation stage order:

- **R34 late proposition omission:** a grounded serious source-fidelity finding
  at the final bounded iteration now compares all integrity-valid evaluated
  versions and may restore the strongest earlier complete version. Exact ties
  prefer the earlier version. The retained text still receives `REVIEW REQUIRED`
  because it has not received a new critic pass.
- **R35 entity annotation contamination:** reusable entity targets strip only a
  trailing exact copy of their own English original (with an optional life-date
  range). First-occurrence display remains owned by annotation metadata.
- **R36 brand/concept collision:** lowercase one-word entity mappings are scoped
  by source role even when their correction has high authority. Curated lexical
  terms remain authoritative; publisher cues and source capitalization still
  admit genuine entity use.
- **R37 unsafe durable wording/style:** unresolved grounded semantic issues at
  confidence `>= 0.75`, and objective grammar/register issues at confidence
  `>= 0.70`, retain short/long continuity as advisory but cannot become reliable
  retrieval or active style evidence.
- **R39 research provenance:** book-identity evidence and term evidence are now
  separate. `source_supported` requires that a cited title/snippet contain the
  normalized term or an explicit alias.
- **R40 duplicate worker generation:** SQLite now stores worker generation,
  heartbeat, stage, chunk, pause request, acknowledgement, and release reason.
  `pausing` is distinct from `paused`; pause is acknowledged after the atomic
  chunk+memory checkpoint, and a second live worker cannot claim the job. A
  periodic lease heartbeat also keeps long provider calls from being mistaken
  for dead workers when they exceed the stale-lease threshold.
- **R38 source de-hyphenation:** remains open. The existing source-local evidence
  rule is retained because removing every line-break ASCII hyphen would corrupt
  genuine compounds. No unsafe heuristic was introduced merely to clear the row.
- **Deferred table work:** unchanged by user decision. Native RTL contents-table
  verification remains in the audit; general scholarly-table reconstruction is
  still deferred.

Regression evidence added in `tests/test_v1019_source_memory_worker.py` covers
both the positive behavior and overreach boundaries for source-version recovery,
memory/style quarantine, entity sense scoping, annotation normalization, research
evidence, durable pause ownership, and long-call lease heartbeats. Focused
historical verification passed `55` tests before the full-suite release gate.
The final v10.19 release gate passed `820` tests; the sole warning is Rich's
optional Jupyter `ipywidgets` notice and does not affect the service runtime.

## v10.20 Implementation Status (Local Release Gate)

Implemented on the v10.19 baseline without removing or reordering any pipeline
stage and without adding an unconditional LLM call:

- One canonical final-quality record reconciles the latest source-aware critique,
  committed refiner decisions, refiner vetoes, validated source repair, final
  language review, and needs-review state. Reliable long-term memory and active
  style now consume the same authority decision; short-term continuity remains.
- A serious grounded source obligation that survives all accepted versions can
  trigger the existing bounded paragraph-local repair. The candidate is accepted
  only after integrity, locality, non-regression, and an independent source-aware
  critic show fewer grounded source defects with adequate accuracy/terminology.
- Adjacent multiword duplicate detection ignores optional Persian combining marks
  for matching only. It does not normalize the delivered spelling and does not
  remove repetition that the aligned source itself supports.
- The target-only readability reviewer now recognizes visibly malformed morpheme
  or ZWNJ construction but continues to treat uncommon technical vocabulary and
  synonym preference as non-defects. Source authority and refiner veto remain.
- Provider-call starts update the persistent worker stage and are stored separately
  from completed `llm_call_attempt` events. The API/UI and v10.20 audits expose the
  active operation, unmatched starts, and calls lasting at least ten minutes.
- Docker builds now exclude runtime databases/uploads, backups, local 9router data,
  and Git history from their context. Runtime dependencies occupy a reusable layer;
  marker verification still rejects stale images. Deployment cleanup remains
  Tarjomeh-only and verifies the running 9router image/start time are unchanged.

Local release verification passed `831` tests with one optional Rich/Jupyter
warning; source/test compilation, Bash syntax, and both embedded audit Python
programs passed. Ruff remains at the exact v10.19 baseline of `242` findings.
Mypy reports `88` errors in 17 files versus `89` on an archived checkout of the
exact v10.19 baseline, so this patch adds no static-analysis regression. R38
remains open and general scholarly-table reconstruction remains deferred.

## v10.21 Implementation Status (Local Release Verified; Remote Push Pending)

Implemented on the v10.20 baseline without removing or reordering any pipeline
stage and without adding an unconditional LLM call:

- Grounded source obligations now rank before aggregate blocker count when an
  earlier integrity-valid version must be selected. A new major/critical source
  obligation is candidate-wide evidence and no longer depends on fragile overlap
  with the critic's Persian quote.
- Full-candidate rollback still attempts atomic recovery of independent refiner
  decisions. One bounded post-review must show source monotonicity; otherwise the
  exact pre-refinement baseline is restored and the chunk is marked for review.
- Source-relative adjacent duplicate repair now covers short Persian lexical
  tokens such as `که`, while deliberate source-authored repetition remains
  protected. Existing event names are preserved for audit compatibility.
- Layer-2 summary updates are transactional: both English and Persian sections
  must be present, bounded, and protocol-clean before replacing the prior advisory
  context. The summary does not gain terminology or style authority.
- Durable worker evidence now records acquire, renew, rejected duplicate claim,
  stale reclaim, pause request, pause persistence, and first release. A second
  generic release cannot overwrite the authoritative reason.
- The translator, source-aware critic/refiner with veto, bounded readability
  review, integrity checks, glossary, research, back-translation, four memory
  layers, style memory, export, and checkpoint behavior remain present.
- General scholarly-table reconstruction remains deferred by user decision.

Paired regression tests cover positive behavior and overreach boundaries. The
local release gate passed `839` tests; source/test compilation, Bash syntax, and
both embedded audit Python programs passed. Ruff remains at the exact v10.20
baseline of `242` findings, and mypy remains at the exact v10.20 baseline of `88`
errors in 17 files. Release status remains pending only for clean explicit staging
and pushed-tag verification.

## v10.22 Implementation Status (Local Release Verification Complete)

Implemented on the v10.21 baseline without removing or reordering any pipeline
stage and without adding an unconditional LLM call:

- Governed-phrase matching now requires each occurrence to be a contiguous
  lexical phrase. Latin citations, digits, parentheses, semicolons, and other
  skipped apparatus cannot be collapsed into an artificial Persian phrase; real
  source-unjustified repetition remains blocking and repairable.
- Final English-original cleanup preserves source-grounded lexical expressions,
  retains ungrounded insertions for explicit review instead of silently deleting
  meaning, and removes only established duplicate annotations. Ordered audit
  passes are aggregated through final rendered output instead of overwritten.
- Title-cased multiword entity evidence outranks phonetic loanword similarity.
  Stronger observed entity evidence may correct an earlier category while leaving
  the accepted Persian target and provenance history intact.
- Dedication-like front matter remains in short- and long-term continuity but is
  excluded from active style authority. Representative body prose continues to
  use the existing score, objective-quality, and source-structure gates.
- The already bounded final language repair now receives governed-phrase findings.
  Translation and readability prompts explicitly preserve head-complement and
  list-matrix dependencies; the source-aware refiner keeps its per-issue veto.
- Every terminal chunk-stage exception is persisted before status transition with
  its worker stage and bounded latest-integrity evidence. The v10.22 reports show
  this history, all four memory layers, style admission, research authority,
  active/obsolete LLM accounting, and final-original provenance.
- Deployment cleanup remains Tarjomeh-only, verifies the image before replacement,
  rolls Tarjomeh back on failure, and verifies the running 9router container ID,
  image, and start time did not change.

Local verification passed all `848` tests. Source and test compilation, all three
Bash scripts, and every embedded audit Python block passed. Ruff improved from the
v10.21 baseline of `242` findings to `240`; the new regression test is lint-clean.
Mypy remains at the exact baseline of `88` errors in 17 files, with no v10.22 error.

## External v10.23 Audit Assessment

The external plan dated after the v10.22 run was reviewed against the source PDF,
delivered DOCX, persisted events, current code, and project invariants. It is useful
evidence, not an authoritative implementation specification.

| Proposal | Decision | Reason / correction |
|---|---|---|
| Replace raw-critique style hard failure | ACCEPT WITH CORRECTION | The companion false failure is real. Canonical `final_quality_admission`, memory authority, resolution status, and selected paragraph scope must control the verdict. The existing authority-disagreement check correctly did not fire because chunk 12 had durable authority after repair; there was no pipeline/memory disagreement. |
| Sweep all companion hard-failure contracts | ACCEPT | Verify every field against current emitters and classify it as current, stale, unknown, or redundant. Do not assume `transport_replay_changed_payload` is stale without a counterexample; its current exact-replay contract remains meaningful for quality operations. |
| Strengthen written-ezafe guidance | ACCEPT NARROWLY | The measured drift is real: comparable v10.17/v10.22 exports changed from 158 to 3 final-heh hamzas and 192 to 80 ezafe kasras. Add a precise convention and revise fixed synthetic exemplars already in the prompt; do not add long copyrighted excerpts or deterministic blanket diacritics. Monitor eligible contexts and over-marking, not a required raw count. |
| Pin one approved translation of the strategic-relational sentence | REJECT AS STATED | The defect is real, but one exact Persian answer must not be production authority. Add a general head/complement, matrix-predicate, appositive-attachment, and proposition-preservation gate; use this sentence only as a test fixture with several acceptable paraphrases. |
| Require one Persian rendering of `polity / politics / policy` globally | REJECT GLOBALLY; ACCEPT BOOK-SCOPED | A user-curated phrase mapping may guide this book. It must not enter production code or the shipped global glossary, and the refiner may reject it when context changes. |
| Treat nested `pluralism` annotation as table-related | REJECT DIAGNOSIS; ACCEPT DEFECT | The nested parenthetical occurs in ordinary body prose, not a table fragment. Fix first-occurrence annotation inside parenthetical apparatus by safe deferral or non-nesting reconciliation. |
| Keep tables out of this round | ACCEPT | Native RTL contents remain protected; general scholarly-table reconstruction remains deferred. |

The external plan's proposed test of rerunning the Bash audit “against the existing
JSON” is not directly valid: the script reads SQLite and container files, not an
earlier JSON result. The policy needs pure fixture tests plus a read-only rerun against
the preserved VPS job database or a copied test database.

## Planned v10.23 Patch

This plan is general across books, models, providers, and domains. Real examples
below are tests and evidence only; they are not production substitutions.

### 1. Source-Obligation Ledger Through Every Candidate

Create one per-chunk source-obligation record for explicit quantities, negation,
modality, named entities, citations, matrix relations, coordinated propositions,
and structure announcements. Evaluate the initial draft, full refiner candidate,
coherent local salvage, glossary repair, rollback target, and final candidate against
the same record. Once a candidate resolves a grounded obligation, a later candidate
or rollback cannot reintroduce it.

Before: `three issues` can return when rollback selects an older valid-integrity
baseline. After: the count repair survives while an unrelated defective territoriality
edit remains rejected. The final output preserves the source's `two issues` even when
the source later enumerates first/second/third.

### 2. Structure-Aware Admission and Bounded Repair

Move actionable structure evidence from report-only final QA into candidate admission.
An introduced source-count change or unauthorized reconciliation rejects that candidate.
If all retained versions share a grounded mismatch, route only the affected paragraph
through the existing bounded source-aware repair. Accept it only when structure and
source obligations improve, deterministic integrity passes, and no language dimension
regresses. If no source-valid version remains, pause before memory and complete export
with durable evidence rather than committing inaccurate text.

The final source-aware gate runs after glossary correction, canonical Persian
typography, source-grounded language cleanup, and English-anchor reconciliation,
but before final memory commitment and complete export. This ordering prevents a
late formatting or annotation pass from reviving a source defect after an earlier
candidate was validated.

### 3. Fluent Academic Dependency Review

Expand the existing defect-triggered readability route for interrupted head/complement
relations, dangling relative/subordinate clauses, missing matrix predicates, unbalanced
explanatory dashes, opaque English-order nominal stacks, and appositives that change the
governing relation. The target-only reviewer remains advisory; the source-aware refiner
keeps per-issue veto, and all accepted wording passes source and integrity validation.
No unconditional LLM stage or blind edit is added.

Before: the condensation noun is separated from its balance-of-forces complement.
After: any accurate Persian construction is allowed when it clearly states that state
power is a mediated condensation *of* the changing balance and preserves reflection,
refraction, agency, and the coordinated conceptual series.

### 4. Sense- and Phrase-Scoped Terminology Memory

Keep curated, reviewed-advisory, observed entity, and contextual evidence separate.
Record coordinated conceptual families as phrase-level, book-scoped evidence with
source role and sense. A publisher/brand, citation, title, or entity spelling cannot
control a lexical concept with the same surface form. Contextual automatic mappings
that add source-external scope remain advisory and cannot become canonical.

The book may curate a coherent `polity / politics / policy` family, but production
logic neither requires those exact Persian words nor exports them to other books.

### 5. Four-Layer Memory and Summary Admission

Feed final structure/source disposition into the canonical memory decision. Layer 1
keeps provenance and compatible-sense retrieval; Layer 2 commits atomically only when
both summaries preserve explicit quantities and material propositions; Layer 3 marks
only source-valid fluent text reliable; Layer 4 retains safe continuity with explicit
trust. An unresolved source-invalid candidate cannot teach terminology, summary,
reliable wording, or style.

### 6. Paragraph-Scoped Style Authority

Persist the selected source/target paragraph index and hashes, quality score, excluded
issue IDs, final resolution statuses, and source genre with every style decision. Admit
only representative body prose whose selected paragraph is source-faithful, fluent,
academic, and free of unresolved objective or terminology findings. A clean paragraph
may remain useful when a different paragraph in the chunk is under review; raw resolved
critique severity alone cannot create a hard audit failure.

### 7. Research Evidence Boundaries

Retain the current advisory-only authority. A proposal is exact-term evidence only
when the cited excerpt contains its normalized term or recorded alias and the source
identity matches the book/author. Generic excerpts remain book context. Research may
disambiguate but cannot correct the source, force a glossary choice, or override a
source-aware refiner decision.

### 8. Ezafe, Latin Titles, and Parenthetical Anchors

Add a restrained prompt convention for final-heh hamza and disambiguating ezafe kasra,
and update the existing fixed synthetic academic exemplars to carry both conventions.
Typography continues to preserve model-authored marks rather than inserting them.
Protect internal punctuation of source-grounded Latin titles. When a first-occurrence
English anchor falls inside parenthetical apparatus, defer it to the next eligible prose
occurrence or retain explicit review evidence instead of creating nested parentheses.
Treat source-grounded units such as `pt` as scholarly/typesetting apparatus rather
than unexplained English prose. Preserve their source meaning while auditing the
surrounding numeral presentation for consistency.

Apply a role-aware numeral policy: Persian digits in running Persian prose; source
digits in citations, ISBNs, addresses, page references, legal identifiers, edition
data, and other scholarly apparatus. Add positive and over-correction tests so a
typography repair cannot localize protected identifiers or leave ordinary prose
numbers inconsistently Latin without review evidence.

Keep malformed ZWNJ construction, including an adverb or modifier incorrectly welded
to the following word, as an objective target-language defect. Detection must be
linguistic and pattern-based, not a hard-coded list from this book; accepted repairs
remain subject to source-aware refiner veto and final admission.

### 9. Audit Contract and Long-Call Observability

Inventory all 32 companion hard checks against current event fields. Keep current checks,
repair only proven stale/redundant ones, and report unknowns honestly. Extract the
resolution-aware audit policy into a testable pure helper shared by report scripts.
Persist/report database chunk index, UI-facing number, operation, worker generation,
stage, start time, and elapsed time for long calls. Long successful calls remain neither
failures nor retries.

### 10. Executable Regression Release Gate

Add end-to-end adverse-sequence tests in addition to helper tests: source `two`, initial
target `three`, a mixed good/bad refiner candidate, local salvage, later rollback, final
typography, memory admission, and export. The result must retain the source-faithful count
and original good territoriality clause. Add paired tests for genuine `three`, unrelated
`two authors`, deliberate repetition, multiple grammatical paraphrases, resolved versus
unresolved style issues, ezafe under/over-marking, Latin-title punctuation, and nested
anchor deferral. Run all historical protected tests and compare stage counts, refiner
veto, four memory layers, style authority, research authority, checkpoint behavior,
DOCX integrity, and LLM accounting before release.

The human evidence in this ledger remains a local operator record and must not be staged,
committed, or pushed unless the user explicitly requests it. Executable generalized
fixtures belong in the test suite so a documented regression can actually block release.

## v10.23 Implementation Status (Local Release Verification Complete)

Implemented without removing, bypassing, or reordering a pipeline feature and without
adding an unconditional LLM call:

- Actionable deterministic source-structure evidence now participates in full refiner
  admission, coherent local salvage, rollback/version ranking, bounded final repair,
  and the final gate before memory and export. A refiner remains free to reject judge
  advice, but no candidate may silently rewrite an explicit source count or analogous
  high-confidence structure fact merely to reconcile the author's text.
- If every retained version has an actionable source mismatch, the existing bounded
  source-aware repair receives paragraph-scoped evidence. Its output must improve the
  objective structure result, pass integrity, and avoid language-quality regression.
  An unresolved mismatch is persisted as a terminal quality failure before any trusted
  memory or completed export is committed.
- The shared editorial contract now asks for clear, idiomatic academic Persian, intact
  head/complement and appositive relations, and conventional but restrained ezafe. This
  changes guidance, not source authority: accuracy and completeness still outrank
  fluency, and the source-aware refiner retains its per-issue veto.
- Style decisions map a selected local body paragraph back to its original source
  paragraph index. Canonical final quality and paragraph-scoped evidence control audit
  authority; a resolved critique on another paragraph cannot manufacture a style hard
  failure. The four memory layers and style retrieval remain available.
- First-occurrence English anchors are deferred rather than nested when the only target
  occurrence is already inside parenthetical apparatus. Source-grounded compact units
  are treated as apparatus, and uppercase-leading Latin title/name runs preserve their
  internal punctuation without shielding ordinary lowercase English prose.
- LLM attempt events expose database chunk index, UI-facing chunk number, and current
  worker stage in addition to the existing operation, generation, timing, and outcome
  evidence. Long successful calls remain successes, not retries or failures.
- Research remains attributable and advisory-only. No book-specific term, preferred
  Persian sentence, model name, or provider behavior was added to production logic.
- The release scripts retain complete stage, memory, style, research, active/obsolete
  LLM, and failure-rate audits. Their default job is truly `LATEST`; explicit IDs remain
  supported. Deployment first performs host/Tarjomeh cleanup and, only when necessary,
  archives and removes the inactive Tarjomeh runtime image to make build space. It never
  stops, removes, retags, prunes, or writes the active 9router image/container/data, and
  verifies 9router identity, start time, image, and mounts after deployment.

Full scholarly-table reconstruction remains deferred by the user's decision. v10.23
does not claim that work. Statuses above remain `MONITOR` until a fresh source/output,
memory, event, and audit bundle provides live evidence.

Local verification passed all `858` tests. Source and test compilation, all three
Bash scripts, and all three embedded Python heredocs passed. Ruff remains at the exact
v10.22 baseline of `240` findings, with no finding in the new regression test. Mypy
remains at the exact baseline of `88` errors in 17 files and reports no error in a
v10.23-changed source file. The release is pending only explicit staging, commit/tag,
push, and fresh VPS evidence.

## Version History

| Version | Job | Result | Important notes |
|---|---|---|---|
| v10.15 | `5210f0ebe736` | REVIEW | Readability loop added; later analysis exposed salvage/predicate weaknesses. |
| v10.16 | `d3508b55f454` | REVIEW | Runtime stable and many regressions protected; R05-R11, R14-R15, and R19 remain open. |
| v10.17 | `3c10682243bc` | REVIEW | Stable runtime and safer authority/rollback behavior; central fluency defect, duplicated appositive, two invalid structured refiners, and weak advisory style/summary evidence remain. |
| v10.18 | `81a12bf4722f` | REVIEW | Central syntax, abbreviations, style-floor filtering, checkpoint/export, and bounded SSE recovery improved. A source proposition was omitted, `polity` crossed brand/concept senses, a defective phrase entered reliable memory, and one worker generation was superseded before chunk commit. |
| v10.19 (`cd732c5`) | awaiting VPS run | PENDING | Source-version recovery, grounded memory/style quarantine, entity sense scoping, term-specific research evidence, and durable single-worker pause ownership are implemented and locally verified with 820 tests. |
| v10.19 | `fd366e891f28` | REVIEW | Runtime and transport were stable, but all-version source omission, one memory/style authority disagreement, a diacritic-bearing duplicate phrase, malformed Persian morphology, and opaque long-call progress remain. |
| v10.20 (`61a2273`) | awaiting VPS run | PENDING | Canonical final authority, bounded source-obligation repair, duplicate-phrase repair, malformed-word review, truthful long-call visibility, and lower-space Docker deployment are locally verified with 831 tests; live evidence pending. |
| v10.20 | `c34fd358f20b` | REVIEW | Runtime recovery and memory/style authority worked, but a late candidate omitted `accumulation`, short `که که` duplication survived, central prose remained opaque, and the first worker release reason was overwritten. |
| v10.21 | `9c2bd6c99419` | REVIEW | Provider calls were stable, but a citation-bound false governed-repetition finding paused chunk 12 twice; export removed `(longue durée)` and hid the first-pass evidence; entity categories and the sole active style sample also needed refinement. |
| v10.22 | `e2ca5e8049a7` | REVIEW | Stable 108-call run and normal checkpoint pause; citation-bound repetition, grounded-original preservation, entity categories, and representative style filtering improved. Source-count rollback, central syntax, legal-title punctuation, nested anchoring, and a stale style-audit hard failure remain. |
| v10.23 (`1855f98`) | awaiting VPS run | PENDING | Source-monotonic structure admission, bounded final structure repair, paragraph-scoped style evidence, non-nested anchor deferral, source-grounded units, Latin-title punctuation, richer LLM chunk/stage evidence, and guarded low-space Tarjomeh-only deployment are implemented locally. Live validation is required. |
| v10.24 (`3f651d4`) | awaiting VPS run | PENDING | Unique source-confirmed note recovery, independent monotonic refiner salvage, final canonical admission, lazy public memory imports, two-stage observed entity authority, bilingual-summary source/lexical guards, paragraph-level source-genre style filtering, and overlap-safe Latin title protection are implemented locally. No book-specific production wording or unconditional LLM stage was added; live validation is required. |
| v10.24.1 | awaiting VPS deployment retry | PENDING | Maintenance correction for a stale deployment source-marker assertion. The v10.24 image passed every substantive marker, including all new protections, but deployment stopped before recreation because the verifier searched for removed comment wording. Production pipeline behavior is unchanged. |
| v10.24.1 | `e4589658195e` | REVIEW | Stable provider transport and normal checkpoint completion after one resume. Database chunk 11 stopped on a unique internal sentence-terminal note marker. Explicit source `two issues`, proposition coverage, four-layer trust separation, research authority, and prior mechanical protections held; dedication-style admission and opaque academic syntax remained. |
| v10.25 (`pending`) | awaiting VPS run | PENDING | General aligned sentence-note recovery, boundary-safe local salvage, grounded objective language repair, canonical citation/memory identity, first-person dedication filtering, semantic runtime probes, and shared-layer low-space deployment pass all 883 local tests. Live evidence remains required. |
| v10.25 | `f48bf43e6000` | REVIEW | Stable 150-call checkpoint run with no provider/content failure. Note recovery, source count, refiner veto, canonical memory identity, dedication filtering, and prior regressions held; second-stage rollback granularity, role-scoped mappings/originals, dash attachment, contextual terminology, and English-summary digits remain open. |
| v10.26 (`pending`) | awaiting VPS run | PENDING | Transactional second-stage replay, typed dash/attachment evidence, source-scoped originals, role- and scope-safe terminology memory, normalized English Layer-2 digits, evidence-bound research metadata, and long-call progress events are locally implemented; live evidence is required. |
| v10.26 | `7eb03e1025a7` | REVIEW | Checkpoint run with 120 active and 126 lifetime attempts. One incomplete stream recovered. Typed dash, source scope, summary digit, research, and worker evidence held, but an untyped count/category collision caused a false terminal stop; foreign-expression retention, automatic mapping boundaries, representative style evidence, and canonical render identity remain partial. |
| v10.27 (`pending`) | awaiting VPS run | PENDING | Typed nonterminal structure evidence, source-grounded foreign-expression anchors, Layer-1 coordination/boundary safety, genre-aware style records, canonical assembly identity, stable runtime capabilities, and an isolated 9router-aware model benchmark are implemented locally. Live validation is required. |
| v10.27 | `e622b56eec7c` | REVIEW | Stable 112-call checkpoint run with no LLM failure. Memory trust and advisory research held, but one material proposition was omitted, canonical identifiers drifted, style had one warming-up sample, and checkpoint export failed on an 82-source/81-target paragraph reconstruction. Genre metadata was useful but not established; the offline model benchmark was not run. |
| v10.28 (`0cf1bb3`) | not separately translated | HOTFIX | Checkpoint output is published before visible pause and export failure becomes resumable `paused_error`. It fixes state ordering but does not by itself repair v10.27's paragraph-identity mismatch. |
| v10.29 (`pending`) | awaiting VPS run | PENDING | Canonical chunk paragraph identity, verifiable critic sentence coverage, final identifier admission, paragraph-scoped repeated-span salvage, truthful v10.29 audits, and a shared-layer low-space deployment pass all 927 local tests. Inferred paragraph boundaries are review-only before memory/style admission. No pipeline stage, memory layer, refiner veto, or model configuration was removed. |

## v10.24 Live Validation

- A missing note marker may be restored only beside one unique source-confirmed
  parenthetical anchor or at one uniquely aligned paragraph ending. Multiple or
  absent anchors remain blocking and cannot be guessed.
- A rejected coherent refiner candidate is decomposed only into its accepted,
  non-overlapping local edits. Each edit must independently preserve source
  structure, predicates, repetition constraints, and integrity; the refiner's
  explicit veto remains authoritative.
- Exact identifiers and uniquely proven note markers are canonicalized before a
  new deterministic final integrity admission. Finished text cannot enter any
  memory layer or export without passing that gate.
- Public imports of `tarjomeh.quality`, `tarjomeh.glossary`, and
  `tarjomeh.memory.MemoryManager` are cycle-free in a fresh interpreter.
- A first observed Persian rendering of a semantically translated institution,
  organization, publication, product, theory, or legal instrument remains
  advisory. It requires two independent context-compatible observations before
  source-observed authority; curated and reviewed authority rules are unchanged.
- Bilingual-summary updates are transactional: explicit English/Persian
  structure conflicts and high-confidence one-letter drift from accepted Persian
  reject the candidate while retaining the previous summary.
- Dedications, memorial lines, and acknowledgment/thanks prose cannot become
  style authority. Representative body paragraphs in the same chunk remain
  eligible, so continuity and style memory are not needlessly emptied.
- Overlapping scholarly spans are protected longest-first. Internal punctuation
  in a source-grounded Latin title remains Latin without shielding unrelated
  surrounding prose from Persian typography.
- Full table reconstruction remains deferred. No v10.24 claim is made for it.

The `e4589658195e` source/output, memory, event, report, and LLM evidence closes
the original pending run. It validates source-count preservation, final canonical
admission, four-layer trust separation, advisory research, and prior typography
protections. It also supplies the counterexamples recorded in R19/R33, R58, R65,
and R66; those rows remain below `PROTECTED` until a v10.25 live run exercises the
new behavior.

Local release verification: all `868` tests pass. Source/test compilation,
all three Bash scripts, and all three embedded Python heredocs pass. Ruff has
`238` baseline findings (an improvement from v10.23's recorded `240`) and no
finding in the new v10.24 test or lazy public-memory module. Mypy remains at
the exact established baseline of `88` errors in 17 files. Live translation
quality and runtime claims remain pending VPS evidence.

## v10.25 Pending Live Validation

- The internal note marker must be restored only in the uniquely aligned
  sentence-boundary case; reused numbers, split/merged sentences, and ambiguous
  alignment must remain blocking.
- A local refiner replacement may remove one duplicated Persian function word at
  its exact edit boundary only when the aligned source does not repeat it. Content
  words and source-authored repetition must remain unchanged.
- Objective predicate, governor, attachment, scope, and calque findings may enter
  the existing bounded repair route. Acceptance still requires fewer grounded
  defects, no new source obligation, no new critique regression, intact identifiers,
  and all quality dimensions at or above the safety floor.
- Citation house style must be canonical before DB and memory admission. The
  final-candidate event, saved chunk, Layer-3 target, chunk index, and SHA-256 must
  agree exactly; audits hard-fail any mismatch.
- First-person dedication paragraphs are excluded from style authority while all
  four continuity layers retain them. Ordinary analytical uses of `dedicated` or
  `thanks to` must remain eligible.
- Runtime checks must exercise the new note/citation behavior or require paired
  symbols and emitted evidence. A removed comment or rephrased prompt must not
  create a false stale-container verdict.
- Deployment must retain the healthy old image during the shared-layer build,
  preserve rollback until health/version checks pass, and prove 9router image,
  container, start time, and mounts are unchanged.
- Full table reconstruction remains deferred by user decision.

Local release verification: all `883` tests pass. Source and test compilation,
all three Bash scripts, their embedded Python, and both external audit wrappers
pass. Ruff remains at the exact v10.24.1 baseline of `238` findings, and mypy
remains at the exact baseline of `88` errors in 17 files. No new lint or type
finding was introduced. Live translation quality and runtime claims remain
pending a fresh v10.25 VPS run.

## v10.25 Live Validation

- Job `f48bf43e6000` reached the configured checkpoint with 16 of 222 chunks
  finished (12 completed, 4 needs review). No terminal chunk failure is recorded.
  The apparent stop was a 625-second successful readability call. One refiner
  returned invalid structured JSON after bounded repair; prior valid text was kept.
- The aligned internal note-marker repair worked without guessing ambiguous markers.
  The explicit source announcement `two issues` remained two, canonical final text
  hashes agreed across event, chunk, and Layer 3, and the first-person dedication was
  excluded from style authority while continuity memory remained populated.
- The refiner veto and all four memory layers remained active. Research stayed
  attributable and advisory-only. No previous identifier, citation, mixed-script,
  blank-page, entity-boundary, or Persian typography regression was observed.
- Remaining defects are real but bounded: second-stage rollback may discard an
  unrelated valid local edit; explanatory and relational dashes are counted as one
  class; lowercase lexical uses may inherit a single-word brand mapping; globally
  known originals may be treated as locally licensed; context-expanded or
  number-shifted terminology can look reusable; and English Layer-2 digits can be
  localized. These are `OPEN` until v10.26 live evidence is available.
- Full scholarly-table reconstruction remains deferred by user decision.

## v10.26 Pending Live Validation

- A late grounded regression must reject only the implicated refiner edit. Any
  unrelated edit may survive only when replayed from the exact valid baseline and
  independently accepted by the deterministic integrity gate; an ambiguous mapping
  restores the exact baseline and requires review.
- Explanatory em dashes and relational en dashes must be audited by role. A balanced
  em-dash aside plus a relational en dash is valid, while an object marker detached
  from its governor by either dash is objective repair evidence.
- Single-word publisher, product, publication, and organization mappings must not
  capture lowercase lexical uses. Stored context remains available, but retrieval and
  first-occurrence originals are source- and role-scoped.
- Automatic and accepted-review terminology with expanded scope or explicit
  source/target number drift must remain contextual rather than becoming book-wide
  Layer-1 authority. The accepted paragraph remains available to Layers 3 and 4.
- The English half of Layer 2 must use ASCII digits; the Persian half is unchanged.
  Summary updates remain transactional and argument-only.
- Long provider calls must emit periodic in-progress evidence while preserving the
  same request, retry budget, worker lease, and eventual success/failure semantics.
- Research records may retain a semantic role and an exact evidence quote only when
  that quote exists in supplied evidence. Research remains advisory and adds no call.
- The final source-aware fluency path remains conditional and bounded. Accuracy and
  completeness outrank fluency, per-issue refiner rejection remains intact, and no
  pipeline stage or memory layer is removed.
- Generalized tests, all historical protected tests, release scripts, embedded Python,
  Bash syntax, lint baseline, type baseline, runtime image identity, full stage/event
  audit, memory/style authority, research isolation, LLM failure accounting, and
  unchanged 9router identity must pass before live validation.
- Local release verification: all `893` tests pass. Source compilation, all three
  v10.26 Bash scripts, their embedded Python, and both external audit wrappers pass.
  Ruff reports `235` findings, improving the v10.25 baseline of `238`; mypy remains
  at the exact established baseline of `88` errors in 17 files. Live translation,
  runtime identity, and VPS audit claims remain pending a fresh v10.26 run.

## v10.27 Pending Live Validation

- `eight sources` plus an unrelated `chapter 5` reference must not create a
  structure mismatch or stop. A genuine `two issues` to `three issues` change must
  remain blocking with exact typed source and target spans.
- Review-level ambiguous structure evidence may mark the chunk for review but cannot
  become a terminal sequential error. Exact blocking evidence still prevents trusted
  memory and completed export.
- Automatic Layer-1 targets must exclude recognized local syntax, preserve all members
  of coordinated source concepts, and honor source-derived entity roles over an LLM
  category guess. Rejected mappings remain available through ordinary paragraph
  continuity in Layers 3 and 4.
- Style checkpoints must expose representative versus fallback records, paragraph role,
  broad genre, source indices, and quality score. The profile is `established` only
  after three representative samples; old checkpoints remain usable as fallback.
- A compact source-authored foreign expression with clear non-English orthography must
  follow the normal first-occurrence original policy. Ordinary English, possessives,
  names, and citations must not be reclassified by this rule.
- The assembled pre-render document must match persisted canonical chunk text under
  whitespace-only normalization. An unexplained lexical mutation blocks export; source-
  grounded term-note and anchor presentation remains separately audited.
- The runtime must report the v10.27 capability manifest, and critical audit claims must
  also pass behavioral probes. The offline benchmark must report requested and actually
  served 9router models, source disqualifications, weighted quality, tokens, failures,
  and latency without modifying jobs, memory, or production model settings.
- Full historical tests, script syntax, embedded audit Python, four-layer memory, style
  authority, research isolation, refiner veto, worker lifecycle, active/obsolete LLM
  accounting, output validity, and unchanged 9router identity must pass. Full scholarly-
  table reconstruction remains deferred by user decision.
- Local release verification: all `912` tests pass. Source/test compilation, all three
  v10.27 Bash scripts, their embedded audit Python, both external audit wrappers, and
  the offline runtime behavior contract pass. Ruff remains at the exact v10.26 baseline
  of `235` findings, and mypy remains at the exact baseline of `88` errors in 17 files.
  No production LLM stage, memory layer, refiner veto, or model setting was removed or
  changed by the benchmark tooling. Live translation evidence remains required.

## v10.28 Pending Live Validation

- A chapter review checkpoint must publish its requested partial output before the
  job becomes visibly `paused`. For DOCX, the stored `output_path` must point to a
  valid, downloadable Word document containing completed chapters only.
- If checkpoint export fails, the boundary must remain unclaimed and the job must
  enter resumable `paused_error`; a later resume must retry the same checkpoint
  rather than silently advancing beyond it.
- The web upload contract must preserve the exact `stop_after_chapter` selected by
  the user. Chapter selection, parsing, translation, critique, refinement, all four
  memory layers, style memory, research, and final quality gates are unchanged.
- Regression coverage includes the prior TXT checkpoint flow, a real DOCX partial
  export followed by full-book resume, state visibility during export, retryability
  after exporter failure, and upload-form configuration propagation.
- This is an export-state hotfix only. It adds no LLM call, changes no model setting,
  and weakens no source-fidelity or canonical-text check. Full scholarly-table
  reconstruction remains deferred by user decision.
- Local release verification: all `918` tests pass, including real partial-DOCX
  generation and resume. Source/test compilation, all three v10.28 Bash scripts,
  and their embedded Python pass. Ruff remains at the exact v10.27 baseline of
  `235` findings; mypy remains at `88` errors in 17 files. Live checkpoint and
  container evidence remain pending deployment.

## v10.29 Pending Live Validation

- Every newly completed chunk stores one atomic source/target paragraph-identity
  record beside its translation and memory checkpoint. Exact persisted boundaries
  drive normal assembly, checkpoint DOCX generation, and later re-export.
- Legacy text may be partitioned only without lexical change. An inferred mapping is
  marked `NEEDS_REVIEW` before memory/style admission: it remains available as
  advisory continuity but cannot teach reliable terminology or active style.
- The existing critic call must account for every stable source sentence ID exactly
  once. Any uncovered meaning requires a grounded omission/accuracy issue. Invalid
  coverage uses only the existing bounded structured-response repair; no unconditional
  LLM stage or model-setting change was added.
- Final canonical admission reruns deterministic source-bound artifact restoration
  after typography and citation normalization, then blocks unresolved exact identifier
  or labeled-identifier loss. It never invents identifiers absent from the source.
- A refiner repair whose Persian quote appears elsewhere may be localized by its
  grounded source-paragraph ID. Ambiguity within that paragraph, overlap, source
  structure drift, predicate loss, repetition, integrity failure, or refiner veto
  still rejects the edit.
- Runtime probes and both audits verify paragraph hashes/offsets, critic coverage,
  source identifiers, four-layer memory/style authority, research isolation, active
  and lifetime LLM failures, output-file validity, and unchanged 9router identity.
- Local release verification: all `927` tests pass. Source/test compilation, all three
  v10.29 Bash scripts, their embedded Python, runtime behavior probes, and external
  wrappers pass. Ruff improves from `235` to `233` findings; mypy remains at the exact
  v10.28 baseline of `88` errors. Live checkpoint DOCX and VPS audit evidence remain
  required before any `MONITOR`/`PARTIAL` row is promoted.

## Update Procedure

After each test run:

1. Add a version-history row with job ID, models/providers, configuration, and
   verdict.
2. Compare the English source, DOCX, QA report, memory audit, companion audit,
   pipeline events, and LLM-call accounting.
3. Update each affected regression row with concrete evidence.
4. Add every newly discovered defect as a new permanent ID.
5. Mark a row `PROTECTED` only after code inspection, output evidence, and a
   regression test agree.
6. Keep deferred work visible. Never delete evidence merely because a later run
   does not reproduce it.
7. Before release, fail the gate when new live source/output evidence contradicts
   any `PROTECTED` row; update that row to `OPEN` or `PARTIAL` before planning the
   next patch.
8. Keep local operator evidence unstaged and unpushed unless the user explicitly
   authorizes publishing it. Commit generalized executable regression fixtures,
   not book-specific production rules.
