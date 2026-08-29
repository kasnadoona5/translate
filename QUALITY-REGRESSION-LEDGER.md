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

- Deployed evidence baseline: `v10.17.0`, commit `e118adc`
- Candidate code: `v10.18.0` (local until release verification and push)
- Test job: `3c10682243bc`
- Source: *The State: Past, Present, Future* (Bob Jessop, 2016)
- Runtime result: stable transport; checkpoint pause worked; no empty completion;
  106 current-generation LLM attempts completed without transport failure.
- Translation verdict: not ready for the longer run because a central theoretical
  sentence remains malformed, an appositive was duplicated, two structured refiner
  responses became QA-unavailable, and advisory summary/style evidence is not yet
  clean enough to guide later chapters.

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

| ID | Behavior | v10.16 status | Evidence / required behavior |
|---|---|---|---|
| R01 | Source-authored inconsistent enumeration is preserved | PROTECTED | Source says “two issues” and later enumerates first/second/third; output must not silently change the source. |
| R02 | Refiner may reject critic suggestions | PROTECTED | Full candidates and local edits are independently gated; rejected edits do not become memory. |
| R03 | Page headers do not split or contaminate prose | PROTECTED | “Third” and “Fourth” paragraphs remain separate despite page boundaries. |
| R04 | Exact adjacent duplicated phrases are blocked | PROTECTED | Earlier repetitions such as duplicated “complementarity” remain absent. |
| R05 | Non-adjacent grammatical duplication is blocked | PARTIAL | Governed-span checks ran, but chunk 13 duplicated the fourth-element appositive (`... یعنی ... : یعنی ...`). Extend detection to repeated explanatory/appositive frames without deleting source-authored repetition. |
| R06 | Valid restructuring is not rejected by predicate-count heuristics | MONITOR | v10.17 replaces the paragraph-wide count with an edit-local predicate-evidence guard; valid restructuring tests pass, live confirmation pending. |
| R07 | Coordinated source meanings survive and recurring conceptual families remain consistent | PARTIAL | Source actions remained present, but `polity / politics / policy` drifted away from the previously accepted related Persian family. This is not an omission, but it is a project terminology-consistency regression. Phrase-level curated choices may guide the book; production logic must not hard-code this example. |
| R08 | Matrix predicate and attachment remain grammatical | OPEN | The v10.17 strategic-relational sentence regressed from the clearer v10.11/v10.14 structures to `تراکمی نهادی و گفتمانی‌میانجی‌شده است — یعنی بازتاب و انکساری — از توازن...`. The modifier stack, malformed ZWNJ, and interrupted head/complement attachment are independent defects. Admission must test grammatical dependencies and source propositions rather than require one exact Persian wording. |
| R09 | Academic Persian is readable without semantic weakening | PARTIAL | Review exists, but opaque calques and broken matrix clauses still pass or valid repairs are rolled back. |
| R10 | Singular/plural agreement is correct | PROTECTED | The v10.17 output uses `تاریخ‌های دولت‌ها و نظام‌های دولت‌ها ... دارند`; the prior agreement defect did not return. |
| R11 | First-occurrence English anchors are complete and non-duplicated | OPEN | Final QA still reports missing targets for `apparatus` and `methodological individualism`; `(United Kingdom)` was added although the source surface is `UK`. |
| R12 | Citations, ISBNs, superscripts, and source identifiers survive | PROTECTED | Current output preserves these structures; future typography changes must retain this gate. |
| R13 | Persian typography does not corrupt citations or model diacritics | PROTECTED | Hazm `می` splitting and authored ezafe/tanwin regressions remain covered. |
| R14 | Contextual phrases cannot become canonical terminology | PROTECTED | Live memory has explicit authority classes; 81 contextual mappings remained advisory and no unsafe automatic mapping became canonical. |
| R15 | Accepted terminology cannot add source-external scope | PROTECTED | Single-passage corrections remained `reviewed_advisory`; only two curated glossary mappings were canonical. |
| R16 | Layer 2 summary stays advisory and source-consistent | PARTIAL | Trust remained advisory, but the Persian summary copied the malformed `تراکمی نهادی و گفتمانی‌میانجی‌شده` construction and may bias continuity. |
| R17 | Layer 3 preserves retrieval while exposing reliability | PROTECTED | Reliable and advisory entries coexist; low-trust entries must not be forced as terminology. |
| R18 | Layer 4 preserves immediate continuity without granting authority | PROTECTED | Trusted, advisory-review, and structural-only entries remain available with explicit trust. |
| R19 | Style samples are accurate, fluent, representative prose | PARTIAL | Memory population genuinely recovered (7 reliable long-term entries, 5 style samples, and a style profile), but population is not the same as clean authority. At least one admitted low-scoring or awkward sample remains, so style admission still needs semantic and fluency evidence in addition to surface density. |
| R20 | Research cannot override source, glossary, or accepted translation | PROTECTED | Suggestions are unapproved/advisory; continue testing source identity and evidence quality. |
| R21 | Research evidence is attributable and book/author matched | PARTIAL | Sources and intended use are attributable, but many term proposals cite the same generic abstract snippet without the proposed term appearing in that excerpt. They remained unapproved, so no authority leak occurred. |
| R22 | Broken chunks cannot contaminate trusted/style memory | PROTECTED | Needs-review language defects were excluded from trusted/style admission. |
| R23 | Intentional major-section page breaks do not create blank pages | PROTECTED | Current DOCX has no blank paragraphs/pages from the prior regression. |
| R24 | Contents table is RTL and readable | PROTECTED | OOXML contains RTL table direction; companion audit currently under-reports this and should be corrected. |
| R25 | General scholarly tables preserve usable RTL layout | DEFERRED | Table 1.1 remains line-oriented; user explicitly deferred full table reconstruction. |
| R26 | LLM transport/recovery is observable and bounded | PROTECTED | Current run had no empty/zero-token completion; keep reporting current and lifetime rates. |
| R27 | Structured helper success is distinguished from usable structured output | OPEN | Transport failure rate was 0%, but chunks 7 and 10 became `qa_unavailable` after invalid refiner JSON and bounded repair. Reports must expose both rates and deterministic normalization should precede any repair call. |
| R28 | Final QA and companion reports agree on anchor state | OPEN | Per-chunk entity coverage reported no missing anchors while final QA reported two missing first-occurrence targets. Both scopes are valid, but the companion summary must expose the final-export result. |
| R29 | Research excerpts support the exact proposed terminology | OPEN | A term may be called `source_supported` only when its normalized term or a recorded alias occurs in the cited excerpt; otherwise it is book-context evidence only. |
| R30 | Structural table content does not gain duplicate rows | DEFERRED | Table 1.1 repeats `ویژگی‌های اساسی آپاراتوس دولت`. Full scholarly table reconstruction remains deferred by the user; retain source-relative duplicate evidence for the later table phase. |
| R31 | Latin scholarly abbreviations survive Persian typography | MONITOR | v10.18 protects and canonicalizes bounded forms such as `e.g.`, `i.e.`, `cf.`, `ibid.`, and `viz.` through spacing and punctuation normalization; paired tests confirm ordinary mixed Latin punctuation remains localizable. Fresh DOCX evidence is pending. |
| R32 | Objective readability defects can reach bounded repair regardless of reviewer severity labels | MONITOR | v10.18 routes exact-span minor findings only when the English rationale identifies an objective grammar/dependency/spacing defect. Style, synonym, punctuation-preference, and repetition advice remain non-promoted; refiner veto and source-aware admission are tested and intact. Fresh event evidence is pending. |
| R33 | Low-quality prose cannot become active style authority merely because it is complete | MONITOR | v10.18 introduces a configurable score floor for new style admission and excludes persisted below-floor samples from the active style prompt without deleting memory. Unit tests pass; fresh output and memory-event evidence are still required. |

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

Candidate status: the full historical suite passed, so R31 and R32 are now
`MONITOR`; they remain unprotected until a fresh source/output run confirms the
delivered document. R08, R09, R16, and R19 remain `PARTIAL` because
the v10.17 output itself is still the latest live evidence.

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

## Acceptance Criteria for the Next Release

1. All v10.16 open language examples are corrected without altering the source.
2. A valid syntactic restructuring is no longer rejected merely for reducing a
   finite-verb count.
3. Non-adjacent grammatical duplication is detected without flagging deliberate
   repetition or source-authored inconsistency.
4. Contextual/generic memory entries cannot become canonical automatically.
5. All four memory layers remain populated and trust-aware; style memory contains
   at least one quality-approved fluent prose sample when eligible prose exists.
6. Research remains attributable and advisory.
7. Refiner veto, integrity protection, bounded retries, and checkpoint/resume all
   retain their current behavior.
8. The full historical test suite and new paired regression tests pass.
9. Manual source-versus-output review finds no omission, addition, malformed
   Persian sentence, missing required anchor, or revived protected regression.

## Version History

| Version | Job | Result | Important notes |
|---|---|---|---|
| v10.15 | `5210f0ebe736` | REVIEW | Readability loop added; later analysis exposed salvage/predicate weaknesses. |
| v10.16 | `d3508b55f454` | REVIEW | Runtime stable and many regressions protected; R05-R11, R14-R15, and R19 remain open. |
| v10.17 | `3c10682243bc` | REVIEW | Stable runtime and safer authority/rollback behavior; central fluency defect, duplicated appositive, two invalid structured refiners, and weak advisory style/summary evidence remain. |
| v10.18 candidate | pending live run | MONITOR | Objective minor grammar routing, final body-prose readability coverage, Latin scholarly-abbreviation protection, conceptual-series guidance, and a configurable active-style floor implemented; 813 tests pass. |

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
