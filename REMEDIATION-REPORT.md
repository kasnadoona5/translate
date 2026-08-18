# Tarjomeh remediation report

Everything below was implemented in `C:\Users\admin\Downloads\VSCODE\Translate-v84-fix`,
branch `fix/remediation`, on top of baseline commit `f6b863e`.

| | |
|---|---|
| Fixes shipped | Stage 1 (20) + Phase 8 (6 parts) + items 15, 16, 12, 19 + Tier 1 (4) + 10.1 + Option A |
| Commits | 21 total |
| Source files changed | 11 changed + 2 new; **50 of 61 untouched** |
| Tests | **352 baseline → 636 passing** (+284 new, 0 failures) |
| mypy | 93 errors before, **89 after** — four *fewer*; no new findings |
| ruff | 244 before, **240 after** — 4 *fewer*; no new findings |
| Prompt files edited | **none** |
| Pipeline order changed | **no** |
| Memory layers changed | **no** |

## Rules this work was held to

Your seven non-negotiable rules were treated as constraints on every change:

- Never silently correct a source inconsistency → the Unicode check baselines against
  the **source**, so source damage is a *warning*, never a rejection or a repair.
- Never accept a semantic edit without source evidence → no new auto-correction was added.
- Preserve the previous valid translation when an automatic edit fails → all new
  blocking findings retain prior text; none delete or rewrite.
- Initial translation failure still blocks sequential progress → untouched.
- Critic/refiner failure retains the valid translation and marks review → untouched.
- `needs_review` usable for continuity, never as trusted style/terminology → untouched.
- No book-specific words, authors, phrases or page numbers hardcoded → **verified**:
  every new pattern is structural (Unicode category, regex on digits/punctuation).
  Nothing references this book, Jessop, or any page number.

Two further constraints you set:

- **Quality unchanged.** No change alters attempt-one prompt, model, reasoning setting,
  streaming, or provider routing. Phase 8 touches only the *token allowance*.
- **General, not tuned to one book.** The two constants that would have tuned to this
  book (`predictive_min_tokens`, `adaptive_max_tokens`) were left alone — see
  "Withdrawn" below.

## How to verify

```bash
cd /c/Users/admin/Downloads/VSCODE/Translate-v84-fix
./.venv/Scripts/python.exe -m pytest -q                      # 636 passed
./.venv/Scripts/python.exe -m ruff check src/tarjomeh/       # 240 (all pre-existing)
./.venv/Scripts/python.exe -m mypy src/tarjomeh/             # 89 (all pre-existing)
```

Windows note: prefix with `PYTHONIOENCODING=utf-8` for anything printing Persian —
the console is cp1252 and will otherwise raise `UnicodeEncodeError`.

## Summary

Output impact is stated per fix. "None" means the fix cannot change translated text.

| Fix | File | Output impact | Commit |
|---|---|---|---|
| 7.1 `serve` crashed without `--host/--port` | `cli/main.py` | none | `3b437a1` |
| 7.8 `tarjomeh.parsers` was not a package | `parsers/__init__.py` (new) | none | `3b437a1` |
| 7.9 job ids reached the filesystem unvalidated | `jobs/database.py` | none | `3b437a1` |
| 6.4 critique-threshold fallback disagreed with its default | `web/app.py` | review flagging only | `3b437a1` |
| 11.2 hazm split decimals into `N ٫ N` | `persian/typography.py` | **yes** — repairs numerals | `3b437a1` |
| 1.1 UI authorised every request when token unset | `web/app.py` | none | `daa5655` |
| 1.2 API keys persisted to the job DB and served by the API | `config.py`, `pipeline.py`, `database.py` | none | `daa5655` |
| 2.1 jobs never reached `RUNNING`/`FAILED` | `pipeline.py`, `web/app.py`, `cli/main.py` | none | `b7f859a` |
| 2.2 two workers could own one job | `web/app.py` | none (prevents corruption) | `b7f859a` |
| 2.3 rejected uploads were never deleted | `web/app.py` | none | `b7f859a` |
| 2.4 a bad glossary upload destroyed the good one | `web/app.py` | none (prevents loss) | `b7f859a` |
| 4.1 resume attached translations to a changed source | `pipeline.py` | **yes** — refuses corrupt resume | `faf3e74` |
| 4.2 chunk + memory committed separately | `database.py`, `pipeline.py` | none | `faf3e74` |
| 7.2 CLI overrides applied after validation | `config.py`, `cli/main.py` | none | `8620891` |
| 7.3 `--model` never reached Ollama | `config.py` | none for OpenRouter | `8620891` |
| 7.10 one bad search candidate skipped the rest | `context/web_searcher.py` | context quality ↑ | `8620891` |
| 7.4 `_run_async` raised inside a running loop | `pipeline.py` | none | `c6976e1` |
| 7.5 HTTP clients were never closed | `llm_client.py`, `pipeline.py`, `web/app.py`, `cli/main.py` | none | `c6976e1` |
| 7.6 OCR leaked a temp dir and broke resume | `pipeline.py` | none | `c6976e1` |
| 7.7 search budget + glossary writes unlocked | `search_providers.py`, `web_searcher.py`, `web/app.py` | none (prevents loss) | `c6976e1` |
| 8.1–8.6 token budget and length recovery | `llm_client.py` | none on prompts | `3457644` |
| 0e/0f Unicode corruption reached the reader | `quality/integrity.py` | **yes** — blocks corrupt text | `51adba0` |

---

## The 85K escalation — your main question

You asked whether the 50K/85K output budgets were your implementation or just DeepSeek.
**Both, but the compounding was yours**, and that part is now fixed.

DeepSeek genuinely loops (85,000 reasoning tokens with `visible_output: false`). What
turned one loop into a permanent condition was four separate defects in
`core/llm_client.py`, all of which fed each other:

1. **`_record_budget_observation` learned from failures.** On `finish_reason == "length"`
   it recorded `ceil(failed_budget × 1.50)` as *observed demand*. So the cap size set the
   ratchet step: failing at 61,944 recorded 92,916; failing at 85,000 recorded 127,500.
2. **`_history_high_water` never decayed.** It was an all-time `max()`, while the sample
   counter was capped at `history_window`. One bad call raised the floor forever.
3. **The key was the combo name.** `payload["model"]` is `combo1`, not the model that
   answered. One combo fronts 14 models, so DeepSeek's looping statistics became the
   budget MiniMax inherited.
4. **`_adaptive_ceiling` could not bind.** `context_room` used `max()`, so it reported
   more answer room than the window physically has, and a trailing
   `max(payload_max, ceiling)` floored the result at whatever had already been requested.

Observed climb in your log: 32,280 → 45,963 → 46,100 → 108,892 → 150,494, with the
calculated request reaching 192,353 against an 85,000 clamp. 36 of 127 calls failed
(28.35%); one critique took 830,713 ms (13.8 min).

**What changed** (all in `core/llm_client.py`):

| Part | Before | After |
|---|---|---|
| 8.1 `_adaptive_ceiling` (L421) | `max()` inflated room; result re-floored | genuine upper bound: `min(85000, window − prompt − safety)`, floor 0 |
| 8.2 `_record_budget_observation` (L374) | recorded truncated calls at 1.5× | records only `finish_reason == "stop"` with non-empty content |
| 8.3 `_budget_key` (L339, new) | `(combo_name, operation)` | `(operation, transport_profile, serving_model)` |
| 8.4 `_history_demand` (L358, new) | all-time `high_water` | **p90** over the rolling window |
| 8.5 retry growth | unchanged, but unclamped | unchanged, now clamped by the corrected ceiling |
| 8.6 at-ceiling retry | repeated an identical request 3× | escalates to fallback model, else raises `TruncatedCompletionError` |

`response_model` is now threaded into all four recording call sites (L1210, L1285,
L1455, L1530 — sync success/failure and async success/failure).

**Deliberately NOT changed** — you pushed back on all three and you were right:

- `predictive_min_tokens` stays **50000**, `adaptive_max_tokens` stays **85000**.
  Your point was correct: `max_tokens` is a cap, not a quota. Requesting 50,000 and
  using 3,000 bills 3,000, so the constants cost nothing on a call that behaves.
  Lowering them risked truncating a legitimately long critique on a denser book —
  i.e. tuning to this one book.
- **`completion_tokens` is still the metric**, not `visible_tokens`. The budget must
  cover reasoning *plus* answer; sizing from ~3K of visible output would under-budget a
  reasoning model and cause *more* truncation than exists today. The bug was never the
  metric — it was learning from *failed* calls.
- **Reasoning is not disabled from attempt 1.** Verified in code: translation never
  loses reasoning (`_PRESERVE_REASONING_ON_ALL_LENGTH_RETRIES`), and
  `_use_no_reasoning_recovery` gates critique/refinement to the *last* attempt only.
  This already worked correctly; it is now locked by tests.

Your own log contains the proof the 8.6 escalation is the right move: chunk 5 attempt 4
ran `THINK:off` and succeeded in 31,380 tokens after three failures at 85,000.

### One contract change to review

`tests/test_stabilization.py::test_length_failure_raises_next_first_attempt_immediately`
asserted that a truncation raised the **next** call's **first** attempt straight to
`adaptive_max_tokens`. **That assertion was the ratchet.** It is renamed
`test_length_failure_raises_only_its_own_retry` and now asserts the next call opens at
the predictive floor again. Its within-request assertion (50,000 → 75,000) is unchanged,
because escalating *this* call's own retry is correct and was kept.

---

## Detail by fix

### `3b437a1` — crashes and safety

- **7.1** `cli/main.py` `cmd_serve`: `config.server` is a dataclass, not a mapping, so
  `.get()` raised `AttributeError` whenever `--host`/`--port` were omitted — `tarjomeh
  serve` could never start without both flags. Now reads `config.server.host/.port`, and
  **refuses to bind to a non-loopback address without `UI_SECRET_TOKEN`** rather than
  silently exposing every route.
- **7.8** `parsers/__init__.py` **created**. `src/tarjomeh/parsers/` had no `__init__.py`
  at all — the only subpackage of twelve without one — so
  `from tarjomeh.parsers import get_parser` (`adapters/langgraph_adapter.py:49`) failed.
  Exports the document model plus a `get_parser()` resolving via `EXTENSION_PARSER_MAP`.
- **7.9** `jobs/database.py`: `_JOB_ID_RE` (L24) restricts job ids to `[A-Za-z0-9_-]{1,64}`
  and `create_job` raises `ValueError` otherwise — ids reach the filesystem. `cleanup_job`
  no longer globs `f"{job_id}.*"` (metacharacter risk) and now also removes
  `jobs/ocr/<job_id>`.
- **6.4** `web/app.py` `_job_critique_threshold`: the exception fallback returned `7.0`
  while its own default is `9.0`, so a malformed config silently loosened review flagging
  by two points. Now `9.0`.
- **11.2** `persian/typography.py` `_SPACED_DECIMAL_RE` (L67): hazm pads U+066B like
  sentence punctuation, turning `10.5` and table number `1.1` into `N ٫ N` — all 13 table
  numbers in your DOCX were malformed this way. Rejoined after normalisation, restricted
  to spaces/tabs so a paragraph boundary is never merged.

### `daa5655` — security

- **1.1** `web/app.py` `_require_auth` (L283): an unset `UI_SECRET_TOKEN` was treated as
  "development mode" and **authorised every request**, including document download and
  job submission. Under the shipped Docker config (binds `0.0.0.0`, publishes 8080) that
  is remotely exploitable. Now returns **503** and names `TARJOMEH_ALLOW_INSECURE_UI` as
  the deliberate opt-out for a trusted localhost session. Token comparison moved to
  `secrets.compare_digest` over UTF-8 bytes (`_tokens_match`, L276) — constant-time, and
  it does not raise on a non-ASCII token. A missing `FLASK_SECRET_KEY` now warns.
- **1.2** credentials no longer reach the job DB or the API:
  - `config.py`: `_SECRET_CONFIG_PATHS` (L363), `redact_config_secrets` (L369),
    `to_dict(redact_secrets=…)` (L952), `_rehydrate_redacted_secrets` (L628) called from
    `from_dict` before `validate()` so a persisted config reloads its key from the
    environment instead of failing on an empty list.
  - `pipeline.py`: both `create_job` call sites persist a redacted config.
  - `database.py`: `_scrub_config_secrets` (L33) applied on every `get_job`/`list_jobs`
    read, plus a one-time `_init_db` migration stripping plaintext keys written by
    earlier releases.
  - Env-over-TOML key precedence is **unchanged** — verified byte-identical against
    baseline before and after.

### `b7f859a` — job lifecycle

- **2.1** `create_job` writes `PENDING` and only the resume branch ever wrote `RUNNING`,
  so a fresh job sat at `PENDING` for its whole life; `JobStatus.FAILED` was defined but
  written by **no code path at all**. Since `get_job`/`list_jobs` map both `pending` and
  `running` to `processing`, a job that died from a parse or export error showed as
  "processing" forever. Now: `pipeline.py` marks `RUNNING` on every entry path;
  `_record_job_failure` (`web/app.py` L202) persists `FAILED` from both workers, reading
  `raw_status` (because `get_job` rewrites `status` for the frontend) and refusing to
  overwrite a terminal state; `cli/main.py` persists `PAUSED` on Ctrl+C — the message
  already told you to resume a state nothing had written — and `FAILED` on an exception.
- **2.2** resume read `_active_jobs` and submitted with **no lock in between**, so two
  concurrent POSTs both passed the check and both submitted: doubled LLM spend and two
  writers racing on the same chunks, memory state and output file. Retranslate had no
  guard at all. Now `_active_jobs_lock` + `_try_claim_job` (L148) /
  `_assign_job_worker` (L163) / `_release_job_claim` (L169) make check-and-reserve one
  atomic step; a `_JobClaim` placeholder (L135) holds the slot until the `Future` exists
  and reports `done() == False` so a racing request loses. Retranslate returns **409**
  when busy.
- **2.3** the upload was saved *before* the config-JSON, numeric and chapter-selection
  checks, each of which returned 400 without deleting it — at up to
  `MAX_CONTENT_LENGTH` (500 MB) per attempt, with no job row to clean it up later. All
  four early returns now go through `_reject()` (L441), which unlinks first.
- **2.4** the glossary upload saved straight to its destination and validated afterwards.
  Same filename means same path, so uploading a malformed `philosophy.csv` overwrote the
  good one and the error branch then `unlink`ed it. Now validates a staged `.part` copy in
  the same directory and `os.replace`s it in only on success.

### `faf3e74` — resume integrity

- **4.1** resume re-parsed the *current* input file and repopulated translations keyed on
  `chunk_index` **alone**. Editing the source between pause and resume silently attached
  old translation *N* to unrelated new source *N*, with no error anywhere. Added
  `ResumeSourceMismatchError` (L98), `_chunk_fingerprint` (L2117 — whitespace-insensitive
  SHA-256, so a reflow is not mistaken for an edit) and `_verify_resume_alignment`
  (L2122), which reports mismatched/dropped/added indices, logs the event, parks the job
  at `PAUSED_ERROR` and raises. Applied to the resume path and to `retranslate_chunk`.
  No schema change was needed — `save_chunks` already persists `chunk.text`.
- **4.2** `update_chunk`, `save_memory_state` and `save_job_artifact` each opened their
  own connection and committed separately, so a crash between them left a chunk marked
  `COMPLETED` whose memory contribution was never saved — and resume trusts both. New
  `commit_chunk_checkpoint` (`database.py` L788) does all three in **one transaction**.
  The SQL mirrors the three existing methods exactly; only the transaction boundary
  changed.

### `8620891` — CLI, config, search

- **7.2** `from_toml` called `validate()` before `cmd_translate` applied any override, and
  `validate()` requires a non-empty `llm.openrouter.api_keys` when provider is
  `openrouter`. So `tarjomeh translate book.pdf --provider ollama` failed on a config
  whose file-level provider was OpenRouter, and the `ValueError` escaped as a traceback.
  `load()` (L411) and `from_toml()` (L457) now take `validate=True`; `cmd_translate`
  builds overrides **first**, loads with `validate=not overrides`, and reports a clean
  "Configuration error" with exit 1 instead of raising. `cmd_serve` got the same handler.
- **7.3** the Ollama payload reads `config.llm.ollama.model`, but `--model` and
  `TRANSLATOR_MODEL` only wrote `config.llm.model` — so for Ollama both were silently
  ignored. Mirrored at the two points where "the user asked for this model" is known.
  `llm_client.py` was deliberately left alone: a `llm.model or llm.ollama.model` fallback
  there would break configs that intentionally set different models per provider.
- **7.10** `web_searcher.py` called `item.get()` on every element of the parsed term list.
  One non-dict element raised `AttributeError`, which the broad outer handler turned into
  "skip every remaining term in this chunk" plus an error stub replacing `last_report`.
  Malformed elements are now skipped individually.

### `c6976e1` — plumbing

- **7.4** `_run_async` (L2022) built a *second* event loop and called
  `run_until_complete` when a loop was already running — which raises. Not reachable via
  Flask/WSGI, but it broke Jupyter/ASGI embedding. Now `asyncio.run` when nothing is
  running, else a single-worker thread owning its own loop.
- **7.5** `close()`/`aclose()` had **zero call sites anywhere in `src/`**, and `aclose()`
  iterated `self._thread_local.clients`, which by definition holds only the *calling*
  thread's clients — so every worker thread's `AsyncClient` leaked its pool for the
  process lifetime. Added a process-wide `_all_async_clients` registry that `aclose()`
  (L231) drains; `_aclient` (L192) now keys on the *running* loop and evicts entries whose
  loop is gone (necessary because 7.4 creates a fresh loop per call). Added
  `TranslationPipeline.close()` (L2085) with `__enter__`/`__exit__`, an `id()` set so a
  shared critic client is not closed twice, and `_close_pipeline()` (`web/app.py` L192)
  called from `run_job`, `run_resume`, retranslate and export.
- **7.6** OCR ran **before** `create_job` and rebound `input_path` to a
  `tempfile.mkdtemp()` directory nothing ever removed — so every OCR run leaked a
  full-size PDF, and `jobs.input_path` recorded a temp path resume could no longer find.
  OCR now runs after the job record exists, writes to `jobs/ocr/<job_id>/`, and records an
  `ocr_output` artifact that resume reuses instead of re-OCRing.
- **7.7** three unsynchronised races: the search budget check and increment were separate
  statements, so concurrent worker threads could both pass the check and **overrun a paid
  budget** (now one locked check-and-reserve); one `WebContextSearcher` is shared by every
  worker but `cache`/`result_audit` were unguarded (lock added, never held across an
  `await`); and every glossary route does load → mutate → full rewrite, now serialised by
  `_serialise_glossary_writes` (L177) across the whole sequence — the read is part of the
  transaction, so locking only the write would still lose edits.

### `51adba0` — Unicode corruption

`quality/integrity.py`: `corruption_artifacts` (L321) and `describe_corruption` (L332)
detect U+FFFD, private-use characters (U+E000–U+F8FF, i.e. an unrestored
`PersianTypographer` sentinel), and C0/C1 control characters other than tab/newline/CR.
The gate check follows the existing `mixed_script_corruption` idiom but baselines against
**both source and previous**.

Why it matters beyond the one shipped character: corruption that damaged a **digit** was
caught incidentally by `numbers_missing`, but corruption that damaged a **letter** was
invisible. Verified directly against pre-fix code — a candidate with a corrupted Persian
letter gave `blocking=[]` before and `blocking=['unicode_corruption']` after.

---

## Deliberate deviations from the plan

Your agent should know these, because the plan says otherwise.

1. **Fix 0d withdrawn — the plan's diagnosis is wrong.** The plan claims
   `integrity.py` classifies bare years as `citation`, that the citation role "demands the
   exact Latin form", and that this deadlocks every correct repair. **It does not
   reproduce.** `_numeric_text()` already applies `_DIGIT_MAP` to *both* sides before
   extracting occurrences, so source `the 1990s` vs candidate `دهه ۱۹۹۰` reconciles with
   `missing={}` and **passes**, even though the source occurrence is classified
   `citation`. The role label is consumed only for QA reporting
   (`integrity.py:1082`/`1091`) and never gates script; parenthetical citations are
   guarded separately by `protected_source_citations`. Reclassifying bare years would
   have changed QA reporting and weakened citation integrity **while fixing nothing**.
   The chunk-7 rejection is fully explained by the U+FFFD itself — the corrupted
   candidate genuinely lacks a well-formed `1990` — which `51adba0` addresses instead.
2. **11.2 partial.** The plan also wanted decimals added to `_SCHOLARLY_PROTECTED_RE`.
   Dropped: that would keep decimals Latin (`10.5`), whereas your critic and refiner had
   already established the correct output is Persian with `٫` (`۱۰٫۵`). Only the
   post-hazm rejoin is implemented.
3. **`ResumeSourceMismatchError`**, not the plan's `ResumeSourceMismatch` — satisfies
   ruff `N818`. Nothing outside that commit referenced the name.
4. **Renamed two locals in the OCR block** (`ocr_artifact`, `ocr_cached_path`). The plan's
   `existing` shadowed a later `existing` in the same `run()` scope and introduced 4 new
   mypy errors.
5. **Fix 5.1 and 6.1 remain CUT, 3.2/6.2/5.2/6.3/10.1 remain DEFERRED**, as agreed.

## Not done

Stages 3–6 of the merged order are **not** implemented. They are new subsystems rather
than edits, they change translated output, and the plan itself mandates characterization
tests before enforcement (item 20).

| Remaining | Why it was not shipped here |
|---|---|
| Phase 3.1 paragraph-identity gate alignment | Stage 3; needs the item-20 characterization corpus |
| Item 12 source-aware structural audit | New module `quality/structure_audit.py` |
| Item 13 unified candidate admission gate | Reroutes every stage through one entry point |
| Item 14 refiner protection | Depends on item 13 |
| Item 15 **repair** half | Only the *blocking* half shipped (`51adba0`). Repair-from-unambiguous-source, the pre-typography scan and the export-time scan are outstanding |
| Item 16 duplicate/overlapping edit guard | Confirmed real (chunk 5 `mqm-449dcdb6827b`) but rejects text accepted today |
| Item 17 tables as real `w:tbl` | Your DOCX has **zero** `w:tbl`; 82 cells are loose paragraphs and single cells split across four. Needs `pdf_parser` cell rejoining first |
| Item 18 memory/research trust | Mostly already implemented; the additions depend on 15 and 16 |
| Item 19 QA/UI/export reporting | Should follow 12–16 so it has something to report |
| Phase 8.7 `expanded_final_attempt` | Left `false` for zero behavioural change, as the plan advises |
| Before/after payload audit | Asserted by tests (attempt-1 payload byte-identical) but not run against a live book |

**The one thing I could not verify:** no end-to-end run against a real book. Everything
here is verified by unit/integration tests, linters, and direct before/after comparison
against the pre-fix code. A full translation of a scanned PDF would additionally exercise
7.6 (OCR needs the optional `ocrmypdf`) and confirm the payload audit.

## Action items for you

1. **Rotate any API key used while the pre-fix code ran.** Fix 1.2 closed a real hole:
   the full config, including `api_keys`, was written into `jobs.config` and returned by
   `/api/jobs`. I checked both working copies and `jobs/jobs.db` currently contains **0
   jobs** in each, so I cannot confirm a key is sitting in these particular files — but
   the code path that exposed them was live, so treat any key used then as disclosed.
2. **Set `UI_SECRET_TOKEN`.** The UI now returns 503 without it. For a trusted
   localhost-only session use `TARJOMEH_ALLOW_INSECURE_UI=true` deliberately.
3. **Set `FLASK_SECRET_KEY`**, or every signed-in browser session dies on restart
   (now warned).
4. **Do not ship the Werkzeug dev server.** `cmd_serve` still ends in `app.run(...)` and
   `Dockerfile:34` makes that the entrypoint. Outside this fix list, still recommended.

---

# Follow-up round: Tier 1 memory fixes and 10.1

Added after the first report. This is the subset that improves memory and
reliability without granting any new gate the power to reject a translation.

| Fix | Output impact | Commit |
|---|---|---|
| 10.2 + 11.6 front matter stops teaching terminology | **yes** — better prompts everywhere | `82e34a7` |
| 5.2 chapter summaries keyed by position | memory only | `82e34a7` |
| 6.3 FixedChunker carries structural policy | memory only (latent) | `82e34a7` |
| 3.1 paragraph-identity gate aligned; degrades instead of aborting | **yes** — stops losing whole jobs | `9373726` |
| 10.1 a table never shares a chunk with prose | **yes** — moves chunk boundaries | `362eba5` |

Note on 10.2: memory feeds prompts, so a memory change *is* a translation
change. That is the mechanism by which it helps; there is no version of this
that improves memory and leaves output byte-identical. What is unchanged is
pipeline order, prompt templates, model, and reasoning policy.

## Three further plan diagnoses that do not reproduce

Fix `0d` was withdrawn in the first round. Verification against the code and
against the real source PDF withdrew two more, and deferred a third. Your agent
should not implement any of them as written.

**10.3 "empty completions are handled inconsistently" — already correct.**
`_attempt_event` already computes a content-based `visible_output_present` plus a
`token_accounting_status` separating `reported` / `provider_usage_missing` /
`no_visible_output`. The three empty-checks in the sync path, the async path and
the recorder are already logically equivalent
(`not content or not content.strip()` is the same test as
`not bool((content or "").strip())`). The audited
`prompt=0 completion=0 finish=None success=True` was *accurate*: content was
present and the provider reported no counters, which is precisely what
`provider_usage_missing_but_content_present: 6` records. Nothing changed — the
LLM hot path is not worth touching to "fix" correct code.

**11.5 "TOC merged into run-on paragraphs" — wrong mechanism, and the harm is
already gone.** Measured on the real PDF: the offending paragraphs (indices 25,
27, 29, 30 on page 8) each contain about five page-number-like tokens and all
carry `cross_page_join=False`. They were never produced by
`_merge_continuation_paragraphs`, so the plan's proposed change to the merge
predicate is a **no-op** — they arrive as single blocks from PyMuPDF. Both real
harms are already handled:

* memory — the TOC sits in a chapter titled `Contents`, which fix 10.2 now
  excludes from proper-noun extraction. Confirmed against the PDF: the parser
  reports chapters `Contents`, `Tables` and `Abbreviations`, all in the
  front-matter set, while `Preface` is deliberately kept;
* style — measured across all 66 chunks of the book, **zero** front-matter
  chunks are `style_eligible`, so the style profile was never at risk.

What remains is cosmetic run-on text in a section readers skim, and the only
available fix is a "line ends with a page number" heuristic that risks misfiring
on real prose. Deliberately not done.

**Item 17 "export real Word tables" — measured; it needs a confidence gate
before it is safe.** On page 19 (Table 1.1, 82 table paragraphs) the parser
exposes `bbox`, `page`, `font_size` and `reading_order`, but there are **37
distinct x0 values and 78 distinct y0 values** across those 82 paragraphs: every
wrapped cell line gets its own y and its own indent, and the x values cluster in
near-duplicate pairs (80.9/81.2, 128.0/128.2, 175.0/175.2, 233.0/233.3). Neither
axis separates cleanly, so a grid can only be recovered by clustering both — with
a real chance of placing cells in the wrong column. A misaligned six-column
table actively misinforms the reader; 82 loose paragraphs are ugly but readable.
The exporter is already built for the handover: `apply_structural_format` styles
table paragraphs and documents that native cells are emitted "only when a parser
provides an actual matrix".

Prerequisites, in order:

1. cell rejoining in `pdf_parser` — x/y clustering with tolerance;
2. a confidence test that refuses to emit a grid when the column count is
   ambiguous;
3. the plan's specified fallback — a bounded preformatted block with a review
   marker;
4. validation across the book's 369 table paragraphs on six or more pages, plus a
   second book.

## Why Tier 3 was held back

The mechanism is not hypothetical — it is in the audited log. Every Tier 3 item
is a gate, and when a gate rejects a candidate the pipeline retains the PREVIOUS
text:

```
Decision mqm-50974ca7046d: accepted -> [corrected text] [commit=not_committed; integrity=rejected]
INTEGRITY REJECTED: stage=refinement blocking=1 prior translation retained
```

One gate rejected a better candidate and the corrupted text shipped. Items 12,
13, 14, 16 and 15-repair would add five more reject-or-rewrite authorities, each
firing on Persian output using patterns designed for English structure. Item 16
in particular must separate repetition the translation *introduced* from
repetition already present in the source, and Persian legitimately repeats where
English does not. Item 15's repair half does not reject but *rewrites*, so a
wrong reconstruction damages an already-correct translation.

Recommended path: build these as **detection and reporting only** — item 12 is
already specified that way ("Output impact: none directly"), and item 16 can mark
`needs_review` instead of rejecting — then promote them to blocking only after the
item-20 corpus shows they do not fire on correct translations.

---

# Change map — which file, which symbol, which commit

Everything below is keyed by **symbol name, not line number**. Line numbers drift:
`core/pipeline.py` moved by roughly 200 lines over this work, so grep the symbol
rather than trusting the number. Line numbers are given as of commit `39d209b`.

To see any fix's exact diff:

```bash
git show <commit> -- <path>          # one file from one commit
git log --oneline f6b863e..HEAD      # every commit in this work
git diff f6b863e..HEAD -- <path>     # cumulative change to one file
```

| Fix | File | Symbols (new / rewritten) | Line | Commit |
|---|---|---|---|---|
| 7.1 | `cli/main.py` | `cmd_serve` — dataclass access + non-loopback refusal | — | `3b437a1` |
| 7.8 | `parsers/__init__.py` | **new file**; `get_parser`, model re-exports | 1 | `3b437a1` |
| 7.9 | `jobs/database.py` | `_JOB_ID_RE` (new); `create_job`, `cleanup_job` | 24 | `3b437a1` |
| 6.4 | `web/app.py` | `_job_critique_threshold` — fallback 7.0 → 9.0 | — | `3b437a1` |
| 11.2 / 11.3 | `persian/typography.py` | `_SPACED_DECIMAL_RE` (new); `process` | 67 | `3b437a1` |
| 1.1 | `web/app.py` | `_insecure_ui_allowed`, `_tokens_match` (new); `_require_auth` (rewritten) | 269, 276, 283 | `daa5655` |
| 1.2 | `core/config.py` | `_SECRET_CONFIG_PATHS`, `redact_config_secrets`, `_rehydrate_redacted_secrets` (new); `to_dict` | 363, 369, 628, 952 | `daa5655` |
| 1.2 | `jobs/database.py` | `_scrub_config_secrets` (new); `_init_db` migration, `get_job`, `list_jobs` | 33 | `daa5655` |
| 1.2 | `core/pipeline.py` | both `create_job` call sites → `to_dict(redact_secrets=True)` | — | `daa5655` |
| 2.1 | `core/pipeline.py` | `run` — `update_job_status(RUNNING)` on every path | — | `b7f859a` |
| 2.1 | `web/app.py` | `_record_job_failure` (new) | 202 | `b7f859a` |
| 2.1 | `cli/main.py` | `cmd_translate` — persists PAUSED / FAILED | — | `b7f859a` |
| 2.2 | `web/app.py` | `_JobClaim`, `_try_claim_job`, `_assign_job_worker`, `_release_job_claim` (new) | 135, 148, 163, 169 | `b7f859a` |
| 2.3 | `web/app.py` | `_reject` (new, nested in `api_translate`) | 441 | `b7f859a` |
| 2.4 | `web/app.py` | glossary upload — staged `.part` + `os.replace` | — | `b7f859a` |
| 4.1 | `core/pipeline.py` | `ResumeSourceMismatchError`, `_chunk_fingerprint`, `_verify_resume_alignment` (new) | 101, 2192, 2197 | `faf3e74` |
| 4.2 | `jobs/database.py` | `commit_chunk_checkpoint` (new) | 788 | `faf3e74` |
| 7.2 | `core/config.py` | `load`, `from_toml` — `validate=` keyword | 411, 457 | `8620891` |
| 7.2 | `cli/main.py` | `cmd_translate`, `cmd_serve` — override order + clean errors | — | `8620891` |
| 7.3 | `core/config.py` | `_apply_env_overrides`, `update_from_overrides` — mirror to `llm.ollama.model` | — | `8620891` |
| 7.10 | `context/web_searcher.py` | term loop — skip non-dict candidates | — | `8620891` |
| 7.4 | `core/pipeline.py` | `_run_async` (rewritten) | 2097 | `c6976e1` |
| 7.5 | `core/llm_client.py` | `_aclient`, `aclose` (rewritten); `_all_async_clients` registry | 192, 231 | `c6976e1` |
| 7.5 | `core/pipeline.py` | `close`, `__enter__`, `__exit__` (new) | 2160 | `c6976e1` |
| 7.5 | `web/app.py` | `_close_pipeline` (new) | 192 | `c6976e1` |
| 7.6 | `core/pipeline.py` | `run` — OCR after job creation, `jobs/ocr/<job_id>/`, `ocr_output` artifact | — | `c6976e1` |
| 7.7 | `context/search_providers.py` | `_budget_lock` — atomic check-and-reserve | — | `c6976e1` |
| 7.7 | `context/web_searcher.py` | `_lock` — guards `cache` / `result_audit` | — | `c6976e1` |
| 7.7 | `web/app.py` | `_glossary_write_lock`, `_serialise_glossary_writes` (new) | 45, 177 | `c6976e1` |
| 8.1–8.6 | `core/llm_client.py` | `_budget_key`, `_history_demand` (new); `_record_budget_observation`, `_adaptive_ceiling`, `_recovery_payload` | 339, 358, 374, 421 | `3457644` |
| 0e / 0f | `quality/integrity.py` | `corruption_artifacts`, `describe_corruption` (new); gate check | 449, 460 | `51adba0` |
| 10.2 / 11.6 | `core/pipeline.py` | `_FRONT_MATTER_TITLES`, `_is_front_matter` (new); both extraction sites | 1326, 1346 | `82e34a7` |
| 5.2 | `core/pipeline.py` | chapter-end test + grouping → `_chunk_chapter_position` | — | `82e34a7` |
| 6.3 | `chunking/chunker.py` | `FixedChunker.chunk` — `para_style_eligible`, structural metadata | 376 | `82e34a7` |
| 3.1 | `core/pipeline.py` | `ParagraphIdentityError`, `_used_paragraph_protocol` (new); both assembly sites | 1834, 1841 | `9373726` |
| 10.1 | `chunking/chunker.py` | `SemanticChunker.chunk` — `buffer_is_table` table boundary | 179, 194, 228 | `362eba5` |
| 16 | `quality/integrity.py` | `_repeated_spans`, `newly_repeated_spans` (new); `duplicate_paragraph` baselined | 332, 349 | `1424cd7` |
| 15 repair | `quality/integrity.py` | `repair_corruption` (new) | 377 | `f5559ac` |
| 15 repair | `core/pipeline.py` | `_translate_single_chunk` — repair before gate and typography | — | `f5559ac` |
| 12 | `quality/structure_audit.py` | **new file**; `announced_counts`, `audit_structure`, `audit_payload` | 127, 186, 273 | `d56716a` |
| 12 | `core/pipeline.py` | `structure_audit` chunk event (report-only) | — | `d56716a` |
| 19 | `web/app.py` | `api_job_qa_report` — new categories, `actionable_structure` | 1271 | `39d209b` |

## Reverse index — what touched each file

| File | Fixes |
|---|---|
| `core/pipeline.py` | 1.2, 2.1, 4.1, 4.2, 7.4, 7.5, 7.6, 10.2/11.6, 5.2, 3.1, 15-repair, 12 |
| `web/app.py` | 6.4, 1.1, 2.1, 2.2, 2.3, 2.4, 7.5, 7.7, 19 |
| `quality/integrity.py` | 0e/0f, 16, 15-repair |
| `core/config.py` | 1.2, 7.2, 7.3 |
| `jobs/database.py` | 7.9, 1.2, 4.2 |
| `core/llm_client.py` | 7.5, 8.1–8.6 |
| `cli/main.py` | 7.1, 2.1, 7.2 |
| `chunking/chunker.py` | 6.3, 10.1 |
| `context/web_searcher.py` | 7.10, 7.7 |
| `context/search_providers.py` | 7.7 |
| `persian/typography.py` | 11.2 / 11.3 |
| `parsers/__init__.py` | 7.8 (new file) |
| `quality/structure_audit.py` | 12 (new file) |

---

# Round three: items 16, 15-repair, 12, 19

Chosen to cut human-review load rather than add to it.

| Item | What changed | Review load |
|---|---|---|
| **16** | `duplicate_paragraph` now baselined; new `duplicate_span_introduced` | **down** — removes a false positive, adds a real catch |
| **15 repair** | `repair_corruption` reconstructs only what the source settles | **down** — unambiguous damage stops needing a human |
| **12** | `structure_audit.py`, report-only | neutral — only 2 of 4 classifications escalate |
| **19** | QA report shows the new categories | neutral — separates "note only" from "review required" |

## Corrections to earlier rounds of this report

Two items were described here as unimplemented. Both were **substantially built
already**, and the earlier description was wrong:

* **Item 16** — the per-issue salvage path in
  `_salvage_local_refinement_edits` already rejected an edit that doubled an
  ADJACENT word (`newly_repeated_adjacent_words`), already guarded overlapping
  issue spans (`overlapping_local_span`), and already enforced span uniqueness,
  locality and size ratios. The genuine gap was the **full-candidate refinement
  path**, which had no repetition check beyond whole-paragraph duplication.
* **Item 14** — the same salvage path already requires an accepted edit to name
  a real span, be uniquely locatable, stay within size bounds, and not cross a
  paragraph boundary. The gap is semantic weakening and unrelated rewriting on
  the full-candidate path.

A third correction, on my own work: the first version of the
`duplicate_paragraph` baseline used set subtraction against the source. That
silently did nothing, because the source is English and the candidate Persian, so
normalised spans never intersect. Only the **count** of duplicated paragraphs is
comparable across languages. Caught while testing and replaced.

## The refiner keeps its veto

Confirmed by `git diff f6b863e..HEAD`: **zero changes** to the refiner's
per-issue decision handling. `_salvage_local_refinement_edits` still treats
anything that is not `accepted` or `partially_applied` as `refiner_rejected`, so
the refiner accepts, partially applies or rejects each MQM finding
independently. Item 14 adds conditions on what an *accepted* edit may commit; it
never forces the refiner to accept anything.

## Still outstanding

| Item | Why it is still open |
|---|---|
| **20** verification corpus | Prerequisite for promoting 12/16 to blocking |
| **17 / 11.4** real Word tables | Needs cell rejoining + a confidence gate. Measured: 37 distinct x0 and 78 distinct y0 values across 82 table paragraphs on one page |
| **13** unified admission gate | 8 scattered `evaluate()` call sites to route through one entry point with uniform restore |
| **14** refiner protection | Semantic weakening / unrelated rewriting on the full-candidate path |
| **18** (2 of 5 parts) | Summary terminology reconciliation; broaden research-source survival |

---

# Round four: Option A — the restore contract, and crash isolation

Two problems, both of which let damage reach the reader.

## 1. A rejection had nothing to restore

Four gate call sites passed no `previous`, so a rejection could only be *logged*.
The final gate matters most: it is the last thing between a candidate and the
export, and it logged `integrity_final_failed` and then returned the rejected
text anyway.

**That is chunk 7.** The gate correctly said "reject". "Reject" meant "retain
prior". Prior was also damaged. So the damage shipped.

`core/pipeline.py` now tracks `last_accepted_translation` — the most recent text
that passed a gate — updated after the initial translation, after each accepted
refinement, and after each accepted glossary auto-correction. The final gate
receives it as `previous`, which is also what lets the corruption and duplicate
checks tell damage *this* step introduced from damage carried in from earlier. On
rejection the last accepted text is restored, the swap is logged as
`integrity_final_restored`, and the chunk is marked for review.

The other three sites are deliberately left without `previous`: `initial_translation`
and the two adaptive-recovery assembly stages all produce the **first** text for a
chunk, so no prior valid version exists by definition. The two recovery stages
already escalate on rejection via a strict retry.

**Considered and rejected:** blocking the initial translation on rejection.
Critique/refine exists precisely to repair a poor first draft — in the audited
run the refiner produced the correct numeral — so stopping there would prevent
the repair rather than enable it.

## 2. A report-only helper could stop a book

`repair_corruption` and `audit_payload` sat unguarded in
`_translate_single_chunk`. Because `retry.pause_on_sequential_error` defaults to
**True** and academic mode forces `parallel_workers = 1`, an exception from
either one would set `PAUSED_ERROR` on the **first** affected chunk — not after
`max_consecutive_errors = 3` — and resume would hit the same chunk again. A hard
loop, with the book stuck.

Both are now wrapped. On failure they log `structure_audit_failed` /
`unicode_corruption_repair_failed` and the translation continues. A report-only
feature must be report-only in its failure mode too.

This was not hypothetical. The `isdigit`/`isdecimal` defect fixed in `d84e647`
was exactly this shape: `str.isdigit()` is True for superscript footnote markers
but `int()` rejects them, so a chunk carrying a footnote marker *and* an
enumeration would have raised `ValueError`. It never reached the user —
`audit_payload` did not exist before `d56716a`, and the user's book runs predate
it — but the exposure was real, and it was found by running the audit over the
real delivered translation rather than over fixtures. That is the argument for
item 20 in one incident.

## Change map additions

| Fix | File | Symbols | Commit |
|---|---|---|---|
| Option A | `core/pipeline.py` | `last_accepted_translation` (new local, 3 update points); final gate `previous=`; `integrity_final_restored`; `try/except` around both helpers | `71ed5bf` |
| crash fix | `quality/structure_audit.py` | `announced_counts` — `isdecimal` not `isdigit` | `d84e647` |

## Test coverage added

| File | Tests | Covers |
|---|---|---|
| `tests/test_remediation_stage1.py` | 96 | fixes 1.1, 1.2, 2.1–2.4, 4.1, 4.2, 7.2–7.7, 7.9, 7.10 |
| `tests/test_llm_budget.py` | 27 | Phase 8 — split into 14 invariants + 13 target behaviours |
| `tests/test_unicode_integrity.py` | 28 | Unicode corruption + item 15 repair refusals |
| `tests/test_tier1_memory.py` | 45 | 10.2/11.6 front matter, 5.2 positions, 6.3 chunker policy |
| `tests/test_paragraph_identity.py` | 15 | 3.1 gate alignment and graceful degradation |
| `tests/test_table_chunking.py` | 11 | 10.1 table boundaries; no paragraph lost or reordered |
| `tests/test_duplicate_guard.py` | 17 | item 16 — mostly false-positive guards |
| `tests/test_structure_audit.py` | 39 | item 12 — weighted toward silence, plus the superscript crash |
| `tests/test_admission_restore.py` | 6 | Option A — restore contract and crash isolation |

Two design notes on this suite:

- **The Phase 8 file is deliberately split.** 14 tests are *invariants* that passed
  identically before and after — the 50K/85K constants, translation keeping reasoning,
  no-reasoning gated to the last attempt, attempt-one payload untouched, small prompts
  keeping the full allowance. They are the evidence that Phase 8 changed nothing it
  should not. The other 13 failed before `3457644` and pass after.
- **Both concurrency tests were checked for whether they actually discriminate.** My
  first search-budget test passed with *and* without the fix, because within a single
  event loop the check and the increment cannot interleave — the race is across worker
  *threads*. It was replaced with one that fails ~4/5 runs on unfixed code (using 6–9 of
  a budget of 3). The glossary test fails 3/3 unfixed (only 3–4 of 12 adds survive).

## Two process notes

- **My patch harness had an idempotency bug.** For additive edits whose replacement
  contains the anchor, the "already applied" guard failed, and a script re-run after a
  mid-run failure applied two edits twice. Caught via a ruff `B025` warning. It had
  introduced one real defect — the OCR block existed twice, which would have run OCR on
  an already-OCR'd file. Fixed, guard rewritten to use an explicit marker, and every
  symbol added across all batches audited: each appears exactly once.
- **The 13 ruff `B023` hits are false positives.** I flagged these earlier as a possible
  uncatalogued bug. They are in `core/term_notes.py` (10), `memory/bilingual_summary.py`
  (1) and `tests/test_stabilization.py` (2) — not in the translation loop. In every case
  the closure is consumed by a `.sub()`/`.subn()` call in the same iteration that defines
  it, so the loop variable can never have advanced. No fix needed.
