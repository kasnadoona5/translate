# Tarjomeh (ترجمه)

**Professional Academic Book Translation System — English → Persian**

Tarjomeh translates full-length academic books (political theory, sociology,
philosophy) from English into publication-quality Iranian Persian (فارسی),
with book-wide terminology consistency, an independent quality judge, and
correct scholarly apparatus. Licensed under **AGPL-3.0**.

## Highlights (v0.2.0)

- **Multi-format**: PDF, EPUB, DOCX, TXT, SRT, Markdown in → PDF, EPUB, DOCX, TXT, SRT out
- **4-layer translation memory** (DocMTAgent/DelTA-inspired): proper-noun records,
  running bilingual summary, long-term TF-IDF retrieval, short-term window —
  the same name/term is rendered identically across the whole book
- **Two-model quality pipeline**: a translator model plus an **independent judge**
  (`[llm.critic]`) that scores every chunk on accuracy / fluency / terminology /
  register and drives the refine loop — judged against the actual glossary, not blind
- **Scholarly-safe mode**: citations, years, page numbers, and footnote markers
  survive untouched — `(Marx 1973, 408)` never becomes `(مارکس ۱۹۷۳، ۴۰۸)`
- **Academic register anchoring**: curated EN→FA exemplars of university-press
  Persian prose are injected into every academic-mode prompt
- **First-occurrence tracking**: `هابرماس (Habermas)` parenthetical appears once
  per book, not once per chunk
- **Correct RTL output**: PDF uses a verified wrap-then-bidi-per-line pipeline;
  DOCX/EPUB set native RTL properties; Vazirmatn fonts are bundled
- **Glossary system**: BabelDOC-compatible CSV with per-term context (Marx's vs
  Bourdieu's "capital"), auto-extraction, compliance checking, auto-correction
- **Robust LLM client**: key rotation, exponential backoff, empty-completion
  retry, automatic `max_tokens` escalation on truncation, per-client endpoints
  (OpenRouter / 9router / any OpenAI-compatible gateway / Ollama)
- **Crash-safe jobs**: SQLite checkpointing, resume, cooperative pause,
  auto-pause after consecutive failures, webhook notifications
- **Web UI + REST API**: drag-drop upload, live SSE progress, job history,
  token-authenticated; full `/api/*` for scripting
- **Docker**: one-command VPS deployment

## Quick Start

```bash
git clone https://gitlab.com/personal-group4462928/translate.git
cd translate

# 1. Configuration — ONE file to fill
cp config.example.toml config.toml    # defaults; you normally never edit this
cp .env.example .env
nano .env                             # ← models + keys + token go HERE

# 2a. Docker (recommended)
docker-compose up --build -d          # web UI on port 8080

# 2b. Or local install (Python 3.11+)
pip install -e .
tarjomeh serve --port 8080
```

Open `http://localhost:8080/?token=YOUR_UI_SECRET_TOKEN` (the token you set in `.env`).

## Choosing models (translator + judge)

Everything is set in **`.env`** — two simple blocks. Each block can point at
OpenRouter **or** 9router (or any OpenAI-compatible gateway), independently:

```bash
# ---- TRANSLATOR (main model) ----
TRANSLATOR_API_BASE=                  # empty = OpenRouter; or http://172.17.0.1:20128/v1 for 9router
TRANSLATOR_API_KEY=sk-or-xxxxxxxx
TRANSLATOR_MODEL=deepseek/deepseek-v4-flash

# ---- CRITIC / JUDGE (optional second model that grades every chunk) ----
CRITIC_ENABLED=true
CRITIC_API_BASE=                      # may differ from the translator's endpoint
CRITIC_API_KEY=sk-or-xxxxxxxx
CRITIC_MODEL=anthropic/claude-sonnet-5
```

| Setup | Translator | Judge | When |
|---|---|---|---|
| **A** | OpenRouter | OpenRouter (stronger model) | Simplest; OpenRouter handles provider fallback |
| **B** | 9router combo | second 9router combo | You manage fallback combos in 9router |
| **C** ⭐ | 9router (cheap/free) | OpenRouter (strong, paid) | Best cost/quality: bulk translation cheap, independent reliable grading |

Copy-paste blocks for all three are inside `.env.example`.

> ⚠ Never use a free/congested model as the judge — the critique loop
> multiplies calls, and an unreliable judge poisons refinement.
>
> Advanced users can still fine-tune modes, chunking, typography, etc. in
> `config.toml`; `.env` values win for models/keys/endpoints.

## CLI

```bash
tarjomeh translate book.pdf --mode academic --output-format docx
tarjomeh translate article.epub --mode fast --output-format txt
tarjomeh jobs list | status <id> | resume <id> | cleanup <id>
tarjomeh glossary validate glossary/academic_political_theory.csv
tarjomeh serve --port 8080
```

### Quality regression evaluation

Compare two completed jobs locally without making any LLM calls or changing
their translations:

```bash
tarjomeh eval BASELINE_JOB_ID CANDIDATE_JOB_ID
tarjomeh eval BASELINE_JOB_ID CANDIDATE_JOB_ID --format json --output evaluation.json
tarjomeh eval BASELINE_JOB_ID CANDIDATE_JOB_ID --fail-on-regression
```

The evaluator checks chunk/source alignment, empty or duplicate output,
paragraph and heading structure, length anomalies, numbers, note markers,
persisted glossary compliance, QA score changes, and output-file validity.
Reports are available as text, JSON, or CSV. The web workspace also supports
blind A/B decisions and stores explicitly approved corrections in a private
benchmark table for future regression runs.

### Post-edit integrity and QA reliability

The enabled-by-default integrity gate checks every refinement, glossary repair,
and manual chunk retranslation before it can replace persisted Persian text.
Lossy proposals are rejected, the prior translation is retained, and the chunk
is marked for review with structured evidence. Final chunks are also checked
for missing numbers, notes, paragraphs, protected terminology, duplicated text,
and leaked JSON control fields.

Critic and refiner JSON is schema-validated and repaired with a bounded retry
(`translation.qa_json_retries`). Persistent malformed output is recorded as
`qa_unavailable`; it is never treated as a valid low score or approval.
Back-translation reports structured number, entity, negation, omission, and
addition risks while lexical overlap remains advisory.

## Translation modes

| Setting | Fast | Quality | Academic |
|---|---|---|---|
| Chunk size (tokens) | 3000 | 1500 | 1000 |
| Self-critique + refine | ❌ | ✅ (1 iter) | ✅ (2 iter) |
| Back-translation QA | ❌ | 5% sample | 20% sample |
| Web context search | ❌ | ✅ | ✅ |
| Parallelism | configurable | 1 | 1 (sequential, memory-coherent) |

## Glossary format (BabelDOC-compatible)

```csv
source,target,tgt_lng,context,domain
hegemony,هژمونی,fa,Gramsci's concept of cultural dominance,political theory
discourse,گفتمان,fa,Foucault's concept,philosophy
```

The `context` column disambiguates author-specific senses and is shown to both
the translator and the judge.

Phase 3 adds optional namespacing columns after the BabelDOC-compatible fields:

```csv
source,target,tgt_lng,context,domain,sense,author
capital,سرمایه,fa,Marx's economic category,political economy,economic,Marx
capital,سرمایه فرهنگی,fa,Bourdieu's cultural/social capital,sociology,cultural,Bourdieu
```

Use `[glossary].paths` in `config.toml` to load extra domain glossaries while
keeping `[glossary].path` as the primary, highest-precedence glossary.

## Deploying / updating a VPS

```bash
ssh root@YOUR_VPS
cd /opt/translate
git pull origin main
cp config.example.toml config.toml   # one-time: refresh defaults (your .env is never touched)
nano .env                            # models + keys + token (see .env.example)
docker-compose down && docker system prune -f
docker-compose up --build -d
```

Watch a run: `docker logs -f translate_tarjomeh_1` — with a judge configured
you'll see `Critic model active: ... (translator: ...)` at job start.

## Versions

Releases are marked with git tags: `v1.0` (initial engine) → `v2.0` / `v3.0`
(RTL PDF fix, paragraph alignment, independent critic, scholarly-safe
citations, academic exemplars, bundled fonts). Retrieve any version with
`git checkout <tag>`.

## Development

```bash
pip install -e .[dev]
python -m pytest tests/ -q       # 64 tests
python -m compileall src/
```

## Architecture

```
Input → Parse (PDF/EPUB/DOCX/TXT/SRT/MD) → [OCR] → Semantic chunking
  → per chunk: 4-layer memory → [web context] → translate (exemplar-anchored)
    → judge critique → refine loop → glossary compliance [+ auto-correct]
    → [back-translation QA] → memory update → checkpoint
  → assemble (paragraph-aligned) → Persian typography (scholarly-safe)
  → export (RTL-correct PDF/EPUB/DOCX/TXT/SRT) → [webhook]
```

Built on ideas from TranslateBooksWithLLMs, Andrew Ng's translation-agent,
DocMTAgent/DelTA, PDFMathTranslate/DocLayout-YOLO, BabelDOC, epub-translator,
and Aphra.

## License

AGPL-3.0 — see [LICENSE](LICENSE). Bundled Vazirmatn font is under the
SIL Open Font License (see `src/tarjomeh/persian/fonts/OFL.txt`).
