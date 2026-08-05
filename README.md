# Tarjomeh

English-to-Persian academic book translation with persistent terminology,
book-level memory, independent quality review, and RTL document export.

This README documents release **v8.7.1**. Tarjomeh is licensed under AGPL-3.0.

## What Tarjomeh Does

Tarjomeh is designed for long academic books, especially political theory,
philosophy, sociology, and political economy.

Current capabilities include:

- PDF, EPUB, DOCX, TXT, Markdown, and SRT input
- DOCX, EPUB, PDF, TXT, Markdown, and SRT output
- Academic, quality, and fast translation modes
- Four-layer memory:
  - proper-noun and first-occurrence records
  - running bilingual book summary
  - long-term retrieval from earlier translations
  - short-term recent-chunk context
- Context-aware, namespaced CSV glossaries
- Automatic terminology extraction with reviewable terms
- Glossary compliance checking and guarded auto-correction
- Optional pre-translation book research with source tracking
- Optional per-chunk web context for ambiguous terminology
- Independent structured-MQM critic and bounded balanced refinement loop
- Back-translation sampling for semantic-risk detection
- Post-edit integrity checks that reject lossy model edits
- First-occurrence English originals inline or as document notes
- Citation, number, footnote-marker, and scholarly-apparatus protection
- Correct Persian RTL DOCX/PDF/EPUB formatting
- SQLite checkpoints, pause/resume, and incomplete-export protection
- Web review, chunk retranslation, QA reports, glossary approval, and re-export
- Quality regression comparison between completed jobs
- OpenRouter, 9router/OpenAI-compatible gateways, and Ollama
- Normal JSON and SSE chat-completion response compatibility

## Architecture

```text
Document
  -> parser
  -> optional book research and automatic term extraction
  -> semantic chunks
  -> for each chunk, sequentially in academic mode:
       memory context
       optional web context
       glossary and first-occurrence policy
       initial translation
       integrity check
       grounded structured-MQM critic
       per-issue balanced refinement decisions
       glossary compliance and guarded correction
       optional back-translation
       memory update and SQLite checkpoint
  -> completeness gate
  -> document assembly
  -> Persian typography and English-original audit
  -> RTL export
  -> review and QA report
```

The translator does not blindly obey the critic. Grounded critic findings are
advice, and each receives a persisted accepted, rejected, or partially-applied
decision. Integrity failures retain
the previous valid translation. Persistent QA-provider failures retain valid
translation content, mark the chunk `needs_review`, and remain visible in the
QA report.

## Requirements

### Docker deployment

- Linux VPS or local machine
- Git
- Docker
- Docker Compose
  - `docker-compose` 1.29 is supported
  - Compose v2 users may replace `docker-compose` with `docker compose`
- At least one OpenRouter-compatible model/API endpoint or an Ollama server

### Local Python deployment

- Python 3.11 or newer
- A C build toolchain may be needed by some dependencies

## Configuration Overview

Tarjomeh uses two local files:

- `.env`: secrets, endpoints, model names, and UI authentication
- `config.toml`: translation behavior and non-secret defaults

Create them once:

```bash
cp .env.example .env
cp config.example.toml config.toml
chmod 600 .env
```

Never commit `.env`. It is ignored by Git.

Configuration precedence, from lowest to highest:

1. Built-in defaults
2. `config.toml`
3. `.env` translator/critic overrides
4. CLI options or per-job Web UI settings

Editing `.env` or `config.toml` does not modify existing job snapshots. Restart
the application before starting a new job.

## Editing `.env`

Open the file:

```bash
nano .env
```

Do not add spaces around `=`. Quote a value when it contains spaces or `#`.

### Required translator settings

```dotenv
TRANSLATOR_API_BASE=
TRANSLATOR_API_KEY=replace-with-real-key
TRANSLATOR_MODEL=replace-with-model-or-combo
```

`TRANSLATOR_API_BASE`:

- Empty: direct OpenRouter endpoint
- `https://openrouter.ai/api/v1`: explicit direct OpenRouter
- `http://172.17.0.1:20128/v1`: example 9router endpoint on the Docker host
- Any other OpenAI-compatible `/v1` endpoint

Use the actual Docker-host gateway on your server rather than assuming it is
always `172.17.0.1`.

### Independent critic settings

```dotenv
CRITIC_ENABLED=true
CRITIC_API_BASE=
CRITIC_API_KEY=replace-with-real-key
CRITIC_MODEL=replace-with-critic-model-or-combo
```

Blank critic endpoint/key fields inherit the translator endpoint/key. For a
genuinely independent critic, set a different model and, when appropriate, a
different endpoint or provider account.

If `CRITIC_ENABLED=false` and `CRITIC_MODEL` is empty, the translator client is
used for QA. Setting `CRITIC_MODEL` activates the critic even when the enabled
flag is omitted.

### Recovery settings

```dotenv
# Translator: 12K legacy floor with predictive first-attempt sizing
TRANSLATOR_MAX_TOKENS=12000
TRANSLATOR_RECOVERY_MODEL=
TRANSLATOR_RECOVERY_MAX_ATTEMPTS=2
TRANSLATOR_RECOVERY_MAX_TOKENS=24000

# Critic: predictive normal attempt plus bounded recovery
CRITIC_RECOVERY_MODEL=
CRITIC_RECOVERY_MAX_ATTEMPTS=4
CRITIC_RECOVERY_MAX_TOKENS=50000
```

The normal request keeps the same prompt, model, temperature, and reasoning
policy, but its output allowance is sized before sending. Tarjomeh combines an
answer estimate, a conservative first-chunk reasoning reserve, model/operation
history, and a 1.25 uncertainty margin. Predictive calls receive at least the
configured 50,000-token allowance when context capacity permits. Successful
short calls never reduce that floor or an operation's learned high-water mark.

Configure the adaptive guardrails in `config.toml`:

```toml
[llm.recovery]
predictive_first_attempt = true
predictive_min_tokens = 50000
bootstrap_reasoning_tokens = 24000
adaptive_max_tokens = 65536
context_window_tokens = 131072
context_safety_tokens = 2048
history_window = 20
```

After a length failure, Attempt 2 recalculates from that request's direct usage
evidence and must request a 50% larger allowance when context capacity permits.
That lower bound becomes the operation's monotonic high-water mark for later
calls. Proper-noun extraction, memory summaries, web-term detection, research,
translation, critique, and refinement maintain separate histories.
Provider reasoning counters are normalized when a gateway reports them in a
different token scale. A configured translator fallback adds one final bounded
model rung. Repeated truncation activates validated paragraph recovery, then
sentence-group recovery only when necessary; recovered parts never receive the
complete source chunk as context.

Recommended critic recovery sequence:

1. Predictively sized configured critic request
2. Same critic with low reasoning after a length failure
3. Optional critic-specific fallback model, or a no-reasoning recovery when no
   fallback is configured
4. Final no-reasoning attempt within the adaptive context and output ceiling

No-reasoning mode is limited to recovery and compact JSON repair. It does not
replace the normal translator, critic, or refiner request.

If all critic attempts fail, a valid translation is retained and marked for
human review. An initial translation failure remains strict and prevents
incomplete export.

### Web research settings

```dotenv
TAVILY_API_KEY=
BRAVE_SEARCH_API_KEY=
GOOGLE_API_KEY=
GOOGLE_CX=
```

Tavily is preferred when configured. Search findings and Phase 7 terminology
are suggestions only and cannot silently overwrite curated glossary entries.

### Web UI security

```dotenv
UI_SECRET_TOKEN=replace-with-a-long-random-value
FLASK_SECRET_KEY=replace-with-another-long-random-value
```

Open:

```text
http://SERVER_IP:8080/?token=YOUR_UI_SECRET_TOKEN
```

Use HTTPS through a reverse proxy for an internet-facing production server.
Do not share URLs containing the token. Tarjomeh redacts token-like query
parameters from new request logs, but URLs may still exist in browser history.

## Provider Examples

### Direct OpenRouter for both models

```dotenv
TRANSLATOR_API_BASE=https://openrouter.ai/api/v1
TRANSLATOR_API_KEY=sk-or-replace-me
TRANSLATOR_MODEL=your-translator-model

CRITIC_ENABLED=true
CRITIC_API_BASE=https://openrouter.ai/api/v1
CRITIC_API_KEY=sk-or-replace-me
CRITIC_MODEL=your-independent-critic-model
```

### 9router for both models

```dotenv
TRANSLATOR_API_BASE=http://172.17.0.1:20128/v1
TRANSLATOR_API_KEY=replace-with-9router-api-key
TRANSLATOR_MODEL=translator-combo

CRITIC_ENABLED=true
CRITIC_API_BASE=http://172.17.0.1:20128/v1
CRITIC_API_KEY=replace-with-9router-api-key
CRITIC_MODEL=critic-combo
CRITIC_RECOVERY_MODEL=critic-fallback-combo
```

### Mixed routing

Translator through 9router, critic directly through OpenRouter:

```dotenv
TRANSLATOR_API_BASE=http://172.17.0.1:20128/v1
TRANSLATOR_API_KEY=replace-with-9router-api-key
TRANSLATOR_MODEL=translator-combo

CRITIC_ENABLED=true
CRITIC_API_BASE=https://openrouter.ai/api/v1
CRITIC_API_KEY=sk-or-replace-me
CRITIC_MODEL=your-independent-critic-model
```

The reverse arrangement also works.

## Multiple API Keys and Rotation

The flat `.env` variables configure one translator key and one critic key.
For multiple direct OpenRouter keys, leave `TRANSLATOR_API_KEY` and
`CRITIC_API_KEY` empty and reference named environment variables in
`config.toml`.

`.env`:

```dotenv
OPENROUTER_TRANSLATOR_KEY_1=sk-or-first
OPENROUTER_TRANSLATOR_KEY_2=sk-or-second
OPENROUTER_CRITIC_KEY_1=sk-or-critic-first
OPENROUTER_CRITIC_KEY_2=sk-or-critic-second
```

`config.toml`:

```toml
[llm.openrouter]
api_keys = [
  "${OPENROUTER_TRANSLATOR_KEY_1}",
  "${OPENROUTER_TRANSLATOR_KEY_2}",
]

[llm.critic]
enabled = true
model = "your-independent-critic-model"
api_keys = [
  "${OPENROUTER_CRITIC_KEY_1}",
  "${OPENROUTER_CRITIC_KEY_2}",
]
```

Keys rotate round-robin across calls. Retried OpenRouter-compatible requests
also advance to the next configured key. A non-empty `TRANSLATOR_API_KEY` or
`CRITIC_API_KEY` flat override replaces the corresponding TOML list with that
single key.

9router provider/account fallback is configured in the 9router dashboard.
Tarjomeh sends the combo name as the model and treats the combo as one endpoint.

## Installing 9router on a VPS

9router is optional. Tarjomeh does not manage its container.

The official Docker layout persists data under `/app/data`:

```bash
mkdir -p /root/.9router

docker pull decolua/9router:latest

docker run -d \
  --name 9router \
  --restart unless-stopped \
  -p 20128:20128 \
  -v /root/.9router:/app/data \
  -e DATA_DIR=/app/data \
  decolua/9router:latest
```

Open `http://SERVER_IP:20128`, secure the dashboard, connect providers, create
translator/critic combos, and generate or copy the API key accepted by its
`/v1` endpoint.

Official documentation:

- <https://github.com/decolua/9router>
- <https://github.com/decolua/9router/blob/master/DOCKER.md>

Check the service:

```bash
docker ps --filter name=9router
docker logs --tail 100 9router
curl -H "Authorization: Bearer YOUR_9ROUTER_KEY" \
  http://127.0.0.1:20128/v1/models
```

Find the Docker-host gateway visible to Tarjomeh:

```bash
docker inspect translate_tarjomeh_1 \
  --format '{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}'
```

Use the resulting address in `TRANSLATOR_API_BASE` and/or `CRITIC_API_BASE`.
For example: `http://172.17.0.1:20128/v1`.

Port `20128` should not be left publicly accessible without authentication and
network restrictions. Tarjomeh only needs internal access to it.

Tarjomeh v8.7 accepts both ordinary JSON and OpenAI-compatible SSE
`chat.completion.chunk` responses from gateways such as 9router.

### Updating the standalone 9router container

Update 9router separately from Tarjomeh. Confirm its persistent mount first:

```bash
docker inspect 9router --format \
  '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}'
```

The expected host mount is `/root/.9router -> /app/data`. Then:

```bash
tar -C /root -czf "/root/9router-backup-$(date +%Y%m%d-%H%M%S).tar.gz" .9router

OLD_IMAGE=$(docker inspect -f '{{.Image}}' 9router)
docker tag "$OLD_IMAGE" 9router:rollback-before-update
docker pull decolua/9router:latest

docker rm -f 9router
docker run -d \
  --name 9router \
  --restart unless-stopped \
  -p 20128:20128 \
  -v /root/.9router:/app/data \
  -e DATA_DIR=/app/data \
  decolua/9router:latest

docker ps --filter name=9router
docker logs --tail 100 9router
```

This procedure does not touch Tarjomeh. Keep the data backup until provider
connections, combos, and the `/v1/models` endpoint have been verified.

## First VPS Installation

The following uses the Compose v1 command available on older Ubuntu servers.

```bash
apt-get update
apt-get install -y git docker.io docker-compose
systemctl enable --now docker

git clone https://gitlab.com/personal-group4462928/translate.git /opt/translate
cd /opt/translate

cp .env.example .env
cp config.example.toml config.toml
chmod 600 .env
nano .env

mkdir -p jobs output glossary

docker-compose build tarjomeh
docker-compose up -d --no-deps tarjomeh
```

Verify:

```bash
git rev-parse --short HEAD
git tag --points-at HEAD
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
curl http://127.0.0.1:8080/api/health
docker logs --tail 100 translate_tarjomeh_1
```

Persistent host data:

| Host path | Container path | Purpose |
|---|---|---|
| `/opt/translate/jobs` | `/app/jobs` | uploads, outputs, SQLite jobs |
| `/opt/translate/glossary` | `/app/glossary` | curated and approved terms |
| `/opt/translate/output` | `/app/output` | explicit CLI output |
| `/opt/translate/config.toml` | `/app/config.toml` | read-only app configuration |
| `/opt/translate/.env` | Compose environment | secrets and endpoints |
| `/root/.9router` | `/app/data` in 9router | 9router database/configuration |

Back up `jobs`, `glossary`, `.env`, `config.toml`, and `/root/.9router`.

## Safely Updating Tarjomeh on a Small VPS

Do not use `docker-compose down`, `docker system prune -a`, or
`docker volume prune`. Do not remove or recreate 9router while updating
Tarjomeh.

```bash
cd /opt/translate

# Inspect first. Untracked config.toml is expected on older installations.
git status --short
git checkout main
git fetch origin main --tags
git pull --ff-only origin main

# Reclaim only safe build/dangling-image space.
docker builder prune -f
docker image prune -f

# Build before removing the currently working app container.
docker-compose build tarjomeh

# Compose 1.29 may raise KeyError: ContainerConfig during recreation.
# Remove only Tarjomeh, never 9router.
docker rm -f translate_tarjomeh_1 2>/dev/null || true
docker-compose up -d --no-deps tarjomeh

docker builder prune -f
docker image prune -f

git rev-parse --short HEAD
git tag --points-at HEAD
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
df -h /
```

The Git update does not overwrite `.env`, jobs, outputs, or glossary data.
Do not copy `config.example.toml` over an existing `config.toml` during an
update. Review and merge new options deliberately.

## Applying `.env` or `config.toml` Changes on the VPS

A rebuild is not required for configuration-only changes:

```bash
cd /opt/translate
nano .env

docker rm -f translate_tarjomeh_1
docker-compose up -d --no-deps tarjomeh

docker ps --filter name=translate_tarjomeh_1
docker logs --tail 100 translate_tarjomeh_1
```

This recreates only Tarjomeh and leaves 9router unchanged.

## Running from the Web UI

1. Open `http://SERVER_IP:8080/?token=YOUR_UI_SECRET_TOKEN`.
2. Upload a supported document.
3. Select mode, output format, and document layout.
4. Select **Detect chapters** when you need chapter-level control.
5. Translate the whole document, selected chapters, pause after one chapter,
   or pause after every chapter.
6. Review advanced terminology, research, and QA options.
7. Start the job and keep the job ID.
8. Use History to monitor, pause/resume, review chunks, download output, or
   download the QA report.

An intentional chapter checkpoint exports a downloadable preview containing
every completed chapter through that boundary. The job remains paused and its
four-layer memory is retained; Resume starts with the next untranslated
chapter. A selected-chapter job is complete when its selected scope finishes.
Chapter detection is strongest for PDF bookmarks, EPUB structure, DOCX
Heading 1 styles, Markdown `#` headings, and explicit `Chapter N` TXT lines.
Review the detected list before starting because unstructured or scanned PDFs
may appear as a single chapter.

Recommended production academic settings:

- Mode: Academic
- Output: DOCX
- Layout: Persian only, unless bilingual output is required
- Critique threshold: `9`
- Refinement attempts: `2`
- Integrity gate: enabled
- Glossary compliance and guarded auto-correction: enabled
- Back-translation sample: `15-20%`
- Book research: enabled when external search is configured
- Critic fallback: an independent reliable model/combo when available

A threshold of `10` is useful for stress testing but can trigger unnecessary
stylistic reversals.

Generated Web UI files are normally stored under:

```text
/opt/translate/jobs/uploads/JOB_ID_translated.docx
```

The recommended way to retrieve them is the History download button.

## Running from the CLI

Local installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

cp .env.example .env
cp config.example.toml config.toml
nano .env

tarjomeh translate book.pdf \
  --mode academic \
  --output-format docx \
  --bilingual target_only \
  --book-research \
  --output output/book-translated.docx
```

Inside the Docker container:

```bash
docker cp book.pdf translate_tarjomeh_1:/tmp/book.pdf

docker exec -it translate_tarjomeh_1 \
  tarjomeh translate /tmp/book.pdf \
  --mode academic \
  --output-format docx \
  --bilingual target_only \
  --book-research \
  --output /app/output/book-translated.docx
```

Job management:

```bash
tarjomeh jobs list
tarjomeh jobs status JOB_ID
tarjomeh jobs resume JOB_ID
tarjomeh jobs export JOB_ID --format docx --output output/re-export.docx
tarjomeh jobs cleanup JOB_ID
```

Glossary validation:

```bash
tarjomeh glossary validate glossary/academic_political_theory.csv
```

## Translation Modes

| Behavior | Fast | Quality | Academic |
|---|---:|---:|---:|
| Default chunk tokens | 3000 | 1500 | 1000 |
| Critique | Off | On | On |
| Refinement attempts | 0 | 1 | 2 |
| Back-translation sample | 0% | 5% | 20% |
| Web context | Off | On | On |
| Parallel execution | Configurable | Configurable | Forced sequential |
| Four-layer memory | Available | Available | Recommended |

Academic mode is sequential so later chunks do not advance without the memory
of an earlier successful chunk.

## Phase 6: First-Occurrence Originals

Configure in the UI, CLI, or `config.toml`:

```toml
[output]
term_notes = "inline" # inline | footnote | endnote | both
```

- `inline`: Persian rendering followed by the English original once
- `footnote`: linked note behavior for supported document formats
- `endnote`: collect first-occurrence originals as notes
- `both`: inline original plus note

DOCX, EPUB, and Markdown have note support. Unsupported output formats fall
back conservatively to inline originals.

## Phase 7: Pre-Translation Research

Enable:

```toml
[translation]
enable_book_research = true

[web_search]
provider = "auto"
phase7_max_queries = 8
max_queries_per_chunk = 2
max_queries_per_book = 100
```

The one-time seed pass:

1. identifies the book, author, domain, and conceptual vocabulary
2. creates bounded external search queries
3. summarizes book/domain context
4. proposes reviewable glossary candidates
5. stores sources, status, and evidence in the job artifact/QA report

Its normal synthesis also follows the 12K/24K whole-request policy. If both
attempts truncate, evidence is analyzed in bounded batches and consolidated.
The job is marked `degraded` when this fallback was needed; translation still
continues with the source-backed context and review-only suggestions recovered.

Research suggestions are marked automatic and never overwrite curated user
terms. Approve useful suggestions through the glossary UI.

## Glossaries

Base columns:

```csv
source,target,tgt_lng,context,domain
```

Optional context-aware columns:

```csv
source,target,tgt_lng,context,domain,sense,author
```

Example:

```csv
capital,sarmayeh,fa,Marx economic category,political economy,economic,Marx
capital,sarmayeh-ye farhangi,fa,Bourdieu cultural capital,sociology,cultural,Bourdieu
```

The example uses Latin transliteration only to keep this README encoding-safe;
real glossary targets should be Persian Unicode.

Load multiple glossaries:

```toml
[glossary]
path = "glossary/academic_political_theory.csv"
paths = [
  "glossary/analytic_philosophy.csv",
  "glossary/islamic_philosophy.csv",
]
```

The primary glossary has highest precedence. Auto-extracted/research terms do
not overwrite curated entries.

## Review and Quality Regression

The Web UI provides:

- per-chunk source and translation review
- persisted MQM issues, per-issue decisions, and integrity evidence
- manual edits and chunk retranslation
- `needs_review` filtering
- glossary approval/edit/download
- QA report download
- output re-export without retranslation
- blind comparison between completed jobs

CLI comparison does not make LLM calls:

```bash
tarjomeh eval BASELINE_JOB_ID CANDIDATE_JOB_ID
tarjomeh eval BASELINE_JOB_ID CANDIDATE_JOB_ID \
  --format json \
  --output evaluation.json
tarjomeh eval BASELINE_JOB_ID CANDIDATE_JOB_ID --fail-on-regression
```

## Monitoring and Troubleshooting

### Health and logs

```bash
curl http://127.0.0.1:8080/api/health
docker logs -f --tail 200 translate_tarjomeh_1
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
df -h /
```

### Confirm the loaded release

```bash
cd /opt/translate
git rev-parse --short HEAD
git tag --points-at HEAD
```

### Test 9router from the host

```bash
curl -H "Authorization: Bearer YOUR_9ROUTER_KEY" \
  http://127.0.0.1:20128/v1/models
```

### Test 9router from Tarjomeh

```bash
docker exec -i translate_tarjomeh_1 python - <<'PY'
import os
import httpx

base = os.getenv("TRANSLATOR_API_BASE", "http://172.17.0.1:20128/v1").rstrip("/")
key = os.getenv("TRANSLATOR_API_KEY", "")
headers = {"Authorization": f"Bearer {key}"} if key else {}
response = httpx.get(f"{base}/models", headers=headers, timeout=15)
print("endpoint:", base)
print("status:", response.status_code)
PY
```

Use the actual Docker gateway. The command reads the configured key without
printing it.

### `KeyError: 'ContainerConfig'`

This is a Docker Compose 1.29 recreation problem. Build first, remove only the
Tarjomeh container, and recreate only its service:

```bash
docker-compose build tarjomeh
docker rm -f translate_tarjomeh_1
docker-compose up -d --no-deps tarjomeh
```

### `finish_reason=length`

The completion reached its provider output limit. Tarjomeh records a prompt hash
and token evidence, then retries with a larger evidence-based bounded budget.
Critic recovery can use a separate fallback. Partial translations are never
accepted as complete.

### `malformed_response` containing `data:`

Confirm the VPS is on v8.7 or later. v8.7 added compatibility for
OpenAI-style SSE streams returned by OpenAI-compatible gateways.

### Job paused on a translation failure

This is intentional in sequential academic mode. Resume after the provider is
healthy. Later chunks are not translated without the missing chunk's memory.

### `needs_review`

The output is present, but a QA disagreement, rejected integrity edit,
unavailable QA provider, or unresolved threshold requires human review. Inspect
the chunk and QA report in the Web UI.

### Disk pressure

Safe cleanup:

```bash
docker builder prune -f
docker image prune -f
df -h /
```

Do not run volume prune or broad destructive cleanup on a production server.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python -m pytest -q
python -m compileall -q src tests
```

The v9.5 suite contains 193 passing tests.

## Release History

- v9.5: grounded sentence-level QA, conservative Persian orthography, exact
  first-occurrence anchoring, deterministic citation cleanup, curated-glossary
  critic guards, and a non-blocking high-risk concept review queue
- v9.4: 50K monotonic first-attempt floor and operation-isolated budgeting
- v9.3: predictive first-attempt budgets, evidence-based recovery, prompt diagnostics
- v9.2: validated adaptive translation recovery and assembly integrity
- v9.0: per-issue MQM refinement decisions
- v8.7.1: complete installation, environment, VPS, and 9router documentation
- v8.7: OpenAI-compatible SSE response assembly while retaining normal JSON
- v8.6: bounded critic recovery, QA failure containment, sequential continuity
- v8.5: integrity/QA reliability refinements
- v8.0-v8.4: regression harness, post-edit integrity, stabilization
- v7.x: book research and reliable adaptive web search
- v6.x: first-occurrence term notes
- v4.x: review UI, QA evidence, glossary workflow, re-export
- v3.x: context-aware namespaced glossary

Use release tags for inspection:

```bash
git tag --sort=-version:refname
git show v8.7
```

## License

Tarjomeh is licensed under AGPL-3.0. See [LICENSE](LICENSE).

The bundled Vazirmatn font is licensed under the SIL Open Font License.
