# Tarjomeh (ترجمه)

**Professional Academic Book Translation System — English → Persian**

A Python-based translation tool for translating full-length academic books (political theory, sociology, philosophy) from English into publication-quality Iranian Persian (فارسی).

Licensed under **AGPL-3.0**.

## Features

- **Multi-format support**: PDF, EPUB, DOCX, TXT, SRT, Markdown
- **4-layer memory system**: Proper noun records, bilingual summaries, long-term TF-IDF retrieval, short-term sliding window — inspired by DocMTAgent/DelTA
- **Quality pipeline**: Translation → self-critique → refinement → back-translation QA
- **Persian RTL typography**: ZWNJ normalization, Persian numerals, proper punctuation via `hazm`
- **Bilingual output**: Inline, side-by-side, or target-only modes in PDF/EPUB/DOCX/TXT
- **Academic glossary**: BabelDOC-compatible CSV with 60+ pre-populated political theory terms
- **Web context search**: Per-chunk term disambiguation (Aphra-inspired)
- **Checkpoint/resume**: SQLite-based job tracking, crash-safe
- **Multiple LLM backends**: OpenRouter (Claude, GPT-4, etc.) and Ollama (local/private)
- **3 translation modes**: Fast, Quality, Academic — with full preset matrix
- **Web UI**: Dark theme, RTL-aware, token-based authentication
- **REST API**: Full `/api/*` endpoints for scripting
- **Docker support**: One-command VPS deployment

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/tarjomeh/tarjomeh.git
cd tarjomeh

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install
pip install -e .

# Copy and edit configuration
cp config.example.toml config.toml
cp .env.example .env
# Edit .env with your API keys
```

### CLI Usage

```bash
# Translate a book (academic mode, PDF output)
tarjomeh translate book.pdf --mode academic --output-format pdf

# Fast mode for quick drafts
tarjomeh translate article.epub --mode fast --output-format txt

# Resume interrupted translation
tarjomeh jobs resume <job_id>

# List all jobs
tarjomeh jobs list

# Clean up job artifacts
tarjomeh jobs cleanup <job_id>

# Validate glossary
tarjomeh glossary validate glossary/academic_political_theory.csv

# Start web UI
tarjomeh serve --port 8080
```

### Docker Deployment

```bash
# Copy configuration
cp config.example.toml config.toml
cp .env.example .env
# Edit .env with your API keys

# Deploy
docker compose up -d

# Access via SSH tunnel (recommended for VPS)
ssh -L 8080:localhost:8080 user@your-vps
# Then open http://localhost:8080?token=your-secret-token
```

## Translation Modes

| Setting | Fast | Quality | Academic |
|---|---|---|---|
| Chunk size (tokens) | 3000 | 1500 | 1000 |
| Self-critique | ❌ | ✅ | ✅ |
| Refinement | ❌ | ✅ (1 iter) | ✅ (2 iter) |
| Back-translation QA | ❌ | 5% sample | 20% sample |
| Web context search | ❌ | ✅ | ✅ |
| Parallel workers | Configurable | 1 | 1 (sequential) |

## Glossary Format

BabelDOC-compatible CSV:

```csv
source,target,tgt_lng,context,domain
hegemony,هژمونی,fa,Gramsci's concept,political theory
discourse,گفتمان,fa,Foucault's concept,philosophy
```

## Configuration

See [`config.example.toml`](config.example.toml) for all options.

Key sections:
- `[llm]` — Provider, model, temperature
- `[translation]` — Mode, language, country (Iran/Afghanistan), domain
- `[glossary]` — Glossary path, auto-extraction
- `[memory]` — 4-layer memory settings
- `[persian]` — Typography options (ZWNJ, numerals, punctuation)
- `[notifications]` — Webhook for job completion/error alerts

## Architecture

Built on insights from 8 open-source projects: TranslateBooksWithLLMs, Turjuman, Andrew Ng's translation-agent, DocMTAgent/DelTA, PDFMathTranslate, BabelDOC, epub-translator, and Aphra.

### Pipeline Flow

```
Input → Parse → [OCR] → Chunk → Extract Terms (TOC + Ch.1)
  → Per-chunk: Memory → [Web Context] → Translate
    → [Critique → Refine] → [Glossary Check] → [Back-translate]
    → Update Memory → Checkpoint
  → Assemble → Persian Typography → Export → [Notify]
```

### Key Design Decisions

- **Custom state machine** (default) with optional LangGraph adapter
- **Sequential translation** in academic mode for memory coherence
- **hazm** for Persian tokenization and ZWNJ normalization
- **tiktoken** for accurate LLM token counting
- **Token-based auth** + localhost binding for VPS security

## Optional Dependencies

```bash
# Layout-aware PDF parsing (GPU recommended)
pip install tarjomeh[pdf-layout]

# OCR for scanned PDFs (requires Tesseract)
pip install tarjomeh[ocr]

# LangGraph adapter
pip install tarjomeh[langgraph]

# All optional dependencies
pip install tarjomeh[all]
```

## License

AGPL-3.0 — See [LICENSE](LICENSE) for details.

Compatible with TranslateBooksWithLLMs (AGPL-3.0) and BabelDOC (AGPL-3.0).
