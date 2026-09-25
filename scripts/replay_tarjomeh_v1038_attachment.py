"""Read-only, opt-in attachment-review trial for saved Tarjomeh chunks.

The script never edits the job database or changes production configuration.
Its output is evidence for human adjudication, not an automatic release gate.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import LLMClient
from tarjomeh.core.pipeline import _is_front_matter
from tarjomeh.quality.critique import TranslationCritique


def frozen_chunks(
    db_path: Path, job_id: str, limit: int, indices: set[int] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    database = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    database.row_factory = sqlite3.Row
    try:
        job = database.execute("SELECT config FROM jobs WHERE id=?", (job_id,)).fetchone()
        if job is None:
            raise ValueError(f"Job not found: {job_id}")
        rows = database.execute(
            "SELECT chunk_index, text, translation, metadata FROM chunks "
            "WHERE job_id=? AND status IN ('completed','needs_review') "
            "AND text IS NOT NULL AND translation IS NOT NULL "
            "ORDER BY chunk_index", (job_id,),
        ).fetchall()
        chunks = []
        for row in rows:
            chunk = dict(row)
            if not chunk["text"].strip() or not chunk["translation"].strip():
                continue
            metadata = json.loads(chunk.get("metadata") or "{}")
            if indices is not None and chunk["chunk_index"] not in indices:
                continue
            if _is_front_matter(SimpleNamespace(
                chapter_title=metadata.get("chapter_title", ""), metadata=metadata
            )):
                continue
            chunks.append(chunk)
            if len(chunks) >= limit:
                break
        for chunk in chunks:
            chunk["text_hash"] = hashlib.sha256(chunk["text"].encode()).hexdigest()
            chunk["translation_hash"] = hashlib.sha256(chunk["translation"].encode()).hexdigest()
        return json.loads(job["config"]), chunks
    finally:
        database.close()


def critic_client(config: TarjomehConfig) -> LLMClient:
    judge = config.llm.critic
    if not judge.is_active:
        return LLMClient(config)
    selected = copy.deepcopy(config)
    if judge.provider:
        selected.llm.provider = judge.provider
    if judge.model:
        selected.llm.model = judge.model
        selected.llm.ollama.model = judge.model
    if any(key.strip() for key in judge.api_keys):
        selected.llm.openrouter.api_keys = list(judge.api_keys)
    if judge.api_base.strip():
        selected.llm.openrouter.api_base = judge.api_base.strip()
    selected.llm.temperature = judge.temperature
    selected.llm.recovery.model = judge.recovery_model.strip()
    selected.llm.recovery.max_attempts = judge.recovery_max_attempts
    selected.llm.recovery.max_tokens = max(
        selected.llm.max_tokens, judge.recovery_max_tokens
    )
    selected.llm.recovery.expanded_final_attempt = True
    return LLMClient(selected)


async def run_trial(config: TarjomehConfig, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    llm = critic_client(config)
    attempts: list[dict[str, Any]] = []
    llm.set_attempt_observer(lambda event: attempts.append(dict(event)))
    reviewer = TranslationCritique(llm, max_parse_retries=config.translation.qa_json_retries)
    records: list[dict[str, Any]] = []
    try:
        for chunk in chunks:
            source = chunk["text"]
            target = chunk["translation"]
            record: dict[str, Any] = {
                "chunk_index": chunk["chunk_index"],
                "kind": chunk.get("kind", "live_text"),
                "source_hash": chunk["text_hash"],
                "target_hash": chunk["translation_hash"],
                "context": "frozen source and final target; terminology and original prompt context unavailable",
                "runs": [],
            }
            for operation in ("ordinary_1", "ordinary_2", "focused"):
                start = time.monotonic()
                first_attempt = len(attempts)
                try:
                    if operation == "focused":
                        result = await reviewer.review_attachment(source, target)
                        details = result.issues
                    else:
                        result = await reviewer.critique(source, target)
                        details = result.issue_details
                    outcome = {
                        "valid": result.valid,
                        "errors": result.validation_errors,
                        "findings": details,
                        "reported_count": getattr(result, "reported_issue_count", len(details)),
                        "raw_response": result.raw_response,
                    }
                except Exception as exc:
                    outcome = {"valid": False, "errors": [f"{type(exc).__name__}: {exc}"], "findings": []}
                outcome.update({
                    "operation": operation,
                    "elapsed_seconds": round(time.monotonic() - start, 3),
                    "attempts": attempts[first_attempt:],
                })
                record["runs"].append(outcome)
            records.append(record)
    finally:
        llm.close()
    return records


def report_text(job_id: str, records: list[dict[str, Any]]) -> str:
    lines = [
        f"# Attachment reviewer trial: {job_id}", "",
        "Production reviewer: OFF. Human adjudication and user approval required.",
        "Original prompt context was not persisted in full; all three trial calls use the same frozen reconstructed context.",
        "Compare focused findings against both ordinary calls AND existing QA, readability, language and structure reports before crediting a catch.",
        "Synthetic sensitivity examples earn no true-positive credit. Uncertain findings count against precision.",
        "Go/no-go: at least two real unique source-grounded catches, precision >= 80% (uncertain in denominator), known-good controls silent, user approval.",
        "A run with zero findings, served-model mismatch, missing controls, or invalid responses cannot pass.",
        "Any new focused finding that would invoke the existing final repair adds a refiner call and re-run checks; count these during adjudication.", "",
        "| Chunk | Kind | Pass | Seconds | Valid | Served models | Findings |",
        "| --- | --- | --- | ---: | --- | --- | ---: |",
    ]
    for record in records:
        for run in record["runs"]:
            models = sorted({
                str(a.get("response_model") or a.get("served_model"))
                for a in run["attempts"] if a.get("response_model") or a.get("served_model")
            })
            lines.append(
                f"| {record['chunk_index']} | {record['kind']} | {run['operation']} | {run['elapsed_seconds']:.1f} "
                f"| {run['valid']} | {', '.join(models) or 'unreported'} | {len(run['findings'])} |"
            )
    total_attempts = sum(
        len(run["attempts"]) for record in records for run in record["runs"]
    )
    total_tokens = sum(
        int(attempt.get("total_tokens") or 0)
        for record in records for run in record["runs"]
        for attempt in run["attempts"]
    )
    total_seconds = sum(
        run["elapsed_seconds"] for record in records for run in record["runs"]
    )
    control_count = sum(record["kind"] == "known_good" for record in records)
    synthetic_count = sum(
        record["kind"] == "synthetic_positive" for record in records
    )
    mismatches = []
    for record in records:
        served = {
            str(attempt.get("response_model"))
            for run in record["runs"] for attempt in run["attempts"]
            if attempt.get("response_model") and attempt.get("visible_output_present")
        }
        if len(served) != 1:
            mismatches.append(record["chunk_index"])
    lines.extend([
        "", f"Observed attempts: {total_attempts}; tokens: {total_tokens}; "
        f"wall time: {total_seconds:.1f}s.",
        f"Inconclusive served-model chunks: {mismatches or 'none'}.",
        f"Controls: {control_count}; synthetic sensitivity cases: {synthetic_count}.",
        "Re-run inconclusive chunks with --indices; if the same model cannot serve all three, do not score them.",
    ])
    lines.extend(["", "## Adjudication", ""])
    for record in records:
        focused = record["runs"][-1]
        for item in focused["findings"]:
            lines.extend([
                f"- Chunk {record['chunk_index']}, source `{item.get('source_quote', '')}`, "
                f"target `{item.get('current_persian_quote', '')}`",
                f"  - Head/dependent: `{item.get('source_head', '')}` / "
                f"`{item.get('source_dependent', '')}` -> "
                f"`{item.get('persian_head', '')}` / `{item.get('persian_dependent', '')}`",
                "  - Human verdict (true / false / uncertain): PENDING",
                "  - Already reported by existing checks or either ordinary pass: PENDING",
                "  - Source-meaning and refiner-gate check: PENDING",
            ])
    lines.extend(["", "Default activation remains BLOCKED until adjudication, precision, real unique catches, controls, and cost are approved by the user.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("--db", type=Path, default=Path("/app/jobs/jobs.db"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--indices", type=int, nargs="*", help="Re-run selected database chunk indices")
    parser.add_argument("--cases", type=Path, help="Optional known-good/synthetic fixture JSON")
    parser.add_argument("--execute", action="store_true", help="Permit LLM calls for the frozen trial")
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 16:
        parser.error("--limit must be between 1 and 16")
    raw_config, chunks = frozen_chunks(
        args.db, args.job_id, args.limit,
        set(args.indices) if args.indices else None,
    )
    if args.cases:
        data = json.loads(args.cases.read_text(encoding="utf-8"))
        for number, item in enumerate(data.get("cases", [])):
            if item.get("kind") not in {"known_good", "synthetic_positive"}:
                parser.error("case kind must be known_good or synthetic_positive")
            source = str(item.get("source_text", ""))
            target = str(item.get("translation", ""))
            if not source or not target:
                parser.error("every case requires source_text and translation")
            chunks.append({
                "chunk_index": f"case:{number}:{item.get('id', '')}",
                "kind": item["kind"], "text": source, "translation": target,
                "text_hash": hashlib.sha256(source.encode()).hexdigest(),
                "translation_hash": hashlib.sha256(target.encode()).hexdigest(),
            })
    if not args.execute:
        print(json.dumps({"job_id": args.job_id, "chunks": [
            {key: row[key] for key in ("chunk_index", "text_hash", "translation_hash")}
            for row in chunks
        ], "llm_calls": 0, "database_writes": 0}, indent=2))
        return
    config = TarjomehConfig.from_dict(raw_config, credential_source=TarjomehConfig.load())
    if str(config.translation.mode).casefold() != "academic":
        parser.error("the focused academic attachment trial requires academic mode")
    records = asyncio.run(run_trial(config, chunks))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "attachment_trial.json").write_text(
        json.dumps({"job_id": args.job_id, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out / "attachment_trial.md").write_text(
        report_text(args.job_id, records), encoding="utf-8"
    )
    print(f"Trial written to {args.out}; production reviewer remains OFF")


if __name__ == "__main__":
    main()
