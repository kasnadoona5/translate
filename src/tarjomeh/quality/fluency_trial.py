"""Opt-in, read-only prompt experiment. Never activates a production policy.

Run with ``python -m tarjomeh.quality.fluency_trial JOB --out DIRECTORY``.
The default only lists saved chunk hashes. ``--execute`` permits LLM calls;
``--with-refinement`` additionally tests one refiner proposal on a frozen copy.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import difflib
import hashlib
import json
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import LLMClient
from tarjomeh.core.pipeline import _candidate_regression_details, audit_translation_language
from tarjomeh.quality.critique import TranslationCritique
from tarjomeh.quality.integrity import PostEditIntegrityGate
from tarjomeh.quality.refiner import TranslationRefiner

TRIAL_GUIDANCE = (
    "\n\nOFFLINE EXPERIMENT ONLY: assess understandable, fluent academic Persian "
    "without simplifying the argument. Preserve the source head and every dependent, "
    "antecedent, qualifier, list scope, count, negation, temporal relation and modality. "
    "An opaque nominal stack or dangling complement is actionable only when exact "
    "source and Persian spans demonstrate the defect. Name the dependency in the "
    "rationale; do not mistake correct conceptual complexity for an error. A rewrite "
    "must include the complete affected relation, not just a nicer fragment. "
    "Use the existing response schema and allow refiner rejection."
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TrialClient:
    """Append experimental guidance only in this isolated process."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    async def chat(self, prompt: str) -> str:
        return await self.client.chat(prompt + TRIAL_GUIDANCE)


def critic_configuration(config: TarjomehConfig) -> TarjomehConfig:
    selected = copy.deepcopy(config)
    critic = config.llm.critic
    if critic.is_active:
        if critic.provider:
            selected.llm.provider = critic.provider
        if critic.model:
            selected.llm.model = selected.llm.ollama.model = critic.model
        if critic.api_base.strip():
            selected.llm.openrouter.api_base = critic.api_base.strip()
        if any(key.strip() for key in critic.api_keys):
            selected.llm.openrouter.api_keys = list(critic.api_keys)
        selected.llm.temperature = critic.temperature
        selected.llm.recovery.model = critic.recovery_model.strip()
        selected.llm.recovery.max_attempts = critic.recovery_max_attempts
        selected.llm.recovery.max_tokens = max(selected.llm.max_tokens, critic.recovery_max_tokens)
        selected.llm.recovery.expanded_final_attempt = True
    return selected


def saved_pairs(database_path: Path, job_id: str, indices: set[int], limit: int):
    with closing(
        sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    ) as database:
        database.row_factory = sqlite3.Row
        job = database.execute("SELECT config FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not job:
            raise ValueError("Saved job not found")
        if database.execute(
            "SELECT 1 FROM jobs WHERE status IN "
            "('pending','running','processing','pausing','resuming') LIMIT 1"
        ).fetchone():
            raise ValueError("Wait until active translation jobs have paused or finished")
        has_workers = database.execute(
            "SELECT 1 FROM sqlite_master WHERE name='job_workers'"
        ).fetchone()
        if (
            has_workers
            and database.execute(
                "SELECT 1 FROM job_workers WHERE state IN ('active','pausing') LIMIT 1"
            ).fetchone()
        ):
            raise ValueError("Active worker lease; do not compete with production")
        rows = database.execute(
            "SELECT chunk_index,text,translation,metadata FROM chunks WHERE job_id=? "
            "AND status IN ('completed','needs_review') ORDER BY chunk_index",
            (job_id,),
        ).fetchall()
        pairs = []
        for row in rows:
            if indices and row["chunk_index"] not in indices:
                continue
            metadata = json.loads(row["metadata"] or "{}")
            roles = set(metadata.get("structural_roles", []) or [])
            if not roles or roles - {"body", "body_prose", "prose"}:
                continue
            if not row["text"] or not row["translation"]:
                continue
            pairs.append(
                {
                    "chunk_index": row["chunk_index"],
                    "kind": "saved_text",
                    "source": row["text"],
                    "target": row["translation"],
                    "source_sha256": digest(row["text"]),
                    "target_sha256": digest(row["translation"]),
                }
            )
            if len(pairs) >= limit:
                break
        return json.loads(job["config"]), pairs


def controls() -> list[dict[str, Any]]:
    examples = [
        (
            "known_good",
            "The sensor may record two signals.",
            (
                "\u0645\u0645\u06a9\u0646 \u0627\u0633\u062a \u062d\u0633\u06af\u0631 "
                "\u062f\u0648 \u0633\u06cc\u06af\u0646\u0627\u0644 \u062b\u0628\u062a "
                "\u06a9\u0646\u062f."
            ),
        ),
        (
            "known_good",
            "This result does not establish causation.",
            (
                "\u0627\u06cc\u0646 \u0646\u062a\u06cc\u062c\u0647 \u0639\u0644\u06cc"
                "\u062a \u0631\u0627 \u0627\u062b\u0628\u0627\u062a \u0646\u0645\u06cc"
                "\u200c\u06a9\u0646\u062f."
            ),
        ),
        (
            "synthetic_sensitivity",
            "The sensor records two signals.",
            (
                "\u062d\u0633\u06af\u0631 \u0633\u0647 \u0633\u06cc\u06af\u0646\u0627"
                "\u0644 \u062b\u0628\u062a \u0645\u06cc\u200c\u06a9\u0646\u062f."
            ),
        ),
    ]
    return [
        {
            "chunk_index": f"control:{i}",
            "kind": kind,
            "source": source,
            "target": target,
            "source_sha256": digest(source),
            "target_sha256": digest(target),
        }
        for i, (kind, source, target) in enumerate(examples)
    ]


async def replay(config: TarjomehConfig, pairs: list[dict[str, Any]], *, with_refinement=False):
    critic_llm = LLMClient(critic_configuration(config))
    translator_llm = LLMClient(config) if with_refinement else None
    attempts: list[dict[str, Any]] = []
    critic_llm.set_attempt_observer(lambda event: attempts.append(dict(event)))
    if translator_llm:
        translator_llm.set_attempt_observer(lambda event: attempts.append(dict(event)))
    ordinary = TranslationCritique(critic_llm, max_parse_retries=config.translation.qa_json_retries)
    experimental = TranslationCritique(
        TrialClient(critic_llm), max_parse_retries=config.translation.qa_json_retries
    )
    records = []
    try:
        for pair in pairs:
            source, target = pair["source"], pair["target"]
            record = {
                **pair,
                "context_sha256": digest(""),
                "context_status": (
                    "same reconstructed empty review context; not a live-prompt replay"
                ),
                "runs": [],
                "adjudication": "PENDING human source comparison",
            }
            results = []
            for name, reviewer in (
                ("ordinary_1", ordinary),
                ("ordinary_2", ordinary),
                ("trial_guidance", experimental),
            ):
                start, offset = time.monotonic(), len(attempts)
                try:
                    result = await reviewer.critique(source, target)
                    outcome = result.to_dict()
                except Exception as exc:
                    result = None
                    outcome = {"valid": False, "error_type": type(exc).__name__}
                results.append(result)
                record["runs"].append(
                    {
                        "operation": name,
                        "result": outcome,
                        "seconds": round(time.monotonic() - start, 3),
                        "attempts": attempts[offset:],
                    }
                )
            trial = results[-1]
            if (
                translator_llm
                and trial
                and trial.valid
                and trial.coverage_complete
                and trial.issue_details
            ):
                refiner = TranslationRefiner(
                    TrialClient(translator_llm),
                    max_iterations=1,
                    max_parse_retries=config.translation.qa_json_retries,
                )
                start, offset = time.monotonic(), len(attempts)
                refined = await refiner.refine_with_decision(source, target, trial)
                record["refiner_trial"] = {
                    **asdict(refined),
                    "seconds": round(time.monotonic() - start, 3),
                    "attempts": attempts[offset:],
                }
                if refined.valid and refined.translation != target:
                    candidate = refined.translation
                    start, offset = time.monotonic(), len(attempts)
                    renewed = await ordinary.critique(source, candidate)
                    changed = [
                        candidate[j1:j2]
                        for tag, _, _, j1, j2 in difflib.SequenceMatcher(
                            None, target, candidate, autojunk=False
                        ).get_opcodes()
                        if tag != "equal"
                    ]
                    record["renewed_checks"] = {
                        "critique": renewed.to_dict(),
                        "attempts": attempts[offset:],
                        "seconds": round(time.monotonic() - start, 3),
                        "candidate_sha256": digest(candidate),
                        "integrity": asdict(
                            PostEditIntegrityGate().evaluate(source, candidate, previous=target)
                        ),
                        "language": audit_translation_language(source, candidate),
                        "edit_attribution_blockers": _candidate_regression_details(
                            renewed,
                            trial,
                            changed,
                            source_text=source,
                            previous_text=target,
                            candidate_text=candidate,
                        ),
                        "limits": (
                            "glossary, typography, canonical identity, DOCX and memory gates "
                            "not replayed; never production-admitted"
                        ),
                    }
            successful = [
                item for run in record["runs"] for item in run["attempts"] if item.get("success")
            ]
            served = {item.get("response_model") for item in successful}
            record["responses_valid"] = all(
                result and result.valid and result.coverage_complete for result in results
            )
            record["served_model_status"] = (
                "matched"
                if successful and len(served) == 1 and None not in served and "" not in served
                else "inconclusive_rerun_same_chunk"
            )
            records.append(record)
    finally:
        critic_llm.close()
        if translator_llm:
            translator_llm.close()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("--db", type=Path, default=Path("/app/jobs/jobs.db"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--indices", type=int, nargs="*", default=[])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--with-refinement", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 16:
        parser.error("limit must be 1 through 16")
    raw_config, pairs = saved_pairs(args.db, args.job_id, set(args.indices), args.limit)
    if not pairs:
        parser.error("No selected, completed pure-body chunks; no LLM requests made")
    if not args.execute:
        print(
            json.dumps(
                {
                    "job_id": args.job_id,
                    "chunks": [
                        {
                            key: item[key]
                            for key in ("chunk_index", "source_sha256", "target_sha256")
                        }
                        for item in pairs
                    ],
                    "llm_calls": 0,
                    "database_writes": 0,
                    "production_changes": 0,
                },
                indent=2,
            )
        )
        return
    if args.out.exists():
        parser.error("Use a new output directory; refusing to overwrite trial evidence")
    config = TarjomehConfig.from_dict(raw_config, credential_source=TarjomehConfig.load())
    if str(config.translation.mode).casefold() != "academic":
        parser.error("This fluency trial is for academic jobs only")
    records = asyncio.run(replay(config, pairs + controls(), with_refinement=args.with_refinement))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "fluency_trial.json").write_text(
        json.dumps(
            {
                "job_id": args.job_id,
                "records": records,
                "database_writes": 0,
                "production_prompt_changes": 0,
                "trial_guidance_sha256": digest(TRIAL_GUIDANCE),
                "verdict": "PENDING_HUMAN_REVIEW",
                "synthetic_findings_earn_credit": False,
                "precision_definition": "true / (true + false + uncertain)",
                "baseline": "both ordinary passes and all existing reports on this exact target",
                "gates": (
                    "controls silent; real incremental benefit; no meaning-changing proposal; "
                    "matched models; user approval"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Trial saved: {args.out}. Production prompts and call budget unchanged.")


if __name__ == "__main__":
    main()
