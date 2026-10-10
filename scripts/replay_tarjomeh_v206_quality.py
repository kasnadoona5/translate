"""Opt-in, read-only controlled trial of guidance inside the existing stages.

Default: freeze inputs and print a plan, with no model calls. --execute explicitly
permits paid calls. Production job state, memory and configuration are never written.
This tool does not implement or waive publication/admission gates.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.evidence_audit import model_identity_counts
from tarjomeh.core.llm_client import LLMClient
from tarjomeh.core.pipeline import TranslationPipeline
from tarjomeh.core.prompts import CRITIQUE_PROMPT, REFINE_PROMPT
from tarjomeh.quality.critique import CritiqueResult, TranslationCritique
from tarjomeh.quality.refiner import TranslationRefiner
from tarjomeh.quality.trial import comparable_served_models, experimental_templates, sha256

ATTEMPT_FIELDS = (
    "model", "response_model", "success", "operation", "attempt", "failure_type",
    "completion_tokens", "prompt_tokens", "total_tokens", "duration_seconds", "max_tokens",
)


def read_job(db_path: Path, job_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        db.row_factory = sqlite3.Row
        job = db.execute("SELECT config FROM jobs WHERE id=?", (job_id,)).fetchone()
        if job is None:
            raise ValueError("Job not found")
        rows = db.execute(
            "SELECT chunk_index,text,translation FROM chunks WHERE job_id=? "
            "AND status IN ('completed','needs_review') ORDER BY chunk_index", (job_id,),
        ).fetchall()
        if len(rows) != 16 or any(not r["text"] or not r["translation"] for r in rows):
            raise ValueError("The full trial requires exactly 16 saved, translated chunks")
        cases = [
            {"id": f"chunk:{r['chunk_index']}", "kind": "real_saved_text",
             "source": r["text"], "target": r["translation"]}
            for r in rows
        ]
        return json.loads(job["config"]), cases


def freeze_cases(cases: list[dict[str, Any]], controls: list[dict[str, Any]],
                 context: dict[str, Any]) -> list[dict[str, Any]]:
    if len(cases) != 16 or len(controls) != 2:
        raise ValueError("Need 16 real saved chunks and two property-specific known-good controls")
    if any(c.get("kind") != "real_saved_text" for c in cases):
        raise ValueError("Synthetic cases cannot replace real saved chunks")
    if any(c.get("kind") != "known_good" or not c.get("property") for c in controls):
        raise ValueError("Each control needs kind=known_good and its tested property")
    frozen = []
    ids = set()
    for case in [*cases, *controls]:
        identifier = str(case.get("id") or "")
        source, target = case.get("source"), case.get("target")
        if not identifier or identifier in ids or not isinstance(source, str) or not source.strip():
            raise ValueError("Invalid or duplicate case identity/source")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("Missing target")
        ids.add(identifier)
        supplied = context.get(identifier, {})
        terminology = str(supplied.get("terminology") or "")
        review_context = str(supplied.get("review_context") or "")
        existing = supplied.get("existing_reports", [])
        if not isinstance(existing, list):
            raise ValueError("existing_reports must be a list")
        for report in existing:
            if (not isinstance(report, dict) or report.get("source_sha256") != sha256(source)
                    or report.get("target_sha256") != sha256(target)):
                raise ValueError("Existing report is not bound to this exact source/target")
        frozen.append({
            "id": identifier, "kind": case["kind"], "property": case.get("property"),
            "source": source, "target": target,
            "source_sha256": sha256(source), "target_sha256": sha256(target),
            "terminology": terminology, "review_context": review_context,
            "context_sha256": sha256(json.dumps([terminology, review_context], ensure_ascii=False)),
            "context_provenance": str(
                supplied.get("provenance") or "reconstructed_empty_not_historical"
            ),
            "existing_reports": existing,
            "all_existing_checks_accounted_for": (
                supplied.get("all_existing_checks_accounted_for") is True
            ),
        })
    return frozen


def write_json(path: Path, payload: Any) -> None:
    # Exclusive writes prevent accidental overwrites and repeated paid work on resume.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    path.chmod(0o600)


def critic_client(config: TarjomehConfig) -> LLMClient:
    # Reuse production role selection without constructing a pipeline or its DB.
    translator = LLMClient(config)
    try:
        critic = TranslationPipeline._build_critic_client(
            SimpleNamespace(llm_client=translator), config,
        )
    except Exception:
        translator.close()
        raise
    if critic is not translator:
        translator.close()
    return critic


def load_trial_config(raw_config: dict[str, Any]) -> TarjomehConfig:
    # Validate the merged saved job, not an incomplete credential-source profile.
    return TarjomehConfig.from_dict(
        raw_config, credential_source=TarjomehConfig.load(validate=False),
    )


def adjudication_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for record in records:
        case = record["case"]
        proposed = record["runs"][-1].get("result", {})
        for item in [*proposed.get("issue_details", []),
                     *proposed.get("ignored_issue_details", [])]:
            exact = bool(
                item.get("source_quote") and item["source_quote"] in case["source"]
                and item.get("current_persian_quote")
                and item["current_persian_quote"] in case["target"]
            )
            table.append({
                "case_id": case["id"], "finding": item,
                "source_sha256": case["source_sha256"], "target_sha256": case["target_sha256"],
                "context_sha256": case["context_sha256"],
                "real_saved_text": case["kind"] == "real_saved_text",
                "exact_quotes_checked": exact,
                "verdict": "pending" if exact else "unsupported",
                "missed_by_both_ordinary": None, "missed_by_all_existing_reports": None,
                "reason": "", "newly_triggers_existing_final_repair": None,
                "ordinary_baselines": [r.get("result", {}) for r in record["runs"][:2]],
                "existing_reports": case["existing_reports"],
            })
    return table


async def run_reviews(
    config: TarjomehConfig, cases: list[dict[str, Any]], out: Path,
) -> dict[str, Any]:
    records = []
    templates = experimental_templates()
    for case in cases:
        runs = []
        # A fresh client per arm makes budget history comparable between arms.
        for arm in ("ordinary_1", "ordinary_2", "proposed"):
            events = []
            client = critic_client(copy.deepcopy(config))
            client.set_attempt_observer(lambda event, sink=events: sink.append({
                key: event[key] for key in ATTEMPT_FIELDS if key in event
            }))
            print(f"START {case['id']} {arm}", flush=True)
            started = time.monotonic()
            try:
                reviewer = TranslationCritique(
                    client, quality_threshold=config.translation.critique_threshold,
                    max_parse_retries=config.translation.qa_json_retries,
                    prompt_template=templates["critique"] if arm == "proposed" else None,
                )
                result = await reviewer.critique(
                    case["source"], case["target"], case["terminology"], case["review_context"],
                )
                run = {"valid": result.valid, "result": result.to_dict()}
            except Exception as exc:
                run = {"valid": False, "error_type": type(exc).__name__}
            finally:
                client.close()
            run.update({"arm": arm, "attempts": events,
                        "seconds": round(time.monotonic() - started, 3)})
            write_json(out / f"review-{len(records):02d}-{arm}.json", run)
            runs.append(run)
            print(f"DONE {case['id']} {arm} valid={run['valid']} "
                  f"attempts={len(events)}", flush=True)
        records.append({"case": case, "runs": runs,
                        "served_models_comparable": comparable_served_models(runs)})
    attempts = [a for record in records for run in record["runs"] for a in run["attempts"]]
    return {
        "records": records, "logical_critique_calls": 3 * len(cases),
        "adjudication_table": adjudication_table(records),
        "property_controls": [
            {"case_id": r["case"]["id"], "property": r["case"]["property"],
             "human_verdict": "pending"}
            for r in records if r["case"]["kind"] == "known_good"
        ],
        "physical_accounting": model_identity_counts(attempts),
        "completion_tokens": sum(int(a.get("completion_tokens") or 0) for a in attempts),
        "seconds": round(sum(run["seconds"] for r in records for run in r["runs"]), 3),
        "status": "AWAITING_HUMAN_ADJUDICATION" if all(
            r["served_models_comparable"] for r in records
        ) else "INCONCLUSIVE",
        "production_guidance": "OFF",
        "database_writes": 0,
        "admission_and_export_trial": "NOT_RUN_not_a_reviewer_only_claim",
    }


def validate_refinement_cases(
    frozen: list[dict[str, Any]], adjudicated: list[dict[str, Any]],
) -> None:
    by_id = {case["id"]: case for case in frozen}
    seen = set()
    if not adjudicated or len(adjudicated) > 18:
        raise ValueError("Need one to 18 human-adjudicated refinement cases")
    for item in adjudicated:
        case = by_id.get(item.get("id"))
        if case is None or item["id"] in seen or item.get("human_approved") is not True:
            raise ValueError("Unknown, duplicate or unapproved refinement case")
        seen.add(item["id"])
        if item.get("target_sha256") != case["target_sha256"]:
            raise ValueError("Adjudication is for a different target")
        if (item.get("source_sha256") != case["source_sha256"]
                or item.get("context_sha256") != case["context_sha256"]):
            raise ValueError("Adjudication source or context differs")
        details = item.get("issue_details")
        if not isinstance(details, list) or not 1 <= len(details) <= 8:
            raise ValueError("Need one to eight identical grounded issues for both arms")
        ids = set()
        for detail in details:
            identifier = detail.get("issue_id")
            if (not identifier or identifier in ids or not detail.get("source_quote")
                    or detail["source_quote"] not in case["source"]
                    or not detail.get("current_persian_quote")
                    or detail["current_persian_quote"] not in case["target"]):
                raise ValueError("Issue IDs and exact source/current target quotes are required")
            ids.add(identifier)


async def run_refinements(config: TarjomehConfig, cases: list[dict[str, Any]],
                          adjudicated: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    validate_refinement_cases(cases, adjudicated)
    by_id = {case["id"]: case for case in cases}
    records, logical_calls = [], 0
    for number, item in enumerate(adjudicated):
        case = by_id[item["id"]]
        details = item["issue_details"]
        critique = CritiqueResult(
            issues=[str(d.get("rationale") or d["issue_id"]) for d in details],
            issue_details=details,
        )
        for arm in ("ordinary", "proposed"):
            client = LLMClient(copy.deepcopy(config))
            events = []
            client.set_attempt_observer(lambda event, sink=events: sink.append({
                k: event[k] for k in ATTEMPT_FIELDS if k in event
            }))
            print(f"START {case['id']} refiner {arm}", flush=True)
            started = time.monotonic()
            try:
                logical_calls += 1
                refiner = TranslationRefiner(
                    client, mode=config.translation.mode,
                    max_parse_retries=config.translation.qa_json_retries,
                    prompt_template=(
                        experimental_templates()["refinement"] if arm == "proposed" else None
                    ),
                )
                result = await refiner.refine_with_decision(
                    case["source"], case["target"], critique,
                    case["terminology"], case["review_context"],
                )
                run = {"valid": result.valid, "result": asdict(result)}
            except Exception as exc:
                run = {"valid": False, "error_type": type(exc).__name__}
            finally:
                client.close()
            run.update({"id": case["id"], "arm": arm, "attempts": events,
                        "admission": "NOT_RUN_requires_disposable_pipeline_gates"})
            # Changed text receives the ordinary critic; never a judge-model winner.
            if run["valid"] and result.translation != case["target"]:
                judge = critic_client(copy.deepcopy(config))
                validation_events = []
                judge.set_attempt_observer(lambda event, sink=validation_events: sink.append({
                    k: event[k] for k in ATTEMPT_FIELDS if k in event
                }))
                try:
                    logical_calls += 1
                    reviewed = await TranslationCritique(
                        judge, quality_threshold=config.translation.critique_threshold,
                        max_parse_retries=config.translation.qa_json_retries,
                    ).critique(case["source"], result.translation,
                               case["terminology"], case["review_context"])
                    run["candidate_review"] = reviewed.to_dict()
                except Exception as exc:
                    run["candidate_review"] = {"valid": False, "error_type": type(exc).__name__}
                finally:
                    judge.close()
                run["validation_attempts"] = validation_events
            run["seconds"] = round(time.monotonic() - started, 3)
            write_json(out / f"refinement-{number:02d}-{arm}.json", run)
            records.append(run)
            print(f"DONE {case['id']} refiner {arm} valid={run['valid']}", flush=True)
    attempts = [a for r in records for a in [*r["attempts"], *r.get("validation_attempts", [])]]
    return {"records": records, "logical_calls": logical_calls,
            "maximum_logical_calls": 72, "production_guidance": "OFF", "database_writes": 0,
            "physical_accounting": model_identity_counts(attempts),
            "completion_tokens": sum(int(a.get("completion_tokens") or 0) for a in attempts),
            "seconds": round(sum(r["seconds"] for r in records), 3),
            "admission_and_export_trial": "NOT_RUN_activation_blocked"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--db", type=Path)
    inputs.add_argument(
        "--bundle", type=Path, help="Exact saved cases plus job config, no DB needed",
    )
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--refine-decisions", type=Path)
    parser.add_argument("--rerun-case", help="One explicitly requested model-mismatch rerun")
    parser.add_argument("--previous-results", type=Path, help="Required proof for the single rerun")
    args = parser.parse_args()
    if args.bundle:
        bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
        if bundle.get("job_id") != args.job_id:
            parser.error("Bundle belongs to another job")
        raw_config, cases = bundle["config"], bundle["cases"]
    else:
        raw_config, cases = read_job(args.db or Path("/app/jobs/jobs.db"), args.job_id)
    controls = json.loads(args.controls.read_text(encoding="utf-8"))
    context = json.loads(args.context.read_text(encoding="utf-8")) if args.context else {}
    frozen = freeze_cases(cases, controls, context)
    if bool(args.rerun_case) != bool(args.previous_results):
        parser.error("--rerun-case requires --previous-results and vice versa")
    if args.rerun_case:
        previous = json.loads(args.previous_results.read_text(encoding="utf-8"))
        if previous.get("is_bounded_rerun"):
            parser.error("A bounded rerun cannot be rerun again")
        if previous.get("settings_sha256") != sha256(json.dumps(raw_config, sort_keys=True)):
            parser.error("Rerun job settings differ")
        if previous.get("experimental_template_hashes") != {
            k: sha256(v) for k, v in experimental_templates().items()
        }:
            parser.error("Rerun templates differ")
        matched = [r for r in previous.get("records", [])
                   if r.get("case", {}).get("id") == args.rerun_case]
        selected = [c for c in frozen if c["id"] == args.rerun_case]
        if (len(matched) != 1 or len(selected) != 1
                or matched[0].get("served_models_comparable") is not False
                or matched[0]["case"] != selected[0]):
            parser.error("Rerun must match the exact frozen, inconclusive case")
        if args.refine_decisions:
            parser.error("Model-mismatch rerun applies only to the review phase")
        frozen = selected
    templates = experimental_templates()
    manifest = {"job_id": args.job_id, "cases": frozen,
                "production_guidance": "OFF", "database_writes": 0,
                "settings_sha256": sha256(json.dumps(raw_config, sort_keys=True)),
                "template_hashes": {"ordinary_critic": sha256(CRITIQUE_PROMPT),
                                    "ordinary_refiner": sha256(REFINE_PROMPT),
                                    **{k: sha256(v) for k, v in templates.items()}},
                "maximum_logical_calls_before_helpers": 126,
                "original_context_reconstructed_cases": [c["id"] for c in frozen if
                    c["context_provenance"] != "exact_saved_and_hash_verified"]}
    # Never create a client or load credentials during a dry run.
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_json(args.out / "frozen.json", manifest)
    if not args.execute:
        print(json.dumps({"cases": len(frozen), "llm_calls": 0, "database_writes": 0,
                          "status": "DRY_RUN", "out": str(args.out)}, indent=2))
        return
    config = load_trial_config(raw_config)
    if config.translation.mode != "academic":
        parser.error("This full academic guidance trial requires academic mode")
    if args.refine_decisions:
        approved = json.loads(args.refine_decisions.read_text(encoding="utf-8"))
        result = asyncio.run(run_refinements(config, frozen, approved, args.out))
    else:
        result = asyncio.run(run_reviews(config, frozen, args.out))
    result["is_bounded_rerun"] = bool(args.rerun_case)
    result["settings_sha256"] = manifest["settings_sha256"]
    result["experimental_template_hashes"] = {k: sha256(v) for k, v in templates.items()}
    write_json(args.out / "results.json", result)
    print(f"Trial complete: {args.out}; production guidance remains OFF", flush=True)


if __name__ == "__main__":
    main()
