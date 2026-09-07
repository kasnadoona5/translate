"""Isolated, source-aware A/B benchmark for translation routes and models."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import LLMClient
from tarjomeh.core.prompts import GENERAL_EDITORIAL_CONTRACT
from tarjomeh.core.structured_output import parse_structured_output
from tarjomeh.quality.integrity import extract_identifiers, extract_numbers
from tarjomeh.quality.structure_audit import audit_payload

BENCHMARK_SCHEMA = 1
WEIGHTS = {
    "accuracy": 35,
    "completeness_structure": 25,
    "fluency": 20,
    "terminology": 10,
    "register": 10,
}

_DEFAULT_ACADEMIC_SUITE: dict[str, Any] = {
    "schema": 1,
    "id": "academic-core-v1",
    "description": (
        "Synthetic source-grounded cases for academic English-to-Persian "
        "model comparison."
    ),
    "passages": [
        {
            "id": "explicit-count-and-enumeration",
            "domain": "social theory",
            "source": (
                "This chapter addresses two issues. First, it distinguishes "
                "institutional authority from coercive capacity. Second, it "
                "explains why that distinction matters for comparative analysis."
            ),
            "notes": "Preserve the explicit count and both coordinated claims.",
        },
        {
            "id": "coordination-and-voice",
            "domain": "political sociology",
            "source": (
                "The framework was extended by later research, and its central "
                "assumptions were qualified when comparative evidence revealed "
                "a different causal sequence."
            ),
            "notes": "Preserve both actions, their voice, and the causal relation.",
        },
        {
            "id": "conceptual-family-and-citation",
            "domain": "political theory",
            "source": (
                "The analysis separates polity, politics, and policy while "
                "preserving their conceptual relationship (Author 1996, 42-44)."
            ),
            "notes": "Keep the triad distinct and the citation unchanged.",
        },
        {
            "id": "foreign-expression-and-qualification",
            "domain": "historical sociology",
            "source": (
                "A longue dur\u00e9e account may clarify the institutional pattern, "
                "but it does not by itself establish that the outcome was necessary."
            ),
            "notes": (
                "Retain the source-authored foreign expression and the modal "
                "limitation."
            ),
        },
    ],
}


def load_benchmark_suite(path: Path) -> dict[str, Any]:
    """Load and validate a frozen benchmark suite."""
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    elif path.as_posix().endswith("benchmarks/suites/academic-core.json"):
        # Docker installs only the Python package. Keep the transparent JSON
        # corpus in the repository while providing the exact frozen default to
        # installed runtimes without expanding the production image.
        data = copy.deepcopy(_DEFAULT_ACADEMIC_SUITE)
    else:
        raise FileNotFoundError(path)
    if not isinstance(data, dict) or not isinstance(data.get("passages"), list):
        raise ValueError("Benchmark suite must contain a passages list")
    passages: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in data["passages"]:
        if not isinstance(raw, dict):
            raise ValueError("Each benchmark passage must be an object")
        passage_id = str(raw.get("id", "")).strip()
        source = str(raw.get("source", "")).strip()
        if not passage_id or not source or passage_id in seen:
            raise ValueError("Passage IDs must be unique and source text cannot be empty")
        seen.add(passage_id)
        passages.append({
            "id": passage_id,
            "source": source,
            "domain": str(raw.get("domain", "academic prose")),
            "notes": str(raw.get("notes", "")),
        })
    if not passages:
        raise ValueError("Benchmark suite cannot be empty")
    return {
        "schema": int(data.get("schema", BENCHMARK_SCHEMA)),
        "id": str(data.get("id", path.stem)),
        "description": str(data.get("description", "")),
        "passages": passages,
    }


def _translation_prompt(passage: dict[str, Any]) -> str:
    return (
        "Translate the complete source passage into publication-quality Iranian "
        "Persian. Accuracy and completeness outrank fluency; fluency must not "
        "simplify, omit, reinterpret, or weaken any proposition. Preserve paragraph "
        "boundaries, numbers, names, citations, logical relations, coordinated "
        "actions, and source-authored foreign expressions. Output Persian only.\n\n"
        + GENERAL_EDITORIAL_CONTRACT
        + "\n\nSOURCE:\n"
        + passage["source"]
    )


def _judge_prompt(passage: dict[str, Any], translation: str) -> str:
    return f"""You are an independent senior editor evaluating an English-to-Persian
academic translation. Score only the supplied source and candidate. Natural Persian
must remain fully faithful: do not reward simplification, omission, reinterpretation,
or weakened qualification. Preserve deliberate complexity when it can be expressed
clearly. Return JSON only.

SOURCE:
{passage['source']}

CANDIDATE:
{translation}

Return exactly:
{{
  "scores": {{
    "accuracy": <0-10>,
    "completeness_structure": <0-10>,
    "fluency": <0-10>,
    "terminology": <0-10>,
    "register": <0-10>
  }},
  "blocking_source_error": <true|false>,
  "findings": [{{
    "category": "...", "source_quote": "...",
    "candidate_quote": "...", "reason": "..."
  }}],
  "summary": "..."
}}
"""


def deterministic_source_checks(source: str, translation: str) -> dict[str, Any]:
    """Return source-grounded checks whose failures disqualify a candidate."""
    source_numbers = extract_numbers(source)
    target_numbers = extract_numbers(translation)
    source_identifiers = extract_identifiers(source)
    target_identifiers = extract_identifiers(translation)
    source_paragraphs = [part for part in source.split("\n\n") if part.strip()]
    target_paragraphs = [part for part in translation.split("\n\n") if part.strip()]
    structure = list(audit_payload(source, translation).get("findings", []))
    blocking_structure = [
        finding for finding in structure
        if str(finding.get("details", {}).get("admission", "blocking")) == "blocking"
    ]
    failures: list[str] = []
    if not translation.strip():
        failures.append("empty_translation")
    if source_numbers - target_numbers:
        failures.append("missing_or_changed_number")
    if source_identifiers - target_identifiers:
        failures.append("missing_identifier")
    if len(source_paragraphs) != len(target_paragraphs):
        failures.append("paragraph_count_changed")
    if blocking_structure:
        failures.append("explicit_source_structure_changed")
    return {
        "passed": not failures,
        "failures": failures,
        "missing_numbers": list((source_numbers - target_numbers).elements()),
        "missing_identifiers": list((source_identifiers - target_identifiers).elements()),
        "source_paragraphs": len(source_paragraphs),
        "target_paragraphs": len(target_paragraphs),
        "blocking_structure": blocking_structure,
    }


def _bounded_score(value: Any) -> float:
    try:
        return round(min(10.0, max(0.0, float(value))), 3)
    except (TypeError, ValueError):
        return 0.0


def _normalized_judgement(raw: str) -> dict[str, Any]:
    data = parse_structured_output(raw, expected=dict)
    scores = data.get("scores", {})
    if not isinstance(scores, dict):
        scores = {}
    normalized = {name: _bounded_score(scores.get(name)) for name in WEIGHTS}
    weighted = round(sum(
        normalized[name] * weight / 10.0
        for name, weight in WEIGHTS.items()
    ), 3)
    findings = data.get("findings", [])
    return {
        "scores": normalized,
        "weighted_score": weighted,
        "blocking_source_error": bool(data.get("blocking_source_error")),
        "findings": findings if isinstance(findings, list) else [],
        "summary": str(data.get("summary", "")),
    }


def _attempt_totals(events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "attempts": len(events),
        "failures": sum(not bool(event.get("success")) for event in events),
        "prompt_tokens": sum(int(event.get("prompt_tokens", 0) or 0) for event in events),
        "completion_tokens": sum(
            int(event.get("completion_tokens", 0) or 0) for event in events
        ),
        "served_models": dict(Counter(
            str(event.get("response_model") or event.get("model") or "unknown")
            for event in events if event.get("success")
        )),
    }


class ModelBenchmark:
    """Compare configured model routes without touching production state."""

    def __init__(
        self,
        config: TarjomehConfig,
        client_factory: Callable[[TarjomehConfig], Any] = LLMClient,
    ) -> None:
        self.config = config
        self.client_factory = client_factory

    def _client(self, model: str, events: list[dict[str, Any]]) -> Any:
        config = copy.deepcopy(self.config)
        config.llm.model = model
        config.llm.temperature = 0.0
        client = self.client_factory(config)
        if hasattr(client, "set_attempt_observer"):
            client.set_attempt_observer(lambda event: events.append(dict(event)))
        return client

    def run(
        self,
        suite: dict[str, Any],
        models: list[str],
        judge_model: str,
    ) -> dict[str, Any]:
        if len(set(models)) < 2:
            raise ValueError("Benchmark comparison requires at least two distinct models")
        if not judge_model.strip():
            raise ValueError("An explicit judge model is required")
        if judge_model in set(models):
            raise ValueError(
                "The benchmark judge must be independent of the translation routes"
            )
        run_seed = json.dumps({
            "suite": suite["id"], "models": models, "judge": judge_model,
            "time": time.time_ns(),
        }, sort_keys=True)
        run_id = hashlib.sha256(run_seed.encode("utf-8")).hexdigest()[:12]
        results: list[dict[str, Any]] = []
        for model in models:
            translation_events: list[dict[str, Any]] = []
            judge_events: list[dict[str, Any]] = []
            translator = self._client(model, translation_events)
            judge = self._client(judge_model, judge_events)
            try:
                for passage in suite["passages"]:
                    started = time.monotonic()
                    translation = translator.complete(
                        messages=[{"role": "user", "content": _translation_prompt(passage)}],
                        _operation="benchmark_translation",
                        _recovery_source_text=passage["source"],
                    ).strip()
                    latency = round(time.monotonic() - started, 3)
                    deterministic = deterministic_source_checks(
                        passage["source"], translation
                    )
                    raw_judgement = judge.complete(
                        messages=[{
                            "role": "user",
                            "content": _judge_prompt(passage, translation),
                        }],
                        _operation="benchmark_judge",
                    )
                    judgement = _normalized_judgement(raw_judgement)
                    eligible = bool(
                        deterministic["passed"]
                        and not judgement["blocking_source_error"]
                    )
                    results.append({
                        "model": model,
                        "passage_id": passage["id"],
                        "source": passage["source"],
                        "translation": translation,
                        "latency_seconds": latency,
                        "deterministic": deterministic,
                        "judgement": judgement,
                        "eligible": eligible,
                    })
            finally:
                if hasattr(translator, "close"):
                    translator.close()
                if hasattr(judge, "close"):
                    judge.close()
            for result in results:
                if result["model"] == model:
                    result["route_evidence"] = {
                        "requested_model": model,
                        "translation": _attempt_totals(translation_events),
                        "judge_requested_model": judge_model,
                        "judge": _attempt_totals(judge_events),
                    }
        summaries = []
        for model in models:
            model_results = [item for item in results if item["model"] == model]
            eligible_results = [item for item in model_results if item["eligible"]]
            summaries.append({
                "model": model,
                "passages": len(model_results),
                "eligible_passages": len(eligible_results),
                "disqualified_passages": len(model_results) - len(eligible_results),
                "average_quality_score": round(
                    sum(
                        item["judgement"]["weighted_score"]
                        for item in eligible_results
                    ) / max(1, len(eligible_results)), 3
                ),
                "latency_seconds": round(
                    sum(item["latency_seconds"] for item in model_results), 3
                ),
            })
        summaries.sort(key=lambda item: (
            item["disqualified_passages"], -item["average_quality_score"],
            item["latency_seconds"],
        ))
        return {
            "schema": BENCHMARK_SCHEMA,
            "id": run_id,
            "suite": {key: suite.get(key) for key in ("id", "description", "schema")},
            "judge_model": judge_model,
            "weights": WEIGHTS,
            "ranking_policy": (
                "Fewest source-grounded disqualifications, then highest weighted "
                "quality score, then lowest latency. Results never change runtime config."
            ),
            "summaries": summaries,
            "results": results,
        }


def write_benchmark_reports(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write transparent machine and human-readable benchmark artifacts."""
    run_dir = output_dir / str(report["id"])
    run_dir.mkdir(parents=True, exist_ok=False)
    json_path = run_dir / "benchmark.json"
    csv_path = run_dir / "benchmark.csv"
    markdown_path = run_dir / "benchmark.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=[
        "model", "passage_id", "eligible", "weighted_score",
        "latency_seconds", "deterministic_failures", "human_score", "human_notes",
    ])
    writer.writeheader()
    for item in report["results"]:
        writer.writerow({
            "model": item["model"],
            "passage_id": item["passage_id"],
            "eligible": item["eligible"],
            "weighted_score": item["judgement"]["weighted_score"],
            "latency_seconds": item["latency_seconds"],
            "deterministic_failures": ";".join(item["deterministic"]["failures"]),
            "human_score": "",
            "human_notes": "",
        })
    csv_path.write_text(csv_buffer.getvalue(), encoding="utf-8-sig")

    lines = [
        f"# Model benchmark {report['id']}", "",
        f"Suite: `{report['suite']['id']}`", "",
        f"Judge route: `{report['judge_model']}`", "",
        "The ranking is advisory. It does not modify Tarjomeh configuration.", "",
        "| Rank | Requested model | Eligible | Disqualified | Quality | Latency (s) |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, summary in enumerate(report["summaries"], 1):
        lines.append(
            f"| {rank} | `{summary['model']}` | {summary['eligible_passages']} | "
            f"{summary['disqualified_passages']} | {summary['average_quality_score']:.3f} | "
            f"{summary['latency_seconds']:.3f} |"
        )
    lines.extend(["", "## Passage evidence", ""])
    for item in report["results"]:
        served = item["route_evidence"]["translation"]["served_models"]
        lines.extend([
            f"### {item['model']} / {item['passage_id']}", "",
            f"Eligible: `{item['eligible']}`  ",
            f"Score: `{item['judgement']['weighted_score']}`  ",
            f"Served model(s): `{json.dumps(served, ensure_ascii=False)}`  ",
            f"Deterministic failures: `{item['deterministic']['failures']}`", "",
            "```text", item["translation"], "```", "",
        ])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}
