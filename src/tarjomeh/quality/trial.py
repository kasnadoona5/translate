"""Isolated experimental guidance and scoring. Never imported by the pipeline."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from tarjomeh.core.prompts import CRITIQUE_PROMPT, REFINE_PROMPT


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _move_section(template: str, start: str, end: str, before: str) -> str:
    first = template.index(start)
    last = template.index(end, first)
    section = template[first:last]
    remaining = template[:first] + template[last:]
    return remaining.replace(before, section + "\n" + before, 1)


def experimental_templates() -> dict[str, str]:
    """Front-load existing safeguards, keeping schemas, budgets and caps intact."""
    critic = _move_section(
        CRITIQUE_PROMPT,
        "- Compare every source sentence",
        "- Put every stable source label",
        "### Source (English)",
    )
    critic = _move_section(
        critic,
        "- Build a compact proposition checklist",
        "- Check every source-authored announced quantity",
        "### Source (English)",
    )
    critic = critic.replace(
        "participants, negation/modality,",
        "participants, qualifier/alias scope, tense/aspect, negation/modality,",
        1,
    )
    refiner = _move_section(
        REFINE_PROMPT,
        "8. Include the complete Persian translation",
        "9. Return ONLY valid JSON",
        "### Source (English)",
    )
    refiner = refiner.replace(
        "For an accepted edit, also quote original_span verbatim from the current Persian",
        "Before editing, copy original_span verbatim from the CURRENT Persian, never a\n"
        "past candidate or the critic's suggested correction. For an accepted edit, original_span\n"
        "must be copied verbatim from the current Persian",
        1,
    )
    return {"critique": critic, "refinement": refiner}


def comparable_served_models(runs: list[dict[str, Any]]) -> bool:
    """Failed routes with no response are not successful model mismatches."""
    if not runs or any(not run.get("valid") for run in runs):
        return False
    models = []
    for run in runs:
        successes = [a for a in run.get("attempts", []) if a.get("success") is True]
        if not successes or any(not str(a.get("response_model") or "").strip() for a in successes):
            return False
        models.extend(str(a["response_model"]).strip() for a in successes)
    return len(set(models)) == 1


def adjudication_metrics(findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Uncertain, unsupported and unadjudicated findings never disappear."""
    counts = Counter(
        "unsupported" if item.get("verdict") == "true"
        and item.get("exact_quotes_checked") is not True
        else str(item.get("verdict") or "pending")
        for item in findings
    )
    total = len(findings)
    true = counts["true"]
    useful_keys = {
        (item["case_id"], item["finding"]["source_quote"],
         item["finding"]["current_persian_quote"])
        for item in findings
        if item.get("verdict") == "true"
        and item.get("real_saved_text") is True
        and item.get("missed_by_both_ordinary") is True
        and item.get("missed_by_all_existing_reports") is True
        and item.get("exact_quotes_checked") is True
        and item.get("case_id") and isinstance(item.get("finding"), dict)
        and item["finding"].get("source_quote")
        and item["finding"].get("current_persian_quote")
    }
    useful = len(useful_keys)
    return {
        "finding_count": total,
        "verdict_counts": dict(counts),
        "precision": true / total if total else 0.0,
        "real_unique_useful_findings": useful,
        "numeric_thresholds_met": bool(total and true / total >= 0.8 and useful >= 2),
        "activation": "OFF_requires_human_evidence_approval_and_complete_admission_trial",
    }
