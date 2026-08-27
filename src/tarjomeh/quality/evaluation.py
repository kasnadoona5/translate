"""Deterministic translation regression evaluation.

This module deliberately does not call an LLM. It compares persisted jobs and
checks observable integrity signals so quality changes can be measured without
changing the translation pipeline.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import uuid
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from tarjomeh.jobs.database import JobDatabase
from tarjomeh.quality.integrity import (
    available_note_markers,
    extract_note_markers,
    extract_numbers,
)


_SPACE_RE = re.compile(r"\s+")


def source_hash(text: str) -> str:
    """Return a stable identity for a normalized source segment."""
    normalized = _SPACE_RE.sub(" ", text or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text or "") if part.strip()]


def _latest_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    matches = [event for event in events if event.get("event_type") == event_type]
    return matches[-1] if matches else None


def _check(
    check_id: str,
    passed: bool,
    severity: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "passed": passed,
        "severity": severity,
        "message": message,
        "details": details or {},
    }


def inspect_chunk(
    source: str,
    translation: str,
    status: str,
    events: list[dict[str, Any]],
    duplicate_count: int = 1,
) -> dict[str, Any]:
    """Inspect one source/translation pair using deterministic evidence."""
    source = source or ""
    translation = translation or ""
    checks: list[dict[str, Any]] = []
    checks.append(_check(
        "translation_present", bool(translation.strip()), "blocking",
        "Translation is present." if translation.strip() else "Translation is empty.",
    ))
    status_ok = status in {"completed", "needs_review"}
    checks.append(_check(
        "terminal_chunk_status", status_ok, "blocking",
        f"Chunk status is {status}.", {"status": status},
    ))

    ratio = len(translation.strip()) / max(1, len(source.strip()))
    ratio_ok = not source.strip() or 0.25 <= ratio <= 3.0
    checks.append(_check(
        "length_ratio", ratio_ok, "blocking",
        f"Target/source character ratio is {ratio:.3f}.", {"ratio": round(ratio, 4)},
    ))
    if ratio_ok and source.strip():
        checks.append(_check(
            "length_ratio_advisory", 0.45 <= ratio <= 2.0, "warning",
            "Length ratio is within the normal advisory range."
            if 0.45 <= ratio <= 2.0 else "Length ratio is unusual and merits review.",
            {"ratio": round(ratio, 4)},
        ))

    source_numbers = extract_numbers(source)
    target_numbers = extract_numbers(translation)
    missing_numbers = list((source_numbers - target_numbers).elements())
    checks.append(_check(
        "numbers_preserved", not missing_numbers, "blocking",
        "All source numbers are preserved."
        if not missing_numbers else "One or more source numbers are missing or changed.",
        {"missing": missing_numbers, "source": list(source_numbers.elements()),
         "target": list(target_numbers.elements())},
    ))

    source_notes = extract_note_markers(source)
    required_notes = Counter(source_notes)
    target_notes = available_note_markers(translation, required_notes)
    missing_notes = list((required_notes - target_notes).elements())
    checks.append(_check(
        "note_markers_preserved", not missing_notes, "blocking",
        "Footnote/endnote markers are preserved."
        if not missing_notes else "Footnote/endnote markers are missing.",
        {"missing": missing_notes},
    ))

    source_paragraphs = _paragraphs(source)
    target_paragraphs = _paragraphs(translation)
    minimum = max(1, (len(source_paragraphs) + 1) // 2)
    paragraph_ok = len(source_paragraphs) <= 1 or len(target_paragraphs) >= minimum
    checks.append(_check(
        "paragraph_structure", paragraph_ok, "warning",
        "Paragraph structure is plausibly preserved."
        if paragraph_ok else "Several source paragraphs may have been collapsed or omitted.",
        {"source_count": len(source_paragraphs), "target_count": len(target_paragraphs)},
    ))

    first_source_line = source.strip().splitlines()[0] if source.strip() else ""
    likely_heading = bool(first_source_line and len(first_source_line) <= 100 and "\n\n" in source)
    target_first = translation.strip().splitlines()[0] if translation.strip() else ""
    heading_ok = not likely_heading or bool(target_first and len(target_first) <= 140 and "\n" in translation)
    checks.append(_check(
        "heading_structure", heading_ok, "warning",
        "Leading heading structure is preserved or not applicable."
        if heading_ok else "A short leading source heading may have lost its structure.",
        {"likely_source_heading": likely_heading},
    ))

    duplicate_ok = not (translation.strip() and len(translation.strip()) > 50 and duplicate_count > 1)
    checks.append(_check(
        "unexpected_duplicate", duplicate_ok, "blocking",
        "Translation is unique among source-distinct chunks."
        if duplicate_ok else "The same substantial translation appears in multiple chunks.",
        {"duplicate_count": duplicate_count},
    ))

    glossary_event = _latest_event(events, "glossary_compliance_final")
    if glossary_event:
        violations = int(glossary_event.get("payload", {}).get("violation_count", 0) or 0)
        checks.append(_check(
            "glossary_compliance", violations == 0, "blocking",
            "Persisted glossary compliance passed."
            if violations == 0 else "Persisted glossary violations remain.",
            {"violation_count": violations},
        ))

    critique = _latest_event(events, "critique_completed")
    score = None
    if critique:
        raw_score = critique.get("payload", {}).get("scores", {}).get("average")
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = None

    failures = [item for item in checks if not item["passed"]]
    return {
        "checks": checks,
        "blocking_failures": sum(item["severity"] == "blocking" for item in failures),
        "warnings": sum(item["severity"] == "warning" for item in failures),
        "critique_average": score,
        "length_ratio": round(ratio, 4),
    }


def _translation_duplicates(chunks: list[dict[str, Any]]) -> Counter[str]:
    by_translation: dict[str, set[str]] = {}
    for chunk in chunks:
        translation = _SPACE_RE.sub(" ", chunk.get("translation") or "").strip()
        if len(translation) <= 50:
            continue
        digest = hashlib.sha256(translation.encode("utf-8")).hexdigest()
        by_translation.setdefault(digest, set()).add(source_hash(chunk.get("text") or ""))
    return Counter({key: len(sources) for key, sources in by_translation.items()})


def inspect_output(job: dict[str, Any]) -> dict[str, Any]:
    """Check that a persisted job output still exists and has a valid envelope."""
    raw_path = job.get("output_path") or ""
    path = Path(raw_path) if raw_path else None
    exists = bool(path and path.is_file())
    size = path.stat().st_size if exists and path else 0
    valid = exists and size > 0
    reason = "ok" if valid else "output file is missing or empty"
    suffix = path.suffix.lower() if path else ""
    if valid and suffix in {".docx", ".epub"}:
        valid = zipfile.is_zipfile(path)
        if valid:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
            required = {"[Content_Types].xml", "word/document.xml"} if suffix == ".docx" else {"mimetype", "META-INF/container.xml"}
            valid = required.issubset(names)
        reason = "ok" if valid else "output has an invalid document package"
    elif valid and suffix == ".pdf":
        try:
            with path.open("rb") as stream:
                valid = stream.read(5) == b"%PDF-"
        except OSError:
            valid = False
        reason = "ok" if valid else "output does not have a PDF header"
    return {"valid": valid, "exists": exists, "size": size, "path": raw_path, "reason": reason}


class QualityEvaluator:
    """Compare two persisted jobs without modifying either job."""

    def __init__(self, db: JobDatabase | None = None) -> None:
        self.db = db or JobDatabase()

    def evaluate(self, baseline_job_id: str, candidate_job_id: str) -> dict[str, Any]:
        if baseline_job_id == candidate_job_id:
            raise ValueError("Baseline and candidate jobs must be different")
        baseline_job = self.db.get_job(baseline_job_id)
        candidate_job = self.db.get_job(candidate_job_id)
        if not baseline_job or not candidate_job:
            missing = baseline_job_id if not baseline_job else candidate_job_id
            raise ValueError(f"Job not found: {missing}")
        for label, job in (("Baseline", baseline_job), ("Candidate", candidate_job)):
            if job.get("raw_status", job.get("status")) != "completed":
                raise ValueError(f"{label} job must be completed")

        evaluation_id = uuid.uuid4().hex[:12]
        baseline_chunks = {int(c["chunk_index"]): c for c in self.db.get_chunks(baseline_job_id)}
        candidate_chunks = {int(c["chunk_index"]): c for c in self.db.get_chunks(candidate_job_id)}
        baseline_events = self._events_by_chunk(baseline_job_id)
        candidate_events = self._events_by_chunk(candidate_job_id)
        baseline_dupes = _translation_duplicates(list(baseline_chunks.values()))
        candidate_dupes = _translation_duplicates(list(candidate_chunks.values()))

        results = []
        for index in sorted(set(baseline_chunks) | set(candidate_chunks)):
            old = baseline_chunks.get(index, {})
            new = candidate_chunks.get(index, {})
            old_source = old.get("text") or ""
            new_source = new.get("text") or ""
            source = new_source or old_source
            old_translation = old.get("translation") or ""
            new_translation = new.get("translation") or ""
            old_digest = hashlib.sha256(
                _SPACE_RE.sub(" ", old_translation).strip().encode("utf-8")
            ).hexdigest()
            new_digest = hashlib.sha256(
                _SPACE_RE.sub(" ", new_translation).strip().encode("utf-8")
            ).hexdigest()
            baseline_result = inspect_chunk(
                old_source, old_translation, old.get("status", "missing"),
                baseline_events.get(index, []), baseline_dupes.get(old_digest, 1),
            )
            candidate_result = inspect_chunk(
                new_source, new_translation, new.get("status", "missing"),
                candidate_events.get(index, []), candidate_dupes.get(new_digest, 1),
            )
            old_failed = {c["id"] for c in baseline_result["checks"] if not c["passed"]}
            new_failed = {c["id"] for c in candidate_result["checks"] if not c["passed"]}
            a_is_candidate = int(hashlib.sha256(
                f"{evaluation_id}:{index}".encode("ascii")
            ).hexdigest(), 16) % 2 == 0
            results.append({
                "chunk_index": index,
                "source_hash": source_hash(source),
                "source": source,
                "source_aligned": bool(old_source and new_source and source_hash(old_source) == source_hash(new_source)),
                "baseline_translation": old_translation,
                "candidate_translation": new_translation,
                "baseline": baseline_result,
                "candidate": candidate_result,
                "regressions": sorted(new_failed - old_failed),
                "improvements": sorted(old_failed - new_failed),
                "changed": old_translation != new_translation,
                "a_is_candidate": a_is_candidate,
            })

        summary = self._summarize(results)
        summary["baseline_output"] = inspect_output(baseline_job)
        summary["candidate_output"] = inspect_output(candidate_job)
        if not summary["candidate_output"]["valid"]:
            summary["release_blocked"] = True
        self.db.save_evaluation(
            baseline_job_id, candidate_job_id, summary, results, evaluation_id=evaluation_id,
        )
        return self.db.get_evaluation(evaluation_id) or {}

    def _events_by_chunk(self, job_id: str) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for event in self.db.get_chunk_events(job_id):
            grouped.setdefault(int(event["chunk_index"]), []).append(event)
        return grouped

    @staticmethod
    def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
        baseline_blocking = sum(item["baseline"]["blocking_failures"] for item in results)
        candidate_blocking = sum(item["candidate"]["blocking_failures"] for item in results)
        baseline_warnings = sum(item["baseline"]["warnings"] for item in results)
        candidate_warnings = sum(item["candidate"]["warnings"] for item in results)
        alignment_failures = sum(not item["source_aligned"] for item in results)
        return {
            "chunks_compared": len(results),
            "changed_chunks": sum(bool(item["changed"]) for item in results),
            "source_alignment_failures": alignment_failures,
            "baseline_blocking_failures": baseline_blocking,
            "candidate_blocking_failures": candidate_blocking,
            "baseline_warnings": baseline_warnings,
            "candidate_warnings": candidate_warnings,
            "integrity_regressions": sum(len(item["regressions"]) for item in results),
            "integrity_improvements": sum(len(item["improvements"]) for item in results),
            "release_blocked": (
                candidate_blocking > 0
                or alignment_failures > 0
                or any(item["regressions"] for item in results)
            ),
        }


def evaluation_text_report(evaluation: dict[str, Any]) -> str:
    """Render a concise human-readable report."""
    summary = evaluation.get("summary", {})
    lines = [
        "Tarjomeh Quality Regression Report",
        f"Evaluation: {evaluation.get('id')}",
        f"Baseline: {evaluation.get('baseline_job_id')}",
        f"Candidate: {evaluation.get('candidate_job_id')}",
        f"Created: {evaluation.get('created_at')}",
        "",
        f"Chunks compared: {summary.get('chunks_compared', 0)}",
        f"Changed chunks: {summary.get('changed_chunks', 0)}",
        f"Source alignment failures: {summary.get('source_alignment_failures', 0)}",
        f"Blocking failures: {summary.get('baseline_blocking_failures', 0)} -> {summary.get('candidate_blocking_failures', 0)}",
        f"Warnings: {summary.get('baseline_warnings', 0)} -> {summary.get('candidate_warnings', 0)}",
        f"Integrity regressions: {summary.get('integrity_regressions', 0)}",
        f"Integrity improvements: {summary.get('integrity_improvements', 0)}",
        f"Release blocked: {summary.get('release_blocked', False)}",
        f"Baseline output: {summary.get('baseline_output', {}).get('reason', 'not checked')}",
        f"Candidate output: {summary.get('candidate_output', {}).get('reason', 'not checked')}",
        "",
    ]
    for chunk in evaluation.get("chunks", []):
        if not chunk.get("regressions") and not chunk.get("improvements"):
            continue
        lines.append(
            f"Chunk {chunk['chunk_index']}: regressions={','.join(chunk['regressions']) or '-'} "
            f"improvements={','.join(chunk['improvements']) or '-'}"
        )
    return "\n".join(lines).rstrip() + "\n"


def evaluation_csv_report(evaluation: dict[str, Any]) -> str:
    """Render one comparison row per chunk as CSV."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "evaluation_id", "chunk_index", "source_aligned", "changed",
        "baseline_blocking", "candidate_blocking", "baseline_warnings",
        "candidate_warnings", "baseline_critique", "candidate_critique",
        "regressions", "improvements", "preference",
    ])
    for chunk in evaluation.get("chunks", []):
        writer.writerow([
            evaluation.get("id"), chunk["chunk_index"], chunk["source_aligned"],
            chunk["changed"], chunk["baseline"]["blocking_failures"],
            chunk["candidate"]["blocking_failures"], chunk["baseline"]["warnings"],
            chunk["candidate"]["warnings"], chunk["baseline"]["critique_average"],
            chunk["candidate"]["critique_average"], ";".join(chunk["regressions"]),
            ";".join(chunk["improvements"]),
            (chunk.get("preference") or {}).get("preference", ""),
        ])
    return output.getvalue()


def evaluation_json_report(evaluation: dict[str, Any]) -> str:
    return json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
