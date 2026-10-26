"""Pure provenance and history selection for explicitly reviewed revisions.

Hashes bind a human decision to text; they never prove semantic correctness.
Ordinary histories without revision metadata keep their existing selection rule.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from typing import Any

_HASH = re.compile(r"[0-9a-f]{64}")


def document_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def history_key(entry: dict[str, Any]) -> tuple[str, str]:
    return document_hash(entry.get("summary", {})), document_hash(entry.get("coverage", []))


def validate_manifest(manifest: Any) -> None:
    """Validate structure only; canonical/source proof is checked by the writer."""
    from tarjomeh.memory.bilingual_summary import BilingualSummary
    from tarjomeh.quality.structure_audit import audit_payload

    required = {
        "schema",
        "job_id",
        "document_sha256",
        "expected_memory_sha256",
        "previous_summary",
        "candidate_summary",
        "coverage",
        "supersedes",
        "evidence",
    }
    if not isinstance(manifest, dict) or set(manifest) != required or manifest["schema"] != 1:
        raise ValueError("Invalid summary revision manifest schema")
    if not isinstance(manifest["job_id"], str) or not manifest["job_id"].strip():
        raise ValueError("A revision needs an exact job ID")
    for key in ("document_sha256", "expected_memory_sha256"):
        if not isinstance(manifest[key], str) or not _HASH.fullmatch(manifest[key]):
            raise ValueError("Invalid revision hash")
    for key in ("previous_summary", "candidate_summary"):
        value = manifest[key]
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "english_summary",
                "persian_summary",
                "input_trust",
                "trust_reasons",
            }
            or any(
                not isinstance(value[k], str)
                for k in (
                    "english_summary",
                    "persian_summary",
                    "input_trust",
                )
            )
            or not isinstance(value["trust_reasons"], list)
            or any(not isinstance(r, str) for r in value["trust_reasons"])
        ):
            raise ValueError("Invalid summary fields")
    old, new = manifest["previous_summary"], manifest["candidate_summary"]
    normalized = BilingualSummary()
    normalized.deserialize(old)
    if (
        new["input_trust"] != normalized.input_trust
        or new["trust_reasons"] != normalized.trust_reasons
    ):
        raise ValueError("Summary revision cannot change contributing input trust")
    candidate = BilingualSummary()
    candidate.deserialize(new)
    if candidate.serialize() != new or not candidate.candidate_quality()["accepted"]:
        raise ValueError("Invalid bilingual summary candidate")
    for section in (new["english_summary"], new["persian_summary"]):
        stack = []
        for character in section:
            if character in "([{":
                stack.append(character)
            elif character in ")]}" and (
                not stack or stack.pop() != {")": "(", "]": "[", "}": "{"}[character]
            ):
                raise ValueError("Unbalanced candidate delimiters")
        if stack:
            raise ValueError("Unbalanced candidate delimiters")
    alignment = audit_payload(new["english_summary"], new["persian_summary"])
    if "translation_structure_mismatch" in alignment.get("classifications", []):
        raise ValueError("Candidate has a bilingual structure mismatch")
    coverage = manifest["coverage"]
    if not isinstance(coverage, list) or not coverage:
        raise ValueError("A revision needs canonical coverage")
    indices = []
    for ref in coverage:
        if (
            not isinstance(ref, dict)
            or set(ref)
            != {
                "chunk_index",
                "source_sha256",
                "target_sha256",
            }
            or type(ref["chunk_index"]) is not int
            or ref["chunk_index"] < 0
            or any(
                not isinstance(ref[k], str) or not _HASH.fullmatch(ref[k])
                for k in ("source_sha256", "target_sha256")
            )
        ):
            raise ValueError("Invalid canonical reference")
        indices.append(ref["chunk_index"])
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate canonical reference")
    if (
        not isinstance(manifest["supersedes"], list)
        or not manifest["supersedes"]
        or any(
            not isinstance(key, list)
            or len(key) != 2
            or any(not isinstance(part, str) or not _HASH.fullmatch(part) for part in key)
            for key in manifest["supersedes"]
        )
    ):
        raise ValueError("Explicit superseded summary identities are required")
    if (
        new == old
        or list(history_key({"summary": new, "coverage": coverage})) in manifest["supersedes"]
    ):
        raise ValueError("A revision must change the summary and cannot supersede itself")
    if not isinstance(manifest["evidence"], list) or not manifest["evidence"]:
        raise ValueError("Source evidence is required")
    for item in manifest["evidence"]:
        if (
            not isinstance(item, dict)
            or set(item)
            != {
                "chunk_index",
                "source_sha256",
                "start",
                "end",
                "quote",
                "claim",
            }
            or type(item["chunk_index"]) is not int
            or type(item["start"]) is not int
            or type(item["end"]) is not int
            or not 0 <= item["start"] < item["end"]
            or not isinstance(item["quote"], str)
            or not item["quote"].strip()
            or not isinstance(item["claim"], str)
            or not item["claim"].strip()
            or not isinstance(item["source_sha256"], str)
            or not _HASH.fullmatch(item["source_sha256"])
        ):
            raise ValueError("Invalid source evidence")


def history_policy(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve append-only review/revocation records without using their prose."""
    revisions: dict[str, dict[str, Any]] = {}
    active: set[str] = set()
    invalid: list[int] = []
    allowed_markers: set[int] = set()
    has_revisions = False
    for index, entry in enumerate(history):
        if not isinstance(entry, dict):
            continue
        has_revisions |= "reviewed_revision" in entry or "summary_revision_rollback" in entry
        try:
            if "reviewed_revision" in entry:
                record = entry["reviewed_revision"]
                manifest = record["manifest"]
                validate_manifest(manifest)
                digest = document_hash(manifest)
                if (
                    record.get("schema") != 1
                    or record.get("revision_id") != digest
                    or record.get("approval_sha256") != digest
                    or not isinstance(record.get("reviewer"), str)
                    or not record["reviewer"].strip()
                    or entry.get("summary") != manifest["candidate_summary"]
                    or entry.get("coverage") != manifest["coverage"]
                    or digest in revisions
                ):
                    raise ValueError("Invalid reviewed history record")
                revisions[digest] = {
                    "index": index,
                    "manifest": manifest,
                    "key": history_key(entry),
                }
                active.add(digest)
                allowed_markers.add(index)
            elif "summary_revision_rollback" in entry:
                digest = entry["summary_revision_rollback"]["revision_id"]
                old = revisions[digest]["manifest"]
                if (
                    digest not in active
                    or entry.get("summary") != old["previous_summary"]
                    or entry.get("coverage") != old["coverage"]
                ):
                    raise ValueError("Invalid summary rollback history")
                active.remove(digest)
                allowed_markers.add(index)
        except (ValueError, TypeError, KeyError, AttributeError):
            invalid.append(index)
    blocked = {
        tuple(key) for digest in active for key in revisions[digest]["manifest"]["supersedes"]
    }
    approved_keys = {revisions[digest]["key"] for digest in active}
    denied = {record["key"] for digest, record in revisions.items() if digest not in active}
    for index in invalid:
        try:
            denied.add(history_key(history[index]))
        except (ValueError, TypeError):
            # Unserializable corrupt metadata is never an eligible candidate.
            continue
    denied.difference_update(approved_keys)
    return {
        "blocked": blocked,
        "denied": denied,
        "invalid": invalid,
        "active_revisions": sorted(active),
        "allowed_markers": allowed_markers,
        "has_revisions": has_revisions,
    }


def select_summary(
    history: list[dict[str, Any]],
    proven: Callable[[dict[str, Any]], bool],
    *,
    require_coverage: bool = True,
) -> dict[str, Any]:
    policy = history_policy(history)
    for index in range(len(history) - 1, -1, -1):
        entry = history[index]
        if not isinstance(entry, dict) or (require_coverage and not entry.get("coverage")):
            continue
        if (
            "reviewed_revision" in entry or "summary_revision_rollback" in entry
        ) and index not in policy["allowed_markers"]:
            continue
        if history_key(entry) in policy["blocked"] | policy["denied"]:
            continue
        if all(proven(ref) for ref in entry.get("coverage", [])):
            return entry
    return {}


def revision_report(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Additive diagnostics only; no admission, semantic verdict or mutation."""
    policy = history_policy(history)
    return {
        "active_revision_ids": policy["active_revisions"],
        "invalid_history_indices": policy["invalid"],
        "superseded_identity_count": len(policy["blocked"]),
        "revoked_or_invalid_identity_count": len(policy["denied"]),
        "authority": "explicit_human_review_not_semantic_certification",
        "prompt_templates_changed": False,
        "additional_llm_calls": 0,
    }
