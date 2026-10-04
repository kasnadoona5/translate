"""Bounded, hash-addressed evidence for context actually sent to the translator."""

from __future__ import annotations

import hashlib
from typing import Any

CONTEXT_COMPONENTS = frozenset({"glossary", "layer_2", "style", "book_context"})


def component_evidence(
    components: dict[str, Any],
    existing: dict[str, Any],
    *,
    max_entries: int = 128,
    max_chars: int = 12000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deduplicate complete bounded components; never capture keys or full prompts.

    These are storage bounds, not prompt limits. Larger components are hashed
    and explicitly unavailable rather than clipped and presented as complete.
    """
    added: dict[str, Any] = {}
    references: dict[str, Any] = {}
    for name, value in components.items():
        if not isinstance(value, str):
            references[name] = {
                "sha256": None, "chars": None, "available": False,
                "reason": "non_text_component",
            }
            continue
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        key = f"{name}:{digest}"
        reference: dict[str, Any] = {"sha256": digest, "chars": len(value)}
        if name not in CONTEXT_COMPONENTS or not value:
            reference["available"] = False
            reference["reason"] = "hash_only_component" if value else "empty"
        elif (
            isinstance(existing.get(key), dict)
            and existing[key].get("text") == value
            and existing[key].get("sha256") == digest
        ):
            reference.update({"available": True, "artifact_entry": key})
        elif len(value) > max_chars:
            reference.update({"available": False, "reason": "component_storage_bound"})
        elif len(existing) + len(added) >= max_entries:
            reference.update({"available": False, "reason": "job_storage_bound"})
        else:
            added[key] = {
                "component": name,
                "sha256": digest,
                "text": value,
                "chars": len(value),
                "complete": True,
            }
            reference.update({"available": True, "artifact_entry": key})
        references[name] = reference
    return added, references
