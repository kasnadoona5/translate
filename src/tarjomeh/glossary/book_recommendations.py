"""Non-binding, paragraph-identified book-term advice and report-only presence."""

from __future__ import annotations

import json
from typing import Any

from tarjomeh.glossary.book_review import (
    BOOK_WIDE_RECOMMENDED,
    _sha,
    _term_pattern,
    _verified_source_paragraphs,
    find_term_occurrences,
)
from tarjomeh.glossary.compliance import target_present

_PREFIX = "BOOK_TERM_RECOMMENDATIONS_JSON="
_POLICY = (
    "Optional book-term recommendations, NOT mandatory. They are approved only in "
    "their evidence paragraph; elsewhere they are advice, not approval of a sense. "
    "Use different wording when source meaning or fluent academic Persian requires it. "
    "Non-use alone is NOT an MQM issue, score deduction, reason to edit, REVIEW, "
    "or memory/style penalty. Genuine source-meaning errors still require normal review."
    " paragraph_index is zero-based: 0 means source paragraph p1."
)


def resolve_book_recommendations(
    source: str,
    metadata: dict[str, Any],
    chunk_index: int,
    review: dict[str, Any],
    body_index: dict[str, Any],
    mandatory_terms: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"recommendations": [], "deferred": [], "authority": "advisory_only"}
    if review.get("phase") != "started" or review.get("review_version") != 3:
        return result
    proposals = [
        item
        for item in review.get("proposals", [])
        if isinstance(item, dict) and item.get("status") == "approved"
    ]
    for term in proposals:
        if term.get("scope_mode") != BOOK_WIDE_RECOMMENDED:
            continue
        english, target = str(term.get("source", "")).strip(), str(term.get("target", "")).strip()
        if not english or not target or not _term_pattern(english).search(source):
            continue
        reason = ""
        if not review.get("source_sha256") or (
            body_index.get("source_sha256") != review["source_sha256"]
        ):
            reason = "body_index_source_identity_unavailable"
        options = [
            item
            for item in proposals
            if str(item.get("source", "")).casefold() == english.casefold()
        ]
        if len(options) != 1:
            reason = "ambiguous_approved_sense"
        if any(
            str(item.get("source", "")).casefold() == english.casefold()
            and str(item.get("target", "")) != target
            for item in mandatory_terms or []
        ):
            reason = "curated_mandatory_conflict"
        # Recommendation scope never grants a new English-original permission.
        if term.get("keep_original"):
            reason = "optional_original_insertion_policy_not_proven"
        if reason:
            result["deferred"].append({"source": english, "reason": reason})
            continue
        entries = [
            item for item in body_index.get("entries", []) if item.get("chunk_index") == chunk_index
        ]
        if _verified_source_paragraphs(source, metadata) is None:
            result["deferred"].append(
                {"source": english, "reason": "source_paragraph_identity_unavailable"}
            )
            continue
        locations = find_term_occurrences(
            english,
            entries,
            {chunk_index: (source, metadata)},
            include_paragraph=True,
        )
        grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
        for location in locations:
            if term.get("chapter_positions") and (
                location["chapter_position"] not in term["chapter_positions"]
            ):
                continue
            if location["paragraph_sha256"] == term.get("source_evidence_sha256"):
                continue
            grouped.setdefault(
                (location["paragraph_index"], location["paragraph_sha256"]), []
            ).append(location)
        for (global_index, digest), occurrences in grouped.items():
            first = occurrences[0]
            identity = _sha(
                json.dumps(
                    [
                        review["source_sha256"],
                        english,
                        target,
                        term.get("sense_id", ""),
                        chunk_index,
                        global_index,
                        digest,
                    ],
                    ensure_ascii=True,
                )
            )
            result["recommendations"].append(
                {
                    "recommendation_id": identity,
                    "scope_mode": BOOK_WIDE_RECOMMENDED,
                    "source": english,
                    "target": target,
                    "sense_id": term.get("sense_id", ""),
                    "english_explanation": str(term.get("english_explanation", ""))[:300],
                    "book_source_sha256": review["source_sha256"],
                    "chunk_index": chunk_index,
                    "paragraph_index": first["local_paragraph"],
                    "global_paragraph_index": global_index,
                    "source_paragraph_hash": digest,
                    "chapter_position": first["chapter_position"],
                    "source_quote": first["paragraph"][
                        max(0, first["start"] - 150) : first["end"] + 150
                    ],
                    "occurrences": [
                        {"start": item["start"], "end": item["end"], "matched": item["matched"]}
                        for item in occurrences
                    ],
                }
            )
    return result


def format_book_recommendations(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    prompt_items = [
        {
            key: item[key]
            for key in (
                "recommendation_id",
                "source",
                "target",
                "sense_id",
                "english_explanation",
                "paragraph_index",
                "source_paragraph_hash",
            )
        }
        for item in items
    ]
    return _POLICY + "\n" + _PREFIX + json.dumps(prompt_items, ensure_ascii=True)


def recommendations_from_context(context: str) -> list[dict[str, Any]]:
    for line in context.splitlines():
        if line.startswith(_PREFIX):
            try:
                values = json.loads(line[len(_PREFIX) :])
            except (ValueError, TypeError):
                return []
            if not isinstance(values, list):
                return []
            return [
                item
                for item in values
                if isinstance(item, dict)
                and all(
                    isinstance(item.get(key), str) and item[key]
                    for key in (
                        "recommendation_id",
                        "source_paragraph_hash",
                        "source",
                        "target",
                    )
                )
            ]
    return []


def parse_advisory_observations(
    value: Any,
    recommendations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Optional data errors never invalidate core critique or request repair."""
    if value is None:
        return [], []
    if not isinstance(value, list):
        return [], ["advisory_term_observations_must_be_array"]
    known = {item["recommendation_id"]: item for item in recommendations}
    accepted, ignored = [], []
    for item in value:
        reference = (
            known.get(str(item.get("recommendation_id", ""))) if isinstance(item, dict) else None
        )
        if not reference or item.get("source_paragraph_hash") != reference["source_paragraph_hash"]:
            ignored.append("unknown_advisory_recommendation_or_paragraph")
            continue
        accepted.append(
            {
                "recommendation_id": reference["recommendation_id"],
                "source_paragraph_hash": reference["source_paragraph_hash"],
                "note": str(item.get("note", ""))[:500],
                "authority": "advisory_only",
            }
        )
    return accepted, ignored


def recommendation_presence(
    source: str,
    translation: str,
    metadata: dict[str, Any],
    recommendations: list[dict[str, Any]],
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Read only hash-bound canonical paragraph units, never infer alignment."""
    sources = _verified_source_paragraphs(source, metadata)
    offsets = identity.get("target_offsets", [])
    target_hashes = identity.get("target_hashes", [])
    proven = bool(
        sources
        and not identity.get("reconstructed", True)
        and identity.get("canonical_target_hash") == _sha(translation)
        and identity.get("source_hashes") == [_sha(item) for item in sources]
        and identity.get("paragraph_indices") == metadata.get("paragraph_indices")
        and len(offsets) == len(target_hashes) == len(sources)
    )
    rows = []
    for item in recommendations:
        row = {
            **item,
            "canonical_target_hash": _sha(translation),
            "target_excerpt": "",
            "presence": "alignment_unavailable_usage_unknown",
        }
        local = item["paragraph_index"]
        if proven and type(local) is int and 0 <= local < len(sources):
            span = offsets[local]
            if (
                isinstance(span, list)
                and len(span) == 2
                and all(type(value) is int for value in span)
                and 0 <= span[0] < span[1] <= len(translation)
            ):
                target = translation[span[0] : span[1]]
                if (
                    _sha(target) == target_hashes[local]
                    and _sha(sources[local]) == item["source_paragraph_hash"]
                ):
                    row.update(
                        {
                            "target_excerpt": target[:700],
                            "presence": (
                                "lexically_present"
                                if target_present(item["target"], target)
                                else "preferred_wording_not_found"
                            ),
                        }
                    )
        rows.append(row)
    return {
        "authority": "report_only",
        "locations": rows,
        "normalization": "existing glossary target_present; no synonym equivalence",
        "semantic_accuracy": "not_verified_by_lexical_presence",
        "occurrence_counts": {
            state: sum(len(row["occurrences"]) for row in rows if row["presence"] == state)
            for state in (
                "lexically_present",
                "preferred_wording_not_found",
                "alignment_unavailable_usage_unknown",
            )
        },
    }


def recommendation_evidence(
    events: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    identities: dict[str, Any],
) -> dict[str, Any]:
    """Recheck the current generation against saved final canonical identities."""
    scopes: dict[int, dict[str, Any]] = {}
    for event in events:
        index = event.get("chunk_index")
        if type(index) is not int:
            continue
        if event.get("event_type") == "chunk_started":
            scopes.pop(index, None)
        if event.get("event_type") == "book_term_recommendations_resolved":
            scopes[index] = event.get("payload") or {}
    locations, deferred = [], []
    for chunk in chunks:
        if chunk.get("status") not in {"completed", "needs_review"}:
            continue
        index = int(chunk["chunk_index"])
        scope = scopes.get(index, {})
        metadata = chunk.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except ValueError:
                metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        report = recommendation_presence(
            str(chunk.get("text", "")),
            str(chunk.get("translation") or ""),
            metadata,
            scope.get("recommendations", []),
            identities.get(str(index), {}),
        )
        locations.extend(report["locations"])
        deferred.extend({"chunk_index": index, **item} for item in scope.get("deferred", []))
    return {
        "authority": "report_only",
        "locations": locations,
        "deferred": deferred,
        "occurrence_counts": {
            state: sum(len(row["occurrences"]) for row in locations if row["presence"] == state)
            for state in (
                "lexically_present",
                "preferred_wording_not_found",
                "alignment_unavailable_usage_unknown",
            )
        },
        "semantic_accuracy": "not_verified_by_lexical_presence",
        "normalization": "existing glossary target_present; no synonym equivalence",
    }
