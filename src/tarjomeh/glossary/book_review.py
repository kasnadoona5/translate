"""Source-paragraph matching for explicitly reviewed book terminology."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from tarjomeh.core.paragraph_protocol import split_paragraphs
from tarjomeh.glossary.compliance import ComplianceReport, Violation, target_present

_BODY_ROLES = frozenset({
    "body", "academic_argument", "expository_nonfiction",
    "narrative_prose", "dialogue",
})
_PUBLISHER_LINE = re.compile(
    r"(?:\b(?:press|publish(?:er|ing)?|books|ltd|inc|copyright|isbn)\b|\u00a9)",
    re.IGNORECASE,
)
REVIEWED_OCCURRENCES = "reviewed_occurrences"
# Reviews created from v10.40.3 on accept only evidence or per-occurrence scope.
REVIEW_VERSION = 2
_OCCURRENCE_KEYS = ("paragraph_index", "paragraph_sha256", "start", "end", "matched")


_TATWEEL = chr(0x0640)
_ZWNJ = chr(0x200C)


def persian_option_defect(target: str) -> str:
    """Return why a proposed Persian rendering is malformed, or "".

    A malformed option (a tatweel joiner, doubled or stray ZWNJ, or a
    Latin/Persian mixed token) is withheld from review and prompts; the
    English candidate itself stays available.
    """
    from tarjomeh.quality.integrity import mixed_script_artifacts

    value = str(target or "").strip()
    if not value:
        return ""
    if _TATWEEL in value:
        return "tatweel_in_persian_option"
    if _ZWNJ * 2 in value:
        return "double_zwnj_in_persian_option"
    if re.search(rf"(?:^|\s){_ZWNJ}|{_ZWNJ}(?:\s|$)", value):
        return "stray_zwnj_in_persian_option"
    if mixed_script_artifacts(value):
        return "mixed_script_persian_option"
    return ""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _term_pattern(source: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(source)}(?!\w)", re.IGNORECASE)


def _is_publisher_line(paragraph: str) -> bool:
    return len(paragraph) < 140 and bool(_PUBLISHER_LINE.search(paragraph))


_PERSIAN_WORD = (
    chr(0x0621) + "-" + chr(0x063A) + chr(0x0641) + "-" + chr(0x064A)
    + chr(0x066E) + "-" + chr(0x06D3) + chr(0x06FA) + "-" + chr(0x06FF)
)
_JOINERS = chr(0x200C) + chr(0x200D)


def _rendering_count(target: str, translation: str) -> int:
    """Count word-initial occurrences of a Persian rendering (suffixes allowed)."""
    from tarjomeh.glossary.compliance import _normalise_persian

    normalised = _normalise_persian(target)
    parts = [
        f"[{_JOINERS}]*".join(re.escape(char) for char in part)
        for part in re.split(rf"[\s{_JOINERS}]+", normalised)
        if part
    ]
    if not parts:
        return 0
    flexible = rf"[\s{_JOINERS}]*".join(parts)
    return len(re.findall(
        rf"(?<![{_PERSIAN_WORD}]){flexible}", _normalise_persian(translation)
    ))


def _reviewed_paragraph_decision(
    term: dict[str, Any],
    paragraph: str,
    global_index: int | None,
) -> dict[str, Any] | None:
    """Decide one paragraph for a per-occurrence approval, or None if unticked.

    Mandatory only when every occurrence of the term in the paragraph was
    ticked and each ticked identity (hash, offsets, text) still verifies.
    """
    if global_index is None:
        return None
    approved = [
        item for item in term.get("approved_occurrences", []) or []
        if isinstance(item, dict) and item.get("paragraph_index") == global_index
    ]
    if not approved:
        return None
    if any(item.get("paragraph_sha256") != _sha(paragraph) for item in approved):
        return {"reason": "approved_occurrence_changed", "count": 0}
    spans = {
        (match.start(), match.end())
        for match in _term_pattern(str(term.get("source", ""))).finditer(paragraph)
    }
    ticked: set[tuple[int, int]] = set()
    for item in approved:
        start, end = int(item.get("start", -1)), int(item.get("end", -1))
        if (start, end) not in spans or paragraph[start:end] != item.get("matched"):
            return {"reason": "approved_occurrence_changed", "count": 0}
        ticked.add((start, end))
    if ticked != spans:
        return {"reason": "partial_occurrence_approval", "count": len(spans)}
    return {"reason": "", "count": len(spans)}


def body_paragraph_index(document: Any, chunks: list[Any]) -> list[dict[str, Any]]:
    """Locate every body-prose paragraph by chunk and verified span, without text.

    Uses the same body filter as term sampling (no front or back matter,
    references, index, headings or tables) and the chunk's verified paragraph
    spans, so each location can be re-read and re-verified later.
    """
    from tarjomeh.context.book_term_candidates import body_term_paragraphs

    eligible = {id(paragraph) for paragraph in body_term_paragraphs(document)}
    body_indices = {
        index for index, paragraph in enumerate(document.all_paragraphs)
        if id(paragraph) in eligible
    }
    entries: list[dict[str, Any]] = []
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        paragraphs = _verified_source_paragraphs(chunk.text, metadata)
        indices = metadata.get("paragraph_indices") or []
        if paragraphs is None or len(indices) != len(paragraphs):
            continue
        for local, (global_index, paragraph) in enumerate(
            zip(indices, paragraphs, strict=True)
        ):
            if int(global_index) in body_indices:
                entries.append({
                    "chunk_index": int(chunk.index),
                    "local_paragraph": local,
                    "paragraph_index": int(global_index),
                    "paragraph_sha256": _sha(paragraph),
                    "chapter_position": int(metadata.get("chapter_position", 1) or 1),
                })
    return entries


def find_term_occurrences(
    source: str,
    index_entries: list[dict[str, Any]],
    chunk_records: dict[int, tuple[str, dict[str, Any]]],
    *,
    include_paragraph: bool = False,
) -> list[dict[str, Any]]:
    """List every body occurrence of *source* with a unique identity.

    An occurrence is identified by its global paragraph index, the paragraph
    hash, its exact character offsets and the matched text, so identical
    paragraphs in two places, or two uses in one paragraph, never collide.
    Short publisher lines never count as occurrences.
    """
    term = str(source or "").strip()
    if not term:
        return []
    pattern = _term_pattern(term)
    cache: dict[int, list[str] | None] = {}
    occurrences: list[dict[str, Any]] = []
    for entry in index_entries:
        chunk_index = int(entry.get("chunk_index", -1))
        record = chunk_records.get(chunk_index)
        if record is None:
            continue
        if chunk_index not in cache:
            cache[chunk_index] = _verified_source_paragraphs(*record)
        paragraphs = cache[chunk_index]
        local = int(entry.get("local_paragraph", -1))
        if paragraphs is None or not 0 <= local < len(paragraphs):
            continue
        paragraph = paragraphs[local]
        if _sha(paragraph) != entry.get("paragraph_sha256") or _is_publisher_line(paragraph):
            continue
        for match in pattern.finditer(paragraph):
            occurrence = {
                "chunk_index": chunk_index,
                "local_paragraph": local,
                "paragraph_index": int(entry["paragraph_index"]),
                "paragraph_sha256": entry["paragraph_sha256"],
                "chapter_position": int(entry.get("chapter_position", 1) or 1),
                "start": match.start(),
                "end": match.end(),
                "matched": match.group(),
            }
            if include_paragraph:
                occurrence["paragraph"] = paragraph
            occurrences.append(occurrence)
    return occurrences


def verify_ticked_occurrences(
    ticked: list[dict[str, Any]],
    occurrences: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Match submitted occurrence identities against the recomputed index."""
    known = {tuple(item[key] for key in _OCCURRENCE_KEYS): item for item in occurrences}
    verified: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in ticked:
        if not isinstance(item, dict):
            return [], "invalid_occurrence"
        try:
            key = (
                int(item["paragraph_index"]), str(item["paragraph_sha256"]),
                int(item["start"]), int(item["end"]), str(item["matched"]),
            )
        except (KeyError, TypeError, ValueError):
            return [], "invalid_occurrence"
        if key in seen:
            return [], "duplicate_occurrence"
        if key not in known:
            return [], "occurrence_not_found_in_source"
        seen.add(key)
        occurrence = known[key]
        verified.append({
            name: occurrence[name]
            for name in (
                "chunk_index", "local_paragraph", "chapter_position", *_OCCURRENCE_KEYS,
            )
        })
    return verified, ""


def _verified_source_paragraphs(text: str, metadata: dict[str, Any]) -> list[str] | None:
    spans = metadata.get("source_paragraph_spans")
    hashes = metadata.get("source_paragraph_hashes")
    indices = metadata.get("paragraph_indices")
    if isinstance(spans, list) and isinstance(hashes, list) and isinstance(indices, list):
        if not spans or not (len(spans) == len(hashes) == len(indices)):
            return None
        paragraphs = []
        for span, digest in zip(spans, hashes, strict=True):
            if (not isinstance(span, list) or len(span) != 2
                    or not all(isinstance(value, int) for value in span)):
                return None
            start, end = span
            if start < 0 or end <= start or end > len(text):
                return None
            value = text[start:end]
            if hashlib.sha256(value.encode("utf-8")).hexdigest() != digest:
                return None
            paragraphs.append(value)
        return paragraphs
    paragraphs = split_paragraphs(text)
    return paragraphs if len(paragraphs) == 1 else None


def resolve_reviewed_book_terms(
    source_text: str,
    metadata: dict[str, Any],
    chapter_position: int,
    decisions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return exact in-scope matches and review-only uncertain occurrences."""
    relevant = [
        term for term in decisions
        if term.get("status") == "approved"
        and str(term.get("source", "")).strip()
        and (not term.get("chapter_positions")
             or chapter_position in term["chapter_positions"])
        and re.search(
            rf"(?<!\w){re.escape(str(term.get('source', '')))}(?!\w)",
            source_text, re.IGNORECASE,
        )
    ]
    if not relevant:
        return [], []
    paragraphs = _verified_source_paragraphs(source_text, metadata)
    if paragraphs is None:
        return [], [{"reason": "uncertain_paragraph_alignment"}]
    roles = metadata.get("structural_roles")
    if not isinstance(roles, list) or len(roles) != len(paragraphs):
        return [], [{"reason": "uncertain_paragraph_role"}]
    matches: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    global_indices = metadata.get("paragraph_indices")
    if not isinstance(global_indices, list) or len(global_indices) != len(paragraphs):
        global_indices = [None] * len(paragraphs)
    for paragraph_index, (paragraph, role) in enumerate(zip(paragraphs, roles, strict=True)):
        candidates: dict[str, list[dict[str, Any]]] = {}
        occurrence_counts: dict[int, int] = {}
        for term in relevant:
            source = str(term.get("source", "")).strip()
            if not source:
                continue
            scope_mode = str(term.get("scope_mode", "evidence_paragraph"))
            if scope_mode == REVIEWED_OCCURRENCES:
                decision = _reviewed_paragraph_decision(
                    term, paragraph, global_indices[paragraph_index]
                )
                if decision is None:
                    continue
                if decision["reason"]:
                    review.append({
                        "paragraph_index": paragraph_index,
                        "source": source,
                        "reason": decision["reason"],
                        "source_occurrence_count": decision["count"],
                    })
                    continue
                candidates.setdefault(source.casefold(), []).append(term)
                occurrence_counts[id(term)] = int(decision["count"])
                continue
            if scope_mode == "evidence_paragraph" and (
                hashlib.sha256(paragraph.encode("utf-8")).hexdigest()
                != str(term.get("source_evidence_sha256", ""))
            ):
                continue
            if scope_mode not in {"evidence_paragraph", "all_body"}:
                review.append({"source": source, "reason": "invalid_approved_scope"})
                continue
            pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)", re.IGNORECASE)
            count = len(pattern.findall(paragraph))
            if count:
                candidates.setdefault(source.casefold(), []).append(term)
                occurrence_counts[id(term)] = count
        for source_key, options in candidates.items():
            scope_reason = ""
            if role not in _BODY_ROLES:
                scope_reason = "non_body_paragraph"
            elif len(paragraph) < 140 and _PUBLISHER_LINE.search(paragraph):
                scope_reason = "possible_publisher_line"
            elif len(options) != 1:
                scope_reason = "ambiguous_approved_sense"
            term = options[0]
            if term.get("keep_original") and str(term.get("source", "")) not in paragraph:
                scope_reason = "original_not_exact_in_source"
            item = {
                "paragraph_index": paragraph_index,
                "source": term["source"],
                "target": term.get("target", ""),
                "keep_original": bool(term.get("keep_original")),
                "source_paragraph_hash": hashlib.sha256(paragraph.encode("utf-8")).hexdigest(),
                "sense_id": str(term.get("sense_id", "")),
            }
            if id(term) in occurrence_counts:
                item["source_occurrence_count"] = occurrence_counts[id(term)]
            if scope_reason:
                review.append({**item, "reason": scope_reason})
                continue
            matches.append(item)
            if int(item.get("source_occurrence_count", 1)) > 1:
                # Paragraph-level compliance cannot prove each rendering, so a
                # repeated approved term is given to the translator but is
                # always reported for review, never as verified.
                review.append({**item, "reason": "multiple_occurrences_unverifiable"})
    return matches, review


def book_term_review_evidence(
    review: dict[str, Any], events: list[dict[str, Any]], chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Report current approvals separately from scope and lexical presence."""
    proposals = [item for item in (review.get("proposals") or []) if isinstance(item, dict)]
    approved = [item for item in proposals if item.get("status") == "approved"]
    ticks = [tick for term in approved for tick in (term.get("approved_occurrences") or [])
             if isinstance(tick, dict)]
    by_chunk: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        if not isinstance(event.get("chunk_index"), int):
            continue
        index = event["chunk_index"]
        if event.get("event_type") == "chunk_started":
            by_chunk[index] = []
        by_chunk.setdefault(index, []).append(event)
    retained = {int(chunk["chunk_index"]): chunk for chunk in chunks
                if chunk.get("status") in {"completed", "needs_review"}}
    evidence_events = []
    scoped = set()
    lexical = []
    for index, generation in by_chunk.items():
        evidence_events.extend(event for event in generation if event.get("event_type") in {
            "book_term_scope_resolved", "book_term_final_compliance",
        })
        latest_scope = next((event.get("payload", {}) for event in reversed(generation)
                             if event.get("event_type") == "book_term_scope_resolved"), {})
        chunk = retained.get(index, {})
        metadata = chunk.get("metadata") or {}
        metadata_valid = True
        if isinstance(metadata, str):
            import json

            try:
                metadata = json.loads(metadata)
            except (ValueError, TypeError):
                metadata = {}
                metadata_valid = False
        if not isinstance(metadata, dict):
            metadata = {}
            metadata_valid = False
        source_parts = _verified_source_paragraphs(str(chunk.get("text", "")), metadata)
        targets = split_paragraphs(str(chunk.get("translation") or ""))
        for match in latest_scope.get("matched", []):
            paragraph = match.get("paragraph_index")
            scoped.add((index, paragraph, match.get("source_paragraph_hash")))
            proven = bool(
                metadata_valid and source_parts is not None and len(source_parts) == len(targets)
                and isinstance(paragraph, int) and 0 <= paragraph < len(source_parts)
                and _sha(source_parts[paragraph]) == match.get("source_paragraph_hash")
            )
            lexical.append({
                "db_chunk_index": index, "ui_chunk_number": index + 1, **match,
                "presence": (
                    "lexically_present" if target_present(
                        str(match.get("target", "")), targets[paragraph]
                    )
                    else "missing"
                ) if proven else "REVIEW",
                "semantic_accuracy": "requires_source_based_human_review",
            })
    return {
        "phase": review.get("phase", "off"), "source_sha256": review.get("source_sha256"),
        "approved_proposal_count": len(approved), "ticked_occurrence_count": len(ticks),
        "ticked_paragraph_count": len({
            (tick.get("paragraph_index"), tick.get("paragraph_sha256")) for tick in ticks
        }),
        "scoped_paragraph_count": len(scoped),
        "lexically_present_paragraph_term_count": sum(
            item["presence"] == "lexically_present" for item in lexical
        ),
        "proposal_decisions": proposals, "scope_and_compliance_events": evidence_events,
        "paragraph_term_presence": lexical,
        "policy": (
            "approval, scope and lexical presence are distinct; none proves semantic accuracy"
        ),
    }


def check_reviewed_book_terms(
    translation: str,
    source_text: str,
    metadata: dict[str, Any],
    matches: list[dict[str, Any]],
) -> tuple[ComplianceReport, list[dict[str, Any]]]:
    """Check only proven source/target paragraph pairs, never the whole chunk."""
    source_parts = _verified_source_paragraphs(source_text, metadata)
    target_parts = split_paragraphs(translation)
    if source_parts is None or len(source_parts) != len(target_parts):
        return ComplianceReport(), [{"reason": "uncertain_target_paragraph_alignment"}]
    violations: list[Violation] = []
    review: list[dict[str, Any]] = []
    checked = 0
    for match in matches:
        index = match["paragraph_index"]
        if type(index) is not int or not 0 <= index < len(source_parts) or (
            hashlib.sha256(source_parts[index].encode("utf-8")).hexdigest()
            != match["source_paragraph_hash"]
        ):
            return ComplianceReport(), [{"reason": "source_paragraph_changed"}]
        source_count = len(_term_pattern(str(match["source"])).findall(source_parts[index]))
        recorded_count = match.get("source_occurrence_count")
        if not source_count or (
            recorded_count is not None
            and (type(recorded_count) is not int or recorded_count != source_count)
        ):
            review.append({
                "paragraph_index": index,
                "source": match["source"],
                "reason": "source_occurrence_count_mismatch",
                "source_occurrence_count": source_count,
                "recorded_occurrence_count": recorded_count,
            })
            continue
        present = target_present(str(match["target"]), target_parts[index])
        if not present:
            violations.append(Violation(
                term=str(match["source"]),
                expected=str(match["target"]),
                chunk_location=f"Paragraph {index + 1}",
            ))
        if source_count > 1:
            # Presence proves one rendering somewhere in the paragraph, not
            # one per occurrence; repeated terms are never counted as checked.
            rendered = _rendering_count(str(match["target"]), target_parts[index])
            reasons = ["multiple_occurrences_unverifiable"]
            if rendered < source_count:
                reasons.append("rendering_count_short")
            review.append({
                "paragraph_index": index,
                "source": match["source"],
                "target": match["target"],
                "reasons": reasons,
                "source_occurrence_count": source_count,
                "target_rendering_count": rendered,
                "compliance": "lexically_present" if present else "missing",
            })
        else:
            checked += 1
    return ComplianceReport(violations=violations, total_checked=checked), review


def format_reviewed_book_terms(matches: list[dict[str, Any]]) -> str:
    if not matches:
        return ""
    lines = [
        "Approved book terms. Apply each rendering only in its named source "
        "paragraph and sense; preserve all other paragraphs and the source meaning:"
    ]
    for match in matches:
        count = int(match.get("source_occurrence_count", 1) or 1)
        repeated = f" (each of its {count} occurrences)" if count > 1 else ""
        lines.append(
            f"- Source paragraph {match['paragraph_index'] + 1}: "
            f"{match['source']} -> {match['target']}{repeated}"
        )
    return "\n".join(lines)
