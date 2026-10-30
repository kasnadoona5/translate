"""Hash-bound PDF cue evidence; never infer footnotes from plain word+digits."""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from typing import Any

_DIGITS = str.maketrans("\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9"
                        "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669",
                        "01234567890123456789")
_SUPER = str.maketrans("\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079", "0123456789")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def paragraph_note_evidence(source: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Resolve recorded geometry against exact paragraph bytes and context.

    A superscript inside a formula, identifier or leading enumeration is not
    enough to prove a note. Unlocated and stale records stay diagnostic only.
    """
    from tarjomeh.parsers.pdf_parser import locate_source_marker

    cues, rejected = {}, []
    records = metadata.get("superscript_markers", []) or []
    if not isinstance(records, list):
        records = []
    for record in records:
        if not isinstance(record, dict):
            rejected.append({"reason": "invalid_record"})
            continue
        raw = str(record.get("text", "")).strip()
        marker = raw.translate(_DIGITS).translate(_SUPER)
        offset = locate_source_marker(source, record)
        reason = ""
        if not re.fullmatch(r"[0-9]{1,3}", marker):
            reason = "not_numeric_note"
        elif offset is None:
            reason = "source_location_unproven"
        elif record.get("offset") is not None and record["offset"] != offset:
            reason = "stale_source_offset"
        elif source[offset:offset + len(raw)] != raw:
            reason = "source_surface_mismatch"
        else:
            before, after = source[:offset], source[offset + len(raw):]
            stem = re.search(r"[^\W\d_]+$", before, re.UNICODE)
            decimal_tail = len(after) > 1 and after[0] in ".\u066b" and after[1].isdecimal()
            if (not before.strip() or (after and after[0].isalnum()) or decimal_tail
                    or (stem and len(stem.group()) < 3)
                    or re.search(r"[=+*/^]\s*\S*$", before)):
                reason = "formula_identifier_or_enumeration"
        if reason:
            rejected.append({"marker": marker, "reason": reason})
            continue
        cues[(offset, offset + len(raw))] = {
            "start": offset, "end": offset + len(raw), "marker": marker,
            "surface": raw, "record": deepcopy(record),
        }
    return {"schema": 1, "source_sha256": text_hash(source),
            "cues": list(cues.values()), "rejected": rejected,
            "authority": "located_pdf_superscript_not_target_semantic_alignment"}


def verified_note_cues(source: str, evidence: dict | None) -> list[dict]:
    if evidence is None:
        return []
    if (not isinstance(evidence, dict) or evidence.get("schema") != 1
            or evidence.get("source_sha256") != text_hash(source)):
        raise ValueError("Source note evidence is stale or malformed")
    if "paragraphs" in evidence:
        cues, previous_end = [], 0
        for part in evidence["paragraphs"]:
            start, end = part["span"]
            if (type(start) is not int or type(end) is not int
                    or not 0 <= previous_end <= start < end <= len(source)):
                raise ValueError("Invalid source note paragraph span")
            cues.extend({**cue, "start": start + cue["start"], "end": start + cue["end"]}
                        for cue in verified_note_cues(source[start:end], part["evidence"]))
            previous_end = end
        return cues
    rebuilt = paragraph_note_evidence(source, {
        "superscript_markers": [cue["record"] for cue in evidence.get("cues", [])]
    })
    if rebuilt["cues"] != evidence.get("cues"):
        raise ValueError("Source note evidence failed offset/context verification")
    return deepcopy(rebuilt["cues"])


def attach_chunk_note_evidence(document: Any, chunks: list) -> None:
    """Add evidence after chunking; source bytes, indices and parser stay intact."""
    paragraphs = document.all_paragraphs
    for chunk in chunks:
        indices = chunk.metadata.get("paragraph_indices", [])
        spans = chunk.metadata.get("source_paragraph_spans", [])
        hashes = chunk.metadata.get("source_paragraph_hashes", [])
        if not indices or len(indices) != len(spans) or len(indices) != len(hashes):
            continue
        parts = []
        for index, span, digest in zip(indices, spans, hashes, strict=True):
            if type(index) is not int or not 0 <= index < len(paragraphs):
                break
            start, end = span
            paragraph = paragraphs[index]
            if chunk.text[start:end] != paragraph.text or text_hash(paragraph.text) != digest:
                break
            parts.append({"span": list(span), "paragraph_index": index,
                          "evidence": paragraph_note_evidence(paragraph.text, paragraph.metadata)})
        else:
            if any(part["evidence"]["cues"] or part["evidence"]["rejected"] for part in parts):
                chunk.metadata["source_note_evidence_v1"] = {
                    "schema": 1, "source_sha256": text_hash(chunk.text), "paragraphs": parts,
                }


def scoped_note_evidence(source: str, original: str, evidence: dict | None) -> dict | None:
    """Retain geometry only for an exact, unique recovery-source slice."""
    if evidence is None:
        return None
    cues = verified_note_cues(original, evidence)
    if source == original:
        return evidence
    if not source or original.count(source) != 1:
        return None
    offset = original.index(source)
    records = []
    for cue in cues:
        if offset <= cue["start"] < cue["end"] <= offset + len(source):
            start, end = cue["start"] - offset, cue["end"] - offset
            records.append({"text": source[start:end], "offset": start,
                            "context_before": source[max(0, start - 24):start],
                            "context_after": source[end:end + 24]})
    return paragraph_note_evidence(source, {"superscript_markers": records})


def revision_audit_evidence(events: list, artifacts: dict) -> dict:
    """Report recorded proofs without fabricating historical note admission."""
    return {
        "source_note_admissions": [
            {"chunk_index": event.get("chunk_index"), "stage": event.get("payload", {}).get("stage"),
             "evidence": event["payload"]["source_note_evidence"]}
            for event in events if isinstance(event.get("payload"), dict)
            and event["payload"].get("source_note_evidence") is not None
        ],
        "rejected_repairs": [event for event in events
            if "reject" in event.get("event_type", "") and "note_marker" in str(event.get("payload", {}))],
        "delivered_note_cues": artifacts.get("render_identity_audit", {}).get("delivered", {}).get("source_note_cues"),
        "text_revision_receipts": [
            {key: value.get(key) for key in ("status", "revision_id", "reviewer", "after_snapshot_sha256", "publication")}
            for name, value in artifacts.items() if name.startswith("reviewed_text_revision:") and isinstance(value, dict)
        ],
        "publication_request_at_revision": artifacts.get("text_revision_publication_pending_v1"),
        "current_publication_authority": "consult_output_coverage_and_render_identity_not_the_historical_request",
        "historical_missing_evidence": "unknown_not_zero",
        "semantic_accuracy": "requires_source_based_human_review",
        "additional_llm_calls": 0,
    }


def geometry_plain_target_cues(source: str, target: str, evidence: dict) -> list[dict]:
    """Recognize an existing unique plain cue, never insert or move a digit.

    A source quantity with the same value, multiple target candidates, a label,
    formula or a leading quantity defeats this proof. Rich cues remain handled
    by the inventory's existing physical-span deduplication.
    """
    from collections import Counter
    from tarjomeh.quality.integrity import _structural_label, extract_numbers

    cues = verified_note_cues(source, evidence)
    counts = Counter(cue["marker"] for cue in cues)
    ordinary_source = source
    for cue in sorted(cues, key=lambda item: item["start"], reverse=True):
        ordinary_source = ordinary_source[:cue["start"]] + ordinary_source[cue["end"]:]
    numbers = extract_numbers(ordinary_source)
    cardinal = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                "seventeen", "eighteen", "nineteen", "twenty")
    ordinal = ("zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
               "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
               "seventeenth", "eighteenth", "nineteenth", "twentieth")
    result = []
    normalized = target.translate(_DIGITS)
    for marker, count in counts.items():
        value = int(marker)
        if count != 1 or numbers[marker] or value >= len(cardinal):
            continue
        if re.search(rf"\b(?:{cardinal[value]}|{ordinal[value]})\b", ordinary_source, re.IGNORECASE):
            continue
        matches = list(re.finditer(rf"(?<!\d){re.escape(marker)}(?!\d)", normalized))
        if len(matches) != 1:
            continue
        match = matches[0]
        before, after = target[:match.start()], target[match.end():]
        decimal_tail = len(after) > 1 and after[0] in ".\u066b" and after[1].isdecimal()
        if (not re.search(r"[^\W\d_]{2,}$", before, re.UNICODE)
                or (after and after[0].isalnum()) or decimal_tail
                or _structural_label(before[-80:], after[:80])
                or re.search(r"[=+*/^]\s*\S*$", before)
                or before.count("(") != before.count(")")
                or before.count("[") != before.count("]")):
            continue
        result.append({"start": match.start(), "end": match.end(), "marker": marker})
    return result
