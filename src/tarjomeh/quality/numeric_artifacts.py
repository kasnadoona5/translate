"""Source-bound numeric projections for proven PDF text-layer artifacts."""

from __future__ import annotations

import re

from tarjomeh.quality.note_evidence import text_hash, verified_note_cues

_DIGITS = str.maketrans("\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9"
                        "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669",
                        "01234567890123456789")
_SOURCE_FORMULA = re.compile(r"(?<![\w+\-])C[ \t]+[O0][ \t]+2(?![\w.\u066b+\-])")
_TARGET_FORMULA = re.compile(
    r"(?<![\w+\-])(?:CO[2\u2082]|\u062f\u06cc[ \u200c-]*\u0627\u06a9\u0633\u06cc\u062f[ \u200c]+"
    r"\u06a9\u0631\u0628\u0646)(?![\w+\-])"
)
_SOURCE_RATIO_TAIL = re.compile(r"(?<![\w.:/])([0-9]+)[ \t]*:[ \t]*([0-9]+)\.$")
# A translated date can precede the ratio with the same connector ("to").
_TARGET_RATIO = re.compile(
    r"(?<![\w.:/+\-])(?=(([0-9]+)(?:[ \t]*:[ \t]*|[ \t]+\u0628\u0647[ \t]+)"
    r"([0-9]+)(?![\w:/+\-]|[.\u066b][0-9])))"
)
_PARAGRAPH = re.compile(r"\n\s*\n")


def line_formula_records(spans: list[dict], text: str, direction: tuple) -> list[dict]:
    """Record an isolated subscript on the same horizontal line as C O/0."""
    if direction != (1.0, 0.0) or not _SOURCE_FORMULA.search(text):
        return []
    visible = [span for span in spans if span.get("text", "").strip()]
    geometry_count = 0
    for left, right in zip(visible, visible[1:]):
        if not re.search(r"(?<!\w)C[ \t]+[O0]$", left["text"].rstrip()) or right["text"].strip() != "2":
            continue
        size = left["size"]
        offset = right["origin"][1] - left["origin"][1]
        gap = right["bbox"][0] - left["bbox"][2]
        if (size > 0 and 0 < right["size"] <= 0.75 * size
                and 0.04 * size <= offset <= 0.55 * size and -0.2 * size <= gap <= size):
            geometry_count += 1
    matches = list(_SOURCE_FORMULA.finditer(text))
    if geometry_count != 1 or len(matches) != 1:
        return []
    match = matches[0]
    return [{"surface": match.group(), "before": text[max(0, match.start() - 20):match.start()],
             "after": text[match.end():match.end() + 4]}]


def attach_formula_evidence(document, chunks: list) -> None:
    """Bind the parser's geometry records to exact saved paragraph slices."""
    paragraphs = document.all_paragraphs
    for chunk in chunks:
        records = []
        indices = chunk.metadata.get("paragraph_indices", [])
        spans = chunk.metadata.get("source_paragraph_spans", [])
        hashes = chunk.metadata.get("source_paragraph_hashes", [])
        if not len(indices) == len(spans) == len(hashes):
            continue
        for index, (start, end), digest in zip(indices, spans, hashes):
            if type(index) is not int or not 0 <= index < len(paragraphs):
                records = []
                break
            paragraph = paragraphs[index]
            part = paragraph.text
            if not 0 <= start < end <= len(chunk.text) or chunk.text[start:end] != part or text_hash(part) != digest:
                records = []
                break
            located = {}
            for record in paragraph.metadata.get("formula_geometry", []):
                matches = [m for m in _SOURCE_FORMULA.finditer(part)
                           if m.group() == record["surface"]
                           and part[:m.start()].endswith(record["before"])
                           and part[m.end():].startswith(record["after"])]
                if len(matches) == 1:
                    match = matches[0]
                    located[match.span()] = match.group()
            if located:
                records.append({"paragraph": [start, end], "paragraph_sha256": digest,
                                "spans": [[start + a, start + b, surface] for (a, b), surface in sorted(located.items())]})
        if records:
            chunk.metadata["source_formula_evidence_v1"] = {"source_sha256": text_hash(chunk.text), "records": records}


def _parts(text: str) -> list[tuple[int, int]]:
    starts = [0]
    ends = []
    for match in _PARAGRAPH.finditer(text):
        ends.append(match.start())
        starts.append(match.end())
    return list(zip(starts, [*ends, len(text)]))


def _mask(text: str, spans: list[tuple[int, int]]) -> str:
    for start, end in sorted(set(spans), reverse=True):
        text = text[:start] + " " * (end - start) + text[end:]
    return text


def _verified_formula_spans(source: str, formula: dict):
    for record in formula.get("records") or []:
        if not isinstance(record, dict):
            continue
        bounds = record.get("paragraph")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or any(type(n) is not int for n in bounds):
            continue
        a, b = bounds
        if not 0 <= a < b <= len(source) or text_hash(source[a:b]) != record.get("paragraph_sha256"):
            continue
        for span in record.get("spans") or []:
            if not isinstance(span, (list, tuple)) or len(span) != 3:
                continue
            start, end, surface = span
            if (type(start) is int and type(end) is int and a <= start < end <= b
                    and source[start:end] == surface and _SOURCE_FORMULA.fullmatch(surface)):
                yield start, end


def project_numeric_artifacts(source: str, target: str, notes: dict | None,
                              formula: dict | None, *, formula_source: str | None = None
                              ) -> tuple[str, str, list[dict], list[dict]]:
    """Return matched numeric views, unmet typed obligations, and their locations."""
    obligations = []
    source_masks = []
    for cue in verified_note_cues(source, notes):
        start, end = cue["start"], cue["end"]
        prefix = source[max(0, start - 100):start].translate(_DIGITS)
        ratio = _SOURCE_RATIO_TAIL.search(prefix)
        if ratio and start >= 2 and source[start - 2].isdecimal():
            left = max(0, start - 100) + ratio.start()
            if start >= 100 and ratio.start() == 0:
                continue
            obligations.append(("ratio", left, start - 1, ratio.groups()))
            source_masks.append((start, end))
    original = formula_source if formula_source is not None else source
    if formula and formula.get("source_sha256") == text_hash(original):
        offset = 0 if original == source else original.find(source) if source and original.count(source) == 1 else -1
        if offset >= 0:
            for start, end in _verified_formula_spans(original, formula):
                if offset <= start < end <= offset + len(source):
                    obligations.append(("co2", start - offset, end - offset, ()))
    if not obligations:
        return source, target, [], []

    sources, targets = _parts(source), _parts(target)
    if len(sources) != len(targets):
        return source, target, [], []
    missing, evidence, used = [], [], set()
    target_masks = []
    for kind, start, end, value in sorted(set(obligations), key=lambda item: item[1]):
        para = next((i for i, (a, b) in enumerate(sources) if a <= start < end <= b), None)
        if para is None:
            continue
        a, b = targets[para]
        pattern = _TARGET_RATIO if kind == "ratio" else _TARGET_FORMULA
        candidates = [(a + m.start(1), a + m.end(1)) if kind == "ratio" else (a + m.start(), a + m.end())
                      for m in pattern.finditer(target[a:b].translate(_DIGITS))
                      if kind != "ratio" or m.groups()[1:] == value]
        chosen = next((span for span in candidates if all(span[1] <= l or r <= span[0] for l, r in used)), None)
        if chosen is None:
            missing.append({"kind": kind, "source_span": [start, end], "value": list(value)})
        else:
            used.add(chosen)
            source_masks.append((start, end))
            target_masks.append(chosen)
        evidence.append({"kind": kind, "source_span": [start, end], "source_paragraph": para,
                         "target_span": list(chosen) if chosen else None})
    return _mask(source, source_masks), _mask(target, target_masks), missing, evidence


def numeric_artifact_reporting(events: list[dict]) -> list[dict]:
    """Select recorded gate evidence only; an empty list is not a pass."""
    records = []
    for event in events:
        if event.get("event_type") != "integrity_check_completed":
            continue
        payload = event.get("payload") or {}
        findings = [finding for finding in payload.get("findings", [])
                    if finding.get("check_id") in {
                        "source_numeric_artifact_evidence", "source_numeric_artifact_missing",
                    }]
        if findings:
            records.append({"chunk_index": event.get("chunk_index"), "stage": payload.get("stage"),
                            "accepted": payload.get("accepted"), "findings": findings})
    return records
