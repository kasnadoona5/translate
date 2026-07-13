"""Deterministic integrity checks for translation replacement operations."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable


_DIGIT_MAP = str.maketrans(
    "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9"
    "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669",
    "01234567890123456789",
)
_SUPERSCRIPT_MAP = str.maketrans("\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u2070", "1234567890")
_NUMBER_RE = re.compile(r"(?<!\w)[+-]?\d+(?:[.,\u066b\u066c]\d+)*(?:\s*[%\u066a])?")
_NOTE_RE = re.compile(r"\[(\d+)\]|([\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u2070]+)")
_PERSIAN_RE = re.compile(r"[\u0600-\u06ff]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_JSON_LEAK_RE = re.compile(r'(^\s*\{|"(?:translation|decision|rationale)"\s*:)', re.IGNORECASE)
_ENGLISH_PAREN_RE = re.compile(r"\(([A-Za-z][A-Za-z0-9 .,&':;\u2019\-]{1,80})\)")


def normalize_for_match(text: str) -> str:
    """Normalize spacing variants used by Persian terminology."""
    return re.sub(r"[\s\u200c]+", " ", (text or "").strip()).casefold()


def extract_numbers(text: str) -> Counter[str]:
    """Extract comparable Latin/Persian numeric tokens and percentages."""
    normalized = (text or "").translate(_DIGIT_MAP).replace("\u066a", "%")
    # Notes have their own representation-aware check, so [12] and superscript
    # 12 are not incorrectly compared as ordinary prose numbers.
    normalized = _NOTE_RE.sub("", normalized)
    normalized = re.sub(
        r"(?<=\d)\s*(?:percent|per\s+cent|\u062f\u0631\u0635\u062f)\b",
        "%",
        normalized,
        flags=re.IGNORECASE,
    )
    values = []
    for match in _NUMBER_RE.findall(normalized):
        value = re.sub(r"\s+", "", match).replace("\u066b", ".").replace("\u066c", ",")
        values.append(value)
    return Counter(values)


def extract_note_markers(text: str) -> list[str]:
    markers = []
    for bracketed, superscript in _NOTE_RE.findall(text or ""):
        markers.append(bracketed or superscript.translate(_SUPERSCRIPT_MAP))
    return markers


def protected_english_originals(source: str, translation: str) -> list[str]:
    """Return inline English originals that are grounded in the source."""
    source_folded = (source or "").casefold()
    values = []
    for value in _ENGLISH_PAREN_RE.findall(translation or ""):
        cleaned = " ".join(value.split()).strip()
        if cleaned and cleaned.casefold() in source_folded:
            values.append(cleaned)
    return sorted(set(values), key=str.casefold)


def protected_source_citations(source: str, translation: str) -> list[str]:
    """Return numeric English parentheticals that also occur in the source."""
    return [
        value for value in protected_english_originals(source, translation)
        if any(char.isdigit() for char in value)
    ]


@dataclass
class IntegrityFinding:
    check_id: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class IntegrityResult:
    stage: str
    source_chars: int
    previous_chars: int
    candidate_chars: int
    findings: list[IntegrityFinding] = field(default_factory=list)

    @property
    def blocking(self) -> list[IntegrityFinding]:
        return [finding for finding in self.findings if finding.severity == "blocking"]

    @property
    def warnings(self) -> list[IntegrityFinding]:
        return [finding for finding in self.findings if finding.severity == "warning"]

    @property
    def accepted(self) -> bool:
        return not self.blocking

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "accepted": self.accepted,
            "source_chars": self.source_chars,
            "previous_chars": self.previous_chars,
            "candidate_chars": self.candidate_chars,
            "blocking_count": len(self.blocking),
            "warning_count": len(self.warnings),
            "findings": [finding.to_dict() for finding in self.findings],
        }


class PostEditIntegrityGate:
    """Reject high-confidence lossy edits while retaining advisory evidence."""

    def __init__(self, min_retention_ratio: float = 0.65, max_growth_ratio: float = 1.75) -> None:
        self.min_retention_ratio = min_retention_ratio
        self.max_growth_ratio = max_growth_ratio

    def evaluate(
        self,
        source: str,
        candidate: str,
        *,
        previous: str = "",
        stage: str = "final",
        protected_terms: Iterable[str] = (),
        protect_inline_english: bool = False,
        allowed_inline_originals: Iterable[str] | None = None,
        enforce_all_terms: bool = False,
    ) -> IntegrityResult:
        source = source or ""
        previous = previous or ""
        candidate = candidate or ""
        result = IntegrityResult(stage, len(source), len(previous), len(candidate))

        def add(check_id: str, severity: str, message: str, **details: Any) -> None:
            result.findings.append(IntegrityFinding(check_id, severity, message, details))

        if not candidate.strip():
            add("empty_translation", "blocking", "The proposed translation is empty.")
            return result

        source_ratio = len(candidate.strip()) / max(1, len(source.strip()))
        if source.strip() and not 0.25 <= source_ratio <= 3.0:
            add(
                "source_length_ratio", "blocking",
                "The proposed translation has an extreme source-length ratio.",
                ratio=round(source_ratio, 4),
            )

        if previous.strip():
            edit_ratio = len(candidate.strip()) / max(1, len(previous.strip()))
            if edit_ratio < self.min_retention_ratio:
                add(
                    "edit_content_loss", "blocking",
                    "The edit removed a suspicious amount of existing translation.",
                    ratio=round(edit_ratio, 4), minimum=self.min_retention_ratio,
                )
            elif edit_ratio > self.max_growth_ratio:
                add(
                    "edit_content_growth", "blocking",
                    "The edit added a suspicious amount of text.",
                    ratio=round(edit_ratio, 4), maximum=self.max_growth_ratio,
                )

        source_numbers = extract_numbers(source)
        candidate_numbers = extract_numbers(candidate)
        missing_numbers = list((source_numbers - candidate_numbers).elements())
        if missing_numbers:
            add(
                "numbers_missing", "blocking",
                "Source numbers, dates, pages, or percentages are missing or changed.",
                missing=missing_numbers,
            )

        required_notes = Counter(extract_note_markers(source))
        if previous:
            required_notes |= Counter(extract_note_markers(previous))
        missing_notes = list((required_notes - Counter(extract_note_markers(candidate))).elements())
        if missing_notes:
            add(
                "note_markers_missing", "blocking",
                "Footnote or endnote markers were removed.", missing=missing_notes,
            )

        source_paragraphs = _paragraphs(source)
        candidate_paragraphs = _paragraphs(candidate)
        minimum_source_paragraphs = max(1, (len(source_paragraphs) + 1) // 2)
        if len(source_paragraphs) > 1 and len(candidate_paragraphs) < minimum_source_paragraphs:
            add(
                "paragraph_structure_loss", "blocking",
                "Too many source paragraph boundaries disappeared.",
                source_count=len(source_paragraphs), candidate_count=len(candidate_paragraphs),
            )
        if previous:
            previous_paragraphs = _paragraphs(previous)
            minimum_previous = max(1, (len(previous_paragraphs) + 1) // 2)
            if len(previous_paragraphs) > 1 and len(candidate_paragraphs) < minimum_previous:
                add(
                    "edit_paragraph_loss", "blocking",
                    "The edit collapsed too many existing translated paragraphs.",
                    previous_count=len(previous_paragraphs), candidate_count=len(candidate_paragraphs),
                )

        if _JSON_LEAK_RE.search(candidate):
            add(
                "structured_response_leak", "blocking",
                "JSON or refiner control fields leaked into translation text.",
            )

        source_words = _ASCII_WORD_RE.findall(source)
        if len(source_words) >= 8 and len(_PERSIAN_RE.findall(candidate)) < 3:
            add(
                "target_language_missing", "blocking",
                "The proposed output does not contain enough Persian text.",
            )

        duplicate_paragraphs = [
            text for text, count in Counter(
                normalize_for_match(p) for p in candidate_paragraphs if len(p) >= 80
            ).items() if count > 1
        ]
        if duplicate_paragraphs:
            add(
                "duplicate_paragraph", "blocking",
                "A substantial translated paragraph is duplicated.",
                duplicate_count=len(duplicate_paragraphs),
            )

        normalized_previous = normalize_for_match(previous)
        normalized_candidate = normalize_for_match(candidate)
        missing_terms = []
        for term in protected_terms:
            normalized_term = normalize_for_match(term)
            if not normalized_term:
                continue
            required = enforce_all_terms or normalized_term in normalized_previous
            if required and normalized_term not in normalized_candidate:
                missing_terms.append(term)
        if missing_terms:
            add(
                "protected_terminology_removed", "blocking",
                "Mandatory terminology present before the edit was removed.",
                missing=missing_terms,
            )

        if protect_inline_english:
            previous_originals = protected_english_originals(source, previous)
            candidate_originals = protected_english_originals(source, candidate)
            previous_folded = {value.casefold() for value in previous_originals}
            candidate_folded = {value.casefold() for value in candidate_originals}
            allowed_folded = (
                {
                    str(value).strip().casefold()
                    for value in allowed_inline_originals
                    if str(value).strip()
                }
                if allowed_inline_originals is not None else None
            )
            protected_previous = previous_originals
            if allowed_folded is not None:
                protected_previous = [
                    value for value in previous_originals
                    if value.casefold() in allowed_folded
                ]
            missing_originals = [
                value for value in protected_previous
                if value.casefold() not in candidate_folded
            ]
            if missing_originals:
                add(
                    "english_original_removed", "blocking",
                    "A required first-occurrence English original was removed.",
                    missing=missing_originals,
                )

            if allowed_folded is not None:
                unauthorized = [
                    value for value in candidate_originals
                    if value.casefold() not in allowed_folded
                    and value.casefold() not in previous_folded
                    and not any(char.isdigit() for char in value)
                ]
                if unauthorized and previous:
                    add(
                        "unauthorized_english_original_added", "blocking",
                        "The edit added an English parenthetical not authorized by policy.",
                        added=unauthorized,
                    )
                elif unauthorized:
                    add(
                        "unauthorized_english_original_present", "warning",
                        "The initial translation contains an unauthorized English parenthetical.",
                        present=unauthorized,
                    )

            if previous:
                missing_citations = [
                    value for value in protected_source_citations(source, previous)
                    if value.casefold() not in candidate_folded
                ]
                if missing_citations:
                    add(
                        "source_citation_removed", "blocking",
                        "A source-grounded citation parenthetical was removed.",
                        missing=missing_citations,
                    )

        if source_paragraphs and len(source_paragraphs[0]) <= 100 and len(source_paragraphs) > 1:
            if not candidate_paragraphs or len(candidate_paragraphs[0]) > 180:
                add(
                    "heading_structure_uncertain", "warning",
                    "The source begins with a likely heading whose target structure is uncertain.",
                )
        return result


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text or "") if part.strip()]
