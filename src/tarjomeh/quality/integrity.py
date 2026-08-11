"""Deterministic integrity checks for translation replacement operations."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from tarjomeh.glossary.compliance import target_present


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
_SOURCE_PAREN_RE = re.compile(r"\(([^()\n]{2,240})\)")
_SOURCE_QUOTE_RE = re.compile(
    r'(?:[\u201c\u201e"]([^\u201c\u201d\u201e"\n]{2,240})[\u201d"]'
    r"|\u2018([^\u2018\u2019\n]{2,240})\u2019)"
)
_CITATION_YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})[a-z]?\b", re.IGNORECASE)
_CITATION_NAME_RE = re.compile(
    r"(?:[A-Z](?:[A-Za-z'\u2019.-]*|\.)(?:\s+|$)){1,5}$"
)
_NON_ATOMIC_APPARATUS_RE = re.compile(
    r"^(?:as|cf\.?|e\.g\.?|for|i\.e\.?|on|see|that|which|where)\b",
    re.IGNORECASE,
)
_STRUCTURAL_NUMBER_LABELS = {
    "chapter": ("\u0641\u0635\u0644",),
    "part": ("\u0628\u062e\u0634", "\u0642\u0633\u0645\u062a"),
    "volume": ("\u062c\u0644\u062f",),
    "vol": ("\u062c\u0644\u062f",),
    "book": ("\u06a9\u062a\u0627\u0628",),
    "table": ("\u062c\u062f\u0648\u0644",),
    "figure": ("\u0634\u06a9\u0644", "\u0646\u0645\u0648\u062f\u0627\u0631"),
    "section": ("\u0628\u062e\u0634", "\u0642\u0633\u0645\u062a"),
    "appendix": ("\u067e\u06cc\u0648\u0633\u062a",),
    "edition": ("\u0648\u06cc\u0631\u0627\u06cc\u0634", "\u0686\u0627\u067e"),
}
_PERSIAN_UNITS = (
    "\u0635\u0641\u0631", "\u06cc\u06a9", "\u062f\u0648", "\u0633\u0647", "\u0686\u0647\u0627\u0631",
    "\u067e\u0646\u062c", "\u0634\u0634", "\u0647\u0641\u062a", "\u0647\u0634\u062a", "\u0646\u0647",
)
_PERSIAN_TEENS = {
    10: "\u062f\u0647", 11: "\u06cc\u0627\u0632\u062f\u0647", 12: "\u062f\u0648\u0627\u0632\u062f\u0647",
    13: "\u0633\u06cc\u0632\u062f\u0647", 14: "\u0686\u0647\u0627\u0631\u062f\u0647", 15: "\u067e\u0627\u0646\u0632\u062f\u0647",
    16: "\u0634\u0627\u0646\u0632\u062f\u0647", 17: "\u0647\u0641\u062f\u0647", 18: "\u0647\u062c\u062f\u0647",
    19: "\u0646\u0648\u0632\u062f\u0647",
}
_PERSIAN_TENS = {
    20: "\u0628\u06cc\u0633\u062a", 30: "\u0633\u06cc", 40: "\u0686\u0647\u0644", 50: "\u067e\u0646\u062c\u0627\u0647",
    60: "\u0634\u0635\u062a", 70: "\u0647\u0641\u062a\u0627\u062f", 80: "\u0647\u0634\u062a\u0627\u062f", 90: "\u0646\u0648\u062f",
}
_PERSIAN_SPECIAL_ORDINALS = {
    1: {"\u0646\u062e\u0633\u062a", "\u0627\u0648\u0644"},
    3: {"\u0633\u0648\u0645"},
    30: {"\u0633\u06cc \u0627\u0645"},
}


def normalize_for_match(text: str) -> str:
    """Normalize spacing variants used by Persian terminology."""
    return re.sub(r"[\s\u200c]+", " ", (text or "").strip()).casefold()


def extract_numbers(text: str) -> Counter[str]:
    """Extract comparable Latin/Persian numeric tokens and percentages."""
    normalized = (text or "").translate(_DIGIT_MAP).replace("\u066a", "%")
    normalized = re.sub(r"(?<=\d)\s*([.\u066b\u066c])\s*(?=\d)", r"\1", normalized)
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


def classify_numbers(text: str) -> dict[str, Counter[str]]:
    """Classify comparable numbers for QA evidence without changing matching.

    Footnote/endnote markers remain governed by ``extract_note_markers`` and
    are intentionally excluded here. The roles make a failure explainable;
    they do not relax or strengthen the existing numeric integrity decision.
    """
    normalized = (text or "").translate(_DIGIT_MAP).replace("\u066a", "%")
    normalized = re.sub(
        r"(?<=\d)\s*([.\u066b\u066c])\s*(?=\d)", r"\1", normalized
    )
    normalized = _NOTE_RE.sub("", normalized)
    normalized = re.sub(
        r"(?<=\d)\s*(?:percent|per\s+cent|\u062f\u0631\u0635\u062f)\b",
        "%",
        normalized,
        flags=re.IGNORECASE,
    )
    roles = {
        "prose": Counter(),
        "citation": Counter(),
        "structural": Counter(),
    }
    structural_labels = "|".join(re.escape(value) for value in (
        *_STRUCTURAL_NUMBER_LABELS,
        *(target for values in _STRUCTURAL_NUMBER_LABELS.values() for target in values),
        "page", "pages", "p", "pp",
    ))
    for match in _NUMBER_RE.finditer(normalized):
        value = re.sub(r"\s+", "", match.group()).replace("\u066b", ".").replace(
            "\u066c", ","
        )
        before = normalized[max(0, match.start() - 80):match.start()]
        after = normalized[match.end():match.end() + 50]
        in_parenthetical = before.rfind("(") > before.rfind(")") and ")" in after
        is_year = bool(_CITATION_YEAR_RE.fullmatch(value.rstrip("%")))
        citation_cue = bool(re.search(
            r"(?:\b(?:see|cf|ibid|et\s+al|doi)\.?|\u0631\.\s*\u06a9\.)\s*$",
            before,
            re.IGNORECASE,
        ))
        structural_cue = bool(re.search(
            rf"(?:\b(?:{structural_labels})\.?\s*)$",
            before,
            re.IGNORECASE,
        ))
        if is_year or citation_cue or in_parenthetical:
            role = "citation"
        elif structural_cue:
            role = "structural"
        else:
            role = "prose"
        roles[role][value] += 1
    return roles


def _missing_number_roles(source: str, missing: Counter[str]) -> dict[str, list[str]]:
    """Allocate missing numeric occurrences to their source-side QA roles."""
    remaining = Counter(missing)
    output: dict[str, list[str]] = {}
    source_roles = classify_numbers(source)
    for role in ("citation", "structural", "prose"):
        allocated = source_roles[role] & remaining
        if allocated:
            output[role] = list(allocated.elements())
            remaining -= allocated
    if remaining:
        output.setdefault("prose", []).extend(remaining.elements())
    return output


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


def protected_source_apparatus(source: str, translation: str) -> list[str]:
    """Return source-delimited Latin spans retained verbatim in a translation.

    This is structural protection, not a vocabulary list. Ordinary English prose
    that was translated is absent from the result; only parenthetical or quoted
    source material already carried into the prior valid translation is retained.
    """
    translation_folded = normalize_for_match(translation)
    values: list[str] = []
    candidates: list[str] = []
    for parenthetical in _SOURCE_PAREN_RE.findall(source or ""):
        years = list(_CITATION_YEAR_RE.finditer(parenthetical))
        if years:
            for year in years:
                prefix = parenthetical[:year.start()]
                name_match = _CITATION_NAME_RE.search(prefix)
                name = name_match.group().strip() if name_match else ""
                name = re.sub(r"^(?:cf\.?|see)\s+", "", name, flags=re.IGNORECASE)
                candidates.append(" ".join(part for part in (name, year.group()) if part))
            continue
        cleaned = " ".join(parenthetical.split()).strip()
        words = re.findall(r"[A-Za-z][A-Za-z'\u2019-]*", cleaned)
        if (
            1 <= len(words) <= 8
            and not _NON_ATOMIC_APPARATUS_RE.search(cleaned)
            and not re.search(r"[.!?;]", cleaned)
        ):
            candidates.append(cleaned)
    for match in _SOURCE_QUOTE_RE.finditer(source or ""):
        value = match.group(1) or match.group(2) or ""
        prefix = (source or "")[max(0, match.start() - 80):match.start()]
        if re.search(
            r"\b(?:called|label(?:led)?(?:\s+reads)?|original(?:-language)?|"
            r"phrase|term|title|known as)(?:\s+\w+){0,2}\s*$",
            prefix,
            re.IGNORECASE,
        ):
            candidates.append(value)
    for value in candidates:
        cleaned = " ".join(value.split()).strip()
        if not cleaned or not re.search(r"[A-Za-z]", cleaned):
            continue
        if normalize_for_match(cleaned) in translation_folded:
            values.append(cleaned)
    return sorted(set(values), key=str.casefold)


def _persian_number_forms(value: int) -> set[str]:
    """Return conservative cardinal/ordinal Persian forms for 0..99."""
    if not 0 <= value <= 99:
        return set()
    if value < 10:
        cardinal = _PERSIAN_UNITS[value]
    elif value < 20:
        cardinal = _PERSIAN_TEENS[value]
    else:
        tens, unit = divmod(value, 10)
        cardinal = _PERSIAN_TENS[tens * 10]
        if unit:
            cardinal += " \u0648 " + _PERSIAN_UNITS[unit]
    forms = {cardinal, cardinal + "\u0645"}
    forms.update(_PERSIAN_SPECIAL_ORDINALS.get(value, set()))
    return forms


def _structural_number_equivalents(source: str, candidate: str) -> Counter[str]:
    """Count localized number words only in explicit structural contexts."""
    normalized_candidate = normalize_for_match(candidate.translate(_DIGIT_MAP))
    equivalents: Counter[str] = Counter()
    labels = "|".join(re.escape(label) for label in _STRUCTURAL_NUMBER_LABELS)
    leading = re.compile(
        rf"\b(?P<label>{labels})s?\.?\s+(?P<numbers>\d{{1,2}}"
        rf"(?:\s*(?:,\s*(?:and|or)?|and|or|&)\s*\d{{1,2}})*)",
        re.IGNORECASE,
    )
    trailing = re.compile(
        rf"\b(?P<number>\d{{1,2}})\s*[- ]\s*(?P<label>{labels})s?\.?\b",
        re.IGNORECASE,
    )
    occurrences: set[tuple[int, str, int]] = set()
    for match in leading.finditer(source or ""):
        label = match.group("label").casefold().rstrip(".")
        for number_text in re.findall(r"\d{1,2}", match.group("numbers")):
            occurrences.add((match.start(), label, int(number_text)))
    for match in trailing.finditer(source or ""):
        occurrences.add((
            match.start(),
            match.group("label").casefold().rstrip("."),
            int(match.group("number")),
        ))
    persian_word = r"\u0621-\u063a\u0641-\u064a\u066e-\u06d3\u06fa-\u06ff"
    for _position, source_label, number in occurrences:
            target_labels = _STRUCTURAL_NUMBER_LABELS.get(source_label, ())
            forms = _persian_number_forms(number)
            for target_label in target_labels:
                for label_match in re.finditer(
                    rf"(?<![{persian_word}]){re.escape(target_label)}"
                    rf"(?:\s*\u0647\u0627(?:\u06cc)?)?(?![{persian_word}])",
                    normalized_candidate,
                ):
                    window = normalized_candidate[
                        max(0, label_match.start() - 100):label_match.end() + 180
                    ]
                    if any(
                        re.search(
                            rf"(?<![{persian_word}]){re.escape(form)}(?![{persian_word}])",
                            window,
                        )
                        for form in forms
                    ):
                        equivalents[str(number)] += 1
                        break
                else:
                    continue
                break
    return equivalents


def _missing_numbers(source: str, candidate: str) -> Counter[str]:
    missing = extract_numbers(source) - extract_numbers(candidate)
    return missing - _structural_number_equivalents(source, candidate)


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

            missing_apparatus = [
                value for value in protected_source_apparatus(source, previous)
                if normalize_for_match(value) not in normalize_for_match(candidate)
            ]
            if missing_apparatus:
                add(
                    "source_apparatus_removed", "blocking",
                    "Source-authored multilingual or scholarly material was removed.",
                    missing=missing_apparatus,
                )

        missing_number_counts = _missing_numbers(source, candidate)
        previous_missing_number_counts = (
            _missing_numbers(source, previous) if previous else Counter()
        )
        newly_missing_number_counts = (
            missing_number_counts - previous_missing_number_counts
            if previous else missing_number_counts
        )
        newly_missing_numbers = list(newly_missing_number_counts.elements())
        if newly_missing_numbers:
            add(
                "numbers_missing", "blocking",
                "Source numbers, dates, pages, or percentages are missing or changed.",
                missing=newly_missing_numbers,
                missing_by_role=_missing_number_roles(
                    source, newly_missing_number_counts
                ),
            )
        elif missing_number_counts:
            add(
                "numbers_still_missing", "warning",
                "The edit did not introduce numeric loss, but an earlier omission remains.",
                missing=list(missing_number_counts.elements()),
                missing_by_role=_missing_number_roles(source, missing_number_counts),
            )

        required_notes = Counter(extract_note_markers(source))
        if previous:
            required_notes |= Counter(extract_note_markers(previous))
        missing_note_counts = required_notes - Counter(extract_note_markers(candidate))
        previous_missing_note_counts = (
            required_notes - Counter(extract_note_markers(previous))
            if previous else Counter()
        )
        newly_missing_note_counts = (
            missing_note_counts - previous_missing_note_counts
            if previous else missing_note_counts
        )
        newly_missing_notes = list(newly_missing_note_counts.elements())
        if newly_missing_notes:
            add(
                "note_markers_missing", "blocking",
                "Footnote or endnote markers were removed.", missing=newly_missing_notes,
            )
        elif missing_note_counts:
            add(
                "note_markers_still_missing", "warning",
                "The edit preserved all available note markers, but an earlier omission remains.",
                missing=list(missing_note_counts.elements()),
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

        missing_terms = []
        for term in protected_terms:
            if not str(term).strip():
                continue
            required = enforce_all_terms or target_present(str(term), previous)
            if required and not target_present(str(term), candidate):
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
