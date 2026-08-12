"""Deterministic integrity checks for translation replacement operations."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

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
_IDENTIFIER_DIGITS = "0-9\u06f0-\u06f9\u0660-\u0669"
_IDENTIFIER_PATTERNS = (
    re.compile(r"https?://[^\s<>()]+", re.IGNORECASE),
    re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE),
    re.compile(
        rf"\b(?:ISBN(?:-1[03])?|ISSN)\s*:?\s*"
        rf"[{_IDENTIFIER_DIGITS}Xx][{_IDENTIFIER_DIGITS}Xx\-\s]{{6,30}}"
        rf"[{_IDENTIFIER_DIGITS}Xx]",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?<![A-Za-z0-9])(?:97[89][\-\s]?)?"
        rf"[{_IDENTIFIER_DIGITS}][{_IDENTIFIER_DIGITS}\-]{{8,20}}"
        rf"[{_IDENTIFIER_DIGITS}Xx](?![A-Za-z0-9])"
    ),
    re.compile(
        rf"(?<![A-Za-z0-9])(?=[A-Za-z0-9\-/]*[A-Za-z])"
        rf"(?=[A-Za-z0-9\-/]*[{_IDENTIFIER_DIGITS}])"
        rf"[A-Za-z]{{2,}}(?:[-/][A-Za-z{_IDENTIFIER_DIGITS}]{{2,}}){{1,8}}"
        rf"(?![A-Za-z0-9])",
        re.IGNORECASE,
    ),
)
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


@dataclass(frozen=True)
class _NumberOccurrence:
    value: str
    role: str
    position: int
    label: str = ""


@dataclass
class _NumberReconciliation:
    missing: Counter[str] = field(default_factory=Counter)
    missing_roles: dict[str, Counter[str]] = field(default_factory=dict)
    localized_equivalents: list[dict[str, Any]] = field(default_factory=list)

    def roles_payload(self) -> dict[str, list[str]]:
        return {
            role: list(values.elements())
            for role, values in self.missing_roles.items()
            if values
        }


def _numeric_text(text: str) -> str:
    normalized = (text or "").translate(_DIGIT_MAP).replace("\u066a", "%")
    normalized = re.sub(
        r"(?<=\d)\s*([.\u066b\u066c])\s*(?=\d)", r"\1", normalized
    )
    normalized = _NOTE_RE.sub("", normalized)
    return re.sub(
        r"(?<=\d)\s*(?:percent|per\s+cent|\u062f\u0631\u0635\u062f)\b",
        "%",
        normalized,
        flags=re.IGNORECASE,
    )


def _structural_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for source_label, target_labels in _STRUCTURAL_NUMBER_LABELS.items():
        canonical = "volume" if source_label == "vol" else source_label
        aliases[source_label] = canonical
        for target_label in target_labels:
            aliases[target_label] = canonical
    for alias in ("page", "pages", "p", "pp"):
        aliases[alias] = "page"
    return aliases


def _structural_label(before: str, after: str) -> str:
    aliases = _structural_aliases()
    alternatives = "|".join(
        sorted((re.escape(value) for value in aliases), key=len, reverse=True)
    )
    leading = re.search(
        rf"(?:\b|(?<=[\u0600-\u06ff]))(?P<label>{alternatives})s?\.?\s*$",
        before,
        re.IGNORECASE,
    )
    if leading:
        return aliases[leading.group("label").casefold().rstrip(".")]
    listed = re.search(
        rf"(?:\b|(?<=[\u0600-\u06ff]))(?P<label>{alternatives})s?\.?\s+"
        rf"\d{{1,2}}(?:\s*,\s*\d{{1,2}})*"
        rf"(?:\s*,\s*|\s*,?\s*(?:and|or|&)\s*)?$",
        before,
        re.IGNORECASE,
    )
    if listed:
        return aliases[listed.group("label").casefold().rstrip(".")]
    trailing = re.match(
        rf"\s*[-\u2010-\u2014]?\s*(?P<label>{alternatives})s?\b",
        after,
        re.IGNORECASE,
    )
    if trailing:
        return aliases[trailing.group("label").casefold().rstrip(".")]
    return ""


def _number_occurrences(text: str) -> list[_NumberOccurrence]:
    normalized = _numeric_text(text)
    occurrences: list[_NumberOccurrence] = []
    for match in _NUMBER_RE.finditer(normalized):
        value = re.sub(r"\s+", "", match.group()).replace("\u066b", ".").replace(
            "\u066c", ","
        )
        before = normalized[max(0, match.start() - 80):match.start()]
        after = normalized[match.end():match.end() + 50]
        label = _structural_label(before, after)
        in_parenthetical = before.rfind("(") > before.rfind(")") and ")" in after
        is_year = bool(_CITATION_YEAR_RE.fullmatch(value.rstrip("%")))
        citation_cue = bool(re.search(
            r"(?:\b(?:see|cf|ibid|et\s+al|doi)\.?|\u0631\.\s*\u06a9\.)\s*$",
            before,
            re.IGNORECASE,
        ))
        if label:
            role = "structural"
        elif is_year or citation_cue or in_parenthetical:
            role = "citation"
        else:
            role = "prose"
        occurrences.append(_NumberOccurrence(value, role, match.start(), label))
    return occurrences


def classify_numbers(text: str) -> dict[str, Counter[str]]:
    """Classify comparable numbers for QA evidence without changing matching.

    Footnote/endnote markers remain governed by ``extract_note_markers`` and
    are intentionally excluded here. The roles make a failure explainable;
    they do not relax or strengthen the existing numeric integrity decision.
    """
    roles = {
        "prose": Counter(),
        "citation": Counter(),
        "structural": Counter(),
    }
    for occurrence in _number_occurrences(text):
        roles[occurrence.role][occurrence.value] += 1
    return roles


def extract_note_markers(text: str) -> list[str]:
    markers = []
    for bracketed, superscript in _NOTE_RE.findall(text or ""):
        markers.append(bracketed or superscript.translate(_SUPERSCRIPT_MAP))
    return markers


def extract_identifiers(text: str) -> Counter[str]:
    """Extract source identifiers whose spelling, digits, and separators matter."""
    values: list[str] = []
    occupied: list[tuple[int, int]] = []
    for pattern in _IDENTIFIER_PATTERNS:
        for match in pattern.finditer(text or ""):
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            value = " ".join(match.group().split()).strip(".,;)")
            value = re.sub(r"[\u2010-\u2015]", "-", value)
            if value:
                values.append(value.casefold())
                occupied.append(match.span())
    return Counter(values)


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


def _localized_structural_matches(
    source_occurrences: list[_NumberOccurrence],
    candidate: str,
) -> tuple[set[int], list[dict[str, Any]]]:
    """Match conservative Persian number words to explicit source units."""
    normalized_candidate = normalize_for_match(candidate.translate(_DIGIT_MAP))
    persian_word = r"\u0621-\u063a\u0641-\u064a\u066e-\u06d3\u06fa-\u06ff"
    consumed_spans: set[tuple[int, int]] = set()
    matched_source: set[int] = set()
    evidence: list[dict[str, Any]] = []
    canonical_targets: dict[str, set[str]] = {}
    for source_label, target_labels in _STRUCTURAL_NUMBER_LABELS.items():
        canonical = "volume" if source_label == "vol" else source_label
        canonical_targets.setdefault(canonical, set()).update(target_labels)

    for index, occurrence in enumerate(source_occurrences):
        if occurrence.role != "structural" or not occurrence.label:
            continue
        try:
            number = int(occurrence.value)
        except ValueError:
            continue
        forms = _persian_number_forms(number)
        target_labels = canonical_targets.get(occurrence.label, set())
        if not forms or not target_labels:
            continue
        candidates: list[tuple[int, int, str]] = []
        for target_label in target_labels:
            label = re.escape(target_label)
            label_suffix = r"(?:\s*\u0647\u0627(?:\u06cc)?|\u06cc)?"
            for form in forms:
                number_form = re.escape(form)
                # A single plural label can govern a short coordinated list:
                # "chapters second, third and fourth". Keep the search inside
                # that label's current clause and consume each number word once.
                for label_match in re.finditer(
                    rf"(?<![{persian_word}]){label}{label_suffix}(?![{persian_word}])",
                    normalized_candidate,
                ):
                    clause_start = label_match.end()
                    clause = normalized_candidate[clause_start:clause_start + 120]
                    clause = re.split(r"[.!?\u061b;\n]", clause, maxsplit=1)[0]
                    for form_match in re.finditer(
                        rf"(?<![{persian_word}]){number_form}(?![{persian_word}])",
                        clause,
                    ):
                        start = clause_start + form_match.start()
                        end = clause_start + form_match.end()
                        if (start, end) not in consumed_spans:
                            candidates.append((start, end, form_match.group()))
                patterns = (
                    # Label-first forms, including lists such as
                    # "chapters second, third and fourth".
                    rf"(?<![{persian_word}]){label}{label_suffix}"
                    rf"(?P<link>[\s\u060c,\u0648\u06cc\u0627-]{{0,48}})"
                    rf"(?P<number>{number_form})(?![{persian_word}])",
                    # Number-first compounds such as "three-volume" ->
                    # "three volume-adjectival" with ordinary space or ZWNJ.
                    rf"(?<![{persian_word}])(?P<number>{number_form})"
                    rf"[\s-]{{0,3}}{label}{label_suffix}(?![{persian_word}])",
                )
                for pattern in patterns:
                    for match in re.finditer(pattern, normalized_candidate):
                        span = match.span("number")
                        if span not in consumed_spans:
                            candidates.append((span[0], span[1], match.group("number")))
        if not candidates:
            continue
        start, end, rendered = min(candidates, key=lambda item: item[0])
        consumed_spans.add((start, end))
        matched_source.add(index)
        evidence.append({
            "source_value": occurrence.value,
            "source_role": occurrence.role,
            "source_label": occurrence.label,
            "target_form": rendered,
        })
    return matched_source, evidence


def _reconcile_numbers(source: str, candidate: str) -> _NumberReconciliation:
    """Reconcile literal occurrences before safe localized unit equivalents."""
    source_occurrences = _number_occurrences(source)
    target_occurrences = _number_occurrences(candidate)
    unmatched_source = set(range(len(source_occurrences)))
    unmatched_target = set(range(len(target_occurrences)))

    def pair_where(
        predicate: Callable[[_NumberOccurrence, _NumberOccurrence], bool],
    ) -> None:
        for source_index in list(unmatched_source):
            source_item = source_occurrences[source_index]
            target_index = next((
                index for index in unmatched_target
                if target_occurrences[index].value == source_item.value
                and predicate(source_item, target_occurrences[index])
            ), None)
            if target_index is not None:
                unmatched_source.remove(source_index)
                unmatched_target.remove(target_index)

    # Preserve semantic identity where diagnostics can establish it, then retain
    # the legacy value-count behavior as a compatibility fallback.
    pair_where(lambda source_item, target_item: (
        source_item.label and source_item.label == target_item.label
    ))
    pair_where(lambda source_item, target_item: source_item.role == target_item.role)
    pair_where(lambda _source_item, _target_item: True)

    remaining = [source_occurrences[index] for index in sorted(unmatched_source)]
    localized_indexes, evidence = _localized_structural_matches(remaining, candidate)
    remaining = [
        occurrence for index, occurrence in enumerate(remaining)
        if index not in localized_indexes
    ]
    missing = Counter(occurrence.value for occurrence in remaining)
    missing_roles: dict[str, Counter[str]] = {}
    for occurrence in remaining:
        missing_roles.setdefault(occurrence.role, Counter())[occurrence.value] += 1
    return _NumberReconciliation(missing, missing_roles, evidence)


def _missing_numbers(source: str, candidate: str) -> Counter[str]:
    return _reconcile_numbers(source, candidate).missing


def _new_missing_roles(
    current: _NumberReconciliation,
    previous: _NumberReconciliation,
    newly_missing: Counter[str],
) -> dict[str, list[str]]:
    """Describe the roles whose unmatched occurrence counts actually grew."""
    remaining = Counter(newly_missing)
    output: dict[str, list[str]] = {}
    for role in ("citation", "structural", "prose"):
        growth = (
            current.missing_roles.get(role, Counter())
            - previous.missing_roles.get(role, Counter())
        ) & remaining
        if growth:
            output[role] = list(growth.elements())
            remaining -= growth
    if remaining:
        for role in ("citation", "structural", "prose"):
            available = current.missing_roles.get(role, Counter()) & remaining
            if available:
                output.setdefault(role, []).extend(available.elements())
                remaining -= available
    if remaining:
        output.setdefault("prose", []).extend(remaining.elements())
    return output


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

        required_identifiers = extract_identifiers(source)
        candidate_identifiers = extract_identifiers(candidate)
        previous_identifiers = extract_identifiers(previous) if previous else Counter()
        missing_identifiers = required_identifiers - candidate_identifiers
        previous_missing_identifiers = (
            required_identifiers - previous_identifiers if previous else Counter()
        )
        newly_missing_identifiers = (
            missing_identifiers - previous_missing_identifiers
            if previous else missing_identifiers
        )
        if newly_missing_identifiers:
            add(
                "source_identifiers_changed",
                "blocking",
                "A source identifier was removed, localized, or respaced.",
                missing=list(newly_missing_identifiers.elements()),
            )
        elif missing_identifiers:
            add(
                "source_identifiers_still_changed",
                "warning",
                "The edit did not introduce identifier damage, but earlier damage remains.",
                missing=list(missing_identifiers.elements()),
            )

        number_reconciliation = _reconcile_numbers(source, candidate)
        previous_number_reconciliation = (
            _reconcile_numbers(source, previous)
            if previous else _NumberReconciliation()
        )
        missing_number_counts = number_reconciliation.missing
        previous_missing_number_counts = previous_number_reconciliation.missing
        newly_missing_number_counts = (
            missing_number_counts - previous_missing_number_counts
            if previous else missing_number_counts
        )
        if number_reconciliation.localized_equivalents:
            add(
                "numbers_localized_equivalent", "info",
                "Explicit structural numbers were preserved as localized Persian forms.",
                equivalents=number_reconciliation.localized_equivalents,
            )
        newly_missing_numbers = list(newly_missing_number_counts.elements())
        if newly_missing_numbers:
            add(
                "numbers_missing", "blocking",
                "Source numbers, dates, pages, or percentages are missing or changed.",
                missing=newly_missing_numbers,
                missing_by_role=_new_missing_roles(
                    number_reconciliation,
                    previous_number_reconciliation,
                    newly_missing_number_counts,
                ),
            )
        elif missing_number_counts:
            add(
                "numbers_still_missing", "warning",
                "The edit did not introduce numeric loss, but an earlier omission remains.",
                missing=list(missing_number_counts.elements()),
                missing_by_role=number_reconciliation.roles_payload(),
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
