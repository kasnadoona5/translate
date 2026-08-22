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
    re.compile(
        r"(?<![@\w])(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,62})\.)+"
        r"[A-Za-z]{2,63}(?:/[^\s<>()]*)?",
        re.IGNORECASE,
    ),
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
    re.compile(
        rf"(?<![A-Za-z0-9])"
        rf"(?=[A-Za-z{_IDENTIFIER_DIGITS}.\-/\u2010-\u2015]*[A-Za-z])"
        rf"(?=[A-Za-z{_IDENTIFIER_DIGITS}.\-/\u2010-\u2015]*[{_IDENTIFIER_DIGITS}])"
        rf"(?=[A-Za-z{_IDENTIFIER_DIGITS}.\-/\u2010-\u2015]*[./])"
        rf"[A-Za-z{_IDENTIFIER_DIGITS}]{{1,12}}"
        rf"(?:[.\-/\u2010-\u2015][A-Za-z{_IDENTIFIER_DIGITS}]{{1,12}}){{1,5}}"
        rf"(?![A-Za-z0-9])",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?<![A-Za-z0-9])[A-Z]{{1,2}}[{_IDENTIFIER_DIGITS}][A-Z{_IDENTIFIER_DIGITS}]?"
        rf"\s+[{_IDENTIFIER_DIGITS}][A-Z]{{2}}(?![A-Za-z0-9])"
    ),
)
_PROSE_FOOTNOTE_SUFFIX_RE = re.compile(
    r"^[a-z][a-z'\u2019-]{1,40}\.[0-9]{1,3}$"
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
_BIBLIOGRAPHIC_MARKER_RE = re.compile(
    r"^(?:pb|pbk|paperback|hb|hbk|hardback|hardcover|cloth|ebook|e-book|"
    r"ed\.?|eds\.?|rev\.?\s*ed\.?|vol\.?|no\.?)$",
    re.IGNORECASE,
)
_PERSIAN_LETTER_CLASS = r"\u0621-\u063a\u0641-\u064a\u066e-\u06d3\u06fa-\u06ff"
_MIXED_SCRIPT_TOKEN_RE = re.compile(
    rf"(?<![\w/.-])(?:[A-Za-z]+[{_PERSIAN_LETTER_CLASS}]+|"
    rf"[{_PERSIAN_LETTER_CLASS}]+[A-Za-z]+)(?![\w/.-])"
)
_LABELED_IDENTIFIER_RE = re.compile(
    rf"\b(?:ISBN(?:-1[03])?|ISSN)\s*:?\s*"
    rf"[{_IDENTIFIER_DIGITS}Xx][{_IDENTIFIER_DIGITS}Xx\-\s]{{6,30}}"
    rf"[{_IDENTIFIER_DIGITS}Xx]",
    re.IGNORECASE,
)
_FLEXIBLE_IDENTIFIER_LABEL_SUFFIX_RE = re.compile(
    rf"(?:I\s*S\s*B\s*N|I\s*S\s*S\s*N)"
    rf"(?:\s*[-\u2010-\u2015]?\s*"
    rf"[{_IDENTIFIER_DIGITS}]\s*[{_IDENTIFIER_DIGITS}])?\s*:?\s*$",
    re.IGNORECASE,
)
_ABBREVIATION_DEFINITION_RE = re.compile(
    r"(?<!\w)(?P<acronym>[A-Z]{2,}(?:\s+[A-Z]{2,}){0,3})"
    r"(?:\s+(?P<label>Act|Agreement|Agency|Convention|Law|Organisation|"
    r"Organization|Program|Programme|Protocol|Treaty))?\s+"
)
_ABBREVIATION_CONNECTORS = frozenset({
    "and", "by", "for", "from", "in", "of", "on", "the", "to", "with",
})
_PARENTHETICAL_PERSIAN_SUFFIX_RE = re.compile(
    rf"\([^()\n]*[A-Za-z][^()\n]*\)\u200c?"
    rf"[{_PERSIAN_LETTER_CLASS}]{{1,8}}"
)
_LATIN_LETTERS = "A-Za-z\u00c0-\u024f\u1e00-\u1eff"
_LATIN_PROSE_TOKEN_RE = re.compile(
    rf"(?<![{_LATIN_LETTERS}])[{_LATIN_LETTERS}]"
    rf"[{_LATIN_LETTERS}'\u2019-]{{1,}}(?![{_LATIN_LETTERS}])"
)
_ROMAN_NUMERAL_RE = re.compile(r"[ivxlcdm]+", re.IGNORECASE)
_APPARATUS_CHAPTER_RE = re.compile(
    r"\b(?:abbreviations?|bibliograph(?:y|ies)|catalog(?:ue|ing)?|contents?|"
    r"glossar(?:y|ies)|index(?:es)?|references?|works cited)\b",
    re.IGNORECASE,
)
_SOURCE_METADATA_RE = re.compile(
    r"\b(?:ISBN|ISSN|Library of Congress|catalog(?:ue|ing)|classification|"
    r"copyright|all rights reserved)\b",
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
    "year": ("\u0633\u0627\u0644",),
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
_PERSIAN_HUNDREDS = {
    100: "\u0635\u062f", 200: "\u062f\u0648\u06cc\u0633\u062a", 300: "\u0633\u06cc\u0635\u062f",
    400: "\u0686\u0647\u0627\u0631\u0635\u062f", 500: "\u067e\u0627\u0646\u0635\u062f", 600: "\u0634\u0634\u0635\u062f",
    700: "\u0647\u0641\u062a\u0635\u062f", 800: "\u0647\u0634\u062a\u0635\u062f", 900: "\u0646\u0647\u0635\u062f",
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
    if not trailing:
        trailing = re.match(
            rf"\s*[-\u2010-\u2014]\s*\d+(?:[.,]\d+)?\s*"
            rf"(?P<label>{alternatives})s?\b",
            after,
            re.IGNORECASE,
        )
    if trailing:
        return aliases[trailing.group("label").casefold().rstrip(".")]
    return ""


def _ordered_enumeration_values(text: str) -> Counter[str]:
    """Identify source list markers such as ``(1) ... (2) ...`` conservatively."""
    normalized = (text or "").translate(_DIGIT_MAP)
    values = [
        int(match.group(1))
        for match in re.finditer(r"\(\s*(\d{1,2})\s*\)", normalized)
    ]
    if len(values) < 2 or values[0] != 1:
        return Counter()
    if any(current != previous + 1 for previous, current in zip(values, values[1:])):
        return Counter()
    return Counter(str(value) for value in values)


def _number_occurrences(text: str) -> list[_NumberOccurrence]:
    normalized = _numeric_text(text)
    enumeration_values = _ordered_enumeration_values(text)
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
        parenthesized_marker = bool(
            re.search(r"\(\s*$", before)
            and re.match(r"\s*\)", after)
            and enumeration_values[value] > 0
        )
        if parenthesized_marker:
            role = "enumeration"
            label = "enumeration"
            enumeration_values[value] -= 1
        elif label:
            role = "structural"
        elif is_year or citation_cue or in_parenthetical:
            role = "citation"
        else:
            role = "prose"
        occurrences.append(_NumberOccurrence(value, role, match.start(), label))
    return occurrences


# Unicode damage that must never reach the reader. Nothing previously looked
# for any of it, which is how a U+FFFD survived into the delivered DOCX.
#   * U+FFFD  - a decoder or provider replaced a character it could not encode
#   * U+E000-U+F8FF - private use; here it means a PersianTypographer sentinel
#     was never restored (typography.py uses U+E000/U+E001 and base U+E100)
#   * C0/C1 control characters other than tab, newline and carriage return
_REPLACEMENT_CHAR = "\ufffd"
_PRIVATE_USE_RE = re.compile("[\ue000-\uf8ff]")
_CONTROL_CHAR_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


# Item 16. The existing per-issue salvage path already rejects an edit that
# doubles an ADJACENT word (pipeline._salvage_local_refinement_edits). Nothing
# looked for a duplicated multi-word span, and nothing checked the
# full-candidate refinement path at all -- which is how a salvage patch
# duplicated the clause it inserted and the next critique iteration had to find
# it. Spans are compared against the PREVIOUS translation, same language, so
# repetition the translation already contained is never blamed on this edit.
_SPAN_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_MIN_REPEATED_SPAN_WORDS = 8


def _repeated_spans(text: str, min_words: int = _MIN_REPEATED_SPAN_WORDS) -> set[str]:
    """Return normalised word spans of >= *min_words* that occur more than once.

    Only maximal-length duplicates matter for reporting, but every window is
    collected because a shorter window inside a longer duplicate is itself a
    genuine repeat; set arithmetic against the baseline removes the noise.
    """
    words = [word.casefold() for word in _SPAN_TOKEN_RE.findall(text or "")]
    if len(words) < min_words * 2:
        return set()
    seen: dict[str, int] = {}
    for start in range(len(words) - min_words + 1):
        window = " ".join(words[start:start + min_words])
        seen[window] = seen.get(window, 0) + 1
    return {span for span, count in seen.items() if count > 1}


def newly_repeated_spans(
    previous: str,
    candidate: str,
    min_words: int = _MIN_REPEATED_SPAN_WORDS,
) -> list[str]:
    """Word spans the candidate repeats that the previous translation did not."""
    if not (previous or "").strip():
        # An initial translation has no same-language baseline, and source
        # repetition cannot be matched across languages. Staying silent is the
        # only safe answer: this is an EDIT guard.
        return []
    return sorted(
        _repeated_spans(candidate, min_words) - _repeated_spans(previous, min_words)
    )


def _source_grounded_repeated_spans(
    source: str,
    candidate: str,
    spans: list[str],
    min_words: int = _MIN_REPEATED_SPAN_WORDS,
) -> set[str]:
    """Identify candidate repetition supported by aligned source repetition.

    Exact phrases cannot be compared across languages.  Paragraph identity can
    still provide conservative evidence: only exempt a Persian repeated span
    when the corresponding source paragraph(s) themselves contain an exact
    repeated source-language span of the same minimum size.
    """
    source_paragraphs = _paragraphs(source)
    candidate_paragraphs = _paragraphs(candidate)
    if not spans or len(source_paragraphs) != len(candidate_paragraphs):
        return set()

    normalized_candidates = [
        " ".join(word.casefold() for word in _SPAN_TOKEN_RE.findall(paragraph))
        for paragraph in candidate_paragraphs
    ]
    grounded: set[str] = set()
    for span in spans:
        aligned_indices = [
            index for index, paragraph in enumerate(normalized_candidates)
            if span in paragraph
        ]
        if not aligned_indices:
            continue
        aligned_source = "\n\n".join(
            source_paragraphs[index] for index in aligned_indices
        )
        if _repeated_spans(aligned_source, min_words):
            grounded.add(span)
    return grounded


_PERSIAN_DIGITS = "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9"
# A run of digits in any supported script, interrupted by corruption. Requires at
# least one digit AND one corrupt character, so clean numbers are never touched.
_CORRUPT_NUMBER_RE = re.compile(
    "(?<![0-9\u06f0-\u06f9])"
    "(?=[0-9\u06f0-\u06f9\ufffd\ue000-\uf8ff]*[0-9\u06f0-\u06f9])"
    "(?=[0-9\u06f0-\u06f9\ufffd\ue000-\uf8ff]*[\ufffd\ue000-\uf8ff])"
    "[0-9\u06f0-\u06f9\ufffd\ue000-\uf8ff]+"
    "(?![0-9\u06f0-\u06f9])"
)


def repair_corruption(source: str, candidate: str) -> tuple[str, list[dict[str, Any]]]:
    """Reconstruct corrupted numerals that the source disambiguates completely.

    Item 15's rule, and the whole safety argument: a value is repaired **only
    when the source provides one unique reconstruction**. A corrupted decade
    becomes the source's year only when exactly one source number has the same
    length and agrees on every surviving digit. Anything ambiguous is left
    alone, so it still reaches the gate, is still blocked, and still reaches a
    human. Repairing on a guess would corrupt a translation that a reviewer
    could have fixed correctly.

    Returns the (possibly unchanged) text plus a record of every repair, so the
    QA report can show exactly what was reconstructed and from what.
    """
    candidate = candidate or ""
    if not _CORRUPT_NUMBER_RE.search(candidate):
        return candidate, []

    source_numbers = [
        match.group().strip()
        for match in _NUMBER_RE.finditer(_numeric_text(source or ""))
    ]

    repairs: list[dict[str, Any]] = []

    def _rebuild(match: re.Match[str]) -> str:
        token = match.group()
        # Persian output stays Persian; the script is taken from the survivors.
        persian = any(char in _PERSIAN_DIGITS for char in token)
        known = [
            char.translate(_DIGIT_MAP) if char.isdigit() or char in _PERSIAN_DIGITS
            else None
            for char in token
        ]
        matches = [
            number for number in source_numbers
            if len(number) == len(known)
            and all(
                digit is None or digit == number[index]
                for index, digit in enumerate(known)
            )
        ]
        unique = sorted(set(matches))
        if len(unique) != 1:
            repairs.append({
                "token_length": len(token),
                "known_digits": sum(1 for digit in known if digit is not None),
                "source_candidates": len(unique),
                "repaired": False,
                "reason": (
                    "no_source_match" if not unique else "ambiguous_source_match"
                ),
            })
            return token
        rebuilt = unique[0]
        if persian:
            rebuilt = "".join(
                _PERSIAN_DIGITS[int(char)] if char.isdigit() else char
                for char in rebuilt
            )
        repairs.append({
            "token_length": len(token),
            "known_digits": sum(1 for digit in known if digit is not None),
            "source_candidates": 1,
            "repaired": True,
            "reason": "unique_source_reconstruction",
        })
        return rebuilt

    return _CORRUPT_NUMBER_RE.sub(_rebuild, candidate), repairs


def corruption_artifacts(text: str) -> dict[str, int]:
    """Count Unicode damage by category. Empty dict means clean."""
    text = text or ""
    counts = {
        "replacement_character": text.count(_REPLACEMENT_CHAR),
        "unrestored_sentinel": len(_PRIVATE_USE_RE.findall(text)),
        "control_character": len(_CONTROL_CHAR_RE.findall(text)),
    }
    return {name: total for name, total in counts.items() if total}


def describe_corruption(text: str) -> list[str]:
    """Human-readable samples of corrupted spans, for QA evidence."""
    text = text or ""
    samples: list[str] = []
    for match in re.finditer(
        f"[{_REPLACEMENT_CHAR}\ue000-\uf8ff\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]",
        text,
    ):
        start = max(0, match.start() - 12)
        end = min(len(text), match.end() + 12)
        codepoint = f"U+{ord(match.group()):04X}"
        samples.append(f"{codepoint} in {text[start:end]!r}")
        if len(samples) >= 5:
            break
    return samples


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
    for _start, _end, raw_value in _identifier_occurrences(text):
        value = " ".join(raw_value.split()).strip(".,;)")
        value = re.sub(r"[\u2010-\u2015]", "-", value)
        value = re.sub(
            r"^\s*(?:ISBN(?:-1[03])?|ISSN)\s*:?\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )
        if value:
            values.append(value.casefold())
    return Counter(values)


def extract_labeled_identifier_surfaces(text: str) -> Counter[str]:
    """Extract complete ISBN/ISSN surfaces, including their source labels."""
    values = []
    for match in _LABELED_IDENTIFIER_RE.finditer(text or ""):
        value = " ".join(match.group().split()).strip(".,;)")
        value = re.sub(r"[\u2010-\u2015]", "-", value)
        values.append(value.casefold())
    return Counter(values)


def _identifier_identity(value: str) -> str:
    """Return a script-insensitive identity used only to locate damaged IDs."""
    normalized = (value or "").translate(_DIGIT_MAP)
    normalized = re.sub(r"[\u2010-\u2015]", "-", normalized)
    normalized = re.sub(
        r"^\s*(?:ISBN(?:-1[03])?|ISSN)\s*:?\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    return re.sub(r"[^A-Za-z0-9]", "", normalized).casefold()


def _identifier_payload(value: str) -> str:
    """Strip a source label while preserving the exact identifier payload."""
    return re.sub(
        r"^\s*(?:ISBN\s*(?:-\s*(?:10|13))?|ISSN)\s*:?\s*",
        "",
        value or "",
        flags=re.IGNORECASE,
    )


def _identity_candidate_pattern(identity: str) -> re.Pattern[str]:
    """Build a source-grounded matcher for localized/respaced identifier text."""
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    parts: list[str] = []
    for character in identity:
        if character.isdigit():
            number = int(character)
            parts.append(re.escape(
                character + persian_digits[number] + arabic_digits[number]
            ).join(("[", "]")))
        else:
            parts.append(re.escape(character))
    separator = r"[\s._:/\-\u066b\u066c\u200e\u200f\u2010-\u2015]*"
    return re.compile(
        rf"(?<![A-Za-z0-9]){separator.join(parts)}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


def _identifier_occurrences(text: str) -> list[tuple[int, int, str]]:
    """Return non-overlapping identifiers, preferring complete specific spans."""
    candidates: list[tuple[int, int, str, int]] = []
    for priority, pattern in enumerate(_IDENTIFIER_PATTERNS):
        for match in pattern.finditer(text or ""):
            value = match.group().strip(".,;)")
            if _PROSE_FOOTNOTE_SUFFIX_RE.fullmatch(value):
                continue
            if value:
                candidates.append(
                    (match.start(), match.start() + len(value), value, priority)
                )

    # A complete catalogue/grant code may contain a substring that also looks
    # like an ordinary numeric identifier. Selecting by span length first keeps
    # the full source-authored token authoritative without relying on its prefix.
    selected: list[tuple[int, int, str, int]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (-(item[1] - item[0]), item[3], item[0]),
    ):
        start, end, _value, _priority = candidate
        if any(start < kept_end and end > kept_start for kept_start, kept_end, *_ in selected):
            continue
        selected.append(candidate)
    return [
        (start, end, value)
        for start, end, value, _priority in sorted(selected, key=lambda item: item[0])
    ]


def restore_source_identifiers(source: str, translation: str) -> tuple[str, dict[str, Any]]:
    """Restore uniquely matched identifier values without translating nearby labels.

    The repair is deliberately conservative: a localized or respaced candidate is
    replaced only when its alphanumeric identity maps to exactly one source value.
    """
    source_by_identity: dict[str, list[str]] = {}
    for _start, _end, value in _identifier_occurrences(source):
        identity = _identifier_identity(value)
        if identity:
            source_by_identity.setdefault(identity, []).append(value)

    replacements: list[tuple[int, int, str, str]] = []
    candidate_occurrences = _identifier_occurrences(translation)
    occupied = [(start, end) for start, end, _value in candidate_occurrences]
    for identity in sorted(source_by_identity, key=len, reverse=True):
        pattern = _identity_candidate_pattern(identity)
        for match in pattern.finditer(translation or ""):
            overlapping_candidates = [
                item for item in candidate_occurrences
                if match.start() < item[1] and match.end() > item[0]
            ]
            overlaps = [
                (start, end) for start, end, _value in overlapping_candidates
                if match.start() < end and match.end() > start
            ]
            if overlaps and not all(
                match.start() <= start and match.end() >= end
                for start, end in overlaps
            ):
                # A malformed target can expose only a leading fragment as an
                # identifier (for example ``RES-۰۵۱``) while the exact
                # source-grounded numeric identity continues after it.  Such a
                # fragment must not block restoration of the complete source
                # value, but an independently valid source identifier still
                # retains precedence.
                if any(
                    _identifier_identity(value) in source_by_identity
                    for _start, _end, value in overlapping_candidates
                ):
                    continue
            if overlaps:
                overlap_set = set(overlaps)
                candidate_occurrences = [
                    item for item in candidate_occurrences
                    if (item[0], item[1]) not in overlap_set
                ]
                occupied = [
                    item for item in occupied if item not in overlap_set
                ]
            candidate_occurrences.append(
                (match.start(), match.end(), match.group())
            )
            occupied.append(match.span())

    flexible_numeric = re.compile(
        rf"(?<![{_IDENTIFIER_DIGITS}])"
        rf"[{_IDENTIFIER_DIGITS}Xx](?:[\s\-\u2010-\u2015]*"
        rf"[{_IDENTIFIER_DIGITS}Xx]){{8,30}}"
        rf"(?![{_IDENTIFIER_DIGITS}])"
    )
    for match in flexible_numeric.finditer(translation or ""):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        candidate_occurrences.append((match.start(), match.end(), match.group()))

    flexible_domains = re.compile(
        r"(?<![@\w])(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,62})\s*\.\s*)+"
        r"[A-Za-z]{2,63}(?:\s*/\s*[^\s<>()]*)?",
        re.IGNORECASE,
    )
    flexible_codes = re.compile(
        rf"(?<![A-Za-z0-9])"
        rf"(?=[A-Za-z{_IDENTIFIER_DIGITS}\s.\-/\u2010-\u2015]*[A-Za-z])"
        rf"(?=[A-Za-z{_IDENTIFIER_DIGITS}\s.\-/\u2010-\u2015]*[{_IDENTIFIER_DIGITS}])"
        rf"(?=[A-Za-z{_IDENTIFIER_DIGITS}\s.\-/\u2010-\u2015]*[./])"
        rf"[A-Za-z{_IDENTIFIER_DIGITS}]{{1,12}}"
        rf"(?:\s*[.\-/\u2010-\u2015]\s*[A-Za-z{_IDENTIFIER_DIGITS}]{{1,12}}){{1,5}}"
        rf"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    flexible_postcodes = re.compile(
        rf"(?<![A-Za-z0-9])[A-Z]{{1,2}}\s*[{_IDENTIFIER_DIGITS}]"
        rf"[A-Z{_IDENTIFIER_DIGITS}]?\s+[{_IDENTIFIER_DIGITS}]\s*[A-Z]{{2}}"
        rf"(?![A-Za-z0-9])"
    )
    occupied = [(start, end) for start, end, _value in candidate_occurrences]
    for pattern in (flexible_domains, flexible_codes, flexible_postcodes):
        for match in pattern.finditer(translation or ""):
            if any(
                match.start() < end and match.end() > start
                for start, end in occupied
            ):
                continue
            candidate_occurrences.append(
                (match.start(), match.end(), match.group())
            )
            occupied.append(match.span())

    for start, end, value in candidate_occurrences:
        identity = _identifier_identity(value)
        candidates = list(dict.fromkeys(source_by_identity.get(identity, [])))
        source_payloads = list(dict.fromkeys(
            _identifier_payload(candidate) for candidate in candidates
        ))
        if len(source_payloads) != 1:
            continue
        source_value = candidates[0]
        source_payload = source_payloads[0]
        candidate_payload = _identifier_payload(value)
        source_has_label = source_payload != source_value
        candidate_has_label = candidate_payload != value
        replacement_start = start
        replacement_before = value
        if source_has_label:
            replacement = source_value
            if not candidate_has_label:
                prefix_start = max(0, start - 40)
                prefix = (translation or "")[prefix_start:start]
                label_match = _FLEXIBLE_IDENTIFIER_LABEL_SUFFIX_RE.search(prefix)
                if label_match:
                    replacement_start = prefix_start + label_match.start()
                    replacement_before = (
                        (translation or "")[replacement_start:start] + value
                    )
        else:
            replacement = source_value
        if replacement_before != replacement:
            replacements.append((
                replacement_start,
                end,
                replacement_before,
                replacement,
            ))

    repaired = translation
    edits: list[dict[str, str]] = []
    for start, end, before, after in sorted(
        replacements, key=lambda item: item[0], reverse=True
    ):
        repaired = repaired[:start] + after + repaired[end:]
        edits.append({"before": before, "after": after})
    edits.reverse()
    return repaired, {"repair_count": len(edits), "repairs": edits}


def mixed_script_artifacts(text: str) -> list[str]:
    """Return contiguous Latin/Persian tokens that indicate protocol corruption."""
    artifacts = set(_MIXED_SCRIPT_TOKEN_RE.findall(text or ""))
    artifacts.update(_PARENTHETICAL_PERSIAN_SUFFIX_RE.findall(text or ""))
    return sorted(artifacts, key=str.casefold)


def is_bibliographic_marker(value: str) -> bool:
    """Recognize compact source-authored publishing apparatus such as ``(pb)``."""
    return bool(_BIBLIOGRAPHIC_MARKER_RE.fullmatch(" ".join((value or "").split())))


def protected_english_originals(source: str, translation: str) -> list[str]:
    """Return inline English originals that are grounded in the source."""
    source_folded = (source or "").casefold()
    values = []
    for value in _ENGLISH_PAREN_RE.findall(translation or ""):
        cleaned = " ".join(value.split()).strip()
        if cleaned and cleaned.casefold() in source_folded:
            values.append(cleaned)
    return sorted(set(values), key=str.casefold)


def source_abbreviation_expansions(source: str) -> list[str]:
    """Find source-authored acronym definitions from their initials.

    This is intentionally structural rather than lexical.  It covers abbreviation
    lists and compact definitions while refusing ordinary English prose whose
    significant-word initials do not reproduce the preceding acronym.
    """
    text = source or ""
    expansions: list[str] = []
    for match in _ABBREVIATION_DEFINITION_RE.finditer(text):
        acronym = re.sub(r"[^A-Z]", "", match.group("acronym"))
        if not 2 <= len(acronym) <= 20:
            continue
        tail = text[match.end():match.end() + 360]
        words = list(re.finditer(r"[A-Za-z][A-Za-z'\u2019-]*", tail))[:30]
        initials = ""
        for index, word in enumerate(words):
            folded = word.group().casefold()
            if folded not in _ABBREVIATION_CONNECTORS:
                initials += word.group()[0].upper()
            if not acronym.startswith(initials):
                break
            if initials == acronym and index >= 2:
                value = tail[words[0].start():word.end()]
                expansions.append(" ".join(value.split()))
                break
    return sorted(set(expansions), key=str.casefold)


def protected_source_citations(source: str, translation: str) -> list[str]:
    """Return source-grounded author-year atoms retained in a translation.

    Framing prose such as ``see chapter 4`` is intentionally excluded. It may
    be translated while the author, year, and structural number remain guarded
    by apparatus and numeric integrity checks.
    """
    values: list[str] = []
    for parenthetical in _SOURCE_PAREN_RE.findall(source or ""):
        for year in _CITATION_YEAR_RE.finditer(parenthetical):
            prefix = parenthetical[:year.start()]
            name_match = _CITATION_NAME_RE.search(prefix)
            name = name_match.group().strip() if name_match else ""
            name = re.sub(r"^(?:cf\.?|see)\s+", "", name, flags=re.IGNORECASE)
            atom = " ".join(part for part in (name, year.group()) if part)
            if atom and _apparatus_value_present(atom, translation):
                values.append(atom)
    return sorted(set(values), key=str.casefold)


def protected_source_apparatus(source: str, translation: str) -> list[str]:
    """Return source-delimited Latin spans retained verbatim in a translation.

    This is structural protection, not a vocabulary list. Ordinary English prose
    that was translated is absent from the result; only parenthetical or quoted
    source material already carried into the prior valid translation is retained.
    """
    translation_folded = normalize_for_match(translation)
    values: list[str] = []
    candidates: list[str] = []
    candidates.extend(source_abbreviation_expansions(source))
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


def _grounded_phrase_spans(text: str, phrase: str) -> list[tuple[int, int]]:
    """Locate a source-grounded phrase despite harmless whitespace changes."""
    words = re.findall(r"\S+", phrase or "")
    if not words:
        return []
    pattern = re.compile(r"\s+".join(re.escape(word) for word in words), re.IGNORECASE)
    return [match.span() for match in pattern.finditer(text or "")]


def _latin_token_identity(value: str) -> str:
    """Normalize a Latin token for source-provenance comparison only."""
    normalized = re.sub(r"[-\u2010-\u2015]", "", value or "").casefold()
    return re.sub(r"(?:['\u2019]s)$", "", normalized)


def _source_latin_identity_sequence(source: str) -> list[str]:
    return [
        _latin_token_identity(match.group())
        for match in _LATIN_PROSE_TOKEN_RE.finditer(source or "")
    ]


def _source_latin_identities(source: str) -> set[str]:
    return set(_source_latin_identity_sequence(source))


def _contains_identity_sequence(
    source_identities: list[str],
    candidate_identities: list[str],
) -> bool:
    """Return whether a normalized token sequence occurs contiguously."""
    width = len(candidate_identities)
    if not width or width > len(source_identities):
        return False
    return any(
        source_identities[index:index + width] == candidate_identities
        for index in range(len(source_identities) - width + 1)
    )


def _source_grounded_parenthetical_spans(
    source: str,
    translation: str,
) -> list[tuple[int, int]]:
    """Protect compact target parentheticals whose Latin words occur in source."""
    source_identities = _source_latin_identity_sequence(source)
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\(([^()\n]{1,160})\)", translation or ""):
        tokens = list(_LATIN_PROSE_TOKEN_RE.finditer(match.group(1)))
        if not tokens or len(tokens) > 8:
            continue
        identities = [_latin_token_identity(token.group()) for token in tokens]
        if _contains_identity_sequence(source_identities, identities):
            spans.append(match.span())
    return spans


def _citation_author_connector(text: str, start: int, end: int) -> bool:
    """Recognize ``Surname and Surname YEAR`` without treating prose as citation."""
    left = (text or "")[max(0, start - 80):start]
    right = (text or "")[end:min(len(text or ""), end + 100)]
    name = rf"[{_LATIN_LETTERS}][{_LATIN_LETTERS}'\u2019.-]*"
    return bool(
        re.search(rf"{name}\s*$", left)
        and re.match(rf"\s+{name}(?:\s+{name}){{0,3}}\s+\(?\d{{4}}", right)
    )


def unexpected_latin_prose(
    source: str,
    translation: str,
    *,
    allowed_originals: Iterable[str] = (),
    structural_role: str = "body",
    chapter_title: str = "",
) -> list[dict[str, Any]]:
    """Find Latin prose that is not justified by source scholarly apparatus.

    The check is deliberately provenance-based. It does not maintain a list of
    book-specific words: identifiers, approved originals, source-delimited
    multilingual apparatus, citation names, acronyms, and Roman page numbers are
    allowed; an unexplained model note or foreign word in Persian prose is not.
    """
    text = translation or ""
    protected_spans = [
        (start, end)
        for start, end, _value in _identifier_occurrences(text)
    ]
    grounded_phrases = list(protected_source_apparatus(source, text))
    grounded_phrases.extend(
        value for value in allowed_originals
        if value and re.search(re.escape(value), source or "", re.IGNORECASE)
    )
    for phrase in grounded_phrases:
        protected_spans.extend(_grounded_phrase_spans(text, phrase))
    protected_spans.extend(_source_grounded_parenthetical_spans(source, text))

    # Preserve source-attested foreign phrases carrying an apostrophe or
    # non-ASCII Latin spelling. Ordinary English sequences remain subject to
    # the leak check, so untranslated source prose cannot pass this exception.
    phrase_token = rf"[{_LATIN_LETTERS}][{_LATIN_LETTERS}'\u2019-]*"
    source_tokens = list(re.finditer(phrase_token, source or ""))
    for index, token_match in enumerate(source_tokens):
        token = token_match.group()
        if not (
            re.search(r"['\u2019]", token)
            or re.search(r"[^\x00-\x7f]", token)
        ):
            continue
        for start_index in range(max(0, index - 2), index + 1):
            for end_index in range(index, min(len(source_tokens), index + 3)):
                phrase = (source or "")[
                    source_tokens[start_index].start():source_tokens[end_index].end()
                ]
                protected_spans.extend(_grounded_phrase_spans(text, phrase))

    source_identities = _source_latin_identities(source)
    source_is_apparatus = bool(
        str(structural_role or "body").casefold() in {
            "bibliography",
            "catalog",
            "front_matter",
            "heading",
            "index",
            "table",
        }
        or _APPARATUS_CHAPTER_RE.search(chapter_title or "")
        or _SOURCE_METADATA_RE.search(source or "")
    )

    findings: list[dict[str, Any]] = []
    for match in _LATIN_PROSE_TOKEN_RE.finditer(text):
        start, end = match.span()
        token = match.group()
        if any(start < span_end and end > span_start for span_start, span_end in protected_spans):
            continue
        source_present = _latin_token_identity(token) in source_identities
        if source_present and source_is_apparatus:
            continue
        if source_present and (
            token.isupper()
            or _ROMAN_NUMERAL_RE.fullmatch(token)
        ):
            continue
        context = text[max(0, start - 80):min(len(text), end + 80)]
        citation_context = bool(_CITATION_YEAR_RE.search(context))
        if source_present and citation_context and (
            token[:1].isupper()
            or token.casefold() in {"et", "al", "ibid", "doi"}
            or (
                token.casefold() in {"and", "or"}
                and _citation_author_connector(text, start, end)
            )
        ):
            continue
        findings.append({
            "token": token,
            "offset": start,
            "source_grounded": source_present,
            "context": context[:180],
            "reason": "unapproved_latin_prose",
        })
    return findings[:50]


def _apparatus_value_present(value: str, candidate: str) -> bool:
    """Match citation atoms despite harmless citation-style reformatting."""
    normalized_candidate = normalize_for_match(candidate)
    years = _CITATION_YEAR_RE.findall(value or "")
    if not years:
        return normalize_for_match(value) in normalized_candidate
    words = re.findall(r"[A-Za-z][A-Za-z'\u2019.-]*", value or "")
    surnames = [
        word.strip(".").casefold() for word in words
        if word.casefold() not in {"cf", "see"}
    ]
    surname_present = not surnames or surnames[-1] in normalized_candidate
    return surname_present and all(
        normalize_for_match(year) in normalized_candidate for year in years
    )


def _persian_cardinal(value: int) -> str:
    """Return one standard Persian cardinal form for a bounded integer."""
    if not 0 <= value <= 9999:
        return ""
    if value < 10:
        return _PERSIAN_UNITS[value]
    if value < 20:
        return _PERSIAN_TEENS[value]
    if value < 100:
        tens, unit = divmod(value, 10)
        cardinal = _PERSIAN_TENS[tens * 10]
        if unit:
            cardinal += " \u0648 " + _PERSIAN_UNITS[unit]
        return cardinal
    if value < 1000:
        hundreds, remainder = divmod(value, 100)
        cardinal = _PERSIAN_HUNDREDS[hundreds * 100]
        if remainder:
            cardinal += " \u0648 " + _persian_cardinal(remainder)
        return cardinal
    thousands, remainder = divmod(value, 1000)
    prefix = "\u0647\u0632\u0627\u0631" if thousands == 1 else f"{_persian_cardinal(thousands)} \u0647\u0632\u0627\u0631"
    return prefix + ((" \u0648 " + _persian_cardinal(remainder)) if remainder else "")


def _persian_number_forms(value: int) -> set[str]:
    """Return conservative cardinal/ordinal Persian forms for 0..9999."""
    cardinal = _persian_cardinal(value)
    if not cardinal:
        return set()
    forms = {cardinal, cardinal + "\u0645"}
    forms.update(_PERSIAN_SPECIAL_ORDINALS.get(value, set()))
    return forms


def _persian_enumerator_forms(value: int) -> set[str]:
    """Return forms that can replace an ordered parenthesized list marker."""
    if value < 0 or value >= 100:
        return set()
    cardinal = (
        _PERSIAN_UNITS[value]
        if value < 10
        else _PERSIAN_TEENS[value]
        if value < 20
        else _PERSIAN_TENS[(value // 10) * 10]
        + ((" \u0648 " + _PERSIAN_UNITS[value % 10]) if value % 10 else "")
    )
    ordinal = cardinal + "\u0645"
    forms = {cardinal, ordinal, ordinal + "\u06cc", ordinal + "\u06cc\u0646"}
    for special in _PERSIAN_SPECIAL_ORDINALS.get(value, set()):
        forms.update({special, special + "\u06cc", special + "\u06cc\u0646"})
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
    enumeration_cursor = 0
    canonical_targets: dict[str, set[str]] = {}
    for source_label, target_labels in _STRUCTURAL_NUMBER_LABELS.items():
        canonical = "volume" if source_label == "vol" else source_label
        canonical_targets.setdefault(canonical, set()).update(target_labels)

    for index, occurrence in enumerate(source_occurrences):
        if occurrence.role == "enumeration":
            try:
                number = int(occurrence.value)
            except ValueError:
                continue
            candidates: list[tuple[int, int, str]] = []
            for form in _persian_enumerator_forms(number):
                for match in re.finditer(
                    rf"(?<![{persian_word}]){re.escape(form)}(?![{persian_word}])",
                    normalized_candidate[enumeration_cursor:],
                ):
                    start = enumeration_cursor + match.start()
                    end = enumeration_cursor + match.end()
                    if (start, end) not in consumed_spans:
                        candidates.append((start, end, match.group()))
            if not candidates:
                continue
            start, end, rendered = min(candidates, key=lambda item: item[0])
            enumeration_cursor = end
            consumed_spans.add((start, end))
            matched_source.add(index)
            evidence.append({
                "source_value": occurrence.value,
                "source_role": occurrence.role,
                "source_label": occurrence.label,
                "target_form": rendered,
            })
            continue
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
                shared_unit_forms: set[str] = set()
                for other_index, other in enumerate(source_occurrences):
                    if (
                        other_index == index
                        or other.role != "structural"
                        or other.label != occurrence.label
                    ):
                        continue
                    try:
                        shared_unit_forms.update(
                            _persian_number_forms(int(other.value))
                        )
                    except ValueError:
                        continue
                if shared_unit_forms:
                    following_number = "|".join(
                        re.escape(value)
                        for value in sorted(shared_unit_forms, key=len, reverse=True)
                    )
                    patterns += (
                        # A source-grounded range can share one trailing unit,
                        # e.g. ``four hundred to five hundred years``. Both
                        # number forms and an explicit connector are required.
                        rf"(?<![{persian_word}])(?P<number>{number_form})"
                        rf"[\s\u200c]{{0,3}}(?:\u062a\u0627|\u0627\u0644\u06cc|\u0648|"
                        rf"[-\u2010-\u2015])[\s\u200c]{{0,3}}"
                        rf"(?:{following_number})[\s\u200c-]{{0,3}}"
                        rf"{label}{label_suffix}(?![{persian_word}])",
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


def reconcile_numbers(source: str, candidate: str) -> dict[str, Any]:
    """Expose conservative numeric reconciliation to secondary QA checks."""
    result = _reconcile_numbers(source, candidate)
    return {
        "missing": list(result.missing.elements()),
        "missing_by_role": result.roles_payload(),
        "localized_equivalents": list(result.localized_equivalents),
    }


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
                if not _apparatus_value_present(value, candidate)
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

        # Unicode corruption. Baselined against BOTH source and previous so a
        # flaw already present in the source is reported, never treated as
        # something the translation introduced -- source anomalies are
        # preserved and noted, not silently corrected or rejected.
        candidate_corruption = corruption_artifacts(candidate)
        source_corruption = corruption_artifacts(source)
        previous_corruption = corruption_artifacts(previous)
        introduced_corruption = {
            name: total - max(
                source_corruption.get(name, 0), previous_corruption.get(name, 0)
            )
            for name, total in candidate_corruption.items()
            if total > max(
                source_corruption.get(name, 0), previous_corruption.get(name, 0)
            )
        }
        if introduced_corruption:
            add(
                "unicode_corruption", "blocking",
                "The proposed translation contains replacement, private-use, or "
                "control characters that the source does not.",
                categories=introduced_corruption,
                samples=describe_corruption(candidate),
            )
        elif candidate_corruption:
            add(
                "unicode_corruption_still_present", "warning",
                "The edit did not introduce Unicode corruption, but earlier "
                "damage remains.",
                categories=candidate_corruption,
                samples=describe_corruption(candidate),
            )

        mixed_artifacts = mixed_script_artifacts(candidate)
        previous_mixed_artifacts = set(mixed_script_artifacts(previous))
        newly_mixed = [
            value for value in mixed_artifacts if value not in previous_mixed_artifacts
        ]
        if newly_mixed:
            add(
                "mixed_script_corruption", "blocking",
                "A contiguous token unexpectedly mixes Latin and Persian scripts.",
                tokens=newly_mixed,
            )
        elif mixed_artifacts:
            add(
                "mixed_script_corruption_still_present", "warning",
                "The edit did not introduce mixed-script corruption, but earlier damage remains.",
                tokens=mixed_artifacts,
            )

        source_words = _ASCII_WORD_RE.findall(source)
        if len(source_words) >= 8 and len(_PERSIAN_RE.findall(candidate)) < 3:
            add(
                "target_language_missing", "blocking",
                "The proposed output does not contain enough Persian text.",
            )

        def _duplicated(paragraphs: list[str]) -> set[str]:
            return {
                text for text, count in Counter(
                    normalize_for_match(p) for p in paragraphs if len(p) >= 80
                ).items() if count > 1
            }

        # A book that legitimately repeats a long paragraph used to be blocked
        # and sent to human review for nothing. Two different baselines are
        # needed, because only one of them is same-language:
        #   * previous translation -- Persian, so the exact spans cancel;
        #   * source -- English, so normalised spans can NEVER match the
        #     candidate's. Only the COUNT of duplicated paragraphs is
        #     comparable across languages, so it is used as an allowance.
        candidate_duplicates = _duplicated(candidate_paragraphs)
        duplicate_paragraphs = sorted(
            candidate_duplicates - _duplicated(_paragraphs(previous))
        )
        source_duplicate_allowance = len(_duplicated(source_paragraphs))
        if len(duplicate_paragraphs) <= source_duplicate_allowance:
            duplicate_paragraphs = []
        if duplicate_paragraphs:
            add(
                "duplicate_paragraph", "blocking",
                "A substantial translated paragraph is duplicated.",
                duplicate_count=len(duplicate_paragraphs),
            )
        elif candidate_duplicates:
            add(
                "duplicate_paragraph_preserved", "info",
                "A duplicated paragraph mirrors repetition already present in "
                "the source or the previous translation.",
                duplicate_count=len(candidate_duplicates),
            )

        # Item 16: a duplicated multi-word span, which is smaller than a whole
        # paragraph and so invisible to the check above.
        introduced_spans = newly_repeated_spans(previous, candidate)
        grounded_spans = _source_grounded_repeated_spans(
            source, candidate, introduced_spans
        )
        introduced_spans = [
            span for span in introduced_spans if span not in grounded_spans
        ]
        if introduced_spans:
            add(
                "duplicate_span_introduced", "blocking",
                "The edit repeats a passage that the previous translation did not.",
                span_count=len(introduced_spans),
                samples=introduced_spans[:3],
            )
        elif grounded_spans:
            add(
                "duplicate_span_source_grounded", "info",
                "Repeated translated wording aligns with repeated source content.",
                span_count=len(grounded_spans),
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
                source_expansions = {
                    value.casefold()
                    for value in source_abbreviation_expansions(source)
                }
                unauthorized = [
                    value for value in candidate_originals
                    if value.casefold() not in allowed_folded
                    and value.casefold() not in source_expansions
                    and value.casefold() not in previous_folded
                    and not any(char.isdigit() for char in value)
                    and not (
                        is_bibliographic_marker(value)
                        and f"({value})".casefold() in source.casefold()
                    )
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
                    if not _apparatus_value_present(value, candidate)
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
