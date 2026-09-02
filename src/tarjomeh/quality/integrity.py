"""Deterministic integrity checks for translation replacement operations."""

from __future__ import annotations

import re
import unicodedata
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
_PLAIN_NOTE_DIGITS = "0-9\u0660-\u0669\u06f0-\u06f9"
_PLAIN_ATTACHED_NOTE_RE = re.compile(
    rf"(?<![{_PLAIN_NOTE_DIGITS}])"
    rf"(?P<leader>[)\],.;:!?\u060c\u061b\u061f])"
    rf"(?P<marker>[{_PLAIN_NOTE_DIGITS}]{{1,3}})"
    rf"(?=\s*(?:[-\u2010-\u2015]|[A-Za-z\u0600-\u06ff]|$))"
)
_PLAIN_SPACED_NOTE_RE = re.compile(
    rf"(?<![{_PLAIN_NOTE_DIGITS}])"
    rf"(?P<leader>[)\],.;:!?\u060c\u061b\u061f])\s*"
    rf"(?P<marker>[{_PLAIN_NOTE_DIGITS}]{{1,3}})"
    rf"(?=\s*(?:[-\u2010-\u2015\u060c\u061b\u061f,;.!?]|"
    rf"[A-Za-z\u0600-\u06ff]|$))"
)
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
_SOURCE_STRUCTURAL_REFERENCE_RE = re.compile(
    r"\b(?P<see>see\s+)?"
    r"(?P<label>chapter|section|table|figure|part)\s+"
    r"(?P<number>[0-9]{1,3})\b",
    re.IGNORECASE,
)
_PERSIAN_STRUCTURAL_LABELS = {
    "chapter": "فصل",
    "section": "بخش",
    "table": "جدول",
    "figure": "شکل",
    "part": "بخش",
}
_ASCII_TO_PERSIAN_DIGITS = str.maketrans(
    "0123456789", "۰۱۲۳۴۵۶۷۸۹"
)
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
_LOCALIZED_IDENTIFIER_LABEL_SUFFIX_RE = re.compile(
    rf"(?:(?:\u0634\u0645\u0627\u0631\u0647\s+)?"
    rf"\u0634\u0627\u0628\u06a9|"
    rf"\u0634\u0645\u0627\u0631\u0647\s+\u0627\u0633\u062a\u0627\u0646\u062f\u0627\u0631\u062f\s+"
    rf"\u0628\u06cc\u0646\u200c?\u0627\u0644\u0645\u0644\u0644\u06cc\s+\u06a9\u062a\u0627\u0628)"
    rf"(?:\s*[-\u2010-\u2015]?\s*"
    rf"[{_IDENTIFIER_DIGITS}]\s*[{_IDENTIFIER_DIGITS}])?\s*:?\s*$"
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
_PARENTHETICAL_CONTEXT_LEADERS = frozenset({
    "although", "because", "both", "either", "however", "neither", "nor",
    "since", "therefore", "these", "those", "thus", "whereas", "while",
})
_MARKDOWN_EMPHASIS_RE = re.compile(
    r"(?<!\*)\*{1,2}\s*(?P<content>[^*\n]{1,240}?)\s*\*{1,2}(?!\*)"
)
_DETACHED_EZAFE_RE = re.compile(
    r"\)\s+[\u06cc\u064a](?=\s|[\u060c\u061b:,.!?\u061f]|$)"
)
_TATWEEL_SEPARATOR_RE = re.compile(r"(?:(?<=\s)|^)\u0640{2,}(?=\s|$)")
_PERSIAN_SUFFIX_AFTER_ORIGINAL_RE = re.compile(
    rf"(?P<anchor>[{_PERSIAN_LETTER_CLASS}]+"
    rf"(?:\u200c[{_PERSIAN_LETTER_CLASS}]+)*)\s+"
    rf"(?P<original>\((?=[^()\n]*[A-Za-z])[^()\n]{{1,200}}\))"
    rf"(?P<join>\u200c?)(?P<suffix>\u0627\u06cc|\u06cc|\u0647\u0627|\u0627\u0646|\u0627\u062a|"
    rf"\u062a\u0631(?:\u06cc\u0646)?|\u0627\u0645|\u0627\u0634|\u0645\u0627\u0646|\u062a\u0627\u0646|\u0634\u0627\u0646)"
    rf"(?![{_PERSIAN_LETTER_CLASS}])"
)
_NESTED_INLINE_ORIGINAL_RE = re.compile(
    rf"\((?P<before>[^()\n]*[{_PERSIAN_LETTER_CLASS}][^()\n]*?)"
    rf"(?P<original>\((?=[^()\n]*[A-Za-z])[^()\n]{{1,200}}\))"
    rf"(?P<after>[^()\n]*)\)"
)
_FOREIGN_SCRIPT_PATTERNS = {
    "cyrillic": re.compile(r"[\u0400-\u052f]+"),
    "greek": re.compile(r"[\u0370-\u03ff]+"),
    "hebrew": re.compile(r"[\u0590-\u05ff]+"),
    "han": re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+"),
    "hangul": re.compile(r"[\uac00-\ud7af]+"),
    "kana": re.compile(r"[\u3040-\u30ff]+"),
}
_SUSPECT_PDF_WORD_BREAK_RE = re.compile(
    r"(?<![A-Z])(?:[A-Z][a-z]{2,}|[a-z]{3,})[-\u2010-\u2015][a-z]{3,}"
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
_CATALOG_NAME_LINE_RE = re.compile(
    rf"^\s*[{_LATIN_LETTERS}][{_LATIN_LETTERS}'\u2019.-]+\s*,\s*"
    rf"[{_LATIN_LETTERS}][{_LATIN_LETTERS}'\u2019.-]+"
    rf"(?:\s+[{_LATIN_LETTERS}][{_LATIN_LETTERS}'\u2019.-]+){{0,5}}\.?\s*$"
)
_CATALOG_TITLE_LINE_RE = re.compile(
    rf"^\s*[^/\n]{{2,220}}\s+/\s+"
    rf"[{_LATIN_LETTERS}][{_LATIN_LETTERS}'\u2019.-]+"
    rf"(?:\s+[{_LATIN_LETTERS}][{_LATIN_LETTERS}'\u2019.-]+){{0,5}}\.?\s*$"
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
    normalized = _PLAIN_ATTACHED_NOTE_RE.sub(
        lambda match: match.group("leader"), normalized
    )
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
    normalized = _PLAIN_ATTACHED_NOTE_RE.sub(
        lambda match: match.group("leader"), normalized
    )
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


def extract_note_markers(text: str, *, allow_spaced: bool = False) -> list[str]:
    """Extract note markers, including PDF-flattened source superscripts.

    PyMuPDF can preserve superscript evidence in paragraph metadata while its
    plain text surface contains forms such as ``Rousseau)1`` or ``follows,3``.
    Attached punctuation is conservative source evidence. Spaced forms are
    accepted only when comparing a target against source-confirmed markers.
    """
    markers = []
    for bracketed, superscript in _NOTE_RE.findall(text or ""):
        markers.append(bracketed or superscript.translate(_SUPERSCRIPT_MAP))
    matcher = _PLAIN_SPACED_NOTE_RE if allow_spaced else _PLAIN_ATTACHED_NOTE_RE
    for match in matcher.finditer((text or "").translate(_DIGIT_MAP)):
        markers.append(match.group("marker"))
    return markers


def available_note_markers(text: str, required: Counter[str]) -> Counter[str]:
    """Return target note evidence without treating arbitrary numbers as notes."""
    strict = Counter(extract_note_markers(text))
    if not required:
        return strict
    spaced = Counter(extract_note_markers(text, allow_spaced=True))
    return strict | Counter({value: spaced[value] for value in required if spaced[value]})


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


def _identifier_label_identity(value: str) -> str:
    """Return the exact normalized ISBN/ISSN label, without its payload."""
    match = re.match(
        r"^\s*(ISBN\s*(?:-\s*(?:10|13))?|ISSN)\b",
        value or "",
        re.IGNORECASE,
    )
    if not match:
        return ""
    return re.sub(r"\s+", "", match.group(1)).casefold()


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
        candidate_label = _identifier_label_identity(value)
        label_matched = [
            candidate for candidate in candidates
            if _identifier_label_identity(candidate) == candidate_label
        ]
        source_value = label_matched[0] if label_matched else candidates[0]
        source_payload = source_payloads[0]
        candidate_payload = _identifier_payload(value)
        source_has_label = source_payload != source_value
        candidate_has_label = candidate_payload != value
        replacement_start = start
        replacement_before = value
        if len(candidates) > 1 and not candidate_label:
            # The same payload may be printed once as ISBN and elsewhere as
            # ISBN-13. Without a local target label, restore only the literal
            # payload instead of manufacturing either source label.
            replacement = source_payload
        elif source_has_label:
            replacement = source_value
            prefix_start = max(0, start - 80)
            prefix = (translation or "")[prefix_start:start]
            localized_label = _LOCALIZED_IDENTIFIER_LABEL_SUFFIX_RE.search(prefix)
            if candidate_has_label and localized_label:
                replacement_start = prefix_start + localized_label.start()
                replacement_before = (
                    (translation or "")[replacement_start:start] + value
                )
            elif not candidate_has_label:
                if localized_label:
                    replacement = source_payload
                else:
                    label_match = _FLEXIBLE_IDENTIFIER_LABEL_SUFFIX_RE.search(
                        prefix
                    )
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


def repeated_persian_word_artifacts(text: str) -> list[dict[str, Any]]:
    """Report adjacent repeated Persian lexical words without rewriting text."""
    word = rf"[{_PERSIAN_LETTER_CLASS}]{{4,}}"
    pattern = re.compile(
        rf"(?<![{_PERSIAN_LETTER_CLASS}])(?=(?P<first>{word})"
        rf"(?:[ \t\u200c]+|[ \t]*[-\u2010-\u2015][ \t]*)"
        rf"(?P<second>{word})(?![{_PERSIAN_LETTER_CLASS}]))"
    )
    findings: list[dict[str, Any]] = []
    for match in pattern.finditer(text or ""):
        if match.group("first").casefold() != match.group("second").casefold():
            continue
        findings.append({
            "word": match.group("first"),
            "offset": match.start("first"),
            "context": (text or "")[
                max(0, match.start("first") - 60):
                min(len(text or ""), match.end("second") + 60)
            ],
        })
    return findings


_PERSIAN_DIACRITIC_CLASS = r"\u064b-\u065f\u0670\u06d6-\u06ed"
_PERSIAN_LEXICAL_TOKEN_RE = re.compile(
    rf"[{_PERSIAN_LETTER_CLASS}]+[{_PERSIAN_DIACRITIC_CLASS}]*"
    rf"(?:\u200c[{_PERSIAN_LETTER_CLASS}]+[{_PERSIAN_DIACRITIC_CLASS}]*)*"
)
_PERSIAN_COORDINATORS = frozenset({"و", "یا"})
_PERSIAN_GOVERNORS = frozenset({
    "از", "با", "بر", "برای", "به", "بدون", "پیرامون", "درباره", "در", "میان",
})
_SOURCE_GOVERNORS = frozenset({
    "about", "among", "around", "between", "by", "concerning", "for", "from",
    "in", "into", "of", "on", "regarding", "through", "to", "with", "without",
})
_SOURCE_LEXICAL_TOKEN_RE = re.compile(
    r"[A-Za-z\u00c0-\u024f]+(?:['\u2019-][A-Za-z\u00c0-\u024f]+)*"
)
_ADJACENT_REPEAT_SEPARATOR_RE = re.compile(
    r"^[ \t\u200c]+$"
)


def _repeat_token_key(value: str) -> str:
    """Ignore optional Persian combining marks only for corruption matching."""
    return "".join(
        character for character in (value or "").casefold()
        if not unicodedata.combining(character)
    )


def _adjacent_repeated_spans(
    text: str,
    token_re: re.Pattern[str],
    *,
    max_words: int = 6,
) -> list[dict[str, Any]]:
    """Return immediately repeated lexical spans without changing the text.

    Comparing token sequences rather than raw substrings catches both ordinary
    phrases (``A B A B``) and ZWNJ compounds. Sentence terminators are not
    accepted as separators, and neither is punctuation. This keeps legitimate
    punctuated lexical reuse outside the corruption signature while still
    catching replacement-boundary duplication.
    """
    value = text or ""
    tokens = list(token_re.finditer(value))
    findings: list[dict[str, Any]] = []
    index = 0
    while index < len(tokens) - 1:
        matched_width = 0
        maximum = min(max_words, (len(tokens) - index) // 2)
        for width in range(maximum, 0, -1):
            left = tokens[index:index + width]
            right = tokens[index + width:index + (2 * width)]
            if [_repeat_token_key(item.group()) for item in left] != [
                _repeat_token_key(item.group()) for item in right
            ]:
                continue
            if width == 1:
                lexical_length = sum(
                    character.isalpha() for character in left[0].group()
                )
                if lexical_length < 2:
                    continue
            separator = value[left[-1].end():right[0].start()]
            if not _ADJACENT_REPEAT_SEPARATOR_RE.fullmatch(separator):
                continue
            phrase = value[left[0].start():left[-1].end()]
            findings.append({
                "phrase": phrase,
                "normalized_phrase": " ".join(
                    _repeat_token_key(item.group()) for item in left
                ),
                "word_count": width,
                "offset": left[0].start(),
                "second_offset": right[0].start(),
                "end_offset": right[-1].end(),
                "context": value[
                    max(0, left[0].start() - 60):
                    min(len(value), right[-1].end() + 60)
                ],
            })
            matched_width = width
            break
        index += max(1, matched_width * 2)
    return findings


def repeated_persian_adjacent_span_artifacts(text: str) -> list[dict[str, Any]]:
    """Report adjacent repeated Persian compounds or phrases of 1-6 words."""
    return _adjacent_repeated_spans(text, _PERSIAN_LEXICAL_TOKEN_RE)


def newly_repeated_adjacent_spans(
    previous: str,
    candidate: str,
) -> list[dict[str, Any]]:
    """Return adjacent phrase repetitions introduced by the current edit."""
    if not (previous or "").strip():
        return []
    before = Counter(
        (int(item["word_count"]), str(item["normalized_phrase"]))
        for item in repeated_persian_adjacent_span_artifacts(previous)
    )
    excess = Counter(
        (int(item["word_count"]), str(item["normalized_phrase"]))
        for item in repeated_persian_adjacent_span_artifacts(candidate)
    ) - before
    introduced: list[dict[str, Any]] = []
    for item in repeated_persian_adjacent_span_artifacts(candidate):
        key = (int(item["word_count"]), str(item["normalized_phrase"]))
        if excess[key] <= 0:
            continue
        introduced.append(item)
        excess[key] -= 1
    return introduced


def _repeated_governed_spans(
    text: str,
    token_re: re.Pattern[str],
    governors: frozenset[str],
) -> list[dict[str, Any]]:
    """Find a short governed phrase repeated across one uninterrupted clause.

    This targets replacement-boundary damage such as ``about X ... about X``
    when a predicate has been spliced between the two copies. It deliberately
    ignores sentence and strong-clause boundaries and requires at least one
    intervening lexical token, so ordinary adjacent repetition remains owned
    by the existing adjacent-span detector.
    """
    value = text or ""
    tokens = list(token_re.finditer(value))
    normalized = [_repeat_token_key(token.group()) for token in tokens]
    findings: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    def contiguous_phrase(start: int, width: int) -> bool:
        """Require one real lexical phrase, not tokens skipped across apparatus."""
        for token_index in range(start, start + width - 1):
            gap = value[tokens[token_index].end():tokens[token_index + 1].start()]
            if not re.fullmatch(r"[ \t\u00a0]+", gap):
                return False
        return True

    for left_index, governor in enumerate(normalized):
        if governor not in governors:
            continue
        for width in range(3, 1, -1):
            left_end = left_index + width
            if left_end >= len(tokens):
                continue
            if not contiguous_phrase(left_index, width):
                continue
            phrase = normalized[left_index:left_end]
            maximum_right = min(len(tokens) - width + 1, left_end + 10)
            for right_index in range(left_end + 1, maximum_right):
                if normalized[right_index:right_index + width] != phrase:
                    continue
                if not contiguous_phrase(right_index, width):
                    continue
                separator = value[tokens[left_end - 1].end():tokens[right_index].start()]
                if re.search(r"[.!?\u061f\u061b;:\n]", separator):
                    continue
                span = (tokens[left_index].start(), tokens[right_index + width - 1].end())
                if span in seen:
                    continue
                seen.add(span)
                phrase_text = value[
                    tokens[left_index].start():tokens[left_end - 1].end()
                ]
                findings.append({
                    "phrase": phrase_text,
                    "normalized_phrase": " ".join(phrase),
                    "word_count": width,
                    "offset": span[0],
                    "end_offset": span[1],
                    "intervening_text": separator.strip(),
                    "context": value[max(0, span[0] - 60):min(len(value), span[1] + 60)],
                })
                break
            else:
                continue
            break
    return findings


def source_unjustified_repeated_governed_span_artifacts(
    source: str,
    translation: str,
) -> list[dict[str, Any]]:
    """Report repeated Persian complement frames beyond source allowance."""
    source_paragraphs = _paragraph_text_spans(source)
    target_paragraphs = _paragraph_text_spans(translation)
    if len(source_paragraphs) != len(target_paragraphs):
        source_paragraphs = [(0, len(source or ""), source or "")]
        target_paragraphs = [(0, len(translation or ""), translation or "")]
    findings: list[dict[str, Any]] = []
    for source_record, target_record in zip(
        source_paragraphs, target_paragraphs, strict=True
    ):
        allowance = len(_repeated_governed_spans(
            source_record[2], _SOURCE_LEXICAL_TOKEN_RE, _SOURCE_GOVERNORS
        ))
        target_start, _target_end, target_paragraph = target_record
        target_findings = _repeated_governed_spans(
            target_paragraph, _PERSIAN_LEXICAL_TOKEN_RE, _PERSIAN_GOVERNORS
        )
        for finding in target_findings[allowance:]:
            findings.append({
                **finding,
                "offset": target_start + int(finding["offset"]),
                "end_offset": target_start + int(finding["end_offset"]),
            })
    return findings


def newly_source_unjustified_repeated_governed_spans(
    source: str,
    previous: str,
    candidate: str,
) -> list[dict[str, Any]]:
    """Return only newly introduced unsupported governed-phrase repetition."""
    before = Counter(
        (int(item["word_count"]), str(item["normalized_phrase"]))
        for item in source_unjustified_repeated_governed_span_artifacts(
            source, previous
        )
    )
    excess = Counter(
        (int(item["word_count"]), str(item["normalized_phrase"]))
        for item in source_unjustified_repeated_governed_span_artifacts(
            source, candidate
        )
    ) - before
    introduced: list[dict[str, Any]] = []
    for item in source_unjustified_repeated_governed_span_artifacts(
        source, candidate
    ):
        key = (int(item["word_count"]), str(item["normalized_phrase"]))
        if excess[key] <= 0:
            continue
        introduced.append(item)
        excess[key] -= 1
    return introduced


def source_unjustified_repeated_adjacent_span_artifacts(
    source: str,
    translation: str,
) -> list[dict[str, Any]]:
    """Report adjacent target repetition not mirrored in its source paragraph."""
    source_paragraphs = _paragraph_text_spans(source)
    target_paragraphs = _paragraph_text_spans(translation)
    if len(source_paragraphs) != len(target_paragraphs):
        return repeated_persian_adjacent_span_artifacts(translation)
    findings: list[dict[str, Any]] = []
    for source_record, target_record in zip(
        source_paragraphs, target_paragraphs, strict=True
    ):
        source_allowance = Counter(
            int(item["word_count"])
            for item in _adjacent_repeated_spans(
                source_record[2], _SOURCE_LEXICAL_TOKEN_RE
            )
        )
        target_start, _target_end, target_paragraph = target_record
        for finding in repeated_persian_adjacent_span_artifacts(target_paragraph):
            width = int(finding["word_count"])
            if source_allowance[width] > 0:
                source_allowance[width] -= 1
                continue
            findings.append({
                **finding,
                "offset": target_start + int(finding["offset"]),
                "second_offset": target_start + int(finding["second_offset"]),
                "end_offset": target_start + int(finding["end_offset"]),
            })
    return findings


def newly_source_unjustified_repeated_adjacent_spans(
    source: str,
    previous: str,
    candidate: str,
) -> list[dict[str, Any]]:
    """Return only newly introduced adjacent repetition unsupported by source."""
    unjustified = Counter(
        (int(item["word_count"]), str(item["normalized_phrase"]))
        for item in source_unjustified_repeated_adjacent_span_artifacts(
            source, candidate
        )
    )
    findings: list[dict[str, Any]] = []
    for item in newly_repeated_adjacent_spans(previous, candidate):
        key = (int(item["word_count"]), str(item["normalized_phrase"]))
        if unjustified[key] <= 0:
            continue
        findings.append(item)
        unjustified[key] -= 1
    return findings


def repeated_persian_clause_artifacts(text: str) -> list[dict[str, Any]]:
    """Report exact multiword clauses duplicated around a coordinator.

    This is deliberately narrower than semantic repetition detection: both
    lexical spans must be byte-equivalent after case folding, contain at least
    two words, and sit immediately on either side of ``و`` or ``یا``.  The
    function reports evidence only; it never removes prose automatically.
    """
    value = text or ""
    tokens = list(_PERSIAN_LEXICAL_TOKEN_RE.finditer(value))
    findings: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for connector_index, connector in enumerate(tokens):
        if connector.group().casefold() not in _PERSIAN_COORDINATORS:
            continue
        for width in range(6, 1, -1):
            left_start = connector_index - width
            right_end = connector_index + 1 + width
            if left_start < 0 or right_end > len(tokens):
                continue
            left = tokens[left_start:connector_index]
            right = tokens[connector_index + 1:right_end]
            if [_repeat_token_key(item.group()) for item in left] != [
                _repeat_token_key(item.group()) for item in right
            ]:
                continue
            left_gap = value[left[-1].end():connector.start()]
            right_gap = value[connector.end():right[0].start()]
            if left_gap.strip(" \t\u200c،؛,:-‐‑‒–—"):
                continue
            if right_gap.strip(" \t\u200c،؛,:-‐‑‒–—"):
                continue
            span = (left[0].start(), right[-1].end())
            if span in seen:
                continue
            seen.add(span)
            phrase = value[left[0].start():left[-1].end()]
            findings.append({
                "phrase": phrase,
                "offset": span[0],
                "context": value[max(0, span[0] - 60):min(len(value), span[1] + 60)],
            })
            break
    return findings


def source_unjustified_repeated_clause_artifacts(
    source: str,
    translation: str,
) -> list[dict[str, Any]]:
    """Return target clause duplication unless the aligned source repeats too."""
    source_paragraphs = _paragraph_text_spans(source)
    target_paragraphs = _paragraph_text_spans(translation)
    if len(source_paragraphs) != len(target_paragraphs):
        return repeated_persian_clause_artifacts(translation)
    findings: list[dict[str, Any]] = []
    source_repeat_re = re.compile(
        r"\b(?P<phrase>[A-Za-z][A-Za-z'\u2019-]*(?:\s+"
        r"[A-Za-z][A-Za-z'\u2019-]*){1,5})\s+(?:and|or)\s+"
        r"(?P=phrase)\b",
        re.IGNORECASE,
    )
    for source_record, target_record in zip(
        source_paragraphs, target_paragraphs, strict=True
    ):
        source_paragraph = source_record[2]
        target_start, _target_end, target_paragraph = target_record
        if source_repeat_re.search(source_paragraph):
            continue
        for finding in repeated_persian_clause_artifacts(target_paragraph):
            findings.append({
                **finding,
                "offset": target_start + int(finding["offset"]),
            })
    return findings


def _paragraph_text_spans(text: str) -> list[tuple[int, int, str]]:
    """Return non-empty paragraph spans while preserving exact document offsets."""
    value = text or ""
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for separator in re.finditer(r"\n[ \t]*\n", value):
        raw = value[cursor:separator.start()]
        if raw.strip():
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw) - len(raw.rstrip())
            start = cursor + leading
            end = separator.start() - trailing
            spans.append((start, end, value[start:end]))
        cursor = separator.end()
    raw = value[cursor:]
    if raw.strip():
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        start = cursor + leading
        end = len(value) - trailing
        spans.append((start, end, value[start:end]))
    return spans


def source_unjustified_repeated_word_artifacts(
    source: str,
    translation: str,
) -> list[dict[str, Any]]:
    """Report target repetition only where its aligned source paragraph lacks it."""
    source_paragraphs = _paragraph_text_spans(source)
    target_paragraphs = _paragraph_text_spans(translation)
    if len(source_paragraphs) != len(target_paragraphs):
        return repeated_persian_word_artifacts(translation)
    findings: list[dict[str, Any]] = []
    for source_record, target_record in zip(
        source_paragraphs, target_paragraphs, strict=True
    ):
        source_paragraph = source_record[2]
        target_start, _target_end, target_paragraph = target_record
        source_words = re.findall(r"[A-Za-z\u00c0-\u024f]+", source_paragraph)
        source_repeats = any(
            left.casefold() == right.casefold()
            for left, right in zip(source_words, source_words[1:], strict=False)
        )
        if not source_repeats:
            for finding in repeated_persian_word_artifacts(target_paragraph):
                findings.append({
                    **finding,
                    "offset": target_start + int(finding["offset"]),
                })
    return findings


def foreign_script_artifacts(source: str, translation: str) -> list[dict[str, Any]]:
    """Report non-Latin foreign scripts introduced without source evidence."""
    source_text = source or ""
    target = translation or ""
    findings: list[dict[str, Any]] = []
    for script, pattern in _FOREIGN_SCRIPT_PATTERNS.items():
        source_values = set(pattern.findall(source_text))
        for match in pattern.finditer(target):
            value = match.group()
            if value in source_values:
                continue
            findings.append({
                "script": script,
                "text": value,
                "offset": match.start(),
                "context": target[max(0, match.start() - 60):match.end() + 60],
            })
    return findings


def markup_wrapper_artifacts(source: str, translation: str) -> list[dict[str, Any]]:
    """Report Markdown emphasis wrappers that were not authored in the source."""
    if "*" in (source or ""):
        return []
    target = translation or ""
    return [
        {
            "text": match.group(),
            "content": match.group("content").strip(),
            "offset": match.start(),
            "context": target[max(0, match.start() - 60):match.end() + 60],
        }
        for match in _MARKDOWN_EMPHASIS_RE.finditer(target)
    ]


def parenthesis_artifacts(source: str, translation: str) -> list[dict[str, Any]]:
    """Report unbalanced or newly nested ASCII parentheses in the target."""
    def scan(value: str) -> tuple[list[dict[str, Any]], int]:
        stack: list[int] = []
        findings: list[dict[str, Any]] = []
        maximum = 0
        for offset, char in enumerate(value or ""):
            if char == "(":
                stack.append(offset)
                maximum = max(maximum, len(stack))
            elif char == ")":
                if stack:
                    stack.pop()
                else:
                    findings.append({"type": "unmatched_close", "offset": offset})
        findings.extend(
            {"type": "unmatched_open", "offset": offset} for offset in stack
        )
        return findings, maximum

    findings, target_depth = scan(translation or "")
    _source_findings, source_depth = scan(source or "")
    if target_depth > max(1, source_depth):
        findings.append({
            "type": "new_nested_parentheses",
            "target_depth": target_depth,
            "source_depth": source_depth,
        })
    return findings


def detached_ezafe_artifacts(text: str) -> list[dict[str, Any]]:
    """Report a detached Persian ezafe suffix after a closing parenthesis."""
    target = text or ""
    return [
        {
            "text": match.group(),
            "offset": match.start(),
            "context": target[max(0, match.start() - 60):match.end() + 60],
        }
        for match in _DETACHED_EZAFE_RE.finditer(target)
    ]


def tatweel_separator_artifacts(text: str) -> list[dict[str, Any]]:
    """Report elongation glyph runs used as punctuation in Persian prose."""
    target = text or ""
    return [
        {
            "text": match.group(),
            "offset": match.start(),
            "context": target[max(0, match.start() - 60):match.end() + 60],
        }
        for match in _TATWEEL_SEPARATOR_RE.finditer(target)
    ]


def repair_source_grounded_language_artifacts(
    source: str,
    translation: str,
) -> tuple[str, dict[str, Any]]:
    """Apply only language edits whose full evidence is present in the source.

    The repair is intentionally narrow: it removes accidental adjacent lexical
    duplication when the source has no adjacent repetition, and parenthesizes an
    exact source-authored non-English Latin expression already copied into Persian
    prose. It never selects terminology or rewrites a proposition.
    """
    repaired = translation or ""
    edits: list[dict[str, Any]] = []

    # PDF superscripts often surface as an ordinary digit immediately after a
    # closing parenthesis. If the translator keeps that marker but also keeps a
    # now-orphaned source parenthesis, remove only the deterministically
    # unmatched close. Balanced author/citation parentheses remain untouched.
    required_note_markers = Counter(extract_note_markers(source))
    if required_note_markers:
        unmatched_closes = {
            int(item["offset"])
            for item in parenthesis_artifacts(source, repaired)
            if item.get("type") == "unmatched_close"
        }
        note_parenthesis_repairs: list[tuple[int, str]] = []
        for match in _PLAIN_SPACED_NOTE_RE.finditer(repaired.translate(_DIGIT_MAP)):
            marker = match.group("marker")
            close_offset = match.start("leader")
            if (
                match.group("leader") == ")"
                and required_note_markers[marker] > 0
                and close_offset in unmatched_closes
            ):
                note_parenthesis_repairs.append((close_offset, marker))
                required_note_markers[marker] -= 1
        for close_offset, marker in reversed(note_parenthesis_repairs):
            repaired = repaired[:close_offset] + repaired[close_offset + 1:]
            edits.append({
                "type": "orphaned_note_parenthesis",
                "before": ")",
                "after": "",
                "offset": close_offset,
                "note_marker": marker,
            })

    # Translate only exact structural framing copied from a source
    # parenthetical. Author names, years and the surrounding citation remain
    # byte-identical; this is language-level apparatus normalization, not a
    # semantic or terminological rewrite.
    structural_replacements: dict[str, str] = {}
    for parenthetical in _SOURCE_PAREN_RE.findall(source or ""):
        for match in _SOURCE_STRUCTURAL_REFERENCE_RE.finditer(parenthetical):
            before = match.group()
            label = _PERSIAN_STRUCTURAL_LABELS[match.group("label").casefold()]
            number = match.group("number").translate(_ASCII_TO_PERSIAN_DIGITS)
            prefix = "نگاه کنید به " if match.group("see") else ""
            structural_replacements.setdefault(
                before.casefold(), f"{prefix}{label} {number}"
            )
    structural_edits: list[tuple[int, int, str, str]] = []
    for target_parenthetical in _SOURCE_PAREN_RE.finditer(repaired):
        inner = target_parenthetical.group(1)
        inner_start = target_parenthetical.start(1)
        for source_phrase, replacement in structural_replacements.items():
            pattern = re.compile(re.escape(source_phrase), re.IGNORECASE)
            for match in pattern.finditer(inner):
                structural_edits.append((
                    inner_start + match.start(),
                    inner_start + match.end(),
                    match.group(),
                    replacement,
                ))
    for start, end, before, after in sorted(
        structural_edits, key=lambda item: item[0], reverse=True
    ):
        repaired = repaired[:start] + after + repaired[end:]
        edits.append({
            "type": "source_structural_reference",
            "before": before,
            "after": after,
            "offset": start,
        })

    for match in reversed(list(_TATWEEL_SEPARATOR_RE.finditer(repaired))):
        repaired = repaired[:match.start()] + "\u2014" + repaired[match.end():]
        edits.append({
            "type": "tatweel_separator",
            "before": match.group(),
            "after": "\u2014",
            "offset": match.start(),
        })

    for match in reversed(list(_DETACHED_EZAFE_RE.finditer(repaired))):
        before = match.group()
        suffix = before[-1]
        after = f"){suffix}"
        repaired = repaired[:match.start()] + after + repaired[match.end():]
        edits.append({
            "type": "detached_ezafe",
            "before": before,
            "after": after,
            "offset": match.start(),
        })

    source_folded = unicodedata.normalize("NFKC", source or "").casefold()
    for match in reversed(list(_PERSIAN_SUFFIX_AFTER_ORIGINAL_RE.finditer(repaired))):
        original = match.group("original")
        original_text = original[1:-1].strip()
        if unicodedata.normalize("NFKC", original_text).casefold() not in source_folded:
            continue
        joiner = match.group("join") or "\u200c"
        after = (
            f"{match.group('anchor')}{joiner}{match.group('suffix')} "
            f"{original}"
        )
        repaired = repaired[:match.start()] + after + repaired[match.end():]
        edits.append({
            "type": "parenthetical_persian_suffix",
            "before": match.group(),
            "after": after,
            "offset": match.start(),
        })

    for match in reversed(list(_NESTED_INLINE_ORIGINAL_RE.finditer(repaired))):
        original = match.group("original")
        original_text = original[1:-1].strip()
        if unicodedata.normalize("NFKC", original_text).casefold() not in source_folded:
            continue
        after = match.group().replace(original, f"[{original_text}]", 1)
        repaired = repaired[:match.start()] + after + repaired[match.end():]
        edits.append({
            "type": "nested_inline_original",
            "before": match.group(),
            "after": after,
            "offset": match.start(),
        })

    source_paragraphs = _paragraph_text_spans(source)
    target_paragraphs = _paragraph_text_spans(repaired)
    if len(source_paragraphs) == len(target_paragraphs):
        phrase_replacements: list[tuple[int, int, str, str, int]] = []
        for source_record, target_record in zip(
            source_paragraphs, target_paragraphs, strict=True
        ):
            source_paragraph = source_record[2]
            target_start, _target_end, target_paragraph = target_record
            for finding in source_unjustified_repeated_adjacent_span_artifacts(
                source_paragraph, target_paragraph
            ):
                start = int(finding["offset"])
                second = int(finding["second_offset"])
                end = int(finding["end_offset"])
                before = target_paragraph[start:end]
                after = target_paragraph[start:second].rstrip(" \t\u200c")
                if not after:
                    continue
                phrase_replacements.append((
                    target_start + start,
                    target_start + end,
                    before,
                    after,
                    int(finding.get("word_count", 0) or 0),
                ))
        for start, end, before, after, word_count in sorted(
            phrase_replacements, key=lambda item: item[0], reverse=True
        ):
            repaired = repaired[:start] + after + repaired[end:]
            edits.append({
                "type": (
                    "adjacent_duplicate"
                    if word_count == 1 else "adjacent_duplicate_phrase"
                ),
                "before": before,
                "after": after,
                "offset": start,
            })

        # Recompute paragraph offsets after phrase repairs before applying the
        # older single-word repair; the two edit classes must never use stale
        # offsets or overlap.
        target_paragraphs = _paragraph_text_spans(repaired)
        duplicate_replacements: list[tuple[int, int, str, str]] = []
        for source_record, target_record in zip(
            source_paragraphs, target_paragraphs, strict=True
        ):
            source_paragraph = source_record[2]
            target_start, _target_end, target_paragraph = target_record
            source_words = re.findall(
                r"[A-Za-z\u00c0-\u024f]+", source_paragraph
            )
            source_repeats = any(
                left.casefold() == right.casefold()
                for left, right in zip(source_words, source_words[1:], strict=False)
            )
            if not source_repeats:
                for finding in sorted(
                    repeated_persian_word_artifacts(target_paragraph),
                    key=lambda item: int(item["offset"]),
                    reverse=True,
                ):
                    start = int(finding["offset"])
                    word = str(finding["word"])
                    duplicate_match = re.match(
                        rf"{re.escape(word)}"
                        rf"(?P<separator>[ \t\u200c]+|"
                        rf"[ \t]*[-\u2010-\u2015][ \t]*)"
                        rf"{re.escape(word)}",
                        target_paragraph[start:],
                        re.IGNORECASE,
                    )
                    if not duplicate_match:
                        continue
                    before = duplicate_match.group(0)
                    absolute_start = target_start + start
                    duplicate_replacements.append((
                        absolute_start,
                        absolute_start + len(before),
                        before,
                        word,
                    ))
        for start, end, before, after in sorted(
            duplicate_replacements, key=lambda item: item[0], reverse=True
        ):
            repaired = repaired[:start] + after + repaired[end:]
            edits.append({
                "type": "adjacent_duplicate",
                "before": before,
                "after": after,
                "offset": start,
            })

    if "*" not in (source or ""):
        for match in reversed(list(_MARKDOWN_EMPHASIS_RE.finditer(repaired))):
            content = match.group("content").strip()
            before = match.group()
            repaired = repaired[:match.start()] + content + repaired[match.end():]
            edits.append({
                "type": "markdown_emphasis_wrapper",
                "before": before,
                "after": content,
                "offset": match.start(),
            })

    foreign_span_re = re.compile(
        r"(?<![A-Za-z\u00c0-\u024f])"
        r"(?P<phrase>[a-z\u00df-\u024f][A-Za-z\u00c0-\u024f'\u2019-]*"
        r"(?:[ \t]+[a-z\u00df-\u024f][A-Za-z\u00c0-\u024f'\u2019-]*){1,3})"
        r"(?![A-Za-z\u00c0-\u024f])"
    )
    replacements: list[tuple[int, int, str]] = []
    for match in foreign_span_re.finditer(repaired):
        phrase = match.group("phrase")
        if not (
            any(ord(char) > 127 for char in phrase)
            or "'" in phrase
            or "\u2019" in phrase
        ):
            continue
        if unicodedata.normalize("NFKC", phrase).casefold() not in source_folded:
            continue
        before = repaired[:match.start()].rstrip()
        after = repaired[match.end():].lstrip()
        if before.endswith("(") and after.startswith(")"):
            continue
        replacements.append((match.start(), match.end(), phrase))
    for start, end, phrase in reversed(replacements):
        repaired = repaired[:start] + f"({phrase})" + repaired[end:]
        edits.append({
            "type": "source_multilingual_parenthetical",
            "before": phrase,
            "after": f"({phrase})",
            "offset": start,
        })

    edits.sort(key=lambda item: int(item["offset"]))
    return repaired, {
        "repair_count": len(edits),
        "repairs": edits,
        "policy": "exact_source_evidence_only",
    }


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
    """Locate a grounded phrase despite whitespace and typography variants."""
    phrase = (phrase or "").strip()
    if not phrase:
        return []
    parts: list[str] = []
    in_whitespace = False
    for character in phrase:
        if character.isspace():
            if not in_whitespace:
                parts.append(r"\s+")
            in_whitespace = True
            continue
        in_whitespace = False
        if character in {"'", "\u2019"}:
            parts.append(r"['\u2019]")
        elif character in "-\u2010\u2011\u2012\u2013\u2014\u2015":
            parts.append(r"[-\u2010-\u2015]")
        else:
            parts.append(re.escape(character))
    pattern = re.compile("".join(parts), re.IGNORECASE)
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
        content = match.group(1).strip()
        tokens = list(_LATIN_PROSE_TOKEN_RE.finditer(content))
        if not tokens or len(tokens) > 8:
            continue
        if tokens[0].group().casefold() in _PARENTHETICAL_CONTEXT_LEADERS:
            continue
        if _SUSPECT_PDF_WORD_BREAK_RE.search(content):
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
        if not re.search(r"['\u2019]", token):
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
        or _CATALOG_NAME_LINE_RE.fullmatch(source or "")
        or _CATALOG_TITLE_LINE_RE.fullmatch(source or "")
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
        missing_note_counts = required_notes - available_note_markers(
            candidate, required_notes
        )
        previous_missing_note_counts = (
            required_notes - available_note_markers(previous, required_notes)
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

        # Short replacement-boundary corruption needs a separate relative
        # check. The long-span detector above intentionally starts at eight
        # words, while real refiner damage often repeats a one-word compound or
        # a two-to-six-word phrase immediately beside itself.
        blocking_adjacent = newly_source_unjustified_repeated_adjacent_spans(
            source, previous, candidate
        )
        if blocking_adjacent:
            add(
                "adjacent_repeated_span_introduced", "blocking",
                "The edit introduces an adjacent repeated phrase absent from "
                "the aligned source.",
                span_count=len(blocking_adjacent),
                samples=[
                    str(item["phrase"]) for item in blocking_adjacent[:3]
                ],
            )

        blocking_governed = newly_source_unjustified_repeated_governed_spans(
            source, previous, candidate
        )
        if blocking_governed:
            add(
                "governed_span_repetition_introduced", "blocking",
                "The edit duplicates a governed phrase inside one clause even "
                "though the aligned source provides no comparable repetition.",
                span_count=len(blocking_governed),
                samples=[
                    {
                        "phrase": str(item["phrase"]),
                        "intervening_text": str(item["intervening_text"]),
                    }
                    for item in blocking_governed[:3]
                ],
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
