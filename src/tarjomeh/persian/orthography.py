"""Conservative, auditable Persian orthography corrections.

The rules in this module intentionally cover only forms whose correction is
unambiguous. Semantic wording, citations, Latin originals, and unknown words
are left untouched.
"""

from __future__ import annotations

import re
from typing import Any


_PERSIAN_LETTERS = (
    r"\u0621-\u063a\u0641-\u064a\u066e-\u06d3\u06fa-\u06ff"
)


_SAFE_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "standard_ba_in_hal_spacing",
        re.compile(
            rf"(?<![{_PERSIAN_LETTERS}])"
            rf"با(?:\u200c|\s*)این(?:\u200c|\s*)حال"
            rf"(?![{_PERSIAN_LETTERS}])"
        ),
        "با این حال",
    ),
    (
        "verb_prefix_zwnj",
        re.compile(
            rf"(?<![{_PERSIAN_LETTERS}\u200c])(ن?می)\s+(?=[{_PERSIAN_LETTERS}])"
        ),
        r"\1‌",
    ),
    (
        "separated_plural_suffix_zwnj",
        re.compile(
            rf"(?<=[{_PERSIAN_LETTERS}])\s+ها"
            rf"(?=(?:ی(?:ی|م|ت|ش|مان|تان|شان)?)?(?![{_PERSIAN_LETTERS}]))"
        ),
        "‌ها",
    ),
    (
        "separated_final_heh_indefinite_zwnj",
        re.compile(
            rf"ه\u200c?[ \t]+ای"
            rf"(?=(?:\u200c?تر(?:ین)?)?(?![{_PERSIAN_LETTERS}]))"
        ),
        "ه‌ای",
    ),
    (
        "separated_comparative_suffix_zwnj",
        re.compile(
            rf"(?<=[{_PERSIAN_LETTERS}])\s+(تر(?:ین)?)(?![{_PERSIAN_LETTERS}])"
        ),
        r"‌\1",
    ),
    (
        "comparative_after_indefinite_zwnj",
        re.compile(r"(?<=ه\u200cای)تر(?=(?:ین)?(?:\b|$))"),
        "‌تر",
    ),
    (
        "modern_ye_after_waw",
        re.compile(rf"(?<=[{_PERSIAN_LETTERS}])وئی"),
        "ویی",
    ),
    (
        "malformed_plural_heh_ezafe",
        re.compile(r"(?<!ه)(?<=[\u0600-\u06ff])\u200cه\u200cای(?=\b|$)"),
        "\u200cهای",
    ),
)


_ADVISORY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "separated_derivational_suffix",
        re.compile(rf"(?<=[{_PERSIAN_LETTERS}])\s+مندی(?![{_PERSIAN_LETTERS}])"),
    ),
)

_COMPOUND_AXIS_RE = re.compile(
    rf"(?P<base>[{_PERSIAN_LETTERS}]{{2,}})(?:[ \t]+|(?<!\u200c))محور"
    rf"(?![{_PERSIAN_LETTERS}])"
)
_NON_COMPOUND_AXIS_BASES = frozenset({
    "از", "این", "آن", "با", "بر", "به", "در", "روی", "حول", "کنار",
    "هر", "یک", "دو", "سه", "چهار", "پنج", "خود",
})


def _axis_compound_matches(text: str) -> list[re.Match[str]]:
    """Return suffix uses while excluding ordinary prepositional phrases."""
    return [
        match for match in _COMPOUND_AXIS_RE.finditer(text or "")
        if match.group("base") not in _NON_COMPOUND_AXIS_BASES
    ]


def apply_safe_persian_orthography(
    text: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Return *text* with only high-confidence orthographic fixes applied."""
    if not text:
        return text, []

    current = text
    edits: list[dict[str, Any]] = []
    for rule_id, pattern, replacement in _SAFE_RULES:
        matches = list(pattern.finditer(current))
        if not matches:
            continue
        before_values = [match.group(0) for match in matches]
        current, count = pattern.subn(replacement, current)
        edits.append({
            "rule_id": rule_id,
            "count": count,
            "before": before_values[:10],
            "after": replacement,
        })
    axis_matches = _axis_compound_matches(current)
    if axis_matches:
        before_values = [match.group(0) for match in axis_matches]
        for match in reversed(axis_matches):
            current = (
                current[:match.start()]
                + f"{match.group('base')}\u200cمحور"
                + current[match.end():]
            )
        edits.append({
            "rule_id": "compound_axis_zwnj",
            "count": len(axis_matches),
            "before": before_values[:10],
            "after": "<base>\u200cمحور",
        })
    return current, edits


def orthography_issue_count(text: str) -> int:
    """Count safe corrections plus conservative review-only suspicions."""
    safe = sum(len(pattern.findall(text or "")) for _, pattern, _ in _SAFE_RULES)
    advisory = sum(
        len(pattern.findall(text or "")) for _, pattern in _ADVISORY_PATTERNS
    )
    compounds = len(_axis_compound_matches(text or ""))
    return safe + advisory + compounds
