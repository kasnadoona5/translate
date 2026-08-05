"""Conservative, auditable Persian orthography corrections.

The rules in this module intentionally cover only forms whose correction is
unambiguous. Semantic wording, citations, Latin originals, and unknown words
are left untouched.
"""

from __future__ import annotations

import re
from typing import Any


_SAFE_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "capitalism_zwnj",
        re.compile(r"(?<![\w\u200c])سرمایه\s*داری(?![\w\u200c])"),
        "سرمایه\u200cداری",
    ),
    (
        "algorithm_plural_zwnj",
        re.compile(r"(?<![\w\u200c])الگوریتم\s*ها(?![\w\u200c])"),
        "الگوریتم\u200cها",
    ),
    (
        "formation_compound_zwnj",
        re.compile(r"(?<![\w\u200c])صورت\s+بندی(?=$|[\s،؛:,.!?؟])"),
        "صورت\u200cبندی",
    ),
    (
        "heh_ezafe_zwnj",
        re.compile(r"های(?=تر(?:\b|$)|(?:\b|$))"),
        "ه\u200cای",
    ),
    (
        "comparative_after_ezafe_zwnj",
        re.compile(r"ه\u200cایتر(?=\b|$)"),
        "ه\u200cای\u200cتر",
    ),
)


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
    return current, edits


def orthography_issue_count(text: str) -> int:
    """Count safely recognizable noncanonical forms without changing text."""
    return sum(len(pattern.findall(text or "")) for _, pattern, _ in _SAFE_RULES)
