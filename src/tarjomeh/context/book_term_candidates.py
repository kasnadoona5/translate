"""Source-only term-family candidates for human book-scoped review."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from tarjomeh.parsers.base import Document

_COORDINATED_TERMS = re.compile(
    r"\b[A-Za-z][A-Za-z'-]+(?:,\s+[A-Za-z][A-Za-z'-]+){1,4}"
    r",?\s+and\s+[A-Za-z][A-Za-z'-]+\b",
    re.IGNORECASE,
)


def collect_book_term_candidates(document: Document) -> list[dict[str, Any]]:
    """Rank repeated source families; no guessed Persian enters the prompt."""
    counts: Counter[str] = Counter()
    evidence: dict[str, str] = {}
    surfaces: dict[str, str] = {}
    for paragraph in document.all_paragraphs:
        if (
            not paragraph.is_translatable
            or paragraph.is_footnote
            or paragraph.heading_level is not None
            or str(paragraph.metadata.get("structure_role", "body")) != "body"
        ):
            continue
        for match in _COORDINATED_TERMS.finditer(paragraph.text):
            source = " ".join(match.group().split())
            key = source.casefold()
            counts[key] += 1
            surfaces.setdefault(key, source)
            evidence.setdefault(key, paragraph.text.strip()[:500])
    return [
        {
            "source": surfaces[key],
            "target": "",
            "status": "candidate",
            "origin": "source_coordination",
            "confidence": "review_only",
            "source_count": count,
            "source_evidence": evidence[key],
            "reason": "Repeated coordinated source terms; choose one book-scoped rendering.",
        }
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= 2
    ][:30]
