"""BabelDOC-compatible glossary management.

Loads, saves, and queries a CSV-based glossary whose rows contain:

    source, target, tgt_lng, context, domain

Term matching is case-insensitive with word-boundary awareness so that
partial matches (e.g., "state" inside "statement") are avoided.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GlossaryEntry:
    """A single glossary row.

    Attributes:
        source:  English term.
        target:  Persian translation.
        tgt_lng: Target-language ISO code (default ``"fa"``).
        context: Optional context hint for disambiguation.
        domain:  Optional domain tag (e.g. ``"political theory"``).
    """

    source: str
    target: str
    tgt_lng: str = "fa"
    context: str = ""
    domain: str = ""


# ---------------------------------------------------------------------------
# GlossaryManager
# ---------------------------------------------------------------------------

class GlossaryManager:
    """CSV-backed glossary with fast regex-based lookup.

    The CSV format is BabelDOC-compatible::

        source,target,tgt_lng,context,domain
        hegemony,هژمونی,fa,Gramsci,political theory
    """

    def __init__(self) -> None:
        self._entries: list[GlossaryEntry] = []
        # Compiled regex cache:  source_lower -> (pattern, entry)
        self._patterns: dict[str, tuple[re.Pattern[str], GlossaryEntry]] = {}

    # -- I/O ----------------------------------------------------------------

    def load(self, csv_path: str | Path) -> None:
        """Load glossary entries from a BabelDOC-compatible CSV file.

        The first row **must** be a header.  Missing optional columns
        (``context``, ``domain``) are tolerated.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Glossary file not found: {path}")

        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                entry = GlossaryEntry(
                    source=row.get("source", "").strip(),
                    target=row.get("target", "").strip(),
                    tgt_lng=row.get("tgt_lng", "fa").strip(),
                    context=row.get("context", "").strip(),
                    domain=row.get("domain", "").strip(),
                )
                if entry.source and entry.target:
                    self._add_entry(entry)

    def save(self, csv_path: str | Path) -> None:
        """Persist all entries to a BabelDOC-compatible CSV file."""
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["source", "target", "tgt_lng", "context", "domain"],
            )
            writer.writeheader()
            for entry in self._entries:
                writer.writerow({
                    "source": entry.source,
                    "target": entry.target,
                    "tgt_lng": entry.tgt_lng,
                    "context": entry.context,
                    "domain": entry.domain,
                })

    # -- mutation ------------------------------------------------------------

    def add_term(
        self,
        source: str,
        target: str,
        tgt_lng: str = "fa",
        context: str = "",
        domain: str = "",
    ) -> None:
        """Add or update a glossary term.

        If the *source* term already exists (case-insensitive), the entry
        is updated in place.
        """
        key = source.strip().lower()
        for existing in self._entries:
            if existing.source.lower() == key:
                existing.target = target
                existing.tgt_lng = tgt_lng
                existing.context = context or existing.context
                existing.domain = domain or existing.domain
                self._rebuild_pattern(existing)
                return
        entry = GlossaryEntry(
            source=source.strip(),
            target=target.strip(),
            tgt_lng=tgt_lng,
            context=context.strip(),
            domain=domain.strip(),
        )
        self._add_entry(entry)

    def merge_auto_extracted(self, extracted_terms: dict[str, str]) -> None:
        """Merge LLM-extracted terms into the glossary.

        Existing (user-supplied) terms take priority and are never
        overwritten by auto-extracted ones.

        Args:
            extracted_terms: ``{english_term: persian_translation}``
        """
        existing_keys = {e.source.lower() for e in self._entries}
        for source, target in extracted_terms.items():
            if source.strip().lower() not in existing_keys:
                self.add_term(source=source, target=target)

    # -- query --------------------------------------------------------------

    def find_terms(self, text: str) -> list[GlossaryEntry]:
        """Return all glossary entries whose source term appears in *text*.

        Matching is **case-insensitive** with **word boundaries** (``\\b``)
        so ``"state"`` will not match ``"statement"``.
        """
        matches: list[GlossaryEntry] = []
        for _key, (pattern, entry) in self._patterns.items():
            if pattern.search(text):
                matches.append(entry)
        return matches

    def format_for_prompt(self, terms: list[GlossaryEntry] | None = None) -> str:
        """Format glossary entries for injection into a translation prompt.

        Returns a Markdown-style table that the LLM can reference during
        translation.

        Args:
            terms: Subset of entries to format.  If *None*, formats all.
        """
        entries = terms if terms is not None else self._entries
        if not entries:
            return ""

        lines = [
            "## Glossary — Use these exact translations",
            "",
            "| English | Persian (فارسی) | Domain |",
            "|---------|----------------|--------|",
        ]
        for e in entries:
            domain_str = e.domain or "—"
            lines.append(f"| {e.source} | {e.target} | {domain_str} |")
        lines.append("")
        return "\n".join(lines)

    # -- properties ----------------------------------------------------------

    @property
    def entries(self) -> list[GlossaryEntry]:
        """Read-only access to the current entry list."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, source: str) -> bool:
        key = source.strip().lower()
        return any(e.source.lower() == key for e in self._entries)

    # -- internals -----------------------------------------------------------

    def _add_entry(self, entry: GlossaryEntry) -> None:
        self._entries.append(entry)
        self._rebuild_pattern(entry)

    def _rebuild_pattern(self, entry: GlossaryEntry) -> None:
        """Compile a case-insensitive word-boundary regex for *entry*."""
        escaped = re.escape(entry.source)
        pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
        self._patterns[entry.source.lower()] = (pattern, entry)
