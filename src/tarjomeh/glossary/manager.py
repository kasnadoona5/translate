"""BabelDOC-compatible glossary management.

Loads, saves, and queries CSV-based glossaries whose rows contain:

    source, target, tgt_lng, context, domain

Phase-3 glossaries may add optional columns after the BabelDOC-compatible
columns:

    sense, author, is_auto

Term matching is case-insensitive with word-boundary awareness so that
partial matches (e.g., "state" inside "statement") are avoided.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# BabelDOC-compatible base columns. Optional columns are additive so old CSVs
# continue to work unchanged.
BASE_FIELDNAMES = ["source", "target", "tgt_lng", "context", "domain"]
OPTIONAL_FIELDNAMES = ["sense", "author", "is_auto"]
FIELDNAMES = BASE_FIELDNAMES + OPTIONAL_FIELDNAMES

_TOKEN_RE = re.compile(r"[A-Za-z0-9\u0600-\u06FF']+", re.UNICODE)


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
        sense:   Optional named sense (e.g. ``"Marx/economic"``).
        author:  Optional author/school namespace (e.g. ``"Bourdieu"``).
        glossary: Internal source glossary name/path stem.
    """

    source: str
    target: str
    tgt_lng: str = "fa"
    context: str = ""
    domain: str = ""
    sense: str = ""
    author: str = ""
    glossary: str = ""
    is_auto: bool = False


# ---------------------------------------------------------------------------
# GlossaryManager
# ---------------------------------------------------------------------------

class GlossaryManager:
    """CSV-backed glossary collection with fast regex-based lookup.

    The CSV format is BabelDOC-compatible::

        source,target,tgt_lng,context,domain
        hegemony,هژمونی,fa,Gramsci,political theory

    Phase-3 namespaced glossaries can append optional ``sense`` and ``author``
    columns. Multiple entries may share the same English source term; when that
    happens, :meth:`find_terms` selects the best matching sense from the chunk
    text, chapter/section context, and domain.
    """

    def __init__(self) -> None:
        self._entries: list[GlossaryEntry] = []
        # Compiled regex cache; a source term can have several namespaced senses.
        self._patterns: list[tuple[re.Pattern[str], GlossaryEntry]] = []

    # -- I/O ----------------------------------------------------------------

    def load(self, csv_path: str | Path, glossary_name: str | None = None) -> None:
        """Load glossary entries from a BabelDOC-compatible CSV file.

        The first row **must** be a header.  Missing optional columns
        (``context``, ``domain``, ``sense``, ``author``) are tolerated.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Glossary file not found: {path}")

        source_name = glossary_name or path.stem
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                entry = GlossaryEntry(
                    source=row.get("source", "").strip(),
                    target=row.get("target", "").strip(),
                    tgt_lng=row.get("tgt_lng", "fa").strip(),
                    context=row.get("context", "").strip(),
                    domain=row.get("domain", "").strip(),
                    sense=row.get("sense", "").strip(),
                    author=row.get("author", "").strip(),
                    glossary=source_name,
                    is_auto=str(row.get("is_auto", "")).strip().lower() in ("1", "true", "yes"),
                )
                if entry.source and entry.target:
                    self._add_entry(entry)

    def load_many(
        self,
        csv_paths: Iterable[str | Path],
        *,
        ignore_missing: bool = False,
    ) -> None:
        """Load several glossary CSV files into one manager.

        The loading order is preserved and is used as a deterministic fallback
        when duplicate terms have no contextual evidence favouring one sense.
        """
        for csv_path in csv_paths:
            path = Path(csv_path)
            if not path.exists():
                if ignore_missing:
                    continue
                raise FileNotFoundError(f"Glossary file not found: {path}")
            self.load(path)

    def save(self, csv_path: str | Path) -> None:
        """Persist all entries to a BabelDOC-compatible CSV file."""
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=_fieldnames_for_entries(self._entries),
                extrasaction="ignore",
            )
            writer.writeheader()
            for entry in self._entries:
                writer.writerow(_entry_to_row(entry))

    def save_auto_extracted(self, csv_path: str | Path) -> None:
        """Persist only auto-extracted (discovered) terms to a CSV file."""
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        auto_entries = [entry for entry in self._entries if entry.is_auto]
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=_fieldnames_for_entries(auto_entries),
                extrasaction="ignore",
            )
            writer.writeheader()
            for entry in auto_entries:
                writer.writerow(_entry_to_row(entry))

    # -- mutation ------------------------------------------------------------

    def add_term(
        self,
        source: str,
        target: str,
        tgt_lng: str = "fa",
        context: str = "",
        domain: str = "",
        sense: str = "",
        author: str = "",
        glossary: str = "",
        is_auto: bool = False,
    ) -> None:
        """Add or update a glossary term.

        Existing simple behaviour is preserved: a single un-namespaced term is
        updated in place by source. Namespaced terms (``sense``/``author``/
        ``domain``) can coexist with other senses of the same source term.
        """
        source_clean = source.strip()
        target_clean = target.strip()
        key = _entry_key(source_clean, sense, author, domain)

        for existing in self._entries:
            if _entry_key(existing.source, existing.sense, existing.author, existing.domain) == key:
                existing.target = target_clean
                existing.tgt_lng = tgt_lng
                existing.context = context or existing.context
                existing.domain = domain or existing.domain
                existing.sense = sense or existing.sense
                existing.author = author or existing.author
                existing.glossary = glossary or existing.glossary
                existing.is_auto = existing.is_auto or is_auto
                self._rebuild_patterns()
                return

        same_source = [e for e in self._entries if e.source.lower() == source_clean.lower()]
        if not sense and not author and same_source:
            simple = [
                e for e in same_source
                if not e.sense and not e.author and (not domain or e.domain.lower() == domain.lower())
            ]
            if len(simple) == 1:
                existing = simple[0]
                existing.target = target_clean
                existing.tgt_lng = tgt_lng
                existing.context = context or existing.context
                existing.domain = domain or existing.domain
                existing.glossary = glossary or existing.glossary
                existing.is_auto = existing.is_auto or is_auto
                self._rebuild_patterns()
                return

        entry = GlossaryEntry(
            source=source_clean,
            target=target_clean,
            tgt_lng=tgt_lng,
            context=context.strip(),
            domain=domain.strip(),
            sense=sense.strip(),
            author=author.strip(),
            glossary=glossary.strip(),
            is_auto=is_auto,
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

    def find_terms(
        self,
        text: str,
        *,
        context: str = "",
        domain: str = "",
    ) -> list[GlossaryEntry]:
        """Return all glossary entries whose source term appears in *text*.

        Matching is **case-insensitive** with **word boundaries** (``\\b``)
        so ``"state"`` will not match ``"statement"``.

        If several entries share the same English source term, the manager
        chooses the active sense using author/sense/domain/context evidence.
        """
        grouped: dict[str, list[GlossaryEntry]] = {}
        source_order: list[str] = []
        for pattern, entry in self._patterns:
            if pattern.search(text):
                key = entry.source.lower()
                if key not in grouped:
                    grouped[key] = []
                    source_order.append(key)
                grouped[key].append(entry)

        selected: list[GlossaryEntry] = []
        for key in source_order:
            entries = grouped[key]
            if len(entries) == 1:
                selected.append(entries[0])
            else:
                selected.append(self._select_entry(entries, text=text, context=context, domain=domain))
        return selected

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

        include_namespace = any(e.sense or e.author for e in entries)
        if include_namespace:
            lines = [
                "## Glossary — Use these exact translations",
                "",
                "| English | Persian (فارسی) | Sense | Author/School | Context | Domain |",
                "|---------|----------------|-------|---------------|---------|--------|",
            ]
        else:
            lines = [
                "## Glossary — Use these exact translations",
                "",
                "| English | Persian (فارسی) | Context | Domain |",
                "|---------|----------------|---------|--------|",
            ]
        for e in entries:
            # Context carries the author-specific sense (e.g. "Gramsci's concept
            # of cultural dominance") — essential for disambiguating terms like
            # "capital" (Marx) vs "capital" (Bourdieu).
            context_str = _table_cell(e.context or "—")
            domain_str = _table_cell(e.domain or "—")
            if include_namespace:
                lines.append(
                    f"| {_table_cell(e.source)} | {_table_cell(e.target)} | "
                    f"{_table_cell(e.sense or '—')} | {_table_cell(e.author or '—')} | "
                    f"{context_str} | {domain_str} |"
                )
            else:
                lines.append(
                    f"| {_table_cell(e.source)} | {_table_cell(e.target)} | "
                    f"{context_str} | {domain_str} |"
                )
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
        self._patterns.append((pattern, entry))

    def _rebuild_patterns(self) -> None:
        self._patterns = []
        for entry in self._entries:
            self._rebuild_pattern(entry)

    def _select_entry(
        self,
        entries: list[GlossaryEntry],
        *,
        text: str,
        context: str,
        domain: str,
    ) -> GlossaryEntry:
        """Select the most contextually appropriate entry from same-source senses."""
        scored = [
            (self._context_score(entry, text=text, context=context, domain=domain), idx, entry)
            for idx, entry in enumerate(entries)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[0][2]

    @staticmethod
    def _context_score(
        entry: GlossaryEntry,
        *,
        text: str,
        context: str,
        domain: str,
    ) -> int:
        haystack = " ".join([text, context, domain]).casefold()
        haystack_tokens = set(_tokens(haystack))
        score = 0

        author_tokens = set(_tokens(entry.author))
        if author_tokens and author_tokens <= haystack_tokens:
            score += 30
        elif author_tokens and author_tokens & haystack_tokens:
            score += 12

        sense_tokens = set(_tokens(entry.sense))
        if sense_tokens and sense_tokens <= haystack_tokens:
            score += 20
        elif sense_tokens:
            score += min(len(sense_tokens & haystack_tokens) * 4, 16)

        domain_tokens = set(_tokens(entry.domain))
        if domain_tokens and domain_tokens <= haystack_tokens:
            score += 10
        elif domain_tokens:
            score += min(len(domain_tokens & haystack_tokens) * 2, 8)

        context_tokens = set(_tokens(entry.context))
        if context_tokens:
            score += min(len(context_tokens & haystack_tokens), 8)

        if entry.author and entry.author.casefold() in haystack:
            score += 8
        if entry.sense and entry.sense.casefold() in haystack:
            score += 6
        if entry.domain and entry.domain.casefold() in haystack:
            score += 4

        return score


def _entry_key(source: str, sense: str, author: str, domain: str) -> tuple[str, str, str, str]:
    return (
        source.strip().lower(),
        sense.strip().lower(),
        author.strip().lower(),
        domain.strip().lower(),
    )


def _entry_to_row(entry: GlossaryEntry) -> dict[str, str]:
    return {
        "source": entry.source,
        "target": entry.target,
        "tgt_lng": entry.tgt_lng,
        "context": entry.context,
        "domain": entry.domain,
        "sense": entry.sense,
        "author": entry.author,
        "is_auto": "true" if entry.is_auto else "",
    }


def _fieldnames_for_entries(entries: Iterable[GlossaryEntry]) -> list[str]:
    entries_list = list(entries)
    if any(entry.sense or entry.author or entry.is_auto for entry in entries_list):
        return FIELDNAMES
    return BASE_FIELDNAMES


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def _table_cell(value: str) -> str:
    return value.replace("|", "\\|").strip()
