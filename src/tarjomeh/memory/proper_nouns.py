"""Layer 1 of the four-layer memory system: Proper Nouns & Transliterations.

Stores established transliterations and translations of person names, places,
institutions, and publications to ensure consistency across the book — and
tracks which names have already been INTRODUCED (appeared in translated text),
so the "English form in parentheses on first occurrence" convention is applied
exactly once per book instead of once per chunk.
"""

from __future__ import annotations

import re
from typing import Any


INLINE_ORIGINAL_CATEGORIES = frozenset({
    "proper_noun", "person", "place", "institution", "organization",
    "publication", "product", "theory", "approved_term",
})
_CATEGORY_ALIASES = {
    "organisation": "organization",
    "book": "publication",
    "article": "publication",
    "journal": "publication",
    "work": "publication",
    "named_theory": "theory",
}
_PERSIAN_LETTER_RE = re.compile(r"[\u0621-\u063a\u0641-\u064a\u066e-\u06d3\u06fa-\u06ff]")
_UNUSABLE_TARGET_RE = re.compile(
    r"\b(?:n/?a|none|unknown|not available|no persian|no established|"
    r"not supported|insufficient evidence|untranslated)\b",
    re.IGNORECASE,
)


def is_usable_memory_mapping(english: str, persian: str) -> bool:
    """Reject malformed/placeholder auto mappings from prompt-time memory."""
    source = " ".join((english or "").split()).strip()
    target = " ".join((persian or "").split()).strip()
    if not source or not target or "_" in target:
        return False
    if source.casefold() == target.casefold():
        return False
    if _UNUSABLE_TARGET_RE.search(target):
        return False
    if not _PERSIAN_LETTER_RE.search(target):
        return False
    return True


def _normalise_category(category: str) -> str:
    value = (category or "proper_noun").strip().lower().replace("-", "_").replace(" ", "_")
    return _CATEGORY_ALIASES.get(value, value)


class ProperNouns:
    """Proper noun translation memory layer.

    Stores mappings from English proper nouns to their Persian equivalents,
    plus an *introduced* flag per noun (has it already appeared in a
    translated chunk?).
    """

    def __init__(self) -> None:
        self._nouns: dict[str, str] = {}
        self._categories: dict[str, str] = {}
        self._introduced: set[str] = set()

    def add_noun(self, english: str, persian: str, category: str = "proper_noun") -> None:
        """Add or update a proper noun mapping.

        Parameters
        ----------
        english:
            English proper noun (key).
        persian:
            Established Persian translation/transliteration.
        """
        en_key = english.strip()
        fa_val = persian.strip()
        if en_key and fa_val:
            # Never overwrite an established rendering with a new suggestion —
            # consistency across the book beats a "better" late transliteration.
            self._nouns.setdefault(en_key, fa_val)
            new_category = _normalise_category(category)
            current_category = self._categories.get(en_key)
            if (
                current_category is None
                or new_category == "approved_term"
                or (
                    current_category not in INLINE_ORIGINAL_CATEGORIES
                    and new_category in INLINE_ORIGINAL_CATEGORIES
                )
            ):
                self._categories[en_key] = new_category

    def mark_introduced(self, english: str) -> None:
        """Mark a noun as already introduced (parenthetical already shown)."""
        en_key = english.strip()
        if en_key in self._nouns:
            self._introduced.add(en_key)

    def mark_seen_in_text(self, source_text: str) -> None:
        """Mark every known noun that appears in *source_text* as introduced.

        Called after a chunk is translated: any known name occurring in that
        chunk's source has now had its first appearance, so later chunks must
        not repeat the English parenthetical.
        """
        for en in self._nouns:
            if en in self._introduced or not self.is_inline_eligible(en):
                continue
            if re.search(rf"\b{re.escape(en)}\b", source_text, re.IGNORECASE):
                self._introduced.add(en)

    def is_introduced(self, english: str) -> bool:
        """Return True if the noun's first occurrence has already happened."""
        return english.strip() in self._introduced

    def category_for(self, english: str) -> str:
        """Return the semantic category retained from extraction."""
        return self._categories.get(english.strip(), "proper_noun")

    def is_inline_eligible(self, english: str) -> bool:
        """Return whether a noun may carry a first-occurrence English original."""
        return self.category_for(english) in INLINE_ORIGINAL_CATEGORIES

    def inline_eligible_nouns(self) -> dict[str, str]:
        """Return only deterministic inline-original candidates."""
        return {
            source: target for source, target in self._nouns.items()
            if self.is_inline_eligible(source)
        }

    def pending_inline_originals(self, source_text: str) -> dict[str, str]:
        """Return eligible, not-yet-introduced originals present in one chunk."""
        return {
            source: target for source, target in self._nouns.items()
            if self.is_inline_eligible(source)
            and source not in self._introduced
            and re.search(rf"\b{re.escape(source)}\b", source_text, re.IGNORECASE)
        }

    def get_context(self, include_inline_originals: bool = True) -> str:
        """Return a formatted string representing the proper nouns dictionary.

        Each entry carries an introduction marker the translation prompt is
        instructed to honour:
        - ``[introduced]`` — do NOT repeat the English parenthetical
        - ``[first occurrence pending]`` — add ``(English)`` after the Persian
          on first use.

        Returns
        -------
        str
            A hyphenated list of 'English -> Persian' proper nouns, or empty string.
        """
        if not self._nouns:
            return ""
        lines = []
        for en, fa in sorted(self._nouns.items()):
            if not is_usable_memory_mapping(en, fa):
                continue
            category = self.category_for(en)
            if not self.is_inline_eligible(en):
                marker = (
                    f"[advisory terminology only; category={category}; never add "
                    "an English parenthetical; glossary and source context override it]"
                )
            elif en in self._introduced:
                marker = "[introduced]"
            elif not include_inline_originals:
                marker = (
                    "[first occurrence pending — use the established Persian "
                    "rendering; the exporter adds the English-original note]"
                )
            else:
                marker = f"[first occurrence pending — add ({en}) after the Persian]"
            lines.append(f"- {en} -> {fa}  {marker}")
        return "\n".join(lines)

    def serialize(self) -> dict[str, Any]:
        """Serialize the layer state for database checkpointing."""
        return {
            "nouns": dict(self._nouns),
            "categories": dict(self._categories),
            "introduced": sorted(self._introduced),
        }

    def deserialize(self, data: dict[str, Any]) -> None:
        """Restore the layer state from serialized data.

        Accepts both the current shape (``{"nouns": {...}, "introduced": [...]}``)
        and the legacy flat ``{english: persian}`` dict from older checkpoints.
        """
        if not data:
            self._nouns = {}
            self._categories = {}
            self._introduced = set()
            return

        if "nouns" in data and isinstance(data.get("nouns"), dict):
            self._nouns = dict(data["nouns"])
            stored_categories = data.get("categories", {})
            self._categories = {
                source: _normalise_category(str(stored_categories.get(source, "proper_noun")))
                for source in self._nouns
            }
            self._introduced = set(data.get("introduced", []))
        else:
            # Legacy checkpoint: flat mapping, no introduction tracking.
            self._nouns = {k: v for k, v in data.items() if isinstance(v, str)}
            self._categories = {source: "proper_noun" for source in self._nouns}
            self._introduced = set()

    def __len__(self) -> int:
        return len(self._nouns)

    def __repr__(self) -> str:
        return (
            f"ProperNouns(count={len(self._nouns)}, "
            f"introduced={len(self._introduced)})"
        )
