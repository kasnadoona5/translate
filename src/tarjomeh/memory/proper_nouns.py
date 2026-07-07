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


class ProperNouns:
    """Proper noun translation memory layer.

    Stores mappings from English proper nouns to their Persian equivalents,
    plus an *introduced* flag per noun (has it already appeared in a
    translated chunk?).
    """

    def __init__(self) -> None:
        self._nouns: dict[str, str] = {}
        self._introduced: set[str] = set()

    def add_noun(self, english: str, persian: str) -> None:
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
            if en in self._introduced:
                continue
            if re.search(rf"\b{re.escape(en)}\b", source_text):
                self._introduced.add(en)

    def is_introduced(self, english: str) -> bool:
        """Return True if the noun's first occurrence has already happened."""
        return english.strip() in self._introduced

    def get_context(self) -> str:
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
            if en in self._introduced:
                marker = "[introduced]"
            else:
                marker = f"[first occurrence pending — add ({en}) after the Persian]"
            lines.append(f"- {en} -> {fa}  {marker}")
        return "\n".join(lines)

    def serialize(self) -> dict[str, Any]:
        """Serialize the layer state for database checkpointing."""
        return {
            "nouns": dict(self._nouns),
            "introduced": sorted(self._introduced),
        }

    def deserialize(self, data: dict[str, Any]) -> None:
        """Restore the layer state from serialized data.

        Accepts both the current shape (``{"nouns": {...}, "introduced": [...]}``)
        and the legacy flat ``{english: persian}`` dict from older checkpoints.
        """
        if not data:
            self._nouns = {}
            self._introduced = set()
            return

        if "nouns" in data and isinstance(data.get("nouns"), dict):
            self._nouns = dict(data["nouns"])
            self._introduced = set(data.get("introduced", []))
        else:
            # Legacy checkpoint: flat mapping, no introduction tracking.
            self._nouns = {k: v for k, v in data.items() if isinstance(v, str)}
            self._introduced = set()

    def __len__(self) -> int:
        return len(self._nouns)

    def __repr__(self) -> str:
        return (
            f"ProperNouns(count={len(self._nouns)}, "
            f"introduced={len(self._introduced)})"
        )
