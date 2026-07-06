"""Layer 1 of the four-layer memory system: Proper Nouns & Transliterations.

Stores established transliterations and translations of person names, places,
institutions, and publications to ensure consistency across the book.
"""

from __future__ import annotations

from typing import Any


class ProperNouns:
    """Proper noun translation memory layer.

    Stores mappings from English proper nouns to their Persian equivalents.
    """

    def __init__(self) -> None:
        self._nouns: dict[str, str] = {}

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
            self._nouns[en_key] = fa_val

    def get_context(self) -> str:
        """Return a formatted string representing the proper nouns dictionary.

        Returns
        -------
        str
            A hyphenated list of 'English -> Persian' proper nouns, or empty string.
        """
        if not self._nouns:
            return ""
        lines = []
        for en, fa in sorted(self._nouns.items()):
            lines.append(f"- {en} -> {fa}")
        return "\n".join(lines)

    def serialize(self) -> dict[str, str]:
        """Serialize the layer state for database checkpointing."""
        return dict(self._nouns)

    def deserialize(self, data: dict[str, str]) -> None:
        """Restore the layer state from serialized data."""
        self._nouns = dict(data)

    def __len__(self) -> int:
        return len(self._nouns)

    def __repr__(self) -> str:
        return f"ProperNouns(count={len(self._nouns)})"
