"""Short-term translation memory — sliding window of recent translations.

Layer 4 of the four-layer memory system.  Keeps the last *N* translated
paragraph pairs in a :class:`collections.deque` for immediate context.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class TranslationPair:
    """A matched source / translation segment."""

    source: str
    translation: str
    trust: str = "trusted"
    structural_role: str = "body"
    chapter_title: str = ""


class ShortTermMemory:
    """Sliding window of the most recent translated paragraphs.

    Parameters:
        window_size: Maximum number of pairs to keep (default ``4``).

    Usage::

        stm = ShortTermMemory(window_size=4)
        stm.add("The state apparatus …", "دستگاه دولتی …")
        context = stm.get_context()  # list of (source, translation)
    """

    def __init__(self, window_size: int = 4) -> None:
        if window_size < 1:
            raise ValueError("window_size must be ≥ 1")
        self.window_size = window_size
        self._window: deque[TranslationPair] = deque(maxlen=window_size)

    def add(
        self,
        source: str,
        translation: str,
        *,
        trust: str = "trusted",
        structural_role: str = "body",
        chapter_title: str = "",
    ) -> None:
        """Push a new source / translation pair into the window.

        The oldest pair is evicted automatically when the window is full.
        """
        self._window.append(TranslationPair(
            source=source,
            translation=translation,
            trust=trust,
            structural_role=structural_role,
            chapter_title=chapter_title,
        ))

    def get_context(self) -> list[tuple[str, str]]:
        """Return the current window as ``[(source, translation), …]``.

        The list is in chronological order (oldest first).
        """
        return [(p.source, p.translation) for p in self._window]

    def get_entries(self) -> list[TranslationPair]:
        """Return continuity entries with their trust and structural roles."""
        return list(self._window)

    def clear(self) -> None:
        """Remove all entries from the window."""
        self._window.clear()

    def serialize(self) -> list[dict[str, str]]:
        """Serialise the window for checkpoint persistence."""
        return [
            {
                "source": p.source,
                "translation": p.translation,
                "trust": p.trust,
                "structural_role": p.structural_role,
                "chapter_title": p.chapter_title,
            }
            for p in self._window
        ]

    @classmethod
    def deserialize(
        cls,
        data: list[dict[str, Any]],
        window_size: int = 4,
    ) -> ShortTermMemory:
        """Restore a :class:`ShortTermMemory` from serialised data."""
        stm = cls(window_size=window_size)
        for pair in data:
            if not isinstance(pair, dict):
                continue
            stm.add(
                str(pair.get("source", "")),
                str(pair.get("translation", "")),
                trust=str(pair.get("trust", "trusted")),
                structural_role=str(pair.get("structural_role", "body")),
                chapter_title=str(pair.get("chapter_title", "")),
            )
        return stm

    def __len__(self) -> int:
        return len(self._window)

    def __repr__(self) -> str:
        return f"ShortTermMemory(window_size={self.window_size}, filled={len(self._window)})"
