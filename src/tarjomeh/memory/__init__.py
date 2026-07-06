"""Memory sub-package — four-layer translation memory system.

Layers:
    1. Proper-noun records  (:mod:`proper_nouns`)
    2. Bilingual summary    (:mod:`bilingual_summary`)
    3. Long-term TF-IDF     (:mod:`long_term`)
    4. Short-term window    (:mod:`short_term`)

Orchestrated by :class:`MemoryManager`.
"""

from __future__ import annotations

from tarjomeh.memory.manager import MemoryContext, MemoryManager

__all__ = ["MemoryContext", "MemoryManager"]
