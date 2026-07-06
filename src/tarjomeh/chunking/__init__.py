"""Chunking sub-package — splits parsed documents into translation-ready chunks.

Provides :class:`SemanticChunker` (respects document structure) and
:class:`FixedChunker` (simple token-count fallback).
"""

from __future__ import annotations

from tarjomeh.chunking.chunker import Chunk, FixedChunker, SemanticChunker

__all__ = ["Chunk", "FixedChunker", "SemanticChunker"]
