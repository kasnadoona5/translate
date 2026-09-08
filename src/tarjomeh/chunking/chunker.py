"""Semantic and fixed-size document chunkers for translation.

This module provides two chunking strategies:

* :class:`SemanticChunker` — respects chapter / section / paragraph
  boundaries and never splits mid-paragraph.
* :class:`FixedChunker` — simple token-count fallback that may split
  paragraphs when the semantic approach is not applicable.

Both produce :class:`Chunk` dataclass instances ready for the
translation pipeline.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Lightweight ``Document`` protocol so chunkers don't depend on the concrete
# parser module (which may not be written yet).
# ---------------------------------------------------------------------------

class _Paragraph(Protocol):
    """Minimal paragraph-like object expected inside a Section."""

    text: str
    metadata: dict[str, object]


class _Section(Protocol):
    """Minimal section-like object expected inside a Chapter."""

    title: str
    paragraphs: list[_Paragraph]


class _Chapter(Protocol):
    """Minimal chapter-like object expected inside a Document."""

    title: str
    number: int | None
    metadata: dict[str, object]
    sections: list[_Section]


class Document(Protocol):
    """Structural protocol for a parsed document.

    The concrete ``Document`` class lives in the parser package; chunkers
    depend only on this protocol so the two packages stay decoupled.

    Expected shape::

        Document
        └── chapters: list[Chapter]
            ├── title: str
            └── sections: list[Section]
                ├── title: str
                └── paragraphs: list[Paragraph]
                    └── text: str
    """

    chapters: list[_Chapter]


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single translation-ready text chunk.

    Attributes:
        index:          Zero-based ordinal position in the document.
        text:           Source text to translate.
        chapter_title:  Title of the enclosing chapter.
        section_title:  Title of the enclosing section (may be ``""``).
        metadata:       Arbitrary key/value bag.  The chunker stores
                        ``overlap_prefix`` here when overlap is enabled.
        token_count:    Number of tokens as counted by *token_counter*.
    """

    index: int
    text: str
    chapter_title: str
    section_title: str
    metadata: dict[str, object] = field(default_factory=dict)
    token_count: int = 0


# ---------------------------------------------------------------------------
# Sentence splitting helper
# ---------------------------------------------------------------------------

_SENTENCE_BOUNDARY = re.compile(
    r"""
    (?<=                 # lookbehind – end-of-sentence char
        [.!?…]           # ASCII or Unicode ellipsis
        |[۔؟!]           # Persian/Arabic sentence-enders
    )
    \s+                  # whitespace that separates sentences
    """,
    re.VERBOSE | re.UNICODE,
)


def _split_sentences(text: str) -> list[str]:
    """Split *text* into a list of sentences.

    Uses a simple regex that handles English and Persian punctuation.
    The result preserves original text exactly (no stripping).
    """
    parts = _SENTENCE_BOUNDARY.split(text.strip())
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------

class SemanticChunker:
    """Structure-aware chunker that respects chapter/section/paragraph boundaries.

    Parameters:
        max_tokens:        Maximum tokens per chunk.
        overlap_sentences: Number of trailing sentences from the previous
                           chunk to prepend as context (stored in
                           ``metadata['overlap_prefix']``).
        token_counter:     Callable ``(str) -> int`` — typically wrapping
                           :func:`tiktoken.encoding_for_model().encode`.
    """

    def __init__(
        self,
        max_tokens: int = 1000,
        overlap_sentences: int = 2,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self._count: Callable[[str], int] = token_counter or self._default_counter

    # -- public API ---------------------------------------------------------

    def chunk(self, document: Document) -> list[Chunk]:
        """Split *document* into a flat list of :class:`Chunk` objects.

        The algorithm walks chapters → sections → paragraphs and groups
        paragraphs into chunks without exceeding *max_tokens*.  It never
        splits a paragraph.  When a single paragraph exceeds *max_tokens*
        it is emitted as its own (over-size) chunk.
        """
        chunks: list[Chunk] = []
        previous_sentences: list[str] = []

        para_to_idx = {id(p): i for i, p in enumerate(document.all_paragraphs)}

        for chapter_position, chapter in enumerate(document.chapters, 1):
            chapter_metadata = getattr(chapter, "metadata", {}) or {}
            original_position = int(
                chapter_metadata.get(
                    "tarjomeh_chapter_position", chapter_position
                )
            )
            chapter_number = getattr(chapter, "number", None)
            for section in chapter.sections:
                buffer: list[str] = []
                buffer_tokens = 0
                buffer_para_indices: list[int] = []
                buffer_roles: list[str] = []
                buffer_style_eligible: list[bool] = []
                buffer_is_table: bool | None = None

                for para in section.paragraphs:
                    para_text = para.text.strip()
                    if not para_text:
                        continue
                    para_tokens = self._count(para_text)
                    para_metadata = getattr(para, "metadata", {}) or {}
                    para_is_table = bool(para_metadata.get("is_table"))
                    # Tables must not share a chunk with prose. The chunkers
                    # grouped by token budget and section only, so an 85-paragraph
                    # table was sent to the model as one prose chunk
                    # (prompt_tokens=29335, truncated at 50,000) and came back
                    # with every column boundary lost.
                    table_boundary = (
                        buffer_is_table is not None
                        and para_is_table != buffer_is_table
                    )

                    # Would adding this paragraph exceed the budget, or cross a
                    # table/prose boundary?
                    if buffer and (
                        (buffer_tokens + para_tokens) > self.max_tokens
                        or table_boundary
                    ):
                        # Flush current buffer as a chunk.
                        chunk = self._make_chunk(
                            index=len(chunks),
                            texts=buffer,
                            tokens=buffer_tokens,
                            chapter_title=chapter.title,
                            section_title=section.title,
                            chapter_position=original_position,
                            chapter_number=chapter_number,
                            chapter_metadata=chapter_metadata,
                            previous_sentences=previous_sentences,
                            para_indices=buffer_para_indices,
                            structural_roles=buffer_roles,
                            style_eligibility=buffer_style_eligible,
                        )
                        chunks.append(chunk)
                        previous_sentences = _split_sentences(chunk.text)[
                            -self.overlap_sentences :
                        ]
                        buffer = []
                        buffer_tokens = 0
                        buffer_para_indices = []
                        buffer_roles = []
                        buffer_style_eligible = []
                        buffer_is_table = None

                    if buffer_is_table is None:
                        buffer_is_table = para_is_table
                    buffer.append(para_text)
                    buffer_tokens += para_tokens
                    buffer_para_indices.append(para_to_idx[id(para)])
                    role = str(para_metadata.get("structure_role", "body"))
                    buffer_roles.append(role)
                    buffer_style_eligible.append(
                        role == "body"
                        and not bool(para_metadata.get("is_footnote"))
                        and not para_is_table
                        and not bool(para_metadata.get("heading_level"))
                    )

                # Flush remaining buffer for this section.
                if buffer:
                    chunk = self._make_chunk(
                        index=len(chunks),
                        texts=buffer,
                        tokens=buffer_tokens,
                        chapter_title=chapter.title,
                        section_title=section.title,
                        chapter_position=original_position,
                        chapter_number=chapter_number,
                        chapter_metadata=chapter_metadata,
                        previous_sentences=previous_sentences,
                        para_indices=buffer_para_indices,
                        structural_roles=buffer_roles,
                        style_eligibility=buffer_style_eligible,
                    )
                    chunks.append(chunk)
                    previous_sentences = _split_sentences(chunk.text)[
                        -self.overlap_sentences :
                    ]

        return chunks

    # -- internals ----------------------------------------------------------

    def _make_chunk(
        self,
        *,
        index: int,
        texts: list[str],
        tokens: int,
        chapter_title: str,
        section_title: str,
        chapter_position: int,
        chapter_number: int | None,
        chapter_metadata: dict[str, object],
        previous_sentences: list[str],
        para_indices: list[int],
        structural_roles: list[str],
        style_eligibility: list[bool],
    ) -> Chunk:
        combined = "\n\n".join(texts)
        source_spans: list[list[int]] = []
        source_hashes: list[str] = []
        cursor = 0
        for text in texts:
            source_spans.append([cursor, cursor + len(text)])
            source_hashes.append(
                hashlib.sha256(text.encode("utf-8")).hexdigest()
            )
            cursor += len(text) + 2
        metadata: dict[str, object] = {}
        if previous_sentences and self.overlap_sentences > 0:
            metadata["overlap_prefix"] = " ".join(previous_sentences)
        metadata["paragraph_indices"] = list(para_indices)
        metadata["paragraph_protocol_version"] = 1
        metadata["source_paragraph_spans"] = source_spans
        metadata["source_paragraph_hashes"] = source_hashes
        metadata["structural_roles"] = list(structural_roles)
        metadata["style_eligible"] = bool(
            style_eligibility and all(style_eligibility)
        )
        metadata["style_body_paragraphs"] = [
            index for index, eligible in enumerate(style_eligibility) if eligible
        ]
        metadata["chapter_position"] = chapter_position
        metadata["chapter_number"] = chapter_number
        for key in ("start_page", "end_page"):
            if key in chapter_metadata:
                metadata[key] = chapter_metadata[key]
        return Chunk(
            index=index,
            text=combined,
            chapter_title=chapter_title,
            section_title=section_title,
            metadata=metadata,
            token_count=tokens,
        )

    @staticmethod
    def _default_counter(text: str) -> int:
        """Rough word-count fallback when no *token_counter* is supplied."""
        return len(text.split())


# ---------------------------------------------------------------------------
# FixedChunker  (fallback)
# ---------------------------------------------------------------------------

class FixedChunker:
    """Token-count–based chunker that may split paragraphs.

    Used as a fallback when the document structure is missing or unreliable.
    It still tracks the current chapter context when available.

    Parameters:
        max_tokens:    Maximum tokens per chunk.
        token_counter: Callable ``(str) -> int``.
    """

    def __init__(
        self,
        max_tokens: int = 1000,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self._count: Callable[[str], int] = token_counter or (lambda t: len(t.split()))

    def chunk(self, document: Document) -> list[Chunk]:
        """Split *document* into fixed-size token chunks.

        Paragraphs may be split if they exceed *max_tokens*.  Chapter and
        section titles are carried forward so downstream code still has
        structural context.
        """
        chunks: list[Chunk] = []
        para_to_idx = {id(p): i for i, p in enumerate(document.all_paragraphs)}

        for chapter_position, chapter in enumerate(document.chapters, 1):
            chapter_metadata = getattr(chapter, "metadata", {}) or {}
            original_position = int(
                chapter_metadata.get(
                    "tarjomeh_chapter_position", chapter_position
                )
            )
            chapter_number = getattr(chapter, "number", None)
            for section in chapter.sections:
                for para in section.paragraphs:
                    para_text = para.text.strip()
                    if not para_text:
                        continue
                    sub_chunks = self._split_text(
                        text=para_text,
                        chapter_title=chapter.title,
                        section_title=section.title,
                        start_index=len(chunks),
                    )
                    # Structural policy must travel with the chunk. Without
                    # it MemoryManager defaults style_eligible to True, so
                    # headings, tables and footnotes became style exemplars and
                    # polluted the long-term style profile.
                    para_metadata = getattr(para, "metadata", {}) or {}
                    para_role = str(para_metadata.get("structure_role", "body"))
                    para_style_eligible = (
                        para_role == "body"
                        and not bool(para_metadata.get("is_footnote"))
                        and not bool(para_metadata.get("is_table"))
                        and not bool(para_metadata.get("heading_level"))
                    )
                    for c in sub_chunks:
                        c.metadata["paragraph_indices"] = [para_to_idx[id(para)]]
                        c.metadata["source_paragraph_spans"] = [[0, len(c.text)]]
                        c.metadata["source_paragraph_hashes"] = [
                            hashlib.sha256(c.text.encode("utf-8")).hexdigest()
                        ]
                        c.metadata["chapter_position"] = original_position
                        c.metadata["chapter_number"] = chapter_number
                        c.metadata["structural_roles"] = [para_role]
                        c.metadata["style_eligible"] = para_style_eligible
                        c.metadata["style_body_paragraphs"] = (
                            [0] if para_style_eligible else []
                        )
                        for key in ("start_page", "end_page"):
                            if key in chapter_metadata:
                                c.metadata[key] = chapter_metadata[key]
                    chunks.extend(sub_chunks)

        return chunks

    # -- internals ----------------------------------------------------------

    def _split_text(
        self,
        text: str,
        chapter_title: str,
        section_title: str,
        start_index: int,
    ) -> list[Chunk]:
        """Split *text* into chunks of at most *max_tokens* tokens.

        Tries to break on sentence boundaries first, falling back to word
        boundaries.
        """
        sentences = _split_sentences(text)
        result: list[Chunk] = []
        buffer: list[str] = []
        buffer_tokens = 0

        for sent in sentences:
            sent_tokens = self._count(sent)

            # Single sentence exceeds budget → split on words.
            if sent_tokens > self.max_tokens:
                if buffer:
                    result.append(
                        self._emit(buffer, buffer_tokens, chapter_title, section_title,
                                   start_index + len(result))
                    )
                    buffer, buffer_tokens = [], 0
                result.extend(
                    self._split_by_words(sent, chapter_title, section_title,
                                         start_index + len(result))
                )
                continue

            if buffer and (buffer_tokens + sent_tokens) > self.max_tokens:
                result.append(
                    self._emit(buffer, buffer_tokens, chapter_title, section_title,
                               start_index + len(result))
                )
                buffer, buffer_tokens = [], 0

            buffer.append(sent)
            buffer_tokens += sent_tokens

        if buffer:
            result.append(
                self._emit(buffer, buffer_tokens, chapter_title, section_title,
                           start_index + len(result))
            )

        return result

    def _split_by_words(
        self,
        text: str,
        chapter_title: str,
        section_title: str,
        start_index: int,
    ) -> list[Chunk]:
        """Last-resort word-level splitting for very long sentences."""
        words = text.split()
        result: list[Chunk] = []
        buf: list[str] = []
        buf_tokens = 0

        for word in words:
            word_tokens = self._count(word)
            if buf and (buf_tokens + word_tokens) > self.max_tokens:
                combined = " ".join(buf)
                result.append(Chunk(
                    index=start_index + len(result),
                    text=combined,
                    chapter_title=chapter_title,
                    section_title=section_title,
                    token_count=buf_tokens,
                ))
                buf, buf_tokens = [], 0
            buf.append(word)
            buf_tokens += word_tokens

        if buf:
            combined = " ".join(buf)
            result.append(Chunk(
                index=start_index + len(result),
                text=combined,
                chapter_title=chapter_title,
                section_title=section_title,
                token_count=buf_tokens,
            ))
        return result

    @staticmethod
    def _emit(
        sentences: list[str],
        tokens: int,
        chapter_title: str,
        section_title: str,
        index: int,
    ) -> Chunk:
        return Chunk(
            index=index,
            text=" ".join(sentences),
            chapter_title=chapter_title,
            section_title=section_title,
            token_count=tokens,
        )
