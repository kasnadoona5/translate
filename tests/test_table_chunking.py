"""Tests for fix 10.1 - tables get their own chunk.

The chunkers grouped by token budget and section only. ``is_table`` was recorded
by the PDF parser but never used to bound a chunk, so Table 1.1 ("Six approaches
to the analysis of the state") arrived as one 85-paragraph prose chunk:
prompt_tokens=29335, truncated at 50,000, completed only on recovery at 77,820,
and the result lost every column boundary.

This change moves chunk boundaries, so it changes translations. The tests below
pin the boundary behaviour and confirm no paragraph is lost or reordered.
"""

from __future__ import annotations

import pytest

from tarjomeh.chunking.chunker import SemanticChunker
from tarjomeh.parsers.base import Chapter, Document, Paragraph, Section

BODY: dict = {}
TABLE = {"is_table": True, "structure_role": "table"}


def _document(paragraphs) -> Document:
    return Document(
        title="Book",
        chapters=[
            Chapter(
                title="Chapter",
                number=1,
                sections=[
                    Section(
                        title="",
                        level=1,
                        paragraphs=[
                            Paragraph(text=text, metadata=dict(metadata))
                            for text, metadata in paragraphs
                        ],
                    )
                ],
            )
        ],
    )


def _chunk_all(paragraphs, max_tokens: int = 100_000):
    return SemanticChunker(
        max_tokens=max_tokens,
        overlap_sentences=0,
        token_counter=lambda text: len(text.split()),
    ).chunk(_document(paragraphs))


def _roles(chunk) -> set[str]:
    return set(chunk.metadata.get("structural_roles", []))


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------

def test_a_table_never_shares_a_chunk_with_prose() -> None:
    paragraphs = (
        [("Prose one about the state.", BODY), ("Prose two about power.", BODY)]
        + [(f"Approach {i} | Focus {i}", TABLE) for i in range(8)]
        + [("Prose three, after the table.", BODY)]
    )
    chunks = _chunk_all(paragraphs)
    assert len(chunks) == 3
    assert [_roles(c) for c in chunks] == [{"body"}, {"table"}, {"body"}]


def test_no_chunk_mixes_table_and_prose_even_under_a_generous_budget() -> None:
    """The budget was never the constraint - it was the missing boundary."""
    paragraphs = [
        ("Prose.", BODY), ("Row a | Row b", TABLE), ("More prose.", BODY),
        ("Row c | Row d", TABLE),
    ]
    for chunk in _chunk_all(paragraphs):
        assert len(_roles(chunk)) == 1, f"mixed chunk: {_roles(chunk)}"


def test_consecutive_table_paragraphs_stay_together() -> None:
    """A table must not be shattered either - it should be exactly one chunk."""
    paragraphs = [(f"Cell {i} | Value {i}", TABLE) for i in range(20)]
    chunks = _chunk_all(paragraphs)
    assert len(chunks) == 1
    assert len(chunks[0].metadata["structural_roles"]) == 20


def test_a_table_larger_than_the_budget_still_splits_on_the_budget() -> None:
    """The token cap must keep working; the boundary is additive."""
    paragraphs = [(f"Cell {i} " + "word " * 30, TABLE) for i in range(10)]
    chunks = _chunk_all(paragraphs, max_tokens=60)
    assert len(chunks) > 1
    for chunk in chunks:
        assert _roles(chunk) == {"table"}


def test_a_table_chunk_is_never_a_style_exemplar() -> None:
    """Confirmed necessary: chunk 8's roles were 82 x table yet it came back
    structure_eligible=True."""
    chunks = _chunk_all([(f"Cell {i}", TABLE) for i in range(5)])
    assert chunks[0].metadata["style_eligible"] is False
    assert chunks[0].metadata["style_body_paragraphs"] == []


def test_prose_only_documents_are_unaffected() -> None:
    """No table => the boundary must never fire, so existing books chunk the
    same way they did before."""
    paragraphs = [(f"Prose paragraph {i} about the state.", BODY) for i in range(6)]
    chunks = _chunk_all(paragraphs)
    assert len(chunks) == 1
    assert _roles(chunks[0]) == {"body"}
    assert chunks[0].metadata["style_eligible"] is True


# ---------------------------------------------------------------------------
# Nothing may be lost
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("table_count", [1, 3, 8, 20])
def test_every_paragraph_survives_and_keeps_its_order(table_count) -> None:
    paragraphs = (
        [("Opening prose.", BODY)]
        + [(f"Row {i} | Value {i}", TABLE) for i in range(table_count)]
        + [("Closing prose.", BODY)]
    )
    chunks = _chunk_all(paragraphs)

    seen_indices = [
        index for chunk in chunks for index in chunk.metadata["paragraph_indices"]
    ]
    assert seen_indices == sorted(seen_indices), "source order must be preserved"
    assert seen_indices == list(range(len(paragraphs))), "no paragraph may be dropped"

    combined = "\n\n".join(chunk.text for chunk in chunks)
    for text, _metadata in paragraphs:
        assert text in combined


def test_paragraph_indices_and_roles_stay_the_same_length() -> None:
    paragraphs = [("Prose.", BODY), ("Row | Row", TABLE), ("Prose.", BODY)]
    for chunk in _chunk_all(paragraphs):
        assert len(chunk.metadata["paragraph_indices"]) == len(
            chunk.metadata["structural_roles"]
        )
