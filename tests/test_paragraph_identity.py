"""Tests for fix 3.1 - the paragraph-identity gate.

Translation applies the marker protocol only when a chunk has MORE THAN ONE
paragraph. Assembly used to enforce strict identity whenever
``paragraph_protocol_version`` was set, which ``SemanticChunker`` sets on every
chunk. So a single-paragraph chunk was translated with no markers, no protocol
instruction and no repair pass, then judged as if it had all three - and a model
returning one paragraph as two blocks aborted the run after all LLM spend.

The two halves of the fix are tested separately: the gate must now agree with
what translation did, and a genuine mismatch must degrade rather than abort.
"""

from __future__ import annotations

import pytest

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.pipeline import (
    ParagraphIdentityError,
    _align_chunk_translation,
    _used_paragraph_protocol,
)


class _Para:
    """Minimal stand-in for a parsed paragraph."""

    def __init__(self, text: str, heading_level: int | None = None) -> None:
        self.text = text
        self.heading_level = heading_level
        self.metadata: dict = {}


def _chunk(text: str, *, version: int | None = 1, indices=None) -> Chunk:
    metadata: dict = {}
    if indices is not None:
        metadata["paragraph_indices"] = list(indices)
    if version is not None:
        metadata["paragraph_protocol_version"] = version
    return Chunk(
        index=0, text=text, chapter_title="c", section_title="", metadata=metadata
    )


# ---------------------------------------------------------------------------
# The gate must mirror _translate_single_chunk exactly
# ---------------------------------------------------------------------------

def test_single_paragraph_chunk_is_not_strict() -> None:
    """The bug: this chunk gets no markers at translation time."""
    assert _used_paragraph_protocol(_chunk("Only one paragraph.")) is False


@pytest.mark.parametrize(
    "text",
    [
        "First paragraph.\n\nSecond paragraph.",
        "First.\n \nSecond.",          # blank line with whitespace still splits
        "A.\n\n\n\nB.",
    ],
)
def test_multi_paragraph_chunks_stay_strict(text) -> None:
    """Protection for genuine multi-paragraph chunks must not be weakened."""
    assert _used_paragraph_protocol(_chunk(text)) is True


@pytest.mark.parametrize("version", [None, 0])
def test_a_chunk_without_the_protocol_version_is_not_strict(version) -> None:
    assert _used_paragraph_protocol(_chunk("A.\n\nB.", version=version)) is False


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_empty_text_is_not_strict(text) -> None:
    assert _used_paragraph_protocol(_chunk(text)) is False


def test_gate_matches_encode_paragraphs_marker_count() -> None:
    """_used_paragraph_protocol must agree with the translation-time condition,
    which is derived from encode_paragraphs()."""
    from tarjomeh.core.paragraph_protocol import encode_paragraphs

    for text in ("one only", "a\n\nb", "a\n\nb\n\nc", "", "  \n\n  "):
        chunk = _chunk(text)
        _, markers = encode_paragraphs(chunk.text)
        assert _used_paragraph_protocol(chunk) is bool(len(markers) > 1), text


# ---------------------------------------------------------------------------
# Behaviour at assembly
# ---------------------------------------------------------------------------

def test_single_paragraph_off_count_translation_no_longer_raises() -> None:
    """A model returning one source paragraph as two blocks used to kill the run."""
    chunk = _chunk("Only one paragraph.", indices=[0])
    aligned = _align_chunk_translation(
        original_paragraphs=[_Para("Only one paragraph.")],
        para_indices=[0],
        tgt_paras=["block one", "block two"],
        chunk_translation="block one\n\nblock two",
        strict_paragraph_identity=_used_paragraph_protocol(chunk),
    )
    assert len(aligned) == 1
    assert aligned[0][0] == 0
    # Nothing may be silently dropped: both blocks must survive.
    assert "block one" in aligned[0][1]
    assert "block two" in aligned[0][1]


def test_multi_paragraph_mismatch_still_raises_the_specific_error() -> None:
    chunk = _chunk("A\n\nB", indices=[0, 1])
    with pytest.raises(ParagraphIdentityError):
        _align_chunk_translation(
            original_paragraphs=[_Para("A"), _Para("B")],
            para_indices=[0, 1],
            tgt_paras=["only one block"],
            chunk_translation="only one block",
            strict_paragraph_identity=_used_paragraph_protocol(chunk),
        )


def test_paragraph_identity_error_is_a_value_error() -> None:
    """Existing `except ValueError` handlers must keep working unchanged."""
    assert issubclass(ParagraphIdentityError, ValueError)


def test_a_matching_multi_paragraph_translation_aligns_one_to_one() -> None:
    chunk = _chunk("A\n\nB", indices=[0, 1])
    aligned = _align_chunk_translation(
        original_paragraphs=[_Para("A"), _Para("B")],
        para_indices=[0, 1],
        tgt_paras=["alpha", "beta"],
        chunk_translation="alpha\n\nbeta",
        strict_paragraph_identity=_used_paragraph_protocol(chunk),
    )
    assert aligned == [(0, "alpha"), (1, "beta")]


def test_non_strict_fallback_preserves_every_translated_block() -> None:
    """The degradation path must not lose text - that is the whole point."""
    aligned = _align_chunk_translation(
        original_paragraphs=[_Para("A"), _Para("B")],
        para_indices=[0, 1],
        tgt_paras=["one", "two", "three"],
        chunk_translation="one\n\ntwo\n\nthree",
        strict_paragraph_identity=False,
    )
    combined = " ".join(text for _pid, text in aligned)
    for block in ("one", "two", "three"):
        assert block in combined
