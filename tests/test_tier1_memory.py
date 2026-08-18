"""Tests for the Tier 1 memory-quality fixes: 10.2, 11.6, 5.2, 6.3.

These changes alter what enters memory, and memory feeds later translation
prompts, so the risk that matters is a *false positive*: discarding legitimate
terminology is worse than keeping some noise. The front-matter tests below are
weighted accordingly - most of them assert that real chapters are KEPT.
"""

from __future__ import annotations

import pytest

from tarjomeh.chunking.chunker import Chunk, FixedChunker, SemanticChunker
from tarjomeh.core.pipeline import _is_front_matter
from tarjomeh.parsers.base import Chapter, Document, Paragraph, Section


def _chunk(chapter_title, roles=None) -> Chunk:
    metadata = {} if roles is None else {"structural_roles": roles}
    return Chunk(
        index=0,
        text="some text",
        chapter_title=chapter_title,
        section_title="",
        metadata=metadata,
    )


def _document(chapters) -> Document:
    """Build a Document from ``[(title, number, [(text, metadata), ...])]``."""
    built = []
    for title, number, paragraphs in chapters:
        built.append(
            Chapter(
                title=title,
                number=number,
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
        )
    return Document(title="Book", chapters=built)


# ---------------------------------------------------------------------------
# 10.2 / 11.6 — front matter must not teach terminology
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "title",
    [
        "Contents", "CONTENTS", " contents: ", "Table of Contents",
        "Tables", "List of Tables", "Figures", "List of Figures",
        "Abbreviations", "List of Abbreviations",
        "Copyright", "Title Page", "Dedication", "Acknowledgements",
    ],
)
def test_front_matter_titles_are_excluded(title) -> None:
    """44 of 83 stored nouns came from here: street names, printers, ISBN."""
    assert _is_front_matter(_chunk(title))


@pytest.mark.parametrize(
    "title",
    [
        "The Concept of the State",
        "Introduction",          # a real chapter in most books
        "Conclusion",
        "Notes",                 # endnotes carry real terminology
        "Bibliography",
        "References",
        "Index",
        "Preface",
        "Chapter 1",
        "",                      # unknown structure
        None,
    ],
)
def test_real_chapters_are_kept(title) -> None:
    """The expensive failure mode: discarding legitimate terminology."""
    assert not _is_front_matter(_chunk(title))


@pytest.mark.parametrize(
    "roles,expected",
    [
        (["body", "body"], False),
        (["body", "table", "heading"], False),   # any body prose => keep
        (["table", "table"], True),              # no body prose at all
        (["heading"], True),
        (["footnote"], True),
        ([], False),                             # no evidence => keep
        (None, False),                           # no metadata => keep
    ],
)
def test_structural_evidence_decides_only_when_present(roles, expected) -> None:
    assert _is_front_matter(_chunk("Chapter 1", roles)) is expected


def test_a_body_chunk_inside_front_matter_titles_is_still_excluded() -> None:
    """Title evidence is stronger than role evidence: the copyright page is
    prose, but it is publisher prose."""
    assert _is_front_matter(_chunk("Copyright", ["body", "body"]))


# ---------------------------------------------------------------------------
# 5.2 — chapter identity is position, not title
# ---------------------------------------------------------------------------

DUPLICATE_TITLES = [
    ("Introduction", 1, [("Alpha paragraph about the state.", {})]),
    ("Introduction", 2, [("Beta paragraph about power.", {})]),
]


@pytest.mark.parametrize("chunker_cls", [SemanticChunker, FixedChunker])
def test_same_titled_chapters_get_distinct_positions(chunker_cls) -> None:
    """Two chapters sharing a title used to merge into one summary, and the
    end-of-chapter trigger misfired."""
    chunks = chunker_cls(max_tokens=1000, token_counter=lambda t: len(t.split())).chunk(
        _document(DUPLICATE_TITLES)
    )
    positions = [c.metadata.get("chapter_position") for c in chunks]
    titles = {c.chapter_title for c in chunks}

    assert titles == {"Introduction"}, "titles must genuinely collide for this test"
    assert len(set(positions)) == 2, f"positions must distinguish them, got {positions}"
    assert None not in positions


def test_chunk_chapter_position_helper_defaults_safely() -> None:
    from tarjomeh.core.pipeline import TranslationPipeline

    position = TranslationPipeline._chunk_chapter_position
    assert position(_chunk("X")) == 1
    chunk = _chunk("X")
    chunk.metadata["chapter_position"] = 7
    assert position(chunk) == 7


# ---------------------------------------------------------------------------
# 6.3 — FixedChunker carries structural metadata
# ---------------------------------------------------------------------------

def _fixed_chunks(paragraphs):
    return FixedChunker(max_tokens=1000, token_counter=lambda t: len(t.split())).chunk(
        _document([("Chapter 1", 1, paragraphs)])
    )


def test_fixed_chunker_emits_structural_policy() -> None:
    """It previously omitted these, so MemoryManager defaulted style_eligible
    to True and headings/tables became style exemplars."""
    chunks = _fixed_chunks([("Ordinary body prose about the state.", {})])
    assert len(chunks) == 1
    metadata = chunks[0].metadata
    assert metadata["structural_roles"] == ["body"]
    assert metadata["style_eligible"] is True
    assert metadata["style_body_paragraphs"] == [0]


@pytest.mark.parametrize(
    "metadata,role",
    [
        ({"structure_role": "heading"}, "heading"),
        ({"structure_role": "footnote"}, "footnote"),
        ({"structure_role": "table"}, "table"),
    ],
)
def test_fixed_chunker_marks_non_body_ineligible_for_style(metadata, role) -> None:
    chunks = _fixed_chunks([("Some non body text.", metadata)])
    assert chunks[0].metadata["structural_roles"] == [role]
    assert chunks[0].metadata["style_eligible"] is False
    assert chunks[0].metadata["style_body_paragraphs"] == []


@pytest.mark.parametrize("flag", ["is_footnote", "is_table", "heading_level"])
def test_fixed_chunker_respects_boolean_structure_flags(flag) -> None:
    """A paragraph flagged structurally must not become a style exemplar even
    when its structure_role still says body."""
    chunks = _fixed_chunks([("Text.", {"structure_role": "body", flag: 1})])
    assert chunks[0].metadata["style_eligible"] is False


def test_fixed_chunker_still_sets_the_metadata_it_always_did() -> None:
    """Regression guard: the new keys must not displace the existing ones."""
    chunks = _fixed_chunks([("Body prose.", {}), ("More prose.", {})])
    for chunk in chunks:
        assert "paragraph_indices" in chunk.metadata
        assert chunk.metadata["chapter_position"] == 1
        assert chunk.metadata["chapter_number"] == 1


def test_a_split_paragraph_keeps_its_role_on_every_piece() -> None:
    """FixedChunker may split one paragraph across chunks."""
    long_table = " ".join(f"cell{i}" for i in range(400))
    chunks = FixedChunker(
        max_tokens=50, token_counter=lambda t: len(t.split())
    ).chunk(_document([("Chapter 1", 1, [(long_table, {"structure_role": "table"})])]))

    assert len(chunks) > 1, "the paragraph must actually split for this test"
    for chunk in chunks:
        assert chunk.metadata["structural_roles"] == ["table"]
        assert chunk.metadata["style_eligible"] is False
