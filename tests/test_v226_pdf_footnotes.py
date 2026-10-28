"""Versioned false-footnote protection, never a text or export waiver."""
import copy
from types import SimpleNamespace

import pytest

from tarjomeh.parsers.pdf_parser import PyMuPDFParser, _median_font_size


def block(text, size, y, **extra):
    return dict(text=text, font_size=size, bbox=(50.0, y, 370.0, y + 35),
                page_height=648.0, orientation="horizontal", is_table=False,
                source_fragment_id=f"fragment-{y}", **extra)


def page_shape(text="Without engaging directly with Nietzsche's observation, the writer offers another response."):
    return [block("2", 22.0, 77), block("A Conceptual Chapter", 18.0, 123),
            block("This ordinary body paragraph establishes the usual printed font. " * 12, 10.5, 277),
            block(text, 10.5, 542)]


def extract(blocks, version, toc):
    class Document(list):
        @property
        def page_count(self):
            return len(self)

    doc = Document([SimpleNamespace(rect=SimpleNamespace(height=648.0))])
    parser = PyMuPDFParser(structure_version=version)
    if toc:
        return parser._extract_pages(doc, 0, 1, [copy.deepcopy(blocks)])
    chapters = parser._parse_without_toc(doc, [copy.deepcopy(blocks)])
    return [p for c in chapters for p in c.all_paragraphs]


@pytest.mark.parametrize("toc", [False, True])
def test_reproduced_heading_inflation_does_not_make_body_a_footnote(toc):
    blocks = page_shape()
    assert _median_font_size(blocks) == 14.25
    before = copy.deepcopy(blocks)
    old = extract(blocks, 4, toc)[-1]
    new = extract(blocks, 5, toc)[-1]
    assert old.metadata["is_footnote"] is True
    assert not new.metadata.get("is_footnote", False)
    assert new.text == old.text == blocks[-1]["text"]
    assert new.metadata["structure_role"] == "body"
    assert new.metadata["footnote_classification"]["reason"] == "matches_dominant_body_font"
    assert blocks == before


@pytest.mark.parametrize("toc", [False, True])
def test_smaller_font_genuine_note_keeps_legacy_behavior(toc):
    blocks = page_shape()
    blocks[-1]["font_size"] = 8.0
    blocks[-1]["text"] = "A smaller-font explanatory note."
    for version in (4, 5):
        assert extract(blocks, version, toc)[-1].metadata["is_footnote"] is True


@pytest.mark.parametrize("prefix", ["1 ", "12. ", "۲ ", "٣) ", "* ", "\u2020 ", "\u2021 ", "\u00b9 ",
                                   "[1] ", "(۲) ", "[٣] "])
def test_same_font_leading_note_marker_is_not_reclassified(prefix):
    blocks = page_shape(prefix + "This is an explanatory note.")
    assert extract(blocks, 5, True)[-1].metadata["is_footnote"] is True


def test_source_confirmed_leading_superscript_does_not_require_a_space():
    blocks = page_shape("1An explanatory note.")
    blocks[-1]["superscript_markers"] = [dict(text="1", context_before="", context_after="An explanatory")]
    assert extract(blocks, 5, True)[-1].metadata["is_footnote"] is True


@pytest.mark.parametrize('prefix', ['*', '\u2020', '\u00b9', '1'])
def test_undelimited_possible_note_marker_keeps_legacy_role(prefix):
    assert extract(page_shape(prefix + 'An explanatory note.'), 5, True)[-1].metadata['is_footnote'] is True


def test_unlocated_superscript_evidence_is_not_proof_of_unmarked_body():
    blocks = page_shape('aAn explanatory note.')
    blocks[-1]['has_superscript'] = True
    assert extract(blocks, 5, True)[-1].metadata['is_footnote'] is True


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_old_versions_keep_old_roles_and_have_no_new_metadata(version):
    paragraph = extract(page_shape(), version, True)[-1]
    assert paragraph.metadata["is_footnote"] is True
    assert "footnote_classification" not in paragraph.metadata


@pytest.mark.parametrize("exclusion", ["is_table", "heading", "contents", "rotated", "bottom", "empty"])
def test_ineligible_reference_cannot_overrule_legacy_note(exclusion):
    blocks = page_shape()
    body = blocks[2]
    if exclusion == "is_table":
        body["is_table"] = True
    elif exclusion == "heading":
        body["structure_role"] = "heading"
    elif exclusion == "contents":
        body["structure_role"] = "contents_entry"
    elif exclusion == "rotated":
        body["orientation"] = "rotated"
    elif exclusion == "bottom":
        body["bbox"] = (50, 530, 370, 570)
    else:
        body["text"] = ""
    assert extract(blocks, 5, True)[-1].metadata["is_footnote"] is True


def test_exact_vote_tie_is_not_body_proof():
    blocks = page_shape()
    blocks[2]["text"] = "body"
    blocks.insert(3, block("note", 9.0, 350))
    blocks.insert(1, block("Another Heading", 18.0, 110))
    assert extract(blocks, 4, True)[-1].metadata["is_footnote"] is True
    assert extract(blocks, 5, True)[-1].metadata["is_footnote"] is True


def test_new_default_is_five_and_heading_decisions_stay_unchanged():
    assert PyMuPDFParser().structure_version == 5
    old = extract(page_shape(), 4, True)
    new = extract(page_shape(), 5, True)
    assert [(p.text, p.metadata.get("heading_level")) for p in old] == [
        (p.text, p.metadata.get("heading_level")) for p in new]


def test_runtime_probe_is_pure_and_rejects_a_disabled_safeguard(monkeypatch):
    from pathlib import Path
    import tarjomeh.parsers.pdf_parser as parser_module
    from tarjomeh.runtime import _v226_behavior_probes

    def forbidden(*args, **kwargs):
        raise AssertionError("Pure geometry probe attempted file I/O")

    monkeypatch.setattr(Path, "open", forbidden)
    assert _v226_behavior_probes() == {"versioned_body_footnote_classification": True}
    original = parser_module._classify_footnote
    monkeypatch.setattr(parser_module, "_classify_footnote",
                        lambda block, median, height, evidence=None: original(block, median, height))
    assert _v226_behavior_probes() == {"versioned_body_footnote_classification": False}


def test_empty_or_nonfinite_body_reference_cannot_supply_proof():
    from tarjomeh.parsers.pdf_parser import _page_body_font_evidence

    assert _page_body_font_evidence([], 12.0, 648)['font_size'] is None
    for size in (float('nan'), float('inf'), 0.0, -1.0):
        reference = _page_body_font_evidence([block('Body text.', size, 200)], 14.25, 648)
        assert reference['font_size'] is None
