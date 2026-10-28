"""Version-six boundaries preserve notes without changing stored parsers."""
from copy import deepcopy

import fitz
import pytest

from test_v226_pdf_footnotes import extract, page_shape
from tarjomeh.parsers.pdf_parser import PyMuPDFParser


@pytest.mark.parametrize("toc", [False, True])
@pytest.mark.parametrize("prefix", ["[1]", "(1)", "[\u06f1]", "(\u0661)", "[*]", "(\u2020)"])
def test_wrapped_markers_need_no_whitespace_in_version_six(toc, prefix):
    blocks = page_shape(prefix + "A note is retained.")
    before = deepcopy(blocks)
    old = extract(blocks, 5, toc)[-1]
    new = extract(blocks, 6, toc)[-1]
    assert not old.metadata.get("is_footnote", False)
    assert new.metadata["is_footnote"] is True
    assert new.metadata["footnote_classification"]["reason"] == "leading_source_note_marker"
    assert old.text == new.text == blocks[-1]["text"]
    assert blocks == before


@pytest.mark.parametrize("toc", [False, True])
def test_unlocated_superscript_retains_role_but_does_not_claim_position(toc):
    blocks = page_shape("A continuation with uncertain superscript position.")
    blocks[-1].update(has_superscript=True, superscript_markers=[])
    old = extract(blocks, 5, toc)[-1]
    new = extract(blocks, 6, toc)[-1]
    assert old.metadata["is_footnote"] and new.metadata["is_footnote"]
    assert old.metadata["footnote_classification"]["reason"] == "leading_source_note_marker"
    assert new.metadata["footnote_classification"]["reason"] == "unlocated_source_superscript"


@pytest.mark.parametrize("toc", [False, True])
@pytest.mark.parametrize("text", ["[1] A note.", "(1) A note.", "1A note.", "*A note."])
def test_previously_supported_note_markers_are_preserved(toc, text):
    for version in (5, 6):
        assert extract(page_shape(text), version, toc)[-1].metadata["is_footnote"]


@pytest.mark.parametrize("text", ["[1)Not a complete wrapper.", "(1]Not a wrapper.",
                                  "[1.2]An identifier.", "Body cites [1] internally."])
def test_marker_extension_does_not_whitelist_unmatched_or_internal_wrappers(text):
    for version in (5, 6):
        assert not extract(page_shape(text), version, True)[-1].metadata.get("is_footnote", False)


def test_new_default_and_old_explicit_versions():
    assert PyMuPDFParser().structure_version == 6
    for version in range(1, 6):
        assert PyMuPDFParser(structure_version=version).structure_version == version
    assert not extract(page_shape(), 6, True)[-1].metadata.get("is_footnote", False)


@pytest.mark.parametrize("toc", [False, True])
@pytest.mark.parametrize("marker", ["[1]A note.", "(1)A note.", "[1] A note."])
def test_real_pdf_both_routes_preserve_wrapped_note(tmp_path, toc, marker):
    path = tmp_path / "notes.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=420, height=648)
        page.insert_text((50, 77), "2", fontsize=22)
        page.insert_text((50, 123), "Methods", fontsize=18)
        page.insert_textbox(fitz.Rect(50, 277, 370, 470),
                            "A body sentence establishes the body font. " * 12, fontsize=10.5)
        page.insert_text((50, 542), marker, fontsize=10.5)
        if toc:
            doc.set_toc([[1, "Methods", 1]])
        doc.save(path)
    original = path.read_bytes()
    results = {}
    for version in (4, 5, 6):
        parsed = PyMuPDFParser(structure_version=version).parse(path)
        paragraph = next(p for p in parsed.all_paragraphs if marker in p.text)
        results[version] = paragraph.metadata.get("is_footnote", False)
        assert paragraph.text == marker
    assert results == {4: True, 5: marker == "[1] A note.", 6: True}
    assert path.read_bytes() == original
