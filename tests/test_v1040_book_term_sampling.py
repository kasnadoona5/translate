from tarjomeh.context.book_term_candidates import (
    body_term_paragraphs,
    book_term_extraction_sample,
    collect_book_term_candidates,
)
from tarjomeh.parsers.base import Chapter, Document, Paragraph, Section
from tarjomeh.parsers.pdf_parser import _extract_page_blocks


def _chapter(title, *texts):
    return Chapter(
        title=title,
        sections=[Section(title="", level=2,
                          paragraphs=[Paragraph(text) for text in texts])],
    )


def test_body_sampling_excludes_back_matter_but_not_prose_citations():
    document = Document(title="Test", chapters=[
        _chapter("Preface", "Preface says neoliberalism neoliberalism."),
        _chapter("The argument",
                 "Jessop (1982) argues that neoliberalism changes institutions. " * 3,
                 "Neoliberalism is discussed again in this substantive chapter. " * 3),
        _chapter("References", "Smith, J. (1982). Neoliberalism in context."),
        _chapter("Index", "neoliberalism 12, 29, 32"),
    ])
    body = [paragraph.text for paragraph in body_term_paragraphs(document)]
    assert len(body) == 2
    assert body[0].startswith("Jessop (1982) argues")
    sample = book_term_extraction_sample(document)
    assert "Jessop (1982) argues" in sample
    assert "Smith, J. (1982)" not in sample
    assert "neoliberalism 12" not in sample
    assert "neoliberalism" in [item["source"].casefold()
                             for item in collect_book_term_candidates(document)]


def test_reference_entry_filter_does_not_drop_argument_sentence():
    document = Document(title="Test", chapters=[
        _chapter("Chapter 1",
                 "Jessop, B. (1982). A bibliographic entry.",
                 "Jessop (1982) argues that the state changes.")
    ])
    assert [p.text for p in body_term_paragraphs(document)] == [
        "Jessop (1982) argues that the state changes."
    ]


def test_repeated_phrase_and_attested_plural_family_are_review_only():
    document = Document(title="Test", chapters=[
        _chapter("Argument",
                 "State power shapes the field. Imagined communities arise.",
                 "State power changes. Imagined communities persist.",
                 "State power matters. An imagined community is discussed."),
        _chapter("References", "Imagined communities 1, 2, 3."),
    ])
    candidates = collect_book_term_candidates(document)
    by_source = {item["source"].casefold(): item for item in candidates}
    assert "state power" in by_source
    assert "imagined community" in by_source
    assert "Imagined communities" in by_source["imagined community"]["source_variants"]
    assert all(item["target"] == "" and item["confidence"] == "review_only"
               for item in candidates)


def test_abbreviations_page_is_candidate_evidence_not_body_sample():
    document = Document(title="Test", chapters=[
        _chapter("Abbreviations", "SRA: strategic relational approach"),
        _chapter("Chapter 1", "The approach concerns state power."),
    ])
    candidates = collect_book_term_candidates(document)
    assert any(item["source"] == "SRA" and item["origin"] == "abbreviations_page"
               for item in candidates)
    sample = book_term_extraction_sample(document)
    assert "SRA | SRA: strategic relational approach" in sample
    assert "SRA: strategic relational approach\nSRA: strategic relational approach" not in sample


def test_aligned_pdf_italic_is_only_a_review_candidate():
    paragraph = Paragraph("The modus operandi is examined.", metadata={
        "italic_source_spans": ["modus operandi", "unmatched expression"],
    })
    document = Document(title="Test", chapters=[Chapter(
        title="Chapter 1", sections=[Section(title="", level=2, paragraphs=[paragraph])],
    )])
    candidates = collect_book_term_candidates(document)
    assert any(item["source"] == "modus operandi"
               and item["origin"] == "aligned_pdf_italic"
               and item["target"] == "" for item in candidates)
    assert not any(item["source"] == "unmatched expression" for item in candidates)


def test_pdf_italic_span_requires_unique_exact_block_text():
    class Page:
        rect = type("Rect", (), {"height": 800, "width": 600})()

        def get_text(self, *_args, **_kwargs):
            return {"blocks": [{"type": 0, "bbox": (0, 0, 200, 30), "lines": [{
                "bbox": (0, 0, 200, 30), "spans": [
                    {"text": "The ", "flags": 0, "size": 12, "font": "Regular"},
                    {"text": "modus operandi", "flags": 2, "size": 12, "font": "Italic"},
                    {"text": " matters.", "flags": 0, "size": 12, "font": "Regular"},
                ],
            }]}]}

    blocks = _extract_page_blocks(Page(), geometry_order=False)
    assert blocks[0]["italic_source_spans"] == ["modus operandi"]
