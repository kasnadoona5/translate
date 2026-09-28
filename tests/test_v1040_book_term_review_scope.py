import hashlib

from tarjomeh.glossary.book_review import (
    check_reviewed_book_terms,
    resolve_reviewed_book_terms,
)


def _metadata(paragraphs, roles):
    text = "\n\n".join(paragraphs)
    spans = []
    cursor = 0
    for paragraph in paragraphs:
        spans.append([cursor, cursor + len(paragraph)])
        cursor += len(paragraph) + 2
    return text, {
        "paragraph_indices": list(range(len(paragraphs))),
        "source_paragraph_spans": spans,
        "source_paragraph_hashes": [
            hashlib.sha256(p.encode("utf-8")).hexdigest() for p in paragraphs
        ],
        "structural_roles": roles,
    }


def test_approved_term_requires_body_paragraph_not_heading_or_publisher():
    source, metadata = _metadata(
        ["Polity", "Published by Polity Press", "The polity has institutions."],
        ["heading", "body", "body"],
    )
    matches, review = resolve_reviewed_book_terms(
        source, metadata, 1,
        [{"source": "polity", "target": "صورت سیاسی", "status": "approved", "scope_mode": "all_body"}],
    )
    assert [item["paragraph_index"] for item in matches] == [2]
    assert {item["reason"] for item in review} == {
        "non_body_paragraph", "possible_publisher_line"
    }


def test_two_senses_in_same_scope_are_review_not_mandatory():
    source, metadata = _metadata(["Capital has two meanings here."], ["body"])
    decisions = [
        {"source": "capital", "target": "سرمایه", "sense_id": "economic", "status": "approved", "scope_mode": "all_body"},
        {"source": "capital", "target": "پایتخت", "sense_id": "city", "status": "approved", "scope_mode": "all_body"},
    ]
    matches, review = resolve_reviewed_book_terms(source, metadata, 2, decisions)
    assert matches == []
    assert review[0]["reason"] == "ambiguous_approved_sense"
    decisions[1]["chapter_positions"] = [3]
    matches, review = resolve_reviewed_book_terms(source, metadata, 2, decisions)
    assert len(matches) == 1 and not review


def test_uncertain_alignment_and_foreign_original_do_not_gain_authority():
    source = "An introduction.\n\nThe modus operandi is examined."
    decision = {"source": "modus operandi", "target": "شیوهٔ عمل",
                "keep_original": True, "status": "approved", "scope_mode": "all_body"}
    matches, review = resolve_reviewed_book_terms(
        source, {"structural_roles": ["body", "body"]}, 1, [decision]
    )
    assert not matches and review[0]["reason"] == "uncertain_paragraph_alignment"
    text, metadata = _metadata(["The modus operandi is examined."], ["body"])
    matches, review = resolve_reviewed_book_terms(text, metadata, 1, [decision])
    assert len(matches) == 1 and matches[0]["keep_original"] and not review


def test_approved_term_is_checked_in_its_own_paragraph_only():
    source, metadata = _metadata(
        ["The polity is examined.", "Another sentence is here."],
        ["body", "body"],
    )
    matches, review = resolve_reviewed_book_terms(
        source, metadata, 1,
        [{"source": "polity", "target": "\u0633\u06cc\u0627\u0633\u062a", "status": "approved", "scope_mode": "all_body"}],
    )
    assert not review and len(matches) == 1
    wrong, uncertain = check_reviewed_book_terms(
        "\u062f\u0648\u0644\u062a\n\n\u0633\u06cc\u0627\u0633\u062a", source, metadata, matches
    )
    assert not uncertain and not wrong.compliant
    assert wrong.violations[0].chunk_location == "Paragraph 1"
    right, uncertain = check_reviewed_book_terms(
        "\u0633\u06cc\u0627\u0633\u062a\n\n\u062f\u0648\u0644\u062a", source, metadata, matches
    )
    assert right.compliant and not uncertain
    _, uncertain = check_reviewed_book_terms(
        "\u0633\u06cc\u0627\u0633\u062a", source, metadata, matches
    )
    assert uncertain[0]["reason"] == "uncertain_target_paragraph_alignment"


def test_evidence_scope_does_not_apply_to_other_senses_or_paragraphs():
    source, metadata = _metadata(
        ["Capital is an economic relation.", "The capital is a city."],
        ["body", "body"],
    )
    decisions = [{
        "source": "capital", "target": "\u0633\u0631\u0645\u0627\u06cc\u0647",
        "status": "approved", "scope_mode": "evidence_paragraph",
        "source_evidence_sha256": hashlib.sha256(
            b"Capital is an economic relation."
        ).hexdigest(),
    }]
    matches, review = resolve_reviewed_book_terms(source, metadata, 1, decisions)
    assert [item["paragraph_index"] for item in matches] == [0]
    assert not review
