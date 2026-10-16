import pytest

from tarjomeh.context.book_researcher import BookResearcher


@pytest.mark.parametrize("date", ["2004/2005", "2004 or 2005", "2004-2005", "2004"])
def test_publisher_before_date_is_not_uploaded_edition_evidence(date):
    original = f"A study (published by Example, {date}). It examines institutions."
    text, conflicts = BookResearcher._reconcile_publication_context(
        original, [{"kind": "first_publication", "year": 2005}],
    )
    assert "published by" not in text
    assert "It examines institutions." in text
    assert conflicts[0]["claim"] == f"(published by Example, {date})"
    assert conflicts[0]["source_evidence"]
    assert original == f"A study (published by Example, {date}). It examines institutions."


@pytest.mark.parametrize("publisher", ["Example", "Example Press"])
def test_typed_first_publication_may_match_but_reprint_does_not_prove_it(publisher):
    value = f"A study (first published by {publisher}, 2005)."
    assert BookResearcher._reconcile_publication_context(
        value, [{"kind": "first_publication", "year": 2005}],
    ) == (value, [])
    text, conflicts = BookResearcher._reconcile_publication_context(
        value, [{"kind": "reprint", "year": 2005}],
    )
    assert "2005" not in text and conflicts


def test_other_works_and_historical_dates_are_not_uploaded_metadata():
    value = ("An earlier book (published by Example Press, 1992/1993) is discussed. "
             "The movement began in 1848 (Smith, 1999).")
    assert BookResearcher._reconcile_publication_context(value, []) == (value, [])
