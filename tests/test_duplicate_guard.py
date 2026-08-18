"""Tests for item 16 - the duplicate / overlapping edit guard.

Two gaps were closed:

* ``duplicate_paragraph`` blocked whenever a long paragraph appeared twice in the
  candidate, with no baseline at all. A book that legitimately repeats a
  paragraph was blocked and sent to human review for nothing.
* Nothing looked for a duplicated multi-word SPAN. The per-issue salvage path
  rejects an edit that doubles an adjacent word, but a restated clause is
  smaller than a paragraph and larger than a word, so it passed every check -
  which is how a salvage patch duplicated the clause it inserted.

The two baselines are deliberately different, and that asymmetry is the point:
the previous translation is the same language as the candidate, so exact spans
cancel; the source is not, so only the COUNT of duplicated paragraphs is
comparable across languages.
"""

from __future__ import annotations

import pytest

from tarjomeh.quality.integrity import (
    PostEditIntegrityGate,
    newly_repeated_spans,
)

# >= 80 characters, so the paragraph-level check genuinely fires.
FA_LONG = (
    "دولت يک رابطه اجتماعى است و بايد در هر تحليل جدى از قدرت سياسى و نهادهاى "
    "حقوقى چنين بررسى شود."
)
FA_MID = "چند جمله ميانى اينجا قرار دارد تا پاراگراف‌ها از هم جدا شوند."
EN_LONG = (
    "The state is a social relation and must be analysed as such in every "
    "serious account of political power and legal institutions today."
)

DUPLICATED = f"{FA_LONG}\n\n{FA_MID}\n\n{FA_LONG}"
CLEAN = f"{FA_LONG}\n\n{FA_MID}"


def _findings(source: str, candidate: str, previous: str = ""):
    result = PostEditIntegrityGate().evaluate(
        source, candidate, previous=previous, stage="final"
    )
    return (
        [f.check_id for f in result.blocking],
        {f.check_id for f in result.findings},
    )


def test_the_long_paragraph_fixture_is_long_enough() -> None:
    """Guards the test itself: the check only looks at paragraphs >= 80 chars."""
    assert len(FA_LONG) >= 80


# ---------------------------------------------------------------------------
# duplicate_paragraph - no longer a false positive
# ---------------------------------------------------------------------------

def test_translation_that_invents_a_duplicate_is_blocked() -> None:
    blocking, _ = _findings(EN_LONG, DUPLICATED)
    assert "duplicate_paragraph" in blocking


def test_repetition_the_source_itself_contains_is_allowed() -> None:
    """The false positive: a repeated epigraph or restated definition."""
    source = f"{EN_LONG}\n\nSome intervening prose to separate them.\n\n{EN_LONG}"
    blocking, ids = _findings(source, DUPLICATED)
    assert "duplicate_paragraph" not in blocking
    assert "duplicate_paragraph_preserved" in ids


def test_repetition_already_in_the_previous_translation_is_allowed() -> None:
    """An edit elsewhere must not be blamed for damage it did not introduce."""
    blocking, ids = _findings(EN_LONG, DUPLICATED, previous=DUPLICATED)
    assert "duplicate_paragraph" not in blocking
    assert "duplicate_paragraph_preserved" in ids


def test_a_clean_translation_reports_nothing_about_duplication() -> None:
    blocking, ids = _findings(EN_LONG, CLEAN)
    assert "duplicate_paragraph" not in blocking
    assert "duplicate_paragraph_preserved" not in ids


def test_a_repair_that_removes_the_duplication_is_never_blocked() -> None:
    blocking, _ = _findings(EN_LONG, CLEAN, previous=DUPLICATED)
    assert "duplicate_paragraph" not in blocking


# ---------------------------------------------------------------------------
# duplicate_span_introduced - the new span-level check
# ---------------------------------------------------------------------------

PREV = (
    "دولت مدرن بر نهادهاى حقوقى استوار است و قدرت سياسى را سازمان مى‌دهد. "
    "اين سازمان‌دهى در سطوح گوناگون رخ مى‌دهد و پيامدهاى فراوان دارد."
)
RESTATED = (
    PREV + " دولت مدرن بر نهادهاى حقوقى استوار است و قدرت سياسى را سازمان مى‌دهد."
)


def test_an_edit_that_restates_a_clause_is_blocked() -> None:
    """Smaller than a paragraph, larger than a word - previously invisible."""
    blocking, _ = _findings(EN_LONG, RESTATED, previous=PREV)
    assert "duplicate_span_introduced" in blocking


def test_an_ordinary_edit_is_not_blocked() -> None:
    edited = PREV.replace("فراوان", "گسترده")
    blocking, _ = _findings(EN_LONG, edited, previous=PREV)
    assert "duplicate_span_introduced" not in blocking


def test_a_no_op_edit_is_not_blocked() -> None:
    blocking, _ = _findings(EN_LONG, PREV, previous=PREV)
    assert "duplicate_span_introduced" not in blocking


def test_repetition_already_present_is_not_attributed_to_this_edit() -> None:
    blocking, _ = _findings(EN_LONG, RESTATED, previous=RESTATED)
    assert "duplicate_span_introduced" not in blocking


def test_an_initial_translation_is_never_flagged_for_spans() -> None:
    """No same-language baseline exists, and source spans cannot be matched
    across languages. Silence is the only safe answer - this is an EDIT guard."""
    assert newly_repeated_spans("", RESTATED) == []
    blocking, _ = _findings(EN_LONG, RESTATED, previous="")
    assert "duplicate_span_introduced" not in blocking


# ---------------------------------------------------------------------------
# The span helper itself
# ---------------------------------------------------------------------------

def test_newly_repeated_spans_detects_the_restated_clause() -> None:
    assert newly_repeated_spans(PREV, RESTATED)


def test_newly_repeated_spans_ignores_short_repeats() -> None:
    """Short phrases recur legitimately in Persian; the window is 8 words."""
    previous = "قدرت سياسى مهم است."
    candidate = "قدرت سياسى مهم است. قدرت سياسى مهم است."
    assert newly_repeated_spans(previous, candidate) == []


@pytest.mark.parametrize("text", ["", "   ", "یک دو سه"])
def test_newly_repeated_spans_tolerates_trivial_input(text) -> None:
    assert newly_repeated_spans(text, text) == []
    assert newly_repeated_spans("baseline text here", text) == []


def test_a_longer_window_can_be_requested() -> None:
    """min_words is tunable, so enforcement can be loosened without a rewrite."""
    assert newly_repeated_spans(PREV, RESTATED, min_words=200) == []
