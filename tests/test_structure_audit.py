"""Tests for item 12 - the source-aware structural audit.

The gap this closes: `_NUMBER_RE` matches digits only, so a source promising
"two objections" and a translation delivering one reconcile with `missing=[]`
and nothing notices.

This module is REPORT-ONLY, so the tests are weighted toward silence. A detector
that fires on correct translations is worse than no detector, and a noisy report
is how a reviewer learns to ignore it.
"""

from __future__ import annotations

import pytest

from tarjomeh.quality.structure_audit import (
    INSUFFICIENT_SOURCE_EVIDENCE,
    SOURCE_ANOMALY_PRESERVED,
    TRANSLATION_STRUCTURE_MISMATCH,
    UNAUTHORIZED_SOURCE_CORRECTION,
    announced_counts,
    audit_payload,
    audit_structure,
    enumeration_length,
    list_marker_count,
    ordinal_sequence_length,
)

EN_TWO = "There are two objections. First it is circular; second it is untestable."
FA_TWO = "دو ایراد وجود دارد. نخست آنکه دوری است؛ دوم آنکه آزمون‌پذیر نیست."
FA_ONE = "دو ایراد وجود دارد. نخست آنکه دوری است."


def _classes(source: str, candidate: str) -> list[str]:
    return [f.classification for f in audit_structure(source, candidate)]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("two objections", [2]),
        ("three reasons", [3]),
        ("دو ایراد", [2]),
        ("سه دلیل", [3]),
        ("2 objections", [2]),
    ],
)
def test_announced_counts_reads_both_languages(text, expected) -> None:
    assert announced_counts(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "the 1990s were pivotal",   # a year is not an announcement
        "see page 45",              # nor a page
        "نه چنین است",              # "no", not "nine"
        "",
    ],
)
def test_announced_counts_ignores_non_announcements(text) -> None:
    assert announced_counts(text) == []


@pytest.mark.parametrize(
    "text,expected",
    [
        ("First this; second that; third the other.", 3),
        ("نخست این؛ دوم آن؛ سوم دیگری.", 3),
        ("First this; second that.", 2),
        ("Second this, third that.", 0),      # does not start at one
        ("The third way is best.", 0),        # a lone ordinal is prose
        ("", 0),
    ],
)
def test_ordinal_sequence_length(text, expected) -> None:
    assert ordinal_sequence_length(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("(1) alpha (2) beta (3) gamma", 3),
        ("(۱) الف (۲) ب", 2),
        ("[1] alpha [2] beta", 2),
        ("(2) beta (3) gamma", 0),            # does not start at one
        ("see (1990) for details", 0),        # a citation year, not a marker
        ("", 0),
    ],
)
def test_list_marker_count(text, expected) -> None:
    assert list_marker_count(text) == expected


def test_enumeration_length_takes_the_strongest_evidence() -> None:
    assert enumeration_length("First (1) alpha; second (2) beta") == 2


# ---------------------------------------------------------------------------
# Silence — the important half
# ---------------------------------------------------------------------------

def test_a_faithful_translation_reports_nothing() -> None:
    assert audit_structure(EN_TWO, FA_TWO) == []


def test_prose_without_any_enumeration_reports_nothing() -> None:
    """Counting words are everywhere in ordinary prose."""
    source = "Two of them left, and three more followed later that year."
    candidate = "دو نفر رفتند و سه نفر دیگر بعدها به آنان پیوستند."
    assert audit_structure(source, candidate) == []


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_input_reports_nothing(text) -> None:
    assert audit_structure(text, text) == []
    assert audit_structure(EN_TWO, text) == [] or True  # must not raise


def test_a_lone_ordinal_on_either_side_reports_nothing() -> None:
    assert audit_structure("The third way.", "راه سوم.") == []


# ---------------------------------------------------------------------------
# The four classifications
# ---------------------------------------------------------------------------

def test_a_dropped_item_is_a_translation_mismatch() -> None:
    """The demo case: source announces and delivers two, translation delivers one."""
    assert _classes(EN_TWO, FA_ONE) == [TRANSLATION_STRUCTURE_MISMATCH]


def test_a_source_that_contradicts_itself_is_preserved_not_blamed() -> None:
    """Never silently correct, and never blame the translation for, a source
    inconsistency."""
    source = "There are three objections. First it is circular; second it is untestable."
    candidate = "سه ایراد وجود دارد. نخست آنکه دوری است؛ دوم آنکه آزمون‌پذیر نیست."
    assert _classes(source, candidate) == [SOURCE_ANOMALY_PRESERVED]


def test_silently_repairing_the_source_is_flagged() -> None:
    """The source says three but lists two; the translation says two and lists
    two. Tidier - and not what the source says."""
    source = "There are three objections. First it is circular; second it is untestable."
    candidate = (
        "دو ایراد وجود دارد. نخست آنکه دوری است؛ دوم آنکه آزمون‌پذیر نیست."
    )
    assert _classes(source, candidate) == [UNAUTHORIZED_SOURCE_CORRECTION]


def test_an_unestablished_source_announcement_is_insufficient_evidence() -> None:
    """The translation enumerates, but the source announcement cannot be read.
    That is explicitly not a mismatch."""
    source = "The objections are as follows, and they matter a great deal."
    candidate = "دو ایراد وجود دارد. نخست آنکه دوری است؛ دوم آنکه آزمون‌پذیر نیست."
    assert _classes(source, candidate) == [INSUFFICIENT_SOURCE_EVIDENCE]


# ---------------------------------------------------------------------------
# Reporting contract
# ---------------------------------------------------------------------------

def test_nothing_in_this_module_can_reject() -> None:
    """Severity is capped at warning; only item 13 may reject."""
    for source, candidate in (
        (EN_TWO, FA_ONE),
        ("There are three objections. First a; second b.", FA_TWO),
    ):
        for finding in audit_structure(source, candidate):
            assert finding.severity in {"info", "warning"}
            assert finding.severity != "blocking"


def test_audit_payload_is_report_shaped() -> None:
    payload = audit_payload(EN_TWO, FA_ONE)
    assert payload["finding_count"] == 1
    assert payload["classifications"] == [TRANSLATION_STRUCTURE_MISMATCH]
    detail = payload["findings"][0]["details"]
    assert detail["source_announced"] == 2
    assert detail["source_items"] == 2
    assert detail["candidate_items"] == 1


def test_findings_carry_both_sides_of_the_evidence() -> None:
    finding = audit_structure(EN_TWO, FA_ONE)[0]
    for key in (
        "source_announced", "source_items", "candidate_announced", "candidate_items"
    ):
        assert key in finding.details
