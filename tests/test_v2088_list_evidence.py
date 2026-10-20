"""Scoped v20.8.8 regression controls."""

import pytest

from tarjomeh.quality.structure_audit import (
    TRANSLATION_STRUCTURE_MISMATCH,
    audit_payload,
    list_marker_count,
    list_marker_evidence,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1) observe; 2) compare; 3) explain.", 3),
        ("۱) مشاهده؛ ۲) مقایسه؛ ۳) تبیین.", 3),
        ("(1) observe; (2) compare; (3) explain.", 3),
        ("[1] observe; [2] compare.", 2),
        ("see (2020) and (2021)", 0),
        ("2) compare; 3) explain.", 0),
        ("1) observe; 3) explain; 2) compare.", 1),
        ("1) observe; 2) compare; 2) duplicate; 3) explain.", 2),
        ("(1] observe; (2) compare.", 0),
        ("(1) first\n\n(2) unrelated", 1),
        ("1) a; 2) b.\n\n1) x; 2) y; 3) z.", 3),
    ],
)
def test_marker_runs_are_local_ordered_evidence(text, expected):
    assert list_marker_count(text) == expected


def test_five_tasks_are_not_compared_with_later_three_chapters():
    source = "There are five tasks: (1) observe; (2) compare; (3) explain; (4) test; (5) report."
    target = "پنج وظیفه وجود دارد: ۱) مشاهده؛ ۲) مقایسه؛ ۳) تبیین؛ ۴) آزمون؛ ۵) گزارش."
    result = audit_payload(source, target)
    assert TRANSLATION_STRUCTURE_MISMATCH not in result["classifications"]
    assert result["list_marker_evidence"]["candidate"][0]["count"] == 5
    assert list_marker_evidence(target)[0]["markers"][0]["format"] == "closing_parenthesis"


def test_nested_announcement_in_item_is_not_an_unrelated_list():
    assert list_marker_count("1) discuss two issues; 2) compare; 3) explain.") == 3


def test_complete_prefix_does_not_hide_duplicate_extra_item():
    source = "There are two reasons: (1) alpha; (2) beta."
    target = "دو دلیل: ۱) الف؛ ۲) ب؛ ۲) ج."
    assert TRANSLATION_STRUCTURE_MISMATCH in audit_payload(source, target)["classifications"]


@pytest.mark.parametrize(
    "target",
    [
        "سه دلیل وجود دارد: ۱) الف؛ ۲) ب.",
        "دو دلیل وجود دارد: ۱) الف؛ ۲) ب.",
        "سه دلیل وجود دارد: ۱) الف؛ ۳) ج؛ ۲) ب.",
    ],
)
def test_true_count_or_missing_items_still_fail(target):
    source = "There are three reasons: (1) alpha; (2) beta; (3) gamma."
    assert TRANSLATION_STRUCTURE_MISMATCH in audit_payload(source, target)["classifications"]
