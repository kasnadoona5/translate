"""Grouped integer equivalence affects offline reporting, not production gates."""

import pytest

from tarjomeh.quality.evaluation import inspect_chunk
from tarjomeh.quality.integrity import reconcile_numbers


def numbers(source, target):
    return next(
        c
        for c in inspect_chunk(source, target, "completed", [])["checks"]
        if c["id"] == "numbers_preserved"
    )


@pytest.mark.parametrize(
    "source,target",
    [
        ("The sample contains 1,000 records.", "این نمونه شامل ۱۰۰۰ رکورد است."),
        ("The sample contains 1000 records.", "این نمونه شامل ۱٬۰۰۰ رکورد است."),
        ("There are 12,345,678 records.", "تعداد رکوردها ۱۲۳۴۵۶۷۸ است."),
        (
            "There are 1,000 records and 1,000 files.",
            "تعداد رکوردها ۱۰۰۰ و تعداد پرونده‌ها ۱۰۰۰ است.",
        ),
    ],
)
def test_grouped_prose_integer_equivalents_are_reported(source, target):
    result = numbers(source, target)
    assert result["passed"]
    assert result["details"]["grouped_prose_equivalents"]
    assert reconcile_numbers(source, target)["missing"]  # Production remains conservative.


@pytest.mark.parametrize(
    "source,target",
    [
        ("There are 1,000 records.", "تعداد رکوردها ۲۰۰۰ است."),
        ("There are 1,000 records and 1,000 files.", "تعداد رکوردها ۱۰۰۰ است."),
        ("There are 1,00 records.", "تعداد رکوردها ۱۰۰ است."),
        ("The value is 1,000.5.", "مقدار ۱۰۰۰٫۵ است."),
        ("See Chapter 1,000.", "بنگرید به فصل ۱۰۰۰."),
        ("Code A1,000 is retained.", "کد A1000 حفظ شده است."),
        ("Identifier: 1,000", "شناسه: ۱۰۰۰"),
        ("There are 01,000 records.", "تعداد رکوردها ۱۰۰۰ است."),
        ("The interval is 1,000-2,000.", "بازه ۱۰۰۰-۲۰۰۰ است."),
    ],
)
def test_changed_missing_or_ambiguous_numbers_do_not_gain_a_pass(source, target):
    assert not numbers(source, target)["passed"]


def test_note_protection_is_not_satisfied_by_numeric_equivalence():
    result = inspect_chunk(
        "There are 1,000 records [2].", "تعداد رکوردها ۱۰۰۰ است.", "completed", []
    )
    assert not next(c for c in result["checks"] if c["id"] == "note_markers_preserved")["passed"]


@pytest.mark.parametrize("label", ["Account", "reference", "sample"])
def test_default_prose_role_without_quantity_evidence_is_not_enough(label):
    assert not numbers(f"{label} 1,000.", f"{label} 1000.")["passed"]
