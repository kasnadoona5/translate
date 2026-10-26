"""Grouped numeric diagnostics need target quantities, not arbitrary labels."""

from copy import deepcopy

import pytest

from tarjomeh.quality.evaluation import inspect_chunk
from tarjomeh.quality.integrity import reconcile_numbers


def numeric_check(source, target):
    return next(
        check for check in inspect_chunk(source, target, "completed", [])["checks"]
        if check["id"] == "numbers_preserved"
    )


@pytest.mark.parametrize(
    "target",
    [
        "Account 1000", "Reference 1000", "Sample 1000", "Code: 1000",
        "حساب ۱۰۰۰", "شمارهٔ حساب ۱۰۰۰", "ارجاع ١٠٠٠", "نمونه ۱۰۰۰",
        "Account 1000. There are 2000 records.",
        "There are files. Reference 1000",
        "تعداد رکوردها مشخص است. شماره حساب ۱۰۰۰",
        "شامل اسناد است. حساب ۱۰۰۰",
        "تعداد رکوردها\n\nحساب ۱۰۰۰",
        "تعداد شماره حساب ۱۰۰۰",
        "تعداد ۱۰۰۰",
        "۱۰۰۰ است.",
        "There are\n\n1000 records.",
        "There are 1000\n\nAccount information.",
        "دارای ۱۰۰۰\n\nحساب بانکی",
    ],
)
def test_grouped_quantity_cannot_match_an_unproven_target_label(target):
    result = numeric_check("There are 1,000 records.", target)
    assert not result["passed"]
    assert result["details"]["missing"] == ["1,000"]
    assert result["details"]["grouped_prose_equivalents"] == []


@pytest.mark.parametrize("number", ["1000", "۱۰۰۰", "١٠٠٠", "۱٬۰۰۰"])
@pytest.mark.parametrize(
    "sentence",
    [
        "There are {number} observations.",
        "The sample contains {number} observations.",
        "این نمونه شامل {number} مشاهده است.",
        "این نمونه حاوی {number} مشاهده است.",
        "این نمونه دارای {number} مشاهده است.",
        "تعداد مشاهده‌ها {number} است.",
        "تعدادِ مشاهده‌ها {number} است.",
    ],
)
def test_explicit_quantities_in_both_scripts_still_match(number, sentence):
    target = sentence.format(number=number)
    result = numeric_check("There are 1,000 observations.", target)
    assert result["passed"]
    if reconcile_numbers("There are 1,000 observations.", target)["missing"]:
        proofs = result["details"]["grouped_prose_equivalents"]
        assert len(proofs) == 1
        assert proofs[0]["target_quantity_context"] in {
            "english_count_construction", "persian_count_construction",
        }


@pytest.mark.parametrize(
    "target,missing",
    [
        ("Account 1000. There are 1000 files.", 1),
        ("تعداد پرونده‌ها ۱۰۰۰ است. حساب ۱۰۰۰", 1),
        ("تعداد پرونده‌ها ۱۰۰۰ و تعداد تصویرها ۱۰۰۰ است.", 0),
    ],
)
def test_only_proven_target_occurrences_are_consumed_once(target, missing):
    result = numeric_check("There are 1,000 files and 1,000 images.", target)
    assert len(result["details"]["missing"]) == missing
    assert len(result["details"]["grouped_prose_equivalents"]) == 2 - missing


def test_literal_numbers_and_input_objects_are_unchanged():
    events = [{"event_type": "unrelated", "payload": {"text": "kept"}}]
    before = deepcopy(events)
    source, target = "Reference 1000", "Reference 1000"
    result = inspect_chunk(source, target, "completed", events)
    assert next(c for c in result["checks"] if c["id"] == "numbers_preserved")["passed"]
    assert events == before
    assert reconcile_numbers("There are 1,000 records.", "تعداد رکوردها ۱۰۰۰ است.")["missing"]
