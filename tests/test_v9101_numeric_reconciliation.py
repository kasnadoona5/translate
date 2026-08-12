"""General regressions for occurrence-aware numeric integrity reconciliation."""

from __future__ import annotations

from tarjomeh.quality.integrity import PostEditIntegrityGate, classify_numbers


def _finding(result, check_id: str):
    return next(item for item in result.findings if item.check_id == check_id)


def test_duplicate_number_roles_reconcile_literal_reference_and_localized_unit() -> None:
    result = PostEditIntegrityGate().evaluate(
        "The argument follows chapter 3 and a 3-volume study.",
        "استدلال از فصل ۳ و یک مطالعه سه‌جلدی پیروی می‌کند.",
        stage="final_translation",
    )

    assert result.accepted
    evidence = _finding(result, "numbers_localized_equivalent")
    assert evidence.severity == "info"
    assert evidence.details["equivalents"] == [{
        "source_value": "3",
        "source_role": "structural",
        "source_label": "volume",
        "target_form": "سه",
    }]


def test_localized_unit_does_not_hide_missing_chapter_reference() -> None:
    result = PostEditIntegrityGate().evaluate(
        "The argument follows chapter 3 and a 3-volume study.",
        "استدلال بر یک مطالعه سه‌جلدی تکیه دارد.",
        stage="final_translation",
    )

    assert not result.accepted
    missing = _finding(result, "numbers_missing")
    assert missing.details["missing"] == ["3"]
    assert missing.details["missing_by_role"] == {"structural": ["3"]}


def test_unrelated_number_word_and_unit_do_not_form_an_equivalence() -> None:
    result = PostEditIntegrityGate().evaluate(
        "The authors completed a 3-volume study.",
        "نویسندگان سه دیدگاه را بررسی کردند و جلد کتاب را بازطراحی کردند.",
    )

    assert not result.accepted
    assert _finding(result, "numbers_missing").details["missing"] == ["3"]


def test_protected_numeric_roles_remain_blocking() -> None:
    cases = (
        ("The study appeared in 2016.", "این پژوهش منتشر شد.", "citation", "2016"),
        ("See page 12.", "به صفحه مراجعه کنید.", "structural", "12"),
        ("The sample was 40 percent.", "نمونه گزارش شد.", "prose", "40%"),
        ("Items 1 and 2 are required.", "دو مورد الزامی است.", "prose", "1"),
    )
    gate = PostEditIntegrityGate()
    for source, target, role, value in cases:
        result = gate.evaluate(source, target)
        assert not result.accepted
        missing = _finding(result, "numbers_missing")
        assert value in missing.details["missing"]
        assert value in missing.details["missing_by_role"][role]


def test_persian_and_arabic_digits_still_match_latin_source_numbers() -> None:
    gate = PostEditIntegrityGate()
    persian = gate.evaluate(
        "Chapter 7 cites 2016 and reports 40 percent.",
        "فصل ۷ به ۲۰۱۶ استناد می‌کند و ۴۰ درصد را گزارش می‌دهد.",
    )
    arabic = gate.evaluate(
        "Chapter 7 cites 2016 and reports 40 percent.",
        "فصل ٧ به ٢٠١٦ استناد می‌کند و ٤٠ درصد را گزارش می‌دهد.",
    )

    assert persian.accepted
    assert arabic.accepted


def test_structural_role_detects_number_before_or_after_unit() -> None:
    roles = classify_numbers("Chapter 3 compares a 3-volume study with page 12.")

    assert roles["structural"] == {"3": 2, "12": 1}


def test_existing_localized_chapter_words_remain_supported() -> None:
    result = PostEditIntegrityGate().evaluate(
        "Chapters 2, 3, and 4 develop the framework.",
        "فصل‌های دوم، سوم و چهارم چارچوب را بسط می‌دهند.",
    )

    assert result.accepted
    equivalents = _finding(
        result, "numbers_localized_equivalent"
    ).details["equivalents"]
    assert [item["source_value"] for item in equivalents] == ["2", "3", "4"]
