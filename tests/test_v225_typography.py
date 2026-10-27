"""Numeral/noun boundaries must survive final typography without guesses."""
import pytest

from tarjomeh.persian.typography import PersianTypographer


@pytest.fixture(scope="module")
def typography():
    return PersianTypographer()


@pytest.mark.parametrize("text", [
    "سه راه برای بررسی مسئله وجود دارد.",
    "او سه راه را شناسایی کرد. این سه راه را خلاصه کرد.",
    "بیست و سه راه دیگر وجود دارد.",
    "سه شنبه",
    "سه راه\n\nدو راه دیگر",
])
def test_authored_numeral_boundary_is_preserved(typography, text):
    assert typography.process(text) == text
    assert typography.process(typography.process(text)) == text


@pytest.mark.parametrize("text,expected", [
    ("سه‌راه شلوغ است.", "سه‌راه شلوغ است."),
    ("سه‌شنبه", "سه‌شنبه"),
    ("کتاب ها را می خوانم.", "کتاب‌ها را می‌خوانم."),
    ("نمی رود", "نمی‌رود"),
    ("آن ها", "آن‌ها"),
    ("۳ راه", "۳ راه"),
    ("3 راه", "۳ راه"),
])
def test_compounds_affixes_and_digits_keep_existing_behavior(typography, text, expected):
    assert typography.process(text) == expected


def test_disabled_normalization_is_not_reenabled(monkeypatch):
    p = PersianTypographer({"normalize_zwnj": False})
    def forbidden():
        raise AssertionError("Hazm must remain disabled")
    monkeypatch.setattr(p, "_get_hazm_normalizer", forbidden)
    assert p.process("سه راه و کتاب ها") == "سه راه و کتاب‌ها"
