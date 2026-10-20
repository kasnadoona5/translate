"""Scoped v20.8.8 regression controls."""

import pytest

from tarjomeh.quality.integrity import (
    object_marker_dash_artifacts,
    object_marker_dash_classification,
)


@pytest.mark.parametrize(
    "target",
    [
        "من کتاب را — که سال‌ها دنبالش بودم — خریدم.",
        "کنشگران تاریخ را — تاریخ خود و دیگران را — می‌سازند.",
        "کنشگران — در این نهادها — تاریخ را می‌سازند.",
        "من کتاب را — که دنبالش بودم — خریدم. آن را — پس از سال‌ها — خواندم.",
    ],
)
def test_valid_paired_asides_do_not_become_object_marker_defects(target):
    result = object_marker_dash_classification("Source with a parenthetical aside.", target)
    assert result["pairs"]
    assert not result["findings"]
    assert "no_source_alignment" in result["scope"]


@pytest.mark.parametrize(
    "target",
    [
        "تاریخ دیگران—را می‌سازند.",
        "تاریخ دیگران — را می‌سازند.",
        "من کتاب را — که سال‌ها دنبالش بودم خریدم.",
        "من کتاب را — که دنبالش بودم – خریدم.",
        "من کتاب را — (که دنبالش بودم — خریدم).",
        "من کتاب را — که دنبالش بودم. — خریدم.",
    ],
)
def test_true_or_ambiguous_detachment_does_not_get_a_pair_exemption(target):
    assert object_marker_dash_artifacts("Source.", target)


def test_even_dash_count_is_not_a_paragraph_exemption():
    target = "کنشگران — در نهادها — تاریخ دیگران—را می‌سازند — سپس می‌روند."
    assert object_marker_dash_artifacts("Actors act.", target)


def test_competing_dash_type_does_not_start_a_new_proven_pair():
    target = "ابتدا — جمله، کتاب را – توضیح – خریدم."
    assert object_marker_dash_artifacts("Source.", target)
