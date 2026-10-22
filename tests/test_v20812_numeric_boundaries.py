"""Structured numbers must not masquerade as standalone quantities."""

import pytest

from tarjomeh.core.pipeline import _actionable_structure_findings, _blocking_structure_findings
from tarjomeh.quality.structure_audit import announced_count_evidence, audit_structure

SOURCE = "2.2 Aspects of the traditional three-element theory 36"
TARGET = (
    "2.2 \u062c\u0646\u0628\u0647\u200c\u0647\u0627\u06cc\u06cc \u0627\u0632 "
    "\u0646\u0638\u0631\u06cc\u0647\u0654 \u0633\u0646\u062a\u06cc\u0650 "
    "\u0639\u0646\u0627\u0635\u0631 \u0633\u0647\u200c\u06af\u0627\u0646\u0647 36"
)


def test_live_table_label_does_not_invent_a_quantity_mismatch():
    assert announced_count_evidence(TARGET) == []
    assert audit_structure(SOURCE, TARGET) == []
    assert _blocking_structure_findings(_actionable_structure_findings(SOURCE, TARGET)) == []


@pytest.mark.parametrize(
    "number",
    [
        "2.2",
        "2.2.1",
        "12.3",
        "0.2",
        "2\u066b2",
        "\u06f2.\u06f2",
        "\u0662.\u0662",
        "\u06f2\u066b\u06f2",
        "\u0662\u066b\u0662",
        "2.\u06f2.\u0663",
    ],
)
@pytest.mark.parametrize(
    "wrapper", ["{} elements", "({}) elements", "Table {} elements", "{}-element theory"]
)
def test_complete_numeric_span_never_supplies_a_bare_count(number, wrapper):
    assert announced_count_evidence(wrapper.format(number)) == []


@pytest.mark.parametrize("label", ["2.2", "\u06f2.\u06f2", "\u0662\u066b\u0662", "2.2.1"])
def test_label_does_not_hide_a_real_following_quantity(label):
    source = f"{label} Three elements"
    good = f"{label} \u0633\u0647 \u0639\u0646\u0635\u0631"
    bad = f"{label} \u062f\u0648 \u0639\u0646\u0635\u0631"
    assert [item.value for item in announced_count_evidence(source)] == [3]
    assert [item.value for item in announced_count_evidence(good)] == [3]
    assert audit_structure(source, good) == []
    assert _blocking_structure_findings(_actionable_structure_findings(source, bad))


@pytest.mark.parametrize(
    "text,value",
    [
        ("2 elements", 2),
        ("\u06f2 \u062c\u0646\u0628\u0647", 2),
        ("\u0663 elements", 3),
        ("three-element", 3),
        ("\u062f\u0648 \u062c\u0646\u0628\u0647", 2),
        ("2.2: 3 elements", 3),
        ("2. 3 elements", 3),
    ],
)
def test_standalone_quantities_and_compounds_keep_their_evidence(text, value):
    assert [item.value for item in announced_count_evidence(text)] == [value]


def test_mixed_body_and_table_paragraphs_preserve_real_mismatch_detection():
    source = SOURCE + "\n\nThree elements are considered."
    good = (
        TARGET
        + "\n\n\u0633\u0647 \u0639\u0646\u0635\u0631 \u0628\u0631\u0631\u0633\u06cc "
        "\u0645\u06cc\u200c\u0634\u0648\u062f."
    )
    bad = (
        TARGET
        + "\n\n\u062f\u0648 \u0639\u0646\u0635\u0631 \u0628\u0631\u0631\u0633\u06cc "
        "\u0645\u06cc\u200c\u0634\u0648\u062f."
    )
    assert audit_structure(source, good) == []
    findings = _blocking_structure_findings(_actionable_structure_findings(source, bad))
    assert len(findings) == 1
    assert findings[0]["details"]["source_paragraph"] == 1
