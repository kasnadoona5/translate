"""Tests for Unicode corruption detection in the integrity gate.

Before this check, nothing in ``src/`` referenced U+FFFD or the private-use
typography sentinels at all, which is how exactly one U+FFFD survived into the
delivered DOCX. Corruption that damages a number was caught incidentally by
``numbers_missing``; corruption that damages a *letter* was invisible.

The non-negotiable rule that shapes these tests: damage already present in the
source is preserved and reported, never treated as something the translation
introduced, and never silently corrected.
"""

from __future__ import annotations

import pytest

from tarjomeh.quality.integrity import (
    PostEditIntegrityGate,
    corruption_artifacts,
    describe_corruption,
)

SOURCE = "The modern state rests on institutional power and legal authority."
CLEAN = "دولت مدرن بر قدرت نهادی و اقتدار قانونی استوار است."

REPLACEMENT = "�"
SENTINEL_OPEN = ""
SENTINEL_CLOSE = ""


def _blocking(candidate: str, *, source: str = SOURCE, previous: str = "") -> list[str]:
    result = PostEditIntegrityGate().evaluate(
        source, candidate, previous=previous, stage="final"
    )
    return [finding.check_id for finding in result.blocking]


def _corruption_findings(
    candidate: str, *, source: str = SOURCE, previous: str = ""
) -> list[tuple[str, str]]:
    result = PostEditIntegrityGate().evaluate(
        source, candidate, previous=previous, stage="final"
    )
    return [
        (finding.check_id, finding.severity)
        for finding in result.findings
        if "corruption" in finding.check_id and "mixed_script" not in finding.check_id
    ]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_clean_text_has_no_artifacts() -> None:
    assert corruption_artifacts(CLEAN) == {}
    assert corruption_artifacts("") == {}
    assert corruption_artifacts(None) == {}
    assert describe_corruption(CLEAN) == []


@pytest.mark.parametrize(
    "damaged,category",
    [
        (CLEAN.replace("نهادی", f"نهاد{REPLACEMENT}ی"), "replacement_character"),
        (f"{SENTINEL_OPEN}0{SENTINEL_CLOSE}" + CLEAN, "unrestored_sentinel"),
        (CLEAN.replace(" ", "\x0b", 1), "control_character"),
        (CLEAN + "\x7f", "control_character"),
    ],
)
def test_each_corruption_category_is_detected(damaged, category) -> None:
    assert category in corruption_artifacts(damaged)


def test_tab_newline_and_carriage_return_are_not_corruption() -> None:
    """Real translations contain these; flagging them would be a false positive."""
    assert corruption_artifacts("first\tsecond\nthird\r\nfourth") == {}


def test_ordinary_persian_typography_is_not_corruption() -> None:
    """ZWNJ, Persian digits and the decimal separator must all pass."""
    text = "می‌شود ۱۹۹۰ و ۱۰٫۵ درصد ، ؛ ؟"
    assert corruption_artifacts(text) == {}


def test_describe_corruption_reports_codepoints_and_is_bounded() -> None:
    damaged = CLEAN + (REPLACEMENT * 20)
    samples = describe_corruption(damaged)
    assert samples, "corruption must be described for QA evidence"
    assert len(samples) <= 5, "evidence must stay bounded"
    assert "U+FFFD" in samples[0]


# ---------------------------------------------------------------------------
# Gate behaviour
# ---------------------------------------------------------------------------

def test_a_clean_candidate_is_not_blocked() -> None:
    assert "unicode_corruption" not in _blocking(CLEAN)


def test_corruption_that_spares_every_number_is_still_blocked() -> None:
    """The real gap. numbers_missing only noticed damage that hit a digit, so
    a corrupted letter passed every check and shipped."""
    damaged = CLEAN.replace("نهادی", f"نهاد{REPLACEMENT}ی")
    assert "unicode_corruption" in _blocking(damaged)


def test_an_unrestored_sentinel_is_blocked() -> None:
    """A PersianTypographer sentinel that was never restored is corruption."""
    damaged = CLEAN.replace("قدرت", f"{SENTINEL_OPEN}{SENTINEL_CLOSE}قدرت")
    assert "unicode_corruption" in _blocking(damaged)


def test_a_control_character_is_blocked() -> None:
    assert "unicode_corruption" in _blocking(CLEAN.replace(" ", "\x0b", 1))


def test_damage_already_in_the_source_is_a_note_not_a_rejection() -> None:
    """Never silently correct - and never blame the translation for - a source
    inconsistency. The translation faithfully carries the source's own damage.
    """
    damaged_source = f"The modern st{REPLACEMENT}ate rests on institutional power."
    faithful = CLEAN.replace("دولت", f"دول{REPLACEMENT}ت")
    findings = _corruption_findings(faithful, source=damaged_source)
    assert findings == [("unicode_corruption_still_present", "warning")]
    assert "unicode_corruption" not in _blocking(faithful, source=damaged_source)


def test_pre_existing_damage_carried_forward_is_a_note_not_a_rejection() -> None:
    """A refinement that fails to repair earlier damage must not be rejected for
    it - the prior valid translation is what gets retained either way."""
    damaged = CLEAN.replace("نهادی", f"نهاد{REPLACEMENT}ی")
    findings = _corruption_findings(damaged, previous=damaged)
    assert findings == [("unicode_corruption_still_present", "warning")]
    assert "unicode_corruption" not in _blocking(damaged, previous=damaged)


def test_newly_added_damage_is_blocked_even_when_earlier_damage_exists() -> None:
    """One pre-existing U+FFFD must not license a second one."""
    previous = CLEAN.replace("نهادی", f"نهاد{REPLACEMENT}ی")
    worse = previous.replace("قانونی", f"قانون{REPLACEMENT}ی")
    assert "unicode_corruption" in _blocking(worse, previous=previous)


def test_a_repair_that_removes_damage_is_never_blocked_for_corruption() -> None:
    """The refiner fixing the corruption must be allowed through."""
    previous = CLEAN.replace("نهادی", f"نهاد{REPLACEMENT}ی")
    assert _corruption_findings(CLEAN, previous=previous) == []
    assert "unicode_corruption" not in _blocking(CLEAN, previous=previous)
