"""Paragraph-proven ISBN label restoration (live job 6b90f8a9b937, chunk 0)."""

from __future__ import annotations

from tarjomeh.core.pipeline import _restore_source_bound_artifacts
from tarjomeh.quality.integrity import (
    extract_identifiers,
    extract_labeled_identifier_surfaces,
    restore_source_identifiers,
)

SHABAK = "شابک"
SOURCE_P9 = "ISBN-13: 978-0-7456-3304-6 ISBN-13: 978-0-7456-3305-3(pb)"
SOURCE_P14 = (
    "pages cm Includes bibliographical references and index. "
    "ISBN 978-0-7456-3304-6 (hardback) – "
    "ISBN 978-0-7456-3305-3 (paperback) 1. State, The. I. Title."
)
SOURCE = SOURCE_P9 + "\n\n" + SOURCE_P14
TARGET_P9_LATIN = (
    "ISBN-13: 978-0-7456-3304-6 ISBN-13: 978-0-7456-3305-3 "
    "(جلد شومیز)"
)
TARGET_P9_LOCALIZED = (
    f"{SHABAK}-۱۳: 978-0-7456-3304-6 "
    f"{SHABAK}-۱۳: 978-0-7456-3305-3 "
    "(جلد شومیز)"
)
TARGET_P14_LOCALIZED = (
    "صفحات. "
    f"{SHABAK} 978-0-7456-3304-6 (جلد سخت) "
    f"– {SHABAK} 978-0-7456-3305-3 "
    "(جلد شومیز)"
)


def _assert_admissible(repaired: str) -> None:
    assert not extract_identifiers(SOURCE) - extract_identifiers(repaired)
    assert not (
        extract_labeled_identifier_surfaces(SOURCE)
        - extract_labeled_identifier_surfaces(repaired)
    )
    assert SHABAK not in repaired


def test_localized_label_in_one_paragraph_is_restored_from_that_paragraph() -> None:
    target = TARGET_P9_LATIN + "\n\n" + TARGET_P14_LOCALIZED

    repaired, report = _restore_source_bound_artifacts(SOURCE, target)

    _assert_admissible(repaired)
    assert report["identifier_repair_count"] == 2
    assert repaired.split("\n\n")[0] == TARGET_P9_LATIN
    assert "ISBN 978-0-7456-3304-6 (جلد" in repaired


def test_localized_labels_in_both_paragraphs_get_each_paragraph_label() -> None:
    target = TARGET_P9_LOCALIZED + "\n\n" + TARGET_P14_LOCALIZED

    repaired, report = _restore_source_bound_artifacts(SOURCE, target)

    _assert_admissible(repaired)
    assert report["identifier_repair_count"] == 4
    first, second = repaired.split("\n\n")
    assert first.startswith(
        "ISBN-13: 978-0-7456-3304-6 ISBN-13: 978-0-7456-3305-3"
    )
    assert "ISBN-13" not in second


def test_localized_label_with_persian_digits_is_restored() -> None:
    target = (
        f"{SHABAK}-۱۳: ۹۷۸-۰-۷۴۵۶-"
        "۳۳۰۴-۶ "
        f"{SHABAK}-۱۳: ۹۷۸-۰-۷۴۵۶-"
        "۳۳۰۵-۳"
        "\n\n"
        f"{SHABAK} ۹۷۸-۰-۷۴۵۶-"
        "۳۳۰۴-۶ – "
        f"{SHABAK} ۹۷۸-۰-۷۴۵۶-"
        "۳۳۰۵-۳"
    )

    repaired, _report = _restore_source_bound_artifacts(SOURCE, target)

    _assert_admissible(repaired)


def test_unaligned_paragraphs_do_not_guess_between_two_source_labels() -> None:
    target = f"{SHABAK} 978-0-7456-3304-6 و {SHABAK} 978-0-7456-3305-3"

    repaired, report = restore_source_identifiers(SOURCE, target)

    assert repaired == target
    assert report["repair_count"] == 0


def test_payload_printed_twice_in_aligned_paragraph_stays_unlabeled() -> None:
    payload = "978-0-7456-3304-6"
    source = f"ISBN-13: {payload}. Later record: ISBN {payload}."
    target = f"{SHABAK} {payload}"

    repaired, report = restore_source_identifiers(source, target)

    assert repaired == target
    assert report["repair_count"] == 0
