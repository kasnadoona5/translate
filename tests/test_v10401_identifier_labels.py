"""Target-proven ISBN label restoration (live job 6b90f8a9b937, chunk 0)."""

from __future__ import annotations

from tarjomeh.core.pipeline import _restore_source_bound_artifacts
from tarjomeh.quality.integrity import (
    extract_identifiers,
    extract_labeled_identifier_surfaces,
    restore_source_identifiers,
)

SHABAK = "شابک"
SHABAK_13 = f"{SHABAK}-۱۳:"
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
    f"{SHABAK_13} 978-0-7456-3304-6 "
    f"{SHABAK_13} 978-0-7456-3305-3 "
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


def test_bare_localized_label_is_resolved_by_eliminating_latin_labels() -> None:
    # Live attempt 1: the ISBN-13 record kept Latin labels, the other record
    # used a bare Persian label.
    target = TARGET_P9_LATIN + "\n\n" + TARGET_P14_LOCALIZED

    repaired, report = _restore_source_bound_artifacts(SOURCE, target)

    _assert_admissible(repaired)
    assert report["identifier_repair_count"] == 2
    assert repaired.split("\n\n")[0] == TARGET_P9_LATIN
    assert "ISBN 978-0-7456-3304-6 (جلد" in repaired


def test_suffixed_and_bare_localized_labels_are_both_resolved() -> None:
    # Live attempt 2: both records used Persian labels.
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
    first_payload = (
        "۹۷۸-۰-۷۴۵۶-"
        "۳۳۰۴-۶"
    )
    second_payload = (
        "۹۷۸-۰-۷۴۵۶-"
        "۳۳۰۵-۳"
    )
    target = (
        f"{SHABAK_13} {first_payload} {SHABAK_13} {second_payload}"
        "\n\n"
        f"{SHABAK} {first_payload} – {SHABAK} {second_payload}"
    )

    repaired, _report = _restore_source_bound_artifacts(SOURCE, target)

    _assert_admissible(repaired)


def test_swapped_paragraphs_take_labels_from_target_evidence_not_position() -> None:
    # Equal paragraph counts are not alignment proof: the ISBN-13 record is
    # translated second here, so paragraph position would assign it wrongly.
    target = TARGET_P14_LOCALIZED + "\n\n" + TARGET_P9_LATIN

    repaired, _report = _restore_source_bound_artifacts(SOURCE, target)

    _assert_admissible(repaired)
    first, second = repaired.split("\n\n")
    assert "ISBN-13" not in first
    assert "ISBN 978-0-7456-3304-6" in first
    assert second == TARGET_P9_LATIN


def test_swapped_localized_paragraphs_use_explicit_label_suffix() -> None:
    target = TARGET_P14_LOCALIZED + "\n\n" + TARGET_P9_LOCALIZED

    repaired, _report = _restore_source_bound_artifacts(SOURCE, target)

    _assert_admissible(repaired)
    first, second = repaired.split("\n\n")
    assert "ISBN-13" not in first
    assert second.startswith(
        "ISBN-13: 978-0-7456-3304-6 ISBN-13: 978-0-7456-3305-3"
    )


def test_swapped_paragraphs_with_only_bare_localized_labels_stay_unresolved() -> None:
    bare_p9 = TARGET_P9_LOCALIZED.replace(SHABAK_13, SHABAK)
    target = TARGET_P14_LOCALIZED + "\n\n" + bare_p9

    repaired, report = restore_source_identifiers(SOURCE, target)

    assert repaired == target
    assert report["repair_count"] == 0
    assert (
        extract_labeled_identifier_surfaces(SOURCE)
        - extract_labeled_identifier_surfaces(repaired)
    )


def test_bare_localized_labels_without_label_evidence_stay_unresolved() -> None:
    target = (
        f"{SHABAK} 978-0-7456-3304-6 و {SHABAK} 978-0-7456-3305-3"
    )

    repaired, report = restore_source_identifiers(SOURCE, target)

    assert repaired == target
    assert report["repair_count"] == 0


def test_contradictory_label_suffixes_leave_payload_unresolved() -> None:
    # Both target records claim ISBN-13, but the source prints it only once.
    claimed_13 = TARGET_P14_LOCALIZED.replace(
        f"{SHABAK} 978", f"{SHABAK_13} 978"
    )
    target = TARGET_P9_LOCALIZED + "\n\n" + claimed_13

    repaired, report = restore_source_identifiers(SOURCE, target)

    assert repaired == target
    assert report["repair_count"] == 0


def test_payload_printed_twice_in_one_paragraph_stays_unlabeled() -> None:
    payload = "978-0-7456-3304-6"
    source = f"ISBN-13: {payload}. Later record: ISBN {payload}."
    target = f"{SHABAK} {payload}"

    repaired, report = restore_source_identifiers(source, target)

    assert repaired == target
    assert report["repair_count"] == 0
