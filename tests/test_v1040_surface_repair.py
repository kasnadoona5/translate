"""Conservative body-prose surface fixes preserve source obligations."""

from tarjomeh.quality.integrity import repair_proven_surface_artifacts


def test_unique_duplicate_coordinator_and_double_zwnj_are_idempotent():
    source = "Polity and power are distinct."
    target = "\u0633\u06cc\u0627\u0633\u062a\u200c\u200c\u0647\u0627 \u0648 \u0648 \u0642\u062f\u0631\u062a"
    repaired, edits = repair_proven_surface_artifacts(source, target)
    assert repaired == "\u0633\u06cc\u0627\u0633\u062a\u200c\u0647\u0627 \u0648 \u0642\u062f\u0631\u062a"
    assert {item["type"] for item in edits} == {"double_zwnj", "duplicate_coordinator"}
    assert repair_proven_surface_artifacts(source, repaired) == (repaired, [])


def test_source_repetition_mixed_role_and_uncertain_alignment_are_unchanged():
    target = "\u062f\u0648\u0644\u062a \u0648 \u0648 \u062c\u0627\u0645\u0639\u0647"
    assert repair_proven_surface_artifacts("and and", target) == (target, [])
    assert repair_proven_surface_artifacts("State and society", target,
                                          structural_role="heading") == (target, [])
    assert repair_proven_surface_artifacts("One.\n\nTwo.", target) == (target, [])


def test_surface_fix_preserves_exact_paragraph_separators():
    source = "One and two.\n\nThree."
    target = "  \u06cc\u06a9 \u0648 \u0648 \u062f\u0648  \n \n\u0633\u0647  "
    repaired, edits = repair_proven_surface_artifacts(source, target)
    assert repaired == "  \u06cc\u06a9 \u0648 \u062f\u0648  \n \n\u0633\u0647  "
    assert len(edits) == 1
