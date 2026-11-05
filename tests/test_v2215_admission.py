"""Raw protected-term segments and reference/list boundaries remain conservative."""
import hashlib
from copy import deepcopy

import pytest

from tarjomeh.core.pipeline import _reviewed_term_candidate_admission
from tarjomeh.quality.structure_audit import audit_payload, list_marker_count

TERM = "\u062f\u0648\u0644\u062a \u0645\u0644\u06cc"
SOURCE = "The national state changes. A separate claim changes. The national state remains."
LEFT = TERM + " \u062f\u0631 \u0622\u063a\u0627\u0632 \u067e\u062f\u06cc\u062f\u0627\u0631 \u0627\u0633\u062a."
RIGHT = TERM + " \u062f\u0631 \u067e\u0627\u06cc\u0627\u0646 \u0628\u0627\u0642\u06cc \u0627\u0633\u062a."
OTHER = "\u0627\u06cc\u0646 \u0627\u062f\u0639\u0627 \u0645\u0633\u062a\u0642\u0644 \u0627\u0633\u062a."
SPLIT = "\u0627\u06cc\u0646 \u0627\u062f\u0639\u0627\u0633\u062a. \u0622\u0646 \u0645\u0633\u062a\u0642\u0644 \u0627\u0633\u062a."
BEFORE = LEFT + " " + OTHER + " " + RIGHT


def matches(source=SOURCE, target=TERM):
    return [{"paragraph_index": 0, "global_paragraph_index": 0,
             "source": "national state", "target": target,
             "source_paragraph_hash": hashlib.sha256(source.encode()).hexdigest(),
             "source_occurrence_count": 2}]


def admit(before=BEFORE, after=None, source=SOURCE, reviewed=None):
    return _reviewed_term_candidate_admission(
        source, before, after if after is not None else before, {},
        matches(source) if reviewed is None else reviewed)


@pytest.mark.parametrize("replacement", [
    OTHER.replace("\u0645\u0633\u062a\u0642\u0644", "\u062c\u062f\u06cc\u062f"), SPLIT,
    OTHER.replace(".", "\u060c \u0628\u0627 \u0634\u0631\u0637\u06cc \u062f\u06cc\u06af\u0631."),
])
def test_independent_same_paragraph_edits_carry_review(replacement):
    after = BEFORE.replace(OTHER, replacement)
    current = matches(); original = deepcopy(current)
    result = admit(after=after, reviewed=current)
    assert result["accepted"]
    assert result["unchanged_reviews"][0]["carry_proof"] == "protected_raw_segments_unchanged"
    assert result["unchanged_reviews"][0]["reasons"] == ["multiple_occurrences_unverifiable"]
    assert current == original


def test_unprotected_sentences_can_merge():
    before = BEFORE.replace(OTHER, SPLIT)
    assert admit(before, BEFORE)["accepted"]


@pytest.mark.parametrize("after", [
    BEFORE.replace(TERM, TERM + "\u0650", 1),
    BEFORE.replace(TERM, TERM + "\u200c\u0647\u0627", 1),
    BEFORE.replace(TERM, "other", 1),
    BEFORE.replace(LEFT, LEFT.replace("\u0627\u0633\u062a", "\u0646\u06cc\u0633\u062a")),
    BEFORE + " " + TERM + " C.",
    BEFORE.replace(LEFT, LEFT.replace(".", "\u060c \u0628\u0627 \u0642\u06cc\u062f\u06cc \u062f\u06cc\u06af\u0631.")),
    RIGHT + " " + OTHER + " " + LEFT,
])
def test_changed_protected_text_or_counts_still_block(after):
    assert after != BEFORE
    assert not admit(after=after)["accepted"]


@pytest.mark.parametrize("marker", [
    "\u06af. \u0648. \u0641. \u0647\u06af\u0644", "G. W. F. Hegel", "e.g.", "pp. 3", "\u2026", "...",
    "\u06af\u0650. \u0648\u0650. \u0641\u0650. \u0647\u06af\u0644", "\u00c9. Durand",
])
def test_ambiguous_sentence_boundaries_retain_old_conservative_rule(marker):
    before = BEFORE.replace(LEFT, LEFT.replace(".", " " + marker + "."))
    assert not admit(before, before.replace(OTHER, SPLIT))["accepted"]
    assert admit(before, before)["accepted"]


def test_ambiguous_identical_protected_segments_block():
    before = BEFORE.replace(RIGHT, LEFT)
    assert not admit(before, before.replace(OTHER, SPLIT))["accepted"]


def test_source_hash_or_occurrence_mismatch_still_blocks():
    reviewed = matches(); reviewed[0]["source_paragraph_hash"] = "0" * 64
    assert not admit(after=BEFORE.replace(OTHER, SPLIT), reviewed=reviewed)["accepted"]


def test_no_actual_violation_waiver():
    reviewed = matches(); reviewed[0]["original_introduction_required"] = True
    result = admit(after=BEFORE.replace(OTHER, SPLIT), reviewed=reviewed)
    assert not result["accepted"] and result["violation_count"]


def test_a_changed_sentence_containing_term_stays_blocked():
    before = TERM + " and varieties of a system. " + TERM + " B."
    assert not admit(before, before.replace("varieties", "types"))["accepted"]


LIST_SOURCE = "The study has five tasks: 1) record; 2) measure; 3) compare; 4) test; 5) report."
LIST_TARGET = "The study has five tasks: 1) record; 2) measure; 3) compare; 4) test; 5) report."


@pytest.mark.parametrize("reference", [
    "(Chapters 2-4)", "(pages 2\u20134)",
    "(\u0641\u0635\u0644\u200c\u0647\u0627\u06cc \u06f2 \u062a\u0627 \u06f4)",
    "(\u0641\u0635\u0648\u0644 \u06f2 \u062a\u0627 \u06f4)",
])
def test_headed_reference_tail_does_not_corrupt_a_real_list(reference):
    target = LIST_TARGET + " " + reference + " contains further discussion."
    assert list_marker_count(target) == 5
    assert not audit_payload(LIST_SOURCE, target)["classifications"]


def test_reference_word_elsewhere_in_parentheses_cannot_hide_real_labels():
    text = "(The chapters discuss five tasks: 1) record; 2) measure; 3) compare; 4) test; 5) report)"
    assert list_marker_count(text) == 5


@pytest.mark.parametrize("wrong", [
    LIST_TARGET.replace("4) test; ", ""),
    LIST_TARGET.replace("4) test; 5) report", "5) report; 4) test"),
    LIST_TARGET.replace("5) report", "4) report"),
    LIST_TARGET.replace("five tasks", "four tasks"),
])
def test_real_count_and_list_failures_remain(wrong):
    assert "translation_structure_mismatch" in audit_payload(LIST_SOURCE, wrong)["classifications"]
