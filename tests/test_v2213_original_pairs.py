"""Bounded spelling equivalence must not expand original-annotation authority."""
from copy import deepcopy
import hashlib

import pytest

from tarjomeh.glossary.book_review import (
    BOOK_WIDE_RECOMMENDED, bind_original_introductions,
    check_reviewed_book_terms, original_pair_present, resolve_reviewed_book_terms,
    scoped_original_permissions,
)


TARGET = "\u0646\u0642\u062f \u0627\u06cc\u062f\u0626\u0648\u0644\u0648\u0698\u06cc"
SOURCE = "Ideologiekritik"


@pytest.mark.parametrize("rendered", [TARGET, TARGET + "\u0650",
    TARGET.replace(" ", "\u200c"), TARGET.replace("\u06cc", "\u064a")])
def test_exact_approved_pair_accepts_only_bounded_spelling_variants(rendered):
    item = dict(target=TARGET, source=SOURCE)
    before = deepcopy(item)
    paragraph = rendered + " (" + SOURCE + ")."
    assert original_pair_present(item, paragraph)
    assert item == before


@pytest.mark.parametrize("target", ["\u0634\u06cc\u0648\u0647", "\u0634\u06cc\u0648\u0647\u0654", "\u0634\u06cc\u0648\u06c0"])
@pytest.mark.parametrize("rendered", ["\u0634\u06cc\u0648\u0647", "\u0634\u06cc\u0648\u0647\u0654", "\u0634\u06cc\u0648\u06c0", "\u0634\u06cc\u0648\u0647\u0650"])
def test_final_heh_ezafe_equivalence(target, rendered):
    assert original_pair_present(dict(target=target, source="modus operandi"),
                                 rendered + " (modus operandi)")


@pytest.mark.parametrize("approved,rendered", [
    ("\u06a9\u0646\u0634", "\u0643\u0646\u0634\u0650"),
    ("\u0643\u0646\u0634", "\u06a9\u0646\u0634"),
    ("\u062e\u0637\u200c\u0645\u0634\u06cc", "\u062e\u0637 \u0645\u0634\u06cc"),
    (TARGET + "\u0650", TARGET),
])
def test_variants_are_symmetric_without_removing_a_required_word_boundary(approved, rendered):
    assert original_pair_present(dict(target=approved, source="test concept"),
                                 rendered + " (test concept)")


@pytest.mark.parametrize("target,source", [("", "concept"), (" ", "concept"), (TARGET, ""),
                                         ("\u0650", "concept"), (" \u0650", "concept"),
                                         ("\u200c", "concept"), ("\u0651\u0650", "concept"),
                                         ("\ufeff", "concept"), ("\u200b", "concept"),
                                         ("\u200d", "concept"), ("\u0301", "concept"),
                                         ("\u0640", "concept")])
def test_empty_approval_cannot_authorize_an_annotation(target, source):
    assert not original_pair_present(dict(target=target, source=source), " (" + source + ")")


@pytest.mark.parametrize("target", ["\u0650", "\u200c", "\u0651\u0650", "\ufeff",
                                    "\u200b", "\u200d", "\u0301", "\u0640"])
def test_mark_only_target_cannot_authorize_bare_latin_in_real_consumers(target):
    source, metadata, matches = bound_case()
    matches[0]["target"] = target
    rendered = target.removesuffix("\u0650") + " (locus verum).\n\n\u0627\u062f\u0639\u0627\u06cc\u06cc \u062f\u06cc\u06af\u0631."
    report, _ = check_reviewed_book_terms(rendered, source, metadata, matches)
    assert report.violations
    assert scoped_original_permissions(source, rendered, matches) == []


@pytest.mark.parametrize("target", ["++", "model"])
def test_nonempty_exact_approved_symbols_and_other_scripts_keep_existing_behavior(target):
    assert original_pair_present(dict(target=target, source="concept"), target + " (concept)")


@pytest.mark.parametrize("paragraph", [
    TARGET + " (ideologiekritik)", TARGET + " (Wrong original)",
    TARGET + "\u200c\u0647\u0627 (" + SOURCE + ")",
    TARGET + "\u06cc (" + SOURCE + ")", TARGET.replace(" ", "") + " (" + SOURCE + ")",
    "\u0628\u06cc\u200c" + TARGET + " (" + SOURCE + ")",
    "\u0646\u0627" + TARGET + " (" + SOURCE + ")",
    TARGET + "\u0654 (" + SOURCE + ")", TARGET + "\u0650\u0650 (" + SOURCE + ")",
    TARGET + " [" + SOURCE + "]", TARGET + " (" + SOURCE + ")suffix",
    (TARGET + "\u0650 (" + SOURCE + ") ") * 2,
])
def test_wrong_derived_embedded_or_duplicate_pairs_remain_rejected(paragraph):
    assert not original_pair_present(dict(target=TARGET, source=SOURCE), paragraph)


def bound_case():
    from test_v2085_book_review import _records

    source, metadata, entries, records = _records(["Its locus verum matters.", "Another claim."])
    term = dict(source="locus verum", target=TARGET, status="approved", keep_original=True,
                scope_mode=BOOK_WIDE_RECOMMENDED,
                source_evidence_sha256=hashlib.sha256("Its locus verum matters.".encode()).hexdigest())
    matches, review = resolve_reviewed_book_terms(source, metadata, 1, [term])
    assert not review
    bind_original_introductions(matches, dict(review_version=4, proposals=[term]), entries, records)
    return source, metadata, matches


def test_real_compliance_and_integrity_consumers_share_bounded_pair_proof():
    from tarjomeh.quality.integrity import PostEditIntegrityGate

    source, metadata, matches = bound_case()
    before = deepcopy(matches)
    previous = TARGET + "\u0650.\n\n\u0627\u062f\u0639\u0627\u06cc\u06cc \u062f\u06cc\u06af\u0631."
    good = previous.replace(TARGET + "\u0650", TARGET + "\u0650 (locus verum)", 1)
    report, _ = check_reviewed_book_terms(good, source, metadata, matches)
    assert not report.violations
    assert scoped_original_permissions(source, good, matches) == ["locus verum"]
    assert PostEditIntegrityGate(paragraph_originals=matches).evaluate(
        source, good, previous=previous, protect_inline_english=True,
        allowed_inline_originals=[]).accepted
    assert matches == before


@pytest.mark.parametrize("rendered", ["\u0634\u06cc\u0648\u0647", "\u0634\u06cc\u0648\u0647\u0654", "\u0634\u06cc\u0648\u06c0", "\u0634\u06cc\u0648\u0647\u0650"])
def test_final_heh_pair_compliance_does_not_depend_on_optional_hazm(monkeypatch, rendered):
    import tarjomeh.glossary.compliance as compliance

    monkeypatch.setattr(compliance, "_HAZM_CHECKED", True)
    monkeypatch.setattr(compliance, "_HAZM_NORMALIZER", None)
    source, metadata, matches = bound_case()
    matches[0]["target"] = "\u0634\u06cc\u0648\u0647"
    target = rendered + " (locus verum).\n\n\u0627\u062f\u0639\u0627\u06cc\u06cc \u062f\u06cc\u06af\u0631."
    report, review = check_reviewed_book_terms(target, source, metadata, matches)
    assert not report.violations and not review
    assert scoped_original_permissions(source, target, matches) == ["locus verum"]


@pytest.mark.parametrize("defect", ["other_paragraph", "duplicate", "bare_duplicate", "stale_source", "not_required"])
def test_permissions_still_require_exact_source_unique_original_and_proven_scope(defect):
    source, metadata, matches = bound_case()
    pair = TARGET + "\u0650 (locus verum)"
    target = pair + ".\n\n\u0627\u062f\u0639\u0627\u06cc\u06cc \u062f\u06cc\u06af\u0631."
    if defect == "other_paragraph":
        target = TARGET + ".\n\n" + pair
    elif defect == "duplicate":
        target += " " + pair
    elif defect == "bare_duplicate":
        target += " locus verum"
    elif defect == "stale_source":
        matches[0]["source_paragraph_hash"] = "0" * 64
    else:
        matches[0]["original_introduction_required"] = False
    assert scoped_original_permissions(source, target, matches) == []


def test_repeated_source_stays_reported_uncertain_without_guessed_introduction():
    from test_v2085_book_review import _records

    paragraph = "Its locus verum matters; locus verum remains."
    source, metadata, entries, records = _records([paragraph])
    term = dict(source="locus verum", target=TARGET, status="approved", keep_original=True,
                scope_mode=BOOK_WIDE_RECOMMENDED,
                source_evidence_sha256=hashlib.sha256(paragraph.encode()).hexdigest())
    matches, review = resolve_reviewed_book_terms(source, metadata, 1, [term])
    assert "multiple_occurrences_unverifiable" in [item["reason"] for item in review]
    bind_original_introductions(matches, dict(review_version=4, proposals=[term]), entries, records)
    assert not matches[0]["original_introduction_required"]


def test_runtime_probe_is_pure_and_detects_both_broken_matchers(monkeypatch):
    import tarjomeh.glossary.book_review as module
    from tarjomeh.runtime import _v2213_behavior_probes

    assert _v2213_behavior_probes() == {"bounded_approved_original_pair_spelling": True}
    for value in (False, True):
        monkeypatch.setattr(module, "original_pair_present", lambda *args: value)
        assert not _v2213_behavior_probes()["bounded_approved_original_pair_spelling"]
