from copy import deepcopy

import pytest

from tarjomeh.core.pipeline import _salvage_local_refinement_edits
from tarjomeh.quality.integrity import PostEditIntegrityGate


def salvage(source, previous, proposed, issues, decisions, **kwargs):
    return _salvage_local_refinement_edits(
        source=source, previous=previous, proposed=proposed,
        issue_details=issues, issue_decisions=decisions,
        integrity_gate=PostEditIntegrityGate(), protected_terms=[],
        protect_inline_english=False, allowed_inline_originals=[], **kwargs,
    )


def issue_decision(old, new, *, issue_id="one", category="addition"):
    return (
        dict(issue_id=issue_id, source_segment_id="p1:s1",
             source_quote="has been asserted", current_persian_quote=old,
             category=category, severity="minor", confidence=0.7),
        dict(issue_id=issue_id, decision="accepted", original_span=old,
             resulting_span=new),
    )


def test_short_deletion_can_use_unchanged_paragraph_context():
    source = "The author's right has been asserted under the Act of 1988."
    old = "به ثبت رسیده و احراز شده است."
    new = "احراز شده است."
    previous = "حق نویسنده بر وفق قانون مصوب 1988 " + old
    proposed = previous.replace(old, new)
    issue, decision = issue_decision(old, new)
    final, enriched, report = salvage(source, previous, proposed, [issue], [decision])
    assert final == proposed
    assert report["committed_count"] == 1
    proof = report["attempts"][0]["contextual_contraction"]
    assert proof["original_span"] == old
    assert proof["resulting_span"] == new
    assert proof["before_sha256"] != proof["candidate_sha256"]
    assert enriched[0]["original_span"] == old
    assert enriched[0]["raw_resulting_span"] == new
    assert enriched[0]["resulting_span"] == new
    replay, _, replay_report = salvage(source, previous, proposed, [issue], enriched)
    assert replay == proposed and replay_report["committed_count"] == 1


@pytest.mark.parametrize("category,quote,segment", [
    ("fluency", "has been asserted", "p1:s1"),
    ("addition", "invented source quote", "p1:s1"),
    ("addition", "has been asserted", "p2:s1"),
])
def test_contraction_requires_existing_source_grounding(category, quote, segment):
    old, new = "به ثبت رسیده و احراز شده است.", "احراز شده است."
    previous = "حق نویسنده بر وفق قانون مصوب 1988 " + old
    issue, decision = issue_decision(old, new, category=category)
    issue.update(source_quote=quote, source_segment_id=segment)
    final, _, _ = salvage("The right has been asserted.", previous,
                           previous.replace(old, new), [issue], [decision])
    assert final == previous


def test_noop_does_not_veto_valid_overlapping_edit():
    previous = "این نظریه دولت را می‌سازز و توضیح می‌دهد."
    proposed = previous.replace("می‌سازز", "می‌سازد")
    bad_i, bad_d = issue_decision("می‌سازز", "می‌سازز", issue_id="noop", category="fluency")
    good_i, good_d = issue_decision("می‌سازز", "می‌سازد", issue_id="good", category="fluency")
    final, enriched, _ = salvage("The theory constructs and explains the state.",
                                previous, proposed, [bad_i, good_i], [bad_d, good_d])
    assert final == proposed
    assert enriched[0]["commit_reason"] == "no_textual_change"
    assert enriched[1]["commit_status"] == "committed_local"


def test_invalid_candidate_decision_does_not_claim_duplicate_ownership():
    previous = "این نظریه دولت را می‌سازز و توضیح می‌دهد."
    proposed = previous.replace("می‌سازز", "می‌سازد")
    good_i, good_d = issue_decision("می‌سازز", "می‌سازد", category="fluency")
    bad_i, bad_d = deepcopy(good_i), deepcopy(good_d)
    bad_i["issue_id"] = bad_d["issue_id"] = "bad"
    bad_i["current_persian_quote"] = "نقل قولی که در متن نیست"
    final, enriched, _ = salvage("The theory constructs and explains the state.",
                                previous, proposed, [bad_i, good_i], [bad_d, good_d])
    assert final == proposed
    assert enriched[0]["commit_status"] == "not_committed"
    assert enriched[1]["commit_status"] == "committed_local"


def test_missing_candidate_result_does_not_veto_valid_edit():
    previous = "این نظریه دولت را می‌سازز و توضیح می‌دهد."
    proposed = previous.replace("می‌سازز", "می‌سازد")
    good_i, good_d = issue_decision("می‌سازز", "می‌سازد", category="fluency")
    bad_i, bad_d = issue_decision("می‌سازز", "می‌سازند", issue_id="bad", category="fluency")
    final, enriched, _ = salvage("The theory constructs and explains the state.",
                                previous, proposed, [bad_i, good_i], [bad_d, good_d])
    assert final == proposed
    assert enriched[0]["commit_reason"] == "resulting_span_not_in_candidate"


def test_two_valid_overlapping_decisions_still_reject():
    previous = "این نظریه دولت را می‌سازز و توضیح می‌دهد."
    proposed = previous.replace("می‌سازز", "می‌سازد")
    a_i, a_d = issue_decision("می‌سازز", "می‌سازد", category="fluency")
    b_i, b_d = issue_decision("دولت را می‌سازز", "دولت را می‌سازد", issue_id="two", category="fluency")
    final, enriched, report = salvage("The theory constructs and explains the state.",
                                      previous, proposed, [a_i, b_i], [a_d, b_d])
    assert final == previous
    assert report["committed_count"] == 0
    assert {x["commit_reason"] for x in enriched} == {"overlapping_local_span"}


def test_context_padding_cannot_block_an_independent_valid_edit():
    old, new = "به ثبت رسیده و احراز شده است.", "احراز شده است."
    previous = "عنوان 1988 " + old
    proposed = "عنوان ۱۹۸۸ " + new
    a_i, a_d = issue_decision(old, new)
    b_i, b_d = issue_decision("عنوان 1988", "عنوان ۱۹۸۸", issue_id="other", category="typography")
    # A meaningful lexical correction occupying all available left context.
    b_d["resulting_span"] = "عنوان نو 1988"
    proposed = previous.replace("عنوان 1988", "عنوان نو 1988").replace(old, new)
    final, _, _ = salvage("The title has been asserted in 1988.", previous, proposed,
                           [a_i, b_i], [a_d, b_d])
    assert final == previous.replace("عنوان 1988", "عنوان نو 1988")


def test_contraction_cannot_delete_a_source_number():
    old, new = "1988 به ثبت رسیده و احراز شده است.", "احراز شده است."
    previous = "حق نویسنده بر وفق قانون مصوب " + old
    issue, decision = issue_decision(old, new)
    final, _, _ = salvage("The right has been asserted under the Act of 1988.",
                           previous, previous.replace(old, new), [issue], [decision])
    assert final == previous


def test_unchanged_context_cannot_mask_removal_of_the_local_predicate():
    from tarjomeh.core.pipeline import _contextual_local_contraction

    old, new = "شرح داده شده است", "شرح"
    before = "متن حاضر درست است " + old + " و ادامه دارد."
    assert _contextual_local_contraction(
        source="The explanation has been asserted.", previous=before,
        proposed=before.replace(old, new), source_segment_id="p1:s1",
        source_quote="has been asserted", current_span=old, resulting_span=new,
        category="addition",
    ) is None
