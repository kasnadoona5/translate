from test_v1035_final_quality_authority import _salvage


def _issue(key="a", quote="می‌سازز", **extra):
    return {"issue_id": key, "source_segment_id": "p1:s1", "current_persian_quote": quote, **extra}


def test_rejected_overlap_cannot_block_or_satisfy_accepted_edit():
    before = "این نظریه دولت را می‌سازز و توضیح می‌دهد."
    after = before.replace("می‌سازز", "می‌سازد")
    issues = [_issue(), _issue("b", "دولت را می‌سازز")]
    decisions = [
        {"issue_id": "a", "decision": "accepted", "resulting_span": "می‌سازد"},
        {"issue_id": "b", "decision": "rejected", "resulting_span": "دولت را می‌سازد"},
    ]
    final, enriched, _ = _salvage(before, after, issues, decisions)
    assert final == after and enriched[0]["commit_status"] == "committed_local"
    assert enriched[1]["commit_status"] == "not_committed"


def test_rejected_duplicate_never_inherits_commit():
    before = "این نظریه را می‌سازز."
    after = before.replace("می‌سازز", "می‌سازد")
    issues = [_issue(), _issue("b")]
    decisions = [
        {"issue_id": key, "decision": choice, "resulting_span": "می‌سازد"}
        for key, choice in [("a", "accepted"), ("b", "rejected")]
    ]
    final, enriched, _ = _salvage(before, after, issues, decisions)
    assert final == after and enriched[1]["commit_status"] == "not_committed"


def test_candidate_comparison_uses_same_safe_orthography_as_result_span():
    before = "این امر انجام می‌گردز."
    raw_after = before.replace("می‌گردز", "می شود")
    final, enriched, _ = _salvage(
        before,
        raw_after,
        [_issue(quote="می‌گردز")],
        [{"issue_id": "a", "decision": "accepted", "resulting_span": "می شود"}],
    )
    assert final == before.replace("می‌گردز", "می‌شود")
    assert enriched[0]["raw_resulting_span"] == "می شود"


def test_objective_spacing_edit_is_not_a_noop():
    before = "این امر انجام می شود."
    after = before.replace("می شود", "می‌شود")
    issue = _issue(
        quote="می شود",
        category="typography",
        severity="major",
        rationale="Incorrect spacing in the Persian verb prefix.",
    )
    final, _, _ = _salvage(
        before,
        after,
        [issue],
        [{"issue_id": "a", "decision": "accepted", "resulting_span": "می‌شود"}],
    )
    assert final == after


def test_spacing_without_objective_evidence_stays_unchanged():
    before = "این امر انجام می شود."
    final, _, _ = _salvage(
        before,
        before.replace("می شود", "می‌شود"),
        [_issue(quote="می شود")],
        [{"issue_id": "a", "decision": "accepted", "resulting_span": "می‌شود"}],
    )
    assert final == before


def test_decimal_punctuation_conversion_is_not_sentence_removal():
    before = "مقدار برابر 3.5 است."
    after = "مقدار برابر ۳٫۵ است."
    final, _, _ = _salvage(
        before,
        after,
        [_issue(quote="3.5")],
        [{"issue_id": "a", "decision": "accepted", "resulting_span": "۳٫۵"}],
    )
    assert final == after


def test_real_sentence_terminal_removal_still_blocks():
    before = "این نظریه است. آن تحلیل است."
    after = before.replace("است. آن", "است و آن")
    final, _, _ = _salvage(
        before,
        after,
        [_issue(quote="است. آن")],
        [{"issue_id": "a", "decision": "accepted", "resulting_span": "است و آن"}],
    )
    assert final == before


def test_internal_scholarly_dots_not_counted_as_sentence_stops():
    from tarjomeh.core.pipeline import _sentence_terminal_count

    assert _sentence_terminal_count("e.g. 3.5") == 1
    assert _sentence_terminal_count("e.g. 3.5.") == 2
    assert _sentence_terminal_count("ibid.") == 1
