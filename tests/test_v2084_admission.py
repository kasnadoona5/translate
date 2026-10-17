from copy import deepcopy

from test_v208_quality_safeguards import AFTER, BEFORE, SOURCE, old_issue

from tarjomeh.core.pipeline import _classify_candidate_quality_count, _proven_unchanged_issue


def test_r109_requires_raw_paragraph_bytes_not_stripped_identity():
    issue = old_issue()
    assert not _proven_unchanged_issue(
        issue, source_text=SOURCE, previous_text=BEFORE, candidate_text=AFTER + " "
    )


def test_low_confidence_old_major_is_carried_but_not_resolved():
    issue = {**old_issue(), "severity": "major", "category": "accuracy", "confidence": 0.60}
    snapshot = deepcopy(issue)
    remaining, carried, vetoed = _classify_candidate_quality_count(
        [issue], source_text=SOURCE, previous_text=BEFORE, candidate_text=AFTER
    )
    assert not remaining and carried == [issue] and not vetoed and issue == snapshot


def test_veto_requires_exact_fingerprint_and_unchanged_paragraph():
    issue = old_issue()
    remaining, carried, vetoed = _classify_candidate_quality_count(
        [issue],
        source_text=SOURCE,
        previous_text=BEFORE,
        candidate_text=AFTER,
        vetoed_issues=[issue],
    )
    assert not remaining and not carried and vetoed == [issue]
    changed = AFTER + " متن تازه"
    remaining, carried, vetoed = _classify_candidate_quality_count(
        [issue],
        source_text=SOURCE,
        previous_text=BEFORE,
        candidate_text=changed,
        vetoed_issues=[issue],
    )
    assert remaining == [issue] and not carried and not vetoed


def test_unlocated_new_issue_cannot_be_excluded():
    issue = {**old_issue(), "source_segment_id": ""}
    remaining, carried, vetoed = _classify_candidate_quality_count(
        [issue],
        source_text=SOURCE,
        previous_text=BEFORE,
        candidate_text=AFTER,
        vetoed_issues=[issue],
    )
    assert remaining == [issue] and not carried and not vetoed


def test_low_confidence_major_does_not_block_actual_extra_final_admission(tmp_path):
    from test_v208_quality_safeguards import run_extra_case
    from test_v2081_quality_safeguards import _changed_issue, _decisions

    from tarjomeh.quality.critique import CritiqueResult

    old = {**old_issue(), "severity": "major", "category": "accuracy", "confidence": 0.60}
    changed = _changed_issue()
    review = CritiqueResult(
        accuracy=9, fluency=9, terminology=9, register=9, average=9, issue_details=[old]
    )
    retained, report, final, _ = run_extra_case(
        tmp_path, SOURCE, BEFORE, AFTER, [changed], review, _decisions(changed)
    )
    assert retained == AFTER and report["accepted"]
    assert final["unresolved_grounded_issue_ids"] == ["old"]
    assert not final["durable_authority"]
