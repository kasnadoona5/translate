"""Both count sides and applied edit ownership must use the same evidence."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from test_v208_quality_safeguards import run_extra_case
from tarjomeh.core.pipeline import _classify_candidate_quality_change, _recover_non_regressed_local_edits
from tarjomeh.quality.critique import CritiqueResult
from tarjomeh.quality.integrity import PostEditIntegrityGate


SOURCE = "They act.\n\nA civic institution meets.\n\nA public office opens."
BEFORE = "\u0622\u0646\u0647\u0627 \u0645\u06cc\u200c\u0631\u0648\u0646\u062f.\n\n\u0646\u0647\u0627\u062f \u0645\u062f\u0646\u06cc \u062f\u06cc\u062f\u0627\u0631 \u0645\u06cc\u200c\u06a9\u0646\u062f.\n\n\u062f\u0641\u062a\u0631 \u0639\u0645\u0648\u0645\u06cc \u0628\u0627\u0632 \u0645\u06cc\u200c\u0634\u0648\u062f."
NO_GAIN = BEFORE.replace("\u0622\u0646\u0647\u0627", "\u0622\u0646\u0627\u0646")
FIXED = BEFORE.replace("\u0645\u06cc\u200c\u0631\u0648\u0646\u062f", "\u06a9\u0646\u0634 \u0645\u06cc\u200c\u06a9\u0646\u0646\u062f")


def issue(key, paragraph, source, target, category="accuracy"):
    return dict(issue_id=key, source_segment_id=f"p{paragraph}:s1", source_quote=source,
                current_persian_quote=target, suggested_correction="\u06a9\u0646\u0634 \u0645\u06cc\u200c\u06a9\u0646\u0646\u062f",
                category=category, severity="minor", confidence=0.65,
                rationale="Source-aware lexical issue.")


def findings():
    return [issue("A", 1, "They act", BEFORE.split("\n\n")[0]),
            issue("B1", 2, "civic institution", "\u0646\u0647\u0627\u062f \u0645\u062f\u0646\u06cc", "terminology"),
            issue("B2", 3, "public office", "\u062f\u0641\u062a\u0631 \u0639\u0645\u0648\u0645\u06cc", "terminology")]


@pytest.mark.parametrize("new_finding", [False, True])
def test_actual_extra_final_cannot_claim_gain_from_untouched_findings(tmp_path, new_finding):
    baseline = findings()
    persistent = {**baseline[0], "issue_id": "A_again", "current_persian_quote": NO_GAIN.split("\n\n")[0]}
    details = [persistent, *baseline[1:]]
    if new_finding:
        details.append(issue("N", 1, "They", "\u0622\u0646\u0627\u0646", "terminology"))
    review = CritiqueResult(accuracy=9, fluency=9, terminology=9, register=9, average=9, issue_details=details)
    decisions = [dict(issue_id="A", decision="accepted", original_span=baseline[0]["current_persian_quote"],
                      resulting_span=persistent["current_persian_quote"])]
    retained, report, _final, calls = run_extra_case(tmp_path, SOURCE, BEFORE, NO_GAIN, baseline, review, decisions)
    assert retained == BEFORE and not report["accepted"]
    assert report["before_actionable_count_raw"] == 3
    assert report["before_actionable_count"] == 1
    assert report["after_actionable_count"] == 1 + int(new_finding)
    assert set(report["before_count_excluded_unchanged_issue_ids"]) == {"B1", "B2"}
    assert calls == ["recorded_refiner", "recorded_critic"]


def test_actual_independent_fix_still_passes_and_old_findings_remain_unresolved(tmp_path):
    baseline = findings()
    review = CritiqueResult(accuracy=9, fluency=9, terminology=9, register=9, average=9,
                            issue_details=baseline[1:])
    decisions = [dict(issue_id="A", decision="accepted", original_span=baseline[0]["current_persian_quote"],
                      resulting_span=FIXED.split("\n\n")[0])]
    retained, report, final, _ = run_extra_case(tmp_path, SOURCE, BEFORE, FIXED, baseline, review, decisions)
    assert retained == FIXED and report["accepted"]
    assert report["before_actionable_count"] == 1 and report["after_actionable_count"] == 0
    assert set(final["unresolved_grounded_issue_ids"]) == {"B1", "B2"}
    assert not final["durable_authority"]


@pytest.mark.parametrize("choice", ["rejected", "accepted", "partially_applied"])
def test_rejected_or_noop_decision_cannot_establish_regression_ownership(choice):
    before = "\u0627\u06cc\u0646 \u0646\u0638\u0631\u06cc\u0647 \u0645\u06cc\u200c\u0633\u0627\u0632\u0632. \u0645\u062a\u0646 \u0628\u062f \u0627\u0633\u062a."
    proposed = before.replace("\u0645\u06cc\u200c\u0633\u0627\u0632\u0632", "\u0645\u06cc\u200c\u0633\u0627\u0632\u062f")
    issues = [issue("safe", 1, "A theory acts.", "\u0645\u06cc\u200c\u0633\u0627\u0632\u0632"),
              issue("veto", 1, "A theory acts.", "\u0645\u062a\u0646 \u0628\u062f")]
    decisions = [dict(issue_id="safe", decision="accepted", original_span="\u0645\u06cc\u200c\u0633\u0627\u0632\u0632", resulting_span="\u0645\u06cc\u200c\u0633\u0627\u0632\u062f"),
                 dict(issue_id="veto", decision=choice, original_span="\u0645\u062a\u0646 \u0628\u062f", resulting_span="\u0645\u062a\u0646 \u0628\u062f")]
    snapshot = deepcopy(decisions)
    retained, _decisions, report, excluded = _recover_non_regressed_local_edits(
        source="A theory acts.", baseline=before, proposed=proposed, issue_details=issues,
        issue_decisions=decisions, regressions=[dict(current_persian_quote="\u0645\u062a\u0646 \u0628\u062f", source_segment_id="p1:s1",
                                                   category="accuracy", severity="major", confidence=0.9)],
        integrity_gate=PostEditIntegrityGate(), protected_terms=[], protect_inline_english=False,
        allowed_inline_originals=[])
    assert retained == before and not excluded
    assert report["policy"] == "ambiguous_regression_restored_exact_baseline"
    assert decisions == snapshot


@pytest.mark.parametrize("case", ["raw_space", "unlocated", "source_scope", "same_paragraph"])
def test_symmetric_counts_do_not_relax_unchanged_scope_proof(case):
    before = findings()
    candidate = FIXED
    if case == "raw_space":
        parts = candidate.split("\n\n")
        parts[1] += " "
        candidate = "\n\n".join(parts)
    elif case == "unlocated":
        before[1].pop("source_segment_id")
    elif case == "source_scope":
        before[1]["source_segment_id"] = "p1:s1"
    else:
        before[1]["source_segment_id"] = "p1:s1"
        before[1]["source_quote"] = "They act"
        before[1]["current_persian_quote"] = BEFORE.split("\n\n")[0]
    _remaining, _carried, _vetoed, report = _classify_candidate_quality_change(
        before, before[1:], source_text=SOURCE, previous_text=BEFORE, candidate_text=candidate)
    assert "B1" not in report["before_count_excluded_unchanged_issue_ids"]


def test_count_dedup_preserves_distinct_findings_at_the_same_quote():
    a = findings()[0]
    distinct = dict(a, issue_id="distinct", suggested_correction="\u062a\u0635\u062d\u06cc\u062d \u062f\u06cc\u06af\u0631")
    _remaining, _carried, _vetoed, report = _classify_candidate_quality_change(
        [a, dict(a, issue_id="duplicate"), distinct], [a, distinct], source_text=SOURCE,
        previous_text=BEFORE, candidate_text=NO_GAIN)
    assert report["before_actionable_count_raw"] == 3
    assert report["before_actionable_count"] == report["after_actionable_count"] == 2


def test_both_count_reports_retain_all_raw_provider_alias_ids():
    baseline = findings()
    alias = dict(baseline[1], issue_id="provider-alias")
    _remaining, _carried, _vetoed, report = _classify_candidate_quality_change(
        [*baseline, alias], [*baseline[1:], alias], source_text=SOURCE,
        previous_text=BEFORE, candidate_text=FIXED)
    assert report["before_actionable_count_raw"] == 4
    assert report["after_actionable_count_raw"] == 3
    assert report["before_actionable_count"] == 1 and report["after_actionable_count"] == 0
    for key in ("before_count_excluded_unchanged_issue_ids", "count_excluded_unchanged_issue_ids"):
        assert report[key] == ["B1", "B2", "provider-alias"]


@pytest.mark.parametrize("fixed", [False, True])
def test_real_exact_final_path_uses_symmetric_counts_and_retains_valid_baseline(fixed):
    from test_v20812_pipeline_export import run_chunk
    from tarjomeh.quality.refiner import RefinementResult

    baseline = findings()
    for item in baseline:
        item.update(severity="major", confidence=0.9)
    baseline[0]["current_persian_quote"] = "\u0645\u06cc\u200c\u0631\u0648\u0646\u062f"
    candidate = FIXED if fixed else NO_GAIN
    candidate_details = baseline[1:] if fixed else baseline
    pipeline, kwargs = run_chunk(SOURCE, BEFORE)
    pipeline.config.translation.enable_critique = True
    pipeline.config.translation.max_refine_iterations = 0
    pipeline.config.translation.extra_final_refine_attempts = 0
    pipeline._canonicalize_final_translation = lambda job, idx, chunk, text: text
    kwargs["chunk"].metadata = dict(paragraph_indices=[0, 1, 2], structural_roles=["body"] * 3)
    calls = []

    async def critique(source, target, **extra):
        calls.append(target)
        details = [] if len(calls) == 1 else baseline if len(calls) == 2 else candidate_details
        return CritiqueResult(accuracy=9, fluency=9, terminology=9, register=9, average=9,
                               issue_details=deepcopy(details))

    async def refine(source, target, review, **extra):
        return RefinementResult(translation=candidate, decision="revised", rationale="Recorded local edit.",
                                issue_decisions=[dict(issue_id="A", decision="accepted",
                                                     original_span=BEFORE.split("\n\n")[0],
                                                     resulting_span=candidate.split("\n\n")[0])])

    kwargs["critique_tool"] = SimpleNamespace(critique=critique)
    kwargs["refiner_tool"] = SimpleNamespace(refine_with_decision=refine)
    retained = pipeline._translate_single_chunk(**kwargs)
    events = [(call.args[2], call.args[3]) for call in pipeline.db.log_chunk_event.call_args_list]
    report = next(payload for kind, payload in events if kind == "exact_final_quality_repair")
    assert report["accepted"] is fixed
    assert report["before_actionable_count_raw"] == 3
    assert report["before_actionable_count"] == 1
    assert report["after_actionable_count"] == (0 if fixed else 1)
    assert retained == (FIXED if fixed else BEFORE)
    assert len(calls) == 3
    assert not any(kind == "integrity_final_failed" for kind, _ in events)


@pytest.mark.parametrize("unlocated", [False, True])
def test_only_applied_local_edits_can_own_each_regression(unlocated):
    before, proposed = "bad has a separator and safe.", "worse has a separator and clean."
    issues = [dict(issue_id=key, source_segment_id="p1:s1", current_persian_quote=word)
              for key, word in (("bad", "bad"), ("safe", "safe"))]
    decisions = [dict(issue_id=key, decision="accepted", original_span=old, resulting_span=new)
                 for key, old, new in (("bad", "bad", "worse"), ("safe", "safe", "clean"))]
    regressions = [dict(current_persian_quote="worse", source_segment_id="p1:s1")]
    if unlocated:
        regressions.append(dict(current_persian_quote="unlocated issue"))
    from tarjomeh.core.pipeline import _decisions_without_regressed_edits

    retained, excluded = _decisions_without_regressed_edits(
        decisions, regressions, source_text="A complete claim.", previous_text=before,
        candidate_text=proposed, issue_details=issues)
    if unlocated:
        assert retained == excluded == []
    else:
        assert [item["issue_id"] for item in retained] == ["safe"] and excluded == ["bad"]


@pytest.mark.parametrize("stage", ["extra_final_refinement_completed", "exact_final_quality_repair"])
@pytest.mark.parametrize("legacy", [False, True])
def test_audit_reports_symmetric_evidence_without_inventing_historical_proof(stage, legacy):
    from tarjomeh.core.evidence_audit import targeted_quality_evidence

    payload = dict(before_actionable_count=1, after_actionable_count=0,
                   after_actionable_count_raw=2, count_excluded_unchanged_issue_ids=["B1", "B2"])
    if not legacy:
        payload.update(before_actionable_count_raw=3,
                       before_count_excluded_unchanged_issue_ids=["B1", "B2"],
                       before_count_excluded_refiner_veto_issue_ids=[],
                       count_comparison_policy="symmetric_before_after_unchanged_scope")
    original = deepcopy(payload)
    result = targeted_quality_evidence({}, [dict(event_type=stage, chunk_index=4, payload=payload)], [])
    row = result["final_repair_counts"][0]
    assert row["symmetric_count_evidence_available"] is not legacy
    assert row["before_actionable_count_raw"] == (None if legacy else 3)
    assert row["before_count_excluded_unchanged_issue_ids"] == (None if legacy else ["B1", "B2"])
    assert row["count_excluded_unchanged_issue_ids"] == ["B1", "B2"]
    assert payload == original


@pytest.mark.parametrize("segment", ["", "p99:s1", "p0:s1"])
def test_unproved_source_segment_cannot_own_even_a_unique_applied_span(segment):
    from tarjomeh.core.pipeline import _decisions_without_regressed_edits

    retained, excluded = _decisions_without_regressed_edits(
        [dict(issue_id="A", decision="accepted", original_span="bad", resulting_span="worse")],
        [dict(current_persian_quote="worse", source_segment_id=segment)],
        source_text="A complete claim.", previous_text="bad.", candidate_text="worse.",
        issue_details=[dict(issue_id="A", current_persian_quote="bad", source_segment_id=segment)])
    assert retained == excluded == []
