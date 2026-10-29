import inspect
from copy import deepcopy

from tarjomeh.core.pipeline import (
    TranslationPipeline, _critique_for_event, _critique_requires_refinement,
    _merge_readability_evidence, _post_rollback_review_evidence,
)
from tarjomeh.quality.critique import CritiqueResult


def merged_review():
    source = "They build and explain the institution."
    target = "آنها نهاد را می‌سازز و توضیح می‌دهند."
    detail = dict(issue_id="typo", category="fluency", severity="minor", confidence=0.5,
                  source_segment_id="p1:s1", source_quote="build and explain",
                  current_persian_quote="می‌سازز", suggested_correction="می‌سازند",
                  rationale="Malformed Persian verb spelling.")
    critique = CritiqueResult(accuracy=9, fluency=9, terminology=9, register=9,
                              average=9, issues=["Malformed verb"], issue_details=[detail])
    advice = dict(category="fluency", severity="major", confidence=0.95,
                  current_persian_quote="می‌سازز", suggested_correction="می‌سازند",
                  rationale="Malformed verb orthography in the Persian predicate.")
    return source, target, critique, advice


def test_matched_readability_updates_disposition_not_original_call_record():
    source, target, critique, advice = merged_review()
    assert not _critique_requires_refinement(critique, 9)
    original_call = _critique_for_event(critique, 9, 1, candidate_text=target)
    original_copy = deepcopy(original_call)
    matched, unmatched = _merge_readability_evidence(
        critique, [advice], source_text=source, translation=target,
    )
    assert matched == 1 and unmatched == []
    result = _post_rollback_review_evidence(critique, 9, 1, target, False, [])
    assert result["review_required"]
    assert result["merged_source_review"]["force_refinement"]
    assert result["merged_source_review"]["candidate_target_hash"] == result["candidate_target_hash"]
    assert original_call == original_copy


def test_prior_blocking_disposition_cannot_be_cleared_by_good_merged_review():
    critique = CritiqueResult(accuracy=10, fluency=10, terminology=10, register=10, average=10)
    details = ["atomic_recovery_source_regression"]
    result = _post_rollback_review_evidence(critique, 9, 1, "متن.", True, details)
    assert result["review_required"] and result["review_details"] == details
    assert details == ["atomic_recovery_source_regression"]


def test_unmatched_target_preference_has_no_new_routing_authority():
    source, target, critique, advice = merged_review()
    advice.update(current_persian_quote="عبارت دیگری", rationale="I prefer another word.")
    matched, _ = _merge_readability_evidence(critique, [advice], source_text=source, translation=target)
    assert not matched
    assert not _post_rollback_review_evidence(critique, 9, 1, target, False, [])["review_required"]


def test_pipeline_persists_final_merged_disposition_before_memory_gates():
    source = inspect.getsource(TranslationPipeline._translate_single_chunk)
    merge = source.index("_merge_readability_evidence(\n", source.index('"post_rollback_source_review_unavailable"'))
    final = source.index("merged_disposition = _post_rollback_review_evidence(", merge)
    assert merge < final < source.index('"post_rollback_final_validation",', final)
    assert 'post_validation.update(merged_disposition)' in source[final:]


def test_real_post_rollback_path_records_merged_review_without_an_extra_call():
    from types import SimpleNamespace
    from test_v20812_pipeline_export import run_chunk
    from tarjomeh.quality.refiner import RefinementResult

    source, target, retained_review, advice = merged_review()
    changed = target.replace("می‌سازز", "تباه می‌کنند")
    pipeline, kwargs = run_chunk(source, target)
    pipeline.config.translation.enable_critique = True
    pipeline.config.translation.max_refine_iterations = 2
    pipeline.config.translation.extra_final_refine_attempts = 0
    pipeline._canonicalize_final_translation = lambda job, idx, chunk, text: text
    kwargs["chunk"].metadata = dict(paragraph_indices=[0], structural_roles=["body"])
    kwargs["chunk"].chapter_title = "Institutions"
    initial = deepcopy(retained_review)
    initial.average = initial.fluency = 8
    initial.issue_details[0].update(severity="major", confidence=0.95)
    regressed = CritiqueResult(accuracy=7, fluency=9, terminology=9, register=9, average=8.5,
        issues=["New source meaning error"], issue_details=[dict(
            issue_id="regression", category="accuracy", severity="major", confidence=0.99,
            source_segment_id="p1:s1", source_quote="build", current_persian_quote="تباه می‌کنند",
            suggested_correction="می‌سازند", rationale="The source says build, not destroy.",
        )])
    calls, refinements = [], []

    async def critique(english, persian, **extra):
        calls.append(persian)
        return deepcopy(initial if len(calls) == 1 else regressed if persian == changed else retained_review)

    async def readability(persian):
        return SimpleNamespace(valid=True, issues=[deepcopy(advice)], validation_errors=[])

    async def refine(english, persian, review, **extra):
        refinements.append(persian)
        if len(refinements) > 1:
            return RefinementResult(translation=persian, decision="unchanged", rationale="No further edit.")
        return RefinementResult(translation=changed, decision="revised", rationale="Local correction.",
            issue_decisions=[dict(issue_id="typo", decision="accepted", original_span="می‌سازز",
                                 resulting_span="تباه می‌کنند")])

    kwargs["critique_tool"] = SimpleNamespace(critique=critique, review_persian_readability=readability)
    kwargs["refiner_tool"] = SimpleNamespace(refine_with_decision=refine)
    retained = pipeline._translate_single_chunk(**kwargs)
    events = [(c.args[2], c.args[3]) for c in pipeline.db.log_chunk_event.call_args_list]
    reports = [p for kind, p in events if kind == "post_rollback_final_validation"]
    assert reports, (calls, refinements, [(kind, p) for kind, p in events if kind.startswith("refinement")])
    report = reports[0]
    assert report["review_required"]
    assert report["merged_source_review"]["force_refinement"]
    assert report["readability_review"]["matched_source_grounded_count"] == 1
    assert report["merged_source_review"]["candidate_target_hash"] == report["candidate_target_hash"]
    assert retained == target
    # Initial, rejected candidate, retained rollback, and existing final review.
    assert len(calls) == 4
    assert any(kind == "critique_needs_review" for kind, _ in events)
    from tarjomeh.core.pipeline import _chunk_style_policy
    saved_events = [dict(event_type=kind, payload=payload) for kind, payload in events]
    policy = _chunk_style_policy(SimpleNamespace(get_chunk_events=lambda *a: saved_events),
                                 "synthetic", 0, target)
    assert not policy["approved"]
