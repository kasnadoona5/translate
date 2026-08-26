from __future__ import annotations

from types import SimpleNamespace

from tarjomeh.core.pipeline import (
    _candidate_regression_details,
    _changed_candidate_spans,
    _chunk_style_policy,
    _objective_unmatched_readability_issues,
    _readability_review_eligible,
    audit_translation_language,
)
from tarjomeh.memory.proper_nouns import is_reusable_terminology_mapping
from tarjomeh.persian.typography import PersianTypographer


def _critique(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "valid": True,
        "accuracy": 9.5,
        "fluency": 9.5,
        "terminology": 9.5,
        "register": 9.5,
        "average": 9.5,
        "issue_details": [],
        "issues": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_complete_refiner_candidate_cannot_introduce_grounded_grammar_defect() -> None:
    baseline = _critique()
    candidate = _critique(issue_details=[{
        "issue_id": "new-syntax",
        "category": "fluency",
        "severity": "minor",
        "confidence": 0.82,
        "source_quote": "results based on the heuristic framework developed here",
        "current_persian_quote": (
            "نتایجی که بر چارچوب نظری اکتشافی که در اینجا بسط یافته استوار خواهند بود"
        ),
        "suggested_correction": (
            "نتایجی که بر چارچوب نظری اکتشافی بسط یافته در اینجا استوار خواهند بود"
        ),
        "rationale": "The nested dependency creates broken grammar and word order.",
    }])

    regressions = _candidate_regression_details(
        candidate,
        baseline,
        ["چارچوب نظری اکتشافی که در اینجا بسط یافته استوار خواهند بود"],
    )

    assert [item["issue_id"] for item in regressions] == ["new-syntax"]


def test_existing_baseline_issue_is_not_mislabeled_as_candidate_regression() -> None:
    detail = {
        "issue_id": "existing",
        "category": "fluency",
        "severity": "major",
        "confidence": 0.95,
        "current_persian_quote": "عبارت دشوار موجود",
        "suggested_correction": "عبارت روشن",
        "rationale": "Broken grammar.",
    }
    assert not _candidate_regression_details(
        _critique(issue_details=[detail]),
        _critique(issue_details=[detail]),
        ["عبارت دشوار موجود"],
    )


def test_broad_candidate_rewrite_keeps_late_changes_in_regression_scope() -> None:
    previous = " ".join(f"old-{index}" for index in range(60))
    candidate = " ".join(f"new-{index}" for index in range(60))

    spans = _changed_candidate_spans(previous, candidate)

    assert len(spans) == 1
    assert "new-59" in spans[0]


def test_readability_review_targets_changed_or_final_candidate_only() -> None:
    chunk = SimpleNamespace(metadata={"structural_roles": ["body"]})
    clean = _critique()

    assert _readability_review_eligible(
        chunk, clean, 9.0, candidate_changed=True, final_candidate=False
    )
    assert not _readability_review_eligible(
        chunk, clean, 9.0, candidate_changed=False, final_candidate=False
    )
    assert not _readability_review_eligible(
        chunk, clean, 9.0, candidate_changed=False, final_candidate=True
    )


def test_unmatched_readability_remains_review_only_and_objective() -> None:
    issues = [
        {
            "severity": "major",
            "current_persian_quote": "نیروها — را می سازند",
            "suggested_correction": "نیروها را می سازند",
            "rationale": "The sentence has broken grammar and an incomplete clause.",
        },
        {
            "severity": "minor",
            "current_persian_quote": "صورت نخست",
            "suggested_correction": "صورت دوم",
            "rationale": "Stylistic preference.",
        },
    ]

    assert _objective_unmatched_readability_issues(issues) == [issues[0]]


def test_scholarly_citation_tails_survive_persian_typography() -> None:
    typographer = PersianTypographer()
    text = (
        "(نگاه کنید به Jessop 1990, 2002, 2011, 2015a)؛ "
        "(نک. به D. E. Smith 1990, 92)؛ "
        "Bartelson 1995; Koselleck 1985"
    )

    result = typographer.process(text)

    assert "Jessop 1990, 2002, 2011, 2015a" in result
    assert "D. E. Smith 1990, 92" in result
    assert "Bartelson 1995؛ Koselleck 1985" in result


def test_spaced_tatweel_becomes_dash_without_turning_word_kashida_into_one() -> None:
    result = PersianTypographer().process("دولت ـ جامعه مدنی و دولــت")

    assert "دولت – جامعه مدنی" in result
    assert result.count("–") == 1
    assert result.endswith("دولت")


def test_source_paired_explanatory_dash_loss_is_review_evidence() -> None:
    report = audit_translation_language(
        "They make history – their own and that of others – in institutions.",
        "آنها در نهادها تاریخ خود و دیگران — را می سازند.",
    )

    assert report["unbalanced_explanatory_dash_count"] == 1
    assert report["review_required"]


def test_contextual_source_clause_cannot_become_global_terminology() -> None:
    assert not is_reusable_terminology_mapping(
        "figurational analyses focus on", "تحلیل های فیگوراتیو"
    )
    assert not is_reusable_terminology_mapping(
        "the state is a relation", "دولت رابطه ای اجتماعی"
    )
    assert is_reusable_terminology_mapping(
        "figurational analysis", "تحلیل فیگوراتیو"
    )


class _EventDB:
    def __init__(self, routed: bool) -> None:
        issue = {
            "issue_id": "minor-one",
            "category": "fluency",
            "severity": "minor",
            "confidence": 0.7,
            "source_segment_id": "p1:s1",
        }
        self.events = [
            {"event_type": "chunk_started", "payload": {}},
            {
                "event_type": "critique_completed",
                "payload": {
                    "valid": True,
                    "blocking_issue_count": 0,
                    "issue_details": [issue],
                    "high_confidence_minor_refinement_issue_ids": (
                        ["minor-one"] if routed else []
                    ),
                    "scores": {
                        "accuracy": 9.5,
                        "fluency": 9.5,
                        "terminology": 9.5,
                        "register": 9.5,
                        "average": 9.5,
                    },
                },
            },
        ]

    def get_chunk_events(self, job_id: str, chunk_index: int):
        return self.events


def test_minor_preferences_do_not_starve_style_memory() -> None:
    clean = _chunk_style_policy(_EventDB(False), "job", 0)
    unresolved = _chunk_style_policy(_EventDB(True), "job", 0)

    assert clean["approved"] and clean["excluded_paragraphs"] == []
    assert unresolved["approved"] and unresolved["excluded_paragraphs"] == [0]
