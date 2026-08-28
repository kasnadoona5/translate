from __future__ import annotations

from types import SimpleNamespace

import pytest

from tarjomeh.core.pipeline import (
    _chunk_memory_admission,
    _chunk_style_policy,
    _promote_objective_readability_issues,
    _salvage_local_refinement_edits,
    audit_translation_language,
)
from tarjomeh.core.prompts import PERSIAN_READABILITY_REVIEW_PROMPT
from tarjomeh.quality.integrity import (
    newly_repeated_adjacent_spans,
    repair_source_grounded_language_artifacts,
    source_unjustified_repeated_adjacent_span_artifacts,
)


class _AcceptedIntegrity:
    accepted = True

    def to_dict(self) -> dict[str, object]:
        return {"accepted": True, "blocking_count": 0, "findings": []}


class _AcceptingGate:
    def evaluate(self, source: str, candidate: str, **kwargs: object):
        return _AcceptedIntegrity()


@pytest.mark.parametrize(
    ("previous", "current_span", "resulting_span"),
    [
        (
            "این بررسی دربارهٔ دولت و قدرت دولت است؛ به‌ویژه در بازار جهانی.",
            "است؛ به‌ویژه",
            "قدرت دولت است، به‌ویژه",
        ),
        (
            "این امر موجه است، بی‌آن‌که دلالت داشته باشد این تنها راه است.",
            "دلالت داشته باشد",
            "بی‌آن‌که دلالت داشته باشد که",
        ),
        (
            "آنچه محل بحث است، مکمل‌بودن — و گاه هم‌شکلی — ویژگی‌هاست.",
            "مکمل‌بودن — و گاه",
            "مکمل‌بودن مکمل‌بودن — و گاه",
        ),
    ],
)
def test_local_salvage_rejects_new_adjacent_phrase_repetition(
    previous: str,
    current_span: str,
    resulting_span: str,
) -> None:
    proposed = previous.replace(current_span, resulting_span, 1)

    final, decisions, report = _salvage_local_refinement_edits(
        source="A source paragraph without adjacent phrase repetition.",
        previous=previous,
        proposed=proposed,
        issue_details=[{
            "issue_id": "readability",
            "current_persian_quote": current_span,
        }],
        issue_decisions=[{
            "issue_id": "readability",
            "decision": "accepted",
            "resulting_span": resulting_span,
        }],
        integrity_gate=_AcceptingGate(),  # type: ignore[arg-type]
        protected_terms=[],
        protect_inline_english=False,
        allowed_inline_originals=[],
    )

    assert final == previous
    assert report["committed_count"] == 0
    assert decisions[0]["commit_status"] == "not_committed"
    assert decisions[0]["commit_reason"] == "new_adjacent_phrase_repetition"


def test_local_salvage_cannot_remove_an_existing_sentence_boundary() -> None:
    previous = "این بحث در بازار جهانی قرار دارد. این تمرکز موجه است."
    current_span = "دارد. این تمرکز"
    resulting_span = "دارد این تمرکز"
    proposed = previous.replace(current_span, resulting_span, 1)

    final, decisions, report = _salvage_local_refinement_edits(
        source="The discussion is in the world market. This focus is justified.",
        previous=previous,
        proposed=proposed,
        issue_details=[{
            "issue_id": "punctuation",
            "current_persian_quote": current_span,
        }],
        issue_decisions=[{
            "issue_id": "punctuation",
            "decision": "accepted",
            "resulting_span": resulting_span,
        }],
        integrity_gate=_AcceptingGate(),  # type: ignore[arg-type]
        protected_terms=[],
        protect_inline_english=False,
        allowed_inline_originals=[],
    )

    assert final == previous
    assert report["committed_count"] == 0
    assert decisions[0]["commit_reason"] == "sentence_boundary_removed"


def test_local_salvage_cannot_remove_a_finite_predicate() -> None:
    previous = (
        "این رویکرد، در عوض، بر منطق‌ها و پویایی‌ها تمرکز می‌کند و "
        "فرصت‌ها را بررسی می‌کند."
    )
    current_span = "بر منطق‌ها و پویایی‌ها تمرکز می‌کند و فرصت‌ها"
    resulting_span = "بر منطق‌ها و پویایی‌ها و فرصت‌ها"
    proposed = previous.replace(current_span, resulting_span, 1)

    final, decisions, report = _salvage_local_refinement_edits(
        source="Instead, it focuses on the logics and examines the opportunities.",
        previous=previous,
        proposed=proposed,
        issue_details=[{
            "issue_id": "predicate",
            "current_persian_quote": current_span,
        }],
        issue_decisions=[{
            "issue_id": "predicate",
            "decision": "accepted",
            "resulting_span": resulting_span,
        }],
        integrity_gate=_AcceptingGate(),  # type: ignore[arg-type]
        protected_terms=[],
        protect_inline_english=False,
        allowed_inline_originals=[],
    )

    assert final == previous
    assert report["committed_count"] == 0
    assert decisions[0]["commit_reason"] == "local_predicate_evidence_removed"


def test_relative_repetition_guard_detects_only_new_damage() -> None:
    previous = "قدرت دولت اهمیت دارد و این بحث ادامه می‌یابد."
    damaged = "قدرت دولت قدرت دولت اهمیت دارد و این بحث ادامه می‌یابد."

    findings = newly_repeated_adjacent_spans(previous, damaged)

    assert [item["phrase"] for item in findings] == ["قدرت دولت"]
    assert newly_repeated_adjacent_spans(damaged, damaged) == []


def test_punctuated_lexical_reuse_is_not_boundary_corruption() -> None:
    text = "در چند دههٔ آینده، آیندهٔ دولت دگرگون خواهد شد."

    assert newly_repeated_adjacent_spans("", text) == []


def test_source_aligned_adjacent_repetition_is_not_reported() -> None:
    source = "The state state distinction is quoted exactly here."
    target = "تمایز دولت دولت در اینجا عیناً نقل شده است."

    assert source_unjustified_repeated_adjacent_span_artifacts(
        source, target
    ) == []


def test_local_salvage_may_replace_one_finite_predicate_with_another() -> None:
    previous = "این رویکرد بر روابط اجتماعی تمرکز می‌کند."
    current_span = "تمرکز می‌کند"
    resulting_span = "تأکید می‌ورزد"
    proposed = previous.replace(current_span, resulting_span, 1)

    final, decisions, report = _salvage_local_refinement_edits(
        source="This approach emphasizes social relations.",
        previous=previous,
        proposed=proposed,
        issue_details=[{
            "issue_id": "predicate",
            "current_persian_quote": current_span,
        }],
        issue_decisions=[{
            "issue_id": "predicate",
            "decision": "accepted",
            "resulting_span": resulting_span,
        }],
        integrity_gate=_AcceptingGate(),  # type: ignore[arg-type]
        protected_terms=[],
        protect_inline_english=False,
        allowed_inline_originals=[],
    )

    assert final == proposed
    assert report["committed_count"] == 1
    assert decisions[0]["commit_status"] == "committed_local"


def test_final_language_audit_reports_source_unjustified_phrase_repetition() -> None:
    report = audit_translation_language(
        "State power matters in this argument.",
        "قدرت دولت قدرت دولت در این استدلال اهمیت دارد.",
    )

    assert report["review_required"] is True
    assert report["repeated_adjacent_span_count"] == 1
    assert report["repeated_adjacent_span_artifacts"][0]["phrase"] == "قدرت دولت"


def test_source_conflicting_readability_advice_is_not_promoted() -> None:
    source = (
        "It addresses two issues. First, it studies formation. "
        "Second, it studies change. Third, it goes beyond territoriality."
    )
    translation = (
        "این فصل به دو مسئله می‌پردازد. نخست، شکل‌گیری را بررسی می‌کند. "
        "دوم، تغییر را بررسی می‌کند. سوم، از قلمروگرایی فراتر می‌رود."
    )
    critique = SimpleNamespace(issue_details=[], issues=[])
    suppressed: list[dict[str, object]] = []

    promoted = _promote_objective_readability_issues(
        critique,
        [{
            "severity": "major",
            "current_persian_quote": "دو مسئله",
            "suggested_correction": "سه مسئله",
            "rationale": "Broken grammar: the announcement conflicts with the list.",
        }],
        translation,
        source_text=source,
        suppressed=suppressed,
    )

    assert promoted == []
    assert critique.issue_details == []
    assert suppressed[0]["reason"] == "source_structure_conflict"
    assert "unauthorized_source_correction" in suppressed[0]["classifications"]


def test_source_restoring_readability_advice_remains_refiner_eligible() -> None:
    source = "It addresses two issues. First, formation. Second, change."
    translation = (
        "این فصل به سه مسئله می‌پردازد. نخست، شکل‌گیری. دوم، تغییر."
    )
    critique = SimpleNamespace(issue_details=[], issues=[])
    suppressed: list[dict[str, object]] = []

    promoted = _promote_objective_readability_issues(
        critique,
        [{
            "severity": "major",
            "current_persian_quote": "سه مسئله",
            "suggested_correction": "دو مسئله",
            "rationale": (
                "Broken grammar: the announced quantity conflicts with the "
                "two enumerated items."
            ),
        }],
        translation,
        source_text=source,
        suppressed=suppressed,
    )

    assert len(promoted) == 1
    assert promoted[0]["suggested_correction"] == "دو مسئله"
    assert suppressed == []


def test_source_grounded_structural_reference_is_localized_without_touching_citation() -> None:
    source = "The argument continues elsewhere (see chapter 4; Jessop 1990)."
    target = "این استدلال در جایی دیگر ادامه می‌یابد (see chapter 4؛ Jessop 1990)."

    repaired, report = repair_source_grounded_language_artifacts(source, target)

    assert "(نگاه کنید به فصل ۴؛ Jessop 1990)" in repaired
    assert "Jessop 1990" in repaired
    assert any(
        item["type"] == "source_structural_reference"
        for item in report["repairs"]
    )


def test_structural_reference_outside_source_parenthetical_is_unchanged() -> None:
    source = "Chapter 4 develops this argument in the main prose."
    target = "Chapter 4 این استدلال را بسط می‌دهد."

    repaired, report = repair_source_grounded_language_artifacts(source, target)

    assert repaired == target
    assert not any(
        item["type"] == "source_structural_reference"
        for item in report["repairs"]
    )


def test_structural_reference_repair_is_scoped_to_target_parenthetical() -> None:
    source = (
        "Chapter 4 is a heading reference; the citation follows "
        "(see chapter 4; Jessop 1990)."
    )
    target = (
        "Chapter 4 عنوان فصل است؛ ارجاع در ادامه می‌آید "
        "(see chapter 4؛ Jessop 1990)."
    )

    repaired, _report = repair_source_grounded_language_artifacts(
        source, target
    )

    assert repaired.startswith("Chapter 4 عنوان فصل است")
    assert "(نگاه کنید به فصل ۴؛ Jessop 1990)" in repaired


class _MemoryDB:
    def get_chunk_events(self, _job_id: str, _chunk_index: int):
        return [
            {"event_type": "chunk_started", "payload": {}},
            {
                "event_type": "critique_completed",
                "payload": {
                    "valid": True,
                    "blocking_issue_count": 0,
                    "scores": {
                        "accuracy": 9.5,
                        "fluency": 9.5,
                        "terminology": 9.5,
                        "register": 9.5,
                        "average": 9.5,
                    },
                    "issue_details": [],
                },
            },
            {
                "event_type": "language_quality_review",
                "payload": {
                    "review_reason": "objective_final_language_artifact",
                    "repeated_adjacent_span_count": 1,
                },
            },
        ]


def test_final_phrase_corruption_retains_continuity_but_loses_memory_authority() -> None:
    policy = _chunk_memory_admission(_MemoryDB(), "job", 0, 9.0)
    style = _chunk_style_policy(_MemoryDB(), "job", 0)

    assert policy["continuity_retained"] is True
    assert policy["short_term_trust"] == "advisory_review"
    assert policy["long_term_reliable"] is False
    assert style["approved"] is False


def test_readability_contract_distinguishes_matrix_from_relative_predicate() -> None:
    assert "relative clause" in PERSIAN_READABILITY_REVIEW_PROMPT
    assert "matrix predicate" in PERSIAN_READABILITY_REVIEW_PROMPT
