from __future__ import annotations

from types import SimpleNamespace

from tarjomeh.core.pipeline import (
    _anchor_only_repair_is_valid,
    _merge_readability_evidence,
    _salvage_local_refinement_edits,
    _salvage_regression_details,
)
from tarjomeh.memory.proper_nouns import (
    is_reusable_terminology_mapping,
    is_usable_observed_mapping,
)
from tarjomeh.persian.typography import PersianTypographer
from tarjomeh.quality.critique import TranslationCritique


class _AcceptedIntegrity:
    accepted = True

    def to_dict(self) -> dict[str, object]:
        return {"accepted": True, "blocking_count": 0, "findings": []}


class _RecordingGate:
    def __init__(self) -> None:
        self.candidates: list[str] = []

    def evaluate(self, source: str, candidate: str, **kwargs: object):
        self.candidates.append(candidate)
        return _AcceptedIntegrity()


def test_same_paragraph_salvage_is_evaluated_as_one_coherent_candidate() -> None:
    previous = (
        "\u0627\u06cc\u0646 \u062c\u0645\u0644\u0647 \u0633\u062e\u062a \u0648 "
        "\u0646\u0627\u0631\u0648\u0634\u0646 \u0627\u0633\u062a."
    )
    proposed = (
        "\u0627\u06cc\u0646 \u062c\u0645\u0644\u0647 \u062f\u0634\u0648\u0627\u0631 \u0648 "
        "\u0645\u0628\u0647\u0645 \u0627\u0633\u062a."
    )
    gate = _RecordingGate()

    final, decisions, report = _salvage_local_refinement_edits(
        source="This sentence is difficult and unclear.",
        previous=previous,
        proposed=proposed,
        issue_details=[
            {"issue_id": "a", "current_persian_quote": "\u0633\u062e\u062a"},
            {"issue_id": "b", "current_persian_quote": "\u0646\u0627\u0631\u0648\u0634\u0646"},
        ],
        issue_decisions=[
            {
                "issue_id": "a",
                "decision": "accepted",
                "resulting_span": "\u062f\u0634\u0648\u0627\u0631",
            },
            {"issue_id": "b", "decision": "accepted", "resulting_span": "\u0645\u0628\u0647\u0645"},
        ],
        integrity_gate=gate,  # type: ignore[arg-type]
        protected_terms=[],
        protect_inline_english=False,
        allowed_inline_originals=[],
    )

    assert final == proposed
    assert gate.candidates == [proposed]
    assert report["committed_count"] == 2
    assert all(item["commit_status"] == "committed_local" for item in decisions)


def test_source_critic_can_roll_back_a_serious_salvage_regression() -> None:
    salvage = {
        "attempts": [
            {
                "committed": True,
                "resulting_span": (
                    "\u0645\u06cc\u0627\u0646 \u062f\u0648\u0644\u062a \u062a\u0627 "
                    "\u062f\u0648\u0644\u062a \u0631\u0627"
                ),
            }
        ]
    }
    critique = SimpleNamespace(
        issue_details=[
            {
                "severity": "major",
                "category": "fluency",
                "confidence": 0.95,
                "current_persian_quote": (
                    "\u0645\u06cc\u0627\u0646 \u062f\u0648\u0644\u062a \u062a\u0627 "
                    "\u062f\u0648\u0644\u062a \u0631\u0627"
                ),
            }
        ]
    )

    assert len(_salvage_regression_details(critique, salvage)) == 1


def test_readability_advice_requires_matching_source_grounded_issue() -> None:
    critique = SimpleNamespace(
        issue_details=[
            {
                "issue_id": "mqm-one",
                "category": "fluency",
                "current_persian_quote": (
                    "\u0627\u06cc\u0646 \u0639\u0628\u0627\u0631\u062a "
                    "\u0646\u0627\u0631\u0648\u0634\u0646 \u0627\u0633\u062a"
                ),
            }
        ]
    )
    matched, unmatched = _merge_readability_evidence(
        critique,
        [
            {
                "severity": "major",
                "current_persian_quote": (
                    "\u0639\u0628\u0627\u0631\u062a \u0646\u0627\u0631\u0648\u0634\u0646"
                ),
                "suggested_correction": "\u0639\u0628\u0627\u0631\u062a \u0631\u0648\u0634\u0646",
                "rationale": "The dependency is incomplete.",
            },
            {
                "severity": "minor",
                "current_persian_quote": "\u0646\u0635 \u062f\u06cc\u06af\u0631",
                "suggested_correction": "\u0628\u062f\u06cc\u0644",
                "rationale": "Preference only.",
            },
        ],
    )

    assert matched == 1
    assert len(unmatched) == 1
    assert critique.issue_details[0]["readability_advisory"]["authority"] == (
        "target_only_advisory"
    )


def test_readability_parser_rejects_ungrounded_target_span() -> None:
    result = TranslationCritique._parse_readability_response(
        '{"issues":[{"severity":"major",'
        '"current_persian_quote":"\u0639\u0628\u0627\u0631\u062a \u063a\u0627\u06cc\u0628",'
        '"suggested_correction":"\u0639\u0628\u0627\u0631\u062a \u0631\u0648\u0634\u0646",'
        '"rationale":"Broken grammar."}]}',
        "\u0645\u062a\u0646 \u0645\u0648\u062c\u0648\u062f \u0627\u0633\u062a.",
    )

    assert not result.valid
    assert result.issues == []


def test_low_authority_memory_rejects_surrounding_persian_syntax() -> None:
    assert not is_usable_observed_mapping(
        "Example Research Institute",
        (
            "\u0628\u0647\u200c\u0648\u0633\u06cc\u0644\u0647\u0654 \u0645\u0648\u0633\u0633\u0647 "
            "\u067e\u0698\u0648\u0647\u0634\u06cc \u0646\u0645\u0648\u0646\u0647"
        ),
        "organization",
    )
    assert not is_reusable_terminology_mapping(
        "social theory",
        (
            "\u0646\u0638\u0631\u06cc\u0647 \u0627\u062c\u062a\u0645\u0627\u0639\u06cc "
            "\u0631\u0627 \u0628\u0631\u0631\u0633\u06cc \u0645\u06cc\u200c\u06a9\u0646\u062f"
        ),
    )
    assert is_reusable_terminology_mapping(
        "social theory",
        "\u0646\u0638\u0631\u06cc\u0647 \u0627\u062c\u062a\u0645\u0627\u0639\u06cc",
    )


def test_typography_preserves_authored_marks_and_repairs_only_false_mi() -> None:
    typographer = PersianTypographer()
    text = (
        "\u0645\u06cc\u0627\u0646\u062c\u06cc\u200c\u06af\u0631\u06cc\u0650 "
        "\u0646\u0647\u0627\u062f\u06cc "
        "\u0645\u06cc\u200c\u0634\u0648\u062f. "
        "\u0631.\u06a9. M. J. Smith 1990"
    )

    result = typographer.process(text)

    assert "\u0645\u06cc\u0627\u0646\u062c\u06cc\u200c\u06af\u0631\u06cc\u0650" in result
    assert "\u0645\u06cc\u200c\u0634\u0648\u062f" in result
    assert "\u0631.\u06a9." in result
    assert "M. J. Smith 1990" in result


def test_anchor_repair_cannot_move_or_merge_structural_paragraphs() -> None:
    before = (
        "\u0639\u0646\u0648\u0627\u0646\n\n"
        "\u0646\u0648\u06cc\u0633\u0646\u062f\u0647 \u0646\u0645\u0648\u0646\u0647 "
        "\u0646\u0648\u0634\u062a."
    )
    valid = (
        "\u0639\u0646\u0648\u0627\u0646\n\n"
        "\u0646\u0648\u06cc\u0633\u0646\u062f\u0647 \u0646\u0645\u0648\u0646\u0647 "
        "(Example Author) "
        "\u0646\u0648\u0634\u062a."
    )
    moved = (
        "\u0639\u0646\u0648\u0627\u0646 \n"
        "\u0646\u0648\u06cc\u0633\u0646\u062f\u0647 \u0646\u0645\u0648\u0646\u0647 "
        "(Example Author) "
        "\u0646\u0648\u0634\u062a."
    )
    categories = {"Example Author": "person"}

    assert _anchor_only_repair_is_valid(before, valid, ["Example Author"], categories)
    assert not _anchor_only_repair_is_valid(before, moved, ["Example Author"], categories)
