from copy import deepcopy

from tarjomeh.core.pipeline import _rebind_canonical_critique


def test_quote_rebinding_uses_full_actual_diff_and_preserves_history_and_suggestion():
    before = "شمار 12 مورد است."
    after = "شمار ۱۲ مورد است."
    payload = {
        "issue_details": [
            {
                "issue_id": "a",
                "source_segment_id": "p1:s1",
                "source_quote": "12 items",
                "current_persian_quote": "12 مورد",
                "suggested_correction": "12 پیشنهاد",
            }
        ]
    }
    original = deepcopy(payload)
    rebound = _rebind_canonical_critique(payload, "There are 12 items.", before, after)
    detail = rebound["issue_details"][0]
    assert payload == original and detail["current_persian_quote"] == "۱۲ مورد"
    assert detail["suggested_correction"] == "12 پیشنهاد"
    assert detail["canonical_quote_binding"]["original_quote"] == "12 مورد"
    assert detail["canonical_quote_binding"]["status"] == "rebound"


def test_protected_year_quote_stays_unchanged():
    text = "این اثر در 1973 منتشر شد."
    payload = {"issue_details": [{"source_segment_id": "p1:s1", "current_persian_quote": "1973"}]}
    result = _rebind_canonical_critique(payload, "Published in 1973.", text, text)
    assert result["issue_details"][0]["current_persian_quote"] == "1973"


def test_ambiguous_or_unlocated_quote_is_reported_not_invented():
    payload = {"issue_details": [{"source_segment_id": "p1:s1", "current_persian_quote": "12"}]}
    result = _rebind_canonical_critique(payload, "12 and 12.", "12 و 12.", "۱۲ و ۱۲.")
    assert result["canonical_quote_unmatched_issue_ids"]
    assert (
        result["issue_details"][0]["canonical_quote_binding"]["status"]
        == "quote_unmatched_after_canonicalization"
    )


def test_lexical_change_cannot_be_rebound():
    payload = {"issue_details": [{"source_segment_id": "p1:s1", "current_persian_quote": "یک"}]}
    result = _rebind_canonical_critique(payload, "One item.", "یک مورد.", "دو مورد.")
    assert result["canonical_quote_unmatched_issue_ids"]


async def test_unmatched_canonical_quote_cannot_trigger_a_refiner_call():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tarjomeh.quality.critique import CritiqueResult
    from tarjomeh.quality.refiner import TranslationRefiner

    llm = SimpleNamespace(chat=AsyncMock())
    refiner = TranslationRefiner(llm_client=llm)
    issue = {
        "issue_id": "a",
        "canonical_quote_binding": {"status": "quote_unmatched_after_canonicalization"},
    }
    result = await refiner.refine_with_decision(
        "A source.", "متن.", CritiqueResult(issues=["unresolved"], issue_details=[issue])
    )
    assert result.translation == "متن." and result.decision == "preserved"
    llm.chat.assert_not_awaited()
