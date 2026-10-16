"""Template contract only, not a semantic-quality certification of model output."""

from string import Formatter

from tarjomeh.core.prompts import SUMMARY_UPDATE_PROMPT


def test_existing_summary_operation_keeps_format_and_inputs():
    fields = {field for _, field, _, _ in Formatter().parse(SUMMARY_UPDATE_PROMPT) if field}
    assert fields == {"current_summary", "new_content", "translation"}
    assert "each ≤ 300 words" in SUMMARY_UPDATE_PROMPT
    assert "## English Summary" in SUMMARY_UPDATE_PROMPT
    assert "## خلاصه فارسی" in SUMMARY_UPDATE_PROMPT
    rendered = SUMMARY_UPDATE_PROMPT.format(
        current_summary="Previous context.", new_content="The council and board act.",
        translation="شورا و هیئت کنش دارند.",
    )
    assert "The council and board act." in rendered


def test_failed_guidance_experiment_is_withdrawn_without_changing_source_authority():
    # The agreed live semantic gate failed; this is not a semantic-fix claim.
    assert "subject, predicate and" not in SUMMARY_UPDATE_PROMPT
    assert "conjunction is not opposition" not in SUMMARY_UPDATE_PROMPT
    assert "Treat the English source as factual authority" in SUMMARY_UPDATE_PROMPT
    assert "8. Return plain text under the required headers." in SUMMARY_UPDATE_PROMPT
