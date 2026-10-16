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


def test_general_subject_and_relation_guidance_is_explicit():
    assert "subject, predicate and" in SUMMARY_UPDATE_PROMPT
    assert "conjunction is not opposition" in SUMMARY_UPDATE_PROMPT
    assert "previous summary is fallible context" in SUMMARY_UPDATE_PROMPT
    assert "negation, modality, quantities and temporal scope" in SUMMARY_UPDATE_PROMPT
    assert "narrower supported statement" in SUMMARY_UPDATE_PROMPT
    assert "without simplifying or" in SUMMARY_UPDATE_PROMPT
