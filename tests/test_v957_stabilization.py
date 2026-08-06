from __future__ import annotations

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import LLMClient
from tarjomeh.core.pipeline import (
    _refinement_decisions_with_commit_state,
    _salvage_local_refinement_edits,
)
from tarjomeh.core.term_notes import ensure_inline_proper_noun_originals
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.persian.typography import PersianTypographer
from tarjomeh.quality.integrity import PostEditIntegrityGate


def test_translation_reasoning_modes_are_provider_neutral() -> None:
    config = TarjomehConfig()
    config.llm.openrouter.api_keys = ["test"]
    client = LLMClient(config)
    try:
        _, _, original = client._prepare_request([{"role": "user", "content": "x"}])
        auto = client._initial_payload_for_operation(original, "translation")
        assert auto["reasoning"] == {"exclude": True}

        config.llm.translation_reasoning = "enabled"
        enabled = client._initial_payload_for_operation(original, "translation")
        assert enabled["reasoning"] == {"enabled": True, "exclude": True}

        config.llm.translation_reasoning = "disabled"
        disabled = client._initial_payload_for_operation(original, "translation")
        assert disabled["reasoning"] == {"effort": "none", "exclude": True}
    finally:
        client.close()


def test_helper_first_attempt_preserves_thinking_and_recovery_disables_it() -> None:
    config = TarjomehConfig()
    config.llm.openrouter.api_keys = ["test"]
    client = LLMClient(config)
    try:
        _, _, original = client._prepare_request([{"role": "user", "content": "x"}])
        initial = client._initial_payload_for_operation(
            original, "proper_noun_initial"
        )
        assert initial["reasoning"] == {"exclude": True}

        recovered, _ = client._recovery_payload(
            initial,
            attempt=1,
            max_attempts=2,
            failure_reason="malformed_response",
            operation="proper_noun_initial",
        )
        assert recovered["reasoning"] == {"effort": "none", "exclude": True}
    finally:
        client.close()


def test_local_refinement_salvage_commits_only_gated_translator_decisions() -> None:
    previous = (
        "\u0627\u06cc\u0646 \u062a\u0631\u062c\u0645\u0647 \u062f\u0642\u06cc\u0642 \u0627\u0633\u062a \u0648 \u0633\u0631\u0645\u0627\u06cc\u0647 \u0631\u0627 \u0628\u0647 \u0631\u0648\u0634\u0646\u06cc \u062a\u0648\u0636\u06cc\u062d \u0645\u06cc\u200c\u062f\u0647\u062f."
    )
    replacement = "\u062f\u0631\u0633\u062a"
    proposed = f"\u0627\u06cc\u0646 \u062a\u0631\u062c\u0645\u0647 {replacement} \u0627\u0633\u062a."
    final, decisions, report = _salvage_local_refinement_edits(
        source="This accurate translation clearly explains capital in the academic text.",
        previous=previous,
        proposed=proposed,
        issue_details=[{
            "issue_id": "mqm-1",
            "current_persian_quote": "\u062f\u0642\u06cc\u0642",
        }],
        issue_decisions=[{
            "issue_id": "mqm-1",
            "decision": "accepted",
            "resulting_span": replacement,
        }],
        integrity_gate=PostEditIntegrityGate(),
        protected_terms=["\u0633\u0631\u0645\u0627\u06cc\u0647"],
        protect_inline_english=False,
        allowed_inline_originals=[],
    )
    assert replacement in final
    assert "\u0633\u0631\u0645\u0627\u06cc\u0647" in final
    assert report["committed_count"] == 1
    assert decisions[0]["commit_status"] == "committed_local"


def test_full_candidate_decisions_report_actual_commit_state() -> None:
    decisions = _refinement_decisions_with_commit_state(
        [{
            "issue_id": "mqm-1",
            "decision": "accepted",
            "resulting_span": "\u062f\u0631\u0633\u062a",
        }],
        [{"issue_id": "mqm-1", "current_persian_quote": "\u062f\u0642\u06cc\u0642"}],
        "\u0627\u06cc\u0646 \u0645\u062a\u0646 \u062f\u0631\u0633\u062a \u0627\u0633\u062a.",
        candidate_accepted=True,
    )
    assert decisions[0]["commit_status"] == "committed_full_candidate"
    assert decisions[0]["integrity_status"] == "accepted"


def test_english_only_first_occurrence_is_paired_with_known_persian() -> None:
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(
        index=0,
        source_text="The Intacta RR2 Pro platform was introduced.",
        translated_text=(
            "\u067e\u0644\u062a\u0641\u0631\u0645 \u00abIntacta RR \u06f2 Pro\u00bb \u0645\u0639\u0631\u0641\u06cc \u0634\u062f."
        ),
    )])
    report = ensure_inline_proper_noun_originals(
        document,
        {"Intacta RR2 Pro": "\u0627\u06cc\u0646\u062a\u06a9\u062a\u0627 \u0622\u0631\u0622\u0631 \u06f2 \u067e\u0631\u0648"},
        PersianTypographer(),
        {"Intacta RR2 Pro": "proper_noun"},
        return_report=True,
    )
    assert report["paired_repair_count"] == 1
    assert report["anchored_count"] == 1
    assert "\u0627\u06cc\u0646\u062a\u06a9\u062a\u0627 \u0622\u0631\u0622\u0631 \u06f2 \u067e\u0631\u0648 (Intacta RR2 Pro)" in document.paragraphs[0].translated_text
