from __future__ import annotations

import queue
import time

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import LLMClient
from tarjomeh.core.pipeline import (
    _salvage_local_refinement_edits,
    _validate_recovery_part,
)
from tarjomeh.quality.integrity import (
    PostEditIntegrityGate,
    unexpected_latin_prose,
)
from tarjomeh.web.app import (
    _progress_queues,
    _remove_progress_queue_if_current,
    _schedule_progress_queue_cleanup,
)


def test_identifier_only_recovery_does_not_require_persian_script() -> None:
    source = "JC11.J47 2015 320.1-dc23 2015013426"

    result = _validate_recovery_part(source, source)

    assert result["valid"] is True
    assert result["target_script_required"] is False
    assert "target_language_missing" not in result["errors"]


def test_prose_recovery_still_requires_persian_script() -> None:
    result = _validate_recovery_part(
        "The state remains a contested social relation.",
        "The state remains a contested social relation.",
    )

    assert result["valid"] is False
    assert result["target_script_required"] is True
    assert "target_language_missing" in result["errors"]


def test_local_salvage_rejects_overlapping_issue_spans() -> None:
    previous = "\u0645\u062a\u0646 \u0627\u0632 \u0646\u0642\u0627\u0637 \u0639\u0632\u06cc\u0645\u062a \u0646\u0638\u0631\u06cc \u0645\u062a\u0639\u062f\u062f \u0622\u063a\u0627\u0632 \u0645\u06cc\u200c\u0634\u0648\u062f."
    proposed = "\u0645\u062a\u0646 \u0627\u0632 \u0645\u0628\u0627\u0646\u06cc \u0646\u0638\u0631\u06cc \u0645\u062a\u0639\u062f\u062f \u0622\u063a\u0627\u0632 \u0645\u06cc\u200c\u0634\u0648\u062f."
    final, decisions, report = _salvage_local_refinement_edits(
        source="The text begins from multiple theoretical starting points.",
        previous=previous,
        proposed=proposed,
        issue_details=[
            {
                "issue_id": "one",
                "current_persian_quote": "\u0627\u0632 \u0646\u0642\u0627\u0637 \u0639\u0632\u06cc\u0645\u062a \u0646\u0638\u0631\u06cc",
            },
            {
                "issue_id": "two",
                "current_persian_quote": "\u0646\u0642\u0627\u0637 \u0639\u0632\u06cc\u0645\u062a \u0646\u0638\u0631\u06cc \u0645\u062a\u0639\u062f\u062f",
            },
        ],
        issue_decisions=[
            {
                "issue_id": "one",
                "decision": "accepted",
                "resulting_span": "\u0627\u0632 \u0645\u0628\u0627\u0646\u06cc \u0646\u0638\u0631\u06cc",
            },
            {
                "issue_id": "two",
                "decision": "accepted",
                "resulting_span": "\u0645\u0628\u0627\u0646\u06cc \u0646\u0638\u0631\u06cc \u0645\u062a\u0639\u062f\u062f",
            },
        ],
        integrity_gate=PostEditIntegrityGate(),
        protected_terms=[],
        protect_inline_english=False,
        allowed_inline_originals=[],
    )

    assert final == previous
    assert report["committed_count"] == 0
    assert {item["commit_reason"] for item in decisions} == {
        "overlapping_local_span"
    }


def test_local_salvage_rejects_new_adjacent_word_repetition() -> None:
    previous = "\u0627\u06cc\u0646 \u0645\u062a\u0646 \u0642\u0627\u0628\u0644 \u0645\u0637\u0627\u0644\u0639\u0647 \u0627\u0633\u062a."
    repeated = "\u0642\u0627\u0628\u0644 \u0642\u0627\u0628\u0644 \u0645\u0637\u0627\u0644\u0639\u0647"
    final, decisions, report = _salvage_local_refinement_edits(
        source="This text is readable.",
        previous=previous,
        proposed=f"\u0627\u06cc\u0646 \u0645\u062a\u0646 {repeated} \u0627\u0633\u062a.",
        issue_details=[{
            "issue_id": "one",
            "current_persian_quote": "\u0642\u0627\u0628\u0644 \u0645\u0637\u0627\u0644\u0639\u0647",
        }],
        issue_decisions=[{
            "issue_id": "one",
            "decision": "accepted",
            "resulting_span": repeated,
        }],
        integrity_gate=PostEditIntegrityGate(),
        protected_terms=[],
        protect_inline_english=False,
        allowed_inline_originals=[],
    )

    assert final == previous
    assert report["committed_count"] == 0
    assert decisions[0]["commit_reason"] == "new_adjacent_word_repetition"


def test_source_attested_apostrophe_phrase_is_not_a_language_leak() -> None:
    source = "The modern state develops its own raison d'etat over time."
    target = "\u062f\u0648\u0644\u062a \u0645\u062f\u0631\u0646 \u0628\u0647\u200c\u062a\u062f\u0631\u06cc\u062c raison d'etat \u062e\u0648\u062f \u0631\u0627 \u067e\u062f\u06cc\u062f \u0645\u06cc\u200c\u0622\u0648\u0631\u062f."

    assert unexpected_latin_prose(source, target) == []


def test_stale_progress_cleanup_cannot_remove_replacement_queue() -> None:
    old_queue = queue.Queue()
    new_queue = queue.Queue()
    _progress_queues["generation-test"] = old_queue
    try:
        _schedule_progress_queue_cleanup(
            "generation-test", old_queue, delay=0.01
        )
        _progress_queues["generation-test"] = new_queue
        time.sleep(0.05)

        assert _progress_queues["generation-test"] is new_queue
        assert _remove_progress_queue_if_current(
            "generation-test", old_queue
        ) is False
    finally:
        _progress_queues.pop("generation-test", None)


def test_successful_content_without_usage_is_not_zero_output() -> None:
    config = TarjomehConfig()
    config.llm.openrouter.api_keys = ["test"]
    client = LLMClient(config)
    try:
        event = client._attempt_event(
            operation="translation",
            attempt=0,
            max_attempts=2,
            payload={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Translate."}],
                "max_tokens": 1000,
                "stream": True,
            },
            started=time.monotonic(),
            success=True,
            finish_reason="stop",
            status_code=200,
            usage={},
            content="\u062a\u0631\u062c\u0645\u0647 \u0645\u0639\u062a\u0628\u0631",
        )
    finally:
        client.close()

    assert event["visible_output_present"] is True
    assert event["visible_completion_tokens"] > 0
    assert event["usage_reported"] is False
    assert event["token_accounting_status"] == "provider_usage_missing"
