import itertools
import json
import time

import pytest

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import LLMClient
from tarjomeh.core.pipeline import _critique_requires_refinement, _merge_readability_evidence
from tarjomeh.quality.critique import CritiqueResult, TranslationCritique

QUOTE = "\u0639\u0628\u0627\u0631\u062a \u0646\u0627\u0642\u0635 \u0627\u0633\u062a"
FIX = "\u0639\u0628\u0627\u0631\u062a \u06a9\u0627\u0645\u0644 \u0627\u0633\u062a"


def test_overlapping_readability_advice_is_order_independent():
    major = dict(
        severity="major",
        current_persian_quote=QUOTE,
        suggested_correction=FIX,
        rationale="Broken grammar and missing predicate.",
    )
    minor = dict(
        severity="minor",
        current_persian_quote=QUOTE[:-4],
        suggested_correction=FIX[:-4],
        rationale="Modifier wording.",
    )
    results = []
    for order in itertools.permutations([major, minor]):
        parsed = TranslationCritique._parse_readability_response(
            json.dumps({"issues": list(order)}),
            QUOTE + ".",
        )
        assert parsed.valid
        detail = dict(
            category="fluency",
            severity="minor",
            confidence=0.5,
            source_quote="A complete claim.",
            source_segment_id="p1:s1",
            current_persian_quote=QUOTE,
        )
        critique = CritiqueResult(average=9, issue_details=[detail])
        _merge_readability_evidence(
            critique, parsed.issues, source_text="A complete claim.", translation=QUOTE + "."
        )
        assert _critique_requires_refinement(critique, 9)
        assert len(critique.issue_details) == 1
        assert len(detail["readability_advisories"]) == 2
        assert detail["severity"] == "minor" and detail["confidence"] == 0.5
        results.append(detail)
    assert results[0] == results[1]


def test_same_span_alternatives_survive_but_invalid_siblings_stay_invalid():
    advice = dict(
        severity="major",
        current_persian_quote=QUOTE,
        suggested_correction=FIX,
        rationale="Broken grammar.",
    )
    weaker = {**advice, "severity": "minor", "rationale": "Word choice."}
    parsed = TranslationCritique._parse_readability_response(
        json.dumps({"issues": [advice, weaker, advice, {}]}),
        QUOTE,
    )
    assert not parsed.valid and parsed.validation_errors
    assert len(parsed.issues) == 2


@pytest.mark.parametrize(
    "usage,consistency,derived",
    [
        (
            {"prompt_tokens": 5663, "completion_tokens": 50000, "total_tokens": 5663},
            "inconsistent",
            55663,
        ),
        ({"prompt_tokens": 4, "completion_tokens": 7, "total_tokens": 11}, "consistent", 11),
        ({"prompt_tokens": 4}, "unknown", None),
        ({}, "unknown", None),
        ({"prompt_tokens": True, "completion_tokens": 7}, "unknown", None),
    ],
)
def test_token_counter_diagnostics_preserve_provider_usage(usage, consistency, derived):
    client = LLMClient.__new__(LLMClient)
    client.config = TarjomehConfig()
    client._prompt_metrics = lambda _: {}
    client.count_tokens = lambda _: 0
    event = client._attempt_event(
        operation="translation",
        attempt=0,
        max_attempts=1,
        payload={},
        started=time.monotonic(),
        success=False,
        finish_reason="length",
        status_code=200,
        usage=usage,
    )
    assert event["token_counter_consistency"] == consistency
    assert event.get("derived_component_token_sum") == derived
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert event[key] == usage.get(key, 0)
    if derived is not None:
        assert event["derived_component_token_sum_is_billing"] is False


def test_equal_environment_policy_produces_no_difference():
    config = TarjomehConfig()
    diagnostics = []
    config._apply_env_overrides(
        persisted_job=True,
        load_diagnostics=diagnostics,
        environment={
            "TRANSLATOR_MODEL": config.llm.model,
            "CRITIC_ENABLED": str(config.llm.critic.enabled),
            "TRANSLATOR_MAX_TOKENS": str(config.llm.max_tokens),
        },
    )
    assert diagnostics == []


def test_different_environment_policy_discloses_names_not_values():
    config = TarjomehConfig()
    original = config.llm.model
    diagnostics = []
    config._apply_env_overrides(
        persisted_job=True,
        load_diagnostics=diagnostics,
        environment={"TRANSLATOR_MODEL": "private-route", "TRANSLATOR_API_KEY": "private-key"},
    )
    assert config.llm.model == original
    assert diagnostics[0]["ignored_environment_variables"] == ["TRANSLATOR_MODEL"]
    assert "private" not in json.dumps(diagnostics)


def test_names_only_diagnostics_are_persisted_once_and_reach_qa(tmp_path, monkeypatch):
    from unittest.mock import patch

    from tarjomeh.core.pipeline import TranslationPipeline
    from tarjomeh.jobs.database import JobDatabase
    from tarjomeh.web.app import create_app

    config = TarjomehConfig()
    config._load_diagnostics = [
        {
            "ignored_environment_variables": ["TRANSLATOR_MODEL"],
            "policy": "persisted_job_settings_preserved",
        }
    ]
    source = tmp_path / "source.txt"
    source.write_text("A claim.", encoding="utf-8")
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("settings", source, config.to_dict())
    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.config, pipeline.db = config, db
    pipeline._record_settings_diagnostics("settings")
    pipeline._record_settings_diagnostics("settings")
    assert len(db.get_chunk_events("settings")) == 1
    monkeypatch.setenv("UI_SECRET_TOKEN", "test")
    with patch("tarjomeh.jobs.database.JobDatabase", return_value=db):
        response = (
            create_app(config)
            .test_client()
            .get(
                "/api/jobs/settings/qa-report",
                headers={"Authorization": "Bearer test"},
            )
        )
    assert response.status_code == 200
    assert "ignored environment names: TRANSLATOR_MODEL" in response.get_data(as_text=True)
