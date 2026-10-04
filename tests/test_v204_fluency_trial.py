"""Offline experiment boundaries; no real model or production job mutations."""

import json
import sqlite3
import sys
from unittest.mock import AsyncMock

import pytest

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.quality import fluency_trial


def saved_database(path):
    with sqlite3.connect(path) as database:
        database.execute("CREATE TABLE jobs (id TEXT, config TEXT, status TEXT)")
        database.execute(
            "CREATE TABLE chunks (job_id TEXT,chunk_index INTEGER,text TEXT,transla"
            "tion TEXT,metadata TEXT,status TEXT)"
        )
        database.execute(
            "INSERT INTO jobs VALUES (?,?,?)",
            (
                "saved",
                json.dumps({"translation": {"mode": "academic"}}),
                "paused",
            ),
        )
        database.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?)",
            (
                "saved",
                9,
                "Two signals are recorded.",
                (
                    "\u062f\u0648 \u0633\u06cc\u06af\u0646\u0627\u0644 \u062b\u0628\u062a "
                    "\u0645\u06cc\u200c\u0634\u0648\u062f."
                ),
                json.dumps({"structural_roles": ["body"]}),
                "completed",
            ),
        )


def test_dry_run_never_loads_credentials_calls_llm_or_changes_database(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "saved.db"
    saved_database(path)
    before = path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run must not access credentials or an LLM")

    monkeypatch.setattr(TarjomehConfig, "load", forbidden)
    monkeypatch.setattr(fluency_trial, "LLMClient", forbidden)
    out = tmp_path / "evidence"
    monkeypatch.setattr(
        sys, "argv", ["trial", "saved", "--db", str(path), "--out", str(out), "--indices", "9"]
    )
    fluency_trial.main()
    report = json.loads(capsys.readouterr().out)
    assert report["llm_calls"] == report["database_writes"] == report["production_changes"] == 0
    assert report["chunks"][0]["chunk_index"] == 9
    assert path.read_bytes() == before and not out.exists()


def test_active_worker_lease_refuses_even_a_paused_job(tmp_path):
    path = tmp_path / "saved.db"
    saved_database(path)
    with sqlite3.connect(path) as database:
        database.execute("CREATE TABLE job_workers (state TEXT)")
        database.execute("INSERT INTO job_workers VALUES ('active')")
    with pytest.raises(ValueError, match="worker lease"):
        fluency_trial.saved_pairs(path, "saved", set(), 2)


def test_trial_critic_keeps_saved_9router_route_and_budget():
    config = TarjomehConfig()
    config.llm.openrouter.api_base = "http://172.17.0.1:20128/v1"
    config.llm.openrouter.api_keys = ["local-placeholder"]
    config.llm.critic.model = "combo1"
    chosen = fluency_trial.critic_configuration(config)
    assert chosen.llm.openrouter.api_base == config.llm.openrouter.api_base
    assert chosen.llm.openrouter.api_keys == config.llm.openrouter.api_keys
    assert chosen.llm.max_tokens == config.llm.max_tokens
    assert config.llm.critic.model == "combo1"


@pytest.mark.asyncio
async def test_experimental_instruction_is_not_written_into_production_prompts():
    from tarjomeh.core import prompts

    original = prompts.CRITIQUE_PROMPT
    client = AsyncMock()
    wrapper = fluency_trial.TrialClient(client)
    await wrapper.chat("ordinary input")
    client.chat.assert_awaited_once_with("ordinary input" + fluency_trial.TRIAL_GUIDANCE)
    assert original == prompts.CRITIQUE_PROMPT


def test_synthetic_cases_have_no_real_job_id_or_credit():
    cases = fluency_trial.controls()
    assert len(cases) == 3
    assert all(case["chunk_index"].startswith("control:") for case in cases)
    assert [case["kind"] for case in cases].count("known_good") == 2
