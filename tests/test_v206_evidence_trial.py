"""General controls for truthful evidence and opt-in existing-stage experiments."""

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.evidence_audit import model_identity_counts
from tarjomeh.core.prompts import CRITIQUE_PROMPT, REFINE_PROMPT
from tarjomeh.quality.critique import CritiqueResult, TranslationCritique
from tarjomeh.quality.refiner import RefinementResult, TranslationRefiner
from tarjomeh.quality.trial import (
    adjudication_metrics,
    comparable_served_models,
    experimental_templates,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v206_trial", ROOT / "scripts/replay_tarjomeh_v206_quality.py",
)
trial = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trial)


def test_model_route_never_becomes_returned_identity():
    result = model_identity_counts([
        {"model": "combo-route", "success": False},
        {"model": "combo-route", "success": True, "response_model": "actual-model"},
    ])
    assert result["served_models"] == {"unknown": 1, "actual-model": 1}
    assert result["successful_served_models"] == {"actual-model": 1}
    assert result["requested_models"] == {"combo-route": 2}
    assert result["failure_rate"] == 0.5
    assert model_identity_counts([])["failure_rate"] is None


def test_no_production_prompt_changes_or_new_runtime_reviewer():
    reviewer = TranslationCritique(object())
    refiner = TranslationRefiner(object())
    assert reviewer._prompt_template == CRITIQUE_PROMPT
    assert refiner._prompt_template == REFINE_PROMPT
    assert "quality.trial" not in (ROOT / "src/tarjomeh/core/pipeline.py").read_text("utf-8")
    assert "prompt_template=" not in (ROOT / "src/tarjomeh/core/pipeline.py").read_text("utf-8")


def test_experimental_templates_keep_all_schemas_limits_and_source_first_rules():
    templates = experimental_templates()
    for template in templates.values():
        template.format(source_text="Source", translation="Target", terminology="Terms",
                        review_context="Context", critique="{}")
        assert "GENERAL" not in template  # No unresolved placeholder contract.
    assert "Report at most 8 actionable issues" in templates["critique"]
    assert "original at most 320 characters, result at most 480" in templates["refinement"]
    assert "do not change a" in templates["refinement"]
    schema = templates["critique"].split('Return a JSON object', 1)[1].split('Scoring guide', 1)[0]
    assert schema == (
        CRITIQUE_PROMPT.split('Return a JSON object', 1)[1].split('Scoring guide', 1)[0]
    )
    assert templates["refinement"].split('9. Return ONLY valid JSON', 1)[1] == (
        REFINE_PROMPT.split('9. Return ONLY valid JSON', 1)[1]
    )


def test_success_identity_not_failed_route_controls_comparability():
    def run(model):
        return {"valid": True, "attempts": [
            {"success": False, "model": "route"},
            {"success": True, "response_model": model},
        ]}
    assert comparable_served_models([run("served"), run("served"), run("served")])
    assert not comparable_served_models([run("served"), run("other")])
    assert not comparable_served_models([run("")])
    assert not comparable_served_models([{**run("served"), "valid": False}])


def test_uncertain_unsupported_and_synthetic_never_make_trial_pass():
    true = {"verdict": "true", "real_saved_text": True,
            "missed_by_both_ordinary": True, "missed_by_all_existing_reports": True,
            "exact_quotes_checked": True, "case_id": "one",
            "finding": {"source_quote": "One scope", "current_persian_quote": "one target"}}
    other = {**true, "case_id": "two"}
    assert adjudication_metrics([true, other])["numeric_thresholds_met"]
    assert not adjudication_metrics([true, true])["numeric_thresholds_met"]
    noisy = adjudication_metrics([true, other, {"verdict": "uncertain"}])
    assert not noisy["numeric_thresholds_met"]
    assert not adjudication_metrics([true, {**other, "real_saved_text": False}])[
        "numeric_thresholds_met"
    ]
    assert adjudication_metrics([{**true, "exact_quotes_checked": False}])["precision"] == 0
    assert not adjudication_metrics([])["numeric_thresholds_met"]


def frozen():
    cases = [{"id": str(i), "kind": "real_saved_text", "source": "A source.",
              "target": "A target."} for i in range(16)]
    controls = [{"id": f"good:{i}", "kind": "known_good", "property": "alias scope",
                 "source": "Another source.", "target": "Another target."} for i in range(2)]
    return trial.freeze_cases(cases, controls, {})


def test_freezing_requires_full_controls_and_exact_baseline_hashes():
    cases = frozen()
    assert len(cases) == 18
    assert all(c["context_provenance"] == "reconstructed_empty_not_historical" for c in cases)
    assert not any(c["all_existing_checks_accounted_for"] for c in cases)
    with pytest.raises(ValueError):
        trial.freeze_cases([], [], {})
    with pytest.raises(ValueError, match="exact source/target"):
        trial.freeze_cases(cases[:16], cases[16:], {"0": {"existing_reports": [{}]}})


def test_output_does_not_overwrite_paid_trial(tmp_path):
    target = tmp_path / "trial.json"
    trial.write_json(target, {"first": True})
    with pytest.raises(FileExistsError):
        trial.write_json(target, {"first": False})
    assert json.loads(target.read_text("utf-8")) == {"first": True}


def test_job_read_is_read_only_and_dry_run_needs_no_credentials(tmp_path, monkeypatch):
    database = tmp_path / "jobs.db"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE jobs(id TEXT, config TEXT)")
        db.execute("CREATE TABLE chunks(job_id TEXT, chunk_index INT, text TEXT, "
                   "translation TEXT, status TEXT)")
        db.execute("INSERT INTO jobs VALUES ('job','{}')")
        db.executemany("INSERT INTO chunks VALUES ('job',?,'Source.','Target.','completed')",
                       [(i,) for i in range(16)])
    before = sha256(database.read_bytes().hex())
    monkeypatch.setattr(TarjomehConfig, "load", lambda *a, **kw: pytest.fail("Credentials loaded"))
    raw, cases = trial.read_job(database, "job")
    assert raw == {} and len(cases) == 16
    assert sha256(database.read_bytes().hex()) == before


def test_role_client_preserves_job_router_and_closes_unused_client(monkeypatch):
    config = TarjomehConfig()
    config.llm.openrouter.api_base = "http://localhost:20128/v1"
    config.llm.openrouter.api_keys = ["test-only-key"]
    config.llm.critic.model = "test-critic"
    created = []

    class Client:
        def __init__(self, selected):
            self.config = selected
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(trial, "LLMClient", Client)
    monkeypatch.setattr("tarjomeh.core.pipeline.LLMClient", Client)
    selected = trial.critic_client(config)
    assert selected.config.llm.model == "test-critic"
    assert selected.config.llm.openrouter.api_base == config.llm.openrouter.api_base
    assert selected.config.llm.openrouter.api_keys == config.llm.openrouter.api_keys
    assert created[0].closed and not selected.closed


def test_refinement_cannot_take_stale_context_or_ungrounded_quotes():
    cases = frozen()
    item = {"id": "0", "human_approved": True, "target_sha256": cases[0]["target_sha256"],
            "source_sha256": cases[0]["source_sha256"],
            "context_sha256": cases[0]["context_sha256"], "issue_details": [
                {"issue_id": "one", "source_quote": "source", "current_persian_quote": "target"}
            ]}
    trial.validate_refinement_cases(cases, [item])
    with pytest.raises(ValueError, match="context"):
        trial.validate_refinement_cases(cases, [{**item, "context_sha256": "stale"}])
    with pytest.raises(ValueError, match="exact"):
        trial.validate_refinement_cases(cases, [{**item, "issue_details": [
            {"issue_id": "one", "source_quote": "guessed", "current_persian_quote": "target"}
        ]}])


def test_new_deploy_retains_verified_rollback_and_never_prunes_router():
    source = (ROOT / "scripts/deploy_tarjomeh_v206.sh").read_text("utf-8")
    final = source.split('echo "========== TARJOMEH-ONLY FINAL CLEANUP =========="', 1)[1]
    assert 'docker image rm "$ROLLBACK"' not in final
    assert 'retained for rollback' in final
    assert 'docker system prune' not in source
    assert 'docker image prune' not in source


def test_companion_counts_requests_separately_in_each_operation():
    source = (ROOT / "scripts/audit_tarjomeh_v206_companion.sh").read_text("utf-8")
    assert 'item.get("response_model") or item.get("model")' not in source
    assert 'stats["requested_models"][requested] += 1' in source


def test_saved_job_is_validated_after_loading_incomplete_credential_source(monkeypatch):
    loaded = []
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEYS", raising=False)

    def load(*, validate):
        loaded.append(validate)
        return TarjomehConfig()

    monkeypatch.setattr(TarjomehConfig, "load", load)
    raw = {"llm": {"provider": "openrouter", "openrouter": {
        "api_keys": ["job-test-key"], "api_base": "http://localhost:20128/v1",
    }}}
    config = trial.load_trial_config(raw)
    assert loaded == [False]
    assert config.llm.openrouter.api_base == "http://localhost:20128/v1"
    with pytest.raises(ValueError, match="api_keys"):
        trial.load_trial_config({"llm": {"provider": "openrouter", "openrouter": {
            "api_keys": [],
        }}})


@pytest.mark.asyncio
async def test_full_reviewer_trial_is_54_calls_and_never_self_approves(tmp_path, monkeypatch):
    prompts = []

    class Client:
        def set_attempt_observer(self, observer):
            self.observer = observer

        def set_operation(self, operation):
            self.operation = operation

        def close(self):
            pass

        async def chat(self, prompt):
            prompts.append(prompt)
            self.observer({"success": True, "model": "route", "response_model": "served",
                           "operation": self.operation, "completion_tokens": 20,
                           "api_key": "must-not-export"})
            return json.dumps({"scores": dict.fromkeys(
                ("accuracy", "fluency", "terminology", "register"), 9), "overall": 9,
                "source_coverage": {"checked_source_segment_ids": ["p1:s1"],
                                    "uncovered_source_segment_ids": [], "complete": True},
                "issues": []})

    monkeypatch.setattr(trial, "critic_client", lambda config: Client())
    config = TarjomehConfig()
    config.translation.qa_json_retries = 0
    result = await trial.run_reviews(config, frozen(), tmp_path)
    assert len(prompts) == result["logical_critique_calls"] == 54
    assert result["physical_accounting"]["physical_attempts"] == 54
    assert result["status"] == "AWAITING_HUMAN_ADJUDICATION"
    assert result["database_writes"] == 0 and result["production_guidance"] == "OFF"
    assert "must-not-export" not in json.dumps(result)
    assert all(r["served_models_comparable"] for r in result["records"])
    assert len(list(tmp_path.glob("review-*.json"))) == 54


@pytest.mark.asyncio
async def test_false_advice_veto_trial_uses_identical_issues_and_no_recritique(
    tmp_path, monkeypatch,
):
    calls = []

    class Client:
        def __init__(self, config):
            pass

        def set_attempt_observer(self, observer):
            self.observer = observer

        def set_operation(self, operation):
            pass

        def close(self):
            pass

        async def chat(self, prompt):
            calls.append(prompt)
            self.observer({"success": True, "response_model": "translator"})
            return json.dumps({"translation": "A target.", "decision": "preserved",
                               "rationale": "Source does not support the advice.",
                               "issue_decisions": [{"issue_id": "one", "decision": "rejected",
                                                    "resulting_span": "A target.",
                                                    "rationale": "Unsupported advice."}]})

    monkeypatch.setattr(trial, "LLMClient", Client)
    cases = frozen()
    decisions = [{"id": "0", "human_approved": True,
                  "source_sha256": cases[0]["source_sha256"],
                  "target_sha256": cases[0]["target_sha256"],
                  "context_sha256": cases[0]["context_sha256"],
                  "issue_details": [{"issue_id": "one", "source_quote": "source",
                                     "current_persian_quote": "target",
                                     "suggested_correction": "unsupported",
                                     "rationale": "bad advice"}]}]
    config = TarjomehConfig()
    config.translation.qa_json_retries = 0
    result = await trial.run_refinements(config, cases, decisions, tmp_path)
    assert len(calls) == result["logical_calls"] == 2
    assert all(r["result"]["issue_decisions"][0]["decision"] == "rejected"
               for r in result["records"])
    assert all("candidate_review" not in r for r in result["records"])
    assert result["admission_and_export_trial"] == "NOT_RUN_activation_blocked"


@pytest.mark.asyncio
async def test_changed_trial_candidates_keep_saved_job_threshold(tmp_path, monkeypatch):
    settings = []

    class Client:
        def __init__(self, config):
            pass

        def set_attempt_observer(self, observer):
            pass

        def close(self):
            pass

    class Refiner:
        def __init__(self, *args, **kwargs):
            pass

        async def refine_with_decision(self, *args):
            return RefinementResult("A changed target.", decision="refined")

    class Critic:
        def __init__(self, client, **kwargs):
            settings.append(kwargs)

        async def critique(self, *args):
            return CritiqueResult()

    monkeypatch.setattr(trial, "LLMClient", Client)
    monkeypatch.setattr(trial, "critic_client", lambda config: Client(config))
    monkeypatch.setattr(trial, "TranslationRefiner", Refiner)
    monkeypatch.setattr(trial, "TranslationCritique", Critic)
    cases = frozen()
    decisions = [{"id": "0", "human_approved": True,
                  "source_sha256": cases[0]["source_sha256"],
                  "target_sha256": cases[0]["target_sha256"],
                  "context_sha256": cases[0]["context_sha256"],
                  "issue_details": [{"issue_id": "one", "source_quote": "source",
                                     "current_persian_quote": "target"}]}]
    config = TarjomehConfig()
    config.translation.critique_threshold = 8.5
    config.translation.qa_json_retries = 2
    result = await trial.run_refinements(config, cases, decisions, tmp_path)
    assert result["logical_calls"] == 4
    assert settings == [{"quality_threshold": 8.5, "max_parse_retries": 2}] * 2
    assert all("candidate_review" in r for r in result["records"])
