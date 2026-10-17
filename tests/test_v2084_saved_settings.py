import pytest

from tarjomeh.core.config import TarjomehConfig


@pytest.mark.parametrize("persisted", [True, False])
def test_saved_policy_isolated_but_live_transport_and_credentials_allowed(monkeypatch, persisted):
    monkeypatch.setenv("TRANSLATOR_MODEL", "new-route")
    monkeypatch.setenv("TRANSLATOR_REASONING", "auto")
    monkeypatch.setenv("TRANSLATOR_MAX_TOKENS", "18000")
    monkeypatch.setenv("CRITIC_ENABLED", "false")
    monkeypatch.setenv("TRANSLATOR_API_KEY", "secret-test-key")
    monkeypatch.setenv("TRANSLATOR_API_BASE", "http://router.invalid/v1")
    monkeypatch.setenv("LLM_READ_TIMEOUT_SECONDS", "321")
    diagnostics = []
    config = TarjomehConfig.from_dict(
        {
            "llm": {
                "model": "saved-route",
                "translation_reasoning": "disabled",
                "max_tokens": 12000,
                "critic": {"enabled": True},
            }
        },
        persisted_job=persisted,
        load_diagnostics=diagnostics,
    )
    assert config.llm.model == ("saved-route" if persisted else "new-route")
    assert config.llm.translation_reasoning == ("disabled" if persisted else "auto")
    assert config.llm.max_tokens == (12000 if persisted else 18000)
    assert config.llm.critic.enabled is persisted
    assert config.llm.openrouter.api_keys == ["secret-test-key"]
    assert config.llm.openrouter.api_base == "http://router.invalid/v1"
    assert config.llm.transport.read_timeout_seconds == 321
    assert config.translation.extra_final_refine_attempts == 0
    assert bool(diagnostics) is persisted
    assert "secret-test-key" not in str(diagnostics)


def test_saved_recovery_budget_not_overridden(monkeypatch):
    monkeypatch.setenv("TRANSLATOR_RECOVERY_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("CRITIC_RECOVERY_MODEL", "other")
    config = TarjomehConfig.from_dict(
        {
            "llm": {
                "provider": "ollama",
                "recovery": {"max_attempts": 2},
                "critic": {"recovery_model": "saved"},
            }
        },
        persisted_job=True,
    )
    assert config.llm.recovery.max_attempts == 2
    assert config.llm.critic.recovery_model == "saved"
