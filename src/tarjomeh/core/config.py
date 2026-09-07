"""Configuration management for the Tarjomeh translation system.

Loads settings from a TOML file (using Python 3.11+ ``tomllib``), applies
mode presets (fast / quality / academic), expands ``${ENV_VAR}`` patterns,
and supports CLI override merging.
"""

from __future__ import annotations

import logging
import os
import re
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mode presets
# ---------------------------------------------------------------------------
_MODE_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "chunking.max_chunk_tokens": 3000,
        "translation.enable_critique": False,
        "translation.enable_back_translation": False,
        "translation.enable_web_context": False,
        "translation.back_translation_sample_pct": 0,
        "translation.max_refine_iterations": 0,
        # parallel_workers stays user-configurable in fast mode
    },
    "quality": {
        "chunking.max_chunk_tokens": 1500,
        "translation.enable_critique": True,
        "translation.enable_back_translation": True,
        "translation.enable_web_context": True,
        "translation.back_translation_sample_pct": 5,
        "translation.max_refine_iterations": 1,
        # parallel_workers stays user-configurable in quality mode
    },
    "academic": {
        "chunking.max_chunk_tokens": 1000,
        "translation.enable_critique": True,
        "translation.enable_back_translation": True,
        "translation.enable_web_context": True,
        "translation.back_translation_sample_pct": 20,
        "translation.max_refine_iterations": 2,
        "translation.parallel_workers": 1,  # forced to 1
    },
}

# ---------------------------------------------------------------------------
# Environment variable expansion
# ---------------------------------------------------------------------------
_ENV_PATTERN = re.compile(r"\$\{([^}]+)}")


def _expand_env(value: str) -> str:
    """Replace ``${VAR}`` patterns with the corresponding environment variable."""

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        env_val = os.environ.get(var_name, "")
        if not env_val:
            logger.warning("Environment variable %s is not set", var_name)
        return env_val

    return _ENV_PATTERN.sub(_replace, value)


def _expand_env_recursive(obj: Any) -> Any:
    """Walk a nested dict / list and expand ``${VAR}`` in every string leaf."""
    if isinstance(obj, str):
        return _expand_env(obj)
    if isinstance(obj, dict):
        return {k: _expand_env_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_recursive(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LLMOpenRouterConfig:
    """OpenRouter-compatible endpoint settings (OpenRouter, 9router, …)."""

    api_keys: list[str] = field(default_factory=lambda: [])
    site_url: str = "https://tarjomeh.local"
    app_name: str = "Tarjomeh"
    # Endpoint base URL. Resolution order:
    #   this field  →  OPENROUTER_API_BASE env var  →  https://openrouter.ai/api/v1
    # Point it at a gateway like 9router (e.g. "http://9router:20128/v1") to
    # route requests through it.
    api_base: str = ""
    exclude_reasoning: bool = True


@dataclass
class LLMOllamaConfig:
    """Ollama-specific LLM settings."""

    host: str = "http://localhost:11434"
    model: str = "llama3.1:70b"


@dataclass
class LLMCriticConfig:
    """Optional independent judge model for critique & back-translation QA.

    Using a separate (ideally stronger) model to grade translations removes
    the correlated blind spots of a model judging its own output. Active when
    ``enabled`` is true OR a ``model`` is explicitly set. Unset fields inherit
    from the main ``[llm]`` block.
    """

    enabled: bool = False
    provider: str = ""      # "" → inherit llm.provider
    model: str = ""         # "" → inherit llm.model (critic effectively disabled)
    api_keys: list[str] = field(default_factory=list)  # [] → inherit openrouter keys
    temperature: float = 0.0  # judges should be near-deterministic
    # "" → inherit llm.openrouter.api_base. Set to route the judge through a
    # DIFFERENT endpoint than the translator (e.g. translator via 9router,
    # judge via OpenRouter directly: "https://openrouter.ai/api/v1").
    api_base: str = ""
    recovery_model: str = ""
    recovery_max_attempts: int = 4
    recovery_max_tokens: int = 50000

    @property
    def is_active(self) -> bool:
        return self.enabled or bool(self.model.strip())


@dataclass
class LLMRecoveryConfig:
    """Adaptive output budgeting and bounded recovery settings."""

    enabled: bool = True
    max_attempts: int = 2
    model: str = ""
    reasoning_effort: str = "low"
    max_tokens: int = 24000
    predictive_first_attempt: bool = True
    predictive_min_tokens: int = 50000
    bootstrap_reasoning_tokens: int = 24000
    adaptive_max_tokens: int = 85000
    context_window_tokens: int = 131072
    context_safety_tokens: int = 2048
    history_window: int = 20
    expanded_final_attempt: bool = False
    final_reasoning_effort: str = "none"


@dataclass
class LLMTransportConfig:
    """Portable HTTP/SSE behavior for OpenAI-compatible model endpoints."""

    # Protocol profile, not a model profile. ``auto`` recognizes common
    # endpoints conservatively; an explicit value is useful behind a private
    # reverse proxy whose URL does not identify the gateway.
    profile: str = "auto"
    streaming: bool = True
    connect_timeout_seconds: float = 30.0
    read_timeout_seconds: float = 900.0
    write_timeout_seconds: float = 120.0
    pool_timeout_seconds: float = 30.0
    # A read timeout has an unknown provider outcome and must not fan out into
    # the full quality retry budget. One means one bounded follow-up attempt.
    unknown_outcome_retries: int = 1
    # 9router's token-saver mode can alter the response contract. Tarjomeh
    # already owns retry/budget policy, so disable that gateway feature.
    bypass_9router_token_saver: bool = True


@dataclass
class LLMConfig:
    """LLM provider settings."""

    provider: str = "openrouter"
    model: str = "anthropic/claude-sonnet-4-5-20250514"
    temperature: float = 0.3
    max_tokens: int = 12000
    # Controls translator/recovery requests only. "auto" preserves the
    # endpoint/model default, while explicit values use the portable
    # OpenRouter-compatible reasoning contract.
    translation_reasoning: str = "auto"
    openrouter: LLMOpenRouterConfig = field(default_factory=LLMOpenRouterConfig)
    ollama: LLMOllamaConfig = field(default_factory=LLMOllamaConfig)
    critic: LLMCriticConfig = field(default_factory=LLMCriticConfig)
    recovery: LLMRecoveryConfig = field(default_factory=LLMRecoveryConfig)
    transport: LLMTransportConfig = field(default_factory=LLMTransportConfig)


@dataclass
class TranslationConfig:
    """Translation behaviour settings."""

    source_lang: str = "English"
    target_lang: str = "Persian (Farsi)"
    country: str = "Iran"
    mode: str = "academic"
    style_register: str = "academic"
    domain: str = "political theory, sociology, philosophy"
    enable_critique: bool = True
    enable_back_translation: bool = True
    back_translation_sample_pct: int = 20
    enable_web_context: bool = True
    parallel_workers: int = 1
    max_refine_iterations: int = 2
    critique_threshold: float = 9.0
    enable_integrity_gate: bool = True
    integrity_min_retention_ratio: float = 0.65
    integrity_max_growth_ratio: float = 1.75
    qa_json_retries: int = 1
    # Optional one-time research pass before chunk translation. Disabled by
    # default because it adds web searches and one LLM call.
    enable_book_research: bool = False
    # Optional 1-based parser chapter positions. An empty list translates the
    # entire document. These positions come from the chapter inspection API,
    # not from potentially missing/duplicated printed chapter numbers.
    chapter_selection: list[int] = field(default_factory=list)
    # Intentional review checkpoints. ``stop_after_chapter`` pauses once after
    # that 1-based parser chapter; ``pause_after_each_chapter`` checkpoints at
    # every chapter boundary except the end of the selected document.
    stop_after_chapter: int = 0
    pause_after_each_chapter: bool = False


@dataclass
class WebSearchConfig:
    """External search providers, budgets, and reliability controls."""

    provider: str = "auto"
    fallback_providers: list[str] = field(
        default_factory=lambda: ["brave", "google", "duckduckgo"]
    )
    max_results: int = 5
    timeout_seconds: float = 15.0
    max_retries: int = 2
    phase7_max_queries: int = 8
    max_queries_per_chunk: int = 2
    max_queries_per_book: int = 100


@dataclass
class ChunkingConfig:
    """Text chunking settings."""

    strategy: str = "semantic"
    max_chunk_tokens: int = 1000
    overlap_sentences: int = 2


@dataclass
class GlossaryConfig:
    """Glossary settings."""

    path: str = "glossary/academic_political_theory.csv"
    # Optional extra glossary CSVs. The primary ``path`` is loaded first so
    # existing projects keep their precedence and behaviour.
    paths: list[str] = field(default_factory=list)
    enable_auto_extraction: bool = True
    # Auto-extracted terms remain reviewable suggestions by default. Enable
    # this only when the user wants unapproved discoveries enforced exactly.
    enforce_auto_extracted_terms: bool = False
    enable_compliance_check: bool = True
    enable_auto_correction: bool = True


@dataclass
class OutputConfig:
    """Output format settings."""

    format: str = "pdf"
    bilingual_mode: str = "inline"
    # "inline" keeps the historical Persian (English) behaviour unchanged.
    term_notes: str = "inline"
    # Start each detected source chapter on a new DOCX page. Other exporters
    # ignore this flag and retain their established behavior.
    chapter_page_breaks: bool = True


@dataclass
class PersianConfig:
    """Persian typography settings."""

    convert_numerals: bool = True
    normalize_zwnj: bool = True
    fix_punctuation: bool = True
    # Shield scholarly apparatus (citations, years, page numbers, footnote
    # markers) from numeral/punctuation conversion: (Marx 1973, 408) stays
    # exactly as-is. Essential for academic books.
    scholarly_mode: bool = True
    font_family: str = "Vazirmatn"


@dataclass
class MemoryConfig:
    """Translation memory settings."""

    enable_4layer: bool = True
    summary_update_interval: str = "chapter"
    long_term_retrieval_k: int = 5
    short_term_window: int = 4
    style_min_score: float = 75.0
    style_min_representative_samples: int = 3


@dataclass
class PDFConfig:
    """PDF parsing settings."""

    parser_backend: str = "pymupdf"
    enable_ocr: bool = False


@dataclass
class RetryConfig:
    """Retry / error-handling settings."""

    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True
    max_consecutive_errors: int = 3
    pause_on_sequential_error: bool = True


@dataclass
class NotificationsConfig:
    """Webhook notification settings."""

    webhook_url: str = ""
    notify_on_complete: bool = True
    notify_on_error: bool = True


@dataclass
class ServerConfig:
    """Web-server settings."""

    host: str = "127.0.0.1"
    port: int = 8080


@dataclass
class SystemPromptConfig:
    """System prompt template settings."""

    template: str = ""


# ---------------------------------------------------------------------------
# Credential redaction
# ---------------------------------------------------------------------------

# Configuration paths holding credentials. These are stripped before a config
# is persisted to the job database or returned by the HTTP API; the live
# values always come from the environment instead. Declared at module level
# so it is never mistaken for a dataclass field.
_SECRET_CONFIG_PATHS: tuple[tuple[str, ...], ...] = (
    ("llm", "openrouter", "api_keys"),
    ("llm", "critic", "api_keys"),
)


def redact_config_secrets(data: dict[str, Any]) -> dict[str, Any]:
    """Blank every credential field in a serialised config, in place."""
    for path in _SECRET_CONFIG_PATHS:
        node: Any = data
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict) and node.get(path[-1]):
            node[path[-1]] = []
    return data


# ---------------------------------------------------------------------------
# Main configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TarjomehConfig:
    """Root configuration for the Tarjomeh translation system.

    Typically constructed via :meth:`from_toml`.  After construction the
    caller may further adjust values with :meth:`update_from_overrides`.
    """

    llm: LLMConfig = field(default_factory=LLMConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    glossary: GlossaryConfig = field(default_factory=GlossaryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    persian: PersianConfig = field(default_factory=PersianConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    pdf: PDFConfig = field(default_factory=PDFConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    system_prompt: SystemPromptConfig = field(default_factory=SystemPromptConfig)

    # -- Construction helpers -----------------------------------------------

    @classmethod
    def load(
        cls, path: str | Path | None = None, *, validate: bool = True
    ) -> TarjomehConfig:
        """Load configuration from a file path.

        If path is None, looks for config.toml, and if not found, falls back
        to config.example.toml or returns a default configuration.

        Pass ``validate=False`` when CLI overrides are about to be merged:
        ``--provider ollama`` must be able to satisfy a file-level config
        whose provider requires an API key.
        """
        if path is None:
            path = Path("config.toml")
            if not path.exists():
                path = Path("config.example.toml")
                if not path.exists():
                    logger.warning("No config file found. Using defaults + .env overrides.")
                    config = cls()
                    config._apply_env_overrides()
                    return config
        return cls.from_toml(path, validate=validate)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        credential_source: TarjomehConfig | None = None,
    ) -> TarjomehConfig:
        """Create a TarjomehConfig from a dictionary.

        Processing order (consistent with from_toml):
        1. Expand env variables recursively.
        2. Apply mode preset defaults.
        3. Apply simple flat .env overrides (TRANSLATOR_* / CRITIC_*).
        4. Enforce default country.
        """
        expanded = _expand_env_recursive(data)
        config = cls._from_raw(expanded)
        config._apply_mode_preset(expanded)
        config._apply_env_overrides()
        # Persisted configs carry redacted credentials; refill from the
        # environment before validation rejects the empty key list.
        config._rehydrate_redacted_secrets(credential_source)
        if not config.translation.country:
            config.translation.country = "Iran"
        config.validate()
        return config

    @classmethod
    def from_toml(
        cls, path: str | Path, *, validate: bool = True
    ) -> TarjomehConfig:
        """Load configuration from a TOML file.

        Processing order:
        1. Parse TOML and expand ``${VAR}`` environment variables.
        2. Apply mode preset defaults for any keys not explicitly set by
           the user.
        3. Enforce mode-level constraints (e.g. academic → parallel_workers=1).

        Parameters
        ----------
        path:
            Filesystem path to the ``.toml`` configuration file.

        Returns
        -------
        TarjomehConfig
            Fully resolved configuration instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with path.open("rb") as fh:
            raw: dict[str, Any] = tomllib.load(fh)

        # Expand environment variables in all string values
        raw = _expand_env_recursive(raw)

        # Build the config instance from raw TOML data
        config = cls._from_raw(raw)

        # Apply mode presets — only for keys that the user did NOT set
        config._apply_mode_preset(raw)

        # Simple flat .env overrides (TRANSLATOR_* / CRITIC_*) win over TOML
        config._apply_env_overrides()

        # Ensure country has a default
        if not config.translation.country:
            config.translation.country = "Iran"

        # Callers that are about to merge CLI overrides defer validation;
        # update_from_overrides() validates once the merge is complete.
        if validate:
            config.validate()
        return config

    @classmethod
    def _from_raw(cls, raw: dict[str, Any]) -> TarjomehConfig:
        """Populate a :class:`TarjomehConfig` from parsed TOML data."""
        config = cls()

        _populate_dataclass(config.llm, raw.get("llm", {}))
        _populate_dataclass(config.llm.openrouter, raw.get("llm", {}).get("openrouter", {}))
        _populate_dataclass(config.llm.ollama, raw.get("llm", {}).get("ollama", {}))
        _populate_dataclass(config.llm.critic, raw.get("llm", {}).get("critic", {}))
        _populate_dataclass(config.llm.recovery, raw.get("llm", {}).get("recovery", {}))
        _populate_dataclass(config.llm.transport, raw.get("llm", {}).get("transport", {}))
        _populate_dataclass(config.translation, raw.get("translation", {}))
        _populate_dataclass(config.web_search, raw.get("web_search", {}))
        _populate_dataclass(config.chunking, raw.get("chunking", {}))
        _populate_dataclass(config.glossary, raw.get("glossary", {}))
        _populate_dataclass(config.output, raw.get("output", {}))
        _populate_dataclass(config.persian, raw.get("persian", {}))
        _populate_dataclass(config.memory, raw.get("memory", {}))
        _populate_dataclass(config.pdf, raw.get("pdf", {}))
        _populate_dataclass(config.retry, raw.get("retry", {}))
        _populate_dataclass(config.notifications, raw.get("notifications", {}))
        _populate_dataclass(config.server, raw.get("server", {}))
        _populate_dataclass(config.system_prompt, raw.get("system_prompt", {}))

        return config

    def _apply_env_overrides(self) -> None:
        """Apply simple flat .env overrides so users configure everything
        from ONE file (.env) without touching config.toml.

        Translator (main model):
            TRANSLATOR_API_BASE   endpoint ("" = OpenRouter; 9router URL to route via it)
            TRANSLATOR_API_KEY    API key
            TRANSLATOR_MODEL      model id or 9router combo name
            TRANSLATOR_MAX_TOKENS optional output budget (default 12000)
            TRANSLATOR_REASONING  auto / enabled / disabled
            TRANSLATOR_RECOVERY_MAX_ATTEMPTS bounded whole-request attempts
            TRANSLATOR_RECOVERY_MAX_TOKENS recovery output budget (default 24000)

        Critic / judge (optional second model):
            CRITIC_ENABLED        true / false
            CRITIC_API_BASE       endpoint (may differ from the translator's)
            CRITIC_API_KEY        API key (may differ)
            CRITIC_MODEL          model id or combo name

        Legacy OPENROUTER_API_KEY / OPENROUTER_API_BASE keep working as
        translator fallbacks (handled via ${VAR} expansion and the client's
        endpoint resolution).
        """
        env = os.environ.get

        value = env("TRANSLATOR_MODEL", "").strip()
        if value:
            self.llm.model = value
            # The Ollama request path reads llm.ollama.model directly, so an
            # explicit model choice has to reach it too or it is ignored.
            self.llm.ollama.model = value
        value = env("TRANSLATOR_API_KEY", "").strip()
        if value:
            self.llm.openrouter.api_keys = [value]
        value = env("TRANSLATOR_API_BASE", "").strip()
        if value:
            self.llm.openrouter.api_base = value
        value = env("TRANSLATOR_MAX_TOKENS", "").strip()
        if value.isdigit():
            self.llm.max_tokens = int(value)
        value = env("TRANSLATOR_REASONING", "").strip().lower()
        if value:
            self.llm.translation_reasoning = value

        value = env("TRANSLATOR_RECOVERY_MODEL", "").strip()
        if value:
            self.llm.recovery.model = value
        value = env("TRANSLATOR_RECOVERY_MAX_ATTEMPTS", "").strip()
        if value.isdigit():
            self.llm.recovery.max_attempts = int(value)
        value = env("TRANSLATOR_RECOVERY_MAX_TOKENS", "").strip()
        if value.isdigit():
            self.llm.recovery.max_tokens = int(value)

        value = env("LLM_STREAMING", "").strip().lower()
        if value in ("1", "true", "yes", "on"):
            self.llm.transport.streaming = True
        elif value in ("0", "false", "no", "off"):
            self.llm.transport.streaming = False
        value = env("LLM_TRANSPORT_PROFILE", "").strip().lower()
        if value:
            self.llm.transport.profile = value
        value = env("LLM_READ_TIMEOUT_SECONDS", "").strip()
        if value:
            try:
                self.llm.transport.read_timeout_seconds = float(value)
            except ValueError:
                logger.warning(
                    "Ignoring invalid LLM_READ_TIMEOUT_SECONDS=%r", value
                )

        value = env("CRITIC_ENABLED", "").strip().lower()
        if value in ("1", "true", "yes", "on"):
            self.llm.critic.enabled = True
        elif value in ("0", "false", "no", "off"):
            self.llm.critic.enabled = False
        value = env("CRITIC_MODEL", "").strip()
        if value:
            self.llm.critic.model = value
        value = env("CRITIC_API_KEY", "").strip()
        if value:
            self.llm.critic.api_keys = [value]
        value = env("CRITIC_API_BASE", "").strip()
        if value:
            self.llm.critic.api_base = value
        value = env("CRITIC_RECOVERY_MODEL", "").strip()
        if value:
            self.llm.critic.recovery_model = value
        value = env("CRITIC_RECOVERY_MAX_ATTEMPTS", "").strip()
        if value.isdigit():
            self.llm.critic.recovery_max_attempts = int(value)
        value = env("CRITIC_RECOVERY_MAX_TOKENS", "").strip()
        if value.isdigit():
            self.llm.critic.recovery_max_tokens = int(value)

    def _rehydrate_redacted_secrets(
        self,
        credential_source: TarjomehConfig | None = None,
    ) -> None:
        """Restore credentials that were stripped before persistence.

        Job records store empty ``api_keys``; the live key comes from the
        environment. ``llm.critic.api_keys == []`` already means "inherit
        the OpenRouter keys", so only the primary list needs a fallback
        here - ``CRITIC_API_KEY`` is handled by _apply_env_overrides.
        """
        source = credential_source
        if source is None and Path("config.toml").is_file():
            try:
                with Path("config.toml").open("rb") as handle:
                    raw = _expand_env_recursive(tomllib.load(handle))
                source = type(self)._from_raw(raw)
            except Exception:
                logger.warning(
                    "Could not read live credentials from config.toml.",
                    exc_info=True,
                )

        if not any(self.llm.openrouter.api_keys):
            fallback = (
                os.environ.get("TRANSLATOR_API_KEY", "").strip()
                or os.environ.get("OPENROUTER_API_KEY", "").strip()
            )
            if fallback:
                self.llm.openrouter.api_keys = [fallback]
            elif source is not None and any(source.llm.openrouter.api_keys):
                self.llm.openrouter.api_keys = list(source.llm.openrouter.api_keys)

        if (
            self.llm.critic.is_active
            and not any(self.llm.critic.api_keys)
            and source is not None
            and any(source.llm.critic.api_keys)
        ):
            self.llm.critic.api_keys = list(source.llm.critic.api_keys)

    def _apply_mode_preset(self, raw: dict[str, Any]) -> None:
        """Apply mode-specific defaults for keys not explicitly provided."""
        mode = self.translation.mode
        preset = _MODE_PRESETS.get(mode)
        if preset is None:
            raise ValueError(
                f"Unknown translation mode '{mode}'. "
                f"Choose from: {', '.join(_MODE_PRESETS)}"
            )

        for dotted_key, preset_value in preset.items():
            section_name, field_name = dotted_key.split(".", 1)
            raw_section = raw.get(section_name, {})

            # Only apply the preset if the user did NOT explicitly set this key
            if field_name not in raw_section:
                section_obj = getattr(self, section_name)
                setattr(section_obj, field_name, preset_value)

        # Academic mode *always* forces parallel_workers to 1
        if mode == "academic":
            self.translation.parallel_workers = 1

    # -- CLI override merging -----------------------------------------------

    def update_from_overrides(self, overrides: dict[str, Any]) -> None:
        """Merge CLI overrides into the configuration.

        Parameters
        ----------
        overrides:
            A flat dictionary of dotted keys, e.g.
            ``{"translation.mode": "fast", "llm.temperature": 0.5}``.
            Nested sub-keys like ``llm.openrouter.api_keys`` are supported
            up to two levels.
        """
        # If translation mode is changed in overrides, apply its presets first
        if "translation.mode" in overrides:
            new_mode = overrides["translation.mode"]
            preset = _MODE_PRESETS.get(new_mode)
            if preset:
                for dotted_key, preset_value in preset.items():
                    if dotted_key not in overrides:
                        section_name, field_name = dotted_key.split(".", 1)
                        section_obj = getattr(self, section_name)
                        setattr(section_obj, field_name, preset_value)

        for dotted_key, value in overrides.items():
            parts = dotted_key.split(".")
            if len(parts) == 2:
                section_name, field_name = parts
                section_obj = getattr(self, section_name, None)
                if section_obj is None:
                    logger.warning("Unknown config section: %s", section_name)
                    continue
                if not hasattr(section_obj, field_name):
                    logger.warning("Unknown config field: %s", dotted_key)
                    continue
                setattr(section_obj, field_name, value)
            elif len(parts) == 3:
                # e.g. llm.openrouter.api_keys
                section_name, sub_section, field_name = parts
                section_obj = getattr(self, section_name, None)
                if section_obj is None:
                    logger.warning("Unknown config section: %s", section_name)
                    continue
                sub_obj = getattr(section_obj, sub_section, None)
                if sub_obj is None:
                    logger.warning("Unknown config sub-section: %s.%s", section_name, sub_section)
                    continue
                if not hasattr(sub_obj, field_name):
                    logger.warning("Unknown config field: %s", dotted_key)
                    continue
                setattr(sub_obj, field_name, value)
            else:
                logger.warning("Invalid override key format: %s", dotted_key)

        # Re-enforce mode constraints after overrides
        if self.translation.mode == "academic":
            self.translation.parallel_workers = 1

        # --model must apply to whichever provider is active. llm.ollama.model
        # is a separate field the Ollama request path reads directly, so
        # without this `--provider ollama --model X` silently ignored X.
        if "llm.model" in overrides and "llm.ollama.model" not in overrides:
            self.llm.ollama.model = overrides["llm.model"]

        self.validate()

    # -- Validation ---------------------------------------------------------

    def validate(self) -> None:
        """Validate that required fields are present and values are sane.

        Raises
        ------
        ValueError
            If any required field is missing or has an invalid value.
        """
        errors: list[str] = []

        # LLM provider must be known
        if self.llm.provider not in ("openrouter", "ollama"):
            errors.append(
                f"llm.provider must be 'openrouter' or 'ollama', got '{self.llm.provider}'"
            )

        # Critic provider (when set) must be known
        if self.llm.critic.provider and self.llm.critic.provider not in ("openrouter", "ollama"):
            errors.append(
                f"llm.critic.provider must be 'openrouter' or 'ollama', "
                f"got '{self.llm.critic.provider}'"
            )

        # OpenRouter requires at least one API key
        if self.llm.provider == "openrouter" and not any(self.llm.openrouter.api_keys):
            errors.append(
                "llm.openrouter.api_keys must contain at least one non-empty API key "
                "(check that ${OPENROUTER_API_KEY} is set)"
            )

        # Model must be set
        if not self.llm.model:
            errors.append("llm.model must be specified")

        valid_reasoning_modes = ("auto", "enabled", "disabled")
        if self.llm.translation_reasoning not in valid_reasoning_modes:
            errors.append(
                "llm.translation_reasoning must be one of "
                f"{valid_reasoning_modes}"
            )

        # Mode must be known
        if self.translation.mode not in _MODE_PRESETS:
            errors.append(
                f"translation.mode must be one of {list(_MODE_PRESETS)}, "
                f"got '{self.translation.mode}'"
            )

        # Style register check
        valid_registers = ("academic", "literary", "general")
        if self.translation.style_register not in valid_registers:
            errors.append(
                f"translation.style_register must be one of {valid_registers}, "
                f"got '{self.translation.style_register}'"
            )

        # Chunking strategy
        valid_strategies = ("semantic", "fixed")
        if self.chunking.strategy not in valid_strategies:
            errors.append(
                f"chunking.strategy must be one of {valid_strategies}, "
                f"got '{self.chunking.strategy}'"
            )

        # Output format
        valid_formats = ("pdf", "epub", "docx", "txt", "srt", "markdown")
        if self.output.format not in valid_formats:
            errors.append(
                f"output.format must be one of {valid_formats}, got '{self.output.format}'"
            )

        valid_term_notes = ("inline", "footnote", "endnote", "both")
        if self.output.term_notes not in valid_term_notes:
            errors.append(
                f"output.term_notes must be one of {valid_term_notes}, "
                f"got '{self.output.term_notes}'"
            )

        valid_bilingual_modes = ("inline", "side_by_side", "target_only")
        if self.output.bilingual_mode not in valid_bilingual_modes:
            errors.append(
                f"output.bilingual_mode must be one of {valid_bilingual_modes}, "
                f"got '{self.output.bilingual_mode}'"
            )

        # Numeric bounds
        if self.chunking.max_chunk_tokens < 100:
            errors.append("chunking.max_chunk_tokens must be >= 100")

        if self.translation.parallel_workers < 1:
            errors.append("translation.parallel_workers must be >= 1")

        if not 0 <= self.translation.max_refine_iterations <= 5:
            errors.append("translation.max_refine_iterations must be 0-5")

        if not 1 <= self.translation.critique_threshold <= 10:
            errors.append("translation.critique_threshold must be 1-10")

        if not 0.3 <= self.translation.integrity_min_retention_ratio <= 1.0:
            errors.append("translation.integrity_min_retention_ratio must be 0.3-1.0")

        if not 1.0 <= self.translation.integrity_max_growth_ratio <= 3.0:
            errors.append("translation.integrity_max_growth_ratio must be 1.0-3.0")

        if not 0 <= self.translation.qa_json_retries <= 2:
            errors.append("translation.qa_json_retries must be 0-2")

        if self.retry.max_retries < 0:
            errors.append("retry.max_retries must be >= 0")

        if not 1 <= self.llm.recovery.max_attempts <= 4:
            errors.append("llm.recovery.max_attempts must be 1-4")
        if self.llm.recovery.reasoning_effort not in ("none", "low", "medium"):
            errors.append("llm.recovery.reasoning_effort must be none, low, or medium")
        if self.llm.recovery.max_tokens < 512:
            errors.append("llm.recovery.max_tokens must be >= 512")
        if self.llm.recovery.bootstrap_reasoning_tokens < 0:
            errors.append("llm.recovery.bootstrap_reasoning_tokens must be >= 0")
        if self.llm.recovery.predictive_min_tokens < self.llm.max_tokens:
            errors.append(
                "llm.recovery.predictive_min_tokens must be >= llm.max_tokens"
            )
        if self.llm.recovery.adaptive_max_tokens < self.llm.max_tokens:
            errors.append(
                "llm.recovery.adaptive_max_tokens must be >= llm.max_tokens"
            )
        if (
            self.llm.recovery.predictive_min_tokens
            > self.llm.recovery.adaptive_max_tokens
        ):
            errors.append(
                "llm.recovery.predictive_min_tokens must be <= adaptive_max_tokens"
            )
        if self.llm.recovery.context_safety_tokens < 0:
            errors.append("llm.recovery.context_safety_tokens must be >= 0")
        if self.llm.recovery.context_window_tokens <= (
            self.llm.recovery.context_safety_tokens + self.llm.max_tokens
        ):
            errors.append(
                "llm.recovery.context_window_tokens must leave room for the prompt"
            )
        if not 3 <= self.llm.recovery.history_window <= 100:
            errors.append("llm.recovery.history_window must be 3-100")
        if not 1 <= self.llm.critic.recovery_max_attempts <= 4:
            errors.append("llm.critic.recovery_max_attempts must be 1-4")
        if self.llm.critic.recovery_max_tokens < 512:
            errors.append("llm.critic.recovery_max_tokens must be >= 512")
        for field_name in (
            "connect_timeout_seconds", "read_timeout_seconds",
            "write_timeout_seconds", "pool_timeout_seconds",
        ):
            if getattr(self.llm.transport, field_name) <= 0:
                errors.append(f"llm.transport.{field_name} must be > 0")
        if not 0 <= self.llm.transport.unknown_outcome_retries <= 2:
            errors.append("llm.transport.unknown_outcome_retries must be 0-2")
        valid_transport_profiles = (
            "auto", "9router", "openrouter", "opencode",
            "openai_compatible", "anthropic_compatible",
        )
        if self.llm.transport.profile not in valid_transport_profiles:
            errors.append(
                "llm.transport.profile must be one of "
                f"{valid_transport_profiles}"
            )

        if self.translation.back_translation_sample_pct < 0 or \
           self.translation.back_translation_sample_pct > 100:
            errors.append("translation.back_translation_sample_pct must be 0-100")

        if self.translation.stop_after_chapter < 0:
            errors.append("translation.stop_after_chapter must be >= 0")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in self.translation.chapter_selection
        ):
            errors.append(
                "translation.chapter_selection must contain positive integers"
            )

        valid_search_providers = {
            "auto", "tavily", "brave", "google", "duckduckgo"
        }
        if self.web_search.provider not in valid_search_providers:
            errors.append(
                "web_search.provider must be one of "
                f"{sorted(valid_search_providers)}"
            )
        invalid_fallbacks = set(self.web_search.fallback_providers) - (
            valid_search_providers - {"auto"}
        )
        if invalid_fallbacks:
            errors.append(
                "web_search.fallback_providers contains unknown providers: "
                + ", ".join(sorted(invalid_fallbacks))
            )
        if self.web_search.phase7_max_queries < 5:
            errors.append("web_search.phase7_max_queries must be >= 5")
        if self.web_search.max_results < 1 or self.web_search.max_results > 20:
            errors.append("web_search.max_results must be 1-20")
        if self.web_search.timeout_seconds <= 0:
            errors.append("web_search.timeout_seconds must be > 0")
        if self.web_search.max_retries < 0 or self.web_search.max_retries > 5:
            errors.append("web_search.max_retries must be 0-5")
        if self.web_search.max_queries_per_chunk < 0:
            errors.append("web_search.max_queries_per_chunk must be >= 0")
        if self.web_search.max_queries_per_book < self.web_search.phase7_max_queries:
            errors.append(
                "web_search.max_queries_per_book must be >= phase7_max_queries"
            )

        if errors:
            combined = "\n  • ".join(errors)
            raise ValueError(f"Configuration validation failed:\n  • {combined}")

    # -- Serialisation helpers ----------------------------------------------

    def to_dict(self, *, redact_secrets: bool = False) -> dict[str, Any]:
        """Serialise the entire configuration to a plain dictionary.

        With ``redact_secrets=True`` every credential field is emptied,
        making the result safe to store in the job database and to return
        from the API. :meth:`from_dict` re-supplies credentials from the
        environment.
        """
        from dataclasses import asdict
        data = asdict(self)
        return redact_config_secrets(data) if redact_secrets else data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _populate_dataclass(instance: Any, data: dict[str, Any]) -> None:
    """Set attributes on *instance* from *data*, ignoring unknown keys and
    nested dicts (which are handled separately)."""
    known_fields = {f.name for f in fields(instance)}
    for key, value in data.items():
        if key in known_fields and not isinstance(value, dict):
            setattr(instance, key, value)
