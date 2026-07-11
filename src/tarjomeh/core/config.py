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

    @property
    def is_active(self) -> bool:
        return self.enabled or bool(self.model.strip())


@dataclass
class LLMConfig:
    """LLM provider settings."""

    provider: str = "openrouter"
    model: str = "anthropic/claude-sonnet-4-5-20250514"
    temperature: float = 0.3
    max_tokens: int = 8192
    openrouter: LLMOpenRouterConfig = field(default_factory=LLMOpenRouterConfig)
    ollama: LLMOllamaConfig = field(default_factory=LLMOllamaConfig)
    critic: LLMCriticConfig = field(default_factory=LLMCriticConfig)


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
    critique_threshold: float = 7.0
    # Optional one-time research pass before chunk translation. Disabled by
    # default because it adds web searches and one LLM call.
    enable_book_research: bool = False


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
    enable_compliance_check: bool = True
    enable_auto_correction: bool = True


@dataclass
class OutputConfig:
    """Output format settings."""

    format: str = "pdf"
    bilingual_mode: str = "inline"
    # "inline" keeps the historical Persian (English) behaviour unchanged.
    term_notes: str = "inline"


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
    def load(cls, path: str | Path | None = None) -> TarjomehConfig:
        """Load configuration from a file path.

        If path is None, looks for config.toml, and if not found, falls back
        to config.example.toml or returns a default configuration.
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
        return cls.from_toml(path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TarjomehConfig:
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
        if not config.translation.country:
            config.translation.country = "Iran"
        config.validate()
        return config

    @classmethod
    def from_toml(cls, path: str | Path) -> TarjomehConfig:
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
        _populate_dataclass(config.translation, raw.get("translation", {}))
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
            TRANSLATOR_MAX_TOKENS optional output budget (default 8192)

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
        value = env("TRANSLATOR_API_KEY", "").strip()
        if value:
            self.llm.openrouter.api_keys = [value]
        value = env("TRANSLATOR_API_BASE", "").strip()
        if value:
            self.llm.openrouter.api_base = value
        value = env("TRANSLATOR_MAX_TOKENS", "").strip()
        if value.isdigit():
            self.llm.max_tokens = int(value)

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

        # Numeric bounds
        if self.chunking.max_chunk_tokens < 100:
            errors.append("chunking.max_chunk_tokens must be >= 100")

        if self.translation.parallel_workers < 1:
            errors.append("translation.parallel_workers must be >= 1")

        if self.retry.max_retries < 0:
            errors.append("retry.max_retries must be >= 0")

        if self.translation.back_translation_sample_pct < 0 or \
           self.translation.back_translation_sample_pct > 100:
            errors.append("translation.back_translation_sample_pct must be 0-100")

        if errors:
            combined = "\n  • ".join(errors)
            raise ValueError(f"Configuration validation failed:\n  • {combined}")

    # -- Serialisation helpers ----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the entire configuration to a plain dictionary."""
        from dataclasses import asdict
        return asdict(self)


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
