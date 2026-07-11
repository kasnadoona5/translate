"""Core components of the Tarjomeh translation system.

Provides the foundational building blocks: configuration management, LLM client,
prompt templates, state machine, and translation pipeline orchestrator.
"""

from __future__ import annotations

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import LLMClient
from tarjomeh.core.state_machine import StateMachine

__all__ = [
    "TarjomehConfig",
    "LLMClient",
    "StateMachine",
    "TranslationPipeline",
]


def __getattr__(name: str):
    """Load the pipeline lazily so leaf core modules remain independently importable."""
    if name == "TranslationPipeline":
        from tarjomeh.core.pipeline import TranslationPipeline
        return TranslationPipeline
    raise AttributeError(name)
