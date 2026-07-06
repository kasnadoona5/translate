"""External framework adapters for Tarjomeh.

Provides optional integration with LangGraph and other orchestration
frameworks via lazy-loaded adapters.
"""

from tarjomeh.adapters.langgraph_adapter import (
    LangGraphTranslationAdapter,
    HAS_LANGGRAPH,
)

__all__ = ["LangGraphTranslationAdapter", "HAS_LANGGRAPH"]
