"""Context sub-package — web search for term disambiguation.

Provides :class:`WebContextSearcher` and concrete search providers
(:class:`DuckDuckGoProvider`, :class:`GoogleSearchProvider`).
"""

from __future__ import annotations

from tarjomeh.context.search_providers import (
    BaseSearchProvider,
    DuckDuckGoProvider,
    GoogleSearchProvider,
    SearchResult,
)
from tarjomeh.context.web_searcher import WebContextSearcher

__all__ = [
    "BaseSearchProvider",
    "DuckDuckGoProvider",
    "GoogleSearchProvider",
    "SearchResult",
    "WebContextSearcher",
]
