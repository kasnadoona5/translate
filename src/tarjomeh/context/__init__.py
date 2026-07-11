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
__all__ = [
    "BookResearchResult",
    "BookResearcher",
    "BaseSearchProvider",
    "DuckDuckGoProvider",
    "GoogleSearchProvider",
    "SearchResult",
    "WebContextSearcher",
]


def __getattr__(name: str):
    """Load orchestration classes lazily to avoid core/context import cycles."""
    if name == "WebContextSearcher":
        from tarjomeh.context.web_searcher import WebContextSearcher
        return WebContextSearcher
    if name in ("BookResearchResult", "BookResearcher"):
        from tarjomeh.context.book_researcher import (
            BookResearchResult,
            BookResearcher,
        )
        return {
            "BookResearchResult": BookResearchResult,
            "BookResearcher": BookResearcher,
        }[name]
    raise AttributeError(name)
