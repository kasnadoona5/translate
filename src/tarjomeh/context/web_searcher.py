"""Web context searcher for term disambiguation using web search.

Identifies ambiguous terms in chunks using the LLM, searches the web for their
definitions, caches results, and returns them for prompt injection.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.chunking.chunker import Chunk
from tarjomeh.context.search_providers import (
    BaseSearchProvider,
    SearchProviderChain,
    build_search_provider,
)

logger = logging.getLogger(__name__)


class WebContextSearcher:
    """Aphra-inspired web context searcher.

    Analyzes chunks, runs web searches, caches results, and returns definitions.
    """

    def __init__(self, config: TarjomehConfig, llm_client: Any) -> None:
        self.config = config
        self.llm_client = llm_client
        self.cache: dict[str, str] = {}  # term_lower -> definition_str

        phase7_reserve = (
            config.web_search.phase7_max_queries
            if config.translation.enable_book_research else 0
        )
        remaining_budget = max(
            0,
            config.web_search.max_queries_per_book
            - phase7_reserve,
        )
        self.provider: BaseSearchProvider = build_search_provider(
            config,
            query_budget=remaining_budget,
        )
        self.last_report: dict[str, Any] = {}

    async def get_context_for_chunk(self, chunk: Chunk, memory_context_str: str = "") -> str:
        """Analyze chunk, search for ambiguous terms, and return definitions."""
        if not self.config.translation.enable_web_context:
            return ""

        from tarjomeh.core.prompts import WEB_CONTEXT_PROMPT

        # Identify ambiguous terms
        known_terms = str(list(self.cache.keys()))
        prompt = WEB_CONTEXT_PROMPT.format(text=chunk.text, known_terms=known_terms)

        try:
            if hasattr(self.llm_client, "set_operation"):
                self.llm_client.set_operation("web_context_term_detection")
            response = await self.llm_client.chat(prompt)
            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                lines = [ln for ln in lines if not ln.strip().startswith("```")]
                cleaned = "\n".join(lines).strip()

            items = json.loads(cleaned)
            if not isinstance(items, list):
                return ""

            query_count = 0
            diagnostic_start = len(
                self.provider.diagnostics
                if isinstance(self.provider, SearchProviderChain) else []
            )
            for item in items:
                term = item.get("term", "").strip()
                search_query = item.get("search_query", "").strip()
                
                if not term or not search_query:
                    continue
                
                term_lower = term.lower()
                if term_lower in self.cache:
                    continue
                if query_count >= self.config.web_search.max_queries_per_chunk:
                    continue

                # Run search query
                query_count += 1
                results = await self.provider.search(search_query)
                if results:
                    def_str = "\n".join(f"- {r.snippet} (source: {r.url})" for r in results[:3])
                    self.cache[term_lower] = def_str
                else:
                    self.cache[term_lower] = ""

            diagnostics = (
                self.provider.diagnostics[diagnostic_start:]
                if isinstance(self.provider, SearchProviderChain) else []
            )
            self.last_report = {
                "candidate_count": len(items),
                "new_query_count": query_count,
                "diagnostics": diagnostics,
                "budget_used": getattr(self.provider, "queries_used", None),
                "budget_limit": getattr(self.provider, "query_budget", None),
            }

        except Exception as exc:
            logger.warning("Web context term search/extraction failed: %s", exc)
            self.last_report = {"error": f"{type(exc).__name__}: {exc}"}

        # Match cached terms in the current chunk text
        matched_definitions = []
        
        import re
        for term_lower, definition in self.cache.items():
            if not definition:
                continue
            escaped_term = re.escape(term_lower)
            pattern = re.compile(rf"\b{escaped_term}\b", re.IGNORECASE)
            if pattern.search(chunk.text):
                matched_definitions.append(
                    f"Term: {term_lower.title()}\n"
                    f"Definition:\n{definition}"
                )

        if matched_definitions:
            return "### Web Context Definitions (for term disambiguation):\n\n" + "\n\n".join(matched_definitions)
        
        return ""

    def export_state(self) -> dict[str, Any]:
        """Return resume-safe context and provider caches."""
        return {
            "term_cache": dict(self.cache),
            "provider": (
                self.provider.export_state()
                if isinstance(self.provider, SearchProviderChain) else {}
            ),
        }

    def import_state(self, state: dict[str, Any]) -> None:
        """Restore persisted caches without repeating paid searches."""
        self.cache = {
            str(key): str(value)
            for key, value in state.get("term_cache", {}).items()
        }
        if isinstance(self.provider, SearchProviderChain):
            self.provider.import_state(state.get("provider", {}))
