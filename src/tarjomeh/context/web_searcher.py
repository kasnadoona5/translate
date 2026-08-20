"""Web context searcher for term disambiguation using web search.

Identifies ambiguous terms in chunks using the LLM, searches the web for their
definitions, caches results, and returns them for prompt injection.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from typing import Any

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.context.search_providers import (
    BaseSearchProvider,
    SearchProviderChain,
    build_search_provider,
    rank_search_results,
)
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.structured_output import parse_structured_output

logger = logging.getLogger(__name__)


class WebContextSearcher:
    """Aphra-inspired web context searcher.

    Analyzes chunks, runs web searches, caches results, and returns definitions.
    """

    def __init__(self, config: TarjomehConfig, llm_client: Any) -> None:
        self.config = config
        self.llm_client = llm_client
        self.cache: dict[str, str] = {}  # term_lower -> definition_str
        self.result_audit: dict[str, list[dict[str, Any]]] = {}
        self._inflight_terms: dict[str, concurrent.futures.Future[str]] = {}
        # One searcher instance is shared by every parallel worker, so the
        # cache and audit dicts need a lock. Never held across an await.
        self._lock = threading.Lock()

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
        with self._lock:
            known_terms = str(list(self.cache.keys()))
        prompt = WEB_CONTEXT_PROMPT.format(text=chunk.text, known_terms=known_terms)

        try:
            if hasattr(self.llm_client, "set_operation"):
                self.llm_client.set_operation("web_context_term_detection")
            response = await self.llm_client.chat(prompt)
            items = parse_structured_output(response, expected=list)
            if not isinstance(items, list):
                return ""

            query_count = 0
            accepted_count = 0
            rejected_count = 0
            issued_queries: set[str] = set()
            diagnostic_start = (
                self.provider.diagnostic_count()
                if isinstance(self.provider, SearchProviderChain) else 0
            )
            for item in items:
                if not isinstance(item, dict):
                    # A single malformed element used to raise AttributeError,
                    # which the broad handler below turned into "skip every
                    # remaining term in this chunk" plus an error stub report.
                    logger.debug("Skipping non-object search candidate: %r", item)
                    continue
                term = str(item.get("term", "") or "").strip()
                search_query = str(item.get("search_query", "") or "").strip()
                
                if not term or not search_query:
                    continue
                
                term_lower = term.lower()
                with self._lock:
                    if not hasattr(self, "_inflight_terms"):
                        self._inflight_terms = {}
                    already_cached = term_lower in self.cache
                    pending = self._inflight_terms.get(term_lower)
                if already_cached:
                    continue
                if pending is not None:
                    await asyncio.wrap_future(pending)
                    continue
                if query_count >= self.config.web_search.max_queries_per_chunk:
                    continue

                with self._lock:
                    # Recheck after parsing work in case another worker won.
                    if term_lower in self.cache:
                        continue
                    pending = self._inflight_terms.get(term_lower)
                    if pending is None:
                        pending = concurrent.futures.Future()
                        self._inflight_terms[term_lower] = pending
                        owns_term = True
                    else:
                        owns_term = False
                if not owns_term:
                    await asyncio.wrap_future(pending)
                    continue

                # Run search query
                query_count += 1
                issued_queries.add(" ".join(search_query.casefold().split()))
                def_str = ""
                try:
                    results = await self.provider.search(search_query)
                    ranked, result_diagnostics = rank_search_results(
                        search_query,
                        results,
                        identity=term,
                    )
                    accepted_count += sum(
                        1 for item in result_diagnostics if item.get("accepted")
                    )
                    rejected_count += sum(
                        1 for item in result_diagnostics if not item.get("accepted")
                    )
                    if ranked:
                        def_str = "\n".join(
                            f"- {r.snippet[:650]} (source: {r.url})"
                            for r in ranked[:3]
                        )
                    with self._lock:
                        self.result_audit[term_lower] = result_diagnostics
                finally:
                    with self._lock:
                        self.cache[term_lower] = def_str
                        self._inflight_terms.pop(term_lower, None)
                        if not pending.done():
                            pending.set_result(def_str)

            diagnostics = (
                self.provider.diagnostics_since(diagnostic_start)
                if isinstance(self.provider, SearchProviderChain) else []
            )
            diagnostics = [
                diagnostic for diagnostic in diagnostics
                if " ".join(
                    str(diagnostic.get("query", "")).casefold().split()
                ) in issued_queries
            ]
            with self._lock:
                relevant_terms = {
                    str(item.get("term", "")).strip().lower()
                    for item in items
                    if isinstance(item, dict)
                }
                relevance_snapshot = {
                    term: list(audit)
                    for term, audit in self.result_audit.items()
                    if term in relevant_terms
                }
            self.last_report = {
                "candidate_count": len(items),
                "new_query_count": query_count,
                "accepted_result_count": accepted_count,
                "rejected_result_count": rejected_count,
                "result_relevance": relevance_snapshot,
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
        with self._lock:
            cache_snapshot = list(self.cache.items())
        for term_lower, definition in cache_snapshot:
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
            context = "\n\n".join(matched_definitions)
            return (
                "### Web Context Definitions (advisory, relevance-filtered):\n\n"
                + context[:6000]
            )
        
        return ""

    def export_state(self) -> dict[str, Any]:
        """Return resume-safe context and provider caches."""
        with self._lock:
            term_cache = dict(self.cache)
            result_audit = {
                key: list(value) for key, value in self.result_audit.items()
            }
        return {
            "term_cache": term_cache,
            "result_audit": result_audit,
            "provider": (
                self.provider.export_state()
                if isinstance(self.provider, SearchProviderChain) else {}
            ),
        }

    def import_state(self, state: dict[str, Any]) -> None:
        """Restore persisted caches without repeating paid searches."""
        with self._lock:
            self.cache = {
                str(key): str(value)
                for key, value in state.get("term_cache", {}).items()
            }
            self.result_audit = {
                str(key): list(value)
                for key, value in state.get("result_audit", {}).items()
                if isinstance(value, list)
            }
        if isinstance(self.provider, SearchProviderChain):
            self.provider.import_state(state.get("provider", {}))
