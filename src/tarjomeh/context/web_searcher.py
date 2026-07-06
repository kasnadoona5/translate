"""Web context searcher for term disambiguation using web search.

Identifies ambiguous terms in chunks using the LLM, searches the web for their
definitions, caches results, and returns them for prompt injection.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.chunking.chunker import Chunk
from tarjomeh.context.search_providers import (
    BaseSearchProvider,
    DuckDuckGoProvider,
    GoogleSearchProvider,
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

        # Resolve search provider (Google if api key + cx are present, else DuckDuckGo)
        google_key = os.environ.get("GOOGLE_API_KEY")
        google_cx = os.environ.get("GOOGLE_CX")
        
        if google_key and google_cx:
            logger.info("Web search using GoogleSearchProvider")
            self.provider: BaseSearchProvider = GoogleSearchProvider(google_key, google_cx)
        else:
            logger.info("Web search using DuckDuckGoProvider")
            self.provider = DuckDuckGoProvider()

    async def get_context_for_chunk(self, chunk: Chunk, memory_context_str: str = "") -> str:
        """Analyze chunk, search for ambiguous terms, and return definitions."""
        if not self.config.translation.enable_web_context:
            return ""

        from tarjomeh.core.prompts import WEB_CONTEXT_PROMPT

        # Identify ambiguous terms
        known_terms = str(list(self.cache.keys()))
        prompt = WEB_CONTEXT_PROMPT.format(text=chunk.text, known_terms=known_terms)

        try:
            response = await self.llm_client.chat(prompt)
            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                lines = [ln for ln in lines if not ln.strip().startswith("```")]
                cleaned = "\n".join(lines).strip()

            items = json.loads(cleaned)
            if not isinstance(items, list):
                return ""

            for item in items:
                term = item.get("term", "").strip()
                search_query = item.get("search_query", "").strip()
                
                if not term or not search_query:
                    continue
                
                term_lower = term.lower()
                if term_lower in self.cache:
                    continue

                # Run search query
                results = await self.provider.search(search_query)
                if results:
                    def_str = "\n".join(f"- {r.snippet} (source: {r.url})" for r in results[:3])
                    self.cache[term_lower] = def_str
                else:
                    self.cache[term_lower] = ""

        except Exception as exc:
            logger.warning("Web context term search/extraction failed: %s", exc)

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
