"""Unit tests for Phase 4: Web Context Search Subsystem."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.chunking.chunker import Chunk
from tarjomeh.context.search_providers import DuckDuckGoProvider, GoogleSearchProvider, SearchResult
from tarjomeh.context.web_searcher import WebContextSearcher


class TestSearchProviders(unittest.IsolatedAsyncioTestCase):
    """Test web search providers using mocked HTTP client."""

    @patch("httpx.AsyncClient.get")
    async def test_duckduckgo_provider_success(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = (
            '<html>'
            '<body>'
            '<a href="//duckduckgo.com/l/?kh=-1&uddg=https%3A%2F%2Fexample.com%2Fhegemony" class="result__snippet">definition of hegemony</a>'
            '</body>'
            '</html>'
        )
        mock_get.return_value = mock_response

        provider = DuckDuckGoProvider()
        results = await provider.search("hegemony")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://example.com/hegemony")
        self.assertEqual(results[0].snippet, "definition of hegemony")

    @patch("httpx.AsyncClient.get")
    async def test_google_provider_success(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {
                    "title": "Hegemony Definition",
                    "link": "https://example.com/hegemony",
                    "snippet": "Definition of hegemony in academia"
                }
            ]
        }
        mock_get.return_value = mock_response

        provider = GoogleSearchProvider(api_key="key", cx="cx")
        results = await provider.search("hegemony")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Hegemony Definition")
        self.assertEqual(results[0].url, "https://example.com/hegemony")
        self.assertEqual(results[0].snippet, "Definition of hegemony in academia")


class TestWebContextSearcher(unittest.IsolatedAsyncioTestCase):
    """Test WebContextSearcher orchestration."""

    async def test_web_context_caching_and_formatting(self) -> None:
        config = TarjomehConfig()
        config.translation.enable_web_context = True

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock()
        
        # Mock LLM to return hegemony term requiring search
        mock_llm.chat.return_value = (
            '[{"term": "hegemony", "reason": "multiple meanings", "search_query": "hegemony definition"}]'
        )

        searcher = WebContextSearcher(config, mock_llm)
        
        # Mock search provider
        mock_provider = MagicMock()
        mock_provider.search = AsyncMock()
        mock_provider.search.return_value = [
            SearchResult(title="Result", url="https://example.com", snippet="Hegemony means leadership.")
        ]
        searcher.provider = mock_provider

        chunk = Chunk(index=0, text="The hegemony of the state.", chapter_title="Ch 1", section_title="")
        context_str = await searcher.get_context_for_chunk(chunk)

        # 1. Verify LLM and search were called
        mock_llm.chat.assert_called_once()
        mock_provider.search.assert_called_once_with("hegemony definition")

        # 2. Verify returned context format
        self.assertIn("Hegemony means leadership.", context_str)
        self.assertIn("hegemony", searcher.cache)

        # 3. Call again on a different chunk containing same term -> should use cached value without repeating search!
        mock_llm.chat.reset_mock()
        mock_provider.search.reset_mock()

        chunk2 = Chunk(index=1, text="Another sentence about hegemony.", chapter_title="Ch 1", section_title="")
        context_str2 = await searcher.get_context_for_chunk(chunk2)

        mock_llm.chat.assert_called_once()  # still identifies terms
        mock_provider.search.assert_not_called()  # but uses cache!
        self.assertIn("Hegemony means leadership.", context_str2)


if __name__ == "__main__":
    unittest.main()
