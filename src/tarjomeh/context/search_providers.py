"""Search providers for the Tarjomeh context module.

Implements DuckDuckGo and Google Search providers to retrieve web definitions
for ambiguous academic terms.
"""

from __future__ import annotations

import re
import urllib.parse
import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """A search result returned by a search provider."""
    title: str
    url: str
    snippet: str


class BaseSearchProvider:
    """Abstract base class for web search providers."""

    async def search(self, query: str) -> list[SearchResult]:
        """Perform a web search and return a list of SearchResult objects."""
        raise NotImplementedError


class DuckDuckGoProvider(BaseSearchProvider):
    """DuckDuckGo search provider.

    Does not require API keys or credentials. Parses results from the HTML endpoint.
    """

    async def search(self, query: str) -> list[SearchResult]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        try:
            async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
                response = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    timeout=10.0
                )
                if response.status_code != 200:
                    logger.warning("DuckDuckGo request failed with status code %d", response.status_code)
                    return []

                html_content = response.text
                results = []

                # 1. Try to find result__snippet blocks:
                snippet_matches = re.finditer(
                    r'(?:href="([^"]+)"[^>]*class="result__snippet"|class="result__snippet"[^>]*href="([^"]+)")>([\s\S]*?)</a>',
                    html_content
                )
                for m in snippet_matches:
                    url = m.group(1) or m.group(2)
                    snippet = m.group(3)
                    snippet_clean = re.sub(r'<[^>]+>', '', snippet).strip()
                    if url and snippet_clean:
                        redir_match = re.search(r'uddg=([^&]+)', url)
                        actual_url = urllib.parse.unquote(redir_match.group(1)) if redir_match else url
                        results.append(SearchResult(
                            title="Search Result",
                            url=actual_url,
                            snippet=snippet_clean
                        ))

                # 2. Fallback check for alternate snippet tag placements
                if not results:
                    matches = re.findall(r'href="([^"]+)"[^>]*class="result__snippet"[^>]*>([\s\S]*?)</a>', html_content)
                    if not matches:
                        matches = re.findall(r'class="result__snippet"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>', html_content)
                    for url, snippet in matches:
                        snippet_clean = re.sub(r'<[^>]+>', '', snippet).strip()
                        redir_match = re.search(r'uddg=([^&]+)', url)
                        actual_url = urllib.parse.unquote(redir_match.group(1)) if redir_match else url
                        results.append(SearchResult(
                            title="Search Result",
                            url=actual_url,
                            snippet=snippet_clean
                        ))

                if not results:
                    logger.warning("DuckDuckGo parser was unable to extract any results from HTML response.")

                return results
        except Exception as e:
            logger.warning("DuckDuckGo search failed with exception: %s", e)
            return []


class GoogleSearchProvider(BaseSearchProvider):
    """Google Custom Search API provider.

    Requires Google API Key and Custom Search Engine ID (cx).
    """

    def __init__(self, api_key: str | None = None, cx: str | None = None) -> None:
        self.api_key = api_key
        self.cx = cx

    async def search(self, query: str) -> list[SearchResult]:
        if not self.api_key or not self.cx:
            return []
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://customsearch.googleapis.com/customsearch/v1",
                    params={
                        "key": self.api_key,
                        "cx": self.cx,
                        "q": query
                    },
                    timeout=10.0
                )
                if response.status_code != 200:
                    return []
                data = response.json()
                results = []
                for item in data.get("items", []):
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", "")
                      ))
                return results
        except Exception:
            return []
