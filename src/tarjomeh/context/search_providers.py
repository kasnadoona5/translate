"""Search providers for the Tarjomeh context module.

Implements DuckDuckGo and Google Search providers to retrieve web definitions
for ambiguous academic terms.
"""

from __future__ import annotations

import asyncio
import os
import re
import urllib.parse
import logging
import html
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


_SEARCH_STOPWORDS = {
    "about", "academic", "and", "book", "concept", "concepts", "definition",
    "for", "from", "into", "key", "of", "or", "persian", "review", "scholarship",
    "search", "summary", "term", "terminology", "the", "theory", "translation",
    "with",
}
_LOW_AUTHORITY_HOSTS = {
    "amazon.com", "coursehero.com", "facebook.com", "fandom.com", "studocu.com",
    "youtube.com",
}


def _search_tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", value or "")
        if token.casefold() not in _SEARCH_STOPWORDS
    }


def rank_search_results(
    query: str,
    results: list[SearchResult],
    *,
    identity: str = "",
    title: str = "",
    author: str = "",
    strict_identity: bool = True,
) -> tuple[list[SearchResult], list[dict[str, Any]]]:
    """Rank search evidence conservatively and reject clear identity mismatches."""
    query_tokens = _search_tokens(query)
    identity_tokens = _search_tokens(identity)
    title_tokens = _search_tokens(title)
    author_tokens = _search_tokens(author)
    ranked: list[tuple[float, SearchResult]] = []
    diagnostics: list[dict[str, Any]] = []
    for result in results:
        host = urllib.parse.urlparse(result.url).netloc.casefold().removeprefix("www.")
        haystack = " ".join((result.title, result.snippet, result.url)).casefold()
        haystack_tokens = _search_tokens(haystack)
        identity_coverage = (
            len(identity_tokens & haystack_tokens) / len(identity_tokens)
            if identity_tokens else 1.0
        )
        title_coverage = (
            len(title_tokens & haystack_tokens) / len(title_tokens)
            if title_tokens else 1.0
        )
        author_coverage = (
            len(author_tokens & haystack_tokens) / len(author_tokens)
            if author_tokens else 1.0
        )
        required_identity_coverage = (
            (1.0 if len(identity_tokens) <= 3 else 0.7)
            if strict_identity else 0.5
        )
        query_coverage = (
            len(query_tokens & haystack_tokens) / len(query_tokens)
            if query_tokens else 0.0
        )
        authority = 0.0
        if host.endswith((".edu", ".ac.uk", ".edu.au", ".gov")):
            authority += 0.18
        if any(
            marker in host
            for marker in ("cambridge.org", "oup.com", "politybooks.com", "jstor.org")
        ):
            authority += 0.18
        if any(host == value or host.endswith("." + value) for value in _LOW_AUTHORITY_HOSTS):
            authority -= 0.18
        score = 0.55 * identity_coverage + 0.30 * query_coverage + authority
        reasons: list[str] = []
        if strict_identity and not identity_tokens:
            reasons.append("missing_identity_anchor")
        title_requirement = 1.0 if len(title_tokens) <= 3 else 0.65
        title_mismatch = bool(
            title_tokens and title_coverage < title_requirement
        )
        # A full title match is a strong identity anchor even when a publisher
        # result omits the author. Partial title matches require corroboration
        # by at least one distinctive author token.
        author_support_required = bool(
            author_tokens and title_tokens and title_coverage < 0.9
        )
        author_mismatch = bool(
            author_support_required and not (author_tokens & haystack_tokens)
        )
        if title_mismatch:
            reasons.append("title_identity_mismatch")
        if author_mismatch:
            reasons.append("author_identity_mismatch")
        if (
            identity_tokens
            and not title_tokens
            and identity_coverage < required_identity_coverage
        ):
            reasons.append("identity_mismatch")
        if query_tokens and query_coverage < 0.15:
            reasons.append("low_query_relevance")
        accepted = not reasons and score >= 0.32
        diagnostics.append({
            "title": result.title[:300],
            "url": result.url[:1000],
            "score": round(score, 4),
            "identity_coverage": round(identity_coverage, 4),
            "title_coverage": round(title_coverage, 4),
            "author_coverage": round(author_coverage, 4),
            "accepted": accepted,
            "reasons": reasons or (["below_relevance_threshold"] if not accepted else []),
        })
        if accepted:
            ranked.append((score, result))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [result for _score, result in ranked], diagnostics


class BaseSearchProvider:
    """Abstract base class for web search providers."""

    name = "base"

    def __init__(self) -> None:
        self.last_error = ""
        self.last_status: int | None = None

    @property
    def available(self) -> bool:
        return True

    async def search(self, query: str) -> list[SearchResult]:
        """Perform a web search and return a list of SearchResult objects."""
        raise NotImplementedError


class DuckDuckGoProvider(BaseSearchProvider):
    """DuckDuckGo search provider.

    Does not require API keys or credentials. Parses results from the HTML endpoint.
    """

    name = "duckduckgo"

    def __init__(self) -> None:
        super().__init__()

    async def search(self, query: str) -> list[SearchResult]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        try:
            async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
                results = []
                try:
                    response = await client.get(
                        "https://html.duckduckgo.com/html/",
                        params={"q": query},
                        timeout=10.0
                    )
                    self.last_status = response.status_code
                    if response.status_code == 200:
                        results = self._parse_html(response.text)
                    elif response.status_code == 202:
                        self.last_error = "challenge response (HTTP 202)"
                except Exception as e:
                    logger.warning("html.duckduckgo.com search failed: %s", e)

                if not results:
                    logger.info("html.duckduckgo.com returned no results; trying lite.duckduckgo.com fallback for query: %s", query)
                    try:
                        lite_response = await client.get(
                            "https://lite.duckduckgo.com/lite/",
                            params={"q": query},
                            timeout=10.0
                        )
                        self.last_status = lite_response.status_code
                        if lite_response.status_code == 200:
                            results = self._parse_lite(lite_response.text)
                        elif lite_response.status_code == 202:
                            self.last_error = "challenge response (HTTP 202)"
                    except Exception as e:
                        logger.warning("lite.duckduckgo.com search fallback failed: %s", e)

                logger.info("DuckDuckGo query '%s' returned %d results", query, len(results))
                return results
        except Exception as e:
            logger.warning("DuckDuckGo search failed with exception: %s", e)
            return []

    def _parse_html(self, html_content: str) -> list[SearchResult]:
        results = []
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
        return results

    def _parse_lite(self, html_content: str) -> list[SearchResult]:
        results = []
        pattern = re.compile(
            r'<a[^>]*href=["\']([^"\']+)["\'][^>]*class=["\']result-link["\'][^>]*>(.*?)</a>[\s\S]*?class=["\']result-snippet["\'][^>]*>([\s\S]*?)</td>',
            re.IGNORECASE
        )
        for m in pattern.finditer(html_content):
            url = m.group(1)
            if "duckduckgo.com/y.js" in url:
                continue
            title = re.sub(r'<[^>]+>', '', m.group(2)).strip()
            snippet = re.sub(r'<[^>]+>', '', m.group(3)).strip()
            title = html.unescape(title)
            snippet = html.unescape(snippet)
            if url and snippet:
                results.append(SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet
                ))
        return results


class GoogleSearchProvider(BaseSearchProvider):
    """Google Custom Search API provider.

    Requires Google API Key and Custom Search Engine ID (cx).
    """

    name = "google"

    def __init__(self, api_key: str | None = None, cx: str | None = None) -> None:
        super().__init__()
        self.api_key = api_key
        self.cx = cx

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.cx)

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
                self.last_status = response.status_code
                if response.status_code != 200:
                    self.last_error = f"HTTP {response.status_code}"
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
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []


class TavilySearchProvider(BaseSearchProvider):
    """Official Tavily JSON search API provider."""

    name = "tavily"

    def __init__(
        self,
        api_key: str | None,
        *,
        max_results: int = 5,
        timeout: float = 15.0,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.max_results = max_results
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str) -> list[SearchResult]:
        if not self.api_key:
            self.last_error = "TAVILY_API_KEY is not configured"
            return []
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "query": query,
                        "search_depth": "basic",
                        "max_results": self.max_results,
                        "include_answer": False,
                        "include_raw_content": False,
                    },
                )
            self.last_status = response.status_code
            if response.status_code != 200:
                self.last_error = f"HTTP {response.status_code}"
                return []
            data = response.json()
            return [
                SearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    snippet=str(item.get("content", "")),
                )
                for item in data.get("results", [])[:self.max_results]
                if item.get("url") and item.get("content")
            ]
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []


class BraveSearchProvider(BaseSearchProvider):
    """Official Brave Search JSON API provider."""

    name = "brave"

    def __init__(
        self,
        api_key: str | None,
        *,
        max_results: int = 5,
        timeout: float = 15.0,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.max_results = max_results
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str) -> list[SearchResult]:
        if not self.api_key:
            self.last_error = "BRAVE_SEARCH_API_KEY is not configured"
            return []
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": self.api_key,
                    },
                    params={
                        "q": query,
                        "count": self.max_results,
                        "safesearch": "moderate",
                    },
                )
            self.last_status = response.status_code
            if response.status_code != 200:
                self.last_error = f"HTTP {response.status_code}"
                return []
            items = response.json().get("web", {}).get("results", [])
            return [
                SearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    snippet=str(item.get("description", "")),
                )
                for item in items[:self.max_results]
                if item.get("url") and item.get("description")
            ]
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []


class SearchProviderChain(BaseSearchProvider):
    """Try configured providers in order with caching and a hard query budget."""

    name = "chain"

    def __init__(
        self,
        providers: list[BaseSearchProvider],
        *,
        max_retries: int = 2,
        query_budget: int = 50,
    ) -> None:
        super().__init__()
        self.providers = providers
        self.max_retries = max(0, max_retries)
        self.query_budget = max(0, query_budget)
        self.queries_used = 0
        self.cache: dict[str, list[SearchResult]] = {}
        self.diagnostics: list[dict[str, Any]] = []

    async def search(self, query: str) -> list[SearchResult]:
        cache_key = " ".join(query.casefold().split())
        if cache_key in self.cache:
            self.diagnostics.append({
                "query": query,
                "provider": "cache",
                "status": "hit",
                "result_count": len(self.cache[cache_key]),
            })
            return self.cache[cache_key]
        if self.queries_used >= self.query_budget:
            self.last_error = "query budget exhausted"
            self.diagnostics.append({
                "query": query,
                "provider": "none",
                "status": "budget_exhausted",
                "result_count": 0,
            })
            return []

        self.queries_used += 1
        for provider in self.providers:
            if not provider.available:
                self.diagnostics.append({
                    "query": query,
                    "provider": provider.name,
                    "status": "unavailable",
                    "error": provider.last_error or "API key not configured",
                    "result_count": 0,
                })
                continue
            for attempt in range(self.max_retries + 1):
                provider.last_error = ""
                provider.last_status = None
                results = await provider.search(query)
                status = "success" if results else "empty"
                self.diagnostics.append({
                    "query": query,
                    "provider": provider.name,
                    "status": status,
                    "http_status": provider.last_status,
                    "attempt": attempt + 1,
                    "error": provider.last_error,
                    "result_count": len(results),
                })
                if results:
                    self.cache[cache_key] = results
                    return results
                if provider.last_status not in (429, 500, 502, 503, 504):
                    break
                await asyncio.sleep(min(0.5 * (2 ** attempt), 2.0))

        self.cache[cache_key] = []
        return []

    def export_state(self) -> dict[str, Any]:
        return {
            "queries_used": self.queries_used,
            "cache": {
                key: [
                    {"title": r.title, "url": r.url, "snippet": r.snippet}
                    for r in results
                ]
                for key, results in self.cache.items()
            },
        }

    def import_state(self, state: dict[str, Any]) -> None:
        self.queries_used = max(0, int(state.get("queries_used", 0)))
        self.cache = {
            str(key): [
                SearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    snippet=str(item.get("snippet", "")),
                )
                for item in items if isinstance(item, dict)
            ]
            for key, items in state.get("cache", {}).items()
            if isinstance(items, list)
        }


def build_search_provider(
    config: Any,
    *,
    query_budget: int | None = None,
) -> SearchProviderChain:
    """Build a key-aware provider chain without exposing credentials."""
    cfg = config.web_search
    factories = {
        "tavily": lambda: TavilySearchProvider(
            os.environ.get("TAVILY_API_KEY"),
            max_results=cfg.max_results,
            timeout=cfg.timeout_seconds,
        ),
        "brave": lambda: BraveSearchProvider(
            os.environ.get("BRAVE_SEARCH_API_KEY"),
            max_results=cfg.max_results,
            timeout=cfg.timeout_seconds,
        ),
        "google": lambda: GoogleSearchProvider(
            os.environ.get("GOOGLE_API_KEY"),
            os.environ.get("GOOGLE_CX"),
        ),
        "duckduckgo": DuckDuckGoProvider,
    }
    if cfg.provider == "auto":
        preferred = ["tavily", "brave", "google", "duckduckgo"]
    else:
        preferred = [cfg.provider, *cfg.fallback_providers]
    names = list(dict.fromkeys(preferred))
    providers = [factories[name]() for name in names]
    return SearchProviderChain(
        providers,
        max_retries=cfg.max_retries,
        query_budget=(
            cfg.max_queries_per_book
            if query_budget is None else query_budget
        ),
    )
    name = "duckduckgo"

    def __init__(self) -> None:
        super().__init__()
