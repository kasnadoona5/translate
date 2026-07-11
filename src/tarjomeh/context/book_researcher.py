"""Optional pre-translation research for book-level terminology seeding."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.parsers.base import Document

logger = logging.getLogger(__name__)


@dataclass
class BookResearchResult:
    """A review-only research result; it is never authoritative."""

    book_context: str = ""
    terms: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    search_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    providers_used: list[str] = field(default_factory=list)
    status: str = "completed"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "book_context": self.book_context,
            "terms": self.terms,
            "sources": self.sources,
            "queries": self.queries,
            "search_diagnostics": self.search_diagnostics,
            "providers_used": self.providers_used,
            "status": self.status,
            "error": self.error,
        }


class BookResearcher:
    """Research a parsed book once and produce bounded glossary suggestions."""

    def __init__(self, config: TarjomehConfig, llm_client: Any) -> None:
        self.config = config
        self.llm_client = llm_client
        from tarjomeh.context.search_providers import build_search_provider
        self.provider = build_search_provider(
            config,
            query_budget=config.web_search.phase7_max_queries,
        )

    async def research(self, document: Document) -> BookResearchResult:
        title = (document.title or "Unknown title").strip()
        author = (document.author or "").strip()
        domain = self.config.translation.domain
        queries = [
            f'"{title}" {author}'.strip(),
            f'"{title}" review summary key concepts',
            f'{author} concepts terminology scholarship'.strip(),
            f'"{title}" {domain} terminology',
            f'{title} {author} Persian scholarship ترجمه فارسی'.strip(),
        ]

        sources: list[dict[str, str]] = []
        try:
            excerpt = self._book_excerpt(document)
            await self._search_queries(queries, sources)
            evidence = self._evidence_text(sources)
            response = await self.llm_client.chat(
                self._prompt(
                    title,
                    author,
                    excerpt,
                    evidence,
                    allow_follow_ups=True,
                )
            )
            data = self._parse_json(response)

            remaining = max(
                0,
                self.config.web_search.phase7_max_queries - len(queries),
            )
            follow_ups = self._normalise_follow_ups(
                data.get("follow_up_queries", []),
                existing=queries,
                limit=remaining,
            )
            if follow_ups:
                queries.extend(follow_ups)
                await self._search_queries(follow_ups, sources)
                evidence = self._evidence_text(sources)
                response = await self.llm_client.chat(
                    self._prompt(
                        title,
                        author,
                        excerpt,
                        evidence,
                        allow_follow_ups=False,
                    )
                )
                data = self._parse_json(response)

            terms = self._normalise_terms(data.get("terms", []), sources)
            if not sources:
                status = "degraded"
            elif not terms:
                status = "completed_without_suggestions"
            else:
                status = "completed"
            diagnostics = list(getattr(self.provider, "diagnostics", []))
            providers_used = list(dict.fromkeys(
                item.get("provider", "")
                for item in diagnostics
                if item.get("status") == "success"
            ))
            return BookResearchResult(
                book_context=str(data.get("book_context", "")).strip()[:4000],
                terms=terms,
                sources=sources,
                queries=queries,
                search_diagnostics=diagnostics,
                providers_used=[name for name in providers_used if name],
                status=status,
            )
        except Exception as exc:
            logger.warning("Book research seed pass failed: %s", exc)
            return BookResearchResult(
                sources=sources,
                queries=queries,
                search_diagnostics=list(
                    getattr(self.provider, "diagnostics", [])
                ),
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _search_queries(
        self,
        queries: list[str],
        sources: list[dict[str, str]],
    ) -> None:
        known_urls = {item["url"] for item in sources}
        for query in queries:
            results = await self.provider.search(query)
            for result in results[:self.config.web_search.max_results]:
                if not result.url or result.url in known_urls:
                    continue
                known_urls.add(result.url)
                sources.append({
                    "query": query,
                    "title": result.title[:300],
                    "url": result.url[:1000],
                    "snippet": result.snippet[:700],
                })

    @staticmethod
    def _evidence_text(sources: list[dict[str, str]]) -> str:
        return chr(10).join(
            f"- {item['title']}: {item['snippet']} ({item['url']})"
            for item in sources
        ) or "(No web results were available.)"

    @staticmethod
    def _normalise_follow_ups(
        raw_queries: Any,
        *,
        existing: list[str],
        limit: int,
    ) -> list[str]:
        if not isinstance(raw_queries, list) or limit <= 0:
            return []
        seen = {" ".join(query.casefold().split()) for query in existing}
        follow_ups: list[str] = []
        for value in raw_queries:
            query = " ".join(str(value).split()).strip()[:300]
            key = query.casefold()
            if not query or key in seen:
                continue
            seen.add(key)
            follow_ups.append(query)
            if len(follow_ups) >= limit:
                break
        return follow_ups

    @staticmethod
    def _book_excerpt(document: Document) -> str:
        parts: list[str] = []
        if document.raw_toc:
            parts.append("Table of contents:" + chr(10) + chr(10).join(document.raw_toc[:80]))
        for paragraph in document.all_paragraphs:
            if paragraph.text.strip():
                parts.append(paragraph.text.strip())
            if sum(len(part) for part in parts) >= 6000:
                break
        return (chr(10) * 2).join(parts)[:7000]

    def _normalise_terms(
        self,
        raw_terms: Any,
        sources: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        if not isinstance(raw_terms, list):
            return []
        known_urls = {item["url"] for item in sources}
        terms: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw_terms[:30]:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            key = source.casefold()
            if not source or not target or key in seen:
                continue
            seen.add(key)
            cited = [
                str(url) for url in item.get("source_urls", [])
                if str(url) in known_urls
            ][:5]
            terms.append({
                "source": source,
                "target": target,
                "context": str(item.get("context", "")).strip()[:1000],
                "domain": str(item.get("domain", "")).strip()
                or self.config.translation.domain,
                "sense": str(item.get("sense", "")).strip(),
                "author": str(item.get("author", "")).strip(),
                "reason": str(item.get("reason", "")).strip()[:1200],
                "confidence": str(item.get("confidence", "low")).strip().lower(),
                "source_urls": cited,
                "evidence_type": (
                    "source_supported" if cited else "book_excerpt_inference"
                ),
                "is_auto": True,
                "status": "suggested",
            })
        return terms

    @staticmethod
    def _parse_json(response: str) -> dict[str, Any]:
        cleaned = (response or "").strip()
        fence = chr(96) * 3
        if cleaned.startswith(fence):
            cleaned = chr(10).join(
                line for line in cleaned.splitlines()
                if not line.strip().startswith(fence)
            ).strip()
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Research response must be a JSON object")
        return data

    @staticmethod
    def _prompt(
        title: str,
        author: str,
        excerpt: str,
        evidence: str,
        *,
        allow_follow_ups: bool,
    ) -> str:
        follow_up_instruction = (
            "- follow_up_queries: at most 3 focused searches that resolve "
            "important remaining author-specific or conceptual uncertainty"
            if allow_follow_ups else
            "- follow_up_queries: an empty array"
        )
        return f"""You are preparing review-only terminology research for an
English-to-Persian academic book translation.

Book title: {title}
Author: {author or "(unknown)"}

Book excerpt:
{excerpt}

Web search evidence:
{evidence}

Return one JSON object with:
- book_context: a short factual domain note for the translator
- terms: at most 30 objects with source, target, context, domain, sense,
  author, reason, confidence (low/medium/high), and source_urls
{follow_up_instruction}

Rules:
- Suggestions are not authoritative. Be conservative.
- Never assume a Persian translation exists. State it only when supported by
  a supplied source.
- Cite only URLs present in the evidence.
- Include a term only when it is likely important across the book.
- Research the book, author, concepts, and domain even when no translation
  or prior Persian scholarship exists.
- Output JSON only."""
