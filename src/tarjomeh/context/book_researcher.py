"""Optional pre-translation research for book-level terminology seeding."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from tarjomeh.context.web_searcher import WebContextSearcher
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
    status: str = "completed"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "book_context": self.book_context,
            "terms": self.terms,
            "sources": self.sources,
            "queries": self.queries,
            "status": self.status,
            "error": self.error,
        }


class BookResearcher:
    """Research a parsed book once and produce bounded glossary suggestions."""

    def __init__(self, config: TarjomehConfig, llm_client: Any) -> None:
        self.config = config
        self.llm_client = llm_client
        self.provider = WebContextSearcher(config, llm_client).provider

    async def research(self, document: Document) -> BookResearchResult:
        title = (document.title or "Unknown title").strip()
        author = (document.author or "").strip()
        identity = " ".join(part for part in (title, author) if part).strip()
        queries = [
            f'"{identity}" Persian translation',
            f'"{identity}" key concepts terminology',
            f'"{identity}" Persian academic scholarship',
        ]

        sources: list[dict[str, str]] = []
        try:
            for query in queries:
                results = await self.provider.search(query)
                for result in results[:3]:
                    sources.append({
                        "query": query,
                        "title": result.title[:300],
                        "url": result.url[:1000],
                        "snippet": result.snippet[:700],
                    })

            excerpt = self._book_excerpt(document)
            evidence = chr(10).join(
                f"- {item['title']}: {item['snippet']} ({item['url']})"
                for item in sources
            ) or "(No web results were available.)"
            response = await self.llm_client.chat(
                self._prompt(title, author, excerpt, evidence)
            )
            data = self._parse_json(response)
            return BookResearchResult(
                book_context=str(data.get("book_context", "")).strip()[:4000],
                terms=self._normalise_terms(data.get("terms", []), sources),
                sources=sources,
                queries=queries,
            )
        except Exception as exc:
            logger.warning("Book research seed pass failed: %s", exc)
            return BookResearchResult(
                sources=sources,
                queries=queries,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )

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
    def _prompt(title: str, author: str, excerpt: str, evidence: str) -> str:
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

Rules:
- Suggestions are not authoritative. Be conservative.
- Do not invent an existing Persian translation.
- Cite only URLs present in the evidence.
- Include a term only when it is likely important across the book.
- Output JSON only."""
