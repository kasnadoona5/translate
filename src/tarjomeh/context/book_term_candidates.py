"""Source-only term-family candidates for human book-scoped review."""

from __future__ import annotations

import re
import hashlib
from collections import Counter
from typing import Any

from tarjomeh.parsers.base import Document

_COORDINATED_TERMS = re.compile(
    r"\b[A-Za-z][A-Za-z'-]+(?:,\s+[A-Za-z][A-Za-z'-]+){1,4}"
    r",?\s+and\s+[A-Za-z][A-Za-z'-]+\b",
    re.IGNORECASE,
)
_PARATEXT_TITLE = re.compile(
    r"^(?:contents|table of contents|copyright|abbreviations|acknowledg(?:e)?ments|"
    r"preface|foreword|references|bibliography|index|endnotes|notes)$",
    re.IGNORECASE,
)
_REFERENCE_ENTRY = re.compile(
    r"^[A-Z][A-Za-z'-]+,\s+(?:[A-Z]\.\s*){1,3}"
    r"(?:\(?(?:18|19|20)\d{2}[a-z]?\)?)[.,:]\s",
)
_WORD = re.compile(r"\b[A-Za-z][A-Za-z'-]{6,}\b")
_TOKEN = re.compile(r"\b[A-Za-z][A-Za-z'-]*\b")
_ABBREVIATION = re.compile(r"^([A-Z][A-Z0-9-]{1,9})(?:\s*[:\-]\s*|\s{2,})([A-Za-z][A-Za-z -]{4,100})$")
_WORD_STOP = frozenset({
    "another", "because", "between", "chapter", "different", "however",
    "include", "including", "itself", "political", "related", "through",
    "whereas", "without", "therefore", "following", "particular",
})
_PHRASE_EDGE_STOP = _WORD_STOP | frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "its", "not", "of", "on", "or", "our",
    "that", "the", "their", "these", "this", "those", "to", "was", "were",
    "which", "with", "would",
})


def _family_key(value: str) -> str:
    return " ".join(value.casefold().split())


def body_term_paragraphs(document: Document):
    """Yield source body prose for sampling; never alter document export."""
    chapters = getattr(document, "chapters", None)
    if chapters is None:
        chapters = [document]
    for chapter in chapters:
        if _PARATEXT_TITLE.fullmatch(str(getattr(chapter, "title", "")).strip()):
            continue
        for paragraph in chapter.all_paragraphs:
            source = paragraph.text.strip()
            if (
                getattr(paragraph, "is_translatable", bool(source))
                and not getattr(paragraph, "is_footnote", False)
                and getattr(paragraph, "heading_level", None) is None
                and getattr(paragraph, "metadata", {}).get("structure_role", "body") == "body"
                and not _REFERENCE_ENTRY.match(source)
            ):
                yield paragraph


def collect_book_term_candidates(document: Document) -> list[dict[str, Any]]:
    """Rank repeated source families; no guessed Persian enters the prompt."""
    counts: Counter[str] = Counter()
    evidence: dict[str, str] = {}
    evidence_hashes: dict[str, str] = {}
    surfaces: dict[str, str] = {}
    word_paragraph_counts: Counter[str] = Counter()
    phrase_paragraph_counts: Counter[str] = Counter()
    variants: dict[str, list[str]] = {}
    italic_keys: set[str] = set()
    for paragraph in body_term_paragraphs(document):
        present_words: set[str] = set()
        present_phrases: set[str] = set()
        for value in paragraph.metadata.get("italic_source_spans", []) or []:
            if (isinstance(value, str) and len(value) >= 5
                    and paragraph.text.count(value) == 1):
                key = value.casefold()
                counts[key] += 2
                italic_keys.add(key)
                surfaces.setdefault(key, value)
                evidence.setdefault(key, paragraph.text.strip()[:500])
                evidence_hashes.setdefault(key, hashlib.sha256(
                    paragraph.text.strip().encode("utf-8")
                ).hexdigest())
        for match in _COORDINATED_TERMS.finditer(paragraph.text):
            source = " ".join(match.group().split())
            key = source.casefold()
            counts[key] += 1
            surfaces.setdefault(key, source)
            evidence.setdefault(key, paragraph.text.strip()[:500])
            evidence_hashes.setdefault(key, hashlib.sha256(
                paragraph.text.strip().encode("utf-8")
            ).hexdigest())
        for match in _WORD.finditer(paragraph.text):
            word = match.group()
            key = word.casefold()
            if key not in _WORD_STOP and not key.endswith("ly"):
                present_words.add(key)
                surfaces.setdefault(key, word)
                evidence.setdefault(key, paragraph.text.strip()[:500])
                evidence_hashes.setdefault(key, hashlib.sha256(
                    paragraph.text.strip().encode("utf-8")
                ).hexdigest())
        tokens = list(_TOKEN.finditer(paragraph.text))
        for width in (2, 3):
            for offset in range(len(tokens) - width + 1):
                token_group = tokens[offset:offset + width]
                if any(not paragraph.text[left.end():right.start()].isspace()
                       for left, right in zip(token_group, token_group[1:])):
                    continue
                group = [match.group() for match in token_group]
                if (group[0].casefold() in _PHRASE_EDGE_STOP
                        or group[-1].casefold() in _PHRASE_EDGE_STOP
                        or not any(len(word) >= 5 for word in group)):
                    continue
                surface = " ".join(group)
                key = _family_key(surface)
                present_phrases.add(key)
                surfaces.setdefault(key, surface)
                evidence.setdefault(key, paragraph.text.strip()[:500])
                evidence_hashes.setdefault(key, hashlib.sha256(
                    paragraph.text.strip().encode("utf-8")
                ).hexdigest())
        word_paragraph_counts.update(present_words)
        phrase_paragraph_counts.update(present_phrases)
    for key, count in phrase_paragraph_counts.items():
        if count >= 3:
            counts[key] += count
    for key in list(phrase_paragraph_counts):
        words = key.split()
        last = words[-1]
        if (len(last) <= 4 or not last.endswith("s")
                or last.endswith(("ss", "us", "is"))):
            continue
        singular_last = last[:-3] + "y" if last.endswith("ies") else last[:-1]
        singular = " ".join([*words[:-1], singular_last])
        if singular not in phrase_paragraph_counts:
            continue
        counts[singular] += phrase_paragraph_counts[key]
        counts.pop(key, None)
        variants.setdefault(singular, []).append(surfaces[key])
    for key, count in word_paragraph_counts.items():
        if count >= 2:
            counts[key] += count
    for chapter in getattr(document, "chapters", []):
        if str(getattr(chapter, "title", "")).strip().casefold() != "abbreviations":
            continue
        for paragraph in chapter.all_paragraphs:
            source = paragraph.text.strip()
            match = _ABBREVIATION.fullmatch(source)
            if match:
                key = match.group(1).casefold()
                counts[key] += 2
                surfaces.setdefault(key, match.group(1))
                evidence.setdefault(key, source)
                evidence_hashes.setdefault(key, hashlib.sha256(
                    source.encode("utf-8")
                ).hexdigest())
    return [
        {
            "source": surfaces[key],
            "target": "",
            "status": "candidate",
            "origin": (
                "source_coordination" if _COORDINATED_TERMS.fullmatch(surfaces[key])
                else "abbreviations_page" if _ABBREVIATION.fullmatch(evidence[key])
                else "aligned_pdf_italic" if key in italic_keys
                else "repeated_source_phrase" if " " in surfaces[key]
                else "repeated_source_word"
            ),
            "confidence": "review_only",
            "source_count": count,
            "source_variants": variants.get(key, []),
            "source_evidence": evidence[key],
            "source_evidence_sha256": evidence_hashes[key],
            "reason": "Source-attested candidate for book-scoped review.",
        }
        for key, count in sorted(
            counts.items(),
            key=lambda item: (
                0 if _COORDINATED_TERMS.fullmatch(surfaces[item[0]]) else
                1 if " " in surfaces[item[0]] else 2,
                -item[1], item[0],
            ),
        )
        if count >= 2
    ][:30]


def book_term_extraction_sample(document: Document) -> str:
    """Bounded source evidence for the existing initial extraction call."""
    candidates = collect_book_term_candidates(document)
    lines = [
        f"{item['source']} | {item['source_evidence'][:180]}"
        for item in candidates[:20]
    ]
    body = [paragraph.text.strip() for paragraph in body_term_paragraphs(document)
            if len(paragraph.text.strip()) >= 80]
    if body:
        positions = sorted({0, len(body) // 2, len(body) - 1})
        lines.extend(body[position][:700] for position in positions)
    return "\n".join(lines)
