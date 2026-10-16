"""Shared current-source entity inventory; candidates confer no memory authority."""

from __future__ import annotations

import re

from tarjomeh.glossary.compliance import term_occurs_only_in_citations
from tarjomeh.memory.proper_nouns import is_safe_automatic_source_span

_LATIN_ENTITY_TOKEN = (
    r"(?:[A-Z\u00c0-\u00d6\u00d8-\u00de]"
    r"[A-Za-z\u00c0-\u024f\u1e00-\u1eff'\u2019-]+|[A-Z]\.)"
    r"(?![A-Za-z0-9])"
)
_LATIN_ENTITY_CONNECTOR = (
    r"(?:and|de|del|der|di|du|la|le|of|the|van|von|&)"
)
_LATIN_ENTITY_RE = re.compile(
    rf"(?<![\w])({_LATIN_ENTITY_TOKEN}"
    rf"(?:[ \t]+(?:{_LATIN_ENTITY_CONNECTOR}[ \t]+)?"
    rf"{_LATIN_ENTITY_TOKEN}){{1,4}})"
)
_NAMED_INSTRUMENT_LABEL = (
    r"(?:Act|Agreement|Charter|Code|Constitution|Convention|Directive|Law|"
    r"Protocol|Regulation|Statute|Treaty)"
)
_NAMED_INSTRUMENT_WORD = (
    r"(?:[A-Z\u00c0-\u00d6\u00d8-\u00de]"
    r"[A-Za-z\u00c0-\u024f\u1e00-\u1eff'\u2019-]+|[A-Z]{2,})"
)
_NAMED_INSTRUMENT_RE = re.compile(
    rf"(?<!\w)(?P<name>{_NAMED_INSTRUMENT_WORD}"
    rf"(?:(?:[ \t]+|,[ \t]*)(?:(?:and|of|the)[ \t]+)?"
    rf"{_NAMED_INSTRUMENT_WORD}){{0,8}}[ \t]+{_NAMED_INSTRUMENT_LABEL})\b"
)
_ENTITY_LEADING_NOISE = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "in", "on", "for",
    "from", "as", "at", "by", "while", "where", "when", "first", "second",
    "third", "fourth", "fifth", "sixth", "chapter", "part", "table", "figure",
    "introduction", "conclusion", "although", "because", "both", "either",
    "however", "neither", "nor", "therefore", "thus", "whereas",
})
_ENTITY_NON_NAME_TOKENS = frozenset({
    "approach", "chapter", "concept", "future", "government", "introduction",
    "market", "method", "part", "politics", "present", "state", "table",
    "theory", "world",
})
_NON_PERSON_ENTITY_ENDINGS = frozenset({
    "academy", "act", "africa", "america", "association", "atlantic", "bank",
    "committee", "company", "copyright", "council", "east", "europe",
    "foundation", "institute", "library", "ministry", "north", "organization",
    "organisation", "pacific", "party", "press", "project", "society", "south",
    "street", "title", "university", "west",
})
_PLACE_ENTITY_ENDINGS = frozenset({
    "africa", "america", "asia", "atlantic", "east", "europe", "north",
    "pacific", "south", "west",
})
_ORGANIZATION_ENTITY_ENDINGS = frozenset({
    "academy", "association", "bank", "committee", "company", "council",
    "foundation", "institute", "library", "ministry", "organization",
    "organisation", "party", "press", "society", "university", "inc",
    "limited", "llc", "ltd", "plc",
})
_ENTITY_TRAILING_ACTIONS = frozenset({
    "created", "edited", "printed", "published", "reproduced", "revised",
    "translated", "typeset",
})
_ENTITY_POSSESSIVE_RE = re.compile(r"(?:['\u2019]s)\Z", re.IGNORECASE)
_INITIALIZED_PERSON_RE = re.compile(
    r"^(?:[A-Z]\.[ \t]+){1,5}"
    r"[A-Z\u00c0-\u00d6\u00d8-\u00de][A-Za-z\u00c0-\u024f'\u2019-]+$"
)
def _canonical_source_entity(value: str) -> str:
    """Normalize citation possessives without changing the printed original."""
    compact = " ".join((value or "").split()).strip(" ,.;:()[]{}")
    return _ENTITY_POSSESSIVE_RE.sub("", compact).strip()


def _bounded_source_entity(value: str) -> str:
    """Trim a parser-joined action after a complete organization boundary."""
    compact = _canonical_source_entity(value)
    tokens = list(re.finditer(r"[A-Za-z\u00c0-\u024f]+", compact))
    if not tokens:
        return ""
    folded = [token.group().casefold() for token in tokens]
    if folded[-1] in _ENTITY_TRAILING_ACTIONS:
        terminal = next((
            index for index in range(len(tokens) - 2, -1, -1)
            if folded[index] in _ORGANIZATION_ENTITY_ENDINGS
        ), None)
        if terminal is not None:
            compact = compact[:tokens[terminal].end()].rstrip()
    return compact


def _source_entity_inventory(source_text: str, limit: int = 24) -> list[str]:
    """Return bounded, high-precision current-passage entity candidates."""
    candidates: list[str] = []
    instrument_spans: list[tuple[int, int]] = []
    for match in _NAMED_INSTRUMENT_RE.finditer(source_text or ""):
        value = _bounded_source_entity(match.group("name"))
        value = re.sub(r"^(?:A|An|The)\s+", "", value)
        if value.casefold() not in {item.casefold() for item in candidates}:
            candidates.append(value)
        instrument_spans.append(match.span("name"))
        if len(candidates) >= limit:
            return candidates
    for match in _LATIN_ENTITY_RE.finditer(source_text or ""):
        if any(
            match.start(1) < end and match.end(1) > start
            for start, end in instrument_spans
        ):
            continue
        matched_value = _bounded_source_entity(match.group(1))
        values = [matched_value]
        if re.search(r"[ \t]+and[ \t]+", matched_value, re.IGNORECASE):
            parts = re.split(
                r"[ \t]+and[ \t]+", matched_value,
                maxsplit=1, flags=re.IGNORECASE,
            )
            ending = re.findall(r"[A-Za-z\u00c0-\u024f]+", matched_value)[-1]
            if (
                ending.casefold() not in _ORGANIZATION_ENTITY_ENDINGS
                and all(2 <= len(part.split()) <= 4 for part in parts)
            ):
                values = parts
        for value in values:
            category = _source_entity_category(source_text, value)
            if not is_safe_automatic_source_span(value, category):
                continue
            tokens = re.findall(
                r"[A-Za-z\u00c0-\u024f\u1e00-\u1eff]+", value
            )
            folded = [token.casefold() for token in tokens]
            if not 2 <= len(tokens) <= 8 or folded[0] in _ENTITY_LEADING_NOISE:
                continue
            if all(token in _ENTITY_NON_NAME_TOKENS for token in folded):
                continue
            if value.isupper() or term_occurs_only_in_citations(source_text, value):
                continue
            if value.casefold() not in {item.casefold() for item in candidates}:
                candidates.append(value)
            if len(candidates) >= limit:
                break
        if len(candidates) >= limit:
            break
    return candidates


def _source_entity_category(source_text: str, candidate: str) -> str:
    """Classify provisional entities conservatively from local source evidence."""
    tokens = re.findall(r"[A-Za-z\u00c0-\u024f]+", candidate or "")
    if not tokens:
        return "source_entity_candidate"
    ending = tokens[-1].casefold()
    if re.search(rf"\b{_NAMED_INSTRUMENT_LABEL}$", candidate):
        return "legal_instrument"
    if ending in _PLACE_ENTITY_ENDINGS:
        return "place"
    if ending in _ORGANIZATION_ENTITY_ENDINGS:
        return "organization"
    occurrence = re.compile(
        rf"(?<!\w){re.escape(candidate)}(?P<possessive>['\u2019]s)?"
        rf"(?P<citation>\s*\(\s*(?:1[5-9]\d{{2}}|20\d{{2}})[a-z]?)?",
        re.IGNORECASE,
    ).search(source_text or "")
    if occurrence and (
        occurrence.group("possessive") or occurrence.group("citation")
    ):
        return "person"
    if _INITIALIZED_PERSON_RE.fullmatch(candidate.strip()):
        return "person"
    if re.search(
        rf"(?<!\w){re.escape(candidate)}\s*,\s*(?:who|whom|whose)\b",
        source_text or "",
        re.IGNORECASE,
    ):
        return "person"
    return "source_entity_candidate"
