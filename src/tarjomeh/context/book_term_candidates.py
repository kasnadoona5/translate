"""Source-only term candidates for human book-scoped review.

Everything here is derived from the English source alone. No Persian is
guessed, and no candidate acquires glossary or prompt authority by being
listed. The same body-only paragraph filter is shared by term review, the
optional research excerpt and the initial extraction sample; excluded
paratext is still translated and exported unchanged.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any

from tarjomeh.parsers.base import Document

_COORDINATED_TERMS = re.compile(
    r"\b[A-Za-z][A-Za-z'-]+(?:,\s+[A-Za-z][A-Za-z'-]+){1,4}"
    r",?\s+and\s+[A-Za-z][A-Za-z'-]+"
    # Keep a capitalized multiword final member intact ("South Africa").
    r"(?:\s+[A-Z][A-Za-z'-]+)?\b",
)
_FRONT_PARATEXT_TITLE = re.compile(
    r"^(?:contents|table of contents|copyright|abbreviations|acknowledg(?:e)?ments|"
    r"preface|foreword|dedication|tables|list of tables|figures|list of figures|"
    r"illustrations|list of illustrations|maps|list of maps)$",
    re.IGNORECASE,
)
_BACK_PARATEXT_TITLE = re.compile(
    r"(?:\bind(?:ex|exes|ices)\b|^(?:notes|endnotes|references|bibliography|"
    r"works cited|further reading|glossary)$)",
    re.IGNORECASE,
)
_CONTENTS_TITLE = re.compile(r"^(?:contents|table of contents)$", re.IGNORECASE)
_REFERENCE_ENTRY = re.compile(
    r"^[A-Z][A-Za-z'-]+,\s+(?:[A-Z]\.\s*){1,3}"
    r"(?:\(?(?:18|19|20)\d{2}[a-z]?\)?)[.,:]\s",
)
_LATIN = "A-Za-zÀ-ɏ"
_WORD = re.compile(
    rf"(?<![{_LATIN}'’-])[{_LATIN}][{_LATIN}'’-]{{6,}}(?![{_LATIN}])"
)
_TOKEN = re.compile(
    rf"(?<![{_LATIN}'’-])[{_LATIN}][{_LATIN}'’-]*(?![{_LATIN}])"
)
# General English adjective/adverb morphology: such single words are
# modifiers ("national", "economic"), rarely a book's terminology.
_MODIFIER_SUFFIX = re.compile(r"(?:al|ic|ive|ous|ary|able|ible|ful|less|ish|ly)$")
_ABBREVIATION = re.compile(
    r"^([A-Z][A-Z0-9-]{1,9}(?:\s[A-Z][A-Z0-9-]{1,9}){0,2})"
    r"(?:\s*[:\-]\s*|\s+)([A-Za-z][A-Za-z ,/&()'–—-]{3,120})$"
)
_LETTER_DASH = re.compile(r"(?<=[A-Za-z])[–—](?=[A-Za-z])")
_WORD_STOP = frozenset({
    "another", "because", "between", "chapter", "chapters", "different",
    "however", "include", "including", "itself", "political", "related",
    "through", "whereas", "without", "therefore", "following", "particular",
    "although", "whether", "towards", "several", "important", "example",
    "general", "specific", "certain", "various", "argument", "question",
})
# General English function words: a phrase starting or ending with one is a
# connector ("rather than", "as well as"), not a term.
_PHRASE_EDGE_STOP = _WORD_STOP | frozenset({
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at",
    "be", "been", "before", "being", "both", "but", "by", "can", "could", "did",
    "do", "does", "each", "even", "first", "for", "from", "had", "has", "have",
    "he", "hence", "her", "here", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "least", "less", "many", "may", "more", "most", "much", "must",
    "new", "no", "nor", "not", "of", "on", "one", "only", "or", "other", "our",
    "over", "rather", "same", "second", "should", "since", "so", "some", "such",
    "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "third", "this", "those", "thus", "to", "two", "under", "upon", "very",
    "was", "we", "well", "were", "what", "when", "where", "which", "while",
    "who", "will", "with", "within", "would", "yet", "you",
})
_QUOTAS = {
    "repeated_source_phrase": 16,
    "fixed_source_expression": 6,
    "compound_source_term": 10,
    "repeated_source_word": 10,
    "aligned_pdf_italic": 8,
    "abbreviations_page": 12,
    "source_coordination": 8,
}
_KIND_ORDER = tuple(_QUOTAS)


def _family_key(value: str) -> str:
    return " ".join(value.casefold().split())


def _singular(word: str) -> str:
    """Conservative English singular used only to merge attested variants."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _coordination_key(value: str) -> str:
    items = [item for item in re.split(r",\s*|\s+and\s+", value.casefold()) if item]
    return " | ".join(_singular(item.strip()) for item in items)


def _normalized(text: str) -> str:
    """Treat a letter-joining en/em dash as a hyphen for candidate keys only."""
    return _LETTER_DASH.sub("-", text)


def _chapter_is_paratext(chapters: list[Any], position: int) -> bool:
    title = str(getattr(chapters[position], "title", "")).strip()
    if _FRONT_PARATEXT_TITLE.fullmatch(title) or _BACK_PARATEXT_TITLE.search(title):
        return True
    # Title and copyright pages precede the contents page.
    for index, chapter in enumerate(chapters[:6]):
        if _CONTENTS_TITLE.fullmatch(str(getattr(chapter, "title", "")).strip()):
            return position < index
    return False


def body_term_paragraphs(document: Document):
    """Yield source body prose for sampling; never alter document export."""
    chapters = getattr(document, "chapters", None)
    if chapters is None:
        chapters = [document]
    for position, chapter in enumerate(chapters):
        if hasattr(chapter, "title") and _chapter_is_paratext(chapters, position):
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


def _sha(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def collect_book_term_candidates(document: Document) -> list[dict[str, Any]]:
    """Rank source-attested candidates by kind; no guessed Persian enters prompts."""
    paragraphs = list(body_term_paragraphs(document))
    kinds: dict[str, str] = {}
    surfaces: dict[str, str] = {}
    evidence: dict[str, str] = {}
    counts: Counter[str] = Counter()
    variants: dict[str, list[str]] = {}

    def note(key: str, surface: str, kind: str, paragraph_text: str) -> None:
        kinds.setdefault(key, kind)
        surfaces.setdefault(key, surface)
        # Prefer running-prose casing over a title-cased first occurrence.
        if surface.islower() and not surfaces[key].islower():
            surfaces[key] = surface
        evidence.setdefault(key, paragraph_text)

    phrase_paragraphs: Counter[str] = Counter()
    long_phrases: set[str] = set()
    word_paragraphs: Counter[str] = Counter()
    token_totals: Counter[str] = Counter()
    phrase_token_totals: dict[str, Counter[str]] = {}
    family_paragraphs: Counter[str] = Counter()
    italic_paragraphs: Counter[str] = Counter()

    for paragraph in paragraphs:
        raw = paragraph.text.strip()
        text = _normalized(paragraph.text)
        for value in paragraph.metadata.get("italic_source_spans", []) or []:
            words = value.split() if isinstance(value, str) else []
            if (isinstance(value, str) and len(value) >= 4
                    and len(words) <= 4
                    and sum(word.casefold() not in _PHRASE_EDGE_STOP
                            for word in words) >= max(1, len(words) - 1)
                    and len(re.findall(rf"(?<!\w){re.escape(value)}(?!\w)",
                                       paragraph.text)) == 1):
                key = "italic:" + value.casefold()
                italic_paragraphs[key] += 1
                note(key, value, "aligned_pdf_italic", raw)
        present_families: set[str] = set()
        for match in _COORDINATED_TERMS.finditer(text):
            surface = " ".join(match.group().split())
            key = "family:" + _coordination_key(surface)
            present_families.add(key)
            note(key, surface, "source_coordination", raw)
            if surface not in variants.setdefault(key, []):
                variants[key].append(surface)
        family_paragraphs.update(present_families)
        tokens = list(_TOKEN.finditer(text))
        token_totals.update(match.group().casefold() for match in tokens)
        present_words: set[str] = set()
        for match in _WORD.finditer(text):
            word = match.group().strip("'’-")
            key = word.casefold()
            if "-" in key:
                # A hyphenated compound ("nation-state") is a candidate term.
                parts = [part for part in key.split("-") if part]
                if len(parts) >= 2 and all(len(part) >= 3 for part in parts):
                    compound = "compound:" + "-".join(
                        [*parts[:-1], _singular(parts[-1])]
                    )
                    present_words.add(compound)
                    note(compound, word, "compound_source_term", raw)
                continue
            if key in _WORD_STOP or _MODIFIER_SUFFIX.search(key):
                continue
            singular = "word:" + _singular(key)
            present_words.add(singular)
            note(singular, word, "repeated_source_word", raw)
            if word not in variants.setdefault(singular, []):
                variants[singular].append(word)
        word_paragraphs.update(present_words)
        present_phrases: set[str] = set()
        for width in (2, 3):
            for offset in range(len(tokens) - width + 1):
                group_matches = tokens[offset:offset + width]
                if any(not text[left.end():right.start()].isspace()
                       for left, right in zip(group_matches, group_matches[1:], strict=False)):
                    continue
                group = [match.group() for match in group_matches]
                substantive = [
                    word for word in group if word.casefold() not in _PHRASE_EDGE_STOP
                ]
                if substantive and all(word[:1].isupper() for word in substantive):
                    # Name pairs and titles ("Miller and Rose") are entities.
                    continue
                if (group[0].casefold() in _PHRASE_EDGE_STOP
                        or group[-1].casefold() in _PHRASE_EDGE_STOP
                        or not any(len(word) >= 3
                                   and word.casefold() not in _PHRASE_EDGE_STOP
                                   for word in group)):
                    continue
                key = _family_key(" ".join(group))
                if any(len(word) >= 5 and word.casefold() not in _PHRASE_EDGE_STOP
                       for word in group):
                    long_phrases.add(key)
                present_phrases.add(key)
                note("phrase:" + key, " ".join(group), "repeated_source_phrase", raw)
                phrase_token_totals.setdefault(key, Counter()).update(
                    word.casefold() for word in group
                )
        phrase_paragraphs.update(present_phrases)

    # Repeated phrases; merge an attested plural into its attested singular.
    merged: Counter[str] = Counter()
    for key, count in phrase_paragraphs.items():
        if key not in long_phrases:
            continue
        words = key.split()
        singular = " ".join([*words[:-1], _singular(words[-1])])
        if singular != key and singular in phrase_paragraphs:
            # Both forms are attested in the source: count them as one term.
            merged[singular] += count
            variants.setdefault("phrase:" + singular, []).append(surfaces["phrase:" + key])
        else:
            merged[key] += count
    phrase_counts: Counter[str] = Counter(
        {key: count for key, count in merged.items() if count >= 3}
    )
    for key, count in phrase_counts.items():
        counts["phrase:" + key] = count

    # Fixed expressions: every substantive word occurs only inside the phrase.
    for key, count in phrase_paragraphs.items():
        if count < 2:
            continue
        inside = phrase_token_totals.get(key, Counter())
        substantive = [
            word for word in inside
            if len(word) >= 3 and word not in _PHRASE_EDGE_STOP
        ]
        surface = surfaces.get("phrase:" + key, "")
        if (substantive and surface.islower()
                and all(token_totals[word] == inside[word] for word in substantive)):
            fixed = "fixed:" + key
            kinds[fixed] = "fixed_source_expression"
            surfaces[fixed] = surfaces["phrase:" + key]
            evidence[fixed] = evidence["phrase:" + key]
            counts[fixed] = count
            counts.pop("phrase:" + key, None)

    # Keep a complete fixed expression ("esprit de corps"), not its fragment.
    fixed_keys = [key for key in counts if key.startswith("fixed:")]
    for key in fixed_keys:
        inner = key[len("fixed:"):]
        if any(other != key and counts[other] == counts[key]
               and f" {inner} " in f" {other[len('fixed:'):]} "
               for other in fixed_keys):
            counts.pop(key, None)

    for key, count in word_paragraphs.items():
        if count >= 2:
            counts[key] = count
    for key, count in italic_paragraphs.items():
        counts[key] = count
    for key, count in family_paragraphs.items():
        if count >= 2:
            counts[key] = count

    chapters = getattr(document, "chapters", []) or []
    for chapter in chapters:
        if str(getattr(chapter, "title", "")).strip().casefold() != "abbreviations":
            continue
        for paragraph in chapter.all_paragraphs:
            source = paragraph.text.strip()
            match = _ABBREVIATION.fullmatch(source)
            if not match or match.group(2).strip().casefold() == "abbreviations":
                continue
            key = "abbr:" + match.group(1).casefold()
            note(key, match.group(1), "abbreviations_page", source)
            counts[key] = max(counts[key], 2)

    ranked: dict[str, list[str]] = {kind: [] for kind in _KIND_ORDER}
    for key, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        kind = kinds.get(key, "")
        if kind in ranked and len(ranked[kind]) < _QUOTAS[kind]:
            ranked[kind].append(key)
    selected = [key for kind in _KIND_ORDER for key in ranked[kind]]
    return [
        {
            "source": surfaces[key],
            "target": "",
            "status": "candidate",
            "origin": kinds[key],
            "confidence": "review_only",
            "source_count": counts[key],
            "source_variants": [
                value for value in variants.get(key, [])
                if value != surfaces[key]
            ],
            "source_evidence": evidence[key][:500],
            "source_evidence_sha256": _sha(evidence[key]),
            "reason": "Source-attested candidate for book-scoped review.",
        }
        for key in selected
    ]


def book_term_extraction_sample(document: Document) -> str:
    """Bounded source evidence for the existing initial extraction call."""
    candidates = collect_book_term_candidates(document)
    lines = [
        f"{item['source']} | {item['source_evidence'][:180]}"
        for item in candidates[:24]
    ]
    body = [paragraph.text.strip() for paragraph in body_term_paragraphs(document)
            if len(paragraph.text.strip()) >= 80]
    if body:
        positions = sorted({0, len(body) // 2, len(body) - 1})
        lines.extend(body[position][:700] for position in positions)
    return "\n".join(lines)
