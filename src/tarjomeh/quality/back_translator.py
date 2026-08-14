"""Back-translation verification for Tarjomeh.

Translates a subset of Persian translations back to English and
compares them with the original source to flag semantic drift.
"""

from __future__ import annotations

import logging
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

from tarjomeh.core.prompts import BACK_TRANSLATE_PROMPT
from tarjomeh.quality.integrity import extract_numbers, reconcile_numbers
from tarjomeh.glossary.compliance import target_present


@dataclass
class BackTranslationResult:
    """Result of a back-translation comparison.

    Attributes
    ----------
    similarity_score : float
        Word-overlap ratio between the original English and the
        back-translated English (0.0 – 1.0).
    flagged : bool
        ``True`` if the chunk should be flagged for human review
        (similarity < threshold).
    differences : list[str]
        Words present in the original but missing from the
        back-translation (indicative of potential meaning loss).
    back_translated : str
        The full back-translated English text.
    """

    similarity_score: float = 0.0
    flagged: bool = False
    differences: list[str] = field(default_factory=list)
    back_translated: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "similarity_score": self.similarity_score,
            "flagged": self.flagged,
            "differences": self.differences,
            "back_translated": self.back_translated,
            "diagnostics": self.diagnostics,
        }


class BackTranslator:
    """Performs probabilistic back-translation QA.

    A configurable percentage of translated chunks are randomly sampled,
    translated back to English, and compared with the original source
    text to detect semantic drift.

    Parameters
    ----------
    llm_client:
        Any object exposing an ``async chat(prompt) -> str`` method.
    sample_pct : float
        Percentage of chunks to back-translate (0–100).  Default ``20``
        (the ``academic`` mode default).
    similarity_threshold : float
        Retained for API compatibility. Lexical overlap is advisory; structured
        number, entity, negation, omission, and addition risks drive flags.
    """

    def __init__(
        self,
        llm_client: Any,
        sample_pct: float = 20.0,
        similarity_threshold: float = 0.5,
    ) -> None:
        self._llm = llm_client
        self._sample_pct = max(0.0, min(100.0, sample_pct))
        self._similarity_threshold = similarity_threshold

    # ── sampling ─────────────────────────────────────────────────────

    def should_sample(self) -> bool:
        """Decide whether the current chunk should be back-translated.

        Uses a simple random draw based on :attr:`sample_pct`.

        Returns
        -------
        bool
            ``True`` if this chunk should be sampled.
        """
        if self._sample_pct <= 0.0:
            return False
        if self._sample_pct >= 100.0:
            return True
        return random.random() * 100.0 < self._sample_pct

    # ── back-translation ─────────────────────────────────────────────

    async def back_translate(self, persian_text: str) -> str:
        """Translate *persian_text* back to English via the LLM.

        Parameters
        ----------
        persian_text:
            Persian text to translate back to English.

        Returns
        -------
        str
            English back-translation.
        """
        prompt = BACK_TRANSLATE_PROMPT.format(persian_text=persian_text)
        if hasattr(self._llm, "set_operation"):
            self._llm.set_operation("back_translation")
        result: str = await self._llm.chat(prompt)

        # Strip accidental markdown fences.
        result = result.strip()
        if result.startswith("```"):
            lines = result.splitlines()
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            result = "\n".join(lines).strip()

        return result

    # ── comparison ───────────────────────────────────────────────────

    def compare(
        self,
        original_english: str,
        back_translated: str,
        translated_text: str = "",
        entity_aliases: dict[str, list[str]] | None = None,
    ) -> BackTranslationResult:
        """Compare the original English with a back-translation.

        Computes word overlap as an advisory metric and structured high-risk
        diagnostics for numbers, entities, negation, omission, and addition:

        .. math::

            \\text{similarity} = \\frac{|W_{\\text{orig}} \\cap W_{\\text{back}}|}
                                      {|W_{\\text{orig}} \\cup W_{\\text{back}}|}

        where *W* is the multiset of lowercased tokens.

        Parameters
        ----------
        original_english:
            The original English source text.
        back_translated:
            The English text produced by back-translating the Persian.
        translated_text:
            Optional final Persian text. Source entities visibly preserved there
            in Latin script are not falsely reported as lost by back-translation.

        Returns
        -------
        BackTranslationResult
            Comparison result with similarity score and flagged status.
        """
        orig_tokens = _tokenize_simple(original_english)
        back_tokens = _tokenize_simple(back_translated)

        orig_counter = Counter(orig_tokens)
        back_counter = Counter(back_tokens)

        # Intersection (element-wise minimum) and union (element-wise maximum).
        intersection_size = sum((orig_counter & back_counter).values())
        union_size = sum((orig_counter | back_counter).values())

        similarity = intersection_size / union_size if union_size > 0 else 0.0

        # Words in the original but missing from the back-translation.
        missing = orig_counter - back_counter
        differences = sorted(missing.elements())

        source_numbers = extract_numbers(original_english)
        back_numbers = extract_numbers(back_translated)
        raw_missing_number_counts = source_numbers - back_numbers
        translated_number_evidence = (
            reconcile_numbers(original_english, translated_text)
            if translated_text else {"missing": [], "localized_equivalents": []}
        )
        translation_missing_counts = Counter(
            translated_number_evidence.get("missing", [])
        )
        missing_number_counts = (
            raw_missing_number_counts & translation_missing_counts
            if translated_text else raw_missing_number_counts
        )
        reconciled_number_counts = raw_missing_number_counts - missing_number_counts
        missing_numbers = list(missing_number_counts.elements())
        added_numbers = list((back_numbers - source_numbers).elements())

        structural_listing = _looks_like_structural_listing(original_english)
        non_prose_front_matter = _looks_like_non_prose_front_matter(
            original_english
        )
        non_prose = structural_listing or non_prose_front_matter
        source_entities = (
            [] if non_prose else _extract_entities(original_english)
        )
        preserved_inline_entities = [
            entity for entity in source_entities
            if translated_text and _entity_is_present(entity, translated_text)
        ]
        aliases_were_supplied = entity_aliases is not None
        entity_aliases = entity_aliases or {}
        reconciled_aliases: list[dict[str, str]] = []
        structurally_reconciled: list[dict[str, str]] = []
        deferred_first_seen_entities: list[str] = []
        for entity in source_entities:
            if entity in preserved_inline_entities:
                continue
            structural_target = _structural_entity_target(
                entity, original_english, translated_text
            )
            if structural_target:
                structurally_reconciled.append({
                    "source": entity,
                    "target": structural_target,
                })
                continue
            aliases: list[str] = []
            for source, values in entity_aliases.items():
                source_grounded = bool(re.search(
                    rf"(?<!\w){re.escape(source)}(?!\w)",
                    original_english,
                    re.IGNORECASE,
                ))
                entity_grounded_in_source = bool(re.search(
                    rf"(?<!\w){re.escape(entity)}(?!\w)",
                    source,
                    re.IGNORECASE,
                ))
                if (
                    source.casefold() == entity.casefold()
                    or (source_grounded and entity_grounded_in_source)
                ):
                    aliases.extend(values)
            matched_alias = next((
                alias for alias in aliases
                if alias and target_present(alias, translated_text)
            ), "")
            if matched_alias:
                reconciled_aliases.append({
                    "source": entity,
                    "target": matched_alias,
                })
            elif aliases_were_supplied and not aliases:
                # Incremental NER intentionally runs after chunk QA. A first-seen
                # name has no trusted Persian alias yet, so final term anchoring
                # must verify it instead of producing a premature omission flag.
                deferred_first_seen_entities.append(entity)
        reconciled_entities = {
            item["source"].casefold() for item in reconciled_aliases
        } | {
            item["source"].casefold() for item in structurally_reconciled
        }
        missing_entities = [
            entity
            for entity in source_entities
            if not _entity_is_present(entity, back_translated)
            and entity not in preserved_inline_entities
            and entity.casefold() not in reconciled_entities
            and entity not in deferred_first_seen_entities
        ]
        source_negations = _negations(original_english)
        back_negations = _negations(back_translated)
        negation_mismatch = bool(source_negations) != bool(back_negations)

        source_sentences = _sentence_count(original_english)
        back_sentences = _sentence_count(back_translated)
        sentence_ratio = back_sentences / max(1, source_sentences)
        possible_omission = not non_prose and (
            (source_sentences >= 2 and sentence_ratio < 0.5)
            or (len(orig_tokens) >= 20 and similarity < 0.2)
        )
        possible_addition = (
            not non_prose and source_sentences >= 1 and sentence_ratio > 2.0
        )
        risk_flags = []
        if missing_numbers:
            risk_flags.append("numbers_missing_or_changed")
        if added_numbers:
            risk_flags.append("numbers_added_or_changed")
        if missing_entities:
            risk_flags.append("named_entities_missing")
        if negation_mismatch:
            risk_flags.append("negation_mismatch")
        if possible_omission:
            risk_flags.append("possible_proposition_omission")
        if possible_addition:
            risk_flags.append("possible_unsupported_addition")

        diagnostics = {
            "risk_flags": risk_flags,
            "missing_numbers": missing_numbers,
            "added_numbers": added_numbers,
            "numbers_reconciled_by_translation": list(
                reconciled_number_counts.elements()
            ),
            "translated_number_evidence": translated_number_evidence,
            "missing_entities": missing_entities,
            "entity_check_skipped_for_structural_listing": structural_listing,
            "qa_risk_skipped_for_non_prose_front_matter": non_prose_front_matter,
            "entities_preserved_inline": preserved_inline_entities,
            "entities_reconciled_by_memory": reconciled_aliases,
            "entities_reconciled_by_structure": structurally_reconciled,
            "entities_deferred_until_term_anchoring": deferred_first_seen_entities,
            "source_negations": source_negations,
            "back_translation_negations": back_negations,
            "negation_mismatch": negation_mismatch,
            "source_sentence_count": source_sentences,
            "back_translation_sentence_count": back_sentences,
            "sentence_count_ratio": round(sentence_ratio, 4),
            "possible_omission": possible_omission,
            "possible_addition": possible_addition,
            "lexical_overlap_advisory": round(similarity, 4),
        }
        # Back-translation is a secondary review signal. It never replaces or
        # edits the Persian translation by itself.
        flagged = bool(risk_flags)

        if flagged:
            logger.warning(
                "Back-translation structured risks: %s (lexical advisory=%.2f)",
                risk_flags,
                similarity,
            )

        return BackTranslationResult(
            similarity_score=round(similarity, 4),
            flagged=flagged,
            differences=differences,
            back_translated=back_translated,
            diagnostics=diagnostics,
        )


# ── Utility ──────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-zA-Z0-9']+")
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][\w'\-]+(?:[ \t]+(?:of|the|and|&|[A-Z][\w'\-]+)){0,5})\b"
)
_ENTITY_STOP = {"the", "a", "an", "this", "that", "these", "those", "in", "on", "chapter", "introduction"}
_ENTITY_LINK_WORDS = {"a", "an", "the", "of", "and"}
_ENTITY_DESCRIPTORS = {"book", "work", "text", "title"}
_ENTITY_TOKEN_EQUIVALENTS = {
    "africa": {"african"},
    "african": {"africa"},
    "america": {"american"},
    "american": {"america"},
    "asia": {"asian"},
    "asian": {"asia"},
    "europe": {"european"},
    "european": {"europe"},
}
_NEGATION_RE = re.compile(r"\b(?:not|no|never|without|neither|nor|cannot|can't|won't|isn't|aren't|didn't|doesn't)\b", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_ROMAN_VALUES = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5,
    "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
}
_PERSIAN_CARDINALS = {
    1: ("یک", "اول", "نخست"),
    2: ("دو", "دوم"),
    3: ("سه", "سوم"),
    4: ("چهار", "چهارم"),
    5: ("پنج", "پنجم"),
    6: ("شش", "ششم"),
    7: ("هفت", "هفتم"),
    8: ("هشت", "هشتم"),
    9: ("نه", "نهم"),
    10: ("ده", "دهم"),
}


def _structural_entity_target(
    entity: str,
    original_english: str,
    translated_text: str,
) -> str:
    """Reconcile translated Part/Chapter headings without hiding name loss."""
    heading = entity or ""
    match = re.match(
        r"^(?P<label>Part|Chapter|Volume|Book|Section)\s+"
        r"(?P<number>[IVX]+|\d+)\b",
        heading,
        re.IGNORECASE,
    )
    if not match:
        for line in (original_english or "").splitlines():
            normalized = " ".join(line.split()).strip()
            heading_match = re.match(
                r"^(?P<label>Part|Chapter|Volume|Book|Section)\s+"
                r"(?P<number>[IVX]+|\d+)\b",
                normalized,
                re.IGNORECASE,
            )
            if not heading_match:
                continue
            heading_shape = ":" in normalized or not re.search(
                r"[.!?][\"')\]]?$", normalized
            )
            contains_entity = bool(re.search(
                rf"(?<!\w){re.escape(entity)}(?!\w)",
                normalized,
                re.IGNORECASE,
            ))
            if heading_shape and contains_entity:
                match = heading_match
                break
    if not match or not translated_text:
        return ""
    label = match.group("label").casefold()
    raw_number = match.group("number").casefold()
    number = int(raw_number) if raw_number.isdigit() else _ROMAN_VALUES.get(raw_number)
    if not number:
        return ""
    labels = {
        "part": ("بخش", "قسمت"),
        "chapter": ("فصل",),
        "volume": ("جلد",),
        "book": ("کتاب",),
        "section": ("بخش", "قسمت"),
    }[label]
    digits = str(number).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))
    forms = (str(number), digits, *_PERSIAN_CARDINALS.get(number, ()))
    label_pattern = "|".join(re.escape(value) for value in labels)
    form_pattern = "|".join(re.escape(value) for value in forms)
    found = re.search(
        rf"(?:{label_pattern})\s+(?P<form>{form_pattern})(?![\u0600-\u06ff\d])",
        translated_text,
    )
    return found.group(0) if found else ""


def _tokenize_simple(text: str) -> list[str]:
    """Extract lowercased word tokens from English text.

    A deliberately simple tokenizer — strips punctuation and
    lowercases everything.  Good enough for word-overlap comparison.
    """
    return [m.group().lower() for m in _WORD_RE.finditer(text)]


def _extract_entities(text: str) -> list[str]:
    entities = []
    for match in _ENTITY_RE.finditer(text or ""):
        value = " ".join(match.group().split()).strip()
        prefix = (text or "")[:match.start()]
        stripped_prefix = prefix.rstrip(" \t")
        at_sentence_start = (
            not stripped_prefix
            or stripped_prefix.endswith((".", "!", "?", "\n"))
        )
        single_word = " " not in value
        if value.split()[-1].casefold() in {"of", "the", "and", "&"}:
            continue
        if single_word and at_sentence_start and not value.isupper():
            continue
        if value.casefold() not in _ENTITY_STOP and len(value) > 2:
            entities.append(value)
    return sorted(set(entities), key=str.casefold)


def _looks_like_structural_listing(text: str) -> bool:
    """Detect TOCs/tables/index-like source where capitals are not entities."""
    value = text or ""
    lines = [" ".join(line.split()) for line in value.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    numbered_lines = sum(bool(re.search(r"\b\d+(?:[-\u2013]\d+)?\s*$", line)) for line in lines)
    short_lines = sum(len(line.split()) <= 16 for line in lines)
    sentence_lines = sum(bool(re.search(r"[.!?][\"')\]]?\s*$", line)) for line in lines)
    return (
        numbered_lines / len(lines) >= 0.35
        and short_lines / len(lines) >= 0.6
        and sentence_lines / len(lines) <= 0.35
    )


def _looks_like_non_prose_front_matter(text: str) -> bool:
    """Detect publishing metadata/dedication blocks without book-specific text."""
    value = text or ""
    folded = value.casefold()
    markers = (
        "all rights reserved", "copyright", "isbn", "issn",
        "library of congress", "british library", "cataloguing",
        "cataloging", "typeset", "printed and bound", "in memoriam",
    )
    marker_count = sum(marker in folded for marker in markers)
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    metadata_lines = sum(bool(re.search(
        r"(?:https?://|www\.|isbn|copyright|\u00a9|\b(?:19|20)\d{2}\b)",
        line,
        re.IGNORECASE,
    )) for line in lines)
    return marker_count >= 2 or (
        marker_count >= 1 and len(lines) >= 3 and metadata_lines >= 2
    )


def _entity_is_present(entity: str, text: str) -> bool:
    """Match an entity by its ordered meaningful tokens.

    Back-translations may insert a harmless descriptor, such as "book" in a
    translated title. Meaningful entity tokens must still occur contiguously
    and in their original order after normalization.
    """
    entity_tokens = [
        token for token in _tokenize_simple(entity)
        if token not in _ENTITY_LINK_WORDS
    ]
    if not entity_tokens:
        return True

    ignored_back_tokens = _ENTITY_LINK_WORDS | (
        _ENTITY_DESCRIPTORS - set(entity_tokens)
    )
    back_tokens = [
        token for token in _tokenize_simple(text)
        if token not in ignored_back_tokens
    ]

    width = len(entity_tokens)
    return any(
        all(
            actual == expected
            or actual in _ENTITY_TOKEN_EQUIVALENTS.get(expected, set())
            for expected, actual in zip(
                entity_tokens,
                back_tokens[index:index + width],
            )
        )
        for index in range(len(back_tokens) - width + 1)
    )


def _negations(text: str) -> list[str]:
    return [match.group().casefold() for match in _NEGATION_RE.finditer(text or "")]


def _sentence_count(text: str) -> int:
    return len([part for part in _SENTENCE_RE.split((text or "").strip()) if part.strip()])
