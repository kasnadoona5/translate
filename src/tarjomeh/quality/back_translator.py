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
from tarjomeh.quality.integrity import extract_numbers


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
        missing_numbers = list((source_numbers - back_numbers).elements())
        added_numbers = list((back_numbers - source_numbers).elements())

        source_entities = _extract_entities(original_english)
        preserved_inline_entities = [
            entity for entity in source_entities
            if translated_text and _entity_is_present(entity, translated_text)
        ]
        missing_entities = [
            entity
            for entity in source_entities
            if not _entity_is_present(entity, back_translated)
            and entity not in preserved_inline_entities
        ]
        source_negations = _negations(original_english)
        back_negations = _negations(back_translated)
        negation_mismatch = bool(source_negations) != bool(back_negations)

        source_sentences = _sentence_count(original_english)
        back_sentences = _sentence_count(back_translated)
        sentence_ratio = back_sentences / max(1, source_sentences)
        possible_omission = (
            (source_sentences >= 2 and sentence_ratio < 0.5)
            or (len(orig_tokens) >= 20 and similarity < 0.2)
        )
        possible_addition = source_sentences >= 1 and sentence_ratio > 2.0
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
            "missing_entities": missing_entities,
            "entities_preserved_inline": preserved_inline_entities,
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
        prefix = (text or "")[:match.start()].rstrip()
        at_sentence_start = not prefix or prefix.endswith((".", "!", "?"))
        single_word = " " not in value
        if value.split()[-1].casefold() in {"of", "the", "and", "&"}:
            continue
        if single_word and at_sentence_start and not value.isupper():
            continue
        if value.casefold() not in _ENTITY_STOP and len(value) > 2:
            entities.append(value)
    return sorted(set(entities), key=str.casefold)


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
