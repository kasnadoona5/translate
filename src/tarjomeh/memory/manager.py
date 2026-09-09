"""Memory manager coordinating the translation memory system.

Integrates a book-level style profile, proper noun records, bilingual summary,
long-term memory, and short-term memory for cohesive book-length translation.
"""

from __future__ import annotations

import logging
import hashlib
import re
from dataclasses import dataclass
from typing import Any

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.structured_output import parse_structured_output
from tarjomeh.core.term_notes import effective_term_notes_mode
from tarjomeh.chunking.chunker import Chunk
from tarjomeh.memory.proper_nouns import (
    ProperNouns,
    has_exact_bilingual_anchor,
    has_minimal_automatic_term_evidence,
    is_automatic_entity_category,
    is_safe_automatic_entity_mapping,
    is_usable_memory_mapping,
    low_authority_mapping_category,
)
from tarjomeh.memory.bilingual_summary import BilingualSummary
from tarjomeh.memory.long_term import LongTermMemory
from tarjomeh.memory.short_term import ShortTermMemory
from tarjomeh.quality.integrity import (
    detached_ezafe_artifacts,
    foreign_script_artifacts,
    markup_wrapper_artifacts,
    mixed_script_artifacts,
    parenthesis_artifacts,
    repeated_persian_clause_artifacts,
    repeated_persian_word_artifacts,
    tatweel_separator_artifacts,
)

logger = logging.getLogger(__name__)

_NON_PROSE_RE = re.compile(
    r"\b(?:copyright|all rights reserved|isbn|contents|tables|abbreviations|"
    r"index of names|subject index|catalog(?:ue|ing)-in-publication)\b",
    re.IGNORECASE,
)
_NONREPRESENTATIVE_STYLE_SOURCE_RE = re.compile(
    r"^\s*(?:(?:I|we)\s+dedicate\s+(?:this\s+(?:book|volume|work)\s+)?to|"
    r"(?:this\s+(?:book|volume|work)\s+is\s+)?dedicated\s+to|"
    r"to\s+the\s+memory\s+of|in\s+memory\s+of|in\s+memoriam|"
    r"acknowledg(?:e|ement|ements|ing)|thanks?\s+(?:are|is|goes?)\s+to)\b",
    re.IGNORECASE,
)
_STYLE_PROTOCOL_RE = re.compile(
    r"(?:^|\s)(?:source|target|translation|rationale|decision)\s*[:=]|"
    r"```|</?(?:analysis|answer|tool|assistant)>|\{\s*\"",
    re.IGNORECASE,
)
_UNTRANSLATED_CITATION_PROSE_RE = re.compile(
    r"\((?:[^()]*(?:\bsee\b|\bcf\.|\be\.g\.|\bi\.e\.|"
    r"\bon (?:these|this|the)\b)[^()]*)\)",
    re.IGNORECASE,
)
_AUTHOR_YEAR_CITATION_RE = re.compile(
    r"\([^()\n]*(?:1[5-9]\d{2}|20\d{2})[a-z]?[^()\n]*\)",
    re.IGNORECASE,
)
_LEADING_NOTE_MARKER_RE = re.compile(
    r"^(?:\[?\d{1,3}\]?|[\u00b9\u00b2\u00b3\u2070-\u2079])\s+"
)


def _numeric_score(value: Any) -> float:
    """Coerce persisted quality metadata without trusting checkpoint types."""
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _summary_lexical_near_misses(
    candidate_persian: str,
    trusted_persian: str,
) -> list[dict[str, str]]:
    """Find high-confidence one-letter drift from already accepted wording."""
    token_re = re.compile(r"[\u0621-\u063a\u0641-\u064a\u066e-\u06d3\u06fa-\u06ff\u200c]+")

    def normalized_tokens(value: str) -> set[str]:
        return {
            re.sub(r"[\u064b-\u065f\u0670\u06d6-\u06ed\u200c]", "", token)
            for token in token_re.findall(value or "")
            if len(re.sub(r"[\u064b-\u065f\u0670\u06d6-\u06ed\u200c]", "", token)) >= 8
        }

    trusted = normalized_tokens(trusted_persian)
    candidate = normalized_tokens(candidate_persian)
    findings: list[dict[str, str]] = []
    for token in sorted(candidate - trusted):
        matches = [
            established for established in trusted
            if len(established) == len(token)
            and token[:3] == established[:3]
            and token[-3:] == established[-3:]
            and sum(
                left != right
                for left, right in zip(token, established, strict=True)
            ) == 1
        ]
        if len(matches) == 1:
            findings.append({
                "candidate": token,
                "established": matches[0],
                "reason": "one_letter_drift_from_accepted_persian",
            })
    return findings[:12]


def _summary_english_prefix_stutters(
    candidate_english: str,
    evidence_english: str,
) -> list[dict[str, str]]:
    """Detect adjacent partial-word restarts absent from source evidence."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]{3,}", candidate_english or "")
    evidence = " ".join((evidence_english or "").casefold().split())
    findings: list[dict[str, str]] = []
    for left, right in zip(words, words[1:], strict=False):
        first = left.casefold().strip("'-")
        second = right.casefold().strip("'-")
        shorter, longer = sorted((first, second), key=len)
        phrase = f"{first} {second}"
        if (
            len(shorter) >= 5
            and shorter != longer
            and longer.startswith(shorter)
            and phrase not in evidence
        ):
            findings.append({
                "candidate": f"{left} {right}",
                "reason": "adjacent_prefix_restart_absent_from_source",
            })
    return findings[:12]


def _clean_style_sample(text: str) -> str:
    """Return a safe style-only sample without mutating persisted memory."""
    sample = _complete_style_sample(text)
    if not sample or _STYLE_PROTOCOL_RE.search(sample):
        return ""
    if sample.lstrip().startswith(("...", "\u2026")) or sample.rstrip().endswith(
        ("...", "\u2026")
    ):
        return ""
    if _UNTRANSLATED_CITATION_PROSE_RE.search(sample):
        return ""
    if mixed_script_artifacts(sample):
        return ""
    if (
        foreign_script_artifacts("", sample)
        or markup_wrapper_artifacts("", sample)
        or parenthesis_artifacts("", sample)
        or detached_ezafe_artifacts(sample)
        or repeated_persian_clause_artifacts(sample)
        or repeated_persian_word_artifacts(sample)
        or tatweel_separator_artifacts(sample)
    ):
        return ""
    persian_chars = len(re.findall(r"[\u0600-\u06ff]", sample))
    latin_words = len(re.findall(r"\b[A-Za-z]{3,}\b", sample))
    if persian_chars < 8 or latin_words > max(12, persian_chars // 18):
        return ""
    return sample


def _style_sample_quality(text: str) -> dict[str, Any]:
    """Score fluent-academic style evidence without choosing terminology.

    The thresholds reject paragraphs that are technically clean but too dense
    to serve as a reusable voice anchor. They do not reject complexity in the
    translation itself; this function controls style-memory admission only.
    """
    sample = _clean_style_sample(text)
    if not sample:
        return {"approved": False, "score": 0.0, "reasons": ["surface_artifact"]}
    sentences = [
        value.strip() for value in re.split(r"(?<=[.!?\u061f])\s+", sample)
        if value.strip()
    ] or [sample]
    word_counts = [
        len(re.findall(r"[\u0600-\u06ffA-Za-z0-9]+", sentence))
        for sentence in sentences
    ]
    maximum_words = max(word_counts, default=0)
    punctuation = len(re.findall(r"[,،;؛:]", sample))
    punctuation_per_sentence = punctuation / max(1, len(sentences))
    parenthetical_chars = sum(
        len(match.group()) for match in re.finditer(r"\([^()]*\)", sample)
    )
    parenthetical_ratio = parenthetical_chars / max(1, len(sample))
    reasons: list[str] = []
    if maximum_words > 65:
        reasons.append("sentence_too_dense_for_style_anchor")
    if punctuation_per_sentence > 8:
        reasons.append("punctuation_stack_too_dense")
    if parenthetical_ratio > 0.24:
        reasons.append("parenthetical_content_dominates_sample")
    score = max(
        0.0,
        100.0
        - max(0, maximum_words - 34) * 1.15
        - max(0.0, punctuation_per_sentence - 3.0) * 4.0
        - parenthetical_ratio * 45.0,
    )
    return {
        "approved": not reasons,
        "score": round(score, 2),
        "reasons": reasons,
        "sentence_count": len(sentences),
        "maximum_sentence_words": maximum_words,
        "punctuation_per_sentence": round(punctuation_per_sentence, 2),
        "parenthetical_ratio": round(parenthetical_ratio, 4),
        "sample": sample,
    }


def _complete_style_sample(text: str, preferred_limit: int = 900) -> str:
    """Select complete early sentences without cutting a sample mid-sentence."""
    normalized = " ".join((text or "").split())
    if not normalized:
        return ""
    sentences = re.findall(r".*?(?:[.!?\u061f]+(?:[\"'\u00bb)]*)|$)", normalized)
    complete = [
        sentence.strip() for sentence in sentences
        if sentence.strip() and re.search(r"[.!?\u061f][\"'\u00bb)]*$", sentence.strip())
    ]
    clean_complete = [
        sentence for sentence in complete
        if not _AUTHOR_YEAR_CITATION_RE.search(sentence)
        and not _LEADING_NOTE_MARKER_RE.search(sentence)
    ]
    if not complete:
        return normalized if len(normalized) <= preferred_limit else ""
    if not clean_complete:
        return ""
    complete = clean_complete

    selected: list[str] = []
    for sentence in complete:
        candidate = " ".join(selected + [sentence])
        if selected and len(candidate) > preferred_limit:
            break
        if not selected and len(sentence) > 1400:
            return ""
        selected.append(sentence)
        if len(candidate) >= preferred_limit:
            break
    return " ".join(selected)


@dataclass
class MemoryContext:
    """Consolidated context from all four memory layers."""

    book_context: str = ""
    style_profile: str = ""
    proper_nouns: str = ""
    bilingual_summary: str = ""
    long_term: str = ""
    short_term: str = ""
    references: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.references is None:
            self.references = {}

    def format(self) -> str:
        """Format the memory layers into a single prompt-ready string."""
        parts = []
        if self.book_context:
            parts.append(
                "--- Reviewable Book Research Context ---"
                + chr(10)
                + self.book_context
            )
        if self.style_profile:
            parts.append(f"--- Style Profile: Book-Level Voice Guide ---\n{self.style_profile}")
        if self.proper_nouns:
            parts.append(f"--- Layer 1: Proper Nouns & Transliterations ---\n{self.proper_nouns}")
        if self.bilingual_summary:
            parts.append(f"--- Layer 2: Bilingual Summary ---\n{self.bilingual_summary}")
        if self.long_term:
            parts.append(f"--- Layer 3: Relevant Past Translations ---\n{self.long_term}")
        if self.short_term:
            parts.append(f"--- Layer 4: Short-term Context ---\n{self.short_term}")
        return "\n\n".join(parts) if parts else "No translation memory context available."


class MemoryManager:
    """Coordinates style profile, proper nouns, summaries, and translation memory."""

    def __init__(self, config: TarjomehConfig) -> None:
        self.config = config
        self.proper_nouns = ProperNouns()
        self.bilingual_summary = BilingualSummary()
        self.book_context = ""
        self.style_profile = ""
        self.style_samples: list[str] = []
        self.style_sample_records: list[dict[str, Any]] = []

        # Pull retrieval_k and window_size from config
        retrieval_k = config.to_dict().get("memory", {}).get("long_term_retrieval_k", 5)
        window_size = config.to_dict().get("memory", {}).get("short_term_window", 4)
        try:
            configured_style_floor = float(
                config.to_dict().get("memory", {}).get("style_min_score", 75.0)
            )
        except (TypeError, ValueError):
            configured_style_floor = 75.0
        self._style_min_score = min(100.0, max(0.0, configured_style_floor))
        self._style_min_representative_samples = max(
            1,
            min(
                5,
                int(getattr(
                    config.memory, "style_min_representative_samples", 3
                ) or 3),
            ),
        )

        self.long_term = LongTermMemory(retrieval_k=retrieval_k)
        self.short_term = ShortTermMemory(window_size=window_size)

    def get_context_for_chunk(self, chunk: Chunk) -> MemoryContext:
        """Retrieve relevant context for translating the given chunk."""
        # Layer 1: Proper Nouns
        note_mode = effective_term_notes_mode(
            self.config.output.term_notes,
            self.config.output.format,
        )
        include_inline_originals = note_mode in (
            "inline",
            "both",
        )
        proper_nouns_str = self.proper_nouns.get_context(
            include_inline_originals=include_inline_originals,
            source_text=chunk.text,
        )

        # Layer 2: Bilingual Summary
        bilingual_summary_str = self.bilingual_summary.get_context()

        # Layer 3: Long-term Memory (TF-IDF)
        relevant_long_term = self.long_term.get_relevant(chunk.text)
        long_term_blocks = []
        for pair in relevant_long_term:
            guidance = (
                "[retrieval: reliable prose]"
                if pair.get("reliable", True)
                else (
                    "[retrieval: advisory continuity; preserve the argument, "
                    "but do not treat wording as terminology or style authority]"
                )
            )
            long_term_blocks.append(
                f"{guidance}\nEN: {pair['source']}\nFA: {pair['translation']}"
            )
        long_term_str = "\n\n".join(long_term_blocks)

        # Layer 4: Short-term Memory (Window)
        pairs = self.short_term.get_entries()
        short_term_str = ""
        if pairs:
            formatted_pairs = []
            for pair in pairs:
                if pair.trust == "advisory_review":
                    guidance = (
                        "[continuity: needs review; preserve argument and references, "
                        "but treat disputed wording as advisory]"
                    )
                elif pair.trust == "structural_only":
                    guidance = (
                        "[structural context only; do not imitate as prose style or "
                        "mandatory terminology]"
                    )
                else:
                    guidance = "[continuity: trusted prose]"
                chapter = (
                    f" Chapter: {pair.chapter_title}." if pair.chapter_title else ""
                )
                formatted_pairs.append(
                    f"{guidance}{chapter}\nEN: {pair.source}\nFA: {pair.translation}"
                )
            short_term_str = "\n\n".join(formatted_pairs)

        prompt_style_profile = (
            self._render_style_profile() or self._legacy_style_profile_for_prompt()
        )
        return MemoryContext(
            book_context=self.book_context,
            style_profile=prompt_style_profile,
            proper_nouns=proper_nouns_str,
            bilingual_summary=bilingual_summary_str,
            long_term=long_term_str,
            short_term=short_term_str,
            references={
                "style_profile_version": len(
                    [sample for sample in self.style_samples if _clean_style_sample(sample)]
                ),
                "style_profile_status": self._style_profile_status(),
                "long_term_entry_ids": [
                    pair.get("entry_id") for pair in relevant_long_term
                ],
                "long_term_trust": [
                    "reliable" if pair.get("reliable", True) else "advisory"
                    for pair in relevant_long_term
                ],
                "short_term_window_size": len(pairs),
                "short_term_trust": [pair.trust for pair in pairs],
                "proper_noun_count": len(self.proper_nouns),
                "has_bilingual_summary": bool(bilingual_summary_str),
            },
        )

    def update_after_translation(
        self,
        chunk: Chunk,
        translation: str,
        *,
        quality_approved: bool = True,
        style_approved: bool | None = None,
        long_term_reliable: bool | None = None,
        short_term_trust: str | None = None,
        reliability_reasons: list[str] | None = None,
        style_excluded_paragraphs: list[int] | None = None,
        style_evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update synchronous memory layers with a new source-translation pair."""
        body_indices = list(
            chunk.metadata.get("style_body_paragraphs", []) or []
        )
        has_body_policy = "style_body_paragraphs" in chunk.metadata
        structure_eligible = bool(
            body_indices if has_body_policy
            else chunk.metadata.get("style_eligible", True)
        ) and not _NON_PROSE_RE.search(chunk.text or "")
        durable_quality = (
            bool(quality_approved)
            if long_term_reliable is None
            else bool(long_term_reliable)
        )
        resolved_long_term_reliable = bool(durable_quality and structure_eligible)
        structural_roles = list(
            chunk.metadata.get("structural_roles", []) or []
        )
        primary_role = (
            structural_roles[0] if len(set(structural_roles)) == 1
            else "mixed"
        ) if structural_roles else "body"
        default_short_term_trust = (
            "trusted"
            if resolved_long_term_reliable
            else ("advisory_review" if structure_eligible else "structural_only")
        )
        resolved_short_term_trust = (
            short_term_trust
            if short_term_trust in {"trusted", "advisory_review", "structural_only"}
            else default_short_term_trust
        )
        if not structure_eligible:
            resolved_short_term_trust = "structural_only"
        canonical_target_hash = hashlib.sha256(
            translation.strip().encode("utf-8")
        ).hexdigest()
        self.short_term.add(
            chunk.text,
            translation,
            trust=resolved_short_term_trust,
            structural_role=primary_role,
            chapter_title=chunk.chapter_title,
        )
        self.long_term.add(
            chunk.text,
            translation,
            reliable=resolved_long_term_reliable,
            chapter_title=chunk.chapter_title,
            chunk_index=chunk.index,
            canonical_target_hash=canonical_target_hash,
        )
        has_structure_policy = "style_eligible" in chunk.metadata
        translation_paragraphs = [
            paragraph.strip() for paragraph in translation.split("\n\n")
            if paragraph.strip()
        ]
        source_paragraphs = [
            paragraph.strip() for paragraph in re.split(r"\n\s*\n", chunk.text or "")
            if paragraph.strip()
        ]
        style_translation = translation
        style_source_indices = list(range(len(translation_paragraphs)))
        excluded_style_indices = {
            int(index) for index in (style_excluded_paragraphs or [])
            if isinstance(index, int) or str(index).isdigit()
        }
        source_genre_excluded_indices: set[int] = set()
        if len(source_paragraphs) == len(translation_paragraphs):
            source_genre_excluded_indices = {
                index for index, paragraph in enumerate(source_paragraphs)
                if _NONREPRESENTATIVE_STYLE_SOURCE_RE.search(paragraph)
            }
            excluded_style_indices.update(source_genre_excluded_indices)
        if not has_body_policy and excluded_style_indices:
            style_source_indices = [
                index for index in style_source_indices
                if index not in excluded_style_indices
            ]
            style_translation = "\n\n".join(
                translation_paragraphs[index] for index in style_source_indices
            )
        if has_body_policy:
            if body_indices and max(body_indices) < len(translation_paragraphs):
                style_source_indices = [
                    index for index in body_indices
                    if index not in excluded_style_indices
                ]
                style_translation = "\n\n".join(
                    translation_paragraphs[index] for index in style_source_indices
                )
            elif not chunk.metadata.get("style_eligible", False):
                style_translation = ""
                style_source_indices = []
        style_quality_approved = (
            bool(quality_approved)
            if style_approved is None else bool(style_approved)
        )
        style_eligible = bool(
            structure_eligible
            and style_quality_approved
            and style_translation.strip()
            and (
                not has_structure_policy
                or (
                    len((chunk.text or "").strip()) >= 240
                    and len((translation or "").strip()) >= 160
                )
            )
        )
        style_count_before = len(self.style_samples)
        style_sample_policy: dict[str, Any] = {
            "candidate_count": 0,
            "accepted": False,
            "reason": "style_not_eligible",
        }
        if source_genre_excluded_indices and not style_translation.strip():
            style_sample_policy["reason"] = (
                "source_genre_not_representative_of_body_voice"
            )
        if style_eligible:
            style_sample_policy = self._update_style_profile(
                style_translation,
                source_paragraph_indices=style_source_indices,
                paragraph_role=primary_role,
                book_genre=self._book_genre(),
                source_chunk_index=chunk.index,
                representative=primary_role in {
                    "body", "academic_argument", "expository_nonfiction",
                    "narrative_prose", "dialogue",
                },
                final_scores=dict((style_evidence or {}).get("final_scores", {})),
                unresolved_issue_paragraphs=dict(
                    (style_evidence or {}).get(
                        "unresolved_issue_paragraphs", {}
                    ) or {}
                ),
            )
            selected = style_sample_policy.get("selected", {}) or {}
            local_index = selected.get("paragraph_index")
            if isinstance(local_index, int) and local_index < len(style_source_indices):
                style_sample_policy["source_paragraph_index"] = (
                    style_source_indices[local_index]
                )
        # Any known proper noun occurring in this chunk has now had its first
        # appearance — later chunks must not repeat the English parenthetical.
        self.proper_nouns.mark_introduced_from_translation(
            chunk.text, translation
        )
        return {
            "short_term_added": True,
            "short_term_trust": resolved_short_term_trust,
            "continuity_retained": True,
            "long_term_added": True,
            "long_term_reliable": resolved_long_term_reliable,
            "long_term_trust": (
                "reliable" if resolved_long_term_reliable else "advisory"
            ),
            "canonical_target_hash": canonical_target_hash,
            "canonical_chunk_index": chunk.index,
            "reliability_reasons": list(dict.fromkeys(
                str(reason).strip()
                for reason in (reliability_reasons or [])
                if str(reason).strip()
            )),
            "style_sample_added": len(self.style_samples) > style_count_before,
            "style_sample_policy": style_sample_policy,
            "structure_eligible": structure_eligible,
            "quality_approved": bool(quality_approved),
            "style_approved": style_quality_approved,
            "style_excluded_paragraphs": sorted(excluded_style_indices),
            "style_source_genre_excluded_paragraphs": sorted(
                source_genre_excluded_indices
            ),
            "structural_roles": structural_roles,
        }

    def _book_genre(self) -> str:
        """Return a stable broad genre without an extra classification call."""
        register = str(self.config.translation.style_register or "").casefold()
        mode = str(self.config.translation.mode or "").casefold()
        if "literary" in register:
            return "literary"
        if mode == "academic" or "academic" in register:
            return "academic"
        return "general"

    def _ensure_style_sample_records(self) -> None:
        """Backfill metadata for checkpoints that predate evidence records."""
        known_hashes = {
            str(record.get("text_hash", ""))
            for record in self.style_sample_records
            if isinstance(record, dict)
        }
        for sample in self.style_samples:
            digest = hashlib.sha256(sample.encode("utf-8")).hexdigest()
            if digest in known_hashes:
                continue
            quality = _style_sample_quality(sample)
            self.style_sample_records.append({
                "text": sample,
                "text_hash": digest,
                "paragraph_role": "legacy_unknown",
                "book_genre": "unknown",
                "representative": False,
                "fallback": True,
                "quality_score": float(quality.get("score", 0.0)),
                "source_chunk_index": None,
                "source_paragraph_index": None,
                "reasons": ["legacy_sample_without_role_metadata"],
            })
            known_hashes.add(digest)

    def _style_profile_status(self) -> str:
        self._ensure_style_sample_records()
        representative = sum(
            bool(record.get("representative"))
            and bool(_clean_style_sample(str(record.get("text", ""))))
            and _numeric_score(record.get("quality_score")) >= self._style_min_score
            for record in self.style_sample_records
        )
        if representative >= self._style_min_representative_samples:
            return "established"
        return "warming_up" if self.style_samples else "empty"

    def _update_style_profile(
        self,
        translation: str,
        *,
        source_paragraph_indices: list[int] | None = None,
        paragraph_role: str = "body",
        book_genre: str = "general",
        source_chunk_index: int | None = None,
        representative: bool = True,
        final_scores: dict[str, Any] | None = None,
        unresolved_issue_paragraphs: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Maintain a compact book-level style guide from early translations."""
        candidates: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        for paragraph_index, paragraph in enumerate(
            re.split(r"\n\s*\n", translation or "")
        ):
            if not paragraph.strip():
                continue
            source_paragraph_index = (
                source_paragraph_indices[paragraph_index]
                if source_paragraph_indices
                and paragraph_index < len(source_paragraph_indices)
                else paragraph_index
            )
            quality = _style_sample_quality(paragraph)
            quality["paragraph_index"] = paragraph_index
            unresolved_here = [
                str(issue_id)
                for issue_id, issue_paragraph in dict(
                    unresolved_issue_paragraphs or {}
                ).items()
                if int(issue_paragraph) == source_paragraph_index
            ]
            if unresolved_here:
                quality = {
                    **quality,
                    "approved": False,
                    "reasons": list(quality.get("reasons", []) or [])
                    + ["unresolved_final_quality_issue"],
                    "unresolved_issue_ids": unresolved_here,
                }
                rejections.append(quality)
                continue
            if (
                quality.get("approved")
                and float(quality.get("score", 0.0)) >= self._style_min_score
            ):
                candidates.append(quality)
            else:
                if (
                    quality.get("approved")
                    and float(quality.get("score", 0.0)) < self._style_min_score
                ):
                    quality = {
                        **quality,
                        "approved": False,
                        "reasons": list(quality.get("reasons", []) or [])
                        + ["style_score_below_floor"],
                    }
                rejections.append(quality)

        report: dict[str, Any] = {
            "candidate_count": len(candidates),
            "accepted": False,
            "reason": "no_fluent_complete_paragraph",
            "rejections": rejections[:5],
            "minimum_score": self._style_min_score,
        }
        self._ensure_style_sample_records()
        if candidates:
            selected = max(candidates, key=lambda item: float(item["score"]))
            text = str(selected["sample"])
            local_index = int(selected.get("paragraph_index", 0) or 0)
            source_paragraph_index = (
                source_paragraph_indices[local_index]
                if source_paragraph_indices
                and local_index < len(source_paragraph_indices)
                else local_index
            )
            record = {
                "text": text,
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "paragraph_role": paragraph_role,
                "book_genre": book_genre,
                "representative": bool(representative),
                "fallback": not bool(representative),
                "quality_score": float(selected.get("score", 0.0)),
                "source_chunk_index": source_chunk_index,
                "source_paragraph_index": source_paragraph_index,
                "canonical_paragraph_hash": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
                "final_scores": {
                    key: _numeric_score(value)
                    for key, value in dict(final_scores or {}).items()
                    if key in {"accuracy", "fluency", "terminology", "register"}
                },
                "unresolved_issue_ids": [
                    str(issue_id)
                    for issue_id, issue_paragraph in dict(
                        unresolved_issue_paragraphs or {}
                    ).items()
                    if int(issue_paragraph) == source_paragraph_index
                ],
                "reasons": [] if representative else [
                    "quality_approved_body_role_uncertain"
                ],
            }
            duplicate = any(
                text == existing
                or (
                    len(text) >= 80
                    and len(existing) >= 80
                    and text[:80] == existing[:80]
                )
                for existing in self.style_samples
            )
            if duplicate:
                report.update({
                    "reason": "duplicate_style_evidence",
                    "selected": selected,
                })
            elif len(self.style_samples) < 5:
                self.style_samples.append(text)
                self.style_sample_records.append(record)
                report.update({
                    "accepted": True,
                    "reason": "quality_approved_sample_added",
                    "selected": selected,
                })
            else:
                existing_quality = [
                    _style_sample_quality(sample) for sample in self.style_samples
                ]
                weakest_index = min(
                    range(len(existing_quality)),
                    key=lambda index: float(existing_quality[index].get("score", 0.0)),
                )
                weakest_score = float(
                    existing_quality[weakest_index].get("score", 0.0)
                )
                fallback_indices = [
                    index for index, existing in enumerate(
                        self.style_sample_records[:len(self.style_samples)]
                    )
                    if bool(existing.get("fallback"))
                ]
                replace_fallback = bool(representative and fallback_indices)
                if replace_fallback:
                    weakest_index = min(
                        fallback_indices,
                        key=lambda index: _numeric_score(
                            self.style_sample_records[index].get(
                                "quality_score", 0.0
                            )
                        ),
                    )
                    weakest_score = _numeric_score(
                        self.style_sample_records[weakest_index].get(
                            "quality_score", 0.0
                        )
                    )
                if replace_fallback or float(selected["score"]) >= weakest_score + 8.0:
                    self.style_samples[weakest_index] = text
                    self.style_sample_records[weakest_index] = record
                    report.update({
                        "accepted": True,
                        "reason": "weaker_style_sample_replaced",
                        "replaced_index": weakest_index,
                        "previous_score": weakest_score,
                        "selected": selected,
                    })
                else:
                    report.update({
                        "reason": "existing_style_set_is_not_weaker",
                        "selected": selected,
                        "weakest_existing_score": weakest_score,
                    })

        self.style_profile = self._render_style_profile()
        report["profile_status"] = self._style_profile_status()
        report["representative_sample_count"] = sum(
            bool(record.get("representative"))
            for record in self.style_sample_records
        )
        return report

    def _render_style_profile(self) -> str:
        """Build a prompt-safe style guide from trusted prose samples."""
        self._ensure_style_sample_records()
        ordered_records = sorted(
            self.style_sample_records,
            key=lambda record: (
                not bool(record.get("representative")),
                -_numeric_score(record.get("quality_score")),
            ),
        )
        representative_count = sum(
            bool(record.get("representative"))
            and bool(_clean_style_sample(str(record.get("text", ""))))
            and _numeric_score(record.get("quality_score")) >= self._style_min_score
            for record in ordered_records
        )
        active_records = (
            [record for record in ordered_records if record.get("representative")]
            if representative_count >= self._style_min_representative_samples
            else ordered_records
        )
        clean_samples = [
            cleaned
            for record in active_records
            if (cleaned := _clean_style_sample(str(record.get("text", ""))))
            and float(_style_sample_quality(cleaned).get("score", 0.0))
            >= self._style_min_score
        ][:5]
        if not clean_samples:
            return ""
        samples = "\n".join(
            f"{i + 1}. {sample}"
            for i, sample in enumerate(clean_samples)
        )
        genre = next((
            str(record.get("book_genre", "general"))
            for record in ordered_records
            if record.get("representative")
        ), self._book_genre())
        genre_guidance = {
            "academic": (
                "Preserve argument structure, conceptual parallelism, explicit "
                "logical relations, and integrated scholarly citations."
            ),
            "literary": (
                "Preserve narrative voice, point of view, dialogue register, "
                "imagery, and intentional rhythm."
            ),
        }.get(genre, "Preserve the source genre's register, voice, and paragraph rhythm.")
        status = self._style_profile_status()
        return (
            "Maintain one coherent scholarly Iranian-Persian voice across the book. "
            "Prefer formal academic diction, precise conceptual renderings, stable "
            "citation handling, and the sentence rhythm established in the samples below. "
            "Do not simplify later chapters into a different register. These samples govern "
            "register and rhythm only: they are not terminology authority, and the source, "
            "curated glossary, and current context override every lexical choice. Do not copy "
            "transliteration artifacts or untranslated citation prose from a sample. "
            f"Genre focus: {genre_guidance} Profile status: {status}.\n\n"
            f"Representative early translation samples:\n{samples}"
        )

    def _legacy_style_profile_for_prompt(self) -> str:
        """Keep old checkpoints usable when they predate persisted samples."""
        value = str(self.style_profile or "").strip()
        if self.style_samples or not value:
            return ""
        if _STYLE_PROTOCOL_RE.search(value) or _UNTRANSLATED_CITATION_PROSE_RE.search(value):
            return ""
        return (
            value
            + "\n\nLegacy profile safety: use this for register and rhythm only; "
            "the source, curated glossary, and current context override lexical choices."
        )

    async def update_proper_nouns(
        self,
        llm_client: Any,
        text: str,
        translation: str = "",
        *,
        source_categories: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Incrementally identify new proper nouns in the text and add them."""
        from tarjomeh.core.prompts import INCREMENTAL_NER_PROMPT, GLOSSARY_EXTRACT_PROMPT

        note_mode = effective_term_notes_mode(
            self.config.output.term_notes,
            self.config.output.format,
        )
        include_inline_originals = note_mode in (
            "inline",
            "both",
        )
        known = self.proper_nouns.get_context(
            include_inline_originals=include_inline_originals
        )
        if not known:
            # First chapter/TOC extract
            prompt = GLOSSARY_EXTRACT_PROMPT.format(text=text)
        else:
            prompt = INCREMENTAL_NER_PROMPT.format(known_entities=known, text=text)
        if translation.strip():
            prompt += (
                "\n\n### Accepted Persian translation\n"
                + translation[:12000]
                + "\n\nFor a proper name, publication, organization, product, or "
                  "named theory that is visibly rendered in the accepted Persian "
                  "translation, copy that exact Persian rendering into "
                  "suggested_persian. Do not infer a different spelling."
            )

        try:
            if hasattr(llm_client, "set_operation"):
                llm_client.set_operation(
                    "proper_noun_incremental" if known else "proper_noun_initial"
                )
            response = await llm_client.chat(prompt)
            items = parse_structured_output(response, expected=(list, dict))
            candidates = items if isinstance(items, list) else (
                items.get("terms", []) or items.get("extracted_terms", []) or []
            )
            accepted = 0
            observed = 0
            aliases_added = 0
            rejected_details: list[dict[str, str]] = []
            inline_categories = {
                "proper_noun", "person", "place", "institution",
                "organization", "organisation", "publication", "product",
                "theory", "named_theory", "approved_term", "book", "article",
                "journal", "work", "technical_loanword", "loanword",
            }
            normalized_translation = re.sub(
                r"[\s\u200c]+", " ", translation.strip()
            ).casefold()
            category_by_source = {
                " ".join(str(source).split()).casefold(): str(value)
                for source, value in (source_categories or {}).items()
                if str(source).strip() and str(value).strip()
            }
            for item in candidates if isinstance(candidates, list) else []:
                if not isinstance(item, dict):
                    continue
                term = str(item.get("term", "")).strip()
                persian = str(item.get("suggested_persian", "")).strip()
                category = str(item.get("category", "term"))
                source_category = category_by_source.get(
                    " ".join(term.split()).casefold(), ""
                )
                if source_category in {
                    "person", "place", "organization", "institution",
                    "legal_instrument", "source_grounded_entity",
                }:
                    category = source_category
                if not is_usable_memory_mapping(term, persian):
                    rejected_details.append({
                        "source": term,
                        "reason": "unusable_memory_mapping",
                    })
                    continue
                effective_category = low_authority_mapping_category(
                    term, persian, category
                )
                if (
                    effective_category == "term"
                    and not has_minimal_automatic_term_evidence(
                        item,
                        translation=translation,
                    )
                ):
                    rejected_details.append({
                        "source": term,
                        "reason": "insufficient_minimal_lexical_evidence",
                    })
                    continue
                rendered = bool(
                    normalized_translation
                    and re.sub(r"[\s\u200c]+", " ", persian).casefold()
                    in normalized_translation
                )
                inline_category = (
                    category.strip().lower().replace("-", "_")
                    in inline_categories
                )
                observed_rendering = bool(
                    rendered
                    and inline_category
                    and has_exact_bilingual_anchor(
                        translation, term, persian, category
                    )
                )
                if (
                    is_automatic_entity_category(category)
                    and not observed_rendering
                ):
                    # A substring in accepted prose is not proof of a complete
                    # entity rendering. Current-chunk anchor reconciliation owns
                    # durable Persian (English) evidence; defer anything else.
                    rejected_details.append({
                        "source": term,
                        "reason": "missing_exact_bilingual_anchor",
                    })
                    continue
                if not is_safe_automatic_entity_mapping(
                    term,
                    persian,
                    category,
                    text,
                    translation=translation,
                ):
                    rejected_details.append({
                        "source": term,
                        "reason": "unsafe_automatic_mapping",
                    })
                    continue
                provenance = (
                    "observed_translation" if observed_rendering
                    else "incremental_extraction"
                )
                outcome = self.proper_nouns.add_noun(
                    term,
                    persian,
                    category=category,
                    provenance=provenance,
                    evidence_key=(
                        "chunk:"
                        + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                    ),
                    context_independent=bool(item.get("context_independent")),
                )
                if outcome.get("action") == "ignored":
                    continue
                if observed_rendering and provenance == "observed_translation":
                    observed += 1
                    if outcome.get("action") == "preserved_higher_authority":
                        aliases_added += int(
                            self.proper_nouns.add_alias(term, persian)
                        )
                accepted += 1
            self.proper_nouns.mark_introduced_from_translation(
                text, translation
            )
            return {
                "status": (
                    "completed" if accepted else "completed_without_suggestions"
                ),
                "candidate_count": len(candidates),
                "accepted_count": accepted,
                "observed_translation_count": observed,
                "alias_count": aliases_added,
                "rejected_count": max(0, len(candidates) - accepted),
                "rejected_details": rejected_details[:50],
            }
        except Exception as exc:
            logger.warning("Failed to extract proper nouns incrementally: %s", exc)
            return {
                "status": "failed",
                "candidate_count": 0,
                "accepted_count": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def reconcile_bilingual_summary(self) -> dict[str, Any]:
        """Remove stale automatic renderings contradicted by accepted corrections."""
        replacements: list[tuple[str, str]] = []
        for source, target in self.proper_nouns.all_nouns().items():
            provenance = self.proper_nouns.provenance_for(source)
            if provenance.get("origin") != "accepted_correction":
                continue
            for record in list(provenance.get("superseded", []) or []):
                if not isinstance(record, dict):
                    continue
                previous = str(record.get("target", "")).strip()
                if previous and previous != target:
                    replacements.append((previous, target))
        return self.bilingual_summary.reconcile_persian_terms(replacements)

    async def update_bilingual_summary(
        self,
        llm_client: Any,
        new_content: str,
        translation: str,
        *,
        input_trust: str = "advisory_inputs",
        trust_reasons: list[str] | None = None,
    ) -> dict[str, Any]:
        """Call the LLM to update the running bilingual summary of translated chapters."""
        from tarjomeh.core.prompts import SUMMARY_UPDATE_PROMPT

        current = self.bilingual_summary.get_context() or "None"
        prompt = SUMMARY_UPDATE_PROMPT.format(
            current_summary=current,
            new_content=new_content,
            translation=translation,
        )

        try:
            if hasattr(llm_client, "set_operation"):
                llm_client.set_operation("bilingual_summary_update")
            response = await llm_client.chat(prompt)
            candidate = BilingualSummary()
            candidate.update(response)
            candidate_quality = candidate.candidate_quality()
            from tarjomeh.quality.structure_audit import audit_payload

            alignment = audit_payload(
                candidate.english_summary,
                candidate.persian_summary,
            )
            lexical_near_misses = _summary_lexical_near_misses(
                candidate.persian_summary,
                self.bilingual_summary.persian_summary + "\n" + translation,
            )
            english_prefix_stutters = _summary_english_prefix_stutters(
                candidate.english_summary,
                self.bilingual_summary.english_summary + "\n" + new_content,
            )
            extra_reasons: list[str] = []
            if "translation_structure_mismatch" in set(
                alignment.get("classifications", []) or []
            ):
                extra_reasons.append("bilingual_summary_structure_mismatch")
            if lexical_near_misses:
                extra_reasons.append("persian_summary_lexical_near_miss")
            if english_prefix_stutters:
                extra_reasons.append("english_summary_lexical_stutter")
            if extra_reasons:
                prior_reasons = candidate_quality.get("reasons", [])
                if not isinstance(prior_reasons, list):
                    prior_reasons = []
                candidate_quality["accepted"] = False
                candidate_quality["reasons"] = list(dict.fromkeys(
                    [str(reason) for reason in prior_reasons if str(reason)]
                    + extra_reasons
                ))
            candidate_quality["bilingual_alignment"] = alignment
            candidate_quality["lexical_near_misses"] = lexical_near_misses
            candidate_quality["english_prefix_stutters"] = (
                english_prefix_stutters
            )
            if not candidate_quality["accepted"]:
                return {
                    "replacement_count": 0,
                    "changes": [],
                    "candidate_quality": candidate_quality,
                    "input_trust": self.bilingual_summary.input_trust,
                    "trust_reasons": list(
                        self.bilingual_summary.trust_reasons
                    ),
                    "context_retained": bool(current != "None"),
                    "candidate_committed": False,
                    "authority": "argument_orientation_only",
                }
            candidate.set_input_trust(input_trust, trust_reasons)
            self.bilingual_summary = candidate
            report = self.reconcile_bilingual_summary()
            report.update({
                "input_trust": self.bilingual_summary.input_trust,
                "trust_reasons": list(self.bilingual_summary.trust_reasons),
                "context_retained": True,
                "candidate_committed": True,
                "candidate_quality": candidate_quality,
                "authority": "argument_orientation_only",
            })
            return report
        except Exception as exc:
            logger.warning("Failed to update running bilingual summary: %s", exc)
            return {
                "replacement_count": 0,
                "changes": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

    def to_dict(self) -> dict[str, Any]:
        """Serialize memory state for database checkpointing."""
        return {
            "book_context": self.book_context,
            "style_profile": self.style_profile,
            "style_samples": self.style_samples,
            "style_sample_records": self.style_sample_records,
            "proper_nouns": self.proper_nouns.serialize(),
            "bilingual_summary": self.bilingual_summary.serialize(),
            "past_translations": self.long_term.serialize(),
            "short_term_context": self.short_term.serialize(),
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore memory state from a checkpoint dictionary."""
        self.book_context = str(data.get("book_context", ""))
        self.style_profile = str(data.get("style_profile", ""))
        self.style_samples = list(data.get("style_samples", []))
        raw_records = data.get("style_sample_records", [])
        self.style_sample_records = [
            dict(record) for record in list(raw_records or [])
            if isinstance(record, dict)
        ]
        self._ensure_style_sample_records()
        self.proper_nouns.deserialize(data.get("proper_nouns", {}))
        self.bilingual_summary.deserialize(data.get("bilingual_summary", {}))
        self.long_term.deserialize(data.get("past_translations", []))
        self.short_term = ShortTermMemory.deserialize(
            data.get("short_term_context", []),
            window_size=self.short_term.window_size
        )
