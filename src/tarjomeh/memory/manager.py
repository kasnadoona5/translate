"""Memory manager coordinating the translation memory system.

Integrates a book-level style profile, proper noun records, bilingual summary,
long-term memory, and short-term memory for cohesive book-length translation.
"""

from __future__ import annotations

import logging
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
    is_automatic_entity_category,
    is_safe_automatic_entity_mapping,
    is_usable_memory_mapping,
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
)

logger = logging.getLogger(__name__)

_NON_PROSE_RE = re.compile(
    r"\b(?:copyright|all rights reserved|isbn|contents|tables|abbreviations|"
    r"index of names|subject index|catalog(?:ue|ing)-in-publication)\b",
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
    ):
        return ""
    persian_chars = len(re.findall(r"[\u0600-\u06ff]", sample))
    latin_words = len(re.findall(r"\b[A-Za-z]{3,}\b", sample))
    if persian_chars < 8 or latin_words > max(12, persian_chars // 18):
        return ""
    return sample


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
        
        # Pull retrieval_k and window_size from config
        retrieval_k = config.to_dict().get("memory", {}).get("long_term_retrieval_k", 5)
        window_size = config.to_dict().get("memory", {}).get("short_term_window", 4)
        
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
        )
        has_structure_policy = "style_eligible" in chunk.metadata
        translation_paragraphs = [
            paragraph.strip() for paragraph in translation.split("\n\n")
            if paragraph.strip()
        ]
        style_translation = translation
        if has_body_policy:
            if body_indices and max(body_indices) < len(translation_paragraphs):
                style_translation = "\n\n".join(
                    translation_paragraphs[index] for index in body_indices
                )
            elif not chunk.metadata.get("style_eligible", False):
                style_translation = ""
        style_quality_approved = (
            bool(quality_approved)
            if style_approved is None else bool(style_approved)
        )
        style_eligible = bool(
            resolved_long_term_reliable
            and style_quality_approved
            and (
                not has_structure_policy
                or (
                    len((chunk.text or "").strip()) >= 240
                    and len((translation or "").strip()) >= 160
                )
            )
        )
        style_count_before = len(self.style_samples)
        if style_eligible:
            self._update_style_profile(style_translation)
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
            "reliability_reasons": list(dict.fromkeys(
                str(reason).strip()
                for reason in (reliability_reasons or [])
                if str(reason).strip()
            )),
            "style_sample_added": len(self.style_samples) > style_count_before,
            "structure_eligible": structure_eligible,
            "quality_approved": bool(quality_approved),
            "style_approved": style_quality_approved,
            "structural_roles": structural_roles,
        }

    def _update_style_profile(self, translation: str) -> None:
        """Maintain a compact book-level style guide from early translations."""
        text = _clean_style_sample(translation)
        if not text:
            return

        if len(self.style_samples) < 5:
            self.style_samples.append(text)

        self.style_profile = self._render_style_profile()

    def _render_style_profile(self) -> str:
        """Build a prompt-safe style guide from trusted prose samples."""
        clean_samples = [
            cleaned for sample in self.style_samples
            if (cleaned := _clean_style_sample(sample))
        ][:5]
        if not clean_samples:
            return ""
        samples = "\n".join(
            f"{i + 1}. {sample}"
            for i, sample in enumerate(clean_samples)
        )
        return (
            "Maintain one coherent scholarly Iranian-Persian voice across the book. "
            "Prefer formal academic diction, precise conceptual renderings, stable "
            "citation handling, and the sentence rhythm established in the samples below. "
            "Do not simplify later chapters into a different register. These samples govern "
            "register and rhythm only: they are not terminology authority, and the source, "
            "curated glossary, and current context override every lexical choice. Do not copy "
            "transliteration artifacts or untranslated citation prose from a sample.\n\n"
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
        self, llm_client: Any, text: str, translation: str = ""
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
            inline_categories = {
                "proper_noun", "person", "place", "institution",
                "organization", "organisation", "publication", "product",
                "theory", "named_theory", "approved_term", "book", "article",
                "journal", "work", "technical_loanword", "loanword",
            }
            normalized_translation = re.sub(
                r"[\s\u200c]+", " ", translation.strip()
            ).casefold()
            for item in candidates if isinstance(candidates, list) else []:
                if not isinstance(item, dict):
                    continue
                term = str(item.get("term", "")).strip()
                persian = str(item.get("suggested_persian", "")).strip()
                category = str(item.get("category", "term"))
                if not is_usable_memory_mapping(term, persian):
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
                    continue
                if not is_safe_automatic_entity_mapping(
                    term,
                    persian,
                    category,
                    text,
                    translation=translation,
                ):
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
            self.bilingual_summary.update(response)
            self.bilingual_summary.set_input_trust(input_trust, trust_reasons)
            report = self.reconcile_bilingual_summary()
            report.update({
                "input_trust": self.bilingual_summary.input_trust,
                "trust_reasons": list(self.bilingual_summary.trust_reasons),
                "context_retained": True,
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
        self.proper_nouns.deserialize(data.get("proper_nouns", {}))
        self.bilingual_summary.deserialize(data.get("bilingual_summary", {}))
        self.long_term.deserialize(data.get("past_translations", []))
        self.short_term = ShortTermMemory.deserialize(
            data.get("short_term_context", []),
            window_size=self.short_term.window_size
        )
