"""Translation pipeline orchestrator for the Tarjomeh translation system.

Orchestrates ingestion, chunking, translation memory context construction,
web search, translation, critique/refinement, and final output exporting.
"""

from __future__ import annotations

import hashlib
import difflib
import logging
import re
import time
import uuid
import threading
import json
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.structured_output import (
    normalize_model_text,
    parse_structured_output,
    protocol_artifacts,
)
from tarjomeh.core.paragraph_protocol import (
    decode_paragraphs,
    encode_paragraphs,
    protocol_instruction,
    repair_prompt as paragraph_repair_prompt,
    split_paragraphs,
)
from tarjomeh.core.llm_client import (
    EmptyCompletionError,
    IncompleteCompletionError,
    LLMClient,
    MalformedLLMResponseError,
    TruncatedCompletionError,
)
from tarjomeh.parsers.base import BaseParser, Document, EXTENSION_PARSER_MAP
from tarjomeh.chunking.chunker import SemanticChunker, FixedChunker, Chunk
from tarjomeh.memory.manager import MemoryManager, MemoryContext
from tarjomeh.memory.proper_nouns import (
    INLINE_ORIGINAL_CATEGORIES,
    has_exact_observed_anchor,
    is_safe_automatic_source_span,
    is_safe_automatic_entity_mapping,
    is_reusable_terminology_mapping,
    looks_like_transliterated_loanword,
    observed_bilingual_target,
    source_term_present,
)
from tarjomeh.context.web_searcher import WebContextSearcher
from tarjomeh.glossary.manager import GlossaryManager
from tarjomeh.glossary.compliance import (
    GlossaryComplianceChecker,
    target_present,
    term_occurs_only_in_citations,
)
from tarjomeh.persian.typography import PersianTypographer
from tarjomeh.persian.orthography import (
    apply_safe_persian_orthography,
    orthography_issue_count,
)
from tarjomeh.core.term_notes import (
    apply_term_notes,
    audit_inline_english_originals,
    effective_term_notes_mode,
    ensure_inline_proper_noun_originals,
    normalize_adjacent_original_citations,
    reconcile_redundant_original_fragments,
)
from tarjomeh.exporters import get_exporter
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.jobs.database import JobDatabase, JobStatus, ChunkStatus
from tarjomeh.quality.critique import TranslationCritique
from tarjomeh.quality.refiner import TranslationRefiner
from tarjomeh.quality.back_translator import BackTranslator
from tarjomeh.quality.structure_audit import audit_payload
from tarjomeh.quality.integrity import (
    PostEditIntegrityGate,
    extract_identifiers,
    extract_labeled_identifier_surfaces,
    repair_corruption,
    mixed_script_artifacts,
    repair_source_grounded_language_artifacts,
    source_unjustified_repeated_word_artifacts,
    detached_ezafe_artifacts,
    foreign_script_artifacts,
    markup_wrapper_artifacts,
    parenthesis_artifacts,
    restore_source_identifiers,
    normalize_for_match,
    protected_english_originals,
    protected_source_apparatus,
    unexpected_latin_prose,
)
from tarjomeh.core.prompts import (
    TRANSLATE_SYSTEM_PROMPT,
    TRANSLATE_CHUNK_PROMPT,
    GLOSSARY_EXTRACT_PROMPT,
    ACADEMIC_EXEMPLARS,
)

logger = logging.getLogger(__name__)


class PipelinePausedException(Exception):
    """Raised when the translation pipeline is cooperatively paused."""
    pass


class ResumeSourceMismatchError(RuntimeError):
    """Raised when a resumed job's input no longer matches its saved chunks."""


class ChapterCheckpointReached(PipelinePausedException):
    """Raised after an intentional chapter-boundary review checkpoint."""

    def __init__(self, chapter_position: int, chapter_title: str) -> None:
        self.chapter_position = chapter_position
        self.chapter_title = chapter_title
        super().__init__(
            f"Chapter {chapter_position} checkpoint reached: {chapter_title}"
        )


def build_chapter_manifest(document: Document) -> list[dict[str, Any]]:
    """Return stable, reviewable chapter metadata in parser order."""
    manifest: list[dict[str, Any]] = []
    for position, chapter in enumerate(document.chapters, 1):
        chapter.metadata["tarjomeh_chapter_position"] = position
        manifest.append({
            "position": position,
            "number": chapter.number,
            "title": chapter.title or f"Untitled chapter {position}",
            "paragraphs": len(chapter.all_paragraphs),
            "start_page": chapter.metadata.get("start_page"),
            "end_page": chapter.metadata.get("end_page"),
        })
    return manifest


def apply_chapter_selection(
    document: Document,
    selected_positions: list[int] | None,
) -> Document:
    """Restrict a parsed document to explicit 1-based chapter positions."""
    build_chapter_manifest(document)
    selected = sorted({int(value) for value in (selected_positions or [])})
    if not selected:
        return document
    invalid = [value for value in selected if value < 1 or value > len(document.chapters)]
    if invalid:
        raise ValueError(
            "Selected chapter position(s) are outside the parsed document: "
            + ", ".join(str(value) for value in invalid)
        )
    return Document(
        title=document.title,
        chapters=[document.chapters[position - 1] for position in selected],
        metadata=dict(document.metadata),
        raw_toc=document.raw_toc,
    )


def sanitize_document_protocol_artifacts(
    document: TranslatedDocument,
) -> dict[str, Any]:
    """Remove safe leading model wrappers and reject unresolved protocol tags."""
    edits: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for paragraph in document.paragraphs:
        original = paragraph.translated_text
        cleaned, report = normalize_model_text(original)
        removed = int(report.get("removed_reasoning_wrappers", 0))
        if removed:
            paragraph.translated_text = cleaned
            edits.append({
                "paragraph_index": paragraph.index,
                "removed_reasoning_wrappers": removed,
            })
        artifacts = protocol_artifacts(paragraph.translated_text)
        if artifacts:
            unresolved.append({
                "paragraph_index": paragraph.index,
                "artifacts": artifacts,
            })
    return {
        "safe_edit_count": len(edits),
        "remaining_artifact_count": len(unresolved),
        "edits": edits,
        "unresolved": unresolved,
    }


def restore_document_source_identifiers(
    document: TranslatedDocument,
) -> dict[str, Any]:
    """Restore exact source identifiers after every text transformation."""
    repairs: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for paragraph in document.paragraphs:
        repaired, report = restore_source_identifiers(
            paragraph.source_text, paragraph.translated_text
        )
        paragraph.translated_text = repaired
        if report.get("repair_count"):
            repairs.append({
                "paragraph_index": paragraph.index,
                **report,
            })
        missing = list(
            (
                extract_identifiers(paragraph.source_text)
                - extract_identifiers(paragraph.translated_text)
            ).elements()
        )
        missing.extend(
            f"labeled:{value}"
            for value in (
                extract_labeled_identifier_surfaces(paragraph.source_text)
                - extract_labeled_identifier_surfaces(
                    paragraph.translated_text
                )
            ).elements()
        )
        if missing:
            unresolved.append({
                "paragraph_index": paragraph.index,
                "missing": missing,
            })
    return {
        "repair_count": sum(
            int(item.get("repair_count", 0)) for item in repairs
        ),
        "repaired_paragraph_count": len(repairs),
        "unresolved_count": sum(len(item["missing"]) for item in unresolved),
        "repairs": repairs,
        "unresolved": unresolved,
    }


def audit_translation_language(
    source: str,
    translation: str,
    *,
    allowed_originals: list[str] | tuple[str, ...] = (),
    structural_role: str = "body",
    chapter_title: str = "",
) -> dict[str, Any]:
    """Report deterministic foreign-script leakage without rewriting semantics."""
    mixed = mixed_script_artifacts(translation)
    unexpected = unexpected_latin_prose(
        source,
        translation,
        allowed_originals=allowed_originals,
        structural_role=structural_role,
        chapter_title=chapter_title,
    )
    repeated = source_unjustified_repeated_word_artifacts(source, translation)
    foreign_scripts = foreign_script_artifacts(source, translation)
    markup = markup_wrapper_artifacts(source, translation)
    parentheses = parenthesis_artifacts(source, translation)
    detached_ezafe = detached_ezafe_artifacts(translation)
    return {
        "review_required": bool(
            mixed
            or unexpected
            or repeated
            or foreign_scripts
            or markup
            or parentheses
            or detached_ezafe
        ),
        "mixed_script_count": len(mixed),
        "mixed_script_artifacts": mixed,
        "unexpected_latin_count": len(unexpected),
        "unexpected_latin": unexpected,
        "repeated_word_count": len(repeated),
        "repeated_word_artifacts": repeated,
        "foreign_script_count": len(foreign_scripts),
        "foreign_script_artifacts": foreign_scripts,
        "markup_wrapper_count": len(markup),
        "markup_wrapper_artifacts": markup,
        "parenthesis_artifact_count": len(parentheses),
        "parenthesis_artifacts": parentheses,
        "detached_ezafe_count": len(detached_ezafe),
        "detached_ezafe_artifacts": detached_ezafe,
        "policy": (
            "Source-grounded identifiers, citations, approved originals, acronyms, "
            "and multilingual apparatus are allowed; unexplained foreign prose or "
            "scripts, mixed-script suffixes, malformed parentheses, detached ezafe, "
            "markup wrappers, and adjacent lexical duplication are review evidence."
        ),
    }


def audit_document_final_text(
    document: TranslatedDocument,
    *,
    allowed_originals: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Recheck final assembled text after every deterministic transformation."""
    paragraph_findings: list[dict[str, Any]] = []
    unresolved_identifiers: list[dict[str, Any]] = []
    for paragraph in document.paragraphs:
        language = audit_translation_language(
            paragraph.source_text,
            paragraph.translated_text,
            allowed_originals=allowed_originals,
            structural_role=str(
                paragraph.metadata.get("structure_role", "body")
            ),
            chapter_title=str(
                paragraph.metadata.get("chapter_title", "")
            ),
        )
        if language["review_required"]:
            paragraph_findings.append({
                "paragraph_index": paragraph.index,
                **language,
            })
        missing = list((
            extract_identifiers(paragraph.source_text)
            - extract_identifiers(paragraph.translated_text)
        ).elements())
        missing.extend(
            f"labeled:{value}"
            for value in (
                extract_labeled_identifier_surfaces(paragraph.source_text)
                - extract_labeled_identifier_surfaces(
                    paragraph.translated_text
                )
            ).elements()
        )
        if missing:
            unresolved_identifiers.append({
                "paragraph_index": paragraph.index,
                "missing": missing,
            })
    return {
        "review_required": bool(paragraph_findings or unresolved_identifiers),
        "language_finding_count": len(paragraph_findings),
        "unresolved_identifier_count": sum(
            len(item["missing"]) for item in unresolved_identifiers
        ),
        "paragraph_findings": paragraph_findings,
        "unresolved_identifiers": unresolved_identifiers,
    }


_QUALITY_STAGE_ERRORS = (
    EmptyCompletionError,
    IncompleteCompletionError,
    MalformedLLMResponseError,
    TruncatedCompletionError,
    httpx.HTTPStatusError,
    httpx.RequestError,
)


def _qa_provider_failure_payload(
    component: str,
    operation: str,
    client: Any,
    exc: Exception,
) -> dict[str, Any]:
    attempts = (
        client.last_call_attempt_count()
        if hasattr(client, "last_call_attempt_count") else 1
    )
    return {
        "component": component,
        "operation": operation,
        "attempts": attempts,
        "failure_type": type(exc).__name__,
        "error": _truncate_for_event(str(exc), 1000),
        "message": (
            f"{component.replace('_', ' ').title()} remained unavailable after "
            "bounded provider recovery; the prior translation was retained and "
            "the chunk requires human review."
        ),
    }


# ---------------------------------------------------------------------------
# Paragraph redistribution helpers (intra-chunk alignment)
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟…])\s+")


def _split_sentences_fa(text: str) -> list[str]:
    """Split Persian/mixed text into sentences on ., !, ?, ؟ and … boundaries."""
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p for p in parts if p.strip()]


def _split_source_recovery_groups(
    text: str,
    max_chars: int = 1800,
) -> list[str]:
    """Group source sentences for one bounded, paragraph-preserving recovery."""
    sentences = _split_sentences_fa(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []
    groups: list[str] = []
    current: list[str] = []
    current_chars = 0
    for sentence in sentences:
        added = len(sentence) + (1 if current else 0)
        if current and current_chars + added > max_chars:
            groups.append(" ".join(current).strip())
            current = []
            current_chars = 0
        current.append(sentence.strip())
        current_chars += len(sentence) + (1 if len(current) > 1 else 0)
    if current:
        groups.append(" ".join(current).strip())
    return groups


def _distribute_translation(translation: str, n_parts: int, src_weights: list[int]) -> list[str]:
    """Distribute a translation blob across *n_parts* paragraphs proportionally.

    Used when the LLM returns a different number of ``\\n\\n`` paragraphs than
    the source chunk had. Instead of dumping everything into the first
    paragraph and blanking the rest (which visibly breaks bilingual layouts),
    sentences are packed into parts whose target sizes are proportional to the
    source paragraphs' lengths. No content is ever dropped: the final part
    always receives the remaining sentences.
    """
    translation = translation.strip()
    if n_parts <= 1:
        return [translation]

    sentences = _split_sentences_fa(translation)
    if len(sentences) <= 1:
        # A single unsplittable run of text — nothing to distribute.
        return [translation] + [""] * (n_parts - 1)

    if len(src_weights) != n_parts or sum(src_weights) <= 0:
        src_weights = [1] * n_parts

    total_weight = sum(src_weights)
    total_len = sum(len(s) for s in sentences)

    parts: list[str] = []
    si = 0
    for pi in range(n_parts):
        remaining_parts = n_parts - pi
        remaining = sentences[si:]
        if not remaining:
            parts.append("")
            continue
        if remaining_parts == 1:
            # Last part takes everything left — guarantees no content loss.
            parts.append(" ".join(remaining))
            si = len(sentences)
            continue

        target_len = total_len * src_weights[pi] / total_weight
        # Leave at least one sentence for each remaining part when possible.
        max_take = max(1, len(remaining) - (remaining_parts - 1))
        taken: list[str] = []
        taken_len = 0
        for s in remaining[:max_take]:
            if taken and taken_len >= target_len:
                break
            taken.append(s)
            taken_len += len(s)
        parts.append(" ".join(taken))
        si += len(taken)

    return parts


_COMMON_HEADING_TRANSLATIONS = {
    "abstract": "چکیده",
    "acknowledgements": "سپاسگزاری",
    "acknowledgments": "سپاسگزاری",
    "appendix": "پیوست",
    "bibliography": "کتاب‌نامه",
    "chapter": "فصل",
    "conclusion": "نتیجه‌گیری",
    "contents": "فهرست",
    "epilogue": "پس‌گفتار",
    "foreword": "پیش‌گفتار",
    "index": "نمایه",
    "introduction": "مقدمه",
    "notes": "یادداشت‌ها",
    "preface": "دیباچه",
    "prologue": "پیش‌درآمد",
    "references": "منابع",
}


def _fallback_heading_translation(source_heading: str) -> str:
    """Return a conservative heading translation when the model omitted one."""
    cleaned = " ".join(source_heading.split()).strip()
    key = re.sub(r"[^a-z0-9 ]+", "", cleaned.lower()).strip()
    return _COMMON_HEADING_TRANSLATIONS.get(key, cleaned)


def _looks_like_heading_translation(candidate: str, source_heading: str) -> bool:
    """Heuristic guard so body text is never assigned to a heading slot."""
    text = candidate.strip()
    if not text:
        return False
    # Headings can be longer than their English source, but not whole pages.
    max_len = max(120, len(source_heading.strip()) * 12)
    if len(text) > max_len:
        return False
    sentence_marks = sum(text.count(mark) for mark in ".!?؟؛…")
    source_sentence_marks = sum(source_heading.count(mark) for mark in ".!?؟؛…")
    if source_sentence_marks == 0 and sentence_marks > 0:
        return False
    return sentence_marks <= 1


def _truncate_for_event(value: str, limit: int = 2000) -> str:
    """Keep structured QA events useful without letting the DB grow wildly."""
    text = value or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


def _integrity_repair_prompt(
    *,
    source: str,
    rejected_translation: str,
    integrity_payload: dict[str, Any],
    terminology: str,
    inline_policy: str,
    paragraph_count: int,
    paragraph_instruction: str = "",
) -> str:
    """Build a complete-translation repair from deterministic gate evidence."""
    blocking = [
        finding for finding in integrity_payload.get("findings", [])
        if finding.get("severity") == "blocking"
    ]
    evidence = json.dumps(blocking, ensure_ascii=False, indent=2)
    return f"""\
The completed Persian translation below failed deterministic integrity checks.
Repair only the evidenced defects while preserving all correct content.

### English source
{source}

### Rejected Persian translation
{rejected_translation}

### Blocking integrity evidence
{evidence}

### Terminology policy
{terminology}

### English-original policy
{inline_policy}

Requirements:
1. Return the complete corrected Persian translation, not a patch or explanation.
2. Preserve every source proposition, paragraph, citation, year, number, name,
   footnote marker, and source-authored multilingual expression.
3. Repair mixed-script corruption and omissions only from source evidence; never guess.
4. Preserve unrelated wording. Keep exactly {paragraph_count} paragraph(s).
5. Use publication-quality formal Iranian Persian and established academic equivalents;
   avoid opaque calques or transliteration when a precise standard Persian rendering exists.
6. Output only the corrected translation.
{paragraph_instruction}"""


def _paragraph_count(text: str) -> int:
    return len([p for p in (text or "").split("\n\n") if p.strip()])


def _recovery_segment_instruction(segment_id: str) -> str:
    return (
        "\n\n### Recovery output contract\n"
        "Return exactly one Persian translation block and no other text:\n"
        f"<<<TRANSLATION {segment_id}>>>\n"
        "[Persian translation of the target source only]\n"
        f"<<<END {segment_id}>>>"
    )


def _parse_recovery_segment(raw: str, segment_id: str) -> tuple[str, dict[str, Any]]:
    """Extract and validate one model-neutral adaptive-recovery segment."""
    text = (raw or "").strip()
    start = f"<<<TRANSLATION {segment_id}>>>"
    end = f"<<<END {segment_id}>>>"
    diagnostics: dict[str, Any] = {
        "segment_id": segment_id,
        "raw_chars": len(text),
        "valid": False,
        "errors": [],
    }
    if text.count(start) != 1 or text.count(end) != 1:
        diagnostics["errors"].append("segment_envelope_mismatch")
        return "", diagnostics
    start_pos = text.find(start) + len(start)
    end_pos = text.find(end, start_pos)
    if end_pos < start_pos:
        diagnostics["errors"].append("segment_envelope_order")
        return "", diagnostics
    residue = (text[:text.find(start)] + text[end_pos + len(end):]).strip()
    residue = residue.replace("```text", "").replace("```", "").strip()
    if residue or re.search(r"<<<(?:TRANSLATION|END)\s+[^>]+>>>", text[:text.find(start)] + text[end_pos + len(end):]):
        diagnostics["errors"].append("segment_output_residue")
    candidate = text[start_pos:end_pos].strip()
    diagnostics["output_chars"] = len(candidate)
    diagnostics["output_paragraphs"] = _paragraph_count(candidate)
    if re.search(r"<<<(?:TRANSLATION|END)\s+[^>]+>>>", candidate):
        diagnostics["errors"].append("foreign_segment_marker")
    if not candidate:
        diagnostics["errors"].append("empty_segment")
    if diagnostics["output_paragraphs"] != 1:
        diagnostics["errors"].append("paragraph_parity")
    diagnostics["valid"] = not diagnostics["errors"]
    return candidate, diagnostics


def _word_shingles(text: str, width: int = 8) -> set[tuple[str, ...]]:
    words = re.findall(r"[\w\u0600-\u06ff]+", (text or "").casefold())
    if len(words) < width:
        return set()
    return {tuple(words[i:i + width]) for i in range(len(words) - width + 1)}


def _validate_recovery_part(
    source: str,
    candidate: str,
    *,
    previous_target: str = "",
    source_context: str = "",
    structural_role: str = "body",
) -> dict[str, Any]:
    """Apply conservative reject-only checks before recovery assembly."""
    source_chars = len((source or "").strip())
    output_chars = len((candidate or "").strip())
    ratio = output_chars / max(1, source_chars)
    errors: list[str] = []
    minimum_ratio, maximum_ratio = (
        (0.25, 3.0) if source_chars >= 80 else (0.15, 8.0)
    )
    if not minimum_ratio <= ratio <= maximum_ratio:
        errors.append("source_output_size_ratio")
    source_identifiers = extract_identifiers(source)
    candidate_identifiers = extract_identifiers(candidate)
    source_words = re.findall(r"[A-Za-z\u00c0-\u024f]{3,}", source or "")
    prose_required = bool(source_words)
    if str(structural_role or "body").casefold() in {
        "bibliography", "catalog", "index", "table",
    }:
        prose_required = False
    elif source_identifiers and len(source_words) <= 2:
        prose_required = False

    if prose_required and len(re.findall(r"[\u0600-\u06ff]", candidate or "")) < 3:
        errors.append("target_language_missing")
    if source_identifiers and bool(source_identifiers - candidate_identifiers):
        errors.append("source_identifier_missing")

    context_words = re.findall(r"[A-Za-z][A-Za-z'-]+", source_context or "")
    context_sequences = {
        " ".join(context_words[i:i + 6]).casefold()
        for i in range(max(0, len(context_words) - 5))
    }
    candidate_folded = (candidate or "").casefold()
    if any(sequence in candidate_folded for sequence in context_sequences):
        errors.append("surrounding_source_repeated")

    prior_shingles = _word_shingles(previous_target)
    candidate_shingles = _word_shingles(candidate)
    if prior_shingles and candidate_shingles:
        common_shingles = len(prior_shingles & candidate_shingles)
        overlap = common_shingles / max(1, len(candidate_shingles))
        if len(candidate_shingles) >= 5 and common_shingles >= 3 and overlap >= 0.45:
            errors.append("previous_translation_repeated")
    else:
        overlap = 0.0
    return {
        "valid": not errors,
        "errors": errors,
        "source_chars": source_chars,
        "output_chars": output_chars,
        "size_ratio": round(ratio, 4),
        "previous_overlap": round(overlap, 4),
        "context_chars": len(source_context or ""),
        "target_script_required": prose_required,
        "source_identifier_count": len(source_identifiers),
    }


def _output_extension(fmt: str) -> str:
    return "md" if fmt.lower() == "markdown" else fmt.lower()


def _term_notes_instruction(mode: str) -> str:
    """Return prompt policy without letting the model invent note numbering."""
    if mode in ("footnote", "endnote"):
        return (
            "Use the established Persian rendering consistently. Do not add an "
            "English parenthetical solely because this is the first occurrence; "
            "the application will add a numbered note after translation. This "
            "instruction overrides inline-parenthetical examples."
        )
    if mode == "both":
        return (
            "For a proper noun's first occurrence, include the English original "
            "in parentheses after the Persian rendering. The application will "
            "also add a numbered note; never invent note numbers yourself."
        )
    return (
        "Consult the proper-noun list in memory. Names marked [introduced] use "
        "the established Persian rendering without another parenthetical. Names "
        "marked [first occurrence pending] get the English original in "
        "parentheses after the Persian rendering exactly once."
    )


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


def _source_entity_categories(
    source_text: str,
    candidates: list[str],
) -> dict[str, str]:
    return {
        candidate: _source_entity_category(source_text, candidate)
        for candidate in candidates
    }


_LANGUAGE_QUALITY_COUNT_FIELDS = (
    "mixed_script_count",
    "unexpected_latin_count",
    "repeated_word_count",
    "foreign_script_count",
    "markup_wrapper_count",
    "parenthesis_artifact_count",
    "detached_ezafe_count",
)


def _language_quality_strictly_improves(
    before: dict[str, Any],
    after: dict[str, Any],
) -> bool:
    """Accept a local repair only when no deterministic category regresses."""
    before_counts = tuple(int(before.get(key, 0) or 0) for key in _LANGUAGE_QUALITY_COUNT_FIELDS)
    after_counts = tuple(int(after.get(key, 0) or 0) for key in _LANGUAGE_QUALITY_COUNT_FIELDS)
    return all(new <= old for old, new in zip(before_counts, after_counts, strict=True)) and any(
        new < old for old, new in zip(before_counts, after_counts, strict=True)
    )


def _language_repair_is_local(
    before: str,
    after: str,
    *,
    structural_role: str,
) -> bool:
    """Reject broad rewrites from the final objective-artifact repair pass."""
    if not before or not after:
        return False
    similarity = difflib.SequenceMatcher(None, before, after, autojunk=False).ratio()
    role = str(structural_role or "body").casefold()
    minimum = 0.55 if role in {"contents_entry", "heading", "title"} else 0.82
    length_ratio = len(after) / max(1, len(before))
    return similarity >= minimum and 0.75 <= length_ratio <= 1.25


def _targeted_language_repair_prompt(
    source: str,
    translation: str,
    findings: dict[str, Any],
    *,
    structural_role: str,
) -> str:
    """Build one paragraph-local repair request from deterministic evidence."""
    evidence = {
        key: findings.get(key, [])
        for key in (
            "mixed_script_artifacts",
            "unexpected_latin",
            "foreign_script_artifacts",
            "markup_wrapper_artifacts",
            "parenthesis_artifacts",
            "detached_ezafe_artifacts",
        )
        if findings.get(key)
    }
    return f"""\
The accepted Persian paragraph below has objective surface-language artifacts.
Correct only the evidenced artifacts while preserving its full meaning.

### Structural role
{structural_role}

### English source paragraph
{source}

### Accepted Persian paragraph
{translation}

### Deterministic findings
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Requirements:
1. Return exactly one complete Persian paragraph and nothing else.
2. Preserve every proposition, qualification, relation, citation, number, name,
   required English parenthetical, and list or table label.
3. Change only what is necessary to remove the listed foreign-script, untranslated
   ordinary prose, markup, parenthesis, or detached-ezafe artifact.
4. Do not choose new terminology, summarize, add commentary, or alter source facts.
5. Use fluent formal Iranian Persian. For a contents/title row, preserve the title's
   meaning and page label while repairing Persian syntax.
"""


def _high_confidence_person_candidates(
    candidates: list[str],
    source_text: str = "",
) -> list[str]:
    """Narrow review-blocking candidates to plausible multi-token people."""
    selected: list[str] = []
    for candidate in candidates:
        tokens = re.findall(r"[A-Za-z\u00c0-\u024f\u1e00-\u1eff]+", candidate)
        if not 2 <= len(tokens) <= 4:
            continue
        if re.search(r"\s+(?:and|&)\s+", candidate, re.IGNORECASE):
            continue
        if tokens[-1].casefold() in _NON_PERSON_ENTITY_ENDINGS:
            continue
        if any(token.casefold() in _ENTITY_NON_NAME_TOKENS for token in tokens):
            continue
        if source_text and _source_entity_category(source_text, candidate) != "person":
            continue
        selected.append(candidate)
    return selected


def _high_confidence_instrument_candidates(
    candidates: list[str],
) -> list[str]:
    """Return source-explicit named legal or institutional instruments."""
    return [
        candidate for candidate in candidates
        if re.search(rf"\b{_NAMED_INSTRUMENT_LABEL}$", candidate)
    ]


def _observed_anchor_target(
    translation: str,
    source: str,
    category: str = "proper_noun",
) -> str:
    """Read a Persian name immediately preceding an exact English original."""
    return observed_bilingual_target(
        translation,
        _canonical_source_entity(source),
        category,
    )


def _reconcile_current_entity_anchors(
    memory_manager: MemoryManager,
    source_entities: list[str],
    translation: str,
    categories: dict[str, str] | None = None,
    source_text: str = "",
) -> dict[str, Any]:
    """Persist only mappings explicitly evidenced by ``Persian (English)``."""
    observed: list[dict[str, Any]] = []
    missing: list[str] = []
    for source in source_entities:
        category = (categories or {}).get(source, "proper_noun")
        target = _observed_anchor_target(translation, source, category)
        if not target:
            missing.append(source)
            continue
        stored_category = (
            category
            if category in INLINE_ORIGINAL_CATEGORIES
            else "source_grounded_entity"
        )
        if not is_safe_automatic_entity_mapping(
            source,
            target,
            stored_category,
            source_text or source,
            translation=translation,
            require_observed_anchor=True,
        ):
            missing.append(source)
            continue
        outcome = memory_manager.proper_nouns.add_noun(
            source,
            target,
            category=stored_category,
            provenance="observed_translation",
        )
        if outcome.get("action") != "ignored":
            observed.append({"source": source, "target": target, **outcome})
    return {
        "candidate_count": len(source_entities),
        "observed_count": len(observed),
        "observed": observed,
        "missing": missing,
    }


def _accepted_grounded_inline_originals(
    source_text: str,
    translation: str,
) -> dict[str, str]:
    """Classify compact, source-grounded ``Persian (Latin)`` anchors.

    This recovers accepted names and transliterated technical loanwords without
    maintaining a vocabulary for one author or book. Citations, years, and
    ordinary untranslated prose are deliberately excluded.
    """
    categories: dict[str, str] = {}
    for match in re.finditer(r"\(([^()\n]{1,160})\)", translation or ""):
        source = _canonical_source_entity(match.group(1))
        words = re.findall(r"[A-Za-z\u00c0-\u024f\u1e00-\u1eff]+", source)
        if (
            not 1 <= len(words) <= 8
            or any(char.isdigit() for char in source)
            or not source_term_present(source_text, source)
        ):
            continue
        category = _source_entity_category(source_text, source)
        target = _observed_anchor_target(translation, source, category)
        if not target:
            continue
        if category == "source_entity_candidate":
            if looks_like_transliterated_loanword(source, target):
                category = "technical_loanword"
            elif len(words) >= 2 and all(
                word[:1].isupper() or len(word) == 1 for word in words
            ):
                category = "source_grounded_entity"
            else:
                continue
        categories[source] = category
    return categories


def _canonicalize_chunk_inline_originals(
    memory_manager: MemoryManager,
    chunk: Chunk,
    translation: str,
) -> tuple[str, dict[str, Any]]:
    """Apply the same first-occurrence policy before memory and export.

    Keeping this deterministic pass at the chunk boundary makes the database,
    continuity memory, and later document assembly start from one canonical
    accepted text. A paragraph mismatch is reported and left untouched.
    """
    source_paragraphs = split_paragraphs(chunk.text)
    target_paragraphs = split_paragraphs(translation)
    before_hash = hashlib.sha256(
        normalize_for_match(translation).encode("utf-8")
    ).hexdigest()
    report: dict[str, Any] = {
        "stage": "pre_memory_inline_reconciliation",
        "source_paragraphs": len(source_paragraphs),
        "target_paragraphs": len(target_paragraphs),
        "before_hash": before_hash,
        "after_hash": before_hash,
        "changed": False,
        "valid": len(source_paragraphs) == len(target_paragraphs),
    }
    if not report["valid"]:
        report["reason"] = "paragraph_count_mismatch"
        return translation, report

    pending = memory_manager.proper_nouns.pending_inline_originals(chunk.text)
    if not pending:
        report["reason"] = "no_pending_originals"
        return translation, report

    state = memory_manager.proper_nouns.serialize()
    categories = {
        source: str(state.get("categories", {}).get(source, "proper_noun"))
        for source in pending
    }
    aliases = {
        source: list(state.get("aliases", {}).get(source, []))
        for source in pending
        if state.get("aliases", {}).get(source)
    }
    document = TranslatedDocument(
        paragraphs=[
            TranslatedParagraph(
                index=index,
                source_text=source,
                translated_text=target,
            )
            for index, (source, target) in enumerate(
                zip(source_paragraphs, target_paragraphs, strict=True)
            )
        ]
    )
    typographer = PersianTypographer(
        memory_manager.config.to_dict().get("persian")
    )
    anchor_report = ensure_inline_proper_noun_originals(
        document,
        pending,
        typographer,
        categories,
        aliases=aliases,
        return_report=True,
    )
    original_report = audit_inline_english_originals(document, pending)
    final_anchor_report = ensure_inline_proper_noun_originals(
        document,
        pending,
        typographer,
        categories,
        aliases=aliases,
        return_report=True,
    )
    canonical = "\n\n".join(
        paragraph.translated_text.strip() for paragraph in document.paragraphs
    )
    after_hash = hashlib.sha256(
        normalize_for_match(canonical).encode("utf-8")
    ).hexdigest()
    report.update({
        "pending_count": len(pending),
        "inserted_count": int(anchor_report.get("inserted_count", 0))
        + int(final_anchor_report.get("inserted_count", 0)),
        "removed_unauthorized_count": int(
            original_report.get("removed_unauthorized_count", 0)
        ),
        "after_hash": after_hash,
        "changed": before_hash != after_hash,
        "reason": "canonicalized",
    })
    return canonical, report


def _anchor_only_repair_is_valid(
    before: str,
    after: str,
    required_originals: list[str],
    categories: dict[str, str],
) -> bool:
    """Accept a model anchor repair only when exact parentheticals were inserted."""
    stripped = after or ""
    for source in required_originals:
        if not has_exact_observed_anchor(stripped, source):
            return False
        category = categories.get(source, "source_entity_candidate")
        if not _observed_anchor_target(stripped, source, category):
            return False
        words = [re.escape(part) for part in source.split()]
        body = r"\s+".join(words)
        stripped = re.sub(
            rf"\s*\(\s*{body}(?:\s+(?:1[5-9]\d{{2}}|20\d{{2}})[a-z]?)?"
            rf"(?:\s*,\s*[^()\n]{{1,120}})?\s*\)",
            "",
            stripped,
            count=1,
            flags=re.IGNORECASE,
        )
    return normalize_for_match(stripped) == normalize_for_match(before)


def _required_anchor_repair_prompt(
    source: str,
    translation: str,
    missing: list[str],
) -> str:
    originals = "\n".join(f"- ({value})" for value in missing)
    return f"""\
The Persian translation is complete, but required first-occurrence English
originals are missing. Insert only the exact parentheticals listed below after
their corresponding Persian person name or named legal instrument.

### English source
{source}

### Accepted Persian translation
{translation}

### Exact parentheticals to insert
{originals}

Requirements:
1. Return the complete Persian translation only.
2. Insert each exact parenthetical once, immediately after its Persian rendering.
3. Do not add a parenthetical if the corresponding Persian rendering is absent.
4. Do not change, remove, reorder, or re-punctuate any existing character.
5. Preserve every paragraph boundary, citation, number, and source expression.
"""


def _research_context_for_memory(artifact: dict[str, Any] | None) -> str:
    """Format unapproved research as non-authoritative prompt context."""
    usable_statuses = {
        "completed", "completed_without_suggestions", "degraded"
    }
    if not artifact or artifact.get("status") not in usable_statuses:
        return ""
    parts = [str(artifact.get("book_context", "")).strip()]
    suggestions = []
    for item in artifact.get("terms", []):
        if not isinstance(item, dict) or item.get("status") != "suggested":
            continue
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        if source and target:
            confidence = str(item.get("confidence", "unknown"))
            evidence = str(item.get("evidence_type", "unspecified"))
            suggestions.append(
                f"- {source} -> {target} "
                f"[confidence={confidence}; evidence={evidence}; unapproved]"
            )
    if suggestions:
        parts.extend([
            "Unapproved research suggestions follow. They are contextual hints, "
            "not mandatory terminology. Never override the curated glossary with them.",
            chr(10).join(suggestions),
        ])
    return (chr(10) * 2).join(part for part in parts if part)


def _glossary_entries_for_event(entries: list[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for entry in entries:
        payload.append({
            "source": getattr(entry, "source", ""),
            "target": getattr(entry, "target", ""),
            "domain": getattr(entry, "domain", ""),
            "context": _truncate_for_event(getattr(entry, "context", ""), 500),
            "sense": getattr(entry, "sense", None),
            "author": getattr(entry, "author", None),
            "glossary": getattr(entry, "glossary", None),
            "is_auto": bool(getattr(entry, "is_auto", False)),
            "include_original": bool(getattr(entry, "include_original", False)),
        })
    return payload


_BLOCKING_CRITIQUE_RE = re.compile(
    r"^\[(?:CRITICAL/[^\]]+|MAJOR/(?:accuracy|terminology))\]",
    re.IGNORECASE,
)


def _blocking_critique_issues(critique: Any) -> list[str]:
    """Return critique issues that must force refinement, regardless of average."""
    details = list(getattr(critique, "issue_details", []) or [])
    if details:
        blocking: list[str] = []
        for detail in details:
            severity = str(detail.get("severity", "")).strip().lower()
            category = str(detail.get("category", "")).strip().lower()
            required = (
                severity == "critical"
                and category in {
                    "accuracy", "omission", "number", "citation", "name",
                }
            ) or (
                severity == "major"
                and category in {"accuracy", "terminology"}
            )
            if required:
                blocking.append(
                    str(detail.get("formatted") or detail.get("issue_id") or detail)
                )
        return blocking
    return [
        str(issue)
        for issue in getattr(critique, "issues", []) or []
        if _BLOCKING_CRITIQUE_RE.match(str(issue).strip())
    ]


def _filter_critique_policy_conflicts(
    critique: Any,
    source_text: str,
    allowed_originals: list[str],
) -> list[dict[str, Any]]:
    """Remove only critic issues that contradict deterministic inline policy."""
    details = list(getattr(critique, "issue_details", []) or [])
    if not details:
        return []

    allowed = {value.casefold() for value in allowed_originals}
    kept_details: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for detail in details:
        current = str(detail.get("current_translation", ""))
        suggested = str(detail.get("suggested_fix", ""))
        current_originals = {
            value.casefold(): value
            for value in protected_english_originals(source_text, current)
        }
        suggested_originals = {
            value.casefold(): value
            for value in protected_english_originals(source_text, suggested)
        }
        removed = set(current_originals) - set(suggested_originals)
        added = set(suggested_originals) - set(current_originals)
        removed_authorized = sorted(
            current_originals[value] for value in removed if value in allowed
        )
        added_unauthorized = sorted(
            suggested_originals[value] for value in added
            if value not in allowed and not any(char.isdigit() for char in value)
        )
        removed_citations = sorted(
            current_originals[value] for value in removed
            if any(char.isdigit() for char in value)
        )
        current_apparatus = protected_source_apparatus(source_text, current)
        suggested_normalized = normalize_for_match(suggested)
        removed_source_apparatus = sorted(
            value for value in current_apparatus
            if normalize_for_match(value) not in suggested_normalized
        )
        if (
            removed_authorized
            or added_unauthorized
            or removed_citations
            or removed_source_apparatus
        ):
            conflicts.append({
                "issue": detail.get("formatted", ""),
                "removed_authorized": removed_authorized,
                "added_unauthorized": added_unauthorized,
                "removed_citations": removed_citations,
                "removed_source_apparatus": removed_source_apparatus,
            })
        else:
            kept_details.append(detail)

    if conflicts:
        critique.issue_details = kept_details
        critique.issues = [
            str(detail.get("formatted", ""))
            for detail in kept_details
            if str(detail.get("formatted", "")).strip()
        ]
    return conflicts


def _filter_critique_glossary_conflicts(
    critique: Any,
    entries: list[Any],
    *,
    include_auto: bool = False,
) -> list[dict[str, Any]]:
    """Withhold terminology advice that removes a mandatory rendering."""
    details = list(getattr(critique, "issue_details", []) or [])
    if not details:
        return []
    mandatory_entries = [
        entry for entry in entries
        if (include_auto or not bool(getattr(entry, "is_auto", False)))
        and str(getattr(entry, "source", "")).strip()
        and str(getattr(entry, "target", "")).strip()
    ]
    kept: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for detail in details:
        category = str(detail.get("category", "")).lower()
        source_quote = str(detail.get("source_quote", ""))
        current = str(detail.get("current_persian_quote", ""))
        suggested = str(detail.get("suggested_correction", ""))
        conflict_entries = []
        if category in {"terminology", "name"}:
            for entry in mandatory_entries:
                source = str(entry.source)
                target = str(entry.target)
                if (
                    re.search(rf"(?<!\w){re.escape(source)}(?!\w)", source_quote, re.I)
                    and GlossaryComplianceChecker._target_present(target, current)
                    and not GlossaryComplianceChecker._target_present(
                        target, suggested
                    )
                ):
                    conflict_entries.append({"source": source, "target": target})
        if conflict_entries:
            conflict = {
                **detail,
                "ignored_reason": "curated_glossary_conflict",
                "curated_entries": conflict_entries,
            }
            conflicts.append(conflict)
        else:
            kept.append(detail)
    if conflicts:
        critique.issue_details = kept
        critique.issues = [
            str(detail.get("formatted", "")) for detail in kept
            if str(detail.get("formatted", "")).strip()
        ]
        critique.ignored_issue_details = list(
            getattr(critique, "ignored_issue_details", []) or []
        ) + conflicts
    return conflicts


def _high_risk_concepts(critique: Any) -> list[dict[str, Any]]:
    """Collect grounded, review-only concept risks from validated issues."""
    risks: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for detail in list(getattr(critique, "issue_details", []) or []):
        for risk in list(detail.get("risk_flags", []) or []):
            key = (
                str(risk.get("term", "")).casefold(),
                str(risk.get("reason", "")),
                str(detail.get("source_segment_id", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            risks.append({
                "term": risk.get("term", ""),
                "reason": risk.get("reason", ""),
                "source_segment_id": detail.get("source_segment_id", ""),
                "source_quote": detail.get("source_quote", ""),
                "current_persian_quote": detail.get("current_persian_quote", ""),
                "suggested_correction": detail.get("suggested_correction", ""),
                "issue_id": detail.get("issue_id", ""),
                "severity": detail.get("severity", ""),
                "category": detail.get("category", ""),
            })
    return risks


def _inline_eligible_nouns_from_state(noun_state: Any) -> dict[str, str]:
    """Read eligible noun mappings from current or legacy memory checkpoints."""
    if not isinstance(noun_state, dict):
        return {}
    nouns = noun_state.get("nouns", {})
    if not isinstance(nouns, dict):
        return {}
    categories = noun_state.get("categories", {})
    if not isinstance(categories, dict):
        categories = {}
    return {
        source: target
        for source, target in nouns.items()
        if str(categories.get(source, "proper_noun")) in INLINE_ORIGINAL_CATEGORIES
    }


def _normalized_translation_version(value: str) -> str:
    return " ".join((value or "").split()).casefold()


def _refinement_decisions_with_commit_state(
    decisions: list[dict[str, Any]],
    issue_details: list[dict[str, Any]],
    candidate: str,
    *,
    candidate_accepted: bool,
) -> list[dict[str, Any]]:
    """Describe what was actually committed for each refiner decision."""
    issues = {
        str(issue.get("issue_id", "")).strip(): issue
        for issue in issue_details
        if str(issue.get("issue_id", "")).strip()
    }
    enriched: list[dict[str, Any]] = []
    for raw in decisions:
        decision = dict(raw)
        issue_id = str(decision.get("issue_id", "")).strip()
        choice = str(decision.get("decision", "")).strip().lower()
        current_span = str(
            issues.get(issue_id, {}).get("current_persian_quote", "")
        ).strip()
        resulting_span = str(decision.get("resulting_span", "")).strip()
        if choice == "rejected":
            status, integrity_status, reason = (
                "not_committed", "not_applicable", "refiner_rejected"
            )
        elif not candidate_accepted:
            status, integrity_status, reason = (
                "not_committed", "rejected", "candidate_integrity_rejected"
            )
        elif (
            current_span
            and resulting_span
            and normalize_for_match(current_span) == normalize_for_match(resulting_span)
        ):
            status, integrity_status, reason = (
                "not_committed", "accepted", "no_textual_change"
            )
        elif resulting_span and resulting_span in candidate:
            status, integrity_status, reason = (
                "committed_full_candidate", "accepted", "candidate_committed"
            )
        else:
            status, integrity_status, reason = (
                "not_committed", "accepted", "resulting_span_not_found"
            )
        decision.update({
            "commit_status": status,
            "integrity_status": integrity_status,
            "commit_reason": reason,
        })
        enriched.append(decision)
    return enriched


def _salvage_local_refinement_edits(
    *,
    source: str,
    previous: str,
    proposed: str,
    issue_details: list[dict[str, Any]],
    issue_decisions: list[dict[str, Any]],
    integrity_gate: PostEditIntegrityGate,
    protected_terms: list[str],
    protect_inline_english: bool,
    allowed_inline_originals: list[str],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Recover only unique, local refiner edits that independently pass integrity."""
    issues = {
        str(issue.get("issue_id", "")).strip(): issue
        for issue in issue_details
        if str(issue.get("issue_id", "")).strip()
    }
    current = previous
    enriched: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    committed = 0

    spans: dict[str, tuple[int, int]] = {}
    for raw in issue_decisions:
        issue_id = str(raw.get("issue_id", "")).strip()
        current_span = str(
            issues.get(issue_id, {}).get("current_persian_quote", "")
        ).strip()
        if current_span and previous.count(current_span) == 1:
            start = previous.find(current_span)
            spans[issue_id] = (start, start + len(current_span))
    overlapping_ids: set[str] = set()
    ordered_spans = sorted(spans.items(), key=lambda item: item[1])
    for position, (issue_id, (start, end)) in enumerate(ordered_spans):
        for other_id, (other_start, other_end) in ordered_spans[position + 1:]:
            if other_start >= end:
                break
            if start < other_end and other_start < end:
                overlapping_ids.update({issue_id, other_id})

    def newly_repeated_adjacent_words(before: str, after: str) -> list[str]:
        token_re = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)

        def repeats(value: str) -> set[str]:
            words = [word.casefold() for word in token_re.findall(value)]
            return {
                words[index]
                for index in range(1, len(words))
                if words[index] == words[index - 1]
            }

        return sorted(repeats(after) - repeats(before))

    for raw in issue_decisions:
        decision = dict(raw)
        issue_id = str(decision.get("issue_id", "")).strip()
        choice = str(decision.get("decision", "")).strip().lower()
        current_span = str(
            issues.get(issue_id, {}).get("current_persian_quote", "")
        ).strip()
        resulting_span = str(decision.get("resulting_span", "")).strip()
        resulting_span = apply_safe_persian_orthography(resulting_span)[0]
        reason = ""
        integrity_payload: dict[str, Any] | None = None

        if choice not in {"accepted", "partially_applied"}:
            reason = "refiner_rejected"
        elif not current_span or not resulting_span:
            reason = "missing_local_span"
        elif normalize_for_match(current_span) == normalize_for_match(resulting_span):
            reason = "no_textual_change"
        elif issue_id in overlapping_ids:
            reason = "overlapping_local_span"
        elif current.count(current_span) != 1:
            reason = "current_span_not_unique"
        elif resulting_span not in proposed:
            reason = "resulting_span_not_in_candidate"
        elif "\n\n" in current_span or "\n\n" in resulting_span:
            reason = "paragraph_boundary_edit"
        elif len(current_span) > 320 or len(resulting_span) > 480:
            reason = "edit_not_local"
        elif not 0.5 <= len(resulting_span) / max(1, len(current_span)) <= 2.0:
            reason = "local_size_ratio_out_of_bounds"
        else:
            candidate = current.replace(current_span, resulting_span, 1)
            repeated_words = newly_repeated_adjacent_words(current, candidate)
            if repeated_words:
                reason = "new_adjacent_word_repetition"
                integrity_payload = {"repeated_words": repeated_words}
            else:
                integrity = integrity_gate.evaluate(
                    source,
                    candidate,
                    previous=current,
                    stage="refinement_local_salvage",
                    protected_terms=protected_terms,
                    protect_inline_english=protect_inline_english,
                    allowed_inline_originals=allowed_inline_originals,
                )
                integrity_payload = integrity.to_dict()
                if integrity.accepted:
                    current = candidate
                    committed += 1
                    reason = "local_edit_committed"
                else:
                    reason = "local_integrity_rejected"

        was_committed = reason == "local_edit_committed"
        decision.update({
            "resulting_span": resulting_span,
            "commit_status": "committed_local" if was_committed else "not_committed",
            "integrity_status": (
                "accepted" if was_committed
                else "rejected" if integrity_payload is not None
                else "not_applicable"
            ),
            "commit_reason": reason,
        })
        enriched.append(decision)
        attempts.append({
            "issue_id": issue_id,
            "decision": choice,
            "committed": was_committed,
            "reason": reason,
            "before_chars": len(current_span),
            "after_chars": len(resulting_span),
            "integrity": integrity_payload,
        })

    return current, enriched, {
        "attempted_count": len(attempts),
        "committed_count": committed,
        "attempts": attempts,
    }


def _critique_candidate_rank(critique: Any) -> tuple[float, float, float, float]:
    scores = [
        float(getattr(critique, name, 0.0))
        for name in ("accuracy", "fluency", "terminology", "register")
    ]
    blocking = len(_blocking_critique_issues(critique))
    return (
        -float(blocking),
        float(getattr(critique, "average", 0.0)),
        min(scores) if scores else 0.0,
        float(getattr(critique, "accuracy", 0.0)),
    )


def _critique_passes_quality_gate(critique: Any, threshold: float) -> bool:
    """Pass when no validated issue justifies another bounded refinement."""
    return bool(
        getattr(critique, "valid", True)
        and not _critique_requires_refinement(critique, threshold)
    )


_HIGH_CONFIDENCE_MINOR_CATEGORIES = frozenset({
    "accuracy",
    "omission",
    "terminology",
})
_HIGH_CONFIDENCE_MINOR_THRESHOLD = 0.85
_STRUCTURAL_FLUENCY_MINOR_THRESHOLD = 0.70
_OBJECTIVE_FLUENCY_MINOR_THRESHOLD = 0.75
_OBJECTIVE_FLUENCY_RATIONALE_RE = re.compile(
    r"\b(?:agreement|ambig(?:uity|uous)|attachment|broken grammar|calque|dependency|"
    r"fragment|modifier stack|nominali[sz]ation|parallelism|participle|referent|"
    r"incomplete (?:clause|coordination|sentence)|missing (?:predicate|verb)|"
    r"malformed|orthograph(?:y|ic)|punctuation|redundan(?:cy|t)|source order|spacing|syntax|"
    r"typograph(?:y|ic)|ungrammatical|word order|zwnj)\b",
    re.IGNORECASE,
)


def _high_confidence_semantic_minor_issues(critique: Any) -> list[dict[str, Any]]:
    """Return grounded minor semantic concerns worth one bounded review pass."""
    routed: list[dict[str, Any]] = []
    for detail in list(getattr(critique, "issue_details", []) or []):
        severity = str(detail.get("severity", "")).strip().lower()
        category = str(detail.get("category", "")).strip().lower()
        try:
            confidence = float(detail.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        source_quote = str(detail.get("source_quote", "")).strip()
        current = " ".join(
            str(detail.get("current_persian_quote", "")).split()
        )
        suggested = " ".join(
            str(detail.get("suggested_correction", "")).split()
        )
        if (
            severity == "minor"
            and category in _HIGH_CONFIDENCE_MINOR_CATEGORIES
            and confidence >= _HIGH_CONFIDENCE_MINOR_THRESHOLD
            and source_quote
            and current
            and suggested
            and current != suggested
        ):
            routed.append(detail)
    return routed


def _high_confidence_structural_fluency_minor_issues(
    critique: Any,
) -> list[dict[str, Any]]:
    """Route long, grounded sentence-level fluency defects for one review pass."""
    routed: list[dict[str, Any]] = []
    for detail in list(getattr(critique, "issue_details", []) or []):
        if (
            str(detail.get("severity", "")).strip().lower() != "minor"
            or str(detail.get("category", "")).strip().lower() != "fluency"
        ):
            continue
        try:
            confidence = float(detail.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        source_quote = " ".join(str(detail.get("source_quote", "")).split())
        current = " ".join(
            str(detail.get("current_persian_quote", "")).split()
        )
        suggested = " ".join(
            str(detail.get("suggested_correction", "")).split()
        )
        evidence = " ".join(
            str(detail.get(field, ""))
            for field in ("issue_id", "rationale", "explanation", "error_type")
        )
        source_words = re.findall(r"[A-Za-z][A-Za-z'\u2019-]*", source_quote)
        current_words = re.findall(r"[\u0600-\u06ff]+", current)
        suggested_words = re.findall(r"[\u0600-\u06ff]+", suggested)
        ratio = len(suggested) / max(1, len(current))
        long_structural_defect = bool(
            confidence >= _STRUCTURAL_FLUENCY_MINOR_THRESHOLD
            and len(source_words) >= 12
            and len(current_words) >= 8
            and len(suggested_words) >= 8
            and 0.75 <= ratio <= 1.5
        )
        objective_local_defect = bool(
            confidence >= _OBJECTIVE_FLUENCY_MINOR_THRESHOLD
            and source_quote
            and current
            and suggested
            and 0.5 <= ratio <= 1.8
            and _OBJECTIVE_FLUENCY_RATIONALE_RE.search(evidence)
        )
        if (
            (long_structural_defect or objective_local_defect)
            and normalize_for_match(current) != normalize_for_match(suggested)
        ):
            routed.append(detail)
    return routed


def _high_confidence_minor_issues(critique: Any) -> list[dict[str, Any]]:
    routed = _high_confidence_semantic_minor_issues(critique)
    seen = {str(item.get("issue_id", "")) for item in routed}
    for detail in _high_confidence_structural_fluency_minor_issues(critique):
        issue_id = str(detail.get("issue_id", ""))
        if issue_id not in seen:
            routed.append(detail)
            seen.add(issue_id)
    return routed


def _critique_requires_refinement(critique: Any, threshold: float) -> bool:
    """Apply Q3 severity routing while preserving legacy score-only critiques."""
    if not getattr(critique, "valid", True):
        return False
    if _blocking_critique_issues(critique):
        return True
    details = list(getattr(critique, "issue_details", []) or [])
    if not details:
        return bool(getattr(critique, "issues", []) or []) and not critique.passes_threshold(
            threshold
        )
    has_substantive_nonblocking = any(
        str(detail.get("severity", "")).strip().lower() in {"critical", "major"}
        for detail in details
    )
    return bool(
        _high_confidence_minor_issues(critique)
        or (
            has_substantive_nonblocking
            and not critique.passes_threshold(threshold)
        )
    )


def _critique_for_event(critique: Any, threshold: float, iteration: int) -> dict[str, Any]:
    blocking_issues = _blocking_critique_issues(critique)
    routed_minor_issues = _high_confidence_minor_issues(critique)
    return {
        "iteration": iteration,
        "valid": bool(getattr(critique, "valid", True)),
        "attempts": int(getattr(critique, "attempts", 1)),
        "validation_errors": list(getattr(critique, "validation_errors", []) or []),
        "threshold": threshold,
        "passes_average_threshold": bool(critique.passes_threshold(threshold)),
        "passes_threshold": not _critique_requires_refinement(critique, threshold),
        "force_refinement": _critique_requires_refinement(critique, threshold),
        "high_confidence_minor_refinement_count": len(routed_minor_issues),
        "high_confidence_minor_refinement_issue_ids": [
            detail.get("issue_id") for detail in routed_minor_issues
        ],
        "scores": {
            "accuracy": critique.accuracy,
            "fluency": critique.fluency,
            "terminology": critique.terminology,
            "register": critique.register,
            "average": critique.average,
        },
        "issue_count": len(critique.issues),
        "blocking_issue_count": len(blocking_issues),
        "blocking_issues": [_truncate_for_event(str(issue), 1000) for issue in blocking_issues],
        "issues": [_truncate_for_event(str(issue), 1000) for issue in critique.issues],
        "issue_details": [
            {
                key: detail.get(key)
                for key in (
                    "issue_id", "category", "severity", "confidence",
                    "source_segment_id", "source_quote", "current_persian_quote",
                    "suggested_correction", "rationale", "risk_flags",
                    "suggestion_orthography_normalized",
                    "source_segment_id_corrected",
                )
            }
            for detail in list(getattr(critique, "issue_details", []) or [])
        ],
        "raw_response_preview": _truncate_for_event(critique.raw_response, 3000),
    }


def _inline_aliases_from_state(noun_state: Any) -> dict[str, list[str]]:
    """Read backward-compatible first-occurrence aliases from memory state."""
    if not isinstance(noun_state, dict):
        return {}
    aliases = noun_state.get("aliases", {})
    eligible = _inline_eligible_nouns_from_state(noun_state)
    if not isinstance(aliases, dict):
        return {}
    return {
        source: [str(value) for value in values if str(value).strip()]
        for source, values in aliases.items()
        if source in eligible and isinstance(values, list)
    }


def _adjacent_source_context(chunks: list[Chunk], index: int) -> str:
    """Return one bounded source neighbour on either side for Q3 review only."""
    parts: list[str] = []
    if index > 0:
        parts.append("Previous source:\n" + chunks[index - 1].text[-1800:])
    if index + 1 < len(chunks):
        parts.append("Next source:\n" + chunks[index + 1].text[:1800])
    return "\n\n".join(parts)


def _bounded_qa_context(chunk: Chunk, adjacent: str, memory: Any) -> str:
    """Build compact critic/refiner context without changing translation input."""
    parts = [
        f"Chapter: {chunk.chapter_title or '(untitled)'}",
        f"Section: {chunk.section_title or '(none)'}",
    ]
    if adjacent:
        parts.append(adjacent[:3600])
    for label, value, limit in (
        ("Style rules", getattr(memory, "style_profile", ""), 1200),
        ("Summary", getattr(memory, "bilingual_summary", ""), 1200),
        ("Relevant proper nouns", getattr(memory, "proper_nouns", ""), 1200),
        ("Relevant long-term memory", getattr(memory, "long_term", ""), 1200),
        ("Recent Persian context", getattr(memory, "short_term", ""), 900),
    ):
        text = str(value or "").strip()
        if text:
            parts.append(f"{label}:\n{text[:limit]}")
    return "\n\n".join(parts)


# Front matter (copyright page, contents, list of tables/abbreviations) yields
# publisher and address noise, not book terminology. In the audited run 44 of 83
# stored proper nouns came from it -- street names, printers, binderies, "ISBN",
# "hardback" -- none of which recur in the book, yet all of which were injected
# into every later translation prompt as established renderings.
_FRONT_MATTER_TITLES = frozenset({
    "contents",
    "table of contents",
    "tables",
    "list of tables",
    "figures",
    "list of figures",
    "abbreviations",
    "list of abbreviations",
    "copyright",
    "copyright page",
    "frontmatter",
    "front matter",
    "title page",
    "dedication",
    "acknowledgements",
    "acknowledgments",
})


def _is_front_matter(chunk: Any) -> bool:
    """Whether a chunk is front matter and must not teach terminology.

    Deliberately conservative: a chunk qualifies only when its chapter title is
    a known front-matter heading, or when it contains no body prose at all.
    Misclassifying a real chapter would DISCARD legitimate terminology, which is
    worse than the noise this filter removes.
    """
    title = (getattr(chunk, "chapter_title", "") or "").strip().casefold()
    title = title.strip(":.-–— ")
    if title in _FRONT_MATTER_TITLES:
        return True
    metadata = getattr(chunk, "metadata", None) or {}
    roles = [str(role) for role in (metadata.get("structural_roles") or [])]
    # No structural evidence => assume body, so nothing is discarded by guessing.
    return bool(roles) and not any(role == "body" for role in roles)


def _chunk_needs_review(db: Any, job_id: str, chunk_index: int) -> bool:
    """Return True when QA recorded an unresolved critic/translator disagreement."""
    events = db.get_chunk_events(job_id, chunk_index)
    last_start = 0
    for i, event in enumerate(events):
        if event.get("event_type") == "chunk_started":
            last_start = i
    review_events = {
        "critique_needs_review",
        "qa_unavailable",
        "integrity_edit_rejected",
        "integrity_final_failed",
        "language_quality_review",
        "back_translation_flagged",
        "glossary_needs_review",
        "chunk_review_required",
    }
    current_events = events[last_start:]
    if any(event.get("event_type") in review_events for event in current_events):
        return True
    actionable_structure = {
        "translation_structure_mismatch",
        "unauthorized_source_correction",
    }
    return any(
        event.get("event_type") == "structure_audit"
        and actionable_structure.intersection(
            event.get("payload", {}).get("classifications", []) or []
        )
        for event in current_events
    )


def _chunk_style_approved(db: Any, job_id: str, chunk_index: int) -> bool:
    """Admit only clean, high-quality prose into the persistent style guide."""
    if _chunk_needs_review(db, job_id, chunk_index):
        return False
    events = db.get_chunk_events(job_id, chunk_index)
    last_start = 0
    for index, event in enumerate(events):
        if event.get("event_type") == "chunk_started":
            last_start = index
    critiques = [
        event.get("payload", {}) or {}
        for event in events[last_start:]
        if event.get("event_type") == "critique_completed"
    ]
    if not critiques:
        return True
    latest = critiques[-1]
    if not bool(latest.get("valid", True)):
        return False
    if int(latest.get("blocking_issue_count", 0) or 0):
        return False
    # Minor, non-blocking observations should not leave a long book without a
    # voice anchor. Major or critical unresolved issues still disqualify the
    # sample, while continuity memory remains available under its own trust.
    issue_details = list(latest.get("issue_details", []) or [])
    if any(
        str(detail.get("severity", "")).strip().casefold()
        in {"critical", "major"}
        for detail in issue_details
        if isinstance(detail, dict)
    ):
        return False
    scores = latest.get("scores", {}) or {}
    dimensions = ("accuracy", "fluency", "terminology", "register")
    try:
        return (
            all(float(scores.get(name, 0)) >= 8.0 for name in dimensions)
            and float(scores.get("average", 0)) >= 9.0
        )
    except (TypeError, ValueError):
        return False


def _chunk_memory_admission(
    db: Any,
    job_id: str,
    chunk_index: int,
) -> dict[str, Any]:
    """Separate continuity context from durable wording/style authority."""
    events = db.get_chunk_events(job_id, chunk_index)
    last_start = 0
    for index, event in enumerate(events):
        if event.get("event_type") == "chunk_started":
            last_start = index
    current_events = events[last_start:]
    needs_review = _chunk_needs_review(db, job_id, chunk_index)
    reasons: list[str] = []
    if needs_review:
        reasons.append("unresolved_qa_review")

    if any(
        event.get("event_type") == "mqm_minor_only_deferred"
        for event in current_events
    ):
        reasons.append("deferred_mqm_advice")

    critiques = [
        event.get("payload", {}) or {}
        for event in current_events
        if event.get("event_type") == "critique_completed"
    ]
    if critiques:
        latest = critiques[-1]
        if not bool(latest.get("valid", True)):
            reasons.append("invalid_final_critique")
        if int(latest.get("blocking_issue_count", 0) or 0):
            reasons.append("blocking_critique_issue")
        scores = latest.get("scores", {}) or {}
        try:
            average = float(scores.get("average", 0) or 0)
            accuracy = float(scores.get("accuracy", 0) or 0)
            terminology = float(scores.get("terminology", 0) or 0)
        except (TypeError, ValueError):
            average = accuracy = terminology = 0.0
        if average < 8.5:
            reasons.append("final_critique_below_memory_floor")
        if accuracy < 8.0 or terminology < 8.0:
            reasons.append("semantic_dimension_below_memory_floor")

    reasons = list(dict.fromkeys(reasons))
    durable_reliable = not reasons
    return {
        "quality_approved": not needs_review,
        "long_term_reliable": durable_reliable,
        "short_term_trust": (
            "trusted" if durable_reliable else "advisory_review"
        ),
        "reliability_reasons": reasons,
        "continuity_retained": True,
    }


def _chapter_summary_memory_admission(
    db: Any,
    job_id: str,
    chunks: list[Chunk],
    boundary_index: int,
) -> dict[str, Any]:
    """Describe source-memory trust for a chapter summary without omitting it."""
    chapter_position = int(
        chunks[boundary_index].metadata.get("chapter_position", 1)
    )
    reasons: list[str] = []
    contributing = 0
    for index, chunk in enumerate(chunks[:boundary_index + 1]):
        if int(chunk.metadata.get("chapter_position", 1)) != chapter_position:
            continue
        policies = [
            event.get("payload", {}) or {}
            for event in db.get_chunk_events(job_id, index)
            if event.get("event_type") == "memory_update_policy"
        ]
        if not policies:
            reasons.append("missing_memory_policy")
            continue
        contributing += 1
        policy = policies[-1]
        if not bool(policy.get("long_term_reliable", False)):
            policy_reasons = list(policy.get("reliability_reasons", []) or [])
            reasons.extend(policy_reasons or ["advisory_contributing_chunk"])
    reasons = list(dict.fromkeys(str(reason) for reason in reasons if str(reason)))
    return {
        "input_trust": (
            "reviewed_inputs" if contributing and not reasons else "advisory_inputs"
        ),
        "trust_reasons": reasons,
        "contributing_chunks": contributing,
        "context_retained": True,
    }


def _reconcile_committed_terminology(
    db: Any,
    job_id: str,
    chunk_index: int,
    memory_manager: MemoryManager,
    source_text: str,
    final_translation: str,
) -> dict[str, Any]:
    """Promote only committed, source-grounded corrections into safe memory."""
    report: dict[str, Any] = {
        "reconciled": [],
        "aliases_added": [],
        "context_deferred": [],
        "skipped": [],
    }
    if _chunk_needs_review(db, job_id, chunk_index):
        report["policy"] = "skipped_unresolved_chunk"
        return report

    issues = {
        str(item.get("issue_id", "")): item
        for item in db.get_qa_issues(job_id, chunk_index)
        if str(item.get("issue_id", ""))
    }
    known = memory_manager.proper_nouns.all_nouns()
    for decision in db.get_issue_decisions(job_id, chunk_index):
        decision_payload = decision.get("payload", {})
        if not isinstance(decision_payload, dict):
            decision_payload = {}
        commit_status = str(
            decision_payload.get("commit_status", "")
        )
        if not commit_status.startswith("committed"):
            continue
        issue_id = str(decision.get("issue_id", ""))
        issue = issues.get(issue_id, {})
        category = str(issue.get("category", "")).strip().lower()
        if category not in {"accuracy", "terminology", "name"}:
            continue
        source_quote = " ".join(str(issue.get("source_quote", "")).split())
        matching_sources = [
            source for source in known
            if re.search(
                rf"(?<!\w){re.escape(source)}(?!\w)",
                source_quote,
                flags=re.IGNORECASE,
            )
            and re.search(
                rf"(?<!\w){re.escape(source)}(?!\w)",
                source_text,
                flags=re.IGNORECASE,
            )
        ]
        derived_source = False
        container_sources: list[str] = []
        if matching_sources:
            source = max(matching_sources, key=len)
        else:
            quote_words = re.findall(r"[A-Za-z][A-Za-z'\-]*", source_quote)
            if not (
                1 <= len(quote_words) <= 4
                and len(source_quote) <= 80
                and re.fullmatch(
                    r"[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,3}",
                    source_quote,
                )
                and re.search(
                    rf"(?<!\w){re.escape(source_quote)}(?!\w)",
                    source_text,
                    re.IGNORECASE,
                )
            ):
                continue
            source = source_quote
            derived_source = True
            container_sources = [
                known_source for known_source in known
                if re.search(
                    rf"(?<!\w){re.escape(source_quote)}(?!\w)",
                    known_source,
                    re.IGNORECASE,
                )
                and re.search(
                    rf"(?<!\w){re.escape(known_source)}(?!\w)",
                    source_text,
                    re.IGNORECASE,
                )
            ]
        if category == "accuracy":
            quote_words = re.findall(r"[A-Za-z][A-Za-z'\-]*", source_quote)
            source_words = re.findall(r"[A-Za-z][A-Za-z'\-]*", source)
            if len(quote_words) > len(source_words) + 2:
                continue

        target_candidates = [
            str(
                decision.get("resulting_span", "")
                or decision_payload.get("resulting_span", "")
            ).strip(),
            str(issue.get("suggested_correction", "")).strip(),
        ]
        target = next((
            value for value in target_candidates
            if value
            and len(value) <= 120
            and "\n" not in value
            and not re.search(r"[.!?\u061f]", value)
            and len(re.findall(r"[\u0600-\u06ff]+", value)) <= 10
            and normalize_for_match(value) in normalize_for_match(final_translation)
        ), "")
        if not target:
            report["skipped"].append({
                "issue_id": issue_id,
                "source": source,
                "reason": "no_compact_committed_target_in_final_translation",
            })
            continue

        if not is_reusable_terminology_mapping(source, target):
            report["context_deferred"].append({
                "issue_id": issue_id,
                "source": source,
                "target": target,
                "reason": "contextual_correction_not_reusable_as_global_term",
            })
            continue

        provenance = memory_manager.proper_nouns.provenance_for(source)
        if int(provenance.get("authority", 50)) >= 100:
            report["skipped"].append({
                "issue_id": issue_id,
                "source": source,
                "reason": "curated_authority_preserved",
            })
            continue
        outcome = memory_manager.proper_nouns.add_noun(
            source,
            target,
            category=(
                "term" if derived_source
                else memory_manager.proper_nouns.category_for(source)
            ),
            provenance="accepted_correction",
        )
        if outcome.get("action") in {"replaced_lower_authority", "confirmed"}:
            report["reconciled"].append({"issue_id": issue_id, **outcome})
        elif memory_manager.proper_nouns.add_alias(source, target):
            report["aliases_added"].append({
                "issue_id": issue_id,
                "source": source,
                "target": target,
            })
        if outcome.get("action") in {
            "added", "replaced_lower_authority", "confirmed"
        }:
            for container in container_sources:
                if memory_manager.proper_nouns.mark_context_conflict(
                    container,
                    corrected_source=source,
                    corrected_target=target,
                ):
                    report["context_deferred"].append({
                        "issue_id": issue_id,
                        "source": container,
                        "conflict_source": source,
                        "accepted_target": target,
                    })
    report["policy"] = (
        "Only committed compact, context-independent terminology corrections from "
        "review-clean chunks may supersede lower-authority automatic memory; "
        "sentence-specific corrections remain in continuity memory and curated "
        "mappings remain protected."
    )
    return report


def _advisory_terminology_consistency(
    memory_manager: MemoryManager,
    source_text: str,
    final_translation: str,
) -> dict[str, Any]:
    """Report established rendering drift without forcing an automatic edit."""
    checked: list[dict[str, Any]] = []
    inconsistent: list[dict[str, Any]] = []
    noun_state = memory_manager.proper_nouns.serialize()
    for source, target in (noun_state.get("nouns", {}) or {}).items():
        if memory_manager.proper_nouns.is_context_deferred(source):
            continue
        if not re.search(
            rf"(?<!\w){re.escape(source)}(?!\w)", source_text, re.IGNORECASE
        ):
            continue
        if term_occurs_only_in_citations(source_text, source):
            continue
        variants = [target] + memory_manager.proper_nouns.aliases_for(source)
        present = next((
            value for value in variants
            if value and target_present(value, final_translation)
        ), "")
        evidence = {
            "source": source,
            "expected": target,
            "matched_variant": present,
            "provenance": memory_manager.proper_nouns.provenance_for(source),
        }
        checked.append(evidence)
        if not present:
            inconsistent.append(evidence)
    return {
        "checked_count": len(checked),
        "inconsistent_count": len(inconsistent),
        "inconsistencies": inconsistent[:50],
        "policy": (
            "Advisory evidence only; no translation is rewritten. Curated glossary "
            "compliance remains the mandatory terminology authority."
        ),
    }


def _chunk_review_reason_payload(
    db: Any,
    job_id: str,
    chunk_index: int,
) -> dict[str, Any] | None:
    """Build one explicit review summary from the current chunk attempt."""
    events = db.get_chunk_events(job_id, chunk_index)
    last_start = 0
    for index, event in enumerate(events):
        if event.get("event_type") == "chunk_started":
            last_start = index
    current = events[last_start:]
    reason_map = {
        "critique_needs_review": "quality_disagreement",
        "qa_unavailable": "qa_unavailable",
        "integrity_edit_rejected": "automatic_edit_rejected",
        "integrity_final_failed": "final_integrity_failed",
        "language_quality_review": "final_language_quality_risk",
        "back_translation_flagged": "back_translation_risk",
        "glossary_needs_review": "glossary_noncompliance",
    }
    reasons: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for event in current:
        event_type = str(event.get("event_type", ""))
        payload = event.get("payload", {}) or {}
        if event_type == "chunk_review_required":
            existing = list(payload.get("reasons", []) or [])
            if not existing and payload.get("reason"):
                existing = [{
                    "reason": str(payload.get("reason")),
                    "detail": str(payload.get("detail") or ""),
                    "source_event": event_type,
                }]
            for item in existing:
                reason = str(item.get("reason", "")).strip()
                detail = str(item.get("detail", "")).strip()
                if reason and (reason, detail) not in seen:
                    seen.add((reason, detail))
                    reasons.append({
                        "reason": reason,
                        "detail": detail,
                        "source_event": str(
                            item.get("source_event") or event_type
                        ),
                    })
            continue
        reason = reason_map.get(event_type)
        if not reason:
            continue
        detail = str(
            payload.get("review_reason")
            or payload.get("component")
            or payload.get("stage")
            or ""
        )
        key = (reason, detail)
        if key in seen:
            continue
        seen.add(key)
        reasons.append({
            "reason": reason,
            "detail": detail,
            "source_event": event_type,
        })
    if not reasons:
        return None
    return {
        "reasons": reasons,
        "reason_codes": [item["reason"] for item in reasons],
        "translation_available": True,
        "automatic_pipeline_continued": True,
        "human_review_required": True,
    }


def _explicit_chunk_review_payload(
    reason: str,
    *,
    detail: str = "",
    message: str = "",
    **evidence: Any,
) -> dict[str, Any]:
    """Build the canonical review schema for direct review decisions."""
    item = {
        "reason": str(reason).strip() or "unspecified",
        "detail": str(detail).strip(),
        "source_event": "chunk_review_required",
    }
    return {
        "reason": item["reason"],
        "reason_codes": [item["reason"]],
        "reasons": [item],
        "message": message,
        "translation_available": True,
        "automatic_pipeline_continued": True,
        "human_review_required": True,
        **evidence,
    }


def _ensure_chunk_review_reason(
    db: Any,
    job_id: str,
    chunk_index: int,
) -> dict[str, Any] | None:
    """Persist a canonical reason once for each current chunk attempt."""
    payload = _chunk_review_reason_payload(db, job_id, chunk_index)
    if payload is None:
        return None
    events = db.get_chunk_events(job_id, chunk_index)
    last_start = 0
    for index, event in enumerate(events):
        if event.get("event_type") == "chunk_started":
            last_start = index
    summaries = [
        event for event in events[last_start:]
        if event.get("event_type") == "chunk_review_required"
    ]
    if not summaries or summaries[-1].get("payload") != payload:
        db.log_chunk_event(
            job_id, chunk_index, "chunk_review_required", payload
        )
    return payload


def _compliance_report_for_event(report: Any) -> dict[str, Any]:
    violations = []
    for violation in getattr(report, "violations", []):
        violations.append({
            "term": getattr(violation, "term", ""),
            "expected": getattr(violation, "expected", ""),
            "actual": getattr(violation, "actual", ""),
            "status": getattr(violation, "status", ""),
            "chunk_location": getattr(violation, "chunk_location", ""),
        })
    return {
        "compliant": bool(getattr(report, "compliant", False)),
        "total_checked": int(getattr(report, "total_checked", 0)),
        "citation_exemptions": list(
            getattr(report, "citation_exemptions", []) or []
        ),
        "violation_count": len(violations),
        "violations": violations,
    }


class ParagraphIdentityError(ValueError):
    """Raised when a chunk translation violates strict paragraph identity.

    A ValueError subclass so existing handlers keep working unchanged.
    """


def _used_paragraph_protocol(chunk: Chunk) -> bool:
    """Whether *chunk* was actually translated under the marker protocol.

    Mirrors ``use_paragraph_protocol`` in ``_translate_single_chunk`` exactly:
    ``encode_paragraphs`` emits one marker per ``split_paragraphs`` element, so
    ``len(split_paragraphs(text)) > 1`` is the same condition.

    This alignment is the whole fix. Translation applies the protocol only when
    a chunk has MORE THAN ONE paragraph, but assembly used to enforce strict
    identity whenever paragraph_protocol_version was set -- which SemanticChunker
    sets on EVERY chunk. So a single-paragraph chunk was translated with no
    markers, no protocol instruction and no repair pass, then judged as if it
    had them. A model returning that one paragraph as two blank-line-separated
    blocks aborted the run after all LLM spend.
    """
    raw_version = chunk.metadata.get("paragraph_protocol_version", 0)
    # A non-integer version falls back to 0, i.e. NOT strict. That is the
    # fail-safe direction: lenient alignment degrades, strict alignment aborts.
    version = raw_version if isinstance(raw_version, int) else 0
    return bool(len(split_paragraphs(chunk.text)) > 1 and version >= 1)


def _align_chunk_translation(
    *,
    original_paragraphs: list[Any],
    para_indices: list[int],
    tgt_paras: list[str],
    chunk_translation: str,
    strict_paragraph_identity: bool = False,
) -> list[tuple[int, str]]:
    """Align translated paragraphs to source paragraph ids, protecting headings.

    LLMs sometimes obey the paragraph count but omit a short heading at the
    start of a chunk. Without this guard, the first body paragraph becomes a
    DOCX heading and the real heading disappears.
    """
    has_heading = any(
        pid < len(original_paragraphs)
        and original_paragraphs[pid].heading_level is not None
        for pid in para_indices
    )
    if strict_paragraph_identity:
        if len(para_indices) != len(tgt_paras):
            raise ParagraphIdentityError(
                "Stable paragraph identity failed during assembly: "
                f"expected {len(para_indices)}, received {len(tgt_paras)}."
            )
        return list(zip(para_indices, tgt_paras))
    if not has_heading:
        if len(para_indices) == len(tgt_paras):
            return list(zip(para_indices, tgt_paras))

        src_paras_chunk = [
            original_paragraphs[pid].text.strip()
            for pid in para_indices
            if pid < len(original_paragraphs)
        ]
        weights = [len(s) for s in src_paras_chunk]
        parts = _distribute_translation(chunk_translation, len(para_indices), weights)
        return list(zip(para_indices, parts))

    aligned: list[tuple[int, str]] = []
    body_indices: list[int] = []
    target_pos = 0

    for pid in para_indices:
        if pid >= len(original_paragraphs):
            continue
        orig_para = original_paragraphs[pid]
        if orig_para.heading_level is None:
            body_indices.append(pid)
            continue

        candidate = tgt_paras[target_pos] if target_pos < len(tgt_paras) else ""
        if _looks_like_heading_translation(candidate, orig_para.text):
            aligned.append((pid, candidate))
            target_pos += 1
        else:
            aligned.append((pid, _fallback_heading_translation(orig_para.text)))

    remaining_targets = tgt_paras[target_pos:]
    if body_indices:
        if len(remaining_targets) == len(body_indices):
            body_parts = remaining_targets
        else:
            body_text = "\n\n".join(remaining_targets).strip()
            body_weights = [
                len(original_paragraphs[pid].text.strip())
                for pid in body_indices
                if pid < len(original_paragraphs)
            ]
            body_parts = _distribute_translation(body_text, len(body_indices), body_weights)
        aligned.extend(zip(body_indices, body_parts))
    elif remaining_targets and aligned:
        pid, current = aligned[-1]
        aligned[-1] = (pid, f"{current}\n\n" + "\n\n".join(remaining_targets))

    order = {pid: i for i, pid in enumerate(para_indices)}
    aligned.sort(key=lambda item: order.get(item[0], 10**9))
    return aligned


class PipelineResult:
    """The result returned upon successful pipeline completion."""

    def __init__(self, output_path: Path, total_chunks: int, duration: float, warnings: list[str] | None = None) -> None:
        self.output_path = output_path
        self.total_chunks = total_chunks
        self.duration = duration
        self.warnings = warnings or []

    @property
    def duration_str(self) -> str:
        minutes, seconds = divmod(int(self.duration), 60)
        return f"{minutes}m {seconds}s"


class TranslationPipeline:
    """The main entry point for running a translation job.

    Coordinates document parsing, chunking, LLM execution stages, and final file export.
    """

    def __init__(self, config: TarjomehConfig) -> None:
        self.config = config
        self.llm_client = LLMClient(config)
        self.critic_client = self._build_critic_client(config)
        self.db = JobDatabase()
        self.current_job_id: str | None = None
        self._async_loop: Any = None
        self._async_thread: threading.Thread | None = None
        self._async_loop_lock = threading.Lock()
        self._async_loop_ready = threading.Event()
        self._closed = False
        observed: set[int] = set()
        for client in (self.llm_client, self.critic_client):
            if id(client) not in observed:
                client.set_attempt_observer(self._record_llm_attempt)
                observed.add(id(client))

    def _record_llm_attempt(self, event: dict[str, Any]) -> None:
        """Persist sanitized provider completion evidence for the active job."""
        payload = dict(event)
        job_id = payload.pop("job_id", None) or self.current_job_id
        chunk_index = payload.pop("chunk_index", None)
        if not job_id:
            return
        if chunk_index is None:
            # Keep the human-readable log while also retaining the full
            # sanitized evidence used by VPS audits and QA diagnostics.
            self.db.log_chunk_event(
                job_id,
                -1,
                "llm_call_attempt",
                payload,
            )
            self.db.log_event(
                job_id,
                "INFO" if payload.get("success") else "WARNING",
                "LLM {operation} attempt {attempt}/{max_attempts}: {result}".format(
                    operation=payload.get("operation", "completion"),
                    attempt=payload.get("attempt", 1),
                    max_attempts=payload.get("max_attempts", 1),
                    result=(
                        payload.get("finish_reason") or "completed"
                        if payload.get("success")
                        else payload.get("failure_reason", "failed")
                    ),
                ),
            )
            return
        self.db.log_chunk_event(
            job_id,
            int(chunk_index),
            "llm_call_attempt",
            payload,
        )

    def _build_critic_client(self, config: TarjomehConfig) -> LLMClient:
        """Build the judge client for critique / back-translation QA.

        When ``[llm.critic]`` is active, a second :class:`LLMClient` is
        constructed against the critic model so translations are graded by an
        independent (ideally stronger) judge instead of the model scoring its
        own output. Unset critic fields inherit from the main ``[llm]`` block.
        Falls back to the translator client when the critic is not configured.
        """
        critic = getattr(config.llm, "critic", None)
        if critic is None or not critic.is_active:
            return self.llm_client

        import copy
        critic_config = copy.deepcopy(config)
        if critic.provider:
            critic_config.llm.provider = critic.provider
        if critic.model:
            critic_config.llm.model = critic.model
            # Ollama reads its model name from llm.ollama.model
            critic_config.llm.ollama.model = critic.model
        if any(k.strip() for k in critic.api_keys):
            critic_config.llm.openrouter.api_keys = list(critic.api_keys)
        if critic.api_base.strip():
            # Judge can use a different endpoint than the translator
            # (e.g. translator via 9router, judge via OpenRouter directly).
            critic_config.llm.openrouter.api_base = critic.api_base.strip()
        critic_config.llm.temperature = critic.temperature
        # Critic recovery is independent from translator recovery. Predictive
        # sizing may change only the allowance; bounded follow-ups use these.
        critic_config.llm.recovery.model = critic.recovery_model.strip()
        critic_config.llm.recovery.max_attempts = critic.recovery_max_attempts
        critic_config.llm.recovery.max_tokens = max(
            critic_config.llm.max_tokens,
            critic.recovery_max_tokens,
        )
        critic_config.llm.recovery.expanded_final_attempt = True

        logger.info(
            "Critic model active: %s via %s (translator: %s)",
            critic_config.llm.model if critic_config.llm.provider == "openrouter"
            else critic_config.llm.ollama.model,
            critic_config.llm.provider,
            config.llm.model,
        )
        return LLMClient(critic_config)

    def _get_parser(self, file_path: Path) -> BaseParser:
        """Resolve and instantiate the correct parser for the file extension."""
        suffix = file_path.suffix.lower()
        if suffix not in EXTENSION_PARSER_MAP:
            raise ValueError(f"Unsupported file format: {suffix}")

        import importlib
        module_path, class_name = EXTENSION_PARSER_MAP[suffix].rsplit(".", 1)
        module = importlib.import_module(module_path)
        parser_cls = getattr(module, class_name)
        return parser_cls()

    def _send_webhook(self, event: str, message: str, status: str) -> None:
        """Send webhook notification for pipeline status updates."""
        url = self.config.notifications.webhook_url
        if not url:
            return

        if status == JobStatus.COMPLETED and not self.config.notifications.notify_on_complete:
            return
        if status in (JobStatus.FAILED, JobStatus.PAUSED_ERROR) and not self.config.notifications.notify_on_error:
            return

        try:
            import httpx
            payload = {
                "event": event,
                "job_id": self.current_job_id,
                "status": status,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
            }
            httpx.post(url, json=payload, timeout=5.0)
        except Exception as e:
            logger.warning("Failed to send pipeline webhook: %s", e)

    def _run_async(self, coro: Any) -> Any:
        """Run a coroutine on the pipeline's single owned event loop.

        One loop per operation defeats HTTP connection pooling and can bind a
        cached ``AsyncClient`` to a loop that has already been destroyed.  A
        dedicated loop also works when the caller already owns an event loop
        (ASGI, notebooks) without nesting ``run_until_complete``.
        """
        import asyncio

        if not hasattr(self, "_async_loop_lock"):
            self._async_loop = None
            self._async_thread = None
            self._async_loop_lock = threading.Lock()
            self._async_loop_ready = threading.Event()
            self._closed = False
        with self._async_loop_lock:
            if self._closed:
                close_coro = getattr(coro, "close", None)
                if callable(close_coro):
                    close_coro()
                raise RuntimeError("TranslationPipeline is closed.")
            if self._async_loop is None:
                loop = asyncio.new_event_loop()
                self._async_loop = loop
                self._async_loop_ready.clear()

                def run_loop() -> None:
                    asyncio.set_event_loop(loop)
                    self._async_loop_ready.set()
                    try:
                        loop.run_forever()
                    finally:
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()
                        if pending:
                            loop.run_until_complete(
                                asyncio.gather(*pending, return_exceptions=True)
                            )
                        loop.run_until_complete(loop.shutdown_asyncgens())
                        loop.close()

                self._async_thread = threading.Thread(
                    target=run_loop,
                    name="tarjomeh-async",
                    daemon=True,
                )
                self._async_thread.start()
            loop = self._async_loop

        self._async_loop_ready.wait()
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    def _prepare_book_research(
        self,
        job_id: str,
        document: Document,
        progress_callback: Callable[[str, float, str], None] | None = None,
    ) -> dict[str, Any] | None:
        """Load or run the opt-in, one-time review-only research pass."""
        artifact = self.db.get_job_artifact(job_id, "book_research")
        if artifact is not None or not self.config.translation.enable_book_research:
            return artifact

        if progress_callback:
            progress_callback(
                "Research",
                0.08,
                "Researching book context and terminology suggestions...",
            )
        from tarjomeh.context.book_researcher import BookResearcher

        result = self._run_async(
            BookResearcher(self.config, self.llm_client).research(document)
        )
        artifact = result.to_dict()
        self.db.save_job_artifact(job_id, "book_research", artifact)
        if artifact.get("status") in {
            "completed", "completed_without_suggestions", "degraded", "partial"
        }:
            status = artifact.get("status")
            self.db.log_event(
                job_id,
                "WARNING" if status in {"degraded", "partial"} else "INFO",
                f"Book research {status} with "
                f"{len(artifact.get('terms', []))} reviewable suggestion(s) "
                f"and {len(artifact.get('sources', []))} source result(s).",
            )
        else:
            self.db.log_event(
                job_id,
                "WARNING",
                "Book research failed; translation will continue without a seed. "
                + str(artifact.get("error", "")),
            )
        return artifact

    def close(self) -> None:
        """Release HTTP resources held by the translator and critic clients.

        close()/aclose() previously had zero call sites anywhere in src/, so
        every job leaked its connection pools for the process lifetime.
        """
        if getattr(self, "_closed", False):
            return

        seen: set[int] = set()
        for client in (
            getattr(self, "llm_client", None),
            getattr(self, "critic_client", None),
        ):
            # _build_critic_client returns self.llm_client when the critic is
            # disabled, so the same object can appear twice.
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            try:
                self._run_async(client.aclose())
            except Exception:
                logger.debug("Async client close failed", exc_info=True)
            try:
                client.close()
            except Exception:
                logger.debug("Sync client close failed", exc_info=True)

        if not hasattr(self, "_async_loop_lock"):
            self._async_loop = None
            self._async_thread = None
            self._async_loop_lock = threading.Lock()
            self._async_loop_ready = threading.Event()
        with self._async_loop_lock:
            self._closed = True
            loop = self._async_loop
            thread = self._async_thread
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10.0)

    def __enter__(self) -> TranslationPipeline:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @staticmethod
    def _chunk_fingerprint(text: str) -> str:
        """Whitespace-insensitive fingerprint of chunk source text."""
        normalized = " ".join((text or "").split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _verify_resume_alignment(self, job_id: str, chunks: list[Chunk]) -> None:
        """Refuse to resume when the input no longer matches the saved chunks.

        Completed translations are reused by chunk index alone, so a source
        edit between pause and resume would silently attach translation N to
        different source text. Fail loudly instead of corrupting the book.
        """
        saved = {
            int(record["chunk_index"]): (record.get("text") or "")
            for record in self.db.get_chunks(job_id)
        }
        if not saved:
            return

        mismatched = [
            idx for idx, text in sorted(saved.items())
            if idx < len(chunks)
            and self._chunk_fingerprint(text)
            != self._chunk_fingerprint(chunks[idx].text)
        ]
        dropped = sorted(idx for idx in saved if idx >= len(chunks))
        added = max(0, len(chunks) - len(saved))
        if not mismatched and not dropped and not added:
            return

        message = (
            "Cannot resume: the input document no longer matches the chunks "
            f"saved for this job (saved={len(saved)} current={len(chunks)} "
            f"mismatched={mismatched[:10]} dropped={dropped[:10]} "
            f"added={added}). Resuming would attach existing translations to "
            "different source text. Restore the original input file, or start "
            "a new job."
        )
        self.db.log_event(job_id, "ERROR", message)
        self.db.update_job_status(job_id, JobStatus.PAUSED_ERROR, message)
        raise ResumeSourceMismatchError(message)

    def run(
        self,
        input_path: Path,
        output_path: Path | None = None,
        job_id: str | None = None,
        progress_callback: Callable[[str, float, str], None] | None = None,
    ) -> PipelineResult:
        """Execute the translation pipeline on the input document."""
        t0 = time.monotonic()
        input_path = Path(input_path)
        self.warnings = []

        # 1. Establish Job ID and Database Record
        is_resume = False
        if job_id:
            self.current_job_id = job_id
            job_record = self.db.get_job(job_id)
            if job_record:
                is_resume = True
                logger.info("Resuming existing job: %s", job_id)
            else:
                logger.info("Starting new job with supplied ID: %s", job_id)
                self.db.create_job(
                    job_id, input_path, self.config.to_dict(redact_secrets=True)
                )
        else:
            self.current_job_id = uuid.uuid4().hex[:12]
            self.db.create_job(
                self.current_job_id,
                input_path,
                self.config.to_dict(redact_secrets=True),
            )

        job_id = self.current_job_id
        # create_job() writes PENDING and only the resume branch used to write
        # RUNNING, so a fresh job stayed PENDING for its entire life. Both
        # render identically in the UI, which hid the difference. Mark every
        # path explicitly.
        self.db.update_job_status(job_id, JobStatus.RUNNING)

        # OCR runs AFTER the job record exists, so jobs.input_path holds the
        # ORIGINAL file (resume could not find the old temp path), and the
        # derived PDF lands in a durable, job-scoped directory instead of a
        # tempfile.mkdtemp() that nothing ever removed.
        if input_path.suffix.lower() == ".pdf" and getattr(
            self.config.pdf, "enable_ocr", False
        ):
            ocr_artifact = self.db.get_job_artifact(job_id, "ocr_output") or {}
            ocr_cached_path = (
                Path(ocr_artifact["path"]) if ocr_artifact.get("path") else None
            )
            if ocr_cached_path is not None and ocr_cached_path.is_file():
                logger.info("Reusing existing OCR output for job %s", job_id)
                input_path = ocr_cached_path
            else:
                if progress_callback:
                    progress_callback(
                        "Ingestion", 0.02, "Running OCR Preprocessing..."
                    )
                from tarjomeh.parsers.ocr_preprocessor import OCRPreprocessor

                ocr_processor = OCRPreprocessor(
                    output_dir=Path("jobs") / "ocr" / job_id
                )
                input_path = ocr_processor.preprocess(input_path)
                self.db.save_job_artifact(job_id, "ocr_output", {
                    "path": str(input_path),
                })

        # Determine default output path if not provided
        if output_path is None:
            fmt = self.config.output.format.lower()
            extension = _output_extension(fmt)
            output_path = input_path.parent / f"{input_path.stem}_translated.{extension}"
        else:
            output_path = Path(output_path)

        # 2. Parse and Chunk
        if progress_callback:
            progress_callback("Ingestion", 0.05, "Parsing document...")

        document: Document
        research_document: Document
        chunks: list[Chunk]

        if is_resume:
            # Reconstruct document structure and loaded chunks from DB
            parser = self._get_parser(input_path)
            structure_artifact = self.db.get_job_artifact(
                job_id, "document_structure_version"
            ) or {}
            structure_version = int(structure_artifact.get("version", 1))
            if hasattr(parser, "structure_version"):
                parser.structure_version = structure_version
            document = parser.parse(input_path)
            build_chapter_manifest(document)
            research_document = document
            document = apply_chapter_selection(
                document, self.config.translation.chapter_selection
            )
            
            # Re-generate chunks to match indices
            def token_counter(text: str) -> int:
                return self.llm_client.count_tokens(text)

            if self.config.chunking.strategy == "semantic":
                chunker = SemanticChunker(
                    max_tokens=self.config.chunking.max_chunk_tokens,
                    overlap_sentences=self.config.chunking.overlap_sentences,
                    token_counter=token_counter
                )
            else:
                chunker = FixedChunker(
                    max_tokens=self.config.chunking.max_chunk_tokens,
                    token_counter=token_counter
                )
            chunks = chunker.chunk(document)
            # Translations below are reused by chunk index alone. Verify the
            # source still matches before trusting any of them.
            self._verify_resume_alignment(job_id, chunks)
        else:
            # Fresh parse and chunk
            parser = self._get_parser(input_path)
            document = parser.parse(input_path)
            structure_version = int(getattr(parser, "structure_version", 1))
            self.db.save_job_artifact(job_id, "document_structure_version", {
                "version": structure_version,
                "parser": type(parser).__name__,
            })
            chapter_manifest = build_chapter_manifest(document)
            research_document = document
            self.db.save_job_artifact(job_id, "chapter_manifest", {
                "chapters": chapter_manifest,
                "selected_positions": list(
                    self.config.translation.chapter_selection
                ),
                "stop_after_chapter": self.config.translation.stop_after_chapter,
                "pause_after_each_chapter": (
                    self.config.translation.pause_after_each_chapter
                ),
            })
            document = apply_chapter_selection(
                document, self.config.translation.chapter_selection
            )

            def token_counter(text: str) -> int:
                return self.llm_client.count_tokens(text)

            if self.config.chunking.strategy == "semantic":
                chunker = SemanticChunker(
                    max_tokens=self.config.chunking.max_chunk_tokens,
                    overlap_sentences=self.config.chunking.overlap_sentences,
                    token_counter=token_counter
                )
            else:
                chunker = FixedChunker(
                    max_tokens=self.config.chunking.max_chunk_tokens,
                    token_counter=token_counter
                )
            chunks = chunker.chunk(document)
            self.db.save_chunks(job_id, chunks)

        structure_audit = research_document.metadata.get(
            "pdf_structure_audit"
        )
        if isinstance(structure_audit, dict):
            self.db.save_job_artifact(
                job_id, "pdf_structure_audit", structure_audit
            )
            removed_count = int(
                structure_audit.get("removed_furniture_count", 0)
            )
            table_count = int(structure_audit.get("table_block_count", 0))
            self.db.log_event(
                job_id,
                "INFO",
                "PDF structure analysis removed "
                f"{removed_count} recurrent header/footer item(s) and detected "
                f"{table_count} table block(s).",
            )
            if table_count:
                self.db.log_event(
                    job_id,
                    "WARNING",
                    "Detected table-like PDF content is preserved in reading order; "
                    "complex table layout may require manual DOCX formatting.",
                )

        total_chunks = len(chunks)
        if total_chunks == 0:
            raise ValueError("Document contains no translatable content.")

        # 3. Setup Glossary
        glossary_manager = GlossaryManager()
        glossary_paths: list[Path] = []
        primary_glossary = getattr(self.config.glossary, "path", "")
        if primary_glossary:
            glossary_paths.append(Path(primary_glossary))
        for extra_path in getattr(self.config.glossary, "paths", []) or []:
            p = Path(extra_path)
            if p not in glossary_paths:
                glossary_paths.append(p)
        glossary_manager.load_many(glossary_paths, ignore_missing=True)
        if is_resume:
            extracted_artifact = self.db.get_job_artifact(
                job_id, "auto_extracted_terms"
            ) or {}
            persisted_terms = extracted_artifact.get("terms", {})
            if isinstance(persisted_terms, dict):
                glossary_manager.merge_auto_extracted(persisted_terms)

        # 4. Setup Memory Manager
        memory_manager = MemoryManager(self.config)
        if is_resume:
            saved_mem = self.db.get_memory_state(job_id)
            if saved_mem:
                memory_manager.from_dict(saved_mem)

        # 5. Extract terms (TOC + first chapter) if configured and new job
        if not is_resume and self.config.glossary.enable_auto_extraction:
            if progress_callback:
                progress_callback("Glossary", 0.10, "Extracting specialized terms...")
            try:
                self.db.log_chunk_event(
                    job_id, 0, "auto_extraction_started", {
                        "source_chars": len(chunks[0].text),
                    }
                )
                # Run NER extraction on first chunk as a proxy for TOC / Ch 1
                first_text = chunks[0].text
                sys_prompt = "You are a terminology extraction assistant."
                ner_response = self.llm_client.complete(
                    messages=[{"role": "user", "content": GLOSSARY_EXTRACT_PROMPT.format(text=first_text)}],
                    system_prompt=sys_prompt,
                    response_format={"type": "json_object"},
                    _operation="auto_term_extraction",
                )
                ner_data = parse_structured_output(
                    ner_response, expected=(list, dict)
                )
                # The model may return either a JSON object ({"terms": [...]})
                # or a bare JSON array of term objects — handle both shapes.
                if isinstance(ner_data, list):
                    extracted_terms = ner_data
                elif isinstance(ner_data, dict):
                    extracted_terms = ner_data.get("terms", []) or ner_data.get("extracted_terms", []) or []
                else:
                    extracted_terms = []
                auto_terms: dict[str, dict[str, str]] = {}
                for item in extracted_terms:
                    term = item.get("term")
                    persian = item.get("suggested_persian")
                    if term and persian:
                        category = str(item.get("category", "term"))
                        auto_terms[term] = {
                            "target": persian,
                            "context": item.get("context", ""),
                            "domain": item.get("domain", "") or self.config.translation.domain,
                            "sense": item.get("sense", ""),
                            "author": item.get("author", ""),
                            "category": category,
                        }
                        memory_manager.proper_nouns.add_noun(
                            term,
                            persian,
                            category=category,
                            provenance="auto_extraction",
                        )
                glossary_manager.merge_auto_extracted(auto_terms)
                self.db.save_job_artifact(
                    job_id,
                    "auto_extracted_terms",
                    {"terms": auto_terms, "count": len(auto_terms)},
                )
                self.db.log_chunk_event(
                    job_id, 0, "auto_extraction_completed", {
                        "model_candidates": len(extracted_terms),
                        "accepted_terms": len(auto_terms),
                        "terms": [
                            {
                                "source": source,
                                "target": value["target"],
                                "domain": value["domain"],
                            }
                            for source, value in list(auto_terms.items())[:50]
                        ],
                    }
                )
            except Exception as e:
                logger.warning("Automatic term extraction failed: %s", e)
                self.db.log_chunk_event(
                    job_id, 0, "auto_extraction_failed", {
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
        elif not is_resume:
            self.db.log_chunk_event(
                job_id, 0, "auto_extraction_skipped", {"enabled": False}
            )

        # Curated glossary terms opt into English originals only explicitly.
        for entry in glossary_manager.entries:
            if bool(getattr(entry, "include_original", False)):
                memory_manager.proper_nouns.add_noun(
                    entry.source,
                    entry.target,
                    category="approved_term",
                    provenance="curated_glossary",
                )

        research_artifact = self._prepare_book_research(
            job_id,
            research_document,
            progress_callback,
        )
        if research_artifact and research_artifact.get("status") in {
            "completed", "completed_without_suggestions", "degraded", "partial"
        }:
            memory_manager.book_context = _research_context_for_memory(
                research_artifact
            )

        # 6. Translate Chunks (Sequential or Concurrent)
        translations: dict[int, str] = {}
        
        # Load existing translations if resuming
        if is_resume:
            for c_record in self.db.get_chunks(job_id):
                if c_record["status"] in (ChunkStatus.COMPLETED, ChunkStatus.NEEDS_REVIEW):
                    translations[c_record["chunk_index"]] = c_record["translation"]

        # Run translation loop
        consecutive_errors = 0
        max_errors = self.config.retry.max_consecutive_errors

        web_searcher = WebContextSearcher(self.config, self.llm_client)
        if is_resume:
            saved_search_state = self.db.get_job_artifact(
                job_id, "web_search_state"
            )
            if saved_search_state:
                web_searcher.import_state(saved_search_state)
        compliance_checker = GlossaryComplianceChecker()
        # Critique and back-translation QA run on the independent judge model
        # (critic_client); translation and refinement stay on the translator.
        critique_tool = TranslationCritique(
            llm_client=self.critic_client,
            max_parse_retries=self.config.translation.qa_json_retries,
        )
        refiner_tool = TranslationRefiner(
            llm_client=self.llm_client,
            max_iterations=self.config.translation.max_refine_iterations,
            max_parse_retries=self.config.translation.qa_json_retries,
        )
        back_translator = BackTranslator(llm_client=self.critic_client, sample_pct=self.config.translation.back_translation_sample_pct)

        try:
            # Separate execution paths based on workers
            workers = self.config.translation.parallel_workers
            if (
                self.config.translation.stop_after_chapter > 0
                or self.config.translation.pause_after_each_chapter
            ):
                # Chapter checkpoints require deterministic source order. The
                # translation/QA stages themselves remain unchanged.
                workers = 1
            if workers > 1:
                # Concurrent Pass (Fast / Quality modes only)
                logger.info("Running parallel translation with %d workers", workers)
                lock = threading.Lock()

                def process_chunk_parallel(idx: int) -> tuple[int, str]:
                    chunk = chunks[idx]
                    translation = self._translate_single_chunk(
                        idx=idx,
                        chunk=chunk,
                        memory_manager=memory_manager,
                        web_searcher=web_searcher,
                        glossary_manager=glossary_manager,
                        compliance_checker=compliance_checker,
                        critique_tool=critique_tool,
                        refiner_tool=refiner_tool,
                        back_translator=back_translator,
                        translations=translations,
                        job_id=job_id,
                        lock=lock,
                        adjacent_source_context=_adjacent_source_context(chunks, idx),
                    )

                    # Update shared memory and database safely under lock
                    with lock:
                        translations[idx] = translation
                        memory_admission = _chunk_memory_admission(
                            self.db, job_id, idx
                        )
                        memory_policy = memory_manager.update_after_translation(
                            chunk,
                            translation,
                            quality_approved=memory_admission["quality_approved"],
                            style_approved=_chunk_style_approved(
                                self.db, job_id, idx
                            ),
                            long_term_reliable=memory_admission["long_term_reliable"],
                            short_term_trust=memory_admission["short_term_trust"],
                            reliability_reasons=memory_admission["reliability_reasons"],
                        )
                        self.db.log_chunk_event(
                            job_id, idx, "memory_update_policy", memory_policy
                        )
                        
                        if self.config.memory.enable_4layer:
                            # Seen-state is grounded in the source occurrence and is
                            # safe even when the target remains advisory.
                            memory_manager.proper_nouns.mark_introduced_from_translation(
                                chunk.text, translation
                            )
                            if _chunk_needs_review(self.db, job_id, idx):
                                self.db.log_chunk_event(
                                    job_id,
                                    idx,
                                    "proper_noun_extraction",
                                    {
                                        "status": "deferred_untrusted_chunk",
                                        "reason": (
                                            "The translation remains available as "
                                            "advisory short-term context, but cannot "
                                            "teach durable noun mappings until review."
                                        ),
                                    },
                                )
                            elif _is_front_matter(chunk):
                                # 10.2/11.6: the copyright page taught 44 bogus
                                # mappings, which also authorised inline English
                                # glossing of street names and printers.
                                self.db.log_chunk_event(
                                    job_id,
                                    idx,
                                    "proper_noun_extraction",
                                    {
                                        "status": "skipped_front_matter",
                                        "reason": (
                                            "Front matter yields publisher and "
                                            "address noise, not book terminology."
                                        ),
                                        "chapter_title": (
                                            chunk.chapter_title or ""
                                        ),
                                    },
                                )
                            else:
                                try:
                                    noun_report = self._run_async(
                                        memory_manager.update_proper_nouns(
                                            self.llm_client, chunk.text, translation
                                        )
                                    )
                                    self.db.log_chunk_event(
                                        job_id, idx, "proper_noun_extraction",
                                        noun_report,
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "Incremental proper noun extraction failed: %s", e
                                    )

                            is_chapter_end = False
                            if idx == total_chunks - 1:
                                is_chapter_end = True
                            else:
                                next_chunk = chunks[idx + 1]
                                # Titles are not unique: edited volumes repeat
                                # "Introduction", "Conclusion", "Notes". Two such
                                # chapters merged into one summary and this
                                # end-of-chapter test misfired. chapter_position
                                # is set by both chunkers and IS unique.
                                if self._chunk_chapter_position(
                                    next_chunk
                                ) != self._chunk_chapter_position(chunk):
                                    is_chapter_end = True

                            if is_chapter_end:
                                chap_source = []
                                chap_trans = []
                                for i in range(idx + 1):
                                    c = chunks[i]
                                    if self._chunk_chapter_position(
                                        c
                                    ) == self._chunk_chapter_position(chunk):
                                        chap_source.append(c.text)
                                        chap_trans.append(translations.get(i, ""))
                                
                                new_content = "\n\n".join(chap_source)
                                chap_translation = "\n\n".join(chap_trans)
                                try:
                                    summary_admission = _chapter_summary_memory_admission(
                                        self.db, job_id, chunks, idx
                                    )
                                    summary_report = self._run_async(memory_manager.update_bilingual_summary(
                                        self.llm_client,
                                        new_content=new_content,
                                        translation=chap_translation,
                                        input_trust=summary_admission["input_trust"],
                                        trust_reasons=summary_admission["trust_reasons"],
                                    ))
                                    summary_report.update({
                                        "contributing_chunks": summary_admission[
                                            "contributing_chunks"
                                        ],
                                    })
                                    self.db.log_chunk_event(
                                        job_id,
                                        idx,
                                        "bilingual_summary_memory_policy",
                                        summary_report,
                                    )
                                    if summary_report.get("replacement_count"):
                                        self.db.log_chunk_event(
                                            job_id,
                                            idx,
                                            "bilingual_summary_reconciled",
                                            summary_report,
                                        )
                                except Exception as e:
                                    logger.warning("Bilingual summary update failed: %s", e)

                        _ensure_chunk_review_reason(self.db, job_id, idx)
                        final_status = (
                            ChunkStatus.NEEDS_REVIEW
                            if _chunk_needs_review(self.db, job_id, idx)
                            else ChunkStatus.COMPLETED
                        )
                        # One transaction: a crash between these three
                        # writes used to leave a COMPLETED chunk whose memory
                        # contribution was missing on resume.
                        self.db.commit_chunk_checkpoint(
                            job_id,
                            idx,
                            final_status,
                            translation,
                            memory_manager.to_dict(),
                            search_state=web_searcher.export_state(),
                        )
                    
                    return idx, translation

                # Submit remaining chunks
                pending_indices = [i for i in range(total_chunks) if i not in translations]
                
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    future_to_idx = {
                        executor.submit(process_chunk_parallel, idx): idx for idx in pending_indices
                    }

                    completed_count = len(translations)
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            _, trans = future.result()
                            translations[idx] = trans
                            completed_count += 1
                            consecutive_errors = 0  # reset on success
                            
                            if progress_callback:
                                pct = 0.10 + (completed_count / total_chunks) * 0.80
                                progress_callback("Translation", pct, f"Translated chunk {completed_count}/{total_chunks}")
                        except Exception as e:
                            # Propagate cooperative pause exceptions out of parallel workers immediately
                            if isinstance(e, PipelinePausedException) or (hasattr(e, "__cause__") and isinstance(e.__cause__, PipelinePausedException)):
                                executor.shutdown(wait=False, cancel_futures=True)
                                raise PipelinePausedException("Job paused cooperatively")
                            
                            logger.error("Failed to translate chunk %d: %s", idx, e)
                            self.db.log_event(job_id, "ERROR", f"Chunk {idx} failed: {type(e).__name__}: {e}")
                            self.db.update_chunk(job_id, idx, ChunkStatus.ERROR)
                            consecutive_errors += 1
                            
                            if consecutive_errors >= max_errors:
                                # Pause and trigger error termination
                                self.db.update_job_status(job_id, JobStatus.PAUSED_ERROR, str(e))
                                self._send_webhook(
                                    "tarjomeh.job.paused_error",
                                    f"Pipeline paused after {consecutive_errors} consecutive failures: {e}",
                                    JobStatus.PAUSED_ERROR,
                                )
                                # Cancel remaining tasks
                                executor.shutdown(wait=False, cancel_futures=True)
                                raise RuntimeError(
                                    f"Pipeline terminated due to {consecutive_errors} consecutive failures. "
                                    f"Last error: {e}"
                                ) from e

            else:
                # Sequential Pass (Academic mode default)
                for idx in range(total_chunks):
                    if idx in translations:
                        continue

                    chunk = chunks[idx]
                    if progress_callback:
                        pct = 0.10 + (idx / total_chunks) * 0.80
                        chapter_position = self._chunk_chapter_position(chunk)
                        chapter_label = chunk.chapter_title or f"Chapter {chapter_position}"
                        progress_callback(
                            "Translation",
                            pct,
                            f"Chapter {chapter_position}: {chapter_label} - "
                            f"translating chunk {idx + 1}/{total_chunks}...",
                        )

                    try:
                        # Check pause and translate
                        translation = self._translate_single_chunk(
                            idx=idx,
                            chunk=chunk,
                            memory_manager=memory_manager,
                            web_searcher=web_searcher,
                            glossary_manager=glossary_manager,
                            compliance_checker=compliance_checker,
                            critique_tool=critique_tool,
                            refiner_tool=refiner_tool,
                            back_translator=back_translator,
                            translations=translations,
                            job_id=job_id,
                            adjacent_source_context=_adjacent_source_context(chunks, idx),
                        )

                        # Successful translation updates
                        translations[idx] = translation
                        consecutive_errors = 0

                        memory_admission = _chunk_memory_admission(
                            self.db, job_id, idx
                        )
                        memory_policy = memory_manager.update_after_translation(
                            chunk,
                            translation,
                            quality_approved=memory_admission["quality_approved"],
                            style_approved=_chunk_style_approved(
                                self.db, job_id, idx
                            ),
                            long_term_reliable=memory_admission["long_term_reliable"],
                            short_term_trust=memory_admission["short_term_trust"],
                            reliability_reasons=memory_admission["reliability_reasons"],
                        )
                        self.db.log_chunk_event(
                            job_id, idx, "memory_update_policy", memory_policy
                        )

                        if self.config.memory.enable_4layer:
                            # Seen-state is grounded in the source occurrence and is
                            # safe even when the target remains advisory.
                            memory_manager.proper_nouns.mark_introduced_from_translation(
                                chunk.text, translation
                            )
                            if _chunk_needs_review(self.db, job_id, idx):
                                self.db.log_chunk_event(
                                    job_id,
                                    idx,
                                    "proper_noun_extraction",
                                    {
                                        "status": "deferred_untrusted_chunk",
                                        "reason": (
                                            "The translation remains available as "
                                            "advisory short-term context, but cannot "
                                            "teach durable noun mappings until review."
                                        ),
                                    },
                                )
                            elif _is_front_matter(chunk):
                                # 10.2/11.6: the copyright page taught 44 bogus
                                # mappings, which also authorised inline English
                                # glossing of street names and printers.
                                self.db.log_chunk_event(
                                    job_id,
                                    idx,
                                    "proper_noun_extraction",
                                    {
                                        "status": "skipped_front_matter",
                                        "reason": (
                                            "Front matter yields publisher and "
                                            "address noise, not book terminology."
                                        ),
                                        "chapter_title": (
                                            chunk.chapter_title or ""
                                        ),
                                    },
                                )
                            else:
                                try:
                                    noun_report = self._run_async(
                                        memory_manager.update_proper_nouns(
                                            self.llm_client, chunk.text, translation
                                        )
                                    )
                                    self.db.log_chunk_event(
                                        job_id, idx, "proper_noun_extraction",
                                        noun_report,
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "Incremental proper noun extraction failed: %s", e
                                    )

                            is_chapter_end = False
                            if idx == total_chunks - 1:
                                is_chapter_end = True
                            else:
                                next_chunk = chunks[idx + 1]
                                # Titles are not unique: edited volumes repeat
                                # "Introduction", "Conclusion", "Notes". Two such
                                # chapters merged into one summary and this
                                # end-of-chapter test misfired. chapter_position
                                # is set by both chunkers and IS unique.
                                if self._chunk_chapter_position(
                                    next_chunk
                                ) != self._chunk_chapter_position(chunk):
                                    is_chapter_end = True

                            if is_chapter_end:
                                chap_source = []
                                chap_trans = []
                                for i in range(idx + 1):
                                    c = chunks[i]
                                    if self._chunk_chapter_position(
                                        c
                                    ) == self._chunk_chapter_position(chunk):
                                        chap_source.append(c.text)
                                        chap_trans.append(translations.get(i, ""))
                                
                                new_content = "\n\n".join(chap_source)
                                chap_translation = "\n\n".join(chap_trans)
                                try:
                                    summary_admission = _chapter_summary_memory_admission(
                                        self.db, job_id, chunks, idx
                                    )
                                    summary_report = self._run_async(memory_manager.update_bilingual_summary(
                                        self.llm_client,
                                        new_content=new_content,
                                        translation=chap_translation,
                                        input_trust=summary_admission["input_trust"],
                                        trust_reasons=summary_admission["trust_reasons"],
                                    ))
                                    summary_report.update({
                                        "contributing_chunks": summary_admission[
                                            "contributing_chunks"
                                        ],
                                    })
                                    self.db.log_chunk_event(
                                        job_id,
                                        idx,
                                        "bilingual_summary_memory_policy",
                                        summary_report,
                                    )
                                    if summary_report.get("replacement_count"):
                                        self.db.log_chunk_event(
                                            job_id,
                                            idx,
                                            "bilingual_summary_reconciled",
                                            summary_report,
                                        )
                                except Exception as e:
                                    logger.warning("Bilingual summary update failed: %s", e)

                        _ensure_chunk_review_reason(self.db, job_id, idx)
                        final_status = (
                            ChunkStatus.NEEDS_REVIEW
                            if _chunk_needs_review(self.db, job_id, idx)
                            else ChunkStatus.COMPLETED
                        )
                        # One transaction: a crash between these three
                        # writes used to leave a COMPLETED chunk whose memory
                        # contribution was missing on resume.
                        self.db.commit_chunk_checkpoint(
                            job_id,
                            idx,
                            final_status,
                            translation,
                            memory_manager.to_dict(),
                            search_state=web_searcher.export_state(),
                        )

                        if self._claim_chapter_checkpoint(job_id, chunks, idx):
                            raise ChapterCheckpointReached(
                                self._chunk_chapter_position(chunk),
                                chunk.chapter_title,
                            )

                    except PipelinePausedException as e:
                        raise e
                    except Exception as e:
                        logger.error("Failed to translate chunk %d: %s", idx, e)
                        self.db.log_event(job_id, "ERROR", f"Chunk {idx} failed: {type(e).__name__}: {e}")
                        self.db.update_chunk(job_id, idx, ChunkStatus.ERROR)
                        consecutive_errors += 1

                        pause_for_order = bool(
                            getattr(
                                self.config.retry,
                                "pause_on_sequential_error",
                                True,
                            )
                        )
                        if pause_for_order or consecutive_errors >= max_errors:
                            self.db.update_job_status(job_id, JobStatus.PAUSED_ERROR, str(e))
                            self._send_webhook(
                                "tarjomeh.job.paused_error",
                                (
                                    f"Pipeline paused at chunk {idx} to preserve "
                                    f"sequential memory order: {e}"
                                    if pause_for_order else
                                    f"Pipeline paused after {consecutive_errors} "
                                    f"consecutive failures: {e}"
                                ),
                                JobStatus.PAUSED_ERROR,
                            )
                            raise RuntimeError(
                                (
                                    f"Pipeline paused at chunk {idx} after a genuine "
                                    "chunk-stage failure so later chunks do not advance "
                                    f"without its memory. Last error: {e}"
                                    if pause_for_order else
                                    f"Pipeline terminated due to {consecutive_errors} "
                                    f"consecutive failures. Last error: {e}"
                                )
                            ) from e
        except ChapterCheckpointReached as checkpoint:
            self.db.update_job_status(job_id, JobStatus.PAUSED)
            selected_positions = list(
                self.config.translation.chapter_selection
            )
            preview_positions = (
                [
                    position for position in selected_positions
                    if position <= checkpoint.chapter_position
                ]
                if selected_positions
                else list(range(1, checkpoint.chapter_position + 1))
            )
            partial_path = self.export_completed_job(
                job_id,
                output_path,
                chapter_positions=preview_positions,
            )
            message = (
                f"Review checkpoint after chapter {checkpoint.chapter_position}: "
                f"{checkpoint.chapter_title}. Partial output is ready."
            )
            self.db.log_event(job_id, "INFO", message)
            if progress_callback:
                progress_callback(
                    "Paused", len(translations) / total_chunks, message
                )
            duration = time.monotonic() - t0
            return PipelineResult(
                partial_path,
                len(translations),
                duration,
                warnings=["Chapter review checkpoint"],
            )
        except PipelinePausedException:
            logger.info("Pipeline paused cooperatively for job %s", job_id)
            if progress_callback:
                progress_callback("Paused", len(translations) / total_chunks, "Job paused cooperatively.")
            duration = time.monotonic() - t0
            return PipelineResult(output_path, len(translations), duration, warnings=["Job paused"])

        missing_indices = [
            idx for idx in range(total_chunks)
            if not translations.get(idx, "").strip()
        ]
        if missing_indices:
            preview = ", ".join(str(idx) for idx in missing_indices[:10])
            if len(missing_indices) > 10:
                preview += ", ..."
            message = (
                f"Translation incomplete: {len(missing_indices)}/{total_chunks} chunk(s) "
                f"have no completed translation. Missing chunk indices: {preview}. "
                "Output was not exported; resume the job after fixing the failed chunks."
            )
            logger.error(message)
            self.db.log_event(job_id, "ERROR", message)
            self.db.update_job_status(job_id, JobStatus.PAUSED_ERROR, message)
            self._send_webhook(
                "tarjomeh.job.paused_error",
                message,
                JobStatus.PAUSED_ERROR,
            )
            if progress_callback:
                progress_callback("Paused", len(translations) / total_chunks, message)
            raise RuntimeError(message)

        # 7. Assemble Document
        if progress_callback:
            progress_callback("Assembly", 0.90, "Reassembling translated paragraphs...")

        original_paragraphs = document.all_paragraphs
        # Pre-populate translated paragraphs list
        translated_paragraphs: list[TranslatedParagraph | None] = [None] * len(original_paragraphs)

        # Track sequential index fallback
        fallback_idx = 0

        for idx in range(total_chunks):
            chunk = chunks[idx]
            chunk_translation = translations.get(idx, "")
            tgt_paras = [p.strip() for p in chunk_translation.split("\n\n") if p.strip()]
            # chunk.metadata is dict[str, object], so narrow once here rather
            # than leaving every downstream len()/index/call site untyped.
            _raw_indices = chunk.metadata.get("paragraph_indices")
            para_indices: list[int] = (
                [int(value) for value in _raw_indices]
                if isinstance(_raw_indices, list) else []
            )

            if para_indices:
                try:
                    aligned = _align_chunk_translation(
                        original_paragraphs=original_paragraphs,
                        para_indices=para_indices,
                        tgt_paras=tgt_paras,
                        chunk_translation=chunk_translation,
                        strict_paragraph_identity=_used_paragraph_protocol(chunk),
                    )
                except ParagraphIdentityError as exc:
                    # Never discard fully-paid translations over a formatting
                    # mismatch. Reconstruct the alignment, flag the chunk for
                    # human review, and finish the export.
                    logger.warning(
                        "Chunk %d: %s Falling back to proportional alignment.",
                        idx, exc,
                    )
                    self.db.log_chunk_event(
                        job_id, idx, "paragraph_identity_degraded", {
                            "expected": len(para_indices),
                            "received": len(tgt_paras),
                        },
                    )
                    self.db.update_chunk(
                        job_id, idx, ChunkStatus.NEEDS_REVIEW, chunk_translation
                    )
                    self.warnings.append(
                        f"Chunk {idx}: paragraph alignment was reconstructed "
                        "from an off-count translation; review recommended."
                    )
                    aligned = _align_chunk_translation(
                        original_paragraphs=original_paragraphs,
                        para_indices=para_indices,
                        tgt_paras=tgt_paras,
                        chunk_translation=chunk_translation,
                        strict_paragraph_identity=False,
                    )
                if len(para_indices) != len(tgt_paras):
                    logger.warning(
                        "Chunk %d: translation has %d paragraph(s) but source has %d; "
                        "redistributed proportionally across source paragraphs.",
                        idx, len(tgt_paras), len(para_indices),
                    )

                for pid, t in aligned:
                    if pid < len(original_paragraphs):
                        orig_para = original_paragraphs[pid]
                        existing = translated_paragraphs[pid]
                        if existing is None:
                            translated_paragraphs[pid] = TranslatedParagraph(
                                index=pid,
                                source_text=orig_para.text,
                                translated_text=t,
                                heading_level=orig_para.heading_level,
                                metadata=orig_para.metadata,
                            )
                        elif t:
                            # Same paragraph index seen again (e.g. FixedChunker
                            # split one long paragraph into several sub-chunks):
                            # APPEND rather than overwrite so no sub-chunk
                            # translation is lost.
                            existing.translated_text = (
                                f"{existing.translated_text.rstrip()} {t}".strip()
                                if existing.translated_text.strip()
                                else t
                            )
            else:
                # Fallback to sequential mapping
                src_paras = [p.strip() for p in chunk.text.split("\n\n") if p.strip()]
                if len(src_paras) != len(tgt_paras):
                    aligned_pairs = []
                    for i in range(max(len(src_paras), len(tgt_paras))):
                        s = src_paras[i] if i < len(src_paras) else ""
                        t = tgt_paras[i] if i < len(tgt_paras) else ""
                        if s or t:
                            aligned_pairs.append((s, t))
                else:
                    aligned_pairs = list(zip(src_paras, tgt_paras))

                for s, t in aligned_pairs:
                    if fallback_idx < len(original_paragraphs):
                        orig_para = original_paragraphs[fallback_idx]
                        translated_paragraphs[fallback_idx] = TranslatedParagraph(
                            index=fallback_idx,
                            source_text=s,
                            translated_text=t,
                            heading_level=orig_para.heading_level,
                            metadata=orig_para.metadata,
                        )
                        fallback_idx += 1

        # Fill any missing/skipped paragraphs with empty translations
        final_translated_paragraphs: list[TranslatedParagraph] = []
        for pid in range(len(original_paragraphs)):
            pt = translated_paragraphs[pid]
            if pt is None:
                orig_para = original_paragraphs[pid]
                final_translated_paragraphs.append(
                    TranslatedParagraph(
                        index=pid,
                        source_text=orig_para.text,
                        translated_text="",
                        heading_level=orig_para.heading_level,
                        metadata=orig_para.metadata,
                    )
                )
            else:
                final_translated_paragraphs.append(pt)

        trans_doc = TranslatedDocument(
            title=document.title,
            author=document.author,
            paragraphs=final_translated_paragraphs,
            metadata={
                **document.metadata,
                "chapter_page_breaks": bool(
                    self.config.output.chapter_page_breaks
                ),
            },
        )

        # 8. Persian Typography Post-Processing
        if progress_callback:
            progress_callback("Typography", 0.95, "Applying Persian typography rules...")

        typographer = PersianTypographer(self.config.to_dict().get("persian"))
        final_orthography_edits: list[dict[str, Any]] = []
        for p in trans_doc.paragraphs:
            p.translated_text, edits = typographer.process_with_report(
                p.translated_text
            )
            for edit in edits:
                final_orthography_edits.append({
                    "paragraph_index": p.index,
                    **edit,
                })
        remaining_orthography_issues = sum(
            orthography_issue_count(p.translated_text)
            for p in trans_doc.paragraphs
        )
        self.db.save_job_artifact(job_id, "persian_orthography_audit", {
            "edit_count": sum(
                int(edit.get("count", 0)) for edit in final_orthography_edits
            ),
            "remaining_issue_count": remaining_orthography_issues,
            "coverage": "deterministic_patterns_only",
            "ambiguous_forms_auto_classified": False,
            "edits": final_orthography_edits,
        })
        if remaining_orthography_issues:
            warning = (
                "Persian orthography audit found "
                f"{remaining_orthography_issues} unresolved deterministic issue(s)."
            )
            self.warnings.append(warning)
            self.db.log_event(job_id, "WARNING", warning)

        protocol_audit = sanitize_document_protocol_artifacts(trans_doc)
        self.db.save_job_artifact(
            job_id, "protocol_integrity_audit", protocol_audit
        )
        if protocol_audit["safe_edit_count"]:
            self.db.log_event(
                job_id,
                "WARNING",
                "Removed recognized leading model-protocol wrapper(s) from "
                f"{protocol_audit['safe_edit_count']} paragraph(s).",
            )
        if protocol_audit["remaining_artifact_count"]:
            raise RuntimeError(
                "Export blocked: unresolved model-protocol artifacts remain "
                "in translated text."
            )

        requested_note_mode = self.config.output.term_notes
        note_mode = effective_term_notes_mode(
            requested_note_mode, self.config.output.format
        )
        note_formats = {"docx", "epub", "markdown"}
        noun_state = memory_manager.proper_nouns.serialize()
        proper_nouns = memory_manager.proper_nouns.inline_eligible_nouns()
        noun_aliases = memory_manager.proper_nouns.inline_eligible_aliases()
        noun_categories = dict(noun_state.get("categories", {}))
        if note_mode in {"inline", "both"}:
            initial_anchor_audit = ensure_inline_proper_noun_originals(
                trans_doc,
                proper_nouns,
                typographer,
                noun_categories,
                aliases=noun_aliases,
                return_report=True,
            )
        citation_audit = normalize_adjacent_original_citations(
            trans_doc, proper_nouns
        )
        fragment_audit = reconcile_redundant_original_fragments(
            trans_doc, proper_nouns
        )
        self.db.save_job_artifact(
            job_id, "original_fragment_reconciliation", fragment_audit
        )
        self.db.save_job_artifact(
            job_id, "citation_format_audit", citation_audit
        )
        original_audit = audit_inline_english_originals(
            trans_doc,
            proper_nouns if note_mode in {"inline", "both"} else {},
        )
        self.db.save_job_artifact(
            job_id, "english_original_audit", original_audit
        )
        self.db.log_event(
            job_id,
            "INFO",
            "English-original audit: "
            f"unauthorized={original_audit['removed_unauthorized_count']}, "
            f"duplicates={original_audit['removed_duplicate_count']}, "
            f"citations_preserved={original_audit['preserved_citation_count']}.",
        )
        if note_mode in {"inline", "both"}:
            final_anchor_audit = ensure_inline_proper_noun_originals(
                trans_doc,
                proper_nouns,
                typographer,
                noun_categories,
                aliases=noun_aliases,
                return_report=True,
            )
            anchor_audit = {
                **final_anchor_audit,
                "initial_inserted_count": int(
                    initial_anchor_audit.get("inserted_count", 0)
                ),
                "initial_missing_target_count": int(
                    initial_anchor_audit.get("missing_target_count", 0)
                ),
                "final_reconciliation": True,
            }
            self.db.save_job_artifact(
                job_id, "english_original_anchor_audit", anchor_audit
            )
            restored = int(initial_anchor_audit.get("inserted_count", 0))
            restored += int(final_anchor_audit.get("inserted_count", 0))
            paired_repaired = int(
                initial_anchor_audit.get("paired_repair_count", 0)
            ) + int(final_anchor_audit.get("paired_repair_count", 0))
            if restored or paired_repaired or anchor_audit.get("repositioned_count"):
                self.db.log_event(
                    job_id, "INFO",
                    "Anchored first-occurrence English originals: "
                    f"inserted={restored}, "
                    f"paired_repaired={paired_repaired}, "
                    f"repositioned={anchor_audit.get('repositioned_count', 0)}, "
                    f"ambiguous={anchor_audit.get('ambiguous_count', 0)}.",
                )
            final_citation_audit = normalize_adjacent_original_citations(
                trans_doc, proper_nouns
            )
            final_fragment_audit = reconcile_redundant_original_fragments(
                trans_doc, proper_nouns
            )
            fragment_audit = {
                "removed_count": int(fragment_audit.get("removed_count", 0))
                + int(final_fragment_audit.get("removed_count", 0)),
                "changes": list(fragment_audit.get("changes", []) or [])
                + list(final_fragment_audit.get("changes", []) or []),
                "final_anchor_reconciliation": True,
            }
            self.db.save_job_artifact(
                job_id, "original_fragment_reconciliation", fragment_audit
            )
            citation_audit = {
                "normalized_count": int(citation_audit.get("normalized_count", 0))
                + int(final_citation_audit.get("normalized_count", 0)),
                "changes": list(citation_audit.get("changes", []) or [])
                + list(final_citation_audit.get("changes", []) or []),
                "final_anchor_reconciliation": True,
            }
            self.db.save_job_artifact(
                job_id, "citation_format_audit", citation_audit
            )
        if note_mode != "inline" and self.config.output.format in note_formats:
            notes = apply_term_notes(
                trans_doc,
                glossary_manager,
                proper_nouns,
                typographer,
                domain=self.config.translation.domain,
                mode=note_mode,
                aliases=noun_aliases,
            )
            self.db.save_job_artifact(job_id, "term_notes", {
                "mode": note_mode,
                "notes": notes,
            })
            self.db.log_event(
                job_id,
                "INFO",
                f"Generated {len(notes)} first-occurrence term note(s).",
            )
        if requested_note_mode != note_mode:
            warning = (
                f"Term notes are not rendered for {self.config.output.format}; "
                "English originals were preserved inline instead."
            )
            self.warnings.append(warning)
            self.db.log_event(job_id, "WARNING", warning)

        identifier_audit = restore_document_source_identifiers(trans_doc)
        self.db.save_job_artifact(
            job_id, "final_identifier_reconciliation", identifier_audit
        )
        if identifier_audit["repair_count"]:
            self.db.log_event(
                job_id, "INFO",
                "Restored exact source identifiers after final typography: "
                f"{identifier_audit['repair_count']} repair(s).",
            )

        final_text_audit = audit_document_final_text(
            trans_doc,
            allowed_originals=tuple(proper_nouns),
        )
        self.db.save_job_artifact(
            job_id, "final_text_quality_audit", final_text_audit
        )
        if final_text_audit["review_required"]:
            self.db.log_event(
                job_id,
                "WARNING",
                "Final text audit found review evidence: "
                f"language_findings={final_text_audit['language_finding_count']}, "
                "unresolved_identifiers="
                f"{final_text_audit['unresolved_identifier_count']}.",
            )

        # 9. Export
        if progress_callback:
            progress_callback("Export", 0.98, f"Exporting to {self.config.output.format.upper()}...")

        exporter_cls = get_exporter(self.config.output.format)
        exporter = exporter_cls(self.config.to_dict().get(self.config.output.format))
        exporter.export(
            document=trans_doc,
            output_path=output_path,
            bilingual_mode=self.config.output.bilingual_mode,
        )

        # Update Job Status in DB
        duration = time.monotonic() - t0
        self.db.update_job_status(job_id, JobStatus.COMPLETED, output_path=output_path)

        # Send Webhook complete notification
        self._send_webhook(
            "tarjomeh.job.completed",
            f"Translation completed in {duration:.1f}s. Output exported to {output_path}",
            JobStatus.COMPLETED,
        )

        if progress_callback:
            progress_callback("Complete", 1.0, f"Finished! Output at {output_path.name}")

        return PipelineResult(output_path, total_chunks, duration, warnings=self.warnings)

    def export_completed_job(
        self,
        job_id: str,
        output_path: Path | None = None,
        *,
        output_format: str | None = None,
        bilingual_mode: str | None = None,
        chapter_positions: list[int] | None = None,
    ) -> Path:
        """Re-export a completed/partially-reviewed job without LLM calls."""
        job = self.db.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found.")

        input_path = Path(job["input_path"])
        if output_format:
            self.config.output.format = output_format
        if bilingual_mode:
            self.config.output.bilingual_mode = bilingual_mode

        fmt = self.config.output.format.lower()
        if output_path is None:
            extension = _output_extension(fmt)
            output_path = input_path.parent / f"{input_path.stem}_reexported.{extension}"
        else:
            output_path = Path(output_path)

        structure_artifact = self.db.get_job_artifact(
            job_id, "document_structure_version"
        ) or {}
        document, chunks = self._parse_and_chunk(
            input_path,
            chapter_positions=chapter_positions,
            structure_version=int(structure_artifact.get("version", 1)),
        )
        translations: dict[int, str] = {}
        for c_record in self.db.get_chunks(job_id):
            if c_record["status"] in (ChunkStatus.COMPLETED, ChunkStatus.NEEDS_REVIEW) and c_record.get("translation"):
                translations[int(c_record["chunk_index"])] = c_record["translation"]

        missing = [idx for idx in range(len(chunks)) if not translations.get(idx, "").strip()]
        if missing:
            raise RuntimeError(
                "Cannot re-export: missing completed translation for chunk(s) "
                + ", ".join(str(i) for i in missing[:20])
            )

        trans_doc = self._assemble_translated_document(document, chunks, translations)
        protocol_audit = sanitize_document_protocol_artifacts(trans_doc)
        self.db.save_job_artifact(
            job_id, "protocol_integrity_audit", protocol_audit
        )
        if protocol_audit["remaining_artifact_count"]:
            raise RuntimeError(
                "Re-export blocked: unresolved model-protocol artifacts remain "
                "in translated text."
            )
        note_mode = effective_term_notes_mode(self.config.output.term_notes, fmt)
        memory_state = self.db.get_memory_state(job_id) or {}
        noun_state = memory_state.get("proper_nouns", {})
        proper_nouns = _inline_eligible_nouns_from_state(noun_state)
        noun_categories = dict(noun_state.get("categories", {})) \
            if isinstance(noun_state, dict) else {}
        noun_aliases = _inline_aliases_from_state(noun_state)

        typographer = PersianTypographer(self.config.to_dict().get("persian"))
        if note_mode in {"inline", "both"}:
            initial_anchor_audit = ensure_inline_proper_noun_originals(
                trans_doc,
                dict(proper_nouns),
                typographer,
                noun_categories,
                aliases=noun_aliases,
                return_report=True,
            )
        citation_audit = normalize_adjacent_original_citations(
            trans_doc, dict(proper_nouns)
        )
        fragment_audit = reconcile_redundant_original_fragments(
            trans_doc, dict(proper_nouns)
        )
        self.db.save_job_artifact(
            job_id, "original_fragment_reconciliation", fragment_audit
        )
        self.db.save_job_artifact(
            job_id, "citation_format_audit", citation_audit
        )
        original_audit = audit_inline_english_originals(
            trans_doc,
            dict(proper_nouns) if note_mode in {"inline", "both"} else {},
        )
        self.db.save_job_artifact(
            job_id, "english_original_audit", original_audit
        )
        if note_mode in {"inline", "both"}:
            anchor_audit = ensure_inline_proper_noun_originals(
                trans_doc,
                dict(proper_nouns),
                typographer,
                noun_categories,
                aliases=noun_aliases,
                return_report=True,
            )
            anchor_audit.update({
                "initial_inserted_count": int(
                    initial_anchor_audit.get("inserted_count", 0)
                ),
                "initial_missing_target_count": int(
                    initial_anchor_audit.get("missing_target_count", 0)
                ),
                "final_reconciliation": True,
            })
            self.db.save_job_artifact(
                job_id, "english_original_anchor_audit", anchor_audit
            )
            final_citation_audit = normalize_adjacent_original_citations(
                trans_doc, dict(proper_nouns)
            )
            final_fragment_audit = reconcile_redundant_original_fragments(
                trans_doc, dict(proper_nouns)
            )
            fragment_audit = {
                "removed_count": int(fragment_audit.get("removed_count", 0))
                + int(final_fragment_audit.get("removed_count", 0)),
                "changes": list(fragment_audit.get("changes", []) or [])
                + list(final_fragment_audit.get("changes", []) or []),
                "final_anchor_reconciliation": True,
            }
            self.db.save_job_artifact(
                job_id, "original_fragment_reconciliation", fragment_audit
            )
            citation_audit = {
                "normalized_count": int(citation_audit.get("normalized_count", 0))
                + int(final_citation_audit.get("normalized_count", 0)),
                "changes": list(citation_audit.get("changes", []) or [])
                + list(final_citation_audit.get("changes", []) or []),
                "final_anchor_reconciliation": True,
            }
            self.db.save_job_artifact(
                job_id, "citation_format_audit", citation_audit
            )
        if note_mode != "inline" and fmt in {"docx", "epub", "markdown"}:
            glossary_manager = GlossaryManager()
            glossary_paths = []
            if self.config.glossary.path:
                glossary_paths.append(Path(self.config.glossary.path))
            glossary_paths.extend(
                Path(path) for path in (self.config.glossary.paths or [])
                if Path(path) not in glossary_paths
            )
            glossary_manager.load_many(glossary_paths, ignore_missing=True)

            note_artifact = self.db.get_job_artifact(job_id, "term_notes") or {}
            persisted_terms = {
                str(note.get("original", "")): str(
                    note.get("transliteration", "")
                )
                for note in note_artifact.get("notes", [])
                if isinstance(note, dict)
                and note.get("original")
                and note.get("transliteration")
            }
            apply_term_notes(
                trans_doc,
                glossary_manager,
                dict(proper_nouns),
                typographer,
                domain=self.config.translation.domain,
                mode=note_mode,
                extra_terms=persisted_terms,
                aliases=noun_aliases,
            )
        identifier_audit = restore_document_source_identifiers(trans_doc)
        self.db.save_job_artifact(
            job_id, "final_identifier_reconciliation", identifier_audit
        )
        final_text_audit = audit_document_final_text(
            trans_doc,
            allowed_originals=tuple(proper_nouns),
        )
        self.db.save_job_artifact(
            job_id, "final_text_quality_audit", final_text_audit
        )
        exporter_cls = get_exporter(fmt)
        exporter = exporter_cls(self.config.to_dict().get(fmt))
        exporter.export(
            document=trans_doc,
            output_path=output_path,
            bilingual_mode=self.config.output.bilingual_mode,
        )
        self.db.update_job_status(
            job_id,
            job.get("raw_status", job["status"]),
            output_path=output_path,
        )
        self.db.log_event(job_id, "INFO", f"Re-exported job to {output_path}")
        return output_path

    def retranslate_chunk(self, job_id: str, chunk_index: int) -> str:
        """Retranslate one chunk for editor review without touching others."""
        job = self.db.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found.")

        structure_artifact = self.db.get_job_artifact(
            job_id, "document_structure_version"
        ) or {}
        document, chunks = self._parse_and_chunk(
            Path(job["input_path"]),
            structure_version=int(structure_artifact.get("version", 1)),
        )
        # Same index-only assumption as resume, so the same guard applies.
        self._verify_resume_alignment(job_id, chunks)
        if chunk_index < 0 or chunk_index >= len(chunks):
            raise ValueError(f"Chunk {chunk_index} is out of range.")

        translations: dict[int, str] = {}
        for c_record in self.db.get_chunks(job_id):
            if c_record.get("translation"):
                translations[int(c_record["chunk_index"])] = c_record["translation"]

        glossary_manager = GlossaryManager()
        glossary_paths: list[Path] = []
        primary_glossary = getattr(self.config.glossary, "path", "")
        if primary_glossary:
            glossary_paths.append(Path(primary_glossary))
        for extra_path in getattr(self.config.glossary, "paths", []) or []:
            p = Path(extra_path)
            if p not in glossary_paths:
                glossary_paths.append(p)
        glossary_manager.load_many(glossary_paths, ignore_missing=True)

        memory_manager = MemoryManager(self.config)
        saved_mem = self.db.get_memory_state(job_id)
        if saved_mem:
            memory_manager.from_dict(saved_mem)
        for entry in glossary_manager.entries:
            if bool(getattr(entry, "include_original", False)):
                memory_manager.proper_nouns.add_noun(
                    entry.source,
                    entry.target,
                    category="approved_term",
                    provenance="curated_glossary",
                )
        research_artifact = self.db.get_job_artifact(job_id, "book_research")
        if research_artifact is not None:
            memory_manager.book_context = _research_context_for_memory(
                research_artifact
            )

        web_searcher = WebContextSearcher(self.config, self.llm_client)
        saved_search_state = self.db.get_job_artifact(
            job_id, "web_search_state"
        )
        if saved_search_state:
            web_searcher.import_state(saved_search_state)
        compliance_checker = GlossaryComplianceChecker()
        critique_tool = TranslationCritique(
            llm_client=self.critic_client,
            max_parse_retries=self.config.translation.qa_json_retries,
        )
        refiner_tool = TranslationRefiner(
            llm_client=self.llm_client,
            max_iterations=self.config.translation.max_refine_iterations,
            max_parse_retries=self.config.translation.qa_json_retries,
        )
        back_translator = BackTranslator(
            llm_client=self.critic_client,
            sample_pct=self.config.translation.back_translation_sample_pct,
        )

        previous_translation = translations.get(chunk_index, "")
        translation = self._translate_single_chunk(
            idx=chunk_index,
            chunk=chunks[chunk_index],
            memory_manager=memory_manager,
            web_searcher=web_searcher,
            glossary_manager=glossary_manager,
            compliance_checker=compliance_checker,
            critique_tool=critique_tool,
            refiner_tool=refiner_tool,
            back_translator=back_translator,
            translations=translations,
            job_id=job_id,
            adjacent_source_context=_adjacent_source_context(chunks, chunk_index),
        )
        if (
            getattr(self.config.translation, "enable_integrity_gate", True)
            and previous_translation.strip()
        ):
            manual_gate = PostEditIntegrityGate(
                min_retention_ratio=getattr(
                    self.config.translation, "integrity_min_retention_ratio", 0.65
                ),
                max_growth_ratio=getattr(
                    self.config.translation, "integrity_max_growth_ratio", 1.75
                ),
            )
            eligible_originals = {
                value.casefold()
                for value in memory_manager.proper_nouns.inline_eligible_nouns()
            }
            manual_allowed_originals = [
                value for value in protected_english_originals(
                    chunks[chunk_index].text, previous_translation
                )
                if value.casefold() in eligible_originals
            ]
            manual_result = manual_gate.evaluate(
                chunks[chunk_index].text,
                translation,
                previous=previous_translation,
                stage="manual_retranslation",
                protect_inline_english=effective_term_notes_mode(
                    self.config.output.term_notes,
                    self.config.output.format,
                ) in {"inline", "both"},
                allowed_inline_originals=manual_allowed_originals,
            )
            self.db.log_chunk_event(
                job_id, chunk_index, "integrity_check_completed",
                manual_result.to_dict(),
            )
            if not manual_result.accepted:
                self.db.log_chunk_event(
                    job_id, chunk_index, "integrity_edit_rejected",
                    manual_result.to_dict(),
                )
                translation = previous_translation
        _ensure_chunk_review_reason(self.db, job_id, chunk_index)
        final_status = (
            ChunkStatus.NEEDS_REVIEW
            if _chunk_needs_review(self.db, job_id, chunk_index)
            else ChunkStatus.COMPLETED
        )
        self.db.update_chunk(job_id, chunk_index, final_status, translation)
        memory_admission = _chunk_memory_admission(
            self.db, job_id, chunk_index
        )
        memory_policy = memory_manager.update_after_translation(
            chunks[chunk_index],
            translation,
            quality_approved=memory_admission["quality_approved"],
            style_approved=_chunk_style_approved(
                self.db, job_id, chunk_index
            ),
            long_term_reliable=memory_admission["long_term_reliable"],
            short_term_trust=memory_admission["short_term_trust"],
            reliability_reasons=memory_admission["reliability_reasons"],
        )
        self.db.log_chunk_event(
            job_id, chunk_index, "memory_update_policy", memory_policy
        )
        self.db.save_memory_state(job_id, memory_manager.to_dict())
        self.db.log_event(job_id, "INFO", f"Retranslated chunk {chunk_index}.")
        return translation

    @staticmethod
    def _chunk_chapter_position(chunk: Chunk) -> int:
        return int(chunk.metadata.get("chapter_position", 1))

    def _claim_chapter_checkpoint(
        self,
        job_id: str,
        chunks: list[Chunk],
        chunk_index: int,
    ) -> bool:
        """Record a configured checkpoint at a completed chapter boundary."""
        if chunk_index >= len(chunks) - 1:
            return False
        current = self._chunk_chapter_position(chunks[chunk_index])
        following = self._chunk_chapter_position(chunks[chunk_index + 1])
        if current == following:
            return False

        requested = (
            self.config.translation.pause_after_each_chapter
            or self.config.translation.stop_after_chapter == current
        )
        if not requested:
            return False

        artifact = self.db.get_job_artifact(job_id, "chapter_checkpoints") or {}
        reached = {
            int(value) for value in artifact.get("reached_positions", [])
        }
        if current in reached:
            return False
        reached.add(current)
        self.db.save_job_artifact(job_id, "chapter_checkpoints", {
            "reached_positions": sorted(reached),
            "latest_position": current,
            "latest_title": chunks[chunk_index].chapter_title,
        })
        return True

    def _parse_and_chunk(
        self,
        input_path: Path,
        chapter_positions: list[int] | None = None,
        structure_version: int = 4,
    ) -> tuple[Document, list[Chunk]]:
        parser = self._get_parser(input_path)
        if hasattr(parser, "structure_version"):
            parser.structure_version = max(1, int(structure_version))
        document = parser.parse(input_path)
        selection = (
            chapter_positions
            if chapter_positions is not None
            else self.config.translation.chapter_selection
        )
        document = apply_chapter_selection(document, selection)

        def token_counter(text: str) -> int:
            return self.llm_client.count_tokens(text)

        if self.config.chunking.strategy == "semantic":
            chunker = SemanticChunker(
                max_tokens=self.config.chunking.max_chunk_tokens,
                overlap_sentences=self.config.chunking.overlap_sentences,
                token_counter=token_counter,
            )
        else:
            chunker = FixedChunker(
                max_tokens=self.config.chunking.max_chunk_tokens,
                token_counter=token_counter,
            )
        return document, chunker.chunk(document)

    def _assemble_translated_document(
        self,
        document: Document,
        chunks: list[Chunk],
        translations: dict[int, str],
    ) -> TranslatedDocument:
        """Assemble translated chunks into document paragraphs."""
        original_paragraphs = document.all_paragraphs
        translated_paragraphs: list[TranslatedParagraph | None] = [None] * len(original_paragraphs)
        fallback_idx = 0

        for idx, chunk in enumerate(chunks):
            chunk_translation = translations.get(idx, "")
            tgt_paras = [p.strip() for p in chunk_translation.split("\n\n") if p.strip()]
            # chunk.metadata is dict[str, object], so narrow once here rather
            # than leaving every downstream len()/index/call site untyped.
            _raw_indices = chunk.metadata.get("paragraph_indices")
            para_indices: list[int] = (
                [int(value) for value in _raw_indices]
                if isinstance(_raw_indices, list) else []
            )

            if para_indices:
                try:
                    aligned = _align_chunk_translation(
                        original_paragraphs=original_paragraphs,
                        para_indices=para_indices,
                        tgt_paras=tgt_paras,
                        chunk_translation=chunk_translation,
                        strict_paragraph_identity=_used_paragraph_protocol(chunk),
                    )
                except ParagraphIdentityError as exc:
                    # No job_id in scope here, so this path only warns and
                    # reconstructs; the run() assembly above records the event.
                    logger.warning(
                        "Chunk %d: %s Falling back to proportional alignment.",
                        idx, exc,
                    )
                    aligned = _align_chunk_translation(
                        original_paragraphs=original_paragraphs,
                        para_indices=para_indices,
                        tgt_paras=tgt_paras,
                        chunk_translation=chunk_translation,
                        strict_paragraph_identity=False,
                    )
                for pid, t in aligned:
                    if pid < len(original_paragraphs):
                        orig_para = original_paragraphs[pid]
                        existing = translated_paragraphs[pid]
                        if existing is None:
                            translated_paragraphs[pid] = TranslatedParagraph(
                                index=pid,
                                source_text=orig_para.text,
                                translated_text=t,
                                heading_level=orig_para.heading_level,
                                metadata=orig_para.metadata,
                            )
                        elif t:
                            existing.translated_text = (
                                f"{existing.translated_text.rstrip()} {t}".strip()
                                if existing.translated_text.strip()
                                else t
                            )
            else:
                src_paras = [p.strip() for p in chunk.text.split("\n\n") if p.strip()]
                aligned_pairs = list(zip(src_paras, tgt_paras)) if len(src_paras) == len(tgt_paras) else [
                    (
                        src_paras[i] if i < len(src_paras) else "",
                        tgt_paras[i] if i < len(tgt_paras) else "",
                    )
                    for i in range(max(len(src_paras), len(tgt_paras)))
                ]
                for s, t in aligned_pairs:
                    if fallback_idx < len(original_paragraphs):
                        orig_para = original_paragraphs[fallback_idx]
                        translated_paragraphs[fallback_idx] = TranslatedParagraph(
                            index=fallback_idx,
                            source_text=s,
                            translated_text=t,
                            heading_level=orig_para.heading_level,
                            metadata=orig_para.metadata,
                        )
                        fallback_idx += 1

        final_translated_paragraphs: list[TranslatedParagraph] = []
        for pid, orig_para in enumerate(original_paragraphs):
            pt = translated_paragraphs[pid]
            final_translated_paragraphs.append(
                pt if pt is not None else TranslatedParagraph(
                    index=pid,
                    source_text=orig_para.text,
                    translated_text="",
                    heading_level=orig_para.heading_level,
                    metadata=orig_para.metadata,
                )
            )

        typographer = PersianTypographer(self.config.to_dict().get("persian"))
        for p in final_translated_paragraphs:
            p.translated_text = typographer.process(p.translated_text)

        return TranslatedDocument(
            title=document.title,
            author=document.author,
            paragraphs=final_translated_paragraphs,
            metadata={
                **document.metadata,
                "chapter_page_breaks": bool(
                    self.config.output.chapter_page_breaks
                ),
            },
        )

    def _translate_single_chunk(
        self,
        idx: int,
        chunk: Chunk,
        memory_manager: MemoryManager,
        web_searcher: WebContextSearcher,
        glossary_manager: GlossaryManager,
        compliance_checker: GlossaryComplianceChecker,
        critique_tool: TranslationCritique,
        refiner_tool: TranslationRefiner,
        back_translator: BackTranslator,
        translations: dict[int, str],
        job_id: str,
        lock: threading.Lock | None = None,
        adjacent_source_context: str = "",
    ) -> str:
        # Cooperative pause check
        job_record = self.db.get_job(job_id)
        if job_record and job_record.get("status") == JobStatus.PAUSED:
            logger.info("Pipeline paused cooperatively for job %s", job_id)
            raise PipelinePausedException("Job paused cooperatively")

        if hasattr(self.llm_client, "set_trace_context"):
            self.llm_client.set_trace_context(job_id, idx)
        critic_client = getattr(self, "critic_client", self.llm_client)
        if hasattr(critic_client, "set_trace_context"):
            critic_client.set_trace_context(job_id, idx)
        self.db.clear_chunk_qa_records(job_id, idx)
        self.db.log_chunk_event(job_id, idx, "chunk_started", {
            "source_chars": len(chunk.text),
            "source_paragraphs": _paragraph_count(chunk.text),
            "paragraph_indices": chunk.metadata.get("paragraph_indices", []),
            "chapter_title": chunk.chapter_title,
            "section_title": chunk.section_title,
            "token_count": chunk.token_count,
        })

        # Translation memory retrieval
        if lock:
            with lock:
                mem_context = memory_manager.get_context_for_chunk(chunk)
        else:
            mem_context = memory_manager.get_context_for_chunk(chunk)
        qa_context = _bounded_qa_context(
            chunk, adjacent_source_context, mem_context
        )
        self.db.log_chunk_event(job_id, idx, "mqm_review_context", {
            "chars": len(qa_context),
            "has_adjacent_source": bool(adjacent_source_context),
            "has_style_rules": bool(mem_context.style_profile),
            "has_summary": bool(mem_context.bilingual_summary),
            "has_relevant_memory": bool(
                mem_context.proper_nouns
                or mem_context.long_term
                or mem_context.short_term
            ),
            "preview": _truncate_for_event(qa_context, 2400),
        })
        self.db.log_chunk_event(job_id, idx, "memory_context", {
            "has_style_profile": bool(mem_context.style_profile),
            "has_proper_nouns": bool(mem_context.proper_nouns),
            "has_long_term": bool(mem_context.long_term),
            "has_short_term": bool(mem_context.short_term),
            "has_bilingual_summary": bool(mem_context.bilingual_summary),
            "style_profile_preview": _truncate_for_event(mem_context.style_profile, 1000),
            "proper_nouns_preview": _truncate_for_event(mem_context.proper_nouns, 1000),
            "long_term_preview": _truncate_for_event(mem_context.long_term, 1000),
            "short_term_preview": _truncate_for_event(mem_context.short_term, 1000),
            "bilingual_summary_preview": _truncate_for_event(mem_context.bilingual_summary, 1000),
            "references": dict(mem_context.references),
        })

        # Web context (Aphra-style)
        web_context_str = ""
        if self.config.translation.enable_web_context:
            web_context_str = self._run_async(web_searcher.get_context_for_chunk(chunk, mem_context.format()))
            self.db.log_chunk_event(job_id, idx, "web_context", {
                "enabled": True,
                "has_context": bool(web_context_str.strip()),
                "chars": len(web_context_str),
                "preview": _truncate_for_event(web_context_str, 2000),
                "search_report": web_searcher.last_report,
            })
        else:
            self.db.log_chunk_event(job_id, idx, "web_context", {"enabled": False})

        # Prep prompts
        if lock:
            with lock:
                prev_trans = translations.get(idx - 1, "")
        else:
            prev_trans = translations.get(idx - 1, "")

        matched_entries = glossary_manager.find_terms(
            chunk.text,
            context=f"{chunk.chapter_title}\n{chunk.section_title}",
            domain=self.config.translation.domain,
        )
        contextual_advisories = getattr(
            glossary_manager, "last_contextual_advisories", []
        )
        if not isinstance(contextual_advisories, list):
            contextual_advisories = []
        citation_exempt_entries = [
            entry
            for entry in matched_entries
            if term_occurs_only_in_citations(chunk.text, entry.source)
        ]
        citation_exempt_sources = {
            entry.source.casefold() for entry in citation_exempt_entries
        }
        active_entries = [
            entry
            for entry in matched_entries
            if entry.source.casefold() not in citation_exempt_sources
        ]
        memory_reconciliations: list[dict[str, Any]] = []

        def reconcile_curated_memory() -> None:
            for entry in active_entries:
                if bool(getattr(entry, "is_auto", False)):
                    continue
                existing = memory_manager.proper_nouns.provenance_for(entry.source)
                include_original = bool(getattr(entry, "include_original", False))
                if not existing and not include_original:
                    continue
                outcome = memory_manager.proper_nouns.add_noun(
                    entry.source,
                    entry.target,
                    category=("approved_term" if include_original else "term"),
                    provenance="curated_glossary",
                )
                if outcome.get("action") in {
                    "replaced_lower_authority", "confirmed"
                }:
                    memory_reconciliations.append(outcome)

        if lock:
            with lock:
                reconcile_curated_memory()
        else:
            reconcile_curated_memory()
        if memory_reconciliations:
            self.db.log_chunk_event(
                job_id,
                idx,
                "terminology_memory_reconciled",
                {
                    "count": len(memory_reconciliations),
                    "mappings": memory_reconciliations[:50],
                    "policy": (
                        "Curated terminology supersedes stale automatic memory; "
                        "automatic suggestions cannot overwrite curated mappings."
                    ),
                },
            )
        enforce_auto_terms = bool(
            getattr(
                self.config.glossary,
                "enforce_auto_extracted_terms",
                False,
            )
        )
        enforced_entries = [
            entry
            for entry in active_entries
            if not bool(getattr(entry, "is_auto", False)) or enforce_auto_terms
        ]
        advisory_entries = [
            entry
            for entry in active_entries
            if bool(getattr(entry, "is_auto", False)) and not enforce_auto_terms
        ] + contextual_advisories
        if contextual_advisories:
            self.db.log_chunk_event(
                job_id,
                idx,
                "glossary_context_deferred",
                {
                    "count": len(contextual_advisories),
                    "entries": _glossary_entries_for_event(
                        contextual_advisories
                    ),
                    "policy": (
                        "Sense-qualified glossary rows without supporting author, "
                        "sense, or local-context evidence are advisory, not mandatory."
                    ),
                },
            )
        self.db.log_chunk_event(job_id, idx, "glossary_matches", {
            "matched_count": len(matched_entries) + len(contextual_advisories),
            "mandatory_count": len(enforced_entries),
            "advisory_count": len(advisory_entries),
            "auto_term_policy": (
                "mandatory" if enforce_auto_terms else "advisory"
            ),
            "entries": _glossary_entries_for_event(
                matched_entries + contextual_advisories
            ),
            "context_deferred_count": len(contextual_advisories),
            "citation_exemptions": [
                entry.source for entry in citation_exempt_entries
            ],
        })
        protected_targets = [
            str(getattr(entry, "target", "")).strip()
            for entry in enforced_entries
            if str(getattr(entry, "target", "")).strip()
            and (
                not bool(getattr(entry, "is_auto", False))
                or enforce_auto_terms
            )
        ]
        integrity_enabled = bool(
            getattr(self.config.translation, "enable_integrity_gate", True)
        )
        integrity_gate = PostEditIntegrityGate(
            min_retention_ratio=getattr(
                self.config.translation, "integrity_min_retention_ratio", 0.65
            ),
            max_growth_ratio=getattr(
                self.config.translation, "integrity_max_growth_ratio", 1.75
            ),
        )
        # Context-aware glossary table: includes each term's Context column
        # (author-specific sense, e.g. Marx's vs Bourdieu's "capital").
        glossary_prompt_parts = [
            glossary_manager.format_for_prompt(enforced_entries),
            (
                glossary_manager.format_advisory_for_prompt(advisory_entries)
                if advisory_entries
                else ""
            ),
        ]
        glossary_terms_str = "\n\n".join(
            part for part in glossary_prompt_parts if part
        ) or "(no glossary terms matched in this chunk)"

        style_register = self.config.translation.style_register
        if style_register == "academic":
            from tarjomeh.core.prompts import ACADEMIC_REGISTER_MODIFIER
            style_register_value = f"{style_register}\n{ACADEMIC_REGISTER_MODIFIER}"
            exemplars = ACADEMIC_EXEMPLARS
        else:
            style_register_value = style_register
            exemplars = ""

        term_notes_instruction = _term_notes_instruction(
            effective_term_notes_mode(
                self.config.output.term_notes,
                self.config.output.format,
            )
        )
        effective_notes_mode = effective_term_notes_mode(
            self.config.output.term_notes,
            self.config.output.format,
        )
        protect_inline_english = effective_notes_mode in {"inline", "both"}
        if lock:
            with lock:
                pending_originals = memory_manager.proper_nouns.pending_inline_originals(
                    chunk.text
                )
        else:
            pending_originals = memory_manager.proper_nouns.pending_inline_originals(
                chunk.text
            )
        known_source_entities = {
            value.casefold()
            for value in memory_manager.proper_nouns.all_nouns()
        }
        source_entity_candidates = (
            [
                value for value in _source_entity_inventory(chunk.text)
                if not memory_manager.proper_nouns.is_introduced(value)
                and value.casefold() not in known_source_entities
            ]
            if not _is_front_matter(chunk) else []
        )
        source_entity_categories = _source_entity_categories(
            chunk.text, source_entity_candidates
        )
        required_person_candidates = _high_confidence_person_candidates(
            source_entity_candidates, chunk.text
        )
        required_instrument_candidates = _high_confidence_instrument_candidates(
            source_entity_candidates
        )
        required_entity_candidates = list(dict.fromkeys([
            *required_person_candidates,
            *required_instrument_candidates,
        ]))
        allowed_inline_originals = (
            sorted(
                set(pending_originals).union(source_entity_candidates),
                key=str.casefold,
            )
            if protect_inline_english else []
        )
        established_originals_text = (
            ", ".join(f"({value})" for value in pending_originals)
            or "(none)"
        )
        candidate_originals_text = (
            ", ".join(f"({value})" for value in source_entity_candidates)
            or "(none)"
        )
        required_originals_text = (
            ", ".join(f"({value})" for value in required_entity_candidates)
            or "(none)"
        )
        inline_policy_context = (
            "\n\n### Deterministic English-original allowlist for this chunk\n"
            f"Established pending originals: {established_originals_text}\n"
            f"Current-source entity candidates: {candidate_originals_text}\n"
            f"Required high-confidence people/instruments: {required_originals_text}\n"
            "For every candidate that is genuinely a person, place, institution, "
            "publication, product, or named theory in this passage, render it as "
            "Persian followed immediately by its exact English original in parentheses. "
            "A candidate is permission, not a command: reject title-cased ordinary "
            "prose that is not an entity. Only listed originals may be added. "
            "Every required high-confidence item must appear exactly once as its "
            "Persian rendering followed by the listed English original. "
            "Ordinary concepts and all unlisted terms must remain Persian-only. "
            "Source citations are separate and must be preserved."
        )
        self.db.log_chunk_event(job_id, idx, "inline_original_policy", {
            "enabled": protect_inline_english,
            "allowed_originals": allowed_inline_originals,
            "established_originals": list(pending_originals),
            "source_entity_candidates": source_entity_candidates,
            "required_person_candidates": required_person_candidates,
            "required_instrument_candidates": required_instrument_candidates,
            "required_entity_candidates": required_entity_candidates,
            "categories": {
                source: (
                    memory_manager.proper_nouns.category_for(source)
                    if source in pending_originals
                    else source_entity_categories.get(
                        source, "source_entity_candidate"
                    )
                )
                for source in allowed_inline_originals
            },
        })
        sys_prompt = TRANSLATE_SYSTEM_PROMPT.format(
            domain=self.config.translation.domain,
            style_register=style_register_value,
            country=self.config.translation.country,
            term_notes_instruction=term_notes_instruction,
        )
        source_paragraphs = [
            paragraph.strip()
            for paragraph in chunk.text.split("\n\n")
            if paragraph.strip()
        ]
        n_source_paras = len(chunk.metadata.get("paragraph_indices", [])) or \
            len(source_paragraphs)
        encoded_source, paragraph_markers = encode_paragraphs(chunk.text)
        use_paragraph_protocol = bool(
            len(paragraph_markers) > 1
            and int(chunk.metadata.get("paragraph_protocol_version", 0)) >= 1
        )

        def build_translation_prompt(
            source_text: str,
            previous: str,
            recovery_context: str = "",
            recovery_segment_id: str = "",
        ) -> str:
            paragraph_count = len([
                paragraph for paragraph in source_text.split("\n\n")
                if paragraph.strip()
            ])
            prompt = TRANSLATE_CHUNK_PROMPT.format(
                exemplars=exemplars,
                glossary_terms=glossary_terms_str,
                memory_context=mem_context.format() + inline_policy_context,
                web_context=web_context_str,
                previous_translation=previous,
                source_text=source_text,
                paragraph_count=paragraph_count,
                term_notes_instruction=term_notes_instruction,
            )
            if recovery_context:
                scope = (
                    "### CONTEXT ONLY - DO NOT TRANSLATE\n"
                    "The bounded text below is context, not a translation target.\n"
                    "<<<CONTEXT>>>\n"
                    f"{recovery_context[:600]}\n"
                    "<<<END CONTEXT>>>\n\n"
                )
                prompt = prompt.replace(
                    "### Source text to translate\n",
                    scope + "### Source text to translate\n",
                    1,
                )
            if recovery_segment_id:
                prompt += _recovery_segment_instruction(recovery_segment_id)
            return prompt

        user_content = build_translation_prompt(chunk.text, prev_trans)
        if use_paragraph_protocol:
            user_content = build_translation_prompt(encoded_source, prev_trans)
            user_content += protocol_instruction(paragraph_markers)

        # Terminology context for the judge & refiner: matched glossary terms
        # plus the established proper-noun renderings, so the "terminology"
        # dimension is scored against the actual mandate instead of blind.
        terminology_ctx = glossary_terms_str
        if mem_context.proper_nouns:
            terminology_ctx += (
                "\n\n### Mandatory established proper-noun renderings\n"
                + mem_context.proper_nouns
            )
        terminology_ctx += (
            "\n\n### First-occurrence English-original policy\n"
            + term_notes_instruction
            + " During critique and refinement, preserve any required English "
              "original already present unless it is factually incorrect. Do "
              "not infer that every ordinary glossary concept requires an "
              "English parenthetical; follow the proper-noun state and the "
              "configured policy above exactly."
            + inline_policy_context
        )
        self.db.log_chunk_event(job_id, idx, "terminology_context", {
            "chars": len(terminology_ctx),
            "preview": _truncate_for_event(terminology_ctx, 2000),
        })

        # Translate
        self.db.update_chunk(job_id, idx, ChunkStatus.TRANSLATING)
        initial_integrity = None
        try:
            translation = self.llm_client.complete(
                messages=[{"role": "user", "content": user_content}],
                system_prompt=sys_prompt,
                _operation="translation",
                _recovery_source_text=chunk.text,
            )
            if use_paragraph_protocol:
                protocol_result = decode_paragraphs(
                    translation, paragraph_markers
                )
                self.db.log_chunk_event(
                    job_id, idx, "paragraph_protocol_checked", {
                        "stage": "initial_translation",
                        "valid": protocol_result.valid,
                        "expected_markers": paragraph_markers,
                        "errors": protocol_result.errors,
                    },
                )
                if not protocol_result.valid:
                    repaired = self.llm_client.complete(
                        messages=[{
                            "role": "user",
                            "content": paragraph_repair_prompt(
                                translation,
                                paragraph_markers,
                                protocol_result.errors,
                            ),
                        }],
                        system_prompt=(
                            "You repair paragraph labels only. Preserve every word "
                            "of the supplied Persian translation."
                        ),
                        _operation="translation_paragraph_repair",
                        _recovery_source_text=chunk.text,
                    )
                    protocol_result = decode_paragraphs(
                        repaired, paragraph_markers
                    )
                    self.db.log_chunk_event(
                        job_id, idx, "paragraph_protocol_repair", {
                            "stage": "initial_translation",
                            "valid": protocol_result.valid,
                            "errors": protocol_result.errors,
                        },
                    )
                if not protocol_result.valid:
                    raise TruncatedCompletionError(
                        "Multi-paragraph output failed stable paragraph identity; "
                        "using bounded paragraph recovery."
                    )
                translation = protocol_result.text
        except (
            TruncatedCompletionError,
            EmptyCompletionError,
            IncompleteCompletionError,
        ) as recovery_trigger:
            recovery_paragraphs = source_paragraphs or [chunk.text.strip()]
            if isinstance(recovery_trigger, EmptyCompletionError):
                recovery_reason = "repeated_empty_completion"
            elif isinstance(recovery_trigger, IncompleteCompletionError):
                recovery_reason = "repeated_incomplete_stream"
            elif "paragraph identity" in str(recovery_trigger).casefold():
                recovery_reason = "paragraph_protocol_invalid"
            else:
                recovery_reason = "repeated_finish_reason_length"
            self.db.log_chunk_event(
                job_id,
                idx,
                "translation_adaptive_split",
                {
                    "reason": recovery_reason,
                    "part_count": len(recovery_paragraphs),
                    "message": (
                        "The complete chunk remained unusable after bounded "
                        "same-request recovery; validated paragraph-boundary "
                        "recovery was activated."
                    ),
                },
            )

            def request_recovery_part(
                source_text: str,
                previous: str,
                segment_id: str,
                source_context: str = "",
            ) -> str:
                last_diagnostics: dict[str, Any] = {}
                for validation_attempt in range(2):
                    effective_context = source_context if validation_attempt == 0 else ""
                    effective_previous = previous if validation_attempt == 0 else ""
                    prompt = build_translation_prompt(
                        source_text,
                        effective_previous,
                        recovery_context=effective_context,
                        recovery_segment_id=segment_id,
                    )
                    raw = self.llm_client.complete(
                        messages=[{"role": "user", "content": prompt}],
                        system_prompt=sys_prompt,
                        _operation="translation_split_recovery",
                        _recovery_source_text=source_text,
                    ).strip()
                    candidate, envelope = _parse_recovery_segment(raw, segment_id)
                    content_check = _validate_recovery_part(
                        source_text,
                        candidate,
                        previous_target=previous,
                        source_context=effective_context,
                        structural_role=str(
                            chunk.metadata.get("structural_role", "body")
                        ),
                    )
                    errors = list(envelope.get("errors", []))
                    errors.extend(content_check.get("errors", []))
                    last_diagnostics = {
                        **envelope,
                        **content_check,
                        "segment_id": segment_id,
                        "validation_attempt": validation_attempt + 1,
                        "strict_target_only": validation_attempt > 0,
                        "valid": not errors,
                        "errors": sorted(set(errors)),
                    }
                    self.db.log_chunk_event(
                        job_id,
                        idx,
                        "translation_recovery_part",
                        last_diagnostics,
                    )
                    if not errors:
                        return candidate
                raise ValueError(
                    f"Adaptive recovery segment {segment_id} failed validation: "
                    + ", ".join(last_diagnostics.get("errors", []))
                )

            def translate_sentence_groups(
                source_paragraph: str,
                previous: str,
                paragraph_index: int,
                reason: str,
                strict_target_only: bool = False,
            ) -> str:
                sentence_groups = _split_source_recovery_groups(source_paragraph)
                if len(sentence_groups) <= 1:
                    raise TruncatedCompletionError(
                        "Source paragraph cannot be split at sentence boundaries."
                    )
                self.db.log_chunk_event(
                    job_id,
                    idx,
                    "translation_sentence_split",
                    {
                        "paragraph_index": paragraph_index,
                        "part_count": len(sentence_groups),
                        "reason": reason,
                    },
                )
                sentence_translations: list[str] = []
                sentence_continuity = previous
                for sentence_index, sentence_group in enumerate(sentence_groups):
                    neighbors = ""
                    if not strict_target_only:
                        context_parts = []
                        if sentence_index > 0:
                            context_parts.append(
                                "Previous source sentence group:\n"
                                + sentence_groups[sentence_index - 1][-300:]
                            )
                        if sentence_index + 1 < len(sentence_groups):
                            context_parts.append(
                                "Next source sentence group:\n"
                                + sentence_groups[sentence_index + 1][:300]
                            )
                        neighbors = "\n\n".join(context_parts)[:600]
                    sentence_translation = request_recovery_part(
                        sentence_group,
                        sentence_continuity,
                        f"c{idx}.p{paragraph_index}.s{sentence_index}",
                        neighbors,
                    )
                    sentence_translations.append(sentence_translation)
                    sentence_continuity = sentence_translation
                return " ".join(sentence_translations).strip()

            def recover_all_parts(strict_target_only: bool = False) -> str:
                recovered_parts: list[str] = []
                continuity = prev_trans
                for part_index, source_paragraph in enumerate(recovery_paragraphs):
                    part_previous = "" if strict_target_only else continuity
                    if len(recovery_paragraphs) == 1:
                        groups = _split_source_recovery_groups(source_paragraph)
                        if len(groups) > 1:
                            recovered = translate_sentence_groups(
                                source_paragraph,
                                part_previous,
                                part_index,
                                "single_paragraph_chunk_exhausted_output_budget",
                                strict_target_only,
                            )
                            recovered_parts.append(recovered)
                            continuity = recovered
                            continue
                    try:
                        recovered = request_recovery_part(
                            source_paragraph,
                            part_previous,
                            f"c{idx}.p{part_index}",
                        )
                    except TruncatedCompletionError:
                        recovered = translate_sentence_groups(
                            source_paragraph,
                            part_previous,
                            part_index,
                            "paragraph_recovery_exhausted_output_budget",
                            strict_target_only,
                        )
                    recovered_parts.append(recovered)
                    continuity = recovered
                assembled = "\n\n".join(recovered_parts)
                if _paragraph_count(assembled) != len(recovery_paragraphs):
                    raise ValueError(
                        "Adaptive recovery assembly changed paragraph count."
                    )
                return assembled

            def log_recovery_protocol(stage: str, assembled: str) -> None:
                if not use_paragraph_protocol:
                    return
                candidate_count = _paragraph_count(assembled)
                self.db.log_chunk_event(
                    job_id,
                    idx,
                    "paragraph_protocol_checked",
                    {
                        "stage": stage,
                        "valid": candidate_count == len(recovery_paragraphs),
                        "expected_paragraphs": len(recovery_paragraphs),
                        "candidate_paragraphs": candidate_count,
                        "recovery_assembly": True,
                    },
                )

            translation = recover_all_parts()
            log_recovery_protocol("adaptive_recovery_assembly", translation)

            translation, identifier_repairs = restore_source_identifiers(
                chunk.text, translation
            )
            if identifier_repairs["repair_count"]:
                self.db.log_chunk_event(
                    job_id, idx, "source_identifiers_restored", {
                        "stage": "adaptive_recovery_assembly",
                        **identifier_repairs,
                    },
                )

            if integrity_enabled:
                initial_integrity = integrity_gate.evaluate(
                    chunk.text,
                    translation,
                    stage="adaptive_recovery_assembly",
                    protect_inline_english=protect_inline_english,
                    allowed_inline_originals=allowed_inline_originals,
                )
                self.db.log_chunk_event(
                    job_id, idx, "integrity_check_completed",
                    initial_integrity.to_dict(),
                )
                if not initial_integrity.accepted:
                    self.db.log_chunk_event(
                        job_id,
                        idx,
                        "translation_recovery_assembly_rejected",
                        {
                            **initial_integrity.to_dict(),
                            "action": "strict_target_only_retry",
                        },
                    )
                    translation = recover_all_parts(strict_target_only=True)
                    log_recovery_protocol(
                        "adaptive_recovery_strict_assembly", translation
                    )
                    translation, identifier_repairs = restore_source_identifiers(
                        chunk.text, translation
                    )
                    if identifier_repairs["repair_count"]:
                        self.db.log_chunk_event(
                            job_id, idx, "source_identifiers_restored", {
                                "stage": "adaptive_recovery_strict_assembly",
                                **identifier_repairs,
                            },
                        )
                    initial_integrity = integrity_gate.evaluate(
                        chunk.text,
                        translation,
                        stage="adaptive_recovery_strict_assembly",
                        protect_inline_english=protect_inline_english,
                        allowed_inline_originals=allowed_inline_originals,
                    )
                    self.db.log_chunk_event(
                        job_id, idx, "integrity_check_completed",
                        initial_integrity.to_dict(),
                    )
                    if not initial_integrity.accepted:
                        self.db.log_chunk_event(
                            job_id,
                            idx,
                            "translation_recovery_assembly_rejected",
                            {
                                **initial_integrity.to_dict(),
                                "action": "continue_to_quality_repair",
                            },
                        )
        if not translation or not translation.strip():
            raise ValueError(f"LLM returned an empty or whitespace-only translation for chunk {idx}.")
        translation, orthography_edits = apply_safe_persian_orthography(
            translation
        )
        translation, identifier_repairs = restore_source_identifiers(
            chunk.text, translation
        )
        if identifier_repairs["repair_count"]:
            self.db.log_chunk_event(
                job_id, idx, "source_identifiers_restored", {
                    "stage": "initial_translation",
                    **identifier_repairs,
                },
            )
        if orthography_edits:
            self.db.log_chunk_event(
                job_id, idx, "persian_orthography_normalized", {
                    "stage": "initial_translation",
                    "edits": orthography_edits,
                    "edit_count": sum(
                        int(edit.get("count", 0)) for edit in orthography_edits
                    ),
                },
            )
        # Item 15: reconstruct corrupted numerals the source disambiguates
        # completely, BEFORE the integrity gate and before typography. Without
        # this, a single stray replacement character costs a human review even
        # when the source makes the fix unambiguous. Anything ambiguous is left
        # untouched, so it still blocks and still reaches a reviewer.
        # Guarded: a defect in a repair helper must never pause a book. The
        # translation is already valid without the repair, so on failure we log
        # and carry on rather than letting the exception reach the chunk loop -
        # where pause_on_sequential_error (default True) would stop the job on
        # the very first affected chunk.
        try:
            translation, corruption_repairs = repair_corruption(
                chunk.text, translation
            )
        except Exception:
            logger.exception("Corruption repair failed for chunk %s", idx)
            self.db.log_chunk_event(
                job_id, idx, "unicode_corruption_repair_failed", {
                    "stage": "initial_translation",
                }
            )
            corruption_repairs = []
        if corruption_repairs:
            self.db.log_chunk_event(
                job_id, idx, "unicode_corruption_repair", {
                    "stage": "initial_translation",
                    "repaired": sum(
                        1 for entry in corruption_repairs if entry["repaired"]
                    ),
                    "left_for_review": sum(
                        1 for entry in corruption_repairs if not entry["repaired"]
                    ),
                    "details": corruption_repairs[:10],
                },
            )
        # Always evaluate the finalized first draft after deterministic repairs.
        # Adaptive recovery may have evaluated an earlier assembly, but identifier,
        # orthography, and corruption repair can legitimately change that result.
        baseline_integrity_accepted = not integrity_enabled
        if integrity_enabled:
            initial_integrity = integrity_gate.evaluate(
                chunk.text,
                translation,
                stage="initial_translation",
                protect_inline_english=protect_inline_english,
                allowed_inline_originals=allowed_inline_originals,
            )
            self.db.log_chunk_event(
                job_id, idx, "integrity_check_completed", initial_integrity.to_dict()
            )
            if not initial_integrity.accepted:
                self.db.log_chunk_event(
                    job_id,
                    idx,
                    "integrity_initial_failed",
                    {
                        **initial_integrity.to_dict(),
                        "action": "quarantine_until_repaired",
                    },
                )
                self.db.log_chunk_event(
                    job_id,
                    idx,
                    "translation_baseline_quarantined",
                    {
                        "translation_chars": len(translation),
                        "repair_path": (
                            "targeted_integrity_then_critique_refinement_and_glossary"
                        ),
                        "memory_eligible": False,
                        "export_eligible": False,
                    },
                )

                repair_source = chunk.text
                repair_translation = translation
                repair_markers: list[str] = []
                marked_translation, target_markers = encode_paragraphs(translation)
                if use_paragraph_protocol and target_markers == paragraph_markers:
                    repair_source = encoded_source
                    repair_translation = marked_translation
                    repair_markers = paragraph_markers
                repair_prompt = _integrity_repair_prompt(
                    source=repair_source,
                    rejected_translation=repair_translation,
                    integrity_payload=initial_integrity.to_dict(),
                    terminology=terminology_ctx,
                    inline_policy=inline_policy_context,
                    paragraph_count=n_source_paras,
                    paragraph_instruction=(
                        protocol_instruction(repair_markers)
                        if repair_markers else ""
                    ),
                )
                repair_event: dict[str, Any] = {
                    "stage": "initial_translation_integrity_repair",
                    "attempted": True,
                    "accepted": False,
                    "blocking_checks": [
                        finding.check_id for finding in initial_integrity.blocking
                    ],
                }
                try:
                    self.llm_client.limit_next_call_attempts(2)
                    repaired_translation = self.llm_client.complete(
                        messages=[{"role": "user", "content": repair_prompt}],
                        system_prompt=sys_prompt,
                        _operation="translation_integrity_repair",
                        _recovery_source_text=chunk.text,
                    )
                    protocol_valid = True
                    protocol_errors: list[str] = []
                    if repair_markers:
                        repair_protocol = decode_paragraphs(
                            repaired_translation, repair_markers
                        )
                        protocol_valid = repair_protocol.valid
                        protocol_errors = repair_protocol.errors
                        self.db.log_chunk_event(
                            job_id, idx, "paragraph_protocol_checked", {
                                "stage": "initial_translation_integrity_repair",
                                "valid": protocol_valid,
                                "expected_markers": repair_markers,
                                "errors": protocol_errors,
                            },
                        )
                        if protocol_valid:
                            repaired_translation = repair_protocol.text
                    if protocol_valid and repaired_translation.strip():
                        repaired_translation, repair_orthography = (
                            apply_safe_persian_orthography(repaired_translation)
                        )
                        repaired_translation, repair_identifiers = (
                            restore_source_identifiers(
                                chunk.text, repaired_translation
                            )
                        )
                        try:
                            repaired_translation, repair_corruptions = (
                                repair_corruption(
                                    chunk.text, repaired_translation
                                )
                            )
                        except Exception:
                            logger.exception(
                                "Corruption repair failed during integrity recovery "
                                "for chunk %s",
                                idx,
                            )
                            repair_corruptions = []
                        repair_integrity = integrity_gate.evaluate(
                            chunk.text,
                            repaired_translation,
                            stage="initial_translation_integrity_repair",
                            protected_terms=protected_targets,
                            protect_inline_english=protect_inline_english,
                            allowed_inline_originals=allowed_inline_originals,
                            enforce_all_terms=False,
                        )
                        self.db.log_chunk_event(
                            job_id,
                            idx,
                            "integrity_check_completed",
                            repair_integrity.to_dict(),
                        )
                        repair_event.update({
                            "accepted": repair_integrity.accepted,
                            "translation_chars": len(repaired_translation),
                            "integrity": repair_integrity.to_dict(),
                            "orthography_edit_count": sum(
                                int(edit.get("count", 0))
                                for edit in repair_orthography
                            ),
                            "identifier_repair_count": int(
                                repair_identifiers.get("repair_count", 0)
                            ),
                            "corruption_repair_count": len(
                                repair_corruptions
                            ),
                        })
                        if repair_integrity.accepted:
                            translation = repaired_translation
                            initial_integrity = repair_integrity
                            self.db.log_chunk_event(
                                job_id,
                                idx,
                                "translation_baseline_repaired",
                                {
                                    "stage": (
                                        "initial_translation_integrity_repair"
                                    ),
                                    "translation_chars": len(translation),
                                },
                            )
                    else:
                        repair_event["protocol_errors"] = protocol_errors
                except _QUALITY_STAGE_ERRORS as exc:
                    repair_event.update({
                        "failure_type": type(exc).__name__,
                        "error": str(exc),
                    })
                except Exception as exc:
                    logger.exception(
                        "Targeted integrity recovery failed for chunk %s", idx
                    )
                    repair_event.update({
                        "failure_type": type(exc).__name__,
                        "error": str(exc),
                    })
                self.db.log_chunk_event(
                    job_id, idx, "translation_integrity_repair", repair_event
                )
            baseline_integrity_accepted = initial_integrity.accepted

        self.db.update_chunk(job_id, idx, ChunkStatus.TRANSLATED, translation)
        # A repairable first draft may reach the bounded quality loop, but it is
        # not trusted state.  Only an integrity-passing version can be restored,
        # committed to memory, or exported by the caller.
        last_accepted_translation: str | None = (
            translation if baseline_integrity_accepted else None
        )
        self.db.log_chunk_event(job_id, idx, "translation_completed", {
            "translation_chars": len(translation),
            "translation_paragraphs": _paragraph_count(translation),
            "expected_paragraphs": n_source_paras,
            "baseline_integrity_accepted": baseline_integrity_accepted,
        })

        # Critique and Refine (judge scores against the terminology mandate)
        if self.config.translation.enable_critique:
            threshold = getattr(self.config.translation, "critique_threshold", 9.0)
            current_integrity_accepted = baseline_integrity_accepted
            accepted_versions = [translation] if current_integrity_accepted else []
            evaluated_versions: list[tuple[str, Any]] = []
            for ref_iter in range(self.config.translation.max_refine_iterations + 1):
                try:
                    critique_rep = self._run_async(
                        critique_tool.critique(
                            chunk.text,
                            translation,
                            terminology=terminology_ctx,
                            review_context=qa_context,
                        )
                    )
                except _QUALITY_STAGE_ERRORS as exc:
                    payload = _qa_provider_failure_payload(
                        "critic", "critique", critic_client, exc
                    )
                    payload["iteration"] = ref_iter
                    self.db.log_chunk_event(
                        job_id, idx, "qa_unavailable", payload
                    )
                    self.db.log_event(
                        job_id,
                        "WARNING",
                        f"Critique unavailable for Chunk {idx}: {type(exc).__name__}",
                    )
                    break
                self.db.update_chunk(job_id, idx, ChunkStatus.CRITIQUED)
                if getattr(critique_rep, "attempts", 1) > 1:
                    retry_payload = {
                        "attempts": critique_rep.attempts,
                        "recovered": bool(getattr(critique_rep, "valid", False)),
                        "validation_errors": list(
                            getattr(critique_rep, "validation_errors", []) or []
                        ),
                    }
                    self.db.log_chunk_event(
                        job_id, idx, "critic_response_invalid", retry_payload
                    )
                    self.db.log_chunk_event(
                        job_id, idx, "critic_response_retried", retry_payload
                    )
                policy_conflicts = []
                glossary_conflicts = []
                if getattr(critique_rep, "valid", True):
                    policy_conflicts = _filter_critique_policy_conflicts(
                        critique_rep, chunk.text, allowed_inline_originals
                    )
                    glossary_conflicts = _filter_critique_glossary_conflicts(
                        critique_rep,
                        enforced_entries,
                        include_auto=enforce_auto_terms,
                    )
                ignored_issues = list(
                    getattr(critique_rep, "ignored_issue_details", []) or []
                )
                if ignored_issues:
                    self.db.log_chunk_event(
                        job_id,
                        idx,
                        "critique_noop_issues_filtered",
                        {
                            "iteration": ref_iter,
                            "count": len(ignored_issues),
                            "issues": ignored_issues,
                            "reason_counts": {
                                reason: sum(
                                    1 for issue in ignored_issues
                                    if issue.get("ignored_reason") == reason
                                )
                                for reason in {
                                    str(issue.get("ignored_reason", "unknown"))
                                    for issue in ignored_issues
                                }
                            },
                            "message": (
                                "No-op, duplicate, excess, or ungrounded critic "
                                "items were withheld from refinement."
                            ),
                        },
                    )
                if policy_conflicts:
                    self.db.log_chunk_event(
                        job_id, idx, "critique_policy_conflicts_filtered", {
                            "iteration": ref_iter,
                            "count": len(policy_conflicts),
                            "allowed_originals": allowed_inline_originals,
                            "conflicts": policy_conflicts,
                            "message": (
                                "Only critic instructions contradicting deterministic "
                                "inline-original or source-apparatus policy were withheld "
                                "from refinement."
                            ),
                        }
                    )
                if glossary_conflicts:
                    self.db.log_chunk_event(
                        job_id, idx, "critique_glossary_conflicts_filtered", {
                            "iteration": ref_iter,
                            "count": len(glossary_conflicts),
                            "conflicts": glossary_conflicts,
                            "message": (
                                "Critic terminology advice that removed a curated "
                                "rendering was withheld from refinement."
                            ),
                        }
                    )
                concept_risks = _high_risk_concepts(critique_rep)
                if concept_risks:
                    self.db.log_chunk_event(
                        job_id, idx, "high_risk_concepts_flagged", {
                            "iteration": ref_iter,
                            "count": len(concept_risks),
                            "concepts": concept_risks,
                            "blocking": False,
                            "message": (
                                "Context-sensitive concepts were queued for review; "
                                "no translation was forced or paused."
                            ),
                        }
                    )
                if getattr(critique_rep, "valid", True):
                    self.db.save_qa_issues(
                        job_id,
                        idx,
                        ref_iter,
                        list(getattr(critique_rep, "issue_details", []) or []),
                    )
                self.db.log_chunk_event(
                    job_id,
                    idx,
                    "critique_completed",
                    _critique_for_event(critique_rep, threshold, ref_iter),
                )
                if (
                    getattr(critique_rep, "issue_details", None)
                    and not critique_rep.passes_threshold(threshold)
                    and not _critique_requires_refinement(critique_rep, threshold)
                ):
                    self.db.log_chunk_event(
                        job_id,
                        idx,
                        "mqm_minor_only_deferred",
                        {
                            "iteration": ref_iter,
                            "critique_average": critique_rep.average,
                            "issue_ids": [
                                detail.get("issue_id")
                                for detail in critique_rep.issue_details
                            ],
                            "message": (
                                "Minor-only MQM advice was recorded without "
                                "starting another refinement loop."
                            ),
                        },
                    )
                if not getattr(critique_rep, "valid", True):
                    if getattr(critique_rep, "attempts", 1) <= 1:
                        self.db.log_chunk_event(
                            job_id, idx, "critic_response_invalid", {
                                "attempts": getattr(critique_rep, "attempts", 1),
                                "recovered": False,
                                "validation_errors": list(
                                    getattr(critique_rep, "validation_errors", []) or []
                                ),
                            },
                        )
                    self.db.log_chunk_event(job_id, idx, "qa_unavailable", {
                        "component": "critic",
                        "iteration": ref_iter,
                        "attempts": getattr(critique_rep, "attempts", 1),
                        "validation_errors": list(
                            getattr(critique_rep, "validation_errors", []) or []
                        ),
                        "message": (
                            "Critic output remained invalid after bounded repair; "
                            "translation kept and chunk requires human review."
                        ),
                    })
                    break
                if current_integrity_accepted:
                    evaluated_versions.append((translation, critique_rep))
                if (
                    current_integrity_accepted
                    and _critique_passes_quality_gate(critique_rep, threshold)
                ):
                    break
                if ref_iter == self.config.translation.max_refine_iterations:
                    blocking_issues = _blocking_critique_issues(critique_rep)
                    review_reason = (
                        "blocking_critique_disagreement"
                        if blocking_issues else "quality_threshold_unmet"
                    )
                    message = (
                        "Unresolved blocking critique disagreement after maximum "
                        "refinement attempts; output kept but chunk needs human review."
                        if blocking_issues else
                        "Configured quality threshold remained unmet after maximum "
                        "refinement attempts; output kept but chunk needs human review."
                    )
                    self.db.log_chunk_event(job_id, idx, "critique_needs_review", {
                        "iteration": ref_iter,
                        "critique_average": critique_rep.average,
                        "blocking_issue_count": len(blocking_issues),
                        "blocking_issues": [
                            _truncate_for_event(str(issue), 1000)
                            for issue in blocking_issues
                        ],
                        "review_reason": review_reason,
                        "message": message,
                    })
                    break
                before_translation = translation
                before_chars = len(before_translation)
                refinement_source = chunk.text
                refinement_translation = translation
                refinement_context = qa_context
                refinement_markers: list[str] = []
                source_for_refinement, source_refinement_markers = (
                    encode_paragraphs(chunk.text)
                )
                translation_for_refinement, target_refinement_markers = (
                    encode_paragraphs(translation)
                )
                if (
                    len(source_refinement_markers) > 1
                    and source_refinement_markers == target_refinement_markers
                ):
                    refinement_source = source_for_refinement
                    refinement_translation = translation_for_refinement
                    refinement_markers = source_refinement_markers
                    refinement_context = (
                        qa_context + protocol_instruction(refinement_markers)
                    )
                try:
                    refinement = self._run_async(
                        refiner_tool.refine_with_decision(
                            refinement_source,
                            refinement_translation,
                            critique_rep,
                            terminology=terminology_ctx,
                            review_context=refinement_context,
                        )
                    )
                except _QUALITY_STAGE_ERRORS as exc:
                    payload = _qa_provider_failure_payload(
                        "refiner", "refinement", self.llm_client, exc
                    )
                    payload["iteration"] = ref_iter + 1
                    self.db.log_chunk_event(
                        job_id, idx, "qa_unavailable", payload
                    )
                    self.db.log_event(
                        job_id,
                        "WARNING",
                        f"Refinement unavailable for Chunk {idx}: {type(exc).__name__}",
                    )
                    break
                if getattr(refinement, "attempts", 1) > 1:
                    refiner_retry_payload = {
                        "component": "refiner",
                        "operation": "refinement",
                        "failure_type": "structured_response_invalid",
                        "iteration": ref_iter + 1,
                        "attempts": refinement.attempts,
                        "recovered": bool(getattr(refinement, "valid", False)),
                        "validation_errors": list(
                            getattr(refinement, "validation_errors", []) or []
                        ),
                    }
                    self.db.log_chunk_event(
                        job_id, idx, "refiner_response_invalid",
                        refiner_retry_payload,
                    )
                    self.db.log_chunk_event(
                        job_id, idx, "refiner_response_retried",
                        refiner_retry_payload,
                    )
                if not getattr(refinement, "valid", True):
                    invalid_payload = {
                        "component": "refiner",
                        "operation": "refinement",
                        "failure_type": "structured_response_invalid",
                        "iteration": ref_iter + 1,
                        "attempts": getattr(refinement, "attempts", 1),
                        "validation_errors": list(
                            getattr(refinement, "validation_errors", []) or []
                        ),
                        "message": (
                            "Refiner output remained invalid after bounded repair; "
                            "prior translation retained."
                        ),
                    }
                    if getattr(refinement, "attempts", 1) <= 1:
                        self.db.log_chunk_event(
                            job_id, idx, "refiner_response_invalid", invalid_payload
                        )
                    self.db.log_chunk_event(
                        job_id, idx, "qa_unavailable",
                        invalid_payload,
                    )
                    break

                proposed_translation = refinement.translation
                if refinement_markers:
                    refinement_protocol = decode_paragraphs(
                        proposed_translation, refinement_markers
                    )
                    self.db.log_chunk_event(
                        job_id, idx, "paragraph_protocol_checked", {
                            "stage": "refinement",
                            "iteration": ref_iter + 1,
                            "valid": refinement_protocol.valid,
                            "expected_markers": refinement_markers,
                            "errors": refinement_protocol.errors,
                        },
                    )
                    if not refinement_protocol.valid:
                        self.db.log_chunk_event(
                            job_id, idx, "integrity_edit_rejected", {
                                "stage": "refinement_paragraph_protocol",
                                "accepted": False,
                                "blocking_count": 1,
                                "findings": [{
                                    "check_id": "paragraph_identity_changed",
                                    "severity": "blocking",
                                    "message": (
                                        "Refinement changed stable paragraph identity."
                                    ),
                                    "errors": refinement_protocol.errors,
                                }],
                            },
                        )
                        proposed_translation = before_translation
                    else:
                        proposed_translation = refinement_protocol.text
                proposed_translation, orthography_edits = (
                    apply_safe_persian_orthography(proposed_translation)
                )
                if orthography_edits:
                    self.db.log_chunk_event(
                        job_id, idx, "persian_orthography_normalized", {
                            "stage": "refinement",
                            "iteration": ref_iter + 1,
                            "edits": orthography_edits,
                            "edit_count": sum(
                                int(edit.get("count", 0))
                                for edit in orthography_edits
                            ),
                        },
                    )
                edit_accepted = True
                integrity_payload: dict[str, Any] | None = None
                issue_details = list(
                    getattr(critique_rep, "issue_details", []) or []
                )
                issue_decisions = list(
                    getattr(refinement, "issue_decisions", []) or []
                )
                local_salvage: dict[str, Any] | None = None
                salvaged_translation = before_translation
                if integrity_enabled:
                    edit_integrity = integrity_gate.evaluate(
                        chunk.text,
                        proposed_translation,
                        previous=(
                            before_translation
                            if current_integrity_accepted else ""
                        ),
                        stage="refinement",
                        protected_terms=protected_targets,
                        protect_inline_english=protect_inline_english,
                        allowed_inline_originals=allowed_inline_originals,
                    )
                    integrity_payload = edit_integrity.to_dict()
                    self.db.log_chunk_event(
                        job_id, idx, "integrity_check_completed", integrity_payload
                    )
                    edit_accepted = edit_integrity.accepted
                    if not edit_accepted:
                        self.db.log_chunk_event(
                            job_id, idx, "integrity_edit_rejected", integrity_payload
                        )
                        if current_integrity_accepted:
                            (
                                salvaged_translation,
                                issue_decisions,
                                local_salvage,
                            ) = _salvage_local_refinement_edits(
                                source=chunk.text,
                                previous=before_translation,
                                proposed=proposed_translation,
                                issue_details=issue_details,
                                issue_decisions=issue_decisions,
                                integrity_gate=integrity_gate,
                                protected_terms=protected_targets,
                                protect_inline_english=protect_inline_english,
                                allowed_inline_originals=allowed_inline_originals,
                            )
                            if local_salvage["committed_count"]:
                                self.db.log_chunk_event(
                                    job_id,
                                    idx,
                                    "refinement_local_edits_recovered",
                                    {
                                        "iteration": ref_iter + 1,
                                        **local_salvage,
                                    },
                                )
                if edit_accepted:
                    issue_decisions = _refinement_decisions_with_commit_state(
                        issue_decisions,
                        issue_details,
                        proposed_translation,
                        candidate_accepted=True,
                    )
                convergence_reason = ""
                translation = (
                    proposed_translation
                    if edit_accepted
                    else salvaged_translation
                    if (local_salvage or {}).get("committed_count")
                    else before_translation
                )
                baseline_was_quarantined = not current_integrity_accepted
                if edit_accepted or (local_salvage or {}).get("committed_count"):
                    current_integrity_accepted = True
                if baseline_was_quarantined and current_integrity_accepted:
                    self.db.log_chunk_event(
                        job_id,
                        idx,
                        "translation_baseline_repaired",
                        {
                            "stage": "refinement",
                            "iteration": ref_iter + 1,
                            "translation_chars": len(translation),
                        },
                    )
                translation_changed = (
                    _normalized_translation_version(translation)
                    != _normalized_translation_version(before_translation)
                )
                if translation_changed:
                    proposed_key = _normalized_translation_version(translation)
                    current_key = _normalized_translation_version(before_translation)
                    prior_keys = [
                        _normalized_translation_version(value)
                        for value in accepted_versions
                    ]
                    if proposed_key == current_key:
                        convergence_reason = "refinement_no_change"
                    elif proposed_key in prior_keys[:-1]:
                        convergence_reason = "refinement_oscillation"

                self.db.save_issue_decisions(
                    job_id,
                    idx,
                    ref_iter + 1,
                    ref_iter,
                    issue_decisions,
                    candidate_accepted=edit_accepted,
                )
                if convergence_reason == "refinement_oscillation":
                    best_index, (translation, _) = max(
                        enumerate(evaluated_versions),
                        key=lambda item: (
                            _critique_candidate_rank(item[1][1]),
                            item[0],
                        ),
                    )
                    current_integrity_accepted = True
                    last_accepted_translation = translation
                    self.db.log_chunk_event(
                        job_id,
                        idx,
                        "refinement_convergence_stopped",
                        {
                            "reason": convergence_reason,
                            "iteration": ref_iter + 1,
                            "selected_evaluated_version": best_index,
                            "candidate_count": len(accepted_versions) + 1,
                            "message": (
                                "A previously accepted translation reappeared; "
                                "the best integrity-passing evaluated version was retained."
                            ),
                        },
                    )
                elif (
                    translation_changed
                    and not convergence_reason
                    and current_integrity_accepted
                ):
                    accepted_versions.append(translation)
                    last_accepted_translation = translation

                self.db.update_chunk(job_id, idx, ChunkStatus.REFINED, translation)
                self.db.log_chunk_event(job_id, idx, "refinement_completed", {
                    "iteration": ref_iter + 1,
                    "critique_average": critique_rep.average,
                    "critique_issue_count": len(critique_rep.issues),
                    "blocking_issue_count": len(_blocking_critique_issues(critique_rep)),
                    "decision": refinement.decision,
                    "rationale": _truncate_for_event(refinement.rationale, 1000),
                    "issue_decisions": issue_decisions,
                    "before_chars": before_chars,
                    "after_chars": len(translation),
                    "proposed_chars": len(proposed_translation),
                    "integrity_accepted": edit_accepted,
                    "candidate_integrity_accepted": edit_accepted,
                    "committed_edit_count": sum(
                        1 for decision in issue_decisions
                        if str(decision.get("commit_status", "")).startswith("committed")
                    ),
                    "commit_mode": (
                        "full_candidate" if edit_accepted
                        else "local_salvage"
                        if (local_salvage or {}).get("committed_count")
                        else "preserved"
                    ),
                    "local_salvage": local_salvage,
                    "integrity": integrity_payload,
                    "paragraphs_after": _paragraph_count(translation),
                    "convergence_reason": convergence_reason or None,
                })
                if convergence_reason:
                    self.db.log_chunk_event(
                        job_id,
                        idx,
                        "critique_needs_review",
                        {
                            "iteration": ref_iter + 1,
                            "critique_average": critique_rep.average,
                            "blocking_issue_count": len(
                                _blocking_critique_issues(critique_rep)
                            ),
                            "review_reason": convergence_reason,
                            "message": (
                                "Refinement stopped because another pass would "
                                "repeat or reverse an accepted edit."
                            ),
                        },
                    )
                    break
        else:
            self.db.log_chunk_event(job_id, idx, "critique_skipped", {"enabled": False})

        # Glossary Compliance
        if self.config.glossary.enable_compliance_check:
            report = compliance_checker.check(
                translation=translation,
                source_text=chunk.text,
                glossary_manager=glossary_manager,
                chunk_location=f"Chunk {idx}",
                entries=enforced_entries,
            )
            self.db.log_chunk_event(
                job_id,
                idx,
                "glossary_compliance_checked",
                _compliance_report_for_event(report),
            )
            if not report.compliant:
                enable_auto_correct = getattr(self.config.glossary, "enable_auto_correction", True)
                if enable_auto_correct:
                    attempts = 0
                    max_attempts = 2
                    correction_feedback = ""
                    while not report.compliant and attempts < max_attempts:
                        attempts += 1
                        violations_text = "\n".join(
                            f"- English: {v.term} -> expected Persian: {v.expected} (status: {v.status})"
                            for v in report.violations
                        )
                        allowed_originals_folded = {
                            value.casefold() for value in allowed_inline_originals
                        }
                        protected_originals = (
                            [
                                value
                                for value in protected_english_originals(
                                    chunk.text, translation
                                )
                                if value.casefold() in allowed_originals_folded
                            ]
                            if protect_inline_english else []
                        )
                        protected_originals_text = (
                            ", ".join(f"({value})" for value in protected_originals)
                            or "(none)"
                        )
                        correction_source = chunk.text
                        correction_translation = translation
                        correction_markers: list[str] = []
                        marked_source, source_markers = encode_paragraphs(
                            chunk.text
                        )
                        marked_translation, target_markers = encode_paragraphs(
                            translation
                        )
                        if len(source_markers) > 1 and source_markers == target_markers:
                            correction_source = marked_source
                            correction_translation = marked_translation
                            correction_markers = source_markers
                        correction_prompt = f"""\
The following translation violated the glossary compliance checks.

English Source:
{correction_source}

Current Translation:
{correction_translation}

Glossary violations found:
{violations_text}

Authorized first-occurrence English originals already present:
{protected_originals_text}

Do not add English parentheticals for any other term.

Please re-translate the text, ensuring that you use each required glossary term's
lexical rendering. Use standard Persian orthography and ZWNJ placement even if a
listed spacing variant is malformed; spacing-only equivalents remain compliant.
Preserve every protected English original above exactly once. Do not remove or relocate
those parentheticals while correcting glossary terminology. Preserve paragraph structure,
citations, numbers, names, and all text unrelated to the listed violations.
{correction_feedback}
Output ONLY the corrected Persian translation.
"""
                        correction_prompt += protocol_instruction(
                            correction_markers
                        )
                        before_correction = translation
                        try:
                            proposed_correction = self.llm_client.complete(
                                messages=[{"role": "user", "content": correction_prompt}],
                                system_prompt=sys_prompt,
                                _operation="glossary_auto_correction",
                            )
                        except _QUALITY_STAGE_ERRORS as exc:
                            self.db.log_chunk_event(
                                job_id,
                                idx,
                                "qa_unavailable",
                                _qa_provider_failure_payload(
                                    "glossary_auto_correction",
                                    "glossary_auto_correction",
                                    self.llm_client,
                                    exc,
                                ),
                            )
                            break
                        proposed_correction, orthography_edits = (
                            apply_safe_persian_orthography(proposed_correction)
                        )
                        correction_protocol_valid = True
                        correction_protocol_errors: list[str] = []
                        if correction_markers:
                            correction_protocol = decode_paragraphs(
                                proposed_correction, correction_markers
                            )
                            correction_protocol_valid = correction_protocol.valid
                            correction_protocol_errors = correction_protocol.errors
                            self.db.log_chunk_event(
                                job_id, idx, "paragraph_protocol_checked", {
                                    "stage": "glossary_auto_correction",
                                    "attempt": attempts,
                                    "valid": correction_protocol.valid,
                                    "expected_markers": correction_markers,
                                    "errors": correction_protocol.errors,
                                },
                            )
                            if correction_protocol.valid:
                                proposed_correction = correction_protocol.text
                        if orthography_edits:
                            self.db.log_chunk_event(
                                job_id, idx, "persian_orthography_normalized", {
                                    "stage": "glossary_auto_correction",
                                    "attempt": attempts,
                                    "edits": orthography_edits,
                                    "edit_count": sum(
                                        int(edit.get("count", 0))
                                        for edit in orthography_edits
                                    ),
                                },
                            )
                        correction_changed = (
                            (proposed_correction or "").strip()
                            != (before_correction or "").strip()
                        )
                        correction_accepted = correction_protocol_valid
                        correction_integrity: dict[str, Any] | None = None
                        if not correction_protocol_valid:
                            correction_feedback = (
                                "The previous candidate changed paragraph labels: "
                                + ", ".join(correction_protocol_errors)
                                + ". Preserve every required label exactly."
                            )
                            self.db.log_chunk_event(
                                job_id, idx, "integrity_edit_rejected", {
                                    "stage": "glossary_paragraph_protocol",
                                    "accepted": False,
                                    "blocking_count": 1,
                                    "findings": [{
                                        "check_id": "paragraph_identity_changed",
                                        "severity": "blocking",
                                        "message": (
                                            "Glossary correction changed stable "
                                            "paragraph identity."
                                        ),
                                        "errors": correction_protocol_errors,
                                    }],
                                },
                            )
                        elif integrity_enabled:
                            correction_result = integrity_gate.evaluate(
                                chunk.text,
                                proposed_correction,
                                previous=before_correction,
                                stage="glossary_auto_correction",
                                protected_terms=protected_targets,
                                protect_inline_english=protect_inline_english,
                                allowed_inline_originals=allowed_inline_originals,
                                enforce_all_terms=True,
                            )
                            correction_integrity = correction_result.to_dict()
                            self.db.log_chunk_event(
                                job_id, idx, "integrity_check_completed",
                                correction_integrity,
                            )
                            correction_accepted = correction_result.accepted
                            if not correction_accepted:
                                self.db.log_chunk_event(
                                    job_id, idx, "integrity_edit_rejected",
                                    correction_integrity,
                                )
                                blocking_checks = [
                                    finding.get("check_id", "")
                                    for finding in correction_integrity.get("findings", [])
                                    if finding.get("severity") == "blocking"
                                ]
                                correction_feedback = (
                                    "The previous correction candidate was rejected by "
                                    "deterministic integrity checks: "
                                    + ", ".join(blocking_checks)
                                    + ". Produce a minimally edited correction that "
                                      "retains all protected content."
                                )
                        if correction_accepted:
                            baseline_was_quarantined = (
                                last_accepted_translation is None
                            )
                            translation = proposed_correction
                            last_accepted_translation = translation
                            if baseline_was_quarantined:
                                self.db.log_chunk_event(
                                    job_id,
                                    idx,
                                    "translation_baseline_repaired",
                                    {
                                        "stage": "glossary_auto_correction",
                                        "attempt": attempts,
                                        "translation_chars": len(translation),
                                    },
                                )
                            report = compliance_checker.check(
                                translation=translation,
                                source_text=chunk.text,
                                glossary_manager=glossary_manager,
                                chunk_location=f"Chunk {idx}",
                                entries=enforced_entries,
                            )
                            if not report.compliant:
                                remaining_terms = "; ".join(
                                    f"{v.term} -> {v.expected}"
                                    for v in report.violations
                                )
                                no_change = (
                                    "The previous correction made no textual change. "
                                    if not correction_changed else ""
                                )
                                correction_feedback = (
                                    no_change
                                    + "The candidate passed integrity, but these exact "
                                      "glossary requirements remain unmet: "
                                    + remaining_terms
                                    + ". Make only the minimum necessary edits and include "
                                      "each expected Persian form exactly."
                                )
                        self.db.log_chunk_event(job_id, idx, "glossary_auto_correct_attempt", {
                            "attempt": attempts,
                            "translation_chars": len(translation),
                            "proposed_chars": len(proposed_correction or ""),
                            "changed": correction_changed,
                            "integrity_accepted": correction_accepted,
                            "integrity": correction_integrity,
                            **_compliance_report_for_event(report),
                        })

                if not report.compliant:
                    for v in report.violations:
                        warn_msg = f"Glossary violation: Term '{v.term}' expected '{v.expected}'"
                        logger.warning("%s in %s", warn_msg, v.chunk_location)
                        if lock:
                            with lock:
                                self.warnings.append(f"{v.chunk_location}: {warn_msg}")
                        else:
                            self.warnings.append(f"{v.chunk_location}: {warn_msg}")
                    review_payload = _compliance_report_for_event(report)
                    review_payload["message"] = (
                        "Mandatory glossary requirements remained unmet after "
                        "the configured compliance and correction checks."
                    )
                    self.db.log_chunk_event(
                        job_id,
                        idx,
                        "glossary_needs_review",
                        review_payload,
                    )
            self.db.log_chunk_event(
                job_id,
                idx,
                "glossary_compliance_final",
                _compliance_report_for_event(report),
            )
        else:
            self.db.log_chunk_event(job_id, idx, "glossary_compliance_skipped", {"enabled": False})

        if integrity_enabled:
            translation, identifier_repairs = restore_source_identifiers(
                chunk.text, translation
            )
            if identifier_repairs["repair_count"]:
                self.db.log_chunk_event(
                    job_id, idx, "source_identifiers_restored", {
                        "stage": "final_translation",
                        **identifier_repairs,
                    },
                )
            final_integrity = integrity_gate.evaluate(
                chunk.text,
                translation,
                # Supplying `previous` is what lets the gate tell damage this
                # last step introduced from damage carried in from earlier. The
                # corruption and duplicate checks both baseline against it.
                previous=last_accepted_translation or "",
                stage="final_translation",
                protected_terms=protected_targets,
                protect_inline_english=protect_inline_english,
                allowed_inline_originals=allowed_inline_originals,
                enforce_all_terms=bool(self.config.glossary.enable_compliance_check),
            )
            final_integrity_payload = final_integrity.to_dict()
            self.db.log_chunk_event(
                job_id, idx, "integrity_check_completed", final_integrity_payload
            )
            if not final_integrity.accepted:
                self.db.log_chunk_event(
                    job_id, idx, "integrity_final_failed", final_integrity_payload
                )
                # Rejection now restores the last text that passed a gate,
                # instead of returning the rejected text. Previously the final
                # gate could only log, so damage introduced after the last
                # accepted version was still what shipped.
                restored = (last_accepted_translation or "").strip()
                restored_is_valid = False
                if restored and restored != (translation or "").strip():
                    restored_integrity = integrity_gate.evaluate(
                        chunk.text,
                        restored,
                        stage="final_translation_restored",
                        protected_terms=protected_targets,
                        protect_inline_english=protect_inline_english,
                        allowed_inline_originals=allowed_inline_originals,
                        enforce_all_terms=bool(
                            self.config.glossary.enable_compliance_check
                        ),
                    )
                    self.db.log_chunk_event(
                        job_id,
                        idx,
                        "integrity_check_completed",
                        restored_integrity.to_dict(),
                    )
                    restored_is_valid = restored_integrity.accepted
                    if restored_is_valid:
                        self.db.log_chunk_event(
                            job_id, idx, "integrity_final_restored", {
                                "restored_from": "last_accepted_translation",
                                "rejected_chars": len(translation or ""),
                                "restored_chars": len(restored),
                            },
                        )
                        translation = restored
                _ensure_chunk_review_reason(self.db, job_id, idx)
                if not restored_is_valid:
                    raise ValueError(
                        "No integrity-valid translation remained after bounded "
                        "critique, refinement, and glossary repair."
                    )
            else:
                last_accepted_translation = translation

        if protect_inline_english:
            grounded_categories = _accepted_grounded_inline_originals(
                chunk.text, translation
            )
            if _is_front_matter(chunk):
                grounded_categories = {
                    source: category
                    for source, category in grounded_categories.items()
                    if category in {"person", "publication", "legal_instrument"}
                }
            for source, category in grounded_categories.items():
                if source.casefold() not in {
                    value.casefold() for value in source_entity_candidates
                }:
                    source_entity_candidates.append(source)
                source_entity_categories[source] = category

        # Canonicalize the accepted text before measuring required-anchor
        # coverage. This pass may remove an unauthorized parenthetical, so an
        # earlier coverage result would describe text that never reaches memory
        # or export.
        inline_reconciliation = {
            "stage": "pre_memory_inline_reconciliation",
            "valid": True,
            "changed": False,
            "reason": "policy_disabled",
        }
        if protect_inline_english:
            if lock:
                with lock:
                    translation, inline_reconciliation = (
                        _canonicalize_chunk_inline_originals(
                            memory_manager, chunk, translation
                        )
                    )
            else:
                translation, inline_reconciliation = (
                    _canonicalize_chunk_inline_originals(
                        memory_manager, chunk, translation
                    )
                )
        self.db.log_chunk_event(
            job_id,
            idx,
            "memory_export_consistency",
            inline_reconciliation,
        )

        entity_coverage: dict[str, Any] = {
            "candidate_count": len(source_entity_candidates),
            "observed_count": 0,
            "observed": [],
            "missing": [],
            "policy_enabled": protect_inline_english,
        }
        if protect_inline_english and source_entity_candidates:
            if lock:
                with lock:
                    entity_coverage.update(_reconcile_current_entity_anchors(
                        memory_manager,
                        source_entity_candidates,
                        translation,
                        source_entity_categories,
                        chunk.text,
                    ))
            else:
                entity_coverage.update(_reconcile_current_entity_anchors(
                    memory_manager,
                    source_entity_candidates,
                    translation,
                    source_entity_categories,
                    chunk.text,
                ))
        missing_required_anchors = [
            source for source in required_entity_candidates
            if not has_exact_observed_anchor(translation, source)
        ]
        anchor_repair_event: dict[str, Any] = {
            "attempted": False,
            "accepted": False,
            "missing": missing_required_anchors,
            "policy": "exact_parenthetical_insertions_only",
        }
        if protect_inline_english and missing_required_anchors:
            anchor_repair_event["attempted"] = True
            try:
                self.llm_client.limit_next_call_attempts(2)
                anchor_candidate = self.llm_client.complete(
                    messages=[{
                        "role": "user",
                        "content": _required_anchor_repair_prompt(
                            chunk.text,
                            translation,
                            missing_required_anchors,
                        ),
                    }],
                    system_prompt=sys_prompt,
                    _operation="translation_entity_anchor_repair",
                    _recovery_source_text=chunk.text,
                ).strip()
                insertion_only = _anchor_only_repair_is_valid(
                    translation,
                    anchor_candidate,
                    missing_required_anchors,
                    source_entity_categories,
                )
                anchor_integrity = integrity_gate.evaluate(
                    chunk.text,
                    anchor_candidate,
                    previous=translation,
                    stage="entity_anchor_repair",
                    protected_terms=protected_targets,
                    protect_inline_english=protect_inline_english,
                    allowed_inline_originals=allowed_inline_originals,
                    enforce_all_terms=bool(
                        self.config.glossary.enable_compliance_check
                    ),
                )
                self.db.log_chunk_event(
                    job_id,
                    idx,
                    "integrity_check_completed",
                    anchor_integrity.to_dict(),
                )
                anchor_repair_event.update({
                    "insertion_only": insertion_only,
                    "integrity_accepted": anchor_integrity.accepted,
                    "candidate_chars": len(anchor_candidate),
                })
                if insertion_only and anchor_integrity.accepted:
                    translation = anchor_candidate
                    last_accepted_translation = translation
                    anchor_repair_event["accepted"] = True
                    if lock:
                        with lock:
                            entity_coverage.update(
                                _reconcile_current_entity_anchors(
                                    memory_manager,
                                    source_entity_candidates,
                                    translation,
                                    source_entity_categories,
                                    chunk.text,
                                )
                            )
                    else:
                        entity_coverage.update(_reconcile_current_entity_anchors(
                            memory_manager,
                            source_entity_candidates,
                            translation,
                            source_entity_categories,
                            chunk.text,
                        ))
            except _QUALITY_STAGE_ERRORS as exc:
                anchor_repair_event.update({
                    "failure_type": type(exc).__name__,
                    "error": str(exc),
                })
            except Exception as exc:
                logger.exception("Entity-anchor repair failed for chunk %s", idx)
                anchor_repair_event.update({
                    "failure_type": type(exc).__name__,
                    "error": str(exc),
                })
        self.db.log_chunk_event(
            job_id, idx, "entity_anchor_repair", anchor_repair_event
        )
        if protect_inline_english:
            if lock:
                with lock:
                    memory_manager.proper_nouns.mark_introduced_from_translation(
                        chunk.text, translation
                    )
            else:
                memory_manager.proper_nouns.mark_introduced_from_translation(
                    chunk.text, translation
                )
        if protect_inline_english and entity_coverage.get("missing"):
            # The deterministic reconciliation above may have supplied an
            # established first-occurrence anchor that the model omitted. Judge
            # coverage from the canonical text that enters memory and export.
            entity_coverage["missing"] = [
                source
                for source in entity_coverage.get("missing", [])
                if not has_exact_observed_anchor(translation, str(source))
            ]
        entity_coverage["required_person_candidates"] = required_person_candidates
        entity_coverage["required_instrument_candidates"] = (
            required_instrument_candidates
        )
        entity_coverage["required_entity_candidates"] = required_entity_candidates
        missing_entities = entity_coverage.get("missing", [])
        if not isinstance(missing_entities, (list, tuple, set)):
            missing_entities = []
        entity_coverage["required_missing"] = [
            source for source in missing_entities
            if source in required_entity_candidates
        ]
        self.db.log_chunk_event(
            job_id, idx, "source_entity_coverage", entity_coverage
        )
        if protect_inline_english and entity_coverage.get("required_missing"):
            self.db.log_chunk_event(
                job_id,
                idx,
                "chunk_review_required",
                _explicit_chunk_review_payload(
                    "missing_first_occurrence_entity_original",
                    detail="source_entity_coverage",
                    missing=entity_coverage["required_missing"],
                    message=(
                        "One or more high-confidence current-source entities did "
                        "not produce an unambiguous Persian (English) anchor. The "
                        "translation was retained without guessing an insertion."
                    ),
                ),
            )

        language_candidate, safe_language_repair = (
            repair_source_grounded_language_artifacts(chunk.text, translation)
        )
        safe_language_repair["accepted"] = False
        if language_candidate != translation:
            language_integrity = integrity_gate.evaluate(
                chunk.text,
                language_candidate,
                previous=translation,
                stage="source_grounded_language_repair",
                protected_terms=protected_targets,
                protect_inline_english=protect_inline_english,
                allowed_inline_originals=allowed_inline_originals,
                enforce_all_terms=bool(
                    self.config.glossary.enable_compliance_check
                ),
            )
            self.db.log_chunk_event(
                job_id,
                idx,
                "integrity_check_completed",
                language_integrity.to_dict(),
            )
            safe_language_repair["integrity_accepted"] = (
                language_integrity.accepted
            )
            if language_integrity.accepted:
                translation = language_candidate
                last_accepted_translation = translation
                safe_language_repair["accepted"] = True
        self.db.log_chunk_event(
            job_id,
            idx,
            "source_grounded_language_repair",
            safe_language_repair,
        )

        # Audit the text that will actually continue into memory and export,
        # after every refinement, correction and integrity restoration. This is
        # report-only: actionable findings mark review and lower memory trust,
        # but never rewrite the author's source structure automatically.
        try:
            structure_payload = {
                "stage": "final_translation",
                **audit_payload(chunk.text, translation),
            }
        except Exception:
            logger.exception("Structure audit failed for chunk %s", idx)
            self.db.log_chunk_event(
                job_id, idx, "structure_audit_failed", {"stage": "final_translation"}
            )
            structure_payload = {"finding_count": 0, "classifications": []}
        if structure_payload["finding_count"]:
            self.db.log_chunk_event(job_id, idx, "structure_audit", structure_payload)
            if {
                "translation_structure_mismatch",
                "unauthorized_source_correction",
            }.intersection(structure_payload.get("classifications", [])):
                _ensure_chunk_review_reason(self.db, job_id, idx)

        structural_roles = {
            str(role).casefold()
            for role in list(chunk.metadata.get("structural_roles", []) or [])
            if str(role).strip()
        }
        language_role = (
            next(iter(structural_roles))
            if len(structural_roles) == 1 and "body" not in structural_roles
            else "body"
        )
        allowed_language_originals = tuple(
            set(memory_manager.proper_nouns.inline_eligible_nouns()).union(
                source_entity_candidates
            )
        )
        language_quality = audit_translation_language(
            chunk.text,
            translation,
            allowed_originals=allowed_language_originals,
            structural_role=language_role,
            chapter_title=chunk.chapter_title,
        )
        repairable_language_finding = any(
            int(language_quality.get(key, 0) or 0)
            for key in (
                "mixed_script_count",
                "unexpected_latin_count",
                "foreign_script_count",
                "markup_wrapper_count",
                "parenthesis_artifact_count",
                "detached_ezafe_count",
            )
        )
        targeted_language_repair: dict[str, Any] = {
            "attempted": False,
            "accepted_count": 0,
            "paragraphs": [],
            "policy": (
                "one bounded paragraph-local target-only repair; unaffected "
                "paragraphs remain byte-identical and deterministic integrity "
                "must pass"
            ),
        }
        if repairable_language_finding:
            source_parts = split_paragraphs(chunk.text)
            target_parts = split_paragraphs(translation)
            targeted_language_repair["attempted"] = True
            targeted_language_repair["source_paragraphs"] = len(source_parts)
            targeted_language_repair["target_paragraphs"] = len(target_parts)
            if len(source_parts) == len(target_parts):
                repaired_parts = list(target_parts)
                for paragraph_index, (source_part, target_part) in enumerate(
                    zip(source_parts, target_parts, strict=True)
                ):
                    paragraph_quality = audit_translation_language(
                        source_part,
                        target_part,
                        allowed_originals=allowed_language_originals,
                        structural_role=language_role,
                        chapter_title=chunk.chapter_title,
                    )
                    paragraph_repairable = any(
                        int(paragraph_quality.get(key, 0) or 0)
                        for key in (
                            "mixed_script_count",
                            "unexpected_latin_count",
                            "foreign_script_count",
                            "markup_wrapper_count",
                            "parenthesis_artifact_count",
                            "detached_ezafe_count",
                        )
                    )
                    if not paragraph_repairable:
                        continue
                    paragraph_event: dict[str, Any] = {
                        "paragraph_index": paragraph_index,
                        "accepted": False,
                    }
                    try:
                        self.llm_client.limit_next_call_attempts(1)
                        candidate_part = self.llm_client.complete(
                            messages=[{
                                "role": "user",
                                "content": _targeted_language_repair_prompt(
                                    source_part,
                                    target_part,
                                    paragraph_quality,
                                    structural_role=language_role,
                                ),
                            }],
                            system_prompt=sys_prompt,
                            _operation="translation_language_repair",
                            _recovery_source_text=source_part,
                        ).strip()
                        candidate_quality = audit_translation_language(
                            source_part,
                            candidate_part,
                            allowed_originals=allowed_language_originals,
                            structural_role=language_role,
                            chapter_title=chunk.chapter_title,
                        )
                        candidate_parts = list(repaired_parts)
                        candidate_parts[paragraph_index] = candidate_part
                        candidate_translation = "\n\n".join(candidate_parts)
                        candidate_integrity = integrity_gate.evaluate(
                            chunk.text,
                            candidate_translation,
                            previous="\n\n".join(repaired_parts),
                            stage="targeted_language_repair",
                            protected_terms=protected_targets,
                            protect_inline_english=protect_inline_english,
                            allowed_inline_originals=allowed_inline_originals,
                            enforce_all_terms=bool(
                                self.config.glossary.enable_compliance_check
                            ),
                        )
                        self.db.log_chunk_event(
                            job_id,
                            idx,
                            "integrity_check_completed",
                            candidate_integrity.to_dict(),
                        )
                        improved = _language_quality_strictly_improves(
                            paragraph_quality, candidate_quality
                        )
                        paragraph_event.update({
                            "candidate_chars": len(candidate_part),
                            "integrity_accepted": candidate_integrity.accepted,
                            "strictly_improved": improved,
                            "local_edit": _language_repair_is_local(
                                target_part,
                                candidate_part,
                                structural_role=language_role,
                            ),
                        })
                        if (
                            candidate_part
                            and len(split_paragraphs(candidate_part)) == 1
                            and candidate_integrity.accepted
                            and improved
                            and paragraph_event["local_edit"]
                        ):
                            repaired_parts[paragraph_index] = candidate_part
                            paragraph_event["accepted"] = True
                            targeted_language_repair["accepted_count"] += 1
                    except _QUALITY_STAGE_ERRORS as exc:
                        paragraph_event.update({
                            "failure_type": type(exc).__name__,
                            "error": str(exc),
                        })
                    except Exception as exc:
                        logger.exception(
                            "Targeted language repair failed for chunk %s paragraph %s",
                            idx,
                            paragraph_index,
                        )
                        paragraph_event.update({
                            "failure_type": type(exc).__name__,
                            "error": str(exc),
                        })
                    targeted_language_repair["paragraphs"].append(paragraph_event)
                if targeted_language_repair["accepted_count"]:
                    translation = "\n\n".join(repaired_parts)
                    last_accepted_translation = translation
                    language_quality = audit_translation_language(
                        chunk.text,
                        translation,
                        allowed_originals=allowed_language_originals,
                        structural_role=language_role,
                        chapter_title=chunk.chapter_title,
                    )
            else:
                targeted_language_repair["reason"] = "paragraph_count_mismatch"
        self.db.log_chunk_event(
            job_id,
            idx,
            "targeted_language_repair",
            targeted_language_repair,
        )
        self.db.log_chunk_event(
            job_id, idx, "language_quality_checked", language_quality
        )
        if language_quality["review_required"]:
            self.db.log_chunk_event(
                job_id,
                idx,
                "language_quality_review",
                {
                    **language_quality,
                    "review_reason": "objective_final_language_artifact",
                    "message": (
                        "Objective language artifacts remained after deterministic "
                        "and bounded local repair. The text was retained for review "
                        "and was not admitted to trusted retrieval or style memory."
                    ),
                },
            )

        # Back translation verification
        if self.config.translation.enable_back_translation:
            if back_translator.should_sample():
                self.db.log_chunk_event(job_id, idx, "back_translation_sampled", {"sampled": True})
                try:
                    back_translated = self._run_async(
                        back_translator.back_translate(translation)
                    )
                except _QUALITY_STAGE_ERRORS as exc:
                    self.db.log_chunk_event(
                        job_id,
                        idx,
                        "qa_unavailable",
                        _qa_provider_failure_payload(
                            "back_translation",
                            "back_translation",
                            critic_client,
                            exc,
                        ),
                    )
                    back_translated = ""
                bt_result = (
                    back_translator.compare(
                        chunk.text,
                        back_translated,
                        translation,
                        entity_aliases={
                            source: [target]
                            + memory_manager.proper_nouns.aliases_for(source)
                            for source, target in memory_manager.proper_nouns
                            .inline_eligible_nouns().items()
                        },
                    )
                    if back_translated else None
                )
                if bt_result is not None:
                    self.db.log_chunk_event(job_id, idx, "back_translation_completed", {
                        "similarity_score": bt_result.similarity_score,
                        "flagged": bt_result.flagged,
                        "difference_count": len(bt_result.differences),
                        "differences_preview": bt_result.differences[:50],
                        "back_translated_preview": _truncate_for_event(bt_result.back_translated, 2000),
                        "diagnostics": bt_result.diagnostics,
                    })
                if bt_result is not None and bt_result.flagged:
                    self.db.log_chunk_event(job_id, idx, "back_translation_flagged", {
                        "risk_flags": bt_result.diagnostics.get("risk_flags", []),
                        "diagnostics": bt_result.diagnostics,
                        "message": (
                            "Structured back-translation diagnostics require human review; "
                            "the Persian translation was not changed automatically."
                        ),
                    })
                    self.db.log_event(
                        job_id,
                        "WARNING",
                        f"Back-translation flagged for Chunk {idx}: "
                        f"{bt_result.diagnostics.get('risk_flags', [])}",
                    )
            else:
                self.db.log_chunk_event(job_id, idx, "back_translation_skipped", {
                    "enabled": True,
                    "sampled": False,
                    "sample_pct": self.config.translation.back_translation_sample_pct,
                })
        else:
            self.db.log_chunk_event(job_id, idx, "back_translation_skipped", {"enabled": False})

        if lock:
            with lock:
                reconciliation_report = _reconcile_committed_terminology(
                    self.db,
                    job_id,
                    idx,
                    memory_manager,
                    chunk.text,
                    translation,
                )
        else:
            reconciliation_report = _reconcile_committed_terminology(
                self.db,
                job_id,
                idx,
                memory_manager,
                chunk.text,
                translation,
            )
        if (
            reconciliation_report.get("reconciled")
            or reconciliation_report.get("aliases_added")
            or reconciliation_report.get("skipped")
        ):
            self.db.log_chunk_event(
                job_id,
                idx,
                "accepted_terminology_reconciled",
                reconciliation_report,
            )
        summary_reconciliation = memory_manager.reconcile_bilingual_summary()
        if summary_reconciliation.get("replacement_count"):
            self.db.log_chunk_event(
                job_id,
                idx,
                "bilingual_summary_reconciled",
                summary_reconciliation,
            )

        consistency_report = _advisory_terminology_consistency(
            memory_manager, chunk.text, translation
        )
        self.db.log_chunk_event(
            job_id,
            idx,
            "terminology_consistency_advisory",
            consistency_report,
        )

        self.db.log_chunk_event(job_id, idx, "chunk_completed", {
            "final_translation_chars": len(translation),
            "final_translation_paragraphs": _paragraph_count(translation),
        })

        return translation
