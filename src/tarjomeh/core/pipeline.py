"""Translation pipeline orchestrator for the Tarjomeh translation system.

Orchestrates ingestion, chunking, translation memory context construction,
web search, translation, critique/refinement, and final output exporting.
"""

from __future__ import annotations

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
from tarjomeh.memory.proper_nouns import INLINE_ORIGINAL_CATEGORIES
from tarjomeh.context.web_searcher import WebContextSearcher
from tarjomeh.glossary.manager import GlossaryManager
from tarjomeh.glossary.compliance import GlossaryComplianceChecker
from tarjomeh.persian.typography import PersianTypographer
from tarjomeh.core.term_notes import (
    apply_term_notes,
    audit_inline_english_originals,
    effective_term_notes_mode,
    ensure_inline_proper_noun_originals,
)
from tarjomeh.exporters import get_exporter
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.jobs.database import JobDatabase, JobStatus, ChunkStatus
from tarjomeh.quality.critique import TranslationCritique
from tarjomeh.quality.refiner import TranslationRefiner
from tarjomeh.quality.back_translator import BackTranslator
from tarjomeh.quality.integrity import (
    PostEditIntegrityGate,
    protected_english_originals,
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


def _paragraph_count(text: str) -> int:
    return len([p for p in (text or "").split("\n\n") if p.strip()])


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
        if removed_authorized or added_unauthorized or removed_citations:
            conflicts.append({
                "issue": detail.get("formatted", ""),
                "removed_authorized": removed_authorized,
                "added_unauthorized": added_unauthorized,
                "removed_citations": removed_citations,
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
    """Average score must pass and no major conceptual/terminology issue may remain."""
    return bool(
        getattr(critique, "valid", True)
        and critique.passes_threshold(threshold)
        and not _blocking_critique_issues(critique)
    )


def _critique_for_event(critique: Any, threshold: float, iteration: int) -> dict[str, Any]:
    blocking_issues = _blocking_critique_issues(critique)
    return {
        "iteration": iteration,
        "valid": bool(getattr(critique, "valid", True)),
        "attempts": int(getattr(critique, "attempts", 1)),
        "validation_errors": list(getattr(critique, "validation_errors", []) or []),
        "threshold": threshold,
        "passes_average_threshold": bool(critique.passes_threshold(threshold)),
        "passes_threshold": bool(critique.passes_threshold(threshold) and not blocking_issues),
        "force_refinement": bool(blocking_issues),
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
        "raw_response_preview": _truncate_for_event(critique.raw_response, 3000),
    }


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
        "back_translation_flagged",
        "glossary_needs_review",
    }
    return any(event.get("event_type") in review_events for event in events[last_start:])


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
        "violation_count": len(violations),
        "violations": violations,
    }


def _align_chunk_translation(
    *,
    original_paragraphs: list[Any],
    para_indices: list[int],
    tgt_paras: list[str],
    chunk_translation: str,
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
        # Critic recovery is independent from translator recovery. Its normal
        # request is unchanged; only bounded follow-up attempts use these.
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
        """Run an async coroutine synchronously, managing event loops properly."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        else:
            return loop.run_until_complete(coro)

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
            "completed", "completed_without_suggestions", "degraded"
        }:
            status = artifact.get("status")
            self.db.log_event(
                job_id,
                "WARNING" if status == "degraded" else "INFO",
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

        # Run OCR preprocessing first if enabled
        if input_path.suffix.lower() == ".pdf" and getattr(self.config.pdf, "enable_ocr", False):
            if progress_callback:
                progress_callback("Ingestion", 0.02, "Running OCR Preprocessing...")
            from tarjomeh.parsers.ocr_preprocessor import OCRPreprocessor
            ocr_processor = OCRPreprocessor()
            input_path = ocr_processor.preprocess(input_path)

        # 1. Establish Job ID and Database Record
        is_resume = False
        if job_id:
            self.current_job_id = job_id
            job_record = self.db.get_job(job_id)
            if job_record:
                is_resume = True
                logger.info("Resuming existing job: %s", job_id)
                self.db.update_job_status(job_id, JobStatus.RUNNING)
            else:
                logger.info("Starting new job with supplied ID: %s", job_id)
                self.db.create_job(job_id, input_path, self.config.to_dict())
        else:
            self.current_job_id = uuid.uuid4().hex[:12]
            self.db.create_job(self.current_job_id, input_path, self.config.to_dict())

        job_id = self.current_job_id

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
        else:
            # Fresh parse and chunk
            parser = self._get_parser(input_path)
            document = parser.parse(input_path)
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
                ner_data = json.loads(ner_response)
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
                            term, persian, category=category
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
                    entry.source, entry.target, category="approved_term"
                )

        research_artifact = self._prepare_book_research(
            job_id,
            research_document,
            progress_callback,
        )
        if research_artifact and research_artifact.get("status") in {
            "completed", "completed_without_suggestions", "degraded"
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
                    )

                    # Update shared memory and database safely under lock
                    with lock:
                        translations[idx] = translation
                        memory_manager.update_after_translation(chunk, translation)
                        
                        if self.config.memory.enable_4layer:
                            try:
                                self._run_async(memory_manager.update_proper_nouns(self.llm_client, chunk.text))
                                memory_manager.proper_nouns.mark_seen_in_text(chunk.text)
                            except Exception as e:
                                logger.warning("Incremental proper noun extraction failed: %s", e)

                            is_chapter_end = False
                            if idx == total_chunks - 1:
                                is_chapter_end = True
                            else:
                                next_chunk = chunks[idx + 1]
                                if next_chunk.chapter_title != chunk.chapter_title:
                                    is_chapter_end = True

                            if is_chapter_end:
                                chap_source = []
                                chap_trans = []
                                for i in range(idx + 1):
                                    c = chunks[i]
                                    if c.chapter_title == chunk.chapter_title:
                                        chap_source.append(c.text)
                                        chap_trans.append(translations.get(i, ""))
                                
                                new_content = "\n\n".join(chap_source)
                                chap_translation = "\n\n".join(chap_trans)
                                try:
                                    self._run_async(memory_manager.update_bilingual_summary(
                                        self.llm_client,
                                        new_content=new_content,
                                        translation=chap_translation
                                    ))
                                except Exception as e:
                                    logger.warning("Bilingual summary update failed: %s", e)

                        final_status = (
                            ChunkStatus.NEEDS_REVIEW
                            if _chunk_needs_review(self.db, job_id, idx)
                            else ChunkStatus.COMPLETED
                        )
                        self.db.update_chunk(job_id, idx, final_status, translation)
                        self.db.save_memory_state(job_id, memory_manager.to_dict())
                        self.db.save_job_artifact(
                            job_id,
                            "web_search_state",
                            web_searcher.export_state(),
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
                        )

                        # Successful translation updates
                        translations[idx] = translation
                        consecutive_errors = 0

                        memory_manager.update_after_translation(chunk, translation)

                        if self.config.memory.enable_4layer:
                            try:
                                self._run_async(memory_manager.update_proper_nouns(self.llm_client, chunk.text))
                                memory_manager.proper_nouns.mark_seen_in_text(chunk.text)
                            except Exception as e:
                                logger.warning("Incremental proper noun extraction failed: %s", e)

                            is_chapter_end = False
                            if idx == total_chunks - 1:
                                is_chapter_end = True
                            else:
                                next_chunk = chunks[idx + 1]
                                if next_chunk.chapter_title != chunk.chapter_title:
                                    is_chapter_end = True

                            if is_chapter_end:
                                chap_source = []
                                chap_trans = []
                                for i in range(idx + 1):
                                    c = chunks[i]
                                    if c.chapter_title == chunk.chapter_title:
                                        chap_source.append(c.text)
                                        chap_trans.append(translations.get(i, ""))
                                
                                new_content = "\n\n".join(chap_source)
                                chap_translation = "\n\n".join(chap_trans)
                                try:
                                    self._run_async(memory_manager.update_bilingual_summary(
                                        self.llm_client,
                                        new_content=new_content,
                                        translation=chap_translation
                                    ))
                                except Exception as e:
                                    logger.warning("Bilingual summary update failed: %s", e)

                        final_status = (
                            ChunkStatus.NEEDS_REVIEW
                            if _chunk_needs_review(self.db, job_id, idx)
                            else ChunkStatus.COMPLETED
                        )
                        self.db.update_chunk(job_id, idx, final_status, translation)
                        self.db.save_memory_state(job_id, memory_manager.to_dict())
                        self.db.save_job_artifact(
                            job_id,
                            "web_search_state",
                            web_searcher.export_state(),
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
            para_indices = chunk.metadata.get("paragraph_indices", [])

            if para_indices:
                aligned = _align_chunk_translation(
                    original_paragraphs=original_paragraphs,
                    para_indices=para_indices,
                    tgt_paras=tgt_paras,
                    chunk_translation=chunk_translation,
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
            metadata=document.metadata,
        )

        # 8. Persian Typography Post-Processing
        if progress_callback:
            progress_callback("Typography", 0.95, "Applying Persian typography rules...")

        typographer = PersianTypographer(self.config.to_dict().get("persian"))
        for p in trans_doc.paragraphs:
            p.translated_text = typographer.process(p.translated_text)

        requested_note_mode = self.config.output.term_notes
        note_mode = effective_term_notes_mode(
            requested_note_mode, self.config.output.format
        )
        note_formats = {"docx", "epub", "markdown"}
        noun_state = memory_manager.proper_nouns.serialize()
        proper_nouns = memory_manager.proper_nouns.inline_eligible_nouns()
        if note_mode in {"inline", "both"}:
            restored = ensure_inline_proper_noun_originals(
                trans_doc, proper_nouns, typographer
            )
            if restored:
                self.db.log_event(
                    job_id, "INFO",
                    f"Restored {restored} first-occurrence English original(s).",
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
        if note_mode != "inline" and self.config.output.format in note_formats:
            notes = apply_term_notes(
                trans_doc,
                glossary_manager,
                proper_nouns,
                typographer,
                domain=self.config.translation.domain,
                mode=note_mode,
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

        document, chunks = self._parse_and_chunk(
            input_path, chapter_positions=chapter_positions
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
        note_mode = effective_term_notes_mode(self.config.output.term_notes, fmt)
        memory_state = self.db.get_memory_state(job_id) or {}
        noun_state = memory_state.get("proper_nouns", {})
        proper_nouns = _inline_eligible_nouns_from_state(noun_state)

        typographer = PersianTypographer(self.config.to_dict().get("persian"))
        if note_mode in {"inline", "both"}:
            ensure_inline_proper_noun_originals(
                trans_doc, dict(proper_nouns), typographer
            )
        original_audit = audit_inline_english_originals(
            trans_doc,
            dict(proper_nouns) if note_mode in {"inline", "both"} else {},
        )
        self.db.save_job_artifact(
            job_id, "english_original_audit", original_audit
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

        document, chunks = self._parse_and_chunk(Path(job["input_path"]))
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
                    entry.source, entry.target, category="approved_term"
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
        final_status = (
            ChunkStatus.NEEDS_REVIEW
            if _chunk_needs_review(self.db, job_id, chunk_index)
            else ChunkStatus.COMPLETED
        )
        self.db.update_chunk(job_id, chunk_index, final_status, translation)
        memory_manager.update_after_translation(chunks[chunk_index], translation)
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
    ) -> tuple[Document, list[Chunk]]:
        parser = self._get_parser(input_path)
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
            para_indices = chunk.metadata.get("paragraph_indices", [])

            if para_indices:
                aligned = _align_chunk_translation(
                    original_paragraphs=original_paragraphs,
                    para_indices=para_indices,
                    tgt_paras=tgt_paras,
                    chunk_translation=chunk_translation,
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
            metadata=document.metadata,
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
        self.db.log_chunk_event(job_id, idx, "glossary_matches", {
            "matched_count": len(matched_entries),
            "entries": _glossary_entries_for_event(matched_entries),
        })
        protected_targets = [
            str(getattr(entry, "target", "")).strip()
            for entry in matched_entries
            if str(getattr(entry, "target", "")).strip()
            and not bool(getattr(entry, "is_auto", False))
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
        glossary_terms_str = glossary_manager.format_for_prompt(matched_entries) \
            or "(no glossary terms matched in this chunk)"

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
        allowed_inline_originals = (
            sorted(pending_originals, key=str.casefold)
            if protect_inline_english else []
        )
        allowed_originals_text = (
            ", ".join(f"({value})" for value in allowed_inline_originals)
            or "(none)"
        )
        inline_policy_context = (
            "\n\n### Deterministic English-original allowlist for this chunk\n"
            f"Authorized first-occurrence originals: {allowed_originals_text}\n"
            "Only these listed originals may be added as English parentheticals. "
            "Ordinary concepts and all unlisted terms must remain Persian-only. "
            "Source citations are separate and must be preserved."
        )
        self.db.log_chunk_event(job_id, idx, "inline_original_policy", {
            "enabled": protect_inline_english,
            "allowed_originals": allowed_inline_originals,
            "categories": {
                source: memory_manager.proper_nouns.category_for(source)
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

        def build_translation_prompt(source_text: str, previous: str) -> str:
            paragraph_count = len([
                paragraph for paragraph in source_text.split("\n\n")
                if paragraph.strip()
            ])
            return TRANSLATE_CHUNK_PROMPT.format(
                exemplars=exemplars,
                glossary_terms=glossary_terms_str,
                memory_context=mem_context.format() + inline_policy_context,
                web_context=web_context_str,
                previous_translation=previous,
                source_text=source_text,
                paragraph_count=paragraph_count,
                term_notes_instruction=term_notes_instruction,
            )

        user_content = build_translation_prompt(chunk.text, prev_trans)

        # Terminology context for the judge & refiner: matched glossary terms
        # plus the established proper-noun renderings, so the "terminology"
        # dimension is scored against the actual mandate instead of blind.
        terminology_ctx = glossary_terms_str
        if mem_context.proper_nouns:
            terminology_ctx += (
                "\n\n### Established proper-noun renderings\n" + mem_context.proper_nouns
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
        try:
            translation = self.llm_client.complete(
                messages=[{"role": "user", "content": user_content}],
                system_prompt=sys_prompt,
                _operation="translation",
            )
        except TruncatedCompletionError:
            if len(source_paragraphs) <= 1:
                raise
            self.db.log_chunk_event(
                job_id,
                idx,
                "translation_adaptive_split",
                {
                    "reason": "repeated_finish_reason_length",
                    "part_count": len(source_paragraphs),
                    "message": (
                        "The complete chunk exhausted its bounded output budget; "
                        "paragraph-boundary recovery was activated."
                    ),
                },
            )
            recovered_parts: list[str] = []
            continuity = prev_trans
            for part_index, source_paragraph in enumerate(source_paragraphs):
                part_prompt = build_translation_prompt(
                    source_paragraph, continuity
                )
                recovered = self.llm_client.complete(
                    messages=[{"role": "user", "content": part_prompt}],
                    system_prompt=sys_prompt,
                    _operation="translation_split_recovery",
                ).strip()
                if not recovered:
                    raise ValueError(
                        f"Adaptive translation part {part_index} was empty."
                    )
                recovered_parts.append(recovered)
                continuity = recovered
            translation = "\n\n".join(recovered_parts)
        if not translation or not translation.strip():
            raise ValueError(f"LLM returned an empty or whitespace-only translation for chunk {idx}.")
        self.db.update_chunk(job_id, idx, ChunkStatus.TRANSLATED, translation)
        self.db.log_chunk_event(job_id, idx, "translation_completed", {
            "translation_chars": len(translation),
            "translation_paragraphs": _paragraph_count(translation),
            "expected_paragraphs": n_source_paras,
        })
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
                    job_id, idx, "integrity_initial_failed", initial_integrity.to_dict()
                )

        # Critique and Refine (judge scores against the terminology mandate)
        if self.config.translation.enable_critique:
            threshold = getattr(self.config.translation, "critique_threshold", 9.0)
            accepted_versions = [translation]
            evaluated_versions: list[tuple[str, Any]] = []
            for ref_iter in range(self.config.translation.max_refine_iterations + 1):
                try:
                    critique_rep = self._run_async(
                        critique_tool.critique(
                            chunk.text,
                            translation,
                            terminology=terminology_ctx,
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
                if getattr(critique_rep, "valid", True):
                    policy_conflicts = _filter_critique_policy_conflicts(
                        critique_rep, chunk.text, allowed_inline_originals
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
                            "message": (
                                "Critic advice requiring no textual change was "
                                "withheld from refinement."
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
                                "Only critic instructions contradicting the deterministic "
                                "English-original policy were withheld from refinement."
                            ),
                        }
                    )
                self.db.log_chunk_event(
                    job_id,
                    idx,
                    "critique_completed",
                    _critique_for_event(critique_rep, threshold, ref_iter),
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
                evaluated_versions.append((translation, critique_rep))
                if _critique_passes_quality_gate(critique_rep, threshold):
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
                try:
                    refinement = self._run_async(
                        refiner_tool.refine_with_decision(
                            chunk.text,
                            translation,
                            critique_rep,
                            terminology=terminology_ctx,
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
                        {"component": "refiner", **invalid_payload},
                    )
                    break

                proposed_translation = refinement.translation
                edit_accepted = True
                integrity_payload: dict[str, Any] | None = None
                if integrity_enabled:
                    edit_integrity = integrity_gate.evaluate(
                        chunk.text,
                        proposed_translation,
                        previous=before_translation,
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
                convergence_reason = ""
                if edit_accepted:
                    proposed_key = _normalized_translation_version(proposed_translation)
                    current_key = _normalized_translation_version(before_translation)
                    prior_keys = [
                        _normalized_translation_version(value)
                        for value in accepted_versions
                    ]
                    if proposed_key == current_key:
                        convergence_reason = "refinement_no_change"
                    elif proposed_key in prior_keys[:-1]:
                        convergence_reason = "refinement_oscillation"

                translation = proposed_translation if edit_accepted else before_translation
                if convergence_reason == "refinement_oscillation":
                    best_index, (translation, _) = max(
                        enumerate(evaluated_versions),
                        key=lambda item: (
                            _critique_candidate_rank(item[1][1]),
                            item[0],
                        ),
                    )
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
                elif edit_accepted and not convergence_reason:
                    accepted_versions.append(translation)

                self.db.update_chunk(job_id, idx, ChunkStatus.REFINED, translation)
                self.db.log_chunk_event(job_id, idx, "refinement_completed", {
                    "iteration": ref_iter + 1,
                    "critique_average": critique_rep.average,
                    "critique_issue_count": len(critique_rep.issues),
                    "blocking_issue_count": len(_blocking_critique_issues(critique_rep)),
                    "decision": refinement.decision,
                    "rationale": _truncate_for_event(refinement.rationale, 1000),
                    "before_chars": before_chars,
                    "after_chars": len(translation),
                    "proposed_chars": len(proposed_translation),
                    "integrity_accepted": edit_accepted,
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
                        correction_prompt = f"""\
The following translation violated the glossary compliance checks.

English Source:
{chunk.text}

Current Translation:
{translation}

Glossary violations found:
{violations_text}

Authorized first-occurrence English originals already present:
{protected_originals_text}

Do not add English parentheticals for any other term.

Please re-translate the text, ensuring that you use the expected glossary terms exactly as prescribed.
Preserve every protected English original above exactly once. Do not remove or relocate
those parentheticals while correcting glossary terminology. Preserve paragraph structure,
citations, numbers, names, and all text unrelated to the listed violations.
{correction_feedback}
Output ONLY the corrected Persian translation.
"""
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
                        correction_changed = (
                            (proposed_correction or "").strip()
                            != (before_correction or "").strip()
                        )
                        correction_accepted = True
                        correction_integrity: dict[str, Any] | None = None
                        if integrity_enabled:
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
                            translation = proposed_correction
                            report = compliance_checker.check(
                                translation=translation,
                                source_text=chunk.text,
                                glossary_manager=glossary_manager,
                                chunk_location=f"Chunk {idx}",
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
            final_integrity = integrity_gate.evaluate(
                chunk.text,
                translation,
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
                    back_translator.compare(chunk.text, back_translated)
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

        self.db.log_chunk_event(job_id, idx, "chunk_completed", {
            "final_translation_chars": len(translation),
            "final_translation_paragraphs": _paragraph_count(translation),
        })

        return translation
