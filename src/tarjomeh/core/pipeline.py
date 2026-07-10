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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import LLMClient
from tarjomeh.parsers.base import BaseParser, Document, EXTENSION_PARSER_MAP
from tarjomeh.chunking.chunker import SemanticChunker, FixedChunker, Chunk
from tarjomeh.memory.manager import MemoryManager, MemoryContext
from tarjomeh.context.web_searcher import WebContextSearcher
from tarjomeh.glossary.manager import GlossaryManager
from tarjomeh.glossary.compliance import GlossaryComplianceChecker
from tarjomeh.persian.typography import PersianTypographer
from tarjomeh.exporters import get_exporter
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.jobs.database import JobDatabase, JobStatus, ChunkStatus
from tarjomeh.quality.critique import TranslationCritique
from tarjomeh.quality.refiner import TranslationRefiner
from tarjomeh.quality.back_translator import BackTranslator
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
        })
    return payload


_BLOCKING_CRITIQUE_RE = re.compile(r"^\[(?:CRITICAL|MAJOR)/(?:accuracy|terminology)\]", re.IGNORECASE)


def _blocking_critique_issues(critique: Any) -> list[str]:
    """Return critique issues that must force refinement, regardless of average."""
    return [
        str(issue)
        for issue in getattr(critique, "issues", []) or []
        if _BLOCKING_CRITIQUE_RE.match(str(issue).strip())
    ]


def _critique_passes_quality_gate(critique: Any, threshold: float) -> bool:
    """Average score must pass and no major conceptual/terminology issue may remain."""
    return bool(critique.passes_threshold(threshold) and not _blocking_critique_issues(critique))


def _critique_for_event(critique: Any, threshold: float, iteration: int) -> dict[str, Any]:
    blocking_issues = _blocking_critique_issues(critique)
    return {
        "iteration": iteration,
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
    return any(
        event.get("event_type") == "critique_needs_review"
        for event in events[last_start:]
    )


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
                logger.warning("Job ID %s not found in DB. Starting fresh.", job_id)
                self.db.create_job(job_id, input_path, self.config.to_dict())
        else:
            self.current_job_id = uuid.uuid4().hex[:12]
            self.db.create_job(self.current_job_id, input_path, self.config.to_dict())

        job_id = self.current_job_id

        # Determine default output path if not provided
        if output_path is None:
            fmt = self.config.output.format.lower()
            output_path = input_path.parent / f"{input_path.stem}_translated.{fmt}"
        else:
            output_path = Path(output_path)

        # 2. Parse and Chunk
        if progress_callback:
            progress_callback("Ingestion", 0.05, "Parsing document...")

        document: Document
        chunks: list[Chunk]

        if is_resume:
            # Reconstruct document structure and loaded chunks from DB
            parser = self._get_parser(input_path)
            document = parser.parse(input_path)
            
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
                # Run NER extraction on first chunk as a proxy for TOC / Ch 1
                first_text = chunks[0].text
                sys_prompt = "You are a terminology extraction assistant."
                ner_response = self.llm_client.complete(
                    messages=[{"role": "user", "content": GLOSSARY_EXTRACT_PROMPT.format(text=first_text)}],
                    system_prompt=sys_prompt,
                    response_format={"type": "json_object"}
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
                for item in extracted_terms:
                    term = item.get("term")
                    persian = item.get("suggested_persian")
                    if term and persian:
                        glossary_manager.add_term(source=term, target=persian, tgt_lng="fa", context=item.get("context", ""), domain=self.config.translation.domain, is_auto=True)
            except Exception as e:
                logger.warning("Automatic term extraction failed: %s", e)

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
        critique_tool = TranslationCritique(llm_client=self.critic_client)
        refiner_tool = TranslationRefiner(llm_client=self.llm_client, max_iterations=self.config.translation.max_refine_iterations)
        back_translator = BackTranslator(llm_client=self.critic_client, sample_pct=self.config.translation.back_translation_sample_pct)

        try:
            # Separate execution paths based on workers
            workers = self.config.translation.parallel_workers
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
                        progress_callback("Translation", pct, f"Translating chunk {idx + 1}/{total_chunks}...")

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

                    except PipelinePausedException as e:
                        raise e
                    except Exception as e:
                        logger.error("Failed to translate chunk %d: %s", idx, e)
                        self.db.log_event(job_id, "ERROR", f"Chunk {idx} failed: {type(e).__name__}: {e}")
                        self.db.update_chunk(job_id, idx, ChunkStatus.ERROR)
                        consecutive_errors += 1

                        if consecutive_errors >= max_errors:
                            self.db.update_job_status(job_id, JobStatus.PAUSED_ERROR, str(e))
                            self._send_webhook(
                                "tarjomeh.job.paused_error",
                                f"Pipeline paused after {consecutive_errors} consecutive failures: {e}",
                                JobStatus.PAUSED_ERROR,
                            )
                            raise RuntimeError(
                                f"Pipeline terminated due to {consecutive_errors} consecutive failures. "
                                f"Last error: {e}"
                            ) from e
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
            output_path = input_path.parent / f"{input_path.stem}_reexported.{fmt}"
        else:
            output_path = Path(output_path)

        document, chunks = self._parse_and_chunk(input_path)
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
        exporter_cls = get_exporter(fmt)
        exporter = exporter_cls(self.config.to_dict().get(fmt))
        exporter.export(
            document=trans_doc,
            output_path=output_path,
            bilingual_mode=self.config.output.bilingual_mode,
        )
        self.db.update_job_status(job_id, job["status"], output_path=output_path)
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

        web_searcher = WebContextSearcher(self.config, self.llm_client)
        compliance_checker = GlossaryComplianceChecker()
        critique_tool = TranslationCritique(llm_client=self.critic_client)
        refiner_tool = TranslationRefiner(
            llm_client=self.llm_client,
            max_iterations=self.config.translation.max_refine_iterations,
        )
        back_translator = BackTranslator(
            llm_client=self.critic_client,
            sample_pct=self.config.translation.back_translation_sample_pct,
        )

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

    def _parse_and_chunk(self, input_path: Path) -> tuple[Document, list[Chunk]]:
        parser = self._get_parser(input_path)
        document = parser.parse(input_path)

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

        sys_prompt = TRANSLATE_SYSTEM_PROMPT.format(
            domain=self.config.translation.domain,
            style_register=style_register_value,
            country=self.config.translation.country,
        )
        n_source_paras = len(chunk.metadata.get("paragraph_indices", [])) or \
            len([p for p in chunk.text.split("\n\n") if p.strip()])
        user_content = TRANSLATE_CHUNK_PROMPT.format(
            exemplars=exemplars,
            glossary_terms=glossary_terms_str,
            memory_context=mem_context.format(),
            web_context=web_context_str,
            previous_translation=prev_trans,
            source_text=chunk.text,
            paragraph_count=n_source_paras,
        )

        # Terminology context for the judge & refiner: matched glossary terms
        # plus the established proper-noun renderings, so the "terminology"
        # dimension is scored against the actual mandate instead of blind.
        terminology_ctx = glossary_terms_str
        if mem_context.proper_nouns:
            terminology_ctx += (
                "\n\n### Established proper-noun renderings\n" + mem_context.proper_nouns
            )
        self.db.log_chunk_event(job_id, idx, "terminology_context", {
            "chars": len(terminology_ctx),
            "preview": _truncate_for_event(terminology_ctx, 2000),
        })

        # Translate
        self.db.update_chunk(job_id, idx, ChunkStatus.TRANSLATING)
        translation = self.llm_client.complete(
            messages=[{"role": "user", "content": user_content}],
            system_prompt=sys_prompt
        )
        if not translation or not translation.strip():
            raise ValueError(f"LLM returned an empty or whitespace-only translation for chunk {idx}.")
        self.db.update_chunk(job_id, idx, ChunkStatus.TRANSLATED, translation)
        self.db.log_chunk_event(job_id, idx, "translation_completed", {
            "translation_chars": len(translation),
            "translation_paragraphs": _paragraph_count(translation),
            "expected_paragraphs": n_source_paras,
        })

        # Critique and Refine (judge scores against the terminology mandate)
        if self.config.translation.enable_critique:
            threshold = getattr(self.config.translation, "critique_threshold", 7.0)
            for ref_iter in range(self.config.translation.max_refine_iterations + 1):
                critique_rep = self._run_async(
                    critique_tool.critique(chunk.text, translation, terminology=terminology_ctx)
                )
                self.db.update_chunk(job_id, idx, ChunkStatus.CRITIQUED)
                self.db.log_chunk_event(
                    job_id,
                    idx,
                    "critique_completed",
                    _critique_for_event(critique_rep, threshold, ref_iter),
                )
                if _critique_passes_quality_gate(critique_rep, threshold):
                    break
                if ref_iter == self.config.translation.max_refine_iterations:
                    blocking_issues = _blocking_critique_issues(critique_rep)
                    self.db.log_chunk_event(job_id, idx, "critique_needs_review", {
                        "iteration": ref_iter,
                        "critique_average": critique_rep.average,
                        "blocking_issue_count": len(blocking_issues),
                        "blocking_issues": [
                            _truncate_for_event(str(issue), 1000)
                            for issue in blocking_issues
                        ],
                        "message": (
                            "Unresolved major critique disagreement after maximum "
                            "refinement attempts; output kept but chunk needs human review."
                        ),
                    })
                    break
                before_chars = len(translation)
                refinement = self._run_async(
                    refiner_tool.refine_with_decision(
                        chunk.text,
                        translation,
                        critique_rep,
                        terminology=terminology_ctx,
                    )
                )
                translation = refinement.translation
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
                    "paragraphs_after": _paragraph_count(translation),
                })
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
                    while not report.compliant and attempts < max_attempts:
                        attempts += 1
                        violations_text = "\n".join(
                            f"- English: {v.term} -> expected Persian: {v.expected} (status: {v.status})"
                            for v in report.violations
                        )
                        correction_prompt = f"""\
The following translation violated the glossary compliance checks.

English Source:
{chunk.text}

Current Translation:
{translation}

Glossary violations found:
{violations_text}

Please re-translate the text, ensuring that you use the expected glossary terms exactly as prescribed.
Output ONLY the corrected Persian translation.
"""
                        translation = self.llm_client.complete(
                            messages=[{"role": "user", "content": correction_prompt}],
                            system_prompt=sys_prompt
                        )
                        report = compliance_checker.check(
                            translation=translation,
                            source_text=chunk.text,
                            glossary_manager=glossary_manager,
                            chunk_location=f"Chunk {idx}",
                        )
                        self.db.log_chunk_event(job_id, idx, "glossary_auto_correct_attempt", {
                            "attempt": attempts,
                            "translation_chars": len(translation),
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
            self.db.log_chunk_event(
                job_id,
                idx,
                "glossary_compliance_final",
                _compliance_report_for_event(report),
            )
        else:
            self.db.log_chunk_event(job_id, idx, "glossary_compliance_skipped", {"enabled": False})

        # Back translation verification
        if self.config.translation.enable_back_translation:
            if back_translator.should_sample():
                self.db.log_chunk_event(job_id, idx, "back_translation_sampled", {"sampled": True})
                back_translated = self._run_async(back_translator.back_translate(translation))
                bt_result = back_translator.compare(chunk.text, back_translated)
                self.db.log_chunk_event(job_id, idx, "back_translation_completed", {
                    "similarity_score": bt_result.similarity_score,
                    "flagged": bt_result.flagged,
                    "difference_count": len(bt_result.differences),
                    "differences_preview": bt_result.differences[:50],
                    "back_translated_preview": _truncate_for_event(bt_result.back_translated, 2000),
                })
                if bt_result.flagged:
                    self.db.log_event(
                        job_id,
                        "WARNING",
                        f"Back-translation flagged for Chunk {idx} (Similarity: {bt_result.similarity_score:.2f}). Differences: {bt_result.differences}",
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
