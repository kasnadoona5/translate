"""Translation pipeline orchestrator for the Tarjomeh translation system.

Orchestrates ingestion, chunking, translation memory context construction,
web search, translation, critique/refinement, and final output exporting.
"""

from __future__ import annotations

import logging
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import LLMClient
from tarjomeh.core.state_machine import StateMachine
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
)

logger = logging.getLogger(__name__)


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
        self.db = JobDatabase()
        self.current_job_id: str | None = None

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
        if Path(self.config.glossary.path).exists():
            glossary_manager.load(Path(self.config.glossary.path))

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
                extracted_terms = ner_data.get("terms", []) or ner_data.get("extracted_terms", []) or []
                for item in extracted_terms:
                    term = item.get("term")
                    persian = item.get("suggested_persian")
                    if term and persian:
                        glossary_manager.add_term(source=term, target=persian, tgt_lng="fa", context=item.get("context", ""), domain=self.config.translation.domain)
            except Exception as e:
                logger.warning("Automatic term extraction failed: %s", e)

        # 6. Translate Chunks (Sequential or Concurrent)
        translations: dict[int, str] = {}
        
        # Load existing translations if resuming
        if is_resume:
            for c_record in self.db.get_chunks(job_id):
                if c_record["status"] == ChunkStatus.COMPLETED:
                    translations[c_record["chunk_index"]] = c_record["translation"]

        # Run translation loop
        consecutive_errors = 0
        max_errors = self.config.retry.max_consecutive_errors

        web_searcher = WebContextSearcher(self.config, self.llm_client)
        compliance_checker = GlossaryComplianceChecker()
        critique_tool = TranslationCritique(llm_client=self.llm_client)
        refiner_tool = TranslationRefiner(llm_client=self.llm_client, max_iterations=self.config.translation.max_refine_iterations)
        back_translator = BackTranslator(llm_client=self.llm_client, sample_pct=self.config.translation.back_translation_sample_pct)

        # Separate execution paths based on workers
        workers = self.config.translation.parallel_workers
        if workers > 1:
            # Concurrent Pass (Fast / Quality modes only)
            logger.info("Running parallel translation with %d workers", workers)
            lock = threading.Lock()

            def process_chunk_parallel(idx: int) -> tuple[int, str]:
                nonlocal consecutive_errors
                chunk = chunks[idx]
                
                # Fetch memory context safely under lock
                with lock:
                    mem_context = memory_manager.get_context_for_chunk(chunk)
                
                # Web context (slow, run outside lock)
                web_context_str = ""
                if self.config.translation.enable_web_context:
                    web_context_str = self._run_async(web_searcher.get_context_for_chunk(chunk, mem_context.format()))

                # Prep prompts
                prev_trans = translations.get(idx - 1, "")
                matched_entries = glossary_manager.find_terms(chunk.text)
                glossary_terms_str = "\n".join(f"- {e.source} -> {e.target}" for e in matched_entries)

                sys_prompt = TRANSLATE_SYSTEM_PROMPT.format(
                    domain=self.config.translation.domain,
                    style_register=self.config.translation.style_register,
                    country=self.config.translation.country,
                )
                user_content = TRANSLATE_CHUNK_PROMPT.format(
                    glossary_terms=glossary_terms_str,
                    memory_context=mem_context.format(),
                    web_context=web_context_str,
                    previous_translation=prev_trans,
                    source_text=chunk.text,
                )

                # Call Translation LLM
                self.db.update_chunk(job_id, idx, ChunkStatus.TRANSLATING)
                translation = self.llm_client.complete(
                    messages=[{"role": "user", "content": user_content}],
                    system_prompt=sys_prompt
                )

                # Critique and Refine Loop (if enabled)
                if self.config.translation.enable_critique:
                    for ref_iter in range(self.config.translation.max_refine_iterations + 1):
                        critique_rep = self._run_async(critique_tool.critique(chunk.text, translation))
                        if critique_rep.average >= 7.0 or ref_iter == self.config.translation.max_refine_iterations:
                            break
                        # Refine
                        translation = self._run_async(refiner_tool.refine(chunk.text, translation, critique_rep))

                # Glossary compliance post-check
                if self.config.glossary.enable_compliance_check:
                    report = compliance_checker.check(
                        translation=translation,
                        source_text=chunk.text,
                        glossary_manager=glossary_manager,
                        chunk_location=f"Chunk {idx}",
                    )
                    if not report.compliant:
                        for v in report.violations:
                            warn_msg = f"Glossary violation: Term '{v.term}' expected '{v.expected}'"
                            logger.warning("%s in %s", warn_msg, v.chunk_location)
                            with lock:
                                self.warnings.append(f"{v.chunk_location}: {warn_msg}")

                # Back-translate sample
                if self.config.translation.enable_back_translation:
                    # Deterministic sampling based on chunk index percent
                    sample_interval = int(100 / self.config.translation.back_translation_sample_pct) if self.config.translation.back_translation_sample_pct > 0 else 0
                    if sample_interval > 0 and idx % sample_interval == 0:
                        back_translated = self._run_async(back_translator.back_translate(translation))
                        back_translator.compare(chunk.text, back_translated)

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

                    self.db.update_chunk(job_id, idx, ChunkStatus.COMPLETED, translation)
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
                        logger.error("Failed to translate chunk %d: %s", idx, e)
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
                    # 4-Layer memory retrieval
                    mem_context = memory_manager.get_context_for_chunk(chunk)

                    # Web context (Aphra-style)
                    web_context_str = ""
                    if self.config.translation.enable_web_context:
                        web_context_str = self._run_async(web_searcher.get_context_for_chunk(chunk, mem_context.format()))

                    # Prep prompts
                    prev_trans = translations.get(idx - 1, "")
                    matched_entries = glossary_manager.find_terms(chunk.text)
                    glossary_terms_str = "\n".join(f"- {e.source} -> {e.target}" for e in matched_entries)

                    sys_prompt = TRANSLATE_SYSTEM_PROMPT.format(
                        domain=self.config.translation.domain,
                        style_register=self.config.translation.style_register,
                        country=self.config.translation.country,
                    )
                    user_content = TRANSLATE_CHUNK_PROMPT.format(
                        glossary_terms=glossary_terms_str,
                        memory_context=mem_context.format(),
                        web_context=web_context_str,
                        previous_translation=prev_trans,
                        source_text=chunk.text,
                    )

                    # Translate
                    self.db.update_chunk(job_id, idx, ChunkStatus.TRANSLATING)
                    translation = self.llm_client.complete(
                        messages=[{"role": "user", "content": user_content}],
                        system_prompt=sys_prompt
                    )

                    # Critique and Refine
                    if self.config.translation.enable_critique:
                        for ref_iter in range(self.config.translation.max_refine_iterations + 1):
                            critique_rep = self._run_async(critique_tool.critique(chunk.text, translation))
                            if critique_rep.average >= 7.0 or ref_iter == self.config.translation.max_refine_iterations:
                                break
                            translation = self._run_async(refiner_tool.refine(chunk.text, translation, critique_rep))

                    # Glossary Compliance
                    if self.config.glossary.enable_compliance_check:
                        report = compliance_checker.check(
                            translation=translation,
                            source_text=chunk.text,
                            glossary_manager=glossary_manager,
                            chunk_location=f"Chunk {idx}",
                        )
                        if not report.compliant:
                            for v in report.violations:
                                warn_msg = f"Glossary violation: Term '{v.term}' expected '{v.expected}'"
                                logger.warning("%s in %s", warn_msg, v.chunk_location)
                                self.warnings.append(f"{v.chunk_location}: {warn_msg}")

                    # Back translation verification
                    if self.config.translation.enable_back_translation:
                        sample_interval = int(100 / self.config.translation.back_translation_sample_pct) if self.config.translation.back_translation_sample_pct > 0 else 0
                        if sample_interval > 0 and idx % sample_interval == 0:
                            back_translated = self._run_async(back_translator.back_translate(translation))
                            back_translator.compare(chunk.text, back_translated)

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

                    self.db.update_chunk(job_id, idx, ChunkStatus.COMPLETED, translation)
                    self.db.save_memory_state(job_id, memory_manager.to_dict())

                except Exception as e:
                    logger.error("Failed to translate chunk %d: %s", idx, e)
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

        # 7. Assemble Document
        if progress_callback:
            progress_callback("Assembly", 0.90, "Reassembling translated paragraphs...")

        translated_paragraphs: list[TranslatedParagraph] = []
        original_paragraphs = document.all_paragraphs

        # Align chunk texts back to paragraph objects
        original_idx = 0
        for idx in range(total_chunks):
            chunk_source = chunks[idx].text
            chunk_translation = translations.get(idx, "")

            # Split paragraphs in source vs target chunks
            src_paras = [p.strip() for p in chunk_source.split("\n\n") if p.strip()]
            tgt_paras = [p.strip() for p in chunk_translation.split("\n\n") if p.strip()]

            # Fallback if paragraph counts mismatch
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
                orig_para = original_paragraphs[original_idx] if original_idx < len(original_paragraphs) else None
                heading_lvl = orig_para.heading_level if orig_para else None
                meta = orig_para.metadata if orig_para else {}

                translated_paragraphs.append(
                    TranslatedParagraph(
                        index=original_idx,
                        source_text=s,
                        translated_text=t,
                        heading_level=heading_lvl,
                        metadata=meta,
                    )
                )
                original_idx += 1

        trans_doc = TranslatedDocument(
            title=document.title,
            author=document.author,
            paragraphs=translated_paragraphs,
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
