"""LangGraph adapter for the Tarjomeh translation system.

Optionally wraps the translation pipeline as a LangGraph StateGraph, enabling
native agentic graph orchestration and SQLite checkpointing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, START, END  # type: ignore[import-untyped]
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import-untyped]
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False


class TranslationState(TypedDict):
    """The State dictionary managed by the LangGraph runtime."""

    input_path: Path
    output_path: Path
    job_id: str
    chunks: list[Any]
    translations: dict[int, str]
    current_idx: int
    status: str
    error_message: str


class LangGraphTranslationAdapter:
    """Adapter wrapping Tarjomeh's workflow as a LangGraph StateGraph."""

    def __init__(self, config: Any) -> None:
        self.config = config
        if not HAS_LANGGRAPH:
            raise ImportError(
                "langgraph is required for the LangGraph adapter. "
                "Install it with:  pip install tarjomeh[langgraph]"
            )

    def _parse_and_chunk(self, state: TranslationState) -> dict[str, Any]:
        """Node: Parses document and divides it into translation-ready chunks."""
        from tarjomeh.parsers import get_parser
        from tarjomeh.chunking.chunker import SemanticChunker, FixedChunker

        logger.info("LangGraph Node: Parsing and Chunking doc %s", state["input_path"])
        parser = get_parser(state["input_path"])
        document = parser.parse(state["input_path"])

        # Word count fallback token counter
        def token_counter(t: str) -> int:
            return len(t.split())

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
        return {
            "chunks": chunks,
            "translations": {},
            "current_idx": 0,
            "status": "translating"
        }

    def _translate_next_chunk(self, state: TranslationState) -> dict[str, Any]:
        """Node: Translates the next chunk in order, updating memory."""
        idx = state["current_idx"]
        chunks = state["chunks"]
        translations = dict(state["translations"])
        job_id = state["job_id"]

        if idx >= len(chunks):
            return {"status": "assembling"}

        chunk = chunks[idx]
        logger.info("LangGraph Node: Translating chunk %d/%d", idx + 1, len(chunks))

        # Re-use real pipeline's logic:
        from tarjomeh.core.pipeline import TranslationPipeline
        from tarjomeh.memory.manager import MemoryManager
        from tarjomeh.context.web_searcher import WebContextSearcher
        from tarjomeh.glossary.manager import GlossaryManager
        from tarjomeh.glossary.compliance import GlossaryComplianceChecker
        from tarjomeh.quality.critique import TranslationCritique
        from tarjomeh.quality.refiner import TranslationRefiner
        from tarjomeh.quality.back_translator import BackTranslator

        pipeline = TranslationPipeline(self.config)
        
        # Load glossary
        glossary_manager = GlossaryManager()
        try:
            glossary_paths = []
            if self.config.glossary.path:
                glossary_paths.append(self.config.glossary.path)
            for extra_path in getattr(self.config.glossary, "paths", []) or []:
                if extra_path not in glossary_paths:
                    glossary_paths.append(extra_path)
            glossary_manager.load_many(glossary_paths, ignore_missing=True)
        except Exception as e:
            logger.warning("Could not load glossary: %s", e)

        # Load or create memory state
        memory_manager = MemoryManager(self.config)
        mem_state = pipeline.db.get_memory_state(job_id)
        if mem_state:
            memory_manager.from_dict(mem_state)

        web_searcher = WebContextSearcher(self.config, pipeline.llm_client)
        compliance_checker = GlossaryComplianceChecker()
        critique_tool = TranslationCritique(llm_client=pipeline.llm_client)
        refiner_tool = TranslationRefiner(llm_client=pipeline.llm_client, max_iterations=self.config.translation.max_refine_iterations)
        back_translator = BackTranslator(llm_client=pipeline.llm_client, sample_pct=self.config.translation.back_translation_sample_pct)

        # Call translation logic
        translation = pipeline._translate_single_chunk(
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

        # Save memory state back to DB
        pipeline.db.save_memory_state(job_id, memory_manager.to_dict())
        pipeline.db.update_chunk(job_id, idx, "completed", translation)

        translations[idx] = translation

        return {
            "translations": translations,
            "current_idx": idx + 1
        }

    def _assemble_and_export(self, state: TranslationState) -> dict[str, Any]:
        """Node: Collects all translated chunks and exports to final format."""
        from tarjomeh.exporters import get_exporter
        from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph

        logger.info("LangGraph Node: Assembling and exporting output %s", state["output_path"])
        paragraphs = []
        for idx, chunk in enumerate(state["chunks"]):
            trans = state["translations"].get(idx, "")
            paragraphs.append(
                TranslatedParagraph(
                    index=idx,
                    source_text=chunk.text,
                    translated_text=trans,
                    heading_level=None
                )
            )

        doc = TranslatedDocument(
            title=state["input_path"].stem,
            paragraphs=paragraphs
        )

        exporter_cls = get_exporter(self.config.output.format)
        exporter = exporter_cls(self.config.to_dict().get(self.config.output.format))
        exporter.export(
            document=doc,
            output_path=state["output_path"],
            bilingual_mode=self.config.output.bilingual_mode
        )

        return {"status": "completed"}

    def compile_graph(self, db_path: str = "jobs/langgraph_checkpoint.db") -> Any:
        """Compile the StateGraph and connect the SqliteSaver checkpointer."""
        builder = StateGraph(TranslationState)

        # Register nodes
        builder.add_node("parse_and_chunk", self._parse_and_chunk)
        builder.add_node("translate_next_chunk", self._translate_next_chunk)
        builder.add_node("assemble_and_export", self._assemble_and_export)

        # Setup edges
        builder.add_edge(START, "parse_and_chunk")
        builder.add_edge("parse_and_chunk", "translate_next_chunk")

        # Routing conditional logic
        def route_next(state: TranslationState) -> str:
            if state["current_idx"] < len(state["chunks"]):
                return "translate_next_chunk"
            return "assemble_and_export"

        builder.add_conditional_edges(
            "translate_next_chunk",
            route_next,
            {
                "translate_next_chunk": "translate_next_chunk",
                "assemble_and_export": "assemble_and_export"
            }
        )
        builder.add_edge("assemble_and_export", END)

        # Setup SQLite checkpointer for resume compatibility
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        import sqlite3
        conn = sqlite3.connect(db_path, check_same_thread=False)
        memory = SqliteSaver(conn)

        return builder.compile(checkpointer=memory)
