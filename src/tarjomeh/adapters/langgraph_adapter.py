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

        if idx >= len(chunks):
            return {"status": "assembling"}

        chunk = chunks[idx]
        logger.info("LangGraph Node: Translating chunk %d/%d", idx + 1, len(chunks))

        # Renders dummy translation (real adapter uses LLMClient + memory)
        translation = f" [ترجمه: {chunk.text}]"
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
        memory = SqliteSaver.from_conn_string(db_path)

        return builder.compile(checkpointer=memory)
