"""Unit tests for Phase 6: External Adapters (LangGraph)."""

from __future__ import annotations

import unittest
from tarjomeh.adapters import LangGraphTranslationAdapter, HAS_LANGGRAPH
from tarjomeh.core.config import TarjomehConfig


class TestLangGraphAdapter(unittest.TestCase):
    """Test LangGraph Adapter import and graph compilation behavior."""

    def test_adapter_import_behavior(self) -> None:
        config = TarjomehConfig()
        
        if not HAS_LANGGRAPH:
            # Should raise ImportError when optional dependency is not installed
            with self.assertRaises(ImportError):
                LangGraphTranslationAdapter(config)
        else:
            # If LangGraph is installed, compile the StateGraph and verify nodes
            adapter = LangGraphTranslationAdapter(config)
            self.assertIsNotNone(adapter)
            
            graph = adapter.compile_graph(db_path=":memory:")
            self.assertIsNotNone(graph)


if __name__ == "__main__":
    unittest.main()
