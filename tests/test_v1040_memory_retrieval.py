from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.memory.long_term import LongTermMemory
from tarjomeh.memory.manager import MemoryManager, _summary_english_prefix_stutters


def test_retrieval_requires_content_overlap():
    memory = LongTermMemory()
    memory.add("This is about a different historical question.", "پرسش متفاوت")
    assert memory.get_relevant("This is about state power and institutions.") == []


def test_summary_stutter_does_not_reject_regular_plural():
    assert _summary_english_prefix_stutters("state statements", "The state has statements.") == []
    assert _summary_english_prefix_stutters("relation relations", "") == []
    assert _summary_english_prefix_stutters("statemen statement", "A statement matters.")


def test_exact_layer3_layer4_duplicate_is_not_rendered_twice():
    manager = MemoryManager(TarjomehConfig())
    source = "State power depends on institutions."
    target = "قدرت دولت به نهادها وابسته است."
    manager.long_term.add(source, target, reliable=True)
    manager.short_term.add(source, target, trust="trusted")
    chunk = Chunk(index=1, text=source, chapter_title="Chapter", section_title="")
    context = manager.get_context_for_chunk(chunk)
    assert context.long_term == ""
    assert context.short_term.count(target) == 1
    manager.short_term.clear()
    manager.short_term.add(source, target, trust="advisory_review")
    context = manager.get_context_for_chunk(chunk)
    assert target in context.long_term
