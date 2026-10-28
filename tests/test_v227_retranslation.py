"""An admitted mapping merge must not fail on a keyword-only conflict API."""
from copy import deepcopy

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.memory.manager import MemoryManager


def test_retranslation_merge_records_conflict_without_crashing_or_overwriting():
    memory = MemoryManager(TarjomehConfig())
    memory.proper_nouns.add_noun("Ada Vale", "\u0622\u062f\u0627 \u0648\u06cc\u0644", category="person",
                                provenance="accepted_correction", context_independent=True)
    incoming = dict(nouns={"Ada Vale": "\u0622\u062f\u0627 \u0648\u0627\u06cc\u0644"},
                    categories={"Ada Vale": "person"},
                    provenance={"Ada Vale": {"origin": "observed_translation", "context_independent": True}},
                    aliases={})
    before = deepcopy(incoming)
    memory.merge_admitted_mapping_changes({}, incoming)
    assert incoming == before
    assert memory.proper_nouns.all_nouns()["Ada Vale"] == "\u0622\u062f\u0627 \u0648\u06cc\u0644"
    assert memory.proper_nouns.is_context_deferred("Ada Vale")
    assert memory.proper_nouns.provenance_for("Ada Vale")["conflict_target"] == before["nouns"]["Ada Vale"]


def test_explicit_authority_cannot_be_deferred_by_attempt_local_conflict():
    memory = MemoryManager(TarjomehConfig())
    memory.proper_nouns.add_noun("Ada Vale", "\u0622\u062f\u0627 \u0648\u06cc\u0644", category="person",
                                provenance="curated_glossary")
    incoming = dict(nouns={"Ada Vale": "\u0622\u062f\u0627 \u0648\u0627\u06cc\u0644"}, categories={"Ada Vale": "person"},
                    provenance={"Ada Vale": {"origin": "observed_translation"}}, aliases={})
    before = deepcopy(memory.proper_nouns.serialize())
    memory.merge_admitted_mapping_changes({}, incoming)
    assert memory.proper_nouns.all_nouns() == before["nouns"]
    assert not memory.proper_nouns.is_context_deferred("Ada Vale")
