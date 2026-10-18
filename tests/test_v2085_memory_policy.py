import asyncio
from dataclasses import replace

import pytest

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.glossary.manager import GlossaryEntry
from tarjomeh.memory.manager import MemoryManager, _style_record_is_authoritative

FA = (
    "\u0627\u062f\u0639\u0627\u06cc \u0646\u062e\u0633\u062a \u0628\u0647 \u06cc\u06a9 "
    "\u0634\u0631\u0637 \u0648\u0627\u0628\u0633\u062a\u0647 \u0627\u0633\u062a."
)
SCORES = dict.fromkeys(("accuracy", "fluency", "terminology", "register"), 9.5)


def _manager():
    manager = MemoryManager(TarjomehConfig())
    manager.quality_policy_version = 1
    return manager


def _add(manager, index, target=FA):
    return manager.update_after_translation(
        Chunk(
            index=index,
            text=f"The claim {index} is conditional.",
            chapter_title="Chapter",
            section_title="",
            metadata={"structural_roles": ["body"]},
        ),
        target,
        style_evidence={"final_scores": SCORES},
    )


def test_replacement_keeps_history_and_rebuilds_short_term_in_source_order():
    manager = _manager()
    for index in range(6):
        _add(manager, index)
    _add(manager, 0, FA + " " + FA)
    pairs = manager.long_term.serialize()
    assert len(pairs) == 7
    assert (
        len([pair for pair in pairs if pair["chunk_index"] == 0 and pair.get("active", True)]) == 1
    )
    assert manager.short_term.get_entries()[-1].chunk_index == 5
    assert [pair.chunk_index for pair in manager.short_term.get_entries()] == [2, 3, 4, 5]
    assert all(
        pair.get("active", True) for pair in manager.long_term.get_relevant("claim conditional")
    )
    assert any(
        record.get("inactive_reason") == "canonical_chunk_replaced"
        for record in manager.style_sample_records
    )
    assert not any(
        _style_record_is_authoritative(record)
        for record in manager.style_sample_records
        if record.get("active") is False
    )


def test_summary_rejection_survives_restart_and_next_update_covers_both_chapters():
    manager = _manager()
    _add(manager, 0)

    class Client:
        calls = 0
        response = "bad response"

        async def chat(self, _):
            self.calls += 1
            return self.response

    client = Client()
    source, target, refs = manager.pending_summary_inputs(0)
    result = asyncio.run(
        manager.update_bilingual_summary(
            client,
            source,
            target,
            coverage_refs=refs,
        )
    )
    assert not result["candidate_committed"]
    assert manager.summary_pending == refs and client.calls == 1
    resumed = _manager()
    resumed.from_dict(manager.to_dict())
    _add(resumed, 1)
    source, target, refs = resumed.pending_summary_inputs(1)
    assert [ref["chunk_index"] for ref in refs] == [0, 1]
    assert source.count("The claim") == 2
    client.response = (
        "## English Summary\nThe claims are conditional.\n## \u062e\u0644\u0627\u0635"
        "\u0647 \u0641\u0627\u0631\u0633\u06cc\n" + FA
    )
    result = asyncio.run(
        resumed.update_bilingual_summary(client, source, target, coverage_refs=refs)
    )
    assert result["candidate_committed"]
    assert resumed.summary_pending == [] and resumed.summary_coverage == refs
    assert client.calls == 2
    _add(resumed, 0, FA + " " + FA)
    assert not resumed.bilingual_summary.get_context()
    assert len(resumed.summary_history) == 1
    assert [ref["chunk_index"] for ref in resumed.summary_pending] == [0, 1]


def test_summary_mismatched_inputs_do_not_call_model_or_clear_queue():
    manager = _manager()
    _add(manager, 0)
    _, target, refs = manager.pending_summary_inputs(0)
    with pytest.raises(ValueError, match="canonical pending"):
        asyncio.run(
            manager.update_bilingual_summary(object(), "different", target, coverage_refs=refs)
        )
    assert manager.summary_pending == refs


def test_empty_pending_summary_does_not_make_a_redundant_call():
    manager = _manager()
    report = asyncio.run(manager.update_bilingual_summary(object(), "", "", coverage_refs=[]))
    assert report["status"] == "skipped_no_pending_canonical_inputs"
    assert not report["candidate_committed"] and manager.summary_pending == []


def test_curated_revision_never_turns_old_target_into_alias_and_removal_defers():
    manager = _manager()
    entry = GlossaryEntry("protocol", "\u0631\u0648\u06cc\u0647", include_original=True)
    manager.synchronize_curated([entry])
    revised = replace(entry, target="\u067e\u0631\u0648\u062a\u06a9\u0644")
    report = manager.synchronize_curated([revised])
    assert report["updated"][0]["action"] == "replaced_curated_revision"
    assert manager.proper_nouns.all_nouns()["protocol"] == revised.target
    assert entry.target not in manager.proper_nouns.aliases_for("protocol")
    assert (
        manager.proper_nouns.provenance_for("protocol")["superseded"][0]["target"] == entry.target
    )
    manager.synchronize_curated([])
    assert manager.proper_nouns.is_context_deferred("protocol")
    assert manager.proper_nouns.all_nouns()["protocol"] == revised.target
    manager.synchronize_curated([revised])
    assert not manager.proper_nouns.is_context_deferred("protocol")


def test_changed_meaning_is_withheld_not_equal_authority_replaced():
    manager = _manager()
    entry = GlossaryEntry(
        "protocol", "\u0631\u0648\u06cc\u0647", include_original=True, sense="social"
    )
    manager.synchronize_curated([entry])
    report = manager.synchronize_curated(
        [replace(entry, sense="network", target="\u067e\u0631\u0648\u062a\u06a9\u0644")]
    )
    assert report["deferred"][0]["reason"] == "curated_meaning_changed"
    assert manager.proper_nouns.all_nouns()["protocol"] == entry.target
    assert manager.proper_nouns.is_context_deferred("protocol")


def test_mixed_chunk_uses_actual_body_role_without_weakening_style_floors():
    manager = _manager()
    result = manager._update_style_profile(
        FA,
        source_paragraphs=["Heading", "A conditional claim."],
        source_paragraph_indices=[1],
        source_alignment_proven=True,
        paragraph_role="mixed",
        paragraph_roles=["heading", "body"],
        representative=False,
        final_scores=SCORES,
    )
    assert result["accepted"]
    record = manager.style_sample_records[0]
    assert record["paragraph_role"] == "body" and record["representative"]
    assert _style_record_is_authoritative(record)
    for role in ("notes", "table", "front_matter", "unknown"):
        result = manager._update_style_profile(
            FA,
            source_paragraphs=["A conditional claim."],
            source_paragraph_indices=[0],
            source_alignment_proven=True,
            paragraph_roles=[role],
            final_scores=SCORES,
        )
        assert not result["accepted"]


def test_legacy_memory_policy_is_not_migrated():
    manager = _manager()
    manager.from_dict({})
    assert manager.quality_policy_version == 0
    _add(manager, 0)
    _add(manager, 0)
    assert len(manager.long_term.serialize()) == 2
    assert manager.summary_pending == []


def test_notes_heading_does_not_hide_a_proven_body_style_pair():
    manager = _manager()
    body_source = (
        "The claim depends on one clear condition. Its interpretation must remain "
        "consistent with the textual evidence. This analysis examines that relationship "
        "without losing detail or adding new meaning."
    )
    body_target = (
        "\u0627\u06cc\u0646 \u0627\u062f\u0639\u0627 \u0628\u0647 \u06cc\u06a9 \u0634"
        "\u0631\u0637 \u0631\u0648\u0634\u0646 \u0648\u0627\u0628\u0633\u062a\u0647 "
        "\u0627\u0633\u062a. "
        "\u062a\u0641\u0633\u06cc\u0631 \u0622\u0646 \u0628\u0627\u06cc\u062f \u0628"
        "\u0627 \u0634\u0648\u0627\u0647\u062f \u0645\u062a\u0646 \u0633\u0627\u0632"
        "\u06af\u0627\u0631 \u0628\u0645\u0627\u0646\u062f. "
        "\u062a\u062d\u0644\u06cc\u0644 \u062d\u0627\u0636\u0631 \u0647\u0645\u06cc"
        "\u0646 \u0631\u0627\u0628\u0637\u0647 \u0631\u0627 \u0628\u062f\u0648\u0646 "
        "\u062d\u0630\u0641 \u062c\u0632\u0626\u06cc\u0627\u062a "
        "\u06cc\u0627 \u0627\u0641\u0632\u0648\u062f\u0646 \u0645\u0639\u0646\u0627"
        "\u06cc \u062a\u0627\u0632\u0647 \u0628\u0631\u0631\u0633\u06cc \u0645\u06cc"
        "\u200c\u06a9\u0646\u062f."
    )
    chunk = Chunk(
        index=0,
        text="Notes\n\n" + body_source + "\n\nA separate note explains the editorial convention.",
        chapter_title="Chapter",
        section_title="",
        metadata={
            "structural_roles": ["heading", "body", "note"],
            "style_body_paragraphs": [1],
            "style_eligible": False,
        },
    )
    result = manager.update_after_translation(
        chunk,
        "\u06cc\u0627\u062f\u062f\u0627\u0634\u062a\u200c\u0647\u0627\n\n"
        + body_target
        + "\n\n"
        + FA,
        style_approved=True,
        style_evidence={"final_scores": SCORES},
    )
    assert result["style_sample_added"]
    record = manager.style_sample_records[0]
    assert record["source_text"] == body_source and record["text"] == body_target
    assert record["paragraph_role"] == "body" and _style_record_is_authoritative(record)
