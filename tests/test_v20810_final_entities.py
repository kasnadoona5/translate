"""Final-candidate memory reconciliation never promotes an unresolved chunk."""

import copy
import hashlib

import pytest

from tarjomeh.memory.final_entities import (
    audit_first_occurrences,
    first_occurrence_evidence,
    reconcile_final_entities,
)
from tarjomeh.memory.proper_nouns import ProperNouns

EN = "Example Press"
OLD = "\u0627\u0646\u062a\u0634\u0627\u0631\u0627\u062a \u0646\u0645\u0648\u0646\u0647"
NEW = "\u0627\u0646\u062a\u0634\u0627\u0631\u0627\u062a \u0645\u062b\u0627\u0644"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture():
    source = "Example Press publishes books.\n\nThere is another issue."
    before = OLD + " (Example Press).\n\n" + "\u0645\u0633\u0626\u0644\u0647."
    after = before.replace(OLD, NEW)
    sources, targets = source.split("\n\n"), after.split("\n\n")
    identity = {
        "reconstructed": False,
        "canonical_target_hash": sha(after),
        "paragraph_indices": [10, 11],
        "source_hashes": [sha(p) for p in sources],
        "target_hashes": [sha(p) for p in targets],
        "target_offsets": [[0, len(targets[0])], [len(targets[0]) + 2, len(after)]],
    }
    quality = {
        "candidate_target_hash": sha(after),
        "critique_valid": True,
        "critique_candidate_match": True,
        "candidate_matched_critique_present": True,
        "issues": [{"status": "unresolved_grounded", "source_paragraph_index": 11}],
    }
    decision = {
        "decision": "accepted",
        "commit_status": "committed_local",
        "issue_id": "name-1",
        "original_span": OLD + " (Example Press)",
        "resulting_span": NEW + " (Example Press)",
    }
    events = [
        {
            "event_type": "extra_final_refinement_completed",
            "payload": {
                "accepted": True,
                "baseline_target_hash": sha(before),
                "candidate_target_hash": sha(after),
                "retained_target_hash": sha(after),
                "decisions": [decision],
            },
        }
    ]
    nouns = ProperNouns()
    nouns.add_noun(
        EN,
        OLD,
        category="organization",
        provenance="observed_translation",
        evidence_key="anchor:" + sha(source)[:16],
    )
    return nouns, source, after, identity, quality, events


def reconcile(f):
    nouns, source, target, identity, quality, events = f
    return reconcile_final_entities(
        nouns,
        source,
        target,
        chunk_index=3,
        paragraph_identity=identity,
        quality=quality,
        events=events,
    )


def test_final_edit_refreshes_only_the_proven_automatic_entity():
    f = fixture()
    old_state = f[0].serialize()
    result = reconcile(f)
    assert result["records"][0]["action"] == "replaced_superseded_observation"
    nouns = f[0]
    assert nouns.all_nouns()[EN] == NEW
    assert nouns.provenance_for(EN)["authority"] == old_state["provenance"][EN]["authority"]
    assert (
        nouns.provenance_for(EN)["authority_class"]
        == old_state["provenance"][EN]["authority_class"]
    )
    assert nouns.provenance_for(EN)["superseded"][-1]["target"] == OLD
    assert OLD not in nouns.aliases_for(EN)
    assert not nouns.is_context_deferred(EN)


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "paragraph",
        "identity",
        "quality",
        "scope",
        "rejected",
        "competing",
        "no_decision_id",
        "no_original",
        "new_generation",
    ],
)
def test_missing_or_conflicting_proof_defers_without_guessing(change):
    f = fixture()
    nouns, source, target, identity, quality, events = f
    if change == "hash":
        events[0]["payload"]["baseline_target_hash"] = "0" * 64
    if change == "paragraph":
        quality["issues"][0]["source_paragraph_index"] = 10
    if change == "scope":
        quality["issues"][0].pop("source_paragraph_index")
    if change == "identity":
        identity["reconstructed"] = True
    if change == "quality":
        quality["candidate_target_hash"] = "0" * 64
    if change == "rejected":
        events[0]["payload"]["accepted"] = False
    if change == "competing":
        nouns._provenance[EN]["evidence_keys"].append("anchor:another_source")
    if change == "no_decision_id":
        events[0]["payload"]["decisions"][0]["issue_id"] = ""
    if change == "no_original":
        events[0]["payload"]["decisions"][0]["original_span"] = OLD
    if change == "new_generation":
        events.append({"event_type": "chunk_started", "payload": {}})
    result = reconcile(f)
    assert nouns.all_nouns()[EN] == OLD
    assert nouns.is_context_deferred(EN)
    assert result["records"][0]["action"] == "context_deferred"
    assert EN not in nouns.inline_eligible_nouns()


@pytest.mark.parametrize(
    "origin", ["curated_glossary", "approved_research", "accepted_correction", "legacy"]
)
def test_protected_mapping_is_never_changed_or_deferred(origin):
    f = fixture()
    f[0]._provenance[EN]["origin"] = origin
    before = copy.deepcopy(f[0].serialize())
    assert reconcile(f)["records"] == []
    assert f[0].serialize() == before


def test_reconciliation_is_idempotent_and_serialization_retains_proof():
    f = fixture()
    reconcile(f)
    before = copy.deepcopy(f[0].serialize())
    assert reconcile(f)["records"] == []
    restored = ProperNouns()
    restored.deserialize(before)
    assert restored.serialize() == before


def test_true_source_only_is_missing_but_new_bilingual_spelling_is_stale():
    f = fixture()
    nouns, source, target, *_ = f
    nouns.mark_introduced_from_translation(source, target)
    report = audit_first_occurrences(
        nouns.serialize(), [{"chunk_index": 3, "text": source, "translation": target}]
    )
    assert report["missing_first_occurrences"] == []
    assert report["active_stale_mappings"] == [EN]
    assert report["records"][0]["evidence"][0]["source_bound_original_present"]
    missing = audit_first_occurrences(
        nouns.serialize(), [{"text": source, "translation": "\u0645\u062a\u0646"}]
    )
    assert missing["missing_first_occurrences"] == [EN] and not missing["active_stale_mappings"]
    assert not first_occurrence_evidence("Other source.", target, EN, NEW)[
        "first_occurrence_present"
    ]


def test_deferred_mapping_remains_visible_without_active_false_expectation():
    f = fixture()
    f[-1][0]["payload"]["accepted"] = False
    reconcile(f)
    f[0].mark_introduced_from_translation(f[1], f[2])
    report = audit_first_occurrences(f[0].serialize(), [{"text": f[1], "translation": f[2]}])
    assert not report["missing_first_occurrences"] and not report["active_stale_mappings"]
    assert report["records"][0]["classification"] == "deferred_historical_mapping_conflict"


def test_shared_first_use_check_has_lexical_boundaries():
    assert not first_occurrence_evidence(EN, "\u0645" + OLD + "\u0645", EN, OLD)[
        "first_occurrence_present"
    ]


@pytest.mark.parametrize("kind", ["refinement_completed", "exact_final_quality_repair"])
def test_other_committed_edit_events_require_the_same_reversible_proof(kind):
    f = fixture()
    f[-1][0]["event_type"] = kind
    payload = f[-1][0]["payload"]
    payload["issue_decisions"] = payload.pop("decisions")
    assert reconcile(f)["records"][0]["action"] == "replaced_superseded_observation"


@pytest.mark.parametrize(
    "malformation",
    [
        "offset",
        "source_hash",
        "target_hash",
        "duplicate_index",
        "decisions",
        "issues",
        "repeat_source",
        "repeat_anchor",
    ],
)
def test_ambiguous_or_malformed_evidence_never_replaces_the_mapping(malformation):
    f = list(fixture())
    if malformation == "offset":
        f[3]["target_offsets"][0] = [1, 4]
    elif malformation == "source_hash":
        f[3]["source_hashes"][0] = "0" * 64
    elif malformation == "target_hash":
        f[3]["target_hashes"][0] = "0" * 64
    elif malformation == "duplicate_index":
        f[3]["paragraph_indices"] = [10, 10]
    elif malformation == "decisions":
        f[-1][0]["payload"]["decisions"] = None
    elif malformation == "issues":
        f[4]["issues"] = None
    elif malformation == "repeat_source":
        f[1] += " " + EN
    else:
        f[2] += " " + NEW + " (Example Press)"
    reconcile(f)
    assert f[0].all_nouns()[EN] == OLD and f[0].is_context_deferred(EN)


def test_old_diacritic_alias_is_not_reused_as_a_verified_variant():
    f = fixture()
    f[0]._aliases[EN] = [OLD, OLD.replace(" ", "\u0650 ")]
    reconcile(f)
    assert f[0].aliases_for(EN) == []


def test_pipeline_hook_preserves_unrelated_review_and_roundtrips_memory(monkeypatch):
    from types import SimpleNamespace

    import tarjomeh.core.pipeline as pipeline
    from tarjomeh.chunking.chunker import Chunk
    from tarjomeh.core.config import TarjomehConfig
    from tarjomeh.memory.manager import MemoryManager

    f = fixture()
    manager = MemoryManager(TarjomehConfig())
    manager.proper_nouns = f[0]
    events = f[-1] + [
        {"event_type": "chunk_review_required", "payload": {"reason": "other paragraph"}}
    ]
    saved = []
    db = SimpleNamespace(
        get_chunk_events=lambda *a: events,
        log_chunk_event=lambda *a: saved.append(a),
    )
    monkeypatch.setattr(pipeline, "_canonical_final_quality_record", lambda *a: f[4])
    chunk = Chunk(
        index=3,
        text=f[1],
        chapter_title="Chapter",
        section_title="",
        metadata={"paragraph_indices": [10, 11], "structural_roles": ["body", "body"]},
    )
    before = copy.deepcopy(manager.to_dict())
    result = pipeline._reconcile_committed_terminology(
        db, "job", 3, manager, f[1], f[2], chunk=chunk
    )
    assert result["policy"] == "skipped_unresolved_chunk"
    assert result["final_entities"]["records"][0]["action"] == "replaced_superseded_observation"
    assert pipeline._chunk_needs_review(db, "job", 3)
    assert saved[-1][2] == "final_entity_memory_reconciled"
    restored = MemoryManager(TarjomehConfig())
    restored.from_dict(manager.to_dict())
    assert restored.proper_nouns.all_nouns()[EN] == NEW
    after = manager.to_dict()
    changed = [key for key in before if before[key] != after[key]]
    assert changed == ["proper_nouns"]


def test_pdf_word_break_display_is_deduplicated_but_never_authorized():
    from tarjomeh.core.evidence_audit import original_annotation_reporting

    source = "A Scholar-ship was awarded."
    raw = {"unapproved_ungrounded": [{"paragraph_index": 7, "original": "Scholarship"}] * 3}
    before = copy.deepcopy(raw)
    identity = {"0": {"paragraph_indices": [7], "source_hashes": [sha(source)]}}
    report = original_annotation_reporting(raw, [{"chunk_index": 0, "text": source}], identity)
    assert raw == before
    assert report["raw_pass_record_count"] == 3 and report["unique_location_count"] == 1
    location = report["locations"][0]
    assert location["classification"] == "possible_pdf_word_break_not_verified"
    assert location["disposition"] == "REVIEW" and location["pass_record_count"] == 3
    assert report["authority"] == "report_only_no_matching_or_retention_permission"
    identity["0"]["source_hashes"] = ["0" * 64]
    unknown = original_annotation_reporting(raw, [{"chunk_index": 0, "text": source}], identity)
    assert not unknown["locations"][0]["source_evidence_available"]
    assert unknown["locations"][0]["classification"] == "source_original_evidence_unresolved"


def test_genuine_hyphen_does_not_make_a_report_a_permission():
    from tarjomeh.core.evidence_audit import original_annotation_reporting

    source = "A nation-state is discussed."
    identity = {"0": {"paragraph_indices": [7], "source_hashes": [sha(source)]}}
    report = original_annotation_reporting(
        {"unapproved_ungrounded": [{"paragraph_index": 7, "original": "nation-state"}]},
        [{"chunk_index": 0, "text": source}],
        identity,
    )
    assert report["locations"][0]["classification"] == "source_original_evidence_unresolved"
    assert report["locations"][0]["disposition"] == "REVIEW"


def test_reconciled_memory_commits_atomically_and_failed_write_restores_old_state(tmp_path):
    import sqlite3

    from tarjomeh.chunking.chunker import Chunk
    from tarjomeh.jobs.database import ChunkStatus, JobDatabase

    f = fixture()
    database = tmp_path / "jobs.db"
    db = JobDatabase(database)
    db.create_job("atomic", tmp_path / "book.txt", {})
    db.save_chunks("atomic", [Chunk(3, f[1], "Chapter", "", {})])
    before = copy.deepcopy(f[0].serialize())
    db.commit_chunk_checkpoint(
        "atomic", 3, ChunkStatus.NEEDS_REVIEW, "old", {"proper_nouns": before}
    )
    reconcile(f)
    final = f[0].serialize()
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TRIGGER reject_final BEFORE UPDATE ON chunks "
            "BEGIN SELECT RAISE(ABORT, 'simulated interruption'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="simulated interruption"):
        db.commit_chunk_checkpoint(
            "atomic", 3, ChunkStatus.NEEDS_REVIEW, f[2], {"proper_nouns": final}
        )
    restarted = JobDatabase(database)
    assert restarted.get_memory_state("atomic")["proper_nouns"] == before
    assert restarted.get_chunks("atomic")[0]["translation"] == "old"
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TRIGGER reject_final")
    restarted.commit_chunk_checkpoint(
        "atomic",
        3,
        ChunkStatus.NEEDS_REVIEW,
        f[2],
        {"proper_nouns": final},
        paragraph_identity=f[3],
    )
    assert restarted.get_memory_state("atomic")["proper_nouns"] == final
    assert restarted.get_chunks("atomic")[0]["translation"] == f[2]
