"""Local quality coordinates must not grant final-name or export authority."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_v20810_final_entities import EN, NEW, OLD, fixture, reconcile, sha

from tarjomeh.core.pipeline import TranslationPipeline, _canonical_final_quality_record
from tarjomeh.core.render_identity import RenderChangeLedger
from tarjomeh.core.term_notes import audit_inline_english_originals
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.exporters.docx_exporter import DocxExporter
from tarjomeh.memory.proper_nouns import ProperNouns


def bound_fixture(segment, indices=(10, 11)):
    f = list(fixture())
    f[3]["paragraph_indices"] = list(indices)
    detail = {
        "issue_id": "unresolved",
        "category": "accuracy",
        "severity": "major",
        "confidence": 0.95,
        "source_segment_id": segment,
        "source_quote": f[1].split("\n\n")[1 if segment == "p2:s1" else 0],
        "current_persian_quote": NEW,
        "suggested_correction": OLD,
    }
    f[-1] += [
        {
            "event_type": "critique_completed",
            "payload": {
                "valid": True,
                "candidate_target_hash": sha(f[2]),
                "issue_details": [detail],
            },
        },
        {"event_type": "chunk_review_required", "payload": {}},
    ]
    f[4] = _canonical_final_quality_record(
        SimpleNamespace(get_chunk_events=lambda *a: f[-1]), "job", 3, f[2]
    )
    return f


@pytest.mark.parametrize("indices", [(0, 1), (10, 11), (1, 2), (100, 200)])
def test_real_quality_same_local_paragraph_defers_for_any_book_offset(indices):
    f = bound_fixture("p1:s1", indices)
    assert f[4]["issues"][0]["source_paragraph_index"] == 0
    before_quality = copy.deepcopy(f[4])
    record = reconcile(f)["records"][0]
    assert record["action"] == "context_deferred"
    assert record["reason"] == "final_entity_paragraph_has_unresolved_evidence"
    assert record["source_local_paragraph_index"] == 0
    assert record["source_global_paragraph_index"] == indices[0]
    assert f[0].all_nouns()[EN] == OLD and f[0].is_context_deferred(EN)
    assert f[4] == before_quality


@pytest.mark.parametrize("indices", [(0, 1), (10, 11), (1, 2)])
def test_real_quality_unrelated_local_paragraph_still_allows_exact_proof(indices):
    f = bound_fixture("p2:s1", indices)
    assert f[4]["issues"][0]["source_paragraph_index"] == 1
    record = reconcile(f)["records"][0]
    assert record["action"] == "replaced_superseded_observation"
    assert f[0].all_nouns()[EN] == NEW
    assert f[4]["issues"][0]["status"].startswith("unresolved")


@pytest.mark.parametrize("segment", ["", "unknown", "p99:s1", "p0:s1"])
def test_real_quality_unknown_or_invalid_paragraph_never_grants_reuse(segment):
    f = bound_fixture(segment)
    record = reconcile(f)["records"][0]
    assert record["reason"] == "final_entity_issue_scope_unproven"
    assert f[0].is_context_deferred(EN) and f[0].all_nouns()[EN] == OLD


@pytest.mark.parametrize("value", [None, -1, 2, 11, True, False, 0.0, "1"])
def test_unresolved_scope_requires_an_in_range_non_boolean_integer(value):
    f = fixture()
    f[4]["issues"][0]["source_paragraph_index"] = value
    reconcile(f)
    assert f[0].is_context_deferred(EN)


def test_resolved_and_advisory_findings_are_not_new_memory_penalties():
    f = fixture()
    f[4]["issues"] = [
        {"status": status, "source_paragraph_index": None}
        for status in ("resolved_by_committed_refinement", "advisory_only")
    ]
    assert reconcile(f)["records"][0]["action"] == "replaced_superseded_observation"


def test_deferred_history_roundtrips_without_prompt_or_inline_reuse():
    f = bound_fixture("p1:s1")
    reconcile(f)
    saved = copy.deepcopy(f[0].serialize())
    restored = ProperNouns()
    restored.deserialize(saved)
    assert restored.serialize() == saved and restored.all_nouns()[EN] == OLD
    assert restored.is_context_deferred(EN) and EN not in restored.inline_eligible_nouns()
    assert OLD not in restored.get_context(f[1])


@pytest.mark.parametrize("mode", ["target_only", "inline", "side_by_side"])
def test_mapping_deferral_neither_changes_text_nor_blocks_valid_docx(tmp_path, mode):
    f = bound_fixture("p1:s1")
    reconcile(f)
    doc = TranslatedDocument(
        paragraphs=[
            TranslatedParagraph(i, source, target)
            for i, (source, target) in enumerate(
                zip(f[1].split("\n\n"), f[2].split("\n\n"), strict=True)
            )
        ]
    )
    before = copy.deepcopy(doc)
    ledger = RenderChangeLedger(doc)
    authorized = f[0].inline_eligible_nouns()
    assert EN not in authorized
    ledger.set_authorized(authorized)
    ledger.step("english_original_cleanup", audit_inline_english_originals, authorized)
    assert doc == before and ledger.audit()["passed"]
    artifacts = {}
    fake = SimpleNamespace(
        config=SimpleNamespace(output=SimpleNamespace(bilingual_mode=mode)),
        db=SimpleNamespace(save_job_artifact=lambda job, key, data: artifacts.update({key: data})),
    )
    output = tmp_path / "checkpoint.docx"
    TranslationPipeline._export_with_render_identity(
        fake, "synthetic", doc, DocxExporter(), output, ledger, "docx"
    )
    assert output.is_file() and artifacts["render_identity_audit"]["passed"]
    assert doc == before


def test_probe_uses_real_quality_producer_and_cannot_pass_a_noop(monkeypatch):
    import tarjomeh.memory.final_entities as entities
    from tarjomeh.runtime import _v20811_behavior_probes

    def forbidden(*a, **k):
        raise AssertionError("Runtime probe performed file I/O")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "is_file", forbidden)
    assert _v20811_behavior_probes()["final_entity_local_issue_scope"]
    monkeypatch.setattr(entities, "reconcile_final_entities", lambda *a, **k: {"records": []})
    assert not _v20811_behavior_probes()["final_entity_local_issue_scope"]
