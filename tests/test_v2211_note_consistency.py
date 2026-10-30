"""Position-scoped note evidence must agree before and after DOCX rendering."""
from copy import deepcopy

import pytest

from tarjomeh.quality.integrity import PostEditIntegrityGate, source_note_cue_inventory
from tarjomeh.quality.note_evidence import paragraph_note_evidence, revision_audit_evidence
from tarjomeh.exporters.docx_exporter import source_superscript_spans


SOURCE = "1 a hypothetical2 claim; 2 a provisional3 argument; 3 another view."
TARGET = "(۱) ادعای فرضی ۲؛ (۲) استدلال موقت ۳؛ (۳) دیدگاهی دیگر."


def metadata(source, markers=("2", "3")):
    records = []
    for marker in markers:
        start = source.index(marker)
        records.append(dict(text=marker, offset=start,
            context_before=source[max(0, start - 24):start],
            context_after=source[start + len(marker):start + len(marker) + 24]))
    return {"superscript_markers": records}


@pytest.mark.parametrize("digits", ["0123456789", "۰۱۲۳۴۵۶۷۸۹", "٠١٢٣٤٥٦٧٨٩"])
def test_enumeration_and_medial_notes_share_one_physical_inventory(digits):
    target = TARGET.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", digits))
    meta = metadata(SOURCE)
    evidence = paragraph_note_evidence(SOURCE, meta)
    result = source_note_cue_inventory(SOURCE, target, note_evidence=evidence)
    assert result["available"] == {"2": 1, "3": 1} and not result["missing"]
    spans = source_superscript_spans(target, meta, SOURCE)
    assert spans == [(row["start"], row["end"]) for row in result["occurrences"]]
    assert PostEditIntegrityGate(note_evidence=evidence).evaluate(SOURCE, target).accepted


@pytest.mark.parametrize("gap", ["", " ", "\t", "\u00a0"])
def test_unique_medial_cue_allows_horizontal_spacing(gap):
    source = "The hypothetical4 claim is disputed."
    target = "این ادعای فرضی" + gap + "۴ محل اختلاف است."
    meta = metadata(source, ("4",))
    evidence = paragraph_note_evidence(source, meta)
    assert not source_note_cue_inventory(source, target, note_evidence=evidence)["missing"]
    assert source_superscript_spans(target, meta, source)


def test_author_diacritics_are_preserved_not_a_note_boundary_failure():
    source = "The hypothetical4 claim is disputed."
    target = "این ادعای فرضیِ ۴ محل اختلاف است."
    meta = metadata(source, ("4",))
    assert PostEditIntegrityGate(note_evidence=paragraph_note_evidence(source, meta)).evaluate(source, target).accepted
    assert source_superscript_spans(target, meta, source)
    assert "فرضیِ" in target


def test_real_admission_needs_no_repair_for_faithful_enumerated_notes():
    from test_v20812_pipeline_export import run_chunk

    pipeline, kwargs = run_chunk(SOURCE, TARGET)
    kwargs["chunk"].metadata.update(structural_roles=["body"],
        source_note_evidence_v1=paragraph_note_evidence(SOURCE, metadata(SOURCE)))
    pipeline._translate_single_chunk(**kwargs)
    assert [call.kwargs.get("_operation") for call in pipeline.llm_client.complete.call_args_list] == ["translation"]


@pytest.mark.parametrize("target", [
    "(۱) ادعای فرضی؛ (۲) استدلال موقت؛ (۳) دیدگاهی دیگر.",
    "(۱) ادعای فرضی ۲؛ (۲) استدلال موقت؛ (۳) دیدگاهی دیگر.",
])
def test_enumeration_labels_cannot_replace_missing_notes(target):
    evidence = paragraph_note_evidence(SOURCE, metadata(SOURCE))
    result = PostEditIntegrityGate(note_evidence=evidence).evaluate(SOURCE, target)
    assert "note_markers_missing" in {finding.check_id for finding in result.blocking}


@pytest.mark.parametrize("source,target", [
    ("A hypothetical4 claim lists four cases.", "این نوشتار ۴ مورد دارد."),
    ("A hypothetical4 claim lists 4 cases.", "این نوشتار۴ مورد دارد."),
    ("A hypothetical4 claim.\n\nWe list four cases.", "ادعایی مطرح است.\n\nاین نوشتار ۴ مورد دارد."),
    ("A hypothetical4 claim.", "فصل ۴ اینجاست."),
    ("A hypothetical4 claim.", "فصلِ ۴ اینجاست."),
    ("A hypothetical4 claim.", "۴ مورد وجود دارد."),
    ("A hypothetical4 claim.", "H4O"),
    ("A hypothetical4 claim.", "ModelX4"),
    ("A hypothetical4 claim.", "x^ عبارتی 4 است."),
    ("A hypothetical4 claim.", "x + عبارتی 4 است."),
    ("A hypothetical4 claim.", "x = عبارتی 4 است."),
    ("A hypothetical4 claim.", "این فرمول x^4 است."),
    ("A hypothetical4 claim.", "این ادعای فرضی۴ محل اختلاف۴ است."),
])
def test_quantity_reference_formula_or_ambiguity_is_not_a_note(source, target):
    meta = metadata(source, ("4",))
    evidence = paragraph_note_evidence(source, meta)
    assert source_note_cue_inventory(source, target, note_evidence=evidence)["missing"] == ["4"]
    assert not source_superscript_spans(target, meta, source)


def test_quantity_separately_accounted_for_does_not_hide_a_real_note():
    source = "The hypothetical4 claim lists four cases."
    target = "این ادعای فرضی ۴ چهار مورد را فهرست می‌کند."
    evidence = paragraph_note_evidence(source, metadata(source, ("4",)))
    assert not source_note_cue_inventory(source, target, note_evidence=evidence)["missing"]


def test_reference_exclusion_requires_matching_source_role_and_value():
    source = "A hypothetical3 claim refers to chapter 3."
    target = "این ادعای فرضی ۳ در فصل‌های ۳ بررسی می‌شود."
    meta = metadata(source, ("3",))
    proof = paragraph_note_evidence(source, meta)
    assert not source_note_cue_inventory(source, target, note_evidence=proof)["missing"]
    no_reference = "A hypothetical3 claim."
    proof = paragraph_note_evidence(no_reference, metadata(no_reference, ("3",)))
    assert source_note_cue_inventory(no_reference, target, note_evidence=proof)["missing"] == ["3"]
    duplicate_reference = target + " و فصل ۳ نیز چنین است."
    assert source_note_cue_inventory(source, duplicate_reference, note_evidence=paragraph_note_evidence(source, meta))["missing"] == ["3"]


def test_duplicate_geometry_records_cannot_duplicate_styling():
    meta = metadata(SOURCE)
    meta["superscript_markers"] *= 2
    spans = source_superscript_spans(TARGET, meta, SOURCE)
    assert len(spans) == len(set(spans)) == 2


@pytest.mark.parametrize("marker", ["12", "21", "40", "123"])
def test_multidigit_medial_notes_are_not_limited_to_small_cardinals(marker):
    source = f"A hypothetical{marker} claim is disputed."
    target = f"این ادعای فرضی {marker} محل اختلاف است."
    meta = metadata(source, (marker,))
    evidence = paragraph_note_evidence(source, meta)
    assert not source_note_cue_inventory(source, target, note_evidence=evidence)["missing"]
    assert source_superscript_spans(target, meta, source)


def test_unresolved_compound_word_quantity_stays_ambiguous():
    source = "A hypothetical40 claim lists forty cases."
    target = "این نوشتار 40 مورد دارد."
    meta = metadata(source, ("40",))
    assert source_note_cue_inventory(source, target, note_evidence=paragraph_note_evidence(source, meta))["missing"] == ["40"]


def test_current_audit_reports_geometry_scope_and_stale_proof_honestly():
    from tarjomeh.core.evidence_audit import quality_safeguard_evidence

    source = "The hypothetical4 claim is disputed."
    proof = paragraph_note_evidence(source, metadata(source, ("4",)))
    chunk = dict(chunk_index=0, status="needs_review", text=source,
                 translation="این ادعای فرضی محل اختلاف است.", metadata={"source_note_evidence_v1": proof})
    before = deepcopy(chunk)
    def report():
        return quality_safeguard_evidence({}, {}, {}, [chunk], [], {})["current_note_cue_evaluation"][0]
    assert report()["missing"] == ["4"] and report()["evidence_scope"] == "verified_source_geometry"
    assert chunk == before
    chunk["metadata"] = {}
    assert report()["evidence_scope"] == "text_only_geometry_unavailable"
    chunk["metadata"] = {"source_note_evidence_v1": {**proof, "source_sha256": "stale"}}
    assert report()["evidence_scope"] == "text_only_geometry_rejected"
    assert report()["geometry_unavailable_reason"]


def test_rejected_notes_are_structured_cross_stage_and_deduplicated():
    event = dict(rowid=7, chunk_index=0, event_type="extra_final_refinement_completed",
        payload={"accepted": False, "salvage": {"attempts": [{"committed": False,
            "integrity": {"accepted": False, "findings": [{"check_id": "note_markers_surplus"}]}}]}})
    misleading = dict(event_type="integrity_edit_rejected", payload={"rationale": "note_markers_surplus"})
    accepted = dict(event_type="extra_final_refinement_completed", payload={"accepted": True,
        "findings": [{"check_id": "note_markers_surplus", "severity": "warning"}]})
    events = [event, deepcopy(event), misleading, accepted]
    before = deepcopy(events)
    assert revision_audit_evidence(events, {})["rejected_repairs"] == [event]
    assert events == before


def test_new_probes_are_pure(monkeypatch):
    from pathlib import Path
    from tarjomeh.runtime import _v2211_behavior_probes

    def forbidden(*args, **kwargs):
        raise AssertionError("Pure probe attempted file I/O")
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "is_file", forbidden)
    assert all(_v2211_behavior_probes().values())


@pytest.mark.parametrize("capability", ["position_scoped_note_resolution", "geometry_scoped_note_reporting", "structured_note_rejection_reporting"])
def test_new_probes_detect_disabled_producers(monkeypatch, capability):
    import tarjomeh.quality.note_evidence as notes
    import tarjomeh.core.evidence_audit as reports
    from tarjomeh.runtime import _v2211_behavior_probes

    if capability == "position_scoped_note_resolution":
        monkeypatch.setattr(notes, "geometry_plain_target_cues", lambda *args: [])
    elif capability == "geometry_scoped_note_reporting":
        monkeypatch.setattr(reports, "quality_safeguard_evidence", lambda *args:
            {"current_note_cue_evaluation": [{"evidence_scope": "text_only_geometry_unavailable", "missing": []}]})
    else:
        monkeypatch.setattr(notes, "_rejected_note_finding", lambda *args: False)
    assert not _v2211_behavior_probes()[capability]
