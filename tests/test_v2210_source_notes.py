"""Source-position-proven notes and narrowly bounded grouped-number spacing."""

from collections import Counter

import pytest

from tarjomeh.persian.typography import PersianTypographer
from tarjomeh.quality.integrity import PostEditIntegrityGate, extract_note_markers


def record(source, token="4"):
    start = source.index(token)
    return {"text": token, "offset": start,
            "context_before": source[max(0, start - 14):start],
            "context_after": source[start + len(token):start + len(token) + 14]}


def test_medial_source_note_accepts_faithful_repair():
    from tarjomeh.quality.note_evidence import paragraph_note_evidence

    source = "The phantasmagorical4 notion of the state is contested."
    evidence = paragraph_note_evidence(source, {"superscript_markers": [record(source)]})
    result = PostEditIntegrityGate(note_evidence=evidence).evaluate(
        source, "\u0645\u0641\u0647\u0648\u0645 \u0634\u0628\u062d\u200c\u06af\u0648\u0646\u2074 \u062f\u0648\u0644\u062a \u0645\u062d\u0644 \u0627\u062e\u062a\u0644\u0627\u0641 \u0627\u0633\u062a.",
        previous="\u0645\u0641\u0647\u0648\u0645 \u0634\u0628\u062d\u200c\u06af\u0648\u0646 \u062f\u0648\u0644\u062a \u0645\u062d\u0644 \u0627\u062e\u062a\u0644\u0627\u0641 \u0627\u0633\u062a.",
    )
    assert not [f for f in result.blocking if f.check_id.startswith("note_markers_")]


def test_grouped_integer_is_not_spaced_as_punctuation():
    assert PersianTypographer().process("\u06f5\u066c\u06f2\u06f0\u06f0") == "\u06f5\u066c\u06f2\u06f0\u06f0"


@pytest.mark.parametrize("digit", ["4", "\u06f4", "\u0664"])
def test_geometry_is_hash_bound_deduplicated_and_not_a_blanket_regex(digit):
    from tarjomeh.quality.note_evidence import paragraph_note_evidence, verified_note_cues

    source = f"A hypothetical{digit} claim."
    entry = record(source, digit)
    evidence = paragraph_note_evidence(source, {"superscript_markers": [entry, entry]})
    assert extract_note_markers(source, note_evidence=evidence) == ["4"]
    assert extract_note_markers(source) == []
    with pytest.raises(ValueError, match="stale"):
        verified_note_cues(source + "!", evidence)
    stale = {**entry, "offset": entry["offset"] + 1}
    assert not paragraph_note_evidence(source, {"superscript_markers": [stale]})["cues"]


@pytest.mark.parametrize("source", ["H2O is water.", "x2 is a formula.", "ModelX4 is a product."])
def test_plain_word_digits_are_not_geometry(source):
    from tarjomeh.quality.note_evidence import paragraph_note_evidence

    assert not paragraph_note_evidence(source, {})["cues"]
    assert not extract_note_markers(source)


def test_geometry_does_not_override_missing_or_surplus_enforcement():
    from tarjomeh.quality.note_evidence import paragraph_note_evidence

    source = "The hypothetical4 claim remains disputed."
    gate = PostEditIntegrityGate(note_evidence=paragraph_note_evidence(
        source, {"superscript_markers": [record(source)]}))
    assert "note_markers_missing" in {f.check_id for f in gate.evaluate(source, "The hypothetical claim remains disputed.").blocking}
    assert "note_markers_surplus" in {f.check_id for f in gate.evaluate(source, "The hypothetical\u2074 claim\u2074 remains disputed.").blocking}
    assert gate.evaluate(source, "The hypothetical\u2074 claim remains disputed.").source_note_evidence["candidate_sha256"]


@pytest.mark.parametrize("value,expected", [
    ("1 \u066c 234 \u066c 567", "1\u066c234\u066c567"),
    ("\u0665 \u066c \u0662\u0660\u0660", "\u0665\u066c\u0662\u0660\u0660"),
    ("12 \u066c 34", "12 \u066c 34"),
    ("1234 \u066c 567", "1234 \u066c 567"),
    ("1 \u066c 234 \u066c 56", "1 \u066c 234 \u066c 56"),
    ("1 \u066c 23 \u066c 456", "1 \u066c 23 \u066c 456"),
    ("1 \u066c\n234", "1 \u066c\n234"),
    ("1, 234", "1, 234"), ("1\u060c 234", "1\u060c 234"),
    ("5\u066b2", "5\u066b2"), ("x1 \u066c 234", "x1 \u066c 234"),
])
def test_group_spacing_has_complete_numeric_boundaries(value, expected):
    from tarjomeh.persian.typography import rejoin_grouped_integers

    assert rejoin_grouped_integers(value) == expected
    assert rejoin_grouped_integers(expected) == expected


@pytest.mark.parametrize("mode", ["target_only", "inline", "side_by_side"])
def test_docx_geometry_legacy_review_and_canonical_cue_loss(tmp_path, mode):
    from tarjomeh.core.render_identity import RenderChangeLedger, compare_docx_to_rendered
    from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
    from tarjomeh.exporters.docx_exporter import DocxExporter

    source = "A hypothetical4 claim is contested."
    metadata = {"superscript_markers": [record(source)]}
    paragraph = TranslatedParagraph(0, source, "A hypothetical claim is contested.", metadata=metadata)
    document = TranslatedDocument(title="Notes", paragraphs=[paragraph])
    ledger = RenderChangeLedger(document)
    path = tmp_path / "legacy.docx"
    DocxExporter().export(document, path, bilingual_mode=mode)
    check = compare_docx_to_rendered(document, path, mode, canonical_texts=ledger.canonical)
    assert check["passed"]
    assert check["source_note_cues"][0]["legacy_source_fidelity_missing"] == ["4"]
    paragraph.translated_text = "A hypothetical\u2074 claim is contested."
    ledger = RenderChangeLedger(document)
    DocxExporter().export(document, path, bilingual_mode=mode)
    assert compare_docx_to_rendered(document, path, mode, canonical_texts=ledger.canonical)["passed"]
    paragraph.translated_text = "A hypothetical claim is contested."
    DocxExporter().export(document, path, bilingual_mode=mode)
    assert not compare_docx_to_rendered(document, path, mode, canonical_texts=ledger.canonical)["passed"]


def test_real_chunk_admission_does_not_repair_a_faithful_geometry_note():
    from test_v20812_pipeline_export import run_chunk
    from tarjomeh.quality.note_evidence import paragraph_note_evidence

    source = "The hypothetical4 claim is disputed."
    target = "\u0627\u06cc\u0646 \u0627\u062f\u0639\u0627\u06cc \u0641\u0631\u0636\u06cc\u2074 \u0645\u062d\u0644 \u0627\u062e\u062a\u0644\u0627\u0641 \u0627\u0633\u062a."
    pipeline, kwargs = run_chunk(source, target)
    kwargs["chunk"].metadata.update(structural_roles=["body"], source_note_evidence_v1=
        paragraph_note_evidence(source, {"superscript_markers": [record(source)]}))
    assert "\u2074" in pipeline._translate_single_chunk(**kwargs)
    assert [call.kwargs.get("_operation") for call in pipeline.llm_client.complete.call_args_list] == ["translation"]
    events = [(call.args[2], call.args[3]) for call in pipeline.db.log_chunk_event.call_args_list]
    notes = [payload["source_note_evidence"] for kind, payload in events
             if kind == "integrity_check_completed" and payload.get("source_note_evidence")]
    assert notes and all(item["required"] == {"4": 1} and not item["missing"] for item in notes)


def test_real_chunk_admission_cannot_commit_a_missing_geometry_note():
    from test_v20812_pipeline_export import run_chunk
    from tarjomeh.quality.note_evidence import paragraph_note_evidence

    source = "The hypothetical4 claim is disputed."
    target = "\u0627\u06cc\u0646 \u0627\u062f\u0639\u0627\u06cc \u0641\u0631\u0636\u06cc \u0645\u062d\u0644 \u0627\u062e\u062a\u0644\u0627\u0641 \u0627\u0633\u062a."
    pipeline, kwargs = run_chunk(source, target)
    kwargs["chunk"].metadata.update(structural_roles=["body"], source_note_evidence_v1=
        paragraph_note_evidence(source, {"superscript_markers": [record(source)]}))
    with pytest.raises(ValueError):
        pipeline._translate_single_chunk(**kwargs)
    assert not pipeline.db.commit_chunk_checkpoint.called


def test_source_slice_proof_is_unique_and_cannot_cross_a_stale_hash():
    from tarjomeh.quality.note_evidence import paragraph_note_evidence, scoped_note_evidence

    source = "A hypothetical4 claim. Another statement."
    evidence = paragraph_note_evidence(source, {"superscript_markers": [record(source)]})
    scoped = scoped_note_evidence("A hypothetical4 claim.", source, evidence)
    assert extract_note_markers("A hypothetical4 claim.", note_evidence=scoped) == ["4"]
    assert scoped_note_evidence("Changed source.", source, evidence) is None
    with pytest.raises(ValueError):
        scoped_note_evidence("A hypothetical4 claim.", source + " changed", evidence)


@pytest.mark.parametrize("digit", ["4", "\u06f4", "\u0664"])
def test_existing_unique_plain_geometry_cue_is_not_a_missing_note(digit):
    from tarjomeh.quality.integrity import source_note_cue_inventory
    from tarjomeh.quality.note_evidence import paragraph_note_evidence

    source = "A hypothetical4 claim."
    evidence = paragraph_note_evidence(source, {"superscript_markers": [record(source)]})
    target = "\u0627\u062f\u0639\u0627\u06cc \u0641\u0631\u0636\u06cc" + digit + " \u0645\u062d\u0644 \u0627\u062e\u062a\u0644\u0627\u0641 \u0627\u0633\u062a."
    result = source_note_cue_inventory(source, target, note_evidence=evidence)
    assert not result["missing"] and result["available"] == {"4": 1}
    offset = target.index(digit)
    styled = source_note_cue_inventory(source, target, note_evidence=evidence, styled_spans=[(offset, offset + 1)])
    assert styled["available"] == {"4": 1}


@pytest.mark.parametrize("source,target", [
    ("A hypothetical4 claim lists four cases.", "\u0627\u06cc\u0646 \u0646\u0648\u0634\u062a\u0627\u0631\u06f4 \u0645\u0648\u0631\u062f \u062f\u0627\u0631\u062f."),
    ("A hypothetical4 claim lists 4 cases.", "\u0627\u06cc\u0646 \u0646\u0648\u0634\u062a\u0627\u0631\u06f4 \u0645\u0648\u0631\u062f \u062f\u0627\u0631\u062f."),
    ("A hypothetical4 claim.", "\u0641\u0635\u0644\u06f4 \u0627\u06cc\u0646\u062c\u0627\u0633\u062a."),
    ("A hypothetical4 claim.", "4 \u0645\u0648\u0631\u062f \u0648\u062c\u0648\u062f \u062f\u0627\u0631\u062f."),
    ("A hypothetical4 claim.", "H4O"),
])
def test_ordinary_or_misplaced_digits_do_not_satisfy_geometry_notes(source, target):
    from tarjomeh.quality.integrity import source_note_cue_inventory
    from tarjomeh.quality.note_evidence import paragraph_note_evidence

    evidence = paragraph_note_evidence(source, {"superscript_markers": [record(source)]})
    assert source_note_cue_inventory(source, target, note_evidence=evidence)["missing"] == ["4"]


def test_new_runtime_probes_do_not_open_files(monkeypatch):
    from pathlib import Path
    from tarjomeh.runtime import _v2210_behavior_probes

    def forbidden(*args, **kwargs):
        raise AssertionError("Pure probe attempted file I/O")
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "is_file", forbidden)
    assert all(_v2210_behavior_probes().values())


@pytest.mark.parametrize("capability", ["geometry_bound_note_admission", "grouped_integer_spacing", "approved_canonical_text_revision"])
def test_new_runtime_probes_fail_if_their_producer_is_disabled(monkeypatch, capability):
    from tarjomeh.runtime import _v2210_behavior_probes
    import tarjomeh.quality.note_evidence as notes
    import tarjomeh.persian.typography as typography
    import tarjomeh.memory.text_revision as revisions

    if capability == "geometry_bound_note_admission":
        monkeypatch.setattr(notes, "paragraph_note_evidence", lambda *a: None)
    elif capability == "grouped_integer_spacing":
        monkeypatch.setattr(typography, "rejoin_grouped_integers", lambda value: value)
    else:
        monkeypatch.setattr(revisions, "revised_coverage", lambda coverage, changes: coverage)
    assert not _v2210_behavior_probes()[capability]
