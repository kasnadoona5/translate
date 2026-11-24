import hashlib

from types import SimpleNamespace

import pytest
from test_v20812_pipeline_export import run_chunk

from tarjomeh.parsers.pdf_parser import _join_span_texts, _line_superscript_markers
from tarjomeh.quality.integrity import PostEditIntegrityGate
from tarjomeh.quality.note_evidence import attach_chunk_note_evidence, paragraph_note_evidence
from tarjomeh.quality.numeric_artifacts import line_formula_records, numeric_artifact_reporting


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def gate_for(source, metadata):
    chunk = SimpleNamespace(text=source, metadata={"paragraph_indices": [0],
        "source_paragraph_spans": [[0, len(source)]], "source_paragraph_hashes": [digest(source)]})
    document = SimpleNamespace(all_paragraphs=[SimpleNamespace(text=source, metadata=metadata)])
    attach_chunk_note_evidence(document, [chunk])
    return PostEditIntegrityGate(note_source=source,
        note_evidence=chunk.metadata.get("source_note_evidence_v1"),
        formula_evidence=chunk.metadata.get("source_formula_evidence_v1"))


def test_raised_digit_is_not_bound_to_the_same_digit_in_a_year():
    spans = [{"text": "In 1983 the ratio was 46:1.", "flags": 0},
             {"text": "3", "flags": 1}, {"text": " The trend continued.", "flags": 0}]
    offsets = {}
    source = _join_span_texts(spans, offsets)
    metadata = {"superscript_markers": _line_superscript_markers(spans, source, offsets)}
    cues = paragraph_note_evidence(source, metadata)["cues"]
    assert len(cues) == 1 and cues[0]["start"] == source.index(".3") + 1
    gate = gate_for(source, metadata)
    faithful = "\u062f\u0631 1983 \u0646\u0633\u0628\u062a 46 \u0628\u0647 1 \u0628\u0648\u062f.\u00b3 \u0631\u0648\u0646\u062f \u0627\u062f\u0627\u0645\u0647 \u06cc\u0627\u0641\u062a."
    assert gate.evaluate(source, faithful).accepted
    assert not gate.evaluate(source, faithful.replace("46 \u0628\u0647 1", "1 \u0628\u0647 46")).accepted
    no_note = gate.evaluate(source, faithful.replace("\u00b3", ""))
    assert any(f.check_id == "note_markers_missing" for f in no_note.findings)
    assert not gate.evaluate(source, faithful.replace("1983", "1984")).accepted
    assert not PostEditIntegrityGate().evaluate(source, faithful).accepted
    adjacent_date = "\u062f\u0631 1983 \u0628\u0647 46 \u0628\u0647 1 \u0631\u0633\u06cc\u062f.\u00b3 \u0631\u0648\u0646\u062f \u0627\u062f\u0627\u0645\u0647 \u06cc\u0627\u0641\u062a."
    assert gate.evaluate(source, adjacent_date).accepted
    assert not gate.evaluate(source, adjacent_date.replace("1983", "1984")).accepted


@pytest.fixture
def formula_case():
    source = "They measured C 0 2 emissions and recorded 0 kg of residue."
    spans = [dict(text="They measured C 0", size=9, origin=(0, 10), bbox=(0, 0, 60, 10)),
             dict(text="2", size=3, origin=(62, 11), bbox=(62, 0, 65, 11))]
    records = line_formula_records(spans, source, (1.0, 0.0))
    assert records
    return source, spans, records


@pytest.mark.parametrize("formula", ["CO2", "CO\u2082", "\u062f\u06cc\u200c\u0627\u06a9\u0633\u06cc\u062f \u06a9\u0631\u0628\u0646"])
def test_formula_geometry_does_not_consume_a_real_quantity(formula_case, formula):
    source, _, records = formula_case
    target = f"\u0627\u0646\u062a\u0634\u0627\u0631 {formula} \u0633\u0646\u062c\u06cc\u062f\u0647 \u0634\u062f \u0648 0 \u06a9\u06cc\u0644\u0648\u06af\u0631\u0645 \u0628\u0627\u0642\u06cc \u0645\u0627\u0646\u062f."
    gate = gate_for(source, {"formula_geometry": records})
    assert gate.evaluate(source, target).accepted
    assert not gate.evaluate(source, target.replace(" 0 ", " ")).accepted
    assert not gate.evaluate(source, target.replace(formula, "CO3")).accepted
    assert not gate.evaluate(source, target.replace(formula, "CO")).accepted
    assert gate.evaluate("C 0 2 emissions", f"\u0627\u0646\u062a\u0634\u0627\u0631 {formula}").accepted
    assert not gate.evaluate("C 0 2 emissions", "\u0627\u0646\u062a\u0634\u0627\u0631 CO3").accepted
    assert not PostEditIntegrityGate().evaluate(source, target).accepted


def test_unproven_formula_keeps_legacy_number_obligations(formula_case):
    source, spans, records = formula_case
    assert not line_formula_records([spans[0], {**spans[1], "origin": (62, 10)}], source, (1.0, 0.0))
    assert not line_formula_records(spans, source, (0.0, 1.0))
    gate = gate_for(source, {"formula_geometry": records})
    gate.formula_evidence["source_sha256"] = digest("stale")
    target = "\u0627\u0646\u062a\u0634\u0627\u0631 CO2 \u0633\u0646\u062c\u06cc\u062f\u0647 \u0634\u062f \u0648 0 \u06a9\u06cc\u0644\u0648\u06af\u0631\u0645 \u0628\u0627\u0642\u06cc \u0645\u0627\u0646\u062f."
    assert not gate.evaluate(source, target).accepted


def test_true_decimal_stays_required():
    result = PostEditIntegrityGate().evaluate("The value was 1.3 units.",
                                               "\u0645\u0642\u062f\u0627\u0631 1 \u0648\u0627\u062d\u062f \u0628\u0648\u062f.")
    assert not result.accepted
    assert any(item.details.get("missing") == ["1.3"] for item in result.findings)


def test_main_admission_consumes_formula_evidence_without_repair(formula_case):
    source, _, records = formula_case
    target = "\u0627\u0646\u062a\u0634\u0627\u0631 \u062f\u06cc\u200c\u0627\u06a9\u0633\u06cc\u062f \u06a9\u0631\u0628\u0646 \u0633\u0646\u062c\u06cc\u062f\u0647 \u0634\u062f \u0648 0 \u06a9\u06cc\u0644\u0648\u06af\u0631\u0645 \u0628\u0627\u0642\u06cc \u0645\u0627\u0646\u062f."
    pipeline, kwargs = run_chunk(source, target)
    chunk = kwargs["chunk"]
    chunk.metadata.update(paragraph_indices=[0], structural_roles=["body"],
                          source_paragraph_spans=[[0, len(source)]], source_paragraph_hashes=[digest(source)])
    attach_chunk_note_evidence(SimpleNamespace(all_paragraphs=[SimpleNamespace(
        text=source, metadata={"formula_geometry": records})]), [chunk])
    result = pipeline._translate_single_chunk(**kwargs)
    assert "\u062f\u06cc\u200c\u0627\u06a9\u0633\u06cc\u062f \u06a9\u0631\u0628\u0646" in result
    assert [c.kwargs.get("_operation") for c in pipeline.llm_client.complete.call_args_list] == ["translation"]
    events = [{"chunk_index": c.args[1], "event_type": c.args[2], "payload": c.args[3]}
              for c in pipeline.db.log_chunk_event.call_args_list]
    assert not any(e["event_type"] == "integrity_final_failed" for e in events)
    receipts = numeric_artifact_reporting(events)
    assert receipts and any(r["accepted"] and r["findings"][0]["details"]["artifacts"][0]["kind"] == "co2"
                            for r in receipts)
