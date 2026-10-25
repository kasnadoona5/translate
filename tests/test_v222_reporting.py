"""New reports expose recorded evidence without rewriting old events."""

from copy import deepcopy
from pathlib import Path

from tarjomeh.core.evidence_audit import quote_and_numeric_reporting
from tarjomeh.quality.grounding import source_quote_span


def test_quote_reporting_revalidates_recorded_proof_without_input_mutation():
    source = "Vale et al. (2018) describe a trend."
    proof = source_quote_span(source, source)
    chunks = [{"chunk_index": 0, "text": source, "translation": "A trend."}]
    events = [
        {
            "chunk_index": 0,
            "event_type": "critique_completed",
            "payload": {
                "issue_details": [
                    {"source_quote": source, "source_quote_span": proof, "issue_id": "x"}
                ]
            },
        }
    ]
    before = deepcopy((chunks, events))
    report = quote_and_numeric_reporting(events, chunks)
    assert report["recorded_quote_span_count"] == 1
    assert report["recorded_quote_spans"][0]["matches_saved_source"]
    assert (chunks, events) == before
    events[0]["payload"]["issue_details"][0]["source_quote_span"]["start"] = 99
    assert not quote_and_numeric_reporting(events, chunks)["recorded_quote_spans"][0][
        "matches_saved_source"
    ]


def test_absent_historical_proof_is_not_reconstructed_or_called_failed():
    report = quote_and_numeric_reporting(
        [{"payload": {"issue_details": [{"source_quote": "a"}]}}], []
    )
    assert report["recorded_quote_span_count"] == 0
    assert "not_recorded" in report["historical_absence"]


def test_grouping_reporting_is_explicitly_offline():
    report = quote_and_numeric_reporting(
        [],
        [
            {
                "chunk_index": 0,
                "text": "There are 1,000 records.",
                "translation": "There are 1000 records.",
            }
        ],
    )
    assert len(report["current_offline_grouped_equivalences"]) == 1
    assert report["production_numeric_policy_changed"] is False


def test_both_collectors_include_additive_evidence_in_json_and_text():
    root = Path(__file__).resolve().parents[1]
    for kind in ("reports", "companion"):
        text = (root / f"scripts/audit_tarjomeh_v222_{kind}.sh").read_text(encoding="utf-8")
        assert '"v222_quote_and_numeric_evidence": quote_numeric_reporting' in text
        assert "Quote spans and offline numeric evidence:" in text
