import ast
import re
from pathlib import Path

from tarjomeh.core.evidence_audit import readability_call_accounting, targeted_quality_evidence


def test_readability_routes_do_not_mislabel_logical_rollback_as_transport_retries():
    events = [
        {
            "event_type": "llm_call_attempt",
            "payload": {
                "operation": "persian_readability_review",
                "success": success,
            },
        }
        for success in (True, False, True)
    ]
    events += [
        {"event_type": "persian_readability_review", "payload": {}},
        {
            "event_type": "post_rollback_final_validation",
            "payload": {"readability_review_attempted": True},
        },
        {
            "event_type": "post_rollback_final_validation",
            "payload": {"readability_review_attempted": False},
        },
    ]
    result = readability_call_accounting(events)
    assert result["physical_attempts"] == 3 and result["failed_physical_attempts"] == 1
    assert result["ordinary_result_batches"] == 1
    assert result["post_rollback_logical_attempts"] == 1


def test_lexical_deferrals_and_added_outcomes_remain_report_only():
    state = {
        "nouns": {"Canada": "CA کانادا"},
        "categories": {"Canada": "term"},
        "provenance": {
            "Canada": {"origin": "accepted_correction", "alignment_status": "exact_local"}
        },
    }
    events = [
        {
            "event_type": "accepted_terminology_reconciled",
            "chunk_index": 1,
            "payload": {"reconciled": [{"action": "added", "source": "a term"}]},
        }
    ]
    result = targeted_quality_evidence(state, events, [])
    assert not result["deferred_name_mappings"]
    assert (
        result["deferred_terminology_mappings"][0]["reason"]
        == "unrelated_leading_latin_display_label"
    )
    assert result["accepted_terminology_reconciliation"][0]["reconciled"][0]["action"] == "added"
    assert result["policy"] == "report_only_no_authority_or_text_change"


def test_collector_entity_markers_use_behavior_not_file_placement():
    source = (Path(__file__).parents[1] / "scripts/audit_tarjomeh_v2083_companion.sh").read_text(
        encoding="utf-8",
    )
    block = next(
        block
        for block in re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY", source, re.S)
        if "same_line_entity_inventory" in block
    )
    tree = ast.parse(block)
    for name in ("same_line_entity_inventory", "bounded_source_entities"):
        expression = next(
            value
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values, strict=True)
            if isinstance(key, ast.Constant) and key.value == name
        )
        compiled = compile(ast.Expression(expression), "<entity-behavior-contract>", "eval")
        assert eval(compiled, {"runtime_probes": {"shared_source_entity_inventory_contract": True}})
        assert not eval(
            compiled, {"runtime_probes": {"shared_source_entity_inventory_contract": False}}
        )


def test_probe_detects_broken_pipeline_wiring(monkeypatch):
    from tarjomeh.core import pipeline
    from tarjomeh.runtime import _v2083_behavior_probes

    monkeypatch.setattr(pipeline, "_source_entity_inventory", lambda *args: [])
    assert not _v2083_behavior_probes()["shared_source_entity_inventory_contract"]
