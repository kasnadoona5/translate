"""Both existing summary call sites recover without obstructing actual export."""

import ast
import asyncio
import inspect
import logging
import re
import textwrap
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from test_v21_atomic_summary import FA, Client, _manager, _state

from tarjomeh.core.pipeline import TranslationPipeline
from tarjomeh.core.render_identity import RenderChangeLedger
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.exporters.docx_exporter import DocxExporter
from tarjomeh.memory.bilingual_summary import BilingualSummary


@pytest.mark.parametrize("call_site", [0, 1])
@pytest.mark.parametrize("failure", ["reconciliation", "history"])
@pytest.mark.parametrize("mode", ["target_only", "inline", "side_by_side"])
def test_original_summary_call_site_retains_memory_and_exports(
    monkeypatch, tmp_path, call_site, failure, mode,
):
    manager = _manager()
    before = _state(manager)
    full_before = deepcopy(manager.to_dict())
    source, target, refs = manager.pending_summary_inputs(1)
    tree = ast.parse(textwrap.dedent(inspect.getsource(TranslationPipeline.run)))
    blocks = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Try) and node.body
        and isinstance(node.body[0], ast.Assign)
        and isinstance(node.body[0].value, ast.Call)
        and isinstance(node.body[0].value.func, ast.Name)
        and node.body[0].value.func.id == "_chapter_summary_memory_admission"
    ]
    assert len(blocks) == 2
    client = Client()
    database = MagicMock()
    pipeline = SimpleNamespace(
        db=database, llm_client=client, _run_async=asyncio.run,
        config=SimpleNamespace(output=SimpleNamespace(bilingual_mode=mode)),
    )

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic summary preparation error")

    with monkeypatch.context() as context:
        if failure == "reconciliation":
            context.setattr(manager, "reconcile_bilingual_summary", fail)
        else:
            serialize = BilingualSummary.serialize

            def serialize_candidate(summary):
                if summary.english_summary == "The later claim is conditional.":
                    fail()
                return serialize(summary)

            context.setattr(BilingualSummary, "serialize", serialize_candidate)
        namespace = {
            "self": pipeline, "memory_manager": manager, "chunks": [], "idx": 1,
            "job_id": "synthetic-atomic-summary", "new_content": source,
            "chap_translation": target, "coverage_refs": refs,
            "logger": logging.getLogger(__name__),
            "_chapter_summary_memory_admission": lambda *a, **k: {
                "input_trust": "advisory_inputs", "trust_reasons": [],
                "contributing_chunks": len(refs),
            },
        }
        # Execute the original call-site block, not a rewritten copy of its recovery logic.
        module = ast.fix_missing_locations(ast.Module(body=[blocks[call_site]], type_ignores=[]))
        exec(compile(module, "<original-summary-call-site>", "exec"), namespace)
    assert _state(manager) == before and manager.to_dict() == full_before
    assert len(client.prompts) == 1
    event = database.log_chunk_event.call_args
    assert event.args[2] == "bilingual_summary_memory_policy"
    assert event.args[3]["candidate_committed"] is False
    assert event.args[3]["status"] == "failed_retained_previous_summary"

    doc = TranslatedDocument(paragraphs=[
        TranslatedParagraph(0, "The claim is conditional.", FA),
    ])
    canonical = deepcopy(doc)
    artifacts = {}
    database.save_job_artifact.side_effect = lambda job, key, value: artifacts.update({key: value})
    output = tmp_path / "checkpoint.docx"
    TranslationPipeline._export_with_render_identity(
        pipeline, "synthetic-atomic-summary", doc, DocxExporter(), output,
        RenderChangeLedger(doc), "docx",
    )
    assert output.is_file() and doc == canonical
    assert artifacts["render_identity_audit"]["passed"]


@pytest.mark.parametrize("kind", ["reports", "companion"])
def test_retained_failure_reporting_does_not_become_candidate_rejection(kind):
    root = Path(__file__).resolve().parents[1]
    source = (root / f"scripts/audit_tarjomeh_v21_{kind}.sh").read_text(encoding="utf-8")
    # Select the Python here-document containing the collector's summary classification.
    block = next(
        value for value in re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY(?:\r?\n|$)", source, re.S)
        if "rejected_summary_candidates =" in value
    )
    tree = ast.parse(block)
    assignments = [
        node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in
                {"rejected_summary_candidates", "summary_update_failures"} for t in node.targets)
    ]
    assert len(assignments) == 2
    retained = {"candidate_committed": False, "status": "failed_retained_previous_summary"}
    rejected = {"candidate_committed": False, "candidate_quality": {"accepted": False}}
    historical = {"error": "old report without commit evidence"}
    namespace = {
        "summary_events": [retained, rejected, historical],
        "summary_policy_events": [{"payload": x} for x in (retained, rejected, historical)],
    }
    module = ast.Module(body=assignments, type_ignores=[])
    exec(compile(module, "<audit-classification>", "exec"), namespace)
    expected_rejected = rejected if kind == "reports" else {"payload": rejected}
    expected_retained = retained if kind == "reports" else {"payload": retained}
    assert namespace["rejected_summary_candidates"] == [expected_rejected]
    assert namespace["summary_update_failures"] == [expected_retained]
