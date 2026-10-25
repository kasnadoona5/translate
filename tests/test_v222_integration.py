"""Grounded evidence keeps ordinary refinement and publication safeguards."""

import asyncio
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from test_v222_grounding import SOURCE, review

from tarjomeh.core.pipeline import (
    TranslationPipeline,
    _critique_passes_quality_gate,
    _critique_requires_refinement,
)
from tarjomeh.core.render_identity import RenderChangeLedger
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.exporters.docx_exporter import DocxExporter
from tarjomeh.quality.refiner import TranslationRefiner


@pytest.mark.parametrize("decision", ["accepted", "rejected"])
def test_cross_segment_finding_uses_existing_refiner_decision_contract(decision):
    critique = review()
    assert _critique_requires_refinement(critique, 9.0)
    assert not _critique_passes_quality_gate(critique, 9.0)
    original = "The report describes a trend."
    resulting = "describes the measured trend" if decision == "accepted" else "describes a trend"
    candidate = "The report " + resulting + "."
    client = SimpleNamespace(
        chat=AsyncMock(
            return_value=json.dumps(
                {
                    "translation": candidate,
                    "decision": "revised" if decision == "accepted" else "preserved",
                    "rationale": "Independent source comparison.",
                    "issue_decisions": [
                        {
                            "issue_id": critique.issue_details[0]["issue_id"],
                            "decision": decision,
                            "resulting_span": resulting,
                            "rationale": "Source comparison, not quote-location authority.",
                        }
                    ],
                }
            )
        )
    )
    result = asyncio.run(
        TranslationRefiner(client).refine_with_decision(SOURCE, original, critique)
    )
    assert result.valid and result.translation == candidate
    assert result.issue_decisions[0]["decision"] == decision
    client.chat.assert_awaited_once()
    assert SOURCE in client.chat.call_args.args[0]
    # A retained proposal is not a later critique or final quality approval.
    assert not _critique_passes_quality_gate(critique, 9.0)


@pytest.mark.parametrize("mode", ["target_only", "inline", "side_by_side"])
@pytest.mark.parametrize("unexplained_change", [False, True])
def test_quote_location_cannot_change_canonical_text_or_bypass_export(
    tmp_path, mode, unexplained_change
):
    target = (
        "\u06af\u0632\u0627\u0631\u0634 \u0631\u0648\u0646\u062f\u06cc "
        "\u0633\u0646\u062c\u06cc\u062f\u0647 \u0631\u0627 "
        "\u062a\u0648\u0635\u06cc\u0641 \u0645\u06cc\u200c\u06a9\u0646\u062f."
    )
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(0, SOURCE, target)])
    before = deepcopy(document)
    canonical_hash = hashlib.sha256(target.encode()).hexdigest()
    ledger = RenderChangeLedger(document)
    assert review().issue_details[0]["source_quote_span"]["segment_ids"] == ["p1:s1", "p1:s2"]
    artifacts = {}
    db = MagicMock()
    db.save_job_artifact.side_effect = lambda job, key, value: artifacts.update({key: value})
    pipeline = SimpleNamespace(
        db=db, config=SimpleNamespace(output=SimpleNamespace(bilingual_mode=mode))
    )
    output = tmp_path / "checkpoint.docx"
    if unexplained_change:
        document.paragraphs[0].translated_text += " unexplained"
        with pytest.raises(RuntimeError, match="Export blocked"):
            TranslationPipeline._export_with_render_identity(
                pipeline, "offline", document, DocxExporter(), output, ledger, "docx"
            )
        assert not output.exists()
    else:
        TranslationPipeline._export_with_render_identity(
            pipeline, "offline", document, DocxExporter(), output, ledger, "docx"
        )
        assert output.exists() and artifacts["render_identity_audit"]["passed"]
        assert document == before
        assert (
            hashlib.sha256(document.paragraphs[0].translated_text.encode()).hexdigest()
            == canonical_hash
        )
