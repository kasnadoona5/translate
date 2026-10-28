"""Behavior probes must detect regressions, not just advertise capabilities."""
from pathlib import Path

import pytest

from tarjomeh.runtime import _v227_behavior_probes


def test_new_probes_are_pure_and_all_pass(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Pure probe attempted file I/O")

    monkeypatch.setattr(Path, "open", forbidden)
    assert len(_v227_behavior_probes()) == 5
    assert all(_v227_behavior_probes().values())


@pytest.mark.parametrize("kind", ["markers", "uncertainty", "merge", "counts", "attribution"])
def test_probe_fails_when_its_production_behavior_is_disabled(monkeypatch, kind):
    import tarjomeh.core.pipeline as pipeline
    import tarjomeh.memory.manager as manager
    import tarjomeh.parsers.pdf_parser as parser

    if kind in {"markers", "uncertainty"}:
        original = parser._classify_footnote
        monkeypatch.setattr(parser, "_classify_footnote",
                            lambda block, median, height, evidence=None, **kwargs:
                            original(block, median, height, evidence, structure_version=5))
        key = "wrapped_note_marker_boundaries" if kind == "markers" else "note_marker_evidence_precision"
    elif kind == "merge":
        monkeypatch.setattr(manager.MemoryManager, "merge_admitted_mapping_changes", lambda *args: None)
        key = "retranslation_conflict_keyword_merge"
    elif kind == "counts":
        original = pipeline._classify_candidate_quality_change

        def asymmetric(before, after, **kwargs):
            result = original(before, after, **kwargs)
            result[-1]["before_actionable_count"] = len(before)
            return result

        monkeypatch.setattr(pipeline, "_classify_candidate_quality_change", asymmetric)
        key = "symmetric_candidate_quality_counts"
    else:
        monkeypatch.setattr(pipeline, "_decisions_without_regressed_edits", lambda *a, **k: ([], []))
        key = "applied_edit_regression_attribution"
    assert _v227_behavior_probes()[key] is False
