from pathlib import Path
import socket

import pytest

from tarjomeh.runtime import _v228_behavior_probes


def test_probes_are_pure_and_pass(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Pure probe attempted I/O")
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    assert len(_v228_behavior_probes()) == 5
    assert all(_v228_behavior_probes().values())


@pytest.mark.parametrize("function,key,value", [
    ("_contextual_local_contraction", "context_proven_local_contraction", None),
    ("_post_rollback_review_evidence", "merged_post_rollback_disposition",
     dict(review_required=False, merged_source_review=dict(force_refinement=False))),
])
def test_probes_detect_disabled_production_behavior(monkeypatch, function, key, value):
    import tarjomeh.core.pipeline as pipeline
    monkeypatch.setattr(pipeline, function, lambda **kw: value if function.startswith("_contextual") else value)
    if function == "_post_rollback_review_evidence":
        monkeypatch.setattr(pipeline, function, lambda *a, **kw: value)
    assert _v228_behavior_probes()[key] is False


@pytest.mark.parametrize("kind", ["overlap", "scope", "report"])
def test_other_probes_detect_disabled_behavior(monkeypatch, kind):
    import tarjomeh.core.pipeline as pipeline
    import tarjomeh.glossary.book_review as review
    if kind == "overlap":
        monkeypatch.setattr(pipeline, "_salvage_local_refinement_edits",
                            lambda **kw: (kw["previous"], [{"commit_reason": "overlap"}], {"committed_count": 0}))
        key = "eligible_local_overlap_classification"
    elif kind == "scope":
        monkeypatch.setattr(review, "book_term_scope_failure", lambda *a: None)
        key = "approval_evidence_chapter_consistency"
    else:
        monkeypatch.setattr(review, "partition_book_term_proposals", lambda p: (p, []))
        key = "withheld_book_proposal_reporting"
    assert _v228_behavior_probes()[key] is False
