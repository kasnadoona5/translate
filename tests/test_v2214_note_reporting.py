"""Collectors must use source-bound geometry, not plain-text note guesses."""
import ast
import collections
from copy import deepcopy
from pathlib import Path

import pytest

from tarjomeh.quality.integrity import available_note_markers, extract_note_markers
from tarjomeh.quality.note_evidence import paragraph_note_evidence
from tarjomeh.quality.note_reporting import audit_note_markers


def chunk(marker="4", target=None):
    source = f"The hypothetical{marker} claim is disputed."
    start = source.index(marker)
    proof = paragraph_note_evidence(source, {"superscript_markers": [{
        "text": marker, "offset": start,
        "context_before": source[max(0, start - 24):start],
        "context_after": source[start + len(marker):start + len(marker) + 24],
    }]})
    return dict(chunk_index=0, status="needs_review", text=source,
                translation=target or "An assertion\u2074 is disputed.",
                metadata={"source_note_evidence_v1": proof})


@pytest.mark.parametrize("kind", ["reports", "companion"])
def test_actual_collector_does_not_flag_a_proven_note(kind):
    root = Path(__file__).resolve().parents[1]
    current = root / f"scripts/audit_tarjomeh_v2214_{kind}.sh"
    path = current if current.exists() else root / f"scripts/audit_tarjomeh_v2213_{kind}.sh"
    lines = path.read_text(encoding="utf-8").splitlines()
    begin = next(i for i, line in enumerate(lines) if "<<'PY'" in line) + 1
    end = next(i for i in range(begin, len(lines)) if lines[i] == "PY")
    tree = ast.parse("\n".join(lines[begin:end]))
    # Execute the collector's actual reporting statements, not a copied loop.
    nodes = [n for n in tree.body if (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {
            "note_marker_reporting", "surplus_note_markers"} for t in n.targets)
        or isinstance(n, ast.For) and any(isinstance(v, ast.Name)
            and v.id == "surplus_note_markers" for v in ast.walk(n)))]
    scope = dict(collections=collections, chunks=[chunk()], job={},
                 artifact=lambda key: {}, extract_note_markers=extract_note_markers,
                 available_note_markers=available_note_markers)
    if current.exists():
        from tarjomeh.quality.note_reporting import audit_note_markers
        scope["audit_note_markers"] = audit_note_markers
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    assert scope["surplus_note_markers"] == []


@pytest.mark.parametrize("marker,target", [
    ("4", "An assertion\u2074 is disputed."),
    ("7", "An assertion\u2077 is disputed."),
    ("12", "An assertion\u00b9\u00b2 is disputed."),
    ("21", "An assertion[21] is disputed."),
    ("40", "An assertion[\u06f4\u06f0] is disputed."),
])
def test_valid_notes_share_the_physical_inventory(marker, target):
    item = chunk(marker, target)
    before = deepcopy(item)
    report = audit_note_markers({}, [item], {})
    assert not report["surplus_note_markers"]
    assert not report["unresolved_surplus_note_markers"]
    assert report["chunks"][0]["required"] == {marker: 1}
    assert not report["chunks"][0]["missing"]
    assert item == before


@pytest.mark.parametrize("target,extra", [
    ("An assertion\u2074 and another\u2074.", ["4"]),
    ("An assertion\u2074 and another\u2079.", ["9"]),
])
def test_real_surplus_is_still_a_hard_finding(target, extra):
    report = audit_note_markers({}, [chunk(target=target)], {})
    assert report["surplus_note_markers"] == [{"chunk_index": 0, "markers": extra}]
    assert not report["unresolved_surplus_note_markers"]


def test_missing_note_is_not_certified_or_repaired():
    item = chunk(target="An assertion is disputed.")
    before = deepcopy(item)
    assert audit_note_markers({}, [item], {})["chunks"][0]["missing"] == ["4"]
    assert item == before


@pytest.mark.parametrize("metadata", [{}, {"source_note_evidence_v1": {"schema": 1, "source_sha256": "stale"}}, "bad-json"])
def test_unavailable_or_stale_geometry_is_unresolved_not_false_fail_or_pass(metadata):
    item = chunk()
    item["metadata"] = metadata
    report = audit_note_markers({}, [item], {})
    assert not report["surplus_note_markers"]
    assert report["unresolved_surplus_note_markers"] == [{"chunk_index": 0, "markers": ["4"]}]
    assert report["chunks"][0]["evidence_scope"].startswith("text_only_geometry_")


@pytest.mark.parametrize("defect", [None, "file_hash", "paragraph_hash", "source", "indices", "missing_file", "non_pdf", "unlocated", "file_changed_during_parse"])
def test_uploaded_replay_uses_exact_file_parser_and_paragraphs(tmp_path, monkeypatch, defect):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from tarjomeh.core.pipeline import TranslationPipeline
    from tarjomeh.parsers.base import Chapter, Document, Paragraph, Section
    from tarjomeh.quality.note_evidence import text_hash

    item = chunk()
    proof = item["metadata"].pop("source_note_evidence_v1")
    meta = {"superscript_markers": [c["record"] for c in proof["cues"]]}
    source = item["text"]
    document = Document("Unrelated research", [Chapter("Body", sections=[Section("", 2, [Paragraph(source, meta)])])],
                        metadata={"format_type": "pdf"})
    parser = SimpleNamespace(structure_version=1, parse=Mock(return_value=document))
    monkeypatch.setattr(TranslationPipeline, "_get_parser", lambda *args: parser)
    path = tmp_path / "source.pdf"
    path.write_bytes(b"exact frozen uploaded bytes")
    import hashlib
    job = {"input_path": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "config": "{}"}
    identity = {"paragraph_indices": [0], "source_hashes": [text_hash(source)]}
    artifacts = {"document_structure_version": {"version": 6},
                 "canonical_chunk_paragraphs_v1": {"chunks": {"0": identity}}}
    if defect == "file_hash":
        job["source_sha256"] = "stale"
    elif defect == "paragraph_hash":
        identity["source_hashes"] = ["stale"]
    elif defect == "source":
        item["text"] += " Changed."
    elif defect == "indices":
        identity["paragraph_indices"] = [True]
    elif defect == "missing_file":
        path.unlink()
    elif defect == "non_pdf":
        document.metadata["format_type"] = "txt"
    elif defect == "unlocated":
        meta.update(has_superscript=True, superscript_markers=[])
    elif defect == "file_changed_during_parse":
        def changed_file(unused):
            path.write_bytes(b"changed source during parsing")
            return document
        parser.parse.side_effect = changed_file
    before = deepcopy([job, item, artifacts])
    report = audit_note_markers(job, [item], artifacts)
    assert [job, item, artifacts] == before
    assert not report["surplus_note_markers"]
    if defect is None:
        assert not report["unresolved_surplus_note_markers"]
        assert report["chunks"][0]["evidence_scope"] == "verified_uploaded_source_replay"
        assert parser.structure_version == 6
        parser.parse.assert_called_once_with(path)
    else:
        assert report["unresolved_surplus_note_markers"]
        assert report["chunks"][0]["geometry_unavailable_reason"]


def test_probe_is_pure_and_detects_disabled_producer(monkeypatch):
    import tarjomeh.quality.note_reporting as reporting
    from tarjomeh.runtime import _v2214_behavior_probes

    def forbidden(*args, **kwargs):
        raise AssertionError("Probe attempted source I/O")
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "is_file", forbidden)
    assert _v2214_behavior_probes() == {"source_bound_surplus_note_reporting": True}
    original = reporting.audit_note_markers
    def disabled(*args, **kwargs):
        result = original(*args, **kwargs)
        result["surplus_note_markers"] = []
        return result
    monkeypatch.setattr(reporting, "audit_note_markers", disabled)
    assert not _v2214_behavior_probes()["source_bound_surplus_note_reporting"]


def test_verified_absence_does_not_allow_invented_notes():
    item = chunk()
    item["text"] = "An ordinary claim without a note."
    item["metadata"] = {"source_note_evidence_v1": paragraph_note_evidence(item["text"], {})}
    assert audit_note_markers({}, [item], {})["surplus_note_markers"] == [
        {"chunk_index": 0, "markers": ["4"]}]


def test_nonfinal_chunks_never_require_source_replay(monkeypatch):
    from tarjomeh.jobs import text_revision
    def forbidden(*args):
        raise AssertionError("Unexpected source replay")
    monkeypatch.setattr(text_revision, "_source_document", forbidden)
    item = chunk()
    item.update(status="pending", metadata={})
    assert audit_note_markers({"input_path": "unused.pdf"}, [item], {})["chunks"] == []
