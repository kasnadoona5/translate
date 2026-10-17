from copy import deepcopy

from tarjomeh.core.pipeline import _reviewed_term_candidate_admission
from tarjomeh.glossary.book_review import (
    _sha,
    attest_book_term_proposals,
    find_term_occurrences,
    format_reviewed_book_terms,
    resolve_reviewed_book_terms,
)


def _records(parts):
    text = "\n\n".join(parts)
    offset = 0
    spans = []
    for part in parts:
        spans.append([offset, offset + len(part)])
        offset += len(part) + 2
    metadata = {
        "paragraph_indices": list(range(len(parts))),
        "structural_roles": ["body"] * len(parts),
        "source_paragraph_spans": spans,
        "source_paragraph_hashes": [_sha(p) for p in parts],
    }
    index = [
        {
            "chunk_index": 0,
            "local_paragraph": i,
            "paragraph_index": i,
            "paragraph_sha256": _sha(part),
            "chapter_position": 2,
        }
        for i, part in enumerate(parts)
    ]
    return text, metadata, index, {0: (text, metadata)}


def test_evidence_is_full_first_attested_paragraph_for_actual_surface():
    text, metadata, index, records = _records(["Institutions act.", "An institution remains."])
    raw = [{"source": "institution", "source_variants": ["Institutions"], "target": "نهاد"}]
    before = deepcopy(raw)
    result = attest_book_term_proposals(raw, index, records)
    proposal = result[0]
    assert raw == before and proposal["source_evidence"] == "An institution remains."
    assert proposal["source_evidence_sha256"] == _sha(proposal["source_evidence"])
    proposal.update(
        status="approved",
        scope_mode="evidence_paragraph",
        english_explanation="A social institution.",
        sense_id="social",
    )
    matches, uncertain = resolve_reviewed_book_terms(text, metadata, 2, [proposal])
    assert len(matches) == 1 and not uncertain and matches[0]["paragraph_index"] == 1
    context = format_reviewed_book_terms(matches)
    assert "A social institution." in context and "social" in context


def test_selected_body_inventory_does_not_guess_singular_or_missing_evidence():
    _, _, index, records = _records(["Institutions act."])
    result = attest_book_term_proposals(
        [{"source": "institution", "source_variants": ["Institutions"]}, {"source": "absent"}],
        index,
        records,
    )
    assert result[0]["source"] == "Institutions"
    assert find_term_occurrences(result[0]["source"], index, records)
    assert result[1]["evidence_unavailable_reason"] == "no_selected_body_occurrence"


def test_surface_collisions_are_explicit_not_silently_merged():
    _, _, index, records = _records(["Institutions act."])
    result = attest_book_term_proposals(
        [
            {"source": "institution", "source_variants": ["Institutions"]},
            {"source": "Institutions"},
        ],
        index,
        records,
    )
    assert all(
        item["evidence_unavailable_reason"] == "colliding_source_proposals" for item in result
    )


def test_mandatory_term_admission_is_paragraph_scoped_not_global():
    source, metadata, _, _ = _records(["The apparatus acts.", "Another apparatus remains."])
    proposal = {
        "source": "apparatus",
        "target": "دستگاه",
        "status": "approved",
        "scope_mode": "evidence_paragraph",
        "source_evidence_sha256": _sha("The apparatus acts."),
    }
    matches, _ = resolve_reviewed_book_terms(source, metadata, 2, [proposal])
    before = "دستگاه عمل می‌کند.\n\nسازوکار دیگری باقی می‌ماند."
    good = before.replace("سازوکار دیگری", "سامانه‌ای دیگر")
    assert _reviewed_term_candidate_admission(source, before, good, metadata, matches)["accepted"]
    bad = before.replace("دستگاه", "سازوکار")
    assert not _reviewed_term_candidate_admission(source, before, bad, metadata, matches)[
        "accepted"
    ]
    assert _reviewed_term_candidate_admission(source, before, bad, metadata, [])["accepted"]


def test_source_accuracy_finding_is_annotated_not_suppressed_by_approval():
    from tarjomeh.core.pipeline import _annotate_reviewed_term_conflicts
    from tarjomeh.quality.critique import CritiqueResult

    source, metadata, _, _ = _records(["The apparatus acts."])
    term = {
        "source": "apparatus",
        "target": "دستگاه",
        "status": "approved",
        "scope_mode": "evidence_paragraph",
        "source_evidence_sha256": _sha(source),
    }
    matches, _ = resolve_reviewed_book_terms(source, metadata, 2, [term])
    issue = {
        "issue_id": "real",
        "category": "accuracy",
        "severity": "major",
        "source_segment_id": "p1:s1",
        "current_persian_quote": "دستگاه",
        "suggested_correction": "سازوکار",
    }
    critique = CritiqueResult(issue_details=[issue])
    _annotate_reviewed_term_conflicts(critique, source, "دستگاه عمل می‌کند.", matches)
    assert critique.issue_details == [issue] and issue["approved_term_conflicts"]


def test_scoped_term_rejection_payload_cannot_claim_accepted_integrity():
    import ast
    import inspect

    from tarjomeh.core import pipeline

    tree = ast.parse(inspect.getsource(pipeline))
    branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "not term_admission['accepted']"
        and any(
            isinstance(child, ast.Constant) and child.value == "refinement_scoped_term_admission"
            for child in ast.walk(node)
        )
    )
    compiled = compile(ast.Module(body=[branch], type_ignores=[]), "<term-admission-event>", "exec")
    for accepted in (False, True):
        context = {
            "term_admission": {"accepted": accepted},
            "edit_accepted": True,
            "integrity_payload": {"accepted": True, "blocking_count": 0},
        }
        exec(compiled, context)
        assert context["edit_accepted"] is accepted
        assert context["integrity_payload"]["accepted"] is accepted
        if not accepted:
            assert context["integrity_payload"]["blocking_count"] == 1
            assert context["integrity_payload"]["stage"] == "refinement_scoped_term_admission"
