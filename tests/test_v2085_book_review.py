from copy import deepcopy
from unittest.mock import patch

import pytest

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.glossary.book_review import (
    BOOK_WIDE_RECOMMENDED,
    _sha,
    attest_book_term_proposals,
    bind_original_introductions,
    check_reviewed_book_terms,
    resolve_reviewed_book_terms,
)
from tarjomeh.jobs.database import JobDatabase, JobStatus
from tarjomeh.web.app import create_app


def _records(parts):
    text = "\n\n".join(parts)
    start = 0
    spans = []
    for part in parts:
        spans.append([start, start + len(part)])
        start += len(part) + 2
    metadata = dict(
        paragraph_indices=list(range(len(parts))),
        structural_roles=["body"] * len(parts),
        source_paragraph_spans=spans,
        source_paragraph_hashes=[_sha(part) for part in parts],
    )
    entries = [
        dict(
            chunk_index=0,
            local_paragraph=i,
            paragraph_index=i,
            paragraph_sha256=_sha(part),
            chapter_position=1,
        )
        for i, part in enumerate(parts)
    ]
    return text, metadata, entries, {0: (text, metadata)}


@pytest.mark.parametrize("source", ["de facto", "raison d\u2019etat", "modus operandi"])
def test_identical_proposals_merge_only_in_new_review(source):
    _, _, entries, records = _records([f"The {source} distinction matters."])
    proposals = [
        dict(source=source, target="\u0645\u0639\u0627\u062f\u0644", origin=origin)
        for origin in ("fixed_source_expression", "aligned_pdf_italic")
    ]
    before = deepcopy(proposals)
    legacy = attest_book_term_proposals(proposals, entries, records)
    assert len(legacy) == 2 and all(item["evidence_unavailable_reason"] for item in legacy)
    merged = attest_book_term_proposals(proposals, entries, records, review_version=4)
    assert len(merged) == 1
    assert merged[0]["origins"] == ["aligned_pdf_italic", "fixed_source_expression"]
    assert merged[0]["source_count"] == 1
    assert merged[0]["scope_mode"] == BOOK_WIDE_RECOMMENDED
    assert not merged[0]["evidence_unavailable_reason"]
    assert proposals == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("target", "different"),
        ("sense_id", "other"),
        ("english_explanation", "another meaning"),
        ("keep_original", True),
    ],
)
def test_different_meanings_or_targets_still_collide(field, value):
    _, _, entries, records = _records(["The token matters."])
    proposals = [
        dict(source="token", target="same"),
        {**dict(source="token", target="same"), field: value},
    ]
    result = attest_book_term_proposals(proposals, entries, records, review_version=4)
    assert len(result) == 2
    assert all(
        item["evidence_unavailable_reason"] == "colliding_source_proposals" for item in result
    )


@pytest.fixture
def review_api(tmp_path, monkeypatch):
    text, metadata, entries, _ = _records(["The de facto distinction matters."])
    path = tmp_path / "source.txt"
    path.write_text(text, encoding="utf-8")
    config = TarjomehConfig()
    config.glossary.path, config.glossary.paths = "", []
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("terms", path, config.to_dict())
    db.save_chunks(
        "terms",
        [Chunk(index=0, text=text, metadata=metadata, chapter_title="Chapter", section_title="")],
    )
    db.update_job_status("terms", JobStatus.AWAITING_BOOK_TERM_REVIEW)
    digest = db.get_job_source_hash("terms")
    db.save_job_artifact(
        "terms", "book_term_body_index_v1", dict(source_sha256=digest, entries=entries)
    )
    db.save_job_artifact(
        "terms",
        "book_term_review_v1",
        dict(
            phase="awaiting",
            review_version=4,
            source_sha256=digest,
            proposals=[
                dict(
                    source="de facto",
                    target="\u0628\u0627\u0644\u0641\u0639\u0644",
                    status="proposed",
                    source_evidence=text,
                    source_evidence_sha256=_sha(text),
                )
            ],
        ),
    )
    monkeypatch.setenv("UI_SECRET_TOKEN", "test")
    client = create_app(config).test_client()

    def post(payload):
        with patch("tarjomeh.jobs.database.JobDatabase", return_value=db):
            return client.post(
                "/api/jobs/terms/book-term-review",
                headers={"Authorization": "Bearer test"},
                json=payload,
            )

    return db, post


def test_new_api_default_preserves_persian_and_skip_preserves_approval(review_api):
    db, post = review_api
    target = "\u0628\u0627\u0644\u0641\u0639\u0644"
    response = post(
        dict(
            action="decide",
            confirm_bulk=True,
            decisions=[
                dict(
                    index=0,
                    status="approved",
                    target=target,
                    keep_original=True,
                )
            ],
        )
    )
    assert response.status_code == 200, response.json
    term = response.json["review"]["proposals"][0]
    assert term["target"] == target and term["keep_original"]
    assert term["scope_mode"] == BOOK_WIDE_RECOMMENDED
    response = post(dict(action="skip"))
    assert response.status_code == 200
    assert (
        db.get_job_artifact("terms", "book_term_review_v1")["proposals"][0]["status"] == "approved"
    )


def test_missing_target_or_all_body_still_cannot_be_approved(review_api):
    db, post = review_api
    before = db.get_job_artifact("terms", "book_term_review_v1")
    response = post(
        dict(
            action="decide",
            confirm_bulk=True,
            decisions=[
                dict(
                    index=0,
                    status="approved",
                    target="",
                    keep_original=True,
                )
            ],
        )
    )
    assert response.status_code == 400
    assert db.get_job_artifact("terms", "book_term_review_v1") == before
    response = post(
        dict(
            action="decide",
            confirm_bulk=True,
            decisions=[
                dict(
                    index=0,
                    status="approved",
                    scope_mode="all_body",
                )
            ],
        )
    )
    assert response.status_code == 409
    assert db.get_job_artifact("terms", "book_term_review_v1") == before


def test_original_requires_persian_pair_in_approved_paragraph_only():
    source, metadata, entries, records = _records(["Its modus operandi matters.", "Another claim."])
    term = dict(
        source="modus operandi",
        target="\u0634\u06cc\u0648\u0647 \u0639\u0645\u0644",
        status="approved",
        keep_original=True,
        scope_mode=BOOK_WIDE_RECOMMENDED,
        source_evidence_sha256=_sha("Its modus operandi matters."),
    )
    review = dict(review_version=4, proposals=[term])
    matches, _ = resolve_reviewed_book_terms(source, metadata, 1, [term])
    bind_original_introductions(matches, review, entries, records)
    assert matches[0]["original_introduction_required"]
    pair = term["target"] + " (modus operandi)"
    good, _ = check_reviewed_book_terms(pair + ".\n\nAnother claim.", source, metadata, matches)
    assert not good.violations
    for bad in ("modus operandi", term["target"], "(modus operandi)"):
        report, _ = check_reviewed_book_terms(bad + ".\n\n" + pair, source, metadata, matches)
        assert report.violations


def test_scoped_original_permission_cannot_authorize_another_paragraph():
    from tarjomeh.glossary.book_review import scoped_original_permissions
    from tarjomeh.quality.integrity import PostEditIntegrityGate

    source, metadata, entries, records = _records(
        ["Its locus verum matters.", "Another claim matters."]
    )
    target = "\u0634\u06cc\u0648\u0647 \u0639\u0645\u0644"
    term = dict(
        source="locus verum",
        target=target,
        status="approved",
        keep_original=True,
        scope_mode=BOOK_WIDE_RECOMMENDED,
        source_evidence_sha256=_sha("Its locus verum matters."),
    )
    review = dict(review_version=4, proposals=[term])
    matches, _ = resolve_reviewed_book_terms(source, metadata, 1, [term])
    bind_original_introductions(matches, review, entries, records)
    previous = (
        target
        + " \u0645\u0647\u0645 \u0627\u0633\u062a.\n\n"
        + target
        + " \u0628\u0627\u0642\u06cc \u0627\u0633\u062a."
    )
    good = previous.replace(target, target + " (locus verum)", 1)
    assert scoped_original_permissions(source, good, matches) == ["locus verum"]
    wrong = (
        previous.rsplit(target, 1)[0]
        + target
        + " (locus verum) \u0628\u0627\u0642\u06cc \u0627\u0633\u062a."
    )
    assert scoped_original_permissions(source, wrong, matches) == []
    assert scoped_original_permissions(source, good + " locus verum", matches) == []
    gate = PostEditIntegrityGate(paragraph_originals=matches)
    assert gate.evaluate(
        source, good, previous=previous, protect_inline_english=True, allowed_inline_originals=[]
    ).accepted
    assert not gate.evaluate(
        source, wrong, previous=previous, protect_inline_english=True, allowed_inline_originals=[]
    ).accepted


def test_curated_namespaces_and_explicit_collision_resolution_remain_separate():
    from tarjomeh.glossary.book_review import (
        collision_has_explicit_resolution,
        curated_conflicting_entries,
    )
    from tarjomeh.glossary.manager import GlossaryEntry

    rows = [
        GlossaryEntry("token", "first", sense="social"),
        GlossaryEntry("token", "second", sense="technical"),
    ]
    assert curated_conflicting_entries("token", "first", rows)
    assert not curated_conflicting_entries(
        "token", "first", rows, {"curated_namespace": {"sense": "social"}}
    )
    proposals = [
        dict(source="token", target="first", status="proposed"),
        dict(source="token", target="second", status="proposed"),
    ]
    decisions = [
        dict(
            index=0,
            status="approved",
            sense_id="social",
            scope_mode="reviewed_occurrences",
            occurrences=[dict(paragraph_index=0, paragraph_sha256="a" * 64)],
        ),
        dict(
            index=1,
            status="approved",
            sense_id="technical",
            scope_mode="reviewed_occurrences",
            occurrences=[dict(paragraph_index=1, paragraph_sha256="b" * 64)],
        ),
    ]
    assert collision_has_explicit_resolution(0, proposals, decisions)
    assert collision_has_explicit_resolution(1, proposals, decisions)
    overlapping = deepcopy(decisions)
    overlapping[1]["occurrences"] = overlapping[0]["occurrences"]
    assert not collision_has_explicit_resolution(0, proposals, overlapping)
    assert not collision_has_explicit_resolution(0, proposals, [decisions[0]])
    assert collision_has_explicit_resolution(
        0, proposals, [decisions[0], dict(index=1, status="rejected")]
    )
