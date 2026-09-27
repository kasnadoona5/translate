from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tarjomeh.context.book_researcher import BookResearcher
from tarjomeh.context.book_term_candidates import collect_book_term_candidates
from tarjomeh.core.pipeline import (
    _language_quality_strictly_improves,
    _merge_approved_book_terms,
    audit_translation_language,
    repair_document_source_grounded_language_artifacts,
)
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.glossary.manager import GlossaryManager
from tarjomeh.jobs.database import JobDatabase
from tarjomeh.memory.proper_nouns import (
    ProperNouns,
    is_bounded_person_name_target,
)
from tarjomeh.parsers.base import Chapter, Document, Paragraph, Section
from tarjomeh.quality.integrity import repair_source_grounded_language_artifacts


def test_optional_plural_is_admitted_by_final_render_comparator() -> None:
    source = "The discourse(s) remains relevant."
    before = (
        "\u06af\u0641\u062a\u0645\u0627\u0646 (\u0647\u0627) \u0645\u0647\u0645 \u0627\u0633\u062a."
    )
    after, _ = repair_source_grounded_language_artifacts(source, before)
    assert "\u06af\u0641\u062a\u0645\u0627\u0646(\u0647\u0627)" in after
    assert _language_quality_strictly_improves(
        audit_translation_language(source, before),
        audit_translation_language(source, after),
    )
    document = TranslatedDocument(
        title="Test",
        paragraphs=[TranslatedParagraph(index=0, source_text=source, translated_text=before)],
    )
    report = repair_document_source_grounded_language_artifacts(document)
    assert report["accepted_repair_count"] == 1
    assert document.paragraphs[0].translated_text == after


def test_orphan_dash_before_object_marker_is_narrowly_repaired() -> None:
    source = "They make history - their own and that of others - in context."
    target = (
        "\u0622\u0646\u0647\u0627 \u062a\u0627\u0631\u06cc\u062e "
        "\u062e\u0648\u062f \u0648 \u062f\u06cc\u06af\u0631\u0627\u0646 "
        "\u2014 \u0631\u0627 \u0645\u06cc\u200c\u0633\u0627\u0632\u0646\u062f."
    )
    repaired, report = repair_source_grounded_language_artifacts(source, target)
    assert "\u2014 \u0631\u0627" not in repaired
    assert any(item["type"] == "orphaned_object_marker_dash" for item in report["repairs"])
    paired = (
        "\u0622\u0646\u0647\u0627 \u2014 \u062e\u0648\u062f \u0648 "
        "\u062f\u06cc\u06af\u0631\u0627\u0646 \u2014 \u0631\u0627 "
        "\u0645\u06cc\u200c\u0633\u0627\u0632\u0646\u062f."
    )
    assert repair_source_grounded_language_artifacts(source, paired)[0] == paired
    assert (
        repair_source_grounded_language_artifacts(source, target, structural_role="heading")[0]
        == target
    )


def test_contextual_person_name_cannot_be_reusable_even_if_legacy_exact() -> None:
    source = "Manuela Tecusan"
    good = "\u0645\u0627\u0646\u0648\u0626\u0644\u0627 \u062a\u06a9\u0648\u0634\u0627\u0646"
    bad = (
        "\u0648\u06cc\u0631\u0627\u0633\u062a\u0627\u0631\u06cc "
        "\u0639\u0627\u0644\u0645\u0627\u0646\u0647 \u0648 "
        "\u06a9\u0627\u0645\u0644\u0627\u064b \u062a\u062e\u0635\u0635\u06cc\u0650 "
        "\u0645\u0627\u0646\u0648\u0626\u0644\u0627 \u062a\u06a9\u0648\u0634\u0627\u0646"
    )
    assert is_bounded_person_name_target(source, good)
    assert not is_bounded_person_name_target(source, bad)
    nouns = ProperNouns()
    nouns.add_noun(
        source,
        bad,
        category="person",
        provenance="accepted_correction",
        alignment_status="exact_local",
    )
    assert nouns.is_context_deferred(source)
    restored = ProperNouns()
    restored.deserialize(nouns.serialize())
    assert restored.is_context_deferred(source)


def test_book_approval_reused_only_for_identical_source_bytes(tmp_path: Path) -> None:
    db = JobDatabase(tmp_path / "jobs.db")
    first = tmp_path / "a.pdf"
    same = tmp_path / "b.pdf"
    other = tmp_path / "c.pdf"
    first.write_bytes(b"identical source")
    same.write_bytes(first.read_bytes())
    other.write_bytes(b"different source")
    for job_id, path in (("a", first), ("b", same), ("c", other)):
        db.create_job(job_id, path, {})
    term = {"source": "polity, politics, and policy", "target": "approved family"}
    db.save_book_term_decision("a", term, "approved")
    assert db.get_approved_book_terms("b") == [term]
    assert db.get_approved_book_terms("c") == []
    glossary = GlossaryManager()
    snapshot = _merge_approved_book_terms(db, "b", glossary)
    assert snapshot["applied"] == [{"source": term["source"], "target": term["target"]}]
    assert glossary.find_terms("polity, politics, and policy")
    unrelated = GlossaryManager()
    assert not _merge_approved_book_terms(db, "c", unrelated)["applied"]
    assert not unrelated.find_terms("polity, politics, and policy")


def test_curated_conflict_is_not_silently_overwritten(tmp_path: Path) -> None:
    db = JobDatabase(tmp_path / "jobs.db")
    source = tmp_path / "book.pdf"
    source.write_bytes(b"source")
    db.create_job("a", source, {})
    db.save_book_term_decision("a", {"source": "state", "target": "book choice"}, "approved")
    glossary = GlossaryManager()
    glossary.add_term("state", "curated choice")
    report = _merge_approved_book_terms(db, "a", glossary)
    assert report["conflicts"] == [{"source": "state", "target": "book choice"}]
    assert glossary.find_terms("state")[0].target == "curated choice"
    source.write_bytes(b"changed after upload")
    with pytest.raises(ValueError, match="changed"):
        db.get_approved_book_terms("a")


def test_book_approval_replaces_matching_auto_entry_only(tmp_path: Path) -> None:
    db = JobDatabase(tmp_path / "jobs.db")
    source = tmp_path / "book.pdf"
    source.write_bytes(b"book")
    db.create_job("a", source, {})
    db.save_book_term_decision(
        "a",
        {
            "source": "politics",
            "target": "chosen",
            "sense": "political process",
        },
        "approved",
    )
    glossary = GlossaryManager()
    glossary.add_term("politics", "automatic", sense="political process", is_auto=True)
    glossary.add_term("policy", "other automatic", is_auto=True)
    _merge_approved_book_terms(db, "a", glossary)
    assert [entry.target for entry in glossary.entries if entry.source == "politics"] == ["chosen"]
    assert glossary.find_terms("politics in the political process")[0].target == "chosen"
    assert glossary.find_terms("policy")[0].target == "other automatic"


def test_research_samples_later_body_and_family_candidates_are_review_only() -> None:
    paragraphs = [
        Paragraph("Opening academic paragraph " + "context " * 18),
        Paragraph("Middle academic paragraph " + "relations " * 18),
        Paragraph("Late academic paragraph " + "theory " * 18),
        Paragraph("polity, politics, and policy " * 5),
        Paragraph("polity, politics, and policy " * 5),
    ]
    document = Document(
        title="A book",
        raw_toc=["Contents " * 80],
        chapters=[
            Chapter(
                title="Chapter", sections=[Section(title="Section", level=2, paragraphs=paragraphs)]
            )
        ],
    )
    excerpt = BookResearcher._book_excerpt(document)
    assert "Middle academic paragraph" in excerpt
    assert "Late academic paragraph" in excerpt
    candidates = collect_book_term_candidates(document)
    assert candidates[0]["source"] == "polity, politics, and policy"
    assert candidates[0]["target"] == ""
    assert candidates[0]["status"] == "candidate"


def test_research_approval_defaults_to_book_scope_and_shared_needs_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tarjomeh.web.app import create_app

    source = tmp_path / "book.pdf"
    source.write_bytes(b"book bytes")
    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("job", source, {})
    db.save_job_artifact(
        "job",
        "book_research",
        {
            "status": "completed",
            "terms": [
                {"source": "polity", "target": "", "status": "candidate"},
            ],
        },
    )
    glossary_path = tmp_path / "shared.csv"
    monkeypatch.setenv("UI_SECRET_TOKEN", "test-token")
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    headers = {"Authorization": "Bearer test-token"}
    with (
        patch("tarjomeh.jobs.database.JobDatabase", return_value=db),
        patch("tarjomeh.web.app._working_glossary_path", return_value=glossary_path),
    ):
        response = client.post(
            "/api/jobs/job/research/terms/0/approve",
            headers=headers,
            json={"target": "\u067e\u06cc\u06a9\u0631\u0647\u0654 \u0633\u06cc\u0627\u0633\u06cc"},
        )
        assert response.status_code == 200
        assert response.json["scope"] == "book"
        assert not glossary_path.exists()
        assert len(db.get_approved_book_terms("job")) == 1
        db.save_job_artifact(
            "job",
            "book_research",
            {
                "status": "completed",
                "terms": [
                    {
                        "source": "politics",
                        "target": "\u0633\u06cc\u0627\u0633\u062a\u200c\u0648\u0631\u0632\u06cc",
                        "status": "suggested",
                    },
                ],
            },
        )
        unconfirmed = client.post(
            "/api/jobs/job/research/terms/0/approve",
            headers=headers,
            json={"scope": "shared"},
        )
        assert unconfirmed.status_code == 409
        confirmed = client.post(
            "/api/jobs/job/research/terms/0/approve",
            headers=headers,
            json={"scope": "shared", "confirm_shared": True},
        )
        assert confirmed.status_code == 200
        shared = GlossaryManager()
        shared.load(glossary_path)
        assert (
            shared.find_terms("politics")[0].target
            == "\u0633\u06cc\u0627\u0633\u062a\u200c\u0648\u0631\u0632\u06cc"
        )


def test_live_audits_fail_closed_after_writing_hard_failure_evidence() -> None:
    root = Path(__file__).parents[1]
    for name in (
        "audit_tarjomeh_v1039_companion.sh",
        "audit_tarjomeh_v1039_reports.sh",
    ):
        script = (root / "scripts" / name).read_text(encoding="utf-8")
        assert 'if verdict == "FAIL":\n    raise SystemExit(2)' in script
