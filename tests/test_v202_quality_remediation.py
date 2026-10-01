"""General controls for the post-v20 source-first remediation."""

import hashlib
import json
import subprocess
import sys

import pytest

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.context.book_researcher import BookResearcher
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import _previous_context_reference, audit_translation_language
from tarjomeh.jobs.database import JobDatabase
from tarjomeh.memory.manager import (
    MemoryManager,
    _is_paratext_style_source,
    _style_record_is_authoritative,
    _style_record_is_prompt_safe,
)
from tarjomeh.memory.proper_nouns import (
    ProperNouns,
    automatic_terminology_risk_reasons,
    is_reusable_terminology_mapping,
)
from tarjomeh.memory.short_term import ShortTermMemory
from tarjomeh.parsers.base import Chapter, Document, Paragraph, Section
from tarjomeh.quality.integrity import duplicated_comma_artifacts, repair_proven_surface_artifacts


@pytest.mark.parametrize(
    "source",
    [
        "This book was written with support from a university research fellowship. "
        "The usual disclaimers apply.",
        "The writing of this volume was undertaken during a fellowship "
        "funded by a research council.",
        "The usual disclaimers apply.",
        "I alone remain responsible for any errors.",
    ],
)
def test_funding_and_author_disclaimers_are_not_style(source):
    assert _is_paratext_style_source(source)
    target = "این جمله نمونه‌ای روشن از نثر دانشگاهی دقیق و پیوسته است."
    record = {
        "source_text": source,
        "text": target,
        "source_text_hash": hashlib.sha256(source.encode()).hexdigest(),
        "text_hash": hashlib.sha256(target.encode()).hexdigest(),
        "sample_scope": "complete_paragraph",
        "alignment_status": "exact_paragraph",
        "representative": True,
        "quality_score": 95,
        "final_scores": dict.fromkeys(("accuracy", "fluency", "terminology", "register"), 9.5),
    }
    assert not _style_record_is_prompt_safe(record)
    assert not _style_record_is_authoritative(record)


@pytest.mark.parametrize(
    "source",
    [
        "Research funded by public institutions can increase scientific capacity.",
        "This book examines support from international institutions and its political effects.",
        "This book analyzes research fellowships as a funding mechanism.",
        "The authors discuss why the usual disclaimers apply to quantitative inference.",
    ],
)
def test_analytical_funding_prose_is_not_excluded(source):
    assert not _is_paratext_style_source(source)


@pytest.mark.parametrize(
    "source", ["include any necessary credits", "control every experimental variable"]
)
def test_quantified_object_corrections_are_passage_specific(source):
    target = "ارجاعات و قدردانی‌های لازم"
    assert not is_reusable_terminology_mapping(source, target)
    assert "quantified_object_passage_fragment" in automatic_terminology_risk_reasons(
        source, target
    )
    nouns = ProperNouns()
    nouns.deserialize(
        {
            "nouns": {source: target},
            "categories": {source: "term"},
            "provenance": {
                source: {
                    "origin": "accepted_correction",
                    "authority": 80,
                    "context_independent": True,
                    "alignment_status": "exact_local",
                }
            },
        }
    )
    assert source not in nouns.get_context(source_text=source)
    assert nouns.serialize()["nouns"][source] == target


@pytest.mark.parametrize(
    "source,target",
    [
        ("control theory", "نظریهٔ کنترل"),
        ("state building", "دولت‌سازی"),
        ("credit allocation", "تخصیص اعتبار"),
        ("include operation", "عملیات شمول"),
    ],
)
def test_real_nominal_terms_survive(source, target):
    assert is_reusable_terminology_mapping(source, target)


def test_capabilities_stdout_is_a_complete_json_document():
    result = subprocess.run(
        [sys.executable, "-m", "tarjomeh.cli.main", "capabilities", "--verify", "--json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert all(manifest["behavior_probes"].values())


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _previous_identity(trust="trusted"):
    previous = "این متن پیشین است."
    row = {
        "chunk_index": 3,
        "status": "completed",
        "text": "Earlier prose.",
        "translation": previous,
    }
    policy = {"canonical_target_hash": _hash(previous), "short_term_trust": trust}
    references = {
        "short_term_identities": [
            {
                "entry": 4,
                "chunk_index": 3,
                "source_sha256": _hash(row["text"]),
                "target_sha256": _hash(previous),
                "trust": trust,
            }
        ]
    }
    return previous, row, policy, references


@pytest.mark.parametrize("trust", ["trusted", "advisory_review", "structural_only"])
def test_previous_reference_requires_complete_committed_identity(trust):
    previous, row, policy, refs = _previous_identity(trust)
    result = _previous_context_reference(previous, 3, row, policy, refs)
    assert "Layer 4 entry 4" in result and trust in result
    assert previous not in result


@pytest.mark.parametrize("field", ["source_sha256", "target_sha256", "trust", "chunk_index"])
def test_distinct_or_unknown_context_is_never_deduplicated(field):
    previous, row, policy, refs = _previous_identity()
    refs["short_term_identities"][0][field] = "different"
    assert _previous_context_reference(previous, 3, row, policy, refs) == ""


def test_uncommitted_or_ambiguous_previous_context_keeps_both_copies():
    previous, row, policy, refs = _previous_identity()
    row["status"] = "pending"
    assert _previous_context_reference(previous, 3, row, policy, refs) == ""
    row["status"] = "completed"
    refs["short_term_identities"] *= 2
    assert _previous_context_reference(previous, 3, row, policy, refs) == ""
    assert _previous_context_reference(previous, 3, row, {}, {}) == ""


def test_short_term_identity_survives_resume_and_legacy_is_unknown():
    memory = ShortTermMemory()
    memory.add("Source.", "متن.", chunk_index=3)
    assert ShortTermMemory.deserialize(memory.serialize()).get_entries()[0].chunk_index == 3
    assert (
        ShortTermMemory.deserialize([{"source": "Old.", "translation": "متن."}])
        .get_entries()[0]
        .chunk_index
        is None
    )


def test_previous_reference_points_to_complete_rendered_memory(tmp_path):
    config = TarjomehConfig()
    manager = MemoryManager(config)
    previous, row, policy, _refs = _previous_identity()
    manager.short_term.add(row["text"], previous, chunk_index=3)
    context = manager.get_context_for_chunk(
        Chunk(index=4, text="Next prose.", chapter_title="Body", section_title="")
    )
    reference = _previous_context_reference(previous, 3, row, policy, context.references)
    assert "Layer 4 entry 1" in reference
    assert context.short_term.count(previous) == 1
    assert row["text"] in context.short_term
    database = JobDatabase(tmp_path / "jobs.db")
    database.create_job("control", tmp_path / "source.txt", {})
    database.save_chunks(
        "control", [Chunk(index=3, text=row["text"], chapter_title="Body", section_title="")]
    )
    assert database.get_chunk("control", 3)["text"] == row["text"]
    assert database.get_chunk("control", 4) is None


def test_publication_metadata_distinguishes_date_kinds():
    document = Document(
        "Example",
        chapters=[
            Chapter(
                "Copyright",
                sections=[
                    Section(
                        "",
                        2,
                        [
                            Paragraph(
                                "First published in 2016 by Aurora Press\n"
                                "Copyright 2015\nReprinted in 2020"
                            ),
                            Paragraph(
                                "Another author first published in 1900, according to a review."
                            ),
                        ],
                    )
                ],
            )
        ],
    )
    evidence = BookResearcher._publication_metadata_evidence(document)
    assert {(item["kind"], item["year"]) for item in evidence} == {
        ("first_publication", 2016),
        ("copyright", 2015),
        ("reprint", 2020),
    }
    context, conflicts = BookResearcher._reconcile_publication_context(
        "First published in 2016. Copyright 2015. Reprinted in 2020. "
        "It concerns institutional change.",
        evidence,
    )
    assert not conflicts and "institutional change" in context
    context, conflicts = BookResearcher._reconcile_publication_context(
        "The book (Example Press, 2015) concerns politics. It develops a relational account.",
        evidence,
    )
    assert "2015" not in context and "relational account" in context
    assert conflicts[0]["reason"] == "publication_fact_unverified"


def test_conflicting_or_unknown_publication_claims_are_not_guessed():
    evidence = [{"kind": "first_publication", "year": 2016, "quote": "First published in 2016"}]
    context, conflicts = BookResearcher._reconcile_publication_context(
        "First published in 2015. Context survives.", evidence
    )
    assert context == "Context survives." and conflicts[0]["reason"] == "edition_date_conflict"
    context, conflicts = BookResearcher._reconcile_publication_context(
        "Published in 2015. Context survives.", []
    )
    assert context == "Context survives." and conflicts


def test_body_claim_about_another_book_is_not_edition_metadata():
    document = Document(
        "Example",
        chapters=[
            Chapter(
                "Analysis",
                sections=[
                    Section(
                        "",
                        2,
                        [
                            Paragraph("First published in 1900."),
                            Paragraph("This is a discussion of another author's book."),
                        ],
                    )
                ],
            )
        ],
    )
    assert BookResearcher._publication_metadata_evidence(document) == []


def test_unique_body_comma_repair_is_precanonical_and_idempotent():
    source = "Historical institutions, political institutions and social institutions."
    target = "نهادهای تاریخی،، نهادهای سیاسی و نهادهای اجتماعی."
    assert audit_translation_language(source, target)["duplicated_comma_count"] == 1
    result, edits = repair_proven_surface_artifacts(source, target)
    assert result == target.replace("،،", "،")
    assert edits[0]["type"] == "duplicate_comma"
    assert not audit_translation_language(source, result)["duplicated_comma_count"]
    assert repair_proven_surface_artifacts(source, result) == (result, [])


@pytest.mark.parametrize(
    "source,target,role",
    [
        ("Quoted words.", "او گفت «نهادهای تاریخی،، نهادهای سیاسی».", "body"),
        ("Quoted words.", "او گفت ‘نهادهای تاریخی،، نهادهای سیاسی’.", "body"),
        ("Quoted words.", "او گفت 'نهادهای تاریخی،، نهادهای سیاسی'.", "body"),
        ("A citation.", "این ادعا بررسی شد (Name،، 1963).", "body"),
        ("Items.", "نهادها،، روابط", "heading"),
        ("Words,, words.", "نهادها،، روابط.", "body"),
        ("One paragraph.", "نهادها،، روابط.\n\nبخش دیگر.", "body"),
        ("Three elements.", "نهادها،، روابط،، ساختارها.", "body"),
        ("A citation.", "نام،، ۱۳۶۳", "body"),
        ("A number.", "۱۳۶۳،، نهادها", "body"),
        ("Quoted words.", "» نهادها،، روابط.", "body"),
        ("Quoted words.", "نهادها،، روابط «", "body"),
    ],
)
def test_uncertain_comma_runs_remain_unchanged(source, target, role):
    assert repair_proven_surface_artifacts(source, target, structural_role=role) == (target, [])
    if ",," not in source:
        assert duplicated_comma_artifacts(source, target, structural_role=role)
