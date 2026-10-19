import copy

import pytest

from tarjomeh.context.book_researcher import BookResearcher
from tarjomeh.core.pipeline import _research_context_for_memory

TITLE = "The Archive: Past, Present, Future"
EVIDENCE = [{"kind": "first_publication", "year": 2005, "quote": "First published in 2005"}]


@pytest.mark.parametrize("date", ["2004/2005", "2004 or 2005", "2004-2005", "2004"])
def test_title_date_is_identity_bound_and_only_annotation_is_removed(date):
    context = f"{TITLE} ({date}) is authored by Nora Vale. It studies institutions."
    text, conflicts = BookResearcher._reconcile_publication_context(
        context, EVIDENCE, book_title=TITLE
    )
    assert text == f"{TITLE} is authored by Nora Vale. It studies institutions."
    assert len(conflicts) == 1 and conflicts[0]["years"]
    assert conflicts[0]["source_evidence"] == EVIDENCE


def test_exact_single_date_kept_but_other_metadata_does_not_prove_first_edition():
    context = f"{TITLE} (2005) is a study."
    assert BookResearcher._reconcile_publication_context(context, EVIDENCE, book_title=TITLE) == (
        context,
        [],
    )
    assert BookResearcher._reconcile_publication_context(
        context, [{"kind": "copyright", "year": 2005}], book_title=TITLE
    )[1]


@pytest.mark.parametrize(
    "context",
    [
        "A revolution began in 1848 (Vale, 2005).",
        "An earlier work The Archive: Past, Present, Future (2004/2005) influenced this study.",
        "An earlier work, The Archive: Past, Present, Future (2004/2005) influenced it.",
        "A Different Archive (2004/2005) is discussed here.",
        "The Archive: Past, Present, Futures (2004/2005) is a different title.",
    ],
)
def test_other_books_citations_and_historical_dates_are_not_changed(context):
    assert BookResearcher._reconcile_publication_context(context, EVIDENCE, book_title=TITLE) == (
        context,
        [],
    )


def test_legacy_unidentified_title_is_not_guessed():
    text = f"{TITLE} (2004/2005) studies institutions."
    assert BookResearcher._reconcile_publication_context(text, EVIDENCE) == (text, [])


def test_cached_projection_keeps_raw_artifact_and_removes_only_unsafe_context_date():
    artifact = {
        "status": "completed",
        "book_context": f"{TITLE} (2004/2005) studies institutions.",
        "raw_book_context": "raw evidence unchanged",
        "terms": [],
    }
    before = copy.deepcopy(artifact)
    projection = BookResearcher.project_cached_context(
        artifact, book_title=TITLE, publication_evidence=EVIDENCE
    )
    assert artifact == before
    assert projection["new_llm_calls"] == 0 and projection["publication_conflicts"]
    projected = {**artifact, "current_context_projection": projection}
    assert _research_context_for_memory(projected) == f"{TITLE} studies institutions."


def test_pipeline_resume_projects_cached_research_without_research_or_raw_rewrite():
    from types import SimpleNamespace

    from tarjomeh.core.pipeline import TranslationPipeline
    from tarjomeh.parsers.base import Chapter, Document, Paragraph, Section

    source = Document(TITLE, chapters=[Chapter("Copyright", sections=[Section("", 2, [
        Paragraph("First published in 2005", metadata={"structure_role": "copyright"}),
    ])])])
    artifact = {"status": "completed", "source_term_candidates_version": 1, "terms": [],
                "book_context": f"{TITLE} (2004/2005) studies institutions."}
    original = copy.deepcopy(artifact)
    saved = {}
    pipeline = SimpleNamespace(
        config=SimpleNamespace(translation=SimpleNamespace(enable_book_research=True)),
        db=SimpleNamespace(get_job_artifact=lambda job, key: copy.deepcopy(artifact),
                           save_job_artifact=lambda job, key, value: saved.update({key: value})),
    )
    projected = TranslationPipeline._prepare_book_research(pipeline, "synthetic", source)
    assert artifact == original and set(saved) == {"research_context_projection"}
    assert _research_context_for_memory(projected) == f"{TITLE} studies institutions."
    assert _research_context_for_memory(projected, book_title=TITLE) == (
        f"{TITLE} studies institutions."
    )
    assert saved["research_context_projection"]["new_llm_calls"] == 0


def test_direct_retranslation_context_also_rechecks_legacy_cached_dates():
    artifact = {"status": "completed", "book_context": f"{TITLE} (2004/2005) is a study.",
                "publication_evidence": EVIDENCE, "terms": []}
    before = copy.deepcopy(artifact)
    assert _research_context_for_memory(artifact, book_title=TITLE) == f"{TITLE} is a study."
    assert artifact == before


def test_common_single_word_title_is_not_enough_to_delete_a_date():
    context = "Politics (1848) influenced the movement."
    assert BookResearcher._reconcile_publication_context(
        context, EVIDENCE, book_title="Politics"
    ) == (context, [])


def test_current_source_evidence_takes_precedence_over_stale_research_evidence():
    artifact = {"status": "completed", "book_context": f"{TITLE} (2005) is a study.",
                "publication_evidence": [{"kind": "first_publication", "year": 2004}],
                "terms": []}
    artifact["current_context_projection"] = BookResearcher.project_cached_context(
        artifact, book_title=TITLE, publication_evidence=EVIDENCE
    )
    before = copy.deepcopy(artifact)
    assert _research_context_for_memory(artifact, book_title=TITLE) == (
        f"{TITLE} (2005) is a study."
    )
    assert artifact == before
