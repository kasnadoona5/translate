"""Source-first fixes with general positive and over-correction controls."""

import hashlib

import pytest

from tarjomeh.core.pipeline import audit_translation_language
from tarjomeh.core.render_identity import proven_inline_original_spans
from tarjomeh.glossary.book_review import check_reviewed_book_terms, resolve_reviewed_book_terms
from tarjomeh.memory.manager import _style_record_is_authoritative, _style_record_is_prompt_safe
from tarjomeh.memory.proper_nouns import (
    ProperNouns,
    automatic_terminology_risk_reasons,
    is_reusable_terminology_mapping,
)
from tarjomeh.quality.integrity import parenthesis_artifacts


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def term_fixture(scope="evidence_paragraph"):
    source = "A control network connects to a second control network."
    target = "\u0634\u0628\u06a9\u0647\u0654 \u06a9\u0646\u062a\u0631\u0644"
    metadata = {
        "source_paragraph_spans": [[0, len(source)]],
        "source_paragraph_hashes": [sha(source)],
        "paragraph_indices": [5],
        "structural_roles": ["body"],
    }
    decision = {
        "source": "control network",
        "target": target,
        "status": "approved",
        "scope_mode": scope,
        "source_evidence_sha256": sha(source),
    }
    return source, target, metadata, decision


@pytest.mark.parametrize("scope", ["evidence_paragraph", "all_body"])
def test_all_legacy_scopes_record_repeats_without_claiming_verification(scope):
    source, target, metadata, decision = term_fixture(scope)
    matches, reviews = resolve_reviewed_book_terms(source, metadata, 0, [decision])
    assert matches[0]["source_occurrence_count"] == 2
    assert reviews[0]["reason"] == "multiple_occurrences_unverifiable"
    report, reviews = check_reviewed_book_terms(target + ".", source, metadata, matches)
    assert report.total_checked == 0
    assert "rendering_count_short" in reviews[0]["reasons"]


def test_missing_legacy_count_is_recomputed_and_two_renderings_still_need_review():
    source, target, metadata, decision = term_fixture()
    matches, _ = resolve_reviewed_book_terms(source, metadata, 0, [decision])
    del matches[0]["source_occurrence_count"]
    report, reviews = check_reviewed_book_terms(
        target + " \u0648 " + target, source, metadata, matches
    )
    assert report.total_checked == 0 and not report.violations
    assert reviews[0]["source_occurrence_count"] == 2
    assert reviews[0]["target_rendering_count"] == 2


@pytest.mark.parametrize("count", [1, "2", True, 0])
def test_inconsistent_count_never_becomes_checked(count):
    source, target, metadata, decision = term_fixture()
    matches, _ = resolve_reviewed_book_terms(source, metadata, 0, [decision])
    matches[0]["source_occurrence_count"] = count
    report, reviews = check_reviewed_book_terms(target, source, metadata, matches)
    assert report.total_checked == 0
    assert reviews[0]["reason"] == "source_occurrence_count_mismatch"


def test_one_approved_occurrence_keeps_its_lexical_check():
    source, target, _, decision = term_fixture()
    source = "This control network preserves all measurements."
    metadata = {"structural_roles": ["body"]}
    decision["source_evidence_sha256"] = sha(source)
    matches, reviews = resolve_reviewed_book_terms(source, metadata, 0, [decision])
    report, compliance_reviews = check_reviewed_book_terms(target, source, metadata, matches)
    assert not reviews and not compliance_reviews and report.total_checked == 1


@pytest.mark.parametrize("source", ["to make a difference", "to reduce uncertainty"])
def test_infinitival_correction_is_contextual_not_erased(source):
    target = "\u062c\u0647\u062a \u062a\u0623\u062b\u06cc\u0631\u06af\u0630\u0627\u0631\u06cc"
    assert not is_reusable_terminology_mapping(source, target)
    assert "bare_infinitival_passage_fragment" in automatic_terminology_risk_reasons(source, target)
    nouns = ProperNouns()
    nouns.add_noun(
        source,
        target,
        category="term",
        provenance="accepted_correction",
        context_independent=True,
        alignment_status="exact_local",
        evidence_key="a",
    )
    assert source not in nouns.get_context(source_text=source)
    assert nouns.serialize()["nouns"][source] == target


@pytest.mark.parametrize("source", ["state building", "control theory", "right to information"])
def test_nominal_terms_are_not_rejected_for_containing_to(source):
    assert is_reusable_terminology_mapping(
        source, "\u0646\u0638\u0631\u06cc\u0647\u0654 \u06a9\u0646\u062a\u0631\u0644"
    )


def test_explicitly_curated_infinitive_keeps_human_authority():
    nouns = ProperNouns()
    nouns.add_noun(
        "to reduce uncertainty",
        "\u06a9\u0627\u0647\u0634 \u0639\u062f\u0645 \u0642\u0637\u0639\u06cc\u062a",
        category="term",
        provenance="curated_glossary",
        context_independent=True,
    )
    assert nouns.authority_class_for("to reduce uncertainty") == "canonical_curated"


def test_separate_single_dashes_are_not_paired_across_sentences():
    source = "One change - its cause - is documented. Another - its effect is uncertain."
    target = (
        "\u06cc\u06a9 \u062a\u063a\u06cc\u06cc\u0631 \u2014 \u0639\u0644\u062a "
        "\u0622\u0646 \u2014 \u062b\u0628\u062a \u0634\u062f. \u062f\u06cc"
        "\u06af\u0631\u06cc \u2014 \u0627\u062b\u0631\u0634 \u0646\u0627\u0645"
        "\u0639\u0644\u0648\u0645 \u0627\u0633\u062a."
    )
    report = audit_translation_language(source, target)
    assert not report["unbalanced_explanatory_dash_artifacts"]
    source = "One change - its cause is clear. Another - its effect is uncertain."
    report = audit_translation_language(source, target)
    assert not report["unbalanced_explanatory_dash_artifacts"]
    assert not report["explanatory_dash_review"]


def anchor_fixture():
    source = "Several researchers (Ada North and Ben West) studied the network."
    target = (
        "\u067e\u0698\u0648\u0647\u0634\u06af\u0631\u0627\u0646 (\u0622\u062f"
        "\u0627 \u0646\u0648\u0631\u062b (Ada North) \u0648 \u0628\u0646 \u0648"
        "\u0633\u062a (Ben West)) \u0634\u0628\u06a9\u0647 \u0631\u0627 \u0628"
        "\u0631\u0631\u0633\u06cc \u06a9\u0631\u062f\u0646\u062f."
    )
    context = {
        "authorized": {
            "Ada North": "\u0622\u062f\u0627 \u0646\u0648\u0631\u062b",
            "Ben West": "\u0628\u0646 \u0648\u0633\u062a",
        }
    }
    return source, target, context


def test_authorized_nested_annotations_preserve_source_list_parentheses():
    source, target, context = anchor_fixture()
    assert len(proven_inline_original_spans(source, target, context)) == 2
    assert not parenthesis_artifacts(source, target, anchor_context=context)
    assert parenthesis_artifacts(source, target)
    assert not audit_translation_language(source, target, anchor_context=context)[
        "parenthesis_artifacts"
    ]


@pytest.mark.parametrize("aside", ["", " (the network)"])
def test_source_authorized_original_cannot_excuse_a_new_foreign_wrapper(aside):
    source = "Ada North studied" + aside + " carefully."
    name = "\u0622\u062f\u0627 \u0646\u0648\u0631\u062b"
    target = "(" + name + " (Ada North)) \u0628\u0631\u0631\u0633\u06cc \u06a9\u0631\u062f."
    context = {"authorized": {"Ada North": name}}
    assert parenthesis_artifacts(source, target, anchor_context=context)


def test_more_nested_annotations_than_source_list_occurrences_remain_review():
    source = "Researchers (Ada North) studied the network. Ada North returned."
    name = "\u0622\u062f\u0627 \u0646\u0648\u0631\u062b"
    target = "(" + name + " (Ada North)) (" + name + " (Ada North))"
    context = {"authorized": {"Ada North": name}}
    assert parenthesis_artifacts(source, target, anchor_context=context)


@pytest.mark.parametrize(
    "defect", ["unknown", "misplaced", "surplus", "case_surplus", "unclosed", "bracket"]
)
def test_annotation_exemption_never_hides_broken_wrappers(defect):
    source, target, context = anchor_fixture()
    if defect == "unknown":
        context = {"authorized": {}}
    elif defect == "misplaced":
        context["authorized"]["Ada North"] = "\u062f\u06cc\u06af\u0631"
    elif defect == "surplus":
        target = target.replace("(Ada North)", "(Ada North) (Ada North)")
    elif defect == "case_surplus":
        target = target.replace("(Ada North)", "(Ada North) (ada north)")
    elif defect == "unclosed":
        target = target[:-1] + "("
    else:
        target = target.replace("(Ada North)", "[(Ada North)]")
    assert parenthesis_artifacts(source, target, anchor_context=context)


def test_both_style_paths_reject_source_proven_spaced_prefix():
    source = "This (meta)theoretical account preserves the relevant distinctions."
    target = (
        "\u0627\u06cc\u0646 \u062a\u062d\u0644\u06cc\u0644 (\u0641\u0631\u0627)"
        " \u0646\u0638\u0631\u06cc \u062a\u0645\u0627\u06cc\u0632\u0647\u0627"
        "\u06cc \u0645\u0631\u0628\u0648\u0637 \u0631\u0627 \u062d\u0641\u0638 "
        "\u0645\u06cc\u200c\u06a9\u0646\u062f."
    )
    record = {
        "source_text": source,
        "text": target,
        "source_text_hash": sha(source),
        "text_hash": sha(target),
        "sample_scope": "complete_paragraph",
        "alignment_status": "exact_paragraph",
        "quality_score": 95,
        "representative": True,
        "final_scores": dict.fromkeys(("accuracy", "fluency", "terminology", "register"), 9.5),
    }
    assert not _style_record_is_prompt_safe(record)
    assert not _style_record_is_authoritative(record)
    record["text"] = target.replace("(\u0641\u0631\u0627) ", "(\u0641\u0631\u0627)")
    record["text_hash"] = sha(record["text"])
    assert _style_record_is_prompt_safe(record)
    assert _style_record_is_authoritative(record)


def test_prompt_evidence_is_exact_bounded_deduplicated_and_not_a_prompt_cap():
    from tarjomeh.core.prompt_evidence import component_evidence

    components = {
        "source": "original",
        "book_context": "Research note.",
        "style": "Complete sample.",
        "layer_1": "context only",
    }
    added, refs = component_evidence(components, {})
    assert len(added) == 2 and not refs["source"]["available"]
    assert refs["book_context"]["sha256"] == sha(components["book_context"])
    again, reused = component_evidence(components, added)
    assert not again and reused == refs
    bounded, refs = component_evidence(components, {}, max_chars=3)
    assert not bounded and refs["style"]["reason"] == "component_storage_bound"
    assert components["style"] == "Complete sample."
    _, refs = component_evidence(components, {}, max_entries=0)
    assert refs["book_context"]["reason"] == "job_storage_bound"


def test_corrupt_context_snapshot_is_not_claimed_as_exact():
    from tarjomeh.core.prompt_evidence import component_evidence

    text = "Actual research note."
    key = f"book_context:{sha(text)}"
    added, refs = component_evidence({"book_context": text}, {key: {"text": "wrong"}})
    assert added[key]["text"] == text and refs["book_context"]["available"]


def test_non_text_context_is_unavailable_not_coerced_into_invented_evidence():
    from tarjomeh.core.prompt_evidence import component_evidence

    added, refs = component_evidence({"book_context": None, "style": object()}, {})
    assert not added
    assert all(not reference["available"] for reference in refs.values())
    assert all(reference["sha256"] is None for reference in refs.values())


@pytest.mark.parametrize("quoted,malformed", [(False, False), (True, False), (True, True)])
def test_english_attestation_never_certifies_a_persian_option(quoted, malformed):
    from tarjomeh.context.book_researcher import BookResearcher
    from tarjomeh.core.config import TarjomehConfig

    persian = "\u0634\u0628\u06a9\u0647\u0654 \u06a9\u0646\u062a\u0631\u0644"
    if malformed:
        persian += "\u0640"
    url = "https://example.invalid/book"
    researcher = object.__new__(BookResearcher)
    researcher.config = TarjomehConfig()
    sources = [
        {
            "url": url,
            "title": "Example book",
            "snippet": "A control network preserves measurements."
            + (" " + persian if quoted else ""),
            "identity_evidence": {"strong_title_match": True},
        }
    ]
    term = researcher._normalise_terms(
        [{"source": "control network", "target": persian, "source_urls": [url]}],
        sources,
    )[0]
    assert term["term_supported"] and term["english_term_attested"]
    assert term["persian_option_quote_present"] is (quoted and not malformed)
    assert term["persian_semantic_accuracy"] == "unverified"
    assert term["authority"] == "advisory_context_only"


def test_report_rechecks_repeated_terms_and_does_not_trust_legacy_quote_claims():
    from tarjomeh.core.evidence_audit import quality_safeguard_evidence

    source, target, metadata, decision = term_fixture()
    result = quality_safeguard_evidence(
        {
            "terms": [
                {
                    "source": "control network",
                    "target": target,
                    "term_supported": True,
                    "persian_option_quote_present": True,
                    "term_supporting_excerpts": [{"snippet": source}],
                }
            ]
        },
        {},
        {"proposals": [decision]},
        [
            {
                "chunk_index": 0,
                "text": source,
                "translation": target,
                "status": "completed",
                "metadata": metadata,
            }
        ],
        [],
        {},
    )
    terms = result["current_occurrence_evaluation"][0]
    assert terms["lexically_checked"] == 0
    assert terms["compliance_review"][0]["source_occurrence_count"] == 2
    assert not result["research_evidence_kinds"][0]["persian_option_quote_present_rechecked"]
    assert result["current_language_evaluation"][0]["authority"].startswith("report_only")


def test_report_corrupt_snapshot_is_not_a_pass_and_missing_old_context_is_honest():
    from tarjomeh.core.evidence_audit import quality_safeguard_evidence

    result = quality_safeguard_evidence(
        {},
        {},
        {},
        [],
        [
            {
                "chunk_index": 0,
                "event_type": "translation_prompt_composition",
                "payload": {},
            }
        ],
        {"entries": {"bad": {"text": "not hash bound"}}},
    )
    assert result["invalid_prompt_component_evidence"] == ["bad"]
    assert result["historical_context_unavailable_chunks"] == [0]


def test_artifact_storage_limit_is_enforced_at_the_transaction_boundary(tmp_path):
    from tarjomeh.jobs.database import JobDatabase

    db = JobDatabase(tmp_path / "jobs.db")
    db.create_job("test", "input.pdf", {})
    first = JobDatabase(tmp_path / "jobs.db")
    second = JobDatabase(tmp_path / "jobs.db")
    assert first.merge_job_artifact_entry(
        "test", "evidence", "entries", "a", {"text": "a"}, entry_limit=1
    )
    assert (
        second.merge_job_artifact_entry(
            "test", "evidence", "entries", "b", {"text": "b"}, entry_limit=1
        )
        is False
    )
    assert db.get_job_artifact("test", "evidence")["entries"] == {"a": {"text": "a"}}
    assert second.merge_job_artifact_entry(
        "test", "evidence", "entries", "a", {"text": "new"}, entry_limit=1
    )
    assert db.merge_job_artifact_entry("test", "legacy", "entries", "a", {}) is None
