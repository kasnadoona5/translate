"""Source shape controls style use without altering canonical translations."""

import hashlib
from copy import deepcopy

import pytest

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.memory.manager import (
    MemoryManager,
    _style_record_is_authoritative,
    _style_record_is_prompt_safe,
    _style_source_eligibility,
)

FA = (
    "\u0627\u06cc\u0646 \u0627\u062f\u0639\u0627 \u0628\u0647 "
    "\u0634\u0648\u0627\u0647\u062f \u0631\u0648\u0634\u0646 "
    "\u0648\u0627\u0628\u0633\u062a\u0647 \u0627\u0633\u062a."
)
NAME = "\u0648\u06cc\u0644\u0633\u0648\u0646\u060c \u0622\u062f\u0631\u06cc\u0627\u0646\u0627."
SCORES = dict.fromkeys(("accuracy", "fluency", "terminology", "register"), 9.5)


def manager():
    result = MemoryManager(TarjomehConfig())
    result.quality_policy_version = 1
    return result


def record(source="A claim depends on evidence.", target=FA, role="body"):
    m = manager()
    m._update_style_profile(
        target, source_paragraphs=[source], source_paragraph_indices=[0],
        source_alignment_proven=True, paragraph_role=role, source_chunk_index=0,
        final_scores=SCORES,
    )
    return deepcopy(m.style_sample_records[0])


@pytest.mark.parametrize("source", [
    "Wilson, Adriana.", "Garcia-Marquez, Elena.", "Muller, A.",
    "Smith, J. (2020). Evidence and methods.",
    "The survey: methods and evidence / Adriana Wilson.",
])
def test_catalogue_and_reference_shapes_do_not_become_style(source):
    result = _style_source_eligibility(source, paragraph_role="body")
    assert not result["eligible"]
    assert result["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()


@pytest.mark.parametrize("source,role", [
    ("Markets change.", "body"),
    ("Cells divide.", "body"),
    ("However, evidence remains decisive.", "body"),
    ("Wilson (2020) argues that the finding is uncertain.", "body"),
    ('"Go, Alice."', "body"),
    ("Go, Alice.", "dialogue"),
    ("This preface explains the method and its limitations.", "body"),
    ("The ratio is A / B.", "body"),
    ("The ratio measures distance / time.", "body"),
    ("Published by researchers, the findings challenged earlier accounts.", "body"),
])
def test_short_prose_citations_dialogue_and_analytic_preface_remain_eligible(source, role):
    assert _style_source_eligibility(source, paragraph_role=role)["eligible"]


def test_publication_context_does_not_accept_a_mislabelled_dialogue_name():
    result = _style_source_eligibility(
        "Wilson, Adriana.", paragraph_role="dialogue", chapter_title="Publication data",
    )
    assert result["reason"] == "catalogue_name_line"


@pytest.mark.parametrize("source", [
    "First published in 2020 by Example Press.",
    "A catalogue record for this book is available from the library.",
    "The publisher has used its best efforts to ensure the URLs remain active.",
    "Typeset in 11 pt by Example Typesetting.",
])
def test_publishing_notices_do_not_replace_the_rejected_catalogue_sample(source):
    assert _style_source_eligibility(
        source, paragraph_role="body",
    )["reason"] == "publication_notice"


def test_real_mixed_catalogue_regression_does_not_change_four_memory_inputs():
    m = manager()
    source = "Copyright. All rights reserved. " * 9 + "\n\nWilson, Adriana."
    target = FA * 12 + "\n\n" + NAME
    chunk = Chunk(0, source, "Publication data", "", {
        "structural_roles": ["body", "body"], "style_body_paragraphs": [0, 1],
        "style_eligible": True,
    })
    outcome = m.update_after_translation(
        chunk, target, style_approved=True, style_evidence={"final_scores": SCORES},
    )
    assert not outcome["style_sample_added"] and not m.style_samples
    assert m.long_term.serialize()[0]["source"] == source
    assert m.long_term.serialize()[0]["translation"] == target
    assert m.short_term.serialize()[0]["translation"] == target
    assert not m.proper_nouns.all_nouns()
    assert not m.bilingual_summary.get_context()


def test_catalogue_in_mixed_chunk_does_not_hide_the_real_body_paragraph():
    m = manager()
    source = ["Copyright.", "Wilson, Adriana.", "The experiment depends on evidence."]
    target = ["\u062d\u0642\u0648\u0642 \u0646\u0634\u0631.", NAME, FA]
    report = m._update_style_profile(
        "\n\n".join(target), source_paragraphs=source,
        source_paragraph_indices=[0, 1, 2], source_alignment_proven=True,
        paragraph_role="mixed", paragraph_roles=["heading", "body", "body"],
        publication_context=True, representative=False, final_scores=SCORES,
    )
    assert report["accepted"] and len(m.style_sample_records) == 1
    saved = m.style_sample_records[0]
    assert saved["source_text"] == source[2] and saved["text"] == FA
    assert _style_record_is_authoritative(saved)
    assert any("catalogue_name_line" in item["reasons"] for item in report["rejections"])


def bad_record():
    result = record()
    result.update(source_text="Wilson, Adriana.", text=NAME, quality_score=100)
    result["source_text_hash"] = hashlib.sha256(result["source_text"].encode()).hexdigest()
    digest = hashlib.sha256(NAME.encode()).hexdigest()
    result["text_hash"] = result["canonical_paragraph_hash"] = digest
    result.pop("style_source_evidence", None)
    return result


def test_legacy_bad_sample_cannot_use_either_prompt_path_or_occupy_capacity():
    bad = bad_record()
    assert not _style_record_is_authoritative(bad)
    assert not _style_record_is_prompt_safe(bad)
    state = manager().to_dict()
    state.update(style_samples=[NAME], style_sample_records=[bad], style_profile=NAME)
    original = deepcopy(state)
    m = manager()
    m.from_dict(state)
    assert state == original
    assert len(m.style_sample_records) == 1 and m.style_sample_records[0]["text"] == NAME
    assert m.style_samples == [NAME] and not m._effective_style_samples()
    assert m._style_profile_status() == "empty"
    context = m.get_context_for_chunk(Chunk(1, "A new paragraph.", "Chapter", ""))
    assert not context.style_profile and context.references["style_profile_version"] == 0
    assert m.style_samples == [NAME]
    for index in range(5):
        m._update_style_profile(
            FA + str(index) + ".", source_paragraphs=[f"Evidence supports claim {index}."],
            source_paragraph_indices=[0], source_alignment_proven=True,
            final_scores=SCORES,
        )
    assert len(m.style_samples) == 5 and len(m.style_sample_records) == 6
    assert m.style_sample_records[0]["text"] == NAME


def test_catalogue_does_not_count_towards_establishment():
    m = manager()
    m.style_sample_records = [bad_record(), record(), record("Another claim.", FA + " " + FA)]
    m.style_samples = [item["text"] for item in m.style_sample_records]
    assert m._style_profile_status() == "warming_up"
    assert NAME not in m._render_style_profile()
    m.style_sample_records.append(record("A third claim.", FA + " " + FA + " " + FA))
    assert m._style_profile_status() == "established"


def test_below_floor_and_uncertain_alignment_still_cannot_teach_style():
    good = record()
    assert _style_record_is_authoritative(good)
    good["final_scores"]["accuracy"] = 8
    assert not _style_record_is_authoritative(good)
    good["alignment_status"] = "uncertain"
    assert not _style_record_is_prompt_safe(good)


def test_report_distinguishes_history_from_current_eligibility_without_mutation():
    from tarjomeh.core.evidence_audit import style_source_evidence

    state = {"style_sample_records": [bad_record(), record()]}
    before = deepcopy(state)
    report = style_source_evidence(state)
    assert state == before
    assert report["stored_record_count"] == 2
    assert report["source_excluded_count"] == 1
    assert report["effective_prompt_count"] == report["effective_authoritative_count"] == 1
    excluded = report["records"][0]
    assert excluded["stored_active"] and excluded["stored_representative"]
    assert excluded["source_eligibility"]["reason"] == "uncertain_catalogue_fragment"
    assert not excluded["current_prompt_eligible"]


def test_hash_tampering_cannot_gain_style_authority():
    saved = record()
    saved["source_text"] = "Wilson, Adriana."
    assert not _style_record_is_authoritative(saved)
    assert not _style_record_is_prompt_safe(saved)


@pytest.mark.parametrize("policy", [0, 1])
def test_full_capacity_replacement_does_not_overwrite_excluded_history(policy):
    m = manager()
    m.quality_policy_version = policy
    bad = bad_record()
    good = [record(f"Evidence supports claim {i}.", FA + str(i) + ".") for i in range(5)]
    for item in good:
        item.update(representative=False, fallback=True)
    m.style_sample_records = [bad, *good]
    m.style_samples = [item["text"] for item in m.style_sample_records]
    report = m._update_style_profile(
        FA + " " + FA, source_paragraphs=["A new result follows from the evidence."],
        source_paragraph_indices=[0], source_alignment_proven=True, final_scores=SCORES,
    )
    assert report["accepted"] and report["reason"] == "weaker_style_sample_replaced"
    assert m.style_sample_records[0]["source_text"] == bad["source_text"]
    assert m.style_sample_records[0]["text"] == NAME
    assert len(m.style_samples) == 5 and NAME not in m.style_samples


def test_admission_counter_uses_effective_capacity_not_raw_history():
    m = manager()
    m.style_sample_records = [bad_record()]
    m.style_samples = [NAME]
    report = m.update_after_translation(
        Chunk(1, "Evidence supports the new result.", "Chapter", "", {
            "structural_roles": ["body"], "style_body_paragraphs": [0],
        }),
        FA, style_approved=True, style_evidence={"final_scores": SCORES},
    )
    assert report["style_sample_added"]
    assert report["style_sample_policy"]["accepted"]
    assert len(m.style_sample_records) == 2 and m.style_samples == [FA]
