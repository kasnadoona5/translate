"""General reference/count and dash evidence, with over-correction controls."""

import hashlib

import pytest

from tarjomeh.core.evidence_audit import quality_safeguard_evidence
from tarjomeh.core.pipeline import audit_translation_language
from tarjomeh.memory.manager import _style_record_is_authoritative, _style_record_is_prompt_safe
from tarjomeh.quality.integrity import (
    _sentence_scoped_orphan_dash,
    repair_source_grounded_language_artifacts,
)
from tarjomeh.quality.structure_audit import announced_count_evidence, audit_payload


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize("reference", ["(فصل‌های ۲ تا ۴)", "(صفحات ۲–۴)", "(فصل ۴)"])
def test_reference_endpoint_does_not_replace_an_approach_count(reference):
    source = "The analysis combines six approaches. Chapters 2-4 develop an institutional approach."
    target = (
        "این تحلیل شش رهیافت تحلیلی را ترکیب می‌کند. "
        f"{reference} به تعریف چهارعنصری، رویکرد نهادی می‌پردازد."
    )
    assert [item.value for item in announced_count_evidence(target)] == [6]
    assert not any(
        item["classification"] == "translation_structure_mismatch"
        for item in audit_payload(source, target)["findings"]
    )


@pytest.mark.parametrize(
    "text",
    [
        "Chapter 4 develops approaches.",
        "Chapters 2-4 develop approaches.",
        "Pages 2 to 4 concern the main approaches.",
        "(4) The approaches vary.",
        "4. The approach varies.",
        "4: an approach is described.",
        "Four people discuss approaches.",
        "Six in a chapter; four approaches.",
    ],
)
def test_reference_and_clause_boundaries_do_not_invent_announcements(text):
    values = [item.value for item in announced_count_evidence(text)]
    assert values == ([4] if text == "Six in a chapter; four approaches." else [])


@pytest.mark.parametrize(
    "text", ["six analytical approaches", "(six) approaches", "شش رهیافت‌های تحلیلی"]
)
def test_actual_local_approach_count_stays_recognized(text):
    evidence = announced_count_evidence(text)
    assert len(evidence) == 1 and evidence[0].value == 6
    assert evidence[0].semantic_category == "approach"


@pytest.mark.parametrize(
    "source,target",
    [
        ("Six approaches explain the change.", "پنج رهیافت این تغییر را توضیح می‌دهند."),
        ("Three key claims follow.", "دو مدعای کلیدی در پی می‌آیند."),
        ("Two issues are considered.", "سه مسئله بررسی می‌شود."),
    ],
)
def test_actual_wrong_count_remains_blocking(source, target):
    assert any(
        item["classification"] == "translation_structure_mismatch"
        for item in audit_payload(source, target)["findings"]
    )


def test_demonstrative_chapter_topic_does_not_hide_a_quantity_mismatch():
    source = "The chapter addresses two distinct issues in detail."
    target = "این فصل سه مسئله متمایز را با جزئیات بررسی می‌کند."
    assert [item.value for item in announced_count_evidence(target)] == [3]
    assert any(
        item["classification"] == "translation_structure_mismatch"
        for item in audit_payload(source, target)["findings"]
    )
    assert not announced_count_evidence("این فصل ۳ دربارهٔ مسئله‌ای است.")


DASH_SOURCE = "People make history - their own and that of others - within institutions."
GLUED = "انسان‌ها تاریخ خود و دیگران—را در نهادها می‌سازند."


def style_record(target):
    return {
        "source_text": DASH_SOURCE,
        "text": target,
        "source_text_hash": sha(DASH_SOURCE),
        "text_hash": sha(target),
        "alignment_status": "exact_paragraph",
        "sample_scope": "complete_paragraph",
        "representative": True,
        "quality_score": 95,
        "final_scores": dict.fromkeys(("accuracy", "fluency", "terminology", "register"), 9.5),
    }


@pytest.mark.parametrize("target", [GLUED, GLUED.replace("—", " — "), GLUED.replace("—", " – ")])
def test_glued_and_spaced_orphans_have_shared_proof_and_style_veto(target):
    assert _sentence_scoped_orphan_dash(DASH_SOURCE, target) is not None
    assert audit_translation_language(DASH_SOURCE, target)["unbalanced_explanatory_dash_artifacts"]
    assert not _style_record_is_prompt_safe(style_record(target))
    assert not _style_record_is_authoritative(style_record(target))


@pytest.mark.parametrize(
    "target",
    [
        GLUED.replace("—", " "),
        "انسان‌ها تاریخ خود — و تاریخ دیگران — را در نهادها می‌سازند.",
        "انسان‌ها تاریخ خود—و تاریخ دیگران—را در نهادها می‌سازند.",
    ],
)
def test_clean_or_closed_aside_is_not_rejected_by_object_marker_spacing(target):
    assert _sentence_scoped_orphan_dash(DASH_SOURCE, target) is None
    assert not audit_translation_language(DASH_SOURCE, target)[
        "unbalanced_explanatory_dash_artifacts"
    ]
    assert _style_record_is_prompt_safe(style_record(target))
    assert _style_record_is_authoritative(style_record(target))


def test_unproven_sentence_alignment_stays_review_not_automatic_repair():
    source = DASH_SOURCE + " A second event follows."
    assert _sentence_scoped_orphan_dash(source, GLUED) is None
    assert audit_translation_language(source, GLUED)["unbalanced_explanatory_dash_artifacts"]
    spaced = GLUED.replace("—", " — ")
    repaired, evidence = repair_source_grounded_language_artifacts(source, spaced)
    assert repaired == spaced
    assert not any(item["type"] == "orphaned_object_marker_dash" for item in evidence["repairs"])


@pytest.mark.parametrize(
    "target",
    [
        "انسان‌ها تاریخ خود–و تاریخ دیگران—را در نهادها می‌سازند.",
        GLUED + " رویداد بعدی–رخ می‌دهد.",
    ],
)
def test_other_glued_or_mixed_dash_evidence_prevents_a_guessed_repair(target):
    source = DASH_SOURCE + (" A second event follows." if ". " in target else "")
    assert _sentence_scoped_orphan_dash(source, target) is None
    assert repair_source_grounded_language_artifacts(source, target)[0] == target


@pytest.mark.parametrize("target", [GLUED, GLUED.replace("—", " — ")])
def test_proven_repair_removes_only_the_unique_orphan_and_is_idempotent(target):
    repaired, evidence = repair_source_grounded_language_artifacts(DASH_SOURCE, target)
    assert repaired == GLUED.replace("—", " ")
    assert (
        len([item for item in evidence["repairs"] if item["type"] == "orphaned_object_marker_dash"])
        == 1
    )
    assert repair_source_grounded_language_artifacts(DASH_SOURCE, repaired)[0] == repaired


def audit(entries, events=()):
    return quality_safeguard_evidence({}, {}, {}, [], list(events), {"entries": entries})


def snapshot(text, component="style"):
    return f"{component}:{sha(text)}", {
        "component": component,
        "text": text,
        "sha256": sha(text),
        "chars": len(text),
        "complete": True,
        "api_key": "must-not-export",
    }


def test_audit_exports_exact_bounded_allowlisted_component_text():
    key, entry = snapshot("Complete aligned style evidence.")
    result = audit({key: entry})
    exported = result["prompt_component_evidence"]["entries"][key]
    assert exported["text"] == entry["text"] and exported["sha256"] == sha(exported["text"])
    assert "api_key" not in exported and not result["invalid_prompt_component_evidence"]


def test_malformed_entry_collection_cannot_pass_as_a_legacy_absence():
    result = quality_safeguard_evidence({}, {}, {}, [], [], {"entries": []})
    assert "malformed_component_entries" in result["invalid_prompt_component_evidence"]


def test_reference_character_count_must_match_the_saved_entry():
    key, entry = snapshot("Complete aligned style evidence.")
    events = [
        {
            "event_type": "translation_prompt_composition",
            "chunk_index": 0,
            "payload": {
                "component_evidence": {
                    "style": {
                        "artifact_entry": key,
                        "available": True,
                        "sha256": entry["sha256"],
                        "chars": entry["chars"] + 1,
                    }
                }
            },
        }
    ]
    assert key in audit({key: entry}, events)["invalid_prompt_component_evidence"]


@pytest.mark.parametrize(
    "text",
    [
        "Authorization: Bearer credential",
        "api_key=credential",
        "https://user:password@example.invalid/",
        "https://example.invalid/?token=credential",
        "password: credential",
    ],
)
def test_secret_bearing_text_is_unavailable_not_redacted_and_claimed_exact(text):
    key, entry = snapshot(text)
    result = audit({key: entry})
    evidence = result["prompt_component_evidence"]
    assert key not in evidence["entries"]
    assert evidence["withheld"][key]["reason"] == "possible_secret_bearing_text"
    assert "text" not in evidence["withheld"][key]


@pytest.mark.parametrize(
    "component,text", [("source", "Not an allowed component"), ("style", "x" * 12001)]
)
def test_out_of_contract_evidence_cannot_be_exported_or_pass(component, text):
    key, entry = snapshot(text, component)
    result = audit({key: entry})
    assert key in result["invalid_prompt_component_evidence"]
    assert key not in result["prompt_component_evidence"]["entries"]


def test_dangling_or_malformed_reference_is_failure_not_an_audit_crash():
    events = [
        {
            "event_type": "translation_prompt_composition",
            "chunk_index": 0,
            "payload": {
                "component_evidence": {
                    "style": {"available": True, "artifact_entry": "missing"},
                    "glossary": "malformed",
                }
            },
        }
    ]
    result = audit({}, events)
    assert result["invalid_prompt_component_evidence"]


def test_term_sample_coverage_is_report_only_and_missing_options_are_visible():
    source = "A control network preserves pressure."
    result = quality_safeguard_evidence(
        {
            "terms": [
                {"source": "control network", "target": "شبکهٔ کنترل"},
                {"source": "apparatus", "target": ""},
            ]
        },
        {},
        {},
        [],
        [],
        {},
        {"text": source, "sha256": sha(source)},
    )
    coverage = result["term_sample_coverage"]
    assert coverage["sample_available"] and coverage["candidate_count"] == 2
    assert coverage["entries"][0]["lexically_present_in_saved_sample"]
    assert coverage["entries"][0]["persian_option_available"]
    assert not coverage["entries"][1]["lexically_present_in_saved_sample"]
    assert not coverage["entries"][1]["persian_option_available"]


def test_invalid_extraction_hash_does_not_claim_sample_presence():
    result = quality_safeguard_evidence(
        {"terms": [{"source": "control network"}]},
        {},
        {},
        [],
        [],
        {},
        {"text": "control network", "sha256": "incorrect"},
    )
    assert result["invalid_auto_extraction_evidence"]
    assert result["term_sample_coverage"]["entries"][0]["lexically_present_in_saved_sample"] is None
