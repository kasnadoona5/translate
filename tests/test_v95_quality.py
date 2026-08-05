from __future__ import annotations

import json
import asyncio
from types import SimpleNamespace

from tarjomeh.core.pipeline import (
    _filter_critique_glossary_conflicts,
    _high_risk_concepts,
)
from tarjomeh.core.term_notes import (
    ensure_inline_proper_noun_originals,
    normalize_adjacent_original_citations,
)
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.persian.orthography import (
    apply_safe_persian_orthography,
    orthography_issue_count,
)
from tarjomeh.persian.typography import PersianTypographer
from tarjomeh.quality.critique import CritiqueResult, TranslationCritique
from tarjomeh.quality.grounding import indexed_source, resolve_source_segment


class RecordingLLM:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.prompt = ""
        self.operation = ""

    def set_operation(self, operation: str) -> None:
        self.operation = operation

    async def chat(self, prompt: str) -> str:
        self.prompt = prompt
        return json.dumps(self.response, ensure_ascii=False)


def test_safe_orthography_repairs_only_known_noncanonical_forms() -> None:
    original = (
        "او می نویسد و کتاب ها را در خانه ای تر کنار تابلوئی بررسی کرد "
        "(Author 2018, 12)."
    )
    normalized, edits = apply_safe_persian_orthography(original)
    assert normalized == (
        "او می‌نویسد و کتاب‌ها را در خانه‌ای‌تر کنار تابلویی بررسی کرد "
        "(Author 2018, 12)."
    )
    assert sum(edit["count"] for edit in edits) == 5


def test_safe_orthography_preserves_plural_hay_and_repairs_old_corruption() -> None:
    preserved = (
        "نظام‌های اجتماعی و عملیات‌های سرمایه؛ پیامدهای نظری، "
        "تنهایی، رهایی، نهایی، بهایی، خانه‌ای و خانهای"
    )
    assert apply_safe_persian_orthography(preserved) == (preserved, [])
    assert orthography_issue_count(preserved) == 0

    malformed = "نظام‌ه‌ای اجتماعی و عملیات‌ه‌ای سرمایه"
    repaired, edits = apply_safe_persian_orthography(malformed)
    assert repaired == "نظام‌های اجتماعی و عملیات‌های سرمایه"
    assert sum(edit["count"] for edit in edits) == 2
    assert orthography_issue_count(repaired) == 0
    assert apply_safe_persian_orthography(repaired) == (repaired, [])


def test_safe_orthography_requires_whitespace_for_indefinite_heh() -> None:
    original = "خانه ای، نکته‌ ای، پیامدهای، تنهایی و خانهای"
    normalized, edits = apply_safe_persian_orthography(original)

    assert normalized == "خانه‌ای، نکته‌ای، پیامدهای، تنهایی و خانهای"
    assert [edit["rule_id"] for edit in edits] == [
        "separated_final_heh_indefinite_zwnj"
    ]
    assert edits[0]["count"] == 2


def test_publication_original_moves_from_lowercase_concept_to_title() -> None:
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(
        index=0,
        source_text=(
            "The concept methods of inquiry matters. "
            "We introduce Methods of Inquiry."
        ),
        translated_text=(
            "مفهوم روش‌های پژوهش (Methods of Inquiry) مهم است. "
            "کتاب روش‌های پژوهش معرفی می‌شود."
        ),
    )])
    typographer = PersianTypographer({
        "convert_numerals": False,
        "normalize_zwnj": False,
        "fix_punctuation": False,
    })
    report = ensure_inline_proper_noun_originals(
        document,
        {"Methods of Inquiry": "روش‌های پژوهش"},
        typographer,
        {"Methods of Inquiry": "publication"},
        return_report=True,
    )
    text = document.paragraphs[0].translated_text
    assert text.count("(Methods of Inquiry)") == 1
    assert "مفهوم روش‌های پژوهش مهم است" in text
    assert "کتاب روش‌های پژوهش (Methods of Inquiry)" in text
    assert report["repositioned_count"] == 1


def test_citation_merging_requires_source_grounding() -> None:
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(
        index=0,
        source_text="Arendt (1958, 24) discusses judgment.",
        translated_text="آرنت (Arendt) (1958, 24) داوری را بررسی می‌کند.",
    )])
    report = normalize_adjacent_original_citations(
        document, {"Arendt": "آرنت"}
    )
    assert document.paragraphs[0].translated_text.startswith(
        "آرنت (Arendt, 1958, 24)"
    )
    assert report["normalized_count"] == 1


def test_bare_author_year_is_merged_only_when_source_grounded() -> None:
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(
        index=0,
        source_text="Rahman 2012 discusses the archive.",
        translated_text="رحمان (Rahman) 2012 بایگانی را بررسی می‌کند.",
    )])
    report = normalize_adjacent_original_citations(
        document, {"Rahman": "رحمان"}
    )
    assert "رحمان (Rahman, 2012)" in document.paragraphs[0].translated_text
    assert report["normalized_count"] == 1


def test_grounded_critique_infers_segment_and_normalizes_suggestion() -> None:
    raw = json.dumps({
        "scores": {
            "accuracy": 8,
            "fluency": 8,
            "terminology": 7,
            "register": 9,
        },
        "overall": 8,
        "issues": [{
            "category": "terminology",
            "severity": "major",
            "confidence": 0.9,
            "source_quote": "institutional authority",
            "current_persian_quote": "اقتدار نهادی",
            "suggested_correction": "مرجعیت نهادی",
            "rationale": "Use consistent terminology.",
        }],
    }, ensure_ascii=False)
    result = TranslationCritique._parse_response(
        raw,
        "The report examines institutional authority. Another sentence follows.",
        "گزارش اقتدار نهادی را بررسی می‌کند. جمله‌ای دیگر می‌آید.",
    )
    assert result.valid
    issue = result.issue_details[0]
    assert issue["source_segment_id"] == "p1:s1"
    assert issue["suggested_correction"] == "مرجعیت نهادی"
    assert _high_risk_concepts(result)


def test_orthography_only_critic_regression_becomes_noop() -> None:
    raw = json.dumps({
        "scores": {
            "accuracy": 9,
            "fluency": 9,
            "terminology": 8,
            "register": 9,
        },
        "issues": [{
            "category": "typography",
            "severity": "minor",
            "confidence": 0.9,
            "source_quote": "writes",
            "current_persian_quote": "می‌نویسد",
            "suggested_correction": "می نویسد",
            "rationale": "Use a regular space.",
        }],
    }, ensure_ascii=False)
    result = TranslationCritique._parse_response(
        raw, "writes", "می‌نویسد"
    )
    assert result.valid
    assert result.issue_details == []
    assert result.ignored_issue_details[0]["ignored_reason"] == (
        "no_textual_change"
    )
    assert result.ignored_issue_details[0]["suggestion_orthography_normalized"]


def test_terminology_zwnj_variant_is_not_a_semantic_violation() -> None:
    raw = json.dumps({
        "scores": {
            "accuracy": 9,
            "fluency": 9,
            "terminology": 8,
            "register": 9,
        },
        "issues": [{
            "category": "terminology",
            "severity": "major",
            "confidence": 0.95,
            "source_quote": "capitalism",
            "current_persian_quote": "سرمایه‌داری",
            "suggested_correction": "سرمایهداری",
            "rationale": "Match the glossary spacing.",
        }],
    }, ensure_ascii=False)

    result = TranslationCritique._parse_response(
        raw, "capitalism", "سرمایه‌داری"
    )

    assert result.valid
    assert result.issue_details == []
    assert result.ignored_issue_details[0]["ignored_reason"] == (
        "terminology_orthography_equivalent"
    )


def test_terminology_semantic_change_remains_actionable() -> None:
    raw = json.dumps({
        "scores": {
            "accuracy": 8,
            "fluency": 9,
            "terminology": 7,
            "register": 9,
        },
        "issues": [{
            "category": "terminology",
            "severity": "major",
            "confidence": 0.95,
            "source_quote": "capitalism",
            "current_persian_quote": "سرمایه‌داری",
            "suggested_correction": "اقتصاد بازار",
            "rationale": "A different lexical rendering is required.",
        }],
    }, ensure_ascii=False)

    result = TranslationCritique._parse_response(
        raw, "capitalism", "سرمایه‌داری"
    )

    assert result.valid
    assert len(result.issue_details) == 1


def test_curated_glossary_conflict_is_withheld_without_invalidating_critique() -> None:
    detail = {
        "issue_id": "mqm-test",
        "category": "terminology",
        "severity": "major",
        "source_quote": "institution",
        "current_persian_quote": "نهاد",
        "suggested_correction": "مؤسسه",
        "formatted": "issue",
    }
    critique = CritiqueResult(
        issues=["issue"], issue_details=[detail], valid=True
    )
    entry = SimpleNamespace(
        source="institution", target="نهاد", is_auto=False
    )
    conflicts = _filter_critique_glossary_conflicts(critique, [entry])
    assert len(conflicts) == 1
    assert critique.issue_details == []
    assert critique.valid
    assert critique.ignored_issue_details[0]["ignored_reason"] == (
        "curated_glossary_conflict"
    )


def test_auto_glossary_conflict_is_protected_only_in_mandatory_mode() -> None:
    detail = {
        "issue_id": "mqm-auto",
        "category": "terminology",
        "severity": "major",
        "source_quote": "capitalism",
        "current_persian_quote": "سرمایه‌داری",
        "suggested_correction": "اقتصاد بازار",
        "formatted": "issue",
    }
    entry = SimpleNamespace(
        source="capitalism", target="سرمایهداری", is_auto=True
    )
    advisory = CritiqueResult(
        issues=["issue"], issue_details=[dict(detail)], valid=True
    )
    mandatory = CritiqueResult(
        issues=["issue"], issue_details=[dict(detail)], valid=True
    )

    assert _filter_critique_glossary_conflicts(advisory, [entry]) == []
    conflicts = _filter_critique_glossary_conflicts(
        mandatory, [entry], include_auto=True
    )

    assert len(conflicts) == 1
    assert mandatory.issue_details == []


def test_critic_source_is_indexed_once_without_extra_llm_call() -> None:
    llm = RecordingLLM({
        "scores": {
            "accuracy": 10,
            "fluency": 10,
            "terminology": 10,
            "register": 10,
        },
        "overall": 10,
        "issues": [],
    })
    source = "First sentence. Second sentence."
    result = asyncio.run(TranslationCritique(llm).critique(source, "ترجمه."))
    assert result.valid
    assert llm.operation == "critique"
    assert "[p1:s1] First sentence." in llm.prompt
    assert "[p1:s2] Second sentence." in llm.prompt
    assert llm.prompt.count("First sentence.") == 1
    assert indexed_source(source).count("[p1:s") == 2
    assert resolve_source_segment(source, "Second sentence.")[0] == "p1:s2"
    corrected, warnings = resolve_source_segment(
        source, "Second sentence.", "p9:s9"
    )
    assert corrected == "p1:s2"
    assert warnings == ["issue_source_segment_id_corrected"]


def test_web_recovery_controls_follow_runtime_defaults() -> None:
    from tarjomeh.web.app import create_app

    app = create_app()
    app.config["TESTING"] = True
    response = app.test_client().get("/")
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Critic recovery allowance" in page
    assert 'id="cfgCriticRecoveryTokens"' in page
    assert 'value="50000"' in page
    assert "Optional 24K recovery model or combo" not in page


def test_web_critic_allowance_never_displays_below_effective_floor() -> None:
    from tarjomeh.web.app import create_app

    app = create_app()
    app.config["TESTING"] = True
    runtime = app.config["TARJOMEH_CONFIG"]
    runtime.llm.critic.recovery_max_tokens = 24000
    runtime.llm.recovery.predictive_min_tokens = 50000

    page = app.test_client().get("/").get_data(as_text=True)
    assert 'id="cfgCriticRecoveryTokens"' in page
    assert 'value="50000"' in page
