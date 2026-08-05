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
from tarjomeh.persian.orthography import apply_safe_persian_orthography
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
        "سرمایهداری و الگوریتمها در صورت بندی حاشیهایتر "
        "بررسی شدند (Marx 1973, 408)."
    )
    normalized, edits = apply_safe_persian_orthography(original)
    assert normalized == (
        "سرمایه‌داری و الگوریتم‌ها در صورت‌بندی حاشیه‌ای‌تر "
        "بررسی شدند (Marx 1973, 408)."
    )
    assert sum(edit["count"] for edit in edits) == 5


def test_publication_original_moves_from_lowercase_concept_to_title() -> None:
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(
        index=0,
        source_text=(
            "The concept politics of operations matters. "
            "We introduce The Politics of Operations."
        ),
        translated_text=(
            "مفهوم سیاست عملیات (The Politics of Operations) مهم است. "
            "کتاب سیاست عملیات معرفی می‌شود."
        ),
    )])
    typographer = PersianTypographer({
        "convert_numerals": False,
        "normalize_zwnj": False,
        "fix_punctuation": False,
    })
    report = ensure_inline_proper_noun_originals(
        document,
        {"The Politics of Operations": "سیاست عملیات"},
        typographer,
        {"The Politics of Operations": "publication"},
        return_report=True,
    )
    text = document.paragraphs[0].translated_text
    assert text.count("(The Politics of Operations)") == 1
    assert "مفهوم سیاست عملیات مهم است" in text
    assert "کتاب سیاست عملیات (The Politics of Operations)" in text
    assert report["repositioned_count"] == 1


def test_citation_merging_requires_source_grounding() -> None:
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(
        index=0,
        source_text="Marx (1973, 408) discusses capital.",
        translated_text="مارکس (Marx) (1973, 408) سرمایه را بررسی می‌کند.",
    )])
    report = normalize_adjacent_original_citations(
        document, {"Marx": "مارکس"}
    )
    assert document.paragraphs[0].translated_text.startswith(
        "مارکس (Marx, 1973, 408)"
    )
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
            "source_quote": "contemporary capitalism",
            "current_persian_quote": "سرمایه‌داری معاصر",
            "suggested_correction": "کاپیتالیسم معاصر",
            "rationale": "Use consistent terminology.",
        }],
    }, ensure_ascii=False)
    result = TranslationCritique._parse_response(
        raw,
        "The book examines contemporary capitalism. Another sentence follows.",
        "کتاب سرمایه‌داری معاصر را بررسی می‌کند. جمله‌ای دیگر می‌آید.",
    )
    assert result.valid
    issue = result.issue_details[0]
    assert issue["source_segment_id"] == "p1:s1"
    assert issue["suggested_correction"] == "کاپیتالیسم معاصر"
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
            "source_quote": "capitalism",
            "current_persian_quote": "سرمایه‌داری",
            "suggested_correction": "سرمایهداری",
            "rationale": "Remove the half-space.",
        }],
    }, ensure_ascii=False)
    result = TranslationCritique._parse_response(
        raw, "capitalism", "سرمایه‌داری"
    )
    assert result.valid
    assert result.issue_details == []
    assert result.ignored_issue_details[0]["ignored_reason"] == (
        "no_textual_change"
    )
    assert result.ignored_issue_details[0]["suggestion_orthography_normalized"]


def test_curated_glossary_conflict_is_withheld_without_invalidating_critique() -> None:
    detail = {
        "issue_id": "mqm-test",
        "category": "terminology",
        "severity": "major",
        "source_quote": "capitalism",
        "current_persian_quote": "سرمایه‌داری",
        "suggested_correction": "کاپیتالیسم",
        "formatted": "issue",
    }
    critique = CritiqueResult(
        issues=["issue"], issue_details=[detail], valid=True
    )
    entry = SimpleNamespace(
        source="capitalism", target="سرمایه‌داری", is_auto=False
    )
    conflicts = _filter_critique_glossary_conflicts(critique, [entry])
    assert len(conflicts) == 1
    assert critique.issue_details == []
    assert critique.valid
    assert critique.ignored_issue_details[0]["ignored_reason"] == (
        "curated_glossary_conflict"
    )


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
