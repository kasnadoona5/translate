from __future__ import annotations

from types import SimpleNamespace

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import (
    _objective_unmatched_readability_issues,
    _promote_objective_readability_issues,
    _readability_review_eligible,
)
from tarjomeh.core.prompts import (
    GENERAL_EDITORIAL_CONTRACT,
    PERSIAN_READABILITY_REVIEW_PROMPT,
)
from tarjomeh.memory.manager import MemoryManager, _style_sample_quality
from tarjomeh.persian.typography import PersianTypographer


def _issue(
    rationale: str,
    *,
    severity: str = "minor",
    current: str = "تراکمی میانجی‌شده",
    suggested: str = "تراکمی که میانجی‌گری شده است",
) -> dict[str, str]:
    return {
        "severity": severity,
        "current_persian_quote": current,
        "suggested_correction": suggested,
        "rationale": rationale,
    }


def test_minor_objective_grammar_routes_despite_reviewer_severity() -> None:
    attachment = _issue(
        "The modifier stack creates broken grammar and unclear attachment."
    )
    zwnj = _issue(
        "Malformed ZWNJ spacing welds two independent words.",
        current="گفتمانی‌میانجی‌شده",
        suggested="گفتمانی میانجی‌شده",
    )

    assert _objective_unmatched_readability_issues([attachment, zwnj]) == [
        attachment,
        zwnj,
    ]


def test_minor_style_and_repetition_preferences_do_not_route_automatically() -> None:
    repetition = _issue(
        "This repetition is slightly inelegant.",
        current="آینده دولت در دهه آینده",
        suggested="سرنوشت دولت در دهه آینده",
    )
    punctuation = _issue(
        "Punctuation could be stylistically smoother.",
        current="دولت، جامعه",
        suggested="دولت و جامعه",
    )

    assert _objective_unmatched_readability_issues(
        [repetition, punctuation]
    ) == []


def test_minor_grammar_advice_is_promoted_only_to_source_aware_authority() -> None:
    issue = _issue("The phrase has a modifier stack with unclear attachment.")
    translation = (
        "قدرت دولت "
        + issue["current_persian_quote"]
        + " است."
    )
    critique = SimpleNamespace(issue_details=[], issues=[])

    promoted = _promote_objective_readability_issues(
        critique,
        [issue],
        translation,
        source_text="State power is a mediated condensation.",
    )

    assert len(promoted) == 1
    advisory = promoted[0]["readability_advisory"]
    assert advisory["authority"] == "target_only_advisory"
    assert advisory["original_severity"] == "minor"
    assert advisory["routing_basis"] == "objective_minor_grammar"
    assert promoted[0]["severity"] == "major"


def test_final_body_candidate_gets_one_readability_pass_at_threshold() -> None:
    chunk = SimpleNamespace(metadata={"structural_roles": ["body"]})
    critique = SimpleNamespace(valid=True, fluency=9.0, issue_details=[])

    assert _readability_review_eligible(
        chunk,
        critique,
        9.0,
        candidate_changed=False,
        final_candidate=True,
    )


def test_structural_content_remains_excluded_from_readability_pass() -> None:
    chunk = SimpleNamespace(metadata={"structural_roles": ["bibliography"]})
    critique = SimpleNamespace(valid=True, fluency=4.0, issue_details=[])

    assert not _readability_review_eligible(
        chunk,
        critique,
        9.0,
        candidate_changed=True,
        final_candidate=True,
    )


def test_latin_scholarly_abbreviations_survive_mixed_persian_typography() -> None:
    text = (
        "برای نمونه e.g., Bartelson 1995؛ "
        "i.e., روایت دوم؛ cf. Koselleck 1985؛ "
        "ibid., 92؛ viz.، 17"
    )

    result = PersianTypographer().process(text)

    assert "e.g., Bartelson 1995" in result
    assert "i.e.," in result
    assert "cf. Koselleck 1985" in result
    assert "ibid., 92" in result
    assert "viz., 17" in result
    assert "e. g." not in result
    assert "e.g.،" not in result


def test_scholarly_protection_does_not_shield_ordinary_latin_punctuation() -> None:
    result = PersianTypographer().process(
        "این بند term, relation; effect را بررسی می‌کند."
    )

    assert "term، relation؛ effect" in result


def test_low_scoring_clean_prose_cannot_become_style_authority() -> None:
    manager = MemoryManager(TarjomehConfig())
    dense = (
        "این پژوهش با بررسی دقیق تاریخ نهادهای سیاسی و تحول روابط اجتماعی "
        "نشان می‌دهد که مفاهیم نظری تنها در پیوند با زمینه‌های مادی، گفتمانی، "
        "فرهنگی، اقتصادی، حقوقی، زمانی و مکانی و تجربه‌های متنوع کنشگران فردی "
        "و جمعی معنا پیدا می‌کنند و تحلیل روشن آن‌ها مستلزم توجه هم‌زمان به "
        "ساختارها، راهبردها، منافع، تعارض‌ها و سازوکارهاست."
    )
    quality = _style_sample_quality(dense)
    assert quality["approved"] is True
    assert quality["score"] < 75.0

    report = manager._update_style_profile(dense)

    assert report["accepted"] is False
    assert report["minimum_score"] == 75.0
    assert "style_score_below_floor" in report["rejections"][0]["reasons"]
    assert manager.style_samples == []
    assert manager.style_profile == ""


def test_prompt_contract_keeps_conceptual_families_and_attachment_source_aware() -> None:
    assert "recurring coordinated conceptual series" in GENERAL_EDITORIAL_CONTRACT
    assert "Advisory context remains revisable" in GENERAL_EDITORIAL_CONTRACT
    assert "interrupts the dependency" in PERSIAN_READABILITY_REVIEW_PROMPT
    assert "ZWNJ has incorrectly welded" in PERSIAN_READABILITY_REVIEW_PROMPT
