"""Source-local syntax evidence and conservative automatic name boundaries."""

import unicodedata
from copy import deepcopy

import pytest

from tarjomeh.core.pipeline import audit_translation_language
from tarjomeh.core.render_identity import proven_inline_original_spans
from tarjomeh.memory.manager import _clean_style_sample
from tarjomeh.memory.proper_nouns import (
    ProperNouns,
    is_bounded_person_name_target,
    observed_bilingual_target,
)
from tarjomeh.quality.integrity import _source_annotation_spans, parenthesis_artifacts


@pytest.mark.parametrize(
    "source,target",
    [
        ("Jean-Paul Sartre", "اندیشه ژانپل سارتر"),
        ("Jean-Jacques Rousseau", "گفتهٔ ژانژاک روسو"),
        ("Ada Vail", "گفتهٔ آداویل"),
        ("Ada Vail", "گفتۀ آداویل"),
    ],
)
def test_context_is_never_promoted_to_an_automatic_name(source, target):
    assert not observed_bilingual_target(f"به {target} ({source})", source, "person")
    assert not is_bounded_person_name_target(source, target)
    saved = {
        "nouns": {source: target}, "categories": {source: "person"},
        "provenance": {source: {"origin": "observed_translation"}},
    }
    snapshot = deepcopy(saved)
    nouns = ProperNouns()
    nouns.deserialize(saved)
    assert saved == snapshot and nouns.all_nouns() == saved["nouns"]
    assert nouns.is_context_deferred(source)
    assert source not in nouns.get_context(source_text=source)
    assert source not in nouns.inline_eligible_nouns()
    assert source not in nouns.pending_inline_originals(source)
    fresh = ProperNouns()
    fresh.add_noun(source, target, "person", provenance="observed_translation")
    assert source not in fresh.inline_eligible_nouns()


@pytest.mark.parametrize("hyphen", ["-", "\u2010", "\u2011"])
def test_compound_group_boundaries_are_preserved(hyphen):
    source = "Anne" + hyphen + "Marie Vale"
    target = "آن" + hyphen + "ماری ویل"
    assert observed_bilingual_target(
        f"گفتاری—به {target} ({source})", source, "person"
    ) == unicodedata.normalize("NFKC", target)
    assert is_bounded_person_name_target(source, target)
    # Same total number of components is not proof of the same groups.
    assert not is_bounded_person_name_target(source, "اندیشه آنماری ویل")
    assert not is_bounded_person_name_target(source, "آن ماری ویل")
    assert not is_bounded_person_name_target(source, "آن ماری-ویل")


@pytest.mark.parametrize("origin", ["curated_glossary", "approved_research"])
def test_explicit_authority_is_not_demoted_by_automatic_boundary_rule(origin):
    nouns = ProperNouns()
    nouns.deserialize({
        "nouns": {"Anne-Marie Vale": "آن ماری ویل"},
        "categories": {"Anne-Marie Vale": "person"},
        "provenance": {"Anne-Marie Vale": {"origin": origin}},
    })
    assert not nouns.is_context_deferred("Anne-Marie Vale")
    assert "Anne-Marie Vale" in nouns.inline_eligible_nouns()


SOURCE = "The researchers (Ada North, Emmerich de Vale, G. W. F. Hale) studied the network."
TARGET = (
    "پژوهشگران (آدا نورث (Ada North)، امریش دو ویل (Emmerich de Vale) "
    "و گ. و. ف. هیل (G. W. F. Hale)) شبکه را بررسی کردند."
)


def test_source_list_recognition_never_creates_memory_or_render_authority():
    nouns = ProperNouns()
    nouns.deserialize({
        "nouns": {"Emmerich de Vale": "امریش دو ویل"},
        "categories": {"Emmerich de Vale": "person"},
    })
    snapshot = deepcopy(nouns.serialize())
    assert len(_source_annotation_spans(SOURCE, TARGET)) == 3
    assert not parenthesis_artifacts(SOURCE, TARGET)
    assert not audit_translation_language(SOURCE, TARGET)["parenthesis_artifacts"]
    assert _clean_style_sample(TARGET, source=SOURCE)
    assert not proven_inline_original_spans(SOURCE, TARGET, {"authorized": {}})
    assert nouns.serialize() == snapshot
    assert nouns.is_context_deferred("Emmerich de Vale")


@pytest.mark.parametrize(
    "target",
    [
        TARGET.replace("(Ada North)", "[(Ada North)]"),
        TARGET.replace("(Ada North)", "(Ada North) (Ada North)"),
        TARGET.replace("(Ada North)", "(Ada North) (ada north)"),
        TARGET.replace("(Ada North)", "(Ada Other)"),
        TARGET.replace("(Ada North)", "(Emmerich de Vale)").replace(
            "ویل (Emmerich de Vale)", "ویل (Ada North)"
        ),
        TARGET + "(",
        TARGET + ")",
        TARGET.replace("آدا نورث (Ada North)", "(Ada North)"),
    ],
)
def test_unproved_wrappers_remain_review_even_with_authorized_names(target):
    context = {"authorized": {"Ada North": "آدا نورث"}}
    assert parenthesis_artifacts(SOURCE, target, anchor_context=context)


@pytest.mark.parametrize(
    "source", [SOURCE + " Ada North returned.", SOURCE.replace("de Vale", "de Vále")]
)
def test_ambiguous_source_occurrence_or_changed_accents_cannot_prove_list(source):
    assert parenthesis_artifacts(source, TARGET)


def test_foreign_expression_wrapper_and_outside_source_names_remain_review():
    assert parenthesis_artifacts(
        "They used raison d’état.", "از [(raison d’état)] استفاده کردند."
    )
    assert parenthesis_artifacts(
        "Ada North studied (the network).", "(آدا نورث (Ada North)) بررسی کرد."
    )
    assert parenthesis_artifacts(
        "The researchers (Ada North) studied.", "(آدا نورث (Ada North)) (آدا نورث (Ada North))"
    )


def test_entity_inventory_preserves_pipeline_import_contract():
    from tarjomeh.context.source_entities import _source_entity_inventory
    from tarjomeh.core import pipeline

    assert pipeline._source_entity_inventory is _source_entity_inventory
    assert _source_entity_inventory(SOURCE) == ["Ada North", "Emmerich de Vale", "G. W. F. Hale"]


@pytest.mark.parametrize("edge", ["گفتهٔ", "گفتۀ", "تخصصیِ"])
def test_ezafe_at_either_edge_is_not_an_automatic_name_boundary(edge):
    assert not is_bounded_person_name_target("Ada Vail", edge + " آداویل")
    assert not is_bounded_person_name_target("Ada Vail", "آداویل " + edge)


def test_separate_source_lists_cannot_be_reordered():
    source = "One group (Ada North) acts. Another group (Ben Hale) waits."
    target = "گروهی (بن هیل (Ben Hale)) کنش دارد. گروهی (آدا نورث (Ada North)) منتظر است."
    assert parenthesis_artifacts(source, target)


def test_existing_proved_single_surname_anchor_is_not_lost():
    source = "A researcher (Hale) studied the network."
    target = "پژوهشگری (هیل (Hale)) شبکه را بررسی کرد."
    context = {"authorized": {"Hale": "هیل"}}
    assert not parenthesis_artifacts(source, target, anchor_context=context)
    assert parenthesis_artifacts(source, target)
    assert parenthesis_artifacts(source, target.replace("(Hale)", "[(Hale)]"), anchor_context=context)
