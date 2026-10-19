import copy

import pytest

from tarjomeh.memory.proper_nouns import (
    ProperNouns,
    is_usable_observed_mapping,
    observed_bilingual_target,
    organization_lexical_deferral,
)

SOURCE = "Social and Natural Science Research Council"
CLEAN = (
    "\u0634\u0648\u0631\u0627\u06cc \u067e\u0698\u0648\u0647\u0634\u200c\u0647\u0627\u06cc "
    "\u0637\u0628\u06cc\u0639\u06cc \u0648 \u0627\u062c\u062a\u0645\u0627\u0639\u06cc"
)
BAD = "\u0645\u0627\u0644\u06cc " + CLEAN


def test_contextual_organization_cannot_be_observed_or_reused():
    text = "\u0628\u0627 \u062d\u0645\u0627\u06cc\u062a " + BAD + f" ({SOURCE})"
    assert observed_bilingual_target(text, SOURCE, "organization") == ""
    assert not is_usable_observed_mapping(SOURCE, BAD, "organization")
    assert is_usable_observed_mapping(SOURCE, CLEAN, "organization")


@pytest.mark.parametrize(
    "origin", ["observed_translation", "incremental_extraction", "accepted_correction", "legacy"]
)
def test_historical_records_are_retained_but_not_prompted_or_inline(origin):
    state = {
        "nouns": {SOURCE: BAD},
        "categories": {SOURCE: "organization"},
        "introduced": [SOURCE],
        "provenance": {
            SOURCE: {
                "origin": origin,
                "authority": 30,
                "context_independent": True,
                "alignment_status": "exact_local",
            }
        },
    }
    noun = ProperNouns()
    noun.deserialize(copy.deepcopy(state))
    assert noun.all_nouns()[SOURCE] == BAD
    assert noun.context_deferral_reason(SOURCE) == "organization_lexical_start_unproven"
    assert SOURCE not in noun.inline_eligible_nouns()
    assert SOURCE not in noun.pending_inline_originals(SOURCE)
    assert SOURCE not in noun.get_context(source_text=SOURCE)


@pytest.mark.parametrize(
    "source,target",
    [
        (
            "Regional Research Council",
            "\u0634\u0648\u0631\u0627\u06cc \u067e\u0698\u0648\u0647\u0634 "
            "\u0645\u0646\u0637\u0642\u0647\u200c\u0627\u06cc",
        ),
        (
            "Royal Library",
            "\u06a9\u062a\u0627\u0628\u062e\u0627\u0646\u0647 \u0633\u0644\u0637\u0646\u062a\u06cc",
        ),
        ("Vale Ltd", "\u0648\u06cc\u0644 \u0644\u06cc\u0645\u06cc\u062a\u062f"),
        ("East Region", "\u0645\u0646\u0637\u0642\u0647 \u0634\u0631\u0642\u06cc"),
    ],
)
def test_legitimate_heads_brands_and_other_categories_are_not_deferred(source, target):
    assert not organization_lexical_deferral(source, target, "organization")


@pytest.mark.parametrize("origin", ["curated_glossary", "approved_research"])
def test_explicit_authority_is_not_demoted_by_automatic_boundary_rule(origin):
    noun = ProperNouns()
    noun.deserialize(
        {
            "nouns": {SOURCE: BAD},
            "categories": {SOURCE: "organization"},
            "provenance": {SOURCE: {"origin": origin, "authority": 100}},
        }
    )
    assert not noun.context_deferral_reason(SOURCE)
