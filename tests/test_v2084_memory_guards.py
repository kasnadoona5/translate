import pytest

from tarjomeh.memory.proper_nouns import ProperNouns
from tarjomeh.memory.bilingual_summary import BilingualSummary


@pytest.mark.parametrize("source,target", [
    ("Engels", "انگلس"), ("Habermas", "هابرماس"),
    ("Nicos Poulantzas", "نیکوس پولانزاس"),
])
def test_exact_local_name_correction_does_not_use_lexical_plural_guard(source, target):
    nouns = ProperNouns()
    nouns.add_noun(source, target, category="person", provenance="accepted_correction",
                   alignment_status="exact_local")
    assert not nouns.context_deferral_reason(source)


def test_lower_authority_cannot_reclassify_curated_concept():
    nouns = ProperNouns()
    nouns.add_noun("polity", "سامان سیاسی", category="term", provenance="curated_glossary")
    nouns.add_noun("polity", "سامان سیاسی", category="organization", provenance="observed_translation")
    assert nouns.category_for("polity") == "term"


def test_higher_authority_confirmation_can_reclassify():
    nouns = ProperNouns()
    nouns.add_noun("Ada Vail", "آدا ویل", category="term", provenance="auto_extraction")
    nouns.add_noun("Ada Vail", "آدا ویل", category="person", provenance="curated_glossary")
    assert nouns.category_for("Ada Vail") == "person"


@pytest.mark.parametrize("origin", ["observed_translation", "accepted_correction"])
def test_automatic_sightings_cannot_erase_context_conflict(origin):
    nouns = ProperNouns()
    nouns.add_noun("state power", "قدرت دولتی", category="term", provenance="observed_translation")
    nouns.mark_context_conflict("state power", corrected_source="power", corrected_target="قدرت")
    nouns.add_noun("state power", "قدرت دولتی", category="term", provenance=origin,
                   alignment_status="exact_local")
    assert nouns.context_deferral_reason("state power") == "context_conflict"
    assert not nouns.get_context(source_text="state power")


def test_extracted_name_is_labelled_as_proposal_not_observation():
    nouns = ProperNouns()
    nouns.add_noun("Ada Vail", "آدا ویل", category="person", provenance="auto_extraction")
    context = nouns.get_context(source_text="Ada Vail")
    assert "proposal" in context and "observed identity evidence" not in context


def test_summary_subheading_does_not_truncate_section():
    summary = BilingualSummary()
    summary.update("## English Summary\nInitial claim.\n### Detail\nLater claim.\n"
                   "## خلاصه فارسی\nادعای نخست.\n### جزئیات\nادعای بعدی.")
    assert "Later claim." in summary.english_summary
    assert "ادعای بعدی." in summary.persian_summary


def test_summary_body_mention_does_not_switch_languages():
    summary = BilingualSummary()
    summary.update("## English Summary\nA mention of خلاصه فارسی is not a header.\n"
                   "More English content.\n## خلاصه فارسی\nمتن فارسی کافی برای خلاصه.")
    assert "More English content." in summary.english_summary
    assert "More English" not in summary.persian_summary
