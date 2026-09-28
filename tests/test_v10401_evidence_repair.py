"""v10.40.1 evidence-grounded repairs: positive and over-correction tests.

All fixtures are synthetic or short generalized strings; the Jessop PDF is
used only by the separate read-only acceptance run, never committed here.
"""

from tarjomeh.context.book_term_candidates import (
    body_term_paragraphs,
    book_term_extraction_sample,
    collect_book_term_candidates,
)
from tarjomeh.core.term_notes import _extend_persian_anchor_end
from tarjomeh.exporters.docx_exporter import source_superscript_spans
from tarjomeh.memory.manager import _is_paratext_style_source
from tarjomeh.memory.proper_nouns import (
    is_bounded_person_name_target,
    is_reusable_terminology_mapping,
    requires_bounded_name_target,
)
from tarjomeh.parsers.base import Chapter, Document, Paragraph, Section
from tarjomeh.parsers.pdf_parser import (
    _italic_runs,
    _markers_in_text,
    locate_source_marker,
)
from tarjomeh.quality.integrity import (
    extract_identifiers,
    repair_proven_surface_artifacts,
    repair_source_grounded_language_artifacts,
)


def _chapter(title, *texts):
    return Chapter(title=title, sections=[Section(
        title="", level=2, paragraphs=[Paragraph(text) for text in texts],
    )])


# -- note markers ---------------------------------------------------------

def test_marker_is_located_by_context_not_digit_count():
    source = ("In the 1920s and 1930s (1) theorists wrote, (3) others followed,3 "
              "historical accounts, see chapter 3.")
    record = {"text": "3", "context_before": "thers followed,", "context_after": " historical"}
    offset = locate_source_marker(source, record)
    assert offset == source.index("followed,3") + len("followed,")
    kept = _markers_in_text(source, [record])
    assert [item["offset"] for item in kept] == [offset]


def test_marker_without_unique_context_is_dropped_not_guessed():
    record = {"text": "3", "context_before": "", "context_after": ""}
    assert locate_source_marker("Items (3) and chapter 3 and note 3.", record) is None


def test_export_superscripts_real_note_but_not_structural_digits():
    source = "Rational choice is ignored,3 while (1) and (3) are listed (chapter 3)."
    metadata = {"superscript_markers": [{
        "text": "3", "context_before": "is ignored,", "context_after": " while",
    }]}
    target = "انتخاب عقلانی نادیده گرفته می‌شود، ۳ در حالی که (۱) و (۳) فهرست شده‌اند (فصل ۳)."
    spans = source_superscript_spans(target, metadata, source)
    assert len(spans) == 1
    start, end = spans[0]
    assert target[start:end] == "۳" and target[:start].rstrip().endswith("،")


def test_export_leaves_ambiguous_note_plain():
    source = "The argument.3"
    metadata = {"superscript_markers": [
        {"text": "3", "context_before": "argument.", "context_after": ""},
    ]}
    # Two note-like candidates: position is not proven, nothing is superscripted.
    assert source_superscript_spans("نکته. ۳ و استدلال. ۳", metadata, source) == []
    # A digit after a word inside a chapter reference is never a note.
    assert source_superscript_spans("در فصل ۳ آمده است", metadata, source) == []


def test_italic_run_joins_soft_hyphen_line_break():
    lines = [
        {"spans": [{"text": "useful for an ", "flags": 0, "font": "Roman"},
                   {"text": "Ideo­", "flags": 2, "font": "Italic"}]},
        {"spans": [{"text": "logiekritik", "flags": 2, "font": "Italic"},
                   {"text": ". My analyses", "flags": 0, "font": "Roman"}]},
    ]
    assert _italic_runs(lines) == ["Ideologiekritik"]


# -- sampling and term inventory ----------------------------------------------

def test_sampling_excludes_front_matter_indexes_and_title_page():
    document = Document(title="A Book", chapters=[
        _chapter("A Book", "Copyright 2016. The right of the author is asserted here."),
        _chapter("Contents", "1 Introduction 1"),
        _chapter("1 Introduction",
                 "The nation-state and civil society shape the modern argument here."),
        _chapter("Index of Names", "Smith, A. 12, 45, 67"),
        _chapter("Subject Index", "nation-state 3, 17, 22, 90"),
    ])
    body = [p.text for p in body_term_paragraphs(document)]
    assert body == ["The nation-state and civil society shape the modern argument here."]
    sample = book_term_extraction_sample(document)
    assert "Copyright" not in sample and "17, 22" not in sample


def test_inventory_is_general_not_tuned_to_one_book():
    # A different domain: river ecology. Terms must surface by general rules.
    # As in any real book, content words also recur outside the term itself.
    paragraphs = [
        "The floodplain-forest supports riparian habitat and sediment transport in each basin.",
        "Sediment transport rather than rainfall alone shapes the floodplain-forest over decades.",
        "Riparian habitat and sediment transport interact; "
        "ad hoc surveys hide this, rather than reveal it.",
        "Ad hoc surveys of riparian habitat undermine estimates of sediment transport.",
        "Fine sediment, transport corridors, riparian soils and habitat loss recur in surveys.",
    ]
    document = Document(title="Rivers", chapters=[_chapter("Chapter 1", *paragraphs)])
    sources = {item["source"].casefold(): item for item in collect_book_term_candidates(document)}
    assert "sediment transport" in sources
    assert "riparian habitat" in sources
    assert "floodplain-forest" in sources
    assert sources["floodplain-forest"]["origin"] == "compound_source_term"
    assert "ad hoc" in sources and sources["ad hoc"]["origin"] == "fixed_source_expression"
    assert "rather than" not in sources
    assert all(item["target"] == "" for item in sources.values())


def test_name_pairs_are_not_terms_and_lists_keep_multiword_final_item():
    paragraphs = [
        "Miller and Rose argue about Brazil, Russia, India, China, and South Africa today.",
        "As Miller and Rose note, Brazil, Russia, India, China, and South Africa differ.",
        "Miller and Rose again; Brazil, Russia, India, China, and South Africa once more.",
    ]
    document = Document(title="T", chapters=[_chapter("Chapter 1", *paragraphs)])
    sources = [item["source"] for item in collect_book_term_candidates(document)]
    assert "Miller and Rose" not in sources
    assert "Brazil, Russia, India, China, and South Africa" in sources


def test_coordinated_family_variants_merge_by_plural_rules():
    paragraphs = [
        "It studies the territory, place, scale, and network in context.",
        "Again the territories, places, scales, and networks.",
        "Finally the territory, place, scale, and networks, as before.",
    ]
    document = Document(title="T", chapters=[_chapter("Chapter 1", *paragraphs)])
    families = [item for item in collect_book_term_candidates(document)
                if item["origin"] == "source_coordination"]
    assert len(families) == 1 and families[0]["source_count"] == 3


# -- final-text repairs -------------------------------------------------------

def test_double_zwnj_is_collapsed_in_table_and_list_rows():
    source = "SRA strategic–relational approach"
    target = "SRA رویکرد راهبردی‌‌رابطه‌ای"
    repaired, edits = repair_proven_surface_artifacts(
        source, target, structural_role="mixed", paragraph_roles=["table"],
    )
    assert "‌‌" not in repaired and edits[0]["type"] == "double_zwnj"


def test_duplicate_coordinator_stays_body_only():
    repaired, edits = repair_proven_surface_artifacts(
        "It was selected and retained.", "برگزیده شد و و حفظ شد.",
        paragraph_roles=["heading"],
    )
    assert "و و" in repaired and edits == []


def test_nested_foreign_original_uses_bracket_convention_and_is_idempotent():
    # R44 convention: one parenthesis, an inner original in square brackets.
    source = "It acquires its own political rationale (raison d’état) and modus operandi."
    for target in (
        "عقلانیت سیاسی (مصلحت دولت [(raison d’état)]) را",
        "عقلانیت سیاسی (مصلحت دولت (raison d’état)) را",
    ):
        once, _ = repair_source_grounded_language_artifacts(source, target)
        twice, _ = repair_source_grounded_language_artifacts(source, once)
        assert "(مصلحت دولت [raison d’état])" in once
        assert "[(" not in once
        assert once == twice


def test_nested_original_is_not_flattened_without_source_evidence():
    target = "عقلانیت سیاسی (مصلحت دولت (raison d’état)) را"
    repaired, report = repair_source_grounded_language_artifacts(
        "It acquires its own political rationale.", target,
    )
    assert repaired == target and report["repair_count"] == 0


def test_stranded_kasra_moves_back_to_its_word():
    source = "Chapter 9 examines the elective affinities between capitalism and democracy."
    target = "فصل ۹ خویشاوندی‌های انتخابی (elective affinities)ِ میان سرمایه‌داری را بررسی می‌کند."
    repaired, _ = repair_source_grounded_language_artifacts(source, target)
    assert "انتخابیِ (elective affinities) میان" in repaired
    assert ")ِ" not in repaired


def test_kasra_insertion_point_steps_over_combining_mark():
    text = "انتخابیِ میان"
    assert _extend_persian_anchor_end(text, len("انتخابی")) == len("انتخابیِ")


def test_optional_prefix_joined_only_with_unique_source_form():
    once_source = "It implies (meta)theoretical pluralism in analysis."
    target = "این امر مستلزم کثرت‌گرایی (فرا) نظری در تحلیل است."
    repaired, _ = repair_source_grounded_language_artifacts(once_source, target)
    assert "(فرا)نظری" in repaired
    twice_source = "Both (inter)state systems and (inter)state rivalry matter."
    twice_target = "هم نظام‌های (بینا) دولتی و هم رقابت (بینا) دولتی مهم‌اند."
    unchanged, _ = repair_source_grounded_language_artifacts(twice_source, twice_target)
    assert unchanged == twice_target


def test_stray_dash_removed_only_when_sentence_alignment_proves_it():
    source = ("Social forces make history – their own and that of others – in contexts. "
              "These studies consider scope, including multilevel interaction.")
    target = ("نیروهای اجتماعی تاریخ خود و تاریخ دیگران — را رقم می‌زنند. "
              "این مطالعات مجال را — از جمله تعامل چندسطحی — می‌سنجند.")
    repaired, report = repair_source_grounded_language_artifacts(source, target)
    assert "دیگران را رقم" in repaired
    assert "— از جمله" in repaired
    # Unequal sentence counts: no proof, text unchanged.
    merged_target = (
        "نیروهای اجتماعی تاریخ خود و تاریخ دیگران — را رقم می‌زنند و "
        "مطالعات — از جمله تعامل — را می‌سنجند."
    )
    unchanged, _ = repair_source_grounded_language_artifacts(source, merged_target)
    assert unchanged == merged_target


def test_us_zip_code_is_a_source_identifier_but_years_are_not():
    assert extract_identifiers("Malden, MA 02148, USA") == {"ma 02148": 1}
    assert not extract_identifiers("in 1988 and 2016")


# -- memory and style authority -----------------------------------------------

def test_context_bound_phrase_fragments_are_not_reusable_terms():
    for source, target in (
        ("some broad macro-trends", "کلان‌روندهای گسترده"),
        ("another epistemic community", "جامعهٔ معرفتیِ دیگری"),
        ("speculating about possible futures", "گمانه‌زنی دربارهٔ آینده‌های ممکن"),
        ("historical specificity", "تعینِ تاریخیِ"),
    ):
        assert not is_reusable_terminology_mapping(source, target), source
    for source, target in (
        ("state building", "دولت‌سازی"),
        ("path shaping", "شکل‌دهی به مسیر"),
        ("policy paradigms", "پارادایم‌های خط‌مشی"),
        ("historical specificity", "تعین تاریخی"),
    ):
        assert is_reusable_terminology_mapping(source, target), source


def test_name_guard_applies_to_name_shaped_entities_and_allows_initials():
    assert requires_bounded_name_target("source_grounded_entity", "Manuela Tecusan")
    assert not requires_bounded_name_target(
        "source_grounded_entity", "Economic and Social Science Research Council"
    )
    for target in (
        "ویراستاری عالمانه و کاملاً تخصصیِ مانوئلا تکوشان",
        "تخصصیِ مانوئلا تکوشان",
        "و مانوئلا تکوشان",
    ):
        assert not is_bounded_person_name_target("Manuela Tecusan", target)
    assert is_bounded_person_name_target("Manuela Tecusan", "مانوئلا تکوسان")
    assert is_bounded_person_name_target("Hegel", "گ. و. ف. هگل")
    assert is_bounded_person_name_target("Ngai-Ling Sum", "نگای-لینگ سام")


def test_acknowledgement_prose_is_paratext_for_style_only():
    assert _is_paratext_style_source(
        "Special thanks are also due to two editors for gently nudging this book."
    )
    assert _is_paratext_style_source("I am grateful to many colleagues.")
    # Analytical prose that merely uses similar words stays eligible.
    assert not _is_paratext_style_source(
        "Chapter 4 is dedicated to the analysis of class power."
    )
    assert not _is_paratext_style_source(
        "The state benefited from favourable conjunctures."
    )


# -- follow-up audit findings (v10.40.1 review) --------------------------------

def test_name_guard_rejects_descriptor_words():
    assert not is_bounded_person_name_target("Hegel", "فیلسوف هگل")
    assert not is_bounded_person_name_target(
        "Manuela Tecusan", "ویراستاری مانوئلا تکوسان"
    )
    assert is_bounded_person_name_target("Emmerich de Vattel", "امریک دو واتل")
    assert is_bounded_person_name_target("Den Haag", "لاهه")


def test_fixed_expression_merges_accent_variants_and_drops_fragment():
    paragraphs = [
        "A political esprit de corps is unusual in such bureaucracies.",
        "The distinctive ésprit de corps of the service matters here.",
    ]
    document = Document(title="T", chapters=[_chapter("Chapter 1", *paragraphs)])
    sources = [item["source"] for item in collect_book_term_candidates(document)]
    # Accent-only variation does not prove equivalence: abstain, offering
    # neither the clipped fragment nor a merged form.
    assert "de corps" not in sources
    assert not any("esprit de corps" in source for source in sources)


def test_identical_preceding_word_completes_a_fixed_fragment():
    paragraphs = [
        "Its political esprit de corps is unusual in such bureaucracies.",
        "The civil service esprit de corps of the ministry matters here.",
    ]
    document = Document(title="T", chapters=[_chapter("Chapter 1", *paragraphs)])
    fixed = [item["source"] for item in collect_book_term_candidates(document)
             if item["origin"] == "fixed_source_expression"]
    assert "esprit de corps" in fixed and "de corps" not in fixed


def test_accent_distinct_fixed_expressions_are_not_merged():
    paragraphs = [
        "The résumé analysis report ranks applicants.",
        "Each résumé analysis report is archived.",
        "The resume analysis report lists restarted jobs.",
        "Each resume analysis report is rotated nightly.",
    ]
    document = Document(title="T", chapters=[_chapter("Chapter 1", *paragraphs)])
    sources = [item["source"] for item in collect_book_term_candidates(document)
               if item["origin"] == "fixed_source_expression"]
    assert "analysis report" not in sources
    assert not any("résumé" in source and "resume" in source for source in sources)
    merged = [s for s in sources if s.endswith("analysis report") and " " in s]
    assert all(s.split()[0] in {"résumé", "resume"} for s in merged)


def test_uncertain_list_boundary_is_not_clipped():
    paragraphs = [
        "It studies polity, politics, and public policy in Europe.",
        "Again polity, politics, and public policy interact.",
        "Finally polity, politics, and public policy diverge.",
    ]
    document = Document(title="T", chapters=[_chapter("Chapter 1", *paragraphs)])
    families = [item["source"] for item in collect_book_term_candidates(document)
                if item["origin"] == "source_coordination"]
    assert "polity, politics, and public" not in families
    assert families == []


def test_accent_distinct_words_are_not_merged():
    paragraphs = [
        "Recruiters use résumé analysis to rank applicants carefully.",
        "Automated résumé analysis favours keywords over experience.",
        "After a crash, resume analysis restarts the stopped batch job.",
        "Operators trigger resume analysis whenever a worker fails.",
        "Both résumé analysis and resume analysis appear in the manual.",
    ]
    document = Document(title="T", chapters=[_chapter("Chapter 1", *paragraphs)])
    sources = {item["source"]: item for item in collect_book_term_candidates(document)}
    assert "résumé analysis" in sources and "resume analysis" in sources
    assert sources["résumé analysis"]["source_count"] == 3
    assert sources["resume analysis"]["source_count"] == 3


def test_coordinated_list_keeps_three_word_final_member():
    paragraphs = [
        "Trade links Qatar, Oman, Bahrain, and United Arab Emirates closely.",
        "Again Qatar, Oman, Bahrain, and United Arab Emirates cooperate.",
    ]
    document = Document(title="T", chapters=[_chapter("Chapter 1", *paragraphs)])
    families = [item["source"] for item in collect_book_term_candidates(document)
                if item["origin"] == "source_coordination"]
    assert families == ["Qatar, Oman, Bahrain, and United Arab Emirates"]


def test_extraction_sample_covers_every_candidate_kind():
    paragraphs = [
        f"State power and state power again; the nation-state {i} matters, "
        "and polity, politics, and policy recur."
        for i in range(6)
    ]
    document = Document(title="T", chapters=[_chapter("Chapter 1", *paragraphs)])
    sample = book_term_extraction_sample(document)
    assert "state power" in sample.casefold()
    assert "nation-state" in sample
    assert "polity, politics, and policy" in sample


def test_fragment_after_different_content_words_is_withdrawn():
    paragraphs = [
        "Officers shared a strong esprit de corps.",
        "Recruits soon felt the spirit de corps.",
    ]
    document = Document(title="T", chapters=[_chapter("Chapter 1", *paragraphs)])
    sources = [item["source"] for item in collect_book_term_candidates(document)]
    assert "de corps" not in sources


def test_fragment_attested_at_a_clear_boundary_is_kept():
    paragraphs = [
        "Its modus operandi was secret.",
        "A distinctive modus operandi emerged later.",
    ]
    document = Document(title="T", chapters=[_chapter("Chapter 1", *paragraphs)])
    sources = [item["source"] for item in collect_book_term_candidates(document)
               if item["origin"] == "fixed_source_expression"]
    assert "modus operandi" in sources
