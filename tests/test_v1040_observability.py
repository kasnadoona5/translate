from tarjomeh.core.pipeline import audit_document_final_text
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph


def test_final_text_observation_does_not_rewrite_or_block() -> None:
    target = "این دولت‌‌ملت در فصل ۳ بررسی می‌شود."
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(
        index=0,
        source_text="This nation-state is discussed in chapter 3.",
        translated_text=target,
    )])

    report = audit_document_final_text(document)

    assert document.paragraphs[0].translated_text == target
    assert any(
        item["check_id"] == "double_zwnj"
        for item in report["report_only_findings"]
    )
    assert report["report_only_finding_count"] == len(report["report_only_findings"])


def test_final_text_report_only_optional_affix_needs_source_shape() -> None:
    document = TranslatedDocument(paragraphs=[TranslatedParagraph(
        index=0,
        source_text="The (meta)theoretical question remains.",
        translated_text="پرسش (فرا) نظری باقی می‌ماند.",
    )])

    report = audit_document_final_text(document)

    assert any(
        item["check_id"] == "spaced_optional_affix"
        for item in report["report_only_findings"]
    )
