from __future__ import annotations

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import (
    _accepted_grounded_inline_originals,
    _log_chunk_terminal_failure,
)
from tarjomeh.core.term_notes import (
    audit_inline_english_originals,
    merge_inline_english_original_audits,
)
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.memory.proper_nouns import ProperNouns
from tarjomeh.quality.integrity import (
    source_unjustified_repeated_governed_span_artifacts,
)


def _document(source: str, target: str) -> TranslatedDocument:
    return TranslatedDocument(
        paragraphs=[
            TranslatedParagraph(
                index=0,
                source_text=source,
                translated_text=target,
            )
        ]
    )


def test_governed_repetition_does_not_skip_latin_citation_atoms() -> None:
    source = (
        "See Connolly 1969 for the first argument; see Althusser 1970 "
        "for the second argument."
    )
    target = (
        "\u0628\u0647 Connolly 1969\u061b \u0628\u0631\u0627\u06cc "
        "\u0627\u0633\u062a\u062f\u0644\u0627\u0644 \u0646\u062e\u0633\u062a "
        "\u0631.\u06a9.\u061b \u0628\u0647 Althusser 1970\u061b \u0628\u0631\u0627\u06cc "
        "\u0627\u0633\u062a\u062f\u0644\u0627\u0644 \u062f\u0648\u0645 \u0631.\u06a9."
    )

    assert source_unjustified_repeated_governed_span_artifacts(source, target) == []


def test_governed_repetition_still_detects_one_real_lexical_phrase() -> None:
    target = (
        "\u0627\u06cc\u0646 \u0628\u0646\u062f \u062f\u0631\u0628\u0627\u0631\u0647 "
        "\u0632\u0628\u0627\u0646 \u0628\u062d\u062b \u0645\u06cc\u200c\u06a9\u0646\u062f \u0648 "
        "\u062f\u0631\u0628\u0627\u0631\u0647 \u0632\u0628\u0627\u0646 "
        "\u062a\u0648\u0636\u06cc\u062d "
        "\u0645\u06cc\u200c\u062f\u0647\u062f."
    )

    findings = source_unjustified_repeated_governed_span_artifacts(
        "The paragraph discusses language and explains its role.", target
    )

    assert findings
    assert (
        findings[0]["normalized_phrase"]
        == "\u062f\u0631\u0628\u0627\u0631\u0647 \u0632\u0628\u0627\u0646"
    )


def test_source_grounded_parenthetical_is_preserved_not_silently_deleted() -> None:
    document = _document(
        "The very longue duree dynamics matter.",
        "\u067e\u0648\u06cc\u0627\u06cc\u06cc\u200c\u0647\u0627\u06cc "
        "\u0628\u0633\u06cc\u0627\u0631 "
        "\u062f\u0631\u0627\u0632\u0645\u062f\u062a "
        "(longue duree) \u0627\u0647\u0645\u06cc\u062a \u062f\u0627\u0631\u0646\u062f.",
    )

    report = audit_inline_english_originals(document, {})

    assert "(longue duree)" in document.paragraphs[0].translated_text
    assert report["preserved_source_grounded_count"] == 1
    assert report["removed_unauthorized_count"] == 0


def test_ungrounded_parenthetical_is_retained_for_review_with_evidence() -> None:
    document = _document(
        "The argument remains contested.",
        "\u0627\u06cc\u0646 \u0627\u0633\u062a\u062f\u0644\u0627\u0644 (invented label) "
        "\u0647\u0645\u0686\u0646\u0627\u0646 "
        "\u0645\u062d\u0644 \u0645\u0646\u0627\u0642\u0634\u0647 \u0627\u0633\u062a.",
    )

    report = audit_inline_english_originals(document, {})

    assert "(invented label)" in document.paragraphs[0].translated_text
    assert report["unapproved_ungrounded_count"] == 1


def test_original_audit_merge_preserves_every_pass() -> None:
    merged = merge_inline_english_original_audits(
        [
            {
                "authorized_count": 2,
                "kept_authorized": 1,
                "preserved_source_grounded_count": 1,
                "preserved_source_grounded": [{"original": "first"}],
            },
            {
                "authorized_count": 2,
                "kept_authorized": 2,
                "removed_duplicate_count": 1,
                "removed_duplicates": [{"original": "duplicate"}],
            },
        ]
    )

    assert merged["pass_count"] == 2
    assert merged["kept_authorized"] == 2
    assert merged["preserved_source_grounded_count"] == 1
    assert merged["removed_duplicate_count"] == 1


def test_title_cased_person_anchor_outranks_phonetic_loanword_guess() -> None:
    categories = _accepted_grounded_inline_originals(
        "Bob Jessop develops the argument.",
        "\u0628\u0627\u0628 \u062c\u0633\u0648\u067e (Bob Jessop) "
        "\u0627\u0633\u062a\u062f\u0644\u0627\u0644 "
        "\u0631\u0627 \u0628\u0633\u0637 \u0645\u06cc\u200c\u062f\u0647\u062f.",
    )

    assert categories == {"Bob Jessop": "source_grounded_entity"}


def test_observed_entity_category_can_upgrade_earlier_loanword_label() -> None:
    memory = ProperNouns()
    memory.add_noun(
        "Bob Jessop",
        "\u0628\u0627\u0628 \u062c\u0633\u0648\u067e",
        category="technical_loanword",
        provenance="observed_translation",
    )
    memory.add_noun(
        "Bob Jessop",
        "\u0628\u0627\u0628 \u062c\u0633\u0648\u067e",
        category="source_grounded_entity",
        provenance="observed_translation",
    )

    assert memory.category_for("Bob Jessop") == "source_grounded_entity"


def test_dedication_remains_continuity_but_not_active_style_authority() -> None:
    manager = MemoryManager(TarjomehConfig())
    chunk = Chunk(
        0,
        "This book is dedicated to a valued colleague. " * 8,
        "Front matter",
        "",
        metadata={
            "style_eligible": True,
            "style_body_paragraphs": [0],
            "structural_roles": ["body"],
        },
    )
    target = (
        "\u0627\u06cc\u0646 \u06a9\u062a\u0627\u0628 \u0628\u0647 \u06cc\u06a9 "
        "\u0647\u0645\u06a9\u0627\u0631 "
        "\u0627\u0631\u0632\u0634\u0645\u0646\u062f "
        "\u062a\u0642\u062f\u06cc\u0645 \u0645\u06cc\u200c\u0634\u0648\u062f. "
        * 8
    )

    policy = manager.update_after_translation(
        chunk, target, quality_approved=True, style_approved=True
    )

    assert policy["short_term_added"] is True
    assert policy["long_term_added"] is True
    assert policy["style_sample_added"] is False
    assert policy["style_sample_policy"]["reason"] == (
        "source_genre_not_representative_of_body_voice"
    )


class _FailureDB:
    def __init__(self) -> None:
        self.logged: list[tuple[str, int, str, dict[str, object]]] = []

    def get_chunk_events(self, _job: str, _chunk: int):
        return [
            {"event_type": "chunk_started", "payload": {}},
            {
                "event_type": "integrity_final_failed",
                "payload": {
                    "stage": "final_translation",
                    "accepted": False,
                    "blocking_count": 1,
                    "findings": [{"type": "numbers_missing"}],
                },
            },
        ]

    def get_worker_lease(self, _job: str):
        return {
            "worker_id": "worker-one",
            "stage": "critique",
            "state": "active",
        }

    def log_chunk_event(
        self,
        job: str,
        chunk: int,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        self.logged.append((job, chunk, event_type, payload))


def test_terminal_chunk_failure_records_stage_and_integrity_evidence() -> None:
    db = _FailureDB()

    payload = _log_chunk_terminal_failure(
        db, "job", 12, ValueError("no valid candidate")
    )

    assert db.logged[0][2] == "chunk_terminal_failure"
    assert payload["worker_stage"] == "critique"
    assert payload["latest_integrity_stage"] == "final_translation"
    assert payload["latest_integrity_findings"] == [{"type": "numbers_missing"}]
