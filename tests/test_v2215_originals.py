"""Legacy approvals remain paragraph-bound; introduction notes grant no permission."""
import hashlib
from copy import deepcopy

import pytest

from tarjomeh.core.pipeline import audit_translation_language
from tarjomeh.glossary.book_review import bind_original_introductions, scoped_original_permissions

SOURCE = "A new institution is described."
TARGET = "\u0646\u0647\u0627\u062f"
PAIR = TARGET + " (institution)"


def match():
    return dict(source="institution", target=TARGET, sense_id="", keep_original=True,
                paragraph_index=0, global_paragraph_index=0, source_occurrence_count=1,
                source_paragraph_hash=hashlib.sha256(SOURCE.encode()).hexdigest())


@pytest.mark.parametrize("version", [1, 2, 3])
def test_legacy_pair_requires_explicit_version_and_bounded_local_pair(version):
    matches = [match()]
    bind_original_introductions(matches, {"review_version": version}, [], {})
    assert scoped_original_permissions(SOURCE, PAIR, matches) == ["institution"]
    assert scoped_original_permissions(SOURCE, "institution", matches) == []
    assert scoped_original_permissions(SOURCE, PAIR + " institution", matches) == []
    assert scoped_original_permissions(SOURCE, TARGET + " (Institution)", matches) == []
    bad = deepcopy(matches); bad[0]["source_paragraph_hash"] = "bad"
    assert scoped_original_permissions(SOURCE, PAIR, bad) == []


@pytest.mark.parametrize("version", [None, 0, 5, True, "1"])
def test_missing_or_unsupported_version_gives_no_legacy_permission(version):
    matches = [match()]
    bind_original_introductions(matches, {"review_version": version}, [], {})
    assert scoped_original_permissions(SOURCE, PAIR, matches) == []


def test_legacy_permission_does_not_leak_to_another_paragraph(monkeypatch):
    matches = [match()]
    bind_original_introductions(matches, {"review_version": 2}, [], {})
    assert scoped_original_permissions(SOURCE + "\n\n" + SOURCE,
                                       PAIR + "\n\n" + PAIR, matches) == ["institution"]
    allowed = []
    def inspect_permissions(*args, **kwargs):
        allowed.append(list(kwargs.get("allowed_originals", [])))
        return []
    monkeypatch.setattr("tarjomeh.core.pipeline.unexpected_latin_prose", inspect_permissions)
    audit_translation_language(
        SOURCE + "\n\n" + SOURCE, PAIR + "\n\n" + PAIR,
        anchor_context={"scoped_book_originals": matches})
    assert allowed[-2:] == [["institution"], []]


def test_repeated_first_occurrence_has_an_explicit_note_not_a_new_requirement():
    import tarjomeh.glossary.book_review as module
    from unittest.mock import patch

    matches = [match()]; matches[0]["source_occurrence_count"] = 2
    with patch.object(module, "resolve_reviewed_book_terms", return_value=(matches, [])):
        bind_original_introductions(matches, {"review_version": 4},
                                    [{"chunk_index": 0, "chapter_position": 0}],
                                    {0: (SOURCE, {})})
    assert not matches[0]["original_introduction_required"]
    assert matches[0]["original_introduction_reason"] == "repeated_occurrence_in_first_approved_paragraph"
    assert scoped_original_permissions(SOURCE, PAIR, matches) == []
