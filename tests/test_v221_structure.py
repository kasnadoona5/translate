"""Chapter evidence must not reject faithful translations or waive real counts."""

import asyncio
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.quality.structure_audit import (
    announced_count_evidence,
    audit_payload,
    chapter_marker_count,
)

ZWNJ = chr(0x200C)
DIGIT_SCRIPTS = (
    "0123456789",
    "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9",
    "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669",
)


def saved_pair():
    data = json.loads(
        (Path(__file__).parent / "fixtures/v221_summary_count.json").read_text(encoding="utf-8")
    )
    for value in data.values():
        assert not value["truncated"]
        assert hashlib.sha256(value["text"].encode()).hexdigest() == value["sha256"]
        assert len(value["text"]) == value["chars"]
    return data["source"]["text"], data["target"]["text"]


def chapter_pair(count=10, joiner=ZWNJ, digits="0123456789"):
    word = {4: "four", 9: "nine", 10: "ten"}[count]
    fa_word = {4: "\u0686\u0647\u0627\u0631", 9: "\u0646\u0647", 10: "\u062f\u0647"}[count]
    source = (
        f"This {word}-chapter report covers "
        + ", ".join(f"Chapter {i}" for i in range(1, count + 1))
        + "."
    )
    prefix = "\u0627\u06cc\u0646 \u06af\u0632\u0627\u0631\u0634 " + fa_word + joiner
    target = (
        prefix
        + "\u0641\u0635\u0644\u06cc \u0634\u0627\u0645\u0644 "
        + ", ".join(
            "\u0641\u0635\u0644 " + str(i).translate(str.maketrans("0123456789", digits))
            for i in range(1, count + 1)
        )
        + "."
    )
    return source, target


def mismatch(source, target):
    return "translation_structure_mismatch" in audit_payload(source, target)["classifications"]


def test_exact_saved_faithful_summary_is_not_a_zero_item_mismatch():
    source, target = saved_pair()
    assert chapter_marker_count(source) == chapter_marker_count(target) == 10
    evidence = announced_count_evidence(target)
    assert any(item.value == 10 and item.semantic_category == "chapter" for item in evidence)
    assert not mismatch(source, target)


@pytest.mark.parametrize("count", [9, 10])
@pytest.mark.parametrize("joiner", ["\u200c", "-"])
def test_joined_quantitative_adjectives_have_exact_evidence(count, joiner):
    source, target = chapter_pair(count, joiner)
    items = [
        item for item in announced_count_evidence(target) if item.semantic_category == "chapter"
    ]
    assert len(items) == 1 and items[0].value == count
    item = items[0]
    expected_count = {9: "\u0646\u0647", 10: "\u062f\u0647"}[count]
    assert target[item.count_span[0] : item.count_span[1]] == expected_count
    assert target[item.noun_span[0] : item.noun_span[1]] == "\u0641\u0635\u0644\u06cc"
    assert item.exact_span == target[item.count_span[0] : item.noun_span[1]]
    assert not mismatch(source, target)


@pytest.mark.parametrize(
    "text",
    [
        "\u0646\u0647 \u0641\u0635\u0644\u06cc",
        "\u062f\u0647 \u0641\u0635\u0644\u06cc",
        "\u0646\u0647 \u0627\u06cc\u0646 \u0641\u0635\u0644",
        "\u0646\u0647\u200c\u0641\u0635\u0644",
        "\u062f\u0647\u200c\u0631\u0648\u0632\u0647",
        "\u062f\u0647--\u0641\u0635\u0644\u06cc",
        "\u0628\u062f\u0647\u200c\u0641\u0635\u0644\u06cc",
        "\u062f\u0647\u200c\u0641\u0635\u0644\u06cc\u062a",
        "\u062f\u0647\u200c\u0641\u0635\u0644\u06cc\u200c\u0627\u0634",
    ],
)
def test_ambiguous_words_and_longer_tokens_are_not_new_cardinals(text):
    assert not announced_count_evidence(text)


@pytest.mark.parametrize("digits", DIGIT_SCRIPTS)
def test_chapter_label_scripts_are_equivalent_on_either_side(digits):
    source, target = chapter_pair(4, digits=digits)
    localized_source = source.translate(str.maketrans("0123456789", digits))
    assert chapter_marker_count(localized_source) == chapter_marker_count(target) == 4
    assert not mismatch(localized_source, target)
    assert mismatch(localized_source, target.replace("\u0641\u0635\u0644 " + digits[4], ""))


@pytest.mark.parametrize("count", [9, 10])
def test_wrong_quantity_missing_chapter_and_reordering_still_fail(count):
    source, target = chapter_pair(count)
    assert mismatch(source, target.replace("\u0641\u0635\u0644 " + str(count), ""))
    assert mismatch(source, target.replace("\u0641\u0635\u0644 2", "\u0641\u0635\u0644 7", 1))
    wrong = target.replace(
        {9: "\u0646\u0647", 10: "\u062f\u0647"}[count] + "\u200c\u0641\u0635\u0644\u06cc",
        "\u062f\u0648\u0627\u0632\u062f\u0647\u200c\u0641\u0635\u0644\u06cc",
    )
    assert mismatch(source, wrong)


def test_unrelated_list_does_not_supply_missing_chapters():
    source, target = chapter_pair()
    target = target.replace("\u0641\u0635\u0644 10", "")
    source += "\n\nThree elements: (1) rain (2) wind (3) snow."
    target += (
        "\n\n\u0633\u0647 \u0639\u0646\u0635\u0631: (1) \u0628\u0627\u0631\u0627\u0646 "
        "(2) \u0628\u0627\u062f (3) \u0628\u0631\u0641."
    )
    assert mismatch(source, target)


@pytest.mark.parametrize("digits", DIGIT_SCRIPTS)
def test_long_numbers_are_not_partially_captured_as_chapter_labels(digits):
    text = "Chapter 101, Chapter 102".translate(str.maketrans("0123456789", digits))
    assert chapter_marker_count(text) == 0


class FrozenClient:
    def __init__(self, source, target):
        self.response = (
            "## English Summary\n"
            + source
            + "\n## \u062e\u0644\u0627\u0635\u0647 \u0641\u0627\u0631\u0633\u06cc\n"
            + target
        )
        self.calls = 0

    async def chat(self, prompt):
        self.calls += 1
        return self.response


def test_mixed_digit_scripts_and_wrapped_chapter_references():
    source, target = chapter_pair()
    for index in range(10, 0, -1):
        digits = DIGIT_SCRIPTS[index % len(DIGIT_SCRIPTS)]
        number = str(index).translate(str.maketrans("0123456789", digits))
        if index == 10:
            number = "\u06f1\u0660"
        target = target.replace(
            "\u0641\u0635\u0644 " + str(index), "(\u0641\u0635\u0644 " + number + ")"
        )
    assert chapter_marker_count(target) == 10
    assert not mismatch(source, target)


def test_a_valid_chapter_count_does_not_hide_duplicated_list_labels():
    source, target = chapter_pair()
    source += "\n\nThree elements: (1) rain (2) wind (3) snow."
    target += "\n\n\u0633\u0647 \u0639\u0646\u0635\u0631: (1) A (2) B (2) B (3) C."
    assert mismatch(source, target)


@pytest.mark.parametrize("invalid", [False, True])
def test_real_summary_admission_and_retention(invalid):
    source, target = saved_pair()
    if invalid:
        target = target.replace(
            "\u062f\u0647\u200c\u0641\u0635\u0644\u06cc",
            "\u062f\u0648\u0627\u0632\u062f\u0647\u200c\u0641\u0635\u0644\u06cc",
        )
    manager = MemoryManager(TarjomehConfig())
    manager.quality_policy_version = 1
    manager.update_after_translation(Chunk(0, source, "Outline", ""), target, style_approved=False)
    manager.bilingual_summary.english_summary = "The previous claim is conditional."
    manager.bilingual_summary.persian_summary = (
        "\u0627\u062f\u0639\u0627\u06cc \u0642\u0628\u0644\u06cc "
        "\u0645\u0634\u0631\u0648\u0637 \u0627\u0633\u062a."
    )
    before = deepcopy(
        (
            manager.bilingual_summary.serialize(),
            manager.summary_history,
            manager.summary_coverage,
            manager.summary_pending,
        )
    )
    layers = deepcopy(
        (
            manager.proper_nouns.serialize(),
            manager.long_term.serialize(),
            manager.short_term.serialize(),
            manager.style_sample_records,
        )
    )
    src, tgt, refs = manager.pending_summary_inputs(0)
    client = FrozenClient(source, target)
    result = asyncio.run(manager.update_bilingual_summary(client, src, tgt, coverage_refs=refs))
    assert result["candidate_committed"] is (not invalid)
    assert client.calls == 1
    if invalid:
        assert before == (
            manager.bilingual_summary.serialize(),
            manager.summary_history,
            manager.summary_coverage,
            manager.summary_pending,
        )
    else:
        assert manager.bilingual_summary.english_summary == source
        assert manager.bilingual_summary.persian_summary == target
        assert manager.summary_pending == []
        assert manager.summary_coverage == refs
    assert layers == (
        manager.proper_nouns.serialize(),
        manager.long_term.serialize(),
        manager.short_term.serialize(),
        manager.style_sample_records,
    )


def test_runtime_probes_are_pure_and_reject_a_disabled_checker(monkeypatch):
    import socket

    import tarjomeh.quality.structure_audit as audit
    from tarjomeh.runtime import _v221_behavior_probes

    def forbidden(*args, **kwargs):
        raise AssertionError("Probe attempted external I/O")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    assert all(_v221_behavior_probes().values())
    monkeypatch.setattr(audit, "audit_payload", lambda *args: {"classifications": []})
    assert not any(_v221_behavior_probes().values())
