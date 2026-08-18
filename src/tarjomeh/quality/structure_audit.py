"""Source-aware structural audit (plan item 12).

Deterministic comparison of *announced structure* between source and
translation: "two objections" followed by an enumeration, ordinal sequences,
and bracketed list markers. This is the gap the numeric checks cannot see -
``_NUMBER_RE`` in :mod:`tarjomeh.quality.integrity` matches digits only, so a
source that promises "two objections" and a translation that delivers one
reconcile with ``missing=[]`` and nothing notices.

**This module reports; it never rejects.** Every function returns findings for
the QA report. Item 13 is the only place that may turn a finding into a
rejection, and only after the item-20 corpus shows the detector does not fire on
correct translations. A detector that flags good work is worse than no detector.

Deliberately conservative in three ways:

* A finding is produced only when an enumeration of at least two items is
  actually present on one side. Counting words appear constantly in ordinary
  prose ("two of them left"), and auditing those would be pure noise.
* The word lists are *language* vocabulary, not book vocabulary. No author,
  title, term or page number from any particular book appears here.
* When the evidence does not support a verdict the result is
  ``insufficient_source_evidence``, never a mismatch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# --- classifications --------------------------------------------------------

SOURCE_ANOMALY_PRESERVED = "source_anomaly_preserved"
TRANSLATION_STRUCTURE_MISMATCH = "translation_structure_mismatch"
UNAUTHORIZED_SOURCE_CORRECTION = "unauthorized_source_correction"
INSUFFICIENT_SOURCE_EVIDENCE = "insufficient_source_evidence"

SEVERITY = {
    SOURCE_ANOMALY_PRESERVED: "info",
    TRANSLATION_STRUCTURE_MISMATCH: "warning",
    UNAUTHORIZED_SOURCE_CORRECTION: "warning",
    INSUFFICIENT_SOURCE_EVIDENCE: "info",
}

# --- language vocabulary ----------------------------------------------------

_CARDINALS: dict[str, int] = {
    # English
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
    # Persian
    "دو": 2,                      # do
    "سه": 3,                      # se
    "چهار": 4,          # chahar
    "پنج": 5,                # panj
    "شش": 6,                      # shesh
    "هفت": 7,                # haft
    "هشت": 8,                # hasht
    # "نه" (nine) and "ده" (ten) are omitted on purpose: they collide with
    # the common function words "no/nor" and the verb stem "give". Because the
    # candidate is the Persian side, a stray match would invent a mismatch.
    # The unambiguous ordinals "نهم"/"دهم" are still recognised below.
    "یازده": 11,   # yazdah
    "دوازده": 12,  # davazdah
}

# Ordinal position -> the words that can express it, in either language.
_ORDINALS: dict[str, int] = {
    # English
    "first": 1, "firstly": 1, "second": 2, "secondly": 2,
    "third": 3, "thirdly": 3, "fourth": 4, "fourthly": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10,
    # Persian
    "نخست": 1,          # nokhost
    "اول": 1,                # avval
    "دوم": 2,                # dovvom
    "سوم": 3,                # sevvom
    "چهارم": 4,
    "پنجم": 5,
    "ششم": 6,
    "هفتم": 7,
    "هشتم": 8,
    "نهم": 9,
    "دهم": 10,
}

_WORD_RE = re.compile(r"[\w؀-ۿ]+", re.UNICODE)
_DIGIT_TRANSLATION = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹"
    "٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)
# "(1)" / "( ۲ )" style list markers, in either digit script.
_LIST_MARKER_RE = re.compile(r"[(\[]\s*([0-9۰-۹]{1,2})\s*[)\]]")


@dataclass
class StructureFinding:
    """One structural observation. ``classification`` drives the action."""

    check_id: str
    classification: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def severity(self) -> str:
        return SEVERITY.get(self.classification, "info")

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "classification": self.classification,
            "severity": self.severity,
            "message": self.message,
            "details": dict(self.details),
        }


def _words(text: str) -> list[str]:
    return [word.casefold() for word in _WORD_RE.findall(text or "")]


def announced_counts(text: str) -> list[int]:
    """Cardinals that could announce an enumeration, in order of appearance.

    Digits are included only when written as a bare small integer, because a
    year or a page number is not an announcement.
    """
    found: list[int] = []
    for word in _words(text):
        if word in _CARDINALS:
            found.append(_CARDINALS[word])
            continue
        # A single bare digit can announce ("2 objections"); a multi-digit
        # number is a year, page or quantity, never an announcement.
        normalised = word.translate(_DIGIT_TRANSLATION)
        if len(normalised) == 1 and normalised.isdigit() and int(normalised) >= 2:
            found.append(int(normalised))
    return found


def ordinal_sequence_length(text: str) -> int:
    """Length of the longest run of ordinals starting at "first".

    A run must start at 1 and increase by 1. Anything else is prose that merely
    happens to contain an ordinal, and is not treated as an enumeration.
    """
    positions = [_ORDINALS[word] for word in _words(text) if word in _ORDINALS]
    if not positions or positions[0] != 1:
        return 0
    length = 1
    for value in positions[1:]:
        if value == length + 1:
            length = value
        elif value <= length:
            continue  # a repeated or back-reference ordinal, not a new item
        else:
            break
    return length


def list_marker_count(text: str) -> int:
    """Length of a ``(1) (2) (3)`` style marker run, starting at 1."""
    values = [
        int(match.group(1).translate(_DIGIT_TRANSLATION))
        for match in _LIST_MARKER_RE.finditer(text or "")
    ]
    if not values or values[0] != 1:
        return 0
    length = 1
    for value in values[1:]:
        if value == length + 1:
            length = value
    return length


def enumeration_length(text: str) -> int:
    """The strongest enumeration evidence available in *text*."""
    return max(ordinal_sequence_length(text), list_marker_count(text))


def audit_structure(source: str, candidate: str) -> list[StructureFinding]:
    """Compare announced structure in *source* against *candidate*.

    Returns findings only; nothing here rejects a translation.
    """
    source_items = enumeration_length(source)
    candidate_items = enumeration_length(candidate)
    if source_items < 2 and candidate_items < 2:
        # No enumeration on either side: there is nothing reliable to audit, and
        # guessing from bare counting words would flood the report.
        return []

    source_counts = announced_counts(source)
    candidate_counts = announced_counts(candidate)
    source_announced = next(
        (value for value in source_counts if value == source_items), None
    )
    if source_announced is None:
        source_announced = source_counts[0] if source_counts else None
    candidate_announced = next(
        (value for value in candidate_counts if value == candidate_items), None
    )
    if candidate_announced is None:
        candidate_announced = candidate_counts[0] if candidate_counts else None

    details: dict[str, Any] = {
        "source_announced": source_announced,
        "source_items": source_items,
        "candidate_announced": candidate_announced,
        "candidate_items": candidate_items,
    }

    if source_announced is None or source_items < 2:
        return [
            StructureFinding(
                "announced_count_unverifiable",
                INSUFFICIENT_SOURCE_EVIDENCE,
                "The translation enumerates items but the source announcement "
                "could not be established; no automatic action.",
                details,
            )
        ]

    source_consistent = source_announced == source_items
    candidate_consistent = (
        candidate_announced is not None and candidate_announced == candidate_items
    )

    if not source_consistent:
        # The source contradicts itself. Preserving that is correct; silently
        # repairing it is not.
        # A silent repair changes the ANNOUNCEMENT to agree with the items it
        # actually lists. Comparing item counts instead would miss it, because a
        # faithful translation and a tidied one list the same number of items.
        if candidate_consistent and candidate_announced != source_announced:
            return [
                StructureFinding(
                    "announced_count_source_corrected",
                    UNAUTHORIZED_SOURCE_CORRECTION,
                    "The source announcement and its enumeration disagree, and "
                    "the translation silently reconciles them.",
                    details,
                )
            ]
        return [
            StructureFinding(
                "announced_count_source_anomaly",
                SOURCE_ANOMALY_PRESERVED,
                "The source announcement and its enumeration disagree; the "
                "translation preserves the discrepancy.",
                details,
            )
        ]

    if candidate_items != source_items or candidate_announced != source_announced:
        return [
            StructureFinding(
                "announced_count_mismatch",
                TRANSLATION_STRUCTURE_MISMATCH,
                "The source announces and delivers a consistent number of "
                "items; the translation does not match it.",
                details,
            )
        ]
    return []


def audit_payload(source: str, candidate: str) -> dict[str, Any]:
    """QA-report friendly summary of :func:`audit_structure`."""
    findings = audit_structure(source, candidate)
    return {
        "finding_count": len(findings),
        "classifications": sorted({f.classification for f in findings}),
        "findings": [f.to_dict() for f in findings],
    }
