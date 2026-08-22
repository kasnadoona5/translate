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

# U+0600-U+06FF includes Persian punctuation such as U+060C, so the old
# expression returned an ordinal with its comma attached. Unicode ``\w``
# already recognises Persian letters and digits; exclude underscore.
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_ANNOUNCEMENT_NOUNS = {
    "issue", "issues", "objection", "objections", "task", "tasks", "point", "points",
    "reason", "reasons", "argument", "arguments", "question", "questions",
    "problem", "problems", "perspective", "perspectives", "approach", "approaches",
    "element", "elements", "objective", "objectives", "theme", "themes",
    "part", "parts", "way", "ways", "claim", "claims", "proposition", "propositions",
    "dimension", "dimensions", "category", "categories", "type", "types",
    "case", "cases", "step", "steps", "factor", "factors", "feature", "features",
    "مسئله", "ایراد", "وظیفه", "نکته", "دلیل", "استدلال", "پرسش", "مشکل", "منظر",
    "رویکرد", "عنصر", "هدف", "مضمون", "بخش", "شیوه", "ادعا", "گزاره",
    "بعد", "دسته", "نوع", "مورد", "گام", "عامل", "ویژگی",
}
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
    words = _words(text)
    found: list[int] = []
    for index, word in enumerate(words):
        value: int | None = None
        if word in _CARDINALS:
            value = _CARDINALS[word]
        else:
            # A single bare digit can announce ("2 objections"); a multi-digit
            # number is a year, page or quantity, never an announcement.
            normalised = word.translate(_DIGIT_TRANSLATION)
            if (
                len(normalised) == 1
                and normalised.isdecimal()
                and int(normalised) >= 2
            ):
                value = int(normalised)
        if value is None:
            continue
        # Bind the count to a nearby structural noun. This distinguishes
        # "two issues" from unrelated prose such as "the two authors", even
        # when an enumeration appears later in the same chunk.
        neighborhood = words[max(0, index - 2): index]
        neighborhood += words[index + 1: index + 6]
        if any(token in _ANNOUNCEMENT_NOUNS for token in neighborhood):
            found.append(value)
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


@dataclass(frozen=True)
class _StructuralEpisode:
    paragraph_index: int
    paragraph_span: int
    announced: int | None
    items: int
    announcement_candidates: tuple[int, ...]


def _paragraphs(text: str) -> list[str]:
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text or "")
        if paragraph.strip()
    ]


def _structural_episodes(
    text: str,
    *,
    minimum_items: int = 2,
) -> list[_StructuralEpisode]:
    """Bind an announcement only to its paragraph or the next list paragraph.

    Chunk-wide maxima confuse unrelated lists, such as a three-part argument
    followed later by eight acknowledgements. The one-paragraph look-ahead is
    deliberately narrow and supports ordinary ``the following`` list layouts.
    """
    paragraphs = _paragraphs(text)
    episodes: list[_StructuralEpisode] = []
    for index, paragraph in enumerate(paragraphs):
        counts = announced_counts(paragraph)
        if not counts:
            continue
        items = enumeration_length(paragraph)
        span = 1
        if items < minimum_items and index + 1 < len(paragraphs):
            next_items = enumeration_length(paragraphs[index + 1])
            if next_items >= 2:
                items = next_items
                span = 2
        if items < minimum_items:
            continue
        announced = next((value for value in counts if value == items), None)
        if announced is None and len(set(counts)) == 1:
            announced = counts[0]
        episodes.append(_StructuralEpisode(
            paragraph_index=index,
            paragraph_span=span,
            announced=announced,
            items=items,
            announcement_candidates=tuple(counts),
        ))
    return episodes


def audit_structure(source: str, candidate: str) -> list[StructureFinding]:
    """Compare announced structure in *source* against *candidate*.

    Returns findings only; nothing here rejects a translation.
    """
    source_episodes = _structural_episodes(source)
    candidate_episodes = _structural_episodes(candidate, minimum_items=1)
    if not source_episodes:
        if not candidate_episodes:
            return []
        episode = candidate_episodes[0]
        return [StructureFinding(
            "announced_count_unverifiable",
            INSUFFICIENT_SOURCE_EVIDENCE,
            "The translation enumerates items but a local source announcement "
            "could not be established; no automatic action.",
            {
                "source_announced": None,
                "source_items": 0,
                "candidate_announced": episode.announced,
                "candidate_items": episode.items,
                "candidate_paragraph": episode.paragraph_index,
            },
        )]

    findings: list[StructureFinding] = []
    for index, source_episode in enumerate(source_episodes):
        candidate_episode = (
            candidate_episodes[index]
            if index < len(candidate_episodes) else None
        )
        candidate_announced = (
            candidate_episode.announced if candidate_episode else None
        )
        candidate_items = candidate_episode.items if candidate_episode else 0
        details: dict[str, Any] = {
            "source_announced": source_episode.announced,
            "source_items": source_episode.items,
            "candidate_announced": candidate_announced,
            "candidate_items": candidate_items,
            "source_paragraph": source_episode.paragraph_index,
            "source_paragraph_span": source_episode.paragraph_span,
            "candidate_paragraph": (
                candidate_episode.paragraph_index if candidate_episode else None
            ),
            "source_announcement_candidates": list(
                source_episode.announcement_candidates
            ),
        }
        if source_episode.announced is None:
            findings.append(StructureFinding(
                "announced_count_unverifiable",
                INSUFFICIENT_SOURCE_EVIDENCE,
                "Several local source counts could apply to this enumeration; "
                "no automatic action.",
                details,
            ))
            continue

        source_consistent = source_episode.announced == source_episode.items
        candidate_consistent = (
            candidate_announced is not None
            and candidate_announced == candidate_items
        )
        if not source_consistent:
            if (
                candidate_consistent
                and candidate_announced != source_episode.announced
            ):
                findings.append(StructureFinding(
                    "announced_count_source_corrected",
                    UNAUTHORIZED_SOURCE_CORRECTION,
                    "The source announcement and its local enumeration disagree, "
                    "and the translation silently reconciles them.",
                    details,
                ))
            else:
                findings.append(StructureFinding(
                    "announced_count_source_anomaly",
                    SOURCE_ANOMALY_PRESERVED,
                    "The source announcement and its local enumeration disagree; "
                    "the translation preserves the discrepancy.",
                    details,
                ))
            continue

        if (
            candidate_items != source_episode.items
            or candidate_announced != source_episode.announced
        ):
            findings.append(StructureFinding(
                "announced_count_mismatch",
                TRANSLATION_STRUCTURE_MISMATCH,
                "The source announces and delivers a locally consistent number "
                "of items; the translation does not match it.",
                details,
            ))
    return findings


def audit_payload(source: str, candidate: str) -> dict[str, Any]:
    """QA-report friendly summary of :func:`audit_structure`."""
    findings = audit_structure(source, candidate)
    return {
        "finding_count": len(findings),
        "classifications": sorted({f.classification for f in findings}),
        "findings": [f.to_dict() for f in findings],
    }
