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
_ANNOUNCEMENT_CATEGORIES: dict[str, str] = {
    # Argumentative units. These are deliberately language concepts rather
    # than book terminology, so ordinary translation synonyms can align.
    **dict.fromkeys((
        "issue", "issues", "objection", "objections", "point", "points",
        "reason", "reasons", "argument", "arguments", "question", "questions",
        "problem", "problems", "claim", "claims", "proposition", "propositions",
        "theme", "themes", "مسئله", "ایراد", "نکته", "دلیل", "استدلال",
        "پرسش", "مشکل", "ادعا", "گزاره", "مضمون",
    ), "argument_item"),
    **dict.fromkeys((
        "perspective", "perspectives", "approach", "approaches", "way", "ways",
        "axis", "axes", "منظر", "رویکرد", "شیوه", "محور", "جهت", "جهات",
        "راه", "روش",
    ), "approach"),
    **dict.fromkeys((
        "element", "elements", "dimension", "dimensions", "factor", "factors",
        "feature", "features", "difference", "differences", "عنصر", "بعد",
        "عامل", "ویژگی", "تفاوت", "جنبه",
    ), "component"),
    **dict.fromkeys((
        "task", "tasks", "objective", "objectives", "step", "steps",
        "وظیفه", "هدف", "گام",
    ), "procedure"),
    **dict.fromkeys((
        "part", "parts", "category", "categories", "type", "types", "case",
        "cases", "بخش", "دسته", "نوع", "مورد",
    ), "classification"),
    **dict.fromkeys(("chapter", "chapters", "فصل"), "chapter"),
    **dict.fromkeys(("source", "sources", "منبع"), "source"),
}
_ANNOUNCEMENT_NOUNS = set(_ANNOUNCEMENT_CATEGORIES)
_PERSIAN_ANNOUNCEMENT_SUFFIXES = (
    "\u200c\u0647\u0627\u06cc\u06cc",
    "\u0647\u0627\u06cc\u06cc",
    "\u200c\u0647\u0627\u06cc",
    "\u0647\u0627\u06cc",
    "\u200c\u0647\u0627",
    "\u0647\u0627",
    "\u06cc",
)
_DIGIT_TRANSLATION = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹"
    "٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)
# "(1)" / "( ۲ )" style list markers, in either digit script.
_LIST_MARKER_RE = re.compile(r"[(\[]\s*([0-9۰-۹]{1,2})\s*[)\]]")
_CHAPTER_MARKER_RE = re.compile(
    r"(?<!\w)(?:chapter|فصل)\s*([0-9۰-۹]{1,2})(?!\d)",
    re.IGNORECASE,
)


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


def _announcement_category(token: str) -> str | None:
    if token in _ANNOUNCEMENT_CATEGORIES:
        return _ANNOUNCEMENT_CATEGORIES[token]
    for suffix in _PERSIAN_ANNOUNCEMENT_SUFFIXES:
        if token.endswith(suffix):
            stem = token[:-len(suffix)]
            if stem in _ANNOUNCEMENT_CATEGORIES:
                return _ANNOUNCEMENT_CATEGORIES[stem]
    return None


def _is_announcement_noun(token: str) -> bool:
    return _announcement_category(token) is not None


@dataclass(frozen=True)
class AnnouncementEvidence:
    """One exact count-before-noun announcement in source or target prose."""

    value: int
    semantic_category: str
    count_span: tuple[int, int]
    noun_span: tuple[int, int]
    exact_span: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "semantic_category": self.semantic_category,
            "count_span": list(self.count_span),
            "noun_span": list(self.noun_span),
            "exact_span": self.exact_span,
            "evidence_level": "exact_typed_local_span",
        }


def announced_count_evidence(text: str) -> list[AnnouncementEvidence]:
    """Return typed announcements, excluding noun-before-number references.

    English and Persian cardinal announcements put the count before their head
    noun (``eight sources`` / ``هشت منبع``). References such as ``chapter 5``
    and ``فصل ۵`` use the reverse order and therefore cannot be mistaken for an
    announced quantity.
    """
    matches = list(_WORD_RE.finditer(text or ""))
    words = [match.group(0).casefold() for match in matches]
    evidence: list[AnnouncementEvidence] = []
    for index, word in enumerate(words):
        value: int | None = _CARDINALS.get(word)
        if value is None:
            normalised = word.translate(_DIGIT_TRANSLATION)
            if (
                len(normalised) == 1
                and normalised.isdecimal()
                and int(normalised) >= 2
            ):
                value = int(normalised)
        if value is None:
            continue
        # Only a following noun can be governed by this count. Stop at another
        # number so two unrelated quantities are never bridged.
        for noun_index in range(index + 1, min(len(words), index + 6)):
            if words[noun_index] in _CARDINALS or words[noun_index].translate(
                _DIGIT_TRANSLATION
            ).isdecimal():
                break
            category = _announcement_category(words[noun_index])
            if category is None:
                continue
            start = matches[index].start()
            end = matches[noun_index].end()
            evidence.append(AnnouncementEvidence(
                value=value,
                semantic_category=category,
                count_span=(matches[index].start(), matches[index].end()),
                noun_span=(matches[noun_index].start(), matches[noun_index].end()),
                exact_span=(text or "")[start:end],
            ))
            break
    return evidence


def announced_counts(text: str) -> list[int]:
    """Cardinals that could announce an enumeration, in order of appearance.

    Digits are included only when written as a bare small integer, because a
    year or a page number is not an announcement.
    """
    return [item.value for item in announced_count_evidence(text)]


def ordinal_sequence_length(text: str) -> int:
    """Length of the longest run of ordinals starting at "first".

    A run must start at 1 and increase by 1. Anything else is prose that merely
    happens to contain an ordinal, and is not treated as an enumeration.
    """
    positions = [_ORDINALS[word] for word in _words(text) if word in _ORDINALS]
    longest = 0
    for start, position in enumerate(positions):
        if position != 1:
            continue
        length = 1
        for value in positions[start + 1:]:
            if value == length + 1:
                length = value
            elif value <= length:
                continue  # repeated/back-reference ordinal
            else:
                break
        longest = max(longest, length)
    return longest


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


def chapter_marker_count(text: str) -> int:
    """Return the longest consecutive ``Chapter 2, 3, 4`` style run."""
    values = [
        int(match.group(1).translate(_DIGIT_TRANSLATION))
        for match in _CHAPTER_MARKER_RE.finditer(text or "")
    ]
    longest = 0
    for start, first in enumerate(values):
        length = 1
        previous = first
        for value in values[start + 1:]:
            if value == previous + 1:
                length += 1
                previous = value
            elif value == previous:
                continue
            else:
                break
        longest = max(longest, length)
    return longest if longest >= 2 else 0


def enumeration_length(text: str) -> int:
    """The strongest enumeration evidence available in *text*."""
    return max(
        ordinal_sequence_length(text),
        list_marker_count(text),
        chapter_marker_count(text),
    )


@dataclass(frozen=True)
class _StructuralEpisode:
    paragraph_index: int
    paragraph_span: int
    announced: int | None
    items: int
    announcement_candidates: tuple[int, ...]
    semantic_category: str
    announcement_evidence: tuple[AnnouncementEvidence, ...]


def _enumeration_evidence(text: str) -> tuple[int, str]:
    """Return the strongest local enumeration and its semantic category."""
    ordinal = ordinal_sequence_length(text)
    listed = list_marker_count(text)
    chapters = chapter_marker_count(text)
    strongest = max(ordinal, listed, chapters)
    if chapters == strongest and chapters > max(ordinal, listed):
        return chapters, "chapter"
    return strongest, "generic"


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
        all_evidence = announced_count_evidence(paragraph)
        if not all_evidence:
            continue
        items, enumeration_category = _enumeration_evidence(paragraph)
        span = 1
        if items < minimum_items and index + 1 < len(paragraphs):
            next_items, next_category = _enumeration_evidence(paragraphs[index + 1])
            if next_items >= 2:
                items = next_items
                enumeration_category = next_category
                span = 2
        if items < minimum_items:
            continue
        evidence = [
            item for item in all_evidence
            if (
                enumeration_category == "generic"
                and item.semantic_category not in {"chapter", "source"}
            )
            or item.semantic_category == enumeration_category
        ]
        if not evidence:
            continue
        categories = {item.semantic_category for item in evidence}
        category = next(iter(categories)) if len(categories) == 1 else "ambiguous"
        counts = [item.value for item in evidence]
        announced = next((value for value in counts if value == items), None)
        if announced is None and len(set(counts)) == 1 and category != "ambiguous":
            announced = counts[0]
        episodes.append(_StructuralEpisode(
            paragraph_index=index,
            paragraph_span=span,
            announced=announced,
            items=items,
            announcement_candidates=tuple(counts),
            semantic_category=category,
            announcement_evidence=tuple(evidence),
        ))
    return episodes


def _direct_announcement_mismatches(
    source: str,
    candidate: str,
    *,
    covered_source_paragraphs: set[int],
) -> list[StructureFinding]:
    """Compare one explicit count announcement per aligned paragraph.

    This fills the page/chunk-boundary gap where a source says "two issues" but
    the enumeration appears later. Multiple announcements are deliberately left
    unverifiable to avoid pairing unrelated quantities by position.
    """
    findings: list[StructureFinding] = []
    source_paragraphs = _paragraphs(source)
    candidate_paragraphs = _paragraphs(candidate)
    for index, source_paragraph in enumerate(source_paragraphs):
        if index in covered_source_paragraphs or index >= len(candidate_paragraphs):
            continue
        source_evidence = announced_count_evidence(source_paragraph)
        candidate_evidence = announced_count_evidence(candidate_paragraphs[index])
        source_by_category: dict[str, list[AnnouncementEvidence]] = {}
        candidate_by_category: dict[str, list[AnnouncementEvidence]] = {}
        for item in source_evidence:
            source_by_category.setdefault(item.semantic_category, []).append(item)
        for item in candidate_evidence:
            candidate_by_category.setdefault(item.semantic_category, []).append(item)
        for category in sorted(source_by_category.keys() & candidate_by_category.keys()):
            source_items = source_by_category[category]
            candidate_items = candidate_by_category[category]
            if len(source_items) != 1 or len(candidate_items) != 1:
                continue
            source_item = source_items[0]
            candidate_item = candidate_items[0]
            if source_item.value == candidate_item.value:
                continue
            findings.append(StructureFinding(
                "announced_count_lexical_mismatch",
                TRANSLATION_STRUCTURE_MISMATCH,
                "The translation changes an explicit source announcement count.",
                {
                    "source_announced": source_item.value,
                    "candidate_announced": candidate_item.value,
                    "semantic_category": category,
                    "source_announcement": source_item.to_dict(),
                    "candidate_announcement": candidate_item.to_dict(),
                    "source_paragraph": index,
                    "candidate_paragraph": index,
                    "source_excerpt": source_paragraph[:360],
                    "candidate_excerpt": candidate_paragraphs[index][:360],
                    "admission": "blocking",
                },
            ))
    return findings


def audit_structure(source: str, candidate: str) -> list[StructureFinding]:
    """Compare announced structure in *source* against *candidate*.

    Returns findings only; nothing here rejects a translation.
    """
    source_episodes = _structural_episodes(source)
    candidate_episodes = _structural_episodes(candidate, minimum_items=1)
    if not source_episodes:
        direct = _direct_announcement_mismatches(
            source,
            candidate,
            covered_source_paragraphs=set(),
        )
        if direct:
            return direct
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
    candidate_by_paragraph = {
        episode.paragraph_index: episode for episode in candidate_episodes
    }
    source_paragraphs = _paragraphs(source)
    candidate_paragraphs = _paragraphs(candidate)
    for source_episode in source_episodes:
        candidate_episode = candidate_by_paragraph.get(
            source_episode.paragraph_index
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
            "semantic_category": source_episode.semantic_category,
            "source_announcement_evidence": [
                item.to_dict() for item in source_episode.announcement_evidence
            ],
            "source_excerpt": source_paragraphs[
                source_episode.paragraph_index
            ][:360],
            "candidate_excerpt": (
                candidate_paragraphs[candidate_episode.paragraph_index][:360]
                if candidate_episode
                and candidate_episode.paragraph_index < len(candidate_paragraphs)
                else ""
            ),
            "admission": (
                "blocking"
                if source_episode.semantic_category != "ambiguous"
                else "review"
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
    findings.extend(_direct_announcement_mismatches(
        source,
        candidate,
        covered_source_paragraphs={
            episode.paragraph_index for episode in source_episodes
        },
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
