"""Post-translation glossary compliance verification.

Checks that every glossary term present in the source text has its
prescribed Persian translation in the output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tarjomeh.glossary.manager import GlossaryEntry, GlossaryManager


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    """A single glossary compliance failure.

    Attributes:
        term:           The English source term from the glossary.
        expected:       The prescribed Persian translation.
        status:         ``"missing"`` or ``"wrong"`` (if a different
                        translation was detected — currently always
                        ``"missing"`` since automatic "wrong" detection
                        requires morphological analysis).
        chunk_location: Human-readable location hint, e.g.
                        ``"Chapter 3 / Section 2, chunk 17"``.
    """

    term: str
    expected: str
    status: str = "missing"
    chunk_location: str = ""


@dataclass
class ComplianceReport:
    """Aggregated result of a glossary compliance check.

    Attributes:
        violations:     List of detected violations.
        total_checked:  How many glossary terms were found in the source.
        compliant:      ``True`` when *violations* is empty.
    """

    violations: list[Violation] = field(default_factory=list)
    total_checked: int = 0
    citation_exemptions: list[str] = field(default_factory=list)

    @property
    def compliant(self) -> bool:
        """Return ``True`` when there are no violations."""
        return len(self.violations) == 0

    def summary(self) -> str:
        """One-line human-readable summary."""
        if self.compliant:
            return f"✅ All {self.total_checked} glossary terms correctly used."
        n = len(self.violations)
        return (
            f"⚠️  {n} glossary violation(s) out of "
            f"{self.total_checked} checked term(s)."
        )


# ---------------------------------------------------------------------------
# Compliance checker
# ---------------------------------------------------------------------------

class GlossaryComplianceChecker:
    """Verify that glossary terms in source text appear correctly in output.

    Usage::

        checker = GlossaryComplianceChecker()
        report = checker.check(
            translation="متن ترجمه‌شده …",
            source_text="The hegemony of …",
            glossary_manager=glossary,
            chunk_location="Chapter 1, chunk 3",
        )
        if not report.compliant:
            for v in report.violations:
                print(v)
    """

    def check(
        self,
        translation: str,
        source_text: str,
        glossary_manager: GlossaryManager,
        chunk_location: str = "",
    ) -> ComplianceReport:
        """Run compliance check and return a :class:`ComplianceReport`.

        Args:
            translation:      The Persian translation output.
            source_text:      The English source text of the same chunk.
            glossary_manager: Active glossary with prescribed terms.
            chunk_location:   Optional location string for diagnostics.

        Returns:
            A :class:`ComplianceReport` with any violations.
        """
        # Step 1: find which glossary terms appear in the source.
        matched_entries = glossary_manager.find_terms(source_text)

        violations: list[Violation] = []
        citation_exemptions: list[str] = []

        for entry in matched_entries:
            if term_occurs_only_in_citations(source_text, entry.source):
                citation_exemptions.append(entry.source)
                continue
            if not self._target_present(entry.target, translation):
                escaped_source = re.escape(entry.source)
                if re.search(rf"\b{escaped_source}\b", translation, re.IGNORECASE):
                    status = "wrong"
                else:
                    status = "missing"

                violations.append(
                    Violation(
                        term=entry.source,
                        expected=entry.target,
                        status=status,
                        chunk_location=chunk_location,
                    )
                )

        return ComplianceReport(
            violations=violations,
            total_checked=len(matched_entries) - len(citation_exemptions),
            citation_exemptions=citation_exemptions,
        )

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _target_present(target: str, translation: str) -> bool:
        """Check if the Persian *target* term exists in *translation*.

        Uses a permissive match that accounts for:
        * Hazm-normalized Persian spelling/spacing
        * Arabic/Persian ي/ی and ك/ک variants
        * Zero-width non-joiner (ZWNJ, ``\\u200c``) versus whitespace variants
        * Common affixed forms such as plurals attached with ZWNJ

        For Persian text we do **not** use ``\\b`` because word-boundary
        semantics are unreliable with the Arabic script.
        """
        normalised_target = _normalise_persian(target)
        normalised_text = _normalise_persian(translation)
        if not normalised_target:
            return False
        # Use letter ranges rather than the whole Arabic block; that block also
        # contains Persian comma/semicolon characters, which are valid term
        # boundaries.
        persian_word = r"\u0621-\u063a\u0641-\u064a\u066e-\u06d3\u06fa-\u06ff"
        if re.search(
            rf"(?<![{persian_word}]){re.escape(normalised_target)}(?![{persian_word}])",
            normalised_text,
        ):
            return True
        intra_word_joiner = rf"[{_ZWNJ}{_ZWJ}]*"
        parts = [
            intra_word_joiner.join(re.escape(char) for char in part)
            for part in re.split(rf"[\s{_ZWNJ}{_ZWJ}]+", normalised_target)
            if part
        ]
        if not parts:
            return False
        flexible_target = rf"[\s{_ZWNJ}{_ZWJ}]*".join(parts)
        # Persian targets can take productive suffixes without a word boundary
        # (for example, a noun becoming an adjective or plural). Keep the
        # leading boundary so embedded substrings in unrelated words still fail.
        joiner = rf"[\s{_ZWNJ}{_ZWJ}]*"
        suffix = (
            rf"(?:{joiner}(?:"
            r"\u0647\u0627(?:\u06cc(?:\u06cc|\u0645|\u062a|\u0634|\u0645\u0627\u0646|\u062a\u0627\u0646|\u0634\u0627\u0646)?)?"
            r"|\u06cc|\u0627\u0646|\u0627\u062a|\u062a\u0631(?:\u06cc\u0646)?|\u0627\u0645|\u0627\u0634|\u0645\u0627\u0646|\u062a\u0627\u0646|\u0634\u0627\u0646"
            r"))?"
        )
        return re.search(
            rf"(?<![{persian_word}]){flexible_target}{suffix}(?![{persian_word}])",
            normalised_text,
        ) is not None


# ---------------------------------------------------------------------------
# Persian normalisation helpers
# ---------------------------------------------------------------------------

_ZWNJ = "\u200c"  # zero-width non-joiner
_ZWJ = "\u200d"   # zero-width joiner
_SCHOLARLY_CITATION_RE = re.compile(
    r"\([^()\n]{0,240}\b(?:1[5-9]\d{2}|20\d{2})[a-z]?\b[^()\n]{0,240}\)",
    re.IGNORECASE,
)


def term_occurs_only_in_citations(source_text: str, term: str) -> bool:
    """Return whether every source occurrence is inside an author-year citation."""
    escaped = re.escape(term.strip())
    if not escaped:
        return False
    occurrences = list(re.finditer(rf"\b{escaped}\b", source_text, re.IGNORECASE))
    if not occurrences:
        return False
    citation_spans = [
        match.span() for match in _SCHOLARLY_CITATION_RE.finditer(source_text)
    ]
    return bool(citation_spans) and all(
        any(
            start <= occurrence.start() and occurrence.end() <= end
            for start, end in citation_spans
        )
        for occurrence in occurrences
    )


def _normalise_persian(text: str) -> str:
    """Normalise Persian text for comparison.

    * Applies hazm normalisation when available
    * Strips diacritics (tashkeel / اعراب)
    * Normalises Arabic ي / ك to Persian ی / ک
    * Collapses multiple spaces around ZWNJ
    """
    try:
        from hazm import Normalizer  # type: ignore[import-untyped]
        text = Normalizer().normalize(text)
    except Exception:
        pass

    # Arabic → Persian letter normalisation
    text = text.replace("\u064a", "\u06cc")   # ي → ی
    text = text.replace("\u0643", "\u06a9")   # ك → ک
    text = text.replace(_ZWJ, _ZWNJ)
    # Remove Arabic tashkeel
    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    # Remove spaces around ZWNJ and collapse repeated ZWNJs.
    text = re.sub(rf"\s*{_ZWNJ}\s*", _ZWNJ, text)
    text = re.sub(rf"{_ZWNJ}+", _ZWNJ, text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _compact_persian(text: str) -> str:
    """Return text without whitespace/joiner distinctions for term matching."""
    return re.sub(rf"[\s{_ZWNJ}{_ZWJ}]+", "", text)
