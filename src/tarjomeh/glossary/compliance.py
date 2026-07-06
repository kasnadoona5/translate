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

        for entry in matched_entries:
            if not self._target_present(entry.target, translation):
                escaped_source = re.escape(entry.source)
                if re.search(rf"\b{escaped_source}\b", translation, re.IGNORECASE):
                    status = "wrong"
                else:
                    target_norm = _normalise_persian(entry.target)
                    translation_norm = _normalise_persian(translation)
                    if len(target_norm) >= 4 and target_norm[:3] in translation_norm:
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
            total_checked=len(matched_entries),
        )

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _target_present(target: str, translation: str) -> bool:
        """Check if the Persian *target* term exists in *translation*.

        Uses a permissive match that accounts for:
        * Zero-width non-joiner (ZWNJ, ``\\u200c``) variations
        * Optional whitespace around the term

        For Persian text we do **not** use ``\\b`` because word-boundary
        semantics are unreliable with the Arabic script.  Instead we check
        plain substring presence after normalising ZWNJ.
        """
        normalised_target = _normalise_persian(target)
        normalised_text = _normalise_persian(translation)
        return normalised_target in normalised_text


# ---------------------------------------------------------------------------
# Persian normalisation helpers
# ---------------------------------------------------------------------------

_ZWNJ = "\u200c"  # zero-width non-joiner
_ZWJ = "\u200d"   # zero-width joiner


def _normalise_persian(text: str) -> str:
    """Normalise Persian text for comparison.

    * Strips diacritics (tashkeel / اعراب)
    * Normalises Arabic ي / ك to Persian ی / ک
    * Collapses multiple spaces
    """
    # Arabic → Persian letter normalisation
    text = text.replace("\u064a", "\u06cc")   # ي → ی
    text = text.replace("\u0643", "\u06a9")   # ك → ک
    # Remove Arabic tashkeel
    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()
