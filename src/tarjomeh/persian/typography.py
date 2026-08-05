"""Persian typography post-processor for Tarjomeh.

Handles ZWNJ normalization, numeral conversion, punctuation fixing,
and tokenization using the hazm NLP library for Persian text.
"""

from __future__ import annotations

import re
from typing import Any

from tarjomeh.persian.orthography import apply_safe_persian_orthography

# ── Western → Persian digit mapping ──────────────────────────────────
_WESTERN_TO_PERSIAN: dict[str, str] = {
    "0": "۰",
    "1": "۱",
    "2": "۲",
    "3": "۳",
    "4": "۴",
    "5": "۵",
    "6": "۶",
    "7": "۷",
    "8": "۸",
    "9": "۹",
}

_DIGIT_PATTERN = re.compile(r"[0-9]")

# ── Latin → Persian punctuation mapping ──────────────────────────────
_PUNCT_MAP: dict[str, str] = {
    "?": "؟",
    ";": "؛",
    ",": "،",
}
# Note: period '.' is intentionally *not* mapped — the spec says
# ". to ." (i.e. period stays period in Persian too).

# ── Scholarly apparatus protection ───────────────────────────────────
# Spans matched here are shielded from numeral AND punctuation conversion
# so citations survive intact: (Marx 1973, 408) must NOT become
# (مارکس ۱۹۷۳، ۴۰۸). Protected patterns:
#   1. Parenthesised runs containing digits but NO Persian/Arabic letters
#      (in-text citations: "(1973, 408)", "(Marx 1867, 92)", "(cf. 2014)")
#   2. Bracketed markers: "[12]", "[see 3-5]"
#   3. Page/volume references: "p. 45", "pp. 12–34", "vol. 3", "no. 7"
#   4. Standalone Gregorian years 1500–2099 (incl. "1973a" style)
_SCHOLARLY_PROTECTED_RE = re.compile(
    r"\([^)؀-ۿ]*[0-9][^)؀-ۿ]*\)"
    r"|\[[^\]]*?[0-9][^\]]*?\]"
    r"|\b(?:pp?|vols?|nos?|chs?|fols?)\.\s*[0-9]+(?:\s*[-–—,]\s*[0-9]+)*"
    r"|\b(?:1[5-9][0-9]{2}|20[0-9]{2})[a-z]?\b",
    re.IGNORECASE,
)

# Digit-free Private-Use-Area sentinels: survive hazm normalisation and are
# untouched by numeral/punctuation conversion.
_SENTINEL_OPEN = ""
_SENTINEL_CLOSE = ""
_SENTINEL_BASE = 0xE100
_SENTINEL_RE = re.compile(f"{_SENTINEL_OPEN}(.){_SENTINEL_CLOSE}")


class PersianTypographer:
    """Applies configurable Persian typography post-processing.

    Each processing step can be independently enabled or disabled via
    the ``[persian]`` section of the Tarjomeh configuration file.

    Parameters
    ----------
    config : dict[str, Any]
        The ``[persian]`` section of the TOML configuration.  Expected
        keys (all default to ``True`` when absent):

        * ``convert_numerals`` – replace Western 0-9 with ۰-۹
        * ``normalize_zwnj``  – apply hazm ZWNJ normalization
        * ``fix_punctuation``  – swap Latin punctuation for Persian
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._convert_numerals: bool = cfg.get("convert_numerals", True)
        self._normalize_zwnj: bool = cfg.get("normalize_zwnj", True)
        self._fix_punctuation: bool = cfg.get("fix_punctuation", True)
        # Scholarly mode shields citations, years, page numbers, and footnote
        # markers from numeral/punctuation conversion (default ON — required
        # for academic books; set false for non-scholarly texts).
        self._scholarly_mode: bool = cfg.get("scholarly_mode", True)

        # Lazy-init hazm normalizer on first use (avoids import cost
        # when hazm isn't available or the step is disabled).
        self._hazm_normalizer: Any | None = None

    # ── private helpers ──────────────────────────────────────────────

    def _get_hazm_normalizer(self) -> Any:
        """Return a cached ``hazm.Normalizer`` instance."""
        if self._hazm_normalizer is None:
            try:
                from hazm import Normalizer  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "hazm is required for ZWNJ normalization. "
                    "Install it with:  pip install hazm"
                ) from exc
            self._hazm_normalizer = Normalizer()
        return self._hazm_normalizer

    # ── public API ───────────────────────────────────────────────────

    def process(self, text: str) -> str:
        """Apply all *enabled* post-processing steps in order.

        Processing order:
        1. ZWNJ normalization (hazm)
        2. Numeral conversion
        3. Punctuation fixing

        Parameters
        ----------
        text:
            Raw Persian text from the translator.

        Returns
        -------
        str
            Post-processed text ready for export.
        """
        if not text:
            return text

        # Shield scholarly apparatus BEFORE any transformation (hazm itself
        # may also touch digits), restore afterwards.
        protected_spans: list[str] = []
        if self._scholarly_mode:
            text = self._protect_scholarly(text, protected_spans)

        if self._normalize_zwnj:
            text = self.normalize_zwnj(text)
        if self._convert_numerals:
            text = self.convert_numerals(text)
        if self._fix_punctuation:
            text = self.fix_punctuation(text)

        if protected_spans:
            text = self._restore_scholarly(text, protected_spans)

        # Hazm deliberately avoids some lexical compounds. Finish with a very
        # small boundary-aware rule set whose edits are always orthographic.
        text, _ = apply_safe_persian_orthography(text)

        return text

    def process_with_report(
        self,
        text: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Process text and return the conservative orthography edit ledger."""
        if not text:
            return text, []
        processed = self.process(text)
        # Report only edits still recognizable from the input. Other enabled
        # typography behavior remains unchanged and is covered by its tests.
        _, edits = apply_safe_persian_orthography(text)
        return processed, edits

    # ── scholarly apparatus protection ───────────────────────────────

    @staticmethod
    def _protect_scholarly(text: str, spans: list[str]) -> str:
        """Replace scholarly-apparatus spans with digit-free PUA sentinels."""

        def _stash(match: re.Match[str]) -> str:
            spans.append(match.group(0))
            return f"{_SENTINEL_OPEN}{chr(_SENTINEL_BASE + len(spans) - 1)}{_SENTINEL_CLOSE}"

        return _SCHOLARLY_PROTECTED_RE.sub(_stash, text)

    @staticmethod
    def _restore_scholarly(text: str, spans: list[str]) -> str:
        """Restore original spans stashed by :meth:`_protect_scholarly`."""

        def _unstash(match: re.Match[str]) -> str:
            index = ord(match.group(1)) - _SENTINEL_BASE
            if 0 <= index < len(spans):
                return spans[index]
            return match.group(0)

        return _SENTINEL_RE.sub(_unstash, text)

    # ── individual steps (also usable standalone) ────────────────────

    def normalize_zwnj(self, text: str) -> str:
        """Normalize Zero-Width Non-Joiner placement using *hazm*.

        The hazm ``Normalizer`` corrects common ZWNJ misplacements in
        Persian text (e.g. verb prefixes, plural suffixes).

        Parameters
        ----------
        text:
            Persian text to normalize.

        Returns
        -------
        str
            Text with corrected ZWNJ characters.
        """
        normalizer = self._get_hazm_normalizer()
        return normalizer.normalize(text)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize Persian text using hazm's word tokenizer.

        Parameters
        ----------
        text:
            Persian sentence or paragraph to tokenize.

        Returns
        -------
        list[str]
            List of word tokens.
        """
        try:
            from hazm import word_tokenize  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "hazm is required for Persian tokenization. "
                "Install it with:  pip install hazm"
            ) from exc
        return word_tokenize(text)

    @staticmethod
    def convert_numerals(text: str) -> str:
        """Convert Western digits (0-9) to Persian digits (۰-۹).

        Parameters
        ----------
        text:
            Text potentially containing Western numerals.

        Returns
        -------
        str
            Text with all Western digits replaced by their Persian
            equivalents.
        """
        return _DIGIT_PATTERN.sub(
            lambda m: _WESTERN_TO_PERSIAN[m.group()], text
        )

    @staticmethod
    def fix_punctuation(text: str) -> str:
        """Convert Western punctuation to Persian equivalents.

        Mapping:
        * ``?`` → ``؟``  (question mark)
        * ``;`` → ``؛``  (semicolon)
        * ``,`` → ``،``  (comma)

        The period (``.``) is intentionally kept as-is because Persian
        typography also uses the standard period.

        Parameters
        ----------
        text:
            Text with potential Latin punctuation.

        Returns
        -------
        str
            Text with Persian punctuation characters.
        """
        for latin, persian in _PUNCT_MAP.items():
            text = text.replace(latin, persian)
        return text
