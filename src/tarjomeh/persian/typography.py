"""Persian typography post-processor for Tarjomeh.

Handles ZWNJ normalization, numeral conversion, punctuation fixing,
and tokenization using the hazm NLP library for Persian text.
"""

from __future__ import annotations

import re
from typing import Any

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

        if self._normalize_zwnj:
            text = self.normalize_zwnj(text)
        if self._convert_numerals:
            text = self.convert_numerals(text)
        if self._fix_punctuation:
            text = self.fix_punctuation(text)

        return text

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
