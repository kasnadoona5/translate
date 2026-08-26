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
_PERSIAN_TO_WESTERN = str.maketrans(
    "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9"
    "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669",
    "01234567890123456789",
)
_PERSIAN_LETTER_RE = re.compile(
    r"[\u0621-\u063a\u0641-\u064a\u066e-\u06d3\u06fa-\u06ff]"
)
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
_LATIN_CATALOG_LINE_RE = re.compile(
    r"^\s*(?:[0-9\u06f0-\u06f9]+\.\s+)?[A-Z][^\n]*"
    r"(?:\bI\.\s+Title\.?|/\s*[A-Z][^\n]*)\s*$"
)
_PERSIAN_PROSE_REFERENCE_RE = re.compile(
    r"(?P<label>\u0641\u0635\u0644|\u062c\u062f\u0648\u0644|\u0634\u06a9\u0644|"
    r"\u0628\u062e\u0634|\u062c\u0644\u062f|\u067e\u06cc\u0648\u0633\u062a)"
    r"(?P<space>[ \t]+)(?P<number>[0-9]{1,3})(?![A-Za-z0-9])"
)

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
_AUTHOR_YEAR_CITATION_RE = re.compile(
    r"(?<![A-Za-z])"
    r"(?:[A-Z]\.\s*){0,5}[A-Z][A-Za-z'\u2019-]+"
    r"(?:\s+(?:(?:and|de|del|der|di|du|la|le|van|von|&)\s+)?"
    r"(?:[A-Z]\.\s*){0,4}[A-Z][A-Za-z'\u2019-]+){0,4}"
    r"\s+(?:1[5-9][0-9]{2}|20[0-9]{2})[a-z]?"
    r"(?:\s*(?:"
    r",\s*(?:(?:1[5-9][0-9]{2}|20[0-9]{2})[a-z]?|[0-9]{1,4})"
    r"(?:[-\u2010-\u2015][0-9]{1,4})?"
    r"|:\s*[0-9]{1,4}(?:[-\u2010-\u2015][0-9]{1,4})?"
    r"))*"
)
_SCHOLARLY_PROTECTED_RE = re.compile(
    r"\b(?:ISBN(?:-1[03])?|ISSN)\s*:?\s*"
    r"[0-9Xx](?:[0-9Xx \t\-‐-―]{6,30})[0-9Xx]\b"
    r"|\([^)؀-ۿ]*[0-9][^)؀-ۿ]*\)"
    r"|\[[^\]]*?[0-9][^\]]*?\]"
    r"|\b(?:pp?|vols?|nos?|chs?|fols?)\.\s*[0-9]+(?:\s*[-–—,]\s*[0-9]+)*"
    r"|\b(?:[A-Z]\.\s*){1,5}[A-Z][A-Za-z'\u2019-]+"
    r"(?:\s+(?:1[5-9][0-9]{2}|20[0-9]{2})[a-z]?)?"
    r"|(?<![\u0600-\u06ff])(?:\u0631\.\s*\u06a9\.|\u0646\u06a9\.)(?![\u0600-\u06ff])"
    r"|\b(?:1[5-9][0-9]{2}|20[0-9]{2})[a-z]?\b",
    re.IGNORECASE,
)

# Preserve only marks already authored by the model. Hazm's default normalizer
# removes them, which otherwise creates silent drift between reviewed memory and
# the exported document. This protection never adds a mark.
_PERSIAN_COMBINING_MARK_RE = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed]+")
_HAZM_FALSE_MI_RE = re.compile(
    r"(?<![\u0600-\u06ff])(?P<prefix>\u0646?\u0645\u06cc)\u200c"
    r"(?P<stem>[\u0621-\u063a\u0641-\u064a\u066e-\u06d3\u06fa-\u06ff]"
    r"(?:[\u0621-\u063a\u0641-\u064a\u066e-\u06d3\u06fa-\u06ff\u200c-])*)"
)
_NON_VERB_HAZM_TAGS = frozenset({"N", "AJ", "NUM", "ADV"})
_SEMANTIC_TATWEEL_SEPARATOR_RE = re.compile(
    r"(?<=[\u0600-\u06ff\u00bb\u201d])\s+\u0640+\s+"
    r"(?=[\u0600-\u06ff\u00ab\u201c])"
)

# Digit-free Private-Use-Area sentinels: survive hazm normalisation and are
# untouched by numeral/punctuation conversion.
_SENTINEL_OPEN = ""
_SENTINEL_CLOSE = ""
_SENTINEL_BASE = 0xE100
_SENTINEL_RE = re.compile(f"{_SENTINEL_OPEN}(.){_SENTINEL_CLOSE}")

# hazm pads U+066B as if it were sentence punctuation, turning a decimal
# like "10.5" or a table number like "1.1" into "N <sep> N". Rejoin after
# normalisation. Restricted to the Persian decimal separator and to
# spaces/tabs (never newlines) so a paragraph boundary is never merged.
_SPACED_DECIMAL_RE = re.compile(
    "([0-9\u06f0-\u06f9])[ \t]*\u066b[ \t]*([0-9\u06f0-\u06f9])"
)


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

        # A spaced tatweel is sometimes model-authored as a semantic dash.
        # Hazm removes it, silently joining the two concepts. Canonicalize only
        # the separator form; decorative kashida inside a word is untouched.
        text = _SEMANTIC_TATWEEL_SEPARATOR_RE.sub(" – ", text)

        if (
            _LATIN_LETTER_RE.search(text)
            and not _PERSIAN_LETTER_RE.search(text)
            and _LATIN_CATALOG_LINE_RE.fullmatch(text)
        ):
            # A Latin-only catalogue or metadata line must not receive Persian
            # punctuation. This also repairs already-normalized resumed text.
            return (
                text.translate(_PERSIAN_TO_WESTERN)
                .replace("\u060c", ",")
                .replace("\u061b", ";")
                .replace("\u061f", "?")
            )

        # Shield scholarly apparatus BEFORE any transformation (hazm itself
        # may also touch digits), restore afterwards.
        protected_spans: list[str] = []
        if self._scholarly_mode:
            text = self._protect_scholarly(text, protected_spans)
        text = self._protect_pattern(
            text, protected_spans, _PERSIAN_COMBINING_MARK_RE
        )

        if self._normalize_zwnj:
            text = self.normalize_zwnj(text)
            text = self._repair_false_mi_splits(text)
            text = _SPACED_DECIMAL_RE.sub("\\1\u066b\\2", text)
        if self._convert_numerals:
            text = self.convert_numerals(text)
        if self._fix_punctuation:
            text = self.fix_punctuation(text)

        if protected_spans:
            text = self._restore_scholarly(text, protected_spans)

        # Hazm deliberately avoids some lexical compounds. Finish with a very
        # small boundary-aware rule set whose edits are always orthographic.
        text, _ = apply_safe_persian_orthography(text)

        if self._convert_numerals:
            text = _PERSIAN_PROSE_REFERENCE_RE.sub(
                lambda match: (
                    match.group("label")
                    + match.group("space")
                    + "".join(
                        _WESTERN_TO_PERSIAN[character]
                        for character in match.group("number")
                    )
                ),
                text,
            )

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
        text = PersianTypographer._protect_pattern(
            text, spans, _AUTHOR_YEAR_CITATION_RE
        )
        return PersianTypographer._protect_pattern(
            text, spans, _SCHOLARLY_PROTECTED_RE
        )

    @staticmethod
    def _protect_pattern(
        text: str,
        spans: list[str],
        pattern: re.Pattern[str],
    ) -> str:
        """Stash regex matches in one shared, collision-free sentinel table."""

        def _stash(match: re.Match[str]) -> str:
            spans.append(match.group(0))
            return f"{_SENTINEL_OPEN}{chr(_SENTINEL_BASE + len(spans) - 1)}{_SENTINEL_CLOSE}"

        return pattern.sub(_stash, text)

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

    def _repair_false_mi_splits(self, text: str) -> str:
        """Undo Hazm ``mi`` splits only for lexicon-backed non-verbs."""
        normalizer = self._get_hazm_normalizer()
        lexicon = getattr(normalizer, "words", {})
        if not isinstance(lexicon, dict):
            return text

        def _join(match: re.Match[str]) -> str:
            joined = match.group("prefix") + match.group("stem")
            record = lexicon.get(joined)
            if not isinstance(record, (tuple, list)) or len(record) < 2:
                return match.group(0)
            try:
                frequency = int(record[0])
            except (TypeError, ValueError):
                return match.group(0)
            raw_tags = record[1]
            tags = (
                {str(tag) for tag in raw_tags}
                if isinstance(raw_tags, (tuple, list, set, frozenset))
                else {str(raw_tags)}
            )
            if frequency > 0 and tags.intersection(_NON_VERB_HAZM_TAGS):
                return joined
            return match.group(0)

        return _HAZM_FALSE_MI_RE.sub(_join, text)

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
