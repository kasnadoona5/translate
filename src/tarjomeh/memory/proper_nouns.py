"""Layer 1 of the four-layer memory system: Proper Nouns & Transliterations.

Stores established transliterations and translations of person names, places,
institutions, and publications to ensure consistency across the book — and
tracks which names have already been INTRODUCED (appeared in translated text),
so the "English form in parentheses on first occurrence" convention is applied
exactly once per book instead of once per chunk.
"""

from __future__ import annotations

import re
from typing import Any


INLINE_ORIGINAL_CATEGORIES = frozenset({
    "proper_noun", "person", "place", "institution", "organization",
    "publication", "product", "theory", "approved_term",
})
_CATEGORY_ALIASES = {
    "organisation": "organization",
    "book": "publication",
    "article": "publication",
    "journal": "publication",
    "work": "publication",
    "named_theory": "theory",
}
_PERSIAN_LETTER_RE = re.compile(r"[\u0621-\u063a\u0641-\u064a\u066e-\u06d3\u06fa-\u06ff]")
_UNUSABLE_TARGET_RE = re.compile(
    r"\b(?:n/?a|none|unknown|not available|no persian|no established|"
    r"not supported|insufficient evidence|untranslated)\b",
    re.IGNORECASE,
)
_CONTEXTUAL_TARGET_TOKEN_RE = re.compile(
    r"(?:^|\s)(?:من|ما|تو|شما|او|ایشان|آنها|آن‌ها|این|آن|همین|همان|"
    r"سایر|دیگر|خود|هستم|هستی|است|هست|هستیم|هستید|هستند|بود|بودند|"
    r"شد|شدند|می‌شود|می‌شوند)(?:\s|$)"
)
_TRAILING_CONNECTIVE_RE = re.compile(
    r"(?:^|\s)(?:از|به|با|در|برای|که|و|یا|اما|تا|را)\s*$"
)
_PROVENANCE_AUTHORITY = {
    "auto_extraction": 10,
    "incremental_extraction": 10,
    "research_suggestion": 10,
    "observed_translation": 30,
    "legacy": 50,
    "accepted_correction": 80,
    "approved_research": 100,
    "curated_glossary": 100,
}


def is_usable_memory_mapping(english: str, persian: str) -> bool:
    """Reject malformed/placeholder auto mappings from prompt-time memory."""
    source = " ".join((english or "").split()).strip()
    target = " ".join((persian or "").split()).strip()
    if not source or not target or "_" in target:
        return False
    if source.casefold() == target.casefold():
        return False
    if _UNUSABLE_TARGET_RE.search(target):
        return False
    if not _PERSIAN_LETTER_RE.search(target):
        return False
    return True


def is_reusable_terminology_mapping(english: str, persian: str) -> bool:
    """Return whether a reviewed rendering is safe as book-wide terminology.

    A sentence-level correction may be perfectly right in its original passage
    while being unsafe as a global ``English -> Persian`` replacement.  This
    conservative test admits compact nominal renderings and defers contextual
    clauses, pronoun-bound phrases, and incomplete connective fragments.
    """
    if not is_usable_memory_mapping(english, persian):
        return False
    source = " ".join((english or "").split()).strip()
    target = " ".join((persian or "").split()).strip()
    source_words = re.findall(r"[A-Za-z][A-Za-z'\-]*", source)
    target_words = re.findall(rf"[{_PERSIAN_LETTER_RE.pattern[1:-1]}]+", target)
    if not 1 <= len(source_words) <= 6 or not target_words:
        return False
    if len(source_words) == 1 and len(target_words) > 3:
        return False
    if len(target_words) > max(5, len(source_words) * 2 + 1):
        return False
    if _CONTEXTUAL_TARGET_TOKEN_RE.search(target):
        return False
    if _TRAILING_CONNECTIVE_RE.search(target):
        return False
    return True


def _source_term_present(source_text: str, term: str) -> bool:
    """Match a stored source term despite PDF whitespace/hyphen line breaks."""
    words = re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", term or "")
    if not words:
        return False
    separator = r"(?:\s+|\s*[-‐-―]\s*)"
    pattern = separator.join(re.escape(word) for word in words)
    return bool(re.search(rf"(?<!\w){pattern}(?!\w)", source_text or "", re.IGNORECASE))


def _normalise_category(category: str) -> str:
    value = (category or "proper_noun").strip().lower().replace("-", "_").replace(" ", "_")
    return _CATEGORY_ALIASES.get(value, value)


def _normalise_target(value: str) -> str:
    return re.sub(r"[\s\u200c]+", " ", (value or "").strip()).casefold()


class ProperNouns:
    """Proper noun translation memory layer.

    Stores mappings from English proper nouns to their Persian equivalents,
    plus an *introduced* flag per noun (has it already appeared in a
    translated chunk?).
    """

    def __init__(self) -> None:
        self._nouns: dict[str, str] = {}
        self._categories: dict[str, str] = {}
        self._provenance: dict[str, dict[str, Any]] = {}
        self._aliases: dict[str, list[str]] = {}
        self._introduced: set[str] = set()

    def _stored_key(self, english: str) -> str | None:
        requested = " ".join((english or "").split()).casefold()
        return next(
            (
                source for source in self._nouns
                if " ".join(source.split()).casefold() == requested
            ),
            None,
        )

    def add_noun(
        self,
        english: str,
        persian: str,
        category: str = "proper_noun",
        *,
        provenance: str = "legacy",
    ) -> dict[str, Any]:
        """Add or reconcile a proper noun mapping by source authority.

        Parameters
        ----------
        english:
            English proper noun (key).
        persian:
            Established Persian translation/transliteration.
        """
        en_key = " ".join(english.split()).strip()
        fa_val = persian.strip()
        if not en_key or not fa_val:
            return {"action": "ignored", "source": en_key}

        origin = provenance if provenance in _PROVENANCE_AUTHORITY else "legacy"
        authority = _PROVENANCE_AUTHORITY[origin]
        stored_key = self._stored_key(en_key)
        previous = self._nouns.get(stored_key or "", "")
        prior = dict(self._provenance.get(stored_key or "", {}))
        prior_authority = int(
            prior.get("authority", _PROVENANCE_AUTHORITY["legacy"])
        )

        if stored_key is None:
            stored_key = en_key
            self._nouns[stored_key] = fa_val
            action = "added"
        elif _normalise_target(previous) == _normalise_target(fa_val):
            action = "confirmed"
        elif authority > prior_authority:
            self._nouns[stored_key] = fa_val
            action = "replaced_lower_authority"
        else:
            action = "preserved_higher_authority"

        observations = int(prior.get("observations", 0)) + 1
        if action in {"added", "replaced_lower_authority"} or (
            action == "confirmed" and authority >= prior_authority
        ):
            history = list(prior.get("superseded", []))
            if action == "replaced_lower_authority" and previous:
                self.add_alias(stored_key, previous)
                history.append({
                    "target": previous,
                    "origin": str(prior.get("origin", "legacy")),
                })
            self._provenance[stored_key] = {
                "origin": origin,
                "authority": authority,
                "observations": observations,
                "superseded": history[-3:],
            }
        elif prior:
            prior["observations"] = observations
            self._provenance[stored_key] = prior

        new_category = _normalise_category(category)
        current_category = self._categories.get(stored_key)
        if (
            current_category is None
            or new_category == "approved_term"
            or action == "replaced_lower_authority"
            or (
                current_category not in INLINE_ORIGINAL_CATEGORIES
                and new_category in INLINE_ORIGINAL_CATEGORIES
            )
        ):
            self._categories[stored_key] = new_category
        return {
            "action": action,
            "source": stored_key,
            "previous_target": previous,
            "target": self._nouns[stored_key],
            "origin": self._provenance.get(stored_key, {}).get("origin", origin),
        }

    def add_alias(self, english: str, persian: str) -> bool:
        """Record a verified rendered variant without changing authority."""
        stored_key = self._stored_key(english)
        target = " ".join((persian or "").split()).strip()
        if not stored_key or not is_usable_memory_mapping(stored_key, target):
            return False
        canonical = self._nouns.get(stored_key, "")
        if _normalise_target(canonical) == _normalise_target(target):
            return False
        aliases = self._aliases.setdefault(stored_key, [])
        if any(_normalise_target(value) == _normalise_target(target) for value in aliases):
            return False
        aliases.append(target)
        self._aliases[stored_key] = aliases[-5:]
        return True

    def aliases_for(self, english: str) -> list[str]:
        """Return known Persian rendering aliases for one source expression."""
        key = self._stored_key(english) or english.strip()
        return list(self._aliases.get(key, []))

    def inline_eligible_aliases(self) -> dict[str, list[str]]:
        """Return aliases only for terms eligible for first-occurrence notes."""
        return {
            source: list(self._aliases.get(source, []))
            for source in self._nouns
            if self.is_inline_eligible(source)
            and not self.is_context_deferred(source)
            and self._aliases.get(source)
        }

    def mark_introduced(self, english: str) -> None:
        """Mark a noun as already introduced (parenthetical already shown)."""
        en_key = self._stored_key(english) or english.strip()
        if en_key in self._nouns:
            self._introduced.add(en_key)

    def mark_seen_in_text(self, source_text: str) -> None:
        """Mark every known noun that appears in *source_text* as introduced.

        Called after a chunk is translated: any known name occurring in that
        chunk's source has now had its first appearance, so later chunks must
        not repeat the English parenthetical.
        """
        for en in self._nouns:
            if en in self._introduced or not self.is_inline_eligible(en):
                continue
            if re.search(rf"\b{re.escape(en)}\b", source_text, re.IGNORECASE):
                self._introduced.add(en)

    def is_introduced(self, english: str) -> bool:
        """Return True if the noun's first occurrence has already happened."""
        return (self._stored_key(english) or english.strip()) in self._introduced

    def category_for(self, english: str) -> str:
        """Return the semantic category retained from extraction."""
        return self._categories.get(
            self._stored_key(english) or english.strip(), "proper_noun"
        )

    def provenance_for(self, english: str) -> dict[str, Any]:
        """Return a copy of the mapping's reconciliation provenance."""
        key = self._stored_key(english) or english.strip()
        return dict(self._provenance.get(key, {}))

    def all_nouns(self) -> dict[str, str]:
        """Return a copy of every proper-noun and terminology mapping."""
        return dict(self._nouns)

    def mark_context_conflict(
        self,
        english: str,
        *,
        corrected_source: str,
        corrected_target: str,
    ) -> bool:
        """Defer a lower-authority container mapping contradicted by review."""
        key = self._stored_key(english)
        if not key:
            return False
        provenance = dict(self._provenance.get(key, {}))
        authority = int(
            provenance.get("authority", _PROVENANCE_AUTHORITY["legacy"])
        )
        if authority >= _PROVENANCE_AUTHORITY["approved_research"]:
            return False
        provenance.update({
            "context_deferred": True,
            "conflict_source": corrected_source,
            "conflict_target": corrected_target,
        })
        self._provenance[key] = provenance
        return True

    def is_context_deferred(self, english: str) -> bool:
        """Return whether a contradicted automatic mapping is prompt-deferred."""
        key = self._stored_key(english) or english.strip()
        provenance = self._provenance.get(key, {})
        if provenance.get("context_deferred"):
            return True
        return bool(
            provenance.get("origin") == "accepted_correction"
            and not is_reusable_terminology_mapping(
                key, self._nouns.get(key, "")
            )
        )

    def is_inline_eligible(self, english: str) -> bool:
        """Return whether a noun may carry a first-occurrence English original."""
        return self.category_for(english) in INLINE_ORIGINAL_CATEGORIES

    def inline_eligible_nouns(self) -> dict[str, str]:
        """Return only deterministic inline-original candidates."""
        return {
            source: target for source, target in self._nouns.items()
            if self.is_inline_eligible(source)
            and not self.is_context_deferred(source)
        }

    def pending_inline_originals(self, source_text: str) -> dict[str, str]:
        """Return eligible, not-yet-introduced originals present in one chunk."""
        return {
            source: target for source, target in self._nouns.items()
            if self.is_inline_eligible(source)
            and not self.is_context_deferred(source)
            and source not in self._introduced
            and re.search(rf"\b{re.escape(source)}\b", source_text, re.IGNORECASE)
        }

    def get_context(
        self,
        include_inline_originals: bool = True,
        source_text: str = "",
    ) -> str:
        """Return a formatted string representing the proper nouns dictionary.

        Each entry carries an introduction marker the translation prompt is
        instructed to honour:
        - ``[introduced]`` — do NOT repeat the English parenthetical
        - ``[first occurrence pending]`` — add ``(English)`` after the Persian
          on first use.

        Returns
        -------
        str
            A hyphenated list of 'English -> Persian' proper nouns, or empty string.
        """
        if not self._nouns:
            return ""
        lines = []
        for en, fa in sorted(self._nouns.items()):
            if not is_usable_memory_mapping(en, fa):
                continue
            if self.is_context_deferred(en):
                continue
            if source_text and not _source_term_present(source_text, en):
                continue
            category = self.category_for(en)
            if not self.is_inline_eligible(en):
                marker = (
                    f"[advisory terminology only; category={category}; never add "
                    "an English parenthetical; glossary and source context override it]"
                )
            elif en in self._introduced:
                marker = "[introduced]"
            elif not include_inline_originals:
                marker = (
                    "[first occurrence pending — use the established Persian "
                    "rendering; the exporter adds the English-original note]"
                )
            else:
                marker = f"[first occurrence pending — add ({en}) after the Persian]"
            lines.append(f"- {en} -> {fa}  {marker}")
        return "\n".join(lines)

    def serialize(self) -> dict[str, Any]:
        """Serialize the layer state for database checkpointing."""
        return {
            "nouns": dict(self._nouns),
            "categories": dict(self._categories),
            "provenance": {
                source: dict(value) for source, value in self._provenance.items()
            },
            "aliases": {
                source: list(values) for source, values in self._aliases.items()
                if values
            },
            "introduced": sorted(self._introduced),
        }

    def deserialize(self, data: dict[str, Any]) -> None:
        """Restore the layer state from serialized data.

        Accepts both the current shape (``{"nouns": {...}, "introduced": [...]}``)
        and the legacy flat ``{english: persian}`` dict from older checkpoints.
        """
        if not data:
            self._nouns = {}
            self._categories = {}
            self._provenance = {}
            self._aliases = {}
            self._introduced = set()
            return

        if "nouns" in data and isinstance(data.get("nouns"), dict):
            self._nouns = dict(data["nouns"])
            stored_categories = data.get("categories", {})
            self._categories = {
                source: _normalise_category(str(stored_categories.get(source, "proper_noun")))
                for source in self._nouns
            }
            stored_provenance = data.get("provenance", {})
            self._provenance = {
                source: dict(stored_provenance.get(source, {}))
                for source in self._nouns
                if isinstance(stored_provenance.get(source), dict)
            }
            for source in self._nouns:
                self._provenance.setdefault(source, {
                    "origin": "legacy",
                    "authority": _PROVENANCE_AUTHORITY["legacy"],
                    "observations": 1,
                    "superseded": [],
                })
            stored_aliases = data.get("aliases", {})
            self._aliases = {
                source: [
                    str(value) for value in stored_aliases.get(source, [])
                    if isinstance(value, str) and value.strip()
                ][-5:]
                for source in self._nouns
                if isinstance(stored_aliases, dict)
                and isinstance(stored_aliases.get(source), list)
            }
            self._introduced = set(data.get("introduced", []))
        else:
            # Legacy checkpoint: flat mapping, no introduction tracking.
            self._nouns = {k: v for k, v in data.items() if isinstance(v, str)}
            self._categories = {source: "proper_noun" for source in self._nouns}
            self._provenance = {
                source: {
                    "origin": "legacy",
                    "authority": _PROVENANCE_AUTHORITY["legacy"],
                    "observations": 1,
                    "superseded": [],
                }
                for source in self._nouns
            }
            self._aliases = {}
            self._introduced = set()

    def __len__(self) -> int:
        return len(self._nouns)

    def __repr__(self) -> str:
        return (
            f"ProperNouns(count={len(self._nouns)}, "
            f"introduced={len(self._introduced)})"
        )
