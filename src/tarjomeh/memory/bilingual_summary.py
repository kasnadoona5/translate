"""Layer 2 of the four-layer memory system: Running Bilingual Summary.

Maintains a running summary of the book content in both English and Persian
to preserve thematic and argumentative progression.
"""

from __future__ import annotations

import html
import re


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_DIRECTIONAL_CONTROL_RE = re.compile(
    r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]"
)
_PROVENANCE_LINE_RE = re.compile(
    r"^(?:provenance\s*:\s*translated content only\.?|"
    r"\u0645\u0646\u0634[\u0623\u0627]\s*:\s*\u0641\u0642\u0637\s+"
    r"\u0645\u062d\u062a\u0648\u0627\u06cc\s+\u062a\u0631\u062c\u0645\u0647[\u200c\s-]*"
    r"\u0634\u062f\u0647\.?)$",
    re.IGNORECASE,
)


def _clean_summary_line(value: str) -> str:
    clean = html.unescape(_HTML_TAG_RE.sub("", value or ""))
    clean = _DIRECTIONAL_CONTROL_RE.sub("", clean)
    return " ".join(clean.split())


def _deduplicate_summary_lines(lines: list[str]) -> list[str]:
    """Remove protocol provenance and exact repeats without rewriting prose."""
    result: list[str] = []
    seen: set[str] = set()
    for value in lines:
        clean = _clean_summary_line(value)
        identity = re.sub(r"[\s\u200c]+", " ", clean).casefold()
        if not clean or _PROVENANCE_LINE_RE.fullmatch(clean) or identity in seen:
            continue
        seen.add(identity)
        result.append(clean)
    return result


class BilingualSummary:
    """Running bilingual chapter summary memory layer.

    Maintains English and Persian summaries of the book.
    """

    def __init__(self) -> None:
        self.english_summary: str = ""
        self.persian_summary: str = ""
        self.input_trust: str = "context_only"
        self.trust_reasons: list[str] = []

    def set_input_trust(
        self,
        trust: str,
        reasons: list[str] | None = None,
    ) -> None:
        """Record how strongly the translated inputs support this summary."""
        self.input_trust = (
            "reviewed_inputs" if trust == "reviewed_inputs" else "advisory_inputs"
        )
        self.trust_reasons = list(dict.fromkeys(
            str(reason).strip() for reason in (reasons or []) if str(reason).strip()
        ))[:12]

    def update(self, raw_llm_output: str) -> None:
        """Parse and update the summaries from the LLM response text.

        Expected LLM output format:
        ## English Summary
        <text>
        ## خلاصه فارسی
        <text>
        """
        eng_parts: list[str] = []
        fa_parts: list[str] = []
        current_section: str | None = None

        for line in raw_llm_output.splitlines():
            line_str = _clean_summary_line(line)
            if not line_str:
                continue
            if line_str == "## English Summary":
                current_section = "english"
                continue
            if line_str == "## خلاصه فارسی" or "خلاصه فارسی" in line_str:
                current_section = "persian"
                continue
            if line_str.startswith("##"):
                current_section = None
                continue

            if current_section == "english":
                eng_parts.append(line_str)
            elif current_section == "persian":
                fa_parts.append(line_str)

        eng_parts = _deduplicate_summary_lines(eng_parts)
        fa_parts = _deduplicate_summary_lines(fa_parts)
        if eng_parts:
            self.english_summary = "\n".join(eng_parts)
        if fa_parts:
            self.persian_summary = "\n".join(fa_parts)

    def get_context(self) -> str:
        """Return prompt-ready bilingual summary string.

        Returns
        -------
        str
            Formatted summary sections or empty string.
        """
        if not self.english_summary and not self.persian_summary:
            return ""

        trust_note = (
            "Input trust: reviewed translated prose."
            if self.input_trust == "reviewed_inputs"
            else "Input trust: advisory; one or more contributing passages need review."
        )
        authority_note = (
            "Memory role: argument orientation only. Never copy its wording as "
            "terminology or style authority; the source, curated glossary, accepted "
            "terminology, and current passage always override it."
        )
        parts = [f"{trust_note}\n{authority_note}"]
        if self.english_summary:
            parts.append(
                "## English Summary\n"
                "Provenance: translated content only.\n"
                f"{self.english_summary}"
            )
        if self.persian_summary:
            parts.append(
                f"## خلاصه فارسی (RTL)\n"
                f"منشأ: فقط محتوای ترجمه‌شده.\n"
                f"{self.persian_summary}"
            )
        return "\n\n".join(parts)

    def reconcile_persian_terms(
        self,
        replacements: list[tuple[str, str]],
    ) -> dict[str, object]:
        """Replace only exact superseded Persian terms in the stored summary.

        This repairs stale memory after a committed terminology correction. It
        neither changes the English summary nor creates terminology authority.
        """
        text = self.persian_summary
        changes: list[dict[str, object]] = []
        persian_letter = r"\u0621-\u063a\u0641-\u064a\u066e-\u06d3\u06fa-\u06ff"
        for previous, current in replacements:
            old = " ".join((previous or "").split()).strip()
            new = " ".join((current or "").split()).strip()
            if (
                not old
                or not new
                or old == new
                or len(old) > 120
                or len(new) > 120
                or not re.search(r"[\u0600-\u06ff]", old + new)
            ):
                continue
            words = [re.escape(word) for word in re.split(r"[\s\u200c]+", old) if word]
            if not words:
                continue
            pattern = re.compile(
                rf"(?<![{persian_letter}])"
                + r"[\s\u200c]+".join(words)
                + rf"(?![{persian_letter}])"
            )
            text, count = pattern.subn(lambda _match: new, text)
            if count:
                changes.append({
                    "previous_target": old,
                    "target": new,
                    "count": count,
                })
        self.persian_summary = text
        return {
            "replacement_count": sum(int(item["count"]) for item in changes),
            "changes": changes,
            "policy": (
                "Only exact superseded Persian renderings from committed, "
                "non-curated terminology corrections are reconciled."
            ),
        }

    def serialize(self) -> dict[str, object]:
        """Serialize the layer state for database checkpointing."""
        return {
            "english_summary": self.english_summary,
            "persian_summary": self.persian_summary,
            "input_trust": self.input_trust,
            "trust_reasons": self.trust_reasons,
        }

    def deserialize(self, data: dict[str, object]) -> None:
        """Restore the layer state from serialized data."""
        self.english_summary = "\n".join(_deduplicate_summary_lines(
            str(data.get("english_summary", "")).splitlines()
        ))
        self.persian_summary = "\n".join(_deduplicate_summary_lines(
            str(data.get("persian_summary", "")).splitlines()
        ))
        stored_trust = str(data.get("input_trust", "context_only"))
        self.input_trust = (
            stored_trust
            if stored_trust in {"reviewed_inputs", "advisory_inputs"}
            else "advisory_inputs"
        )
        raw_reasons = data.get("trust_reasons", [])
        self.trust_reasons = [
            str(reason) for reason in raw_reasons
        ] if isinstance(raw_reasons, list) else []

    def __repr__(self) -> str:
        en_len = len(self.english_summary)
        fa_len = len(self.persian_summary)
        return f"BilingualSummary(en_chars={en_len}, fa_chars={fa_len})"
