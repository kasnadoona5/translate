"""Layer 2 of the four-layer memory system: Running Bilingual Summary.

Maintains a running summary of the book content in both English and Persian
to preserve thematic and argumentative progression.
"""

from __future__ import annotations

import html
import re


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clean_summary_line(value: str) -> str:
    clean = html.unescape(_HTML_TAG_RE.sub("", value or ""))
    return " ".join(clean.replace("\u200e", "").replace("\u200f", "").split())


class BilingualSummary:
    """Running bilingual chapter summary memory layer.

    Maintains English and Persian summaries of the book.
    """

    def __init__(self) -> None:
        self.english_summary: str = ""
        self.persian_summary: str = ""

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

        parts = []
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

    def serialize(self) -> dict[str, str]:
        """Serialize the layer state for database checkpointing."""
        return {
            "english_summary": self.english_summary,
            "persian_summary": self.persian_summary,
        }

    def deserialize(self, data: dict[str, str]) -> None:
        """Restore the layer state from serialized data."""
        self.english_summary = _clean_summary_line(
            data.get("english_summary", "")
        )
        self.persian_summary = _clean_summary_line(
            data.get("persian_summary", "")
        )

    def __repr__(self) -> str:
        en_len = len(self.english_summary)
        fa_len = len(self.persian_summary)
        return f"BilingualSummary(en_chars={en_len}, fa_chars={fa_len})"
