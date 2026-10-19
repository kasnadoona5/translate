"""Shared lexical spans for original insertion and its independent proof."""

from __future__ import annotations

import re
from dataclasses import dataclass

_MARKS = frozenset("\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0654\u0670")
_LETTERS = r"\u0621-\u063a\u0641-\u064a\u066e-\u06d3\u06fa-\u06ff"
ANCHOR_SUFFIX_RE = re.compile(
    r"(?:\u200c?(?:\u0647\u0627(?:\u06cc(?:\u06cc|\u0645|\u062a|\u0634|\u0645\u0627\u0646|\u062a\u0627\u0646|\u0634\u0627\u0646)?)?"
    r"|\u06cc|\u0627\u0646|\u0627\u062a|\u062a\u0631(?:\u06cc\u0646)?|\u0627\u0645|\u0627\u062a|\u0627\u0634|\u0645\u0627\u0646|\u062a\u0627\u0646|\u0634\u0627\u0646))"
)


@dataclass(frozen=True)
class AnchorSpan:
    start: int
    end: int
    rendered: str
    kind: str


def _normalized(text: str) -> tuple[str, list[int]]:
    indices = [i for i, char in enumerate(text) if char not in _MARKS and char != "\u200c"]
    return "".join(text[i] for i in indices), indices


def extend_anchor_end(text: str, end: int) -> int:
    match = ANCHOR_SUFFIX_RE.match(text, end)
    end = match.end() if match else end
    while end < len(text) and text[end] in _MARKS:
        end += 1
    return end


def anchor_spans(text: str, target: str) -> list[AnchorSpan]:
    """Match contiguous words; only internal indefinite yeh may vary.

    Normalization is used for evidence offsets only, never to rewrite text.
    Exact matches retain precedence over the conservative inflection fallback.
    """
    normalized, offsets = _normalized(text or "")
    canonical, _ = _normalized((target or "").strip())
    tokens = canonical.split()
    if not tokens:
        return []
    suffix = _normalized(ANCHOR_SUFFIX_RE.pattern)[0].replace(r"\u200c?", "")
    boundary = rf"(?=$|[^\w]|(?:{suffix})(?!\w))"
    patterns = [("exact", r"[ \t]+".join(re.escape(token) for token in tokens))]
    if len(tokens) >= 2 and all(re.fullmatch(rf"[{_LETTERS}]+", token) for token in tokens):
        patterns.append(
            (
                "internal_indefinite_yeh",
                r"[ \t]+".join(
                    re.escape(token)
                    + (
                        r"\u06cc?"
                        if i < len(tokens) - 1 and not token.endswith("\u06cc") and len(token) >= 3
                        else ""
                    )
                    for i, token in enumerate(tokens)
                ),
            )
        )
    for kind, body in patterns:
        result = []
        for match in re.finditer(rf"(?<!\w){body}{boundary}", normalized):
            start, end = offsets[match.start()], offsets[match.end() - 1] + 1
            while end < len(text) and text[end] in _MARKS:
                end += 1
            result.append(AnchorSpan(start, end, text[start:end], kind))
        if result:
            return result
    return []


def anchor_before(text: str, position: int, targets: list[str]) -> tuple[AnchorSpan, str] | None:
    boundary = len(text[:position].rstrip(" \t\u00bb\u201d"))
    for target in targets:
        spans = anchor_spans(text[:boundary], target)
        for span in spans:
            if extend_anchor_end(text, span.end) == boundary:
                if span.kind != "exact" and len(spans) != 1:
                    continue
                return span, target
    return None
