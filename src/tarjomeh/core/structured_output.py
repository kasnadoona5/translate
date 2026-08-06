"""Provider-neutral normalization for model text and structured responses."""

from __future__ import annotations

import json
import re
from typing import Any


_LEADING_REASONING_BLOCK = re.compile(
    r"\A\s*<\s*(think|analysis|reasoning)\s*>.*?"
    r"<\s*/\s*\1\s*>\s*",
    re.IGNORECASE | re.DOTALL,
)
_PROTOCOL_TAG = re.compile(
    r"<\s*/?\s*(think|analysis|reasoning)\s*>",
    re.IGNORECASE,
)
_MARKDOWN_FENCE = re.compile(
    r"\A\s*```(?:json|javascript|js)?\s*(.*?)\s*```\s*\Z",
    re.IGNORECASE | re.DOTALL,
)


def normalize_model_text(text: str | None) -> tuple[str, dict[str, Any]]:
    """Remove only recognized leading reasoning wrappers from visible output."""
    cleaned = str(text or "")
    removed = 0
    while True:
        match = _LEADING_REASONING_BLOCK.match(cleaned)
        if match is None:
            break
        cleaned = cleaned[match.end():]
        removed += 1
    cleaned = cleaned.strip()
    return cleaned, {
        "removed_reasoning_wrappers": removed,
        "remaining_protocol_artifacts": protocol_artifacts(cleaned),
    }


def protocol_artifacts(text: str | None) -> list[str]:
    """Return known model-protocol markers that remain in visible text."""
    return list(dict.fromkeys(
        match.group(0) for match in _PROTOCOL_TAG.finditer(str(text or ""))
    ))


def parse_structured_output(
    text: str | None,
    *,
    expected: type | tuple[type, ...] = (dict, list),
) -> Any:
    """Extract the first complete JSON value of the expected top-level type."""
    cleaned, _ = normalize_model_text(text)
    fence = _MARKDOWN_FENCE.match(cleaned)
    if fence:
        cleaned = fence.group(1).strip()

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as direct_error:
        decoder = json.JSONDecoder()
        value = None
        for index, char in enumerate(cleaned):
            if char not in "[{":
                continue
            try:
                candidate, _ = decoder.raw_decode(cleaned[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, expected):
                value = candidate
                break
        if value is None:
            raise direct_error

    if not isinstance(value, expected):
        names = ", ".join(
            item.__name__ for item in (
                expected if isinstance(expected, tuple) else (expected,)
            )
        )
        raise ValueError(f"Structured response must be one of: {names}")
    return value
