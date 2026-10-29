"""Source-bound selection of an explicitly requested finalized prefix."""
from __future__ import annotations

from typing import Any


def saved_prefix_count(chunks: list[Any], records: list[dict[str, Any]]) -> int:
    """Reject gaps, duplicate indices and source drift rather than skip them."""
    if not chunks or len(records) != len(chunks):
        raise ValueError('Saved-prefix export requires the original complete chunk index')
    indices = [r.get('chunk_index') for r in records]
    if (any(type(i) is not int for i in indices)
            or sorted(indices) != list(range(len(chunks)))):
        raise ValueError('Saved-prefix export has missing or duplicate chunk indices')
    records = sorted(records, key=lambda r: r['chunk_index'])
    count = 0
    gap = False
    for i, (chunk, record) in enumerate(zip(chunks, records)):
        if type(chunk.index) is not int or chunk.index != i or chunk.text != record.get('text'):
            raise ValueError('Saved-prefix export source no longer matches saved chunks')
        if record.get('status') in {'completed', 'needs_review'} and not str(record.get('translation') or '').strip():
            raise ValueError('Saved-prefix export found an empty finalized translation')
        finalized = (record.get('status') in {'completed', 'needs_review'}
                     and bool(str(record.get('translation') or '').strip()))
        if finalized and gap:
            raise ValueError('Saved-prefix export cannot skip a gap before a finalized chunk')
        if finalized:
            count += 1
        else:
            gap = True
    if not count:
        raise ValueError('Saved-prefix export has no finalized prefix')
    return count
