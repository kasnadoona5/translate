"""Layer 3 of the four-layer memory system: Long-term Memory.

TF-IDF keyword retrieval of past translated segments. Matches the English source of
the current chunk against the English source of previously translated chunks to
ensure technical terminology and stylistic consistency.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any





def _tokenize_english(text: str) -> list[str]:
    """Tokenize and lowercase English text, removing punctuation."""
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


class LongTermMemory:
    """Long-term sentence translation memory.

    Indexes past translation pairs and retrieves the top-K matches using TF-IDF.
    """

    def __init__(self, retrieval_k: int = 5) -> None:
        self.retrieval_k = retrieval_k
        self._pairs: list[dict[str, Any]] = []

    def add(
        self,
        source: str,
        translation: str,
        *,
        reliable: bool = True,
        chapter_title: str = "",
    ) -> None:
        """Add a new translated pair to the memory database."""
        src = source.strip()
        trans = translation.strip()
        if src and trans:
            self._pairs.append({
                "entry_id": len(self._pairs),
                "source": src,
                "translation": trans,
                "reliable": bool(reliable),
                "chapter_title": chapter_title,
            })

    def get_relevant(self, query_text: str) -> list[dict[str, Any]]:
        """Retrieve relevant pairs while retaining advisory continuity.

        Reviewed-but-unresolved translations remain useful evidence about the
        argument and local references.  They are therefore retrievable, but a
        small ranking penalty keeps equally relevant, QA-reliable prose ahead
        of advisory wording.
        """
        if not self._pairs:
            return []

        eligible_pairs = list(self._pairs)
        if not eligible_pairs:
            return []

        query_tokens = _tokenize_english(query_text)
        if not query_tokens:
            return []

        query_counter = Counter(query_tokens)
        num_docs = len(eligible_pairs)

        doc_tokens_list = [_tokenize_english(p["source"]) for p in eligible_pairs]
        doc_counters = [Counter(tokens) for tokens in doc_tokens_list]

        # Gather all unique terms
        all_terms = set()
        for counters in doc_counters:
            all_terms.update(counters.keys())

        # Compute Document Frequency (DF)
        df: Counter[str] = Counter()
        for counters in doc_counters:
            for term in counters:
                df[term] += 1

        # Compute Inverse Document Frequency (IDF)
        idf: dict[str, float] = {}
        for term, freq in df.items():
            idf[term] = math.log(1.0 + num_docs / freq)

        # Calculate cosine similarity scores
        scores: list[tuple[float, dict[str, str]]] = []
        for idx, doc_counter in enumerate(doc_counters):
            dot_product = 0.0
            query_norm = 0.0
            doc_norm = 0.0

            for term in query_counter:
                q_val = query_counter[term] * idf.get(term, 0.0)
                d_val = doc_counter.get(term, 0) * idf.get(term, 0.0)
                dot_product += q_val * d_val
                query_norm += q_val ** 2

            for term in doc_counter:
                d_val = doc_counter[term] * idf.get(term, 0.0)
                doc_norm += d_val ** 2

            if query_norm > 0.0 and doc_norm > 0.0:
                sim = dot_product / (math.sqrt(query_norm) * math.sqrt(doc_norm))
            else:
                sim = 0.0

            trust_weight = 1.0 if eligible_pairs[idx].get("reliable", True) else 0.85
            scores.append((sim * trust_weight, eligible_pairs[idx]))

        # Sort by similarity score descending
        scores.sort(key=lambda x: x[0], reverse=True)

        # Filter only positive matches
        relevant = [pair for sim, pair in scores if sim > 0.0]
        return relevant[:self.retrieval_k]

    def get_context(self, query_text: str) -> str:
        """Return a formatted string representing the retrieved pairs.

        Returns
        -------
        str
            A prompt-ready block listing retrieved source-target pairs.
        """
        relevant = self.get_relevant(query_text)
        if not relevant:
            return ""
        lines = []
        for p in relevant:
            guidance = (
                "[retrieval: reliable prose]"
                if p.get("reliable", True)
                else (
                    "[retrieval: advisory continuity; preserve the argument, "
                    "but do not treat wording as terminology or style authority]"
                )
            )
            lines.append(
                f"{guidance}\nEN: {p['source']}\nFA: {p['translation']}"
            )
        return "\n\n".join(lines)

    def serialize(self) -> list[dict[str, Any]]:
        """Serialize the layer state for database checkpointing."""
        return list(self._pairs)

    def deserialize(self, data: list[dict[str, Any]]) -> None:
        """Restore the layer state from serialized data."""
        self._pairs = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            restored = dict(item)
            restored.setdefault("entry_id", index)
            restored.setdefault("reliable", True)
            self._pairs.append(restored)

    def clear(self) -> None:
        """Clear the memory state."""
        self._pairs.clear()

    def __len__(self) -> int:
        return len(self._pairs)

    def __repr__(self) -> str:
        return f"LongTermMemory(pairs_count={len(self._pairs)}, retrieval_k={self.retrieval_k})"
