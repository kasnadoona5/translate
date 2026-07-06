"""Layer 3 of the four-layer memory system: Long-term Memory.

TF-IDF keyword retrieval of past translated segments. Matches the English source of
the current chunk against the English source of previously translated chunks to
ensure technical terminology and stylistic consistency.
"""

from __future__ import annotations

import math
import re
from collections import Counter





def _tokenize_english(text: str) -> list[str]:
    """Tokenize and lowercase English text, removing punctuation."""
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


class LongTermMemory:
    """Long-term sentence translation memory.

    Indexes past translation pairs and retrieves the top-K matches using TF-IDF.
    """

    def __init__(self, retrieval_k: int = 5) -> None:
        self.retrieval_k = retrieval_k
        self._pairs: list[dict[str, str]] = []

    def add(self, source: str, translation: str) -> None:
        """Add a new translated pair to the memory database."""
        src = source.strip()
        trans = translation.strip()
        if src and trans:
            self._pairs.append({"source": src, "translation": trans})

    def get_relevant(self, query_text: str) -> list[dict[str, str]]:
        """Retrieve top-K translation pairs that are relevant to query_text."""
        if not self._pairs:
            return []

        query_tokens = _tokenize_english(query_text)
        if not query_tokens:
            return []

        query_counter = Counter(query_tokens)
        num_docs = len(self._pairs)

        doc_tokens_list = [_tokenize_english(p["source"]) for p in self._pairs]
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

            scores.append((sim, self._pairs[idx]))

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
            lines.append(f"EN: {p['source']}\nFA: {p['translation']}")
        return "\n\n".join(lines)

    def serialize(self) -> list[dict[str, str]]:
        """Serialize the layer state for database checkpointing."""
        return list(self._pairs)

    def deserialize(self, data: list[dict[str, str]]) -> None:
        """Restore the layer state from serialized data."""
        self._pairs = list(data)

    def clear(self) -> None:
        """Clear the memory state."""
        self._pairs.clear()

    def __len__(self) -> int:
        return len(self._pairs)

    def __repr__(self) -> str:
        return f"LongTermMemory(pairs_count={len(self._pairs)}, retrieval_k={self.retrieval_k})"
