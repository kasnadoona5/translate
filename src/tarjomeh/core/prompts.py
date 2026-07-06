"""Prompt templates for the Tarjomeh English→Persian translation system.

All prompts are purpose-built for academic English-to-Iranian-Persian translation.
Named-entity recognition uses **semantic role detection** (author names, place names,
institutions, book titles) — never capitalization, since Persian has no uppercase.

Templates use ``str.format()`` for variable injection.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. System-level translation prompt
# ---------------------------------------------------------------------------
TRANSLATE_SYSTEM_PROMPT: str = """\
You are a professional Persian translator specialising in academic texts.
Your task is to produce publication-quality Iranian Persian (فارسی ایران) prose
suitable for university-press publication (نثر آکادمیک).

Domain expertise: {domain}
Style register: {style_register}
Target dialect / country: {country}

Core directives:
• Maintain the precise meaning and nuance of the source English text.
• Use formal academic Persian appropriate for scholarly publication in {country}.
• Follow established Persian academic conventions for transliteration of proper nouns.
  - For transliteration, use the accepted Iranian academic standard.
  - On first occurrence of a transliterated proper noun, include the original English
    form in parentheses, e.g. هابرماس (Habermas).
• Preserve the author's argumentative structure and rhetorical style.
• Use the provided glossary terms consistently throughout.
• Use Iranian Persian vocabulary and conventions — not Dari or Afghan Persian.
• Output must be valid right-to-left (RTL) Persian text with correct ZWNJ placement.
• Convert Western numerals to Persian numerals (۰۱۲۳۴۵۶۷۸۹).
• Use standard Persian punctuation: «» for quotation marks, ؛ for semicolons, etc.
• Do NOT add personal commentary, footnotes, or translator's notes unless
  explicitly instructed.
"""

ACADEMIC_REGISTER_MODIFIER: str = """\
Use a highly formal, precise, and scholarly academic register. \
Employ complex sentence structures where appropriate, utilize established scholarly terminology, \
and avoid any colloquialisms, slang, or overly simplified vocabulary. The tone must reflect that of \
a publication by a reputable university press (نثر فاخر و دانشگاهی).\
"""


# ---------------------------------------------------------------------------
# 2. Per-chunk translation prompt
# ---------------------------------------------------------------------------
TRANSLATE_CHUNK_PROMPT: str = """\
Translate the following English academic text into Persian (فارسی).

### Glossary — mandatory terms (use exactly these translations)
{glossary_terms}

### Context from translation memory (previous chunks / running summary)
{memory_context}

### Web-sourced context for ambiguous terms
{web_context}

### Translation of the immediately preceding chunk (for continuity)
{previous_translation}

### Source text to translate
{source_text}

Instructions:
1. Translate the entire source text faithfully into academic Persian.
2. Apply every glossary term exactly as listed above.
3. Ensure stylistic and terminological continuity with the preceding translation.
4. Maintain paragraph structure; do not merge or split paragraphs.
5. Use ZWNJ (‌) correctly in compound verbs and affixed words (e.g. می‌خواهد).
6. For proper nouns appearing for the first time, include the original English form in
   parentheses after the Persian transliteration.
7. Output ONLY the Persian translation — no commentary, preamble, or labels.
"""

# ---------------------------------------------------------------------------
# 3. Critique / QA prompt
# ---------------------------------------------------------------------------
CRITIQUE_PROMPT: str = """\
You are a senior academic Persian editor reviewing a translation from English to \
Iranian Persian. Evaluate the translation rigorously and return a structured JSON report.

### Source (English)
{source_text}

### Translation (Persian)
{translation}

Return a JSON object with exactly this schema:
{{
  "scores": {{
    "accuracy": <1-10>,
    "fluency": <1-10>,
    "terminology": <1-10>,
    "register": <1-10>
  }},
  "overall": <1-10>,
  "issues": [
    {{
      "category": "accuracy" | "fluency" | "terminology" | "register" | "typography",
      "severity": "critical" | "major" | "minor",
      "source_segment": "<relevant English segment>",
      "current_translation": "<current Persian rendering>",
      "suggested_fix": "<improved Persian rendering>",
      "explanation": "<brief explanation in English>"
    }}
  ],
  "praise": "<brief note on what the translation does well>"
}}

Scoring guide:
- accuracy  : semantic equivalence, completeness, no omissions or additions
- fluency   : natural Persian prose flow, correct grammar, proper ZWNJ usage
- terminology: adherence to glossary, consistency of technical terms
- register  : appropriateness of academic tone, avoidance of colloquialisms
- overall   : holistic quality; a score below 7 means the chunk should be revised

Return ONLY valid JSON — no markdown fences, no commentary outside the JSON.
"""

# ---------------------------------------------------------------------------
# 4. Refinement prompt (applies critique feedback)
# ---------------------------------------------------------------------------
REFINE_PROMPT: str = """\
You are refining an academic English-to-Persian translation based on editorial feedback.

### Source (English)
{source_text}

### Current translation (Persian)
{translation}

### Editorial critique (JSON)
{critique}

Instructions:
1. Address every issue listed in the critique, especially those marked "critical" or "major".
2. Preserve parts of the translation that were praised or have no issues.
3. Maintain consistent terminology with the rest of the document.
4. Ensure correct ZWNJ placement, Persian numerals, and RTL punctuation.
5. Output ONLY the refined Persian translation — no commentary or JSON.
"""

# ---------------------------------------------------------------------------
# 5. Back-translation prompt (Persian → English, for quality verification)
# ---------------------------------------------------------------------------
BACK_TRANSLATE_PROMPT: str = """\
Translate the following Persian (فارسی) academic text back into English.
This is for quality-assurance purposes — produce a faithful, literal English rendering
that reflects the Persian text as closely as possible.

Do NOT try to reconstruct the original English source; translate what the Persian
actually says, including any errors or awkward phrasings.

### Persian text
{persian_text}

Output ONLY the English back-translation — no commentary or labels.
"""

# ---------------------------------------------------------------------------
# 6. Glossary / NER extraction prompt — semantic-role based
# ---------------------------------------------------------------------------
GLOSSARY_EXTRACT_PROMPT: str = """\
Analyse the following English academic text and identify all proper nouns and \
specialised terms that require consistent translation into Persian.

IMPORTANT — Identification method:
• Identify entities by their SEMANTIC ROLE in the sentence, NOT by capitalisation.
  Persian has no uppercase letters, so capitalisation-based NER is meaningless for
  our downstream task.
• Detect: author/person names, place names, institution names, organisation names,
  book/article titles, journal names, named theories or doctrines, and specialised
  academic terms specific to the domain.

### Text
{text}

Return a JSON array where each element has this schema:
[
  {{
    "term": "<English term as it appears in text>",
    "category": "person" | "place" | "institution" | "publication" | "theory" | "term",
    "context": "<short phrase showing how the term is used>",
    "suggested_persian": "<suggested Persian transliteration or translation, or null>"
  }}
]

Return ONLY valid JSON — no markdown fences, no commentary.
"""

# ---------------------------------------------------------------------------
# 7. Incremental NER prompt (finds only NEW entities)
# ---------------------------------------------------------------------------
INCREMENTAL_NER_PROMPT: str = """\
Analyse the following English academic text and identify any NEW proper nouns or \
specialised terms that are NOT already in the known-entity list.

IMPORTANT — Identification method:
• Identify entities by their SEMANTIC ROLE in the sentence, NOT by capitalisation.
  Persian has no uppercase letters, so capitalisation-based detection is invalid.
• Detect: person names, place names, institution names, publication titles,
  named theories/doctrines, and domain-specific academic terms.

### Already-known entities (do NOT repeat these)
{known_entities}

### New text to analyse
{text}

Return a JSON array of ONLY newly discovered entities (not in the known list).
Each element must have this schema:
[
  {{
    "term": "<English term>",
    "category": "person" | "place" | "institution" | "publication" | "theory" | "term",
    "context": "<short phrase showing how the term is used>",
    "suggested_persian": "<suggested Persian transliteration or translation, or null>"
  }}
]

If no new entities are found, return an empty array: []
Return ONLY valid JSON — no markdown fences, no commentary.
"""

# ---------------------------------------------------------------------------
# 8. Bilingual summary update prompt
# ---------------------------------------------------------------------------
SUMMARY_UPDATE_PROMPT: str = """\
Update the running bilingual (English / Persian) summary of the book being translated.

### Current summary
{current_summary}

### New source content (English)
{new_content}

### Translation of the new content (Persian)
{translation}

Instructions:
1. Produce an updated bilingual summary that incorporates the key points from the
   new content.
2. The summary has two sections — English and Persian — each ≤ 300 words.
3. The Persian section MUST be valid RTL text:
   - Use right-to-left paragraph direction.
   - Use Persian punctuation (« » for quotes, ، for comma, ؛ for semicolons).
   - Use Persian numerals (۱۲۳ not 123).
   - Place ZWNJ correctly in compound verbs and affixed words.
4. Keep both halves semantically aligned but naturally phrased in each language.
5. Focus on: main arguments, key concepts, named entities, and chapter progression.

Output format (use these exact headers):

## English Summary
<updated English summary>

## خلاصه فارسی
<updated Persian summary — RTL formatted>
"""

# ---------------------------------------------------------------------------
# 9. Web-context identification prompt
# ---------------------------------------------------------------------------
WEB_CONTEXT_PROMPT: str = """\
Analyse the following English academic text and identify terms or concepts that are
ambiguous and would benefit from web-search disambiguation before translation into
Persian.

### Text
{text}

### Already-resolved terms (do NOT include these)
{known_terms}

Focus on:
- Technical terms with multiple possible Persian translations
- Lesser-known scholars whose name transliteration is uncertain
- Theories or frameworks with established but non-obvious Persian equivalents
- Acronyms or abbreviations whose expansion is unclear from context
- Institutional names that may have official Persian translations

Return a JSON array:
[
  {{
    "term": "<the ambiguous English term>",
    "reason": "<why disambiguation is needed for Persian translation>",
    "search_query": "<suggested web-search query to resolve it>"
  }}
]

If no ambiguous terms are found, return an empty array: []
Return ONLY valid JSON — no markdown fences, no commentary.
"""
