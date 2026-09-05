"""Prompt templates for the Tarjomeh English→Persian translation system.

All prompts are purpose-built for academic English-to-Iranian-Persian translation.
Named-entity recognition uses **semantic role detection** (author names, place names,
institutions, book titles) — never capitalization, since Persian has no uppercase.

Templates use ``str.format()`` for variable injection.
"""

from __future__ import annotations


GENERAL_EDITORIAL_CONTRACT: str = """\
### Editorial contract
- Resolve meaning in context; never force an isolated dictionary equivalent.
- Preserve semantic roles and relations: quantity, negation, modality, comparison,
  causality, and time. Preserve source inconsistencies.
- Preserve titles, names, citations, expressions, and note markers.
- Write clear academic Iranian Persian with correct syntax, punctuation, ZWNJ,
  and ezafe; avoid calques and modifier stacks.
- Use final-heh hamza for ezafe and preserve authored kasra; never add blanket diacritics.
- Keep finite matrix clauses. Recast stacked modifiers naturally without breaking
  appositives or head-complement links.
- Render relational frames such as "in X terms" by their contextual function,
  not an opaque English preposition frame.
- Preserve distinctions and parallel form in recurring coordinated conceptual series when
  supported by source or mandatory terminology.
- Emit only requested output; never add commentary or unexplained foreign prose.
- Mandatory entries and names bind. Advisory context remains revisable
  and cannot override source.
"""

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
• Accuracy and completeness outrank fluency. Natural Persian must never simplify,
  omit, generalize, reinterpret, or weaken the source.
• Use formal academic Persian appropriate for scholarly publication in {country}.
• Follow established Persian academic conventions for transliteration of proper nouns.
  - For transliteration, use the accepted Iranian academic standard.
  - {term_notes_instruction}
  - A specialist label explicitly marked as a technical loanword follows the same
    first-occurrence original policy; ordinary translated concepts do not.
• Preserve the author's argumentative structure and rhetorical style.
• Use entries marked mandatory consistently. Evaluate advisory glossary candidates
  against the source context before using, revising, or rejecting them.
• Use Iranian Persian vocabulary and conventions — not Dari or Afghan Persian.
• Output must be valid right-to-left (RTL) Persian text with correct ZWNJ placement.
• Convert Western numerals to Persian numerals (۰۱۲۳۴۵۶۷۸۹) in running prose ONLY.
  SCHOLARLY APPARATUS IS EXEMPT: keep citation years, page numbers, footnote markers,
  and bibliographic references in Western digits and Latin script exactly as in the
  source — e.g. (Marx 1867, 92) stays (Marx 1867, 92), NOT (مارکس ۱۸۶۷، ۹۲).
• Preserve in-text citations, footnote/endnote markers, and reference callouts
  unchanged. Translate quotations into Persian, but keep the quotation's citation
  (author, year, page) intact in its original form.
• Use standard Persian punctuation: «» for quotation marks, ؛ for semicolons, etc.
• Do NOT add personal commentary, footnotes, or translator's notes unless
  explicitly instructed.
""" + "\n" + GENERAL_EDITORIAL_CONTRACT

ACADEMIC_REGISTER_MODIFIER: str = """\
Use a highly formal, precise, fluent, and scholarly academic register. \
Use complex sentence structures only where their dependencies remain clear in Persian; \
otherwise reorganize or split sentences within the same source paragraph. Employ established scholarly terminology, \
and avoid any colloquialisms, slang, or overly simplified vocabulary. Prefer established, transparent \
Persian academic equivalents to opaque calques or phonetic borrowing. Transliterate a specialist \
label only when that borrowing is established in Persian scholarship or no precise Persian equivalent \
exists, and then apply the configured first-occurrence original policy. The tone must reflect that of \
a publication by a reputable university press (نثر فاخر و دانشگاهی). Make every finite \
clause complete, keep referents and modifier attachments explicit, and preserve conceptual \
series with parallel Persian phrasing and equally visible distinctions.\
"""

# Curated gold exemplars of publication-grade academic Persian. Injected into
# the per-chunk prompt in the academic register only — they anchor what
# "university-press Persian prose" concretely looks like (register, proper-noun
# parentheticals, and scholarly-safe citation handling).
ACADEMIC_EXEMPLARS: str = """\
### Exemplars — match this register and these conventions exactly
Example 1 (political theory, formal register):
EN: The state, on this account, is not a neutral arbiter but an ensemble of institutions that crystallizes prevailing relations of power.
FA: دولت، بنا بر این روایت، داور بی‌طرف نیست، بلکه مجموعهٔ نهادهایی است که مناسبات مسلطِ قدرت را تثبیت می‌کند.

Example 2 (philosophy, first-occurrence proper noun gets the English parenthetical):
EN: Hegemony, as Gramsci conceives it, operates less through coercion than through the organization of consent.
FA: هژمونی، آن‌گونه که گرامشی (Gramsci) در نظر دارد، نه چندان از راه اجبار، بلکه از طریق سازمان‌دهیِ رضایت عمل می‌کند.

Example 3 (quotation translated, citation kept in Latin script and Western digits):
EN: As Marx (1867, 92) observes, "the wealth of societies appears as an immense collection of commodities."
FA: چنان‌که مارکس (Marx) (1867, 92) خاطرنشان می‌کند، «ثروتِ جامعه‌ها همچون مجموعهٔ عظیمی از کالاها پدیدار می‌شود».
"""


# ---------------------------------------------------------------------------
# 2. Per-chunk translation prompt
# ---------------------------------------------------------------------------
TRANSLATE_CHUNK_PROMPT: str = """\
Translate the following English academic text into Persian (فارسی).

{exemplars}
### Terminology policy (mandatory entries and advisory candidates are labeled below)
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
   Accuracy and completeness outrank fluency: fluent Persian must never simplify,
   omit, generalize, reinterpret, or weaken the source.
   Before emitting the answer, ensure that every source proposition and
   qualification is represented, every finite Persian clause has an identifiable
   predicate, pronoun referents and modifier attachments are clear, and no English
   modifier stack or nominal chain has been copied mechanically into Persian.
   Treat every content-bearing source head and its complements as an obligation:
   a fluent paraphrase may reorganize them, but it must retain the governing
   concept, action or relation as well as every modifier that limits its scope.
   Treat coordinated source members separately: preserve every verb, complement,
   contrast, and qualification in pairs such as "extended and qualified" rather
   than allowing one fluent Persian phrase to swallow the other member.
   Prefer natural Persian clause order. Split or reorganize sentences inside the
   same paragraph when that improves comprehension without merging claims,
   deleting qualifications, adding interpretation, or changing logical relations.
   Do not improve readability by simplifying the author's theory, replacing a
   precise relation with a looser paraphrase, or suppressing deliberate complexity.
   A parenthetical term or foreign label supplements its full proposition; it must
   not replace the proposition's head, complement, participants, or relation.
   Keep Persian heads visibly connected to their complements; do not interrupt
   that dependency with an ambiguous parenthetical, dash, or modifier stack.
   Close every list-introducing matrix construction with its own predicate after
   translating all of its members.
2. Apply every entry marked mandatory using its prescribed lexical rendering and
   standard Persian orthography. Treat entries marked advisory as non-binding context.
3. Ensure stylistic and terminological continuity with the preceding translation.
   Continuity is evidence, not authority: do not imitate an awkward construction
   merely because it appears in advisory memory. Preserve parallel distinctions in
   coordinated conceptual series with equally clear Persian phrasing.
4. Maintain paragraph structure; do not merge or split paragraphs. The source text contains
   exactly {paragraph_count} paragraph(s) — your translation MUST also contain exactly
   {paragraph_count} paragraph(s), separated by double newlines (\n\n).
   If a paragraph is a short heading or title, translate it as its own short heading
   paragraph; never merge it into the following body paragraph and never omit it.
    Preserve each contents, list, or table row as its own paragraph and keep its final
    source page label or row identifier unchanged.
   You may reorganize or split sentences inside a paragraph when needed for natural
   Persian, but you must preserve every proposition, qualification, contrast, and
   logical relation and must not change the paragraph count.
5. Use ZWNJ (‌) correctly in compound verbs and affixed words (e.g. می‌خواهد).
6. Proper nouns and first-occurrence originals:
   {term_notes_instruction}
   Apply the same rule to entries explicitly marked as transliterated technical
   loanwords. Never infer this for an ordinary concept with a Persian translation.
7. Scholarly apparatus: keep in-text citations, years, page numbers, and footnote markers
   in Latin script and Western digits exactly as in the source (e.g. (Marx 1867, 92)).
   Translate quoted passages and ordinary citation connective prose, but leave citation
   atoms (author, year, page) and original-language expressions untouched.
8. Output ONLY the Persian translation — no commentary, preamble, or labels.
"""

# ---------------------------------------------------------------------------
# 3. Critique / QA prompt
# ---------------------------------------------------------------------------
CRITIQUE_PROMPT: str = """\
You are a senior academic Persian editor reviewing a translation from English to \
Iranian Persian. Evaluate the translation rigorously and return a structured JSON report.

### Source (English)
{source_text}

Sentence labels such as [p1:s2] are stable review coordinates. Copy the label
for the sentence containing each source quote into source_segment_id. Never
copy these labels into the Persian translation.

### Translation (Persian)
{translation}

### Terminology policy (glossary + established proper-noun renderings)
Only entries explicitly marked mandatory may cause a terminology violation.
Advisory auto-extracted candidates are contextual suggestions and may be rejected.
For mandatory entries, enforce the lexical rendering while accepting equivalent
standard Persian ZWNJ/spacing forms; never prefer malformed typography:
{terminology}

### Bounded review context
Use this only to resolve discourse, style, and reference ambiguity. Do not
criticize text outside the current source/translation pair:
{review_context}

""" + GENERAL_EDITORIAL_CONTRACT + """

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
      "category": "accuracy" | "omission" | "addition" | "terminology" |
        "name" | "number" | "citation" | "fluency" | "register" | "typography",
      "severity": "critical" | "major" | "minor",
      "confidence": <0-1>,
      "source_segment_id": "<stable label such as p1:s2>",
      "source_quote": "<exact short quote copied from the current English source>",
      "current_persian_quote": "<exact short quote copied from the current Persian translation>",
      "suggested_correction": "<compact improved Persian span>",
      "rationale": "<brief explanation in English>"
    }}
  ]
}}

Scoring guide:
- accuracy  : semantic equivalence, completeness, no omissions or additions
- fluency   : natural Persian prose flow, correct grammar, proper ZWNJ usage
- terminology: adherence to glossary, consistency of technical terms
- register  : appropriateness of academic tone, avoidance of colloquialisms
- overall   : holistic quality

MQM rules:
- Report at most 8 actionable issues, prioritizing critical and major issues.
- Categories may also be omission, addition, name, number, or citation.
- Critical accuracy/omission/number/citation/name issues and major
  accuracy/terminology issues are blocking.
- Minor style preferences must stay minor and must not be inflated to force edits.
- Every quote must occur verbatim in the current source or translation.
- Every source_segment_id must identify the sentence containing source_quote.
- For semantic issues, the rationale must identify the affected relation or
  proposition in context, not merely offer a different dictionary synonym.
- Treat source-order calques, opaque modifier stacks, unclear attachment or referents,
  excessive nominalization, malformed participles, and coordinated conceptual series
  whose distinctions or parallelism were lost as objective fluency issues.
- Compare every source sentence with its Persian counterpart before scoring. Check
  predicate completeness, semantic roles and valency, scope and modifier attachment,
  coordinated parallel terms, and accidental duplication of one meaning in two
  Persian predicates. A fluent alternative is acceptable only if all source content
  and technical precision remain unchanged.
- Build a compact proposition checklist for each source sentence before assigning
  scores: matrix action, coordinated actions, participants, negation/modality,
  content-bearing nominal heads and complements, explicit quantities, contrasts,
  and qualifications. Report an omission whenever one checklist member has no
  Persian equivalent, even if the remaining sentence is grammatical and
  stylistically polished. Do not let a translated modifier stand in for an omitted
  head concept such as a process, relation, institution, or form of change.
- Check every source-authored announced quantity against the Persian wording itself,
  even when the enumeration continues in another paragraph or chunk. Follow the
  source exactly when its announcement and later list disagree; never silently repair
  an author's inconsistency.
- Check that parenthetical explanations and appositives attach to the correct Persian
  head and do not interrupt or duplicate the finite predicate. Render reflection,
  restatement, or clarification as a grammatically integrated Persian relation rather
  than preserving English punctuation around an unattached phrase.
- A sentence may be formally worded yet still be unpublishable if a Persian reader
  must reconstruct its English syntax to understand it. Report that defect precisely.
- A ZWNJ/spacing-only difference is not a terminology error. Use typography only
  when the current Persian itself violates standard orthography.
- Do not praise, repeat the full source, repeat the full translation, or provide
  commentary outside the JSON. Keep each rationale under 60 words.

Return ONLY valid JSON — no markdown fences, no commentary outside the JSON.
"""

PERSIAN_READABILITY_REVIEW_PROMPT: str = """\
You are a conservative Persian copy editor. Review only the Persian paragraph
below for objective readability defects. You do not have the English source and
must not infer, simplify, add, remove, reinterpret, or weaken meaning.

Report only defects visible in Persian itself: a missing finite predicate, broken
agreement or dependency, an unresolved or dangling referent, malformed punctuation
or parentheses, accidental repetition, an opaque source-order calque, or a modifier
stack whose attachment is grammatically unclear. Also report visibly malformed word
construction, such as an accidental extra morpheme or a ZWNJ that fuses independent
words. Formal complexity, uncommon technical vocabulary, and a merely preferable
synonym are not defects.

Check whether an appositive or parenthetical explanation interrupts the dependency
between a Persian head and its complement, and whether ZWNJ has incorrectly welded
two independent words rather than joining a real compound or affix. Do not replace
technical vocabulary merely because it is uncommon.

For long enumerations, verify that the introductory matrix construction still has
its own finite predicate after all listed members. A predicate inside one list item
does not complete an unfinished frame that governs the whole list.

Also report a governed phrase or complement that is accidentally stated twice in
one clause with competing predicates between its two occurrences. Do not report
deliberate rhetorical repetition when both occurrences have independent grammatical
roles.

For predicate completeness, identify the matrix predicate of each independent
clause. A finite verb inside a relative clause introduced by words such as «که»
does not by itself complete the surrounding matrix clause. Do not propose a lexical
or semantic change when the matrix dependency is already complete.

Persian paragraph:
{translation}

Return ONLY this JSON object with at most 4 issues:
{{
  "issues": [
    {{
      "severity": "major" | "minor",
      "current_persian_quote": "<exact short span copied from the paragraph>",
      "suggested_correction": "<compact Persian replacement span>",
      "rationale": "<brief objective Persian-grammar reason in English>"
    }}
  ]
}}

Every current_persian_quote must occur verbatim. A suggestion is advisory: the
source-aware refiner will independently accept or reject it. Return an empty list
when the Persian is coherent. No markdown and no commentary.
"""

# ---------------------------------------------------------------------------
# 4. Refinement prompt (applies critique feedback)
# ---------------------------------------------------------------------------
REFINE_PROMPT: str = """\
You are refining an academic English-to-Persian translation after an editorial critique.
The critique identifies high-risk passages; it is not automatically authoritative.

### Source (English)
{source_text}

### Current translation (Persian)
{translation}

### Editorial critique (JSON)
{critique}

### Terminology policy (mandatory glossary + advisory discovered candidates)
{terminology}

### Bounded review context
{review_context}

""" + GENERAL_EDITORIAL_CONTRACT + """

Instructions:
1. Evaluate each critique issue in severity order: "critical", then "major", then "minor".
2. If the critique is correct, revise the translation to fix the issue.
3. If the current translation is more accurate in context, preserve it; do not change a
   correct rendering merely because the critic suggested an alternative.
   Accept a minor fluency or style issue only when it identifies an objective defect
   such as ambiguity, broken grammar/agreement, an unnatural calque, or invalid
   orthography. Opaque modifier stacking, unclear dependency or reference, malformed
   participles, and lost parallelism in a coordinated conceptual series are objective
   defects. Reject synonym swaps and stylistic preferences without such evidence.
   A target-only readability advisory, when present inside an issue, is secondary
   evidence only. Accept it only after checking the English source; reject it if it
   changes scope, emphasis, modality, terminology, or any proposition.
4. Preserve every part of the translation that has no validated issue.
5. Apply entries marked mandatory using their prescribed lexical rendering, with
   standard Persian orthography. Treat advisory candidates as optional context and
   do not let them override source meaning or a mandatory entry.
6. Ensure correct ZWNJ placement, Persian numerals in prose, and RTL punctuation — but keep
   citations, years, and page numbers in Latin script and Western digits.
7. Return exactly one compact decision for every issue ID. A serious issue must
   be evaluated, but the critic's suggested wording is never mandatory.
   Use source_segment_id only to locate the issue; never reproduce sentence
   labels in the Persian translation.
8. Include the complete Persian translation exactly once. Do not repeat it in
   issue decisions or rationales.
    Every resulting_span must be copied verbatim from that returned translation.
   You may reorder or split sentences within the same paragraph to repair an objective
   fluency defect, but preserve every proposition and keep paragraph boundaries fixed.
   Before returning the candidate, verify every coordinated source member separately
   and identify a natural Persian matrix predicate for each independent clause.
   Compare the complete candidate with the source again: every content-bearing head,
   complement, action, contrast, quantity and qualification present before editing
   must still be represented after editing. A
   reduction in surface verb count is allowed when Persian grammar expresses the same
   relations more naturally; loss of a source action, contrast, modifier, or scope is
   never allowed.
9. Return ONLY valid JSON with this schema:
{{
  "translation": "<the final Persian translation only>",
  "decision": "revised" | "preserved" | "mixed",
  "rationale": "<brief holistic English note, under 80 words>",
  "issue_decisions": [
    {{
      "issue_id": "<exact issue ID>",
      "decision": "accepted" | "rejected" | "partially_applied",
      "resulting_span": "<short final Persian span, not the full translation>",
      "rationale": "<brief English reason, under 50 words>"
    }}
  ]
}}
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
• Use category "technical_loanword" only when Persian scholarly prose normally
  transliterates the source label rather than replacing it with an established
  Persian lexical equivalent. Do not use it for ordinary translated concepts.
• For category "term", return only a minimal, reusable lexical equivalent. Never
  absorb a nearby subject, object, author, field, time, or other contextual modifier.
  Set context_independent to false when no context-neutral equivalent is evidenced.

### Text
{text}

Return a JSON array where each element has this schema:
[
  {{
    "term": "<English term as it appears in text>",
    "category": "person" | "place" | "institution" | "organization" | "publication" | "product" | "theory" | "technical_loanword" | "term",
    "context": "<short phrase showing how the term is used>",
    "suggested_persian": "<minimal Persian transliteration or lexical equivalent, or null>",
    "exact_source_span": "<the exact source term only>",
    "exact_target_span": "<exact Persian span if accepted text is supplied, otherwise empty>",
    "context_independent": <true only when suggested_persian adds no surrounding context>
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
  named theories/doctrines, and domain-specific academic terms. Classify a
  specialist label as "technical_loanword" only when its Persian rendering is
  a transliteration rather than an established lexical translation.
• For category "term", return only a minimal, reusable lexical equivalent. Never
  include a nearby subject, object, author, field, time, or contextual modifier.
  Set context_independent to false if the accepted Persian supplies only a
  context-bound rendering.

### Already-known entities (do NOT repeat these)
{known_entities}

### New text to analyse
{text}

Return a JSON array of ONLY newly discovered entities (not in the known list).
Each element must have this schema:
[
  {{
    "term": "<English term>",
    "category": "person" | "place" | "institution" | "organization" | "publication" | "product" | "theory" | "technical_loanword" | "term",
    "context": "<short phrase showing how the term is used>",
    "suggested_persian": "<minimal Persian transliteration or lexical equivalent, or null>",
    "exact_source_span": "<the exact source term only>",
    "exact_target_span": "<exact Persian span copied from accepted text, or empty>",
    "context_independent": <true only when suggested_persian adds no surrounding context>
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
6. Summarize only claims supported by the current summary and new translated content.
   Do not extrapolate from a table of contents, book research, or future chapter titles.
7. Treat the English source as factual authority when it conflicts with wording in
   the translation. Preserve explicit quantities and established Persian terms;
   do not coin a near-copy spelling or broaden a term with contextual material.
8. Return plain text under the required headers. Do not emit HTML or XML tags.

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
