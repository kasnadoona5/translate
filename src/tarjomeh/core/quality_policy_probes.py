"""Synthetic behavior probes: no providers, files, database or environment edits."""

from copy import deepcopy
from dataclasses import replace

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import token_counter_diagnostics
from tarjomeh.glossary.book_review import (
    BOOK_WIDE_RECOMMENDED,
    _sha,
    attest_book_term_proposals,
    bind_original_introductions,
    resolve_reviewed_book_terms,
    scoped_original_permissions,
    skip_pending_proposals,
)
from tarjomeh.glossary.manager import GlossaryEntry
from tarjomeh.memory.manager import MemoryManager, _style_record_is_authoritative


def style_source_probes() -> dict[str, bool]:
    from tarjomeh.memory.manager import _style_hash, _style_record_is_prompt_safe

    config = TarjomehConfig()
    text = (
        "\u0627\u062f\u0639\u0627 \u0628\u0647 \u0634\u0648\u0627\u0647\u062f "
        "\u0648\u0627\u0628\u0633\u062a\u0647 \u0627\u0633\u062a."
    )
    memory = MemoryManager(config)
    scores = dict.fromkeys(("accuracy", "fluency", "terminology", "register"), 9.5)
    result = memory._update_style_profile(
        text + "\n\n" + text,
        source_paragraphs=["Wilson, Adriana.", "Evidence supports the claim."],
        source_paragraph_indices=[0, 1], source_alignment_proven=True,
        paragraph_role="mixed", paragraph_roles=["body", "body"],
        publication_context=True, representative=False, final_scores=scores,
    )
    good = memory.style_sample_records[0]
    bad = deepcopy(good)
    bad.update(source_text="Wilson, Adriana.", text=text)
    bad["source_text_hash"] = _style_hash(bad["source_text"])
    bad.pop("style_source_evidence", None)
    state = memory.to_dict()
    state.update(style_sample_records=[bad], style_samples=[text], style_profile=text)
    restored = MemoryManager(config)
    restored.from_dict(state)
    return {
        "source_eligible_style_memory": result["accepted"]
        and len(memory.style_sample_records) == 1
        and good["source_text"] == "Evidence supports the claim."
        and _style_record_is_authoritative(good)
        and any("catalogue_name_line" in row["reasons"] for row in result["rejections"]),
        "historical_style_source_quarantine": not _style_record_is_prompt_safe(bad)
        and not _style_record_is_authoritative(bad)
        and not restored._render_style_profile()
        and not restored._legacy_style_profile_for_prompt()
        and not restored._effective_style_samples()
        and len(restored.style_sample_records) == 1,
    }


def quality_policy_probes() -> dict[str, bool]:
    from tarjomeh.core.pipeline import _critique_requires_refinement, _merge_readability_evidence
    from tarjomeh.quality.critique import CritiqueResult

    config = TarjomehConfig()
    fa = (
        "\u0627\u062f\u0639\u0627 \u0628\u0647 \u06cc\u06a9 \u0634\u0631\u0637 \u0648"
        "\u0627\u0628\u0633\u062a\u0647 \u0627\u0633\u062a."
    )
    source = "The civic protocol matters."
    metadata = {
        "paragraph_indices": [0],
        "structural_roles": ["body"],
        "source_paragraph_spans": [[0, len(source)]],
        "source_paragraph_hashes": [_sha(source)],
    }
    entries = [
        dict(
            chunk_index=0,
            local_paragraph=0,
            paragraph_index=0,
            paragraph_sha256=_sha(source),
            chapter_position=1,
        )
    ]
    records = {0: (source, metadata)}
    proposals = attest_book_term_proposals(
        [
            dict(source="civic protocol", target="\u0631\u0648\u06cc\u0647", origin=origin)
            for origin in ("first", "second")
        ],
        entries,
        records,
        review_version=4,
    )
    term = {**proposals[0], "status": "approved", "keep_original": True}
    matches, _ = resolve_reviewed_book_terms(source, metadata, 1, [term])
    bind_original_introductions(matches, dict(review_version=4, proposals=[term]), entries, records)
    original = f"{term['target']} (civic protocol)"
    skip = skip_pending_proposals([term, dict(status="proposed"), dict(status="rejected")], 4)
    memory = MemoryManager(config)
    memory.quality_policy_version = 1
    for index in range(2):
        memory.update_after_translation(
            Chunk(
                index=index,
                text=source,
                chapter_title="Chapter",
                section_title="",
                metadata=metadata,
            ),
            fa,
            style_approved=False,
        )
    rows = [dict(chunk_index=i, text=source, translation=fa, status="completed") for i in range(2)]
    prior, _ = memory.prior_context_view(1, rows)
    pending = memory.pending_summary_inputs(1)[2]
    memory.update_after_translation(
        Chunk(index=0, text=source, chapter_title="Chapter", section_title="", metadata=metadata),
        fa,
        style_approved=False,
    )
    curated = GlossaryEntry("protocol", "\u0631\u0648\u06cc\u0647", include_original=True)
    memory.synchronize_curated([curated])
    memory.synchronize_curated([replace(curated, target="\u067e\u0631\u0648\u062a\u06a9\u0644")])
    revised = (
        memory.proper_nouns.all_nouns().get("protocol") == "\u067e\u0631\u0648\u062a\u06a9\u0644"
    )
    no_old_alias = curated.target not in memory.proper_nouns.aliases_for("protocol")
    memory.synchronize_curated([])
    deferred = memory.proper_nouns.is_context_deferred("protocol")
    style = MemoryManager(config)
    style.quality_policy_version = 1
    style._update_style_profile(
        fa,
        source_paragraphs=["Heading", source],
        source_paragraph_indices=[1],
        source_alignment_proven=True,
        paragraph_role="mixed",
        paragraph_roles=["heading", "body"],
        representative=False,
        final_scores=dict.fromkeys(("accuracy", "fluency", "terminology", "register"), 9.5),
    )
    bad = "\u0639\u0628\u0627\u0631\u062a \u0646\u0627\u0642\u0635 \u0627\u0633\u062a"
    fixed = "\u0639\u0628\u0627\u0631\u062a \u06a9\u0627\u0645\u0644 \u0627\u0633\u062a"
    major = dict(
        severity="major",
        current_persian_quote=bad,
        suggested_correction=fixed,
        rationale="Broken grammar and missing predicate.",
    )
    minor = dict(
        severity="minor",
        current_persian_quote=bad[:-4],
        suggested_correction=fixed[:-4],
        rationale="Word choice.",
    )
    routed = []
    for advice in ([major, minor], [minor, major]):
        critique = CritiqueResult(
            average=9,
            issue_details=[
                dict(
                    category="fluency",
                    severity="minor",
                    confidence=0.5,
                    source_quote=source,
                    source_segment_id="p1:s1",
                    current_persian_quote=bad,
                )
            ],
        )
        _merge_readability_evidence(critique, advice, source_text=source, translation=bad + ".")
        routed.append(_critique_requires_refinement(critique, 9))
    diagnostics = []
    config._apply_env_overrides(
        persisted_job=True,
        load_diagnostics=diagnostics,
        environment={"TRANSLATOR_MODEL": config.llm.model},
    )
    equal_ignored = not diagnostics
    config._apply_env_overrides(
        persisted_job=True,
        load_diagnostics=diagnostics,
        environment={"TRANSLATOR_MODEL": "different-synthetic-route"},
    )
    counters = token_counter_diagnostics(dict(prompt_tokens=3, completion_tokens=7, total_tokens=3))
    return {
        "order_independent_readability_evidence": all(routed),
        "identical_term_proposal_dedup": len(proposals) == 1 and proposals[0]["source_count"] == 1,
        "fresh_term_scope_default": proposals[0]["scope_mode"] == BOOK_WIDE_RECOMMENDED,
        "persian_plus_scoped_original": scoped_original_permissions(source, original, matches)
        == ["civic protocol"]
        and not scoped_original_permissions(source, "(civic protocol)", matches),
        "pending_only_term_skip": [item["status"] for item in skip]
        == ["approved", "skipped", "rejected"],
        "curated_revision_authority": revised and no_old_alias and deferred,
        "prior_only_retranslation_context": [
            pair["chunk_index"] for pair in prior.long_term.serialize()
        ]
        == [0],
        "active_memory_version_replacement": sum(
            pair.get("active", True) for pair in memory.long_term.serialize()
        )
        == 2,
        "pending_canonical_summary_coverage": [ref["chunk_index"] for ref in pending] == [0, 1],
        "paragraph_role_style_evidence": bool(style.style_sample_records)
        and _style_record_is_authoritative(style.style_sample_records[0]),
        "raw_token_counter_diagnostics": counters["token_counter_consistency"] == "inconsistent"
        and counters["derived_component_token_sum"] == 10,
        "actual_ignored_policy_diagnostics": equal_ignored
        and len(diagnostics) == 1
        and diagnostics[0]["ignored_environment_variables"] == ["TRANSLATOR_MODEL"],
    }


def export_memory_research_probes() -> dict[str, bool]:
    """Synthetic span/authority/metadata behavior, without calls or file I/O."""
    from types import SimpleNamespace

    from tarjomeh.context.book_researcher import BookResearcher
    from tarjomeh.core.render_identity import RenderChangeLedger
    from tarjomeh.core.term_notes import ensure_inline_proper_noun_originals
    from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
    from tarjomeh.memory.proper_nouns import ProperNouns

    target = "\u062a\u062d\u0648\u0644 \u0641\u0646\u06cc"
    inflected = "\u062a\u062d\u0648\u0644\u06cc \u0641\u0646\u06cc"
    doc = TranslatedDocument(title="Synthetic", paragraphs=[TranslatedParagraph(
        index=0, source_text="A technical change matters.", translated_text=inflected)])
    ledger = RenderChangeLedger(doc)
    ledger.set_authorized({"technical change": target})
    ledger.step("inline_original_anchor", ensure_inline_proper_noun_originals,
                {"technical change": target}, SimpleNamespace(process=lambda value: value))
    source = "Regional Research Council"
    bad = (
        "\u0645\u0627\u0644\u06cc \u0634\u0648\u0631\u0627\u06cc "
        "\u067e\u0698\u0648\u0647\u0634 \u0645\u0646\u0637\u0642\u0647\u200c\u0627\u06cc"
    )
    noun = ProperNouns()
    noun.deserialize({"nouns": {source: bad}, "categories": {source: "organization"},
                      "provenance": {source: {"origin": "observed_translation", "authority": 30}}})
    title = "The Archive and Power"
    raw = f"{title} (2004/2005) studies institutions."
    context, conflicts = BookResearcher._reconcile_publication_context(
        raw, [{"kind": "first_publication", "year": 2005}], book_title=title)
    return {
        "inflected_multiword_anchor_proof": bool(ledger.audit()["passed"]
                                                  and ledger.canonical == [inflected]
                                                  and doc.paragraphs[0].translated_text
                                                  == inflected + " (technical change)"),
        "organization_lexical_boundary": bool(noun.all_nouns().get(source) == bad
                                                and noun.context_deferral_reason(source)
                                                == "organization_lexical_start_unproven"
                                                and source not in noun.inline_eligible_nouns()),
        "book_identity_date_guard": bool(conflicts and "2004/2005" not in context
                                         and "studies institutions" in context),
    }


def checkpoint_classification_probes() -> dict[str, bool]:
    """Generic pure controls for document-scope proof and punctuation evidence."""
    from tarjomeh.core.render_identity import RenderChangeLedger
    from tarjomeh.core.term_notes import audit_inline_english_originals
    from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
    from tarjomeh.quality.integrity import object_marker_dash_artifacts
    from tarjomeh.quality.structure_audit import audit_payload, list_marker_count

    name = "Example Press"
    authorized = {name: "\u0646\u0645\u0648\u0646\u0647"}
    text = authorized[name] + " (Example Press)"
    doc = TranslatedDocument(paragraphs=[
        TranslatedParagraph(0, "Published by Example Press.", text),
        TranslatedParagraph(1, "Example Press returned.", text),
    ])
    ledger = RenderChangeLedger(doc)
    ledger.set_authorized(authorized)
    report = ledger.step("english_original_cleanup", audit_inline_english_originals, authorized)
    good = bool(ledger.audit()["passed"] and report["removed_duplicate_count"] == 1
                and doc.paragraphs[0].translated_text == text
                and doc.paragraphs[1].translated_text == authorized[name])
    doc.paragraphs[0].translated_text = authorized[name]
    keeper_required = not ledger.audit()["passed"]
    source = "There are three reasons: (1) observe; (2) compare; (3) explain."
    target = (
        "\u0633\u0647 \u062f\u0644\u06cc\u0644: \u06f1) \u0627\u0644\u0641\u061b "
        "\u06f2) \u0628\u061b \u06f3) \u062c."
    )
    wrong = target.replace("\u06f2)", "\u06f3)")
    paired = (
        "\u0645\u0646 \u06a9\u062a\u0627\u0628 \u0631\u0627 \u2014 "
        "\u06a9\u0647 \u0645\u06cc\u200c\u062e\u0648\u0627\u0633\u062a\u0645 "
        "\u2014 \u062e\u0631\u06cc\u062f\u0645."
    )
    orphan = (
        "\u062a\u0627\u0631\u06cc\u062e \u062f\u06cc\u06af\u0631\u0627\u0646\u2014\u0631\u0627 "
        "\u0645\u06cc\u200c\u0633\u0627\u0632\u0646\u062f."
    )
    return {
        "document_scoped_duplicate_proof": good and keeper_required,
        "closing_parenthesis_list_evidence": list_marker_count(target) == 3
        and "translation_structure_mismatch" not in audit_payload(source, target)["classifications"]
        and "translation_structure_mismatch" in audit_payload(source, wrong)["classifications"],
        "paired_object_marker_dash_classification": not object_marker_dash_artifacts(
            "Source.", paired)
        and bool(object_marker_dash_artifacts("Source.", orphan)),
    }


def original_keeper_probes() -> dict[str, bool]:
    """Keep uncertain containers without disabling proven duplicate cleanup."""
    from tarjomeh.core.render_identity import RenderChangeLedger
    from tarjomeh.core.term_notes import audit_inline_english_originals
    from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph

    original = "Example Press"
    target = "\u0646\u0645\u0648\u0646\u0647"
    checks = []
    for suffix in ("London", "Ltd", "2020"):
        annotation = f"({original}, {suffix})"
        text = f"{target} ({original})"
        doc = TranslatedDocument(paragraphs=[
            TranslatedParagraph(0, f"Published by {annotation}.", annotation),
            TranslatedParagraph(1, f"{original} publishes books.", text),
            TranslatedParagraph(2, f"{original} returned.", text),
        ])
        ledger = RenderChangeLedger(doc)
        ledger.set_authorized({original: target})
        report = ledger.step("english_original_cleanup", audit_inline_english_originals,
                             {original: target})
        good = bool(ledger.audit()["passed"] and report["removed_duplicate_count"] == 1
                    and report["deferred_deduplication_count"] == 1
                    and doc.paragraphs[0].translated_text == annotation
                    and doc.paragraphs[1].translated_text == text
                    and doc.paragraphs[2].translated_text == target)
        doc.paragraphs[1].translated_text = target
        checks.append(good and not ledger.audit()["passed"])
    return {"shared_original_keeper_eligibility": all(checks)}


def final_entity_local_scope_probe() -> bool:
    """Bind real quality-record coordinates to a later chunk, without I/O."""
    import hashlib
    from types import SimpleNamespace

    from tarjomeh.core.pipeline import _canonical_final_quality_record
    from tarjomeh.memory.final_entities import reconcile_final_entities
    from tarjomeh.memory.proper_nouns import ProperNouns

    def sha(text):
        return hashlib.sha256(text.encode()).hexdigest()

    english = "Example Press"
    old = "\u0627\u0646\u062a\u0634\u0627\u0631\u0627\u062a \u0646\u0645\u0648\u0646\u0647"
    new = "\u0627\u0646\u062a\u0634\u0627\u0631\u0627\u062a \u0645\u062b\u0627\u0644"
    source = english + " publishes books.\n\nAnother issue remains."
    before = old + " (" + english + ").\n\n\u0645\u0633\u0626\u0644\u0647."
    after = before.replace(old, new)
    parts = after.split("\n\n")
    identity = {
        "reconstructed": False,
        "canonical_target_hash": sha(after),
        "paragraph_indices": [10, 11],
        "source_hashes": [sha(p) for p in source.split("\n\n")],
        "target_hashes": [sha(p) for p in parts],
        "target_offsets": [[0, len(parts[0])], [len(parts[0]) + 2, len(after)]],
    }
    edit = {
        "event_type": "extra_final_refinement_completed",
        "payload": {
            "accepted": True,
            "baseline_target_hash": sha(before),
            "candidate_target_hash": sha(after),
            "decisions": [
                {
                    "decision": "accepted",
                    "commit_status": "committed_local",
                    "issue_id": "name",
                    "original_span": old + " (" + english + ")",
                    "resulting_span": new + " (" + english + ")",
                }
            ],
        },
    }
    for segment, should_defer in (
        ("p1:s1", True),
        ("p2:s1", False),
        ("p99:s1", True),
        ("p0:s1", True),
    ):
        nouns = ProperNouns()
        nouns.add_noun(
            english,
            old,
            category="organization",
            provenance="observed_translation",
            evidence_key="anchor:" + sha(source)[:16],
        )
        events = [
            edit,
            {
                "event_type": "critique_completed",
                "payload": {
                    "valid": True,
                    "candidate_target_hash": sha(after),
                    "issue_details": [
                        {
                            "issue_id": "unresolved",
                            "category": "accuracy",
                            "severity": "major",
                            "confidence": 0.95,
                            "source_segment_id": segment,
                            "source_quote": english + " publishes books.",
                            "current_persian_quote": new,
                            "suggested_correction": old,
                        }
                    ],
                },
            },
            {"event_type": "chunk_review_required", "payload": {}},
        ]
        quality = _canonical_final_quality_record(
            SimpleNamespace(get_chunk_events=lambda *a, events=events: events), "probe", 3, after
        )
        result = reconcile_final_entities(
            nouns,
            source,
            after,
            chunk_index=3,
            paragraph_identity=identity,
            quality=quality,
            events=events,
        )
        if not result["records"] or nouns.is_context_deferred(english) != should_defer:
            return False
        if nouns.all_nouns()[english] != (old if should_defer else new):
            return False
    return True


def final_entity_probes() -> dict[str, bool]:
    """Exercise final-name and first-use contracts without provider or file I/O."""
    import hashlib

    from tarjomeh.memory.final_entities import audit_first_occurrences, reconcile_final_entities
    from tarjomeh.memory.proper_nouns import ProperNouns

    def sha(text):
        return hashlib.sha256(text.encode()).hexdigest()

    english = "Example Press"
    old = "\u0627\u0646\u062a\u0634\u0627\u0631\u0627\u062a \u0646\u0645\u0648\u0646\u0647"
    new = "\u0627\u0646\u062a\u0634\u0627\u0631\u0627\u062a \u0645\u062b\u0627\u0644"
    source = "Example Press publishes books."
    before, after = old + " (Example Press)", new + " (Example Press)"
    memory = ProperNouns()
    memory.add_noun(
        english,
        old,
        category="organization",
        provenance="observed_translation",
        evidence_key="anchor:" + sha(source)[:16],
    )
    memory.mark_introduced_from_translation(source, after)
    stale = audit_first_occurrences(memory.serialize(), [{"text": source, "translation": after}])
    missing = audit_first_occurrences(
        memory.serialize(), [{"text": source, "translation": "\u0645\u062a\u0646"}]
    )
    identity = {
        "reconstructed": False,
        "canonical_target_hash": sha(after),
        "paragraph_indices": [0],
        "source_hashes": [sha(source)],
        "target_hashes": [sha(after)],
        "target_offsets": [[0, len(after)]],
    }
    quality = {
        "candidate_target_hash": sha(after),
        "critique_valid": True,
        "critique_candidate_match": True,
        "candidate_matched_critique_present": True,
        "issues": [],
    }
    events = [
        {
            "event_type": "extra_final_refinement_completed",
            "payload": {
                "accepted": True,
                "baseline_target_hash": sha(before),
                "candidate_target_hash": sha(after),
                "decisions": [
                    {
                        "decision": "accepted",
                        "commit_status": "committed_local",
                        "issue_id": "name",
                        "original_span": before,
                        "resulting_span": after,
                    }
                ],
            },
        }
    ]
    report = reconcile_final_entities(
        memory,
        source,
        after,
        chunk_index=0,
        paragraph_identity=identity,
        quality=quality,
        events=events,
    )
    return {
        "final_candidate_entity_memory": bool(
            report["records"]
            and report["records"][0]["action"] == "replaced_superseded_observation"
            and memory.all_nouns()[english] == new
            and old not in memory.aliases_for(english)
            and memory.provenance_for(english)["authority"] == 30
        ),
        "first_occurrence_evidence_classification": bool(
            stale["active_stale_mappings"] == [english]
            and not stale["missing_first_occurrences"]
            and missing["missing_first_occurrences"] == [english]
        ),
    }
