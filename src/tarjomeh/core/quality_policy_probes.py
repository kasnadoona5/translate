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
