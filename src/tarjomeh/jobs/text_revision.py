"""Operator-only prepare/check/apply for separately approved canonical edits.

Never called by translation, export, the UI or deployment. No provider requests.
Use a stopped Tarjomeh service and the host maintenance locks for application.
"""

from __future__ import annotations

import argparse
import json
from contextlib import closing
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from tarjomeh.jobs.summary_revision import _approval, _backup, _connect, _hash_file, _read
from tarjomeh.memory.summary_revision import document_hash, text_hash
from tarjomeh.memory.text_revision import revised_coverage

PREFIX = "reviewed_text_revision:"


def _encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _artifacts(conn, job_id):
    return {row["artifact_key"]: json.loads(row["payload"]) for row in conn.execute(
        "SELECT * FROM job_artifacts WHERE job_id=? ORDER BY artifact_key", (job_id,))}


def _snapshot(conn, job_id):
    job, state = _read(conn, job_id)
    tables = {}
    for name in ("chunks", "memory_state", "job_artifacts", "chunk_events", "qa_issues",
                 "issue_decisions", "job_log", "job_workers"):
        rows = [dict(row) for row in conn.execute(
            f'SELECT * FROM "{name}" WHERE job_id=? ORDER BY rowid', (job_id,))]
        if name == "job_artifacts":
            rows = [row for row in rows if not row["artifact_key"].startswith(PREFIX)]
        elif name == "chunk_events":
            rows = [row for row in rows if row["event_type"] != "reviewed_text_revision_receipt"]
        tables[name] = rows
    tables["term_decisions"] = [dict(row) for row in conn.execute(
        "SELECT * FROM book_term_decisions WHERE source_sha256=? ORDER BY source_key,sense_key",
        (job.get("source_sha256"),))]
    output = Path(job["output_path"]) if job.get("output_path") else None
    publication_hash = _hash_file(output) if output and output.exists() else None
    source_hash = _hash_file(Path(job["input_path"]))
    if source_hash != job.get("source_sha256"):
        raise ValueError("Uploaded source bytes changed")
    return job, state, document_hash({"job": job, "tables": tables, "publication": publication_hash,
                                     "source_sha256": source_hash})


def _source_document(job, artifacts):
    from tarjomeh.core.pipeline import TranslationPipeline, apply_chapter_selection

    source = Path(job["input_path"])
    digest = _hash_file(source)
    if job.get("source_sha256") != digest:
        raise ValueError("Source file differs from the recorded uploaded bytes")
    # Parser resolution is pure; do not construct clients or a pipeline worker.
    parser = TranslationPipeline._get_parser(None, source)
    if hasattr(parser, "structure_version"):
        parser.structure_version = artifacts.get("document_structure_version", {}).get("version", 1)
    document = parser.parse(source)
    settings = json.loads(job.get("config") or "{}")
    return apply_chapter_selection(document, settings.get("translation", {}).get("chapter_selection")), digest


def _stage(conn, proposal):
    from tarjomeh.chunking.chunker import Chunk
    from tarjomeh.core.pipeline import (
        _actionable_structure_findings, _blocking_structure_findings,
        _canonical_chunk_paragraph_identity, _reviewed_term_candidate_admission,
        _target_units_from_identity,
    )
    from tarjomeh.glossary.book_review import bind_original_introductions, resolve_reviewed_book_terms
    from tarjomeh.quality.integrity import PostEditIntegrityGate
    from tarjomeh.quality.note_evidence import attach_chunk_note_evidence
    from tarjomeh.glossary.manager import GlossaryManager
    from tarjomeh.glossary.compliance import GlossaryComplianceChecker

    if set(proposal) != {"job_id", "changes"} or not isinstance(proposal["changes"], list) or not proposal["changes"]:
        raise ValueError("Proposal requires job_id and a nonempty changes list")
    job, state, snapshot_hash = _snapshot(conn, proposal["job_id"])
    if state.get("quality_policy_version") != 1:
        raise ValueError("Exact canonical memory policy required")
    artifacts = _artifacts(conn, job["id"])
    document, source_hash = _source_document(job, artifacts)
    settings = json.loads(job.get("config") or "{}")
    glossary_config = settings.get("glossary", {})
    paths = list(dict.fromkeys([p for p in [glossary_config.get("path"),
                                *glossary_config.get("paths", [])] if p]))
    glossary_files = [{"path": str(path), "sha256": _hash_file(Path(path))} for path in paths]
    glossary = GlossaryManager()
    glossary.load_many(paths)
    identities = artifacts.get("canonical_chunk_paragraphs_v1", {}).get("chunks", {})
    canonical = {row["chunk_index"]: dict(row) for row in conn.execute(
        "SELECT * FROM chunks WHERE job_id=? ORDER BY chunk_index", (job["id"],))}
    if state.get("summary_coverage"):
        from tarjomeh.memory.summary_revision import select_summary

        def proven(ref):
            row = canonical.get(ref.get("chunk_index"))
            return bool(row and row["status"] in {"completed", "needs_review"}
                        and ref.get("source_sha256") == text_hash(row["text"])
                        and ref.get("target_sha256") == text_hash(row["translation"]))

        selected = select_summary(state.get("summary_history", []), proven)
        if (selected.get("summary") != state.get("bilingual_summary")
                or selected.get("coverage") != state.get("summary_coverage")):
            raise ValueError("Active summary wording/coverage is not backed by eligible history")
    changes, seen = [], set()
    review = artifacts.get("book_term_review_v1", {})
    if review and review.get("source_sha256") != source_hash:
        raise ValueError("Book-term review source has changed")
    for requested in proposal["changes"]:
        if set(requested) != {"chunk_index", "before", "after", "reason"}:
            raise ValueError("Each correction needs complete before/after text and a reason")
        index = requested["chunk_index"]
        if type(index) is not int or index in seen or index not in canonical:
            raise ValueError("Duplicate or unknown correction chunk")
        seen.add(index)
        row = canonical[index]
        if (row["status"] not in {"completed", "needs_review"}
                or row["translation"] != requested["before"]
                or not isinstance(requested["after"], str) or not requested["after"].strip()
                or requested["after"] == requested["before"] or not str(requested["reason"]).strip()):
            raise ValueError("Correction is stale, empty or unchanged")
        metadata = json.loads(row["metadata"] or "{}")
        chunk = Chunk(index, row["text"], "", "", metadata=metadata)
        spans, indices = metadata.get("source_paragraph_spans", []), metadata.get("paragraph_indices", [])
        if (not indices or len(indices) != len(spans)
                or any(type(pid) is not int or not 0 <= pid < len(document.all_paragraphs)
                       or row["text"][span[0]:span[1]] != document.all_paragraphs[pid].text
                       for pid, span in zip(indices, spans, strict=True))):
            raise ValueError("Source paragraph identity is not proven by the recorded parser")
        before_parts = _target_units_from_identity(requested["before"], identities.get(str(index)))
        if before_parts is None or identities[str(index)].get("reconstructed") is not False:
            raise ValueError("Exact prior paragraph identity required")
        after, identity = _canonical_chunk_paragraph_identity(chunk, requested["after"])
        if (after != requested["after"] or identity["reconstructed"]
                or identity["source_count"] != identity["target_count"]
                or identity["paragraph_indices"] != identities[str(index)]["paragraph_indices"]
                or identity["source_hashes"] != identities[str(index)]["source_hashes"]):
            raise ValueError("Correction changes or reconstructs paragraph boundaries")
        attach_chunk_note_evidence(document, [chunk])
        matches, uncertain = resolve_reviewed_book_terms(
            chunk.text, metadata, metadata.get("chapter_position", 1), review.get("proposals", []))
        bind_original_introductions(matches, review,
            artifacts.get("book_term_body_index_v1", {}).get("entries", []),
            {i: (r["text"], json.loads(r["metadata"] or "{}")) for i, r in canonical.items()})
        terms = _reviewed_term_candidate_admission(chunk.text, requested["before"], after, metadata, matches)
        if uncertain or not terms["accepted"]:
            raise ValueError("Mandatory term scope/compliance is unresolved; withhold correction")
        curated = glossary.find_terms(chunk.text, context=state.get("book_context", ""))
        if glossary_config.get("enable_compliance_check", True) and not GlossaryComplianceChecker().check(
            after, chunk.text, glossary, entries=curated,
        ).compliant:
            raise ValueError("Correction conflicts with a mandatory curated term")
        gate = PostEditIntegrityGate(note_evidence=metadata.get("source_note_evidence_v1"))
        integrity = gate.evaluate(chunk.text, after, previous=requested["before"], stage="reviewed_text_revision")
        if not integrity.accepted:
            raise ValueError("Correction failed deterministic integrity: " + _encode(integrity.to_dict()))
        old_structure = _blocking_structure_findings(_actionable_structure_findings(chunk.text, requested["before"]))
        new_structure = _blocking_structure_findings(_actionable_structure_findings(chunk.text, after))
        if any(finding not in old_structure for finding in new_structure):
            raise ValueError("Correction introduces a source-structure mismatch")
        changes.append({**deepcopy(requested), "source": chunk.text, "identity": identity,
                        "integrity": integrity.to_dict(), "book_terms": terms})
    changes.sort(key=lambda item: item["chunk_index"])
    continuity = {"summary_sha256": document_hash(state.get("bilingual_summary", {})),
                  "summary": deepcopy(state.get("bilingual_summary", {})),
                  "before": deepcopy(state.get("summary_coverage", [])),
                  "after": revised_coverage(state.get("summary_coverage", []), changes)}
    manifest = {"schema": 1, "job_id": job["id"], "document_sha256": source_hash,
                "expected_memory_sha256": document_hash(state), "snapshot_sha256": snapshot_hash,
                "changes": changes, "summary_continuity": continuity,
                "glossary_files": glossary_files,
                "authority": "explicit_human_review_not_model_score_or_semantic_certification"}
    _revised_memory(state, manifest, "PREPARATION_ONLY", "PREPARATION_ONLY")
    return manifest


def _revised_memory(before, manifest, digest, reviewer):
    state = deepcopy(before)
    for change in manifest["changes"]:
        index, old, new = change["chunk_index"], change["before"], change["after"]
        active = [p for p in state.get("past_translations", [])
                  if p.get("active", True) and p.get("chunk_index") == index]
        if len(active) != 1 or active[0].get("translation") != old or active[0].get("source") != change["source"]:
            raise ValueError("Active Layer-3 pair does not match canonical text")
        prior = active[0]
        prior["active"] = False
        revised = deepcopy(prior)
        for field in ("final_scores", "quality_approved", "style_approved", "quality_evidence", "reviewer_scores"):
            revised.pop(field, None)
        revised.update(active=True, entry_id=len(state["past_translations"]), translation=new,
                       canonical_target_hash=text_hash(new), reliable=False, continuity_trust="advisory_review",
                       reviewed_text_revision=digest)
        state["past_translations"].append(revised)
        for pair in state.get("short_term_context", []):
            if pair.get("chunk_index") == index:
                if pair.get("source") != change["source"] or pair.get("translation") != old:
                    raise ValueError("Active Layer-4 pair is stale")
                pair.update(translation=new, trust="advisory_review")
        for key in ("summary_pending",):
            state[key] = revised_coverage(state.get(key, []), [change])
        # Existing style records may share a chunk, but their exact complete
        # paragraph pairs must still occur unchanged. Never teach edited prose.
        for sample in state.get("style_sample_records", []):
            if sample.get("active", True) and sample.get("source_chunk_index") == index:
                source_parts = change["source"].split("\n\n")
                target_parts = new.split("\n\n")
                if (sample.get("source_text"), sample.get("text")) not in list(zip(source_parts, target_parts, strict=True)):
                    raise ValueError("Correction changes an active style sample; withhold it")
    continuity = manifest["summary_continuity"]
    if continuity["before"] != continuity["after"]:
        state["summary_history"].append({
            "summary": deepcopy(state["bilingual_summary"]), "coverage": deepcopy(continuity["after"]),
            "reviewed_text_continuity": {"schema": 1, "revision_id": digest,
                "approval_sha256": digest, "reviewer": reviewer, "manifest": deepcopy(manifest)},
        })
        state["summary_coverage"] = deepcopy(continuity["after"])
    return state


def prepare(path: Path, proposal: dict) -> dict:
    with closing(_connect(path)) as conn:
        conn.execute("BEGIN")
        return _stage(conn, proposal)


def _proposal(manifest):
    return {"job_id": manifest["job_id"], "changes": [
        {key: item[key] for key in ("chunk_index", "before", "after", "reason")}
        for item in manifest["changes"]]}


def _verify(conn, manifest):
    if manifest != _stage(conn, _proposal(manifest)):
        raise ValueError("Stale or altered manifest; prepare again and obtain new approval")


def check(path: Path, manifest: dict) -> dict:
    with closing(_connect(path)) as conn:
        conn.execute("BEGIN")
        digest = document_hash(manifest)
        _, _, current = _snapshot(conn, manifest["job_id"])
        receipt = _artifacts(conn, manifest["job_id"]).get(PREFIX + digest)
        if receipt:
            if (receipt.get("status") != "applied" or receipt.get("manifest") != manifest
                    or receipt.get("after_snapshot_sha256") != current):
                raise ValueError("Applied receipt is stale; read-only investigation required")
            return {"status": "already_applied_verified", "manifest_sha256": digest,
                    "after_snapshot_sha256": current}
        _verify(conn, manifest)
    return {"status": "dry_run_only", "manifest_sha256": document_hash(manifest),
            "semantic_accuracy": "requires_complete_source_based_human_review"}


def apply(path: Path, manifest: dict, *, approval: str, reviewer: str, backup: Path,
          operator_locks_held: bool = False) -> dict:
    digest, job_id = document_hash(manifest), manifest["job_id"]
    _approval(approval, digest, reviewer, operator_locks_held)
    with closing(_connect(path)) as conn:
        conn.execute("BEGIN")
        _, _, current = _snapshot(conn, job_id)
        old = _artifacts(conn, job_id).get(PREFIX + digest)
        if old:
            if (old.get("status") != "applied" or old.get("manifest") != manifest
                    or old.get("after_snapshot_sha256") != current):
                raise ValueError("Applied receipt is stale; read-only investigation required")
            return {**old, "status": "already_applied"}
        _verify(conn, manifest)
    snapshot = _backup(path, backup, job_id, manifest["expected_memory_sha256"])
    with closing(_connect(path, writable=True)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        _verify(conn, manifest)
        _, before = _read(conn, job_id)
        after = _revised_memory(before, manifest, digest, reviewer)
        timestamp = datetime.now(UTC).isoformat()

        def event(index, kind, payload):
            conn.execute("INSERT INTO chunk_events VALUES (?,?,?,?,?)",
                         (job_id, index, timestamp, kind, _encode(payload)))

        def artifact(key, payload):
            conn.execute("INSERT INTO job_artifacts VALUES (?,?,?,?) ON CONFLICT(job_id,artifact_key) "
                         "DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at",
                         (job_id, key, _encode(payload), timestamp))

        artifacts = _artifacts(conn, job_id)
        identities = deepcopy(artifacts["canonical_chunk_paragraphs_v1"])
        from tarjomeh.core.pipeline import _final_candidate_selection_payload, _final_canonical_admission_payload
        for change in manifest["changes"]:
            index, identity = change["chunk_index"], change["identity"]
            identities["chunks"][str(index)] = identity
            conn.execute("UPDATE chunks SET translation=?,status='needs_review' WHERE job_id=? AND chunk_index=?",
                         (change["after"], job_id, index))
            event(index, "final_candidate_selection", _final_candidate_selection_payload(change["after"], identity))
            event(index, "final_canonical_admission", _final_canonical_admission_payload(change["after"], identity))
            event(index, "final_quality_admission", {
                "candidate_target_hash": text_hash(change["after"]), "durable_authority": False,
                "authority_source": "explicit_human_review_not_fresh_model_scores",
                "quality_approved": False, "style_approved": False, "final_scores": {},
                "issue_status_counts": {}, "reviewed_text_revision": digest,
                "historical_model_scores_reused": False,
            })
            event(index, "language_quality_review", {"review_reason": "reviewed_canonical_text_requires_fresh_quality",
                "candidate_target_hash": text_hash(change["after"]), "reviewed_text_revision": digest})
            event(index, "memory_update_policy", {
                "canonical_target_hash": text_hash(change["after"]), "canonical_chunk_index": index,
                "long_term_reliable": False, "long_term_trust": "advisory",
                "short_term_trust": "advisory_review", "style_sample_added": False,
                "quality_approved": False, "style_approved": False, "reviewed_text_revision": digest,
                "reliability_reasons": ["reviewed_canonical_text_without_fresh_model_scores"],
            })
        artifact("canonical_chunk_paragraphs_v1", identities)
        artifact("text_revision_publication_pending_v1", {"revision_id": digest, "state": "pending_explicit_review_export",
                 "previous_publication_retained": True})
        conn.execute("UPDATE memory_state SET state_data=? WHERE job_id=?", (_encode(after), job_id))
        _, _, after_hash = _snapshot(conn, job_id)
        receipt = {"status": "applied", "revision_id": digest, "reviewer": reviewer,
                   "manifest": deepcopy(manifest), "before_state": before, "backup": snapshot,
                   "after_snapshot_sha256": after_hash, "publication": "pending_explicit_review_export"}
        artifact(PREFIX + digest, receipt)
        event(-1, "reviewed_text_revision_receipt", {"revision_id": digest, "approval_sha256": approval,
              "canonical_chunks": [item["chunk_index"] for item in manifest["changes"]], "additional_llm_calls": 0})
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "check", "apply"])
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--approve-sha256", default="")
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--operator-locks-held", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.file.read_text(encoding="utf-8"))
    if args.action == "prepare":
        result = prepare(args.db, payload)
    elif args.action == "check":
        result = check(args.db, payload)
    else:
        if args.backup is None:
            parser.error("application requires a new exclusive backup file")
        result = apply(args.db, payload, approval=args.approve_sha256, reviewer=args.reviewer,
                       backup=args.backup, operator_locks_held=args.operator_locks_held)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
