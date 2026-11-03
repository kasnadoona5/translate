"""Read-only collector evidence. Never grants translation or export permission."""
from __future__ import annotations

import json

from tarjomeh.quality.integrity import source_note_cue_inventory
from tarjomeh.quality.note_evidence import paragraph_note_evidence, text_hash, verified_note_cues


def _reconstructed_proof(source, identity, document):
    if document.format_type != "pdf":
        raise ValueError("Source format has no recorded PDF note geometry")
    indices, hashes = identity.get("paragraph_indices", []), identity.get("source_hashes", [])
    if not indices or len(indices) != len(hashes):
        raise ValueError("Canonical source paragraph identity unavailable")
    paragraphs = document.all_paragraphs
    parts, texts, cursor, previous = [], [], 0, -1
    for index, digest in zip(indices, hashes, strict=True):
        if type(index) is not int or not previous < index < len(paragraphs):
            raise ValueError("Invalid canonical source paragraph order")
        paragraph = paragraphs[index]
        if text_hash(paragraph.text) != digest:
            raise ValueError("Canonical source paragraph hash changed")
        if paragraph.metadata.get("has_superscript") and not paragraph.metadata.get("superscript_markers"):
            raise ValueError("Source contains unlocated superscript geometry")
        evidence = paragraph_note_evidence(paragraph.text, paragraph.metadata)
        if any(r["reason"] not in {"not_numeric_note", "formula_identifier_or_enumeration"}
               for r in evidence["rejected"]):
            raise ValueError("Source note geometry could not be completely located")
        parts.append({"span": [cursor, cursor + len(paragraph.text)],
                      "paragraph_index": index,
                      "evidence": evidence})
        texts.append(paragraph.text)
        cursor += len(paragraph.text) + 2
        previous = index
    if "\n\n".join(texts) != source:
        raise ValueError("Canonical paragraphs do not match the exact saved source")
    proof = {"schema": 1, "source_sha256": text_hash(source), "paragraphs": parts}
    verified_note_cues(source, proof)
    return proof


def audit_note_markers(job, chunks, artifacts):
    """Use saved geometry or an exact uploaded-source replay, never target inference.

    Source replay uses the saved parser and chapter selection, without a worker,
    database writes or provider clients. Absence/rejection of geometry is not
    evidence that a flattened source contained no notes.
    """
    document, attempted, replay_reason = None, False, ""
    rows, surplus, unresolved = [], [], []
    for chunk in chunks:
        if chunk.get("status") not in {"completed", "needs_review"}:
            continue
        source, target = str(chunk.get("text") or ""), str(chunk.get("translation") or "")
        reason = ""
        try:
            metadata = chunk.get("metadata") or {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            proof = metadata.get("source_note_evidence_v1")
            verified_note_cues(source, proof)
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            proof, reason = None, str(exc)
        scope = "verified_source_geometry" if proof is not None else "text_only_geometry_unavailable"
        if proof is None and job.get("input_path"):
            if not attempted:
                attempted = True
                try:
                    from tarjomeh.jobs.text_revision import _source_document
                    from tarjomeh.jobs.summary_revision import _hash_file
                    from pathlib import Path

                    replayed, digest = _source_document(job, artifacts)
                    if _hash_file(Path(job["input_path"])) != digest:
                        raise ValueError("Uploaded source changed during report replay")
                    document = replayed
                except Exception as exc:
                    # Parser/file failures are unavailable report evidence, not
                    # permission to infer notes or change an admission decision.
                    replay_reason = f"{type(exc).__name__}: {exc}"
            if document is not None:
                try:
                    identity = artifacts.get("canonical_chunk_paragraphs_v1", {}).get(
                        "chunks", {}).get(str(chunk["chunk_index"]), {})
                    proof = _reconstructed_proof(source, identity, document)
                    scope = "verified_uploaded_source_replay"
                    reason = ""
                except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
                    reason = str(exc)
            else:
                reason = replay_reason
        if proof is None and reason:
            scope = "text_only_geometry_rejected"
        inventory = source_note_cue_inventory(source, target, note_evidence=proof)
        row = {"chunk_index": chunk.get("chunk_index"),
               "source_sha256": text_hash(source), "target_sha256": text_hash(target),
               **inventory, "evidence_scope": scope,
               "geometry_unavailable_reason": reason,
               "authority": "current_report_only_not_admission_or_semantic_verification"}
        rows.append(row)
        if inventory["surplus"]:
            finding = {"chunk_index": chunk.get("chunk_index"), "markers": inventory["surplus"]}
            (surplus if proof is not None else unresolved).append(finding)
    return {"chunks": rows, "surplus_note_markers": surplus,
            "unresolved_surplus_note_markers": unresolved,
            "uploaded_source_replay_attempted": attempted,
            "production_state_changed": False}
