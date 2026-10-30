"""Pure approval/coverage proof for operator-reviewed canonical text revisions."""

from __future__ import annotations

from copy import deepcopy

from tarjomeh.memory.summary_revision import document_hash, text_hash


def revised_coverage(coverage: list, changes: list) -> list:
    by_index = {item["chunk_index"]: item for item in changes}
    result = deepcopy(coverage)
    for ref in result:
        change = by_index.get(ref["chunk_index"])
        if change is not None:
            if (ref["source_sha256"] != text_hash(change["source"])
                    or ref["target_sha256"] != text_hash(change["before"])):
                raise ValueError("Summary coverage does not bind the prior canonical text")
            ref["target_sha256"] = text_hash(change["after"])
    return result


def validate_continuity(entry: dict, earlier: list) -> tuple[tuple, tuple]:
    """The same summary is authorized only for explicit old->new target hashes."""
    from tarjomeh.memory.summary_revision import history_key, history_policy

    record = entry["reviewed_text_continuity"]
    manifest = record["manifest"]
    digest = document_hash(manifest)
    continuity = manifest["summary_continuity"]
    old = {"summary": entry["summary"], "coverage": continuity["before"]}
    policy = history_policy(earlier)
    if (manifest.get("schema") != 1 or record.get("schema") != 1
            or record.get("revision_id") != digest or record.get("approval_sha256") != digest
            or not isinstance(record.get("reviewer"), str) or not record["reviewer"].strip()
            or document_hash(entry["summary"]) != continuity["summary_sha256"]
            or entry["summary"] != continuity["summary"]
            or entry["coverage"] != continuity["after"]
            or continuity["after"] != revised_coverage(continuity["before"], manifest["changes"])
            or continuity["before"] == continuity["after"]
            or history_key(old) in policy["blocked"] | policy["denied"]
            or not any(history_key(item) == history_key(old) for item in earlier if isinstance(item, dict))):
        raise ValueError("Invalid reviewed text coverage continuity")
    return history_key(old), history_key(entry)


def partition_revision_policy_events(events: list, chunks: list, artifacts: dict, job_id: str) -> dict:
    """Keep raw policy history, but do not apply an old score to reviewed wording.

    Only an exact approved revision chain to the current canonical source/target
    can supersede a hash-bound policy event. Unknown evidence is never excluded.
    Physical requests and all original events remain untouched.
    """
    edges = {}
    canonical = {row["chunk_index"]: row for row in chunks}
    for key, receipt in artifacts.items():
        if not key.startswith("reviewed_text_revision:") or not isinstance(receipt, dict):
            continue
        try:
            manifest = receipt["manifest"]
            digest = document_hash(manifest)
            if (key != "reviewed_text_revision:" + digest or receipt.get("revision_id") != digest
                    or receipt.get("status") != "applied" or manifest.get("job_id") != job_id
                    or not receipt.get("reviewer")):
                continue
            for change in manifest["changes"]:
                index = change["chunk_index"]
                row = canonical.get(index)
                if row and row.get("text") == change["source"]:
                    edges.setdefault(index, []).append((text_hash(change["before"]), text_hash(change["after"])))
        except (KeyError, TypeError, ValueError):
            continue
    superseded = {}
    for index, transitions in edges.items():
        current = text_hash(canonical[index]["translation"])
        reachable, old = {current}, set()
        for _ in range(len(transitions)):
            for before, after in transitions:
                if after in reachable and before != current:
                    reachable.add(before)
                    old.add(before)
        superseded[index] = old
    history, current_events = [], []
    for event in events:
        target = event.get("canonical_target_hash")
        if target and target in superseded.get(event.get("chunk_index"), set()):
            history.append(event)
        else:
            current_events.append(event)
    return {"current": current_events, "superseded_hash_bound_history": history,
            "authority": "revision_identity_only_no_score_promotion_or_call_filtering"}
