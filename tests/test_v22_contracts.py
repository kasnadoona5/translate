"""v22 adds evidence and deterministic fixes, not weaker admission policies."""

import json
import re
import socket
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from tarjomeh.core.evidence_audit import narrow_release_reporting
from tarjomeh.runtime import _v22_behavior_probes, runtime_behavior_probes, runtime_capabilities

ROOT = Path(__file__).resolve().parents[1]
CAPABILITIES = (
    "empty_memory_pair_retention", "docx_heading_font_inheritance",
    "localized_number_evaluation", "score_grounding_diagnostics_only",
    "source_scoped_admission_reporting",
)


def test_pure_v22_probes_need_no_network_or_persistent_state(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("probe performed I/O")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    probes = _v22_behavior_probes()
    assert set(probes) == set(CAPABILITIES) and all(probes.values())
    manifest = runtime_capabilities()
    assert manifest["release"] == "v22.6"
    assert all(manifest["capabilities"][key] for key in CAPABILITIES)


def test_probe_rejects_broken_heading_size_policy(monkeypatch):
    import tarjomeh.exporters.docx_exporter as exporter

    monkeypatch.setattr(exporter, "target_run_font_size", lambda name: 11)
    assert not _v22_behavior_probes()["docx_heading_font_inheritance"]


def test_missing_historical_diagnostics_are_unavailable_not_a_gate():
    events = [
        {"chunk_index": 0, "event_type": "llm_call_attempt", "payload": {"success": True}},
        {"chunk_index": 0, "event_type": "critique_completed", "payload": {}},
        {"chunk_index": 0, "event_type": "critique_completed", "payload": {}},
        {"chunk_index": 0, "event_type": "refinement_completed",
         "payload": {"issue_decisions": [{"issue_id": "old"}]}},
    ]
    before = deepcopy(events)
    result = narrow_release_reporting(events, [])
    assert result["physical_attempt_records"] == 1
    assert result["critique_result_records"] == 2
    assert all(row["availability"] == "unavailable" for row in result["critique_diagnostics"])
    assert result["local_term_admissions"][0]["availability"] == "unavailable_or_not_reached"
    assert events == before and "accepted" not in result
    assert result["authority"] == "report_only_no_gate_score_trust_or_call_change"


def _canonical_router(metadata):
    script = (ROOT / "scripts/deploy_tarjomeh_v22.sh").read_text(encoding="utf-8")
    block = next(block for block in re.findall(r"-c '\n(.*?)\n'", script, re.S)
                 if "rows = json.load(sys.stdin)" in block)
    result = subprocess.run(
        [sys.executable, "-c", block], input=json.dumps(metadata),
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.parametrize("metadata", [[], [{"Id": "one"}], [{"Mounts": None}]])
def test_incomplete_router_inspection_cannot_pass(metadata):
    with pytest.raises(AssertionError):
        _canonical_router(metadata)


def _router():
    return {"Id": "container-one", "Image": "sha256:image-one",
            "State": {"StartedAt": "2026-01-01T00:00:00Z"},
            "Mounts": [
                {"Type": "bind", "Source": "/data", "Destination": "/app/data", "RW": True},
                {"Type": "bind", "Source": "/config", "Destination": "/app/config", "RW": False},
            ]}


def test_same_mounts_reordered_do_not_claim_a_runtime_change():
    before = _router()
    after = deepcopy(before)
    after["Mounts"] = [dict(reversed(list(item.items())))
                       for item in reversed(after["Mounts"])]
    assert _canonical_router([before]) == _canonical_router([after])


@pytest.mark.parametrize("change", ["Id", "Image", "StartedAt", "Source", "Destination", "RW"])
def test_real_router_changes_still_fail_identity_comparison(change):
    before = _router()
    after = deepcopy(before)
    if change in {"Id", "Image"}:
        after[change] = "changed"
    elif change == "StartedAt":
        after["State"][change] = "2026-02-01T00:00:00Z"
    else:
        after["Mounts"][0][change] = False if change == "RW" else "/different"
    assert _canonical_router([before]) != _canonical_router([after])


def test_router_evidence_and_update_lock_are_kept():
    source = (ROOT / "scripts/deploy_tarjomeh_v22.sh").read_text(encoding="utf-8")
    assert "flock -n 8" in source and "/run/lock/9router-updater.lock" in source
    assert "9router.before.json" in source and "9router.after.$stage.json" in source
    assert "9ROUTER CHANGED FIELD:" in source
    assert source.count("verify_router_unchanged ") == 3
    assert "sorted(mounts" in source
    assert '"$(docker inspect -f \'{{json .Mounts}}\' 9router)" = ' not in source


def test_all_previous_runtime_probes_still_pass():
    probes = runtime_behavior_probes()
    assert len(probes) == 164 and all(probes.values())
