"""Static and pure runtime gates for the v10.40.3 release."""

import ast
import re
from pathlib import Path

from tarjomeh.runtime import runtime_behavior_probes, runtime_capabilities

ROOT = Path(__file__).resolve().parents[1]
NEW_CAPABILITIES = (
    "complete_style_pairs",
    "per_paragraph_precanonical_affix_repair",
    "render_identity_audit",
    "reviewed_book_term_occurrences",
    "job_scoped_book_term_add",
    "imprint_name_grounding",
    "malformed_persian_option_screening",
    "prompt_duplication_measurement",
)


def _python_blocks(source: str) -> list[str]:
    return re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY(?:\r?\n|$)", source, re.S)


def test_v10403_audits_parse_fail_closed_and_gate_render_identity() -> None:
    for name in ("companion", "reports"):
        path = ROOT / "scripts" / f"audit_tarjomeh_v10403_{name}.sh"
        source = path.read_text(encoding="utf-8")
        blocks = _python_blocks(source)
        assert blocks, path
        for block in blocks:
            ast.parse(block, filename=str(path))
        assert 'if verdict == "FAIL":\n    raise SystemExit(2)' in source
        assert 'exit "$AUDIT_RC"' in source
        assert "v10403" in source and "v10402" not in source
    companion = (ROOT / "scripts" / "audit_tarjomeh_v10403_companion.sh").read_text(
        encoding="utf-8"
    )
    assert 'hard.append("render_identity_failed")' in companion
    assert '"v10403_evidence": v10403_evidence' in companion
    assert '"runtime_release_contract": runtime_manifest.get("release") == "v10.40.3"' in companion


def test_v10403_deploy_is_tarjomeh_only_and_disk_guarded() -> None:
    path = ROOT / "scripts" / "deploy_tarjomeh_v10403.sh"
    source = path.read_text(encoding="utf-8")
    assert 'TAG="v10.40.3"' in source
    assert 'manifest["release"] == "v10.40.3"' in source
    assert "--no-deps" in source
    assert "docker image prune" not in source
    assert "docker system prune" not in source
    assert 'if [ "$AVAILABLE_KB" -lt 350000 ]' in source
    for marker in ("NINE_CONTAINER_ID", "NINE_IMAGE", "NINE_STARTED", "NINE_MOUNTS"):
        assert source.count(marker) >= 2
    for name in NEW_CAPABILITIES:
        assert f'"{name}"' in source
    for block in _python_blocks(source):
        ast.parse(block, filename=str(path))
    for block in re.findall(r"python -c '\n(.*?)\n'", source, re.S):
        ast.parse(block, filename=str(path))


def test_v10403_runtime_capabilities_and_probes_are_pure_and_enabled() -> None:
    manifest = runtime_capabilities()
    assert manifest["release"] == "v20.5"
    for name in NEW_CAPABILITIES:
        assert manifest["capabilities"][name] is True
    assert manifest["policy_versions"]["render_identity"] == 2
    probes = runtime_behavior_probes()
    for name in (
        "complete_style_pair_is_stored_whole",
        "clipped_legacy_style_is_quarantined",
        "prefix_repaired_per_paragraph_before_canonical",
        "joined_plural_is_not_a_spaced_prefix",
        "render_citation_merge_is_replay_proven",
        "render_change_without_producer_blocks",
        "partial_occurrence_ticks_are_not_prompted",
        "repeated_term_occurrences_stay_review",
        "imprint_names_are_source_grounded",
        "malformed_persian_option_is_withheld",
    ):
        assert probes[name] is True, name
    assert all(probes.values())
