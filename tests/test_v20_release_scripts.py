"""Static gates for the literal v20 deployment and audit entry points."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from tarjomeh.runtime import runtime_behavior_probes, runtime_capabilities

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _python_blocks(source: str) -> list[str]:
    return re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY(?:\r?\n|$)", source, re.S)


def test_v20_deploy_backs_up_sqlite_and_preserves_9router() -> None:
    source = (SCRIPTS / "deploy_tarjomeh_v20.sh").read_text(encoding="utf-8")
    assert 'TAG="v20"' in source
    assert 'manifest["release"] == "v20"' in source
    assert "source.backup(snapshot)" in source
    assert 'snapshot.execute("PRAGMA integrity_check")' in source
    assert 'gzip -t "$BACKUP/jobs.db.gz"' in source
    assert 'if [ "$BACKUP_FREE" -lt $((2 * DB_SIZE + 350000000)) ]' in source
    assert 'if [ "$AVAILABLE_KB" -lt 350000 ]' in source
    assert "--no-deps" in source
    assert "docker image prune" not in source
    assert "docker system prune" not in source
    for marker in ("NINE_CONTAINER_ID", "NINE_IMAGE", "NINE_STARTED", "NINE_MOUNTS"):
        assert source.count(marker) >= 2
    for capability in ("selected_chapter_term_index", "bilingual_docx_identity"):
        assert f'"{capability}"' in source
    for block in _python_blocks(source):
        ast.parse(block)


def test_v20_audits_are_parseable_and_fail_closed() -> None:
    for name in ("companion", "reports"):
        source = (SCRIPTS / f"audit_tarjomeh_v20_{name}.sh").read_text(
            encoding="utf-8"
        )
        assert "v20" in source
        assert 'if verdict == "FAIL":\n    raise SystemExit(2)' in source
        assert 'exit "$AUDIT_RC"' in source
        for block in _python_blocks(source):
            ast.parse(block)
    companion = (SCRIPTS / "audit_tarjomeh_v20_companion.sh").read_text(
        encoding="utf-8"
    )
    assert 'hard.append("render_identity_failed")' in companion
    assert '"delivered_verified": render_delivered.get("verified")' in companion
    assert '"runtime_release_contract": runtime_manifest.get("release") == "v20"' in companion


def test_v20_runtime_capabilities_and_pure_probes() -> None:
    manifest = runtime_capabilities()
    assert manifest["release"] == "v20"
    assert manifest["capabilities"]["selected_chapter_term_index"]
    assert manifest["capabilities"]["bilingual_docx_identity"]
    assert manifest["policy_versions"]["book_term_scope"] == 4
    assert manifest["policy_versions"]["render_identity"] == 2
    probes = runtime_behavior_probes()
    assert probes["selected_chapter_body_terms_keep_local_indices"]
    assert probes["bilingual_docx_layouts_have_expected_text_units"]
    assert all(probes.values())
