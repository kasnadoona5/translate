from __future__ import annotations

import ast
import re
from pathlib import Path

from tarjomeh.runtime import runtime_capabilities


def test_v1028_audit_scripts_embed_valid_python() -> None:
    for relative in (
        "scripts/audit_tarjomeh_v1028_reports.sh",
        "scripts/audit_tarjomeh_v1028_companion.sh",
    ):
        source = Path(relative).read_text(encoding="utf-8")
        blocks = re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY(?:\r?\n|$)", source, re.S)
        assert blocks, relative
        for block in blocks:
            ast.parse(block, filename=relative)


def test_v1028_deployment_is_scoped_and_checks_checkpoint_publish() -> None:
    deployment = Path("scripts/deploy_tarjomeh_v1028.sh").read_text(
        encoding="utf-8"
    )
    assert 'TAG="v10.28.0"' in deployment
    assert 'git diff --quiet v10.27.0 "$TAG"' in deployment
    assert "checkpoint_preview_atomic_publish" in deployment
    assert "rollback_tarjomeh_on_error" in deployment
    for identity in (
        "NINE_IMAGE",
        "NINE_STARTED",
        "NINE_CONTAINER_ID",
        "NINE_MOUNTS",
    ):
        assert identity in deployment
    for unsafe in (
        "docker system prune",
        "docker image prune",
        "docker builder prune",
        "docker rm -f 9router",
        'docker image rm "$NINE_IMAGE"',
    ):
        assert unsafe not in deployment


def test_v1028_runtime_contract_declares_checkpoint_publish() -> None:
    manifest = runtime_capabilities()
    assert manifest["release"] == "v10.34.0"
    assert manifest["capabilities"]["checkpoint_preview_atomic_publish"] is True
    assert manifest["policy_versions"]["checkpoint_export"] == 3
