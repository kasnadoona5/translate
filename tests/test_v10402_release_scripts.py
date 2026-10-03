"""Static and pure runtime gates for the v10.40.2 identifier release."""

import ast
from pathlib import Path
import re

from tarjomeh.runtime import runtime_behavior_probes, runtime_capabilities


ROOT = Path(__file__).resolve().parents[1]


def test_v10402_audits_parse_and_fail_closed() -> None:
    for name in ("companion", "reports"):
        path = ROOT / "scripts" / f"audit_tarjomeh_v10402_{name}.sh"
        source = path.read_text(encoding="utf-8")
        blocks = re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY(?:\r?\n|$)", source, re.S)
        assert blocks, path
        for block in blocks:
            ast.parse(block, filename=str(path))
        assert 'if verdict == "FAIL":\n    raise SystemExit(2)' in source
        assert 'exit "$AUDIT_RC"' in source
        assert "v10402" in source
        assert "v10.40.2" in source or name == "reports"


def test_v10402_deploy_is_tarjomeh_only() -> None:
    path = ROOT / "scripts" / "deploy_tarjomeh_v10402.sh"
    source = path.read_text(encoding="utf-8")
    assert 'TAG="v10.40.2"' in source
    assert 'manifest["release"] == "v10.40.2"' in source
    assert "--no-deps" in source
    assert "docker image prune" not in source
    assert "docker system prune" not in source
    for marker in ("NINE_CONTAINER_ID", "NINE_IMAGE", "NINE_STARTED", "NINE_MOUNTS"):
        assert source.count(marker) >= 2
    for block in re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY(?:\r?\n|$)", source, re.S):
        ast.parse(block, filename=str(path))
    for block in re.findall(r"python -c '\n(.*?)\n'", source, re.S):
        ast.parse(block, filename=str(path))


def test_v10402_runtime_identifier_guards_are_pure_and_enabled() -> None:
    assert runtime_capabilities()["release"] == "v20.3"
    probes = runtime_behavior_probes()
    for name in (
        "repeated_isbn_labels_use_target_evidence",
        "unproven_repeated_isbn_labels_remain_unresolved",
        "persian_book_number_never_becomes_issn",
    ):
        assert probes[name] is True
