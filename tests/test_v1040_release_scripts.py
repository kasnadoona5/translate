"""Static release gates for the opt-in term-review candidate."""

import ast
from pathlib import Path
import re

from tarjomeh.runtime import runtime_behavior_probes, runtime_capabilities


ROOT = Path(__file__).resolve().parents[1]


def test_v1040_audits_parse_and_fail_closed() -> None:
    for name in ("companion", "reports"):
        path = ROOT / "scripts" / f"audit_tarjomeh_v1040_{name}.sh"
        source = path.read_text(encoding="utf-8")
        blocks = re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY(?:\r?\n|$)", source, re.S)
        assert blocks, path
        for block in blocks:
            ast.parse(block, filename=str(path))
        assert 'if verdict == "FAIL":\n    raise SystemExit(2)' in source
        assert 'exit "$AUDIT_RC"' in source
        assert "v10.40" in source
    companion = (ROOT / "scripts" / "audit_tarjomeh_v1040_companion.sh").read_text(
        encoding="utf-8"
    )
    assert "finished_opt_in_job_missing_book_term_review" in companion
    assert "awaiting_user_book_term_decision" in companion


def test_v1040_deploy_keeps_9router_out_of_cleanup() -> None:
    path = ROOT / "scripts" / "deploy_tarjomeh_v1040.sh"
    source = path.read_text(encoding="utf-8")
    assert 'TAG="v10.40.1"' in source
    assert "--no-deps" in source
    assert "less than 350 MB" in source
    assert "docker image prune" not in source
    assert "docker system prune" not in source
    for marker in ("NINE_CONTAINER_ID", "NINE_IMAGE", "NINE_STARTED", "NINE_MOUNTS"):
        assert source.count(marker) >= 2
    for block in re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY(?:\r?\n|$)", source, re.S):
        ast.parse(block, filename=str(path))
    for block in re.findall(r"python -c '\n(.*?)\n'", source, re.S):
        ast.parse(block, filename=str(path))


def test_v1040_runtime_reports_pure_behavior_guards() -> None:
    manifest = runtime_capabilities()
    probes = runtime_behavior_probes()
    assert manifest["release"] == "v20.6"
    for name in (
        "opt_in_book_term_review_is_off_by_default",
        "reviewed_terms_are_paragraph_scoped",
        "source_proven_surface_repair_is_bounded",
    ):
        assert probes[name] is True
