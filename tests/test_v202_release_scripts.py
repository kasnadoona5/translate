"""Release safety checks without contacting Docker, a VPS or an LLM."""

import ast
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _blocks(source):
    return re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY(?:\r?\n|$)", source, re.S)


@pytest.mark.parametrize(
    "name",
    [
        "deploy_tarjomeh_v202.sh",
        "audit_tarjomeh_v202_reports.sh",
        "audit_tarjomeh_v202_companion.sh",
    ],
)
def test_release_shell_and_embedded_python_are_parseable(name):
    path = ROOT / "scripts" / name
    source = path.read_text(encoding="utf-8")
    for block in _blocks(source):
        ast.parse(block)
    for block in re.findall(r"-c '\n(.*?)\n'", source, re.S):
        ast.parse(block)
    bash = shutil.which("bash")
    assert bash, "Bash is required for release validation"
    result = subprocess.run([bash, "-n", path.as_posix()], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_deploy_checks_revision_space_and_only_tarjomeh_cleanup():
    source = (ROOT / "scripts/deploy_tarjomeh_v202.sh").read_text(encoding="utf-8")
    assert 'TAG="v20.2"' in source
    assert "flock -n 9" in source
    assert "source.backup(snapshot)" in source
    assert 'snapshot.execute("PRAGMA integrity_check")' in source
    assert 'gzip -t "$BACKUP/jobs.db.gz"' in source
    assert source.index("VERIFIED TARJOMEH-ONLY PRE-BACKUP CLEANUP") < source.index(
        "MIN_BACKUP_FREE="
    )
    assert '--label "org.opencontainers.image.revision=$(git rev-parse HEAD)"' in source
    assert "index .Config.Labels" in source
    assert "container_references=" in source
    assert "--no-deps" in source
    assert "STOPPED: a job became active during build." in source
    assert "350000000" in source
    for marker in ("NINE_CONTAINER_ID", "NINE_IMAGE", "NINE_STARTED", "NINE_MOUNTS"):
        assert source.count(marker) >= 3
    for forbidden in (
        "docker system prune",
        "docker image prune",
        "docker volume prune",
        "apt-get clean",
        "journalctl --vacuum",
        "docker rm 9router",
        "docker stop 9router",
    ):
        assert forbidden not in source


def test_audits_use_distinct_artifacts_and_do_not_hide_hard_failures():
    for kind, artifact in (("reports", "memory"), ("companion", "companion")):
        source = (ROOT / f"scripts/audit_tarjomeh_v202_{kind}.sh").read_text(encoding="utf-8")
        assert f"_v202_{artifact}_audit.json" in source
        assert f"_v202_{artifact}_audit.txt" in source
        assert 'if verdict == "FAIL":\n    raise SystemExit(2)' in source
        assert 'exit "$AUDIT_RC"' in source
        assert '"v202_remediation"' in source
        assert "requires_source_based_human_review" in source


def test_deployment_sqlite_snapshot_includes_uncheckpointed_wal(tmp_path):
    source = (ROOT / "scripts/deploy_tarjomeh_v202.sh").read_text(encoding="utf-8")
    block = next(block for block in _blocks(source) if "source.backup(snapshot)" in block)
    database = tmp_path / "live.db"
    snapshot = tmp_path / "snapshot.db"
    writer = sqlite3.connect(database)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE evidence (value TEXT)")
        writer.execute("INSERT INTO evidence VALUES ('committed in WAL')")
        writer.commit()
        assert Path(str(database) + "-wal").stat().st_size > 0
        result = subprocess.run(
            [sys.executable, "-c", block, str(database), str(snapshot)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        with sqlite3.connect(snapshot) as copied:
            assert copied.execute("SELECT value FROM evidence").fetchall() == [
                ("committed in WAL",)
            ]
            assert copied.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    finally:
        writer.close()
