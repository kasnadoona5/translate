"""Release safety checks without contacting Docker, a VPS or an LLM."""

import ast
import gzip
import hashlib
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_new_runtime_probes_do_not_read_config_or_open_files(monkeypatch):
    from tarjomeh.runtime import _v205_behavior_probes

    def forbidden(*args, **kwargs):
        raise AssertionError("A synthetic runtime probe tried file I/O")

    monkeypatch.setattr(Path, "is_file", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    assert all(_v205_behavior_probes().values())


def _blocks(source):
    return re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY(?:\r?\n|$)", source, re.S)


@pytest.mark.parametrize(
    "name",
    [
        "deploy_tarjomeh_v205.sh",
        "audit_tarjomeh_v205_reports.sh",
        "audit_tarjomeh_v205_companion.sh",
    ],
)
def test_release_shell_and_embedded_python_are_parseable(name):
    path = ROOT / "scripts" / name
    source = path.read_text(encoding="utf-8")
    for block in _blocks(source):
        ast.parse(block)
        lint = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select", "F821,F822,F823", "-"],
            input=block,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert lint.returncode == 0, lint.stdout + lint.stderr
    for block in re.findall(r"-c '\n(.*?)\n'", source, re.S):
        ast.parse(block)
    bash = shutil.which("bash")
    assert bash, "Bash is required for release validation"
    result = subprocess.run([bash, "-n", path.as_posix()], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_deploy_checks_revision_space_and_only_tarjomeh_cleanup():
    source = (ROOT / "scripts/deploy_tarjomeh_v205.sh").read_text(encoding="utf-8")
    assert 'TAG="v20.5"' in source
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


def test_embedded_image_and_running_source_gates_pass_on_candidate():
    source = (ROOT / "scripts/deploy_tarjomeh_v205.sh").read_text(encoding="utf-8")
    blocks = re.findall(r"-c '\n(.*?)\n'", source, re.S)
    assert len(blocks) == 3
    for block in blocks:
        result = subprocess.run(
            [sys.executable, "-c", block],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=ROOT,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_companion_follows_the_shared_dash_helper_ownership():
    source = (ROOT / "scripts/audit_tarjomeh_v205_companion.sh").read_text(encoding="utf-8")
    block = next(block for block in _blocks(source) if '"typed_dash_attachment_audit"' in block)
    tree = ast.parse(block)
    expression = next(
        value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "typed_dash_attachment_audit"
    )
    namespace = {
        "pipeline_source": (ROOT / "src/tarjomeh/core/pipeline.py").read_text(encoding="utf-8"),
        "integrity_source": (ROOT / "src/tarjomeh/quality/integrity.py").read_text(
            encoding="utf-8"
        ),
    }
    assert eval(
        compile(ast.Expression(expression), "<synthetic-source-contract>", "eval"), namespace
    )


def test_audits_use_distinct_artifacts_and_do_not_hide_hard_failures():
    for kind, artifact in (("reports", "memory"), ("companion", "companion")):
        source = (ROOT / f"scripts/audit_tarjomeh_v205_{kind}.sh").read_text(encoding="utf-8")
        assert f"_v205_{artifact}_audit.json" in source
        assert f"_v205_{artifact}_audit.txt" in source
        assert 'if verdict == "FAIL":\n    raise SystemExit(2)' in source
        assert 'exit "$AUDIT_RC"' in source
        assert '"v205_remediation"' in source
        assert "requires_source_based_human_review" in source


def test_deployment_sqlite_snapshot_includes_uncheckpointed_wal(tmp_path):
    source = (ROOT / "scripts/deploy_tarjomeh_v205.sh").read_text(encoding="utf-8")
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


def test_compressed_database_is_restored_verified_and_never_overwritten(tmp_path):
    source = (ROOT / "scripts/deploy_tarjomeh_v205.sh").read_text(encoding="utf-8")
    block = next(block for block in _blocks(source) if 'destination.open("xb")' in block)
    original = tmp_path / "original.db"
    restored = tmp_path / "restored.db"
    archive = tmp_path / "original.db.gz"
    with sqlite3.connect(original) as database:
        database.execute("CREATE TABLE evidence (value TEXT)")
        database.execute("INSERT INTO evidence VALUES ('consistent snapshot')")
    with gzip.open(archive, "wb") as compressed:
        compressed.write(original.read_bytes())
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    command = [sys.executable, "-c", block, str(archive), str(restored), digest]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert restored.read_bytes() == original.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0 and restored.read_bytes() == original.read_bytes()
    restored.unlink()
    corrupted_hash = subprocess.run(command[:-1] + ["0" * 64], capture_output=True, text=True)
    assert corrupted_hash.returncode != 0


def test_deploy_checks_worker_leases_before_backup_and_before_replacement():
    source = (ROOT / "scripts/deploy_tarjomeh_v205.sh").read_text(encoding="utf-8")
    blocks = [block for block in _blocks(source) if "state IN ('active','pausing')" in block]
    assert len(blocks) == 2
    assert source.index(blocks[0]) < source.index("source.backup(snapshot)")
    assert source.index(blocks[1]) > source.index("IMAGE_V205_CONFIRMED")
