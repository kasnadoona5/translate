"""Current release safety checks; historical tagged scripts remain immutable."""

import ast
import gzip
import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORRECTIVE_CAPABILITIES = (
    "numeral_boundary_preservation", "source_metalinguistic_token_protection",
    "published_output_coverage",
    "cumulative_summary_input_trust",
    "compound_chapter_count_evidence",
    "arabic_indic_chapter_references",
    "cross_segment_quote_grounding",
    "offline_grouped_integer_evaluation",
    "empty_memory_pair_retention",
    "docx_heading_font_inheritance",
    "localized_number_evaluation",
    "score_grounding_diagnostics_only",
    "source_scoped_admission_reporting",
    "atomic_summary_commit", "cumulative_summary_input_trust", "reviewed_summary_revision",
    "numeric_announcement_boundaries",
    "final_entity_local_issue_scope",
    "final_candidate_entity_memory",
    "first_occurrence_evidence_classification",
    "shared_original_keeper_eligibility",
    "document_scoped_duplicate_proof",
    "closing_parenthesis_list_evidence",
    "paired_object_marker_dash_classification",
    "inflected_multiword_anchor_proof",
    "organization_lexical_boundary",
    "book_identity_date_guard",
    "source_eligible_style_memory",
    "historical_style_source_quarantine",
    "order_independent_readability_evidence",
    "identical_term_proposal_dedup",
    "fresh_term_scope_default",
    "persian_plus_scoped_original",
    "pending_only_term_skip",
    "curated_revision_authority",
    "prior_only_retranslation_context",
    "active_memory_version_replacement",
    "pending_canonical_summary_coverage",
    "paragraph_role_style_evidence",
    "raw_token_counter_diagnostics",
    "actual_ignored_policy_diagnostics",
    "persisted_job_policy_isolation",
    "memory_category_and_conflict_authority",
    "accepted_entity_not_lexical_plural",
    "accepted_only_surface_salvage",
    "shared_raw_paragraph_admission",
    "canonical_quote_diff_binding",
    "matched_readability_advisory_routing",
    "selected_body_proposal_evidence",
    "paragraph_term_candidate_admission",
    "summary_exact_section_headers",
    "explicit_hyphen_name_group_alignment",
    "final_heh_ezafe_name_edges",
    "source_local_annotation_syntax",
    "shared_source_entity_inventory_contract",
    "lexical_annotation_memory_deferral",
    "complete_terminology_reconciliation_events",
    "optional_book_term_recommendations",
    "advisory_term_observations_nonblocking",
    "report_only_book_term_presence",
    "publisher_before_date_research_guard",
    "r109_serious_unchanged_count_preserved",
    "bounded_name_suffix_components",
    "conservative_automatic_name_ambiguity",
    "quoted_suffix_source_scope",
)


@pytest.mark.parametrize("kind", ["reports", "companion"])
def test_new_collectors_label_current_evidence_and_publication_scope(kind):
    source = (ROOT / f"scripts/audit_tarjomeh_v225_{kind}.sh").read_text(encoding="utf-8")
    assert "Tarjomeh v22.5" in source
    assert "V22.2" not in source
    assert '"output_coverage": publication_coverage' in source
    assert '"source_metalinguistic_evidence": metalinguistic_reporting' in source
    assert "Current-policy source-authored etymons (not memory approval)" in source
    assert 'str(chunk.get("text") or "")' in source


@pytest.mark.parametrize("kind", ["reports", "companion"])
def test_summary_lineage_reporting_is_read_only_and_honest_about_legacy(kind):
    source = (ROOT / f"scripts/audit_tarjomeh_v225_{kind}.sh").read_text(encoding="utf-8")
    block = next(block for block in _blocks(source) if "summary_provenance_reporting =" in block)
    tree = ast.parse(block)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "summary_provenance_reporting"
                              for t in node.targets))
    state = {"bilingual_summary": {"input_trust": "advisory_inputs", "trust_reasons": ["old"]}}
    payload = {"input_trust": "advisory_inputs", "candidate_committed": True,
               "candidate_quality": {"input_trust_lineage": {"inherited_advisory": True}}}
    events = [{"event_type": "bilingual_summary_memory_policy", "payload": p}
              for p in (payload, {"candidate_committed": False})]
    before = json.dumps([state, events], sort_keys=True)
    scope = {"state": state, "active_events": events}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "reporting", "exec"), scope)
    report = scope["summary_provenance_reporting"]
    assert report["updates"][0]["lineage"] == {"inherited_advisory": True}
    assert report["updates"][1]["lineage"] is None
    assert report["active_input_trust"] == "advisory_inputs"
    assert report["historical_records_relabelled"] is False
    assert report["authority"] == "input_provenance_not_semantic_verification"
    assert json.dumps([state, events], sort_keys=True) == before


def test_new_runtime_probes_do_not_read_config_or_open_files(monkeypatch):
    from tarjomeh.runtime import _v20810_behavior_probes

    def forbidden(*args, **kwargs):
        raise AssertionError("A synthetic runtime probe tried file I/O")

    monkeypatch.setattr(Path, "is_file", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    assert all(_v20810_behavior_probes().values())


def test_entity_probe_cannot_pass_if_producer_silently_does_nothing(monkeypatch):
    import tarjomeh.memory.final_entities as entities
    from tarjomeh.runtime import _v20810_behavior_probes

    monkeypatch.setattr(entities, "reconcile_final_entities", lambda *a, **k: {"records": []})
    assert not _v20810_behavior_probes()["final_candidate_entity_memory"]


@pytest.mark.parametrize("missing", CORRECTIVE_CAPABILITIES)
def test_deploy_cannot_pass_with_a_missing_corrective_capability(monkeypatch, missing):
    import tarjomeh.runtime as runtime

    manifest = runtime.runtime_capabilities()
    del manifest["capabilities"][missing]
    monkeypatch.setattr(runtime, "runtime_capabilities", lambda: manifest)
    source = (ROOT / "scripts/deploy_tarjomeh_v225.sh").read_text(encoding="utf-8")
    block = next(
        block for block in re.findall(r"-c '\n(.*?)\n'", source, re.S) if "required = (" in block
    )
    with pytest.raises(AssertionError):
        exec(compile(block, "<synthetic-deploy-contract>", "exec"), {})


def test_each_collector_requires_corrective_capabilities_and_probes():
    from tarjomeh.runtime import runtime_behavior_probes, runtime_capabilities

    namespace = {
        "runtime_manifest": runtime_capabilities(),
        "runtime_probes": runtime_behavior_probes(),
    }
    for kind in ("reports", "companion"):
        source = (ROOT / f"scripts/audit_tarjomeh_v225_{kind}.sh").read_text(encoding="utf-8")
        block = next(
            block for block in _blocks(source) if "v225_targeted_runtime_contract" in block
        )
        tree = ast.parse(block)
        if kind == "reports":
            expression = next(
                node.test.operand
                for node in ast.walk(tree)
                if isinstance(node, ast.If)
                and isinstance(node.test, ast.UnaryOp)
                and any(
                    isinstance(child, ast.Constant)
                    and child.value == "v225_targeted_runtime_contract_failed"
                    for child in ast.walk(node)
                )
            )
        else:
            expression = next(
                value
                for node in ast.walk(tree)
                if isinstance(node, ast.Dict)
                for key, value in zip(node.keys, node.values, strict=True)
                if isinstance(key, ast.Constant) and key.value == "v225_targeted_runtime_contract"
            )
        compiled = compile(ast.Expression(expression), "<synthetic-audit-contract>", "eval")
        assert eval(compiled, namespace)
        for missing in CORRECTIVE_CAPABILITIES:
            for dictionary in ("runtime_probes",):
                namespace[dictionary][missing] = False
                assert not eval(compiled, namespace)
                namespace[dictionary][missing] = True
            namespace["runtime_manifest"]["capabilities"][missing] = False
            assert not eval(compiled, namespace)
            namespace["runtime_manifest"]["capabilities"][missing] = True


def _blocks(source):
    return re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY(?:\r?\n|$)", source, re.S)


@pytest.mark.parametrize(
    "name",
    [
        "deploy_tarjomeh_v225.sh",
        "audit_tarjomeh_v225_reports.sh",
        "audit_tarjomeh_v225_companion.sh",
        "run_tarjomeh_v225_audits.sh",
        "revise_tarjomeh_summary.sh",
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
    source = (ROOT / "scripts/deploy_tarjomeh_v225.sh").read_text(encoding="utf-8")
    assert 'TAG="v22.5"' in source
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
    assert "VERIFY HOST JOBS BACKUP SOURCE" in source
    for marker in ("NINE_CONTAINER_ID", "NINE_IMAGE", "NINE_STARTED", "NINE_STATE"):
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


@pytest.mark.parametrize(
    "mounts,accepted",
    [
        ([{"Type": "bind", "Source": "/opt/translate/jobs", "Destination": "/app/jobs"}], True),
        ([{"Type": "bind", "Source": "/another/jobs", "Destination": "/app/jobs"}], False),
        ([], False),
        (
            [
                {"Type": "bind", "Source": "/opt/translate/jobs", "Destination": "/app/jobs"},
                {"Type": "bind", "Source": "/another/jobs.db", "Destination": "/app/jobs/jobs.db"},
            ],
            False,
        ),
    ],
)
def test_host_backup_cannot_silently_copy_a_different_database(mounts, accepted):
    source = (ROOT / "scripts/deploy_tarjomeh_v225.sh").read_text(encoding="utf-8")
    block = next(
        block
        for block in re.findall(r"-c '\n(.*?)\n'", source, re.S)
        if "mounts = json.load(sys.stdin)" in block
    )
    result = subprocess.run(
        [sys.executable, "-c", block],
        input=json.dumps(mounts),
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is accepted


@pytest.mark.parametrize("reports,companion,expected", [(0, 0, 0), (2, 0, 2), (0, 2, 2), (2, 3, 3)])
def test_combined_audit_runs_both_and_never_hides_a_failure(reports, companion, expected):
    source = (ROOT / "scripts/run_tarjomeh_v225_audits.sh").read_text(encoding="utf-8")
    source = source.replace("cd /opt/translate", "cd .")
    stub = f"""bash() {{
        case "$1" in
            *reports.sh) echo REPORTS_RAN; return {reports} ;;
            *companion.sh) echo COMPANION_RAN; return {companion} ;;
            *) return 99 ;;
        esac
    }}
    """
    result = subprocess.run(
        [shutil.which("bash"), "-c", stub + source, "audit-wrapper-test", "synthetic-job"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == expected, result.stderr
    assert "REPORTS_RAN" in result.stdout and "COMPANION_RAN" in result.stdout


def test_embedded_image_and_running_source_gates_pass_on_candidate():
    from tarjomeh.runtime import RUNTIME_RELEASE

    source = (ROOT / "scripts/deploy_tarjomeh_v225.sh").read_text(encoding="utf-8")
    blocks = re.findall(r"-c '\n(.*?)\n'", source, re.S)
    assert len(blocks) == 5
    for block in blocks:
        if "mounts = json.load(sys.stdin)" in block or "rows = json.load(sys.stdin)" in block:
            continue
        block = block.replace('== "v22.5"', f'== "{RUNTIME_RELEASE}"')
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
    source = (ROOT / "scripts/audit_tarjomeh_v225_companion.sh").read_text(encoding="utf-8")
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
        source = (ROOT / f"scripts/audit_tarjomeh_v225_{kind}.sh").read_text(encoding="utf-8")
        assert f"_v225_{artifact}_audit.json" in source
        assert f"_v225_{artifact}_audit.txt" in source
        assert 'if verdict == "FAIL":\n    raise SystemExit(2)' in source
        assert 'exit "$AUDIT_RC"' in source
        assert '"v225_remediation"' in source
        assert "requires_source_based_human_review" in source
        assert '"v225_targeted_evidence": quality_v225' in source
        assert 'quality_v225["deferred_name_mappings"]' in source
        assert 'quality_v225["extra_final_counts"]' in source
        assert 'quality_v225["quoted_suffix_reports"]' in source


def test_deployment_sqlite_snapshot_includes_uncheckpointed_wal(tmp_path):
    source = (ROOT / "scripts/deploy_tarjomeh_v225.sh").read_text(encoding="utf-8")
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
    source = (ROOT / "scripts/deploy_tarjomeh_v225.sh").read_text(encoding="utf-8")
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
    source = (ROOT / "scripts/deploy_tarjomeh_v225.sh").read_text(encoding="utf-8")
    blocks = [block for block in _blocks(source) if "state IN ('active','pausing')" in block]
    assert len(blocks) == 2
    assert source.index(blocks[0]) < source.index("source.backup(snapshot)")
    assert source.index(blocks[1]) > source.index("IMAGE_V225_CONFIRMED")
