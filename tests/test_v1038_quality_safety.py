from __future__ import annotations

import json
import importlib.util
import sqlite3
from types import SimpleNamespace
from pathlib import Path

from tarjomeh.core.pipeline import (
    _candidate_regression_details,
    audit_translation_language,
)
from tarjomeh.quality.critique import TranslationCritique
from tarjomeh.quality.integrity import (
    repair_source_grounded_language_artifacts,
)


def _finding(*, paragraph: int = 2, rationale: str = "scope") -> dict[str, object]:
    return {
        "issue_id": "scope-error",
        "category": "accuracy",
        "severity": "major",
        "confidence": 0.8,
        "source_segment_id": f"p{paragraph}:s1",
        "source_quote": "ideational institutionalism",
        "current_persian_quote": "\u0646\u0647\u0627\u062f\u06af\u0631\u0627\u06cc\u06cc\u200c\u0647\u0627\u06cc \u0627\u0646\u062f\u06cc\u0634\u0647\u200c\u0627\u06cc",
        "rationale": rationale,
    }


def _candidate(finding: dict[str, object], *, candidate: str, source: str,
               previous: str) -> tuple[list[dict], list[dict]]:
    old: list[dict] = []
    regressions = _candidate_regression_details(
        SimpleNamespace(issue_details=[finding]),
        SimpleNamespace(issue_details=[]),
        ["\u0631\u0627 \u0645\u06cc\u200c\u0633\u0627\u0632\u0646\u062f"],
        source_text=source,
        previous_text=previous,
        candidate_text=candidate,
        newly_observed_unchanged=old,
    )
    return regressions, old


def test_newly_observed_issue_in_unchanged_paragraph_does_not_veto_edit() -> None:
    source = "They make history.\n\nOnly ideational institutionalism has this name."
    previous = (
        "\u062a\u0627\u0631\u06cc\u062e \u0631\u0627 \u2014 \u0645\u06cc\u200c\u0633\u0627\u0632\u0646\u062f.\n\n"
        "\u0646\u0647\u0627\u062f\u06af\u0631\u0627\u06cc\u06cc\u200c\u0647\u0627\u06cc \u0627\u0646\u062f\u06cc\u0634\u0647\u200c\u0627\u06cc \u0647\u0633\u062a\u0646\u062f."
    )
    candidate = previous.replace("\u0631\u0627 \u2014 \u0645\u06cc", "\u0631\u0627 \u0645\u06cc")
    regressions, old = _candidate(
        _finding(), candidate=candidate, source=source, previous=previous
    )
    assert not regressions
    assert [item["issue_id"] for item in old] == ["scope-error"]


def test_uncertain_or_cross_paragraph_issue_blocks_edit() -> None:
    source = "They make history.\n\nOnly ideational institutionalism has this name."
    previous = "\u0631\u0627 \u2014 \u0645\u06cc\u200c\u0633\u0627\u0632\u0646\u062f.\n\n\u0646\u0647\u0627\u062f\u06af\u0631\u0627\u06cc\u06cc\u200c\u0647\u0627\u06cc \u0627\u0646\u062f\u06cc\u0634\u0647\u200c\u0627\u06cc."
    candidate = previous.replace("\u0631\u0627 \u2014 \u0645\u06cc", "\u0631\u0627 \u0645\u06cc")
    cross = _finding(rationale="The antecedent is in the previous paragraph")
    regressions, old = _candidate(
        cross, candidate=candidate, source=source, previous=previous
    )
    assert len(regressions) == 1 and not old
    uncertain = _finding(paragraph=1)
    regressions, old = _candidate(
        uncertain, candidate=candidate, source=source, previous=previous
    )
    assert len(regressions) == 1 and not old


def test_malformed_readability_sibling_keeps_valid_item_and_errors() -> None:
    target = "\u062c\u0645\u0644\u0647 \u0646\u0627\u0631\u0633\u0627\u0633\u062a."
    raw = json.dumps({"issues": [
        {"severity": "minor", "current_persian_quote": target,
         "suggested_correction": "\u062c\u0645\u0644\u0647 \u0631\u0648\u0634\u0646 \u0627\u0633\u062a.",
         "rationale": "The predicate is awkward."},
        {"severity": "minor", "current_persian_quote": "not in target",
         "suggested_correction": "anything", "rationale": "Not grounded."},
    ]})
    result = TranslationCritique._parse_readability_response(raw, target)
    assert not result.valid
    assert len(result.issues) == 1
    assert any("not_found" in error for error in result.validation_errors)


def test_focused_attachment_requires_both_grounded_dependencies() -> None:
    source = "Only ideational institutionalism has this name."
    target = "\u0646\u0647\u0627\u062f\u06af\u0631\u0627\u06cc\u06cc \u0627\u0646\u062f\u06cc\u0634\u0647\u200c\u0627\u06cc \u0646\u0627\u0645 \u062f\u0627\u0631\u062f."
    issue = {
        "category": "accuracy", "severity": "major", "confidence": 0.8,
        "source_segment_id": "p1:s1",
        "source_quote": "ideational institutionalism",
        "current_persian_quote": "\u0646\u0647\u0627\u062f\u06af\u0631\u0627\u06cc\u06cc \u0627\u0646\u062f\u06cc\u0634\u0647\u200c\u0627\u06cc",
        "source_head": "institutionalism", "source_dependent": "ideational",
        "persian_head": "\u0646\u0647\u0627\u062f\u06af\u0631\u0627\u06cc\u06cc",
        "persian_dependent": "\u0627\u0646\u062f\u06cc\u0634\u0647\u200c\u0627\u06cc",
        "suggested_correction": "\u0646\u0647\u0627\u062f\u06af\u0631\u0627\u06cc\u06cc \u0627\u0646\u062f\u06cc\u0634\u0647\u200c\u0627\u06cc",
        "rationale": "The relative clause has the wrong scope.",
    }
    parsed = TranslationCritique._parse_attachment_response(
        json.dumps({"issues": [issue]}), source, target
    )
    assert parsed.valid
    assert len(parsed.issues) == 1
    issue["source_head"] = "state"
    rejected = TranslationCritique._parse_attachment_response(
        json.dumps({"issues": [issue]}), source, target
    )
    assert not rejected.valid
    assert not rejected.issues
    assert rejected.reported_issue_count == 1


def test_optional_plural_and_quoted_ezafe_require_body_source_evidence() -> None:
    source = "The discourse(s) in this account."
    target = "\u00ab\u06af\u0641\u062a\u0645\u0627\u0646\u00bb \u06cc \u0648 \u06af\u0641\u062a\u0645\u0627\u0646 (\u0647\u0627)"
    repaired, report = repair_source_grounded_language_artifacts(
        source, target, structural_role="body"
    )
    assert "\u00bb\u06cc" in repaired
    assert "\u06af\u0641\u062a\u0645\u0627\u0646(\u0647\u0627)" in repaired
    assert {item["type"] for item in report["repairs"]} >= {
        "detached_ezafe", "source_optional_plural_spacing"
    }
    assert not audit_translation_language(source, repaired)["review_required"]

    untouched, _ = repair_source_grounded_language_artifacts(
        "The discourse matters.", target, structural_role="heading"
    )
    assert untouched == target
    unsupported, _ = repair_source_grounded_language_artifacts(
        "The discourse matters.", "\u06af\u0641\u062a\u0645\u0627\u0646 (\u0647\u0627)",
        structural_role="body",
    )
    assert unsupported == "\u06af\u0641\u062a\u0645\u0627\u0646 (\u0647\u0627)"
    assert audit_translation_language(
        "The discourse matters.", unsupported
    )["spaced_optional_plural_count"] == 1
    unrelated, _ = repair_source_grounded_language_artifacts(
        "The discourse(s) matters.\nAnother sentence.",
        "\u06af\u0641\u062a\u0645\u0627\u0646 \u0645\u0647\u0645 \u0627\u0633\u062a.\n\u0646\u0647\u0627\u062f (\u0647\u0627)",
        structural_role="body",
    )
    assert "\u0646\u0647\u0627\u062f (\u0647\u0627)" in unrelated
    source_authored = "\u00ab\u0646\u0627\u0645\u00bb \u06cc \u0648 \u0646\u0647\u0627\u062f (\u0647\u0627)"
    preserved, _ = repair_source_grounded_language_artifacts(
        source_authored, source_authored
    )
    assert preserved == source_authored
    authored_audit = audit_translation_language(source_authored, preserved)
    assert authored_audit["detached_ezafe_count"] == 0
    assert authored_audit["spaced_optional_plural_count"] == 0


def test_attachment_trial_reads_only_body_chunks_and_never_writes_db(tmp_path: Path) -> None:
    path = tmp_path / "jobs.db"
    with sqlite3.connect(path) as db:
        db.executescript(
            "CREATE TABLE jobs(id TEXT PRIMARY KEY, config TEXT);"
            "CREATE TABLE chunks(job_id TEXT, chunk_index INTEGER, text TEXT, "
            "translation TEXT, metadata TEXT, status TEXT);"
        )
        db.execute("INSERT INTO jobs VALUES (?,?)", ("job", "{}"))
        db.executemany(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?)", [
                ("job", 0, "Copyright.", "\u062d\u0642 \u0646\u0634\u0631.",
                 json.dumps({"chapter_title": "Copyright", "structural_roles": ["heading"]}),
                 "completed"),
                ("job", 1, "The argument continues.", "\u0628\u062d\u062b \u0627\u062f\u0627\u0645\u0647 \u0645\u06cc\u200c\u06cc\u0627\u0628\u062f.",
                 json.dumps({"structural_roles": ["body"]}), "needs_review"),
            ],
        )
    before = path.read_bytes()
    script = Path(__file__).resolve().parents[1] / "scripts" / "replay_tarjomeh_v1038_attachment.py"
    spec = importlib.util.spec_from_file_location("attachment_replay", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config, chunks = module.frozen_chunks(path, "job", 16)
    assert config == {}
    assert [item["chunk_index"] for item in chunks] == [1]
    assert chunks[0]["translation_hash"]
    assert path.read_bytes() == before


def test_v1038_release_scripts_keep_9router_out_of_cleanup() -> None:
    root = Path(__file__).resolve().parents[1]
    deploy = (root / "scripts" / "deploy_tarjomeh_v1038.sh").read_text(
        encoding="utf-8"
    )
    companion = (
        root / "scripts" / "audit_tarjomeh_v1038_companion.sh"
    ).read_text(encoding="utf-8")
    reports = (
        root / "scripts" / "audit_tarjomeh_v1038_reports.sh"
    ).read_text(encoding="utf-8")
    assert 'TAG="v10.38.0"' in deploy
    assert "--no-deps" in deploy
    for marker in ("NINE_CONTAINER_ID", "NINE_IMAGE", "NINE_STARTED", "NINE_MOUNTS"):
        assert deploy.count(marker) >= 2
    assert "docker image prune" not in deploy
    assert "docker system prune" not in deploy
    assert "COPY scripts/" not in deploy
    assert "less than 350 MB" in deploy
    assert "post_edit_attribution" in companion and "post_edit_attribution" in reports
    assert "runtime_behavior_contract" in companion
    assert 'verdict = "FAIL" if hard else "REVIEW" if review else "PASS"' in companion
