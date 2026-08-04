"""Tarjomeh Web UI — Flask application with REST API and SSE progress streaming.

Token-based authentication via UI_SECRET_TOKEN environment variable.
Bound to 127.0.0.1 by default for VPS security (access via SSH tunnel).
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import os
import queue
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from functools import wraps
from pathlib import Path
from threading import Timer
from typing import Any

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    session,
    send_file,
)
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

# Global job executor — bounded to 2 concurrent workers
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tarjomeh-job")
_active_jobs: dict[str, Future] = {}
_progress_queues: dict[str, queue.Queue] = {}
_PROGRESS_QUEUE_TTL_SECONDS = 300.0
_QUERY_SECRET_RE = re.compile(
    r"(?P<prefix>[?&](?:token|api_key|key)=)[^&\s\"']+",
    re.IGNORECASE,
)


def _redact_query_secrets(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return _QUERY_SECRET_RE.sub(r"\g<prefix>[REDACTED]", value)


class _QuerySecretLogFilter(logging.Filter):
    """Prevent URL authentication secrets from entering request logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _redact_query_secrets(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(_redact_query_secrets(arg) for arg in record.args)
        elif isinstance(record.args, dict):
            record.args = {
                key: _redact_query_secrets(value)
                for key, value in record.args.items()
            }
        return True


def _install_request_log_redaction() -> None:
    werkzeug_logger = logging.getLogger("werkzeug")
    if not any(
        isinstance(item, _QuerySecretLogFilter)
        for item in werkzeug_logger.filters
    ):
        werkzeug_logger.addFilter(_QuerySecretLogFilter())


def _schedule_progress_queue_cleanup(job_id: str, delay: float = _PROGRESS_QUEUE_TTL_SECONDS) -> None:
    """Drop finished-job SSE queues even if the browser never consumed them."""
    timer = Timer(delay, lambda: _progress_queues.pop(job_id, None))
    timer.daemon = True
    timer.start()


def _safe_glossary_upload_path(filename: str) -> Path:
    """Return a safe glossary upload path restricted to the glossary folder."""
    safe_name = secure_filename(filename)
    if not safe_name or not safe_name.lower().endswith(".csv"):
        raise ValueError("Must be a CSV file")

    glossary_dir = Path("glossary")
    glossary_dir.mkdir(exist_ok=True)
    root = glossary_dir.resolve()
    save_path = (glossary_dir / safe_name).resolve()
    if save_path.parent != root:
        raise ValueError("Invalid glossary filename")
    return save_path


def _job_critique_threshold(job: dict[str, Any]) -> float:
    """Read the configured critique threshold for review UI flagging."""
    try:
        return float(job.get("config", {}).get("translation", {}).get("critique_threshold", 9.0))
    except (TypeError, ValueError):
        return 7.0


def create_app(config: Any = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        config: TarjomehConfig instance. If None, loads defaults.

    Returns:
        Configured Flask application.
    """
    _install_request_log_redaction()
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )

    app.secret_key = os.environ.get("FLASK_SECRET_KEY", uuid.uuid4().hex)
    app.config["TARJOMEH_CONFIG"] = config
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max upload
    app.config["UPLOAD_FOLDER"] = Path("jobs") / "uploads"
    app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)

    _register_auth(app)
    _register_routes(app)
    _register_api(app)

    return app


def _require_auth(f):
    """Decorator to require authentication for a route."""
    @wraps(f)
    def decorated(*args, **kwargs):
        # Check if auth is configured
        secret = os.environ.get("UI_SECRET_TOKEN", "")
        if not secret:
            # No auth configured — allow access (development mode)
            return f(*args, **kwargs)

        # Check session
        if session.get("authenticated"):
            return f(*args, **kwargs)

        # Check query parameter (first-time auth)
        token = request.args.get("token", "")
        if token == secret:
            session["authenticated"] = True
            session.permanent = True
            return f(*args, **kwargs)

        # Check Authorization header (API access)
        auth_header = request.headers.get("Authorization", "")
        if auth_header == f"Bearer {secret}":
            return f(*args, **kwargs)

        return jsonify({"error": "Authentication required. Pass ?token=YOUR_TOKEN"}), 401

    return decorated


def _register_auth(app: Flask) -> None:
    """Register authentication middleware."""

    @app.before_request
    def check_health():
        """Allow health check without auth."""
        if request.path == "/api/health":
            return None


def _register_routes(app: Flask) -> None:
    """Register web UI routes."""

    @app.route("/")
    @_require_auth
    def index():
        """Serve the main dashboard."""
        return render_template("index.html")


def _register_api(app: Flask) -> None:
    """Register REST API endpoints."""

    @app.route("/api/health")
    def api_health():
        """Health check endpoint (no auth required)."""
        return jsonify({
            "status": "ok",
            "version": "0.1.0",
            "timestamp": datetime.utcnow().isoformat(),
        })

    @app.route("/api/chapters", methods=["POST"])
    @_require_auth
    def api_chapters():
        """Parse an upload and return its detected chapter manifest only."""
        if "file" not in request.files or not request.files["file"].filename:
            return jsonify({"error": "No file provided"}), 400

        from tarjomeh.core.pipeline import build_chapter_manifest
        from tarjomeh.parsers.base import EXTENSION_PARSER_MAP

        upload = request.files["file"]
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in EXTENSION_PARSER_MAP:
            return jsonify({"error": f"Unsupported file format: {suffix}"}), 400

        inspection_path = (
            app.config["UPLOAD_FOLDER"]
            / f"chapter-inspection-{uuid.uuid4().hex[:12]}{suffix}"
        )
        try:
            upload.save(str(inspection_path))
            import importlib

            module_path, class_name = EXTENSION_PARSER_MAP[suffix].rsplit(".", 1)
            parser_cls = getattr(importlib.import_module(module_path), class_name)
            document = parser_cls().parse(inspection_path)
            chapters = build_chapter_manifest(document)
            warnings = []
            if len(chapters) == 1:
                warnings.append(
                    "Only one chapter was detected. Check source bookmarks or heading styles."
                )
            return jsonify({
                "title": document.title,
                "format": document.format_type,
                "chapters": chapters,
                "warnings": warnings,
            })
        except Exception as exc:
            logger.exception("Chapter inspection failed: %s", exc)
            return jsonify({"error": f"Chapter inspection failed: {exc}"}), 400
        finally:
            inspection_path.unlink(missing_ok=True)

    @app.route("/api/translate", methods=["POST"])
    @_require_auth
    def api_translate():
        """Start a new translation job.

        Accepts multipart form data with:
        - file: the document to translate
        - config: JSON string with config overrides (optional)
        """
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "Empty filename"}), 400

        # Save uploaded file
        upload_dir = app.config["UPLOAD_FOLDER"]
        job_id = uuid.uuid4().hex[:12]
        file_ext = Path(file.filename).suffix
        saved_path = upload_dir / f"{job_id}{file_ext}"
        file.save(str(saved_path))

        # Build config overrides from the individual UI form fields and/or a
        # `config` JSON blob (for API callers). The web UI sends flat fields
        # (mode/format/bilingual_mode); map them to dotted config keys so they
        # actually reach the pipeline. Individual fields take precedence.
        config_overrides: dict[str, Any] = {}
        if "config" in request.form:
            try:
                config_overrides.update(json.loads(request.form["config"]))
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid config JSON"}), 400

        _form_field_map = {
            "mode": "translation.mode",
            "format": "output.format",
            "bilingual_mode": "output.bilingual_mode",
            "term_notes": "output.term_notes",
            "search_provider": "web_search.provider",
            "recovery_model": "llm.recovery.model",
            "critic_recovery_model": "llm.critic.recovery_model",
        }
        for form_key, dotted_key in _form_field_map.items():
            value = request.form.get(form_key)
            if value:
                config_overrides[dotted_key] = value
        _boolean_field_map = {
            "enable_book_research": "translation.enable_book_research",
            "enable_critique": "translation.enable_critique",
            "enable_back_translation": "translation.enable_back_translation",
            "enable_integrity_gate": "translation.enable_integrity_gate",
            "enable_web_context": "translation.enable_web_context",
            "enable_auto_extraction": "glossary.enable_auto_extraction",
            "enable_compliance_check": "glossary.enable_compliance_check",
            "enable_auto_correction": "glossary.enable_auto_correction",
            "scholarly_mode": "persian.scholarly_mode",
            "pause_after_each_chapter": (
                "translation.pause_after_each_chapter"
            ),
        }
        for form_key, dotted_key in _boolean_field_map.items():
            value = request.form.get(form_key, "").lower()
            if value in ("true", "false"):
                config_overrides[dotted_key] = value == "true"

        _numeric_field_map = {
            "max_refine_iterations": (
                "translation.max_refine_iterations",
                int,
            ),
            "critique_threshold": ("translation.critique_threshold", float),
            "qa_json_retries": ("translation.qa_json_retries", int),
            "critic_recovery_max_tokens": (
                "llm.critic.recovery_max_tokens",
                int,
            ),
            "back_translation_sample_pct": (
                "translation.back_translation_sample_pct",
                int,
            ),
            "phase7_max_queries": ("web_search.phase7_max_queries", int),
            "max_queries_per_chunk": (
                "web_search.max_queries_per_chunk",
                int,
            ),
            "max_queries_per_book": (
                "web_search.max_queries_per_book",
                int,
            ),
            "stop_after_chapter": (
                "translation.stop_after_chapter",
                int,
            ),
        }
        for form_key, (dotted_key, converter) in _numeric_field_map.items():
            value = request.form.get(form_key)
            if value not in (None, ""):
                try:
                    config_overrides[dotted_key] = converter(value)
                except ValueError:
                    return jsonify({
                        "error": f"Invalid numeric setting: {form_key}"
                    }), 400

        selected_raw = request.form.get("selected_chapters", "").strip()
        if selected_raw:
            try:
                selected = json.loads(selected_raw)
                if not isinstance(selected, list):
                    raise ValueError
                selected = sorted({int(value) for value in selected})
                if any(value < 1 for value in selected):
                    raise ValueError
                config_overrides["translation.chapter_selection"] = selected
            except (TypeError, ValueError, json.JSONDecodeError):
                return jsonify({
                    "error": "selected_chapters must be a JSON list of positive integers"
                }), 400

        # Create progress queue for SSE
        _progress_queues[job_id] = queue.Queue()

        # Submit translation job to thread pool
        def run_job():
            try:
                from tarjomeh.core.config import TarjomehConfig
                from tarjomeh.core.pipeline import TranslationPipeline
                from tarjomeh.jobs.database import JobDatabase, JobStatus

                base_config = app.config.get("TARJOMEH_CONFIG") or TarjomehConfig()
                # Deep-copy so per-job overrides never mutate the shared global
                # config (which would otherwise leak settings into later jobs).
                config = copy.deepcopy(base_config)
                if config_overrides:
                    config.update_from_overrides(config_overrides)

                pipeline = TranslationPipeline(config)

                def progress_callback(stage: str, pct: float, message: str = "") -> None:
                    q = _progress_queues.get(job_id)
                    if q:
                        q.put({"stage": stage, "progress": pct, "message": message})

                result = pipeline.run(
                    input_path=saved_path,
                    job_id=job_id,
                    progress_callback=progress_callback,
                )

                db = JobDatabase()
                current_job = db.get_job(job_id)
                if current_job and current_job.get("status") == JobStatus.PAUSED:
                    q = _progress_queues.get(job_id)
                    if q:
                        q.put({
                            "stage": "paused",
                            "progress": current_job.get("pct", 0),
                            "message": "Job paused. Resume when ready.",
                        })
                    return

                # Signal completion
                q = _progress_queues.get(job_id)
                if q:
                    q.put({
                        "stage": "complete",
                        "progress": 1.0,
                        "message": f"Output: {result.output_path}",
                        "output_path": str(result.output_path),
                    })

                # Send webhook notification
                _send_webhook(app.config.get("TARJOMEH_CONFIG"), job_id, "completed")

            except Exception as e:
                logger.exception(f"Job {job_id} failed: {e}")
                q = _progress_queues.get(job_id)
                if q:
                    q.put({"stage": "error", "progress": 0, "message": str(e)})
                _send_webhook(app.config.get("TARJOMEH_CONFIG"), job_id, "failed", str(e))
            finally:
                _active_jobs.pop(job_id, None)
                _schedule_progress_queue_cleanup(job_id)

        future = _executor.submit(run_job)
        _active_jobs[job_id] = future

        return jsonify({
            "job_id": job_id,
            "status": "submitted",
            "file": file.filename,
            "stream_url": f"/api/jobs/{job_id}/stream",
        }), 202

    @app.route("/api/jobs")
    @_require_auth
    def api_list_jobs():
        """List all translation jobs."""
        from tarjomeh.jobs.database import JobDatabase
        db = JobDatabase()
        jobs = db.list_jobs()
        return jsonify({"jobs": jobs})

    @app.route("/api/jobs/<job_id>")
    @_require_auth
    def api_job_detail(job_id: str):
        """Get detailed job status."""
        from tarjomeh.jobs.database import JobDatabase
        db = JobDatabase()
        job = db.get_job(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        chunks = db.get_chunk_summary(job_id)
        return jsonify({"job": job, "chunks": chunks})

    @app.route("/api/jobs/<job_id>/events")
    @_require_auth
    def api_job_events(job_id: str):
        """Get structured per-chunk QA/progress events for a job."""
        from tarjomeh.jobs.database import JobDatabase
        db = JobDatabase()
        job = db.get_job(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        chunk_index_raw = request.args.get("chunk")
        chunk_index = None
        if chunk_index_raw not in (None, ""):
            try:
                chunk_index = int(chunk_index_raw)
            except ValueError:
                return jsonify({"error": "chunk must be an integer"}), 400

        return jsonify({"events": db.get_chunk_events(job_id, chunk_index)})

    @app.route("/api/evaluations", methods=["POST"])
    @_require_auth
    def api_create_evaluation():
        """Create a persisted deterministic comparison between two jobs."""
        from tarjomeh.quality.evaluation import QualityEvaluator

        data = request.get_json(silent=True) or {}
        baseline = str(data.get("baseline_job_id", "")).strip()
        candidate = str(data.get("candidate_job_id", "")).strip()
        if not baseline or not candidate:
            return jsonify({"error": "baseline_job_id and candidate_job_id are required"}), 400
        try:
            evaluation = QualityEvaluator().evaluate(baseline, candidate)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({
            "evaluation_id": evaluation["id"],
            "summary": evaluation["summary"],
        }), 201

    @app.route("/api/evaluations/<evaluation_id>")
    @_require_auth
    def api_get_evaluation(evaluation_id: str):
        """Return blind A/B translations and deterministic evidence."""
        from tarjomeh.jobs.database import JobDatabase

        evaluation = JobDatabase().get_evaluation(evaluation_id)
        if not evaluation:
            return jsonify({"error": "Evaluation not found"}), 404
        blind_chunks = []
        for item in evaluation.get("chunks", []):
            candidate_first = bool(item.get("a_is_candidate"))
            blind_chunks.append({
                "chunk_index": item["chunk_index"],
                "source": item.get("source", ""),
                "source_aligned": item.get("source_aligned", False),
                "translation_a": item.get(
                    "candidate_translation" if candidate_first else "baseline_translation", ""
                ),
                "translation_b": item.get(
                    "baseline_translation" if candidate_first else "candidate_translation", ""
                ),
                "regression_count": len(item.get("regressions", [])),
                "improvement_count": len(item.get("improvements", [])),
                "preference": item.get("preference"),
            })
        return jsonify({
            "id": evaluation["id"],
            "created_at": evaluation["created_at"],
            "summary": evaluation["summary"],
            "chunks": blind_chunks,
        })

    @app.route("/api/evaluations/<evaluation_id>/report")
    @_require_auth
    def api_evaluation_report(evaluation_id: str):
        """Download a text, JSON, or CSV regression report."""
        from tarjomeh.jobs.database import JobDatabase
        from tarjomeh.quality.evaluation import (
            evaluation_csv_report,
            evaluation_json_report,
            evaluation_text_report,
        )

        evaluation = JobDatabase().get_evaluation(evaluation_id)
        if not evaluation:
            return jsonify({"error": "Evaluation not found"}), 404
        report_format = request.args.get("format", "text").lower()
        renderers = {
            "text": (evaluation_text_report, "text/plain", "txt"),
            "json": (evaluation_json_report, "application/json", "json"),
            "csv": (evaluation_csv_report, "text/csv", "csv"),
        }
        if report_format not in renderers:
            return jsonify({"error": "format must be text, json, or csv"}), 400
        renderer, mimetype, extension = renderers[report_format]
        return Response(
            renderer(evaluation), mimetype=f"{mimetype}; charset=utf-8",
            headers={
                "Content-Disposition":
                    f"attachment; filename={evaluation_id}_evaluation.{extension}"
            },
        )

    @app.route(
        "/api/evaluations/<evaluation_id>/chunks/<int:chunk_index>/preference",
        methods=["POST"],
    )
    @_require_auth
    def api_evaluation_preference(evaluation_id: str, chunk_index: int):
        """Persist a blind human choice and optionally add it to the benchmark."""
        from tarjomeh.jobs.database import JobDatabase
        from tarjomeh.quality.evaluation import source_hash

        db = JobDatabase()
        evaluation = db.get_evaluation(evaluation_id)
        if not evaluation:
            return jsonify({"error": "Evaluation not found"}), 404
        chunk = next(
            (item for item in evaluation.get("chunks", [])
             if int(item["chunk_index"]) == chunk_index),
            None,
        )
        if not chunk:
            return jsonify({"error": "Evaluation chunk not found"}), 404

        data = request.get_json(silent=True) or {}
        preference = str(data.get("preference", "")).lower()
        if preference not in {"a", "b", "equal", "both_need_edit"}:
            return jsonify({"error": "Invalid preference"}), 400
        edited = str(data.get("edited_translation", "")).strip() or None
        notes = str(data.get("notes", "")).strip() or None
        db.save_evaluation_preference(
            evaluation_id, chunk_index, preference, edited, notes,
        )

        approved = edited
        if not approved and preference in {"a", "b"}:
            a_is_candidate = bool(chunk.get("a_is_candidate"))
            selected_candidate = (preference == "a") == a_is_candidate
            approved = chunk.get(
                "candidate_translation" if selected_candidate else "baseline_translation"
            )
        saved_to_benchmark = bool(data.get("save_to_benchmark") and approved)
        if saved_to_benchmark:
            db.save_approved_benchmark(
                source_hash(chunk.get("source", "")),
                chunk.get("source", ""),
                approved or "",
                {
                    "evaluation_id": evaluation_id,
                    "chunk_index": chunk_index,
                    "preference": preference,
                    "baseline_job_id": evaluation["baseline_job_id"],
                    "candidate_job_id": evaluation["candidate_job_id"],
                    "notes": notes or "",
                },
            )
        return jsonify({
            "status": "saved",
            "preference": preference,
            "saved_to_benchmark": saved_to_benchmark,
        })

    @app.route("/api/evaluation-benchmark")
    @_require_auth
    def api_evaluation_benchmark():
        """Return private approved examples for regression administration."""
        from tarjomeh.jobs.database import JobDatabase

        items = JobDatabase().list_approved_benchmark()
        return jsonify({"count": len(items), "items": items})

    @app.route("/api/jobs/<job_id>/review")
    @_require_auth
    def api_job_review(job_id: str):
        """Return chunks plus QA flags for editor review."""
        from tarjomeh.jobs.database import JobDatabase
        db = JobDatabase()
        job = db.get_job(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        events = db.get_chunk_events(job_id)
        events_by_chunk: dict[int, list[dict[str, Any]]] = {}
        for event in events:
            events_by_chunk.setdefault(int(event["chunk_index"]), []).append(event)
        issues_by_chunk: dict[int, list[dict[str, Any]]] = {}
        for issue in db.get_qa_issues(job_id):
            issues_by_chunk.setdefault(int(issue["chunk_index"]), []).append(issue)
        decisions_by_chunk: dict[int, list[dict[str, Any]]] = {}
        for decision in db.get_issue_decisions(job_id):
            decisions_by_chunk.setdefault(
                int(decision["chunk_index"]), []
            ).append(decision)

        critique_threshold = _job_critique_threshold(job)
        review_chunks = []
        for chunk in db.get_chunks(job_id):
            idx = int(chunk["chunk_index"])
            all_chunk_events = events_by_chunk.get(idx, [])
            last_start = 0
            for i, event in enumerate(all_chunk_events):
                if event.get("event_type") == "chunk_started":
                    last_start = i
            chunk_events = all_chunk_events[last_start:]
            critique_events = [
                e for e in chunk_events
                if e["event_type"] == "critique_completed"
            ]
            latest_critique = critique_events[-1] if critique_events else None
            scores = [
                e["payload"].get("scores", {}).get("average")
                for e in critique_events
            ]
            scores = [float(s) for s in scores if s is not None]
            blocking_critique_issues = [
                issue
                for e in ([latest_critique] if latest_critique else [])
                for issue in _blocking_critique_issues(e.get("payload", {}))
            ]
            needs_review_events = [
                e for e in chunk_events
                if e["event_type"] == "critique_needs_review"
            ]
            qa_unavailable_events = [
                e for e in chunk_events if e["event_type"] == "qa_unavailable"
            ]
            integrity_rejections = [
                e for e in chunk_events if e["event_type"] == "integrity_edit_rejected"
            ]
            integrity_final_failures = [
                e for e in chunk_events if e["event_type"] == "integrity_final_failed"
            ]
            final_glossary_events = [
                e for e in chunk_events
                if e["event_type"] == "glossary_compliance_final"
            ]
            glossary_events = final_glossary_events or [
                e for e in chunk_events
                if e["event_type"] == "glossary_compliance_checked"
            ]
            glossary_violations = sum(
                int(e["payload"].get("violation_count", 0))
                for e in glossary_events
            )
            bt_flagged = any(
                bool(e["payload"].get("flagged"))
                for e in chunk_events
                if e["event_type"] == "back_translation_completed"
            )
            low_score = bool(scores and min(scores) < critique_threshold)
            flagged = (
                chunk["status"] != "completed"
                or low_score
                or bool(blocking_critique_issues)
                or bool(needs_review_events)
                or bool(qa_unavailable_events)
                or bool(integrity_rejections)
                or bool(integrity_final_failures)
                or glossary_violations > 0
                or bt_flagged
            )
            review_chunks.append({
                "chunk_index": idx,
                "status": chunk["status"],
                "source": chunk.get("text") or "",
                "translation": chunk.get("translation") or "",
                "critique_average": min(scores) if scores else None,
                "critique_threshold": critique_threshold,
                "blocking_critique_issues": blocking_critique_issues,
                "needs_review": bool(
                    needs_review_events
                    or qa_unavailable_events
                    or integrity_rejections
                    or integrity_final_failures
                    or bt_flagged
                ),
                "qa_unavailable": bool(qa_unavailable_events),
                "integrity_rejections": len(integrity_rejections),
                "integrity_final_failures": len(integrity_final_failures),
                "glossary_violations": glossary_violations,
                "back_translation_flagged": bt_flagged,
                "flagged": flagged,
                "qa_issues": issues_by_chunk.get(idx, []),
                "issue_decisions": decisions_by_chunk.get(idx, []),
                "events": chunk_events,
            })

        return jsonify({"job": job, "critique_threshold": critique_threshold, "chunks": review_chunks})

    @app.route("/api/jobs/<job_id>/qa-report")
    @_require_auth
    def api_job_qa_report(job_id: str):
        """Download a text QA scorecard for a job."""
        from tarjomeh.jobs.database import JobDatabase
        db = JobDatabase()
        job = db.get_job(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        lines = [
            f"Tarjomeh QA Report",
            f"Job: {job_id}",
            f"Input: {job.get('input_path')}",
            f"Status: {job.get('status')}",
            "",
        ]
        job_config = job.get("config", {})
        translation_config = job_config.get("translation", {})
        glossary_config = job_config.get("glossary", {})
        output_config = job_config.get("output", {})
        search_config = job_config.get("web_search", {})
        critic_config = job_config.get("llm", {}).get("critic", {})
        lines.extend([
            "Job Configuration:",
            f"  mode={translation_config.get('mode')}",
            f"  output={output_config.get('format')} term_notes={output_config.get('term_notes')}",
            f"  critique={translation_config.get('enable_critique')} "
            f"threshold={translation_config.get('critique_threshold')} "
            f"refinements={translation_config.get('max_refine_iterations')}",
            f"  critic_recovery_attempts={critic_config.get('recovery_max_attempts')} "
            f"fallback={critic_config.get('recovery_model') or '(same model)'} "
            f"final_tokens={critic_config.get('recovery_max_tokens')}",
            f"  integrity_gate={translation_config.get('enable_integrity_gate')}",
            f"  back_translation={translation_config.get('enable_back_translation')} "
            f"sample_pct={translation_config.get('back_translation_sample_pct')}",
            f"  book_research={translation_config.get('enable_book_research')} "
            f"web_context={translation_config.get('enable_web_context')}",
            f"  chapter_selection={translation_config.get('chapter_selection') or 'all'} "
            f"stop_after={translation_config.get('stop_after_chapter') or 0} "
            f"pause_each={translation_config.get('pause_after_each_chapter', False)}",
            f"  glossary_compliance={glossary_config.get('enable_compliance_check')} "
            f"auto_correction={glossary_config.get('enable_auto_correction')}",
            f"  search_provider={search_config.get('provider')} "
            f"research_queries={search_config.get('phase7_max_queries')} "
            f"chunk_queries={search_config.get('max_queries_per_chunk')} "
            f"book_query_budget={search_config.get('max_queries_per_book')}",
            "",
        ])
        research = db.get_job_artifact(job_id, "book_research")
        chapter_manifest = db.get_job_artifact(job_id, "chapter_manifest")
        chapter_checkpoints = db.get_job_artifact(job_id, "chapter_checkpoints")
        if chapter_manifest is not None:
            lines.extend([
                "Chapter Scope:",
                f"  detected={len(chapter_manifest.get('chapters', []))}",
                f"  selected={chapter_manifest.get('selected_positions') or 'all'}",
                "  checkpoints_reached="
                + str((chapter_checkpoints or {}).get("reached_positions", [])),
                "",
            ])
        if research is not None:
            suggested = [
                term for term in research.get("terms", [])
                if isinstance(term, dict) and term.get("status") == "suggested"
            ]
            approved = [
                term for term in research.get("terms", [])
                if isinstance(term, dict) and term.get("status") == "approved"
            ]
            lines.extend([
                "Book Research:",
                f"  status={research.get('status')}",
                f"  sources={len(research.get('sources', []))}",
                f"  queries={len(research.get('queries', []))}",
                "  providers="
                + ", ".join(research.get("providers_used", [])),
                f"  suggestions={len(suggested)} approved={len(approved)}",
                f"  context={research.get('book_context', '')}",
                "",
            ])
        original_audit = db.get_job_artifact(job_id, "english_original_audit")
        if original_audit is not None:
            lines.extend([
                "English-original audit:",
                f"  authorized={original_audit.get('authorized_count', 0)} kept={original_audit.get('kept_authorized', 0)}",
                f"  unauthorized_removed={original_audit.get('removed_unauthorized_count', 0)} duplicates_removed={original_audit.get('removed_duplicate_count', 0)}",
                f"  citations_preserved={original_audit.get('preserved_citation_count', 0)}",
                "",
            ])
        all_events = db.get_chunk_events(job_id)
        all_chunks = db.get_chunks(job_id)
        events_by_chunk: dict[int, list[dict[str, Any]]] = {}
        for event in all_events:
            events_by_chunk.setdefault(int(event["chunk_index"]), []).append(event)

        review_reasons: set[str] = set()
        if job.get("status") != "completed":
            review_reasons.add(f"job_{job.get('status')}")
        if any(chunk.get("status") == "needs_review" for chunk in all_chunks):
            review_reasons.add("chunk_needs_review")
        if any(
            chunk.get("status") not in {"completed", "needs_review"}
            for chunk in all_chunks
        ):
            review_reasons.add("incomplete_chunk")
        review_event_reasons = {
            "qa_unavailable": "qa_unavailable",
            "glossary_needs_review": "glossary_needs_review",
            "integrity_final_failed": "integrity_final_failed",
        }
        for event in all_events:
            reason = review_event_reasons.get(event["event_type"])
            if reason:
                review_reasons.add(reason)
        lines.extend([
            "QA Verdict:",
            "  status=" + ("review_required" if review_reasons else "pass"),
            "  reasons=" + (
                ", ".join(sorted(review_reasons)) if review_reasons else "none"
            ),
            "",
        ])

        for chunk in all_chunks:
            idx = int(chunk["chunk_index"])
            chunk_events = events_by_chunk.get(idx, [])
            lines.append(f"Chunk {idx} [{chunk['status']}]")
            for event in chunk_events:
                payload = event["payload"]
                if event["event_type"] == "llm_call_attempt" and (
                    payload.get("recovery") or not payload.get("success", False)
                ):
                    lines.append(
                        f"  LLM attempt: operation={payload.get('operation')} "
                        f"attempt={payload.get('attempt')}/{payload.get('max_attempts')} "
                        f"success={payload.get('success')} "
                        f"finish={payload.get('finish_reason')} "
                        f"failure={payload.get('failure_reason', '')} "
                        f"model={payload.get('model')} "
                        f"tokens={payload.get('completion_tokens', 0)}"
                    )
                elif event["event_type"] == "critique_completed":
                    scores = payload.get("scores", {})
                    lines.append(
                        "  Critique: valid={valid} attempts={attempts} avg={average} accuracy={accuracy} fluency={fluency} "
                        "terminology={terminology} register={register} issues={issues} "
                        "blocking={blocking}".format(
                            valid=payload.get("valid", True),
                            attempts=payload.get("attempts", 1),
                            average=scores.get("average"),
                            accuracy=scores.get("accuracy"),
                            fluency=scores.get("fluency"),
                            terminology=scores.get("terminology"),
                            register=scores.get("register"),
                            issues=payload.get("issue_count"),
                            blocking=payload.get("blocking_issue_count", 0),
                        )
                    )
                    for issue in payload.get("issue_details", []):
                        lines.append(
                            "    MQM {issue_id}: {severity}/{category} "
                            "confidence={confidence}".format(
                                issue_id=issue.get("issue_id"),
                                severity=issue.get("severity"),
                                category=issue.get("category"),
                                confidence=issue.get("confidence"),
                            )
                        )
                        lines.append(
                            f"      Source: {issue.get('source_quote', '')}"
                        )
                        lines.append(
                            "      Current: "
                            f"{issue.get('current_persian_quote', '')}"
                        )
                        lines.append(
                            "      Suggested: "
                            f"{issue.get('suggested_correction', '')}"
                        )
                elif event["event_type"] == "refinement_completed":
                    lines.append(
                        f"  Refinement: iteration={payload.get('iteration')} "
                        f"decision={payload.get('decision')} "
                        f"before={payload.get('before_chars')} after={payload.get('after_chars')}"
                    )
                    if payload.get("rationale"):
                        lines.append(f"    Rationale: {payload.get('rationale')}")
                    for decision in payload.get("issue_decisions", []):
                        lines.append(
                            f"    Decision {decision.get('issue_id')}: "
                            f"{decision.get('decision')} -> "
                            f"{decision.get('resulting_span')}"
                        )
                        if decision.get("rationale"):
                            lines.append(
                                f"      {decision.get('rationale')}"
                            )
                elif event["event_type"] == "critique_needs_review":
                    lines.append(
                        f"  NEEDS REVIEW: reason={payload.get('review_reason', 'legacy_unresolved')} "
                        f"blocking={payload.get('blocking_issue_count')} "
                        f"after iteration={payload.get('iteration')}"
                    )
                    if payload.get("message"):
                        lines.append(f"    {payload.get('message')}")
                elif event["event_type"] == "glossary_compliance_final":
                    lines.append(
                        f"  Glossary: compliant={payload.get('compliant')} "
                        f"violations={payload.get('violation_count')}"
                    )
                    for violation in payload.get("violations", []):
                        lines.append(
                            "    "
                            f"{violation.get('term')} -> {violation.get('expected')} "
                            f"status={violation.get('status')}"
                        )
                    for term in payload.get("citation_exemptions", []):
                        lines.append(
                            f"    Citation preserved (not terminology): {term}"
                        )
                elif event["event_type"] == "glossary_needs_review":
                    lines.append(
                        "  GLOSSARY NEEDS REVIEW: "
                        f"violations={payload.get('violation_count')}"
                    )
                    if payload.get("message"):
                        lines.append(f"    {payload.get('message')}")
                elif event["event_type"] == "back_translation_completed":
                    diagnostics = payload.get("diagnostics", {})
                    lines.append(
                        f"  Back-translation: score={payload.get('similarity_score')} "
                        f"flagged={payload.get('flagged')} "
                        f"risks={diagnostics.get('risk_flags', [])}"
                    )
                    if diagnostics.get("missing_entities"):
                        lines.append(
                            "    Missing entities: "
                            + ", ".join(diagnostics["missing_entities"])
                        )
                elif event["event_type"] == "integrity_edit_rejected":
                    lines.append(
                        f"  INTEGRITY REJECTED: stage={payload.get('stage')} "
                        f"blocking={payload.get('blocking_count')} prior translation retained"
                    )
                    for finding in payload.get("findings", []):
                        if finding.get("severity") == "blocking":
                            lines.append(
                                f"    {finding.get('check_id')}: {finding.get('message')}"
                            )
                elif event["event_type"] == "integrity_final_failed":
                    lines.append(
                        f"  INTEGRITY FINAL: blocking={payload.get('blocking_count')} "
                        "chunk requires review"
                    )
                elif event["event_type"] == "qa_unavailable":
                    lines.append(
                        f"  QA UNAVAILABLE: component={payload.get('component')} "
                        f"attempts={payload.get('attempts')}"
                    )
                    if payload.get("message"):
                        lines.append(f"    {payload.get('message')}")
            lines.append("")

        report = "\n".join(lines)
        return Response(
            report,
            mimetype="text/plain; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={job_id}_qa_report.txt"},
        )

    @app.route("/api/jobs/<job_id>/chunks/<int:chunk_index>/retranslate", methods=["POST"])
    @_require_auth
    def api_retranslate_chunk(job_id: str, chunk_index: int):
        """Retranslate a single chunk and update its DB translation."""
        from tarjomeh.core.config import TarjomehConfig
        from tarjomeh.core.pipeline import TranslationPipeline
        from tarjomeh.jobs.database import JobDatabase

        db = JobDatabase()
        job = db.get_job(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        config = TarjomehConfig.from_dict(job.get("config", {}))
        pipeline = TranslationPipeline(config)
        translation = pipeline.retranslate_chunk(job_id, chunk_index)
        return jsonify({
            "status": "retranslated",
            "job_id": job_id,
            "chunk_index": chunk_index,
            "translation": translation,
        })

    @app.route("/api/jobs/<job_id>/export", methods=["POST"])
    @_require_auth
    def api_export_job(job_id: str):
        """Re-export a job from completed chunks without translating."""
        from tarjomeh.core.config import TarjomehConfig
        from tarjomeh.core.pipeline import TranslationPipeline
        from tarjomeh.jobs.database import JobDatabase

        db = JobDatabase()
        job = db.get_job(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        data = request.get_json(silent=True) or {}
        fmt = data.get("format") or job.get("config", {}).get("output", {}).get("format", "docx")
        bilingual = data.get("bilingual_mode") or job.get("config", {}).get("output", {}).get("bilingual_mode")
        config = TarjomehConfig.from_dict(job.get("config", {}))
        pipeline = TranslationPipeline(config)
        input_path = Path(job["input_path"])
        extension = "md" if fmt == "markdown" else fmt
        output_path = input_path.parent / f"{input_path.stem}_reviewed.{extension}"
        exported = pipeline.export_completed_job(
            job_id,
            output_path,
            output_format=fmt,
            bilingual_mode=bilingual,
        )
        return jsonify({"status": "exported", "output_path": str(exported)})

    @app.route("/api/jobs/<job_id>/stream")
    @_require_auth
    def api_job_stream(job_id: str):
        """SSE endpoint for real-time job progress."""
        def generate():
            q = _progress_queues.get(job_id)
            if not q:
                # Job already finished (queue cleaned up) — tell the client the
                # stream is done so its EventSource stops reconnecting.
                yield f"data: {json.dumps({'stage': 'closed'})}\n\n"
                return

            while True:
                try:
                    data = q.get(timeout=30)
                    yield f"data: {json.dumps(data)}\n\n"

                    if data.get("stage") in ("complete", "error", "paused"):
                        # Terminal event delivered — drop the queue so a
                        # reconnecting EventSource gets 'closed' and stops,
                        # instead of looping on keepalives forever.
                        _progress_queues.pop(job_id, None)
                        break
                except queue.Empty:
                    # Comment-only keepalive: keeps the connection alive without
                    # emitting a bogus "message" event to the client.
                    yield ": keepalive\n\n"

        return Response(generate(), mimetype="text/event-stream")

    @app.route("/api/jobs/<job_id>/pause", methods=["POST"])
    @_require_auth
    def api_pause_job(job_id: str):
        """Pause a running job."""
        from tarjomeh.jobs.database import JobDatabase, JobStatus
        db = JobDatabase()
        job = db.get_job(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        # Update DB status to PAUSED
        db.update_job_status(job_id, JobStatus.PAUSED)
        db.log_event(job_id, "INFO", "Pause requested by user.")

        future = _active_jobs.get(job_id)
        if future and not future.done():
            future.cancel()
            return jsonify({
                "status": "pausing",
                "job_id": job_id,
                "message": "Pause requested. Current LLM call may finish before the job stops.",
            })
        return jsonify({"status": "paused", "job_id": job_id})

    @app.route("/api/jobs/<job_id>/resume", methods=["POST"])
    @_require_auth
    def api_resume_job(job_id: str):
        """Resume a paused job."""
        from tarjomeh.jobs.database import JobDatabase, JobStatus
        db = JobDatabase()
        job = db.get_job(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        future = _active_jobs.get(job_id)
        if future and not future.done():
            db.log_event(job_id, "WARNING", "Resume requested while an existing worker was still running.")
            return jsonify({
                "error": "Job is still pausing",
                "status": "pausing",
                "job_id": job_id,
                "message": "Wait until the current LLM call finishes, then resume again.",
            }), 409

        # Update DB status to RUNNING
        db.update_job_status(job_id, JobStatus.RUNNING)
        db.log_event(job_id, "INFO", "Resume requested by user.")

        # Re-submit to executor
        _progress_queues[job_id] = queue.Queue()

        def run_resume():
            try:
                from tarjomeh.core.config import TarjomehConfig
                from tarjomeh.core.pipeline import TranslationPipeline

                config = TarjomehConfig.from_dict(job.get("config", {}))
                pipeline = TranslationPipeline(config)

                def progress_callback(stage: str, pct: float, message: str = "") -> None:
                    q = _progress_queues.get(job_id)
                    if q:
                        q.put({"stage": stage, "progress": pct, "message": message})

                result = pipeline.run(
                    input_path=Path(job["input_path"]),
                    job_id=job_id,
                    progress_callback=progress_callback,
                )

                db_after = JobDatabase()
                current_job = db_after.get_job(job_id)
                if current_job and current_job.get("status") == JobStatus.PAUSED:
                    q = _progress_queues.get(job_id)
                    if q:
                        q.put({
                            "stage": "paused",
                            "progress": current_job.get("pct", 0),
                            "message": "Job paused. Resume when ready.",
                        })
                    return

                q = _progress_queues.get(job_id)
                if q:
                    q.put({
                        "stage": "complete",
                        "progress": 1.0,
                        "output_path": str(result.output_path),
                    })
            except Exception as e:
                logger.exception(f"Resume job {job_id} failed: {e}")
                q = _progress_queues.get(job_id)
                if q:
                    q.put({"stage": "error", "progress": 0, "message": str(e)})
            finally:
                _active_jobs.pop(job_id, None)
                _schedule_progress_queue_cleanup(job_id)

        future = _executor.submit(run_resume)
        _active_jobs[job_id] = future

        return jsonify({"status": "resumed", "job_id": job_id}), 202

    @app.route("/api/jobs/<job_id>/download")
    @_require_auth
    def api_download(job_id: str):
        """Download translated output file."""
        from tarjomeh.jobs.database import JobDatabase
        db = JobDatabase()
        job = db.get_job(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        if not job.get("output_path"):
            return jsonify({"error": "Output not ready"}), 404

        output_path = Path(job["output_path"])
        if not output_path.exists():
            return jsonify({"error": "Output file missing"}), 404

        return send_file(str(output_path.resolve()), as_attachment=True)

    @app.route("/api/jobs/<job_id>/research")
    @_require_auth
    def api_job_research(job_id: str):
        """Return the persisted, review-only book research artifact."""
        from tarjomeh.jobs.database import JobDatabase

        db = JobDatabase()
        if not db.get_job(job_id):
            return jsonify({"error": "Job not found"}), 404
        artifact = db.get_job_artifact(job_id, "book_research")
        return jsonify({
            "job_id": job_id,
            "research": artifact,
        })

    @app.route(
        "/api/jobs/<job_id>/research/terms/<int:index>/<action>",
        methods=["POST"],
    )
    @_require_auth
    def api_job_research_term(job_id: str, index: int, action: str):
        """Approve or reject one job-scoped research suggestion."""
        from tarjomeh.glossary.manager import GlossaryEntry, GlossaryManager
        from tarjomeh.jobs.database import JobDatabase

        if action not in ("approve", "reject"):
            return jsonify({"error": "action must be approve or reject"}), 400
        db = JobDatabase()
        artifact = db.get_job_artifact(job_id, "book_research")
        if artifact is None:
            return jsonify({"error": "Research artifact not found"}), 404
        terms = artifact.get("terms", [])
        if not isinstance(terms, list) or index < 0 or index >= len(terms):
            return jsonify({"error": "Research term not found"}), 404
        term = terms[index]
        if not isinstance(term, dict):
            return jsonify({"error": "Invalid research term"}), 400

        if action == "approve":
            path = _working_glossary_path(app.config.get("TARJOMEH_CONFIG"))
            gm = GlossaryManager()
            if path.exists():
                gm.load(path)
            source_key = str(term.get("source", "")).strip().casefold()
            curated = [
                entry for entry in gm.entries
                if entry.source.casefold() == source_key and not entry.is_auto
            ]
            if curated:
                return jsonify({
                    "error": "A curated glossary term already exists; it was not overwritten.",
                    "existing": _glossary_entry_payload(0, curated[0]),
                }), 409
            entries = [
                entry for entry in gm.entries
                if entry.source.casefold() != source_key
            ]
            entries.append(GlossaryEntry(
                source=str(term.get("source", "")).strip(),
                target=str(term.get("target", "")).strip(),
                context=str(term.get("context", "")).strip(),
                domain=str(term.get("domain", "")).strip(),
                sense=str(term.get("sense", "")).strip(),
                author=str(term.get("author", "")).strip(),
                glossary="book_research_approved",
                is_auto=False,
                include_original=False,
            ))
            _save_glossary_entries(path, entries)

        term["status"] = "approved" if action == "approve" else "rejected"
        db.save_job_artifact(job_id, "book_research", artifact)
        db.log_event(
            job_id,
            "INFO",
            f"Research term {index} {term['status']}: "
            + str(term.get("source", "")),
        )
        return jsonify({"status": term["status"], "term": term})

    @app.route("/api/glossary/upload", methods=["POST"])
    @_require_auth
    def api_upload_glossary():
        """Upload a custom glossary CSV."""
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "Must be a CSV file"}), 400

        try:
            save_path = _safe_glossary_upload_path(file.filename)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        file.save(str(save_path))

        # Validate
        from tarjomeh.glossary.manager import GlossaryManager
        try:
            gm = GlossaryManager()
            gm.load(save_path)
            return jsonify({
                "status": "uploaded",
                "path": str(save_path),
                "terms": len(gm.entries),
            })
        except Exception as e:
            save_path.unlink(missing_ok=True)
            return jsonify({"error": f"Invalid glossary: {e}"}), 400

    @app.route("/api/glossary/terms", methods=["GET", "POST"])
    @_require_auth
    def api_glossary_terms():
        """List or add terms in the working glossary."""
        from tarjomeh.glossary.manager import GlossaryEntry, GlossaryManager

        path = _working_glossary_path(app.config.get("TARJOMEH_CONFIG"))
        gm = GlossaryManager()
        if path.exists():
            gm.load(path)

        if request.method == "GET":
            return jsonify({
                "path": str(path),
                "terms": [_glossary_entry_payload(i, e) for i, e in enumerate(gm.entries)],
            })

        data = request.get_json(silent=True) or {}
        entry = GlossaryEntry(
            source=str(data.get("source", "")).strip(),
            target=str(data.get("target", "")).strip(),
            tgt_lng=str(data.get("tgt_lng", "fa")).strip() or "fa",
            context=str(data.get("context", "")).strip(),
            domain=str(data.get("domain", "")).strip(),
            sense=str(data.get("sense", "")).strip(),
            author=str(data.get("author", "")).strip(),
            is_auto=bool(data.get("is_auto", False)),
            include_original=bool(data.get("include_original", False)),
        )
        if not entry.source or not entry.target:
            return jsonify({"error": "source and target are required"}), 400
        entries = gm.entries + [entry]
        _save_glossary_entries(path, entries)
        return jsonify({"status": "created", "term": _glossary_entry_payload(len(entries) - 1, entry)}), 201

    @app.route("/api/glossary/terms/<int:index>", methods=["PUT", "DELETE"])
    @_require_auth
    def api_glossary_term(index: int):
        """Update or delete a working glossary term by row index."""
        from tarjomeh.glossary.manager import GlossaryEntry, GlossaryManager

        path = _working_glossary_path(app.config.get("TARJOMEH_CONFIG"))
        gm = GlossaryManager()
        if path.exists():
            gm.load(path)
        entries = gm.entries
        if index < 0 or index >= len(entries):
            return jsonify({"error": "term index not found"}), 404

        if request.method == "DELETE":
            removed = entries.pop(index)
            _save_glossary_entries(path, entries)
            return jsonify({"status": "deleted", "term": _glossary_entry_payload(index, removed)})

        data = request.get_json(silent=True) or {}
        current = entries[index]
        entries[index] = GlossaryEntry(
            source=str(data.get("source", current.source)).strip(),
            target=str(data.get("target", current.target)).strip(),
            tgt_lng=str(data.get("tgt_lng", current.tgt_lng)).strip() or "fa",
            context=str(data.get("context", current.context)).strip(),
            domain=str(data.get("domain", current.domain)).strip(),
            sense=str(data.get("sense", current.sense)).strip(),
            author=str(data.get("author", current.author)).strip(),
            glossary=current.glossary,
            is_auto=bool(data.get("is_auto", current.is_auto)),
            include_original=bool(data.get("include_original", current.include_original)),
        )
        if not entries[index].source or not entries[index].target:
            return jsonify({"error": "source and target are required"}), 400
        _save_glossary_entries(path, entries)
        return jsonify({"status": "updated", "term": _glossary_entry_payload(index, entries[index])})

    @app.route("/api/glossary/terms/<int:index>/approve", methods=["POST"])
    @_require_auth
    def api_glossary_approve(index: int):
        """Approve an auto-extracted glossary term."""
        from tarjomeh.glossary.manager import GlossaryEntry, GlossaryManager

        path = _working_glossary_path(app.config.get("TARJOMEH_CONFIG"))
        gm = GlossaryManager()
        if path.exists():
            gm.load(path)
        entries = gm.entries
        if index < 0 or index >= len(entries):
            return jsonify({"error": "term index not found"}), 404

        e = entries[index]
        entries[index] = GlossaryEntry(
            source=e.source,
            target=e.target,
            tgt_lng=e.tgt_lng,
            context=e.context,
            domain=e.domain,
            sense=e.sense,
            author=e.author,
            glossary=e.glossary,
            is_auto=False,
            include_original=e.include_original,
        )
        _save_glossary_entries(path, entries)
        return jsonify({"status": "approved", "term": _glossary_entry_payload(index, entries[index])})

    @app.route("/api/glossary/download")
    @_require_auth
    def api_glossary_download():
        """Download the working glossary CSV."""
        path = _working_glossary_path(app.config.get("TARJOMEH_CONFIG"))
        if not path.exists():
            return jsonify({"error": "Working glossary not found"}), 404
        return send_file(str(path.resolve()), as_attachment=True)


def _working_glossary_path(config: Any) -> Path:
    if config is not None and hasattr(config, "glossary"):
        path = getattr(config.glossary, "path", "")
        if path:
            return Path(path)
    return Path("glossary") / "academic_political_theory.csv"


def _glossary_entry_payload(index: int, entry: Any) -> dict[str, Any]:
    return {
        "index": index,
        "source": entry.source,
        "target": entry.target,
        "tgt_lng": entry.tgt_lng,
        "context": entry.context,
        "domain": entry.domain,
        "sense": entry.sense,
        "author": entry.author,
        "glossary": entry.glossary,
        "is_auto": entry.is_auto,
        "include_original": entry.include_original,
    }


def _blocking_critique_issues(payload: dict[str, Any]) -> list[str]:
    if payload.get("force_refinement"):
        issues = [str(issue) for issue in payload.get("blocking_issues", [])]
        return issues or ["Critique forced refinement."]

    issues = payload.get("issues", []) or []
    blocking_prefixes = (
        "[MAJOR/accuracy]",
        "[CRITICAL/accuracy]",
        "[MAJOR/terminology]",
        "[CRITICAL/terminology]",
    )
    return [
        str(issue)
        for issue in issues
        if str(issue).strip().lower().startswith(tuple(p.lower() for p in blocking_prefixes))
    ]


def _save_glossary_entries(path: Path, entries: list[Any]) -> None:
    from tarjomeh.glossary.manager import _entry_to_row, _fieldnames_for_entries

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=_fieldnames_for_entries(entries),
            extrasaction="ignore",
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(_entry_to_row(entry))


def _send_webhook(config: Any, job_id: str, status: str, error: str = "") -> None:
    """Send webhook notification for job events."""
    if not config:
        return

    webhook_url = ""
    if hasattr(config, "notifications"):
        webhook_url = getattr(config.notifications, "webhook_url", "")
    if not webhook_url:
        return

    notify_complete = getattr(config.notifications, "notify_on_complete", True)
    notify_error = getattr(config.notifications, "notify_on_error", True)

    if status == "completed" and not notify_complete:
        return
    if status == "failed" and not notify_error:
        return

    try:
        import httpx
        payload = {
            "event": f"tarjomeh.job.{status}",
            "job_id": job_id,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if error:
            payload["error"] = error

        httpx.post(webhook_url, json=payload, timeout=10)
    except Exception as e:
        logger.warning(f"Webhook notification failed: {e}")
