"""Tarjomeh Web UI — Flask application with REST API and SSE progress streaming.

Token-based authentication via UI_SECRET_TOKEN environment variable.
Bound to 127.0.0.1 by default for VPS security (access via SSH tunnel).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import queue
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from functools import wraps
from pathlib import Path
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

logger = logging.getLogger(__name__)

# Global job executor — bounded to 2 concurrent workers
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tarjomeh-job")
_active_jobs: dict[str, Future] = {}
_progress_queues: dict[str, queue.Queue] = {}


def create_app(config: Any = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        config: TarjomehConfig instance. If None, loads defaults.

    Returns:
        Configured Flask application.
    """
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
        }
        for form_key, dotted_key in _form_field_map.items():
            value = request.form.get(form_key)
            if value:
                config_overrides[dotted_key] = value

        # Create progress queue for SSE
        _progress_queues[job_id] = queue.Queue()

        # Submit translation job to thread pool
        def run_job():
            try:
                from tarjomeh.core.config import TarjomehConfig
                from tarjomeh.core.pipeline import TranslationPipeline

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

                    if data.get("stage") in ("complete", "error"):
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

        future = _active_jobs.get(job_id)
        if future and not future.done():
            future.cancel()
            return jsonify({"status": "paused", "job_id": job_id})
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

        # Update DB status to RUNNING
        db.update_job_status(job_id, JobStatus.RUNNING)

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

                q = _progress_queues.get(job_id)
                if q:
                    q.put({
                        "stage": "complete",
                        "progress": 1.0,
                        "output_path": str(result.output_path),
                    })
            except Exception as e:
                logger.exception(f"Resume job {job_id} failed: {e}")

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

    @app.route("/api/glossary/upload", methods=["POST"])
    @_require_auth
    def api_upload_glossary():
        """Upload a custom glossary CSV."""
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if not file.filename or not file.filename.endswith(".csv"):
            return jsonify({"error": "Must be a CSV file"}), 400

        glossary_dir = Path("glossary")
        glossary_dir.mkdir(exist_ok=True)
        save_path = glossary_dir / file.filename
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
