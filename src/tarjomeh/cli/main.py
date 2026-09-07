"""Tarjomeh CLI — Command-line interface for academic book translation.

Commands:
    tarjomeh translate <file>   Translate a document
    tarjomeh jobs list          List all translation jobs
    tarjomeh jobs status <id>   Show job status
    tarjomeh jobs resume <id>   Resume a paused/failed job
    tarjomeh jobs cleanup <id>  Clean up job artifacts
    tarjomeh glossary validate  Validate a glossary CSV file
    tarjomeh serve              Start the web UI server
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def cmd_translate(args: argparse.Namespace) -> int:
    """Execute the translate command."""
    from tarjomeh.core.config import TarjomehConfig
    from tarjomeh.core.pipeline import TranslationPipeline

    input_path = Path(args.file)
    if not input_path.exists():
        console.print(f"[red]Error:[/red] File not found: {input_path}")
        return 1

    # Build CLI overrides BEFORE loading, so validation can be deferred until
    # after they are merged. Otherwise `--provider ollama` fails validation
    # against a config.toml whose file-level provider needs an API key, and the
    # ValueError escapes as a traceback.
    overrides: dict = {}
    if args.mode:
        overrides["translation.mode"] = args.mode
    if args.output_format:
        overrides["output.format"] = args.output_format
    if args.bilingual:
        overrides["output.bilingual_mode"] = args.bilingual
    if getattr(args, "term_notes", None):
        overrides["output.term_notes"] = args.term_notes
    if getattr(args, "book_research", False):
        overrides["translation.enable_book_research"] = True
    if args.provider:
        overrides["llm.provider"] = args.provider
    if args.model:
        overrides["llm.model"] = args.model

    config_path = Path(args.config) if args.config else None
    try:
        config = TarjomehConfig.load(config_path, validate=not overrides)
    except FileNotFoundError:
        console.print(
            "[yellow]Warning:[/yellow] No config.toml found. Using defaults. "
            "Copy config.example.toml to config.toml to customize."
        )
        config = TarjomehConfig()
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        return 1

    if overrides:
        # update_from_overrides() validates once the merge is complete.
        try:
            config.update_from_overrides(overrides)
        except ValueError as exc:
            console.print(f"[red]Configuration error:[/red] {exc}")
            return 1
    else:
        config.validate()

    # Determine output path
    output_path = Path(args.output) if args.output else None

    # Create and run pipeline
    pipeline = TranslationPipeline(config)

    console.print(f"\n[bold cyan]ترجمه[/bold cyan] — Translating: [green]{input_path.name}[/green]")
    console.print(f"  Mode: [yellow]{config.translation.mode}[/yellow]")
    console.print(f"  Provider: {config.llm.provider}")
    console.print(f"  Model: {config.llm.model}")
    console.print(f"  Output: {config.output.format} ({config.output.bilingual_mode})")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Translating...", total=100)

        def on_progress(stage: str, pct: float, message: str = "") -> None:
            progress.update(task, completed=pct * 100, description=f"{stage}: {message}")

        try:
            result = pipeline.run(
                input_path=input_path,
                output_path=output_path,
                job_id=args.resume,
                progress_callback=on_progress,
            )
        except KeyboardInterrupt:
            # Persist the pause, otherwise the job stays RUNNING and the
            # message below promises a state that was never written.
            try:
                from tarjomeh.jobs.database import JobDatabase, JobStatus

                if pipeline.current_job_id:
                    db = JobDatabase()
                    db.update_job_status(pipeline.current_job_id, JobStatus.PAUSED)
                    db.log_event(
                        pipeline.current_job_id,
                        "INFO",
                        "Paused by keyboard interrupt.",
                    )
            except Exception:
                logging.exception("Could not persist paused state")
            console.print("\n[yellow]Translation paused.[/yellow] Resume with: "
                          f"tarjomeh jobs resume {pipeline.current_job_id}")
            pipeline.close()
            return 130
        except Exception as e:
            try:
                from tarjomeh.jobs.database import JobDatabase, JobStatus

                if pipeline.current_job_id:
                    db = JobDatabase()
                    current = db.get_job(pipeline.current_job_id)
                    if not current or current.get("raw_status") not in (
                        JobStatus.PAUSED,
                        JobStatus.PAUSED_ERROR,
                        JobStatus.COMPLETED,
                    ):
                        db.update_job_status(
                            pipeline.current_job_id, JobStatus.FAILED, str(e)
                        )
            except Exception:
                logging.exception("Could not persist failed state")
            console.print(f"\n[red]Error:[/red] {e}")
            logging.exception("Translation failed")
            pipeline.close()
            return 1

    pipeline.close()

    console.print(f"\n[bold green]✓ Translation complete![/bold green]")
    console.print(f"  Output: [cyan]{result.output_path}[/cyan]")
    console.print(f"  Chunks: {result.total_chunks}")
    console.print(f"  Duration: {result.duration_str}")
    if result.warnings:
        console.print(f"  Warnings: {len(result.warnings)}")
        for w in result.warnings[:5]:
            console.print(f"    [yellow]⚠[/yellow] {w}")

    return 0


def cmd_jobs(args: argparse.Namespace) -> int:
    """Execute jobs subcommands."""
    from tarjomeh.jobs.database import JobDatabase

    db = JobDatabase()

    if args.jobs_action == "list":
        jobs = db.list_jobs()
        if not jobs:
            console.print("[dim]No jobs found.[/dim]")
            return 0

        table = Table(title="Translation Jobs")
        table.add_column("Job ID", style="cyan", no_wrap=True)
        table.add_column("Input", style="green")
        table.add_column("Status", style="bold")
        table.add_column("Progress")
        table.add_column("Created", style="dim")

        for job in jobs:
            status_color = {
                "completed": "green",
                "running": "yellow",
                "paused": "yellow",
                "paused_error": "red",
                "failed": "red",
                "pending": "dim",
            }.get(job["status"], "white")

            table.add_row(
                job["id"][:8] + "...",
                Path(job["input_path"]).name,
                f"[{status_color}]{job['status']}[/{status_color}]",
                f"{job.get('progress', 0):.0f}%",
                job.get("created_at", ""),
            )

        console.print(table)
        return 0

    elif args.jobs_action == "status":
        if not args.job_id:
            console.print("[red]Error:[/red] Job ID required.")
            return 1
        job = db.get_job(args.job_id)
        if not job:
            console.print(f"[red]Error:[/red] Job {args.job_id} not found.")
            return 1

        console.print(f"\n[bold]Job:[/bold] {job['id']}")
        console.print(f"  Input: {job['input_path']}")
        console.print(f"  Status: {job['status']}")
        console.print(f"  Created: {job.get('created_at', 'N/A')}")
        if job.get("error_message"):
            console.print(f"  Error: [red]{job['error_message']}[/red]")

        chunks = db.get_chunk_summary(args.job_id)
        if chunks:
            console.print(f"\n  Chunks: {chunks['total']}")
            console.print(f"    Completed: [green]{chunks['completed']}[/green]")
            console.print(f"    Pending: [yellow]{chunks['pending']}[/yellow]")
            console.print(f"    Errors: [red]{chunks['errors']}[/red]")
        return 0

    elif args.jobs_action == "resume":
        if not args.job_id:
            console.print("[red]Error:[/red] Job ID required.")
            return 1
        console.print(f"Resuming job {args.job_id}...")
        # Re-run translate with resume flag
        from tarjomeh.core.config import TarjomehConfig
        from tarjomeh.core.pipeline import TranslationPipeline

        job = db.get_job(args.job_id)
        if not job:
            console.print(f"[red]Error:[/red] Job {args.job_id} not found.")
            return 1

        live_config = TarjomehConfig.load(validate=False)
        config = TarjomehConfig.from_dict(
            job.get("config", {}), credential_source=live_config
        )
        pipeline = TranslationPipeline(config)

        try:
            result = pipeline.run(
                input_path=Path(job["input_path"]),
                job_id=args.job_id,
            )
            console.print(f"[bold green]✓ Resume complete![/bold green] Output: {result.output_path}")
        except Exception as e:
            console.print(f"[red]Error resuming:[/red] {e}")
            return 1
        finally:
            pipeline.close()
        return 0

    elif args.jobs_action == "cleanup":
        if not args.job_id:
            console.print("[red]Error:[/red] Job ID required.")
            return 1
        deleted = db.cleanup_job(args.job_id)
        if deleted:
            console.print(f"[green]✓ Cleaned up job {args.job_id}[/green]")
        else:
            console.print(f"[yellow]No artifacts to clean for job {args.job_id}[/yellow]")
        return 0

    elif args.jobs_action == "export":
        if not args.job_id:
            console.print("[red]Error:[/red] Job ID required.")
            return 1

        from tarjomeh.core.config import TarjomehConfig
        from tarjomeh.core.pipeline import TranslationPipeline

        job = db.get_job(args.job_id)
        if not job:
            console.print(f"[red]Error:[/red] Job {args.job_id} not found.")
            return 1

        live_config = TarjomehConfig.load(validate=False)
        config = TarjomehConfig.from_dict(
            job.get("config", {}), credential_source=live_config
        )
        pipeline = TranslationPipeline(config)
        try:
            output = pipeline.export_completed_job(
                args.job_id,
                Path(args.output) if args.output else None,
                output_format=args.format,
                bilingual_mode=args.bilingual,
            )
        except Exception as e:
            console.print(f"[red]Export failed:[/red] {e}")
            return 1
        finally:
            pipeline.close()

        console.print(f"[bold green]✓ Exported without re-translating:[/bold green] {output}")
        return 0

    return 1


def cmd_glossary(args: argparse.Namespace) -> int:
    """Execute glossary subcommands."""
    from tarjomeh.glossary.manager import GlossaryManager

    if args.glossary_action == "validate":
        csv_path = Path(args.file)
        if not csv_path.exists():
            console.print(f"[red]Error:[/red] File not found: {csv_path}")
            return 1

        try:
            gm = GlossaryManager()
            gm.load(csv_path)
            console.print(f"[green]✓ Glossary valid![/green] {len(gm.entries)} terms loaded.")

            table = Table(title=f"Glossary: {csv_path.name}")
            table.add_column("Source", style="cyan")
            table.add_column("Target", style="green")
            table.add_column("Language")
            table.add_column("Domain", style="dim")

            for entry in gm.entries[:20]:
                table.add_row(entry.source, entry.target, entry.tgt_lng, entry.domain)

            if len(gm.entries) > 20:
                table.add_row("...", f"({len(gm.entries) - 20} more)", "", "")

            console.print(table)
        except Exception as e:
            console.print(f"[red]Glossary validation failed:[/red] {e}")
            return 1

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Compare two completed jobs without calling the translation pipeline."""
    from tarjomeh.quality.evaluation import (
        QualityEvaluator,
        evaluation_csv_report,
        evaluation_json_report,
        evaluation_text_report,
    )

    try:
        evaluation = QualityEvaluator().evaluate(args.baseline_job, args.candidate_job)
    except ValueError as exc:
        console.print(f"[red]Evaluation failed:[/red] {exc}")
        return 1

    renderers = {
        "text": evaluation_text_report,
        "json": evaluation_json_report,
        "csv": evaluation_csv_report,
    }
    report = renderers[args.format](evaluation)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        console.print(f"[green]Evaluation saved:[/green] {output_path}")
    else:
        console.print(report, markup=False, highlight=False)

    console.print(f"Evaluation ID: [cyan]{evaluation['id']}[/cyan]")
    blocked = bool(evaluation.get("summary", {}).get("release_blocked"))
    return 2 if args.fail_on_regression and blocked else 0


def cmd_capabilities(args: argparse.Namespace) -> int:
    """Print stable capabilities from the currently loaded installation."""
    import json

    from tarjomeh.runtime import runtime_behavior_probes, runtime_capabilities

    payload = runtime_capabilities()
    if args.verify:
        payload["behavior_probes"] = runtime_behavior_probes()
    if args.json:
        console.print(json.dumps(payload, ensure_ascii=False, indent=2), markup=False)
    else:
        console.print(f"Runtime release: [cyan]{payload['release']}[/cyan]")
        for name, enabled in payload["capabilities"].items():
            console.print(f"  {name}: {'yes' if enabled else 'no'}")
    probes = payload.get("behavior_probes", {})
    return 0 if all(probes.values()) else 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run an isolated model comparison through the configured endpoint."""
    from tarjomeh.core.config import TarjomehConfig
    from tarjomeh.quality.model_benchmark import (
        ModelBenchmark,
        load_benchmark_suite,
        write_benchmark_reports,
    )

    if args.benchmark_action != "compare":
        console.print("[red]Benchmark action required.[/red]")
        return 1
    try:
        config = TarjomehConfig.load(
            Path(args.config) if args.config else None
        )
        suite = load_benchmark_suite(Path(args.suite))
        report = ModelBenchmark(config).run(
            suite, list(args.models), args.judge_model
        )
        paths = write_benchmark_reports(report, Path(args.output_dir))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        console.print(f"[red]Benchmark failed:[/red] {exc}")
        return 1

    console.print(f"Benchmark ID: [cyan]{report['id']}[/cyan]")
    for rank, summary in enumerate(report["summaries"], 1):
        console.print(
            f"  {rank}. {summary['model']}: "
            f"quality={summary['average_quality_score']:.3f}, "
            f"disqualified={summary['disqualified_passages']}, "
            f"latency={summary['latency_seconds']:.3f}s"
        )
    console.print(f"Reports: [green]{paths['markdown'].parent}[/green]")
    console.print("No job, memory layer, or runtime model setting was changed.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the web UI server."""
    from tarjomeh.web.app import create_app
    from tarjomeh.core.config import TarjomehConfig

    config_path = Path(args.config) if args.config else None
    try:
        config = TarjomehConfig.load(config_path)
    except FileNotFoundError:
        config = TarjomehConfig()
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        return 1

    # ServerConfig is a dataclass, not a mapping. .get() raised
    # AttributeError whenever --host/--port were omitted, so `tarjomeh
    # serve` never started without both flags.
    host = args.host or config.server.host
    port = args.port or config.server.port

    _LOOPBACK = {"127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"}
    if host not in _LOOPBACK and not os.environ.get(
        "UI_SECRET_TOKEN", ""
    ).strip():
        console.print(
            f"[red]Refusing to start.[/red] Binding to {host} without "
            "UI_SECRET_TOKEN would expose every API route, including "
            "document download and job submission.\n"
            "  Set UI_SECRET_TOKEN, or bind to 127.0.0.1 and use an SSH tunnel."
        )
        return 1

    app = create_app(config)

    console.print(f"\n[bold cyan]ترجمه[/bold cyan] — Web UI")
    console.print(f"  Listening on: [green]http://{host}:{port}[/green]")
    if host == "127.0.0.1":
        console.print("  [dim]Bound to localhost. Use SSH tunnel for remote access.[/dim]")
    console.print()

    app.run(host=host, port=port, debug=args.debug)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="tarjomeh",
        description="ترجمه — Professional Academic Book Translation System (English → Persian)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # translate
    p_translate = subparsers.add_parser("translate", help="Translate a document")
    p_translate.add_argument("file", help="Input file path")
    p_translate.add_argument("-c", "--config", help="Config file path (default: config.toml)")
    p_translate.add_argument("-m", "--mode", choices=["fast", "quality", "academic"],
                             help="Translation mode")
    p_translate.add_argument("-f", "--output-format", choices=["pdf", "epub", "docx", "markdown", "txt", "srt"],
                             help="Output format")
    p_translate.add_argument("-o", "--output", help="Output file path")
    p_translate.add_argument("-b", "--bilingual", choices=["inline", "side_by_side", "target_only"],
                             help="Bilingual mode")
    p_translate.add_argument(
        "--term-notes",
        choices=["inline", "footnote", "endnote", "both"],
        help="First-occurrence English-original presentation",
    )
    p_translate.add_argument(
        "--book-research",
        action="store_true",
        help="Run the optional pre-translation research seed pass",
    )
    p_translate.add_argument("--provider", choices=["openrouter", "ollama"],
                             help="LLM provider")
    p_translate.add_argument("--model", help="LLM model name")
    p_translate.add_argument("--resume", help="Resume from job ID")

    # jobs
    p_jobs = subparsers.add_parser("jobs", help="Manage translation jobs")
    jobs_sub = p_jobs.add_subparsers(dest="jobs_action")
    jobs_sub.add_parser("list", help="List all jobs")
    p_status = jobs_sub.add_parser("status", help="Show job status")
    p_status.add_argument("job_id", help="Job ID")
    p_resume = jobs_sub.add_parser("resume", help="Resume a job")
    p_resume.add_argument("job_id", help="Job ID")
    p_cleanup = jobs_sub.add_parser("cleanup", help="Clean up job artifacts")
    p_cleanup.add_argument("job_id", help="Job ID")
    p_export = jobs_sub.add_parser("export", help="Re-export a completed job without translating")
    p_export.add_argument("job_id", help="Job ID")
    p_export.add_argument("--format", choices=["pdf", "epub", "docx", "markdown", "txt", "srt"], help="Output format")
    p_export.add_argument("-o", "--output", help="Output file path")
    p_export.add_argument("-b", "--bilingual", choices=["inline", "side_by_side", "target_only"], help="Bilingual mode")

    # glossary
    p_glossary = subparsers.add_parser("glossary", help="Glossary management")
    glossary_sub = p_glossary.add_subparsers(dest="glossary_action")
    p_validate = glossary_sub.add_parser("validate", help="Validate glossary CSV")
    p_validate.add_argument("file", help="Glossary CSV file path")

    # eval
    p_eval = subparsers.add_parser(
        "eval", help="Compare two persisted translation jobs"
    )
    p_eval.add_argument("baseline_job", help="Known baseline job ID")
    p_eval.add_argument("candidate_job", help="Candidate job ID")
    p_eval.add_argument(
        "-f", "--format", choices=["text", "json", "csv"], default="text",
        help="Report format (default: text)",
    )
    p_eval.add_argument("-o", "--output", help="Write the report to a file")
    p_eval.add_argument(
        "--fail-on-regression", action="store_true",
        help="Exit with status 2 when the candidate has a blocking regression",
    )

    p_capabilities = subparsers.add_parser(
        "capabilities", help="Show loaded runtime quality capabilities"
    )
    p_capabilities.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    p_capabilities.add_argument(
        "--verify", action="store_true",
        help="Run offline behavioral probes and fail if any probe fails",
    )

    p_benchmark = subparsers.add_parser(
        "benchmark", help="Run isolated model quality comparisons"
    )
    benchmark_sub = p_benchmark.add_subparsers(dest="benchmark_action")
    p_compare = benchmark_sub.add_parser(
        "compare", help="Compare translation routes on one frozen suite"
    )
    p_compare.add_argument(
        "--models", nargs="+", required=True,
        help="Two or more model or 9router route names",
    )
    p_compare.add_argument(
        "--judge-model", required=True,
        help="Independent judge model or 9router route name",
    )
    p_compare.add_argument(
        "--suite", default="benchmarks/suites/academic-core.json",
        help="Frozen benchmark suite JSON",
    )
    p_compare.add_argument(
        "--output-dir", default="benchmarks/runs",
        help="Parent directory for immutable run artifacts",
    )
    p_compare.add_argument("-c", "--config", help="Config file path")

    # serve
    p_serve = subparsers.add_parser("serve", help="Start web UI server")
    p_serve.add_argument("-c", "--config", help="Config file path")
    p_serve.add_argument("--host", help="Host to bind to")
    p_serve.add_argument("--port", type=int, help="Port to listen on")
    p_serve.add_argument("--debug", action="store_true", help="Enable debug mode")

    return parser


def main() -> int:
    """Main entry point for the CLI."""
    from dotenv import load_dotenv
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "translate": cmd_translate,
        "jobs": cmd_jobs,
        "glossary": cmd_glossary,
        "eval": cmd_evaluate,
        "capabilities": cmd_capabilities,
        "benchmark": cmd_benchmark,
        "serve": cmd_serve,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
