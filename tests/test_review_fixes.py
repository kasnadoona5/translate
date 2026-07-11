from __future__ import annotations

import gc
import tempfile
from pathlib import Path

import pytest

from tarjomeh.context.search_providers import DuckDuckGoProvider
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.glossary.compliance import GlossaryComplianceChecker
from tarjomeh.jobs.database import JobDatabase


def test_glossary_compliance_uses_persian_word_boundaries() -> None:
    present = GlossaryComplianceChecker._target_present
    assert present("کلی", "این یک مفهوم کلی است")
    assert present("کلی", "مفاهیم کلی‌ها")
    assert not present("کلی", "این نمونه شکلی است")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bilingual_mode", "unknown"),
        ("max_refine_iterations", 6),
        ("critique_threshold", 10.5),
    ],
)
def test_new_server_side_bounds_are_validated(field: str, value: object) -> None:
    config = TarjomehConfig()
    if field == "bilingual_mode":
        config.output.bilingual_mode = value
    else:
        setattr(config.translation, field, value)
    with pytest.raises(ValueError):
        config.validate()


def test_job_history_reports_real_output_name_and_format() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = JobDatabase(Path(tmp) / "jobs.db")
        db.create_job(
            "job-1",
            "source.pdf",
            {"output": {"format": "docx"}},
        )
        db.update_job_status(
            "job-1",
            "completed",
            output_path=Path(tmp) / "source_translated.docx",
        )
        job = db.list_jobs()[0]
        assert job["filename"] == "source.pdf"
        assert job["output_filename"] == "source_translated.docx"
        assert job["output_format"] == "docx"
        del db
        gc.collect()


def test_duckduckgo_provider_has_stable_name() -> None:
    assert DuckDuckGoProvider().name == "duckduckgo"