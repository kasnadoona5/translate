import json
from unittest.mock import MagicMock

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import TranslationPipeline
from tarjomeh.jobs.database import JobDatabase


def test_source_absent_name_never_enters_initial_proposals(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("The apparatus mediates an institutional order. " * 5, encoding="utf-8")
    config = TarjomehConfig()
    config.output.format = "txt"
    config.translation.review_book_terms_before_translating = True
    config.translation.enable_book_research = False
    config.translation.enable_web_context = False
    config.glossary.path, config.glossary.paths = "", []
    pipeline = TranslationPipeline(config)
    pipeline.db = JobDatabase(tmp_path / "jobs.db")
    pipeline.llm_client.complete = MagicMock(
        return_value=json.dumps(
            {
                "terms": [
                    {
                        "term": "Absent Person",
                        "suggested_persian": "شخص غایب",
                        "category": "person",
                    }
                ]
            }
        )
    )
    pipeline.run(source, tmp_path / "output.txt", job_id="sample-attestation")
    events = pipeline.db.get_chunk_events("sample-attestation", 0)
    completed = next(
        event["payload"] for event in events if event["event_type"] == "auto_extraction_completed"
    )
    assert completed["accepted_terms"] == 0
    assert completed["rejected_details"] == [
        {
            "source": "Absent Person",
            "reason": "source_term_not_in_frozen_extraction_sample",
        }
    ]
    assert pipeline.llm_client.complete.call_count == 1
    review = pipeline.db.get_job_artifact("sample-attestation", "book_term_review_v1")
    assert all(term["source"] != "Absent Person" for term in review["proposals"])
    assert not (tmp_path / "output.txt").exists()
