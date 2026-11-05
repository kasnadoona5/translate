"""JSON repair costs and logical invalidity are different report-only evidence."""
from copy import deepcopy

from tarjomeh.core.evidence_audit import refiner_recovery_reporting


def event(kind, **payload):
    return dict(chunk_index=0, event_type=kind, payload=payload)


def test_recovery_reports_existing_errors_without_double_counting_retried_event():
    payload = dict(iteration=1, attempts=3, recovered=True,
                   validation_errors=["invalid_json: Expecting value", "translation_is_required"])
    events = [event("refiner_response_invalid", **payload),
              event("refiner_response_retried", **payload),
              event("llm_call_attempt", operation="refinement_json_repair", success=True),
              event("llm_call_attempt", operation="refinement_json_repair", success=False)]
    before = deepcopy(events)
    report = refiner_recovery_reporting(events)
    assert events == before
    assert report["logical_invalid_records"] == 1
    assert report["recovered_logical_records"] == 1
    assert report["physical_json_repair_attempts"] == 2
    assert report["physical_json_repair_failures"] == 1
    assert report["validation_error_categories"] == {"invalid_json": 1, "translation_is_required": 1}


def test_extra_final_cost_does_not_invent_an_unrecorded_validation_error():
    report = refiner_recovery_reporting([event("llm_call_attempt",
        operation="refinement_json_repair", success=True,
        quality_attempt_context={"stage": "extra_final_refinement"})])
    assert report["logical_invalid_records"] == 0
    assert report["physical_json_repair_attempts"] == 1
    assert report["physical_json_repairs_by_stage"] == {"extra_final_refinement": 1}
    assert report["missing_logical_evidence_is_not_zero_invalidity"] is True


def test_clean_or_unparseable_evidence_never_changes_admission_or_exposes_raw_errors():
    assert refiner_recovery_reporting([])["physical_json_repair_attempts"] == 0
    report = refiner_recovery_reporting([event("refiner_response_invalid", recovered=False,
        validation_errors=["api_key=secret", "missing_issue_decision:private-id"] )])
    assert report["validation_error_categories"] == {"unclassified": 1, "missing_issue_decision": 1}
    assert "secret" not in str(report) and "private-id" not in str(report)
    assert report["authority"] == "report_only_no_admission_or_retry_change"
