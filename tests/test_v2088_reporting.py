"""Scoped v20.8.8 regression controls."""

import copy


def test_new_evidence_reports_never_mutate_or_readmit_historical_state():
    from tarjomeh.core.evidence_audit import checkpoint_classification_evidence

    state = {
        "last_rejected_summary": {
            "english_candidate": {"text": "There are two reasons: (1) alpha; (2) beta."},
            "persian_candidate": {"text": "دو دلیل: ۱) الف؛ ۲) ب."},
        }
    }
    before = copy.deepcopy(state)
    report = checkpoint_classification_evidence(state, {}, [])
    assert state == before
    assert report["historical_summary_readmitted"] is False
    assert not report["rejected_summary_current_structure"]["classifications"]
    assert report["semantic_accuracy"] == "requires_source_based_human_review"
