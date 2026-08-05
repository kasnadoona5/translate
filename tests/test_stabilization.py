from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock

from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.llm_client import (
    IncompleteCompletionError,
    LLMClient,
    TruncatedCompletionError,
)
from tarjomeh.core.term_notes import audit_inline_english_originals
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.exporters.docx_exporter import DocxExporter, HAS_DOCX
from tarjomeh.glossary.compliance import (
    GlossaryComplianceChecker,
    term_occurs_only_in_citations,
)
from tarjomeh.glossary.manager import GlossaryManager
from tarjomeh.quality.back_translator import _extract_entities
from tarjomeh.quality.critique import TranslationCritique


def _response(content: str, finish_reason: str = "stop") -> Mock:
    response = Mock()
    response.status_code = 200
    response.headers = {}
    response.raise_for_status = Mock()
    response.text = json.dumps({
        "choices": [{
            "finish_reason": finish_reason,
            "message": {"content": content},
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    })
    return response


class TestBoundedRecovery(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TarjomehConfig()
        self.config.llm.openrouter.api_keys = ["key"]
        self.config.retry.max_retries = 5
        self.config.retry.base_delay = 0
        self.config.retry.max_delay = 0
        self.config.retry.jitter = False
        self.config.llm.recovery.max_attempts = 3
        self.client = LLMClient(self.config)

    def tearDown(self) -> None:
        self.client.close()

    def test_defaults_recommend_nine_and_bound_recovery(self) -> None:
        defaults = TarjomehConfig()
        self.assertEqual(defaults.translation.critique_threshold, 9.0)
        self.assertEqual(defaults.llm.recovery.max_attempts, 2)
        self.assertEqual(defaults.llm.max_tokens, 12000)
        self.assertEqual(defaults.llm.recovery.max_tokens, 24000)
        self.assertTrue(defaults.llm.recovery.predictive_first_attempt)
        self.assertEqual(defaults.llm.recovery.adaptive_max_tokens, 65536)

    def test_first_attempt_changes_only_budget_then_length_recovers(self) -> None:
        messages = [{"role": "user", "content": "Translate this paragraph."}]
        _, _, expected = self.client._prepare_request(messages)
        payloads = []
        responses = iter([
            _response("partial", "length"),
            _response("complete", "stop"),
        ])

        def post(*args, **kwargs):
            payloads.append(json.loads(json.dumps(kwargs["json"])))
            return next(responses)

        events = []
        self.client._client.post = post
        self.client.set_attempt_observer(events.append)
        result = self.client.complete(messages=messages, _operation="translation")

        self.assertEqual(result, "complete")
        self.assertEqual(payloads[0]["messages"], expected["messages"])
        self.assertEqual(payloads[0]["model"], expected["model"])
        self.assertEqual(payloads[0]["temperature"], expected["temperature"])
        self.assertEqual(payloads[0]["reasoning"], expected["reasoning"])
        self.assertGreater(payloads[0]["max_tokens"], expected["max_tokens"])
        self.assertEqual(payloads[1]["reasoning"], expected["reasoning"])
        self.assertGreater(payloads[1]["max_tokens"], payloads[0]["max_tokens"])
        self.assertLessEqual(payloads[1]["max_tokens"], 65536)
        self.assertEqual([event["success"] for event in events], [False, True])
        self.assertEqual(events[0]["failure_reason"], "length")
        self.assertTrue(events[0]["normal_attempt"])
        self.assertEqual(
            events[0]["prompt_sha256"], events[1]["prompt_sha256"]
        )
        self.assertEqual(
            events[0]["preflight_calculation"]["stage"], "preflight"
        )
        calculation = events[1]["recovery_calculation"]
        self.assertEqual(calculation["uncertainty_multiplier"], 1.25)
        self.assertEqual(
            calculation["applied_max_tokens"], payloads[1]["max_tokens"]
        )

    def test_translation_recovery_uses_evidence_based_budget_formula(self) -> None:
        self.config.llm.recovery.max_tokens = 50000
        source = "Academic source sentence. " * 200
        partial = "\u062a\u0631\u062c\u0645\u0647 \u0646\u0627\u062a\u0645\u0627\u0645"
        first = _response(partial, "length")
        body = json.loads(first.text)
        body["usage"] = {
            "prompt_tokens": 3000,
            "completion_tokens": 10000,
            "total_tokens": 13000,
            "completion_tokens_details": {"reasoning_tokens": 9000},
        }
        first.text = json.dumps(body)
        payloads = []
        responses = iter([first, _response("complete", "stop")])

        def post(*args, **kwargs):
            payloads.append(json.loads(json.dumps(kwargs["json"])))
            return next(responses)

        events = []
        self.client._client.post = post
        self.client.set_attempt_observer(events.append)
        self.client.complete(
            messages=[{"role": "user", "content": "Translate."}],
            _operation="translation",
            _recovery_source_text=source,
        )

        calc = events[1]["recovery_calculation"]
        expected = __import__("math").ceil(
            (calc["answer_headroom_tokens"] + calc["reasoning_headroom_tokens"])
            * 1.25
        )
        self.assertEqual(calc["calculated_max_tokens"], expected)
        minimum = payloads[0]["max_tokens"] + max(
            1024, __import__("math").ceil(payloads[0]["max_tokens"] * 0.15)
        )
        self.assertEqual(payloads[1]["max_tokens"], max(expected, minimum))
        self.assertNotIn("_recovery_source_text", payloads[0])

    def test_recovery_normalizes_mismatched_reasoning_token_units(self) -> None:
        source = "Academic source sentence. " * 100
        first = _response("partial", "length")
        body = json.loads(first.text)
        body["usage"] = {
            "prompt_tokens": 4000,
            "completion_tokens": 12000,
            "total_tokens": 16000,
            "completion_tokens_details": {"reasoning_tokens": 20888},
        }
        first.text = json.dumps(body)
        responses = iter([first, _response("complete", "stop")])
        events = []
        self.client._client.post = lambda *args, **kwargs: next(responses)
        self.client.set_attempt_observer(events.append)

        self.client.complete(
            messages=[{"role": "user", "content": "Translate."}],
            _operation="translation",
            _recovery_source_text=source,
        )

        calculation = events[1]["recovery_calculation"]
        self.assertEqual(calculation["reported_reasoning_tokens"], 20888)
        self.assertEqual(calculation["reasoning_estimate_tokens"], 12000)

    def test_success_history_adapts_later_first_attempt_budget(self) -> None:
        payloads = []

        def post(*args, **kwargs):
            payloads.append(json.loads(json.dumps(kwargs["json"])))
            return _response("short complete answer", "stop")

        self.client._client.post = post
        messages = [{"role": "user", "content": "Extract proper nouns."}]
        for _ in range(4):
            self.client.complete(messages=messages, _operation="chat")

        self.assertGreater(payloads[0]["max_tokens"], 12000)
        self.assertEqual(payloads[3]["max_tokens"], 12000)

    def test_second_translation_attempt_uses_configured_fallback(self) -> None:
        self.config.llm.recovery.model = "recovery-combo"
        payloads = []
        responses = iter([
            _response("partial one", "length"),
            _response("partial two", "length"),
            _response("recovered", "stop"),
        ])

        def post(*args, **kwargs):
            payloads.append(dict(kwargs["json"]))
            return next(responses)

        self.client._client.post = post
        result = self.client.complete(
            messages=[{"role": "user", "content": "x"}],
            _operation="translation",
        )
        self.assertEqual(result, "recovered")
        self.assertEqual(payloads[0]["model"], self.config.llm.model)
        self.assertEqual(payloads[1]["model"], self.config.llm.model)
        self.assertEqual(payloads[2]["model"], "recovery-combo")
        self.assertGreater(payloads[1]["max_tokens"], payloads[0]["max_tokens"])
        self.assertGreaterEqual(payloads[2]["max_tokens"], payloads[1]["max_tokens"])
        self.assertEqual(payloads[1]["reasoning"], payloads[0]["reasoning"])
        self.assertEqual(payloads[2]["reasoning"], payloads[0]["reasoning"])

    def test_fourth_length_attempt_expands_only_after_unchanged_first_try(self) -> None:
        self.config.llm.recovery.max_attempts = 4
        self.config.llm.recovery.model = "critic-fallback"
        self.config.llm.recovery.max_tokens = 24000
        self.config.llm.recovery.expanded_final_attempt = True
        messages = [{"role": "user", "content": "Judge this translation."}]
        _, _, expected = self.client._prepare_request(messages)
        payloads = []
        responses = iter([
            _response("partial one", "length"),
            _response("partial two", "length"),
            _response("partial three", "length"),
            _response("recovered", "stop"),
        ])

        def post(*args, **kwargs):
            payloads.append(json.loads(json.dumps(kwargs["json"])))
            return next(responses)

        events = []
        self.client._client.post = post
        self.client.set_attempt_observer(events.append)
        result = self.client.complete(messages=messages, _operation="critique")

        self.assertEqual(result, "recovered")
        self.assertEqual(payloads[0]["messages"], expected["messages"])
        self.assertEqual(payloads[0]["reasoning"], expected["reasoning"])
        self.assertGreater(payloads[0]["max_tokens"], expected["max_tokens"])
        self.assertEqual(payloads[1]["model"], self.config.llm.model)
        self.assertGreater(payloads[1]["max_tokens"], payloads[0]["max_tokens"])
        self.assertEqual(payloads[2]["model"], "critic-fallback")
        self.assertGreaterEqual(payloads[2]["max_tokens"], payloads[1]["max_tokens"])
        self.assertEqual(payloads[3]["model"], "critic-fallback")
        self.assertGreaterEqual(payloads[3]["max_tokens"], 24000)
        self.assertEqual(payloads[3]["reasoning"]["effort"], "none")
        self.assertEqual(events[3]["recovery_stage"], "final_expanded")

    def test_critic_third_attempt_disables_reasoning_without_fallback(self) -> None:
        payloads = []
        responses = iter([
            _response("partial one", "length"),
            _response("partial two", "length"),
            _response("recovered", "stop"),
        ])

        def post(*args, **kwargs):
            payloads.append(json.loads(json.dumps(kwargs["json"])))
            return next(responses)

        self.client._client.post = post
        result = self.client.complete(
            messages=[{"role": "user", "content": "Judge this translation."}],
            _operation="critique",
        )

        self.assertEqual(result, "recovered")
        self.assertGreater(payloads[0]["max_tokens"], 12000)
        self.assertNotEqual(payloads[0].get("reasoning", {}).get("effort"), "none")
        self.assertEqual(payloads[1]["reasoning"]["effort"], "low")
        self.assertEqual(payloads[2]["reasoning"]["effort"], "none")
        self.assertGreaterEqual(payloads[2]["max_tokens"], 24000)

    def test_glossary_recovery_disables_reasoning_after_length(self) -> None:
        payloads = []
        responses = iter([
            _response("partial", "length"),
            _response("recovered", "stop"),
        ])

        def post(*args, **kwargs):
            payloads.append(json.loads(json.dumps(kwargs["json"])))
            return next(responses)

        self.client._client.post = post
        result = self.client.complete(
            messages=[{"role": "user", "content": "Repair terminology."}],
            _operation="glossary_auto_correction",
        )

        self.assertEqual(result, "recovered")
        self.assertNotEqual(payloads[0].get("reasoning", {}).get("effort"), "none")
        self.assertEqual(payloads[1]["reasoning"]["effort"], "none")
        self.assertGreaterEqual(payloads[1]["max_tokens"], 24000)

    def test_json_repair_is_compact_from_its_first_request(self) -> None:
        payloads = []

        def post(*args, **kwargs):
            payloads.append(json.loads(json.dumps(kwargs["json"])))
            return _response('{"scores": {}, "issues": []}', "stop")

        self.client._client.post = post
        self.client.complete(
            messages=[{"role": "user", "content": "Repair JSON."}],
            _operation="critique_json_repair",
        )

        self.assertEqual(payloads[0]["reasoning"]["effort"], "none")
        self.assertGreaterEqual(payloads[0]["max_tokens"], 24000)

    def test_partial_length_response_is_never_returned(self) -> None:
        self.config.llm.recovery.max_attempts = 1
        self.client._client.post = Mock(return_value=_response("partial", "length"))
        with self.assertRaises(TruncatedCompletionError):
            self.client.complete(messages=[{"role": "user", "content": "x"}])

    def test_assembles_openai_compatible_sse_response(self) -> None:
        body = "\n\n".join([
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
            '"model":"qwen3.7-plus","choices":[{"index":0,"delta":'
            '{"role":"assistant"},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
            '"model":"qwen3.7-plus","choices":[{"index":0,"delta":'
            '{"content":"Persian "},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
            '"model":"qwen3.7-plus","choices":[{"index":0,"delta":'
            '{"content":"translation"},"finish_reason":"stop"}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":10,'
            '"completion_tokens":2,"total_tokens":12}}',
            "data: [DONE]",
        ])

        parsed = LLMClient._parse_response_json(body)

        self.assertEqual(parsed["model"], "qwen3.7-plus")
        self.assertEqual(
            parsed["choices"][0]["message"]["content"],
            "Persian translation",
        )
        self.assertEqual(parsed["choices"][0]["finish_reason"], "stop")
        self.assertEqual(parsed["usage"]["total_tokens"], 12)

    def test_rejects_interrupted_sse_response(self) -> None:
        body = (
            'data: {"choices":[{"index":0,"delta":{"content":"partial"},'
            '"finish_reason":null}]}'
        )

        with self.assertRaises(IncompleteCompletionError):
            LLMClient._parse_response_json(body)


class TestStabilizationPolicies(unittest.TestCase):
    def test_final_original_audit_removes_grounded_noise_and_duplicates(self) -> None:
        document = TranslatedDocument(paragraphs=[
            TranslatedParagraph(
                index=0,
                source_text=(
                    "Marx (1973, 408) discussed ground and Monsanto. "
                    "The source adds (Editorial Note)."
                ),
                translated_text=(
                    "Marx-fa (Marx) (1973, 408) earth (ground) and "
                    "Monsanto-fa (Monsanto). (Editorial Note)"
                ),
            ),
            TranslatedParagraph(
                index=1,
                source_text="Monsanto returned.",
                translated_text="Monsanto-fa (Monsanto) returned.",
            ),
        ])
        report = audit_inline_english_originals(
            document,
            {"Marx": "Marx-fa", "Monsanto": "Monsanto-fa"},
        )
        combined = "\n".join(p.translated_text for p in document.paragraphs)
        self.assertNotIn("(ground)", combined)
        self.assertEqual(combined.count("(Monsanto)"), 1)
        self.assertIn("(1973, 408)", combined)
        self.assertIn("(Editorial Note)", combined)
        self.assertEqual(report["removed_unauthorized_count"], 1)
        self.assertEqual(report["removed_duplicate_count"], 1)
        self.assertEqual(report["preserved_citation_count"], 1)

    def test_noop_critic_issue_is_not_sent_to_refinement(self) -> None:
        result = TranslationCritique._parse_response(json.dumps({
            "scores": {
                "accuracy": 9, "fluency": 9, "terminology": 9, "register": 9,
            },
            "issues": [
                {
                    "severity": "minor",
                    "category": "register",
                    "current_translation": "correct",
                    "suggested_fix": "correct",
                    "explanation": "No change needed.",
                },
                {
                    "severity": "major",
                    "category": "accuracy",
                    "current_translation": "A",
                    "suggested_fix": "B",
                    "explanation": "Meaning differs.",
                },
            ],
        }))
        self.assertEqual(len(result.issues), 1)
        self.assertEqual(len(result.ignored_issue_details), 1)
        self.assertIn('current: "A"', result.issues[0])

    def test_noop_noise_cannot_invalidate_actionable_critique(self) -> None:
        long_rationale = "x" * 700
        result = TranslationCritique._parse_response(
            json.dumps({
                "scores": {
                    "accuracy": 9,
                    "fluency": 9,
                    "terminology": 9,
                    "register": 9,
                },
                "issues": [
                    {
                        "severity": "minor",
                        "category": "fluency",
                        "confidence": 0.5,
                        "source_quote": "not present",
                        "current_persian_quote": "also not present",
                        "suggested_correction": "also not present",
                        "rationale": "No change needed.",
                    },
                    {
                        "severity": "major",
                        "category": "accuracy",
                        "confidence": 0.9,
                        "source_quote": "field",
                        "current_persian_quote": "زمینه",
                        "suggested_correction": "میدان",
                        "rationale": long_rationale,
                    },
                ],
            }),
            source_text="The field matters.",
            translation="زمینه مهم است.",
        )

        self.assertTrue(result.valid)
        self.assertEqual(len(result.issues), 1)
        self.assertEqual(len(result.ignored_issue_details), 1)
        self.assertTrue(result.issue_details[0]["rationale_truncated"])
        self.assertEqual(len(result.issue_details[0]["rationale"]), 500)

    def test_one_ungrounded_issue_does_not_discard_valid_critic_items(self) -> None:
        result = TranslationCritique._parse_response(
            json.dumps({
                "scores": {
                    "accuracy": 8,
                    "fluency": 9,
                    "terminology": 9,
                    "register": 9,
                },
                "issues": [
                    {
                        "severity": "major",
                        "category": "accuracy",
                        "confidence": 0.9,
                        "source_quote": "field",
                        "current_persian_quote": "A",
                        "suggested_correction": "B",
                        "rationale": "The contextual sense is technical.",
                    },
                    {
                        "severity": "minor",
                        "category": "fluency",
                        "confidence": 0.8,
                        "source_quote": "invented quote",
                        "current_persian_quote": "A",
                        "suggested_correction": "C",
                        "rationale": "This quote is not grounded.",
                    },
                ],
            }),
            source_text="The field matters.",
            translation="A",
        )

        self.assertTrue(result.valid)
        self.assertEqual(len(result.issue_details), 1)
        self.assertEqual(len(result.ignored_issue_details), 1)
        self.assertEqual(
            result.ignored_issue_details[0]["ignored_reason"],
            "invalid_issue",
        )
        self.assertIn(
            "issue_source_quote_not_found",
            result.ignored_issue_details[0]["validation_errors"],
        )

    def test_author_year_citation_is_not_a_glossary_translation_target(self) -> None:
        glossary = GlossaryManager()
        glossary.add_term("Caceres", "کاسرس")
        checker = GlossaryComplianceChecker()

        source = "The claim has been documented (Caceres 2014, 22)."
        report = checker.check("این ادعا مستند شده است.", source, glossary)

        self.assertTrue(term_occurs_only_in_citations(source, "Caceres"))
        self.assertTrue(report.compliant)
        self.assertEqual(report.total_checked, 0)
        self.assertEqual(report.citation_exemptions, ["Caceres"])

    def test_glossary_term_in_prose_remains_enforced_despite_citation(self) -> None:
        glossary = GlossaryManager()
        glossary.add_term("Caceres", "کاسرس")
        checker = GlossaryComplianceChecker()

        source = "Caceres argues this point (Caceres 2014, 22)."
        report = checker.check("این نکته مطرح می‌شود.", source, glossary)

        self.assertFalse(term_occurs_only_in_citations(source, "Caceres"))
        self.assertFalse(report.compliant)
        self.assertEqual(report.total_checked, 1)

    def test_entity_extraction_does_not_join_heading_and_sentence(self) -> None:
        entities = _extract_entities(
            "Introduction\n\nImagine the scene. The Monsanto Company expanded."
        )
        self.assertNotIn("Introduction Imagine", entities)
        self.assertIn("The Monsanto Company", entities)


@unittest.skipUnless(HAS_DOCX, "python-docx is not installed")
class TestDocxPersianLayout(unittest.TestCase):
    def test_docx_contains_rtl_justification_and_heading_pagination(self) -> None:
        document = TranslatedDocument(paragraphs=[
            TranslatedParagraph(
                index=0,
                source_text="Introduction",
                translated_text="\u0645\u0642\u062f\u0645\u0647",
                heading_level=1,
            ),
            TranslatedParagraph(
                index=1,
                source_text="Monsanto changed the field.",
                translated_text=(
                    "\u0645\u0648\u0646\u0633\u0627\u0646\u062a\u0648 "
                    "(Monsanto) \u0645\u06cc\u062f\u0627\u0646 \u0631\u0627 "
                    "\u062a\u063a\u06cc\u06cc\u0631 \u062f\u0627\u062f."
                ),
            ),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rtl.docx"
            DocxExporter().export(document, path)
            with zipfile.ZipFile(path) as archive:
                xml = archive.read("word/document.xml").decode("utf-8")
        self.assertIn('w:val="both"', xml)
        self.assertIn("<w:bidi", xml)
        self.assertIn("<w:keepNext", xml)
        self.assertIn('w:cs="Vazirmatn"', xml)
        self.assertIn('w:ascii="Times New Roman"', xml)
