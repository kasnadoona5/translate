from __future__ import annotations

import ast
import asyncio
import copy
import json
import re
from pathlib import Path

from tarjomeh.chunking.chunker import Chunk
from tarjomeh.core.config import TarjomehConfig
from tarjomeh.core.pipeline import (
    _blocking_structure_findings,
    _source_foreign_expression_inventory,
    audit_canonical_document_identity,
)
from tarjomeh.exporters.base import TranslatedDocument, TranslatedParagraph
from tarjomeh.memory.manager import MemoryManager
from tarjomeh.memory.proper_nouns import (
    automatic_terminology_risk_reasons,
    observed_bilingual_target,
)
from tarjomeh.quality.model_benchmark import (
    ModelBenchmark,
    deterministic_source_checks,
    write_benchmark_reports,
)
from tarjomeh.quality.structure_audit import (
    TRANSLATION_STRUCTURE_MISMATCH,
    announced_count_evidence,
    audit_payload,
    audit_structure,
)
from tarjomeh.runtime import runtime_capabilities


def test_v1027_audit_scripts_embed_valid_python() -> None:
    for relative in (
        "scripts/audit_tarjomeh_v1027_reports.sh",
        "scripts/audit_tarjomeh_v1027_companion.sh",
    ):
        source = Path(relative).read_text(encoding="utf-8")
        blocks = re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY(?:\r?\n|$)", source, re.S)
        assert blocks, relative
        for block in blocks:
            ast.parse(block, filename=relative)


def test_v1027_deployment_is_tarjomeh_scoped_and_rollback_safe() -> None:
    deployment = Path("scripts/deploy_tarjomeh_v1027.sh").read_text(
        encoding="utf-8"
    )
    reports = Path("scripts/audit_tarjomeh_v1027_reports.sh").read_text(
        encoding="utf-8"
    )
    companion = Path("scripts/audit_tarjomeh_v1027_companion.sh").read_text(
        encoding="utf-8"
    )

    assert 'TAG="v10.27.0"' in deployment
    assert 'git diff --quiet v10.26.0 "$TAG"' in deployment
    assert "rollback_tarjomeh_on_error" in deployment
    assert deployment.count("tarjomeh capabilities --verify --json") == 2
    for identity in (
        "NINE_IMAGE", "NINE_STARTED", "NINE_CONTAINER_ID", "NINE_MOUNTS"
    ):
        assert identity in deployment
    for unsafe in (
        "docker system prune", "docker image prune", "docker builder prune",
        'docker rm -f 9router', 'docker image rm "$NINE_IMAGE"',
    ):
        assert unsafe not in deployment
    assert 'local JOB="${1:-LATEST}"' in reports
    assert 'local JOB="${1:-LATEST}"' in companion


def test_typed_announcement_does_not_bind_source_count_to_chapter_reference() -> None:
    source = (
        "The discussion draws on eight sources, including the account in chapter 5. "
        "First, it considers institutions; second, relations; third, strategy."
    )
    target = (
        "این بحث بر هشت منبع، از جمله روایت فصل ۵، تکیه دارد. "
        "نخست نهادها، دوم روابط و سوم راهبرد را بررسی می‌کند."
    )

    evidence = announced_count_evidence(source)

    assert [(item.value, item.semantic_category) for item in evidence] == [
        (8, "source")
    ]
    assert audit_structure(source, target) == []


def test_typed_announcement_still_blocks_real_source_wording_change() -> None:
    payload = audit_payload(
        "This chapter addresses two issues.",
        "این فصل به سه مسئله می‌پردازد.",
    )

    assert payload["classifications"] == [TRANSLATION_STRUCTURE_MISMATCH]
    finding = payload["findings"][0]
    assert finding["details"]["semantic_category"] == "argument_item"
    assert finding["details"]["source_announcement"]["exact_span"] == "two issues"
    assert finding["details"]["candidate_announcement"]["exact_span"] == "سه مسئله"
    assert _blocking_structure_findings([finding]) == [finding]


def test_review_only_structure_evidence_cannot_stop_sequential_admission() -> None:
    finding = {
        "classification": TRANSLATION_STRUCTURE_MISMATCH,
        "details": {"admission": "review", "evidence_level": "ambiguous"},
    }

    assert _blocking_structure_findings([finding]) == []


def test_observed_organization_anchor_drops_only_leading_context_syntax() -> None:
    english = "Economic and Social Science Research Council"
    translation = (
        "این طرح را که شورای پژوهش‌های اقتصادی و اجتماعی "
        f"({english}) پشتیبانی کرد، بررسی می‌کنیم."
    )

    assert observed_bilingual_target(
        translation, english, category="organization"
    ) == "شورای پژوهش‌های اقتصادی و اجتماعی"


def test_incomplete_coordinated_term_cannot_become_canonical() -> None:
    reasons = automatic_terminology_risk_reasons(
        "institutional isomorphism or complementarity",
        "مکمل‌بودن نهادی",
    )

    assert "coordinated_source_target_incomplete" in reasons


class _EntityResponse:
    async def chat(self, _prompt: str) -> str:
        return (
            '[{"term":"Emmerich de Vattel",'
            '"suggested_persian":"امریش دو واتل",'
            '"category":"technical_loanword",'
            '"context_independent":true}]'
        )

    def set_operation(self, _operation: str) -> None:
        return None


def test_source_entity_role_overrides_mislabeled_technical_loanword() -> None:
    manager = MemoryManager(TarjomehConfig())
    source = "Emmerich de Vattel (1714) developed the argument."
    target = "امریش دو واتل (Emmerich de Vattel, 1714) این استدلال را بسط داد."

    result = asyncio.run(manager.update_proper_nouns(
        _EntityResponse(),
        source,
        target,
        source_categories={"Emmerich de Vattel": "person"},
    ))

    assert result["accepted_count"] == 1
    assert manager.proper_nouns.category_for("Emmerich de Vattel") == "person"


_STYLE_SAMPLE_A = (
    "این استدلال نشان می‌دهد که تحلیل نهادی، بدون توجه به روابط اجتماعی، "
    "نمی‌تواند دگرگونی تاریخی قدرت سیاسی را به‌درستی توضیح دهد."
)
_STYLE_SAMPLE_B = (
    "ازاین‌رو، نویسنده میان صورت حقوقی دولت و مناسباتی که قدرت آن را شکل "
    "می‌دهند تمایزی روشن و از نظر تحلیلی ثمربخش برقرار می‌کند."
)
_STYLE_SAMPLE_C = (
    "این رویکرد، ضمن حفظ دقت مفهومی، پیوند میان ساختارهای نهادی و راهبردهای "
    "کنشگران را در بستر تاریخی گسترده‌تری بررسی می‌کند."
)
_STYLE_SCORES = {
    "accuracy": 9.5,
    "fluency": 9.5,
    "terminology": 9.5,
    "register": 9.5,
}


def test_legacy_style_checkpoint_migrates_as_fallback_evidence() -> None:
    manager = MemoryManager(TarjomehConfig())

    manager.from_dict({"style_samples": [_STYLE_SAMPLE_A]})

    assert manager._style_profile_status() == "warming_up"
    assert manager.style_sample_records[0]["fallback"] is True
    assert manager.style_sample_records[0]["representative"] is False


def test_only_representative_body_prose_establishes_style_profile() -> None:
    manager = MemoryManager(TarjomehConfig())

    fallback = manager._update_style_profile(
        _STYLE_SAMPLE_A,
        paragraph_role="body",
        book_genre="academic",
        representative=False,
    )
    manager._update_style_profile(
        _STYLE_SAMPLE_B,
        paragraph_role="body",
        book_genre="academic",
        representative=True,
        final_scores=_STYLE_SCORES,
    )
    final = manager._update_style_profile(
        _STYLE_SAMPLE_C,
        paragraph_role="body",
        book_genre="academic",
        representative=True,
        final_scores=_STYLE_SCORES,
    )

    assert fallback["profile_status"] == "warming_up"
    assert final["representative_sample_count"] == 2
    assert final["profile_status"] == "warming_up"

    fourth = manager._update_style_profile(
        "در این چشم‌انداز، توضیح دگرگونی سیاسی مستلزم آن است که سازوکارهای "
        "نهادی و انتخاب‌های راهبردی را در پیوندی تاریخی و منسجم با یکدیگر بسنجیم.",
        paragraph_role="body",
        book_genre="academic",
        representative=True,
        final_scores=_STYLE_SCORES,
    )
    assert fourth["profile_status"] == "established"


def test_genre_guidance_changes_style_dimensions_not_terminology_authority() -> None:
    academic = MemoryManager(TarjomehConfig())
    academic._update_style_profile(
        _STYLE_SAMPLE_A,
        paragraph_role="body",
        book_genre="academic",
        representative=True,
        final_scores=_STYLE_SCORES,
    )

    literary_config = TarjomehConfig()
    literary_config.translation.style_register = "literary"
    literary = MemoryManager(literary_config)
    literary._update_style_profile(
        _STYLE_SAMPLE_A,
        paragraph_role="body",
        book_genre="literary",
        representative=True,
        final_scores=_STYLE_SCORES,
    )

    assert "argument structure" in academic.style_profile
    assert "narrative voice" in literary.style_profile
    assert "not terminology authority" in academic.style_profile
    assert "not terminology authority" in literary.style_profile


def test_runtime_capability_manifest_is_versioned_and_complete() -> None:
    manifest = runtime_capabilities()

    assert manifest["release"] == "v10.35.0"
    assert manifest["revision"] >= 1
    assert all(manifest["capabilities"].values())
    assert manifest["capabilities"]["four_layer_memory"] is True
    assert manifest["capabilities"]["refiner_issue_veto"] is True
    assert manifest["capabilities"]["checkpoint_preview_atomic_publish"] is True


def test_canonical_document_identity_ignores_layout_whitespace_only() -> None:
    chunks = [
        Chunk(index=0, text="A", chapter_title="", section_title=""),
        Chunk(index=1, text="B", chapter_title="", section_title=""),
    ]
    translations = {0: "ترجمهٔ نخست.", 1: "ترجمهٔ دوم."}
    document = TranslatedDocument(paragraphs=[
        TranslatedParagraph(0, "A", "ترجمهٔ نخست."),
        TranslatedParagraph(1, "B", "  ترجمهٔ دوم.  "),
    ])

    audit = audit_canonical_document_identity(document, chunks, translations)

    assert audit["lexically_identical"] is True
    assert audit["expected_hash"] == audit["assembled_hash"]


def test_canonical_document_identity_detects_export_text_drift() -> None:
    chunks = [Chunk(index=0, text="A", chapter_title="", section_title="")]
    document = TranslatedDocument(paragraphs=[
        TranslatedParagraph(0, "A", "ترجمهٔ تغییرکرده.")
    ])

    audit = audit_canonical_document_identity(
        document, chunks, {0: "ترجمهٔ پذیرفته‌شده."}
    )

    assert audit["lexically_identical"] is False


def test_source_foreign_expression_inventory_is_orthography_grounded() -> None:
    source = (
        "The account studies the longue durée and the author's later work, "
        "including Emmerich de Vattel."
    )

    assert _source_foreign_expression_inventory(source) == ["longue durée"]


def test_source_foreign_expression_inventory_accepts_internal_apostrophe() -> None:
    assert _source_foreign_expression_inventory(
        "The argument invokes raison d'état in this passage."
    ) == ["raison d'état"]


class _BenchmarkClient:
    translations = {
        "route-a": "این فصل به دو مسئله می‌پردازد.",
        "route-b": "این فصل به سه مسئله می‌پردازد.",
    }

    def __init__(self, config: TarjomehConfig) -> None:
        self.model = config.llm.model
        self.observer = None

    def set_attempt_observer(self, observer: object) -> None:
        self.observer = observer

    def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        operation = kwargs.get("_operation")
        if self.observer:
            self.observer({
                "success": True,
                "model": self.model,
                "response_model": f"served-{self.model}",
                "prompt_tokens": 10,
                "completion_tokens": 5,
            })
        if operation == "benchmark_judge":
            return json.dumps({
                "scores": {
                    "accuracy": 10,
                    "completeness_structure": 10,
                    "fluency": 10,
                    "terminology": 10,
                    "register": 10,
                },
                "blocking_source_error": False,
                "findings": [],
                "summary": "synthetic",
            })
        return self.translations[self.model]

    def close(self) -> None:
        return None


def test_offline_benchmark_disqualifies_source_structure_regression(tmp_path: object) -> None:
    from pathlib import Path

    config = TarjomehConfig()
    original = copy.deepcopy(config.to_dict())
    suite = {
        "schema": 1,
        "id": "test",
        "description": "test",
        "passages": [{
            "id": "count",
            "source": "This chapter addresses two issues.",
            "domain": "test",
            "notes": "",
        }],
    }
    report = ModelBenchmark(config, _BenchmarkClient).run(
        suite, ["route-a", "route-b"], "judge"
    )

    assert report["summaries"][0]["model"] == "route-a"
    bad = next(item for item in report["results"] if item["model"] == "route-b")
    assert bad["eligible"] is False
    assert "explicit_source_structure_changed" in bad["deterministic"]["failures"]
    assert bad["route_evidence"]["translation"]["served_models"] == {
        "served-route-b": 1
    }
    assert config.to_dict() == original

    paths = write_benchmark_reports(report, Path(str(tmp_path)))
    assert all(path.is_file() for path in paths.values())


def test_benchmark_source_checks_preserve_numbers_and_paragraphs() -> None:
    checks = deterministic_source_checks(
        "There were 4 cases.\n\nA second paragraph follows.",
        "چهار مورد وجود داشت.",
    )

    assert checks["passed"] is False
    assert "missing_or_changed_number" in checks["failures"]
    assert "paragraph_count_changed" in checks["failures"]


def test_benchmark_requires_an_independent_judge_route() -> None:
    config = TarjomehConfig()
    suite = {
        "id": "test",
        "passages": [{"id": "one", "source": "One source sentence."}],
    }

    try:
        ModelBenchmark(config, _BenchmarkClient).run(
            suite, ["route-a", "route-b"], "route-a"
        )
    except ValueError as exc:
        assert "independent" in str(exc)
    else:
        raise AssertionError("A translation route was accepted as its own judge")
