from tarjomeh.core.config import TarjomehConfig
from tarjomeh.memory.manager import MemoryManager


def test_selected_style_sample_keeps_its_own_source_paragraph():
    manager = MemoryManager(TarjomehConfig())
    sources = ["The first claim is conditional.", "The second claim has two parts."]
    targets = [
        "ادعای نخست به یک شرط وابسته است.",
        "ادعای دوم دو بخش مستقل دارد و هر دو بخش در استدلال مهم‌اند.",
    ]
    manager._update_style_profile(
        "\n\n".join(targets), source_paragraphs=sources,
        source_paragraph_indices=[0, 1], source_alignment_proven=True,
        final_scores=dict.fromkeys(("accuracy", "fluency", "terminology", "register"), 9.5),
    )
    assert manager.style_sample_records
    record = manager.style_sample_records[-1]
    assert record["source_text"] == sources[record["source_paragraph_index"]]
    assert record["alignment_status"] == "exact_paragraph"


def test_uncertain_style_record_does_not_enter_prompt_during_warmup():
    manager = MemoryManager(TarjomehConfig())
    manager.style_sample_records.append({
        "text": "سپاس از همهٔ کسانی که به این کتاب یاری رساندند.",
        "source_text": "Acknowledgements to the many contributors.",
        "representative": True,
        "quality_score": 100,
        "final_scores": dict.fromkeys(("accuracy", "fluency", "terminology", "register"), 9.5),
    })
    assert manager._render_style_profile() == ""
