from __future__ import annotations

from collections import Counter
from typing import Any

from .config import ExtractorConfig, package_version
from .database import export_database
from .model_setup import prepare_models
from .normalizer import normalize_all
from .runner import process_one_pdf
from .scanner import scan_pdfs


def run_pipeline(config: ExtractorConfig, prepare_model_files: bool = False) -> dict[str, Any]:
    # Configure runtime before importing MinerU in runner.py.
    config.configure_runtime()
    mineru_ver = package_version("mineru")
    if mineru_ver == "not-installed" or not mineru_ver.startswith("4."):
        raise RuntimeError(
            f"Paket MinerU 4.x harus terpasang, tetapi versi ditemukan {mineru_ver!r}. "
            "Install requirements.txt di environment Python yang sama dengan perintah ini."
        )
    print(f"MinerU version : {mineru_ver}")
    print(f"Runtime mode   : {config.runtime_mode.upper()}")
    print(f"Small backend  : {'ONNX' if config.runtime_mode == 'cpu' else 'Torch'}")
    print(f"VLM engine     : {config.effective_vlm_engine or 'not used for this tier'}")
    print(f"Table device   : {'CPU' if config.runtime_mode == 'cpu' else 'CUDA'}")
    print(f"Tier           : {config.tier}")
    print(f"OCR mode       : {config.ocr_mode}")
    print(f"Input          : {config.input_dir}")
    print(f"Output         : {config.output_dir}")

    config.documents_dir.mkdir(parents=True, exist_ok=True)
    config.database_dir.mkdir(parents=True, exist_ok=True)

    input_records = scan_pdfs(config)
    if prepare_model_files:
        prepare_models(config)

    run_results: list[dict[str, str]] = []
    for number, item in enumerate(input_records, start=1):
        print(f"\n===== {number}/{len(input_records)} =====", flush=True)
        result = process_one_pdf(item, config, mineru_ver)
        run_results.append(result)
        if result["status"].startswith("FAILED") and not config.continue_on_error:
            raise RuntimeError(result["message"])

    print("\nRingkasan batch:")
    status_summary = dict(Counter(row["status"] for row in run_results))
    print(status_summary)

    documents, elements, text_blocks, tables, figures, image_inventory = normalize_all(
        input_records, config, mineru_ver
    )
    db_path = export_database(
        config, documents, elements, text_blocks, tables, figures, image_inventory
    )
    return {
        "db_path": str(db_path),
        "status_summary": status_summary,
        "document_count": len(documents),
        "element_count": len(elements),
        "text_block_count": len(text_blocks),
        "table_count": len(tables),
        "figure_count": len(figures),
        "image_count": len(image_inventory),
    }
