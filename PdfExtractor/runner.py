from __future__ import annotations

import json
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ExtractorConfig, package_version
from .utils import json_load

REQUIRED_OUTPUTS = ("markdown.md", "middle_json.json", "structured_content.json")


def output_artifacts_valid(output_dir: Path) -> bool:
    """Validate expected SDK artifacts and JSON shape without assuming all keys exist."""
    try:
        for name in REQUIRED_OUTPUTS:
            target = output_dir / name
            if not target.is_file() or target.stat().st_size == 0:
                return False
        middle = json_load(output_dir / "middle_json.json")
        structured = json_load(output_dir / "structured_content.json")
        if not isinstance(middle, dict) or not isinstance(structured, dict):
            return False
        pages = middle.get("pages") or middle.get("pdf_info") or []
        if not isinstance(pages, list) or not pages:
            return False
        # Empty Markdown can be legitimate for a damaged/blank PDF; it is recorded
        # as a quality warning downstream rather than treated as malformed output.
        (output_dir / "markdown.md").read_text(encoding="utf-8", errors="replace")
        return True
    except Exception:
        return False


def _reusable_success(item: dict[str, Any], config: ExtractorConfig, mineru_ver: str) -> bool:
    outdir = Path(item["doc_output_dir"])
    marker = outdir / "_SUCCESS.json"
    if not config.skip_valid_results or not marker.is_file() or not output_artifacts_valid(outdir):
        return False
    try:
        metadata = json_load(marker)
        return (
            metadata.get("source_sha256") == item.get("sha256")
            and metadata.get("tier") == config.tier
            and metadata.get("ocr_mode") == config.ocr_mode
            and metadata.get("mineru_version") == mineru_ver
            and metadata.get("cpu_only") == config.cpu_only
        )
    except Exception:
        return False


def process_one_pdf(item: dict[str, Any], config: ExtractorConfig, mineru_ver: str) -> dict[str, str]:
    pdf_path = Path(item["source_path"])
    outdir = Path(item["doc_output_dir"])
    if item.get("preflight_error"):
        return {
            "document_id": item["document_id"],
            "status": "FAILED_PREFLIGHT",
            "message": str(item["preflight_error"]),
        }
    if _reusable_success(item, config, mineru_ver):
        marker_data = json_load(outdir / "_SUCCESS.json")
        return {
            "document_id": item["document_id"],
            "status": str(marker_data.get("status", "SUCCESS")),
            "message": "Hasil valid dipakai ulang; parsing dilewati.",
        }

    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    print(
        f"\n[START] {item['source_relative_path']} | tier={config.tier} | "
        f"pages={item.get('page_count_pdf')} | CPU mode={config.cpu_only}",
        flush=True,
    )
    try:
        # Important: MinerU is imported only after configure_runtime() has set the
        # CPU backend environment in pipeline.py.
        from mineru.parser import parse
        from mineru.parser.writer import FileBasedDataWriter

        result = parse(
            str(pdf_path),
            tier=config.tier,
            ocr_mode=config.ocr_mode,
            page_range="all",
        )
        result.save(FileBasedDataWriter(str(outdir)))
        if not output_artifacts_valid(outdir):
            raise RuntimeError(
                "MinerU selesai tetapi markdown.md / middle_json.json / structured_content.json "
                "tidak lengkap atau middle_json.json tidak memiliki daftar halaman valid."
            )
        middle = json_load(outdir / "middle_json.json")
        pages = middle.get("pages") or middle.get("pdf_info") or []
        mineru_page_count = len(pages)
        warnings: list[str] = []
        expected_pages = item.get("page_count_pdf")
        if expected_pages is not None and expected_pages != mineru_page_count:
            warnings.append(
                f"page_count_mismatch: PDF={expected_pages}, MinerU={mineru_page_count}"
            )
        if not (outdir / "markdown.md").read_text(encoding="utf-8", errors="replace").strip():
            warnings.append("markdown_empty")
        status = "SUCCESS_WITH_WARNINGS" if warnings else "SUCCESS"
        marker_data = {
            "document_id": item["document_id"],
            "source_sha256": item["sha256"],
            "tier": config.tier,
            "ocr_mode": config.ocr_mode,
            "mineru_version": mineru_ver,
            "cpu_only": config.cpu_only,
            "small_backend": "onnx" if config.cpu_only else "auto",
            "vlm_engine": "llama-cpp" if config.cpu_only else "auto",
            "table_device": "cpu" if config.cpu_only else "auto",
            "started_at_utc": started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "page_count_mineru": mineru_page_count,
            "status": status,
            "warnings": warnings,
        }
        (outdir / "_SUCCESS.json").write_text(
            json.dumps(marker_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[DONE] {item['file_name']} | MinerU pages={mineru_page_count} | {status}",
            flush=True,
        )
        return {"document_id": item["document_id"], "status": status, "message": "; ".join(warnings)}
    except Exception as exc:
        error_data = {
            "document_id": item["document_id"],
            "source_sha256": item.get("sha256"),
            "tier": config.tier,
            "ocr_mode": config.ocr_mode,
            "mineru_version": mineru_ver,
            "cpu_only": config.cpu_only,
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "exception_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc()[-12000:],
        }
        (outdir / "_ERROR.json").write_text(
            json.dumps(error_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[ERROR] {item['file_name']}: {type(exc).__name__}: {exc}", flush=True)
        if not config.continue_on_error:
            raise
        return {
            "document_id": item["document_id"],
            "status": "FAILED",
            "message": f"{type(exc).__name__}: {exc}",
        }
