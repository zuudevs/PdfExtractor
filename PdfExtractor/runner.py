from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
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
            and metadata.get("runtime_mode", "cpu") == config.runtime_mode
        )
    except Exception:
        return False


def _process_one_pdf_impl(item: dict[str, Any], config: ExtractorConfig, mineru_ver: str) -> dict[str, str]:
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
        f"pages={item.get('page_count_pdf')} | device={config.runtime_mode}",
        flush=True,
    )
    try:
        # Important: MinerU is imported only after configure_runtime() has set the
        # selected CPU/GPU backend environment in pipeline.py.
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
            "runtime_mode": config.runtime_mode,
            "small_backend": "onnx" if config.runtime_mode == "cpu" else "torch",
            "vlm_engine": config.effective_vlm_engine or "not-used",
            "table_device": "cpu" if config.runtime_mode == "cpu" else "cuda",
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
            "runtime_mode": config.runtime_mode,
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



def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def process_one_pdf(item: dict[str, Any], config: ExtractorConfig, mineru_ver: str) -> dict[str, str]:
    """Run a document in a supervised process so native crashes can be recorded.

    Python cannot reliably catch a SIGSEGV raised by a native extension in the
    same process. The parent process can still create an _ERROR.json if the
    isolated MinerU worker dies before Python exception handling runs.
    """
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

    payload = {
        "item": _json_safe(item),
        "config": {
            "input_dir": str(config.input_dir),
            "output_dir": str(config.output_dir),
            "recursive": config.recursive,
            "tier": config.tier,
            "ocr_mode": config.ocr_mode,
            "skip_valid_results": config.skip_valid_results,
            "continue_on_error": config.continue_on_error,
            "model_source": config.model_source,
            "runtime_mode": config.runtime_mode,
            "vlm_engine": config.vlm_engine,
        },
        "mineru_version": mineru_ver,
    }
    outdir.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix="pdfextractor-worker-") as temp_dir:
        temp_path = Path(temp_dir)
        payload_path = temp_path / "payload.json"
        result_path = temp_path / "result.json"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "PdfExtractor.worker",
            str(payload_path),
            str(result_path),
        ]
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=os.environ.copy(),
            text=True,
            capture_output=True,
            errors="replace",
        )

        worker_log = (
            (completed.stdout or "")
            + ("\n[stderr]\n" + completed.stderr if completed.stderr else "")
        )
        if worker_log:
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "_WORKER.log").write_text(worker_log[-200_000:], encoding="utf-8")

        worker_result_error = ""
        if result_path.is_file():
            try:
                result_data = json.loads(result_path.read_text(encoding="utf-8"))
                result = result_data.get("result")
                if isinstance(result, dict) and "status" in result:
                    normalized_result = {str(k): str(v) for k, v in result.items()}
                    if normalized_result["status"].startswith("FAILED") and not (outdir / "_ERROR.json").is_file():
                        failure_details = {
                            "document_id": item["document_id"],
                            "source_path": item.get("source_path"),
                            "tier": config.tier,
                            "runtime_mode": config.runtime_mode,
                            "vlm_engine": config.effective_vlm_engine or "not-used",
                            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                            "exception_type": "WorkerFailure",
                            "error": normalized_result.get("message", "Worker gagal."),
                            "worker_log": str(outdir / "_WORKER.log") if worker_log else None,
                            "worker_traceback": result_data.get("traceback"),
                        }
                        outdir.mkdir(parents=True, exist_ok=True)
                        (outdir / "_ERROR.json").write_text(
                            json.dumps(failure_details, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                    return normalized_result
                worker_result_error = str(result_data.get("error", "Worker gagal tanpa detail."))
            except (OSError, json.JSONDecodeError) as exc:
                worker_result_error = f"Gagal membaca hasil worker: {type(exc).__name__}: {exc}"

        signal_number = -completed.returncode if completed.returncode < 0 else None
        try:
            signal_name = signal.Signals(signal_number).name if signal_number is not None else None
        except ValueError:
            signal_name = f"SIG{signal_number}" if signal_number is not None else None
        is_native_crash = signal_number is not None
        if is_native_crash:
            message = (
                f"Worker MinerU berhenti karena {signal_name} (exit code {completed.returncode}); "
                "kode native tidak sempat menghasilkan exception Python."
            )
        else:
            message = f"Worker MinerU berhenti dengan exit code {completed.returncode} tanpa hasil JSON."
        if worker_result_error:
            message += " " + worker_result_error

        error_data = {
            "document_id": item["document_id"],
            "source_path": item.get("source_path"),
            "source_sha256": item.get("sha256"),
            "tier": config.tier,
            "ocr_mode": config.ocr_mode,
            "mineru_version": mineru_ver,
            "runtime_mode": config.runtime_mode,
            "small_backend": "onnx" if config.runtime_mode == "cpu" else "torch",
            "vlm_engine": config.effective_vlm_engine or "not-used",
            "table_device": "cpu" if config.runtime_mode == "cpu" else "cuda",
            "started_at_utc": started,
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "exception_type": "NativeProcessCrash" if is_native_crash else "WorkerProcessExit",
            "error": message,
            "exit_code": completed.returncode,
            "signal": signal_name,
            "worker_command": command,
            "worker_log": str(outdir / "_WORKER.log") if worker_log else None,
            "stdout_tail": (completed.stdout or "")[-6000:],
            "stderr_tail": (completed.stderr or "")[-6000:],
        }
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "_ERROR.json").write_text(
            json.dumps(error_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[ERROR] {item.get('file_name', item['document_id'])}: {message}", flush=True)
        if not config.continue_on_error:
            raise RuntimeError(message)
        return {
            "document_id": str(item["document_id"]),
            "status": "FAILED_NATIVE_CRASH" if is_native_crash else "FAILED_WORKER",
            "message": message,
        }
