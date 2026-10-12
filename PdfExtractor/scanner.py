from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .config import ExtractorConfig
from .utils import read_pdf_metadata, safe_slug, sha256_file


def scan_pdfs(config: ExtractorConfig) -> list[dict[str, Any]]:
    iterator = config.input_dir.rglob("*") if config.recursive else config.input_dir.glob("*")
    pdf_paths = sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: str(path).lower(),
    )
    if not pdf_paths:
        raise FileNotFoundError(
            f"Tidak ditemukan PDF di {config.input_dir} (recursive={config.recursive}). "
            "Periksa path dan ekstensi file."
        )

    records: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        relative_path = pdf_path.relative_to(config.input_dir).as_posix()
        try:
            file_hash = sha256_file(pdf_path)
            file_size = pdf_path.stat().st_size
            preflight = read_pdf_metadata(pdf_path)
            document_id = hashlib.sha256(
                f"{relative_path}\n{file_hash}".encode("utf-8")
            ).hexdigest()[:20]
            folder_name = f"{safe_slug(pdf_path.stem)}__{document_id}"
            records.append({
                "document_id": document_id,
                "file_name": pdf_path.name,
                "source_relative_path": relative_path,
                "source_path": str(pdf_path),
                "sha256": file_hash,
                "file_size_bytes": file_size,
                "doc_folder_name": folder_name,
                "doc_output_dir": config.documents_dir / folder_name,
                **preflight,
            })
        except Exception as exc:
            document_id = hashlib.sha256(str(pdf_path).encode("utf-8")).hexdigest()[:20]
            folder_name = f"{safe_slug(pdf_path.stem)}__{document_id}"
            records.append({
                "document_id": document_id,
                "file_name": pdf_path.name,
                "source_relative_path": relative_path,
                "source_path": str(pdf_path),
                "sha256": "",
                "file_size_bytes": None,
                "doc_folder_name": folder_name,
                "doc_output_dir": config.documents_dir / folder_name,
                "page_count_pdf": None,
                "pdf_title_metadata": "",
                "pdf_author_metadata": "",
                "preflight_warning": f"Gagal melakukan preflight: {type(exc).__name__}: {exc}",
                "preflight_error": f"{type(exc).__name__}: {exc}",
            })

    print(f"Ditemukan {len(records)} PDF")
    for item in records[:10]:
        print(
            f"- {item['source_relative_path']} | {item.get('page_count_pdf')} halaman | "
            f"{item.get('file_size_bytes')} byte"
        )
    if len(records) > 10:
        print(f"... dan {len(records) - 10} file lainnya")
    return records
