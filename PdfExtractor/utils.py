from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safe_slug(value: str, max_len: int = 70) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return value[:max_len] or "document"


def read_pdf_metadata(pdf_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "page_count_pdf": None,
        "pdf_title_metadata": "",
        "pdf_author_metadata": "",
        "preflight_warning": "",
    }
    try:
        with Path(pdf_path).open("rb") as stream:
            head = stream.read(1024)
        if b"%PDF-" not in head:
            result["preflight_warning"] = (
                "PDF signature %PDF- tidak ditemukan dalam 1024 byte pertama. "
                "File tetap akan dicoba oleh MinerU."
            )
    except Exception as exc:
        result["preflight_warning"] = f"Gagal membaca header PDF: {type(exc).__name__}: {exc}"

    try:
        reader = PdfReader(str(pdf_path), strict=False)
        if reader.is_encrypted:
            msg = "PDF terenkripsi; ekstraksi dapat gagal atau tidak lengkap."
            result["preflight_warning"] = (
                f"{result['preflight_warning']} | {msg}" if result["preflight_warning"] else msg
            )
        result["page_count_pdf"] = len(reader.pages)
        metadata = reader.metadata
        if metadata:
            result["pdf_title_metadata"] = str(metadata.title or "").strip()
            result["pdf_author_metadata"] = str(metadata.author or "").strip()
    except Exception as exc:
        msg = f"Metadata PDF tidak terbaca via pypdf: {type(exc).__name__}: {exc}"
        result["preflight_warning"] = (
            f"{result['preflight_warning']} | {msg}" if result["preflight_warning"] else msg
        )
    return result


def json_load(path: Path) -> Any:
    import json

    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)
