from __future__ import annotations

import html as html_lib
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterator

from .config import ExtractorConfig, package_version
from .utils import json_load

TEXT_KEYS = {"content", "text", "latex", "markdown", "title", "value", "caption", "footnote"}
DOI_REGEX = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def get_pages_and_blocks(middle_data: dict[str, Any]) -> Iterator[tuple[Any, list[Any]]]:
    """Yield (page_idx, blocks), supporting current and older middle JSON shapes."""
    pages = middle_data.get("pages")
    if isinstance(pages, list) and pages:
        for ordinal, page in enumerate(pages):
            if not isinstance(page, dict):
                continue
            page_idx = page.get("page_idx", page.get("page_index", ordinal))
            blocks = page.get("blocks")
            if not isinstance(blocks, list):
                blocks = []
            yield page_idx, blocks
        return

    legacy_pages = middle_data.get("pdf_info")
    if isinstance(legacy_pages, list):
        for ordinal, page in enumerate(legacy_pages):
            if not isinstance(page, dict):
                continue
            page_idx = page.get("page_idx", ordinal)
            blocks = page.get("para_blocks")
            if not isinstance(blocks, list) or not blocks:
                blocks = []
                for key in ("tables", "images", "interline_equations", "discarded_blocks"):
                    candidate = page.get(key, [])
                    if isinstance(candidate, list):
                        blocks.extend(candidate)
            yield page_idx, blocks


def walk_blocks(
    blocks: list[Any], page_idx: Any, parent_path: str = "", parent_type: str = ""
) -> Iterator[tuple[Any, str, str, str, dict[str, Any]]]:
    """Walk nested semantic blocks; line/span content remains in each raw block."""
    if not isinstance(blocks, list):
        return
    for ordinal, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        idx = block.get("index", ordinal)
        current_path = f"{parent_path}.{idx}" if parent_path else str(idx)
        yield page_idx, current_path, parent_path, parent_type, block
        children = block.get("blocks")
        if isinstance(children, list):
            yield from walk_blocks(children, page_idx, current_path, str(block.get("type", "")))


def block_type(block: dict[str, Any]) -> str:
    return str(block.get("type") or block.get("category") or "unknown").strip().lower().replace(" ", "_")


def own_text_and_html(block: dict[str, Any]) -> tuple[str, str]:
    texts: list[str] = []
    htmls: list[str] = []

    def add_unique(target: list[str], value: str) -> None:
        value = str(value).strip()
        if value and value not in target:
            target.append(value)

    def visit(obj: Any, key_name: str | None = None) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in {"blocks", "pages", "page_info"}:
                    continue
                if key.lower() == "html" and isinstance(value, str):
                    add_unique(htmls, value)
                elif key.lower() in TEXT_KEYS and isinstance(value, str):
                    add_unique(texts, value)
                elif isinstance(value, (dict, list)):
                    visit(value, key)
        elif isinstance(obj, list):
            for value in obj:
                visit(value, key_name)

    visit(block)
    text_value = "\n".join(texts).strip()
    html_value = "\n".join(htmls).strip()
    if not text_value and html_value:
        plain = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html_value)
        plain = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|h[1-6])>", "\n", plain)
        plain = re.sub(r"(?s)<[^>]+>", " ", plain)
        plain = html_lib.unescape(plain)
        plain = re.sub(r"[ \t\r\f\v]+", " ", plain)
        plain = re.sub(r"\n\s*\n+", "\n", plain).strip()
        text_value = plain
    return text_value, html_value


def descendant_blocks(block: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield block
    children = block.get("blocks") if isinstance(block, dict) else None
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                yield from descendant_blocks(child)


def all_strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from all_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from all_strings(value)


def extract_asset_paths(obj: Any) -> list[str]:
    paths: set[str] = set()
    path_pattern = re.compile(r"(?:\./)?images/[^\s\"'< >\)\]\}]+", re.IGNORECASE)
    for value in all_strings(obj):
        for match in path_pattern.findall(value):
            cleaned = match.removeprefix("./").strip(".,;:!?`'")
            if cleaned.startswith("images/"):
                paths.add(cleaned)
    return sorted(paths)


def bbox_value(block: dict[str, Any]) -> Any:
    for key in ("bbox", "bbox_2d", "box"):
        value = block.get(key)
        if value is not None:
            return value
    return None


def is_table_type(element_type: str) -> bool:
    return element_type in {"table", "table_block"}


def is_figure_type(element_type: str) -> bool:
    return element_type in {"image", "figure", "chart", "figure_block", "image_block", "chart_block"}


def classify_text_type(element_type: str) -> bool:
    if any(term in element_type for term in ("table", "image", "figure", "chart")):
        return False
    return any(
        term in element_type
        for term in (
            "text", "title", "caption", "footnote", "paragraph", "list", "formula",
            "equation", "abstract", "reference", "heading", "section",
        )
    )


def markdown_title_candidate(markdown_text: str, pdf_title: str = "") -> tuple[str, str]:
    if pdf_title and len(pdf_title.strip()) >= 4:
        return pdf_title.strip(), "pdf_metadata"
    for line in (markdown_text or "").splitlines():
        line = line.strip()
        if line.startswith("#"):
            candidate = line.lstrip("# ").strip()
            if candidate:
                return candidate[:500], "first_markdown_heading"
    for line in (markdown_text or "").splitlines():
        candidate = line.strip()
        if candidate and not candidate.startswith(("!", "|", "---")):
            return candidate[:500], "first_nonempty_markdown_line"
    return "", "not_found"


def collect_doi_candidates(text: str) -> list[str]:
    found: list[str] = []
    for value in DOI_REGEX.findall(text or ""):
        value = value.rstrip(".,;:)]}")
        if value and value not in found:
            found.append(value)
    return found


def make_normalized_records(
    item: dict[str, Any], config: ExtractorConfig, mineru_ver: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    outdir = Path(item["doc_output_dir"])
    middle_path = outdir / "middle_json.json"
    markdown_path = outdir / "markdown.md"
    structured_path = outdir / "structured_content.json"
    marker_path = outdir / "_SUCCESS.json"
    error_path = outdir / "_ERROR.json"

    markdown_text = markdown_path.read_text(encoding="utf-8", errors="replace") if markdown_path.is_file() else ""
    title_candidate, title_source = markdown_title_candidate(markdown_text, item.get("pdf_title_metadata", ""))
    doi_candidates = collect_doi_candidates(markdown_text)
    marker = json_load(marker_path) if marker_path.is_file() else {}
    processing_error = json_load(error_path) if error_path.is_file() else {}
    default_status = (
        "FAILED_PREFLIGHT" if item.get("preflight_error")
        else "FAILED" if processing_error
        else "NOT_PROCESSED"
    )
    status = marker.get("status", default_status)
    warnings = list(marker.get("warnings", []))
    if item.get("preflight_warning"):
        warnings.append(item["preflight_warning"])

    element_records: list[dict[str, Any]] = []
    text_records: list[dict[str, Any]] = []
    table_records: list[dict[str, Any]] = []
    figure_records: list[dict[str, Any]] = []
    page_indices: list[Any] = []
    middle_data: dict[str, Any] = {}

    if middle_path.is_file():
        try:
            middle_data = json_load(middle_path)
            if not isinstance(middle_data, dict):
                raise ValueError("middle_json.json bukan objek JSON.")
            for page_idx, blocks in get_pages_and_blocks(middle_data):
                page_indices.append(page_idx)
                for page, path, parent_path, parent_type, block in walk_blocks(blocks, page_idx):
                    element_type = block_type(block)
                    own_text, own_html = own_text_and_html(block)
                    asset_paths = extract_asset_paths(block)
                    element_id = f"{item['document_id']}:p{page}:b{path}"
                    raw_json = json.dumps(block, ensure_ascii=False, separators=(",", ":"))
                    bbox = bbox_value(block)
                    try:
                        page_number = int(page) + 1
                    except (TypeError, ValueError):
                        page_number = None
                    common = {
                        "element_id": element_id,
                        "document_id": item["document_id"],
                        "page_idx": page,
                        "page_number": page_number,
                        "block_path": path,
                        "parent_path": parent_path,
                        "parent_type": parent_type,
                        "element_type": element_type,
                        "text": own_text,
                        "html": own_html,
                        "bbox_json": json.dumps(bbox, ensure_ascii=False) if bbox is not None else "",
                        "asset_paths_json": json.dumps(asset_paths, ensure_ascii=False),
                        "raw_json": raw_json,
                    }
                    element_records.append(common)

                    if own_text and classify_text_type(element_type):
                        text_records.append({
                            "text_id": element_id,
                            "document_id": item["document_id"],
                            "page_idx": page,
                            "page_number": page_number,
                            "block_path": path,
                            "element_type": element_type,
                            "text": own_text,
                            "bbox_json": common["bbox_json"],
                            "raw_json": raw_json,
                        })

                    if is_table_type(element_type):
                        caption_parts: list[str] = []
                        html_parts: list[str] = []
                        subtree_assets: set[str] = set()
                        for child in descendant_blocks(block):
                            child_type = block_type(child)
                            child_text, child_html = own_text_and_html(child)
                            if "caption" in child_type and child_text and child_text not in caption_parts:
                                caption_parts.append(child_text)
                            if child_html and child_html not in html_parts:
                                html_parts.append(child_html)
                            subtree_assets.update(extract_asset_paths(child))
                        table_records.append({
                            "table_id": element_id,
                            "document_id": item["document_id"],
                            "page_idx": page,
                            "page_number": page_number,
                            "block_path": path,
                            "caption": "\n".join(caption_parts),
                            "table_html": "\n".join(html_parts),
                            "table_text": own_text,
                            "bbox_json": common["bbox_json"],
                            "asset_paths_json": json.dumps(sorted(subtree_assets), ensure_ascii=False),
                            "source_json": raw_json,
                        })

                    if is_figure_type(element_type):
                        caption_parts = []
                        subtree_assets = set()
                        for child in descendant_blocks(block):
                            child_type = block_type(child)
                            child_text, _ = own_text_and_html(child)
                            if "caption" in child_type and child_text and child_text not in caption_parts:
                                caption_parts.append(child_text)
                            subtree_assets.update(extract_asset_paths(child))
                        figure_records.append({
                            "figure_id": element_id,
                            "document_id": item["document_id"],
                            "page_idx": page,
                            "page_number": page_number,
                            "block_path": path,
                            "figure_type": element_type,
                            "caption": "\n".join(caption_parts),
                            "description_text": own_text,
                            "bbox_json": common["bbox_json"],
                            "asset_paths_json": json.dumps(sorted(subtree_assets), ensure_ascii=False),
                            "source_json": raw_json,
                        })
        except Exception as exc:
            warnings.append(f"normalization_error: {type(exc).__name__}: {exc}")
            status = "FAILED_NORMALIZATION"

    referenced_assets: set[str] = set()
    for row in element_records:
        try:
            referenced_assets.update(json.loads(row["asset_paths_json"]))
        except Exception:
            pass
    missing_assets = [rel for rel in sorted(referenced_assets) if not (outdir / rel).is_file()]
    if missing_assets:
        warnings.append(f"missing_asset_files: {len(missing_assets)} referensi gambar tidak ditemukan")

    page_count_mineru = len(set(page_indices))
    expected_pages = item.get("page_count_pdf")
    if expected_pages is not None and page_count_mineru and expected_pages != page_count_mineru:
        message = f"page_count_mismatch: PDF={expected_pages}, MinerU={page_count_mineru}"
        if message not in warnings:
            warnings.append(message)
    if not markdown_text.strip() and status.startswith("SUCCESS"):
        warnings.append("markdown_empty")
    if warnings and status == "SUCCESS":
        status = "SUCCESS_WITH_WARNINGS"

    image_dir = outdir / "images"
    extracted_image_files = (
        [str(path.relative_to(outdir).as_posix()) for path in image_dir.rglob("*") if path.is_file()]
        if image_dir.exists() else []
    )
    metadata = middle_data.get("metadata") if isinstance(middle_data, dict) else {}
    producer = metadata.get("producer") if isinstance(metadata, dict) else {}
    current_version = producer.get("version", "") if isinstance(producer, dict) else ""

    document_record = {
        "document_id": item["document_id"],
        "file_name": item["file_name"],
        "source_relative_path": item["source_relative_path"],
        "source_path": item["source_path"],
        "sha256": item.get("sha256", ""),
        "file_size_bytes": item.get("file_size_bytes"),
        "pdf_title_metadata": item.get("pdf_title_metadata", ""),
        "pdf_author_metadata": item.get("pdf_author_metadata", ""),
        "title_candidate": title_candidate,
        "title_candidate_source": title_source,
        "doi_candidates_json": json.dumps(doi_candidates, ensure_ascii=False),
        "page_count_pdf": expected_pages,
        "page_count_mineru": page_count_mineru,
        "mineru_version": current_version or mineru_ver,
        "tier": marker.get("tier", config.tier),
        "ocr_mode": marker.get("ocr_mode", config.ocr_mode),
        "parse_status": status,
        "warnings_json": json.dumps(list(dict.fromkeys(warnings)), ensure_ascii=False),
        "table_count": len(table_records),
        "figure_count": len(figure_records),
        "text_block_count": len(text_records),
        "element_count": len(element_records),
        "text_chars_markdown": len(markdown_text),
        "asset_file_count": len(extracted_image_files),
        "missing_asset_count": len(missing_assets),
        "output_dir_relative": outdir.relative_to(config.output_dir).as_posix() if outdir.exists() else "",
        "markdown_path_relative": markdown_path.relative_to(config.output_dir).as_posix() if markdown_path.is_file() else "",
        "middle_json_path_relative": middle_path.relative_to(config.output_dir).as_posix() if middle_path.is_file() else "",
        "structured_content_path_relative": structured_path.relative_to(config.output_dir).as_posix() if structured_path.is_file() else "",
        "full_text": markdown_text,
        "error_message": processing_error.get("error", item.get("preflight_error", "")),
    }
    return document_record, element_records, text_records, table_records, figure_records, extracted_image_files


def normalize_all(
    input_records: list[dict[str, Any]], config: ExtractorConfig, mineru_ver: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_documents: list[dict[str, Any]] = []
    all_elements: list[dict[str, Any]] = []
    all_text_blocks: list[dict[str, Any]] = []
    all_tables: list[dict[str, Any]] = []
    all_figures: list[dict[str, Any]] = []
    image_inventory: list[dict[str, Any]] = []

    for item in input_records:
        doc, elements, text_blocks, tables, figures, images = make_normalized_records(item, config, mineru_ver)
        all_documents.append(doc)
        all_elements.extend(elements)
        all_text_blocks.extend(text_blocks)
        all_tables.extend(tables)
        all_figures.extend(figures)
        for asset in images:
            image_inventory.append({
                "document_id": item["document_id"],
                "file_name": item["file_name"],
                "asset_path_relative": (Path("documents") / Path(item["doc_folder_name"]) / asset).as_posix(),
                "exists": (Path(item["doc_output_dir"]) / asset).is_file(),
            })

    print("Dokumen:", len(all_documents))
    print("Elemen:", len(all_elements))
    print("Blok teks:", len(all_text_blocks))
    print("Tabel:", len(all_tables))
    print("Figure/chart/gambar:", len(all_figures))
    print("File aset gambar tersimpan:", len(image_inventory))
    return all_documents, all_elements, all_text_blocks, all_tables, all_figures, image_inventory
