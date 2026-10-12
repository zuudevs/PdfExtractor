from __future__ import annotations

import csv
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .config import ExtractorConfig

DOC_EXPORT_FIELDS = [
    "document_id", "file_name", "source_relative_path", "source_path", "sha256", "file_size_bytes",
    "pdf_title_metadata", "pdf_author_metadata", "title_candidate", "title_candidate_source", "doi_candidates_json",
    "page_count_pdf", "page_count_mineru", "mineru_version", "tier", "ocr_mode", "parse_status", "warnings_json",
    "table_count", "figure_count", "text_block_count", "element_count", "text_chars_markdown", "asset_file_count",
    "missing_asset_count", "output_dir_relative", "markdown_path_relative", "middle_json_path_relative",
    "structured_content_path_relative", "error_message",
]
TABLE_EXPORT_FIELDS = [
    "table_id", "document_id", "page_idx", "page_number", "block_path", "caption", "table_html",
    "table_text", "bbox_json", "asset_paths_json",
]
FIGURE_EXPORT_FIELDS = [
    "figure_id", "document_id", "page_idx", "page_number", "block_path", "figure_type", "caption",
    "description_text", "bbox_json", "asset_paths_json",
]
TEXT_EXPORT_FIELDS = [
    "text_id", "document_id", "page_idx", "page_number", "block_path", "element_type", "text", "bbox_json",
]
AUDIT_EXPORT_FIELDS = [
    "document_id", "file_name", "source_relative_path", "parse_status", "page_count_pdf", "page_count_mineru",
    "table_count", "figure_count", "text_block_count", "element_count", "asset_file_count", "missing_asset_count",
    "warnings_json", "error_message",
]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (row.get(key, "") if row.get(key, "") is not None else "") for key in fieldnames})


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _insert_rows(
    connection: sqlite3.Connection, table_name: str, fields: list[str], rows: list[dict[str, Any]]
) -> None:
    if not rows:
        return
    placeholders = ",".join("?" for _ in fields)
    sql = f"INSERT INTO {table_name} ({','.join(fields)}) VALUES ({placeholders})"
    connection.executemany(sql, [tuple(row.get(field) for field in fields) for row in rows])


def export_database(
    config: ExtractorConfig,
    documents: list[dict[str, Any]],
    elements: list[dict[str, Any]],
    text_blocks: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    figures: list[dict[str, Any]],
    image_inventory: list[dict[str, Any]],
) -> Path:
    database_dir = config.database_dir
    database_dir.mkdir(parents=True, exist_ok=True)

    write_csv(database_dir / "literature_documents.csv", documents, DOC_EXPORT_FIELDS)
    write_csv(database_dir / "literature_tables.csv", tables, TABLE_EXPORT_FIELDS)
    write_csv(database_dir / "literature_figures.csv", figures, FIGURE_EXPORT_FIELDS)
    write_csv(database_dir / "literature_text_blocks.csv", text_blocks, TEXT_EXPORT_FIELDS)
    write_jsonl(database_dir / "literature_elements.jsonl", elements)
    write_jsonl(database_dir / "literature_fulltext.jsonl", [
        {
            "document_id": doc["document_id"],
            "file_name": doc["file_name"],
            "title_candidate": doc["title_candidate"],
            "markdown_path_relative": doc["markdown_path_relative"],
            "full_text": doc["full_text"],
        }
        for doc in documents
    ])
    write_csv(database_dir / "literature_extraction_audit.csv", documents, AUDIT_EXPORT_FIELDS)
    write_csv(database_dir / "literature_image_inventory.csv", image_inventory,
              ["document_id", "file_name", "asset_path_relative", "exists"])

    db_path = database_dir / "literature_database.sqlite"
    temp_db_path = database_dir / "literature_database.building.sqlite"
    if temp_db_path.exists():
        temp_db_path.unlink()

    with sqlite3.connect(temp_db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript("""
        CREATE TABLE documents (
            document_id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            source_relative_path TEXT,
            source_path TEXT,
            sha256 TEXT,
            file_size_bytes INTEGER,
            pdf_title_metadata TEXT,
            pdf_author_metadata TEXT,
            title_candidate TEXT,
            title_candidate_source TEXT,
            doi_candidates_json TEXT,
            page_count_pdf INTEGER,
            page_count_mineru INTEGER,
            mineru_version TEXT,
            tier TEXT,
            ocr_mode TEXT,
            parse_status TEXT,
            warnings_json TEXT,
            table_count INTEGER,
            figure_count INTEGER,
            text_block_count INTEGER,
            element_count INTEGER,
            text_chars_markdown INTEGER,
            asset_file_count INTEGER,
            missing_asset_count INTEGER,
            output_dir_relative TEXT,
            markdown_path_relative TEXT,
            middle_json_path_relative TEXT,
            structured_content_path_relative TEXT,
            full_text TEXT,
            error_message TEXT
        );
        CREATE TABLE elements (
            element_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            page_idx INTEGER,
            page_number INTEGER,
            block_path TEXT,
            parent_path TEXT,
            parent_type TEXT,
            element_type TEXT,
            text TEXT,
            html TEXT,
            bbox_json TEXT,
            asset_paths_json TEXT,
            raw_json TEXT,
            FOREIGN KEY(document_id) REFERENCES documents(document_id)
        );
        CREATE TABLE text_blocks (
            text_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            page_idx INTEGER,
            page_number INTEGER,
            block_path TEXT,
            element_type TEXT,
            text TEXT,
            bbox_json TEXT,
            raw_json TEXT,
            FOREIGN KEY(document_id) REFERENCES documents(document_id)
        );
        CREATE TABLE tables (
            table_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            page_idx INTEGER,
            page_number INTEGER,
            block_path TEXT,
            caption TEXT,
            table_html TEXT,
            table_text TEXT,
            bbox_json TEXT,
            asset_paths_json TEXT,
            source_json TEXT,
            FOREIGN KEY(document_id) REFERENCES documents(document_id)
        );
        CREATE TABLE figures (
            figure_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            page_idx INTEGER,
            page_number INTEGER,
            block_path TEXT,
            figure_type TEXT,
            caption TEXT,
            description_text TEXT,
            bbox_json TEXT,
            asset_paths_json TEXT,
            source_json TEXT,
            FOREIGN KEY(document_id) REFERENCES documents(document_id)
        );
        CREATE TABLE image_assets (
            document_id TEXT NOT NULL,
            file_name TEXT,
            asset_path_relative TEXT,
            exists_on_disk INTEGER,
            FOREIGN KEY(document_id) REFERENCES documents(document_id)
        );
        CREATE INDEX idx_elements_document_page ON elements(document_id, page_number);
        CREATE INDEX idx_text_document_page ON text_blocks(document_id, page_number);
        CREATE INDEX idx_tables_document_page ON tables(document_id, page_number);
        CREATE INDEX idx_figures_document_page ON figures(document_id, page_number);
        CREATE INDEX idx_documents_status ON documents(parse_status);
        """)

        doc_db_fields = [field for field in DOC_EXPORT_FIELDS if field != "source_path"]
        # Source path is retained in SQLite even though it is already exported to CSV.
        doc_db_fields = list(dict.fromkeys(doc_db_fields + ["source_path", "full_text"]))
        _insert_rows(connection, "documents", doc_db_fields, documents)
        _insert_rows(connection, "elements", [
            "element_id", "document_id", "page_idx", "page_number", "block_path", "parent_path", "parent_type",
            "element_type", "text", "html", "bbox_json", "asset_paths_json", "raw_json",
        ], elements)
        _insert_rows(connection, "text_blocks", [
            "text_id", "document_id", "page_idx", "page_number", "block_path", "element_type", "text",
            "bbox_json", "raw_json",
        ], text_blocks)
        _insert_rows(connection, "tables", [
            "table_id", "document_id", "page_idx", "page_number", "block_path", "caption", "table_html",
            "table_text", "bbox_json", "asset_paths_json", "source_json",
        ], tables)
        _insert_rows(connection, "figures", [
            "figure_id", "document_id", "page_idx", "page_number", "block_path", "figure_type", "caption",
            "description_text", "bbox_json", "asset_paths_json", "source_json",
        ], figures)
        _insert_rows(connection, "image_assets", [
            "document_id", "file_name", "asset_path_relative", "exists_on_disk",
        ], [
            {
                "document_id": row["document_id"],
                "file_name": row["file_name"],
                "asset_path_relative": row["asset_path_relative"],
                "exists_on_disk": int(bool(row["exists"])),
            }
            for row in image_inventory
        ])

        try:
            connection.execute(
                "CREATE VIRTUAL TABLE documents_fts USING fts5(document_id UNINDEXED, title_candidate, full_text)"
            )
            connection.executemany(
                "INSERT INTO documents_fts(document_id, title_candidate, full_text) VALUES (?, ?, ?)",
                [(doc["document_id"], doc.get("title_candidate", ""), doc.get("full_text", "")) for doc in documents],
            )
            connection.execute(
                "CREATE VIRTUAL TABLE text_blocks_fts USING fts5(document_id UNINDEXED, element_type, text)"
            )
            connection.executemany(
                "INSERT INTO text_blocks_fts(document_id, element_type, text) VALUES (?, ?, ?)",
                [(row["document_id"], row.get("element_type", ""), row.get("text", "")) for row in text_blocks],
            )
        except sqlite3.OperationalError as exc:
            print("FTS5 tidak tersedia; query LIKE tetap dapat digunakan. Detail:", exc)

    # Atomic replace: a crash during DB construction does not destroy the previous DB.
    os.replace(temp_db_path, db_path)
    print(f"\nDatabase selesai dibuat: {db_path}")
    print(f"CSV/JSONL dan audit: {database_dir}")
    with sqlite3.connect(db_path) as connection:
        print("Jumlah record di SQLite:")
        for table_name in ("documents", "elements", "text_blocks", "tables", "figures", "image_assets"):
            count = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"- {table_name}: {count}")
    return db_path
