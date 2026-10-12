from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.mineru_literature_cpu.mineru_literature_cpu.config import ExtractorConfig
from scripts.mineru_literature_cpu.mineru_literature_cpu.database import export_database


class DatabaseTests(unittest.TestCase):
    def test_database_and_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "input"
            input_dir.mkdir()
            output_dir = root / "output"
            config = ExtractorConfig(input_dir=input_dir, output_dir=output_dir)
            document = {
                "document_id": "doc1", "file_name": "paper.pdf", "source_relative_path": "paper.pdf",
                "source_path": str(input_dir / "paper.pdf"), "sha256": "hash", "file_size_bytes": 10,
                "pdf_title_metadata": "", "pdf_author_metadata": "", "title_candidate": "Paper",
                "title_candidate_source": "first_markdown_heading", "doi_candidates_json": "[]",
                "page_count_pdf": 1, "page_count_mineru": 1, "mineru_version": "4.test", "tier": "standard",
                "ocr_mode": "auto", "parse_status": "SUCCESS", "warnings_json": "[]", "table_count": 1,
                "figure_count": 1, "text_block_count": 1, "element_count": 3, "text_chars_markdown": 12,
                "asset_file_count": 1, "missing_asset_count": 0, "output_dir_relative": "documents/paper__doc1",
                "markdown_path_relative": "documents/paper__doc1/markdown.md",
                "middle_json_path_relative": "documents/paper__doc1/middle_json.json",
                "structured_content_path_relative": "documents/paper__doc1/structured_content.json",
                "full_text": "# Paper", "error_message": "",
            }
            element = {
                "element_id": "doc1:p0:b0", "document_id": "doc1", "page_idx": 0, "page_number": 1,
                "block_path": "0", "parent_path": "", "parent_type": "", "element_type": "text",
                "text": "Some text", "html": "", "bbox_json": "", "asset_paths_json": "[]", "raw_json": "{}",
            }
            text_block = {
                "text_id": "doc1:p0:b0", "document_id": "doc1", "page_idx": 0, "page_number": 1,
                "block_path": "0", "element_type": "text", "text": "Some text", "bbox_json": "", "raw_json": "{}",
            }
            table = {
                "table_id": "doc1:p0:b1", "document_id": "doc1", "page_idx": 0, "page_number": 1,
                "block_path": "1", "caption": "Table 1", "table_html": "<table></table>", "table_text": "A",
                "bbox_json": "", "asset_paths_json": "[]", "source_json": "{}",
            }
            figure = {
                "figure_id": "doc1:p0:b2", "document_id": "doc1", "page_idx": 0, "page_number": 1,
                "block_path": "2", "figure_type": "image", "caption": "Figure 1", "description_text": "",
                "bbox_json": "", "asset_paths_json": "[]", "source_json": "{}",
            }
            image = {"document_id": "doc1", "file_name": "paper.pdf", "asset_path_relative": "documents/paper__doc1/images/fig.png", "exists": True}
            db_path = export_database(config, [document], [element], [text_block], [table], [figure], [image])
            with sqlite3.connect(db_path) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 1)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM tables").fetchone()[0], 1)
                self.assertEqual(connection.execute("SELECT full_text FROM documents").fetchone()[0], "# Paper")
            self.assertTrue((output_dir / "database" / "literature_documents.csv").is_file())
            self.assertTrue((output_dir / "database" / "literature_elements.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
