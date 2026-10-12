from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.PdfExtractor.PdfExtractor.config import ExtractorConfig
from scripts.PdfExtractor.PdfExtractor.normalizer import (
    collect_doi_candidates,
    extract_asset_paths,
    get_pages_and_blocks,
    make_normalized_records,
    markdown_title_candidate,
)


class NormalizerTests(unittest.TestCase):
    def test_title_and_doi(self):
        title, source = markdown_title_candidate("# Example Paper\n\nBody", "")
        self.assertEqual((title, source), ("Example Paper", "first_markdown_heading"))
        self.assertEqual(collect_doi_candidates("DOI: 10.1234/Abc.def."), ["10.1234/Abc.def"])

    def test_assets(self):
        paths = extract_asset_paths({"img": "./images/fig_1.png and images/table_2.jpg"})
        self.assertEqual(paths, ["images/fig_1.png", "images/table_2.jpg"])

    def test_page_walk_and_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            outdir = output_dir / "documents" / "demo__123"
            (outdir / "images").mkdir(parents=True)
            (outdir / "images" / "fig1.png").write_bytes(b"fake image")
            middle = {
                "metadata": {"producer": {"version": "4.test"}},
                "pages": [{
                    "page_idx": 0,
                    "blocks": [
                        {"index": 0, "type": "text", "content": "Introduction text"},
                        {"index": 1, "type": "table", "html": "<table><tr><td>A</td></tr></table>", "blocks": [
                            {"index": 0, "type": "table_caption", "content": "Table 1. Demo"}
                        ]},
                        {"index": 2, "type": "image", "content": "Figure", "url": "images/fig1.png", "blocks": [
                            {"index": 0, "type": "figure_caption", "content": "Figure 1. Demo"}
                        ]},
                    ],
                }],
            }
            (outdir / "middle_json.json").write_text(json.dumps(middle), encoding="utf-8")
            (outdir / "structured_content.json").write_text(json.dumps({"pages": []}), encoding="utf-8")
            (outdir / "markdown.md").write_text("# Example Paper\n\nDOI: 10.1234/Abc.def", encoding="utf-8")
            (outdir / "_SUCCESS.json").write_text(json.dumps({"status": "SUCCESS", "tier": "standard", "ocr_mode": "auto"}), encoding="utf-8")

            config = ExtractorConfig(input_dir=input_dir, output_dir=output_dir)
            item = {
                "document_id": "demo123", "file_name": "demo.pdf", "source_relative_path": "demo.pdf",
                "source_path": str(input_dir / "demo.pdf"), "sha256": "abc", "file_size_bytes": 123,
                "doc_folder_name": "demo__123", "doc_output_dir": outdir,
                "page_count_pdf": 1, "pdf_title_metadata": "", "pdf_author_metadata": "",
            }
            doc, elements, texts, tables, figures, images = make_normalized_records(item, config, "4.test")
            self.assertEqual(doc["title_candidate"], "Example Paper")
            self.assertEqual(doc["table_count"], 1)
            self.assertEqual(doc["figure_count"], 1)
            self.assertGreaterEqual(len(texts), 1)
            self.assertEqual(tables[0]["caption"], "Table 1. Demo")
            self.assertEqual(figures[0]["caption"], "Figure 1. Demo")
            self.assertEqual(images, ["images/fig1.png"])


if __name__ == "__main__":
    unittest.main()
