from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PdfExtractor.config import ExtractorConfig
from PdfExtractor.runner import process_one_pdf


class NativeCrashReportingTests(unittest.TestCase):
    def test_sigsegv_creates_error_json_and_worker_log(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "input"
            input_dir.mkdir()
            config = ExtractorConfig(
                input_dir=input_dir,
                output_dir=root / "output",
                runtime_mode="gpu",
                tier="standard",
                vlm_engine="llama-cpp",
            )
            document_output = config.documents_dir / "paper__123"
            item = {
                "document_id": "123",
                "file_name": "paper.pdf",
                "source_relative_path": "paper.pdf",
                "source_path": str(input_dir / "paper.pdf"),
                "sha256": "test-hash",
                "doc_output_dir": document_output,
                "page_count_pdf": 1,
            }
            mocked = subprocess.CompletedProcess(
                args=["python", "-m", "PdfExtractor.worker"],
                returncode=-11,
                stdout="native worker output",
                stderr="native segmentation fault",
            )
            with patch("PdfExtractor.runner.subprocess.run", return_value=mocked):
                result = process_one_pdf(item, config, "4.0.0")

            self.assertEqual(result["status"], "FAILED_NATIVE_CRASH")
            self.assertTrue((document_output / "_ERROR.json").is_file())
            self.assertTrue((document_output / "_WORKER.log").is_file())
            report = json.loads((document_output / "_ERROR.json").read_text(encoding="utf-8"))
            self.assertEqual(report["exit_code"], -11)
            self.assertEqual(report["signal"], "SIGSEGV")
            self.assertIn("native segmentation fault", report["stderr_tail"])


if __name__ == "__main__":
    unittest.main()
