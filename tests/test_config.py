from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from scripts.PdfExtractor.PdfExtractor.config import ExtractorConfig


class CpuConfigTests(unittest.TestCase):
    def test_cpu_backends_are_forced_before_parser_import(self):
        env_keys = [
            "MINERU_MODEL_SMALL_BACKEND", "MINERU_MODEL_VLM_ENGINE", "MINERU_TABLE_DEVICE",
            "CUDA_VISIBLE_DEVICES", "MINERU_MODEL_SOURCE",
        ]
        old = {key: os.environ.get(key) for key in env_keys}
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                input_dir = root / "in"
                input_dir.mkdir()
                config = ExtractorConfig(input_dir=input_dir, output_dir=root / "out", model_source="auto")
                os.environ["MINERU_MODEL_SOURCE"] = "auto"
                config.configure_runtime()
                self.assertEqual(os.environ["MINERU_MODEL_SMALL_BACKEND"], "onnx")
                self.assertEqual(os.environ["MINERU_MODEL_VLM_ENGINE"], "llama-cpp")
                self.assertEqual(os.environ["MINERU_TABLE_DEVICE"], "cpu")
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")
                self.assertNotIn("MINERU_MODEL_SOURCE", os.environ)
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
