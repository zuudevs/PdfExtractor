from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PdfExtractor.config import ExtractorConfig


class RuntimeConfigTests(unittest.TestCase):
    def _env_snapshot(self, keys):
        return {key: os.environ.get(key) for key in keys}

    def _restore_env(self, old):
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_cpu_backends_are_forced(self):
        env_keys = ["MINERU_MODEL_SMALL_BACKEND", "MINERU_MODEL_VLM_ENGINE", "MINERU_TABLE_DEVICE", "CUDA_VISIBLE_DEVICES", "MINERU_MODEL_SOURCE"]
        old = self._env_snapshot(env_keys)
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                inp = root / "in"
                inp.mkdir()
                config = ExtractorConfig(input_dir=inp, output_dir=root / "out", model_source="auto", runtime_mode="cpu")
                os.environ["MINERU_MODEL_SOURCE"] = "auto"
                config.configure_runtime()
                self.assertEqual(os.environ["MINERU_MODEL_SMALL_BACKEND"], "onnx")
                self.assertEqual(os.environ["MINERU_MODEL_VLM_ENGINE"], "llama-cpp")
                self.assertEqual(os.environ["MINERU_TABLE_DEVICE"], "cpu")
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")
                self.assertNotIn("MINERU_MODEL_SOURCE", os.environ)
        finally:
            self._restore_env(old)

    def test_gpu_backends_are_selected(self):
        env_keys = ["MINERU_MODEL_SMALL_BACKEND", "MINERU_MODEL_VLM_ENGINE", "MINERU_TABLE_DEVICE", "CUDA_VISIBLE_DEVICES", "MINERU_MODEL_SOURCE"]
        old = self._env_snapshot(env_keys)
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                inp = root / "in"
                inp.mkdir()
                config = ExtractorConfig(input_dir=inp, output_dir=root / "out", runtime_mode="gpu")
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                fake_vllm = types.ModuleType("vllm")
                with patch.dict(sys.modules, {"torch": fake_torch, "vllm": fake_vllm}):
                    config.configure_runtime()
                self.assertEqual(os.environ["MINERU_MODEL_SMALL_BACKEND"], "torch")
                self.assertEqual(os.environ["MINERU_MODEL_VLM_ENGINE"], "vllm")
                self.assertEqual(os.environ["MINERU_TABLE_DEVICE"], "cuda")
                self.assertNotEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "-1")
        finally:
            self._restore_env(old)

    def test_gpu_basic_does_not_require_vllm(self):
        env_keys = ["MINERU_MODEL_SMALL_BACKEND", "MINERU_MODEL_VLM_ENGINE", "MINERU_TABLE_DEVICE", "CUDA_VISIBLE_DEVICES", "MINERU_MODEL_SOURCE"]
        old = self._env_snapshot(env_keys)
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                inp = root / "in"
                inp.mkdir()
                config = ExtractorConfig(input_dir=inp, output_dir=root / "out", runtime_mode="gpu", tier="basic")
                with patch.dict(sys.modules, {"torch": fake_torch}):
                    config.configure_runtime()
                self.assertEqual(os.environ["MINERU_MODEL_SMALL_BACKEND"], "torch")
                self.assertEqual(os.environ["MINERU_TABLE_DEVICE"], "cuda")
                self.assertNotIn("MINERU_MODEL_VLM_ENGINE", os.environ)
        finally:
            self._restore_env(old)


    def test_gpu_llama_cpp_engine_does_not_require_vllm(self):
        env_keys = ["MINERU_MODEL_SMALL_BACKEND", "MINERU_MODEL_VLM_ENGINE", "MINERU_TABLE_DEVICE", "CUDA_VISIBLE_DEVICES", "MINERU_MODEL_SOURCE"]
        old = self._env_snapshot(env_keys)
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                inp = root / "in"
                inp.mkdir()
                config = ExtractorConfig(
                    input_dir=inp,
                    output_dir=root / "out",
                    runtime_mode="gpu",
                    vlm_engine="llama-cpp",
                )
                # No fake vllm module is provided: compatibility mode must not import it.
                with patch.dict(sys.modules, {"torch": fake_torch}):
                    config.configure_runtime()
                self.assertEqual(os.environ["MINERU_MODEL_SMALL_BACKEND"], "torch")
                self.assertEqual(os.environ["MINERU_MODEL_VLM_ENGINE"], "llama-cpp")
                self.assertEqual(os.environ["MINERU_TABLE_DEVICE"], "cuda")
                self.assertNotEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "-1")
        finally:
            self._restore_env(old)

    def test_cpu_rejects_vllm_engine(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inp = root / "in"
            inp.mkdir()
            with self.assertRaisesRegex(ValueError, "runtime GPU"):
                ExtractorConfig(
                    input_dir=inp,
                    output_dir=root / "out",
                    runtime_mode="cpu",
                    vlm_engine="vllm",
                )


if __name__ == "__main__":
    unittest.main()
