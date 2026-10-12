from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import Mock, patch

from PdfExtractor.cuda_compat import configure_cudnn_compatibility


def make_fake_torch(conv_side_effect):
    cudnn = types.SimpleNamespace(enabled=True, version=lambda: 92000)
    conv2d = Mock(side_effect=conv_side_effect)
    fake_torch = types.SimpleNamespace(
        __version__="2.11.0+cu130",
        version=types.SimpleNamespace(cuda="13.0"),
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            synchronize=Mock(),
        ),
        backends=types.SimpleNamespace(cudnn=cudnn),
        nn=types.SimpleNamespace(
            functional=types.SimpleNamespace(conv2d=conv2d)
        ),
        randn=Mock(return_value=object()),
        float32=object(),
    )
    return fake_torch, conv2d, cudnn


class CudnnCompatibilityTests(unittest.TestCase):
    def test_cpu_mode_does_not_import_or_probe_torch(self):
        with patch.dict(sys.modules, {"torch": None}):
            self.assertEqual(configure_cudnn_compatibility("cpu"), "not-applicable")

    def test_disables_cudnn_only_for_known_mismatch_when_fallback_passes(self):
        fake_torch, conv2d, cudnn = make_fake_torch(
            [
                RuntimeError("cuDNN error: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH"),
                object(),
            ]
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            result = configure_cudnn_compatibility("gpu")

        self.assertEqual(result, "disabled-after-sublibrary-mismatch")
        self.assertFalse(cudnn.enabled)
        self.assertEqual(conv2d.call_count, 2)

    def test_other_conv_errors_do_not_change_cudnn_setting(self):
        fake_torch, conv2d, cudnn = make_fake_torch(
            [RuntimeError("unrelated CUDA convolution failure")]
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            result = configure_cudnn_compatibility("gpu")

        self.assertEqual(result, "probe-failed")
        self.assertTrue(cudnn.enabled)
        self.assertEqual(conv2d.call_count, 1)

    def test_failed_fallback_restores_original_cudnn_setting(self):
        fake_torch, conv2d, cudnn = make_fake_torch(
            [
                RuntimeError("CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH"),
                RuntimeError("fallback CUDA convolution failed"),
            ]
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "fallback.*juga gagal"):
                configure_cudnn_compatibility("gpu")

        self.assertTrue(cudnn.enabled)
        self.assertEqual(conv2d.call_count, 2)


if __name__ == "__main__":
    unittest.main()
