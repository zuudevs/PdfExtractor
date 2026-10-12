from __future__ import annotations

from typing import Any

_CUDNN_SUBLIBRARY_MISMATCH = "CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH"


def _probe_cuda_convolution(torch: Any) -> None:
    """Exercise the same CUDA Conv2d/cuDNN path used by MinerU's layout model."""
    sample = torch.randn((1, 3, 32, 32), device="cuda", dtype=torch.float32)
    kernel = torch.randn((8, 3, 3, 3), device="cuda", dtype=torch.float32)
    torch.nn.functional.conv2d(sample, kernel, padding=1)
    torch.cuda.synchronize()


def configure_cudnn_compatibility(runtime_mode: str) -> str:
    """Check CUDA convolution and disable cuDNN only for a known sublibrary mismatch.

    Some PyTorch CUDA wheels can encounter a cuDNN sublibrary version mismatch
    when the dynamic loader finds a different system-wide cuDNN sub-library.
    PyTorch's non-cuDNN CUDA convolution fallback can still work in that case.
    The check runs in the isolated extraction worker so the setting applies to
    the actual process that imports and runs MinerU.
    """
    if runtime_mode != "gpu":
        return "not-applicable"

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Mode GPU dipilih, tetapi torch.cuda.is_available() bernilai False "
            "di worker ekstraksi."
        )

    cudnn = torch.backends.cudnn
    try:
        cudnn_version = cudnn.version()
    except Exception:
        cudnn_version = "unknown"
    print(
        f"[CUDA PREFLIGHT] torch={torch.__version__}; CUDA={torch.version.cuda}; "
        f"cuDNN={cudnn_version}; cuDNN enabled={cudnn.enabled}",
        flush=True,
    )

    if not cudnn.enabled:
        print(
            "[CUDA PREFLIGHT] cuDNN sudah dinonaktifkan oleh environment; "
            "uji Conv2d tetap dijalankan.",
            flush=True,
        )
        try:
            _probe_cuda_convolution(torch)
        except Exception as exc:
            raise RuntimeError(
                "Conv2d CUDA gagal walaupun cuDNN sudah dinonaktifkan: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return "already-disabled"

    try:
        _probe_cuda_convolution(torch)
    except RuntimeError as exc:
        if _CUDNN_SUBLIBRARY_MISMATCH not in str(exc):
            print(
                "[CUDA PREFLIGHT] Uji Conv2d gagal, tetapi bukan error "
                "CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH. cuDNN tidak diubah; "
                f"ekstraksi mungkin tetap gagal. Detail: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return "probe-failed"

        print(
            "[CUDA PREFLIGHT] Terdeteksi CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH. "
            "Mencoba fallback CUDA PyTorch dengan cuDNN dinonaktifkan untuk worker ini.",
            flush=True,
        )
        cudnn.enabled = False
        try:
            _probe_cuda_convolution(torch)
        except Exception as fallback_exc:
            cudnn.enabled = True
            raise RuntimeError(
                "Conv2d gagal karena CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH, "
                "dan uji fallback dengan torch.backends.cudnn.enabled=False juga gagal. "
                "Periksa kecocokan wheel PyTorch/cuDNN dan shared library sistem. "
                f"Fallback: {type(fallback_exc).__name__}: {fallback_exc}"
            ) from fallback_exc

        print(
            "[CUDA PREFLIGHT] Fallback Conv2d berhasil. cuDNN tetap dinonaktifkan "
            "di worker ekstraksi ini; proses bisa lebih lambat.",
            flush=True,
        )
        return "disabled-after-sublibrary-mismatch"

    print("[CUDA PREFLIGHT] CUDA Conv2d/cuDNN smoke test berhasil.", flush=True)
    return "enabled"
