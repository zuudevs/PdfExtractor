from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Tier = Literal["flash", "basic", "standard", "advanced"]
ModelSource = Literal["auto", "huggingface", "modelscope"]
RuntimeMode = Literal["cpu", "gpu"]


@dataclass(slots=True)
class ExtractorConfig:
    input_dir: Path
    output_dir: Path
    recursive: bool = True
    tier: Tier = "standard"
    ocr_mode: str = "auto"
    skip_valid_results: bool = True
    continue_on_error: bool = True
    model_source: ModelSource = "auto"
    runtime_mode: RuntimeMode = "cpu"

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        if self.tier not in {"flash", "basic", "standard", "advanced"}:
            raise ValueError(f"Tier tidak didukung: {self.tier!r}")
        if self.model_source not in {"auto", "huggingface", "modelscope"}:
            raise ValueError(f"Sumber model tidak didukung: {self.model_source!r}")
        if self.runtime_mode not in {"cpu", "gpu"}:
            raise ValueError(f"Mode runtime tidak didukung: {self.runtime_mode!r}")
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Folder input tidak ditemukan: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"INPUT_DIR bukan folder: {self.input_dir}")
        if self.input_dir == self.output_dir:
            raise ValueError("INPUT_DIR dan OUTPUT_DIR tidak boleh sama.")
        # Prevent accidentally putting outputs within the input tree (could make
        # recursive scans messy and consume storage in the source collection).
        if self.input_dir in self.output_dir.parents:
            raise ValueError("OUTPUT_DIR tidak boleh berada di dalam INPUT_DIR.")

    @property
    def documents_dir(self) -> Path:
        return self.output_dir / "documents"

    @property
    def database_dir(self) -> Path:
        return self.output_dir / "database"

    def configure_runtime(self) -> None:
        """Set MinerU 4.x model backends before importing MinerU.

        CPU: ONNX small models + llama.cpp VLM + CPU table models.
        NVIDIA GPU (Linux/Colab): Torch small models + vLLM VLM + CUDA table models.
        """
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if self.runtime_mode == "cpu":
            os.environ["MINERU_MODEL_SMALL_BACKEND"] = "onnx"
            os.environ["MINERU_MODEL_VLM_ENGINE"] = "llama-cpp"
            os.environ["MINERU_TABLE_DEVICE"] = "cpu"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            # GPU mode is intended for a compatible NVIDIA CUDA runtime, such as
            # Google Colab T4. Do not set CUDA_VISIBLE_DEVICES=-1 in this mode.
            if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ["MINERU_MODEL_SMALL_BACKEND"] = "torch"
            os.environ["MINERU_TABLE_DEVICE"] = "cuda"
            needs_vlm = self.tier in {"standard", "advanced"}
            if needs_vlm:
                os.environ["MINERU_MODEL_VLM_ENGINE"] = "vllm"
                import importlib.util
                if importlib.util.find_spec("vllm") is None:
                    raise RuntimeError(
                        "Mode GPU Standard/Advanced dipilih, tetapi package vllm tidak ditemukan. "
                        "Di Colab, instal dependensi dengan mineru[full]>=4.0,<5."
                    )
            else:
                # Basic/Flash do not run a local VLM; avoid forcing an absent VLM engine.
                os.environ.pop("MINERU_MODEL_VLM_ENGINE", None)
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "Mode GPU dipilih tetapi torch.cuda.is_available() bernilai False. "
                        "Di Colab, pilih Runtime > Change runtime type > T4 GPU lalu restart runtime."
                    )
            except ImportError as exc:
                raise RuntimeError(
                    "Mode GPU membutuhkan PyTorch dengan dukungan CUDA. "
                    "Gunakan instalasi MinerU full pada runtime NVIDIA yang kompatibel."
                ) from exc
        if self.model_source == "auto":
            # Let MinerU probe configured/default sources. Some releases reject
            # an inherited literal value 'auto' in MINERU_MODEL_SOURCE.
            os.environ.pop("MINERU_MODEL_SOURCE", None)
        else:
            os.environ["MINERU_MODEL_SOURCE"] = self.model_source


def package_version(package_name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"
