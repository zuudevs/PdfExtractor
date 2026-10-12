from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Tier = Literal["flash", "basic", "standard", "advanced"]
ModelSource = Literal["auto", "huggingface", "modelscope"]


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
    cpu_only: bool = True

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        if self.tier not in {"flash", "basic", "standard", "advanced"}:
            raise ValueError(f"Tier tidak didukung: {self.tier!r}")
        if self.model_source not in {"auto", "huggingface", "modelscope"}:
            raise ValueError(f"Sumber model tidak didukung: {self.model_source!r}")
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Folder input tidak ditemukan: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"INPUT_DIR bukan folder: {self.input_dir}")
        if self.input_dir == self.output_dir:
            raise ValueError("INPUT_DIR dan OUTPUT_DIR tidak boleh sama.")

    @property
    def documents_dir(self) -> Path:
        return self.output_dir / "documents"

    @property
    def database_dir(self) -> Path:
        return self.output_dir / "database"

    def configure_runtime(self) -> None:
        """Set backend variables before MinerU is imported in this process.

        MinerU 4.x's documented CPU combination is ONNX for small models and
        llama.cpp for the VLM. Table models are explicitly assigned to CPU.
        """
        if self.cpu_only:
            os.environ["MINERU_MODEL_SMALL_BACKEND"] = "onnx"
            os.environ["MINERU_MODEL_VLM_ENGINE"] = "llama-cpp"
            os.environ["MINERU_TABLE_DEVICE"] = "cpu"
            # Hide CUDA GPUs from libraries that autodetect CUDA. This does not
            # disable CPU execution; it prevents the Colab T4 from being selected.
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if self.model_source == "auto":
            # Let MinerU probe the configured/default sources. An inherited value
            # such as the literal string "auto" may be rejected by some releases.
            os.environ.pop("MINERU_MODEL_SOURCE", None)
        else:
            os.environ["MINERU_MODEL_SOURCE"] = self.model_source


def package_version(package_name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"
