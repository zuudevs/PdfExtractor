from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Tier = Literal["flash", "basic", "standard", "advanced"]
ModelSource = Literal["auto", "huggingface", "modelscope"]
RuntimeMode = Literal["cpu", "gpu"]
VLMEngine = Literal["auto", "llama-cpp", "vllm"]


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
    vlm_engine: VLMEngine = "auto"

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        if self.tier not in {"flash", "basic", "standard", "advanced"}:
            raise ValueError(f"Tier tidak didukung: {self.tier!r}")
        if self.model_source not in {"auto", "huggingface", "modelscope"}:
            raise ValueError(f"Sumber model tidak didukung: {self.model_source!r}")
        if self.runtime_mode not in {"cpu", "gpu"}:
            raise ValueError(f"Mode runtime tidak didukung: {self.runtime_mode!r}")
        if self.vlm_engine not in {"auto", "llama-cpp", "vllm"}:
            raise ValueError(f"VLM engine tidak didukung: {self.vlm_engine!r}")
        if self.tier in {"standard", "advanced"} and self.runtime_mode == "cpu" and self.vlm_engine == "vllm":
            raise ValueError("VLM engine vllm memerlukan runtime GPU; pilih engine llama-cpp atau auto.")
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
    def effective_vlm_engine(self) -> str | None:
        """Resolve the VLM engine for tiers that actually invoke a VLM."""
        if self.tier not in {"standard", "advanced"}:
            return None
        if self.vlm_engine != "auto":
            return self.vlm_engine
        return "llama-cpp" if self.runtime_mode == "cpu" else "vllm"

    @property
    def documents_dir(self) -> Path:
        return self.output_dir / "documents"

    @property
    def database_dir(self) -> Path:
        return self.output_dir / "database"

    def configure_runtime(self) -> None:
        """Set MinerU 4.x backends before importing MinerU.

        The VLM engine is independent from the Torch/CUDA small-model and table
        backends. On GPU, selecting llama-cpp is a compatibility option for
        runtimes where vLLM wheels cannot be imported or initialized.
        """
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        needs_vlm = self.tier in {"standard", "advanced"}

        if self.runtime_mode == "cpu":
            os.environ["MINERU_MODEL_SMALL_BACKEND"] = "onnx"
            os.environ["MINERU_TABLE_DEVICE"] = "cpu"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            if needs_vlm:
                os.environ["MINERU_MODEL_VLM_ENGINE"] = self.effective_vlm_engine or "llama-cpp"
            else:
                os.environ.pop("MINERU_MODEL_VLM_ENGINE", None)
        else:
            # Do not mask an NVIDIA GPU in GPU mode.
            if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ["MINERU_MODEL_SMALL_BACKEND"] = "torch"
            os.environ["MINERU_TABLE_DEVICE"] = "cuda"

            if needs_vlm:
                engine = self.effective_vlm_engine or "vllm"
                os.environ["MINERU_MODEL_VLM_ENGINE"] = engine
                if engine == "vllm":
                    # find_spec() alone does not catch broken wheels or missing shared
                    # libraries, both common causes of opaque Colab extraction failures.
                    try:
                        import vllm  # noqa: F401
                    except Exception as exc:
                        raise RuntimeError(
                            "VLM engine vLLM dipilih, tetapi package vllm gagal diimpor. "
                            "Ini biasanya menandakan wheel/dependensi Python, CUDA, atau "
                            "shared-library tidak cocok dengan runtime Colab. "
                            "Coba --vlm-engine llama-cpp untuk tetap memakai backend "
                            "Torch/CUDA pada small model dan tabel, atau gunakan runtime "
                            "GPU dan wheel vLLM yang kompatibel. "
                            f"Detail impor: {type(exc).__name__}: {exc}"
                        ) from exc
            else:
                os.environ.pop("MINERU_MODEL_VLM_ENGINE", None)

            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "Mode GPU dipilih tetapi torch.cuda.is_available() bernilai False. "
                        "Di Colab, pilih Runtime > Change runtime type > GPU lalu restart runtime."
                    )
            except ImportError as exc:
                raise RuntimeError(
                    "Mode GPU membutuhkan PyTorch dengan dukungan CUDA. "
                    "Gunakan instalasi MinerU [torch]/[full] pada runtime NVIDIA yang kompatibel."
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
