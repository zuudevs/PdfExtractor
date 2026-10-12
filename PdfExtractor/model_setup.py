from __future__ import annotations

import os
import subprocess
from typing import Sequence

from .config import ExtractorConfig


def _run(command: Sequence[str]) -> None:
    printable = " ".join(command)
    print(f"\n[MODEL SETUP] {printable}", flush=True)
    subprocess.run(command, check=True, env=os.environ.copy())


def prepare_models(config: ExtractorConfig) -> None:
    """Download and verify models for the selected runtime/backend combination."""
    if config.tier == "flash":
        print("Tier flash tidak memerlukan model VLM lokal; melewati model download.")
        return

    download_tier = "basic" if config.tier == "basic" else "standard"
    if config.runtime_mode == "cpu":
        common = ["--small-backend", "onnx"]
        if download_tier == "standard":
            common.extend(["--vlm-engine", "llama-cpp"])
    else:
        common = ["--small-backend", "torch"]
        if download_tier == "standard":
            common.extend(["--vlm-engine", "vllm"])

    download = ["mineru-kit", "models", "download", "--tier", download_tier, *common]
    verify = ["mineru-kit", "models", "verify", "--tier", download_tier, *common]
    if config.model_source != "auto":
        download.extend(["--source", config.model_source])
    _run(download)
    _run(verify)
    # Verify with the same selection to prevent runtime/backend mismatch.
    _run(["mineru-kit", "models", "show"])
