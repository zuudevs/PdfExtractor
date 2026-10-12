from __future__ import annotations

import os
import subprocess
import sys
from typing import Sequence

from .config import ExtractorConfig


def _run(command: Sequence[str]) -> None:
    printable = " ".join(command)
    print(f"\n[MODEL SETUP] {printable}", flush=True)
    subprocess.run(command, check=True, env=os.environ.copy())


def prepare_models(config: ExtractorConfig) -> None:
    """Download and verify model bundles using the exact CPU backends used at runtime."""
    if config.tier == "flash":
        print("Tier flash tidak memerlukan tahap persiapan model VLM; melewati model download.")
        return

    if config.tier == "basic":
        download_tier = "basic"
        common = ["--small-backend", "onnx"]
    else:
        # Advanced uses the standard model bundle in MinerU 4.x.
        download_tier = "standard"
        common = ["--small-backend", "onnx", "--vlm-engine", "llama-cpp"]

    download = [
        "mineru-kit", "models", "download", "--tier", download_tier, *common
    ]
    verify = [
        "mineru-kit", "models", "verify", "--tier", download_tier, *common
    ]
    if config.model_source != "auto":
        download.extend(["--source", config.model_source])
    _run(download)
    _run(verify)
