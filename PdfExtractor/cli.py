from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExtractorConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mineru-literature-cpu",
        description="Ekstraksi PDF dengan MinerU 4.x dalam mode CPU dan ekspor database literatur.",
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder berisi PDF.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder output (sebaiknya di luar input).")
    parser.add_argument("--tier", choices=["flash", "basic", "standard", "advanced"], default="standard")
    parser.add_argument("--ocr-mode", default="auto", choices=["auto", "txt", "ocr"])
    parser.add_argument("--non-recursive", action="store_true", help="Jangan mencari PDF di subfolder.")
    parser.add_argument("--no-resume", action="store_true", help="Proses ulang meskipun hasil valid sudah ada.")
    parser.add_argument("--stop-on-error", action="store_true", help="Hentikan batch saat satu PDF gagal.")
    parser.add_argument(
        "--model-source", choices=["auto", "huggingface", "modelscope"], default="auto",
        help="Sumber model. 'auto' membiarkan MinerU memilih sumber yang tersedia.",
    )
    parser.add_argument(
        "--prepare-models", action="store_true",
        help="Unduh dan verifikasi model CPU yang cocok sebelum ekstraksi (berguna saat run pertama).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExtractorConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        recursive=not args.non_recursive,
        tier=args.tier,
        ocr_mode=args.ocr_mode,
        skip_valid_results=not args.no_resume,
        continue_on_error=not args.stop_on_error,
        model_source=args.model_source,
        cpu_only=True,
    )
    # Import pipeline only after the package paths/config are initialized. The
    # MinerU package itself is imported lazily by runner after CPU env vars set.
    from .pipeline import run_pipeline

    result = run_pipeline(config, prepare_model_files=args.prepare_models)
    print("\nRingkasan final:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
