from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExtractorConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-extractor",
        description="Ekstraksi PDF MinerU 4.x ke database literatur; mendukung runtime CPU atau NVIDIA GPU.",
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder berisi PDF.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder output (wajib terpisah dari input).")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="cpu atau gpu (NVIDIA CUDA; mis. Colab T4).")
    parser.add_argument(
        "--vlm-engine", choices=["auto", "llama-cpp", "vllm"], default="auto",
        help="Engine VLM untuk tier standard/advanced. auto memilih llama-cpp pada CPU dan vLLM pada GPU; llama-cpp juga dapat dipakai sebagai fallback kompatibilitas di GPU.",
    )
    parser.add_argument("--tier", choices=["flash", "basic", "standard", "advanced"], default="standard")
    parser.add_argument("--ocr-mode", default="auto", choices=["auto", "txt", "ocr"])
    parser.add_argument("--non-recursive", action="store_true", help="Jangan mencari PDF di subfolder.")
    parser.add_argument("--no-resume", action="store_true", help="Proses ulang walaupun hasil valid sudah ada.")
    parser.add_argument("--stop-on-error", action="store_true", help="Hentikan batch saat satu PDF gagal.")
    parser.add_argument(
        "--model-source", choices=["auto", "huggingface", "modelscope"], default="auto",
        help="Sumber model. 'auto' membiarkan MinerU memilih sumber yang tersedia.",
    )
    parser.add_argument(
        "--prepare-models", action="store_true",
        help="Unduh dan verifikasi model untuk tier dan backend yang dipilih (jalankan pada penggunaan pertama).",
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
        runtime_mode=args.device,
        vlm_engine=args.vlm_engine,
    )
    # Import pipeline only after paths/config are parsed. Pipeline applies backend
    # settings before importing MinerU itself.
    from .pipeline import run_pipeline

    result = run_pipeline(config, prepare_model_files=args.prepare_models)
    print("\nRingkasan final:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
