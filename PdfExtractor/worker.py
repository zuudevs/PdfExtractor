from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from .config import ExtractorConfig
from .runner import _process_one_pdf_impl


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python -m PdfExtractor.worker PAYLOAD.json RESULT.json", file=sys.stderr)
        return 2

    payload_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])
    document_id = "unknown"
    try:
        payload: dict[str, Any] = json.loads(payload_path.read_text(encoding="utf-8"))
        item = payload["item"]
        document_id = str(item.get("document_id", "unknown"))
        config = ExtractorConfig(**payload["config"])
        result = _process_one_pdf_impl(item, config, str(payload["mineru_version"]))
        result_path.write_text(
            json.dumps({"result": result}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 0
    except Exception as exc:
        result_path.write_text(
            json.dumps(
                {
                    "result": {
                        "document_id": document_id,
                        "status": "FAILED_WORKER",
                        "message": f"{type(exc).__name__}: {exc}",
                    },
                    "traceback": traceback.format_exc()[-12000:],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
