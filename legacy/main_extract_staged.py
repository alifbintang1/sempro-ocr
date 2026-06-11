#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from idx_fin_parser.pdf_statements import extract_with_stages


def _write_process_files(data: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "00_overview.json"
    meta = {
        "source_pdf": data.get("source_pdf"),
        "meta": data.get("meta", {}),
        "statement_types": list(data.get("stages", {}).keys()),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    statement_map = {s.get("type"): s for s in data.get("statements", [])}

    for statement_type, stage in data.get("stages", {}).items():
        stmt_dir = out_dir / statement_type
        stmt_dir.mkdir(parents=True, exist_ok=True)

        (stmt_dir / "01_meta.json").write_text(
            json.dumps(
                {
                    "statement_type": statement_type,
                    "start_page": stage.get("start_page"),
                    "pages": stage.get("pages", []),
                    "years": stage.get("years", []),
                    "start_patterns": stage.get("start_patterns", []),
                    "stop_patterns": stage.get("stop_patterns", []),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        (stmt_dir / "02_raw_text_by_page.json").write_text(
            json.dumps(stage.get("raw_text_by_page", []), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        raw_text_joined = []
        for p in stage.get("raw_text_by_page", []):
            page_no = p.get("page")
            page_text = p.get("text", "")
            raw_text_joined.append(f"===== PAGE {page_no} =====")
            raw_text_joined.append(page_text)
            raw_text_joined.append("")
        (stmt_dir / "02_raw_text_combined.txt").write_text("\n".join(raw_text_joined), encoding="utf-8")

        (stmt_dir / "03_normalized_lines.txt").write_text(
            "\n".join(stage.get("normalized_lines", [])),
            encoding="utf-8",
        )

        (stmt_dir / "04_merged_lines.txt").write_text(
            "\n".join(stage.get("merged_lines", [])),
            encoding="utf-8",
        )

        (stmt_dir / "05_sections.json").write_text(
            json.dumps(stage.get("sections", {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        (stmt_dir / "06_final_statement.json").write_text(
            json.dumps(statement_map.get(statement_type, {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract IDX statements with full step-by-step pipeline (PDF -> raw text -> normalized -> merged -> JSON)."
    )
    ap.add_argument("pdf", help="Path to IDX PDF file")
    ap.add_argument("-o", "--out", default="output_stages.json", help="Output JSON path")
    ap.add_argument("--process-dir", default="", help="Optional folder to write per-stage process files")
    ap.add_argument("--ocr", action="store_true", help="Use OCR fallback when PDF text layer is empty")
    ap.add_argument("--ocr-lang", default="ind+eng", help="Tesseract OCR language (default: ind+eng)")
    args = ap.parse_args()

    data = extract_with_stages(args.pdf, use_ocr=args.ocr, ocr_lang=args.ocr_lang)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote staged output: {out_path.resolve()}")

    if args.process_dir:
        process_dir = Path(args.process_dir)
        _write_process_files(data, process_dir)
        print(f"Wrote process files: {process_dir.resolve()}")


if __name__ == "__main__":
    main()
