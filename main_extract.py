#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from idx_fin_parser.pdf_statements import (
    extract_with_stages,
    extract_statement_financial_position,
    extract_statement_profit_loss,
)


def _write_process_files(data: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    statement_map = {s.get("type"): s for s in data.get("statements", [])}

    for statement_type, stage in data.get("stages", {}).items():
        stmt_dir = out_dir / statement_type
        stmt_dir.mkdir(parents=True, exist_ok=True)

        (stmt_dir / "raw_text_by_page.json").write_text(
            json.dumps(stage.get("raw_text_by_page", []), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (stmt_dir / "normalized_lines.txt").write_text(
            "\n".join(stage.get("normalized_lines", [])),
            encoding="utf-8",
        )
        (stmt_dir / "merged_lines.txt").write_text(
            "\n".join(stage.get("merged_lines", [])),
            encoding="utf-8",
        )
        (stmt_dir / "sections.json").write_text(
            json.dumps(stage.get("sections", {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (stmt_dir / "final_statement.json").write_text(
            json.dumps(statement_map.get(statement_type, {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

def build_output(pdf_path: str, use_ocr: bool = False, ocr_lang: str = "ind+eng") -> Dict[str, Any]:
    fp = extract_statement_financial_position(pdf_path, use_ocr=use_ocr, ocr_lang=ocr_lang)
    pl = extract_statement_profit_loss(pdf_path, use_ocr=use_ocr, ocr_lang=ocr_lang)

    return {
        "source_pdf": str(pdf_path),
        "meta": {"use_ocr": use_ocr, "ocr_lang": ocr_lang},
        "statements": [fp.to_dict(), pl.to_dict()],
    }

def main() -> None:
    ap = argparse.ArgumentParser(description="Extract IDX financial statements into JSON.")
    ap.add_argument("pdf", help="Path to IDX PDF file")
    ap.add_argument("-o", "--out", default="output.json", help="Output JSON path")
    ap.add_argument("--ocr", action="store_true", help="Use OCR fallback when PDF text layer is empty")
    ap.add_argument("--ocr-lang", default="ind+eng", help="Tesseract OCR language (default: ind+eng)")
    ap.add_argument("--process-dir", default="", help="Optional folder to write per-stage process files")
    ap.add_argument(
        "--stages-out",
        default="",
        help="Optional path to write step-by-step extraction JSON (raw text, normalized lines, merged lines, parsed sections)",
    )
    args = ap.parse_args()

    data = build_output(args.pdf, use_ocr=args.ocr, ocr_lang=args.ocr_lang)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path.resolve()}")

    if args.stages_out:
        stages_data = extract_with_stages(args.pdf, use_ocr=args.ocr, ocr_lang=args.ocr_lang)
        stages_path = Path(args.stages_out)
        stages_path.write_text(json.dumps(stages_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote stages: {stages_path.resolve()}")

        if args.process_dir:
            process_dir = Path(args.process_dir)
            _write_process_files(stages_data, process_dir)
            print(f"Wrote process files: {process_dir.resolve()}")
    elif args.process_dir:
        stages_data = extract_with_stages(args.pdf, use_ocr=args.ocr, ocr_lang=args.ocr_lang)
        process_dir = Path(args.process_dir)
        _write_process_files(stages_data, process_dir)
        print(f"Wrote process files: {process_dir.resolve()}")

if __name__ == "__main__":
    main()