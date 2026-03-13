#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from idx_fin_parser.pdf_statements import (
    extract_statement_financial_position,
    extract_statement_profit_loss,
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
    args = ap.parse_args()

    data = build_output(args.pdf, use_ocr=args.ocr, ocr_lang=args.ocr_lang)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path.resolve()}")

if __name__ == "__main__":
    main()