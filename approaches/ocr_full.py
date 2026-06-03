"""Tesseract OCR on every page + text-line parser.

force_ocr=True bypasses the text layer entirely — represents an OCR-only
pipeline as if the source PDF had no text layer.
"""
from __future__ import annotations

import time

from idx_fin_parser.pdf_statements import (
    extract_statement_financial_position,
    extract_statement_profit_loss,
)
from idx_fin_parser.unified import build_unified_output

from . import register


def run(pdf_path: str) -> tuple[dict, float]:
    t0 = time.perf_counter()
    fp = extract_statement_financial_position(pdf_path, force_ocr=True)
    pl = extract_statement_profit_loss(pdf_path, force_ocr=True)
    out = build_unified_output(
        source_pdf=str(pdf_path),
        approach="ocr_full",
        statements=[fp.to_dict(), pl.to_dict()],
        meta={"use_ocr": False, "force_ocr": True, "ocr_lang": "ind+eng"},
    )
    return out, time.perf_counter() - t0


register("ocr_full", run)
