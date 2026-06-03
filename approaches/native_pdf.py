"""Custom column-coordinate parser using pdfplumber x/y word positions.

This is the canonical "rule-based, layout-aware" approach in idx_fin_parser/.
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
    fp = extract_statement_financial_position(pdf_path)
    pl = extract_statement_profit_loss(pdf_path)
    out = build_unified_output(
        source_pdf=str(pdf_path),
        approach="native_pdf",
        statements=[fp.to_dict(), pl.to_dict()],
        meta={"use_ocr": False, "force_ocr": False},
    )
    return out, time.perf_counter() - t0


register("native_pdf", run)
