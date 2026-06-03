"""pdfplumber.extract_tables() — library-default table extraction.

Ablation against the custom column-coordinate parser. Same library, but uses
pdfplumber's built-in ruling-line / character-cluster detection instead of our
custom word x/y logic.
"""
from __future__ import annotations

import time

import pdfplumber

from idx_fin_parser.unified import (
    build_statement,
    build_unified_output,
    make_node,
    parse_value,
    split_into_sections,
)
from idx_fin_parser.utils import find_years_in_order, normalize_text

from . import register

START_FP = ["statement of financial position", "laporan posisi keuangan"]
STOP_FP = ["statement of profit or loss", "laporan laba rugi"]
START_PL = ["statement of profit or loss", "laporan laba rugi"]
STOP_PL = [
    "statement of changes in equity", "laporan perubahan ekuitas",
    "statement of cash flows", "laporan arus kas",
    "catatan atas laporan keuangan", "notes to the financial statements",
]


def _find_pages(pdf, start_patterns, stop_patterns):
    start_pats = [p.lower() for p in start_patterns]
    stop_pats = [p.lower() for p in stop_patterns]
    start = None
    for i, p in enumerate(pdf.pages):
        text = (p.extract_text() or "").lower()
        if any(s in text for s in start_pats):
            start = i
            break
    if start is None:
        return []
    pages = [start]
    for i in range(start + 1, len(pdf.pages)):
        text = (pdf.pages[i].extract_text() or "").lower()
        if any(s in text for s in stop_pats):
            break
        pages.append(i)
    return pages


def _table_rows_to_nodes(rows, years):
    year_keys = [f"{y}-12-31" for y in years]
    nodes = []
    for row in rows:
        if not row or not any(row):
            continue
        cells = [normalize_text(c or "") for c in row]
        non_empty = [c for c in cells if c]
        if not non_empty:
            continue
        id_label = non_empty[0]
        en_label = ""
        amounts: list[int | None] = []
        for c in non_empty[1:]:
            v = parse_value(c)
            if v is not None:
                amounts.append(v)
            elif not en_label:
                en_label = c
        amounts = amounts[-len(years):] if amounts else []
        padded = [None] * (len(years) - len(amounts)) + amounts
        values = {yk: v for yk, v in zip(year_keys, padded)}
        nodes.append(make_node(label=id_label, label_en=en_label,
                               values=values, level=0))
    return nodes


_TABLE_SETTINGS = {
    # BBNI has background-coloured cells but no visible ruling lines, so the
    # default 'lines' strategy returns 0 tables. Use whitespace-based detection.
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 3,
    "join_tolerance": 3,
}


def _statement(pdf_path, statement_type, start_patterns, stop_patterns):
    with pdfplumber.open(pdf_path) as pdf:
        pages = _find_pages(pdf, start_patterns, stop_patterns)
        if not pages:
            return build_statement(statement_type, [], [], {})
        rows: list = []
        all_lines = []
        for i in pages:
            page = pdf.pages[i]
            for tbl in page.extract_tables(table_settings=_TABLE_SETTINGS) or []:
                rows.extend(tbl)
            all_lines.extend((page.extract_text() or "").splitlines())
        years = find_years_in_order([normalize_text(t) for t in all_lines])
        nodes = _table_rows_to_nodes(rows, years)
        sections = split_into_sections(nodes)
        return build_statement(statement_type, years, pages, sections)


def run(pdf_path: str) -> tuple[dict, float]:
    t0 = time.perf_counter()
    fp = _statement(pdf_path, "financial_position", START_FP, STOP_FP)
    pl = _statement(pdf_path, "profit_or_loss", START_PL, STOP_PL)
    out = build_unified_output(
        source_pdf=str(pdf_path),
        approach="pdfplumber_tables",
        statements=[fp, pl],
        meta={"note": "uses pdfplumber.extract_tables() defaults"},
    )
    return out, time.perf_counter() - t0


register("pdfplumber_tables", run)
