"""camelot-py table extractor.

Third-party PDF-table tool widely used in finance/data-science. Uses whitespace
clustering (stream flavor) — lattice flavor requires visible ruling lines and
Ghostscript, which the BBNI PDF lacks. Module name kept as 'lattice' for
historical reasons; ``flavor`` controls the actual algorithm.
"""
from __future__ import annotations

import time

import camelot  # raises ImportError if missing — handled by approaches/__init__

from idx_fin_parser.unified import (
    build_statement,
    build_unified_output,
    make_node,
    parse_value,
    split_into_sections,
)
from idx_fin_parser.utils import find_years_in_order, normalize_text

from . import register

FLAVOR = "stream"  # 'lattice' needs Ghostscript + visible borders

START_FP = ["statement of financial position", "laporan posisi keuangan"]
STOP_FP = ["statement of profit or loss", "laporan laba rugi"]
START_PL = ["statement of profit or loss", "laporan laba rugi"]
STOP_PL = [
    "statement of changes in equity", "laporan perubahan ekuitas",
    "statement of cash flows", "laporan arus kas",
    "catatan atas laporan keuangan", "notes to the financial statements",
]


def _find_pages(pdf_path, start_patterns, stop_patterns):
    """Camelot uses 1-indexed pages."""
    import pdfplumber
    start_pats = [p.lower() for p in start_patterns]
    stop_pats = [p.lower() for p in stop_patterns]
    pages_1idx: list[int] = []
    with pdfplumber.open(pdf_path) as pdf:
        start = None
        for i, p in enumerate(pdf.pages):
            text = (p.extract_text() or "").lower()
            if any(s in text for s in start_pats):
                start = i
                break
        if start is None:
            return []
        pages_1idx.append(start + 1)
        for i in range(start + 1, len(pdf.pages)):
            text = (pdf.pages[i].extract_text() or "").lower()
            if any(s in text for s in stop_pats):
                break
            pages_1idx.append(i + 1)
    return pages_1idx


def _rows_to_nodes(df_rows, years):
    year_keys = [f"{y}-12-31" for y in years]
    nodes = []
    for row in df_rows:
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


def _statement(pdf_path, statement_type, start_patterns, stop_patterns):
    pages_1idx = _find_pages(pdf_path, start_patterns, stop_patterns)
    if not pages_1idx:
        return build_statement(statement_type, [], [], {})

    pages_arg = ",".join(str(p) for p in pages_1idx)
    tables = camelot.read_pdf(pdf_path, pages=pages_arg, flavor=FLAVOR)

    all_rows: list[list[str]] = []
    all_text_lines: list[str] = []
    for t in tables:
        for _, r in t.df.iterrows():
            all_rows.append(r.tolist())
            all_text_lines.append(" ".join(str(c) for c in r if c))

    years = find_years_in_order([normalize_text(t) for t in all_text_lines])
    nodes = _rows_to_nodes(all_rows, years)
    sections = split_into_sections(nodes)
    return build_statement(
        statement_type, years,
        [p - 1 for p in pages_1idx],  # back to 0-indexed for schema
        sections,
    )


def run(pdf_path: str) -> tuple[dict, float]:
    t0 = time.perf_counter()
    fp = _statement(pdf_path, "financial_position", START_FP, STOP_FP)
    pl = _statement(pdf_path, "profit_or_loss", START_PL, STOP_PL)
    out = build_unified_output(
        source_pdf=str(pdf_path),
        approach="camelot_stream",
        statements=[fp, pl],
        meta={"library": "camelot-py", "flavor": FLAVOR},
    )
    return out, time.perf_counter() - t0


register("camelot_stream", run)
