"""Naive baseline: text-layer + regex for amounts. No coordinate awareness.

Establishes a *floor* for the thesis comparison — shows how much accuracy comes
from the column-layout logic vs raw text scanning.
"""
from __future__ import annotations

import re
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


_AMOUNT_RE = re.compile(r"\(?\s*-?\d{1,3}(?:[,.]\d{3})+\s*\)?|\(?\s*\d+\s*\)?")

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


def _parse_lines(lines, years):
    year_keys = [f"{y}-12-31" for y in years]
    years_set = set(years)
    nodes = []
    for raw in lines:
        ln = normalize_text(raw)
        if not ln:
            continue
        tokens = []
        for m in _AMOUNT_RE.finditer(ln):
            tok = m.group(0).strip()
            inner = tok.strip("() ").replace(",", "").replace(".", "")
            if inner.isdigit() and len(inner) == 4 and int(inner) in years_set:
                continue
            tokens.append((m.start(), tok))
        if not tokens:
            nodes.append(make_node(label=ln, level=0))
            continue
        first_pos = tokens[0][0]
        label = ln[:first_pos].strip()
        if not label:
            continue
        amounts = [parse_value(t) for _, t in tokens]
        amounts = amounts[-len(years):] if amounts else []
        padded = [None] * (len(years) - len(amounts)) + amounts
        values = {yk: v for yk, v in zip(year_keys, padded)}
        nodes.append(make_node(label=label, values=values, level=0))
    return nodes


def _statement(pdf_path, statement_type, start_patterns, stop_patterns):
    with pdfplumber.open(pdf_path) as pdf:
        pages = _find_pages(pdf, start_patterns, stop_patterns)
        if not pages:
            return build_statement(statement_type, [], [], {})
        all_lines = []
        for i in pages:
            t = pdf.pages[i].extract_text() or ""
            all_lines.extend(t.splitlines())
        years = find_years_in_order([normalize_text(t) for t in all_lines])
        nodes = _parse_lines(all_lines, years)
        sections = split_into_sections(nodes)
        return build_statement(statement_type, years, pages, sections)


def run(pdf_path: str) -> tuple[dict, float]:
    t0 = time.perf_counter()
    fp = _statement(pdf_path, "financial_position", START_FP, STOP_FP)
    pl = _statement(pdf_path, "profit_or_loss", START_PL, STOP_PL)
    out = build_unified_output(
        source_pdf=str(pdf_path),
        approach="baseline_regex",
        statements=[fp, pl],
        meta={"note": "no coordinate awareness; no hierarchy detection"},
    )
    return out, time.perf_counter() - t0


register("baseline_regex", run)
