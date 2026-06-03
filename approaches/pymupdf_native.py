"""PyMuPDF (fitz) word-coordinate parser.

Same algorithm as native_pdf but uses PyMuPDF's text extraction instead of
pdfplumber. Library-comparison: do both libraries return equivalent word
coordinates / does that change downstream parsing quality?
"""
from __future__ import annotations

import time

import fitz  # PyMuPDF

from idx_fin_parser.unified import (
    build_statement,
    build_unified_output,
    split_into_sections,
)
from idx_fin_parser.utils import (
    ItemNode,
    _split_bilingual_row,
    assign_levels,
    find_years_in_order,
    merge_continuation_rows,
    normalize_text,
)
from idx_fin_parser.pdf_statements import _parse_structured_rows_to_tree

from . import register

START_FP = ["statement of financial position", "laporan posisi keuangan"]
STOP_FP = ["statement of profit or loss", "laporan laba rugi"]
START_PL = ["statement of profit or loss", "laporan laba rugi"]
STOP_PL = [
    "statement of changes in equity", "laporan perubahan ekuitas",
    "statement of cash flows", "laporan arus kas",
    "catatan atas laporan keuangan", "notes to the financial statements",
]


def _find_pages(doc: "fitz.Document", start_patterns, stop_patterns):
    start_pats = [p.lower() for p in start_patterns]
    stop_pats = [p.lower() for p in stop_patterns]
    start = None
    for i in range(len(doc)):
        text = doc[i].get_text("text").lower()
        if any(s in text for s in start_pats):
            start = i
            break
    if start is None:
        return []
    pages = [start]
    for i in range(start + 1, len(doc)):
        text = doc[i].get_text("text").lower()
        if any(s in text for s in stop_pats):
            break
        pages.append(i)
    return pages


def _is_amount_word(text: str, years_set: set) -> bool:
    """Lift of utils.is_amount_word without the years_set None fallback."""
    import re
    AMOUNT_WORD_RE = re.compile(r'^\(?\d{1,3}(?:,\d{3})*\)?$|^\(?\d+\)?$')
    t = text.strip()
    if not AMOUNT_WORD_RE.match(t):
        return False
    inner = t.strip("()")
    try:
        v = int(inner.replace(",", ""))
        if years_set and len(inner.replace(",", "")) == 4 and v in years_set:
            return False
    except ValueError:
        pass
    return True


def _extract_rows_pymupdf(page: "fitz.Page", years_set: set) -> list[dict]:
    """PyMuPDF analogue of utils.extract_rows_by_column."""
    # words: list of (x0, y0, x1, y1, text, block, line, word_no)
    words = page.get_text("words")
    if not words:
        return []
    word_dicts = [{
        "x0": w[0], "top": w[1], "x1": w[2], "bottom": w[3], "text": w[4],
    } for w in words]

    # Group by y (3pt tolerance, same as pdfplumber path)
    row_map: dict[int, list] = {}
    for w in word_dicts:
        y_key = round(w["top"] / 3) * 3
        row_map.setdefault(y_key, []).append(w)

    # Merge parenthesised amounts (same logic as utils._merge_paren_amounts)
    import re
    def merge_paren(words):
        out = []
        i = 0
        while i < len(words):
            w = words[i]
            if (w["text"] == "(" and i + 2 < len(words)
                    and re.match(r"^[\d,]+$", words[i + 1]["text"])
                    and words[i + 2]["text"] == ")"):
                out.append({**w, "x1": words[i + 2]["x1"],
                            "text": f"({words[i + 1]['text']})"})
                i += 3
            else:
                out.append(w)
                i += 1
        return out

    # Detect EN column threshold
    en_x_starts: list[float] = []
    for y in sorted(row_map):
        rw = merge_paren(sorted(row_map[y], key=lambda w: w["x0"]))
        amt_idx = [i for i, w in enumerate(rw) if _is_amount_word(w["text"], years_set)]
        if amt_idx and amt_idx[-1] + 1 < len(rw):
            en_x_starts.append(rw[amt_idx[-1] + 1]["x0"])
    en_threshold = min(en_x_starts) if en_x_starts else page.rect.width * 0.65

    rows = []
    for y in sorted(row_map):
        rw = merge_paren(sorted(row_map[y], key=lambda w: w["x0"]))
        amt_idx = [i for i, w in enumerate(rw) if _is_amount_word(w["text"], years_set)]
        x0_first = rw[0]["x0"] if rw else 0.0

        if not amt_idx:
            # Use gap-based bilingual split (same fix as pdfplumber path)
            id_t, en_t = _split_bilingual_row(rw, en_threshold)
            if id_t or en_t:
                rows.append({"id_label": id_t, "amounts": [], "en_label": en_t,
                             "x0_first": x0_first})
            continue

        first, last = amt_idx[0], amt_idx[-1]
        id_label = normalize_text(" ".join(rw[i]["text"] for i in range(first)))
        amounts = [rw[i]["text"] for i in amt_idx]
        en_label = normalize_text(
            " ".join(rw[i]["text"] for i in range(last + 1, len(rw))))
        rows.append({"id_label": id_label, "amounts": amounts,
                     "en_label": en_label, "x0_first": x0_first})

    return merge_continuation_rows(rows)


def _statement(doc, statement_type, start_patterns, stop_patterns):
    pages = _find_pages(doc, start_patterns, stop_patterns)
    if not pages:
        return build_statement(statement_type, [], [], {})

    raw_lines = []
    for i in pages:
        raw_lines.extend(doc[i].get_text("text").splitlines())
    years = find_years_in_order([normalize_text(t) for t in raw_lines])
    years_set = set(years)

    all_rows = []
    for i in pages:
        all_rows.extend(_extract_rows_pymupdf(doc[i], years_set))
    all_rows = assign_levels(all_rows)
    sections = _parse_structured_rows_to_tree(all_rows, years)

    return {
        "type": statement_type,
        "years": years,
        "pages": pages,
        "sections": {k: [n.to_dict() for n in v] for k, v in sections.items()},
    }


def run(pdf_path: str) -> tuple[dict, float]:
    t0 = time.perf_counter()
    doc = fitz.open(pdf_path)
    try:
        fp = _statement(doc, "financial_position", START_FP, STOP_FP)
        pl = _statement(doc, "profit_or_loss", START_PL, STOP_PL)
    finally:
        doc.close()
    out = build_unified_output(
        source_pdf=str(pdf_path),
        approach="pymupdf_native",
        statements=[fp, pl],
        meta={"library": "PyMuPDF (fitz)"},
    )
    return out, time.perf_counter() - t0


register("pymupdf_native", run)
