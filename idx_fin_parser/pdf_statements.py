from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pdfplumber

from .utils import (
    ItemNode,
    extract_amount_tokens,
    find_years_in_order,
    looks_like_section_header,
    normalize_text,
    split_bilingual_if_possible,
    split_label_amounts,
    token_to_int,
)

@dataclass
class StatementResult:
    statement_type: str
    years: List[int]
    pages: List[int]
    sections: Dict[str, List[ItemNode]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.statement_type,
            "years": self.years,
            "pages": self.pages,
            "sections": {k: [n.to_dict() for n in v] for k, v in self.sections.items()},
        }

def _find_page_index(pdf: pdfplumber.PDF, patterns: List[str], start: int = 0) -> Optional[int]:
    pats = [p.lower() for p in patterns]
    for i in range(start, len(pdf.pages)):
        text = (pdf.pages[i].extract_text() or "").lower()
        if any(p in text for p in pats):
            return i
    return None

def _collect_pages_until(pdf: pdfplumber.PDF, start_idx: int, stop_patterns: List[str]) -> List[int]:
    stop_pats = [p.lower() for p in stop_patterns]
    pages: List[int] = []
    for i in range(start_idx, len(pdf.pages)):
        text = (pdf.pages[i].extract_text() or "").lower()
        if i != start_idx and any(p in text for p in stop_pats):
            break
        pages.append(i)
    return pages

def _merge_wrapped_lines(lines: List[str], years: List[int]) -> List[str]:
    """
    Merge wrapped lines caused by PDF line breaking.

    We handle:
    1) Header/label wrapping: consecutive lines WITHOUT amounts where the next line looks like continuation.
    2) Value-row wrapping: a line WITH amounts followed by tail lines WITHOUT amounts (qualifiers like 'pihak ketiga').
    """
    merged: List[str] = []
    i = 0
    while i < len(lines):
        cur = normalize_text(lines[i])
        if not cur:
            i += 1
            continue

        cur_amt = extract_amount_tokens(cur, years)

        # Case 2: value row + wrapped tail lines
        if cur_amt:
            buf = cur
            j = i + 1
            while j < len(lines):
                nxt = normalize_text(lines[j])
                if not nxt:
                    j += 1
                    continue
                nxt_amt = extract_amount_tokens(nxt, years)
                if nxt_amt:
                    break
                if looks_like_section_header(nxt) is not None:
                    break
                if (nxt[:1].islower()) or (len(nxt.split()) <= 4):
                    buf = f"{buf} {nxt}".strip()
                    j += 1
                    continue
                break
            merged.append(buf)
            i = j
            continue

        # Case 1: label/header wrapping
        j = i + 1
        buf = cur
        while j < len(lines):
            nxt = normalize_text(lines[j])
            if not nxt:
                j += 1
                continue
            nxt_amt = extract_amount_tokens(nxt, years)
            if nxt_amt:
                break
            if looks_like_section_header(nxt) is not None:
                break
            if (nxt[:1].islower()) or (len(nxt.split()) <= 3):
                buf = f"{buf} {nxt}".strip()
                j += 1
                continue
            break
        merged.append(buf)
        i = j
    return merged

def _parse_lines_to_tree(lines: List[str], years: List[int]) -> Dict[str, List[ItemNode]]:
    """
    Conservative hierarchy:
    - section headers reset parent
    - "parent" is a no-amount line
    - amount lines attach to current parent if prefix matches
    """
    year_keys = [f"{y}-12-31" for y in years]
    sections: Dict[str, List[ItemNode]] = {}
    current_section = "unknown"
    current_parent: Optional[ItemNode] = None

    def add_root(node: ItemNode) -> None:
        sections.setdefault(current_section, []).append(node)

    lines2 = _merge_wrapped_lines(lines, years)

    for raw in lines2:
        ln = normalize_text(raw)
        if not ln:
            continue

        amt_tokens = extract_amount_tokens(ln, years)
        amt_tokens = amt_tokens[-len(years):] if amt_tokens else []
        left, right = split_label_amounts(ln, amt_tokens)

        # Section header
        sec = looks_like_section_header(left)
        if sec and not amt_tokens:
            current_section = sec
            current_parent = None
            continue

        # Parent line (no amounts)
        if not amt_tokens:
            if not right:
                left, right = split_bilingual_if_possible(left)
            node = ItemNode(label=left, label_right=right)
            add_root(node)
            current_parent = node
            continue

        # Value line
        values: Dict[str, Optional[int]] = {}
        padded = [None] * (len(years) - len(amt_tokens)) + amt_tokens
        for yk, tok in zip(year_keys, padded):
            values[yk] = token_to_int(tok) if tok is not None else None

        node_label = left
        attached = False

        if current_parent:
            pl = current_parent.label.lower()
            cl = node_label.lower()
            if cl.startswith(pl):
                short = node_label[len(current_parent.label):].strip()
                node_label = short if short else node_label
                current_parent.children.append(ItemNode(label=node_label, label_right=right, values=values))
                attached = True

        if not attached:
            add_root(ItemNode(label=node_label, label_right=right, values=values))

        # Totals usually end a block
        if re.search(r"\b(jumlah|total|subtotal)\b", left.lower()):
            current_parent = None

    return sections

def extract_statement_financial_position(pdf_path: str) -> StatementResult:
    with pdfplumber.open(pdf_path) as pdf:
        start = _find_page_index(pdf, ["statement of financial position", "laporan posisi keuangan"])
        if start is None:
            raise ValueError("Could not find 'Statement of financial position' in PDF.")
        pages = _collect_pages_until(pdf, start, ["statement of profit or loss", "laporan laba rugi"])

        all_lines: List[str] = []
        for i in pages:
            txt = pdf.pages[i].extract_text() or ""
            all_lines.extend(txt.splitlines())

        years = find_years_in_order(all_lines)
        sections = _parse_lines_to_tree(all_lines, years)
        return StatementResult("financial_position", years, pages, sections)

def extract_statement_profit_loss(pdf_path: str) -> StatementResult:
    with pdfplumber.open(pdf_path) as pdf:
        start = _find_page_index(pdf, ["statement of profit or loss", "laporan laba rugi"])
        if start is None:
            raise ValueError("Could not find 'Statement of profit or loss' in PDF.")
        pages = _collect_pages_until(
            pdf,
            start,
            ["statement of cash flows", "laporan arus kas", "catatan atas laporan keuangan", "notes to the financial statements"],
        )

        all_lines: List[str] = []
        for i in pages:
            txt = pdf.pages[i].extract_text() or ""
            all_lines.extend(txt.splitlines())

        years = find_years_in_order(all_lines)
        sections = _parse_lines_to_tree(all_lines, years)
        return StatementResult("profit_or_loss", years, pages, sections)