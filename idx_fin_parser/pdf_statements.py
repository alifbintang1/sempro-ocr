from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


def _extract_page_text(page: pdfplumber.page.Page, use_ocr: bool = False, ocr_lang: str = "ind+eng") -> str:
    text = page.extract_text() or ""
    if text.strip() or not use_ocr:
        return text

    try:
        import pytesseract
    except ImportError as exc:
        raise ValueError("OCR mode requires 'pytesseract'. Run: pip install pytesseract") from exc

    if shutil.which("tesseract") is None:
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                pytesseract.pytesseract.tesseract_cmd = candidate
                break

    try:
        pil_img = page.to_image(resolution=250).original
        ocr_text = pytesseract.image_to_string(pil_img, lang=ocr_lang, config="--psm 6")
        return ocr_text or ""
    except pytesseract.pytesseract.TesseractNotFoundError as exc:
        raise ValueError(
            "Tesseract engine not found. Install Tesseract OCR and ensure it is in PATH."
        ) from exc


def _find_page_index(
    pdf: pdfplumber.PDF,
    patterns: List[str],
    start: int = 0,
    use_ocr: bool = False,
    ocr_lang: str = "ind+eng",
) -> Optional[int]:
    pats = [p.lower() for p in patterns]
    for i in range(start, len(pdf.pages)):
        text = _extract_page_text(pdf.pages[i], use_ocr=use_ocr, ocr_lang=ocr_lang).lower()
        if any(p in text for p in pats):
            return i
    return None


def _collect_pages_until(
    pdf: pdfplumber.PDF,
    start_idx: int,
    stop_patterns: List[str],
    use_ocr: bool = False,
    ocr_lang: str = "ind+eng",
) -> List[int]:
    stop_pats = [p.lower() for p in stop_patterns]
    pages: List[int] = []
    for i in range(start_idx, len(pdf.pages)):
        text = _extract_page_text(pdf.pages[i], use_ocr=use_ocr, ocr_lang=ocr_lang).lower()
        if i != start_idx and any(p in text for p in stop_pats):
            break
        pages.append(i)
    return pages


def _merge_wrapped_lines(lines: List[str], years: List[int]) -> List[str]:
    merged: List[str] = []
    i = 0
    while i < len(lines):
        cur = normalize_text(lines[i])
        if not cur:
            i += 1
            continue

        cur_amt = extract_amount_tokens(cur, years)

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


def _parse_merged_lines_to_tree(lines: List[str], years: List[int]) -> Dict[str, List[ItemNode]]:
    year_keys = [f"{y}-12-31" for y in years]
    sections: Dict[str, List[ItemNode]] = {}
    current_section = "unknown"
    current_parent: Optional[ItemNode] = None

    def add_root(node: ItemNode) -> None:
        sections.setdefault(current_section, []).append(node)

    for raw in lines:
        ln = normalize_text(raw)
        if not ln:
            continue

        amt_tokens = extract_amount_tokens(ln, years)
        amt_tokens = amt_tokens[-len(years):] if amt_tokens else []
        left, right = split_label_amounts(ln, amt_tokens)

        sec = looks_like_section_header(left)
        if sec and not amt_tokens:
            current_section = sec
            current_parent = None
            continue

        if not amt_tokens:
            if not right:
                left, right = split_bilingual_if_possible(left)
            node = ItemNode(label=left, label_right=right)
            add_root(node)
            current_parent = node
            continue

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

        if re.search(r"\b(jumlah|total|subtotal)\b", left.lower()):
            current_parent = None

    return sections


def _parse_lines_to_tree(lines: List[str], years: List[int]) -> Dict[str, List[ItemNode]]:
    merged = _merge_wrapped_lines(lines, years)
    return _parse_merged_lines_to_tree(merged, years)


def _extract_statement_generic(
    pdf_path: str,
    statement_type: str,
    start_patterns: List[str],
    stop_patterns: List[str],
    use_ocr: bool = False,
    ocr_lang: str = "ind+eng",
) -> StatementResult:
    with pdfplumber.open(pdf_path) as pdf:
        start = _find_page_index(pdf, start_patterns, use_ocr=use_ocr, ocr_lang=ocr_lang)
        if start is None:
            raise ValueError(f"Could not find '{statement_type}' in PDF.")

        pages = _collect_pages_until(pdf, start, stop_patterns, use_ocr=use_ocr, ocr_lang=ocr_lang)

        all_lines: List[str] = []
        for i in pages:
            txt = _extract_page_text(pdf.pages[i], use_ocr=use_ocr, ocr_lang=ocr_lang)
            all_lines.extend(txt.splitlines())

        years = find_years_in_order(all_lines)
        sections = _parse_lines_to_tree(all_lines, years)
        return StatementResult(statement_type, years, pages, sections)


def _extract_statement_with_stages(
    pdf_path: str,
    statement_type: str,
    start_patterns: List[str],
    stop_patterns: List[str],
    use_ocr: bool = False,
    ocr_lang: str = "ind+eng",
) -> Tuple[StatementResult, Dict[str, Any]]:
    with pdfplumber.open(pdf_path) as pdf:
        start = _find_page_index(pdf, start_patterns, use_ocr=use_ocr, ocr_lang=ocr_lang)
        if start is None:
            raise ValueError(f"Could not find '{statement_type}' in PDF.")

        pages = _collect_pages_until(pdf, start, stop_patterns, use_ocr=use_ocr, ocr_lang=ocr_lang)

        page_texts: List[Dict[str, Any]] = []
        raw_lines: List[str] = []
        for i in pages:
            txt = _extract_page_text(pdf.pages[i], use_ocr=use_ocr, ocr_lang=ocr_lang)
            page_texts.append({"page": i, "text": txt})
            raw_lines.extend(txt.splitlines())

        normalized_lines = [normalize_text(ln) for ln in raw_lines if normalize_text(ln)]
        years = find_years_in_order(normalized_lines)
        merged_lines = _merge_wrapped_lines(normalized_lines, years)
        sections = _parse_merged_lines_to_tree(merged_lines, years)

        result = StatementResult(statement_type, years, pages, sections)
        stages = {
            "statement_type": statement_type,
            "start_patterns": start_patterns,
            "stop_patterns": stop_patterns,
            "start_page": start,
            "pages": pages,
            "raw_text_by_page": page_texts,
            "normalized_lines": normalized_lines,
            "merged_lines": merged_lines,
            "years": years,
            "sections": {k: [n.to_dict() for n in v] for k, v in sections.items()},
        }
        return result, stages


def extract_statement_financial_position(
    pdf_path: str,
    use_ocr: bool = False,
    ocr_lang: str = "ind+eng",
) -> StatementResult:
    return _extract_statement_generic(
        pdf_path,
        statement_type="financial_position",
        start_patterns=["statement of financial position", "laporan posisi keuangan"],
        stop_patterns=["statement of profit or loss", "laporan laba rugi"],
        use_ocr=use_ocr,
        ocr_lang=ocr_lang,
    )


def extract_statement_profit_loss(
    pdf_path: str,
    use_ocr: bool = False,
    ocr_lang: str = "ind+eng",
) -> StatementResult:
    return _extract_statement_generic(
        pdf_path,
        statement_type="profit_or_loss",
        start_patterns=["statement of profit or loss", "laporan laba rugi"],
        stop_patterns=[
            "statement of cash flows",
            "laporan arus kas",
            "catatan atas laporan keuangan",
            "notes to the financial statements",
        ],
        use_ocr=use_ocr,
        ocr_lang=ocr_lang,
    )


def extract_with_stages(
    pdf_path: str,
    use_ocr: bool = False,
    ocr_lang: str = "ind+eng",
) -> Dict[str, Any]:
    fp_result, fp_stage = _extract_statement_with_stages(
        pdf_path,
        statement_type="financial_position",
        start_patterns=["statement of financial position", "laporan posisi keuangan"],
        stop_patterns=["statement of profit or loss", "laporan laba rugi"],
        use_ocr=use_ocr,
        ocr_lang=ocr_lang,
    )

    pl_result, pl_stage = _extract_statement_with_stages(
        pdf_path,
        statement_type="profit_or_loss",
        start_patterns=["statement of profit or loss", "laporan laba rugi"],
        stop_patterns=[
            "statement of cash flows",
            "laporan arus kas",
            "catatan atas laporan keuangan",
            "notes to the financial statements",
        ],
        use_ocr=use_ocr,
        ocr_lang=ocr_lang,
    )

    return {
        "source_pdf": str(pdf_path),
        "meta": {"use_ocr": use_ocr, "ocr_lang": ocr_lang},
        "statements": [fp_result.to_dict(), pl_result.to_dict()],
        "stages": {
            "financial_position": fp_stage,
            "profit_or_loss": pl_stage,
        },
    }
