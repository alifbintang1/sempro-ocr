"""Convert PaddleOCR table HTML output to unified-schema JSON.

Input  : ocr_table_output/json/page_XXX.json  (one PaddleOCR result per page)
Output : ocr_html_unified.json                (single unified-schema file)

Indentation inside the HTML cells is the only hierarchy cue available, so the
level field is a best-effort heuristic (leading spaces / &nbsp;).
"""
from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from idx_fin_parser.unified import (
    APPROACH_OCR_HTML,
    build_statement,
    build_unified_output,
    detect_statement_type,
    flat_rows_to_tree,
    parse_value,
    split_into_sections,
)


def _extract_years_from_df(df: pd.DataFrame) -> List[int]:
    years: List[int] = []
    candidates: List[str] = [str(c) for c in df.columns]
    if len(df) > 0:
        candidates.extend(df.iloc[0].astype(str).tolist())
    for c in candidates:
        for m in re.finditer(r"\b(?:19|20)\d{2}\b", c):
            y = int(m.group(0))
            if y not in years:
                years.append(y)
    return years[:2]


def _detect_level(label: str) -> int:
    s = label.replace("\xa0", " ").expandtabs(4)
    leading = len(s) - len(s.lstrip(" "))
    return min(leading // 2, 5)


def _classify_columns(df: pd.DataFrame) -> Tuple[int, List[int], int | None]:
    """Return (id_col_idx, amount_col_idxs, en_col_idx_or_None)."""
    ncols = len(df.columns)
    amount_cols: List[int] = []
    for ci in range(1, ncols):
        col_vals = df.iloc[:, ci].astype(str).tolist()
        numeric = sum(1 for v in col_vals if parse_value(v) is not None)
        if numeric >= max(1, len(col_vals) // 4):
            amount_cols.append(ci)
    en_col = ncols - 1 if (ncols - 1) not in amount_cols and ncols > 1 else None
    return 0, amount_cols, en_col


def convert_page(json_path: Path, page_num: int) -> dict:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = data[0]
    html = data["table_res_list"][0]["pred_html"]
    df = pd.read_html(StringIO(html))[0].fillna("")

    years = _extract_years_from_df(df)
    id_col, amount_cols, en_col = _classify_columns(df)

    rows: List[dict] = []
    for _, r in df.iterrows():
        id_label_raw = str(r.iloc[id_col])
        id_label = id_label_raw.strip()
        if not id_label:
            continue
        # Skip header row (contains the years)
        if any(str(y) in id_label for y in years) and not any(
            parse_value(r.iloc[c]) is not None for c in amount_cols
        ):
            continue
        en_label = str(r.iloc[en_col]).strip() if en_col is not None else ""
        amounts = [parse_value(r.iloc[c]) for c in amount_cols]
        row: dict = {
            "level": _detect_level(id_label_raw),
            "account": id_label,
            "account_en": en_label,
        }
        for y, amt in zip(years, amounts):
            row[str(y)] = amt
        rows.append(row)

    nodes = flat_rows_to_tree(rows, years, label_en_key="account_en")
    sections = split_into_sections(nodes)
    return build_statement(
        statement_type=detect_statement_type(""),
        years=years,
        pages=[page_num],
        sections=sections,
    )


def convert_dir(json_dir: str = "ocr_table_output/json",
                output_path: str = "ocr_html_unified.json") -> Path:
    base = Path(json_dir)
    statements = []
    for p in sorted(base.glob("page_*.json")):
        m = re.search(r"page_(\d+)", p.stem)
        page_num = int(m.group(1)) if m else 0
        try:
            statements.append(convert_page(p, page_num))
        except Exception as exc:
            print(f"[skip] {p.name}: {exc}")

    output = build_unified_output(
        source_pdf=str(base.parent.name),
        approach=APPROACH_OCR_HTML,
        statements=statements,
        meta={"source_dir": str(base)},
    )
    out = Path(output_path)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    path = convert_dir()
    print(f"Wrote: {path.resolve()}")
