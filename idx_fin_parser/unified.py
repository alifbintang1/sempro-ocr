"""Unified JSON schema for IDX financial-statement extraction.

All approaches (native pdfplumber, OCR fallback, OCR-HTML converter, VLM)
conform to this output:

{
  "schema_version": "1.0",
  "source_pdf":     str,
  "approach":       "native_pdf" | "ocr_fallback" | "ocr_html_table" | "vlm",
  "meta":           {...},
  "statements": [
    {
      "type":     str,                  # e.g. "financial_position"
      "years":    [int, int],
      "pages":    [int, ...],
      "sections": { "<name>": [Node, ...] }
    }
  ]
}

Node = {
  "level":     int,
  "label":     str,                     # Indonesian label
  "label_en":  str,                     # English label
  "row_type":  "group" | "item" | "label_only",
  "values":    { "YYYY-12-31": int|null, ... },
  "children":  [Node, ...]
}
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"

APPROACH_NATIVE = "native_pdf"
APPROACH_OCR_FALLBACK = "ocr_fallback"
APPROACH_OCR_HTML = "ocr_html_table"
APPROACH_VLM = "vlm"


# ── Node / statement builders ─────────────────────────────────────────────

def make_node(
    label: str,
    label_en: str = "",
    values: Optional[Dict[str, Optional[int]]] = None,
    level: int = 0,
    children: Optional[List[Dict[str, Any]]] = None,
    row_type: Optional[str] = None,
) -> Dict[str, Any]:
    values = dict(values) if values else {}
    children = list(children) if children else []
    if row_type is None:
        if any(v is not None for v in values.values()):
            row_type = "item"
        elif children:
            row_type = "group"
        else:
            row_type = "label_only"
    return {
        "level": level,
        "label": label,
        "label_en": label_en,
        "row_type": row_type,
        "values": values,
        "children": children,
    }


def build_statement(
    statement_type: str,
    years: List[int],
    pages: List[int],
    sections: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    return {
        "type": statement_type,
        "years": years,
        "pages": pages,
        "sections": sections,
    }


def build_unified_output(
    source_pdf: str,
    approach: str,
    statements: List[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_pdf": source_pdf,
        "approach": approach,
        "meta": meta or {},
        "statements": statements,
    }


# ── Adapters ──────────────────────────────────────────────────────────────

def itemnode_to_unified(node) -> Dict[str, Any]:
    """Convert idx_fin_parser.utils.ItemNode → unified node dict."""
    children = [itemnode_to_unified(c) for c in (node.children or [])]
    return make_node(
        label=node.label,
        label_en=getattr(node, "label_right", "") or "",
        values=dict(node.values) if node.values else {},
        level=getattr(node, "level", 0),
        children=children,
    )


def parse_value(v: Any) -> Optional[int]:
    """Parse a numeric cell; handles '(1,234)', '1.234', '1,234', None, ''."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip()
    if not s or s.lower() in ("null", "none", "-", "—", "nan"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("() ").replace(",", "").replace(".", "").replace(" ", "")
    if not s:
        return None
    try:
        n = int(s)
    except ValueError:
        return None
    return -n if neg else n


def flat_rows_to_tree(
    rows: List[Dict[str, Any]],
    years: List[int],
    level_key: str = "level",
    label_key: str = "account",
    label_en_key: Optional[str] = "account_en",
) -> List[Dict[str, Any]]:
    """Convert VLM-style flat rows with a `level` field into nested unified nodes.

    Each row may carry per-year values keyed by stringified year ("2025") or by
    canonical key ("2025-12-31"). Order must follow document/DFS order.
    """
    year_keys = [f"{y}-12-31" for y in years]
    year_aliases = [str(y) for y in years]

    roots: List[Dict[str, Any]] = []
    stack: List[Dict[str, Any]] = []

    for row in rows:
        try:
            level = int(row.get(level_key, 0) or 0)
        except (TypeError, ValueError):
            level = 0
        label = (row.get(label_key) or "").strip()
        label_en = (row.get(label_en_key) or "").strip() if label_en_key else ""
        values: Dict[str, Optional[int]] = {}
        for yk, alias in zip(year_keys, year_aliases):
            raw = row.get(alias)
            if raw is None:
                raw = row.get(yk)
            if raw is None:
                # Lenient fallback: any row key containing the year string.
                # Handles VLM outputs like "31 December 2025", "Per 2025", etc.
                for k, v in row.items():
                    if alias in str(k) and k not in (level_key, label_key, label_en_key):
                        raw = v
                        break
            values[yk] = parse_value(raw)
        row_type = row.get("row_type")
        node = make_node(
            label=label,
            label_en=label_en,
            values=values,
            level=level,
            row_type=row_type,
        )
        while stack and stack[-1]["level"] >= level:
            stack.pop()
        if stack:
            stack[-1]["children"].append(node)
        else:
            roots.append(node)
        stack.append(node)
    return roots


def split_into_sections(nodes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group top-level nodes into sections using looks_like_section_header().

    A node whose label matches a known section keyword AND has no values
    becomes a section bucket. Three input styles supported:

      1. native_pdf:  label-only header rows interleaved with sibling content rows
      2. VLM-flat:    section-header group with its rows nested as children
      3. VLM-wrapped: an outer wrapper (e.g. "Laporan posisi keuangan") nests
                      the section headers one level deeper

    For style 3, we unwrap one level if no top-level node is itself a section
    header and at least one of their children is.
    """
    from .utils import looks_like_section_header

    # Unwrap any top-level wrapper that itself is not a section header but
    # whose direct children include section headers. Handles VLM outputs
    # where pages nest content under a "Laporan posisi keuangan" wrapper,
    # while OTHER top-level nodes may also exist at the same level.
    effective: List[Dict[str, Any]] = []
    for n in nodes:
        has_value = any(v is not None for v in (n.get("values") or {}).values())
        is_section = looks_like_section_header(n.get("label", "")) is not None
        child_has_section = any(
            looks_like_section_header(c.get("label", ""))
            for c in n.get("children", [])
        )
        if not has_value and not is_section and child_has_section:
            effective.extend(n.get("children", []))
        else:
            effective.append(n)

    sections: Dict[str, List[Dict[str, Any]]] = {"unknown": []}
    current = "unknown"

    for node in effective:
        sec = looks_like_section_header(node.get("label", ""))
        has_value = any(v is not None for v in (node.get("values") or {}).values())
        if sec and not has_value:
            current = sec
            sections.setdefault(current, [])
            sections[current].extend(node.get("children", []))
            continue
        sections.setdefault(current, []).append(node)

    if len(sections) > 1 and not sections.get("unknown"):
        sections.pop("unknown", None)
    return sections


def years_from_columns(columns: List[Any]) -> List[int]:
    """Extract two-year list from column headers like ['account', '2025', '2024']."""
    years: List[int] = []
    for c in columns:
        for m in re.finditer(r"\b(?:19|20)\d{2}\b", str(c)):
            y = int(m.group(0))
            if y not in years:
                years.append(y)
    return years[:2]


def detect_statement_type(title: str) -> str:
    """Map a free-text title to a canonical statement type."""
    t = (title or "").lower()
    if "posisi keuangan" in t or "financial position" in t:
        return "financial_position"
    if "laba rugi" in t or "profit or loss" in t or "comprehensive income" in t:
        return "profit_or_loss"
    if "perubahan ekuitas" in t or "changes in equity" in t:
        return "changes_in_equity"
    if "arus kas" in t or "cash flows" in t:
        return "cash_flows"
    return "unknown"
