"""Render unified-schema JSON as a PSAK-style financial statement table (HTML).

Mimics the IAI PSAK 201 textbook layout:
- Centered title block with currency note
- Two-column value layout (current + prior period)
- Hierarchical indentation by level
- Section headers in UPPERCASE bold
- Sub-group headers in bold
- "Total / Jumlah" rows bold with horizontal line above
- Indonesian number format (123.456 / negatives in parentheses)

Usage:
  python tools/render_psak.py <unified.json> [--out report.html]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ── Mapping section-internal keys to PSAK display labels ──────────────────
SECTION_LABELS = {
    "assets": "ASET",
    "liabilities": "LIABILITAS",
    "equity": "EKUITAS",
    "temporary_syirkah_funds": "DANA SYIRKAH TEMPORER",
    "operating": "PENDAPATAN DAN BEBAN OPERASIONAL",
    "non_operating": "PENDAPATAN DAN BEBAN BUKAN OPERASIONAL",
    "other_comprehensive_income": "PENDAPATAN KOMPREHENSIF LAIN",
    "earnings_per_share": "LABA (RUGI) PER SAHAM",
}

STATEMENT_TITLE = {
    "financial_position": "Laporan Posisi Keuangan",
    "profit_or_loss": "Laporan Laba Rugi dan Penghasilan Komprehensif Lain",
    "changes_in_equity": "Laporan Perubahan Ekuitas",
    "cash_flows": "Laporan Arus Kas",
}

CSS = """
  @page { size: A4; margin: 18mm; }
  body { font-family: 'Times New Roman', Georgia, serif; max-width: 780px;
         margin: 24px auto; color: #111; font-size: 11pt; }
  h2 { text-align: center; margin: 4px 0; font-size: 13pt; font-weight: bold; }
  .entity { text-align: center; font-weight: bold; font-size: 13pt;
            margin: 8px 0 2px 0; }
  .subtitle { text-align: center; font-style: italic; color: #444;
              margin-bottom: 18px; font-size: 10pt; }
  table { border-collapse: collapse; width: 100%; }
  thead th { text-align: right; border-bottom: 1px solid #222;
             font-weight: bold; padding: 6px 8px; font-size: 10pt; }
  thead th.first { text-align: left; }
  td { padding: 2px 8px; vertical-align: top; }
  td.label { text-align: left; }
  td.value { text-align: right; font-variant-numeric: tabular-nums;
             white-space: nowrap; padding-right: 12px; }
  .l0 { padding-left: 0; }
  .l1 { padding-left: 16px; }
  .l2 { padding-left: 32px; }
  .l3 { padding-left: 48px; }
  .l4 { padding-left: 64px; }
  .l5 { padding-left: 80px; }
  tr.section-header td { padding-top: 12px; font-weight: bold;
                         text-transform: uppercase; }
  tr.subgroup td { font-weight: bold; padding-top: 6px; }
  tr.total td { font-weight: bold; border-top: 1px solid #555;
                padding-top: 4px; padding-bottom: 4px; }
  tr.grand-total td { font-weight: bold; border-top: 1px solid #222;
                      border-bottom: 3px double #222; padding: 4px 8px; }
  .stmt { margin-bottom: 56px; page-break-after: always; }
  .footer-note { font-style: italic; text-align: right; color: #666;
                 font-size: 9pt; margin-top: 8px; }
"""

PAGE = """<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{header}
{stmts}
</body>
</html>"""


def _fmt(v) -> str:
    """Indonesian number format: 1.234.567 with parentheses for negatives."""
    if v is None or v == "":
        return ""
    try:
        n = int(v)
    except (TypeError, ValueError):
        return str(v)
    if n < 0:
        return f"({abs(n):,})".replace(",", ".")
    return f"{n:,}".replace(",", ".")


def _is_total_label(label: str) -> bool:
    l = (label or "").lower()
    return any(k in l for k in ["jumlah", "total ", "subtotal"]) or l.startswith("total")


def _is_grand_total(label: str) -> bool:
    """'Jumlah aset', 'Jumlah liabilitas, dana syirkah temporer dan ekuitas', etc."""
    l = (label or "").lower().strip()
    if l in ("jumlah aset", "total aset", "jumlah liabilitas", "total liabilitas",
            "jumlah ekuitas", "total ekuitas"):
        return True
    return l.startswith("jumlah liabilitas") or l.startswith("total liabilitas")


def _row_class(node: dict, depth: int) -> str:
    label = node.get("label", "")
    has_value = any(v is not None for v in (node.get("values") or {}).values())
    if _is_grand_total(label):
        return "grand-total"
    if _is_total_label(label):
        return "total"
    if not has_value and node.get("children") and depth > 0:
        return "subgroup"
    return ""


def _render_node(node: dict, year_keys: list[str], depth: int = 0) -> str:
    cls = _row_class(node, depth)
    label_cls = f"l{min(depth, 5)}"
    vals = "".join(
        f'<td class="value">{_fmt((node.get("values") or {}).get(y))}</td>'
        for y in year_keys
    )
    out = (f'<tr class="{cls}">'
           f'<td class="label {label_cls}">{node.get("label", "")}</td>'
           f'{vals}</tr>')
    for c in node.get("children", []):
        out += _render_node(c, year_keys, depth + 1)
    return out


def _render_statement(stmt: dict) -> str:
    yrs = stmt.get("years", [])
    if not yrs:
        return ""
    year_keys = [f"{y}-12-31" for y in yrs]

    stmt_type = stmt.get("type", "")
    title = STATEMENT_TITLE.get(stmt_type, stmt_type.replace("_", " ").title())
    yr_str = " dan ".join(f"31 Desember {y}" for y in yrs)
    header_cells = "".join(f"<th>{y}</th>" for y in yrs)

    body = ""
    for sec_name, nodes in stmt.get("sections", {}).items():
        sec_label = SECTION_LABELS.get(sec_name, sec_name.replace("_", " ").upper())
        body += (f'<tr class="section-header">'
                 f'<td class="label l0">{sec_label}</td>'
                 f'{"<td></td>" * len(yrs)}</tr>')
        for n in nodes:
            body += _render_node(n, year_keys, depth=1)

    return f"""
<div class="stmt">
  <h2>{title}</h2>
  <div class="subtitle">per {yr_str}<br/>(dalam jutaan rupiah)</div>
  <table>
    <colgroup>
      <col style="width: 60%"/>
      {"".join('<col style="width: 20%"/>' for _ in yrs)}
    </colgroup>
    <thead>
      <tr><th class="first"></th>{header_cells}</tr>
    </thead>
    <tbody>{body}</tbody>
  </table>
</div>"""


def render(data: dict, entity_name: str = "") -> str:
    src = data.get("source_pdf", "")
    approach = data.get("approach", "")
    title = f"{src} ({approach})"

    entity = (entity_name or data.get("meta", {}).get("issuer_full")
              or "KELOMPOK USAHA XYZ")
    header = f'<div class="entity">{entity}</div>'
    if approach and approach != "ground_truth":
        header += (f'<div class="footer-note">Sumber: ekstraksi otomatis '
                   f'({approach}) dari <code>{src}</code></div>')
    elif approach == "ground_truth":
        header += '<div class="footer-note">Ground truth (manual)</div>'

    stmts = "".join(_render_statement(s) for s in data.get("statements", []))
    return PAGE.format(title=title, css=CSS, header=header, stmts=stmts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("unified_json", help="Path to unified-schema JSON")
    ap.add_argument("--out", default=None, help="Output HTML path")
    ap.add_argument("--entity", default="", help="Entity name override")
    args = ap.parse_args()

    data = json.loads(Path(args.unified_json).read_text(encoding="utf-8"))
    html = render(data, entity_name=args.entity)
    out = Path(args.out) if args.out else Path(args.unified_json).with_suffix(".html")
    out.write_text(html, encoding="utf-8")
    print(f"Wrote: {out.resolve()}")


if __name__ == "__main__":
    main()
