"""Aggregate comparison results across multiple banks into one summary table.

Reads runs/<...>/comparison.json for BBNI, BBRI, BMRI and produces:
  - Per-bank OVERALL Node F1 / Cell F1 / Hierarchy per approach
  - Mean +/- std across banks per approach
  - Markdown + JSON output at runs/cross_bank_summary.{md,json}
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANKS = {
    "BBNI": "runs/FinancialStatement-2025-Tahunan-BBNI/comparison.json",
    "BBRI": "runs/FinancialStatement-2025-Tahunan-BBRI/comparison.json",
    "BMRI": "runs/FinancialStatement-2025-Tahunan-BMRI/comparison.json",
}

APPROACH_ORDER = [
    "native_pdf", "pymupdf_native", "vlm_openai_gpt_4o",
    "vlm_openai_gpt_4o_mini", "ocr_full", "camelot_stream",
    "pdfplumber_tables", "baseline_regex",
]


def load_overall(path):
    data = json.loads((ROOT / path).read_text(encoding="utf-8"))
    out = {}
    for r in data["rows"]:
        if r["statement"] == "OVERALL (macro)":
            out[r["approach"]] = {
                "node_f1": r.get("node_f1"),
                "cell_f1": r.get("cell_f1"),
                "hierarchy_acc": r.get("hierarchy_acc"),
                "runtime_s": r.get("runtime_s"),
            }
    return out


def main():
    per_bank = {b: load_overall(p) for b, p in BANKS.items()}
    approaches = [a for a in APPROACH_ORDER
                  if any(a in per_bank[b] for b in BANKS)]

    rows = []
    for a in approaches:
        node_vals = [per_bank[b][a]["node_f1"] for b in BANKS
                     if a in per_bank[b] and per_bank[b][a]["node_f1"] is not None]
        cell_vals = [per_bank[b][a]["cell_f1"] for b in BANKS
                     if a in per_bank[b] and per_bank[b][a]["cell_f1"] is not None]
        hier_vals = [per_bank[b][a]["hierarchy_acc"] for b in BANKS
                     if a in per_bank[b] and per_bank[b][a]["hierarchy_acc"] is not None]
        row = {"approach": a}
        for b in BANKS:
            row[f"{b}_node"] = per_bank[b].get(a, {}).get("node_f1")
        row["node_mean"] = round(statistics.mean(node_vals), 3) if node_vals else None
        row["node_std"] = round(statistics.pstdev(node_vals), 3) if len(node_vals) > 1 else 0.0
        row["cell_mean"] = round(statistics.mean(cell_vals), 3) if cell_vals else None
        row["hier_mean"] = round(statistics.mean(hier_vals), 3) if hier_vals else None
        rows.append(row)

    # Markdown
    lines = ["# Cross-Bank Comparison Summary", "",
             "Dataset: 3 bank IDX (BBNI, BBRI, BMRI) 2025, halaman 4-19.",
             "Metrik = OVERALL macro (rata-rata financial_position + profit_or_loss).",
             ""]
    hdr = ["approach", "BBNI", "BBRI", "BMRI", "Node F1 mean", "±std",
           "Cell F1 mean", "Hierarchy mean"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        def f(x):
            return "—" if x is None else f"{x:.3f}"
        lines.append("| " + " | ".join([
            r["approach"], f(r["BBNI_node"]), f(r["BBRI_node"]), f(r["BMRI_node"]),
            f(r["node_mean"]), f(r["node_std"]), f(r["cell_mean"]), f(r["hier_mean"]),
        ]) + " |")

    md = "\n".join(lines)
    (ROOT / "runs" / "cross_bank_summary.md").write_text(md, encoding="utf-8")
    (ROOT / "runs" / "cross_bank_summary.json").write_text(
        json.dumps({"per_bank": per_bank, "summary_rows": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(md)
    print(f"\nSaved: runs/cross_bank_summary.md + .json")


if __name__ == "__main__":
    main()
