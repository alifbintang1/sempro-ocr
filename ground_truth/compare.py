"""Run evaluate.py against several prediction JSONs and print a comparison table.

Usage:
  python ground_truth/compare.py pred_native.json pred_ocr.json pred_vlm.json \
         [--gold ground_truth/bbni_2025.json] [--out compare.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate import evaluate  # type: ignore


def _row(approach: str, stype: str, r: dict) -> dict:
    if "error" in r:
        return {"approach": approach, "statement": stype, "error": r["error"]}
    n, c = r["node"], r["cell"]
    return {
        "approach": approach,
        "statement": stype,
        "node_f1": n["f1"],
        "node_p": n["precision"],
        "node_r": n["recall"],
        "cell_f1": c["f1"],
        "cell_correct": c["correct"],
        "cell_gold": c["gold"],
        "cell_mae": c["mae"] if c["mae"] is not None else float("nan"),
        "hierarchy_acc": r["hierarchy_parent_path_acc"],
        "label_sim": r["label_similarity_avg_on_tp"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", help="One or more prediction JSON files")
    default_gold = str(Path(__file__).parent / "bbni_2025.json")
    ap.add_argument("--gold", default=default_gold)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))

    all_reports = []
    rows: list[dict] = []
    for pred_path in args.preds:
        pred = json.loads(Path(pred_path).read_text(encoding="utf-8"))
        report = evaluate(gold, pred)
        all_reports.append({"file": pred_path, "report": report})
        approach = report["approach"]
        for stype, r in report["per_statement"].items():
            rows.append(_row(approach, stype, r))
        rows.append({
            "approach": approach,
            "statement": "OVERALL (macro)",
            "node_f1": report["overall"]["macro_node_f1"],
            "cell_f1": report["overall"]["macro_cell_f1"],
            "hierarchy_acc": report["overall"]["macro_hierarchy_acc"],
        })

    # Pretty table
    header = ["approach", "statement", "node_f1", "node_p", "node_r",
              "cell_f1", "cell_correct", "cell_mae", "hierarchy_acc", "label_sim"]
    widths = {h: max(len(h), max((len(_fmt(r.get(h))) for r in rows), default=0)) for h in header}

    sep = " | "
    print(sep.join(h.ljust(widths[h]) for h in header))
    print("-+-".join("-" * widths[h] for h in header))
    for r in rows:
        print(sep.join(_fmt(r.get(h)).ljust(widths[h]) for h in header))

    if args.out:
        Path(args.out).write_text(
            json.dumps({"rows": rows, "reports": all_reports},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved: {args.out}")


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if v != v:  # nan
            return "-"
        return f"{v:.3f}"
    return str(v)


if __name__ == "__main__":
    main()
