"""Orchestrate all registered extraction approaches and produce a unified
comparison report (accuracy metrics + runtime).

Each approach lives in ``approaches/<name>.py`` and calls
``register(name, fn)``; dropping a new file there makes it appear in the
comparison automatically.

Usage:
  python ground_truth/run_all.py <pdf_path> \
         [--gold ground_truth/bbni_2025.json] \
         [--out-dir runs/<run_name>] \
         [--only native_pdf,ocr_full,...] \
         [--skip native_pdf,...] \
         [--vlm-json pred_vlm.json]   # if VLM was run on Colab
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from approaches import REGISTRY  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate  # type: ignore  # noqa: E402


def _row(approach: str, statement: str, r: dict, runtime: float | None) -> dict:
    if "error" in r:
        return {"approach": approach, "statement": statement, "error": r["error"]}
    n, c = r["node"], r["cell"]
    return {
        "approach": approach,
        "statement": statement,
        "node_f1": round(n["f1"], 3),
        "node_p": round(n["precision"], 3),
        "node_r": round(n["recall"], 3),
        "cell_f1": round(c["f1"], 3),
        "cell_correct": c["correct"],
        "cell_gold": c["gold"],
        "cell_mae": round(c["mae"], 2) if c["mae"] is not None else None,
        "hierarchy_acc": round(r["hierarchy_parent_path_acc"], 3),
        "label_sim": round(r["label_similarity_avg_on_tp"], 3),
        "runtime_s": round(runtime, 2) if runtime is not None else None,
    }


def _overall_row(approach: str, report: dict, runtime: float | None) -> dict:
    o = report["overall"]
    return {
        "approach": approach,
        "statement": "OVERALL (macro)",
        "node_f1": round(o["macro_node_f1"], 3),
        "cell_f1": round(o["macro_cell_f1"], 3),
        "hierarchy_acc": round(o["macro_hierarchy_acc"], 3),
        "runtime_s": round(runtime, 2) if runtime is not None else None,
    }


def _markdown_table(rows: list[dict]) -> str:
    headers = ["approach", "statement", "node_f1", "node_p", "node_r",
               "cell_f1", "cell_correct", "cell_mae", "hierarchy_acc",
               "label_sim", "runtime_s"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        vals = []
        for h in headers:
            v = r.get(h)
            vals.append("—" if v is None else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _load_pred(path: str) -> tuple[dict, float | None] | None:
    p = Path(path)
    if not p.exists():
        print(f"[warn] pred JSON not found at {p}; skipping.")
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    runtime = data.get("meta", {}).get("runtime_seconds")
    return data, runtime


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="Path to PDF file")
    ap.add_argument("--gold", default=str(ROOT / "ground_truth" / "bbni_2025.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--only", default=None, help="Comma-separated approach names")
    ap.add_argument("--skip", default=None, help="Comma-separated approach names")
    ap.add_argument("--vlm-json", default=None,
                    help="Pre-computed prediction JSON(s). Comma-separated paths "
                         "for multiple. Useful for VLM/API results computed "
                         "earlier or in a different env.")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "runs" / pdf_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    only = set(args.only.split(",")) if args.only else None
    skip = set(args.skip.split(",")) if args.skip else set()

    approaches = [(n, f) for n, f in REGISTRY.items() if n not in skip
                  and (only is None or n in only)]

    print(f"Approaches to run: {[n for n, _ in approaches]}")
    print(f"GT: {args.gold}")
    print(f"Out: {out_dir}\n")

    results: list[tuple[str, dict, float | None]] = []
    for idx, (name, fn) in enumerate(approaches, 1):
        print(f"[{idx}/{len(approaches)}] {name}  ...", flush=True)
        try:
            t0 = time.perf_counter()
            pred, approach_runtime = fn(str(pdf_path))
            wall = time.perf_counter() - t0
            runtime = approach_runtime if approach_runtime is not None else wall
        except Exception:
            print(f"      FAILED:")
            traceback.print_exc()
            continue
        (out_dir / f"pred_{name}.json").write_text(
            json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"      done in {runtime:.2f}s")
        results.append((name, pred, runtime))

    # External pre-computed predictions (e.g. VLM run earlier or in Colab)
    if args.vlm_json:
        for path in args.vlm_json.split(","):
            path = path.strip()
            if not path:
                continue
            loaded = _load_pred(path)
            if not loaded:
                continue
            pred, runtime = loaded
            approach_name = pred.get("approach", Path(path).stem)
            (out_dir / f"pred_{approach_name}.json").write_text(
                json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[extra] {approach_name}  loaded from {path}"
                  + (f" (runtime: {runtime:.2f}s)" if runtime else ""))
            results.append((approach_name, pred, runtime))

    # Evaluate
    rows: list[dict] = []
    all_reports: list[dict] = []
    for approach, pred, runtime in results:
        report = evaluate(gold, pred)
        all_reports.append({"approach": approach, "runtime_s": runtime, **report})
        (out_dir / f"report_{approach}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        for stype, r in report["per_statement"].items():
            rows.append(_row(approach, stype, r, runtime))
        rows.append(_overall_row(approach, report, runtime))

    combined = {
        "gold": gold.get("source_pdf"),
        "pdf": str(pdf_path),
        "rows": rows,
        "reports": all_reports,
    }
    (out_dir / "comparison.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    md = (f"# Comparison: {pdf_path.name}\n\n"
          f"Ground truth: `{gold.get('source_pdf')}`  ·  "
          f"Approaches evaluated: {len(results)}\n\n"
          + _markdown_table(rows))
    (out_dir / "comparison.md").write_text(md, encoding="utf-8")

    print(f"\n=== Comparison saved to {out_dir} ===")
    print(_markdown_table(rows))


if __name__ == "__main__":
    main()
