"""Evaluate one extraction approach against the ground-truth JSON.

Usage:
  python ground_truth/evaluate.py <pred.json> [--gold ground_truth/bbni_2025.json]
                                  [--out report.json]

Both pred.json and gold.json must follow the unified schema (see
idx_fin_parser/unified.py). Sample run:

  python main_extract.py docs/FinancialStatement-2025-Tahunan-BBNI.pdf -o pred_native.json
  python ground_truth/evaluate.py pred_native.json

Metrics computed per statement (financial_position, profit_or_loss, ...):

  Node detection (structural)
    - tp / fp / fn / precision / recall / F1
    - keyed by (section, normalised_label_path)
      e.g. ("assets", "/giro pada bank lain/giro pada bank lain pihak ketiga")

  Cell value
    - precision / recall / F1 on (path, year) cells with non-null values
    - MAE on cells present-and-non-null in both gold and pred

  Hierarchy
    - parent_path_accuracy: of all gold nodes whose label also appears in pred,
      what % have at least one occurrence under the same parent path?

  Label fidelity
    - average normalised Levenshtein similarity over matched (tp) labels —
      forgives paraphrasing (relevant for VLM)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterator


# ── Helpers ───────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation — for label matching."""
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + (ca != cb),
            )
        prev = curr
    return prev[-1]


def _lev_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    d = _levenshtein(a, b)
    return 1.0 - d / max(len(a), len(b))


def _harmonic(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


# ── Tree traversal ────────────────────────────────────────────────────────

def _walk(nodes: list[dict], parent_path: str = "") -> Iterator[tuple[str, str, dict]]:
    """DFS yield (path, parent_path, node). Path uses normalised labels."""
    for n in nodes:
        path = parent_path + "/" + _norm(n.get("label", ""))
        yield (path, parent_path, n)
        yield from _walk(n.get("children", []), path)


def _build_path_map(statement: dict) -> dict[tuple[str, str], dict]:
    """{(section, path): node}."""
    out: dict[tuple[str, str], dict] = {}
    for section, nodes in statement.get("sections", {}).items():
        for path, _ppath, node in _walk(nodes):
            out[(section, path)] = node
    return out


def _build_label_index(statement: dict) -> dict[str, list[tuple[str, str, dict]]]:
    """{normalised_label: [(section, parent_path, node), ...]}."""
    out: dict[str, list] = {}
    for section, nodes in statement.get("sections", {}).items():
        for path, ppath, node in _walk(nodes):
            key = _norm(node.get("label", ""))
            out.setdefault(key, []).append((section, ppath, node))
    return out


# ── Per-statement evaluation ──────────────────────────────────────────────

def _score_nodes(gold_keys: set, pred_keys: set) -> dict:
    tp = gold_keys & pred_keys
    fp = pred_keys - gold_keys
    fn = gold_keys - pred_keys
    p = len(tp) / len(pred_keys) if pred_keys else 0.0
    r = len(tp) / len(gold_keys) if gold_keys else 0.0
    return {
        "precision": p, "recall": r, "f1": _harmonic(p, r),
        "tp": len(tp), "fp": len(fp), "fn": len(fn),
        "gold": len(gold_keys), "pred": len(pred_keys),
    }


def _tally_cell(g, pv, acc: dict) -> None:
    """Update running tallies for a single (gold_value, pred_value) cell pair."""
    if g is not None:
        acc["gold"] += 1
    if pv is not None:
        acc["pred"] += 1
    if g is not None and pv == g:
        acc["correct"] += 1
    if g is not None and pv is not None:
        acc["err_sum"] += abs(g - pv)
        acc["err_n"] += 1


def _score_cells(gold_map: dict, pred_map: dict, tp_keys: set) -> dict:
    acc = {"gold": 0, "pred": 0, "correct": 0, "err_sum": 0.0, "err_n": 0}
    for k in tp_keys:
        g_vals = gold_map[k].get("values", {})
        p_vals = pred_map[k].get("values", {})
        for yk in set(g_vals) | set(p_vals):
            _tally_cell(g_vals.get(yk), p_vals.get(yk), acc)
    p = acc["correct"] / acc["pred"] if acc["pred"] else 0.0
    r = acc["correct"] / acc["gold"] if acc["gold"] else 0.0
    return {
        "precision": p, "recall": r, "f1": _harmonic(p, r),
        "correct": acc["correct"], "gold": acc["gold"], "pred": acc["pred"],
        "mae": acc["err_sum"] / acc["err_n"] if acc["err_n"] else None,
    }


def _score_hierarchy(gold_stmt: dict, pred_stmt: dict) -> float:
    gold_lab = _build_label_index(gold_stmt)
    pred_lab = _build_label_index(pred_stmt)
    total = correct = 0
    for label, gold_entries in gold_lab.items():
        if label not in pred_lab:
            continue
        pred_parents = {p[1] for p in pred_lab[label]}
        for _sec, ppath, _node in gold_entries:
            total += 1
            if ppath in pred_parents:
                correct += 1
    return correct / total if total else 0.0


def _avg_label_similarity(gold_map: dict, pred_map: dict, tp_keys: set) -> float:
    if not tp_keys:
        return 0.0
    sims = [
        _lev_ratio(gold_map[k].get("label", ""), pred_map[k].get("label", ""))
        for k in tp_keys
    ]
    return sum(sims) / len(sims)


def evaluate_statement(gold_stmt: dict, pred_stmt: dict) -> dict:
    gold_map = _build_path_map(gold_stmt)
    pred_map = _build_path_map(pred_stmt)
    gold_keys, pred_keys = set(gold_map), set(pred_map)
    tp = gold_keys & pred_keys
    return {
        "node": _score_nodes(gold_keys, pred_keys),
        "cell": _score_cells(gold_map, pred_map, tp),
        "label_similarity_avg_on_tp": _avg_label_similarity(gold_map, pred_map, tp),
        "hierarchy_parent_path_acc": _score_hierarchy(gold_stmt, pred_stmt),
    }


def evaluate(gold: dict, pred: dict) -> dict:
    gold_stmts = {s["type"]: s for s in gold.get("statements", [])}
    pred_stmts = {s["type"]: s for s in pred.get("statements", [])}

    per_stmt: dict[str, Any] = {}
    for stype, gstmt in gold_stmts.items():
        pstmt = pred_stmts.get(stype)
        if pstmt is None:
            per_stmt[stype] = {"error": "missing in prediction"}
            continue
        per_stmt[stype] = evaluate_statement(gstmt, pstmt)

    f1s = [r["node"]["f1"] for r in per_stmt.values() if "node" in r]
    cf1s = [r["cell"]["f1"] for r in per_stmt.values() if "cell" in r]
    h_accs = [r["hierarchy_parent_path_acc"] for r in per_stmt.values() if "hierarchy_parent_path_acc" in r]

    return {
        "approach": pred.get("approach", "unknown"),
        "gold_source": gold.get("source_pdf"),
        "pred_source": pred.get("source_pdf"),
        "overall": {
            "macro_node_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "macro_cell_f1": sum(cf1s) / len(cf1s) if cf1s else 0.0,
            "macro_hierarchy_acc": sum(h_accs) / len(h_accs) if h_accs else 0.0,
        },
        "per_statement": per_stmt,
    }


# ── Reporting ─────────────────────────────────────────────────────────────

def print_report(report: dict) -> None:
    print(f"\n=== approach: {report['approach']}  vs  {report['gold_source']} ===")
    for stype, r in report["per_statement"].items():
        if "error" in r:
            print(f"\n[{stype}]  {r['error']}")
            continue
        n, c = r["node"], r["cell"]
        mae = c["mae"]
        print(f"\n[{stype}]")
        print(f"  Node          P={n['precision']:.3f}  R={n['recall']:.3f}  F1={n['f1']:.3f}"
              f"   tp={n['tp']}  fp={n['fp']}  fn={n['fn']}")
        print(f"  Cell value    P={c['precision']:.3f}  R={c['recall']:.3f}  F1={c['f1']:.3f}"
              f"   correct={c['correct']}/{c['gold']}")
        print(f"  Cell MAE      {mae:,.1f}" if mae is not None else "  Cell MAE      -")
        print(f"  Hierarchy acc {r['hierarchy_parent_path_acc']:.3f}")
        print(f"  Label sim     {r['label_similarity_avg_on_tp']:.3f}")
    o = report["overall"]
    print(f"\n--- overall macro: Node F1={o['macro_node_f1']:.3f}   "
          f"Cell F1={o['macro_cell_f1']:.3f}   "
          f"Hierarchy={o['macro_hierarchy_acc']:.3f} ---")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", help="Path to prediction JSON (unified schema)")
    default_gold = str(Path(__file__).parent / "bbni_2025.json")
    ap.add_argument("--gold", default=default_gold)
    ap.add_argument("--out", default=None, help="Optional path to write JSON report")
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    report = evaluate(gold, pred)
    print_report(report)
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nReport saved: {args.out}")


if __name__ == "__main__":
    main()
