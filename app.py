#!/usr/bin/env python3
"""Prototype web UI untuk sistem ekstraksi laporan keuangan IDX.

Mendemonstrasikan komparasi multi-approach pada satu PDF:
- Upload PDF atau pilih demo BBNI
- Pilih beberapa approach untuk dijalankan
- Hasil ditampilkan side-by-side dengan tabel metrik (jika GT tersedia),
  view PSAK-style, dan raw JSON
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ground_truth"))
sys.path.insert(0, str(ROOT / "tools"))

from approaches import REGISTRY  # auto-loads .env (OPENAI_API_KEY)
from render_psak import render as render_psak_html  # type: ignore
from evaluate import evaluate  # type: ignore

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload

# Demo documents available (each has a manually-transcribed ground truth)
DOCUMENTS = {
    "BBNI": {
        "issuer_full": "PT Bank Negara Indonesia (Persero) Tbk",
        "pdf": ROOT / "docs" / "FinancialStatement-2025-Tahunan-BBNI.pdf",
        "gt": ROOT / "ground_truth" / "bbni_2025.json",
    },
    "BBRI": {
        "issuer_full": "PT Bank Rakyat Indonesia (Persero) Tbk",
        "pdf": ROOT / "docs" / "FinancialStatement-2025-Tahunan-BBRI.pdf",
        "gt": ROOT / "ground_truth" / "bbri_2025.json",
    },
    "BMRI": {
        "issuer_full": "PT Bank Mandiri (Persero) Tbk",
        "pdf": ROOT / "docs" / "FinancialStatement-2025-Tahunan-BMRI.pdf",
        "gt": ROOT / "ground_truth" / "bmri_2025.json",
    },
}


APPROACH_META = {
    "native_pdf": {
        "label": "Native PDF (kustom)",
        "desc": "Custom column-coordinate parser via pdfplumber",
        "runtime": "~4 detik",
        "cost": "Gratis",
        "category": "rule-based",
    },
    "pymupdf_native": {
        "label": "PyMuPDF Native",
        "desc": "Algoritma sama, backend PyMuPDF (37× lebih cepat)",
        "runtime": "~0,1 detik",
        "cost": "Gratis",
        "category": "rule-based",
    },
    "baseline_regex": {
        "label": "Baseline Regex",
        "desc": "Naive text-layer + regex (floor baseline)",
        "runtime": "~4 detik",
        "cost": "Gratis",
        "category": "rule-based",
    },
    "pdfplumber_tables": {
        "label": "pdfplumber Tables",
        "desc": "Built-in extract_tables() dengan strategy text",
        "runtime": "~5 detik",
        "cost": "Gratis",
        "category": "rule-based",
    },
    "camelot_stream": {
        "label": "Camelot (Stream)",
        "desc": "Third-party table extractor",
        "runtime": "~5 detik",
        "cost": "Gratis",
        "category": "rule-based",
    },
    "ocr_full": {
        "label": "OCR Penuh (Tesseract)",
        "desc": "Force OCR setiap halaman, parse text-line",
        "runtime": "~60 detik",
        "cost": "Gratis",
        "category": "ocr",
    },
    "vlm_openai_gpt_4o_mini": {
        "label": "VLM: GPT-4o mini",
        "desc": "OpenAI Vision API, model ringan",
        "runtime": "~5 menit",
        "cost": "~$0,03",
        "category": "vlm",
    },
    "vlm_openai_gpt_4o": {
        "label": "VLM: GPT-4o",
        "desc": "OpenAI Vision API, model penuh",
        "runtime": "~2 menit",
        "cost": "~$0,50",
        "category": "vlm",
    },
}


def _approach_availability() -> dict:
    """Return mapping approach_name → metadata + availability info."""
    out: dict[str, dict] = {}
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    for name, meta in APPROACH_META.items():
        registered = name in REGISTRY
        # VLM approaches require both registration AND API key
        if name.startswith("vlm_openai"):
            available = registered and has_openai
            disabled_reason = "" if available else "OPENAI_API_KEY belum di-set"
        else:
            available = registered
            disabled_reason = "" if available else "Pustaka belum terpasang"
        out[name] = {**meta, "available": available, "disabled_reason": disabled_reason}
    return out


def _run_one(name: str, pdf_path: str) -> dict[str, Any]:
    """Run one approach; return dict with prediction, runtime, PSAK HTML, metrics."""
    try:
        t0 = time.perf_counter()
        pred, approach_runtime = REGISTRY[name](pdf_path)
        wall = time.perf_counter() - t0
        runtime = approach_runtime if approach_runtime else wall
    except Exception as exc:
        return {
            "name": name,
            "label": APPROACH_META.get(name, {}).get("label", name),
            "error": str(exc),
        }

    # Render PSAK style (fragment HTML for iframe srcdoc)
    psak_html = render_psak_html(pred, entity_name="")

    return {
        "name": name,
        "label": APPROACH_META.get(name, {}).get("label", name),
        "runtime": round(runtime, 2),
        "psak_html": psak_html,
        "pred_json": json.dumps(pred, ensure_ascii=False, indent=2),
        "raw_pred": pred,  # for evaluation
        "error": None,
    }


def _gt_stats(gold: dict) -> dict:
    """Count total nodes + nodes-with-value in a ground truth, for the caption."""
    total = filled = 0

    def walk(nodes):
        nonlocal total, filled
        for n in nodes:
            total += 1
            if any(v is not None for v in (n.get("values") or {}).values()):
                filled += 1
            walk(n.get("children", []))

    for s in gold.get("statements", []):
        for nodes in s.get("sections", {}).values():
            walk(nodes)
    return {"total": total, "filled": filled}


def _eval_one(pred: dict, gold: dict) -> dict:
    """Run evaluator + return condensed metric row."""
    report = evaluate(gold, pred)
    overall = report["overall"]
    return {
        "node_f1": round(overall["macro_node_f1"], 3),
        "cell_f1": round(overall["macro_cell_f1"], 3),
        "hierarchy": round(overall["macro_hierarchy_acc"], 3),
        "per_statement": [
            {
                "type": stype,
                "node_f1": round(r["node"]["f1"], 3),
                "cell_f1": round(r["cell"]["f1"], 3),
                "cell_correct": r["cell"]["correct"],
                "cell_gold": r["cell"]["gold"],
            }
            for stype, r in report["per_statement"].items()
            if "node" in r
        ],
    }


# ── Routes ────────────────────────────────────────────────────────────────

def _documents_for_template() -> dict:
    """Document metadata for the index page (only those whose PDF exists)."""
    out = {}
    for code, meta in DOCUMENTS.items():
        out[code] = {
            "issuer_full": meta["issuer_full"],
            "available": meta["pdf"].exists(),
            "has_gt": meta["gt"].exists(),
        }
    return out


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        approaches=_approach_availability(),
        documents=_documents_for_template(),
    )


def _index_error(msg: str):
    return render_template(
        "index.html",
        approaches=_approach_availability(),
        documents=_documents_for_template(),
        error=msg,
    )


@app.route("/run", methods=["POST"])
def run():
    selected = request.form.getlist("approach")
    if not selected:
        return _index_error("Pilih minimal satu approach.")

    # Mode A: upload PDF sendiri (tanpa ground truth → tanpa metrik)
    upload = request.files.get("pdf")
    if upload and upload.filename:
        if not upload.filename.lower().endswith(".pdf"):
            return _index_error("Hanya file PDF yang diterima.")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            upload.save(tmp_path)
        try:
            results = [_run_one(n, tmp_path) for n in selected if n in REGISTRY]
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        for r in results:
            r.pop("raw_pred", None)
        return render_template(
            "result.html",
            filename=upload.filename,
            results=results,
            has_gt=False,
        )

    # Mode B: dokumen demo (dengan ground truth → metrik dihitung)
    code = (request.form.get("document") or "").upper()
    doc = DOCUMENTS.get(code)
    if not doc or not doc["pdf"].exists():
        return _index_error("Pilih dokumen demo atau upload PDF.")

    results = [_run_one(n, str(doc["pdf"])) for n in selected if n in REGISTRY]
    has_gt = doc["gt"].exists()
    gt_stats = None
    if has_gt:
        gold = json.loads(doc["gt"].read_text(encoding="utf-8"))
        gt_stats = _gt_stats(gold)
        for r in results:
            if r.get("error") or not r.get("raw_pred"):
                continue
            r["metrics"] = _eval_one(r["raw_pred"], gold)
    for r in results:
        r.pop("raw_pred", None)

    return render_template(
        "result.html",
        filename=f"{code} — {doc['issuer_full']}",
        results=results,
        has_gt=has_gt,
        gt_label=f"{code} 2025",
        gt_stats=gt_stats,
    )


@app.route("/api/run", methods=["POST"])
def api_run():
    """JSON API endpoint (for headless usage)."""
    data = request.get_json() or {}
    code = (data.get("document") or "BBNI").upper()
    doc = DOCUMENTS.get(code)
    if not doc or not doc["pdf"].exists():
        return jsonify({"error": f"Unknown document: {code}"}), 400
    approaches = data.get("approaches", ["native_pdf"])
    results = []
    for name in approaches:
        if name not in REGISTRY:
            continue
        r = _run_one(name, str(doc["pdf"]))
        r.pop("psak_html", None)
        r.pop("raw_pred", None)
        results.append(r)
    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
