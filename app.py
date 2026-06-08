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

from flask import Flask, jsonify, redirect, render_template, request, url_for

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ground_truth"))
sys.path.insert(0, str(ROOT / "tools"))

from approaches import REGISTRY  # auto-loads .env (OPENAI_API_KEY)
from render_psak import render as render_psak_html  # type: ignore
from evaluate import evaluate  # type: ignore

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

GT_PATH = ROOT / "ground_truth" / "bbni_2025.json"
DEMO_PDF = ROOT / "docs" / "FinancialStatement-2025-Tahunan-BBNI.pdf"


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

@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        approaches=_approach_availability(),
        demo_available=DEMO_PDF.exists(),
    )


@app.route("/run", methods=["POST"])
def run():
    use_demo = request.form.get("demo") == "true"
    if use_demo:
        if not DEMO_PDF.exists():
            return render_template(
                "index.html",
                approaches=_approach_availability(),
                error="Demo PDF tidak ditemukan.",
            )
        pdf_path = str(DEMO_PDF)
        original_filename = DEMO_PDF.name
        tmp_path = None
        is_bbni = True
    else:
        if "pdf" not in request.files:
            return redirect(url_for("index"))
        file = request.files["pdf"]
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            return render_template(
                "index.html",
                approaches=_approach_availability(),
                error="Hanya file PDF yang diterima.",
            )
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)
        pdf_path = tmp_path
        original_filename = file.filename
        is_bbni = "BBNI" in (file.filename or "").upper()

    selected = request.form.getlist("approach")
    if not selected:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return render_template(
            "index.html",
            approaches=_approach_availability(),
            error="Pilih minimal satu approach.",
        )

    # Run each approach
    results = []
    for name in selected:
        if name not in REGISTRY:
            continue
        results.append(_run_one(name, pdf_path))

    # Cleanup temp upload
    if tmp_path:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Evaluate against GT if available + applicable
    has_gt = is_bbni and GT_PATH.exists()
    if has_gt:
        gold = json.loads(GT_PATH.read_text(encoding="utf-8"))
        for r in results:
            if r.get("error") or not r.get("raw_pred"):
                continue
            r["metrics"] = _eval_one(r["raw_pred"], gold)

    # Drop heavy raw_pred from template context (already serialized in pred_json)
    for r in results:
        r.pop("raw_pred", None)

    return render_template(
        "result.html",
        filename=original_filename,
        results=results,
        has_gt=has_gt,
    )


@app.route("/api/run", methods=["POST"])
def api_run():
    """JSON API endpoint (for headless usage)."""
    data = request.get_json() or {}
    pdf_path = data.get("pdf_path", str(DEMO_PDF))
    approaches = data.get("approaches", ["native_pdf"])
    if not Path(pdf_path).exists():
        return jsonify({"error": f"PDF not found: {pdf_path}"}), 400
    results = []
    for name in approaches:
        if name not in REGISTRY:
            continue
        r = _run_one(name, pdf_path)
        r.pop("psak_html", None)  # strip heavy HTML from API response
        results.append(r)
    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
