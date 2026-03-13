#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for

from idx_fin_parser.pdf_statements import (
    extract_statement_financial_position,
    extract_statement_profit_loss,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload


@app.template_filter("format_number")
def format_number(value):
    """Format integer with thousands separator, parentheses for negatives."""
    if value is None:
        return ""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return str(value)
    if v < 0:
        return f"({abs(v):,})".replace(",", ".")
    return f"{v:,}".replace(",", ".")


def _build_result(pdf_path: str, use_ocr: bool = False, ocr_lang: str = "ind+eng") -> dict:
    fp = extract_statement_financial_position(pdf_path, use_ocr=use_ocr, ocr_lang=ocr_lang)
    pl = extract_statement_profit_loss(pdf_path, use_ocr=use_ocr, ocr_lang=ocr_lang)
    return {
        "source_pdf": Path(pdf_path).name,
        "statements": [fp.to_dict(), pl.to_dict()],
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/extract", methods=["POST"])
def extract():
    if "pdf" not in request.files:
        return redirect(url_for("index"))

    file = request.files["pdf"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return render_template("index.html", error="Hanya file PDF yang diterima.")

    use_ocr = request.form.get("use_ocr") == "on"
    ocr_lang = "ind+eng"

    # Save to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        result = _build_result(tmp_path, use_ocr=use_ocr, ocr_lang=ocr_lang)
        result["source_pdf"] = file.filename
        result["meta"] = {"use_ocr": use_ocr, "ocr_lang": ocr_lang}
    except Exception as e:
        os.unlink(tmp_path)
        return render_template("index.html", error=f"Gagal memproses PDF: {e}", use_ocr=use_ocr)

    os.unlink(tmp_path)

    if request.args.get("format") == "json":
        return jsonify(result)

    return render_template("result.html", data=result, raw_json=json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
