"""VLM via OpenAI vision API (gpt-4o / gpt-4o-mini).

Rasterizes PDF pages → image → sends to OpenAI with a strict-JSON extraction
prompt → aggregates per-page results into the unified schema.

Configurable via env vars:
  OPENAI_API_KEY         (required)
  OPENAI_VLM_MODEL       (default: gpt-4o-mini)
  OPENAI_VLM_START_PAGE  (default: auto-detected from PDF text)
  OPENAI_VLM_END_PAGE    (default: auto-detected from PDF text)
  OPENAI_VLM_DPI         (default: 150)
  OPENAI_VLM_MAX_TOKENS  (default: 4096)

The approach name registered is ``vlm_openai_<modelname>`` so multiple models
can co-exist in the same comparison run.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from pathlib import Path

import pdfplumber
from openai import OpenAI

from idx_fin_parser.unified import (
    build_unified_output,
    build_statement,
    detect_statement_type,
    flat_rows_to_tree,
    split_into_sections,
    years_from_columns,
)

from . import register


# ── Prompts (mirror vlm_image_to_table.py / colab notebook for fairness) ──

SYSTEM_PROMPT = """You are an expert document structure extraction system.

Your task:
Extract a financial table from an image into STRICT JSON.

Important rules:
1. Preserve row hierarchy using indentation level.
2. Preserve the exact row order from top to bottom.
3. Extract account labels carefully.
4. Extract numeric values exactly as shown.
5. If a row has no value, return null.
6. If a row is a section/group header, still include it.
7. Negative values written in parentheses must stay as strings, e.g. "(7)".
8. Do not explain anything.
9. Output ONLY valid JSON.
10. Do not wrap JSON in markdown fences.

Return this schema exactly:

{
  "table_title": "string or null",
  "columns": ["account", "YEAR1", "YEAR2"],
  "rows": [
    {
      "level": 0,
      "account": "string",
      "YEAR1": "string or null",
      "YEAR2": "string or null",
      "row_type": "group|item|label_only"
    }
  ]
}

Replace YEAR1, YEAR2 with the actual year values found in the table header.
"""

USER_PROMPT = """Extract the table from this image.

Focus on:
- account names on the left
- value columns on the right
- indentation / hierarchy level
- section headers vs detail rows

Infer `level` like this:
- 0 = main section
- 1 = first indent
- 2 = deeper indent
- 3 = deeper indent if needed

Infer `row_type` like this:
- group = a grouping/header row
- item = a normal row with values
- label_only = a row without values that is not clearly a group header

Return STRICT JSON only.
"""


# ── Config ────────────────────────────────────────────────────────────────

MODEL = os.environ.get("OPENAI_VLM_MODEL", "gpt-4o-mini")
DPI = int(os.environ.get("OPENAI_VLM_DPI", "150"))
MAX_TOKENS = int(os.environ.get("OPENAI_VLM_MAX_TOKENS", "4096"))

START_FP = ["statement of financial position", "laporan posisi keuangan"]
STOP_FP = ["statement of profit or loss", "laporan laba rugi"]
START_PL = ["statement of profit or loss", "laporan laba rugi"]
STOP_PL = [
    "statement of changes in equity", "laporan perubahan ekuitas",
    "statement of cash flows", "laporan arus kas",
    "catatan atas laporan keuangan", "notes to the financial statements",
]


# ── Helpers ───────────────────────────────────────────────────────────────

def _find_pages(pdf, start_patterns, stop_patterns) -> list[int]:
    start_pats = [p.lower() for p in start_patterns]
    stop_pats = [p.lower() for p in stop_patterns]
    start = None
    for i, p in enumerate(pdf.pages):
        text = (p.extract_text() or "").lower()
        if any(s in text for s in start_pats):
            start = i
            break
    if start is None:
        return []
    pages = [start]
    for i in range(start + 1, len(pdf.pages)):
        text = (pdf.pages[i].extract_text() or "").lower()
        if any(s in text for s in stop_pats):
            break
        pages.append(i)
    return pages


def _page_to_base64_png(page, dpi: int) -> str:
    img = page.to_image(resolution=dpi).original
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("model output missing valid JSON object")
    return json.loads(text[start:end + 1])


def _call_openai(client: OpenAI, image_b64: str, model: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_b64}",
                }},
            ]},
        ],
    )
    return resp.choices[0].message.content or ""


def _statement_from_pages(pdf, pages_idx: list[int], statement_type: str,
                           client: OpenAI, model: str,
                           raw_cache: list | None = None) -> dict:
    by_type_rows: list[dict] = []
    all_years: list[int] = []
    pages_used: list[int] = []
    for i in pages_idx:
        page = pdf.pages[i]
        b64 = _page_to_base64_png(page, DPI)
        try:
            raw = _call_openai(client, b64, model)
            data = _extract_json(raw)
        except Exception as exc:
            print(f"      [page {i + 1}] ERROR: {exc}")
            continue
        if raw_cache is not None:
            raw_cache.append({"page": i, "statement_type": statement_type,
                              "raw": raw, "parsed": data})
        cols = data.get("columns", [])
        years = years_from_columns(cols)
        if years and not all_years:
            all_years = years
        rows = data.get("rows", [])
        by_type_rows.extend(rows)
        pages_used.append(i)
        print(f"      [page {i + 1}] {len(rows)} rows")

    nodes = flat_rows_to_tree(by_type_rows, all_years, label_en_key=None)
    sections = split_into_sections(nodes)
    return build_statement(statement_type, all_years, pages_used, sections)


# ── Entry point ───────────────────────────────────────────────────────────

def run(pdf_path: str) -> tuple[dict, float]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    client = OpenAI()
    model = MODEL
    raw_cache: list = []
    t0 = time.perf_counter()
    with pdfplumber.open(pdf_path) as pdf:
        fp_pages = _find_pages(pdf, START_FP, STOP_FP)
        pl_pages = _find_pages(pdf, START_PL, STOP_PL)
        print(f"      FP pages: {[p + 1 for p in fp_pages]}")
        print(f"      PL pages: {[p + 1 for p in pl_pages]}")
        fp = _statement_from_pages(pdf, fp_pages, "financial_position",
                                    client, model, raw_cache)
        pl = _statement_from_pages(pdf, pl_pages, "profit_or_loss",
                                    client, model, raw_cache)
    runtime = time.perf_counter() - t0

    # Save raw cache so the JSON can be rebuilt without re-paying for API calls.
    cache_path = Path(pdf_path).with_suffix(f".vlm_openai_{model}_raw.json")
    cache_path = Path("runs") / Path(pdf_path).stem / cache_path.name
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"model": model, "raw_responses": raw_cache},
                                      ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"      raw cache: {cache_path}")

    out = build_unified_output(
        source_pdf=str(pdf_path),
        approach=f"vlm_openai_{model.replace('-', '_')}",
        statements=[fp, pl],
        meta={"provider": "openai", "model": model, "dpi": DPI,
              "runtime_seconds": runtime,
              "raw_cache": str(cache_path)},
    )
    return out, runtime


def rebuild_from_cache(cache_path: str, pdf_path: str) -> dict:
    """Re-assemble unified output from cached raw VLM responses (no API call)."""
    cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    model = cache["model"]
    by_type: dict[str, dict] = {}
    for entry in cache["raw_responses"]:
        stmt_type = entry["statement_type"]
        data = entry["parsed"]
        years = years_from_columns(data.get("columns", []))
        bucket = by_type.setdefault(
            stmt_type, {"years": years, "pages": [], "rows": []})
        if years and not bucket["years"]:
            bucket["years"] = years
        bucket["pages"].append(entry["page"])
        bucket["rows"].extend(data.get("rows", []))

    statements = []
    for stmt_type, b in by_type.items():
        nodes = flat_rows_to_tree(b["rows"], b["years"], label_en_key=None)
        sections = split_into_sections(nodes)
        statements.append(build_statement(
            stmt_type, b["years"], b["pages"], sections))

    return build_unified_output(
        source_pdf=str(pdf_path),
        approach=f"vlm_openai_{model.replace('-', '_')}",
        statements=statements,
        meta={"provider": "openai", "model": model, "rebuilt_from_cache": True},
    )


# Register under a model-aware name so multiple variants can coexist
_approach_name = f"vlm_openai_{MODEL.replace('-', '_')}"
register(_approach_name, run)
