# vlm_image_to_table.py

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import pypdfium2 as pdfium
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


# =========================================
# CONFIG
# =========================================
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
DEFAULT_DPI = 150
DEFAULT_OUTPUT_DIR = "vlm_output"


# =========================================
# PROMPT
# =========================================
SYSTEM_PROMPT = """
You are an expert document structure extraction system.

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

USER_PROMPT = """
Extract the table from this image.

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


# =========================================
# PDF TO IMAGES
# =========================================
def pdf_to_images(pdf_path: str, start_page: int, end_page: int, dpi: int = DEFAULT_DPI):
    """
    Convert PDF pages to PIL Images.
    start_page and end_page are 1-indexed, inclusive.
    Returns list of (page_num, PIL.Image).
    """
    doc = pdfium.PdfDocument(pdf_path)
    total = len(doc)

    start_idx = max(1, start_page) - 1
    end_idx = min(total, end_page) - 1
    scale = dpi / 72  # pdfium renders at 72 DPI by default

    images = []
    for idx in range(start_idx, end_idx + 1):
        page = doc[idx]
        bitmap = page.render(scale=scale, rotation=0)
        pil_image = bitmap.to_pil()
        images.append((idx + 1, pil_image))

    doc.close()
    return images


def get_total_pages(pdf_path: str) -> int:
    doc = pdfium.PdfDocument(pdf_path)
    total = len(doc)
    doc.close()
    return total


# =========================================
# HELPERS
# =========================================
def extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model output tidak mengandung JSON object yang valid.")

    return json.loads(text[start:end + 1])


def normalize_rows(data: dict):
    rows = data.get("rows", [])
    columns = data.get("columns", [])
    value_cols = [c for c in columns if c != "account"]

    normalized = []
    for r in rows:
        row = {
            "level": int(r.get("level", 0)) if r.get("level") is not None else 0,
            "account": r.get("account"),
            "row_type": r.get("row_type", "item"),
        }
        for col in value_cols:
            row[col] = r.get(col)
        normalized.append(row)

    return normalized


def build_parent_child(rows):
    stack = []
    output = []

    for row in rows:
        level = row["level"]

        while stack and stack[-1]["level"] >= level:
            stack.pop()

        parent_account = stack[-1]["account"] if stack else None
        output.append({**row, "parent_account": parent_account})
        stack.append(row)

    return output


# =========================================
# LOAD MODEL
# =========================================
def load_model(model_name=MODEL_NAME):
    processor = AutoProcessor.from_pretrained(model_name)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    return processor, model


# =========================================
# INFERENCE
# =========================================
def run_inference(image: Image.Image, processor, model):
    """Run VLM inference on a PIL Image."""
    image = image.convert("RGB")

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            temperature=0.0,
        )

    output_text = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0]

    return output_text


# =========================================
# SAVE
# =========================================
def save_page_output(data: dict, page_num: int, output_dir: Path):
    rows = normalize_rows(data)
    rows = build_parent_child(rows)

    label = f"page_{page_num:03d}"
    json_path = output_dir / f"{label}.json"
    csv_path = output_dir / f"{label}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "page": page_num,
                "table_title": data.get("table_title"),
                "columns": data.get("columns", []),
                "rows": rows,
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    df = pd.DataFrame(rows)
    df.insert(0, "page", page_num)
    df.to_csv(csv_path, index=False)

    return rows, data.get("table_title"), data.get("columns", [])


def save_combined_output(all_results: list, output_dir: Path):
    json_path = output_dir / "combined.json"
    csv_path = output_dir / "combined.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    dfs = []
    for result in all_results:
        if not result.get("rows"):
            continue
        df = pd.DataFrame(result["rows"])
        df.insert(0, "page", result["page"])
        dfs.append(df)

    if dfs:
        pd.concat(dfs, ignore_index=True).to_csv(csv_path, index=False)

    return json_path, csv_path


# =========================================
# MAIN
# =========================================
def main():
    parser = argparse.ArgumentParser(
        description="Ekstrak tabel keuangan dari PDF menggunakan VLM (Qwen2-VL)."
    )
    parser.add_argument("pdf", help="Path ke file PDF")
    parser.add_argument(
        "--start-page", type=int, default=1,
        help="Halaman awal (1-indexed, default: 1)"
    )
    parser.add_argument(
        "--end-page", type=int, default=None,
        help="Halaman akhir (1-indexed, default: halaman terakhir)"
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Direktori output (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--dpi", type=int, default=DEFAULT_DPI,
        help=f"DPI render PDF ke gambar (default: {DEFAULT_DPI})"
    )
    parser.add_argument(
        "--model", default=MODEL_NAME,
        help=f"Nama model VLM (default: {MODEL_NAME})"
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {pdf_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_pages = get_total_pages(str(pdf_path))
    start_page = args.start_page
    end_page = args.end_page if args.end_page is not None else total_pages

    if start_page < 1 or start_page > total_pages:
        raise ValueError(f"--start-page {start_page} di luar rentang (1-{total_pages})")
    if end_page < start_page or end_page > total_pages:
        raise ValueError(f"--end-page {end_page} di luar rentang ({start_page}-{total_pages})")

    print(f"PDF     : {pdf_path} ({total_pages} halaman)")
    print(f"Proses  : halaman {start_page} s/d {end_page}")
    print(f"Output  : {output_dir}/")
    print(f"DPI     : {args.dpi}")

    print("\nMengkonversi halaman PDF ke gambar...")
    page_images = pdf_to_images(str(pdf_path), start_page, end_page, args.dpi)

    print(f"Memuat model {args.model}...")
    processor, model = load_model(args.model)

    all_results = []

    for page_num, image in page_images:
        print(f"\n[Halaman {page_num}/{end_page}] Running VLM inference...")
        try:
            raw_output = run_inference(image, processor, model)
            data = extract_json(raw_output)
            rows, title, columns = save_page_output(data, page_num, output_dir)

            all_results.append({
                "page": page_num,
                "table_title": title,
                "columns": columns,
                "rows": rows,
            })

            print(f"[Halaman {page_num}] OK — {len(rows)} baris diekstrak")

        except Exception as e:
            print(f"[Halaman {page_num}] ERROR: {e}")
            all_results.append({
                "page": page_num,
                "error": str(e),
                "rows": [],
            })

    print("\nMenyimpan output gabungan...")
    json_path, csv_path = save_combined_output(all_results, output_dir)

    success = sum(1 for r in all_results if not r.get("error"))
    failed = len(all_results) - success

    print(f"\nSelesai! {success} halaman berhasil, {failed} gagal.")
    print(f"  Per halaman : {output_dir}/page_XXX.json / .csv")
    print(f"  Combined    : {json_path}")
    print(f"               {csv_path}")


if __name__ == "__main__":
    main()
