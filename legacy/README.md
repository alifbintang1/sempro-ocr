# Legacy / Unused Code

Berkas di folder ini **tidak dipakai** oleh pipeline aktif (web app +
`ground_truth/run_all.py` + 8 approach). Disimpan untuk arsip / referensi,
bukan bagian dari sistem yang menghasilkan hasil thesis.

Pipeline aktif: `app.py`, `approaches/`, `idx_fin_parser/`, `ground_truth/`,
`tools/`, `templates/`.

---

## Isi folder ini

| Berkas | Dulu untuk apa | Kenapa tidak terpakai |
|---|---|---|
| `main_extract.py` | CLI ekstraksi awal (1 PDF → output.json) | Digantikan `ground_truth/run_all.py` |
| `main_extract_staged.py` | CLI ekstraksi bertahap (debug per-stage) | Digantikan `run_all.py` + stages di pdf_statements |
| `converter.py` | Konversi OCR-HTML (PaddleOCR `pred_html`) → unified JSON | Membaca `ocr_table_output/` (data dev lama); OCR sekarang via `approaches/ocr_full.py` (Tesseract) |
| `table_extractor.py` | Inspeksi cepat 1 halaman OCR → DataFrame | Skrip eksplorasi awal; tidak terhubung ke pipeline |
| `compile.py` | Gabung semua `.py` jadi `all_python_files.txt` | Utilitas dev; tidak dipakai sistem |
| `vlm_image_to_table.py` | VLM self-hosted Qwen2-VL (CLI lokal) | Tidak dipakai untuk hasil; VLM final pakai OpenAI API (`approaches/vlm_openai.py`) |
| `vlm_colab.ipynb` | VLM self-hosted Qwen2-VL (Google Colab) | Jalur alternatif "self-hosted VLM"; tidak menghasilkan hasil di komparasi |
| `VLM_WORKFLOW.md` | Panduan jalankan Qwen2-VL di Colab | Mendokumentasikan jalur `vlm_image_to_table.py` / notebook di atas |

---

## Catatan

- **Jalur VLM yang dipakai thesis** adalah OpenAI API
  (`approaches/vlm_openai.py`, model `gpt-4o` & `gpt-4o-mini`), bukan Qwen
  self-hosted di atas.
- Kalau suatu saat ingin menjalankan VLM self-hosted (tanpa biaya API),
  jalur Qwen di sini masih lengkap dan bisa dipindah kembali ke root.
- Semua berkas masih dilacak Git, jadi pemindahan ini reversibel.
