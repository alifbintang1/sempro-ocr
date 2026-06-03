# VLM Workflow (Qwen2-VL via Colab)

Karena VLM butuh GPU, jalankan di Colab. Workflow:

## 1. Buka Colab
File [vlm_colab.ipynb](vlm_colab.ipynb) di Google Colab dengan runtime **T4 GPU**.

## 2. Setup (di Colab, sebelum cell 1)
Tambah cell baru di awal yang clone repo:
```python
!git clone <your-repo-url> sempro
%cd sempro
```
Atau upload manual: `vlm_image_to_table.py`, `idx_fin_parser/unified.py`,
`idx_fin_parser/utils.py`.

## 3. Run inferensi
```python
!python vlm_image_to_table.py docs/FinancialStatement-2025-Tahunan-BBNI.pdf \
    --start-page 4 --end-page 19 \
    --output-dir vlm_output
```

Output akan mencakup `vlm_output/unified.json` (skema yang sama dengan native &
OCR).

## 4. Download & evaluasi
Download `vlm_output/unified.json` dari Colab, simpan sebagai `pred_vlm.json`
di project root, lalu:

```bash
python ground_truth/run_all.py docs/FinancialStatement-2025-Tahunan-BBNI.pdf \
    --vlm-json pred_vlm.json --skip-ocr
```

(`--skip-ocr` untuk skip re-run OCR — sudah dilakukan)

## 5. Cek hasil
File [runs/<pdf_stem>/comparison.md](runs/) akan memuat 3-way comparison
lengkap.

---

## Catatan untuk thesis

- **Runtime per page** harus di-record dari Colab (Qwen2-VL 7B di T4 ≈ 15-30s/page).
- **Cost estimation**: Colab Pro ~Rp200k/bulan; gratis hanya untuk eksperimen
  pendek. Setiap PDF ~16 halaman × 25s = ~7 menit GPU time.
- **Non-determinism**: jalankan 2-3 kali untuk lihat variansi (penting untuk
  bab "limitations").
