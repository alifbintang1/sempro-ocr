# Sistem Otomasi Ekstraksi Laporan Keuangan IDX

Prototype sistem ekstraksi laporan keuangan tahunan emiten Bursa Efek
Indonesia (IDX) ke skema JSON terpadu, dengan perbandingan 8 pendekatan
ekstraksi: rule-based custom, library generik, OCR (Tesseract), dan
Vision-Language Model (OpenAI GPT-4o).

Skripsi Teknik Informatika ITB.

---

## Quick Start

```bash
# 1. Clone
git clone <repo-url> sempro
cd sempro

# 2. Setup virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # Linux/Mac

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. (Opsional) Install Tesseract OCR — hanya jika mau pakai approach ocr_full
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# Mac:     brew install tesseract tesseract-lang
# Linux:   sudo apt install tesseract-ocr tesseract-ocr-ind tesseract-ocr-eng

# 5. (Opsional) Setup .env untuk approach VLM
echo "OPENAI_API_KEY=sk-..." > .env

# 6. Jalankan web UI
python app.py
# → buka http://localhost:5001

# 7. Atau headless via CLI: jalankan semua approach pada PDF demo
python ground_truth/run_all.py docs/FinancialStatement-2025-Tahunan-BBNI.pdf
```

---

## Step-by-step Detail

### 1. Prasyarat

| Komponen | Versi minimum | Wajib? |
|---|---|---|
| Python | 3.10 (disarankan 3.12) | ✅ |
| Git | apa pun | ✅ |
| Tesseract OCR | 5.0+ | Optional (untuk approach `ocr_full`) |
| OpenAI API key | — | Optional (untuk approach VLM) |
| GPU | — | Tidak dibutuhkan (VLM via API) |

### 2. Clone repository

```bash
git clone <repo-url> sempro
cd sempro
```

### 3. Buat virtual environment & install dependencies

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verifikasi instalasi:
```bash
python -c "import pdfplumber, fitz, camelot, openai, flask; print('OK')"
```

### 4. (Opsional) Install Tesseract OCR

Diperlukan **hanya** jika ingin pakai approach `ocr_full`. Approach
lain berjalan tanpa Tesseract.

**Windows:**
1. Download installer dari https://github.com/UB-Mannheim/tesseract/wiki
2. Install dengan opsi "Add to PATH" dicentang
3. Pastikan paket bahasa Indonesia (`ind`) terpilih saat install
4. Verifikasi: `tesseract --version`

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-ind tesseract-ocr-eng
```

### 5. (Opsional) Setup OpenAI API key

Diperlukan **hanya** jika ingin pakai approach `vlm_openai_gpt_4o_mini`
atau `vlm_openai_gpt_4o`.

Buat berkas `.env` di root proyek:
```
OPENAI_API_KEY=sk-proj-XXXXXXXXXXXXXXXX
```

(Optional) Set model VLM yang dipakai:
```
OPENAI_VLM_MODEL=gpt-4o-mini   # default; ganti ke gpt-4o untuk akurasi lebih
```

Berkas `.env` sudah tercantum di `.gitignore` jadi tidak akan
ter-commit.

**Estimasi biaya per PDF (BBNI 16 halaman):**
- `gpt-4o-mini`: ~$0,03
- `gpt-4o`: ~$0,50

### 6. Jalankan

#### Opsi A: Web UI (untuk demo)

```bash
python app.py
```

Buka http://localhost:5001 di browser. Pilih:
- Upload PDF Anda **atau** klik tombol demo BBNI 2025
- Centang approach yang ingin dijalankan
- Klik **Jalankan Ekstraksi**

Hasil akan menampilkan:
- Tabel komparasi metrik (jika ground truth tersedia)
- Tab per-approach dengan PSAK view + raw JSON

#### Opsi B: CLI untuk evaluasi 8-way

```bash
# Jalankan semua approach + bandingkan vs ground truth BBNI
python ground_truth/run_all.py docs/FinancialStatement-2025-Tahunan-BBNI.pdf

# Hanya approach tertentu
python ground_truth/run_all.py docs/...pdf --only native_pdf,pymupdf_native

# Skip approach lambat
python ground_truth/run_all.py docs/...pdf --skip ocr_full,vlm_openai_gpt_4o
```

Output di `runs/<pdf_stem>/`:
- `pred_<approach>.json` — prediksi per approach (skema unified)
- `report_<approach>.json` — laporan metrik per approach
- `comparison.json` — agregat machine-readable
- `comparison.md` — tabel paper-ready

#### Opsi C: Render PSAK HTML standalone

```bash
python tools/render_psak.py ground_truth/bbni_2025.json
# → menghasilkan ground_truth/bbni_2025.html
```

#### Opsi D: Generate grafik analisis (untuk thesis)

```bash
python tools/make_figures.py
# → menghasilkan 3 PNG di runs/.../figures/
```

---

## Struktur Direktori

```
sempro/
├── app.py                       # Flask web UI
├── requirements.txt
├── .env                         # API key (tidak ter-commit)
├── .gitignore
│
├── idx_fin_parser/              # Parser inti + skema
│   ├── unified.py               # Skema data terpadu
│   ├── utils.py                 # Ekstraksi kolom, deteksi tahun
│   └── pdf_statements.py        # Pipeline ekstraksi native/OCR
│
├── approaches/                  # 8 pendekatan ekstraksi (plugin)
│   ├── __init__.py              # Registry + .env loader
│   ├── native_pdf.py            # pdfplumber custom column
│   ├── pymupdf_native.py        # PyMuPDF (37× faster)
│   ├── ocr_full.py              # Tesseract force OCR
│   ├── baseline_regex.py        # Naive floor baseline
│   ├── pdfplumber_tables.py     # extract_tables() bawaan
│   ├── camelot_lattice.py       # camelot stream
│   └── vlm_openai.py            # OpenAI Vision API
│
├── ground_truth/
│   ├── build_bbni_2025.py       # Builder ground truth
│   ├── bbni_2025.json           # Gold standard (339 baris)
│   ├── evaluate.py              # Evaluator 5 metrik
│   ├── compare.py               # Comparison side-by-side
│   └── run_all.py               # Orkestrator end-to-end
│
├── tools/
│   ├── render_psak.py           # Renderer HTML PSAK-style
│   └── make_figures.py          # Generator grafik matplotlib
│
├── templates/                   # Template Flask
│   ├── index.html               # Upload + approach selector
│   └── result.html              # Tabel komparasi + per-approach tabs
│
├── docs/                        # PDF input demo
│   └── FinancialStatement-2025-Tahunan-BBNI.pdf
│
├── runs/                        # Output ekstraksi (gitignored)
└── vlm_colab.ipynb              # Self-hosted Qwen2-VL (Colab)
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'fitz'`
PyMuPDF tidak terinstal. Jalankan `pip install -r requirements.txt`.

### `TesseractNotFoundError`
Tesseract binary tidak ditemukan. Pastikan Tesseract terinstal (lihat
Langkah 4) dan PATH-nya ter-set. Pada Windows, default install path
sudah ter-handle di kode (`C:\Program Files\Tesseract-OCR\`).

### `OPENAI_API_KEY not set in environment`
Approach VLM membutuhkan API key. Buat `.env` (lihat Langkah 5) atau
hindari memilih approach VLM di web UI / `--skip vlm_*` di CLI.

### Approach VLM lambat (~5 menit)
Normal. `gpt-4o-mini` lebih lambat tapi murah; `gpt-4o` lebih cepat
tapi mahal. Hasil API di-cache di `runs/.../*_raw.json`, sehingga
re-rebuild tidak perlu re-call API:

```python
from approaches.vlm_openai import rebuild_from_cache
pred = rebuild_from_cache("runs/.../...vlm_openai_gpt_4o_raw.json", "docs/...pdf")
```

### Web UI: "approach VLM disabled"
OPENAI_API_KEY belum ter-set. Setup `.env` (Langkah 5) lalu restart
`python app.py`.

### `pdfplumber` extract_tables menghasilkan 0 tabel
PDF tidak memiliki garis pembatas tabel eksplisit (kasus IDX BBNI).
Kode sudah menggunakan strategi `text` untuk handling ini.

---

## Reproducibility

Untuk mereproduksi hasil komparasi di Bab IV skripsi:

```bash
# 1. Pastikan ground truth ada
ls ground_truth/bbni_2025.json

# 2. Jalankan 8-way comparison (perlu Tesseract + OPENAI_API_KEY untuk full)
python ground_truth/run_all.py docs/FinancialStatement-2025-Tahunan-BBNI.pdf

# 3. Generate grafik untuk thesis
python tools/make_figures.py

# 4. Hasil di runs/FinancialStatement-2025-Tahunan-BBNI/
#    - comparison.md            (tabel paper-ready)
#    - comparison.json          (machine-readable)
#    - figures/*.png            (Gambar 4.3, 4.4, 4.5)
#    - pred_*.json              (prediksi tiap approach)
```

---

## Lisensi

Akademik / non-komersial. Hak cipta penulis skripsi.
