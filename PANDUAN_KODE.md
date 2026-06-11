# Panduan Cara Kerja Kode — Sistem Ekstraksi Laporan Keuangan IDX

Dokumen ini menjelaskan cara kerja setiap bagian kode di repo `sempro/`,
dari modul tingkat rendah sampai antarmuka web. Ditujukan untuk:
- Penulis skripsi yang membaca ulang kode setelah jeda
- Reviewer / dosen pembimbing yang ingin memahami arsitektur
- Pengembang lain yang ingin menambah pendekatan baru

---

## Bab 1. Gambaran Sistem

### 1.1 Apa yang dilakukan sistem ini

Sistem menerima berkas PDF laporan keuangan emiten IDX (Bursa Efek Indonesia)
dan menghasilkan struktur JSON hierarkis yang memuat:

- Header laporan (tahun, halaman, jenis)
- Pembagian seksi (Aset / Liabilitas / Ekuitas / dst.)
- Akun-akun dalam hierarki tree (group → item → sub-item)
- Nilai numerik per tahun pembanding

Sistem mendukung **8 pendekatan ekstraksi yang berbeda** dan menyediakan
*harness* untuk membandingkannya secara kuantitatif terhadap *ground truth*
yang dibentuk manual.

### 1.2 Aliran data tingkat tinggi

```
PDF input
    │
    ▼
┌────────────────────────────────────────────────┐
│  Lapisan Ekstraksi (8 pendekatan modular)      │
│                                                │
│  native_pdf, pymupdf_native, baseline_regex,   │
│  pdfplumber_tables, camelot_stream, ocr_full,  │
│  vlm_openai_gpt_4o_mini, vlm_openai_gpt_4o     │
└────────────────┬───────────────────────────────┘
                 │ Output: unified schema JSON
                 ▼
        ┌────────────────────┐
        │  Lapisan Evaluasi  │  evaluate.py
        │  (5 metrik)        │  run_all.py
        └─────────┬──────────┘
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
   PSAK HTML            Tabel Komparasi
   (render_psak.py)     (comparison.md)
```

### 1.3 Prinsip desain inti

1. **Unified schema** — semua pendekatan menghasilkan struktur JSON yang
   sama. Ini memungkinkan komparasi adil dan komposisi modul.
2. **Plug-in registry** — pendekatan baru ditambahkan dengan satu berkas
   baru di `approaches/`, tanpa mengubah modul lain.
3. **Deterministic core** — modul evaluasi dan penyaji tidak bergantung
   pada model atau heuristik tertentu. Hanya butuh JSON yang valid.

---

## Bab 2. Struktur Direktori

```
sempro/
│
├── idx_fin_parser/              # ← MODUL INTI: skema + utilitas + parser
│   ├── unified.py               # Skema data terpadu + adapter
│   ├── utils.py                 # Utilitas ekstraksi: kolom, tahun, kontinuasi
│   └── pdf_statements.py        # Pipeline ekstraksi native/OCR end-to-end
│
├── approaches/                  # ← PLUG-IN: tiap berkas = 1 pendekatan
│   ├── __init__.py              # Registry + pemuat .env
│   ├── native_pdf.py
│   ├── pymupdf_native.py
│   ├── baseline_regex.py
│   ├── pdfplumber_tables.py
│   ├── camelot_lattice.py
│   ├── ocr_full.py
│   └── vlm_openai.py
│
├── ground_truth/                # ← GROUND TRUTH + EVALUATOR
│   ├── build_bbni_2025.py       # Builder GT yang auditable
│   ├── bbni_2025.json           # Gold standard (339 baris)
│   ├── evaluate.py              # 5 metrik evaluasi
│   ├── compare.py               # Tabel side-by-side
│   └── run_all.py               # Orkestrator end-to-end
│
├── tools/                       # ← MODUL PENYAJI
│   ├── render_psak.py           # JSON → HTML format PSAK 201
│   └── make_figures.py          # Generator grafik analisis
│
├── templates/                   # ← WEB UI
│   ├── index.html
│   └── result.html
│
├── app.py                       # ← ENTRY POINT WEB
├── docs/                        # PDF input
├── runs/                        # Output (gitignored)
├── requirements.txt
├── README.md
└── .env                         # API key (gitignored)
```

---

## Bab 3. Modul Inti: `idx_fin_parser/`

### 3.1 `unified.py` — Skema Data Terpadu

**Tujuan:** Mendefinisikan kontrak data antar modul. Semua approach harus
menghasilkan struktur yang mengikuti skema ini.

#### Struktur skema

```json
{
  "schema_version": "1.0",
  "source_pdf":     "<nama berkas PDF>",
  "approach":       "<nama pendekatan>",
  "meta":           { ... },
  "statements": [
    {
      "type":     "financial_position" | "profit_or_loss" | ...,
      "years":    [2025, 2024],
      "pages":    [4, 5, ..., 12],
      "sections": { "<nama_seksi>": [Node, ...] }
    }
  ]
}
```

#### Struktur Node (rekursif)

```json
{
  "level":     0,
  "label":     "Giro pada bank lain",
  "label_en":  "Current accounts with other banks",
  "row_type":  "group" | "item" | "label_only",
  "values":    { "2025-12-31": 25858094, "2024-12-31": 22031212 },
  "children":  [ /* Node, ... */ ]
}
```

#### Fungsi-fungsi inti

| Fungsi | Tujuan |
|---|---|
| `make_node(label, label_en, values, level, children, row_type)` | Konstruktor Node dengan validasi |
| `build_statement(type, years, pages, sections)` | Konstruktor statement |
| `build_unified_output(source_pdf, approach, statements, meta)` | Konstruktor output lengkap |
| `parse_value(v)` | Parse string angka "1,234" atau "(7)" → int |
| `years_from_columns(columns)` | Ekstrak `[2025, 2024]` dari `["31 December 2025", ...]` |
| `detect_statement_type(title)` | Map "Laporan Posisi Keuangan" → "financial_position" |
| `flat_rows_to_tree(rows, years)` | Konversi flat rows (VLM output) → nested tree |
| `split_into_sections(nodes)` | Pisahkan ke seksi via section header detection |
| `itemnode_to_unified(item_node)` | Konversi ItemNode dataclass → dict skema |

#### Adapter penting: `flat_rows_to_tree`

Mengkonversi keluaran flat (typical untuk VLM) menjadi pohon hierarkis
berdasarkan kolom `level`:

```python
# Input: [
#   {"level": 0, "account": "Aset", "values": {...}},
#   {"level": 1, "account": "Kas", "values": {...}},
#   {"level": 1, "account": "Giro", "values": {...}},
# ]
# Output: tree dengan "Aset" sebagai root, "Kas" dan "Giro" sebagai anak
```

Logika: gunakan stack — pop semua simpul dengan `level >= row.level`,
lalu lampirkan row baru sebagai anak dari simpul teratas stack.

#### Adapter penting: `split_into_sections`

Memisahkan daftar simpul level-0 ke dalam seksi terpisah berdasarkan
deteksi label seksi (mis. "Aset" → seksi `assets`).

**Tiga gaya input ditangani:**

1. **native_pdf style** — label-only header (mis. "Aset") muncul sebagai
   simpul terpisah, diikuti item-itemnya sebagai sibling
2. **VLM flat style** — section header punya `children` yang langsung
   berisi item — children di-*promote* ke seksi
3. **VLM wrapped style** — ada wrapper "Laporan posisi keuangan" yang
   nest seluruh konten — wrapper di-*unwrap* satu level

### 3.2 `utils.py` — Utilitas Ekstraksi

**Tujuan:** Fungsi-fungsi *helper* yang dipakai oleh `pdf_statements.py`
dan modul-modul approach.

#### Fungsi utama

##### `extract_rows_by_column(page, years_set)` — Inti ekstraksi koordinat

Algoritma:

1. **Extract words dengan koordinat** dari halaman PDF
2. **Group ke baris visual** berdasarkan koordinat `top` (toleransi 3pt)
3. **First pass — deteksi ambang kolom EN**: dari baris-baris yang punya
   angka, catat posisi `x0` dari kata pertama setelah angka terakhir.
   Threshold = `min(en_x_starts)`.
4. **Second pass — bangun rows terstruktur**: untuk tiap baris:
   - Kalau punya angka: split ke `id_label | amounts | en_label`
     berdasarkan posisi angka
   - Kalau tanpa angka: gunakan `_split_bilingual_row()` (gap-based)

##### `_split_bilingual_row(row_words, en_x_threshold)` — Pemisah kolom label-only

Mengatasi *failure mode* "EN-bleed":

```python
# Algoritma:
# 1. Hitung celah horizontal antar kata berurutan
# 2. Kalau celah terbesar > 30pt dan > 3× median:
#       split di celah itu (left = ID, right = EN)
# 3. Sebaliknya: fallback ke en_x_threshold
```

##### `assign_levels(rows)` — Deteksi level hierarki

```python
# 1. Ambil x0_first (posisi paling kiri kata pertama) dari rows yang punya
#    amounts (mengabaikan noise dari section header)
# 2. Cluster nilai-nilai x0 ke level diskrit (toleransi 8pt)
# 3. Cluster terkiri = level 0, berikutnya level 1, dst.
# 4. Assign level ke tiap row berdasarkan x0_first-nya
```

##### `merge_continuation_rows(rows)` — Penggabung baris kontinuasi

Mengatasi label yang terpotong ke baris berikutnya. Tiga heuristik:

1. `id_label` baris saat ini kosong (cuma EN text wrap)
2. `id_label` baris saat ini diawali huruf kecil (suffix label ID)
3. Kata terakhir baris sebelumnya = kata penghubung (`bank`, `pada`,
   `dan`, dst.) DAN `x0` baris saat ini sama dengan sebelumnya

##### `looks_like_section_header(label)` — Deteksi nama seksi

```python
# Strict exact-match terhadap whitelist:
# "aset" → "assets"
# "liabilitas" → "liabilities"
# "ekuitas" → "equity"
# "dana syirkah temporer" → "temporary_syirkah_funds"
# dll.
```

Awalnya pakai prefix matching (mis. `l.startswith("liabilitas ")`)
tapi menyebabkan false positive — diperbaiki menjadi exact match
dengan whitelist + blacklist.

##### `find_years_in_order(lines)` — Deteksi tahun pembanding

Cari pola 4 digit (1900-2099) di 50 baris pertama dokumen. Ambil 2
tahun terbanyak muncul dalam urutan yang sama (mis. `[2025, 2024]`).

##### Class `ItemNode` (dataclass)

Representasi simpul dalam memori. Method `to_dict()` mengkonversi ke
dict yang mengikuti skema unified.

### 3.3 `pdf_statements.py` — Pipeline Ekstraksi Native/OCR

**Tujuan:** Mengkoordinasikan deteksi halaman, ekstraksi konten, dan
pembangunan pohon untuk approach `native_pdf` dan `ocr_full`.

#### Fungsi-fungsi inti

##### `_extract_page_text(page, use_ocr, ocr_lang, force_ocr)`

```python
# Logika ekstraksi teks halaman:
#
# if force_ocr:
#     skip text layer, langsung OCR (untuk benchmark)
# else:
#     text = page.extract_text()
#     if text kosong DAN use_ocr=True:
#         fallback ke OCR
#     return text
```

##### `_find_page_index(pdf, start_patterns)` — Cari halaman start statement

Iterasi halaman; kembalikan indeks halaman pertama yang memuat kata
kunci start.

##### `_collect_pages_until(pdf, start_idx, stop_patterns)` — Kumpulkan halaman

Mulai dari `start_idx`, kumpulkan halaman sampai bertemu kata kunci
stop. Tidak termasuk halaman stop.

##### `_extract_statement_generic(pdf_path, ..., force_ocr)`

Pipeline lengkap untuk satu jenis statement:

```python
# 1. Buka PDF
# 2. Cari halaman start
# 3. Kumpulkan halaman target
# 4. Ekstrak text lines + deteksi tahun
# 5. Branch:
#    if text_based (force_ocr atau use_ocr):
#        _merge_wrapped_lines → _parse_merged_lines_to_tree
#    else:
#        extract_rows_by_column → assign_levels →
#        _parse_structured_rows_to_tree
# 6. Return StatementResult
```

##### `_parse_structured_rows_to_tree(rows, years)` — Algoritma tree native

Stack-based attach:

```python
stack = []
current_section = "unknown"

for row in rows:
    # Deteksi section header
    sec = looks_like_section_header(row.id_label)
    if sec and not row.amounts:
        current_section = sec
        stack = []
        continue

    # Buat node
    node = ItemNode(label=..., values=..., level=row.level)

    # Pop stack sampai bisa attach
    while stack and stack[-1].level >= row.level:
        stack.pop()

    if stack:
        stack[-1].children.append(node)
    else:
        sections[current_section].append(node)

    stack.append(node)
```

##### `extract_statement_financial_position()` dan `extract_statement_profit_loss()`

Pemanggil `_extract_statement_generic` dengan parameter spesifik
(start/stop patterns).

##### `extract_with_stages(pdf_path, ...)`

Versi yang juga return informasi stage-by-stage (raw text per halaman,
normalized lines, merged lines, structured rows) — berguna untuk
debugging dan inspeksi proses ekstraksi.

---

## Bab 4. Approach Ekstraksi

Pola umum tiap berkas di `approaches/`:

```python
from . import register

def run(pdf_path: str) -> tuple[dict, float]:
    """
    Input  : path ke PDF
    Output : (unified_dict, elapsed_seconds)
    """
    t0 = time.perf_counter()
    # ... lakukan ekstraksi ...
    runtime = time.perf_counter() - t0
    return unified_output, runtime

register("nama_approach", run)
```

Registrasi otomatis terjadi saat impor `from approaches import REGISTRY`
di `run_all.py` atau `app.py`.

### 4.1 `native_pdf.py` — Custom Column Coordinate

**Pustaka:** `pdfplumber`
**Algoritma:** Tahap 3.3 (`_extract_statement_generic`)
**Output:** Pohon hierarkis dengan section terdeteksi

```python
def run(pdf_path):
    t0 = time.perf_counter()
    fp = extract_statement_financial_position(pdf_path)
    pl = extract_statement_profit_loss(pdf_path)
    out = build_unified_output(
        source_pdf=pdf_path, approach="native_pdf",
        statements=[fp.to_dict(), pl.to_dict()], meta={...},
    )
    return out, time.perf_counter() - t0
```

### 4.2 `pymupdf_native.py` — Ablasi Pustaka

**Pustaka:** `PyMuPDF (fitz)`
**Algoritma:** **Sama dengan `native_pdf`** tapi extract koordinat kata
dengan `page.get_text("words")` dari PyMuPDF.

Modul ini menyalin logika `extract_rows_by_column` ke dalam fungsi lokal
`_extract_rows_pymupdf` yang menerima `fitz.Page` (bukan `pdfplumber.Page`).
Tujuan: ablasi kontribusi pustaka backend terhadap akurasi dan kecepatan.

**Hasil empiris:** Akurasi identik, runtime 37× lebih cepat (0.1s vs 3.9s).

### 4.3 `baseline_regex.py` — Floor Baseline

**Pustaka:** `pdfplumber.extract_text()` (cuma string output)
**Algoritma:**

```python
# Untuk tiap halaman target:
#   text = page.extract_text()
#   for raw_line in text.splitlines():
#       tokens = regex match _AMOUNT_RE
#       if not tokens:
#           emit node(label=raw_line, values={})  # label-only
#       else:
#           label = raw_line[:tokens[0].start]
#           amounts = parse(tokens)
#           emit node(label=label, values=amounts)
# Semua simpul level 0; sections via split_into_sections
```

**Karakteristik:** Tidak ada deteksi hierarki, tidak ada deteksi kolom
bilingual. Berguna sebagai floor untuk mengukur kontribusi sophistication
pendekatan lain.

### 4.4 `pdfplumber_tables.py` — Library Default

**Pustaka:** `pdfplumber.extract_tables()`
**Algoritma:**

```python
# Per halaman:
#   tables = page.extract_tables(table_settings={
#       "vertical_strategy": "text",
#       "horizontal_strategy": "text",
#       "snap_tolerance": 3,
#       "join_tolerance": 3,
#   })
#   for row in concat(tables):
#       cells = [normalize(c) for c in row]
#       first non-empty = label
#       remaining: parse_value → amount, else → en_label
```

**Karakteristik:** Cepat namun tidak adekuat untuk PDF tanpa garis
pembatas tabel. Sering menggabungkan dua sel berurutan menjadi satu
(failure mode catastrophic).

### 4.5 `camelot_lattice.py` — Third-party Tool

**Pustaka:** `camelot-py`, flavor `"stream"`
**Algoritma:**

```python
# 1. Cari halaman target dulu (pakai pdfplumber, lebih cepat)
# 2. tables = camelot.read_pdf(pdf, pages='4,5,...', flavor='stream')
# 3. Untuk tiap tabel:
#       for row in table.df.iterrows():
#           non_empty = [c for c in row if c.strip()]
#           first = id_label, parse_value → amounts, last = en_label
```

**Catatan:** `lattice` flavor tidak dipakai karena butuh garis pembatas
visible dan dependency Ghostscript. Stream lebih cocok untuk layout
borderless seperti BBNI.

### 4.6 `ocr_full.py` — Tesseract Force OCR

**Pustaka:** `pytesseract` + Tesseract binary
**Algoritma:** Memanggil `extract_statement_*` dari `pdf_statements.py`
dengan `force_ocr=True`.

```python
def run(pdf_path):
    t0 = time.perf_counter()
    fp = extract_statement_financial_position(pdf_path, force_ocr=True)
    pl = extract_statement_profit_loss(pdf_path, force_ocr=True)
    ...
```

Internal: setiap halaman target di-rasterize ke PIL Image 250 DPI, lalu
diserahkan ke Tesseract dengan `lang="ind+eng"`. Output teks diparse
dengan text-line parser (bukan column-based) karena koordinat tidak
tersedia.

**Optimisasi penting:** Page detection (mencari halaman start) tetap
pakai text layer, bukan OCR. Tanpa optimisasi ini, OCR seluruh PDF
akan menghabiskan >15 menit.

### 4.7 `vlm_openai.py` — Vision-Language Model API

**Pustaka:** `openai` Python SDK
**Algoritma:**

```python
def run(pdf_path):
    client = OpenAI()  # baca OPENAI_API_KEY dari .env
    model = os.environ.get("OPENAI_VLM_MODEL", "gpt-4o-mini")

    for halaman in halaman_target:
        # 1. Rasterize → PIL Image 150 DPI
        image = halaman.to_image(resolution=150).original
        # 2. Encode ke base64 PNG
        b64 = base64_encode(image.to_png())
        # 3. Panggil API
        response = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64}"}},
                ]},
            ],
        )
        # 4. Parse JSON dari respons
        data = extract_json(response.choices[0].message.content)
        # 5. Tambahkan ke raw_cache (untuk rebuild tanpa re-pay API)
        # 6. Akumulasi rows

    # 7. Build tree dari rows
    nodes = flat_rows_to_tree(rows, years)
    sections = split_into_sections(nodes)
    return unified_output, runtime
```

#### System prompt VLM

```
You are an expert document structure extraction system.
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
```

#### Raw response caching

Setiap panggilan API disimpan ke berkas `*_raw.json` di `runs/`. Fungsi
`rebuild_from_cache(cache_path, pdf_path)` dapat mengkonstruksi ulang
prediksi dari cache **tanpa memanggil API lagi** — berguna setelah
perbaikan algoritma agregasi.

#### Dua model didaftarkan otomatis

Berdasarkan env `OPENAI_VLM_MODEL`:
- `OPENAI_VLM_MODEL=gpt-4o-mini` → `vlm_openai_gpt_4o_mini` di registry
- `OPENAI_VLM_MODEL=gpt-4o` → `vlm_openai_gpt_4o` di registry

---

## Bab 5. Pipeline Evaluasi

### 5.1 `ground_truth/evaluate.py`

**Tujuan:** Menghitung 5 metrik antara satu prediksi dan satu ground truth.

#### Tahap utama

```python
def evaluate(gold, pred) -> dict:
    # 1. Group statements by type (financial_position, profit_or_loss)
    # 2. Untuk setiap pernyataan yang ada di gold:
    #    a. Build path map: dict[(section, normalized_path)] → node
    #    b. Hitung metrik:
    #       - _score_nodes(gold_keys, pred_keys)
    #       - _score_cells(gold_map, pred_map, tp_keys)
    #       - _avg_label_similarity(...)
    #       - _score_hierarchy(...)
    # 3. Hitung macro averages
    # 4. Return report dict
```

#### Detail per metrik

##### Node F1

Pencocokan baris berdasarkan kunci `(section_name, normalized_label_path)`:

```python
gold_keys = {(section, path) for path in gold_paths}
pred_keys = {(section, path) for path in pred_paths}
tp = gold_keys & pred_keys
fp = pred_keys - gold_keys
fn = gold_keys - pred_keys
precision = |tp| / |pred_keys|
recall    = |tp| / |gold_keys|
f1        = 2 * P * R / (P + R)
```

Normalisasi label: lowercase, strip whitespace ganda, hilangkan tanda baca.

##### Cell value F1 + MAE

Pada simpul TP, evaluasi nilai per (path, tahun):

```python
for k in tp:
    for year_key in years:
        g_val = gold_map[k]["values"].get(year_key)
        p_val = pred_map[k]["values"].get(year_key)
        if g_val is not None: cells_gold += 1
        if p_val is not None: cells_pred += 1
        if g_val == p_val and g_val is not None: cells_correct += 1
        if both not None: err_sum += abs(g_val - p_val); err_n += 1
P = cells_correct / cells_pred
R = cells_correct / cells_gold
F1 = 2 P R / (P + R)
MAE = err_sum / err_n
```

##### Hierarchy Accuracy

Berapa % label yang cocok punya `parent_path` yang benar:

```python
for label in gold_labels & pred_labels:
    for gold_entry of label:
        if gold_entry.parent_path in pred_label_parents[label]:
            correct += 1
        total += 1
accuracy = correct / total
```

##### Label Similarity

Rata-rata Levenshtein ratio antar label TP:

```python
similarities = []
for k in tp:
    similarities.append(1 - lev(gold[k].label, pred[k].label) / max_len)
avg = mean(similarities)
```

### 5.2 `ground_truth/run_all.py`

**Tujuan:** Orkestrator yang menjalankan semua approach + menghitung
metrik + menyimpan hasil komparasi.

#### Alur eksekusi

```python
# 1. Parse argumen: pdf path, gold path, --only / --skip filter, --vlm-json
# 2. Muat ground truth
# 3. Filter approaches sesuai --only/--skip
# 4. Loop:
#    for name, fn in approaches:
#        try:
#            pred, runtime = fn(pdf_path)
#            simpan pred ke runs/<stem>/pred_<name>.json
#        except: catat error, lanjut
# 5. Muat external predictions via --vlm-json (untuk cached / Colab results)
# 6. Loop:
#    for approach, pred, runtime in results:
#        report = evaluate(gold, pred)
#        simpan report ke runs/<stem>/report_<name>.json
#        tambahkan baris ke comparison table
# 7. Tulis comparison.json + comparison.md
```

#### CLI flags

| Flag | Fungsi |
|---|---|
| `--gold <path>` | Path ke GT JSON (default: `ground_truth/bbni_2025.json`) |
| `--out-dir <dir>` | Output directory (default: `runs/<pdf_stem>/`) |
| `--only <a,b,c>` | Jalankan hanya approach ini |
| `--skip <a,b,c>` | Skip approach ini |
| `--vlm-json <p1,p2>` | Muat prediksi pre-computed (mis. dari Colab) |

### 5.3 `ground_truth/compare.py`

Mode pembanding lightweight: terima beberapa file prediksi yang sudah
ada, jalankan evaluator, cetak tabel side-by-side. Tidak menjalankan
approach.

```bash
python ground_truth/compare.py pred_native.json pred_ocr.json
```

---

## Bab 6. Modul Penyaji

### 6.1 `tools/render_psak.py`

**Tujuan:** Mengkonversi unified JSON ke HTML dengan layout yang
menyerupai laporan keuangan baku PSAK 201.

#### Fitur layout

| Elemen | Style |
|---|---|
| Judul laporan | Centered, bold, font Times |
| Periode + satuan | Centered, italic |
| Section header | UPPERCASE, bold (ASET, LIABILITAS, EKUITAS) |
| Sub-group | Bold, regular case |
| Item | Indent berdasarkan level (16px per level) |
| Total / Subtotal | Bold + garis horizontal di atas |
| Grand total | Bold + garis ganda di bawah |
| Angka | Format Indonesia: `1.362.054.731`, negatif `(7)` |

#### Cara kerja

```python
def render(data):
    for stmt in data["statements"]:
        for section, nodes in stmt["sections"].items():
            # Tulis section header
            for node in nodes:
                render_node(node, depth=1)

def render_node(node, depth):
    cls = _row_class(node, depth)  # "total" | "subgroup" | ""
    # Format nilai dengan _fmt()
    # Tulis baris HTML dengan padding sesuai depth
    for child in node["children"]:
        render_node(child, depth + 1)
```

### 6.2 `tools/make_figures.py`

Generator grafik matplotlib untuk thesis Bab 4. Menghasilkan 3 PNG:

1. **Gambar 4.3 Cell F1 vs Node F1** — scatter plot trade-off
   akurasi-struktur vs akurasi-nilai
2. **Gambar 4.4 Runtime vs Node F1** — scatter plot log-x speed vs accuracy
3. **Gambar 4.5 Top-4 metrics** — bar chart 3 metrik untuk 4 pendekatan
   teratas

Input: `runs/<pdf_stem>/comparison.json`
Output: `runs/<pdf_stem>/figures/*.png`

### 6.3 `app.py` + `templates/`

**Web UI Flask** untuk demo interaktif.

#### Route

| Route | Method | Fungsi |
|---|---|---|
| `/` | GET | Halaman utama: form upload + approach selector |
| `/run` | POST | Jalankan approach yang dipilih, render hasil |
| `/api/run` | POST | JSON API headless untuk script |

#### Alur user

```
1. Buka /
2. Pilih: upload PDF sendiri ATAU klik "Demo BBNI 2025"
3. Centang approach yang ingin dijalankan
4. Klik "Jalankan Ekstraksi"
5. POST /run → app.py:
    a. Validasi input
    b. Loop: pilih approach → run → simpan pred
    c. Kalau filename mengandung "BBNI" → evaluate vs GT
    d. Render result.html
6. Result page:
    a. Tabel komparasi metrik di atas
    b. Tab per-approach: PSAK view (iframe) + raw JSON
```

#### Template structure

- `index.html`: form upload + grid approach cards dengan metadata
- `result.html`: comparison table + tabs + per-statement breakdown

---

## Bab 7. Ground Truth

### 7.1 `ground_truth/build_bbni_2025.py`

**Tujuan:** Skrip Python yang membangun GT JSON dari transkripsi manual
yang auditable.

#### Cara kerja

Helper functions untuk konstruksi simpul:

```python
def L0(label, en, v25=None, v24=None, children=None): ...
def L1(label, en, v25=None, v24=None, children=None): ...
def L2(label, en, v25=None, v24=None, children=None): ...
def L3(label, en, v25=None, v24=None, children=None): ...

ASSETS = [
    L0("Kas", "Cash", 13_352_065, 13_709_930),
    L0("Giro pada bank lain", "Current accounts with other banks",
       children=[
        L1("Giro pada bank lain pihak ketiga",
           "Current accounts ... third parties",
           25_858_094, 22_031_212),
        L1("Giro pada bank lain pihak berelasi", ...),
        L1("Cadangan kerugian penurunan nilai ...", -7, -13),
    ]),
    ...
]
```

Eksekusi `python ground_truth/build_bbni_2025.py` menghasilkan
`bbni_2025.json` yang valid sesuai skema unified.

**Statistik:**
- 339 baris total
- 114 baris dengan nilai
- Kedalaman maksimum: 4 level (Ekuitas → Saldo laba → Saldo laba yang
  telah ditentukan → Cadangan umum dan wajib)
- Validasi: Total Aset = Total Liab + Dana Syirkah + Ekuitas ✓

---

## Bab 8. Flow End-to-End

### 8.1 Jalankan approach `native_pdf` saja

```bash
python -c "
from approaches.native_pdf import run
pred, runtime = run('docs/FinancialStatement-2025-Tahunan-BBNI.pdf')
print(f'Runtime: {runtime:.2f}s')
print(f'Statements: {len(pred[\"statements\"])}')
"
```

### 8.2 Jalankan 8-way comparison

```bash
python ground_truth/run_all.py docs/FinancialStatement-2025-Tahunan-BBNI.pdf
```

Internal:

```
1. run_all.py muat ground_truth/bbni_2025.json
2. Import approaches → registry filled
3. Untuk tiap approach in registry:
    pred, runtime = REGISTRY[name](pdf_path)
    simpan ke runs/.../pred_<name>.json
4. Untuk tiap pred:
    report = evaluate(gold, pred)
    simpan ke runs/.../report_<name>.json
    tambahkan ke comparison table
5. Tulis comparison.json + comparison.md
```

### 8.3 Render PSAK HTML

```bash
python tools/render_psak.py runs/.../pred_native_pdf.json
# → output: runs/.../pred_native_pdf.html
```

### 8.4 Demo web

```bash
python app.py
# Buka http://localhost:5001
```

---

## Bab 9. Cara Menambah Approach Baru

### 9.1 Buat berkas baru di `approaches/`

```python
# approaches/my_new_approach.py
"""Penjelasan singkat tentang pendekatan."""
from __future__ import annotations

import time
from idx_fin_parser.unified import build_unified_output
from . import register

def run(pdf_path: str) -> tuple[dict, float]:
    t0 = time.perf_counter()

    # ── Lakukan ekstraksi di sini ──
    # ... custom logic ...

    # Bangun output yang valid sesuai skema unified
    out = build_unified_output(
        source_pdf=str(pdf_path),
        approach="my_new_approach",
        statements=[
            {
                "type": "financial_position",
                "years": [2025, 2024],
                "pages": [4, 5, 6, ...],
                "sections": {
                    "assets": [
                        # daftar Node...
                    ],
                    ...
                },
            },
            # ... profit_or_loss ...
        ],
        meta={"custom_param": "value"},
    )

    runtime = time.perf_counter() - t0
    return out, runtime

register("my_new_approach", run)
```

### 9.2 Daftarkan di `approaches/__init__.py`

```python
try:
    from . import my_new_approach  # noqa
except ImportError as exc:
    print(f"[approaches] my_new_approach unavailable: {exc}")
```

### 9.3 Test

```bash
python ground_truth/run_all.py docs/...pdf --only my_new_approach
```

---

## Bab 10. Cara Menambah PDF / Ground Truth Baru

### 10.1 Tambahkan PDF ke `docs/`

```bash
cp /path/to/new_emiten.pdf docs/
```

### 10.2 Bangun ground truth

Buat berkas baru `ground_truth/build_<emiten>_<tahun>.py` dengan pola
yang sama dengan `build_bbni_2025.py`. Transkripsi manual dari PDF.

Eksekusi untuk menghasilkan JSON:

```bash
python ground_truth/build_<emiten>_<tahun>.py
# → menghasilkan ground_truth/<emiten>_<tahun>.json
```

### 10.3 Jalankan komparasi

```bash
python ground_truth/run_all.py docs/<emiten>.pdf \
    --gold ground_truth/<emiten>_<tahun>.json
```

---

## Bab 11. Troubleshooting Kode

| Masalah | Penyebab | Solusi |
|---|---|---|
| Node F1 rendah pada PDF baru | Section header berbeda dari BBNI | Tambahkan ke `_SECTION_EXACT` di `utils.py` |
| EN bleed pada PDF baru | Layout kolom bilingual berbeda | Cek `en_x_threshold` per halaman; mungkin perlu tuning gap-based threshold |
| Hierarki kacau | `assign_levels` tidak detect cluster x0 | Cek distribusi `x0_first`; mungkin tuning `clustering tolerance` (default 8pt) |
| VLM Cell F1 = 0 | Format key tahun berbeda | `flat_rows_to_tree` sudah substring-match; cek format `columns` di raw response |
| OCR sangat lambat | Page detection ikut OCR | Pastikan `_find_page_index` pakai `use_ocr=True` (text fallback), bukan `force_ocr=True` |
| `pdfplumber_tables` MAE catastrophic | Cell merge antar baris | Tuning `table_settings` (snap/join tolerance) atau gunakan pendekatan koordinat |

---

## Bab 12. Reproduksi Hasil Thesis

```bash
# 1. Pastikan setup lengkap (Tesseract + .env dengan OPENAI_API_KEY)
# 2. Jalankan komparasi 8-way
python ground_truth/run_all.py docs/FinancialStatement-2025-Tahunan-BBNI.pdf

# 3. Generate figures Bab 4
python tools/make_figures.py

# 4. Verifikasi output
ls runs/FinancialStatement-2025-Tahunan-BBNI/
# - comparison.md          (Tabel 4.7)
# - comparison.json        (raw data)
# - pred_*.json            (8 prediksi)
# - report_*.json          (8 evaluasi)
# - figures/
#     fig_4_1_*.png        (Gambar 4.3)
#     fig_4_2_*.png        (Gambar 4.4)
#     fig_4_3_*.png        (Gambar 4.5)
```

---

## Bab 13. Daftar Berkas Penting (Quick Reference)

| Berkas | Peran |
|---|---|
| `idx_fin_parser/unified.py` | Definisi skema + adapter |
| `idx_fin_parser/utils.py` | Utilitas ekstraksi (kolom, level, kontinuasi) |
| `idx_fin_parser/pdf_statements.py` | Pipeline native + OCR |
| `approaches/__init__.py` | Registry + .env loader |
| `approaches/*.py` | 8 modul pendekatan |
| `ground_truth/build_bbni_2025.py` | Builder GT |
| `ground_truth/bbni_2025.json` | GT JSON (339 baris) |
| `ground_truth/evaluate.py` | 5 metrik evaluasi |
| `ground_truth/run_all.py` | Orkestrator 8-way |
| `tools/render_psak.py` | Renderer HTML PSAK |
| `tools/make_figures.py` | Generator grafik |
| `app.py` | Flask web UI |
| `templates/index.html` | Halaman upload |
| `templates/result.html` | Halaman hasil |

---

*Dokumen ini ditujukan sebagai panduan teknis lengkap. Untuk pertanyaan
spesifik tentang algoritma atau implementasi, rujuk komentar di kode
sumber di berkas terkait.*
