# Penjelasan Logika Sistem Ekstraksi Laporan Keuangan

Dokumen ini menjelaskan **cara berpikir** di balik sistem ekstraksi PDF
laporan keuangan IDX menjadi data tabular terstruktur. Ditujukan sebagai
bahan presentasi ke dosen pembimbing — fokus pada intuisi dan logika,
bukan implementasi kode.

---

## 1. Masalah Fundamental: PDF Bukan Tabel

### 1.1 PDF tidak menyimpan struktur tabel

Ketika kita melihat laporan keuangan di layar, kita melihat **tabel**
dengan baris-baris akun, kolom-kolom tahun, indentasi yang menandakan
hierarki. Tapi yang sebenarnya disimpan di dalam berkas PDF **bukan**
tabel — yang disimpan adalah serangkaian instruksi:

> "Tulis kata 'Kas' di koordinat (x=94, y=358)"
> "Tulis kata '13.352.065' di koordinat (x=370, y=358)"
> "Tulis kata 'Cash' di koordinat (x=480, y=358)"
> "Tulis kata 'Giro' di koordinat (x=94, y=380)"
> ...

Tidak ada informasi yang mengatakan "ini sel pada baris X kolom Y".
Tidak ada informasi tentang hierarki induk-anak. Yang ada hanya **kata
+ posisi**.

### 1.2 Pekerjaan kita: rekonstruksi struktur

Tugas sistem ekstraksi adalah **merekonstruksi** struktur tabel hierarkis
dari sekumpulan "kata + posisi" yang tidak terstruktur ini.

Ini analog dengan pekerjaan mata manusia: ketika kita membaca laporan
keuangan, otak kita **menginterpretasi** posisi spasial menjadi makna —
"kata yang sejajar secara vertikal dengan toleransi tertentu pasti
satu baris", "kata yang lebih ke kanan adalah kolom angka", "label yang
lebih indent ke dalam adalah sub-item dari label di atasnya".

Yang akan kita bahas: **bagaimana algoritma menirukan interpretasi
visual ini secara otomatis.**

---

## 2. Mengapa PDF Laporan Keuangan IDX Khususnya Sulit

Karakteristik khas dokumen IDX yang menambah kompleksitas:

### 2.1 Bilingual — Indonesia di kiri, Inggris di kanan

Setiap akun ditulis dalam dua bahasa: kolom kiri Indonesia, kolom kanan
Inggris. Algoritma harus tahu di mana batas kolom ini agar tidak
mencampurkan keduanya.

```
Kas                    13.352.065    13.709.930    Cash
Giro pada bank lain                                Current accounts with
                                                   other banks
  Pihak ketiga          25.858.094    22.031.212    Third parties
```

### 2.2 Hierarki implisit lewat indentasi

Tidak ada penanda eksplisit ("ini induk", "ini anak"). Yang ada hanya
**indentasi visual** — semakin dalam indent, semakin dalam levelnya:

```
Aset                                  ← level 0 (section)
  Kas                  13.352.065     ← level 1 (item)
  Giro pada bank lain                 ← level 1 (group)
    Pihak ketiga       25.858.094     ← level 2 (item)
    Pihak berelasi     79.999         ← level 2 (item)
  Penempatan ...                      ← level 1
```

### 2.3 Tidak ada garis pembatas tabel

Berbeda dengan tabel di Excel atau Word, PDF IDX **tidak memiliki garis
pembatas sel** yang bisa dideteksi. Library seperti `camelot` yang
mengandalkan garis pembatas akan gagal di sini.

### 2.4 Banyak template item kosong

Karena IDX menggunakan template XBRL yang seragam untuk semua sektor,
satu PDF biasanya berisi banyak baris kosong (mis. emiten perbankan
konvensional tidak akan mengisi item-item asuransi syariah). Sistem
harus tetap mengekstrak baris-baris kosong ini supaya schema tetap
konsisten antar PDF.

### 2.5 Label yang panjang dan terpotong ke baris berikutnya

Label akun bisa sangat panjang sehingga terpotong ke baris berikutnya.
Algoritma harus tahu kapan dua baris fisik sebenarnya satu baris logis.

```
Cadangan kerugian
penurunan nilai pada      (7)         (13)
giro pada bank lain
```

Tiga baris fisik di atas sebenarnya satu baris logis: "Cadangan
kerugian penurunan nilai pada giro pada bank lain".

---

## 3. Empat Cara Berpikir untuk Memecahkan Masalah Ini

Kami mencoba **empat keluarga pendekatan** yang berbeda dalam cara
mereka "membaca" PDF:

### Keluarga A. Rule-based Berbasis Koordinat Fisik

**Intuisi:** Manfaatkan posisi (x, y) tiap kata yang sudah disediakan
oleh PDF. Inilah informasi paling kaya yang kita punya — gunakan
sepenuhnya.

**Cara berpikir:** Seperti seseorang yang sangat detail melihat dokumen
dengan penggaris. Mengukur jarak antar kata, mengukur indentasi,
mengukur kolom.

### Keluarga B. Rule-based Generik (Library Pihak Ketiga)

**Intuisi:** Pakai library yang sudah jadi (`pdfplumber.extract_tables()`,
`camelot-py`). Pendekatan generik untuk berbagai jenis tabel di dunia
PDF.

**Cara berpikir:** Seperti seseorang yang hanya tahu "ini ada tabel
di dokumen, coba ekstrak". Tidak tahu konteks khusus laporan
keuangan IDX.

### Keluarga C. OCR (Optical Character Recognition)

**Intuisi:** Anggap PDF sebagai **gambar** — abaikan text layer-nya.
Lalu baca teks dari gambar dengan OCR engine (Tesseract).

**Cara berpikir:** Seperti seseorang yang membaca **fotocopy** dokumen
(tidak punya file digital). Bisa membaca huruf tapi kehilangan
informasi koordinat yang akurat.

### Keluarga D. Vision-Language Model (VLM)

**Intuisi:** Berikan gambar halaman ke model AI yang sudah dilatih
memahami struktur dokumen, dan minta dia menulis struktur tabel dalam
format JSON.

**Cara berpikir:** Seperti minta bantuan asisten ahli yang sudah pernah
melihat ribuan laporan keuangan, lalu tinggal disuruh "ekstrak tabel
ini ke JSON".

---

## 4. Detail Cara Kerja Tiap Pendekatan

### 4.1 Native PDF Parser (Custom, Berbasis Koordinat) — Pendekatan Utama

Ini pendekatan inti yang kami kembangkan secara custom. Tujuh tahap:

#### Tahap 1: Temukan Halaman Target

PDF tahunan IDX punya banyak halaman (BBNI 40 halaman) tapi yang kita
butuhkan hanya **Laporan Posisi Keuangan** dan **Laporan Laba Rugi**.

Kita cari halaman awal dengan kata kunci ("statement of financial
position" / "laporan posisi keuangan") dan halaman akhir dengan kata
kunci ("statement of profit or loss" / "laporan laba rugi").

#### Tahap 2: Ekstrak Kata Beserta Koordinat

Untuk tiap halaman target, kita panggil fungsi yang mengembalikan
daftar kata seperti:

```
[{"text": "Kas",        "x0": 94,  "top": 358},
 {"text": "13,352,065", "x0": 370, "top": 358},
 {"text": "Cash",       "x0": 480, "top": 358},
 {"text": "Giro",       "x0": 94,  "top": 380},
 ...]
```

#### Tahap 3: Kelompokkan ke Baris Visual

Kata-kata dengan koordinat `top` yang sama (dalam toleransi 3 poin)
adalah satu baris visual:

```
y=358: ["Kas", "13,352,065", "13,709,930", "Cash"]
y=380: ["Giro", "pada", "bank", "lain", "Current", "accounts", ...]
```

#### Tahap 4: Identifikasi Kolom

Untuk tiap baris, deteksi token angka. Kolom ID = sebelum angka,
kolom Amounts = di antara angka pertama dan terakhir, kolom EN = setelah
angka terakhir.

```
"Kas"           ←  ID
"13,352,065"    ←  Amount 2025
"13,709,930"    ←  Amount 2024
"Cash"          ←  EN
```

**Masalah khusus** untuk baris **tanpa angka** (mis. header "Giro pada
bank lain"): bagaimana kita tahu di mana batas kolom Indonesia vs
Inggris?

**Solusi: deteksi celah terbesar**

Kita ukur jarak horizontal antar kata berurutan. Celah terbesar yang
**signifikan lebih besar** dari celah-celah lain adalah batas kolom.

```
"Giro pada bank lain          Current accounts with other banks"
     ↑                  ↑↑↑                                       
     gap normal        celah besar (= batas kolom)
```

#### Tahap 5: Deteksi Level Hierarki dari Indentasi

Untuk setiap baris, ambil posisi `x` paling kiri dari kata pertama
(`x0_first`). Ini menunjukkan seberapa dalam indentasi.

Pengelompokan (clustering): baris dengan `x0` yang serupa (dalam
toleransi 8 poin) → level yang sama. Cluster terkiri = level 0,
berikutnya = level 1, dst.

```
x0=49   →  level 0 (mis. "Aset")
x0=79   →  level 1 (mis. "Kas")
x0=94   →  level 2 (mis. "Giro pihak ketiga")
x0=109  →  level 3 (mis. "Cadangan umum")
```

#### Tahap 6: Bangun Pohon Hierarkis

Sekarang kita punya daftar baris dengan level. Kita ubah jadi pohon
dengan algoritma "stack-based attach":

```
Bayangkan kita tumpuk simpul ke stack berdasarkan level.

Saat baris baru datang dengan level k:
  1. Pop semua simpul di stack dengan level >= k
     (artinya: tutup semua sub-tree yang lebih dalam)
  2. Lampirkan baris baru sebagai anak dari simpul teratas stack
  3. Push baris baru ke stack
```

Contoh visual:

```
Baris masuk: "Aset" (level 0)
  Stack: []  →  Empty, jadikan root, push.  Stack: [Aset]

Baris masuk: "Kas" (level 1)
  Stack top = Aset (level 0) < 1, attach Kas ke Aset.
  Stack: [Aset, Kas]

Baris masuk: "Giro" (level 1)
  Stack top = Kas (level 1) >= 1, pop Kas.
  Stack top = Aset (level 0) < 1, attach Giro ke Aset.
  Stack: [Aset, Giro]

Baris masuk: "Pihak ketiga" (level 2)
  Stack top = Giro (level 1) < 2, attach Pihak ketiga ke Giro.
  Stack: [Aset, Giro, Pihak ketiga]
```

Hasilnya:
```
Aset
├── Kas
└── Giro
    └── Pihak ketiga
```

#### Tahap 7: Gabungkan Baris Kontinuasi

Label panjang yang terpotong ke baris berikutnya harus digabung.
Tiga heuristik:

1. **Label baris saat ini kosong** — yang ada hanya teks Inggris yang
   wrap. Gabungkan ke baris sebelumnya.

2. **Label baris saat ini diawali huruf kecil** — kemungkinan suffix
   dari kata sebelumnya. Mis. "berkelanjutan" sebagai sambungan
   "pembiayaan yang".

3. **Kata terakhir baris sebelumnya = kata penghubung** (`bank`,
   `pada`, `dan`, `dengan`) DAN posisi `x0` baris saat ini sama dengan
   sebelumnya. Mis. "Penempatan pada Bank" + "Indonesia dan bank lain"
   = "Penempatan pada Bank Indonesia dan bank lain".

---

### 4.2 PyMuPDF Native — Ablasi Library

**Intuisi:** Algoritma sama persis dengan Native PDF di atas, tapi
pustaka yang dipakai untuk ekstrak koordinat berbeda (PyMuPDF
menggantikan pdfplumber).

**Apa yang ingin kita ukur:** Berapa banyak performa berasal dari
**algoritma** vs **pilihan library**?

**Hasil:** Akurasi sama (Node F1 0.719 vs 0.720), tapi **37× lebih
cepat** (0.1 detik vs 3.9 detik). Pelajaran: untuk skala produksi,
ganti backend dari pdfplumber ke PyMuPDF.

---

### 4.3 Baseline Regex — Floor Baseline

**Intuisi:** Bayangkan kita ekstraksi paling naif yang bisa dibayangkan:
ambil teks, cari pola angka dengan regex, label = teks sebelum angka.

**Kenapa kita lakukan ini:** Untuk thesis, kita butuh **floor** —
tolak ukur "se-buruk apa kalau kita tidak melakukan effort sama sekali".
Ini membenarkan: "lihat, custom algorithm meningkatkan akurasi 70×
dibanding regex naif".

**Karakteristik:**
- Tidak ada deteksi hierarki — semua simpul level 0
- Tidak ada deteksi kolom bilingual — label sering bercampur dengan
  teks Inggris
- Tidak ada deteksi continuation — label terpotong tetap terpisah

---

### 4.4 pdfplumber Tables — Library Default

**Intuisi:** "Kan ada fungsi `extract_tables()` di pdfplumber, kenapa
nggak pakai itu saja?"

**Cara kerja:** Library mendeteksi struktur tabel berdasarkan posisi
teks. Asumsi internal: tabel punya garis pembatas atau ruang kosong
yang konsisten antar sel.

**Mengapa gagal di IDX:** Karena PDF IDX tidak punya garis pembatas
eksplisit, library kadang **menggabungkan dua sel berurutan secara
vertikal** menjadi satu — sehingga nilai 660.513.890 dan 616.469.089
ter-extract sebagai satu nilai raksasa "660513890616469089". Ini
disebut **failure mode catastrophic** karena nilai jadi sangat-sangat
salah tanpa peringatan.

---

### 4.5 Camelot Stream — Third-party Tool

**Intuisi:** "Mungkin tool spesialis tabel akan lebih baik."

**Cara kerja:** Camelot adalah pustaka khusus ekstraksi tabel PDF
yang populer di komunitas data finance. Dua mode:
- **Lattice**: butuh garis pembatas visible
- **Stream**: berbasis pengelompokan ruang kosong

Karena tidak ada garis pembatas di IDX, kita pakai mode Stream.

**Mengapa gagal:** Sama dengan pdfplumber_tables — algoritma generik
tidak memahami pola spesifik layout IDX bilingual borderless.

---

### 4.6 OCR Penuh (Tesseract)

**Intuisi:** "Bagaimana kalau kita lupakan text layer, anggap PDF
sebagai gambar, dan baca dengan OCR engine?"

**Kenapa kita lakukan ini:** Untuk menyimulasikan kondisi PDF
**scan** — banyak laporan keuangan lama atau yang di-scan tidak punya
text layer. Pendekatan native gagal total di situ; OCR adalah satu-
satunya pilihan.

**Cara kerja:**
1. Tiap halaman target di-rasterize jadi gambar 250 DPI
2. Gambar diserahkan ke Tesseract engine dengan paket bahasa
   `ind+eng`
3. Tesseract baca tulisan tapi **kehilangan informasi koordinat akurat**
4. Hasilnya cuma string teks — kita parsing dengan logika berbasis baris

**Kelebihan:** Bisa handle PDF scan
**Kekurangan:**
- Lambat (~60 detik per PDF, karena OCR per halaman)
- Tesseract membaca text **line-by-line**, kehilangan boundary kolom
- Row attribution kacau: angka kadang ter-asosiasi ke label salah

**Insight menarik:** Pada baris yang berhasil ter-extract, **nilai
numeriknya benar persis** (Tesseract baca digit dengan akurat). Yang
gagal adalah **struktur baris** — angka tidak ter-asosiasi ke label
yang tepat.

---

### 4.7 VLM (GPT-4o-mini dan GPT-4o)

**Intuisi:** "Bagaimana kalau kita berikan gambar tabel ke AI yang
sudah dilatih memahami dokumen, dan tinggal minta dia menulis JSON
struktur?"

**Cara kerja:**

1. Tiap halaman target di-rasterize jadi gambar
2. Gambar di-encode base64 dan dikirim ke API OpenAI
3. Kita beri **system prompt** yang sangat ketat:

   > "Kamu adalah ahli ekstraksi struktur dokumen. Ekstrak tabel
   > keuangan dari gambar ini ke JSON dengan schema berikut: ..."

4. Kita beri **user prompt** yang mengarahkan model:

   > "Perhatikan indentasi untuk menentukan level hierarki (0 = section,
   > 1 = sub-item, dst.). Pertahankan urutan baris persis seperti di
   > gambar. Jangan jelaskan apa-apa, hanya output JSON."

5. Model membaca gambar + prompt, lalu mengembalikan JSON yang
   (mudah-mudahan) berisi struktur tabel
6. JSON dari semua halaman digabung lalu dibangun pohon dengan
   informasi level yang sudah disediakan model

**Hyperparameter penting:**
- `temperature = 0` agar deterministik sebanyak mungkin
- `max_tokens = 4096` cukup untuk ~30 baris per halaman

**Dua varian:**
- `gpt-4o-mini`: lebih murah (~$0.03 per PDF), akurasi sedang
- `gpt-4o`: lebih mahal (~$0.50 per PDF), akurasi tinggi

**Insight unik VLM:**
- **Memahami struktur dengan sangat baik** — Node F1 GPT-4o = 0.734,
  sedikit melebihi native_pdf
- **Tapi sering salah di nilai** — Cell F1 hanya 0.32
- **Failure mode khas**: VLM kadang menghalusinasi nilai `0` untuk
  template item yang kosong (BBNI tidak mengisi item asuransi
  syariah, tapi VLM "merasa harus mengisi sesuatu" jadi mengisi `0`)

---

## 5. Kunci Penting: Bagaimana Hierarki Dibangun

Ini bagian paling tricky di seluruh sistem. Ada **dua strategi**
yang berbeda fundamental:

### Strategi 1 — Berbasis Koordinat (digunakan Native PDF)

Kita **ukur** posisi indentasi tiap baris, lalu **cluster** ke
level diskrit. Pohon dibangun dengan mempertahankan invariant:
simpul level k jadi anak dari simpul terdekat di atas dengan
level < k.

Ini **deterministik** dan setia pada tata letak fisik dokumen.

### Strategi 2 — Berbasis Output Model (digunakan VLM)

Model AI **langsung memberi tahu kita** level tiap baris di outputnya.
Kita tinggal percaya dan bangun pohon.

Ini **fleksibel** terhadap variasi layout tapi tergantung kualitas
model.

**Keduanya menghasilkan pohon yang sama strukturnya** — yang berbeda
adalah cara mendapatkan informasi level itu.

---

## 6. Kontrak Data: Mengapa Komparasi Adil Mungkin

Inti dari sistem ini adalah **skema data terpadu**: semua pendekatan
ekstraksi menghasilkan JSON dengan struktur yang sama.

```
{
  "source_pdf": ...,
  "approach":   ...,
  "statements": [
    {
      "type":     "financial_position",
      "years":    [2025, 2024],
      "pages":    [4, 5, ...],
      "sections": {
        "assets": [
          {
            "level":     0,
            "label":     "Kas",
            "label_en":  "Cash",
            "values":    {"2025-12-31": 13352065, "2024-12-31": 13709930},
            "children":  [...]
          },
          ...
        ]
      }
    }
  ]
}
```

**Manfaat skema terpadu:**

1. **Komparasi adil** — kita bisa bandingkan output 8 pendekatan
   secara *apple-to-apple*
2. **Reusability** — modul evaluasi dan penyajian tidak peduli
   bagaimana JSON dihasilkan, hanya peduli formatnya
3. **Extensibility** — tambah pendekatan baru tinggal pastikan
   output-nya mengikuti skema

---

## 7. Bagaimana Kita Mengukur Akurasi: 5 Metrik

Setelah ekstraksi selesai, kita bandingkan dengan **ground truth**
(jawaban benar yang ditranskrip manual dari PDF).

### 7.1 Node F1 — Berapa Baris yang Berhasil Dideteksi?

**Pertanyaan:** Dari semua baris di ground truth, berapa yang
berhasil ditemukan oleh pendekatan, di posisi hierarki yang benar?

Kita bandingkan setiap baris berdasarkan **(seksi, jalur normalisasi
label)**, contoh:

```
("assets", "/giro pada bank lain/giro pada bank lain pihak ketiga")
```

Hitung:
- **Precision** = baris yang benar / total baris yang di-predict
- **Recall** = baris yang benar / total baris di ground truth
- **F1** = rata-rata harmonik keduanya (0 = tidak ada match, 1 = sempurna)

### 7.2 Cell F1 — Berapa Nilai Numerik yang Benar?

**Pertanyaan:** Pada baris-baris yang berhasil dicocokkan, berapa
% nilai numerik per (baris, tahun) yang benar persis?

Ini **conditional** pada Node F1 — kalau baris tidak ter-deteksi,
nilai-nya tidak bisa dievaluasi.

**Insight penting:** Cell F1 tinggi BUKAN berarti pendekatan bagus
secara keseluruhan. Bisa jadi pendekatan cuma berhasil deteksi
sedikit baris, tapi di baris yang sedikit itu nilainya benar.

### 7.3 Cell MAE — Seberapa Jauh Salahnya?

**Pertanyaan:** Pada sel yang nilainya salah, seberapa jauh selisihnya?

MAE = 0 berarti "kalau pendekatan mengembalikan nilai, nilainya selalu
benar persis". Kalau salah, biasanya salah karena nilai-nya `null`,
bukan nilai berbeda.

**Failure mode unik:** pdfplumber_tables punya MAE = 1.5 juta —
artinya ada sel yang nilainya beda **jutaan**. Ini karena library
kadang menggabungkan 2 nilai berurutan jadi 1 nilai super besar.

### 7.4 Hierarchy Accuracy — Apakah Induk Benar?

**Pertanyaan:** Di antara label yang sama-sama muncul di gold dan
predict, berapa % yang punya **induk** (parent path) yang benar?

Ini menangkap kasus: label benar tapi ditempatkan di seksi yang salah
(mis. item aset ditempatkan di seksi liabilitas).

### 7.5 Label Similarity — Apakah Teks Label Mirip?

**Pertanyaan:** Berapa kemiripan tekstual antar label yang
dicocokkan (Levenshtein ratio)?

Berguna terutama untuk VLM yang bisa melakukan **paraphrase** —
mengeluarkan "Cash and equivalents" padahal ground truth "Cash".
Metrik ini memberi toleransi terhadap variasi minor.

---

## 8. Mengapa Tidak Ada "Pemenang Mutlak"

Hasil komparasi 8-way menunjukkan trade-off yang menarik:

| Approach | Node F1 (struktur) | Cell F1 (nilai) | Runtime |
|---|---:|---:|---:|
| native_pdf (custom) | 0.720 | **0.996** | 3.9 s |
| pymupdf_native (custom + fast) | 0.719 | 0.991 | **0.1 s** |
| GPT-4o (VLM mahal) | **0.734** | 0.320 | 126 s |
| GPT-4o-mini (VLM murah) | 0.525 | 0.354 | 336 s |
| Yang lain | ≤ 0.04 | bervariasi | bervariasi |

**Insight:**

1. **VLM menang tipis di struktur** tapi **kalah jauh di nilai** —
   VLM bisa memahami layout tabel dengan baik, tapi sering salah baca
   atau halusinasi angka.

2. **Custom rule-based menang di nilai** — karena ekstraksi koordinat
   bersifat deterministik, tidak ada loss informasi.

3. **Speed-vs-accuracy trade-off di VLM** menarik: GPT-4o lebih
   cepat dari mini meskipun model lebih besar (anti-intuisi).

4. **Library generik tidak adekuat** untuk layout bilingual
   borderless IDX — Node F1 < 0.05.

5. **OCR pure** bagus untuk situasi PDF scan, tapi tidak compete untuk
   PDF native dengan text layer yang utuh.

**Kesimpulan praktis:**
- Untuk produksi pada PDF native IDX: gunakan `pymupdf_native`
  (akurasi tinggi + cepat + gratis)
- Untuk PDF scan: gunakan OCR sebagai fallback
- Untuk PDF dengan layout tidak standar: VLM bisa membantu, dengan
  validasi nilai sebagai post-processing

---

## 9. Kontribusi yang Dapat Diklaim

Dari kerja ini, kontribusi yang dapat dipertanggungjawabkan:

### 9.1 Metodologis

- **Skema data terpadu** yang memungkinkan komparasi 8 pendekatan
  ekstraksi PDF dengan paradigma yang sangat berbeda secara adil
- **Algoritma rule-based custom** yang menggabungkan deteksi
  koordinat, *gap-based bilingual split*, clustering level, dan
  *stack-based tree construction*

### 9.2 Empiris

- **Benchmark BBNI 2025** dengan ground truth 339 baris yang
  divalidasi via cross-check identitas akuntansi
- **Trade-off Pareto** ter-quantify antar 8 pendekatan
- **Identifikasi failure mode** spesifik per pendekatan (EN-bleed,
  cell-merge catastrophic, hallucinasi nilai VLM, dll.)

### 9.3 Praktis

- **Sistem prototype** lengkap dengan web UI, CLI, renderer PSAK,
  dan visualisasi
- **Reproducible** — semua hasil dapat di-regenerate dengan satu
  perintah dari repo

---

## 10. Limitasi yang Perlu Disampaikan

Sebagai konteks untuk dosen:

1. **Ukuran sampel kecil** — 1 PDF (BBNI 2025). Klaim statistik
   lintas-emiten/sektor tidak dapat diberikan. Saran: minimal 10
   PDF dari 3 sektor.

2. **Section detector dependency** — perbaikan section detector
   menguntungkan pendekatan dengan ekstraksi bersih (native) dan
   merugikan yang noisy (OCR). Potensi bias evaluasi yang perlu
   di-disclose.

3. **VLM non-deterministik** — meskipun `temperature=0`, satu PDF
   diuji dalam satu sesi. Klaim ilmiah lebih kuat dengan 3 run.

4. **Ground truth manual oleh penulis** — tidak ada peer review per
   baris. Validasi hanya berbasis identitas akuntansi total.

---

## Lampiran: Contoh Visual Output

### Sebelum (PDF asli)

```
Aset                                                Assets
Kas                              13.352.065   13.709.930   Cash
Dana yang dibatasi penggunaannya                            Restricted funds
Giro pada Bank Indonesia         79.989.122   51.669.054   Current accounts
                                                            with Bank Indonesia
Giro pada bank lain                                         Current accounts
                                                            with other banks
  Giro pada bank lain                                       Current accounts
  pihak ketiga                   25.858.094   22.031.212   with other banks
                                                            third parties
  ...
Jumlah aset                   1.362.054.731 1.130.128.862  Total assets
```

### Setelah Ekstraksi (JSON unified)

```json
{
  "type": "financial_position",
  "years": [2025, 2024],
  "sections": {
    "assets": [
      {
        "level": 0,
        "label": "Kas",
        "label_en": "Cash",
        "row_type": "item",
        "values": {"2025-12-31": 13352065, "2024-12-31": 13709930},
        "children": []
      },
      {
        "level": 0,
        "label": "Giro pada bank lain",
        "label_en": "Current accounts with other banks",
        "row_type": "group",
        "values": {"2025-12-31": null, "2024-12-31": null},
        "children": [
          {
            "level": 1,
            "label": "Giro pada bank lain pihak ketiga",
            "values": {"2025-12-31": 25858094, "2024-12-31": 22031212},
            ...
          },
          ...
        ]
      },
      ...,
      {
        "level": 0,
        "label": "Jumlah aset",
        "values": {"2025-12-31": 1362054731, "2024-12-31": 1130128862}
      }
    ]
  }
}
```

### Setelah Render PSAK (HTML)

Output yang menyerupai format buku PSAK 201:

```
                PT Bank Negara Indonesia (Persero) Tbk
                  Laporan Posisi Keuangan
            per 31 Desember 2025 dan 31 Desember 2024
                     (dalam jutaan rupiah)

                                              2025          2024
ASET
    Kas                                 13.352.065    13.709.930
    Dana yang dibatasi penggunaannya
    Giro pada Bank Indonesia            79.989.122    51.669.054
    Giro pada bank lain
        └ Giro pada bank lain
          pihak ketiga                  25.858.094    22.031.212
        └ Giro pada bank lain
          pihak berelasi                    79.999        42.974
        └ Cadangan kerugian penurunan
          nilai pada giro pada bank lain        (7)          (13)
    ...
    Jumlah aset                      1.362.054.731 1.130.128.862
    ─────────────────────────────────────────────────────────────
LIABILITAS
    ...
```

---

*Dokumen ini fokus pada logika konseptual sistem. Untuk detail
implementasi kode, lihat README.md dan dokumentasi inline di
masing-masing modul.*
