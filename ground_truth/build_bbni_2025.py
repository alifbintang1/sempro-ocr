"""Manually-transcribed ground truth for BBNI 2025 annual financial statements.

Source : docs/FinancialStatement-2025-Tahunan-BBNI.pdf, pages 4 – 19
Output : ground_truth/bbni_2025.json

Run    : python ground_truth/build_bbni_2025.py

This is the gold standard used by ground_truth/evaluate.py to score the three
extraction approaches (native_pdf, ocr_fallback, vlm) for the thesis.

Hierarchy convention (matches what the unified schema expects):
  level 0 = direct child of a section ("Aset", "Liabilitas", "Ekuitas", ...)
  level 1 = child of a level-0 group
  level 2 = child of a level-1 group
  level 3 = child of a level-2 group (only inside equity → retained earnings)

Empty rows (no value for either year) are kept as `label_only` nodes so that
extraction approaches that correctly produce them are rewarded.
"""
from __future__ import annotations

import json
from pathlib import Path


def N(label: str, label_en: str, level: int,
      v25: int | None = None, v24: int | None = None,
      children: list | None = None) -> dict:
    children = children or []
    values = {"2025-12-31": v25, "2024-12-31": v24}
    has_v = v25 is not None or v24 is not None
    if has_v:
        row_type = "item"
    elif children:
        row_type = "group"
    else:
        row_type = "label_only"
    return {
        "level": level,
        "label": label,
        "label_en": label_en,
        "row_type": row_type,
        "values": values,
        "children": children,
    }


def L0(lbl, en, v25=None, v24=None, children=None): return N(lbl, en, 0, v25, v24, children)
def L1(lbl, en, v25=None, v24=None, children=None): return N(lbl, en, 1, v25, v24, children)
def L2(lbl, en, v25=None, v24=None, children=None): return N(lbl, en, 2, v25, v24, children)
def L3(lbl, en, v25=None, v24=None, children=None): return N(lbl, en, 3, v25, v24, children)


# ════════════════════════════════════════════════════════════════════════
# FINANCIAL POSITION  (pages 4 – 12)
# ════════════════════════════════════════════════════════════════════════

ASSETS = [
    L0("Kas", "Cash", 13_352_065, 13_709_930),
    L0("Dana yang dibatasi penggunaannya", "Restricted funds"),
    L0("Giro pada Bank Indonesia", "Current accounts with Bank Indonesia",
       79_989_122, 51_669_054),

    L0("Giro pada bank lain", "Current accounts with other banks", children=[
        L1("Giro pada bank lain pihak ketiga",
           "Current accounts with other banks third parties",
           25_858_094, 22_031_212),
        L1("Giro pada bank lain pihak berelasi",
           "Current accounts with other banks related parties",
           79_999, 42_974),
        L1("Cadangan kerugian penurunan nilai pada giro pada bank lain",
           "Allowance for impairment losses for current accounts with other bank",
           -7, -13),
    ]),

    L0("Penempatan pada Bank Indonesia dan bank lain",
       "Placements with Bank Indonesia and other banks", children=[
        L1("Penempatan pada Bank Indonesia dan bank lain pihak ketiga",
           "Placements with Bank Indonesia and other banks third parties",
           31_725_849, 15_455_444),
        L1("Penempatan pada Bank Indonesia dan bank lain pihak berelasi",
           "Placements with Bank Indonesia and other banks related parties",
           2_054_144, 1_620_191),
        L1("Cadangan kerugian penurunan nilai pada penempatan pada bank lain",
           "Allowance for impairment losses for placements with other banks",
           -78, -194),
    ]),

    L0("Piutang asuransi", "Insurance receivables", children=[
        L1("Piutang asuransi pihak ketiga", "Insurance receivables third parties"),
        L1("Piutang asuransi pihak berelasi", "Insurance receivables related parties"),
        L1("Cadangan kerugian penurunan nilai pada piutang asuransi",
           "Allowance for impairment losses for insurance receivables"),
    ]),

    L0("Biaya akuisisi tangguhan", "Deferred acquisition costs"),
    L0("Deposito pada lembaga kliring dan penjaminan",
       "Deposits to clearing and settlement guarantee institution"),

    L0("Efek-efek yang diperdagangkan", "Marketable securities", children=[
        L1("Efek-efek yang diperdagangkan pihak ketiga",
           "Marketable securities third parties", 50_783_955, 38_376_931),
        L1("Efek-efek yang diperdagangkan pihak berelasi",
           "Marketable securities related parties", 12_232_740, 10_157_415),
        L1("Cadangan kerugian penurunan nilai pada efek-efek yang diperdagangkan",
           "Allowance for impairment losses for marketable securities", -15, -273),
    ]),

    L0("Investasi pemegang polis pada kontrak unit-linked",
       "Investments of policyholder in unit-linked contracts"),
    L0("Efek yang dibeli dengan janji dijual kembali",
       "Securities purchased under agreement to resale", 6_910_606, 7_971_923),

    L0("Wesel ekspor dan tagihan lainnya", "Bills and other receivables", children=[
        L1("Wesel ekspor dan tagihan lainnya pihak ketiga",
           "Bills and other receivables third parties", 8_181_871, 7_087_118),
        L1("Wesel ekspor dan tagihan lainnya pihak berelasi",
           "Bills and other receivables related parties", 5_338_937, 6_208_736),
        L1("Cadangan kerugian penurunan nilai pada wesel ekspor dan tagihan lainnya",
           "Allowance for impairment losses for bills and other receivables",
           -67_962, -52_828),
    ]),

    L0("Tagihan akseptasi", "Acceptance receivables", children=[
        L1("Tagihan akseptasi pihak ketiga",
           "Acceptance receivables third parties", 13_528_919, 13_193_510),
        L1("Tagihan akseptasi pihak berelasi",
           "Acceptance receivables related parties", 6_053_827, 2_825_260),
        L1("Cadangan kerugian penurunan nilai pada tagihan akseptasi",
           "Allowance for impairment losses for acceptance receivables",
           -133_986, -93_249),
    ]),

    L0("Tagihan derivatif", "Derivative receivables", children=[
        L1("Tagihan derivatif pihak ketiga",
           "Derivative receivables third parties", 4_654_266, 1_451_146),
        L1("Tagihan derivatif pihak berelasi",
           "Derivative receivables related parties", 768_083, 341_832),
    ]),

    L0("Pinjaman yang diberikan", "Loans", children=[
        L1("Pinjaman yang diberikan pihak ketiga",
           "Loans third parties", 660_513_890, 616_469_089),
        L1("Pinjaman yang diberikan pihak berelasi",
           "Loans related parties", 239_016_801, 159_402_689),
        L1("Cadangan kerugian penurunan nilai pada pinjaman yang diberikan",
           "Allowance for impairment losses for loans", -35_860_626, -38_684_520),
    ]),

    L0("Piutang dari lembaga kliring dan penjaminan",
       "Receivables from clearing and settlement guarantee institution"),

    L0("Piutang nasabah", "Receivables from customers", children=[
        L1("Piutang nasabah pihak ketiga",
           "Receivables from customers third parties"),
        L1("Piutang nasabah pihak berelasi",
           "Receivables from customers related parties"),
        L1("Cadangan kerugian penurunan nilai pada piutang nasabah",
           "Allowance for impairment losses for receivables from customers"),
    ]),

    L0("Piutang murabahah", "Murabahah receivables", children=[
        L1("Piutang murabahah pihak ketiga", "Murabahah receivables third parties"),
        L1("Piutang murabahah pihak berelasi", "Murabahah receivables related parties"),
        L1("Cadangan kerugian penurunan nilai pada piutang murabahah",
           "Allowance for impairment losses for murabahah receivables"),
    ]),

    L0("Piutang istishna", "Istishna receivables", children=[
        L1("Piutang istishna pihak ketiga", "Istishna receivables third parties"),
        L1("Piutang istishna pihak berelasi", "Istishna receivables related parties"),
        L1("Cadangan kerugian penurunan nilai pada piutang istishna",
           "Allowance for impairment losses for istishna receivables"),
    ]),

    L0("Piutang ijarah", "Ijarah receivables", children=[
        L1("Piutang ijarah pihak ketiga", "Ijarah receivables third parties"),
        L1("Piutang ijarah pihak berelasi", "Ijarah receivables related parties"),
        L1("Cadangan kerugian penurunan nilai pada piutang ijarah",
           "Allowance for impairment losses for ijarah receivables"),
    ]),

    L0("Piutang pembiayaan konsumen", "Consumer financing receivables", children=[
        L1("Piutang pembiayaan konsumen pihak ketiga",
           "Consumer financing receivables third parties"),
        L1("Piutang pembiayaan konsumen pihak berelasi",
           "Consumer financing receivables related parties"),
        L1("Cadangan kerugian penurunan nilai pada piutang pembiayaan konsumen",
           "Allowance for impairment losses for consumer financing receivables"),
    ]),

    L0("Pinjaman qardh", "Qardh funds", children=[
        L1("Pinjaman qardh pihak ketiga", "Qardh funds third parties"),
        L1("Pinjaman qardh pihak berelasi", "Qardh funds related parties"),
        L1("Cadangan kerugian penurunan nilai pada pinjaman qardh",
           "Allowance for impairment losses for qardh funds"),
    ]),

    L0("Pembiayaan mudharabah", "Mudharabah financing", children=[
        L1("Pembiayaan mudharabah pihak ketiga", "Mudharabah financing third parties"),
        L1("Pembiayaan mudharabah pihak berelasi", "Mudharabah financing related parties"),
        L1("Cadangan kerugian penurunan nilai pada pembiayaan mudharabah",
           "Allowance for impairment losses for mudharabah financing"),
    ]),

    L0("Pembiayaan musyarakah", "Musyarakah financing", children=[
        L1("Pembiayaan musyarakah pihak ketiga", "Musyarakah financing third parties"),
        L1("Pembiayaan musyarakah pihak berelasi", "Musyarakah financing related parties"),
        L1("Cadangan kerugian penurunan nilai pada pembiayaan musyarakah",
           "Allowance for impairment losses for musyarakah financing"),
    ]),

    L0("Investasi sewa", "Lease investments", children=[
        L1("Investasi sewa pihak ketiga", "Lease investments third parties"),
        L1("Investasi sewa pihak berelasi", "Lease investments related parties"),
        L1("Investasi sewa nilai residu yang terjamin",
           "Lease investments guaranteed residual value"),
        L1("Investasi sewa pendapatan pembiayaan tangguhan",
           "Lease investments deferred financing income"),
        L1("Investasi sewa simpanan jaminan", "Lease investments guarantee deposits"),
        L1("Cadangan kerugian penurunan nilai pada investasi sewa",
           "Allowance for impairment losses for lease investments"),
    ]),

    L0("Tagihan anjak piutang", "Factoring receivables", children=[
        L1("Tagihan anjak piutang pihak ketiga", "Factoring receivables third parties"),
        L1("Tagihan anjak piutang pihak berelasi", "Factoring receivables related parties"),
        L1("Tagihan anjak piutang pada pendapatan anjak piutang tangguhan",
           "Factoring receivables on deferred factoring income"),
        L1("Cadangan kerugian penurunan nilai pada tagihan anjak piutang",
           "Allowance for impairment losses for factoring receivables"),
    ]),

    L0("Piutang lainnya", "Other receivables", children=[
        L1("Piutang lainnya pihak ketiga", "Other receivables third parties"),
        L1("Piutang lainnya pihak berelasi", "Other receivables related parties"),
        L1("Cadangan kerugian penurunan nilai pada piutang lainnya",
           "Allowance for impairment losses for other receivables"),
    ]),

    L0("Aset keuangan lainnya", "Other financial assets"),
    L0("Obligasi pemerintah", "Government bonds", 163_510_426, 132_068_581),
    L0("Aset tidak lancar atau kelompok lepasan diklasifikasikan sebagai dimiliki untuk dijual",
       "Non-current assets or disposal groups classified as held-for-sale"),
    L0("Aset tidak lancar atau kelompok lepasan diklasifikasikan sebagai dimiliki untuk didistribusikan kepada pemilik",
       "Non-current assets or disposal groups classified as held-for-distribution to owners"),
    L0("Uang muka", "Advances"),
    L0("Biaya dibayar dimuka", "Prepaid expenses", 2_274_762, 2_941_109),
    L0("Jaminan", "Guarantees"),
    L0("Pajak dibayar dimuka", "Prepaid taxes", 3_090_304, 18_950),
    L0("Klaim atas pengembalian pajak", "Claims for tax refund"),
    L0("Aset pajak tangguhan", "Deferred tax assets", 2_315_772, 7_145_286),
    L0("Investasi yang dicatat dengan menggunakan metode ekuitas",
       "Investments accounted for using equity method", 652_311, 637_280),

    L0("Investasi pada ventura bersama dan entitas asosiasi",
       "Investments in joint ventures and associates", children=[
        L1("Investasi pada entitas ventura bersama", "Investments in joint ventures"),
        L1("Investasi pada entitas asosiasi", "Investments in associates",
           14_354_281, 12_748_127),
    ]),

    L0("Aset reasuransi", "Reinsurance assets"),
    L0("Aset imbalan pasca kerja", "Post-employment benefit assets"),
    L0("Goodwill", "Goodwill", 727_789, 727_786),
    L0("Aset takberwujud selain goodwill", "Intangible assets other than goodwill",
       14_416, 15_528),
    L0("Properti investasi", "Investment properties"),
    L0("Aset ijarah", "Ijarah assets"),
    L0("Aset tetap", "Property, plant, and equipment", 31_112_635, 30_408_236),
    L0("Aset hak guna", "Right of use assets"),
    L0("Aset pengampunan pajak", "Tax amnesty assets"),
    L0("Agunan yang diambil alih", "Foreclosed assets", 864_637, 914_825),
    L0("Aset lainnya", "Other assets", 18_156_904, 13_319_777),
    L0("Jumlah aset", "Total assets", 1_362_054_731, 1_130_128_862),
]

LIABILITIES = [
    L0("Liabilitas segera", "Obligations due immediately", 5_761_037, 5_514_720),
    L0("Bagi hasil yang belum dibagikan", "Undistributed profit sharing"),
    L0("Dana simpanan syariah", "Sharia deposits"),

    L0("Simpanan nasabah", "Customers deposits", children=[
        L1("Giro", "Current accounts", children=[
            L2("Giro pihak ketiga", "Current accounts third parties",
               275_664_821, 229_995_309),
            L2("Giro pihak berelasi", "Current accounts related parties",
               163_834_043, 75_738_219),
        ]),
        L1("Giro wadiah", "Wadiah demand deposits", children=[
            L2("Giro wadiah pihak ketiga", "Wadiah demand deposits third parties"),
            L2("Giro wadiah pihak berelasi", "Wadiah demand deposits related parties"),
        ]),
        L1("Tabungan", "Savings", children=[
            L2("Tabungan pihak ketiga", "Savings third parties",
               286_152_305, 257_431_407),
            L2("Tabungan pihak berelasi", "Savings related parties",
               307_846, 112_946),
        ]),
        L1("Tabungan wadiah", "Wadiah savings", children=[
            L2("Tabungan wadiah pihak ketiga", "Wadiah savings third parties"),
            L2("Tabungan wadiah pihak berelasi", "Wadiah savings related parties"),
        ]),
        L1("Deposito berjangka", "Time deposits", children=[
            L2("Deposito berjangka pihak ketiga", "Time deposits third parties",
               188_209_389, 205_250_321),
            L2("Deposito berjangka pihak berelasi", "Time deposits related parties",
               126_666_020, 36_982_646),
        ]),
        L1("Deposito wakalah", "Wakalah deposits", children=[
            L2("Deposito wakalah pihak ketiga", "Wakalah deposits third parties"),
            L2("Deposito wakalah pihak berelasi", "Wakalah deposits related parties"),
        ]),
    ]),

    L0("Simpanan dari bank lain", "Other banks deposits", children=[
        L1("Simpanan dari bank lain pihak berelasi",
           "Other banks deposits related parties", 2_945_332, 4_164_697),
        L1("Simpanan dari bank lain pihak ketiga",
           "Other banks deposits third parties", 8_617_451, 14_383_767),
    ]),

    L0("Efek yang dijual dengan janji untuk dibeli kembali",
       "Securities sold with repurchase agreement", 7_251_381, 15_890_945),

    L0("Liabilitas derivatif", "Derivative payables", children=[
        L1("Liabilitas derivatif pihak ketiga",
           "Derivative payables third parties", 977_119, 154_840),
        L1("Liabilitas derivatif pihak berelasi",
           "Derivative payables related parties", 4_421_780, 1_324_345),
    ]),

    L0("Utang asuransi", "Insurance payables"),
    L0("Utang koasuransi", "Coinsurance liabilities"),
    L0("Liabilitas kepada pemegang polis unit-linked",
       "Liabilities to policyholder in unit-linked contracts"),
    L0("Utang bunga", "Interest payables"),

    L0("Liabilitas akseptasi", "Acceptance liabilities", children=[
        L1("Liabilitas akseptasi pihak berelasi",
           "Acceptance liabilities related parties", 501_612, 900_755),
        L1("Liabilitas akseptasi pihak ketiga",
           "Acceptance liabilities third parties", 1_822_893, 3_328_729),
    ]),

    L0("Utang usaha", "Accounts payable"),
    L0("Uang muka dan angsuran", "Advances and installments"),
    L0("Utang dividen", "Dividends payable"),
    L0("Utang dealer", "Dealer payables"),

    L0("Pinjaman yang diterima", "Borrowings", children=[
        L1("Pinjaman yang diterima pihak ketiga", "Borrowings third parties",
           37_612_374, 41_357_233),
        L1("Pinjaman yang diterima pihak berelasi", "Borrowings related parties",
           1_428_013, 1_574_211),
        L1("Pinjaman yang diterima utang pada lembaga kliring dan penjaminan",
           "Borrowings payables to clearing and settlement guarantee institution"),
    ]),

    L0("Efek yang diterbitkan", "Securities issued", children=[
        L1("Utang obligasi", "Bonds payable"),
        L1("Sukuk", "Sukuk"),
        L1("Obligasi subordinasi", "Subordinated bonds", 18_339_988, 17_699_183),
        L1("Surat utang jangka menengah", "Medium term notes"),
        L1("Efek yang diterbitkan lainnya", "Others securities issued",
           14_253_190, 12_974_497),
    ]),

    L0("Liabilitas kontrak asuransi", "Insurance contract liabilities"),
    L0("Utang perusahaan efek", "Securities company payables"),
    L0("Provisi", "Provisions"),
    L0("Liabilitas atas kontrak", "Contract liabilities"),
    L0("Pendapatan ditangguhkan", "Deferred income"),
    L0("Liabilitas sewa pembiayaan", "Finance lease liabilities"),
    L0("Estimasi kerugian komitmen dan kontinjensi",
       "Estimated losses on commitments and contingencies", 1_463_072, 2_283_222),
    L0("Beban akrual", "Accrued expenses", 941_816, 1_529_305),
    L0("Utang pajak", "Taxes payable", 414_884, 317_569),
    L0("Liabilitas pajak tangguhan", "Deferred tax liabilities"),
    L0("Liabilitas pengampunan pajak", "Tax amnesty liabilities"),
    L0("Liabilitas lainnya", "Other liabilities", 29_290_002, 27_525_272),
    L0("Kewajiban imbalan pasca kerja", "Post-employment benefit obligations",
       8_838_995, 7_146_717),

    L0("Pinjaman subordinasi", "Subordinated loans", children=[
        L1("Pinjaman subordinasi pihak ketiga", "Subordinated loans third parties"),
        L1("Pinjaman subordinasi pihak berelasi", "Subordinated loans related parties"),
    ]),

    L0("Jumlah liabilitas", "Total liabilities", 1_185_715_363, 963_580_855),
]

TEMPORARY_SYIRKAH_FUNDS = [
    L0("Bukan bank", "Non-banks", children=[
        L1("Giro mudharabah", "Mudharabah current account", children=[
            L2("Giro mudharabah pihak ketiga",
               "Mudharabah current account third parties"),
            L2("Giro berjangka mudharabah pihak berelasi",
               "Mudharabah current account related parties"),
        ]),
        L1("Tabungan mudharabah", "Mudharabah saving deposits", children=[
            L2("Tabungan mudharabah pihak ketiga",
               "Mudharabah saving deposits third parties"),
            L2("Tabungan mudharabah pihak berelasi",
               "Mudharabah saving deposits related parties"),
        ]),
        L1("Deposito berjangka mudharabah", "Mudharabah time deposits", children=[
            L2("Deposito berjangka mudharabah pihak ketiga",
               "Mudharabah time deposits third parties"),
            L2("Deposito berjangka mudharabah pihak berelasi",
               "Mudharabah time deposits related parties"),
        ]),
    ]),
    L0("Bank", "Bank", children=[
        L1("Giro mudharabah", "Mudharabah current account"),
        L1("Tabungan mudharabah (ummat)", "Mudharabah saving deposits (ummat)"),
        L1("Deposito berjangka mudharabah", "Mudharabah time deposits"),
    ]),
    L0("Efek yang diterbitkan bank", "Bank securities issued", children=[
        L1("Investasi mudharabah antar bank", "Interbank mudharabah investments"),
        L1("Sukuk mudharabah", "Mudharabah sukuk"),
        L1("Sukuk mudharabah subordinasi", "Subordinated mudharabah sukuk"),
    ]),
    L0("Jumlah dana syirkah temporer", "Total temporary syirkah funds"),
    L0("Jumlah akumulasi dana tabarru", "Total accumulated tabarru's funds"),
]

EQUITY = [
    L0("Ekuitas yang diatribusikan kepada pemilik entitas induk",
       "Equity attributable to equity owners of parent entity", children=[
        L1("Saham biasa", "Common stocks", 9_054_807, 9_054_807),
        L1("Saham preferen", "Preferred stocks"),
        L1("Tambahan modal disetor", "Additional paid-in capital",
           17_010_254, 17_010_254),
        L1("Saham treasuri", "Treasury stocks", 0, 0),
        L1("Uang muka setoran modal", "Advances in capital stock"),
        L1("Opsi saham", "Stock options"),
        L1("Penjabaran laporan keuangan", "Translation adjustment"),
        L1("Cadangan revaluasi", "Revaluation reserves",
           16_711_395, 16_711_395),
        L1("Cadangan selisih kurs penjabaran",
           "Reserve of exchange differences on translation",
           -124_345, -96_998),
        L1("Cadangan perubahan nilai wajar aset keuangan nilai wajar melalui pendapatan komprehensif lainnya",
           "Reserve for changes in fair value of fair value through other comprehensive income financial assets",
           2_617_037, -1_465_059),
        L1("Cadangan keuntungan (kerugian) investasi pada instrumen ekuitas",
           "Reserve of gains (losses) from investments in equity instruments"),
        L1("Cadangan pembayaran berbasis saham", "Reserve of share-based payments",
           349_167, 322_589),
        L1("Cadangan lindung nilai arus kas", "Reserve of cash flow hedges"),
        L1("Cadangan pengukuran kembali program imbalan pasti",
           "Reserve of remeasurements of defined benefit plans"),
        L1("Cadangan lainnya", "Other reserves", 2_256_999, 2_256_999),
        L1("Selisih Transaksi Perubahan Ekuitas Entitas Anak/Asosiasi",
           "Difference Due to Changes of Equity in Subsidiary/Associates"),
        L1("Komponen ekuitas lainnya", "Other components of equity"),
        L1("Saldo laba (akumulasi kerugian)", "Retained earnings (deficit)", children=[
            L2("Saldo laba yang telah ditentukan penggunaanya",
               "Appropriated retained earnings", children=[
                L3("Cadangan umum dan wajib", "General and legal reserves",
                   2_778_412, 2_778_412),
                L3("Cadangan khusus", "Specific reserves"),
            ]),
            L2("Saldo laba yang belum ditentukan penggunaannya",
               "Unappropriated retained earnings", 121_076_418, 115_465_415),
        ]),
        L1("Jumlah ekuitas yang diatribusikan kepada pemilik entitas induk",
           "Total equity attributable to equity owners of parent entity",
           171_730_144, 162_071_240),
    ]),
    L0("Proforma ekuitas", "Proforma equity"),
    L0("Kepentingan non-pengendali", "Non-controlling interests", 4_609_224, 4_476_767),
    L0("Jumlah ekuitas", "Total equity", 176_339_368, 166_548_007),
    L0("Jumlah liabilitas, dana syirkah temporer dan ekuitas",
       "Total liabilities, temporary syirkah funds and equity",
       1_362_054_731, 1_130_128_862),
]


# ════════════════════════════════════════════════════════════════════════
# PROFIT OR LOSS  (pages 13 – 19)
# ════════════════════════════════════════════════════════════════════════

OPERATING = [
    L0("Pendapatan bunga", "Interest income", 69_394_154, 66_583_110),
    L0("Beban bunga", "Interest expenses", -29_060_959, -26_102_905),
    L0("Pendapatan pengelolaan dana oleh bank sebagai mudharib",
       "Revenue from fund management as mudharib"),
    L0("Hak pihak ketiga atas bagi hasil dana syirkah temporer",
       "Third parties share on return of temporary syirkah funds"),

    L0("Pendapatan asuransi", "Insurance income", children=[
        L1("Pendapatan dari premi asuransi", "Revenue from insurance premiums"),
        L1("Premi reasuransi", "Reinsurance premiums"),
        L1("Premi retrosesi", "Retrocession premiums"),
        L1("Penurunan (kenaikan) premi yang belum merupakan pendapatan",
           "Decrease (increase) in unearned premiums"),
        L1("Penurunan (kenaikan) pendapatan premi disesikan kepada reasuradur",
           "Decrease (increase) in premium income ceded to reinsurancer"),
        L1("Pendapatan komisi asuransi", "Insurance commission income"),
        L1("Pendapatan bersih investasi", "Net investment income", 2_029_888, 1_255_915),
        L1("Penerimaan ujrah", "Ujrah received"),
        L1("Pendapatan asuransi lainnya", "Other insurance income", 1_881_769, 1_992_607),
    ]),

    L0("Beban asuransi", "Insurance expenses", children=[
        L1("Beban klaim", "Claim expenses", -1_523, -1_752),
        L1("Klaim reasuransi", "Reinsurance claims"),
        L1("Klaim retrosesi", "Retrocession claims"),
        L1("Kenaikan (penurunan) estimasi liabilitas klaim",
           "Increase (decrease) in estimated claims liability"),
        L1("Kenaikan (penurunan) liabilitas manfaat polis masa depan",
           "Increase (decrease) in liability for future policy benefit"),
        L1("Kenaikan (penurunan) provisi yang timbul dari tes kecukupan liabilitas",
           "Increase (decrease) in provision for losses arising from liability adequacy test"),
        L1("Kenaikan (penurunan) liabilitas asuransi yang disesikan kepada reasuradur",
           "Increase (decrease) in insurance liabilities ceded to reinsurers"),
        L1("Kenaikan (penurunan) liabilitas pemegang polis pada kontrak unit-linked",
           "Increase (decrease) in liabilities to policyholder in unit-linked contracts"),
        L1("Beban komisi asuransi", "Insurance commission expenses"),
        L1("Ujrah dibayar", "Ujrah paid"),
        L1("Beban akuisisi dari kontrak asuransi", "Acquisition costs of insurance contracts"),
        L1("Beban asuransi lainnya", "Other insurance expenses", -3_166_941, -2_480_478),
    ]),

    L0("Pendapatan dari pembiayaan", "Financing income", children=[
        L1("Pendapatan dari pembiayaan konsumen", "Revenue from consumer financing"),
        L1("Pendapatan dari sewa pembiayaan", "Revenue from finance lease"),
        L1("Pendapatan dari sewa operasi", "Revenue from operating lease"),
        L1("Pendapatan dari anjak piutang", "Revenue from factoring"),
    ]),

    L0("Pendapatan sekuritas", "Securities income", children=[
        L1("Pendapatan kegiatan penjamin emisi dan penjualan efek",
           "Revenue from underwriting activities and selling fees"),
        L1("Pendapatan pembiayaan transaksi nasabah",
           "Revenue from financing transactions"),
        L1("Pendapatan jasa biro administrasi efek",
           "Revenue from securities administration service"),
        L1("Pendapatan kegiatan jasa manajer investasi",
           "Revenue from investment management services"),
        L1("Pendapatan kegiatan jasa penasehat keuangan",
           "Revenue from financial advisory services"),
        L1("Keuntungan (kerugian) dari transaksi perdagangan efek yang telah direalisasi",
           "Realised gains (losses) on trading of marketable securities",
           2_885_819, 1_769_243),
        L1("Keuntungan (kerugian) perubahan nilai wajar efek",
           "Gains (losses) on changes in fair value of marketable securities",
           141_576, 144_558),
    ]),

    L0("Pendapatan operasional lainnya", "Other operating income", children=[
        L1("Pendapatan investasi", "Investments income", 1_627_444, 1_522_798),
        L1("Pendapatan provisi dan komisi dari transaksi lainnya selain kredit",
           "Provisions and commissions income from transactions other than loan",
           11_167_675, 10_599_228),
        L1("Pendapatan transaksi perdagangan", "Revenue from trading transactions"),
        L1("Pendapatan dividen", "Dividends income"),
        L1("Keuntungan (kerugian) yang telah direalisasi atas instrumen derivatif",
           "Realised gains (losses) from derivative instruments"),
        L1("Penerimaan kembali aset yang telah dihapusbukukan",
           "Revenue from recovery of written-off assets", 5_438_988, 6_025_183),
        L1("Keuntungan (kerugian) selisih kurs mata uang asing",
           "Gains (losses) on changes in foreign exchange rates",
           1_037_143, 1_259_207),
        L1("Keuntungan (kerugian) pelepasan aset tetap",
           "Gains (losses) on disposal of property and equipment"),
        L1("Keuntungan (kerugian) pelepasan agunan yang diambil alih",
           "Gains (losses) on disposal of foreclosed assets"),
        L1("Pendapatan operasional lainnya", "Other operating income",
           1_601_317, 1_328_962),
    ]),

    L0("Pemulihan penyisihan kerugian penurunan nilai",
       "Recovery of impairment loss", children=[
        L1("Pemulihan penyisihan kerugian penurunan nilai aset keuangan",
           "Recovery of impairment loss of financial assets"),
        L1("Pemulihan penyisihan kerugian penurunan nilai aset keuangan - sewa pembiayaan",
           "Recovery of impairment loss of financial assets finance lease"),
        L1("Pemulihan penyisihan kerugian penurunan nilai aset keuangan - piutang pembiayaan konsumen",
           "Recovery of impairment loss of financial assets consumer financing receivables"),
        L1("Pemulihan penyisihan kerugian penurunan nilai aset non-keuangan",
           "Recovery of impairment loss of non-financial assets"),
        L1("Pemulihan penyisihan kerugian penurunan nilai aset non-keuangan - agunan yang diambil alih",
           "Recovery of impairment loss of non-financial assets repossessed collaterals"),
        L1("Pemulihan penyisihan estimasi kerugian atas komitmen dan kontinjensi",
           "Recovery of estimated loss of commitments and contingency"),
    ]),

    L0("Pembentukan kerugian penurunan nilai", "Allowances for impairment losses", children=[
        L1("Pembentukan penyisihan kerugian penurunan nilai aset produktif",
           "Allowances for impairment losses on earnings assets",
           -9_724_370, -8_210_562),
        L1("Pembentukan penyisihan kerugian penurunan nilai aset non-produktif",
           "Allowances for impairment losses on non-earnings assets"),
    ]),

    L0("Pembalikan (beban) estimasi kerugian komitmen dan kontijensi",
       "Reversal (expense) of estimated losses on commitments and contingencies"),

    L0("Beban operasional lainnya", "Other operating expenses", children=[
        L1("Beban umum dan administrasi", "General and administrative expenses",
           -23_923_938, -22_806_989),
        L1("Beban penjualan", "Selling expenses", -1_187_693, -1_103_491),
        L1("Beban sewa, pemeliharaan, dan perbaikan",
           "Rent, maintenance and improvement expenses", -695_137, -679_572),
        L1("Beban provisi dan komisi", "Other fees and commissions expenses", 0, 0),
        L1("Beban operasional lainnya", "Other operating expenses",
           -5_049_588, -4_522_174),
    ]),

    L0("Jumlah laba operasional", "Total profit from operation",
       24_395_624, 26_572_888),
]

NON_OPERATING = [
    L0("Pendapatan bukan operasional", "Non-operating income", 1_839, 0),
    L0("Beban bukan operasional", "Non-operating expenses", 0, -3_797),
    L0("Bagian atas laba (rugi) entitas asosiasi yang dicatat dengan menggunakan metode ekuitas",
       "Share of profit (loss) of associates accounted for using equity method"),
    L0("Bagian atas laba (rugi) entitas ventura bersama yang dicatat menggunakan metode ekuitas",
       "Share of profit (loss) of joint ventures accounted for using equity method"),
    L0("Jumlah laba (rugi) sebelum pajak penghasilan",
       "Total profit (loss) before tax", 24_397_463, 26_569_091),
    L0("Pendapatan (beban) pajak", "Tax benefit (expenses)", -4_286_423, -4_899_694),
    L0("Jumlah laba (rugi) dari operasi yang dilanjutkan",
       "Total profit (loss) from continuing operations", 20_111_040, 21_669_397),
    L0("Laba (rugi) dari operasi yang dihentikan",
       "Profit (loss) from discontinued operations"),
    L0("Jumlah laba (rugi)", "Total profit (loss)", 20_111_040, 21_669_397),
]

OCI = [
    L0("Pendapatan komprehensif lainnya yang tidak akan direklasifikasi ke laba rugi, setelah pajak",
       "Other comprehensive income that will not be reclassified to profit or loss, after tax",
       children=[
        L1("Pendapatan komprehensif lainnya atas keuntungan (kerugian) hasil revaluasi aset tetap, setelah pajak",
           "Other comprehensive income for gains (losses) on revaluation of property and equipment, after tax",
           0, 1_263_566),
        L1("Pendapatan komprehensif lainnya atas pengukuran kembali kewajiban manfaat pasti, setelah pajak",
           "Other comprehensive income for remeasurement of defined benefit obligation, after tax",
           -591_455, 523_859),
        L1("Penyesuaian lainnya atas pendapatan komprehensif lainnya yang tidak akan direklasifikasi ke laba rugi, setelah pajak",
           "Other adjustments to other comprehensive income that will not be reclassified to profit or loss, after tax",
           112_403, -99_722),
        L1("Jumlah pendapatan komprehensif lainnya yang tidak akan direklasifikasi ke laba rugi, setelah pajak",
           "Total other comprehensive income that will not be reclassified to profit or loss, after tax",
           -479_052, 1_687_703),
    ]),

    L0("Pendapatan komprehensif lainnya yang akan direklasifikasi ke laba rugi, setelah pajak",
       "Other comprehensive income that may be reclassified to profit or loss, after tax",
       children=[
        L1("Keuntungan (kerugian) selisih kurs penjabaran, setelah pajak",
           "Gains (losses) on exchange differences on translation, after tax",
           -27_347, -38_631),
        L1("Penyesuaian reklasifikasi selisih kurs penjabaran, setelah pajak",
           "Reclassification adjustments on exchange differences on translation, after tax"),
        L1("Keuntungan (kerugian) yang belum direalisasi atas perubahan nilai wajar aset keuangan melalui penghasilan komprehensif lain, setelah pajak",
           "Unrealised gains (losses) on changes in fair value through other comprehensive income financial assets, after tax",
           5_084_110, -770_837),
        L1("Penyesuaian reklasifikasi atas aset keuangan nilai wajar melalui pendapatan komprehensif lainnya, setelah pajak",
           "Reclassification adjustments on fair value through other comprehensive income financial assets, after tax"),
        L1("Keuntungan (kerugian) lindung nilai arus kas, setelah pajak",
           "Gains (losses) on cash flow hedges, after tax"),
        L1("Penyesuaian reklasifikasi atas lindung nilai arus kas, setelah pajak",
           "Reclassification adjustments on cash flow hedges, after tax"),
        L1("Nilai tercatat dari aset (liabilitas) non-keuangan yang perolehan atau keterjadiannya merupakan suatu prakiraan transaksi yang kemungkinan besar terjadi yang dilindung nilai, setelah pajak",
           "Carrying amount of non-financial asset (liability) whose acquisition or incurrence was hedged on highly probable forecast transaction, adjusted from equity, after tax"),
        L1("Keuntungan (kerugian) lindung nilai investasi bersih kegiatan usaha luar negeri, setelah pajak",
           "Gains (losses) on hedges of net investments in foreign operations, after tax"),
        L1("Penyesuaian reklasifikasi atas lindung nilai investasi bersih kegiatan usaha luar negeri, setelah pajak",
           "Reclassification adjustments on hedges of net investments in foreign operations, after tax"),
        L1("Bagian pendapatan komprehensif lainnya dari entitas asosiasi yang dicatat dengan menggunakan metode ekuitas, setelah pajak",
           "Share of other comprehensive income of associates accounted for using equity method, after tax"),
        L1("Bagian pendapatan komprehensif lainnya dari entitas ventura bersama yang dicatat dengan menggunakan metode ekuitas, setelah pajak",
           "Share of other comprehensive income of joint ventures accounted for using equity method, after tax"),
        L1("Penyesuaian lainnya atas pendapatan komprehensif lainnya yang akan direklasifikasi ke laba rugi, setelah pajak",
           "Other adjustments to other comprehensive income that may be reclassified to profit or loss, after tax",
           -972_629, 152_276),
        L1("Jumlah pendapatan komprehensif lainnya yang akan direklasifikasi ke laba rugi, setelah pajak",
           "Total other comprehensive income that may be reclassified to profit or loss, after tax",
           4_084_134, -657_192),
    ]),

    L0("Jumlah pendapatan komprehensif lainnya, setelah pajak",
       "Total other comprehensive income, after tax", 3_605_082, 1_030_511),
    L0("Jumlah laba rugi komprehensif", "Total comprehensive income",
       23_716_122, 22_699_908),

    L0("Laba (rugi) yang dapat diatribusikan", "Profit (loss) attributable to", children=[
        L1("Laba (rugi) yang dapat diatribusikan ke entitas induk",
           "Profit (loss) attributable to parent entity", 20_040_703, 21_463_599),
        L1("Laba (rugi) yang dapat diatribusikan ke kepentingan non-pengendali",
           "Profit (loss) attributable to non-controlling interests",
           70_337, 205_798),
    ]),

    L0("Laba rugi komprehensif yang dapat diatribusikan",
       "Comprehensive income attributable to", children=[
        L1("Laba rugi komprehensif yang dapat diatribusikan ke entitas induk",
           "Comprehensive income attributable to parent entity",
           23_583_665, 22_551_316),
        L1("Laba rugi komprehensif yang dapat diatribusikan ke kepentingan non-pengendali",
           "Comprehensive income attributable to non-controlling interests",
           132_457, 148_592),
    ]),
]

EPS = [
    L0("Laba per saham dasar diatribusikan kepada pemilik entitas induk",
       "Basic earnings per share attributable to equity owners of the parent entity",
       children=[
        L1("Laba (rugi) per saham dasar dari operasi yang dilanjutkan",
           "Basic earnings (loss) per share from continuing operations", 537, 576),
        L1("Laba (rugi) per saham dasar dari operasi yang dihentikan",
           "Basic earnings (loss) per share from discontinued operations"),
    ]),
    L0("Laba (rugi) per saham dilusian", "Diluted earnings (loss) per share",
       children=[
        L1("Laba (rugi) per saham dilusian dari operasi yang dilanjutkan",
           "Diluted earnings (loss) per share from continuing operations"),
        L1("Laba (rugi) per saham dilusian dari operasi yang dihentikan",
           "Diluted earnings (loss) per share from discontinued operations"),
    ]),
]


# ════════════════════════════════════════════════════════════════════════
# Assemble + write
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    output = {
        "schema_version": "1.0",
        "source_pdf": "FinancialStatement-2025-Tahunan-BBNI.pdf",
        "approach": "ground_truth",
        "meta": {
            "issuer": "BBNI",
            "issuer_full": "PT Bank Negara Indonesia (Persero) Tbk",
            "fiscal_year": 2025,
            "rounding": "in_million_idr",
            "transcribed_pages": list(range(4, 20)),
            "transcribed_by": "manual",
        },
        "statements": [
            {
                "type": "financial_position",
                "years": [2025, 2024],
                "pages": list(range(4, 13)),
                "sections": {
                    "assets": ASSETS,
                    "liabilities": LIABILITIES,
                    "temporary_syirkah_funds": TEMPORARY_SYIRKAH_FUNDS,
                    "equity": EQUITY,
                },
            },
            {
                "type": "profit_or_loss",
                "years": [2025, 2024],
                "pages": list(range(13, 20)),
                "sections": {
                    "operating": OPERATING,
                    "non_operating": NON_OPERATING,
                    "other_comprehensive_income": OCI,
                    "earnings_per_share": EPS,
                },
            },
        ],
    }

    out_path = Path(__file__).parent / "bbni_2025.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Quick sanity stats
    def count(nodes):
        total = leaves = filled = 0
        for n in nodes:
            total += 1
            if n["children"]:
                t, l, f = count(n["children"])
                total += t; leaves += l; filled += f
            else:
                leaves += 1
            if any(v is not None for v in n["values"].values()):
                filled += 1
        return total, leaves, filled

    print(f"Wrote: {out_path}")
    for stmt in output["statements"]:
        for sec_name, nodes in stmt["sections"].items():
            t, l, f = count(nodes)
            print(f"  {stmt['type']:<20s} / {sec_name:<28s} total={t:4d}  leaves={l:4d}  with_values={f:4d}")


if __name__ == "__main__":
    main()
