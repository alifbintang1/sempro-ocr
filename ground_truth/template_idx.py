"""Shared IDX XBRL financial-statement template (taxonomy [4220000] + [4312000]).

All three banks (BBNI, BBRI, BMRI) share the SAME label structure and hierarchy
because they emit the same XBRL taxonomy. Only the numeric values differ, plus
the OCI subsection wording ("setelah pajak" vs "sebelum pajak").

This module defines the structure ONCE and injects values via a lookup dict
keyed by Indonesian label. The four labels that appear twice in the document
are disambiguated with a qualified key (see `_DUP_KEYS`).

Used by build_bbri_2025.py and build_bmri_2025.py. The BBNI builder
(build_bbni_2025.py) predates this template and is kept standalone; this
template is validated to reproduce BBNI's output byte-for-byte (see
`regenerate_bbni()` in this file's __main__ self-test).
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

Values = Callable[[str], Tuple[Optional[int], Optional[int]]]


def make_values_lookup(values: dict) -> Values:
    """Return a V(key) -> (v25, v24) lookup, defaulting to (None, None)."""
    def V(key: str) -> Tuple[Optional[int], Optional[int]]:
        return values.get(key, (None, None))
    return V


def N(label, label_en, level, v25=None, v24=None, children=None) -> dict:
    children = children or []
    vals = {"2025-12-31": v25, "2024-12-31": v24}
    if v25 is not None or v24 is not None:
        row_type = "item"
    elif children:
        row_type = "group"
    else:
        row_type = "label_only"
    return {"level": level, "label": label, "label_en": label_en,
            "row_type": row_type, "values": vals, "children": children}


# ── Financial Position ─────────────────────────────────────────────────────

def build_assets(V: Values) -> list:
    def leaf(lbl, en, key=None):
        v25, v24 = V(key or lbl)
        return N(lbl, en, 1, v25, v24)

    def item0(lbl, en, key=None):
        v25, v24 = V(key or lbl)
        return N(lbl, en, 0, v25, v24)

    def pihak3(prefix, en_prefix):
        # standard "pihak ketiga / pihak berelasi / Cadangan" trio
        return [
            leaf(f"{prefix} pihak ketiga", f"{en_prefix} third parties"),
            leaf(f"{prefix} pihak berelasi", f"{en_prefix} related parties"),
        ]

    return [
        item0("Kas", "Cash"),
        item0("Dana yang dibatasi penggunaannya", "Restricted funds"),
        item0("Giro pada Bank Indonesia", "Current accounts with Bank Indonesia"),
        N("Giro pada bank lain", "Current accounts with other banks", 0, children=[
            leaf("Giro pada bank lain pihak ketiga",
                 "Current accounts with other banks third parties"),
            leaf("Giro pada bank lain pihak berelasi",
                 "Current accounts with other banks related parties"),
            leaf("Cadangan kerugian penurunan nilai pada giro pada bank lain",
                 "Allowance for impairment losses for current accounts with other bank"),
        ]),
        N("Penempatan pada Bank Indonesia dan bank lain",
          "Placements with Bank Indonesia and other banks", 0, children=[
            leaf("Penempatan pada Bank Indonesia dan bank lain pihak ketiga",
                 "Placements with Bank Indonesia and other banks third parties"),
            leaf("Penempatan pada Bank Indonesia dan bank lain pihak berelasi",
                 "Placements with Bank Indonesia and other banks related parties"),
            leaf("Cadangan kerugian penurunan nilai pada penempatan pada bank lain",
                 "Allowance for impairment losses for placements with other banks"),
        ]),
        N("Piutang asuransi", "Insurance receivables", 0, children=[
            leaf("Piutang asuransi pihak ketiga", "Insurance receivables third parties"),
            leaf("Piutang asuransi pihak berelasi", "Insurance receivables related parties"),
            leaf("Cadangan kerugian penurunan nilai pada piutang asuransi",
                 "Allowance for impairment losses for insurance receivables"),
        ]),
        item0("Biaya akuisisi tangguhan", "Deferred acquisition costs"),
        item0("Deposito pada lembaga kliring dan penjaminan",
              "Deposits to clearing and settlement guarantee institution"),
        N("Efek-efek yang diperdagangkan", "Marketable securities", 0, children=[
            leaf("Efek-efek yang diperdagangkan pihak ketiga",
                 "Marketable securities third parties"),
            leaf("Efek-efek yang diperdagangkan pihak berelasi",
                 "Marketable securities related parties"),
            leaf("Cadangan kerugian penurunan nilai pada efek-efek yang diperdagangkan",
                 "Allowance for impairment losses for marketable securities"),
        ]),
        item0("Investasi pemegang polis pada kontrak unit-linked",
              "Investments of policyholder in unit-linked contracts"),
        item0("Efek yang dibeli dengan janji dijual kembali",
              "Securities purchased under agreement to resale"),
        N("Wesel ekspor dan tagihan lainnya", "Bills and other receivables", 0, children=[
            leaf("Wesel ekspor dan tagihan lainnya pihak ketiga",
                 "Bills and other receivables third parties"),
            leaf("Wesel ekspor dan tagihan lainnya pihak berelasi",
                 "Bills and other receivables related parties"),
            leaf("Cadangan kerugian penurunan nilai pada wesel ekspor dan tagihan lainnya",
                 "Allowance for impairment losses for bills and other receivables"),
        ]),
        N("Tagihan akseptasi", "Acceptance receivables", 0, children=[
            leaf("Tagihan akseptasi pihak ketiga", "Acceptance receivables third parties"),
            leaf("Tagihan akseptasi pihak berelasi", "Acceptance receivables related parties"),
            leaf("Cadangan kerugian penurunan nilai pada tagihan akseptasi",
                 "Allowance for impairment losses for acceptance receivables"),
        ]),
        N("Tagihan derivatif", "Derivative receivables", 0, children=[
            leaf("Tagihan derivatif pihak ketiga", "Derivative receivables third parties"),
            leaf("Tagihan derivatif pihak berelasi", "Derivative receivables related parties"),
        ]),
        N("Pinjaman yang diberikan", "Loans", 0, children=[
            leaf("Pinjaman yang diberikan pihak ketiga", "Loans third parties"),
            leaf("Pinjaman yang diberikan pihak berelasi", "Loans related parties"),
            leaf("Cadangan kerugian penurunan nilai pada pinjaman yang diberikan",
                 "Allowance for impairment losses for loans"),
        ]),
        item0("Piutang dari lembaga kliring dan penjaminan",
              "Receivables from clearing and settlement guarantee institution"),
        N("Piutang nasabah", "Receivables from customers", 0, children=[
            leaf("Piutang nasabah pihak ketiga", "Receivables from customers third parties"),
            leaf("Piutang nasabah pihak berelasi", "Receivables from customers related parties"),
            leaf("Cadangan kerugian penurunan nilai pada piutang nasabah",
                 "Allowance for impairment losses for receivables from customers"),
        ]),
        N("Piutang murabahah", "Murabahah receivables", 0, children=[
            leaf("Piutang murabahah pihak ketiga", "Murabahah receivables third parties"),
            leaf("Piutang murabahah pihak berelasi", "Murabahah receivables related parties"),
            leaf("Cadangan kerugian penurunan nilai pada piutang murabahah",
                 "Allowance for impairment losses for murabahah receivables"),
        ]),
        N("Piutang istishna", "Istishna receivables", 0, children=[
            leaf("Piutang istishna pihak ketiga", "Istishna receivables third parties"),
            leaf("Piutang istishna pihak berelasi", "Istishna receivables related parties"),
            leaf("Cadangan kerugian penurunan nilai pada piutang istishna",
                 "Allowance for impairment losses for istishna receivables"),
        ]),
        N("Piutang ijarah", "Ijarah receivables", 0, children=[
            leaf("Piutang ijarah pihak ketiga", "Ijarah receivables third parties"),
            leaf("Piutang ijarah pihak berelasi", "Ijarah receivables related parties"),
            leaf("Cadangan kerugian penurunan nilai pada piutang ijarah",
                 "Allowance for impairment losses for ijarah receivables"),
        ]),
        N("Piutang pembiayaan konsumen", "Consumer financing receivables", 0, children=[
            leaf("Piutang pembiayaan konsumen pihak ketiga",
                 "Consumer financing receivables third parties"),
            leaf("Piutang pembiayaan konsumen pihak berelasi",
                 "Consumer financing receivables related parties"),
            leaf("Cadangan kerugian penurunan nilai pada piutang pembiayaan konsumen",
                 "Allowance for impairment losses for consumer financing receivables"),
        ]),
        N("Pinjaman qardh", "Qardh funds", 0, children=[
            leaf("Pinjaman qardh pihak ketiga", "Qardh funds third parties"),
            leaf("Pinjaman qardh pihak berelasi", "Qardh funds related parties"),
            leaf("Cadangan kerugian penurunan nilai pada pinjaman qardh",
                 "Allowance for impairment losses for qardh funds"),
        ]),
        N("Pembiayaan mudharabah", "Mudharabah financing", 0, children=[
            leaf("Pembiayaan mudharabah pihak ketiga", "Mudharabah financing third parties"),
            leaf("Pembiayaan mudharabah pihak berelasi", "Mudharabah financing related parties"),
            leaf("Cadangan kerugian penurunan nilai pada pembiayaan mudharabah",
                 "Allowance for impairment losses for mudharabah financing"),
        ]),
        N("Pembiayaan musyarakah", "Musyarakah financing", 0, children=[
            leaf("Pembiayaan musyarakah pihak ketiga", "Musyarakah financing third parties"),
            leaf("Pembiayaan musyarakah pihak berelasi", "Musyarakah financing related parties"),
            leaf("Cadangan kerugian penurunan nilai pada pembiayaan musyarakah",
                 "Allowance for impairment losses for musyarakah financing"),
        ]),
        N("Investasi sewa", "Lease investments", 0, children=[
            leaf("Investasi sewa pihak ketiga", "Lease investments third parties"),
            leaf("Investasi sewa pihak berelasi", "Lease investments related parties"),
            leaf("Investasi sewa nilai residu yang terjamin",
                 "Lease investments guaranteed residual value"),
            leaf("Investasi sewa pendapatan pembiayaan tangguhan",
                 "Lease investments deferred financing income"),
            leaf("Investasi sewa simpanan jaminan", "Lease investments guarantee deposits"),
            leaf("Cadangan kerugian penurunan nilai pada investasi sewa",
                 "Allowance for impairment losses for lease investments"),
        ]),
        N("Tagihan anjak piutang", "Factoring receivables", 0, children=[
            leaf("Tagihan anjak piutang pihak ketiga", "Factoring receivables third parties"),
            leaf("Tagihan anjak piutang pihak berelasi", "Factoring receivables related parties"),
            leaf("Tagihan anjak piutang pada pendapatan anjak piutang tangguhan",
                 "Factoring receivables on deferred factoring income"),
            leaf("Cadangan kerugian penurunan nilai pada tagihan anjak piutang",
                 "Allowance for impairment losses for factoring receivables"),
        ]),
        N("Piutang lainnya", "Other receivables", 0, children=[
            leaf("Piutang lainnya pihak ketiga", "Other receivables third parties"),
            leaf("Piutang lainnya pihak berelasi", "Other receivables related parties"),
            leaf("Cadangan kerugian penurunan nilai pada piutang lainnya",
                 "Allowance for impairment losses for other receivables"),
        ]),
        item0("Aset keuangan lainnya", "Other financial assets"),
        item0("Obligasi pemerintah", "Government bonds"),
        item0("Aset tidak lancar atau kelompok lepasan diklasifikasikan sebagai dimiliki untuk dijual",
              "Non-current assets or disposal groups classified as held-for-sale"),
        item0("Aset tidak lancar atau kelompok lepasan diklasifikasikan sebagai dimiliki untuk didistribusikan kepada pemilik",
              "Non-current assets or disposal groups classified as held-for-distribution to owners"),
        item0("Uang muka", "Advances"),
        item0("Biaya dibayar dimuka", "Prepaid expenses"),
        item0("Jaminan", "Guarantees"),
        item0("Pajak dibayar dimuka", "Prepaid taxes"),
        item0("Klaim atas pengembalian pajak", "Claims for tax refund"),
        item0("Aset pajak tangguhan", "Deferred tax assets"),
        item0("Investasi yang dicatat dengan menggunakan metode ekuitas",
              "Investments accounted for using equity method"),
        N("Investasi pada ventura bersama dan entitas asosiasi",
          "Investments in joint ventures and associates", 0, children=[
            leaf("Investasi pada entitas ventura bersama", "Investments in joint ventures"),
            leaf("Investasi pada entitas asosiasi", "Investments in associates"),
        ]),
        item0("Aset reasuransi", "Reinsurance assets"),
        item0("Aset imbalan pasca kerja", "Post-employment benefit assets"),
        item0("Goodwill", "Goodwill"),
        item0("Aset takberwujud selain goodwill", "Intangible assets other than goodwill"),
        item0("Properti investasi", "Investment properties"),
        item0("Aset ijarah", "Ijarah assets"),
        item0("Aset tetap", "Property, plant, and equipment"),
        item0("Aset hak guna", "Right of use assets"),
        item0("Aset pengampunan pajak", "Tax amnesty assets"),
        item0("Agunan yang diambil alih", "Foreclosed assets"),
        item0("Aset lainnya", "Other assets"),
        item0("Jumlah aset", "Total assets"),
    ]


def build_liabilities(V: Values) -> list:
    def leaf(lbl, en, key=None, lvl=1):
        v25, v24 = V(key or lbl)
        return N(lbl, en, lvl, v25, v24)

    return [
        leaf("Liabilitas segera", "Obligations due immediately", lvl=0),
        leaf("Bagi hasil yang belum dibagikan", "Undistributed profit sharing", lvl=0),
        leaf("Dana simpanan syariah", "Sharia deposits", lvl=0),
        N("Simpanan nasabah", "Customers deposits", 0, children=[
            N("Giro", "Current accounts", 1, children=[
                N("Giro pihak ketiga", "Current accounts third parties", 2, *V("Giro pihak ketiga")),
                N("Giro pihak berelasi", "Current accounts related parties", 2, *V("Giro pihak berelasi")),
            ]),
            N("Giro wadiah", "Wadiah demand deposits", 1, children=[
                N("Giro wadiah pihak ketiga", "Wadiah demand deposits third parties", 2, *V("Giro wadiah pihak ketiga")),
                N("Giro wadiah pihak berelasi", "Wadiah demand deposits related parties", 2, *V("Giro wadiah pihak berelasi")),
            ]),
            N("Tabungan", "Savings", 1, children=[
                N("Tabungan pihak ketiga", "Savings third parties", 2, *V("Tabungan pihak ketiga")),
                N("Tabungan pihak berelasi", "Savings related parties", 2, *V("Tabungan pihak berelasi")),
            ]),
            N("Tabungan wadiah", "Wadiah savings", 1, children=[
                N("Tabungan wadiah pihak ketiga", "Wadiah savings third parties", 2, *V("Tabungan wadiah pihak ketiga")),
                N("Tabungan wadiah pihak berelasi", "Wadiah savings related parties", 2, *V("Tabungan wadiah pihak berelasi")),
            ]),
            N("Deposito berjangka", "Time deposits", 1, children=[
                N("Deposito berjangka pihak ketiga", "Time deposits third parties", 2, *V("Deposito berjangka pihak ketiga")),
                N("Deposito berjangka pihak berelasi", "Time deposits related parties", 2, *V("Deposito berjangka pihak berelasi")),
            ]),
            N("Deposito wakalah", "Wakalah deposits", 1, children=[
                N("Deposito wakalah pihak ketiga", "Wakalah deposits third parties", 2, *V("Deposito wakalah pihak ketiga")),
                N("Deposito wakalah pihak berelasi", "Wakalah deposits related parties", 2, *V("Deposito wakalah pihak berelasi")),
            ]),
        ]),
        N("Simpanan dari bank lain", "Other banks deposits", 0, children=[
            leaf("Simpanan dari bank lain pihak berelasi", "Other banks deposits related parties"),
            leaf("Simpanan dari bank lain pihak ketiga", "Other banks deposits third parties"),
        ]),
        leaf("Efek yang dijual dengan janji untuk dibeli kembali",
             "Securities sold with repurchase agreement", lvl=0),
        N("Liabilitas derivatif", "Derivative payables", 0, children=[
            leaf("Liabilitas derivatif pihak ketiga", "Derivative payables third parties"),
            leaf("Liabilitas derivatif pihak berelasi", "Derivative payables related parties"),
        ]),
        leaf("Utang asuransi", "Insurance payables", lvl=0),
        leaf("Utang koasuransi", "Coinsurance liabilities", lvl=0),
        leaf("Liabilitas kepada pemegang polis unit-linked",
             "Liabilities to policyholder in unit-linked contracts", lvl=0),
        leaf("Utang bunga", "Interest payables", lvl=0),
        N("Liabilitas akseptasi", "Acceptance liabilities", 0, children=[
            leaf("Liabilitas akseptasi pihak berelasi", "Acceptance liabilities related parties"),
            leaf("Liabilitas akseptasi pihak ketiga", "Acceptance liabilities third parties"),
        ]),
        leaf("Utang usaha", "Accounts payable", lvl=0),
        leaf("Uang muka dan angsuran", "Advances and installments", lvl=0),
        leaf("Utang dividen", "Dividends payable", lvl=0),
        leaf("Utang dealer", "Dealer payables", lvl=0),
        N("Pinjaman yang diterima", "Borrowings", 0, children=[
            leaf("Pinjaman yang diterima pihak ketiga", "Borrowings third parties"),
            leaf("Pinjaman yang diterima pihak berelasi", "Borrowings related parties"),
            leaf("Pinjaman yang diterima utang pada lembaga kliring dan penjaminan",
                 "Borrowings payables to clearing and settlement guarantee institution"),
        ]),
        N("Efek yang diterbitkan", "Securities issued", 0, children=[
            leaf("Utang obligasi", "Bonds payable"),
            leaf("Sukuk", "Sukuk"),
            leaf("Obligasi subordinasi", "Subordinated bonds"),
            leaf("Surat utang jangka menengah", "Medium term notes"),
            leaf("Efek yang diterbitkan lainnya", "Others securities issued"),
        ]),
        leaf("Liabilitas kontrak asuransi", "Insurance contract liabilities", lvl=0),
        leaf("Utang perusahaan efek", "Securities company payables", lvl=0),
        leaf("Provisi", "Provisions", lvl=0),
        leaf("Liabilitas atas kontrak", "Contract liabilities", lvl=0),
        leaf("Pendapatan ditangguhkan", "Deferred income", lvl=0),
        leaf("Liabilitas sewa pembiayaan", "Finance lease liabilities", lvl=0),
        leaf("Estimasi kerugian komitmen dan kontinjensi",
             "Estimated losses on commitments and contingencies", lvl=0),
        leaf("Beban akrual", "Accrued expenses", lvl=0),
        leaf("Utang pajak", "Taxes payable", lvl=0),
        leaf("Liabilitas pajak tangguhan", "Deferred tax liabilities", lvl=0),
        leaf("Liabilitas pengampunan pajak", "Tax amnesty liabilities", lvl=0),
        leaf("Liabilitas lainnya", "Other liabilities", lvl=0),
        leaf("Kewajiban imbalan pasca kerja", "Post-employment benefit obligations", lvl=0),
        N("Pinjaman subordinasi", "Subordinated loans", 0, children=[
            leaf("Pinjaman subordinasi pihak ketiga", "Subordinated loans third parties"),
            leaf("Pinjaman subordinasi pihak berelasi", "Subordinated loans related parties"),
        ]),
        leaf("Jumlah liabilitas", "Total liabilities", lvl=0),
    ]


def build_syirkah(V: Values) -> list:
    return [
        N("Bukan bank", "Non-banks", 0, children=[
            N("Giro mudharabah", "Mudharabah current account", 1, children=[
                N("Giro mudharabah pihak ketiga", "Mudharabah current account third parties", 2, *V("Giro mudharabah pihak ketiga")),
                N("Giro berjangka mudharabah pihak berelasi", "Mudharabah current account related parties", 2, *V("Giro berjangka mudharabah pihak berelasi")),
            ]),
            N("Tabungan mudharabah", "Mudharabah saving deposits", 1, children=[
                N("Tabungan mudharabah pihak ketiga", "Mudharabah saving deposits third parties", 2, *V("Tabungan mudharabah pihak ketiga")),
                N("Tabungan mudharabah pihak berelasi", "Mudharabah saving deposits related parties", 2, *V("Tabungan mudharabah pihak berelasi")),
            ]),
            N("Deposito berjangka mudharabah", "Mudharabah time deposits", 1, children=[
                N("Deposito berjangka mudharabah pihak ketiga", "Mudharabah time deposits third parties", 2, *V("Deposito berjangka mudharabah pihak ketiga")),
                N("Deposito berjangka mudharabah pihak berelasi", "Mudharabah time deposits related parties", 2, *V("Deposito berjangka mudharabah pihak berelasi")),
            ]),
        ]),
        N("Bank", "Bank", 0, children=[
            N("Giro mudharabah", "Mudharabah current account", 1, *V("Bank|Giro mudharabah")),
            N("Tabungan mudharabah (ummat)", "Mudharabah saving deposits (ummat)", 1, *V("Tabungan mudharabah (ummat)")),
            N("Deposito berjangka mudharabah", "Mudharabah time deposits", 1, *V("Bank|Deposito berjangka mudharabah")),
        ]),
        N("Efek yang diterbitkan bank", "Bank securities issued", 0, children=[
            N("Investasi mudharabah antar bank", "Interbank mudharabah investments", 1, *V("Investasi mudharabah antar bank")),
            N("Sukuk mudharabah", "Mudharabah sukuk", 1, *V("Sukuk mudharabah")),
            N("Sukuk mudharabah subordinasi", "Subordinated mudharabah sukuk", 1, *V("Sukuk mudharabah subordinasi")),
        ]),
        N("Jumlah dana syirkah temporer", "Total temporary syirkah funds", 0, *V("Jumlah dana syirkah temporer")),
        N("Jumlah akumulasi dana tabarru", "Total accumulated tabarru's funds", 0, *V("Jumlah akumulasi dana tabarru")),
    ]


def build_equity(V: Values) -> list:
    def l1(lbl, en, key=None):
        v25, v24 = V(key or lbl)
        return N(lbl, en, 1, v25, v24)
    return [
        N("Ekuitas yang diatribusikan kepada pemilik entitas induk",
          "Equity attributable to equity owners of parent entity", 0, children=[
            l1("Saham biasa", "Common stocks"),
            l1("Saham preferen", "Preferred stocks"),
            l1("Tambahan modal disetor", "Additional paid-in capital"),
            l1("Saham treasuri", "Treasury stocks"),
            l1("Uang muka setoran modal", "Advances in capital stock"),
            l1("Opsi saham", "Stock options"),
            l1("Penjabaran laporan keuangan", "Translation adjustment"),
            l1("Cadangan revaluasi", "Revaluation reserves"),
            l1("Cadangan selisih kurs penjabaran",
               "Reserve of exchange differences on translation"),
            l1("Cadangan perubahan nilai wajar aset keuangan nilai wajar melalui pendapatan komprehensif lainnya",
               "Reserve for changes in fair value of fair value through other comprehensive income financial assets"),
            l1("Cadangan keuntungan (kerugian) investasi pada instrumen ekuitas",
               "Reserve of gains (losses) from investments in equity instruments"),
            l1("Cadangan pembayaran berbasis saham", "Reserve of share-based payments"),
            l1("Cadangan lindung nilai arus kas", "Reserve of cash flow hedges"),
            l1("Cadangan pengukuran kembali program imbalan pasti",
               "Reserve of remeasurements of defined benefit plans"),
            l1("Cadangan lainnya", "Other reserves"),
            l1("Selisih Transaksi Perubahan Ekuitas Entitas Anak/Asosiasi",
               "Difference Due to Changes of Equity in Subsidiary/Associates"),
            l1("Komponen ekuitas lainnya", "Other components of equity"),
            N("Saldo laba (akumulasi kerugian)", "Retained earnings (deficit)", 1, children=[
                N("Saldo laba yang telah ditentukan penggunaanya",
                  "Appropriated retained earnings", 2, children=[
                    N("Cadangan umum dan wajib", "General and legal reserves", 3, *V("Cadangan umum dan wajib")),
                    N("Cadangan khusus", "Specific reserves", 3, *V("Cadangan khusus")),
                ]),
                N("Saldo laba yang belum ditentukan penggunaannya",
                  "Unappropriated retained earnings", 2, *V("Saldo laba yang belum ditentukan penggunaannya")),
            ]),
            l1("Jumlah ekuitas yang diatribusikan kepada pemilik entitas induk",
               "Total equity attributable to equity owners of parent entity"),
        ]),
        N("Proforma ekuitas", "Proforma equity", 0, *V("Proforma ekuitas")),
        N("Kepentingan non-pengendali", "Non-controlling interests", 0, *V("Kepentingan non-pengendali")),
        N("Jumlah ekuitas", "Total equity", 0, *V("Jumlah ekuitas")),
        N("Jumlah liabilitas, dana syirkah temporer dan ekuitas",
          "Total liabilities, temporary syirkah funds and equity", 0,
          *V("Jumlah liabilitas, dana syirkah temporer dan ekuitas")),
    ]


# ── Profit or Loss ─────────────────────────────────────────────────────────

def build_operating(V: Values) -> list:
    def l1(lbl, en, key=None):
        v25, v24 = V(key or lbl)
        return N(lbl, en, 1, v25, v24)
    def l0(lbl, en, key=None):
        v25, v24 = V(key or lbl)
        return N(lbl, en, 0, v25, v24)
    return [
        l0("Pendapatan bunga", "Interest income"),
        l0("Beban bunga", "Interest expenses"),
        l0("Pendapatan pengelolaan dana oleh bank sebagai mudharib",
           "Revenue from fund management as mudharib"),
        l0("Hak pihak ketiga atas bagi hasil dana syirkah temporer",
           "Third parties share on return of temporary syirkah funds"),
        N("Pendapatan asuransi", "Insurance income", 0, children=[
            l1("Pendapatan dari premi asuransi", "Revenue from insurance premiums"),
            l1("Premi reasuransi", "Reinsurance premiums"),
            l1("Premi retrosesi", "Retrocession premiums"),
            l1("Penurunan (kenaikan) premi yang belum merupakan pendapatan",
               "Decrease (increase) in unearned premiums"),
            l1("Penurunan (kenaikan) pendapatan premi disesikan kepada reasuradur",
               "Decrease (increase) in premium income ceded to reinsurancer"),
            l1("Pendapatan komisi asuransi", "Insurance commission income"),
            l1("Pendapatan bersih investasi", "Net investment income"),
            l1("Penerimaan ujrah", "Ujrah received"),
            l1("Pendapatan asuransi lainnya", "Other insurance income"),
        ]),
        N("Beban asuransi", "Insurance expenses", 0, children=[
            l1("Beban klaim", "Claim expenses"),
            l1("Klaim reasuransi", "Reinsurance claims"),
            l1("Klaim retrosesi", "Retrocession claims"),
            l1("Kenaikan (penurunan) estimasi liabilitas klaim",
               "Increase (decrease) in estimated claims liability"),
            l1("Kenaikan (penurunan) liabilitas manfaat polis masa depan",
               "Increase (decrease) in liability for future policy benefit"),
            l1("Kenaikan (penurunan) provisi yang timbul dari tes kecukupan liabilitas",
               "Increase (decrease) in provision for losses arising from liability adequacy test"),
            l1("Kenaikan (penurunan) liabilitas asuransi yang disesikan kepada reasuradur",
               "Increase (decrease) in insurance liabilities ceded to reinsurers"),
            l1("Kenaikan (penurunan) liabilitas pemegang polis pada kontrak unit-linked",
               "Increase (decrease) in liabilities to policyholder in unit-linked contracts"),
            l1("Beban komisi asuransi", "Insurance commission expenses"),
            l1("Ujrah dibayar", "Ujrah paid"),
            l1("Beban akuisisi dari kontrak asuransi", "Acquisition costs of insurance contracts"),
            l1("Beban asuransi lainnya", "Other insurance expenses"),
        ]),
        N("Pendapatan dari pembiayaan", "Financing income", 0, children=[
            l1("Pendapatan dari pembiayaan konsumen", "Revenue from consumer financing"),
            l1("Pendapatan dari sewa pembiayaan", "Revenue from finance lease"),
            l1("Pendapatan dari sewa operasi", "Revenue from operating lease"),
            l1("Pendapatan dari anjak piutang", "Revenue from factoring"),
        ]),
        N("Pendapatan sekuritas", "Securities income", 0, children=[
            l1("Pendapatan kegiatan penjamin emisi dan penjualan efek",
               "Revenue from underwriting activities and selling fees"),
            l1("Pendapatan pembiayaan transaksi nasabah", "Revenue from financing transactions"),
            l1("Pendapatan jasa biro administrasi efek",
               "Revenue from securities administration service"),
            l1("Pendapatan kegiatan jasa manajer investasi",
               "Revenue from investment management services"),
            l1("Pendapatan kegiatan jasa penasehat keuangan",
               "Revenue from financial advisory services"),
            l1("Keuntungan (kerugian) dari transaksi perdagangan efek yang telah direalisasi",
               "Realised gains (losses) on trading of marketable securities"),
            l1("Keuntungan (kerugian) perubahan nilai wajar efek",
               "Gains (losses) on changes in fair value of marketable securities"),
        ]),
        N("Pendapatan operasional lainnya", "Other operating income", 0, children=[
            l1("Pendapatan investasi", "Investments income"),
            l1("Pendapatan provisi dan komisi dari transaksi lainnya selain kredit",
               "Provisions and commissions income from transactions other than loan"),
            l1("Pendapatan transaksi perdagangan", "Revenue from trading transactions"),
            l1("Pendapatan dividen", "Dividends income"),
            l1("Keuntungan (kerugian) yang telah direalisasi atas instrumen derivatif",
               "Realised gains (losses) from derivative instruments"),
            l1("Penerimaan kembali aset yang telah dihapusbukukan",
               "Revenue from recovery of written-off assets"),
            l1("Keuntungan (kerugian) selisih kurs mata uang asing",
               "Gains (losses) on changes in foreign exchange rates"),
            l1("Keuntungan (kerugian) pelepasan aset tetap",
               "Gains (losses) on disposal of property and equipment"),
            l1("Keuntungan (kerugian) pelepasan agunan yang diambil alih",
               "Gains (losses) on disposal of foreclosed assets"),
            N("Pendapatan operasional lainnya", "Other operating income", 1,
              *V("leaf|Pendapatan operasional lainnya")),
        ]),
        N("Pemulihan penyisihan kerugian penurunan nilai",
          "Recovery of impairment loss", 0, children=[
            l1("Pemulihan penyisihan kerugian penurunan nilai aset keuangan",
               "Recovery of impairment loss of financial assets"),
            l1("Pemulihan penyisihan kerugian penurunan nilai aset keuangan - sewa pembiayaan",
               "Recovery of impairment loss of financial assets finance lease"),
            l1("Pemulihan penyisihan kerugian penurunan nilai aset keuangan - piutang pembiayaan konsumen",
               "Recovery of impairment loss of financial assets consumer financing receivables"),
            l1("Pemulihan penyisihan kerugian penurunan nilai aset non-keuangan",
               "Recovery of impairment loss of non-financial assets"),
            l1("Pemulihan penyisihan kerugian penurunan nilai aset non-keuangan - agunan yang diambil alih",
               "Recovery of impairment loss of non-financial assets repossessed collaterals"),
            l1("Pemulihan penyisihan estimasi kerugian atas komitmen dan kontinjensi",
               "Recovery of estimated loss of commitments and contingency"),
        ]),
        N("Pembentukan kerugian penurunan nilai", "Allowances for impairment losses", 0, children=[
            l1("Pembentukan penyisihan kerugian penurunan nilai aset produktif",
               "Allowances for impairment losses on earnings assets"),
            l1("Pembentukan penyisihan kerugian penurunan nilai aset non-produktif",
               "Allowances for impairment losses on non-earnings assets"),
        ]),
        l0("Pembalikan (beban) estimasi kerugian komitmen dan kontijensi",
           "Reversal (expense) of estimated losses on commitments and contingencies"),
        N("Beban operasional lainnya", "Other operating expenses", 0, children=[
            l1("Beban umum dan administrasi", "General and administrative expenses"),
            l1("Beban penjualan", "Selling expenses"),
            l1("Beban sewa, pemeliharaan, dan perbaikan",
               "Rent, maintenance and improvement expenses"),
            l1("Beban provisi dan komisi", "Other fees and commissions expenses"),
            N("Beban operasional lainnya", "Other operating expenses", 1,
              *V("leaf|Beban operasional lainnya")),
        ]),
        l0("Jumlah laba operasional", "Total profit from operation"),
    ]


def build_non_operating(V: Values) -> list:
    def l0(lbl, en, key=None):
        v25, v24 = V(key or lbl)
        return N(lbl, en, 0, v25, v24)
    return [
        l0("Pendapatan bukan operasional", "Non-operating income"),
        l0("Beban bukan operasional", "Non-operating expenses"),
        l0("Bagian atas laba (rugi) entitas asosiasi yang dicatat dengan menggunakan metode ekuitas",
           "Share of profit (loss) of associates accounted for using equity method"),
        l0("Bagian atas laba (rugi) entitas ventura bersama yang dicatat menggunakan metode ekuitas",
           "Share of profit (loss) of joint ventures accounted for using equity method"),
        l0("Jumlah laba (rugi) sebelum pajak penghasilan", "Total profit (loss) before tax"),
        l0("Pendapatan (beban) pajak", "Tax benefit (expenses)"),
        l0("Jumlah laba (rugi) dari operasi yang dilanjutkan",
           "Total profit (loss) from continuing operations"),
        l0("Laba (rugi) dari operasi yang dihentikan", "Profit (loss) from discontinued operations"),
        l0("Jumlah laba (rugi)", "Total profit (loss)"),
    ]


def build_eps(V: Values) -> list:
    def l1(lbl, en, key=None):
        v25, v24 = V(key or lbl)
        return N(lbl, en, 1, v25, v24)
    return [
        N("Laba per saham dasar diatribusikan kepada pemilik entitas induk",
          "Basic earnings per share attributable to equity owners of the parent entity", 0, children=[
            l1("Laba (rugi) per saham dasar dari operasi yang dilanjutkan",
               "Basic earnings (loss) per share from continuing operations"),
            l1("Laba (rugi) per saham dasar dari operasi yang dihentikan",
               "Basic earnings (loss) per share from discontinued operations"),
        ]),
        N("Laba (rugi) per saham dilusian", "Diluted earnings (loss) per share", 0, children=[
            l1("Laba (rugi) per saham dilusian dari operasi yang dilanjutkan",
               "Diluted earnings (loss) per share from continuing operations"),
            l1("Laba (rugi) per saham dilusian dari operasi yang dihentikan",
               "Diluted earnings (loss) per share from discontinued operations"),
        ]),
    ]


# ── OCI (parameterized: after-tax for BBNI/BBRI, before-tax for BMRI) ───────

def build_oci(V: Values, before_tax: bool = False) -> list:
    """OCI section. before_tax=True uses BMRI [4322000] wording + 2 extra lines."""
    sfx = "sebelum pajak" if before_tax else "setelah pajak"
    en = "before tax" if before_tax else "after tax"

    def l1(lbl, en_lbl, key=None):
        v25, v24 = V(key or lbl)
        return N(lbl, en_lbl, 1, v25, v24)

    not_reclass = N(
        f"Pendapatan komprehensif lainnya yang tidak akan direklasifikasi ke laba rugi, {sfx}",
        f"Other comprehensive income that will not be reclassified to profit or loss, {en}",
        0, children=[
            l1(f"Pendapatan komprehensif lainnya atas keuntungan (kerugian) hasil revaluasi aset tetap, {sfx}",
               f"Other comprehensive income for gains (losses) on revaluation of property and equipment, {en}"),
            l1(f"Pendapatan komprehensif lainnya atas pengukuran kembali kewajiban manfaat pasti, {sfx}",
               f"Other comprehensive income for remeasurement of defined benefit obligation, {en}"),
            l1(f"Penyesuaian lainnya atas pendapatan komprehensif lainnya yang tidak akan direklasifikasi ke laba rugi, {sfx}",
               f"Other adjustments to other comprehensive income that will not be reclassified to profit or loss, {en}"),
            l1(f"Jumlah pendapatan komprehensif lainnya yang tidak akan direklasifikasi ke laba rugi, {sfx}",
               f"Total other comprehensive income that will not be reclassified to profit or loss, {en}"),
        ])

    reclass = N(
        f"Pendapatan komprehensif lainnya yang akan direklasifikasi ke laba rugi, {sfx}",
        f"Other comprehensive income that may be reclassified to profit or loss, {en}",
        0, children=[
            l1(f"Keuntungan (kerugian) selisih kurs penjabaran, {sfx}",
               f"Gains (losses) on exchange differences on translation, {en}"),
            l1(f"Penyesuaian reklasifikasi selisih kurs penjabaran, {sfx}",
               f"Reclassification adjustments on exchange differences on translation, {en}"),
            l1(f"Keuntungan (kerugian) yang belum direalisasi atas perubahan nilai wajar aset keuangan melalui penghasilan komprehensif lain, {sfx}",
               f"Unrealised gains (losses) on changes in fair value through other comprehensive income financial assets, {en}"),
            l1(f"Penyesuaian reklasifikasi atas aset keuangan nilai wajar melalui pendapatan komprehensif lainnya, {sfx}",
               f"Reclassification adjustments on fair value through other comprehensive income financial assets, {en}"),
            l1(f"Keuntungan (kerugian) lindung nilai arus kas, {sfx}",
               f"Gains (losses) on cash flow hedges, {en}"),
            l1(f"Penyesuaian reklasifikasi atas lindung nilai arus kas, {sfx}",
               f"Reclassification adjustments on cash flow hedges, {en}"),
            l1(f"Nilai tercatat dari aset (liabilitas) non-keuangan yang perolehan atau keterjadiannya merupakan suatu prakiraan transaksi yang kemungkinan besar terjadi yang dilindung nilai, {sfx}",
               f"Carrying amount of non-financial asset (liability) whose acquisition or incurrence was hedged on highly probable forecast transaction, adjusted from equity, {en}"),
            l1(f"Keuntungan (kerugian) lindung nilai investasi bersih kegiatan usaha luar negeri, {sfx}",
               f"Gains (losses) on hedges of net investments in foreign operations, {en}"),
            l1(f"Penyesuaian reklasifikasi atas lindung nilai investasi bersih kegiatan usaha luar negeri, {sfx}",
               f"Reclassification adjustments on hedges of net investments in foreign operations, {en}"),
            l1(f"Bagian pendapatan komprehensif lainnya dari entitas asosiasi yang dicatat dengan menggunakan metode ekuitas, {sfx}",
               f"Share of other comprehensive income of associates accounted for using equity method, {en}"),
            l1(f"Bagian pendapatan komprehensif lainnya dari entitas ventura bersama yang dicatat dengan menggunakan metode ekuitas, {sfx}",
               f"Share of other comprehensive income of joint ventures accounted for using equity method, {en}"),
            l1(f"Penyesuaian lainnya atas pendapatan komprehensif lainnya yang akan direklasifikasi ke laba rugi, {sfx}",
               f"Other adjustments to other comprehensive income that may be reclassified to profit or loss, {en}"),
            l1(f"Jumlah pendapatan komprehensif lainnya yang akan direklasifikasi ke laba rugi, {sfx}",
               f"Total other comprehensive income that may be reclassified to profit or loss, {en}"),
        ])

    out = [not_reclass, reclass]
    if before_tax:
        out.append(N("Jumlah pendapatan komprehensif lainnya, sebelum pajak",
                      "Total other comprehensive income, before tax", 0,
                      *V("Jumlah pendapatan komprehensif lainnya, sebelum pajak")))
        out.append(N("Pajak atas pendapatan komprehensif lainnya",
                      "Tax on other comprehensive income", 0,
                      *V("Pajak atas pendapatan komprehensif lainnya")))
    out.append(N("Jumlah pendapatan komprehensif lainnya, setelah pajak",
                  "Total other comprehensive income, after tax", 0,
                  *V("Jumlah pendapatan komprehensif lainnya, setelah pajak")))
    out.append(N("Jumlah laba rugi komprehensif", "Total comprehensive income", 0,
                 *V("Jumlah laba rugi komprehensif")))
    out.append(N("Laba (rugi) yang dapat diatribusikan", "Profit (loss) attributable to", 0, children=[
        N("Laba (rugi) yang dapat diatribusikan ke entitas induk",
          "Profit (loss) attributable to parent entity", 1,
          *V("Laba (rugi) yang dapat diatribusikan ke entitas induk")),
        N("Laba (rugi) yang dapat diatribusikan ke kepentingan non-pengendali",
          "Profit (loss) attributable to non-controlling interests", 1,
          *V("Laba (rugi) yang dapat diatribusikan ke kepentingan non-pengendali")),
    ]))
    out.append(N("Laba rugi komprehensif yang dapat diatribusikan",
                 "Comprehensive income attributable to", 0, children=[
        N("Laba rugi komprehensif yang dapat diatribusikan ke entitas induk",
          "Comprehensive income attributable to parent entity", 1,
          *V("Laba rugi komprehensif yang dapat diatribusikan ke entitas induk")),
        N("Laba rugi komprehensif yang dapat diatribusikan ke kepentingan non-pengendali",
          "Comprehensive income attributable to non-controlling interests", 1,
          *V("Laba rugi komprehensif yang dapat diatribusikan ke kepentingan non-pengendali")),
    ]))
    return out


def build_output(source_pdf, issuer, issuer_full, values, oci_before_tax=False) -> dict:
    V = make_values_lookup(values)
    return {
        "schema_version": "1.0",
        "source_pdf": source_pdf,
        "approach": "ground_truth",
        "meta": {
            "issuer": issuer, "issuer_full": issuer_full, "fiscal_year": 2025,
            "rounding": "in_million_idr", "transcribed_pages": list(range(4, 20)),
            "transcribed_by": "manual",
            "oci_basis": "before_tax" if oci_before_tax else "after_tax",
        },
        "statements": [
            {
                "type": "financial_position", "years": [2025, 2024],
                "pages": list(range(4, 13)),
                "sections": {
                    "assets": build_assets(V),
                    "liabilities": build_liabilities(V),
                    "temporary_syirkah_funds": build_syirkah(V),
                    "equity": build_equity(V),
                },
            },
            {
                "type": "profit_or_loss", "years": [2025, 2024],
                "pages": list(range(13, 20)),
                "sections": {
                    "operating": build_operating(V),
                    "non_operating": build_non_operating(V),
                    "other_comprehensive_income": build_oci(V, before_tax=oci_before_tax),
                    "earnings_per_share": build_eps(V),
                },
            },
        ],
    }
