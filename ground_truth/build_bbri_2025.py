"""Ground truth for BBRI 2025 annual financial statements.

Source : docs/FinancialStatement-2025-Tahunan-BBRI.pdf, pages 4 – 19
Output : ground_truth/bbri_2025.json
Run    : python ground_truth/build_bbri_2025.py

Uses the shared IDX template (template_idx.py) which is validated to reproduce
BBNI byte-for-byte. Only the per-bank values differ. Values transcribed manually
from the PDF text layer. Empty (None) values are template defaults.

Cross-checks (verified):
  Total Aset 2,135,371,105 = Liabilitas 1,804,429,671 + Syirkah 0 + Ekuitas 330,941,434
  Laba operasional 73,247,881 + Pend. bukan op (-455,062) = Laba sebelum pajak 72,792,819
  Laba bersih 57,132,365 = induk 56,652,384 + NCI 479,981
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import template_idx as T

VALUES = {
    # ── Assets ──
    "Kas": (32_044_482, 29_783_642),
    "Giro pada Bank Indonesia": (31_929_608, 88_878_969),
    "Giro pada bank lain pihak ketiga": (41_954_524, 25_078_088),
    "Giro pada bank lain pihak berelasi": (490_057, 504_737),
    "Cadangan kerugian penurunan nilai pada giro pada bank lain": (-11_273, -8_378),
    "Penempatan pada Bank Indonesia dan bank lain pihak ketiga": (18_285_606, 52_680_605),
    "Penempatan pada Bank Indonesia dan bank lain pihak berelasi": (2_771_604, 5_193_730),
    "Cadangan kerugian penurunan nilai pada penempatan pada bank lain": (-2_405, -767),
    "Efek-efek yang diperdagangkan pihak ketiga": (123_728_979, 87_391_531),
    "Efek-efek yang diperdagangkan pihak berelasi": (249_003_823, 239_144_169),
    "Cadangan kerugian penurunan nilai pada efek-efek yang diperdagangkan": (-89_519, -58_823),
    "Efek yang dibeli dengan janji dijual kembali": (24_452, 16_845_690),
    "Wesel ekspor dan tagihan lainnya pihak ketiga": (40_664_125, 36_793_795),
    "Wesel ekspor dan tagihan lainnya pihak berelasi": (7_587_958, 3_863_027),
    "Cadangan kerugian penurunan nilai pada wesel ekspor dan tagihan lainnya": (-465_498, -1_075_559),
    "Tagihan akseptasi pihak ketiga": (10_956_139, 8_049_731),
    "Tagihan akseptasi pihak berelasi": (2_122_428, 2_055_642),
    "Cadangan kerugian penurunan nilai pada tagihan akseptasi": (-32_226, -321_683),
    "Tagihan derivatif pihak ketiga": (1_167_029, 1_087_048),
    "Pinjaman yang diberikan pihak ketiga": (1_305_958_960, 1_211_272_379),
    "Pinjaman yang diberikan pihak berelasi": (154_770_458, 87_045_710),
    "Cadangan kerugian penurunan nilai pada pinjaman yang diberikan": (-79_328_619, -76_902_889),
    "Piutang lainnya pihak ketiga": (60_724_493, 56_280_498),
    "Piutang lainnya pihak berelasi": (31_946, 42_192),
    "Cadangan kerugian penurunan nilai pada piutang lainnya": (-3_730_037, -4_160_622),
    "Aset pajak tangguhan": (8_129_522, 12_800_660),
    "Investasi pada entitas ventura bersama": (1_249_588, 1_550_814),
    "Investasi pada entitas asosiasi": (7_585_280, 6_525_753),
    "Aset tetap": (63_294_240, 62_477_965),
    "Aset lainnya": (54_555_381, 39_369_252),
    "Jumlah aset": (2_135_371_105, 1_992_186_906),

    # ── Liabilities ──
    "Liabilitas segera": (39_818_745, 36_821_661),
    "Giro pihak ketiga": (199_866_427, 236_443_958),
    "Giro pihak berelasi": (248_337_243, 138_110_382),
    "Tabungan pihak ketiga": (585_559_949, 544_142_091),
    "Tabungan pihak berelasi": (2_025_913, 284_856),
    "Deposito berjangka pihak ketiga": (277_168_008, 327_804_120),
    "Deposito berjangka pihak berelasi": (153_886_299, 118_664_697),
    "Simpanan dari bank lain pihak berelasi": (1_375_788, 1_156_791),
    "Simpanan dari bank lain pihak ketiga": (16_225_648, 13_522_691),
    "Efek yang dijual dengan janji untuk dibeli kembali": (27_932_749, 25_043_717),
    "Liabilitas derivatif pihak ketiga": (1_101_753, 1_585_120),
    "Liabilitas akseptasi pihak ketiga": (13_078_567, 10_105_373),
    "Pinjaman yang diterima pihak ketiga": (92_649_347, 91_213_516),
    "Pinjaman yang diterima pihak berelasi": (36_536_769, 36_666_288),
    "Utang obligasi": (40_901_648, 32_502_499),
    "Estimasi kerugian komitmen dan kontinjensi": (1_946_135, 2_551_050),
    "Utang pajak": (3_299_310, 2_150_487),
    "Liabilitas lainnya": (34_804_195, 28_674_477),
    "Kewajiban imbalan pasca kerja": (27_428_311, 20_936_335),
    "Pinjaman subordinasi pihak ketiga": (257_538, 262_663),
    "Pinjaman subordinasi pihak berelasi": (229_329, 229_118),
    "Jumlah liabilitas": (1_804_429_671, 1_668_871_890),

    # ── Equity ──
    "Saham biasa": (7_577_950, 7_577_950),
    "Tambahan modal disetor": (75_946_195, 75_880_223),
    "Saham treasuri": (-4_463_270, -4_349_007),
    "Opsi saham": (575_039, 765_435),
    "Cadangan revaluasi": (20_754_251, 20_222_379),
    "Cadangan selisih kurs penjabaran": (227_059, -204_632),
    "Cadangan perubahan nilai wajar aset keuangan nilai wajar melalui pendapatan komprehensif lainnya": (60_966, 51_931),
    "Cadangan pengukuran kembali program imbalan pasti": (-2_198_095, -505_787),
    "Komponen ekuitas lainnya": (2_890_385, -442_530),
    "Cadangan umum dan wajib": (3_022_685, 3_022_685),
    "Saldo laba yang belum ditentukan penggunaannya": (219_640_693, 215_009_704),
    "Jumlah ekuitas yang diatribusikan kepada pemilik entitas induk": (324_033_858, 317_028_351),
    "Kepentingan non-pengendali": (6_907_576, 6_286_665),
    "Jumlah ekuitas": (330_941_434, 323_315_016),
    "Jumlah liabilitas, dana syirkah temporer dan ekuitas": (2_135_371_105, 1_992_186_906),

    # ── Operating ──
    "Pendapatan bunga": (207_783_368, 199_266_252),
    "Beban bunga": (-57_284_939, -56_607_595),
    "Pendapatan dari premi asuransi": (7_662_733, 7_346_611),
    "Beban klaim": (-6_364_550, -6_179_801),
    "Keuntungan (kerugian) dari transaksi perdagangan efek yang telah direalisasi": (3_372_396, 2_209_474),
    "Keuntungan (kerugian) perubahan nilai wajar efek": (160_659, None),
    "Pendapatan provisi dan komisi dari transaksi lainnya selain kredit": (21_447_045, 20_390_833),
    "Penerimaan kembali aset yang telah dihapusbukukan": (20_952_308, 25_363_951),
    "Keuntungan (kerugian) selisih kurs mata uang asing": (2_087_065, 1_187_802),
    "leaf|Pendapatan operasional lainnya": (8_060_685, 5_499_466),
    "Pemulihan penyisihan kerugian penurunan nilai aset non-keuangan": (-82_414, -13_008),
    "Pemulihan penyisihan estimasi kerugian atas komitmen dan kontinjensi": (624_058, 3_596_482),
    "Pembentukan penyisihan kerugian penurunan nilai aset produktif": (-46_723_647, -41_744_402),
    "Beban umum dan administrasi": (-33_776_049, -29_288_456),
    "Beban provisi dan komisi": (-53_113, -93_037),
    "leaf|Beban operasional lainnya": (-54_617_724, -52_718_725),
    "Jumlah laba operasional": (73_247_881, 78_215_847),

    # ── Non-operating ──
    "Pendapatan bukan operasional": (-455_062, -963_653),
    "Jumlah laba (rugi) sebelum pajak penghasilan": (72_792_819, 77_252_194),
    "Pendapatan (beban) pajak": (-15_660_454, -16_945_848),
    "Jumlah laba (rugi) dari operasi yang dilanjutkan": (57_132_365, 60_306_346),
    "Jumlah laba (rugi)": (57_132_365, 60_306_346),

    # ── OCI (after-tax) ──
    "Jumlah pendapatan komprehensif lainnya, setelah pajak": (2_909_006, 1_564_472),
    "Jumlah laba rugi komprehensif": (60_041_371, 61_870_818),
    "Laba (rugi) yang dapat diatribusikan ke entitas induk": (56_652_384, 59_944_649),
    "Laba (rugi) yang dapat diatribusikan ke kepentingan non-pengendali": (479_981, 361_697),
    "Laba rugi komprehensif yang dapat diatribusikan ke entitas induk": (59_265_589, 61_620_303),
    "Laba rugi komprehensif yang dapat diatribusikan ke kepentingan non-pengendali": (775_782, 250_515),

    # ── EPS ──
    "Laba (rugi) per saham dasar dari operasi yang dilanjutkan": (376, 398),
}


def main() -> None:
    output = T.build_output(
        source_pdf="FinancialStatement-2025-Tahunan-BBRI.pdf",
        issuer="BBRI",
        issuer_full="PT Bank Rakyat Indonesia (Persero) Tbk",
        values=VALUES,
        oci_before_tax=False,
    )
    out_path = Path(__file__).parent / "bbri_2025.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
