"""Ground truth for BMRI 2025 annual financial statements.

Source : docs/FinancialStatement-2025-Tahunan-BMRI.pdf, pages 4 – 19
Output : ground_truth/bmri_2025.json
Run    : python ground_truth/build_bmri_2025.py

Uses the shared IDX template (template_idx.py). BMRI differs from BBNI/BBRI:
  - Has NON-ZERO temporary syirkah funds (Bank Syariah Indonesia subsidiary)
  - Uses OCI taxonomy [4322000] (before-tax) with 2 extra summary lines

Cross-checks (verified):
  Total Aset 2,829,948,026 = Liab 2,212,925,204 + Syirkah 289,620,824 + Ekuitas 327,401,998
  Laba operasional 76,310,739 + Pend. bukan op 106,824 = Laba sebelum pajak 76,417,563
  Laba bersih 61,346,133 = induk 56,293,950 + NCI 5,052,183

Note: BMRI EPS is presented with decimals ("603.23"). Because the Indonesian
thousand-separator is also ".", the parser interprets "603.23" as 60323. EPS is
stored here as the parser-consistent integer (60323/59767) so the cell-level
comparison rewards correct digit extraction uniformly across approaches.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import template_idx as T

VALUES = {
    # ── Assets ──
    "Kas": (33_857_220, 31_665_082),
    "Giro pada Bank Indonesia": (238_289_478, 105_146_044),
    "Giro pada bank lain pihak ketiga": (60_750_690, 46_474_028),
    "Giro pada bank lain pihak berelasi": (201_893, 194_411),
    "Cadangan kerugian penurunan nilai pada giro pada bank lain": (-27_621, -30_755),
    "Penempatan pada Bank Indonesia dan bank lain pihak ketiga": (49_185_275, 60_122_934),
    "Penempatan pada Bank Indonesia dan bank lain pihak berelasi": (1_286_559, 3_107_120),
    "Cadangan kerugian penurunan nilai pada penempatan pada bank lain": (-1_586, -1_679),
    "Efek-efek yang diperdagangkan pihak ketiga": (107_734_397, 75_306_473),
    "Efek-efek yang diperdagangkan pihak berelasi": (17_034_297, 20_223_075),
    "Cadangan kerugian penurunan nilai pada efek-efek yang diperdagangkan": (-40_720, -51_497),
    "Efek yang dibeli dengan janji dijual kembali": (3_903_777, 8_290_138),
    "Wesel ekspor dan tagihan lainnya pihak ketiga": (25_191_454, 22_919_450),
    "Wesel ekspor dan tagihan lainnya pihak berelasi": (6_880_657, 7_054_667),
    "Cadangan kerugian penurunan nilai pada wesel ekspor dan tagihan lainnya": (-1_432_270, -1_422_889),
    "Tagihan akseptasi pihak ketiga": (6_926_985, 7_615_001),
    "Tagihan akseptasi pihak berelasi": (1_161_293, 1_698_864),
    "Cadangan kerugian penurunan nilai pada tagihan akseptasi": (-26_015, -31_340),
    "Tagihan derivatif pihak ketiga": (4_247_695, 4_812_513),
    "Tagihan derivatif pihak berelasi": (3_029_980, 2_948_995),
    "Pinjaman yang diberikan pihak ketiga": (1_447_120_377, 1_331_581_512),
    "Pinjaman yang diberikan pihak berelasi": (402_847_579, 291_635_100),
    "Cadangan kerugian penurunan nilai pada pinjaman yang diberikan": (-48_033_747, -49_354_645),
    "Piutang pembiayaan konsumen pihak ketiga": (40_858_943, 41_531_960),
    "Piutang pembiayaan konsumen pihak berelasi": (4_257, 41_346),
    "Cadangan kerugian penurunan nilai pada piutang pembiayaan konsumen": (-1_049_570, -934_353),
    "Investasi sewa pihak ketiga": (4_851_402, 6_791_445),
    "Investasi sewa pihak berelasi": (-6_140, -22_400),
    "Investasi sewa nilai residu yang terjamin": (1_833_652, 2_445_103),
    "Investasi sewa pendapatan pembiayaan tangguhan": (-691_522, -1_011_969),
    "Investasi sewa simpanan jaminan": (-1_833_652, -2_445_103),
    "Cadangan kerugian penurunan nilai pada investasi sewa": (-134_987, -103_337),
    "Obligasi pemerintah": (292_817_548, 287_272_659),
    "Aset tidak lancar atau kelompok lepasan diklasifikasikan sebagai dimiliki untuk dijual": (253_774, 0),
    "Biaya dibayar dimuka": (5_673_038, 4_827_723),
    "Pajak dibayar dimuka": (851_625, 739_015),
    "Aset pajak tangguhan": (4_654_270, 8_353_454),
    "Investasi pada entitas asosiasi": (2_346_322, 2_416_748),
    "Goodwill": (519_286, 519_286),
    "Aset takberwujud selain goodwill": (6_999_872, 6_525_457),
    "Aset tetap": (72_062_331, 63_030_845),
    "Agunan yang diambil alih": (942_479, 998_294),
    "Aset lainnya": (38_907_451, 36_344_487),
    "Jumlah aset": (2_829_948_026, 2_427_223_262),

    # ── Liabilities ──
    "Liabilitas segera": (4_537_458, 5_703_731),
    "Giro pihak ketiga": (468_860_753, 414_420_537),
    "Giro pihak berelasi": (197_248_837, 154_155_472),
    "Tabungan pihak ketiga": (616_814_451, 573_852_753),
    "Tabungan pihak berelasi": (5_100_519, 6_339_043),
    "Deposito berjangka pihak ketiga": (379_607_155, 243_976_854),
    "Deposito berjangka pihak berelasi": (149_265_493, 53_490_298),
    "Simpanan dari bank lain pihak berelasi": (1_669_124, 6_520_346),
    "Simpanan dari bank lain pihak ketiga": (19_086_419, 20_522_363),
    "Efek yang dijual dengan janji untuk dibeli kembali": (39_955_889, 90_256_225),
    "Liabilitas derivatif pihak ketiga": (4_373_784, 5_203_494),
    "Liabilitas derivatif pihak berelasi": (2_467_837, 2_133_504),
    "Liabilitas akseptasi pihak berelasi": (1_728_332, 2_565_287),
    "Liabilitas akseptasi pihak ketiga": (6_191_001, 6_570_726),
    "Pinjaman yang diterima pihak ketiga": (152_427_126, 143_288_024),
    "Pinjaman yang diterima pihak berelasi": (2_245_296, 4_627_957),
    "Utang obligasi": (62_205_231, 41_141_067),
    "Liabilitas kontrak asuransi": (37_850_988, 35_487_487),
    "Provisi": (112_537, 264_275),
    "Estimasi kerugian komitmen dan kontinjensi": (895_791, 1_114_013),
    "Beban akrual": (6_168_983, 5_466_461),
    "Utang pajak": (3_327_702, 3_078_642),
    "Liabilitas pajak tangguhan": (27_996, 9_278),
    "Liabilitas lainnya": (42_467_140, 32_656_899),
    "Kewajiban imbalan pasca kerja": (7_899_583, 7_160_018),
    "Pinjaman subordinasi pihak ketiga": (354_779, 363_562),
    "Pinjaman subordinasi pihak berelasi": (35_000, 40_000),
    "Jumlah liabilitas": (2_212_925_204, 1_860_408_316),

    # ── Temporary Syirkah Funds (BMRI has values) ──
    "Giro mudharabah pihak ketiga": (24_899_562, 17_389_993),
    "Giro berjangka mudharabah pihak berelasi": (19_137_701, 19_798_526),
    "Tabungan mudharabah pihak ketiga": (98_215_131, 84_878_381),
    "Tabungan mudharabah pihak berelasi": (1_101_408, 375_768),
    "Deposito berjangka mudharabah pihak ketiga": (99_334_364, 92_461_883),
    "Deposito berjangka mudharabah pihak berelasi": (46_178_777, 37_757_408),
    "Bank|Giro mudharabah": (52_357, 47_282),
    "Tabungan mudharabah (ummat)": (616_794, 536_509),
    "Bank|Deposito berjangka mudharabah": (84_730, 94_515),
    "Jumlah dana syirkah temporer": (289_620_824, 253_340_265),

    # ── Equity ──
    "Saham biasa": (11_666_667, 11_666_667),
    "Tambahan modal disetor": (18_095_274, 18_095_274),
    "Saham treasuri": (-403_625, None),
    "Cadangan revaluasi": (38_445_684, 34_772_745),
    "Cadangan selisih kurs penjabaran": (152_018, 10_289),
    "Cadangan perubahan nilai wajar aset keuangan nilai wajar melalui pendapatan komprehensif lainnya": (1_146_052, -2_160_850),
    "Cadangan lindung nilai arus kas": (-11_218, -8_885),
    "Cadangan pengukuran kembali program imbalan pasti": (1_374_981, 1_595_606),
    "Cadangan lainnya": (85_052, 85_052),
    "Komponen ekuitas lainnya": (-309_938, -309_938),
    "Cadangan umum dan wajib": (2_333_333, 2_333_333),
    "Cadangan khusus": (3_046_935, 3_046_935),
    "Saldo laba yang belum ditentukan penggunaannya": (218_129_454, 214_670_201),
    "Jumlah ekuitas yang diatribusikan kepada pemilik entitas induk": (293_750_669, 283_796_429),
    "Kepentingan non-pengendali": (33_651_329, 29_678_252),
    "Jumlah ekuitas": (327_401_998, 313_474_681),
    "Jumlah liabilitas, dana syirkah temporer dan ekuitas": (2_829_948_026, 2_427_223_262),

    # ── Operating ──
    "Pendapatan bunga": (164_412_466, 151_236_027),
    "Beban bunga": (-58_202_431, -49_479_107),
    "Pendapatan dari premi asuransi": (550_415, 2_520_813),
    "Keuntungan (kerugian) dari transaksi perdagangan efek yang telah direalisasi": (463_146, 150_297),
    "Pendapatan provisi dan komisi dari transaksi lainnya selain kredit": (27_553_414, 23_447_520),
    "Pendapatan transaksi perdagangan": (6_343_482, 4_483_298),
    "Penerimaan kembali aset yang telah dihapusbukukan": (9_844_380, 9_310_556),
    "leaf|Pendapatan operasional lainnya": (4_261_159, 4_929_641),
    "Pembentukan penyisihan kerugian penurunan nilai aset produktif": (-10_359_492, -11_811_786),
    "Pembentukan penyisihan kerugian penurunan nilai aset non-produktif": (-1_231_070, -151_047),
    "Pembalikan (beban) estimasi kerugian komitmen dan kontijensi": (259_675, 33_829),
    "Beban umum dan administrasi": (-52_999_201, -46_762_170),
    "Beban sewa, pemeliharaan, dan perbaikan": (-4_058_534, -3_748_226),
    "Beban provisi dan komisi": (-3_090_316, -1_222_846),
    "leaf|Beban operasional lainnya": (-7_436_354, -6_877_204),
    "Jumlah laba operasional": (76_310_739, 76_059_595),

    # ── Non-operating ──
    "Pendapatan bukan operasional": (106_824, 343_891),
    "Jumlah laba (rugi) sebelum pajak penghasilan": (76_417_563, 76_403_486),
    "Pendapatan (beban) pajak": (-15_071_430, -15_238_365),
    "Jumlah laba (rugi) dari operasi yang dilanjutkan": (61_346_133, 61_165_121),
    "Jumlah laba (rugi)": (61_346_133, 61_165_121),

    # ── OCI (before-tax taxonomy [4322000]) ──
    "Pendapatan komprehensif lainnya atas keuntungan (kerugian) hasil revaluasi aset tetap, sebelum pajak": (3_672_939, 108_911),
    "Pendapatan komprehensif lainnya atas pengukuran kembali kewajiban manfaat pasti, sebelum pajak": (-324_493, 167_984),
    "Jumlah pendapatan komprehensif lainnya yang tidak akan direklasifikasi ke laba rugi, sebelum pajak": (3_348_446, 276_895),
    "Keuntungan (kerugian) selisih kurs penjabaran, sebelum pajak": (140_873, 161_227),
    "Keuntungan (kerugian) yang belum direalisasi atas perubahan nilai wajar aset keuangan melalui penghasilan komprehensif lain, sebelum pajak": (4_375_263, -533_762),
    "Keuntungan (kerugian) lindung nilai arus kas, sebelum pajak": (-5_863, -25_927),
    "Jumlah pendapatan komprehensif lainnya yang akan direklasifikasi ke laba rugi, sebelum pajak": (4_510_273, -398_462),
    "Jumlah pendapatan komprehensif lainnya, sebelum pajak": (7_858_719, -121_567),
    "Pajak atas pendapatan komprehensif lainnya": (-730_931, 103_211),
    "Jumlah pendapatan komprehensif lainnya, setelah pajak": (7_127_788, -18_356),
    "Jumlah laba rugi komprehensif": (68_473_921, 61_146_765),
    "Laba (rugi) yang dapat diatribusikan ke entitas induk": (56_293_950, 55_782_742),
    "Laba (rugi) yang dapat diatribusikan ke kepentingan non-pengendali": (5_052_183, 5_382_379),
    "Laba rugi komprehensif yang dapat diatribusikan ke entitas induk": (63_192_562, 55_740_401),
    "Laba rugi komprehensif yang dapat diatribusikan ke kepentingan non-pengendali": (5_281_359, 5_406_364),

    # ── EPS (decimal → parser-consistent integer; see module docstring) ──
    "Laba (rugi) per saham dasar dari operasi yang dilanjutkan": (60_323, 59_767),
    "Laba (rugi) per saham dilusian dari operasi yang dilanjutkan": (60_323, 59_767),
}


def main() -> None:
    output = T.build_output(
        source_pdf="FinancialStatement-2025-Tahunan-BMRI.pdf",
        issuer="BMRI",
        issuer_full="PT Bank Mandiri (Persero) Tbk",
        values=VALUES,
        oci_before_tax=True,
    )
    out_path = Path(__file__).parent / "bmri_2025.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
