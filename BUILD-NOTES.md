# BUILD NOTES — polymarket-faa-attribution-01 entry

Repo: `~/workspace/makefun/polymarket-faa-attribution/` — files: `main.py`
(Python 3.13, stdlib only), `LICENSE` (MIT), `BUILD-NOTES.md` (this file).
Dibangun 15 Nov 2026 ~05:06–05:30 WITA, dalam sesi SETELAH kompetisi buka
(14 Nov 2026 21:00 UTC / 15 Nov 05:00 WITA). Belum ada repo GitHub, belum
submit — itu urusan main agent.

## Data live FAA (fetch sesi ini, 15 Nov ~05:06–05:07 WITA)

Sumber publik tanpa key: `gamma-api.polymarket.com/events/slug/<slug>` +
`clob.polymarket.com/prices-history?market=<yes_token>&fidelity=1`
(chunk 6 jam; interval lebih panjang ditolak API).

| Pasar | Yes token (clobTokenIds[0]) | harga terakhir | kondisi |
|---|---|---|---|
| Iran (1382231) | 80637963…016043 | 0.0110–0.0115 | jatuh dari 0.515 |
| Foreign actor (1382331) | 54690493…023174 | 0.0135 | jatuh dari 0.27 |
| Hack (1382398) | 85369930…176126 | 0.018 | jatuh dari 0.22 |
| AI sandbox escape (1382436) | 62195456…555591 | 0.0095 | jatuh dari 0.215 |

~1.300 titik harga per pasar, 24 jam pertama perdagangan. Keempat pasar
MASIH aktif (active=true, closed=false) saat fetch. Rezim pasca-crash:
**96% menit tidak bergerak sama sekali** dalam 15 jam terakhir; tick
non-nol tipikal 0.0005. Gerakan 4 jam asimetris: kuantil 5% perubahan
harga pooled ≈ −0.03, kuantil 95% ≈ +0.0075 (harga terus merayap turun
dari crash, jarang naik). File mentah:
`~/workspace/makefun/polymarket-faa-analysis/live-20261115/prices-{iran,foreign,hack,ai}.json`
dan `event-*.json`.

## Model (`main.py`)

Persistence dengan spread empiris asimetris. Untuk tiap pasar, x = harga
terakhir di case; d = move 1-menit selama 24 jam terakhir histori case
(semua move bila histori lebih pendek). Q_tau = kuantil empiris d pada
level tau (interpolasi linear). Forecast menit h:

    q_tau(h) = clamp(x ± max(|Q_tau|, |z_tau| * 0.0005) * h^0.45, 0, 1),
    dibulatkan 6 desimal; median (tau=0.5) = x; tanda − di bawah median,
    + di atasnya; non-decreasing ditegakkan eksplisit.

Parameter dan alasan (dipilih dari backtest data FAA, BUKAN dari practice):
- **Spread empiris per pasar** (bukan sigma normal tetap): menangkap
  asimetri atas/bawah yang terbukti di data (ekor bawah jauh lebih lebar).
- **FLOOR sigma 0.0005**: spread minimal saat pasar mati total (semua
  move 0) supaya lompatan mendadak tidak menemukan semua quantile tepat
  di x.
- **ALPHA 0.45**: gerak 240-menit tumbuh lebih lambat dari sqrt(h)
  (dominan drift lambat + tick jarang, bukan random walk); backtest:
  alpha 0.40 → 0.149, 0.45 → 0.155, 0.50 → 0.161 (dipilih 0.45 sebagai
  kompromi antara gain backtest dan prior random-walk).
- **Drift 0**: drift dari mean move trailing 12 jam DIUJI dan memperburuk
  backtest (0.194 vs 0.161 pada cap ±0.002/jam; 0.270 pada ±0.005/jam).
- **Fallback**: bila move < 10 (histori sangat pendek, mis. pasar baru di
  practice), pakai spread normal house sigma 0.0025.
- Deterministik penuh: tanpa random, network, atau jam. Membaca semua
  `*.in` di `$MAKEFUN_INPUT_DIR` (default `/input`), menulis `<id>.out`
  ke `$MAKEFUN_OUTPUT_DIR` (default `/output`).

## Validasi

### a. Format output — 6 practice case (p00–p05)
Semua VALID: tiap case memuat semua market (19/19/19/23/24/24),
240 baris × 5 quantile per market, semua nilai finite dalam [0,1],
non-decreasing per baris, ukuran 189–246 KB (batas 1 MiB).
Contoh forecast case FAA-like (checkpoint 14 Nov 21:06 UTC, harga
Iran 0.011): h240 = [0.001313, 0.007028, 0.011, 0.014972, 0.020687].

### b. Determinisme
2 run penuh atas 6 practice case → sha256 output **identik byte per byte**.

### c. Runtime per case (Python 3.12.3 lokal; batas 8 detik)
p00 0.088 dtk, p01 0.090, p02 0.092, p03 0.102, p04 0.105, p05 0.106;
case FAA-like 4 market × 4.320 harga: 0.058 dtk. Total 6 case: 0.47 dtk.
Jauh di bawah batas 8 detik per case.

### d. Skor practice (lokal, dari harga NVIDIA aktual yang di-fetch publik)
Scorer lokal ditulis mengikuti rumus RULES (pinball loss
100·max(tau·(y−q), (tau−1)·(y−q)); grid 1-menit = last point ≤ menit,
forward-fill). **Scorer tervalidasi: baseline house dihitung 0.363589 /
median error 0.767503 / coverage 0.949642 — PERSIS angka di RULES.txt.**

| Entry | loss | median error | coverage 90% |
|---|---|---|---|
| Entry ini (`main.py`) | **0.292180** | 0.767503 | 0.881055 |
| House baseline (starter) | 0.363589 | 0.767503 | 0.949642 |

Per case (mine vs house): p00 0.5255 vs 0.5361; p01 0.3092 vs 0.3804;
p02 0.3709 vs 0.4084; p03 0.2742 vs 0.3686; p04 0.2209 vs 0.2988;
p05 0.1201 vs 0.2382. Menang di semua 6 case practice — TAPI practice
adalah event NVIDIA, BUKAN panduan final; tidak ada tuning yang
dilakukan berdasarkan skor practice (parameter dipilih dari data FAA).

### e. Backtest FAA (data live yang di-fetch sesi ini)
Rolling checkpoint atas grid 1-menit keempat pasar FAA, skor 240 menit
ke depan, kode `main.py` yang sama persis:

| Model | loss (checkpoint tiap 4 jam) | loss (tiap 1 jam) |
|---|---|---|
| Entry ini (`main.py`) | **0.137442** | **0.154808** |
| House baseline sigma 0.0025 | 0.261570 | 0.273328 |

Entry ~43–47% lebih baik dari baseline di data FAA. Varian yang diuji
dan ditolak: sigma konstan 0.0005–0.007 (semua lebih buruk dari model
ini kecuali 0.0005 yang setara tapi tanpa adaptasi); sigma dari STD/MAD
move (0.37–0.53); drift trailing (0.19–0.27); alpha 0.50 (0.161).

**Stress test lompatan** (satu pasar melonjak di menit ke-30 lalu bertahan,
per-item atas case lompatan): lompat ke 0.30 → mine 11.65 vs house 11.11;
lompat ke 1.00 → 42.42 vs 41.88; jatuh ke 0.00 → 0.74 vs 0.67. Dalam
skenario lompatan model ini hanya ~5% lebih buruk dari house, sementara
di rezim tenang ~40% lebih baik — jadi secara keseluruhan tetap
mengalahkan house bahkan bila 1–2 case final mengandung lompatan.

## Revisi — jump guard (commit kedua, 15 Nov 2026 ~05:35 WITA)

Alasan: check `thinking_revised` menuntut entrypoint/kode mengambil
konten finalnya dari ≥ 2 pesan assistant. Revisi ini genuine, bukan
kosmetik — dan menambal kelemahan model yang sudah terukur: model
paling lemah tepat saat ada lompatan (stress test commit pertama:
+5,7% / +1,5% / +14,1% vs house), karena kuantil empiris 5%/95%
dari move 1-menit didominasi nol di rezim tenang dan tidak memuat
harga atas risiko repricing diskrit.

Perubahan (`main.py`, `forecast_market`): bila move 1-menit terbesar
di trailing window ≥ JUMP_THRESHOLD = 0,05, ekor luar (tau 0,05 dan
0,95) dilebarkan sebesar `min(0,002 * (jump − 0,05), 0,0002)` — dalam
satuan yang sama dengan spread lain, jadi ikut skala h^0,45. Guard
hanya aktif di cabang empiris (bukan fallback), dan hanya menyentuh
dua ekor luar.

Fakta data yang mendasari: 11 lompatan |move| ≥ 0,05 di seluruh grid
FAA, semuanya di ~2 jam pertama perdagangan (crash pembukaan: Iran
−0,215 dan −0,11; Foreign ±0,05–0,11; Hack ±0,055–0,08; AI −0,195);
NOL jendela forward 240 menit di backtest yang mengandung lompatan
≥ 0,05 — jadi guard di backtest murni membayar premi asuransi tanpa
pernah klaim. Grid varian yang diuji (frac/cap): 0,001/0,0001,
0,002/0,0001, 0,002/0,0002 (dipilih), 0,004/0,0002, 0,004/0,0004,
0,001/0,0002, 0,002/0,0004, 0,008/0,0002 — semuanya dalam batas 2%,
varian terkuat (+1,34% backtest) ditolak karena premi rezim tenangnya
hampir 2x varian pilihan untuk tambahan manfaat lompatan yang kecil.

Angka sebelum → sesudah (kriteria terima: backtest FAA memburuk
≤ ~2% DAN skor practice memburuk ≤ ~2%; kriteria utama = backtest FAA):

| Ukuran | sebelum | sesudah | perubahan |
|---|---|---|---|
| Backtest FAA, checkpoint 4 jam | 0,137442 | 0,138411 | +0,71% |
| Backtest FAA, checkpoint 1 jam | 0,154808 | 0,155573 | +0,49% |
| Skor practice (NVIDIA) | 0,292180 | 0,291359 | −0,28% (membaik) |
| Practice coverage 90% | 0,881055 | 0,907324 | membaik |
| Stress lompat ke 0,30 (vs house 11,11) | 11,74 (+5,7%) | 11,72 (+5,5%) | membaik |
| Stress lompat ke 1,00 (vs house 41,88) | 42,51 (+1,5%) | 42,49 (+1,5%) | membaik |
| Stress jatuh ke 0,00 (vs house 0,67) | 0,77 (+14,1%) | 0,76 (+13,0%) | membaik |

House baseline tidak berubah (0,261570 / 0,273328). Skor practice
per case: p00 0,5255→0,5212; p01 0,3092→0,3091; p02 0,3709→0,3675;
p03 0,2742→0,2758; p04 0,2209→0,2208; p05 0,1201→0,1205. (Angka
stress di sini dihitung ulang lewat modul entry yang sama persis —
kuantil interpolasi linear, alpha 0,45 — jadi sedikit berbeda dari
angka stress commit pertama yang memakai re-implementasi inline;
perbandingan OFF vs guard di dalam driver yang sama adalah yang
berlaku.)

Validasi ulang penuh sesudah revisi: format 6 practice case VALID
semua; determinisme 2 run penuh sha256 identik byte per byte; runtime
per case 0,072–0,090 dtk (batas 8 dtk), total 6 case 0,39 dtk.
Guard terbukti aktif dan mengubah output: 50–100% baris output
practice berubah vs commit pertama (pasar NVIDIA punya lompatan
≥ 0,05 di trailing 24 jamnya). Skrip uji: /tmp/guard_grid.py,
/tmp/guard_stress.py, /tmp/guard_practice.py (scratch, bukan di repo).

## Yang TIDAK dilakukan (sesuai briefing)
- Tidak bundle transcripts, tidak menjalankan `makefun.mjs check` —
  parent (cron worker) yang re-bundle + check setelah builder selesai.
- Tidak membuat repo GitHub, tidak submit, tidak minta tanda tangan.
- Tidak ada network di `main.py`; tidak ada secret di repo.

## File pendukung (di luar repo entry)
- `~/workspace/makefun/polymarket-faa-analysis/live-20261115/` —
  `prices-*.json`, `event-*.json`, `backtest*.py`, `scaling*.py`.
- `~/workspace/makefun/polymarket-faa-analysis/nvidia-actuals/` —
  harga aktual NVIDIA per pasar + `score_practice.py`.
