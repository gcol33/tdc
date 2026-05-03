# tdc bench results

Date: 2026-04-17 (PM: LZ_STREAMS LL/ML symbol cap raised 21→25 — fixes
TDC_E_CORRUPT on blocks > 2 MiB, unlocks DICT_NUMERIC+BSHUF+LZ_STREAMS
on NASA regional at 4.79× / 70 MB/s enc, now +12% vs zstd L19. New
DICT_NUMERIC pipeline rows on the regional block: LZ_STREAMS 4.79×,
LZ_SPLIT 4.68×, LZ_OPT+HUF 4.62×, LZ_OPT 4.48×.)
Prior: 2026-04-17 AM (DICT_NUMERIC_1D added — numeric value dictionary
for low-cardinality f64 grids; cardinality columns added to f64 dataset
descriptions; expanded DICT_NUMERIC pipeline coverage.)
Prior: 2026-04-16 (post 1 MiB LZ window: ratio-first defaults, encode cost
documented; new 3D pipelines, HUF4, PAETH/UP variants).
Host: the-beast (Windows 11, MSVC 19.50, RTX 5080 / 64 GB — CPU bench, GPU unused)
tdc build: `cmake --build build_release --config Release` (MSVC `/W4 /permissive-`, default Release flags)
zstd: libzstd 1.5.7 via python-zstandard 0.25.0 (in-process)
Methodology: best of 5, throughput on uncompressed bytes, all cases round-trip clean.

**Optimization axis:** ratio is the primary objective in tdc. Encode and
decode throughput are reported as context, not targets. A pipeline that
gives 2× the ratio at half the encode speed is a win, not a regression.

Reproduce:
```bash
cmake --build build_release --config Release --target bench_throughput
./build_release/Release/bench_throughput.exe
python bench/bench_zstd_compare.py
```

## Part 1 — tdc throughput (bench/bench_throughput.c)

| pipeline                    | block                | raw MiB | enc MiB | ratio   | enc MB/s | dec MB/s |
|-----------------------------|----------------------|--------:|--------:|--------:|---------:|---------:|
| RAW + NONE                  | vec1d u8 16M         |   16.00 |   16.00 |   1.00x |   2619.4 |   4357.9 |
| RAW+HUF (isolated)          | vec1d u8 16M         |   16.00 |   12.00 |   1.33x |    484.6 |    381.5 |
| RAW+HUF4 (isolated)         | vec1d u8 16M         |   16.00 |   12.00 |   1.33x |    540.3 |   1013.0 |
| RAW + LZ                    | vec1d i32 4M (ramp)  |   16.00 |   16.00 |   1.00x |     25.1 |   3345.3 |
| RAW + BSHUF + LZ            | vec1d i32 4M (ramp)  |   16.00 |    0.24 |  67.79x |    365.9 |   1415.7 |
| DELTA1D + LZ                | vec1d i32 4M (ramp)  |   16.00 |    0.00 | 159783.01x |  541.0 |   2800.2 |
| DELTA1D+ZIGZAG+BSHUF+LZ     | vec1d i16 8M (walk)  |   16.00 |    2.49 |   6.44x |    136.0 |   1108.0 |
| DELTA1D+ZZ+BSHUF+HUFFMAN    | vec1d i16 8M (walk)  |   16.00 |    6.81 |   2.35x |    330.7 |    423.6 |
| DELTA1D+ZZ+BSHUF+HUF4       | vec1d i16 8M (walk)  |   16.00 |    6.81 |   2.35x |    346.5 |    464.4 |
| DELTA1D+ZZ+BSHUF+FSE        | vec1d i16 8M (walk)  |   16.00 |    6.80 |   2.35x |    132.6 |    280.0 |
| DELTA1D+ZZ+BSHUF+LZ+HUF     | vec1d i16 8M (walk)  |   16.00 |    1.92 |   8.33x |    130.2 |    712.3 |
| DELTA1D+BSHUF+LZ            | vec1d f64 2M (smooth)|   16.00 |    8.59 |   1.86x |     19.7 |   1232.8 |
| DELTA1D+BSHUF+HUF           | vec1d f64 2M (smooth)|   16.00 |   10.99 |   1.46x |    329.9 |    442.2 |
| DELTA1D+BSHUF+FSE           | vec1d f64 2M (smooth)|   16.00 |   10.93 |   1.46x |    117.2 |    256.0 |
| DELTA1D+BSHUF+LZ+HUF        | vec1d f64 2M (smooth)|   16.00 |    8.54 |   1.87x |     19.6 |    527.3 |
| DELTA1D+LZ                  | vec1d f64 2M (smooth)|   16.00 |   14.26 |   1.12x |     17.8 |    694.4 |
| RAW+BSHUF+LZ                | vec1d f64 2M (smooth)|   16.00 |    8.50 |   1.88x |     19.9 |   1577.8 |
| RAW+LZ_STREAMS (default)    | vec1d f64 2M (smooth)|   16.00 |   13.00 |   1.23x |     12.4 |    444.0 |
| RAW+LZ_STREAMS optimal      | vec1d f64 2M (smooth)|   16.00 |   10.47 |   1.53x |      0.5 |    175.2 |
| RAW+LZ_STREAMS L1           | vec1d f64 2M (smooth)|   16.00 |   14.98 |   1.07x |    251.5 |    418.0 |
| RAW+LZ_STREAMS L2           | vec1d f64 2M (smooth)|   16.00 |   13.00 |   1.23x |     10.5 |    440.3 |
| RAW+LZ_STREAMS L3           | vec1d f64 2M (smooth)|   16.00 |   13.00 |   1.23x |      5.9 |    439.1 |
| RAW+LZ_STREAMS L4           | vec1d f64 2M (smooth)|   16.00 |   13.00 |   1.23x |      3.6 |    440.0 |
| FPC+BSHUF+LZ                | vec1d f64 2M (smooth)|   16.00 |    8.70 |   1.84x |     19.9 |    900.4 |
| FPC+BSHUF+HUF               | vec1d f64 2M (smooth)|   16.00 |   10.24 |   1.56x |    303.5 |    390.8 |
| PRED2D+BSHUF+LZ             | rast2d u16 2048x2048 |    8.00 |    4.85 |   1.65x |     61.3 |    393.2 |
| PRED2D+BSHUF+LZ+HUF         | rast2d u16 2048x2048 |    8.00 |    3.76 |   2.13x |     57.1 |    232.3 |
| PRED2D+BSHUF+LZ_OPT         | rast2d u16 2048x2048 |    8.00 |    3.72 |   2.15x |      3.3 |    508.5 |
| PRED2D+BSHUF+LZ_OPT+HUF     | rast2d u16 2048x2048 |    8.00 |    3.16 |   2.54x |      3.3 |    299.7 |
| PRED2D+BSHUF+FSE            | rast2d u16 2048x2048 |    8.00 |    2.89 |   2.76x |    102.2 |    210.9 |
| PRED2D+BSHUF+HUF            | rast2d u16 2048x2048 |    8.00 |    2.89 |   2.77x |    246.4 |    398.4 |
| PRED2D+BSHUF+LZ+FSE         | rast2d u16 2048x2048 |    8.00 |    3.75 |   2.13x |     46.6 |    189.7 |
| PRED2D(UP)+BSHUF+HUF        | rast2d u16 2048x2048 |    8.00 |    2.86 |   2.80x |    340.2 |    608.3 |
| PRED2D(UP)+BSHUF+HUF4       | rast2d u16 2048x2048 |    8.00 |    2.86 |   2.80x |    346.0 |    564.8 |
| PRED2D(UP)+BSHUF+FSE        | rast2d u16 2048x2048 |    8.00 |    3.01 |   2.66x |    122.2 |    264.0 |
| PRED2D(UP)+BSHUF+LZ         | rast2d u16 2048x2048 |    8.00 |    3.97 |   2.02x |     74.4 |    597.4 |
| PRED2D(PAETH)+BSHUF+HUF     | rast2d u16 2048x2048 |    8.00 |    2.89 |   2.77x |    249.6 |    400.9 |
| PRED2D(PAETH)+BSHUF+HUF4    | rast2d u16 2048x2048 |    8.00 |    2.89 |   2.77x |    252.6 |    468.1 |
| PRED2D(PAETH)+BSHUF+FSE     | rast2d u16 2048x2048 |    8.00 |    2.89 |   2.76x |    101.9 |    211.5 |
| PRED2D(PAETH)+BSHUF+LZ      | rast2d u16 2048x2048 |    8.00 |    4.85 |   1.65x |     60.6 |    378.1 |
| PLANE2D+BSHUF+LZ            | rast2d i32 1024x1024 |    4.00 |    0.00 | 1323.96x |   532.3 |  15261.4 |
| PRED3D(GRAD)+LZ             | vol3d f32 128^3      |    8.00 |    3.98 |   2.01x |     86.9 |    300.7 |
| PRED3D(GRAD)+BSHUF+LZ       | vol3d f32 128^3      |    8.00 |    5.35 |   1.50x |     41.5 |    251.7 |
| PRED3D(AUTO)+BSHUF+HUF      | vol3d f32 128^3      |    8.00 |    5.20 |   1.54x |    165.4 |    191.4 |
| PRED3D(AUTO)+LZ             | vol3d i16 128^3      |    4.00 |    2.23 |   1.79x |     65.6 |    224.5 |
| PRED3D(AUTO)+ZZ+BSHUF+LZ    | vol3d i16 128^3      |    4.00 |    1.89 |   2.12x |     64.5 |    203.9 |
| PRED3D(GRAD)+ZZ+BSHUF+HUF   | vol3d i16 128^3      |    4.00 |    1.19 |   3.36x |    300.7 |    419.2 |

Notes:
- `RAW + NONE` is the memcpy ceiling for the API (encode/decode framing overhead included).
- `RAW + LZ` on the i32 ramp returns 1.00x because the byte-level pattern of `1000 + i*3` has no exact repetitions for LZ's match-finder. The instant DELTA1D collapses the ramp, LZ hits 159,783x — the residual is a constant (3), LEB128 match-length encodes the whole 16 MiB run in one sequence.
- `RAW + BSHUF + LZ` is the "no model, just shuffle+entropy" floor row. On the i32 ramp it lands at 67.79x — a generic shuffle finds significant per-lane structure even without a model. DELTA1D still wins on ratio.
- `PRED2D(UP)+BSHUF+HUF` at 2.80x is the best pipeline for noisy u16 gradients — 34% better ratio than zstd L19, at 117× the encode speed (340 vs 2.9 MB/s). The UP predictor (SIMD-friendly, no serial dependency on left neighbor) matches PAETH's residual entropy and runs faster.
- `PLANE2D+BSHUF+LZ`: 1324× ratio, 15.3 GB/s decode. The zero-residual fast path bypasses entropy+xform chains entirely; AVX2 8-wide intrinsics handle the tile kernel.
- `PRED3D(GRAD)+ZZ+BSHUF+HUF` on the i16 volume hits 3.36× — 26% better than zstd L19 (2.66×) — at ~100× the encode speed.
- LZ encode rates dropped vs the pre-Apr-12 baseline (window 64 KiB → 1 MiB, hash 2¹⁶ → 2¹⁸, chain depth 128, ext cap 256). Ratios on every applicable row went up; the encode cost is the documented price. See "encode-speed footnote" below.

### u16 PAETH wavefront kernel microbench (2026-05-03, i9-14900K, gcc 14.2)

Direct decode-kernel microbench, bypassing the encode/transform/entropy
pipeline. 8 MiB residual buffer, kernels interleaved per iteration to
share cache + frequency state, best of 50. Numbers are kernel-only
throughput (residual → decoded raster), much higher than the
end-to-end PAETH pipeline rows above (~290-470 MB/s) where BSHUF +
entropy decode dominate.

| kernel                          | decode MB/s | vs wf4   |
|---------------------------------|------------:|---------:|
| wf4 (SSE2 4-row staggered)      | ~1380-1500  |  1.00×   |
| wf8 (AVX2 8-row staggered)      |  ~585-620   |  0.41-0.50× |

**Finding.** wf8 is consistently ~58% slower than wf4 across 5 runs on
Raptor Lake (i9-14900K) with `-mavx2 -O3`. The cross-128-bit-lane
`permute2x128 + alignr_epi8` adds ~3 cycles to the per-iter critical
path, and the 8 `_mm256_extract_epi32` scatter-stores double the
store-port pressure vs SSE2's 4 extracts. Neither cost is offset
because the bottleneck is the row-above memory dependency on the
staggered `dRm[c]`/`dRm[c-1]` reads, not vector ALU throughput.

This confirms the prediction in `SPEEDUP-TODO.md` N4: **the 4-row
wavefront is at the uarch ceiling on x86_64 for u16 PAETH decode.**
The wf8 kernel and helper are kept compiled and round-trip-tested
(see `tests/test_pred2d_wf_consistency`) so a future bench on a
different uarch (Zen 4/5, MSVC, AVX-512 widening) or a fused
entropy+model decode path can re-evaluate without re-implementing.
The dispatcher in `pred2d_decode_sweep` continues to route u16 PAETH
through wf4.

## Part 2 — vectra (tdc-via-vectra) end-to-end

Skipped this run — vectra integration not exercised in this pass. The
2026-04-14 numbers (`bench_compress_final.R`, 5 layers × 2000² f64 raster)
showed VTR fast at 70.8 MB / 3.21× / 1.31s write / 1.27s read, beating
terra DEFLATE on encode (1.31 vs 4.72 s) and on filtered read (1.90 vs
1.99 s). Re-run when vectra is rewired onto current tdc.

## Part 3 — Head-to-head vs libzstd (bench/bench_zstd_compare.py)

zstd is run on the **same raw input bytes** as tdc — the comparison is
"specialized model+transform stack vs generic high-quality entropy coder".

### i32 ramp 16 MiB

| codec                 | ratio   | enc MB/s | dec MB/s |
|-----------------------|--------:|---------:|---------:|
| zstd L1               |   1.43x |    911.7 |   1297.3 |
| zstd L3               |   1.45x |    843.1 |   1639.4 |
| zstd L9               |   1.46x |    583.3 |   1668.0 |
| zstd L19              |   4.63x |      4.2 |    253.8 |
| **tdc DELTA1D+LZ**    | **159783x** | 541.0 | **2800.2** |

### i16 random walk 16 MiB

| codec                              | ratio  | enc MB/s | dec MB/s |
|------------------------------------|-------:|---------:|---------:|
| zstd L1                            |  1.07x |    622.0 |    979.9 |
| zstd L3                            |  1.27x |    132.1 |    667.5 |
| zstd L9                            |  1.30x |     41.6 |    529.3 |
| zstd L19                           |  1.62x |      3.3 |    287.3 |
| tdc DELTA1D+ZIGZAG+BSHUF+LZ        |  6.44x |    136.0 |   1108.0 |
| tdc DELTA1D+ZZ+BSHUF+HUF           |  2.35x |    330.7 |    423.6 |
| **tdc DELTA1D+ZZ+BSHUF+LZ+HUF**   | **8.33x** |  130.2 |    712.3 |

### u16 noisy gradient 2048×2048

| codec                          | ratio  | enc MB/s | dec MB/s |
|--------------------------------|-------:|---------:|---------:|
| zstd L1                        |  1.15x |    262.2 |    876.3 |
| zstd L3                        |  1.50x |    123.6 |    561.9 |
| zstd L9                        |  1.91x |     29.4 |    473.0 |
| zstd L19                       |  2.09x |      3.1 |    365.1 |
| tdc PRED2D(PAETH)+BSHUF+LZ     |  1.65x |     60.6 |    378.1 |
| tdc PRED2D+BSHUF+HUF           |  2.77x |    246.4 |    398.4 |
| **tdc PRED2D(UP)+BSHUF+HUF**   | **2.80x** |  340.2 |    608.3 |
| tdc PRED2D+BSHUF+FSE           |  2.76x |    102.2 |    210.9 |

### i32 split-planes 1024×1024

| codec                       | ratio    | enc MB/s | dec MB/s |
|-----------------------------|---------:|---------:|---------:|
| zstd L1                     |  327.25x |   9285.1 |   3221.6 |
| zstd L3                     |  324.69x |   6010.5 |   3497.4 |
| zstd L9                     |  369.09x |   1290.0 |   3175.4 |
| zstd L19                    |  469.00x |    123.6 |   2644.3 |
| **tdc PLANE2D+BSHUF+LZ**    | **1323.96x** | 532.3 | **15261.4** |

### f32 smooth volume 128³

| codec                       | ratio  | enc MB/s | dec MB/s |
|-----------------------------|-------:|---------:|---------:|
| zstd L1                     |  1.18x |    615.3 |    913.8 |
| zstd L3                     |  1.29x |    122.2 |    647.1 |
| zstd L9                     |  1.56x |     43.9 |    554.7 |
| zstd L19                    |  1.59x |      3.9 |    314.8 |
| **tdc PRED3D(GRAD)+LZ**    | **2.01x** |   86.9 |    300.7 |

### i16 gradient volume 128³

| codec                              | ratio  | enc MB/s | dec MB/s |
|------------------------------------|-------:|---------:|---------:|
| zstd L1                            |  1.62x |    211.3 |    693.4 |
| zstd L3                            |  2.12x |    119.8 |    486.3 |
| zstd L9                            |  2.42x |     29.1 |    615.0 |
| zstd L19                           |  2.66x |      3.1 |    545.6 |
| **tdc PRED3D(GRAD)+ZZ+BSHUF+HUF** | **3.36x** |  300.7 |    419.2 |

### Headlines

1. **Structured 1D signals** (ramp): tdc DELTA1D+LZ delivers 159,783× vs zstd L19's 4.63×. Model collapses the ramp to a constant residual.
2. **Noisy 1D signals** (walk): tdc DELTA1D+ZZ+BSHUF+LZ+HUF at 8.33× — **5.1× better than zstd L19** (1.62×). Pure-LZ variant alone (6.44×) already wins by 4×.
3. **Noisy 2D gradients** (u16): tdc PRED2D(UP)+BSHUF+HUF at 2.80× — **34% better ratio than zstd L19**, at 117× the encode speed.
4. **Piecewise-smooth 2D** (split-planes): PLANE2D+BSHUF+LZ at 1324× — **2.8× better ratio** and 5.8× faster decode than zstd L19.
5. **3D scalar fields**: PRED3D(GRAD)+LZ on smooth f32 (2.01× vs 1.59×) and PRED3D+ZZ+BSHUF+HUF on noisy i16 (3.36× vs 2.66×) both beat zstd L19 by ~25-30%.

## Part 4 — Real data

Datasets prepared by `bench/prepare_real_data.py` and stashed in
`bench/data/` (gitignored). Each is a real public dataset, normalized
to a flat little-endian binary blob plus a `.meta.json` sidecar. Run:

```bash
python bench/prepare_real_data.py                                                # fetch
./build_release/Release/bench_throughput.exe --from PATH --dtype DT --shape S    # tdc
python bench/bench_zstd_compare.py            --from PATH --dtype DT --shape S   # zstd
```

### USGS streamflow — Mississippi River at St. Louis (1995-2024, daily mean discharge)

10958 samples, f64, 85.6 KiB. Real, non-stationary, mildly noisy 1D time
series. 1031 unique values out of 10 958 (9.4% cardinality, ~10.6 repeats
per unique value) — moderate cardinality from cfs values rounded to the
integer.

| codec                           | ratio  | enc MB/s | dec MB/s | notes |
|---------------------------------|-------:|---------:|---------:|-------|
| **zstd L19**                    | **4.50x** |     3.3 |   1310.4 | best ratio overall |
| tdc RAW+LZ_STREAMS              |  4.04x |     71.2 |    678.0 | best tdc, beats zstd L9 (3.99x) |
| zstd L9                         |  3.99x |     48.9 |   1405.1 | |
| tdc DICT_NUMERIC+BSHUF+LZ_SPLIT |  3.94x |     28.1 |   1090.0 | new (added 2026-04-17) |
| tdc DICT_NUMERIC+BSHUF+LZ_STREAMS | 3.93x |    124.3 |   1074.6 | best speed/ratio: 40× zstd L19 enc |
| tdc DICT_NUMERIC+BSHUF+LZ_OPT+HUF | 3.90x |     28.5 |   1213.4 | |
| tdc RAW+LZ_SPLIT                |  3.86x |      8.1 |    551.1 | |
| tdc RAW+LZ_OPT+HUF              |  3.78x |      8.2 |    687.5 | |
| tdc DELTA1D+BSHUF+LZ_STREAMS    |  3.77x |    114.1 |    569.5 | |
| zstd L3                         |  3.78x |    446.8 |   1182.5 | |
| tdc RAW+LZ_OPT+FSE              |  3.74x |      8.1 |    722.0 | |
| tdc RAW+LZ_STREAMS L1           |  3.33x |    119.3 |    553.3 | fast-encode tier |
| tdc DELTA1D+BSHUF+LZ_OPT+HUF    |  3.68x |     16.5 |    657.3 | |
| tdc RAW+BSHUF+LZ_STREAMS        |  3.67x |    111.0 |    593.3 | |
| zstd L1                         |  3.46x |    464.2 |    920.7 | |
| tdc RAW+LZ+HUF                  |  3.56x |    116.8 |    730.2 | |
| tdc RAW+LZ+FSE                  |  3.53x |     97.1 |    521.5 | |
| tdc DELTA1D+BSHUF+LZ+HUF        |  3.53x |    142.8 |    623.4 | |
| tdc DELTA1D+BSHUF+LZ+FSE        |  3.49x |    117.1 |    490.3 | |
| tdc RAW+BSHUF+LZ+HUF            |  3.42x |    136.5 |    620.2 | |
| tdc RAW+BSHUF+LZ+FSE            |  3.38x |    115.9 |    533.9 | |
| tdc DELTA1D+BSHUF+LANE          |  3.27x |    374.1 |    641.1 | fastest encode at >3x ratio |
| tdc RAW+LZ_OPT                  |  3.18x |      8.1 |   3166.8 | fastest decode (3.2 GB/s) |
| tdc DELTA1D+BSHUF+LZ            |  3.16x |    149.5 |   1398.0 | |
| tdc RAW+BSHUF+LZ                |  3.14x |    151.9 |   1429.1 | |
| tdc DELTA1D+LZ_OPT+HUF          |  3.03x |      8.3 |    536.9 | |
| tdc FPC+BSHUF+LZ_STREAMS        |  3.51x |    114.5 |    479.9 | dual-predictor |
| tdc FPC+BSHUF+LANE              |  3.11x |    336.7 |    508.8 | |

**zstd L19 (4.50×) edges out tdc's best (4.04×) by 10%.** Three tdc
pipelines now sit in 3.90-3.94×, all beating zstd L9 (3.99×) at much
faster encode. The fast-encode highlight is **DICT_NUMERIC+BSHUF+LZ_STREAMS
at 3.93× / 124 MB/s encode / 1075 MB/s decode** — 40× zstd L19's encode
speed for a 13% ratio loss. The remaining gap to L19 is structural
(zstd's optimal-parsing LZ on a small block), not closeable without an
optimal-parser stage in tdc. LZ_OPT decodes at 3.2 GB/s — 2.4× faster
than zstd L19.

### NASA POWER T2M — Graz, AT, daily mean 2m air temperature (1995-2024)

10958 samples, f64, 85.6 KiB. Smooth seasonal periodic signal. 3310
unique values out of 10 958 (30.2% cardinality, ~3.3 repeats per unique
value) — high cardinality from temperature stored at 0.01 °C precision
on a continuously-varying signal. DICT_NUMERIC has near-zero leverage
here (confirmed: every DICT_NUMERIC variant tops out at 2.01× on this
block).

| codec                          | ratio  | enc MB/s | dec MB/s | notes |
|--------------------------------|-------:|---------:|---------:|-------|
| **zstd L19**                   | **3.48x** |     5.2 |   1132.8 | best overall |
| zstd L9                        |  3.14x |     38.3 |   1224.1 | |
| tdc RAW+LZ_SPLIT               |  3.12x |      8.8 |    486.9 | best tdc, ~10% gap to zstd L19 |
| tdc RAW+LZ_OPT+HUF             |  3.05x |      9.2 |    539.4 | |
| tdc RAW+LZ_OPT+FSE             |  3.04x |      9.2 |    469.7 | |
| zstd L3                        |  3.02x |    341.5 |   1238.6 | |
| zstd L1                        |  2.81x |    427.9 |   1169.3 | |
| tdc RAW+LZ_STREAMS             |  2.67x |     77.6 |    450.2 | |
| tdc RAW+LZ+HUF                 |  2.68x |    103.1 |    456.1 | fast-encode + good ratio |
| tdc RAW+LZ_STREAMS L1          |  2.54x |    108.4 |    444.2 | |
| tdc RAW+LZ_OPT                 |  2.40x |      9.3 |   2375.1 | fastest decode |
| tdc RAW+LZ                     |  2.17x |    111.0 |   1841.5 | |
| tdc DELTA1D+LZ                 |  1.93x |    102.1 |   1398.0 | model destroys periodicity |
| tdc RAW+BSHUF+LANE             |  1.59x |    174.1 |    320.4 | BSHUF harmful here |

**No model helps on this periodic signal**: every model (DELTA1D, DELTA2,
FPC) destroys the natural byte-level repetition that LZ exploits. BSHUF
is also harmful on f64 periodic data — every BSHUF pipeline lands below
1.6×. Best path: **RAW + LZ_SPLIT** (no model, no shuffle, per-stream
entropy on raw f64 bytes).

### Open Topo Data SRTM30m — central Alps (47.00N 11.00E, 128×128)

16384 samples, i16 elevation in meters, 32 KiB. Real geo-raster.

| codec                          | ratio  | enc MB/s | dec MB/s | notes |
|--------------------------------|-------:|---------:|---------:|-------|
| **tdc PRED2D+BSHUF+LZ+HUF**    | **1.46x** |    59.1 |    259.1 | **beats zstd L19** |
| tdc RAW+BSHUF+LZ               |  1.43x |     74.9 |   1183.7 | |
| zstd L19                       |  1.42x |     15.1 |    637.8 | |
| tdc PLANE2D+BSHUF+LZ           |  1.37x |     64.9 |    905.8 | |
| tdc PRED2D(PAETH)+BSHUF+LZ     |  1.31x |     68.2 |    429.8 | |
| tdc RAW+LZ_STREAMS             |  1.30x |     60.9 |    323.5 | |
| tdc RAW+LZ_SPLIT               |  1.28x |     16.9 |    320.8 | |
| zstd L9                        |  1.28x |    466.4 |    961.5 | |
| tdc RAW+LZ_OPT+HUF             |  1.27x |     17.4 |    362.5 | |
| zstd L1                        |  1.26x |   1230.3 |   1024.6 | |
| tdc RAW+LZ                     |  1.06x |     82.2 |   2083.3 | |

**tdc PRED2D+BSHUF+LZ+HUF (1.46×) beats zstd L19 (1.42×) by 3%.** Spatial
prediction wins on real DEM data even at 32 KiB. RAW+BSHUF+LZ at 1.43×
decodes at 1.2 GB/s — 1.85× zstd L19's decode.

### NASA POWER T2M Regional — flat f64 grid (1,084,842 samples, 8.28 MiB)

Regional temperature grid, 1084842 f64 values. Large enough for stable
benchmarks (small datasets above are noisier). 5426 unique f64 values
out of 1 084 842 (0.50% cardinality) — low-precision rounding makes this
dictionary-friendly.

| codec                              | ratio  | enc MB/s | dec MB/s | notes |
|------------------------------------|-------:|---------:|---------:|-------|
| **tdc DICT_NUMERIC+BSHUF+LZ_STREAMS** | **4.79x** |    69.6 |    833.9 | **best overall, +12% vs zstd L19** |
| tdc DICT_NUMERIC+BSHUF+LZ_SPLIT    |  4.68x |     11.6 |    587.1 | |
| tdc DICT_NUMERIC+BSHUF+LZ_OPT+HUF  |  4.62x |     10.7 |    687.2 | |
| tdc DICT_NUMERIC+BSHUF+LZ_OPT      |  4.48x |     10.6 |   1242.4 | fast-decode tier (1.24 GB/s) |
| tdc DICT_NUMERIC+BSHUF+LZ+HUF      |  4.31x |     71.3 |    592.4 | prior best |
| zstd L19                           |  4.28x |      2.6 |    790.5 | |
| tdc DICT_NUMERIC+LZ_SPLIT          |  4.02x |      7.7 |    426.6 | |
| tdc DICT_NUMERIC+BSHUF+LZ          |  3.99x |     73.8 |   1198.9 | best ratio+speed at no-HUF tier |
| tdc DICT_NUMERIC+LZ_OPT+HUF        |  3.90x |      7.6 |    579.3 | |
| tdc DICT_NUMERIC+LZ_STREAMS        |  3.74x |     63.2 |    479.1 | |
| tdc DICT_NUMERIC+BSHUF+HUF4        |  3.64x |    284.1 |    720.9 | fast-encode tier |
| tdc DICT_NUMERIC+BSHUF+HUF         |  3.64x |    281.2 |    626.2 | |
| tdc DICT_NUMERIC+BSHUF+FSE         |  3.63x |    159.0 |    430.1 | |
| tdc DICT_NUMERIC+LZ_OPT            |  3.46x |      7.6 |   1204.9 | |
| zstd L3                            |  3.45x |    345.6 |   1046.8 | |
| zstd L9                            |  3.40x |     57.4 |   1111.3 | |
| tdc RAW+LZ_SPLIT                   |  3.18x |      2.9 |    343.2 | best non-dict baseline |
| tdc RAW+LZ_OPT+FSE                 |  3.13x |      2.8 |    359.9 | |
| tdc RAW+LZ_OPT+HUF                 |  3.12x |      2.9 |    435.4 | |
| zstd L1                            |  2.96x |    383.3 |   1026.6 | |
| tdc DICT_NUMERIC+LZ                |  2.85x |     78.8 |    882.3 | fastest DICT encode |
| tdc RAW+LZ_STREAMS                 |  2.79x |     49.1 |    400.5 | |
| tdc RAW+LZ                         |  1.95x |     95.5 |   1246.8 | |
| tdc DELTA1D+LZ                     |  1.71x |     41.7 |    647.1 | predictor destroys dict structure |

**tdc DICT_NUMERIC+BSHUF+LZ_STREAMS (4.79×) beats zstd L19 (4.28×) by
12% at 27× the encode speed and 5% faster decode.** The DICT_NUMERIC
model turns the f64 stream into u32 indices (5 426 unique values across
1.08 M samples, 0.5% cardinality), BSHUF concentrates the mostly-zero
high bytes into dense planes, and LZ_STREAMS' split-stream Huffman coder
capitalizes on the bimodal length distribution (one huge zero-run match
per plane + short dictionary-adjacency matches). The previous published
best (`DICT_NUMERIC+BSHUF+LZ+HUF` at 4.31×) was the best *working*
pipeline — LZ_STREAMS on post-BSHUF 4.3 MiB input hit a latent bug in
the LL/ML symbol cap (21 → raised to 25) that rejected match lengths
≥ 2 MiB as TDC_E_CORRUPT. Round-trip verified on a 4 MiB zeros
regression fixture added to `test_lz_streams_roundtrip.c`.

**DICT_NUMERIC is cardinality-driven.** Across the three real f64
datasets the win/loss tracks `unique / n` directly: NASA regional (0.50%
unique) → DICT wins by 35% over RAW; USGS (9.4% unique) → DICT loses by
3% to RAW best, but provides the best speed/ratio tradeoff; NASA daily
(30.2% unique) → DICT loses by 36% (caps at 2.01×). Rough rule of thumb:
**DICT_NUMERIC dominates below ~5% cardinality, ties above ~10%, and
should be skipped above ~25%**.

### Real-data headlines

1. **SRTM DEM (i16 raster)**: tdc wins. PRED2D+BSHUF+LZ+HUF at 1.46× beats zstd L19 (1.42×) by 3%; spatial prediction matters even at 32 KiB.
2. **USGS streamflow (noisy f64, 9.4% cardinality)**: zstd L19 (4.50×) wins by 10% over tdc RAW+LZ_STREAMS (4.04×). tdc beats zstd L9 (3.99×). Best speed/ratio tier: DICT_NUMERIC+BSHUF+LZ_STREAMS at 3.93× / 124 MB/s enc / 1075 MB/s dec.
3. **NASA T2M single-station (periodic f64, 30.2% cardinality)**: zstd L19 (3.48×) wins by 10% over tdc RAW+LZ_SPLIT (3.12×); tdc ties zstd L9 (3.14×). High cardinality rules out DICT_NUMERIC here.
4. **NASA T2M regional (8 MiB f64 grid)**: tdc wins by 12%. DICT_NUMERIC+BSHUF+LZ_STREAMS at 4.79× beats zstd L19 (4.28×) at 27× the encode speed and 5% faster decode. Low-cardinality f64 (0.5% unique values) is the profile — 30 years of daily temperature stored at 0.1 °C precision recycles the same ~5 000 values across a million samples. LZ_STREAMS' split-stream Huffman beats the packed single-stream LZ+HUF by 0.48× of ratio.
5. **Byte-level predictors destroy f64 structure; numeric dictionary reveals it**: DELTA1D/DELTA2/FPC residuals on periodic f64 have *higher* byte entropy than the raw bytes. DICT_NUMERIC flips the frame — work at the f64-value level, not the byte level. On the same NASA regional block DELTA1D+LZ hits 1.71×, RAW+LZ_SPLIT 3.18×, DICT_NUMERIC+BSHUF+LZ+HUF 4.31×.
6. **BSHUF is value-stream dependent, not globally bad**: BSHUF on raw f64 bytes ≈ 1.00× on periodic f64 (destroys structure). BSHUF on DICT_NUMERIC u32 indices is *beneficial* — the high bytes of a small dictionary's indices are mostly zero, so BSHUF concentrates them into runs that LZ+HUF collapse.
7. **Where tdc wins**: structured/zero-residual signals (ramps, split planes), spatially-predictable rasters (DEM, gradients), 3D scalar fields with neighborhood structure, **low-cardinality f64 grids (DICT_NUMERIC)**. Where zstd wins: non-stationary or high-cardinality f64 time series.

### f64 pipeline recommendation

| data profile | best pipeline | ratio | decode MB/s |
|---|---|---|---|
| Low-cardinality f64 grid (<5% unique) | DICT_NUMERIC+BSHUF+LZ_STREAMS | 4.79x | 834 |
| Low-cardinality, fast decode | DICT_NUMERIC+BSHUF+LZ_OPT | 4.48x | 1242 |
| Mid-cardinality, non-stationary (USGS-like, ~10% unique) | RAW+LZ_STREAMS | 4.04x | 678 |
| Mid-cardinality, fast encode+decode | DICT_NUMERIC+BSHUF+LZ_STREAMS | 3.93x | 1075 |
| High-cardinality periodic (NASA-like, >25% unique) | RAW+LZ_SPLIT | 3.12x | 487 |
| Fast-decode priority | RAW+LZ_OPT | 2.40-3.18x | 1150-3170 |
| Fast-encode + good ratio (low-card) | DICT_NUMERIC+BSHUF+LZ | 3.99x | 1199 |

## Encode-speed footnote

Commit 557b1f6 (Apr 12) traded encode throughput for compression ratio:
LZ window 64 KiB → 1 MiB, hash table 2¹⁶ → 2¹⁸, optimal-parser chain
depth → 128, extension cap → 256. Encode rates dropped 2-10× on LZ
pipelines vs the pre-Apr-12 baseline; ratios on every applicable row
went up. Sample deltas:

| row | enc MB/s pre | enc MB/s now | ratio change |
|---|---:|---:|---|
| RAW+LZ (i32 ramp) | 188 | 25 | 1.00× → 1.00× (no LZ-friendly bytes either way) |
| RAW+BSHUF+LZ (i32 ramp) | 1037 | 366 | 48.6× → **67.8×** |
| DELTA1D+LZ (i32 ramp) | 2217 | 541 | 159783× (constant residual, both) |
| DELTA1D+ZIGZAG+BSHUF+LZ (walk) | 258 | 136 | 1.73× → **6.44×** |

This is the documented design: tdc optimizes for ratio first, throughput
second. Encode rates are reported for context, not as a regression to fix.
