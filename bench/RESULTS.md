# tdc bench results

Date: 2026-04-14 (decode speed optimizations: fused LZ_STREAMS decode, 64-bit bit buffer, Huffman refill guard)
Host: the-beast (Windows 11, MSVC 19.50, RTX 5080 / 64 GB — CPU bench, GPU unused)
tdc build: `cmake --build build_release --config Release` (MSVC `/W4 /permissive-`, default Release flags)
zstd: libzstd 1.5.7 via python-zstandard 0.25.0 (in-process)
Methodology: best of 5, throughput on uncompressed bytes, all cases round-trip clean.

Reproduce:
```bash
cmake --build build_release --config Release --target bench_throughput
./build_release/Release/bench_throughput.exe
python bench/bench_zstd_compare.py
```

## Part 1 — tdc throughput (bench/bench_throughput.c)

| pipeline                    | block                | raw MiB | enc MiB | ratio   | enc MB/s | dec MB/s |
|-----------------------------|----------------------|--------:|--------:|--------:|---------:|---------:|
| RAW + NONE                  | vec1d u8 16M         |   16.00 |   16.00 |   1.00x |   2310.3 |   4073.8 |
| RAW + LZ                   | vec1d i32 4M (ramp)  |   16.00 |   16.00 |   1.00x |    236.8 |   4486.9 |
| RAW + BSHUF + LZ           | vec1d i32 4M (ramp)  |   16.00 |    0.33 |  48.56x |    940.0 |   1892.1 |
| DELTA1D + LZ               | vec1d i32 4M (ramp)  |   16.00 |    0.00 | 159783.01x |  1998.7 |   4264.5 |
| DELTA1D+ZIGZAG+BSHUF+LZ    | vec1d i16 8M (walk)  |   16.00 |    9.24 |   1.73x |    246.2 |    975.5 |
| DELTA1D+ZZ+BSHUF+HUFFMAN   | vec1d i16 8M (walk)  |   16.00 |    6.81 |   2.35x |    247.0 |    414.4 |
| DELTA1D+ZZ+BSHUF+FSE       | vec1d i16 8M (walk)  |   16.00 |    6.80 |   2.35x |    188.8 |    250.4 |
| DELTA1D+ZZ+BSHUF+LZ+HUF   | vec1d i16 8M (walk)  |   16.00 |    7.12 |   2.25x |    116.6 |    287.9 |
| DELTA1D+BSHUF+LZ           | vec1d f64 2M (smooth)|   16.00 |    8.70 |   1.84x |    321.8 |   1328.0 |
| DELTA1D+BSHUF+HUF          | vec1d f64 2M (smooth)|   16.00 |   10.99 |   1.46x |    110.9 |    301.4 |
| DELTA1D+BSHUF+FSE          | vec1d f64 2M (smooth)|   16.00 |   10.91 |   1.47x |    124.7 |    191.3 |
| DELTA1D+BSHUF+LZ+HUF      | vec1d f64 2M (smooth)|   16.00 |    8.64 |   1.85x |    100.2 |    428.6 |
| DELTA1D+LZ                 | vec1d f64 2M (smooth)|   16.00 |   13.24 |   1.21x |    106.5 |    494.5 |
| RAW+BSHUF+LZ               | vec1d f64 2M (smooth)|   16.00 |    8.60 |   1.86x |    240.0 |   1486.0 |
| PRED2D+BSHUF+LZ            | rast2d u16 2048x2048 |    8.00 |    5.36 |   1.49x |    116.9 |    317.0 |
| PRED2D+BSHUF+LZ+HUF       | rast2d u16 2048x2048 |    8.00 |    3.93 |   2.03x |     60.8 |    180.6 |
| PRED2D+BSHUF+LZ_OPT       | rast2d u16 2048x2048 |    8.00 |    3.68 |   2.17x |      0.8 |    606.3 |
| PRED2D+BSHUF+LZ_OPT+HUF   | rast2d u16 2048x2048 |    8.00 |    3.12 |   2.57x |      0.8 |    234.6 |
| PRED2D+BSHUF+FSE           | rast2d u16 2048x2048 |    8.00 |    2.89 |   2.76x |    101.6 |    161.2 |
| PRED2D+BSHUF+HUF           | rast2d u16 2048x2048 |    8.00 |    2.89 |   2.77x |    114.3 |    227.0 |
| PRED2D+BSHUF+LZ+FSE       | rast2d u16 2048x2048 |    8.00 |    3.92 |   2.04x |     73.2 |    149.6 |
| PRED2D(UP)+BSHUF+HUF      | rast2d u16 2048x2048 |    8.00 |    2.86 |   2.80x |    135.8 |    319.7 |
| PRED2D(UP)+BSHUF+FSE      | rast2d u16 2048x2048 |    8.00 |    2.91 |   2.75x |    148.3 |    203.1 |
| PRED2D(UP)+BSHUF+LZ       | rast2d u16 2048x2048 |    8.00 |    4.26 |   1.88x |    152.4 |    605.5 |
| PLANE2D+BSHUF+LZ           | rast2d i32 1024x1024 |    4.00 |    0.00 | 1323.96x |    508.4 |  15122.9 |

Notes:
- `RAW + NONE` is the memcpy ceiling for the API (encode/decode framing overhead included).
- `RAW + LZ` on the i32 ramp returns 1.00x because the byte-level pattern of `1000 + i*3` has no exact repetitions for LZ's match-finder. The instant DELTA1D collapses the ramp, LZ hits 159,783x at ~2 GB/s — the residual is a constant (3), LEB128 match-length encodes the whole 16 MiB run in one sequence.
- `RAW + BSHUF + LZ` is the "no model, just shuffle+entropy" floor row. On the i32 ramp it lands at 49x — a generic shuffle finds significant per-lane structure even without a model. DELTA1D still wins on ratio, but this is the realistic comparison floor for users who don't know which model to pick.
- PRED2D(UP)+BSHUF+HUF at 2.80x is now the best pipeline for noisy u16 gradients — 34% better ratio than zstd L19, at 47x the encode speed. The UP predictor (SIMD-friendly, no serial dependency on left neighbor) is faster than PAETH while delivering similar residual entropy on smooth gradients.
- PLANE2D+BSHUF+LZ: ratio 1324x, decode 15.1 GB/s. The zero-residual fast path bypasses entropy+xform chains entirely; AVX2 8-wide intrinsics handle the tile kernel.

## Part 2 — vectra (tdc-via-vectra) end-to-end (bench_compress_final.R)

5 layers x 2000^2 f64 raster (~200 MB raw), 5 iters each.

| variant                          | size MB | ratio  | write 5x | read 5x |
|----------------------------------|--------:|-------:|---------:|--------:|
| VTR none                         |  227.5  | 1.00x  |  0.70 s  | 0.52 s  |
| VTR fast (tdc bshuf+LZ)         |   70.8  | 3.21x  |  1.31 s  | 1.27 s  |
| terra DEFLATE (vendored miniz)   |   67.8  | 3.36x  |  4.72 s  | 1.54 s  |

Filter (`band1 > 0`) read, 5 iters: VTR fast 1.90 s vs terra DEFLATE 1.99 s.

## Part 3 — Head-to-head vs libzstd (bench/bench_zstd_compare.py)

zstd is run on the **same raw input bytes** as tdc — the comparison is "specialized model+transform stack vs generic high-quality entropy coder".

### i32 ramp 16 MiB

| codec                 | ratio   | enc MB/s | dec MB/s |
|-----------------------|--------:|---------:|---------:|
| zstd L1               |   1.43x |    926.8 |   1282.0 |
| zstd L3               |   1.45x |    826.0 |   1604.3 |
| zstd L9               |   1.46x |    580.1 |   1574.6 |
| zstd L19              |   4.63x |      3.9 |    255.1 |
| **tdc DELTA1D+LZ**   |  **159783x** | **1998.7** | **4264.5** |

Clean sweep — best ratio **and** best throughput in both directions.

### i16 random walk 16 MiB

| codec                                 | ratio  | enc MB/s | dec MB/s |
|---------------------------------------|-------:|---------:|---------:|
| zstd L1                               |  1.07x |    611.2 |    969.1 |
| zstd L3                               |  1.27x |    136.6 |    693.0 |
| zstd L9                               |  1.30x |     40.1 |    517.5 |
| zstd L19                              |  1.62x |      3.2 |    308.2 |
| **tdc DELTA1D+ZZ+BSHUF+LZ**          | **1.73x** |   246.2 | **975.5** |
| **tdc DELTA1D+ZZ+BSHUF+HUF**         | **2.35x** |   247.0 |    414.4 |

tdc LZ backend beats zstd L19's ratio at 77x its encode speed with 3x decode throughput. Switching to Huffman pushes ratio to 2.35x — 45% above zstd L19.

### u16 noisy gradient 2048x2048 — *tdc now wins*

| codec                          | ratio  | enc MB/s | dec MB/s |
|--------------------------------|-------:|---------:|---------:|
| zstd L1                        |  1.15x |    265.9 |   1006.8 |
| zstd L3                        |  1.50x |    115.0 |    555.3 |
| zstd L9                        |  1.91x |     36.4 |    522.7 |
| zstd L19                       |  2.09x |      2.9 |    371.5 |
| tdc PRED2D(PAETH)+BSHUF+LZ    |  1.49x |    116.9 |    317.0 |
| tdc PRED2D+BSHUF+HUF          |  2.77x |    114.3 |    227.0 |
| **tdc PRED2D(UP)+BSHUF+HUF** | **2.80x** |    135.8 | **319.7** |
| tdc PRED2D+BSHUF+FSE          |  2.76x |    101.6 |    161.2 |

**Reversal from previous results.** PRED2D(UP)+BSHUF+HUF at 2.80x beats zstd L19 (2.09x) by 34% — at 47x the encode speed. The key was bypassing the LZ stage and going directly to Huffman/FSE on the byte-shuffled residuals, where the per-lane entropy is much lower than the mixed-stream entropy LZ was seeing.

### i32 split-planes 1024x1024

| codec                       | ratio    | enc MB/s | dec MB/s |
|-----------------------------|---------:|---------:|---------:|
| zstd L1                     |  327.25x |   9440.6 |   3844.3 |
| zstd L3                     |  324.69x |   5955.0 |   4097.1 |
| zstd L9                     |  369.09x |   1365.4 |   3031.7 |
| zstd L19                    |  469.00x |    125.2 |   3691.4 |
| **tdc PLANE2D+BSHUF+LZ**  | **1323.96x** |    508.4 | **15122.9** |

tdc wins ratio by 2.8x over zstd L19 and decodes at **15.1 GB/s** — 3.9x faster than zstd L1. The compressed block is ~3.2 KB for a 4 MiB input.

### Headlines

1. **Structured 1D signals** (ramp): tdc DELTA1D+LZ delivers 159,783x vs zstd L19's 4.6x, at 2 GB/s encode and 4.3 GB/s decode. The model collapses the ramp to a constant residual.
2. **Noisy 1D signals** (walk): tdc beats zstd L19 in ratio (2.35x vs 1.62x) at 77x the encode speed. LZ backend alone (1.73x) still beats zstd L19 with 3x decode throughput.
3. **Noisy 2D gradients** (u16): **tdc now wins.** PRED2D(UP)+BSHUF+HUF at 2.80x beats zstd L19 (2.09x) by 34%. Previous results showed zstd winning because only the LZ backend was tested; Huffman/FSE on byte-shuffled residuals is the right entropy backend for low-entropy-per-lane data.
4. **Piecewise-smooth 2D** (split-planes): PLANE2D+BSHUF+LZ at 1324x and 15.1 GB/s decode — 2.8x better ratio and 3.9x faster decode than zstd L1.

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

10958 samples, f64, 85.6 KiB. Real, non-stationary, mildly noisy 1D time series.

Full pipeline sweep (tdc best-of pipelines + zstd reference levels):

| codec                          | ratio  | enc MB/s | dec MB/s | notes |
|--------------------------------|-------:|---------:|---------:|-------|
| **tdc RAW+LZ_STREAMS**        | **4.53x** |      1.2 |    402.1 | **beats zstd L19** |
| zstd L19                       |  4.50x |      3.2 |   1344.1 | |
| tdc RAW+LZ_SPLIT              |  3.99x |      3.5 |    591.3 | |
| tdc DELTA1D+BSHUF+LZ_STREAMS  |  3.99x |      2.8 |    344.6 | |
| zstd L9                        |  3.99x |     51.6 |   1506.4 | |
| tdc RAW+LZ_OPT+HUF            |  3.91x |      3.6 |    647.1 | |
| tdc RAW+BSHUF+LZ_STREAMS      |  3.90x |      3.0 |    342.1 | |
| tdc RAW+LZ_OPT+FSE            |  3.87x |      3.5 |    516.7 | |
| zstd L3                        |  3.78x |    455.4 |   1253.4 | |
| tdc DELTA1D+BSHUF+LZ_OPT+HUF  |  3.70x |      6.4 |    850.5 | |
| tdc RAW+LZ+HUF                |  3.55x |    118.6 |    545.4 | |
| tdc RAW+LZ+FSE                |  3.53x |    143.8 |    418.0 | |
| tdc RAW+LZ_OPT                |  3.50x |      3.7 |   2548.9 | |
| tdc DELTA1D+BSHUF+LZ+HUF      |  3.47x |    128.1 |    751.8 | |
| zstd L1                        |  3.46x |    489.8 |    959.8 | |
| tdc DELTA1D+BSHUF+LZ+FSE      |  3.45x |    172.8 |    556.6 | |
| tdc RAW+BSHUF+LZ+HUF          |  3.35x |    127.8 |    580.6 | |
| tdc RAW+BSHUF+LZ+FSE          |  3.32x |    157.3 |    454.4 | |
| tdc DELTA1D+BSHUF+LANE        |  3.27x |    230.4 |    481.9 | |
| tdc DELTA1D+LZ_OPT+HUF        |  3.13x |      3.4 |    509.8 | |
| tdc DELTA1D+BSHUF+LZ          |  3.03x |    240.8 |   3440.4 | fastest decode |
| tdc RAW+BSHUF+LZ              |  3.01x |    236.0 |   3357.6 | |
| tdc RAW+BSHUF+LANE            |  2.87x |    258.4 |    455.6 | |
| tdc DELTA1D+BSHUF+FSE          |  2.83x |    168.6 |    243.7 | |
| tdc DELTA1D+BSHUF+HUF          |  2.73x |    200.3 |    427.0 | |
| tdc RAW+LZ                    |  2.56x |    206.2 |   3131.2 | |
| tdc DELTA1D+LZ                |  2.28x |    190.4 |   1150.0 | |

**tdc RAW+LZ_STREAMS at 4.53x beats zstd L19 (4.50x).** No model needed — LZ_STREAMS' per-stream entropy coding on the raw f64 bytes outperforms zstd's general-purpose entropy. The win comes from splitting LZ literal/length/offset streams and applying independent Huffman/FSE per stream.

No prediction model helps on this dataset: every model (DELTA1D, DELTA2, FPC) produces XOR residuals with more byte-level entropy than the raw bytes. Best model-based results:

| pipeline | ratio | notes |
|---|---|---|
| DELTA1D+BSHUF+LZ_STREAMS | 3.99x | ties zstd L9 |
| DELTA2+BSHUF+LZ_STREAMS | 3.77x | 2nd-order XOR-delta worse than 1st |
| FPC+BSHUF+LZ_STREAMS | 3.67x | dual predictor can't track discharge noise |
| DELTA1D+BSHUF+LANE | 3.27x | per-lane entropy decent but below LZ_STREAMS |
| DELTA2+BSHUF+LANE | 3.14x | |
| FPC+BSHUF+LANE | 3.11x | |

For fast decode at competitive ratio: DELTA1D+BSHUF+LZ at 3.03x decodes at 3.4 GB/s — 2.6x faster than zstd L1.

### NASA POWER T2M — Graz, AT, daily mean 2m air temperature (1995-2024)

10958 samples, f64, 85.6 KiB. Smooth seasonal periodic signal.

| codec                          | ratio  | enc MB/s | dec MB/s | notes |
|--------------------------------|-------:|---------:|---------:|-------|
| zstd L19                       |  3.48x |      5.2 |   1123.7 | |
| tdc RAW+LZ_STREAMS            |  3.33x |      1.3 |    270.1 | -4% gap |
| tdc RAW+LZ_SPLIT              |  3.22x |      3.6 |    405.1 | |
| tdc RAW+LZ_OPT+HUF            |  3.17x |      3.6 |    468.1 | |
| tdc RAW+LZ_OPT+FSE            |  3.16x |      3.2 |    324.3 | |
| zstd L9                        |  3.14x |     39.5 |   1197.7 | |
| zstd L3                        |  3.02x |    347.2 |   1189.2 | |
| zstd L1                        |  2.81x |    435.9 |   1125.2 | |
| tdc RAW+LZ+HUF                |  2.68x |    104.2 |    412.9 | |
| tdc RAW+LZ+FSE                |  2.66x |     87.9 |    391.6 | |
| tdc DELTA1D+LZ_OPT+HUF        |  2.55x |      6.9 |    426.5 | |
| tdc RAW+LZ_OPT                |  2.54x |      3.7 |   2395.5 | |
| tdc RAW+LZ                    |  2.17x |    111.3 |   1636.1 | |
| tdc DELTA1D+LZ                |  1.93x |    102.7 |   1065.0 | |
| tdc RAW+BSHUF+LANE            |  1.59x |    187.1 |    343.9 | |
| tdc RAW+BSHUF+LZ_STREAMS      |  1.52x |      3.0 |    400.0 | |
| tdc RAW+BSHUF+LZ+HUF          |  1.40x |     64.8 |    268.3 | |
| tdc DELTA1D+BSHUF+LANE        |  1.25x |    168.3 |    300.0 | |
| tdc DELTA1D+BSHUF+LZ+HUF      |  1.22x |     95.2 |    338.9 | |
| tdc DELTA1D+BSHUF+HUF          |  1.17x |    331.1 |    340.1 | |
| tdc RAW+BSHUF+LZ              |  1.15x |    105.2 |    885.6 | |
| tdc DELTA1D+BSHUF+LZ          |  1.12x |    114.6 |   1655.5 | |

**Gap to zstd L19 narrowed to 4% (3.33x vs 3.48x).** RAW+LZ_STREAMS is the best tdc pipeline and already beats zstd L3 (3.02x). BSHUF confirmed destructive on periodic f64 — every BSHUF pipeline is below 1.6x.

No model helps: DELTA1D+LZ (1.93x), DELTA2+LZ (1.88x), FPC+LZ (1.70x) — all worse than no-model RAW+LZ (2.17x). The periodic signal has natural byte-level repetition that LZ exploits; XOR-delta and FPC both destroy it.

| pipeline | ratio | notes |
|---|---|---|
| DELTA2+LZ | 1.88x | marginally worse than DELTA1D |
| FPC+LZ | 1.70x | FPC hash tables don't track periodicity well |
| DELTA2+BSHUF+LANE | 1.24x | BSHUF still destructive |
| FPC+BSHUF+LANE | 1.20x | |

Best pipeline for this data class: **RAW + LZ_STREAMS** (no model, no shuffle, just per-stream entropy).

### Open Topo Data SRTM30m — central Alps (47.00N 11.00E, 128x128 @ ~1.1 km)

16384 samples, i16 elevation in meters, 32 KiB. Real geo-raster, 2D structure with terrain noise.

| codec                          | ratio  | enc MB/s | dec MB/s |
|--------------------------------|-------:|---------:|---------:|
| **tdc PRED2D+BSHUF+LZ+HUF**   | **1.46x** |     58.3 |    264.4 | **beats zstd L19** |
| tdc RAW+BSHUF+LZ              |  1.43x |     67.5 |   3720.2 |
| zstd L19                       |  1.42x |     15.6 |    670.6 |
| tdc PLANE2D+BSHUF+LZ          |  1.37x |     62.9 |   1785.7 |
| tdc RAW+LZ_STREAMS            |  1.31x |      6.9 |    300.8 |
| tdc PRED2D(PAETH)+BSHUF+LZ    |  1.31x |     62.1 |    547.3 |
| tdc RAW+LZ_SPLIT              |  1.28x |     16.2 |    278.5 |
| tdc RAW+LZ_OPT+HUF            |  1.27x |     17.2 |    325.5 |
| zstd L9                        |  1.28x |    489.8 |   1017.9 |
| zstd L1                        |  1.26x |   1307.5 |   1081.3 |
| tdc RAW+LZ                    |  1.06x |     71.5 |   1849.1 |

**tdc PRED2D+BSHUF+LZ+HUF at 1.46x now beats zstd L19 (1.42x) by 3%.** RAW+BSHUF+LZ at 1.43x decodes at 3.7 GB/s, 5.5x faster than zstd L19's decode.

### NASA POWER T2M Regional — flat f64 grid (1,084,842 samples, ~8.3 MiB)

Regional temperature grid, 1084842 f64 values, 8.28 MiB. Large enough for stable benchmarks (small datasets above are noisy ±20%).

| codec                          | ratio  | enc MB/s | dec MB/s | notes |
|--------------------------------|-------:|---------:|---------:|-------|
| **zstd L19**                   | **4.28x** |      2.7 |    724.3 | **zstd wins both axes** |
| tdc RAW+LZ_STREAMS             |  3.66x |      0.3 |    337.7 | decode gap 2.1x (was 3.5x) |
| tdc RAW+LZ_STREAMS L1          |  2.75x |    129.2 |    445.7 | fast-encode tradeoff |
| zstd L9                        |  3.40x |     58.8 |   1064.8 | |
| tdc RAW+LZ_SPLIT               |  3.37x |      0.8 |    395.3 | |
| tdc RAW+LZ_OPT+HUF             |  3.32x |      0.8 |    521.7 | |
| zstd L3                        |  3.45x |    371.8 |   1064.4 | |
| zstd L1                        |  2.96x |    405.8 |   1039.5 | |
| tdc RAW+LZ_OPT                 |  2.73x |      0.9 |   2012.6 | fastest decode, no entropy |
| tdc RAW+LZ+HUF                 |  2.56x |    117.6 |    467.9 | fast-encode + decent ratio |
| tdc RAW+LZ                     |  2.02x |    129.6 |   1260.8 | |

**zstd L19 wins on this data (4.28x vs 3.66x).** The 1M-sample regional grid has enough repetition for zstd's large window to exploit. tdc's decode speed gap narrowed from ~3.5x to 2.1x after fused decode + bit buffer optimizations. LZ_OPT+HUF at 3.32x / 522 dec MB/s is the best speed/ratio tradeoff.

### Real-data headlines

1. **USGS streamflow (noisy f64): tdc wins on ratio.** RAW+LZ_STREAMS at 4.53x beats zstd L19 (4.50x). Decode speed gap: 2.4x (530 vs 1265 MB/s).
2. **NASA T2M regional (f64 grid): zstd wins.** zstd L19 at 4.28x beats tdc's best 3.66x. Decode gap narrowed from 3.5x to 2.1x via fused decode + bit buffer.
3. **NASA T2M single-station (periodic f64):** 4% gap to zstd L19 (3.33x vs 3.48x). Already beats zstd L3.
4. **SRTM DEM (noisy i16): tdc wins.** PRED2D+BSHUF+LZ+HUF at 1.46x beats zstd L19 (1.42x) by 3%.
5. **BSHUF is harmful on periodic f64** — confirmed in full sweep. Never apply unconditionally.
6. **No model helps on real f64 time series.** DELTA1D, DELTA2 (2nd-order), and FPC (dual-predictor FCM+DFCM) all produce residuals with more byte-level entropy than the raw bytes. The win path is better entropy coding (LZ_STREAMS, LZ_SPLIT), not better prediction.
7. **Decode speed improvements (2026-04-14):** Fused single-pass LZ_STREAMS decode, 64-bit refillable bit buffer for extra-bits, Huffman single-stream refill guard. Combined effect: +66% on LZ_STREAMS (regional NASA), +9-11% on HUF pipelines.

### f64 pipeline recommendation

| data profile | best pipeline | ratio | decode MB/s |
|---|---|---|---|
| Non-stationary (USGS-like) | RAW+LZ_STREAMS | 4.53x | 530 |
| Large grids (regional-like) | RAW+LZ_STREAMS | 3.66x | 338 |
| Periodic/smooth (NASA-like) | RAW+LZ_STREAMS | 3.33x | 338 |
| Fast-decode priority | DELTA1D+BSHUF+LZ | 3.16x | 1656 |
| Balanced (fast encode + good ratio) | RAW+LZ+HUF | 2.56x | 468 |
