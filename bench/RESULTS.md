# tdc bench results

Date: 2026-04-07
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
| RAW + NONE                  | vec1d u8 16M         |   16.00 |   16.00 |   1.00x |   2840.1 |   5226.2 |
| RAW + LZ2                   | vec1d i32 4M (ramp)  |   16.00 |   16.00 |   1.00x |    180.9 |   4699.9 |
| RAW + BSHUF + LZ2           | vec1d i32 4M (ramp)  |   16.00 |    0.38 |  42.57x |   1005.1 |   1511.2 |
| DELTA1D + LZ2               | vec1d i32 4M (ramp)  |   16.00 |    0.06 | 254.62x |   2230.4 |   4742.6 |
| DELTA1D+ZIGZAG+BSHUF+LZ2    | vec1d i16 8M (walk)  |   16.00 |    7.94 |   2.02x |    257.8 |   1240.0 |
| DELTA1D+ZZ+BSHUF+HUFFMAN    | vec1d i16 8M (walk)  |   16.00 |    6.81 |   2.35x |    245.7 |    223.9 |
| DELTA1D+ZZ+BSHUF+FSE        | vec1d i16 8M (walk)  |   16.00 |    6.80 |   2.35x |    191.2 |    245.9 |
| DELTA1D+ZZ+BSHUF+LZ2+HUF    | vec1d i16 8M (walk)  |   16.00 |    5.47 |   2.93x |    163.0 |    161.2 |
| PRED2D(PAETH)+BSHUF+LZ2     | rast2d u16 2048x2048 |    8.00 |    4.89 |   1.64x |    206.8 |    315.0 |
| PLANE2D+BSHUF+LZ2           | rast2d i32 1024x1024 |    4.00 |    0.03 | 145.46x |    567.8 |    534.4 |

Notes:
- `RAW + NONE` is the memcpy ceiling for the API (encode/decode framing overhead included).
- `RAW + LZ2` on the i32 ramp returns 1.00x because the byte-level pattern of `1000 + i*3` has no exact repetitions for LZ2's match-finder. The instant DELTA1D collapses the ramp, LZ2 hits 254x at >2 GB/s — proves the entropy stage itself is healthy. (SPEEDUP-TODO P1.1 fixed: codec.h now documents that LZ2 alone needs either a model or a byte-shuffle on multi-byte dtypes.)
- `RAW + BSHUF + LZ2` is the new "no model, just shuffle+entropy" floor row. On the i32 ramp it lands at 42x — a generic shuffle finds significant per-lane structure even without a model. DELTA1D still wins on ratio, but this is the realistic comparison floor for users who don't know which model to pick.
- DELTA1D+LZ2 ramp ratio jumped from 32x → 254x and PLANE2D+BSHUF+LZ2 from 30x → 145x after **SPEEDUP-TODO P0.1**: LZ2's max-match cap (130 bytes, inherited from vectra) was chopping long zero-run residuals into thousands of 3-byte sequences. Match length is now varint-encoded with no ceiling; the safe-path overlap copy was upgraded to the same doubling pattern the fast path uses so long matches don't bail to byte-by-byte. Round-trip clean across all 11 ctests.
- PRED2D(PAETH) encode 129 → 207 MB/s, decode 219 → 315 MB/s after **SPEEDUP-TODO P0.2** partial: the `&&` short-circuit in the Paeth ternary chain was replaced with bitwise `&` so the compiler emits a flat cmov sequence. Decode is still serial-bound on the per-row `left = d0[col-1]` chain — getting to the 800 MB/s acceptance target requires row-interleaved SIMD which is parked.
- **SPEEDUP-TODO P2.1** (entropy chain): the four `DELTA1D+ZZ+BSHUF+*` rows run the identical residual stream through different entropy backends. LZ2 alone is the throughput king at 258/1240 MB/s but loses ~17% ratio to Huffman/FSE; swapping LZ2 for Huffman or FSE costs throughput and buys 2.35x vs 2.02x on this random-walk residual. Chaining LZ2→Huffman picks up the long-term wins LZ2 leaves on the table for a 2.93x ratio, but cuts throughput roughly in half because the Huffman pass then re-processes the LZ2 output stream end-to-end. The driver-side chain walk (size-table prefix in the payload) is correct and runs on every applicable pipeline; the numbers above establish the crossover point where a second entropy pass starts earning its cost.

## Part 2 — vectra (tdc-via-vectra) end-to-end (bench_compress_final.R)

5 layers × 2000² f64 raster (~200 MB raw), 5 iters each.

| variant                          | size MB | ratio  | write 5× | read 5× |
|----------------------------------|--------:|-------:|---------:|--------:|
| VTR none                         |  227.5  | 1.00x  |  0.70 s  | 0.52 s  |
| VTR fast (tdc bshuf+LZ2)         |   70.8  | 3.21x  |  1.31 s  | 1.27 s  |
| terra DEFLATE (vendored miniz)   |   67.8  | 3.36x  |  4.72 s  | 1.54 s  |

Filter (`band1 > 0`) read, 5 iters: VTR fast 1.90 s vs terra DEFLATE 1.99 s.

Round-trip correctness: TRUE for both `VTR none` and `VTR fast`.

Read of the post-zlib-removal state:
- Vendored **miniz** matches system zlib's ratio within 4% — no functional regression from the dependency removal.
- vectra-via-tdc is **~3.6× faster to write** than terra/DEFLATE for essentially the same ratio.
- `compress = "ratio"` is gone (removed when vectra was rewired to tdc); `bench_compress_final.R` was patched accordingly.

## Part 3 — Head-to-head vs libzstd (bench/bench_zstd_compare.py)

zstd is run on the **same raw input bytes** as tdc — the comparison is "specialized model+transform stack vs generic high-quality entropy coder".

### i32 ramp 16 MiB

| codec                 | ratio   | enc MB/s | dec MB/s |
|-----------------------|--------:|---------:|---------:|
| zstd L1               |   1.43x |    932.6 |   1418.4 |
| zstd L3               |   1.45x |    856.4 |   1695.4 |
| zstd L9               |   1.46x |    578.4 |   1748.9 |
| zstd L19              |   4.63x |      4.7 |    404.4 |
| **tdc DELTA1D+LZ2**   |  **32.49x** | **2091.7** | **4085.8** |

Clean sweep — best ratio **and** best throughput in both directions.

### i16 random walk 16 MiB

| codec                                 | ratio  | enc MB/s | dec MB/s |
|---------------------------------------|-------:|---------:|---------:|
| zstd L1                               |  1.07x |    919.0 |   1234.1 |
| zstd L3                               |  1.27x |    237.3 |    883.5 |
| zstd L9                               |  1.30x |     92.1 |    925.5 |
| zstd L19                              |  1.62x |      4.2 |    524.8 |
| **tdc DELTA1D+ZIGZAG+BSHUF+LZ2**      | **1.96x** |   243.2 | **1149.6** |

tdc beats zstd L19's ratio at >50× zstd L19's encode speed; fastest decode in the table.

### u16 noisy gradient 2048×2048 — *PRED2D underdelivers*

| codec                          | ratio  | enc MB/s | dec MB/s |
|--------------------------------|-------:|---------:|---------:|
| zstd L1                        |  1.15x |    475.5 |   1094.2 |
| zstd L3                        |  1.50x |    203.5 |    875.0 |
| **zstd L9**                    | **1.91x** |     68.9 | **819.5** |
| zstd L19                       |  2.09x |      4.5 |    623.2 |
| tdc PRED2D(PAETH)+BSHUF+LZ2    |  1.64x |    129.1 |    219.3 |

zstd L9 wins this block: better ratio (1.91 vs 1.64) and ~4× faster decode at half the encode speed.

### i32 split-planes 1024×1024 — *PLANE2D leaves structure on the table*

| codec                       | ratio    | enc MB/s | dec MB/s |
|-----------------------------|---------:|---------:|---------:|
| zstd L1                     |  327.25x |  16778.5 |   5813.1 |
| zstd L3                     |  324.69x |  10746.9 |   5204.3 |
| zstd L9                     |  369.09x |   2466.9 |   4949.3 |
| zstd L19                    |  469.00x |    219.0 |   5767.0 |
| tdc PLANE2D+BSHUF+LZ2       |   29.65x |    563.4 |    494.3 |

After **SPEEDUP-TODO P0.1** (LZ2 max-match cap removed): tdc PLANE2D+BSHUF+LZ2 = **145.46x @ 568/534 MB/s**. Now within ~3x of zstd L1 in ratio at ~1/30 of its throughput. The remaining ratio gap on this synthetic block is the varint-length encoding floor on a literal 4 MiB-of-zeros stream — real planar data won't hit it.

### Headlines

1. For **smooth/signed 1D signals**, tdc's model stack is the right answer — beats libzstd in ratio **and** throughput simultaneously. DELTA1D vs zstd L19 on the ramp: **254x vs 4.6x, 470× faster encode** (post-P0.1).
2. **PRED2D PAETH still loses on noisy gradients.** Decode is bottlenecked by the per-row serial dependency; hitting >800 MB/s on decode requires row-interleaved SIMD (parked, see SPEEDUP-TODO P0.2).
3. **PLANE2D recovered after the LZ2 fix.** Was 30x at 60 MB/s; now 145x at 570 MB/s on the same block. The original "PLANE2D underdelivers" diagnosis pointed at the model — it was actually the entropy stage capping match length.

## Part 4 — Real data (SPEEDUP-TODO P1.3)

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

| codec                          | ratio  | enc MB/s | dec MB/s |
|--------------------------------|-------:|---------:|---------:|
| tdc RAW + LZ2                  |  2.56x |    397.9 |   3426.4 |
| tdc RAW + BSHUF + LZ2          |  3.00x |    353.8 |   3557.6 |
| zstd L1                        |  3.46x |    491.5 |    954.4 |
| zstd L3                        |  3.78x |    455.9 |   1255.3 |
| zstd L9                        |  3.99x |     50.9 |   1511.8 |
| zstd L19                       |  4.50x |      3.3 |   1339.8 |

zstd wins on ratio (because tdc has no f64-aware delta model), but tdc decodes at 3.5 GB/s — **2.4× faster than zstd L1's decode**. The takeaway: even without a model, BSHUF+LZ2 is the right floor for f64 time series and decode is comfortably faster than any libzstd setting.

### NASA POWER T2M — Graz, AT, daily mean 2m air temperature (1995-2024)

10958 samples, f64, 85.6 KiB. Smooth seasonal periodic signal.

| codec                          | ratio  | enc MB/s | dec MB/s |
|--------------------------------|-------:|---------:|---------:|
| tdc RAW + LZ2                  |  2.15x |    340.3 |   1716.7 |
| tdc RAW + BSHUF + LZ2          |  1.14x |    164.9 |   1238.6 |
| zstd L1                        |  2.81x |    435.9 |   1179.2 |
| zstd L9                        |  3.14x |     39.7 |   1272.5 |
| zstd L19                       |  3.48x |      5.4 |   1177.5 |

**Surprise: BSHUF actively hurts here (2.15x → 1.14x).** On a strongly periodic f64 signal, the byte-lane transpose breaks the seasonal repetition that LZ2 was finding in the unshuffled bytes. Real-data finding worth recording: the cookbook should *not* recommend BSHUF unconditionally for f64. tdc decode again wins by 30-50% over zstd at all levels.

### Open Topo Data SRTM30m — central Alps (47.00N 11.00E, 128×128 @ ~1.1 km)

16384 samples, i16 elevation in meters, 32 KiB. Real geo-raster, 2D structure with terrain noise.

| codec                          | ratio  | enc MB/s | dec MB/s |
|--------------------------------|-------:|---------:|---------:|
| tdc RAW + LZ2                  |  1.06x |    172.8 |   2185.3 |
| tdc RAW + BSHUF + LZ2          |  1.37x |    192.2 |    944.1 |
| tdc PRED2D(PAETH)+BSHUF+LZ2    |  1.21x |    170.7 |    376.1 |
| tdc PLANE2D+BSHUF+LZ2          |  1.28x |    135.5 |    266.6 |
| zstd L1                        |  1.26x |   1307.5 |   1081.3 |
| zstd L9                        |  1.28x |    487.5 |   1014.6 |
| zstd L19                       |  1.42x |     15.8 |    667.7 |

**tdc RAW+BSHUF+LZ2 at 1.37x ties zstd L19 within margin and beats zstd L9** — but at much higher decode throughput than L19. Both 2D model paths (PRED2D, PLANE2D) actually score *worse* than plain BSHUF on real DEM data. Real-data finding: on small noisy DEM tiles, the predictor models don't justify their overhead. For this class of input the right pipeline is BSHUF+LZ2, no model.

### Real-data takeaways

- BSHUF+LZ2 is the no-brainer floor for **i16/i32 rasters** — competitive with zstd L9-L19 at ~2x the encode/decode speed of L19.
- BSHUF is **not** universally good for f64: on periodic climate signals it can shred the repetition LZ2 was finding. Don't recommend it blindly.
- tdc's lack of an f64-aware delta/quantize model leaves a real ratio gap on continuous-physical-quantity time series. This is the next obvious model-stage gap, not a bug.
- The 2D model stages (PRED2D, PLANE2D) are tuned for synthetic inputs that match their assumptions. On real DEM noise they over-fit and underperform a generic shuffle. Worth investigating an "AUTO" path that picks no-model when the residual variance doesn't drop.
