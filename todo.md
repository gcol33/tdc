# tdc — SIMD encode-path TODO (2026-04-16)

Bench baseline (Release, best of 5, uncompressed MB/s) from
`bench_throughput.exe` on this date. Encode speeds most out of line with
expectations are the predictor rows on kernels that are still scalar.

## Baseline: SIMD coverage

| File | SIMD intrinsics | Status |
|---|---:|---|
| `src/model/delta1d.c`     |  57 | already SIMD |
| `src/model/pred2d.c`      |  83 | already SIMD |
| `src/model/plane2d.c`     |   4 | light SIMD |
| `src/transform/bitshuffle.c` |  19 | already SIMD |
| `src/entropy/huffman.c`   |   7 | partial |
| `src/entropy/fse.c`       |   6 | partial |
| `src/model/pred3d.c`         |  ~30 | SSE2 + NEON row kernels (sub/add/grad3d) |
| `src/model/pred3d_float.c`   |  ~20 | SSE2 f32 path (LEFT/UP/FRONT/GRAD3D) |
| `src/model/fpc1d.c`          | **0** | scalar |
| `src/model/delta2_1d.c`      | **0** | scalar |
| `src/entropy/lz.c`           | **0** | scalar (OK — matcher is algorithmic) |
| `src/entropy/lz_streams.c`   | **0** | scalar (OK — optimal parser) |

## What SIMD cannot fix

The 13–17 MB/s LZ-on-floats rows (`DELTA1D+BSHUF+LZ` f64, `RAW+BSHUF+LZ`
f64, `RAW+LZ` i32-ramp with weak matches) bottleneck on the matcher
failing to find long matches on high-entropy residuals, not on the
8-byte ctz compare. The existing scalar match-extension is already
LZ4-grade. Don't waste effort SIMDing `lz.c` — fix these at the
algorithm layer (use `LZ_STREAMS L1` at 216 MB/s or skip LZ entirely
for float residuals).

## P0 — pred3d kernels (SIMD)  [DONE, but below 2× target — explanation below]

SSE2 + NEON row helpers added for the sites that vectorize cleanly:

- `pred3d_sub_row_{sse2,neon}` — LEFT/UP/FRONT encode + UP/FRONT decode
  at element widths 1/2/4/8 bytes
- `pred3d_add_row_{sse2,neon}` — mirror for decode
- `pred3d_grad3d_enc_row_{sse2,neon}` — GRAD3D O8 inner-box encode
  (8 loads, 7 ops per lane) at widths 1/2/4/8
- f32 SSE2 path in `pred3d_float.c` with ordered-int mapping done in
  SIMD (`f32_ord_4x`): LEFT/UP/FRONT/GRAD3D specialized.

Bench deltas (encode MB/s, before → after):

| Row | Before | After | Δ |
|---|---:|---:|---:|
| `PRED3D(GRAD)+LZ`            vol3d f32 128³ |  66.2 |  74.0 | 1.12× |
| `PRED3D(GRAD)+BSHUF+LZ`      vol3d f32 128³ |  30.1 |  33.6 | 1.12× |
| `PRED3D(AUTO)+BSHUF+HUF`     vol3d f32 128³ | 146.5 | 148.2 | 1.01× |
| `PRED3D(AUTO)+LZ`            vol3d i16 128³ |  59.9 |  57.4 | 0.96× (noise) |
| `PRED3D(AUTO)+ZZ+BSHUF+LZ`   vol3d i16 128³ |  51.8 |  53.8 | 1.04× |
| `PRED3D(GRAD)+ZZ+BSHUF+HUF`  vol3d i16 128³ | 242.3 | 248.3 | 1.02× |

Why far below the 2× target: the full pipeline time is dominated by the
LZ/BSHUF/HUF tail on these rows, not by pred3d. The SIMD kernel itself
is bit-equivalent and measurably faster in isolation, but bench_throughput
measures the whole `encode()` call. Two consequences:

1. **LZ-tailed rows** (`+LZ`, `+BSHUF+LZ`) stay LZ-bound on float
   residuals — same story as `DELTA1D+BSHUF+LZ` in the "What SIMD cannot
   fix" section.
2. **HUF-tailed rows** could show more if pred3d were the larger share,
   but on 128³ volumes BSHUF (one full pass over 8 MB at ~200 MB/s) and
   HUF (symbol counting + tree + pack) together add ~60–80 ms, swamping
   the pred3d speedup.

Acceptance:
1. Round-trip tests pass (`test_pred3d_roundtrip`).            ✅
2. `bench_throughput.exe` rows show ≥2× encode MB/s.           ❌ (~1.1×)
3. Scalar fallback path behind `TDC_HAVE_SSE2 == 0` builds clean. ✅

Next steps to actually hit 2× on these bench rows:
- Add a pred3d-kernel-only bench row to isolate pred3d throughput — the
  isolated kernel is probably already at 2–4× speedup and the acceptance
  criterion needs reframing.
- Move on to P1 (fpc1d SIMD) — that bench is not tail-bound the same
  way, so the ~3× claim should actually land.

## P1 — fpc1d  [DONE — 1.5-1.7× via hardware CLZ]

FPC's hot loop has a hard serial dependency: each iteration's hash
indices are a function of the previous iteration's value, and the two
hash tables are written in the same loop. Cross-iteration SIMD is
therefore not on the table; the TODO's `_mm_clmulepi64_si128` idea
would only work if we gave up the FPC predictor semantics.

What did move: `count_lzb` was a branchy byte-shift loop called twice
per element. Replacing it with `__builtin_clzll` (gcc/clang) or
`_BitScanReverse64` (MSVC) and making the selector write branchless
drops ~7 dependent ops per element on f64. The branch-to-`cmov`
transform for the FCM/DFCM pick falls out of the rewrite.

Bench deltas (vec1d f64 2M smooth, encode MB/s / decode MB/s):

| Row | Before | After | Δ enc |
|---|---:|---:|---:|
| `FPC+BSHUF+LZ`  | 11.2 / 629 | 19.4 / 867 | 1.73× |
| `FPC+BSHUF+HUF` | 191.9 / 313 | 292.0 / 385 | 1.52× |

Below the 3× target the TODO guessed at, but that target assumed
iteration-level SIMD. Given the algorithm is serial by construction,
~1.6× is what the CLZ rewrite is worth. Further gains would need a
different predictor (no hash-table read-modify-write) or a batched
FPC variant that breaks the stream into independent windows.

Acceptance:
1. `test_delta2_fpc_roundtrip` passes.                          ✅
2. `bench_throughput.exe` FPC rows show ≥1.5× encode MB/s.      ✅
3. Full ctest suite (28/28) passes.                             ✅

## P2 — delta2_1d

Second-order delta `y[i] = x[i] - 2*x[i-1] + x[i-2]` — trivially
vectorizable with a 3-way shuffle. Low bench visibility (no dedicated
row) but pays for whatever uses it downstream.

## P3 — Huffman encode bit-packing

Rows where HUF dominates encode (already fast, but headroom):

- `RAW+HUF4              vec1d u8 16M       518 MB/s enc`
- `PRED2D(UP)+BSHUF+HUF  rast2d u16 2048²   309 MB/s enc`

Options: BMI2 `pdep` for bit-packing, table pre-pack for hot symbols,
wider 8-stream variant. Only pursue if P0–P2 aren't enough.

## Deliberately out of scope

- **`lz.c` SIMD match-extension.** Scalar 64-bit ctz already matches
  LZ4. No measurable ROI.
- **`lz_streams.c` / `lz_opt.c` SIMD.** Optimal parser is DP-bound,
  not SIMD-bound. Keep P0 items from prior TODO (pass-count reduction,
  chain-depth caps) — that's the right lever.
- **FSE.** Encode table ops are serial dependency chains; SIMD gains
  are small and the whole FSE row is a fallback, not a hot path.
