# Performance tuning

tdc exposes a small set of knobs and a much larger set of choices that look like knobs but are really pipeline-design decisions. This page walks the ones we reach for most, puts real numbers on every setting, and calls out the failure modes that hide behind clean-looking microbenchmarks. Ratio is the primary axis; we are willing to trade one or two percent of ratio for a large speed win, and almost never willing to go further.

Two warnings worth carrying through every table on this page. First, every block record carries an 80-byte header. On blocks smaller than a few kilobytes that overhead dominates the ratio before anything the codec does matters. The examples here all use inputs of 64 KiB or larger so the header reads as a rounding error. Second, the encoder has a whole-pipeline fast path: when the model emits an all-zero residual, the block record sets `TDC_BLOCK_FLAG_ZERO_RESIDUAL`, writes zero transform and entropy bytes, and the decoder reconstructs the residual from the flag alone. Any benchmark input a model predicts exactly (a tri-affine volume, a constant vector, a perfect linear ramp) will trip that flag, report a spectacular ratio, and produce throughput numbers that do not reflect what the transform or entropy stages do on real data. Every input below keeps deterministic jitter on top of the structural pattern so the residual stays nonzero.

## Methodology

Numbers below came from running two example programs on the author's workstation (Windows 11, MSVC 2022, `/O2 /arch:AVX2`, tdc `0.2.0-dev`). Each row reports a warm-cache median over nine iterations, encoded bytes after the full pipeline (including the 80-byte block record header), ratio, encode MB/s, and decode MB/s. Throughput is reported on uncompressed bytes divided by wallclock time, which is the convention `zstd -b` and the tdc `bench_throughput` harness both use. `enc MB/s` counts the raw input going into `tdc_encode_block`; `dec MB/s` counts the same raw bytes coming out of `tdc_decode_block_into`.

The harness for every table on this page is [`docs/examples/tuning_pipeline_compare.c`](../examples/tuning_pipeline_compare.c) for the pipeline comparison and [`docs/examples/tuning_knob_walk.c`](../examples/tuning_knob_walk.c) for the three single-knob sweeps. Both compile under `TDC_DOC_EXAMPLES` and run against the installed `libtdc`; re-running them reproduces the tables here within noise. The allocator is the plain stdlib wrapper from `docs/examples/quickstart_common.h`: `realloc` on grow, `free` on zero size. Timing is `QueryPerformanceCounter` on Windows and `clock_gettime(CLOCK_MONOTONIC)` on POSIX. The encode loop reuses the output `tdc_buffer` across iterations so only the first iteration pays the realloc path. Decode reads from a buffer the harness pre-allocated using `tdc_decode_peek` followed by `tdc_decode_block_into`, with no per-iteration allocation.

The inputs are tuned to exercise the stages honestly. The pipeline-comparison raster is 128 rows by 256 columns of `int16_t` values drawn from a smooth bi-axial gradient with a small deterministic jitter. The smooth part gives `PRED_2D` real signal to strip; the jitter keeps the zero-residual fast path off so the transform and entropy stages see actual bytes. The LZ sweep uses a 256 by 256 `int16_t` raster with the same pattern at larger size so deeper LZ parsers have something to chase. The QUANTIZE sweep uses a 16,384-element `f64` sinusoid plus sub-unit noise. The byte-shuffle sweep uses a 16,384-element near-linear integer sequence stored at four widths. None of the inputs are synthetic enough to trip the zero-residual flag; the delta streams have a small but nonzero residual magnitude.

Three inputs worth calling out that we deliberately did not benchmark. A pure linear ramp hits the zero-residual path under `DELTA_1D` and reports whatever the block-header overhead dictates, which is not an interesting number. A uniform-random buffer defeats every model equally and the numbers collapse to the RAW line across the whole table. A constant vector produces the zero-residual flag and the same uninformative output as the ramp. Callers debugging a real pipeline should pick an input close to their target workload; the synthetic rasters on this page are calibrated for the shape of the tradeoff curve, not for absolute ratios.

## Pipeline picks

The first decision is the model, the second decision is the transform chain, the third decision is the entropy coder. The model almost always determines the ratio. Running [`tuning_pipeline_compare.c`](../examples/tuning_pipeline_compare.c) on the 128 by 256 `i16` raster produces:

```
input: 65536 bytes (128x256 i16, gradient + jitter)

  spec                                       enc    ratio    encode MB/s    decode MB/s
  -------------------------------------------------------------------------------------------
  RAW + LZ                                  56883 B    1.15x        64 enc MB/s      2501 dec MB/s
  PRED_2D / PAETH + LZ                      44665 B    1.47x        55 enc MB/s       488 dec MB/s
  PRED_2D / PAETH + ZZ + LZ                 43855 B    1.49x        64 enc MB/s       512 dec MB/s
  PRED_2D / PAETH + ZZ + BSHUF + LZ         32922 B    1.99x        92 enc MB/s       676 dec MB/s
  PLANE_2D tile=32 + ZZ + BSHUF + LZ        32959 B    1.99x        86 enc MB/s      2324 dec MB/s
```

The `RAW + LZ` baseline compresses 65,536 bytes down to 56,883. A 1.15x ratio on an input with obvious 2D structure means LZ found a handful of matches inside the low-order byte lane and nothing else. The ratio stays around one because the matcher is byte-level; two adjacent `i16` elements that differ by one share exactly one byte, which is too short to emit as a match. Decode throughput is the fastest on this page (2501 MB/s) because LZ on a mostly-literal stream is essentially a memcpy with occasional short copies.

Adding `PRED_2D / PAETH` as the model jumps the ratio to 1.47x. The predictor emits signed residuals in a narrow range around zero; LZ then finds longer runs across the residual bytes than it did across the raw `i16` bytes. Decode throughput drops from 2501 to 488 MB/s. The gap is the model's per-element inverse loop: for every output pixel the decoder reads three neighbors, runs the Paeth rule, and adds the residual. That is cheap arithmetic but it runs at a fixed per-element cost, so the effective throughput tracks the product of element count and cycles per element rather than the LZ copy rate.

`ZIGZAG` on top of the model recovers one percent of ratio (1.49x vs 1.47x) because on this input the residual magnitudes are small enough that the negative values' 0xFF sign-extension bytes were already short enough for LZ to match as a single token. The transform runs for free in throughput terms; the 64 MB/s encode rate is indistinguishable from the model-only row within timing noise. The real payoff for zigzag comes when it is paired with byte shuffle.

`ZIGZAG + BYTE_SHUFFLE` together jump the ratio to 1.99x, a 35 percent payload reduction (43,855 to 32,922 bytes). The byte shuffle transposes the residual stream by byte lane, grouping the high-byte lane (mostly zeros after zigzag) into one long run that LZ collapses into one or two tokens. The throughput actually improves slightly (92 MB/s encode, 676 MB/s decode) because LZ now has much shorter work on a denser, more-repeatable byte stream. This is our default chain on any integer raster that uses a 2D predictor.

The last row swaps the model to `PLANE_2D` at the default 32x32 tile size. Ratio matches `PRED_2D / PAETH + ZZ + BSHUF + LZ` almost exactly (1.99x vs 1.99x), but decode throughput jumps back up to 2324 MB/s. The plane model's inverse is a single-pass add-back of a precomputed plane over each tile, which vectorizes better than Paeth's per-pixel branch. For a raster that fits both model families equally well in ratio terms, `PLANE_2D` is the better pick on read-heavy pipelines. On a raster where the two differ in ratio, the usual rule holds: pick the model that gives the smaller output.

The pattern across these five rows is the one we reach for first on any structured integer input. Model picks up the structure. `ZIGZAG` fixes the sign encoding. `BYTE_SHUFFLE` separates the byte lanes. `LZ` collapses what the other three left behind. Removing any of the four drops the ratio by a double-digit percentage on this input.

## Knob walk

Three tunables matter in v0: the QUANTIZE step size, the LZ level, and the byte-shuffle grouping. All three are driven by [`tuning_knob_walk.c`](../examples/tuning_knob_walk.c).

### QUANTIZE step

`QUANTIZE` maps `f32` or `f64` to a signed integer target via `stored = round((value - offset) * scale)`. Step size is `1 / scale`; target chooses the byte width. On a 16,384-element `f64` sinusoid in `[-100, 100]` plus sub-unit noise:

```
1. QUANTIZE step sweep (N=16384 f64 sinusoid in [-100, 100]):
   input: 131072 bytes

  scale=1       step=1      i8        2802 B    46.78x       301 enc MB/s      2188 dec MB/s
  scale=10      step=0.1    i16      10836 B    12.10x       235 enc MB/s      1867 dec MB/s
  scale=100     step=0.01   i16      17753 B     7.38x       218 enc MB/s      1798 dec MB/s
  scale=1000    step=0.001  i32      24092 B     5.44x       161 enc MB/s      6271 dec MB/s
```

The curve is monotonic: finer precision costs more bytes and spends more cycles per output sample. Moving step size from 1.0 to 0.001 shrinks the ratio from 46.78x to 5.44x because the target dtype widens and the low-order bytes stop being zero. The right pick is always the coarsest step that still meets the downstream task's tolerance; anything finer is spending bits on noise that will be thrown away by whatever consumes the reconstructed values. An `f64` sensor reading whose measurement noise is 0.1 unit should use `scale=10 target=i16` (step 0.1) and not `scale=1000 target=i32`. The four-times payload cost of the i32 row carries no information.

Decode throughput tracks the dtype width non-monotonically: the i32 row is fastest to decode (6271 MB/s) because the inverse quantize is a fused multiply-add per element and the i32-to-f64 path uses the CPU's native integer-float conversion. The i16 and i8 rows do extra sign-extension and the i8 row's decode speed is the lowest at 2188 MB/s. All decodes are faster than any encode row because encoding pays the cost of the scale-and-round loop plus the whole downstream LZ.

### LZ level

`tdc_entropy_level.level` controls the LZ parser's search depth and lazy-match policy. Level 1 skips hash chains entirely (flat hash, no lazy). Level 0 or 3 (the default) uses depth-4 chains and one lazy step. Level 5 doubles the chain depth; level 7 doubles it again with two lazy steps. The underlying matcher is single-pass greedy; level tunes how many candidate matches the parser considers per position.

On a 256 by 256 `i16` gradient raster with `PRED_2D / PAETH + ZIGZAG + BSHUF`:

```
2. LZ level sweep (256x256 i16 gradient raster):
   input: 131072 bytes  model=PRED_2D/PAETH  xform=ZIGZAG+BSHUF

  LZ level=1                         65883 B     1.99x       296 enc MB/s       661 dec MB/s
  LZ level=2                         65538 B     2.00x        91 enc MB/s       666 dec MB/s
  LZ level=3                         65534 B     2.00x        87 enc MB/s       648 dec MB/s
  LZ level=5                         65534 B     2.00x        93 enc MB/s       652 dec MB/s
  LZ level=7                         65534 B     2.00x        99 enc MB/s       654 dec MB/s
```

Three observations. First, decode throughput is flat across levels (648 to 666 MB/s). Level is encode-time only; the decoder reads the same single-stream LZ format regardless. Second, levels 3 through 7 produce identical ratios on this input (2.00x). The upstream pipeline has already done most of the compression work; what reaches LZ is a near-flat byte stream with the same runs every parser variant finds. Third, level 1 runs 3.3 times faster on encode (296 MB/s vs 87-99 MB/s for levels 2+) and loses only 0.5 percent of ratio (1.99x vs 2.00x), which sits inside our "up to 1-2 percent ratio for a real speed win" trade.

The practical recommendation: use level 1 when the pipeline runs once per byte and the downstream reader never sees the payload twice, or when the encode budget is tight enough that a 3x speed multiplier matters. Use the default (level 0 or 3) when the payload will be read many times and the encode-time cost amortizes. Levels 5 and 7 exist for inputs where the deeper lazy search finds matches the greedy parser misses; on a pipeline that has already run a predictor and a byte shuffle, those inputs are rare. Our rule is to leave the level alone unless a specific workload measurements says otherwise.

### BYTE_SHUFFLE grouping

`BYTE_SHUFFLE` has no params; its grouping width reads from the upstream dtype. At elem_size 1 it falls through to memcpy. At elem_size 2, 4, or 8 it transposes the element stream by byte lane. Walking the knob therefore means running the same logical payload at four widths. The delta stream below has identical per-element structure at every width; only the storage dtype changes.

```
3. BYTE_SHUFFLE grouping sweep (N=16384 DELTA_1D integer streams):
   model=DELTA_1D  entropy=LZ

  i8  elem=1 (BSHUF no-op)  (no BSHUF)     176 B    93.09x        51 enc MB/s       759 dec MB/s
  i8  elem=1 (BSHUF no-op)  + BSHUF     176 B    93.09x        53 enc MB/s       648 dec MB/s
  i16 elem=2  (no BSHUF)               108 B   303.41x       100 enc MB/s      1251 dec MB/s
  i16 elem=2  + BSHUF                  111 B   295.21x        96 enc MB/s       853 dec MB/s
  i32 elem=4  (no BSHUF)               114 B   574.88x       162 enc MB/s     17247 dec MB/s
  i32 elem=4  + BSHUF                  112 B   585.14x       179 enc MB/s      8295 dec MB/s
  i64 elem=8  (no BSHUF)               114 B  1149.75x       298 enc MB/s     18461 dec MB/s
  i64 elem=8  + BSHUF                  112 B  1170.29x       270 enc MB/s      7756 dec MB/s
```

At elem_size 1 the shuffle is a no-op in ratio terms (identical 176 bytes). The encode cost is the one extra pass over the buffer; leaving it in the chain on an `i8` input costs a few microseconds and shaves zero bytes. At elem_size 2 the shuffle actually loses 2.7 percent of ratio on this input (111 vs 108 bytes) because the residual stream is already small enough that the shuffle's lane reordering breaks up short matches LZ was finding across elements. At elem_size 4 and 8 the shuffle wins (112 vs 114 bytes, a 1.8 percent payload reduction) because the high six or seven byte lanes are all zero after `DELTA_1D`, and grouping them together produces a single long zero run LZ collapses to one token.

The byte shuffle also halves decode throughput at elem_size 4 and 8 (17 GB/s to 8 GB/s, 18 GB/s to 7 GB/s). That drop is the inverse shuffle's per-element byte gathering running at the SSE2 vector-path cost. A pipeline with stringent decode latency budgets on a wide-dtype delta stream should benchmark the shuffle's actual contribution; on our knob walk it costs more than it saves. On a less-structured input (a 2D predictor residual, a float quantized through QUANTIZE) the shuffle earns its keep because the high byte lanes are not already near-zero before the model ran.

The general rule: include `BYTE_SHUFFLE` after a model that emits small-magnitude integer residuals at dtype widths of 4 bytes or more. Drop it when elem_size is 1 (no work), and measure it when elem_size is 2 (could go either way). Every decision lives downstream of a known residual distribution.

## When to RAW

`RAW + LZ` is the right pick in three narrow cases. The first is a pre-compressed payload that the caller is routing through tdc for the block-record metadata rather than for compression. An encrypted blob, a zstd-compressed column, a PNG embedded inside a column store — every model in tdc will leave those untouched and the transforms have nothing to exploit. Running a predictor across such data costs cycles and saves no bytes; `RAW + LZ` (or `RAW + ENTROPY_NONE`) avoids the work.

The second case is data whose structure the model family does not capture. A shuffled-hash-key column looks like noise to `DELTA_1D`; the deltas have the same distribution as the values. A satellite image with heavy sensor noise and no low-frequency content defeats `PRED_2D`. The residual is the input minus a random value and has the same entropy. Running `RAW + LZ` captures whatever byte-level repetition exists and skips the model's per-element overhead.

The third case is short blocks where the block record header dominates. A 256-byte sensor reading compressed by any codec will still cost the 80-byte header; running a model on top of that adds its own 1- to 20-byte side metadata overhead and a decode-side per-element cost that is out of proportion to the savings. For blocks under a few hundred elements the call is often to batch them into a larger block before compression rather than to run a model on each one in isolation.

[`docs/examples/models_raw.c`](../examples/models_raw.c) has the full RAW baseline on three inputs: an `i32` ramp (where `RAW + LZ` loses to anything), u8 noise (where every codec collapses to 1.0x), and a 16-byte-periodic `u8` pattern (where RAW + LZ hits 35.93x because LZ matches the period directly). Those three inputs are the shape of the decision: structured and predictable picks a model, structurally-periodic picks `RAW + LZ`, anything else asks whether the model's per-element cost is earning its keep.

## Regression watch

A performance regression in tdc either changes the ratio (a bug in a model or transform) or changes the throughput (a regression in the matcher, the wildcopy, or the model inverse). The two are caught by different benchmarks.

Changes to `src/model/*.c` rerun `tuning_pipeline_compare.c` and the model-specific examples under `docs/examples/models_*.c`. The ratios in the five-row pipeline compare are the first line of defense: a predictor change that loses compression on our canonical raster will move one of those numbers. The knob-walk example's BYTE_SHUFFLE sweep additionally exercises `DELTA_1D` on four dtype widths; a regression in the delta path shows up there.

Changes to `src/transform/shuffle.c` rerun the byte-shuffle sweep plus the chain-compare examples under `docs/examples/xforms_*.c`. The 17 GB/s and 18 GB/s decode numbers on the elem_size 4 and 8 lines are especially sensitive: the SSE2 path is what pushes them that high, so a scalar fallback landing silently would drop decode throughput by an order of magnitude without changing correctness. The `adversarial_inputs` test catches the correctness side; the benchmarks catch the throughput side.

Changes to `src/entropy/lz*.c` rerun the LZ level sweep and the `bench_throughput` harness. The five-level sweep above gives us encode-speed numbers at every level in roughly 300 milliseconds; any change that drops level 1 below 250 MB/s or level 3 below 80 MB/s on our reference raster is a regression we want to see before the commit lands. The ratio column is also a check; a level-assignment change that silently maps level 1 to depth-4 chains would show up as encode speed dropping from 296 to 91 MB/s.

Changes to `src/format/*.c` or the block record layout rerun `docs/examples/format_hexdump.c` plus the full ctest suite. Layout changes tend not to affect throughput (the hot paths do not cross block boundaries often) but they do change the byte counts in every example on this page by 1 to 20 bytes depending on the change. A ratio that moves by exactly one byte across every row is usually a format-side edit; a ratio that moves unevenly is usually a codec-side bug.

The rule for landing a performance-sensitive change: reproduce the row the change targets plus at least one row from each of the other two benchmark families (`tuning_pipeline_compare` and the relevant `tuning_knob_walk` section). A single-row check catches the narrowest class of regressions; two rows across two families catches almost everything.
