# Models

Models own the first stage of the tdc pipeline. A model reads a typed `tdc_block` and emits two byte streams: a residual stream that the transform chain will massage into entropy-coder-friendly bytes, and a tiny side-metadata stream that the decoder needs to run the inverse. Every model id in `codec.h` points to a vtable with `encode` and `decode` function pointers, acceptance bitmasks for dtypes and layouts, and a human-readable name. The dispatcher routes incoming blocks by layout first (`VECTOR_1D`, `RASTER_2D`, `STACK_2D`, `VOLUME_3D`) and by dtype second; a mismatch returns `TDC_E_LAYOUT` or `TDC_E_DTYPE` before any model code runs.

This page walks every model backend tdc ships in v0 as a first-class pick in the common pipelines: `TDC_MODEL_RAW`, `TDC_MODEL_DELTA_1D`, `TDC_MODEL_PRED_2D`, `TDC_MODEL_PLANE_2D`, and `TDC_MODEL_PRED_3D`. Together they cover every layout the library supports. Dictionary models (`TDC_MODEL_DICT_1D`, `TDC_MODEL_DICT_NUMERIC_1D`), float-specific models (`TDC_MODEL_DELTA2_1D`, `TDC_MODEL_FPC_1D`), the sparse-zero specialization, the fused `TDC_MODEL_QUANTIZE_PRED_2D`, and `TDC_MODEL_STACK_2D` are mentioned where they overlap, but each has its own dedicated walkthrough. The quickstart's layout taste tests sit directly on top of the ids described here; reading that page first will make the examples below land faster.

All six examples on this page compile as part of `TDC_DOC_EXAMPLES` in the top-level `CMakeLists.txt`, and `cmake --build build --config Debug` rebuilds them. Every byte count and ratio printed below came from running the built binaries on the author's Windows MSVC build at `tdc 0.2.0-dev`. The 80-byte block record header counts in every reported encoded size, which is why the RAW baseline on a 1 KiB input reports 1104 bytes instead of 1024.

## What it does

A model produces a residual stream whose statistics are friendlier to downstream stages than the input's. The simplest case is `TDC_MODEL_RAW`, which copies the input byte-for-byte; its residual is the input and the transform + entropy stages carry the whole load. At the other end, `TDC_MODEL_PRED_3D / GRAD3D` runs a trilinear predictor that is exact on any tri-affine volume and leaves a zero residual over every interior voxel. Everything else sits on a continuum between those poles.

Concretely, each model in `src/model/*.c` implements the two function pointers in `tdc_model_vt`:

- `encode` reads the `tdc_block`, computes a per-element prediction from available causal neighbors, stores the modular difference `val - pred` (or `val XOR pred` on the float path) as the residual, and writes any decoder-visible state into `side_out`. Side metadata is tiny by contract: `DELTA_1D` stores zero bytes (the seed lives inside the residual stream), `PRED_2D` and `PRED_3D` store one byte (the resolved predictor kind), `PLANE_2D` stores a per-tile varint-packed coefficient table.
- `decode` takes the residual bytes plus the side metadata and reconstructs `out->data` in place by inverting the predictor. Models never resize `out->data`: the block record carried the exact uncompressed byte count, the caller sized the buffer, the model only walks it.

Models are dimension-aware. `DELTA_1D` iterates a flat vector; `PRED_2D` walks a raster row by row; `PRED_3D` walks a volume octant by octant because every face, every edge, and the corner voxel need different in-bounds rules for their neighbors. That iteration logic is what `layout/traverse.c` and `layout/tiling.c` help with; the model calls into layout, layout never calls into models. Byte shuffling, zigzag mapping, quantization, and LZ or Huffman coding live in the transform and entropy stages and do not belong anywhere in this file.

Every model in v0 accepts only fixed-width numeric dtypes. Strings go through `TDC_MODEL_DICT_1D`, which owns its own `offsets[]` side metadata and uses `tdc_decode_block_varlen` to round-trip; the rest of the pipeline only sees numeric residuals. Integer dtypes wrap modularly at their native width on both sides, which is why a wrapping overflow in an `i64` delta still round-trips. Float dtypes go through an ordered-integer mapping (`src/core/float_order.h`) so that the prediction kernels operate entirely on unsigned integers; float subtraction never appears in the residual path.

## When to use / when NOT

Which model to reach for first falls out of the block's layout and dtype.

**RAW** pays off on two inputs. The first is already-random data (LCG noise, hashed keys, ciphertext) where no predictor will find structure; RAW plus a byte-level entropy coder is the honest baseline. The second is short blocks. When the 80-byte block record header already dominates the compressed size, the predictor's residual would be just as small, so the simpler model wins on decode speed. RAW falls over as soon as the data has obvious structure the transforms and the entropy coder cannot see: a monotonic integer column, a smooth raster, a voxel grid with a gradient. A byte-level LZ matcher cannot turn an `i32` ramp `1000 + i*3` into matches because no two adjacent 4-byte elements share a byte; the ratio returns roughly 1.0.

**DELTA_1D** fits a single numeric column with along-axis correlation: sensor timeseries, monotonic ids, quantized signals, float traces with mostly-shared exponents. The model accepts every fixed-width numeric dtype. The integer path writes `val - prev` in modular arithmetic at the input width; the float path XORs raw bits so that consecutive samples sharing an exponent produce high-order zeros. DELTA_1D is the wrong pick on incompressible columns (random draws, hashed ids), on a column where neighboring values are unrelated (shuffled keys), and on anything multi-dimensional; the model rejects `RASTER_2D` and above with `TDC_E_LAYOUT` before running.

**PRED_2D** covers natural rasters where each pixel has useful left / up / up-left neighbors. Accepted dtypes cover `i8`–`i64`, `u8`–`u64`, and all three IEEE floats. Pick `PAETH` when axes disagree about which neighbor is closest (natural images, thermal maps, sparse bi-axial structure); pick `LEFT` or `UP` when only one axis carries signal; pick `AUTO` when the block is small enough that the 10,000-element sample cost is invisible. The model is wrong on incompressible rasters (camera noise, random mixes): the residual is the input minus a random value, still random, and `RAW + LZ` beats it slightly because it skips the model's walking cost. Float rasters that need quantizing should pass through `TDC_MODEL_QUANTIZE_PRED_2D` instead, to avoid the ordering bug where the predictor emits a float residual that the `QUANTIZE` transform then misinterprets.

**PLANE_2D** fits a per-tile `a + b·x + c·y` plane by solving a 3×3 normal-equation system, stores the three int32 fixed-point coefficients as side metadata, and subtracts the plane before writing the residual. The sweet spot is piecewise-planar rasters with long ramps: DEM tiles, depth maps, fluid-field slices, synthetic aperture products. The wrong pick is a raster with no planar structure: PRED_2D's one-byte side metadata beats PLANE_2D's per-tile coefficients on anything that is not locally planar, and on incompressible noise both collapse to roughly 1.0.

**PRED_3D** is the only `VOLUME_3D` model in v0. `GRAD3D` is exact on tri-affine fields and the right pick when the volume stays smooth over its full extent. `PAETH3D` handles volumes with sharp edges better because on each face and each edge run it reduces to a 2D Paeth or 1D nearest-neighbor rule and adapts to the local discontinuity. `AVG3` and the single-axis kinds (`LEFT`, `UP`, `FRONT`) fit volumes where the correlation lives mostly on one axis. Pick `AUTO` when the volume is a black box; the sample-prefix cost is small and the resolved kind lands in the 1-byte side metadata so the decoder dispatches with no ambiguity. PRED_3D is the wrong pick on `STACK_2D` layouts where the frames are independent; reach for `TDC_MODEL_STACK_2D` with `inter_slice = 0` in that case.

## The RAW baseline

`docs/examples/models_raw.c` encodes three blocks with `tdc_codec_spec_raw()` (identity model, no transforms, passthrough entropy) and with `RAW + ENTROPY_LZ`:

```
-- i32 ramp (N=256): --
RAW                        raw= 1024 encoded= 1104 ratio= 0.93x
RAW + LZ                   raw= 1024 encoded= 1116 ratio= 0.92x
-- u8 noise (128x128): --
RAW                        raw=16384 encoded=16464 ratio= 1.00x
RAW + LZ                   raw=16384 encoded=16476 ratio= 0.99x
-- u8 repeating pattern (N=4096, period 16): --
RAW                        raw= 4096 encoded= 4176 ratio= 0.98x
RAW + LZ                   raw= 4096 encoded=  114 ratio=35.93x
```

Three takeaways. First, every RAW block carries the 80-byte block record header; for short inputs the header alone pushes the ratio below 1. Second, LZ on an `i32` ramp finds no matches (no two adjacent 4-byte elements share a byte) and adds the entropy stage's book-keeping on top of the header, hurting the ratio by another dozen bytes. Third, LZ on a 16-byte-periodic byte pattern collapses to a near-optimal compressed size because the matcher sees 4080 bytes of exact repetition after the first 16. The same input fed to any of the structure-aware models below beats 35.93× without breaking a sweat, but the LZ-only path is a useful reference for what the entropy stage alone can do.

The construction path is the same as every other spec in tdc. The caller sets `spec.model = TDC_MODEL_RAW`, optionally fills `spec.entropy[0]`, initializes a `tdc_buffer` with a `realloc_fn`, and calls `tdc_encode_block`. Decode is the usual peek-then-into pair; nothing about RAW requires a special code path.

## DELTA_1D on integers and floats

`docs/examples/models_delta1d.c` exercises DELTA_1D across three inputs: a near-linear `i64` timestamp column, a random-walk `i64` column, and an `f64` polynomial sine wave. The float path takes the XOR-delta variant automatically; the driver dispatches on dtype.

```
-- i64 timestamp column (N=4096): --
  RAW + LZ          raw=32768 enc=20638 ratio=  1.59x
  DELTA+ZZ+BSHUF+LZ raw=32768 enc=  127 ratio=258.02x
-- i64 random walk (N=4096):
  RAW + LZ          raw=32768 enc=16817 ratio=  1.95x
  DELTA+ZZ+BSHUF+LZ raw=32768 enc= 8268 ratio=  3.96x
-- f64 smooth sinusoid (XOR-delta path, N=4096):
  RAW + LZ          raw=32768 enc=32860 ratio=  1.00x
  DELTA + BSHUF + LZ raw=32768 enc=25460 ratio=  1.29x
```

For the timestamp column the delta stream is almost constant (1000 to 1006); zigzag maps those small signed values to single-byte magnitudes, byte shuffle groups the one interesting byte lane, and LZ collapses the long runs. 258× at 32 KiB input, ending at 127 bytes (of which 80 are the block record header). The random walk is harder: deltas span about ±2048 after the step function used in the example, which is a 12-bit range. DELTA still more than doubles the LZ-only ratio because the walk's absolute values roam over the whole `i64` range and have no byte-level repetition for LZ to catch.

The float case is the one to dwell on. `TDC_DT_F64` goes through DELTA_1D's XOR-delta kernel rather than the integer subtraction kernel. `ZIGZAG` is an integer transform, so the spec drops it. The ratio of 1.29 reflects the fact that a polynomial sine wave changes its exponent often enough that consecutive bit patterns do not share as much as the top mantissa bits would suggest. DELTA_1D on floats is a better pick than DELTA_1D on raw bytes but not a great pick on smoothly-varying signals with fast-moving exponents; `TDC_MODEL_DELTA2_1D` and `TDC_MODEL_FPC_1D` are the specialized float predictors to reach for when this line of the table matters.

## PRED_2D kinds compared

`docs/examples/models_pred2d_kinds.c` encodes a 256×256 `u8` raster with a bi-axial gradient plus a low-amplitude diagonal ripple. Every kind runs; the PAETH round trip is verified with `memcmp`.

```
-- 256x256 u8, bi-axial gradient + diagonal ripple (raw=65536): --
  LEFT       enc=  1895 ratio= 34.58x
  UP         enc=   438 ratio=149.63x
  AVERAGE    enc=  2238 ratio= 29.28x
  PAETH      enc=   188 ratio=348.60x
  PAETH roundtrip: ok
  AUTO       enc=   188 ratio=348.60x
-- 256x256 u8, horizontal gradient (UP is exact): --
  LEFT       enc=   106 ratio=618.26x
  UP         enc=   357 ratio=183.57x
  PAETH      enc=   107 ratio=612.49x
  AUTO       enc=   107 ratio=612.49x
```

For the bi-axial input LEFT explains the `x` component and leaves the `y` component in the residual; UP does the reverse; PAETH captures the plane `x + y - up_left` and reduces the residual to the diagonal ripple alone. AVERAGE is the worst of the lot here because `(left + up) / 2` under C truncation is half the plane rather than the plane itself, leaving a systematic bias. AUTO samples the first 10,000 elements, scores every kind by `sum |residual|`, and picks the winner. On this input AUTO and PAETH produce identical output because AUTO resolved to PAETH and wrote the same 1-byte side metadata.

The horizontal gradient flips the usual intuition. UP predicts every interior row exactly, but pays for the 0..255 seed row; LEFT has a constant-1 residual everywhere, which LZ collapses into one long match of 106 bytes. PAETH reduces to LEFT on this input and lands within one byte of LEFT's output. This is what "PAETH is a safe default" means in practice: when the gradient lives on one axis PAETH picks up the winning single-axis predictor, and when the gradient is bi-axial PAETH captures the plane directly.

The params struct is `tdc_pred2d_params { tdc_pred2d_kind kind; }`. Passing NULL is equivalent to `AUTO`. The resolved kind is always recorded as one byte in side metadata, so the decoder never re-runs the scoring loop.

## PLANE_2D on a tiled DEM

`docs/examples/models_plane2d.c` encodes a 512×512 `i16` raster split into four quadrants, each holding a distinct plane `a + b·x + c·y`:

```
-- 512x512 i16 piecewise-planar DEM (raw=524288): --
  PRED_2D / PAETH                  enc=   188 ratio=2788.77x
  PLANE_2D, tile_size=16           enc=  3296 ratio=159.07x
  PLANE_2D, tile_size=32 (default) enc=   928 ratio=564.97x
  PLANE_2D, tile_size=128          enc=   160 ratio=3276.80x
-- 512x512 i16 incompressible noise: --
  PRED_2D / PAETH                  enc=524381 ratio=  1.00x
  PLANE_2D, tile_size=32           enc=526553 ratio=  1.00x
```

PAETH hits 2788× on the DEM because every pixel except the four quadrant seams matches a linear predictor exactly; only the seam pixels generate residual. PLANE_2D at the default tile_size of 32 loses to PAETH here because each 32×32 tile contributes a three-coefficient side record even on the interior of a single quadrant, and the varint encoder can fold redundant neighbors down to about three bytes per tile but not to zero. Raising `tile_size` to 128 amortizes the side metadata across much larger tiles and wins outright (3276× vs 2788×); lowering it to 16 quadruples the tile count and loses. On the noise input both models collapse; the side-metadata cost shows up as 2.2 KiB of dead weight on top of an already-incompressible stream.

Tuning `tile_size` is the one knob PLANE_2D exposes via `tdc_plane2d_params { uint16_t tile_size; }`. A `tile_size` of 0 resolves to the default 32. The right value depends on two things: how long the locally-planar runs are (a raster with 64-pixel planar tiles is best served by `tile_size = 64`) and how many discontinuities the raster holds (a patchy raster needs more, smaller tiles or the predictor is fitting a plane across a seam). PAETH stays the safer default on unknown rasters because its side metadata is one byte regardless of resolution.

PLANE_2D accepts `i8`, `i16`, `i32`, `u8`, `u16`, `u32`. It rejects `i64` and `u64` (the accumulator inside the normal-equation solver is `int64_t` and cannot prove overflow-free at 64-bit input) and rejects floats (quantize through `TDC_MODEL_QUANTIZE_PRED_2D` first). Layout is `RASTER_2D` only.

## PRED_3D kinds compared

`docs/examples/models_pred3d.c` runs every PRED_3D kind on two volumes: a pure tri-affine field (the GRAD3D sweet spot) and a smooth Gaussian bell with a sharp step edge:

```
-- 24x32x40 i32 tri-affine volume (raw=122880): --
  LEFT       enc=  1284 ratio=  95.70x
  UP         enc=   733 ratio= 167.64x
  FRONT      enc=   627 ratio= 195.98x
  AVG3       enc=   135 ratio= 910.22x
  GRAD3D     enc=   129 ratio= 952.56x
  PAETH3D    enc=   137 ratio= 896.93x
  AUTO       enc=   129 ratio= 952.56x
-- 16x32x32 i32 smooth bell + step edge (raw=65536): --
  LEFT       enc=  5129 ratio=  12.78x
  AVG3       enc= 12278 ratio=   5.34x
  GRAD3D     enc=  7822 ratio=   8.38x
  PAETH3D    enc=  7816 ratio=   8.38x
  AUTO       enc=  7822 ratio=   8.38x
```

With the tri-affine volume GRAD3D hits the zero-residual fast path: `pred = a + b + c - ab - ac - bc + abc` is exact for every function linear in all three axes, so every interior voxel predicts its exact value and only the three boundary seed faces contribute residual bytes. AVG3 comes close (910× vs 952×) because on the inner box `(a + b + c) / 3` is a good plane approximation but loses at most one LSB to C integer truncation per voxel. PAETH3D on smooth data picks the face neighbor closest to GRAD3D's linear predictor and lands within one integer of the same answer.

The bell + step volume makes every kind fight the step edge. LEFT wins on this input because the bell has strong x-axis correlation (r² on x alone is close to 1) and the step edge is aligned with the x-axis boundary, so LEFT gets one big residual at the step and short residuals everywhere else. GRAD3D pays for the step seven voxels at a time because every tri-affine term that crosses the discontinuity contributes a residual byte to the stream, which adds up fast across a 64³ volume; PAETH3D adapts to the step but still needs to carry the edge data on its face-slab path. AVG3 does worst here. `(a + b + c) / 3` is a poor predictor near a step: all three neighbors lie on one side of the edge while the target lies on the other, so the residual is about `step / 3` for every voxel in the affected slab.

The right default for natural volumes (medical CT, seismic blocks, climate fields) is `TDC_PRED3D_AUTO` or `TDC_PRED3D_PAETH3D`. AUTO samples the first slab and picks by `sum |residual|`; the cost is a single slab of prediction at encode time and nothing at decode. `GRAD3D` is the right pick when the volume is known to be smooth throughout (synthetic fields, bandlimited simulations). `LEFT / UP / FRONT` exist for the edge cases where a volume is structured on exactly one axis and the single-axis predictor's residual is already tiny.

## Parameter tuning

Every model except RAW and DELTA_1D exposes at least one knob.

`tdc_pred2d_params { kind }` and `tdc_pred3d_params { kind }` select among the enumerated predictor kinds. Passing NULL resolves to `AUTO`, which scores the kinds on a sample prefix and picks the smallest `sum |residual|`. AUTO adds a small encode-time cost (one sample pass per kind) and zero decode cost; the resolved kind lands in side metadata. Call sites that know their data shape can skip AUTO: `PAETH` on natural 2D rasters, `GRAD3D` on smooth volumes, `PAETH3D` on volumes with discontinuities.

`tdc_plane2d_params { uint16_t tile_size }` sets the per-tile square side. Default is 32. The bench line in `models_plane2d.c` shows the sensitivity clearly: on a four-quadrant 512×512 DEM, `tile_size = 128` beats 32 by 3.5× because the locally-planar runs are longer than 128 pixels; on a raster that genuinely changes every 16 pixels, `tile_size = 16` would win. The encoder's side metadata is a delta-coded varint stream over the 3×N tile-coefficient table, so interior tiles that predict exactly from their left neighbor cost about three bytes each; raising `tile_size` beyond the actual planar-run length costs a large per-tile residual. The knob is a single integer; a sensitivity sweep over `{16, 32, 64, 128}` covers most real use cases.

`tdc_stack2d_params { kind, inter_slice }` is described in full in the STACK_2D walkthrough; briefly, `inter_slice = 1` subtracts the previous slice before running the in-plane predictor (right for frame stacks where consecutive frames are nearly identical) and `inter_slice = 0` treats each frame independently (right for independent frames batched for storage).

`tdc_quantize_pred2d_params` fuses `QUANTIZE` with `PRED_2D` for float rasters. The two sub-params (`scale / offset / target` on the quantize side, `kind` on the pred2d side) apply atomically. Unfused, running `model = PRED_2D, xform[0] = QUANTIZE` on an f32 raster runs the predictor first and produces a float residual the quantizer then mangles; the fused model avoids that ordering bug by construction.

Callers pass params as a `const void *` in `tdc_codec_spec.model_params`, and the receiving model casts back to its own params type. Passing NULL is always legal and always resolves to a working default. Adding a field to a params struct stays forward-compatible as long as the new field has a meaningful zero value; otherwise the format version bumps.

## Benchmarks

The numbers below came from running each example once on the author's machine (Windows 10, MSVC 2022, `tdc 0.2.0-dev`, Debug configuration). They are ratio comparisons, not absolute throughput; a separate [performance tuning page](../performance/tuning.md) covers encode and decode speed with warm-cache medians.

| Input (raw bytes) | Spec | Encoded | Ratio |
|---|---|---:|---:|
| 4096-element i64 timestamps (32,768 B) | RAW + LZ | 20,638 | 1.59× |
| 4096-element i64 timestamps | DELTA_1D + ZIGZAG + BSHUF + LZ | 127 | 258.02× |
| 4096-element i64 random walk (32,768 B) | RAW + LZ | 16,817 | 1.95× |
| 4096-element i64 random walk | DELTA_1D + ZIGZAG + BSHUF + LZ | 8,268 | 3.96× |
| 4096-element f64 sinusoid (32,768 B) | RAW + LZ | 32,860 | 1.00× |
| 4096-element f64 sinusoid | DELTA_1D + BSHUF + LZ | 25,460 | 1.29× |
| 256×256 u8 bi-axial gradient (65,536 B) | PRED_2D / PAETH + LZ | 188 | 348.60× |
| 256×256 u8 horizontal gradient (65,536 B) | PRED_2D / LEFT + LZ | 106 | 618.26× |
| 512×512 i16 DEM quadrants (524,288 B) | PRED_2D / PAETH + ZZ + BSHUF + LZ | 188 | 2788.77× |
| 512×512 i16 DEM quadrants | PLANE_2D tile=32 + ZZ + BSHUF + LZ | 928 | 564.97× |
| 512×512 i16 DEM quadrants | PLANE_2D tile=128 + ZZ + BSHUF + LZ | 160 | 3276.80× |
| 24×32×40 i32 tri-affine (122,880 B) | PRED_3D / GRAD3D + ZZ + BSHUF + LZ | 129 | 952.56× |
| 16×32×32 i32 bell + step (65,536 B) | PRED_3D / PAETH3D + ZZ + BSHUF + LZ | 7,816 | 8.38× |

Reading the table: the model stage is where ratio comes from. `RAW + LZ` on the timestamp column returns 1.59× because LZ matches short byte-runs inside the i64 representation, and adding DELTA_1D in front drops the residual to near-constant bytes so the ratio jumps to 258×. `RAW + LZ` on the f64 sinusoid returns 1.00× because the bit patterns are unique enough that LZ finds nothing; adding DELTA_1D's XOR path helps modestly. `PRED_2D / PAETH` on the DEM reaches 2788× because PAETH predicts exactly on three of the four quadrants' interiors, and raising `PLANE_2D`'s tile_size to 128 matches or beats PAETH when the planar runs are longer than the tile.

Two pointers are worth flagging against these numbers. The zero-residual fast path (`TDC_BLOCK_FLAG_ZERO_RESIDUAL`) fires whenever the model reports an all-zero residual, which short-circuits the transform and entropy stages entirely. The tri-affine volume (952×) hits this path; most of the ratio comes from the flag, not from LZ. Synthetic benchmarks that are exactly predictable will therefore overstate real compression on noisy data; the random-walk line and the bell + step volume are calibration points for what a genuinely-noisy input looks like after a well-chosen model.

The next-closest tdc backend for each row is the RAW + LZ baseline above it. A comparison against the closest third-party library (`zstd -3`, say) belongs in the [performance tuning page](../performance/tuning.md), where encode / decode timing and a third-party reference appear side by side.

## Edge cases

`docs/examples/models_edge_cases.c` exercises the corners:

```
-- empty block (n=0, i32, VECTOR_1D): --
  DELTA_1D + LZ                raw=   0 encoded=  92  (empty block, no payload to decode)
-- single element (n=1): --
  DELTA_1D + LZ                raw=   4 encoded=  96  memcmp=ok
-- all-equal block (n=4096, zero-residual fast path): --
  DELTA_1D + LZ                raw=16384 encoded= 106  memcmp=ok
-- layout mismatch (PRED_2D on VECTOR_1D): --
  PRED_2D on VECTOR_1D         error: layout not accepted by stage
-- degenerate raster 1x256 (PRED_2D/PAETH): --
  PAETH on 1xW raster          encoded=100 bytes
```

An empty block (`n_elems = 0`) is valid. The encoder writes the 80-byte block record header plus 12 bytes of per-stage framing and returns `TDC_OK`; the decoder's peek reports zero bytes required and no payload needs copying. Callers that ferry zero-length slices through tdc can rely on this path round-tripping.

A single-element block is also valid. DELTA_1D stores the seed (no deltas to write), the transform chain sees 4 bytes, and the block record lands at 96 bytes (80 bytes of header plus 16 bytes of per-stage framing). The ratio is meaningless on a 4-byte input; what the line demonstrates is that the pipeline does not choke on `n_elems = 1`.

An all-equal block trips the zero-residual fast path. DELTA_1D maps a constant column to a non-zero seed plus zero deltas, which is not quite an all-zero residual but produces a residual that LZ collapses almost entirely. The output lands at 106 bytes on a 16 KiB input. Synthetic benchmarks of "the compressor" run against this kind of input, and the resulting ratios measure the fast path, not the matcher.

Layout mismatches are caught before any model code runs. `PRED_2D` is restricted to `RASTER_2D` via its `accepted_layouts` bitmask; feeding it a `VECTOR_1D` block returns `TDC_E_LAYOUT` with the output buffer untouched. The same gate catches `DELTA_1D` on a 3D block (returns `TDC_E_LAYOUT`) and `PLANE_2D` on an `f64` raster (returns `TDC_E_DTYPE`). Callers can run `tdc_block_validate` ahead of time to catch shape and layout errors, though the encoder will re-check regardless; the validation is cheap enough to leave on in release builds.

A degenerate 1×256 raster is a boundary case for PRED_2D: every pixel's "up" neighbor is out of bounds because `ny = 1`, so the predictor degenerates to LEFT on the single row. The PAETH kernel handles this internally (the tie-break rules fall out of the formula), and the output is 100 bytes, roughly what DELTA_1D on the same 256-byte column would produce. 1-row and 1-column rasters are unusual but valid.

## Integration notes

The five model backends on this page pair naturally with different transform + entropy chains. A short menu:

- **RAW** pairs with `ENTROPY_LZ` or `ENTROPY_LZ_OPT` for the "compress the bytes as-is" case. Add `XFORM_BYTE_SHUFFLE` when the input is a multi-byte dtype whose per-lane statistics differ (a timeseries of `f32` measurements with a near-constant exponent byte, a `u32` column whose high lane is nearly all zero). The shuffle transposes the input by byte lane, so the 4-byte `f32` elements become four contiguous runs (all sign bits, all upper-exponent bytes, all mantissa-high bytes, all mantissa-low bytes) and LZ then matches within each run. Do not chain `XFORM_QUANTIZE` directly on floats under RAW; reach for `TDC_MODEL_QUANTIZE_PRED_2D` instead so the quantization precedes the predictor, and the decoder rehydrates through the paired model automatically.
- **DELTA_1D** on integers: chain `ZIGZAG`, `BYTE_SHUFFLE`, `LZ`. On floats drop zigzag and keep `BYTE_SHUFFLE + LZ`.
- **PRED_2D** on integers: same chain as DELTA_1D. On `u8` rasters zigzag is a no-op (the dtype is already unsigned) and byte shuffle at width 1 is another no-op, so the example in this page runs straight from the model into LZ. On `u16` or `u32` rasters keep byte shuffle in the chain.
- **PLANE_2D** uses the same `ZIGZAG + BYTE_SHUFFLE + LZ` chain as PRED_2D. Side metadata is varint-packed internally, so no separate transform is needed for the coefficient table.
- **PRED_3D** uses `ZIGZAG + BYTE_SHUFFLE + LZ` on integers and `BYTE_SHUFFLE + LZ` on floats. A volume whose residual is already near-zero after GRAD3D benefits most from byte shuffle at the natural element width.

Models conflict with a few chains. `XFORM_QUANTIZE` before a predictor model is almost always wrong: the predictor wants to see the same dtype on encode and decode, so quantization at that point should go through the fused model id (`TDC_MODEL_QUANTIZE_PRED_2D`). Stacking two entropy stages (`entropy[0] = LZ, entropy[1] = HUFFMAN`) is legal and works but tends to lose to a single `ENTROPY_LZ_SPLIT` for the same reason a two-pass image compressor loses to a single pass that sees both streams jointly.

All five backends honor the same allocator convention. The encoder grows `out->data` through the caller's `realloc_fn` (same `(user, ptr, n)` contract as the library's internal allocations). The decoder asks the caller to size `dst->data` ahead of time using `tdc_decode_peek`; the model writes in place and never reallocates. Scratch memory for ordering tables, residual staging, and the encoder's zero-residual check all goes through the same allocator if the caller uses `tdc_decode_block_ex`; otherwise internal scratch falls back to libc `realloc`. R consumers plug an arena here; CLI tools use plain `realloc`.

The [Transforms](transforms.md) and [Entropy](entropy.md) pages cover the downstream stages. The [Predictor math](../theory/predictors.md) page derives Paeth, GRAD3D, and the PLANE_2D normal-equation solve. [Performance tuning](../performance/tuning.md) carries the timed throughput numbers that this page only ratio-summarizes.
