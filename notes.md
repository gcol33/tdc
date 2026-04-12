# tdc engineering notes

Running log of design decisions that aren't obvious from the code and
aren't worth a full doc page. Append-only — older entries may go stale,
that's fine.

## 2026-04-08 — PLANE2D side_meta and the structural predictor

### The question

PLANE2D was losing to zstd L1 on a literally-planar bench input
(`i32 1024x1024 split-planes`): 29.65x ratio vs zstd L1's 327x. SPEEDUP-
TODO P0.1 listed three hypotheses — tile/boundary misalignment, meta
overhead, BSHUF unfriendliness. None of them were measured. I added a
`TDC_PLANE2D_DEBUG`-gated print to `plane2d_encode` that reports per-
encode `n_nonzero`, `energy`, and `side_meta bytes`, then ran the bench.

### What the instrumentation showed

- `n_nonzero=0, energy=0`. The LSQ fit is exact on every tile — the
  boundary at row 512 is tile-aligned at `ty=16`, so no tile straddles
  the plane transition, so no tile accumulates boundary residuals. The
  alignment hypothesis was just wrong.
- `side_meta=12,294 B`. Flat `n_tiles * 3 * i32` layout pays 12 bytes
  per tile regardless of content. With 1024 tiles that's most of the
  compressed block.
- LZ payload on 4 MiB of zeros was 16,461 B. Not tdc's fault in a
  direct sense but worth noting — see the P0.3 entry in SPEEDUP-TODO.

So the instrumentation flipped the diagnosis. The interesting lever
was the side-meta format, not the fit or the boundary.

### What actually worked

Three variants, measured on the same bench block:

| side-meta encoding                           | bytes   | ratio    |
|----------------------------------------------|--------:|---------:|
| flat `n_tiles * 3 * i32`                     | 12,294  |  145.46x |
| zigzag-LEB128 delta, copy-prev-tile          |  5,130  |  193.54x |
| + structural predictor (`b * tile_size`)     |  3,086  |  213.70x |

The "copy-prev-tile" predictor stores `cf[k] - prev_tile[k]` for each
coefficient, using top-of-column-0 prediction at row transitions. This
already zeros out `b` and `c` on interior tiles of a uniform plane,
which drops 2 of 3 coefficients to a single zigzag-zero byte each. But
`a` changes by a constant stride across tiles, so it still costs ~3
bytes per tile.

The structural predictor exploits the fact that plane coefficients are
not independent: if the same global plane covers tile `(tx, ty)` and
its right neighbour `(tx+1, ty)`, then the neighbour's constant term
satisfies `a_right = a_left + b_left * tile_size` by construction. The
predictor uses `prev_b_fp * tile_size` as the a-delta (and `prev_c_fp`
for the top-neighbour case). On a single uniform plane, all three
delta components drop to zero; the varint stream is three one-byte
zeros per interior tile.

Interior tiles in the bench now cost 3 bytes. The only non-zero bytes
in side_meta are:
- tile (0, 0), which stores its coefficients verbatim (no predictor);
- the first tile of each new row where the top-prediction is also
  exact (0 bytes — most rows);
- the single row-transition tile at ty=16 where the two halves of the
  bench input meet;
- the 31 interior tiles on every row that also cost 3 bytes each.

Final side_meta is 3086 B ≈ 3 * 1024 + slack for the header + initial
tile + row transition. Within a factor of 2 of the information floor
for this input.

### Decoder rewrite was a separate win

The original decoder walked the pixel grid linearly and computed
`tx = px / tile_size, ty = py / tile_size` per pixel, then indexed into
a materialized coefficient table. Two observations killed that shape:

1. The varint stream cannot be random-accessed without a separate
   first-pass walk.
2. The per-pixel divide is a real cost at 4 MiB.

The rewrite walks tiles in the outer loop — reading three varints,
applying the predictor, reconstructing the tile's pixels inline — and
never materializes the coefficient table. Decode MB/s went 494 → 835.

### Dead ends not retried

- **Adaptive tile size.** The fit is already exact on this input. Any
  finer tiling just adds meta.
- **Non-structural delta predictors** (e.g. second-order difference on
  `a` without using `b`). The structural predictor is strictly better
  because it uses a coefficient that is already in the stream, and
  it's exact whenever the plane itself is uniform.
- **Compressing side_meta through a nested model/transform/entropy
  chain.** Would need a second pipeline path in the encoder and a
  longer block-header section. Not worth it when a 20-line varint
  walk hits the acceptance target.

### What is parked

LZ match-length varint (see SPEEDUP-TODO P0.3). On this bench,
payload is 16,461 B because each 255-byte stride of the 4 MiB
back-reference costs one output byte. Replacing the chain with LEB128
would drop the payload to ~4 bytes and make the block ~3-4 KB total.
Out of scope for P0.1, in scope as its own item.


## 2026-04-07 — pred3d AVG3 divisor + Tier 3 octant prologues

### The question

For the 3D average-of-three predictor `pred = (a + b + c) / k`, what's
the right `k` and the right loop shape?

Three candidates:

1. **`/count` (branchy)** — `k` = number of in-bounds neighbors (1, 2,
   or 3). One integrated loop walks the whole volume; per voxel it
   checks `x>0`, `y>0`, `z>0`, sums whichever neighbors are in-bounds,
   and divides by the runtime count.
2. **`/3` (branchy)** — same single integrated loop and same per-voxel
   bounds checks, but always divides by the constant 3 (out-of-bounds
   neighbors contribute 0 to the sum). Subtly different prediction at
   the boundary, but in the boundary the residuals are tiny anyway.
3. **Tier 3 (octant prologues)** — split the volume into 8 octants
   based on (z>0, y>0, x>0). Each octant gets its own loop with the
   bounds checks already resolved at compile time. The inner box (O8,
   `z>0 && y>0 && x>0`) is the hot loop and divides by the compile-time
   constant 3.

### The numbers

`bench/bench_pred3d_avg3.c`, 256³ u16 volume = 32 MiB raw, MSVC `/O2`,
best of 5:

| variant                | ms     | MB/s    | speedup |
|------------------------|--------|---------|---------|
| /count (branchy)       | 19.42  |  1648.0 | 1.00×   |
| /3     (branchy)       | 11.39  |  2809.6 | 1.70×   |
| Tier 3 inner box only  |  4.62  |  6929.7 | 4.20×   |

(Earlier run on the same hardware: 1735 / 2960 / 7008. Within noise.)

### Why Tier 3 wins

Two compounding effects:

- **Compile-time-constant divisor.** In the Tier 3 inner box, `k` is
  literally `3` in the source. MSVC folds `(a+b+c)/3` into a
  multiply-by-magic + shift, no `idiv`. In the `/count` variant `k`
  comes from a runtime counter, so the compiler emits an actual
  division. `/3` already gets the magic-multiply optimization, which
  is why it's 1.7× faster than `/count` despite having the same loop
  shape.
- **No per-voxel bounds checks.** The `/count` and `/3` variants do
  three branches per voxel (`x>0`, `y>0`, `z>0`). Inside the inner box
  these are dead — but the compiler can't prove it without the loop
  split. After the split the inner box has zero bounds checks, the
  loop body is straight-line int math, and it vectorizes cleanly.

The other 7 octants are O(n²) work compared to O(n³) for the inner box,
so their per-voxel cost is irrelevant at any reasonable volume size.
At 256³ the inner box is 99.5% of the voxels.

### Decision

`pred3d.c` ships Tier 3 octant prologues for AVG3, GRAD3D, and PAETH3D.
LEFT/UP/FRONT stay as plain single-axis loops — they only have one
neighbor, so there's no boundary-vs-interior shape difference to
exploit.

### PAETH3D face semantics (separate fix in the same pass)

The original branchy PAETH3D loop computed `paeth(a, b, c, p)` where
out-of-bounds neighbors were 0. On a face (e.g. `z == 0`) this becomes
`paeth(a, b, 0, a+b-ab)` — uses 0 as the third candidate instead of
`ab`. That's not 2D Paeth on the in-bounds triple, it's a different
predictor that happens to be wrong on the boundary.

Tier 3 fixes this for free: each face octant calls strict 2D Paeth on
the in-bounds triple. O4 (`z==0`, interior xy) is `paeth(a, b, ab, a+b-ab)`,
O6 (`y==0`) is `paeth(a, c, ac, a+c-ac)`, O7 (`x==0`) is
`paeth(b, c, bc, b+c-bc)`. Edges reduce to a single neighbor, the corner
is 0.

The reference scorer in `tests/test_pred3d_roundtrip.c` (`ipaeth` helper
+ per-octant switch in `pred3d_compute`) was written to match these
semantics, so the AUTO-select argmin test still passes.

### Dead ends not retried

- **Single integrated loop with `__builtin_expect` / branch hints** — not
  worth trying. The bounds branches inside the inner box are
  perfectly predicted anyway; the cost is the divide and the lost
  vectorization, neither of which a hint fixes.
- **SIMD intrinsics for the inner box** — possibly worth doing later,
  but Tier 3 already gives 4× and the compiler is auto-vectorizing the
  scalar inner loop. Revisit only if AVG3 shows up in a real profile.


## 2026-04-09 — Huffman 11-bit table + PLANE2D integer dispatch

### The question

After the P0 pass, two pipelines were still visibly trailing zstd on
decode even though tdc's ratio was competitive:

- `DELTA+ZZ+BSHUF+HUF` on i16 walk: 222 MB/s decode vs zstd L3 693.
- `PLANE2D+BSHUF+LZ` on i32 split-planes: 489 MB/s decode vs zstd L3
  3808 (zstd is bandwidth-bound on a 13-byte payload, we're not —
  still, the gap was too wide to attribute only to that).

Both smelled like hot-loop shape, not algorithmic floor. The question
was whether the gap was bit-serial decode and per-pixel
double-precision eval (fixable), or a deeper µarch ceiling (not).

### What the instrumentation showed

Eyeballing the hot loops was enough to diagnose both:

- `huffman.c` decode walked canonical codes one bit at a time
  (~8 bit-pops per symbol on a walk-residual alphabet). zstd decodes
  Huffman through a primary table lookup that resolves the common
  short codes in a single load.
- `plane2d.c` decode evaluated the predictor in `double`:
  `round((a + b*lx + c*ly) / 256.0)`. One `idiv`-class op, one
  `cvttsd2si`, and a per-pixel `switch (dtype)` for the store width.
  At 4 MiB of i32 that's 1 M float ops and 1 M indirect branches for
  what should be integer adds and a constant-width store.

Neither required a format change.

### What actually worked

**Huffman decode (`src/entropy/huffman.c`).** Added an 11-bit primary
lookup table sized `1 << 11 = 2048` entries, each holding
`{symbol, code_len}`. Populated at the end of
`huffman_build_decoder_tables` by iterating the canonical code table
and stamping every 11-bit window that starts with the code's bit
pattern. The decode hot path peeks 11 bits, indexes the table, and
advances the bitstream by `code_len`. Slow path (codes > 11 bits)
falls back to the original canonical walk — rare on practical
alphabets because the top 8-10 bits cover > 99% of symbols once the
stream has any skew at all. The 2048-entry table is 4 KiB on the
stack; no allocator plumbing needed.

| pipeline / input                 | dec before | dec after | Δ      |
|----------------------------------|-----------:|----------:|-------:|
| DELTA+ZZ+BSHUF+HUF  walk i16     |      222   |     385   | +74%   |
| DELTA+ZZ+BSHUF+LZ+HUF walk i16  |      162   |     411   | +154%  |

The LZ+HUF chain wins more than HUF alone because on that pipeline
the Huffman decode was a larger share of total wallclock — LZ decode
was already fast, so fixing Huffman moved the whole bar.

**PLANE2D decode (`src/model/plane2d.c`).** Two changes in the same
pass:

1. Replaced the double eval with an exact integer form. The fit
   stores `b`, `c` as 8.8 fixed-point already (`cf_fp`), so the
   per-pixel predictor is `a + ((b_fp * lx + c_fp * ly + 128) >> 8)`,
   using unsigned modular arithmetic and a round-to-nearest bias. Bit-
   identical to the double version because the fit coefficients
   themselves round-trip through the fixed-point quantization — the
   old `/256.0 + round()` was computing the same number with a
   `cvttsd2si` at the end.
2. Replaced the per-pixel `switch (dtype)` with a per-tile
   width-based dispatch. The model only cares about store width (1/2/4
   bytes) and the arithmetic happens in `u32` for all of them; signed
   dtypes wrap identically under two's complement. Three typed inner
   loops (`u8`, `u16`, `u32`) selected once at tile entry, zero
   per-pixel branches.

| pipeline / input            | dec before | dec after | Δ    |
|-----------------------------|-----------:|----------:|-----:|
| PLANE2D+BSHUF+LZ i32 split |      489   |     868   | +78% |

Encode side picks up the integer eval for free because
`plane2d_encode` calls the same `plane2d_eval` helper during residual
computation — 333 → 398 MB/s on the same bench, no loop changes.

### Dead ends not retried

- **16-bit Huffman primary table.** 65536 entries ≈ 256 KiB on stack
  is a non-starter; heap-allocating through `realloc_fn` would work
  but the build cost (population loop runs `1 << (16 - code_len)`
  stamps per code) eats the decode savings on short streams. 11 bits
  is the sweet spot documented in the zstd source for exactly this
  reason.
- **Pre-shifting `b_fp` and `c_fp` column-wise.** Tried mentally:
  `b_fp * lx` could be strength-reduced to an add-per-column, but the
  inner loop is row-major and `ly` is the outer index, so the
  add-per-column only saves one multiply per row. MSVC's auto-
  vectorizer already collapses the `b_fp * lx` pattern into a
  broadcast multiply. Not worth the loop restructure.
- **SIMD plane2d eval.** Parked for a potential next pass; see below.

### What is parked

- **SIMD plane2d eval.** The integer form is now trivially SIMD-
  friendly (`a + ((b*lx + c*ly + 128) >> 8)` over 4-8 lanes of `lx`
  vector). Realistic 3-4× on PLANE2D decode on top of what landed
  here, landing somewhere near 3 GB/s — competitive with zstd even
  though zstd has the bandwidth advantage on this input. Not done in
  this pass because the Huffman fix already pulled more bench gap in
  less code, and SIMD plane2d ties in with the RESULTS.md note about
  PLANE2D being the bench case most sensitive to the "zstd is memcpy-
  ing 13 bytes" asymmetry.
- **FSE precomputed slot table.** FSE walk decode still at ~256 MB/s.
  Same algorithmic shape as the Huffman fix would apply (pack
  `sym / nbits / state_delta` into one struct indexed by state), 1.5-
  2× expected. Lower priority because FSE is only lit up by one bench
  case right now.
- **Dual-stream Huffman with 64-bit accumulator.** The 385 MB/s post-
  fix number is bounded by single-stream branch/load latency; going
  higher means decoding two or four streams in parallel, which is a
  format change. Out of scope until a real-data bench surfaces
  Huffman decode as a bottleneck that matters.


## 2026-04-09 — SSE2 unshuffle for 2- and 4-byte elements

### The question

After the Huffman + PLANE2D pass, PLANE2D+BSHUF+LZ on i32 was still at
868 MB/s decode vs zstd L3's 3808 MB/s. I went in assuming the model
stage was the next lever. The stage timers (`TDC_STAGE_TIMERS=1`) said
otherwise, and the diagnosis flipped.

### What the instrumentation showed

Per-stage decode breakdown on the 4 MiB i32 split-plane block:

| stage   | wallclock | throughput |
|---------|----------:|-----------:|
| entropy (LZ)  |  0.6 ms |  ~6500 MB/s |
| **xform (BSHUF inverse)** | **2.6 ms** | **~1500 MB/s** |
| model (plane2d decode)    |  1.1 ms |  ~3600 MB/s |

Model was already fast — the per-row strength reduction I had just
written (hoisting `c*ly` and replacing `b*lx` with an additive
accumulator, see the `PLANE2D_DEC_TILE_BODY` macro) produced no
visible bench movement. Total decode time is dominated by the BSHUF
inverse stage at 1.5 GB/s, and a faster plane2d model stage was just
shaving a small tail. The model strength reduction stayed anyway —
cleaner code, one less multiply per pixel, no downside — but it was
not the right lever.

Reading `src/transform/shuffle.c` showed why BSHUF was slow: the
`unshuffle_dispatch` had SIMD only for `elem_size == 8` (inherited
from vectra's `byte_unshuffle_8_sse2`). Everything else — including
`elem_size == 4` (i32/u32/f32) and `elem_size == 2` (i16/u16) — fell
through to `unshuffle_scalar`. i32 and i16 are the dominant residual
types for DELTA1D and raster pipelines, so the scalar path was hit by
nearly every pipeline in the bench.

### What actually worked

Added two SSE2 fast paths:

- **`unshuffle_2_sse2`** — single-stage unpack. Load two 16-byte lanes
  (`lane0` = all `b0` bytes, `lane1` = all `b1` bytes), emit 16 fully
  interleaved 2-byte elements via one `unpacklo_epi8` + one
  `unpackhi_epi8`. 16 elements per iteration, 32 bytes out.

- **`unshuffle_4_sse2`** — two-stage unpack. First stage pairs
  `(b0, b1)` via `unpacklo/hi_epi8(r0, r1)` and `(b2, b3)` via
  `unpacklo/hi_epi8(r2, r3)`. Second stage glues the pairs into full
  4-byte groups via `unpacklo/hi_epi16(a0, a2)` and `(a1, a3)`. 16
  elements per iteration, 64 bytes out.

NEON counterparts added in the same pass using `vzip1q_u8/u16` /
`vzip2q_u8/u16` — one-to-one translations of the SSE2 unpack cascade.
Dispatcher now covers `elem_size ∈ {2, 4, 8}` under SSE2 and NEON,
with the scalar path kept as fallback for exotic widths and targets.

### Bench impact (best of 5, synthetic block suite)

| pipeline                             | dec before | dec after | Δ    |
|--------------------------------------|-----------:|----------:|-----:|
| PLANE2D+BSHUF+LZ i32                |      868   |    1509   | +74% |
| RAW+BSHUF+LZ i32 ramp               |     1508   |    2443   | +62% |
| DELTA1D+ZZ+BSHUF+LZ walk i16        |     1154   |    1824   | +58% |
| DELTA1D+ZZ+BSHUF+LZ+HUF walk i16    |      288   |     478   | +66% |
| DELTA1D+ZZ+BSHUF+HUFFMAN walk i16    |      381   |     424   | +11% |
| PRED2D(PAETH)+BSHUF+LZ u16          |      278   |     317   | +14% |

The big +50–75% gains are on pipelines where BSHUF was a dominant
term in the decode wallclock (LZ decode of the residual stream is
fast, so anything that was previously BSHUF-bound now pulls through
the entropy/xform pair much faster). Pipelines where the entropy
stage itself dominates — Huffman-only and FSE-only walk cases —
inherit only a small tail benefit (~10–15%) because the BSHUF stage
was never the bottleneck there.

All 15 ctests pass. The existing `test_byte_shuffle_roundtrip.c`
already covers F32 (elem_size=4) and I16 (elem_size=2) with element
counts 16, 17, 512, 1024 — so both full-SIMD blocks and odd tails
are exercised by the round-trip suite. No new tests were required.

### Where this leaves the zstd comparison

- **PLANE2D+BSHUF+LZ i32**: tdc 1509 vs zstd L3 3808 — gap halved
  (was -77%, now -60%). The remaining gap is bandwidth: zstd
  memcpies ~13 bytes of compressed input, tdc does a real 4 MiB
  BSHUF pass plus the model reconstruction. Closing it further
  needs a model-side SIMD plane2d eval (still parked) **and** LZ
  decode that stays fast on the ~4 MiB zero-run output.
- **DELTA1D+ZZ+BSHUF+LZ walk i16**: tdc 1824 vs zstd best 1009 —
  tdc is now 1.81× ahead (was 1.25×).
- **RAW+BSHUF+LZ ramp i32**: tdc 2443 vs zstd L3 1641 — tdc 1.49×
  ahead. This case used to be basically tied.

### Dead ends not retried

- **Encode-side shuffle SIMD.** The encode path still calls
  `shuffle_scalar` for all elem_sizes. The inverse transpose
  (pack interleaved → planar lanes) needs a different SSE2 sequence:
  either PSHUFB (SSSE3, new dependency) or mask-shift-packus for the
  2-byte case and a second unpack stage for the 4-byte case. Left
  for a follow-up pass — encode isn't the dominant cost in tdc's
  write-once/read-many target workloads, and the user's original
  summary explicitly parked encode throughput as the "LZ match
  finder on noisy residuals" algorithmic floor, not a SIMD problem.
- **`elem_size == 16` for SIMD path**. Not needed — tdc has no
  16-byte fixed-width dtype. If one is ever added (e.g. complex128
  as a primitive), the same 3-stage unpack cascade as the 8-byte
  path extends by one more level.

### What is parked

- **SIMD plane2d eval.** Still a real lever — the strength-reduced
  decoder is at ~3.6 GB/s model-stage throughput and SIMD would
  plausibly take it to 6+ GB/s. Low priority now that BSHUF is no
  longer the ceiling; PLANE2D decode total is bandwidth-contending
  with LZ decode output and full SIMD on the model stage would
  only move total throughput by ~15%.
- **FSE precomputed slot table.** FSE walk decode still at ~262
  MB/s. Packed-u32 slot table (`sym << 24 | cum << 12 | freq-1`)
  eliminates the dependent `norm[s]` and `cum[s]` loads — same
  trick as the Huffman 11-bit table. Expected 1.5–2×. Left for a
  next pass when FSE shows up in more than one bench case.
