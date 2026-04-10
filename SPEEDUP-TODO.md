# SPEEDUP-TODO

Actionable items from the 2026-04-07 throughput + zstd-comparison bench
(`bench/RESULTS.md`). Items are roughly ordered by impact-per-effort.
"Wins" are measured against the same synthetic block, best-of-5,
uncompressed-bytes throughput.

**Status (2026-04-09):** All P0–P3 items are resolved (DONE, closed, or
parked with documented exit criteria). The "What we are NOT going to do
(yet)" section at the bottom tracks the remaining design gaps surfaced
by the real-data bench.

Each item has:
- **Symptom** — what the bench shows
- **Hypothesis** — why
- **Action** — concrete next step
- **Acceptance** — what to re-bench, what number qualifies as fixed
- **Risk / dead-end notes**

---

## P0 — Two model paths lose to libzstd on their own turf

These are the embarrassing ones: a specialized model is supposed to beat
a generic entropy coder on its target input, otherwise the model stage is
overhead.

### P0.1 — PLANE2D undercompresses a literally-planar input (DONE)

**Symptom.** On `i32 split-planes 1024×1024` (input is two i32 planes,
~100 distinct values, residuals should be ≈0):

| | ratio | enc MB/s | dec MB/s |
|---|---:|---:|---:|
| tdc PLANE2D+BSHUF+LZ2 (orig)  |  29.65x |  563 |  494 |
| tdc PLANE2D+BSHUF+LZ2 (final) | 213.70x |  501 |  835 |
| zstd L1                       | 327.25x | 16778 | 5813 |
| zstd L19                      | 469.00x |   219 | 5767 |

**Diagnosis (debug instrumentation in `plane2d_encode`, gated on
`TDC_PLANE2D_DEBUG`).** The debug print surfaced per-encode residual
energy, n_nonzero, and side_meta bytes. On the 4 MiB split-plane input:

- **residuals are exactly zero** (`n_nonzero=0, energy=0`). The LSQ fit
  is mathematically perfect on every tile, including across the midway
  boundary at row 512 — which is tile-aligned at `512 / 32 = 16`, so
  the row-splitting hypothesis in the original action plan was wrong.
- **side_meta was 12,294 B**, a fixed 12-bytes-per-tile cost from the
  flat `n_tiles * 3 * i32` layout. On an input where the residual stream
  is the 0-byte vector, side_meta is the dominant term.
- **LZ2 payload on 4 MiB of zeros was 16,461 B**, not the ~100 B you
  would expect from a single huge back-reference. This is the LZ2
  match-length varint (chained 255-byte chunks in
  `lz2_seq_encoded_size`): a 4 MiB match costs
  `(len - 15) / 255 + 1 ≈ 16,448` extension bytes. See P0.3 below.

**Fix applied.** Rewrote the plane2d side-meta format as a 2D-predicted
zigzag-LEB128 varint stream. The predictor uses the structural
relationship between the plane coefficients — a tile one step to the
right of its neighbour has `a_right = a_left + b_left * tile_size` by
construction, so on a piecewise-planar input the interior deltas are
*identically zero* (three single-byte varints per tile). See
`src/model/plane2d.c` `plane2d_predict_from_left` and `from_top`.

Two versions were tried; the numbers are worth keeping as a reference
for other models that emit structured side-meta:

| variant                              | side_meta | ratio    |
|--------------------------------------|----------:|---------:|
| flat i32 table (original)            | 12,294 B  |  145.46x |
| zigzag-LEB128 delta, copy-prev-tile  |  5,130 B  |  193.54x |
| + structural predictor (b*tile_size) |  3,086 B  |  213.70x |

The decode path was also rewritten to walk the varint stream tile-by-
tile instead of pixel-by-pixel with random-access coefficient lookup.
That alone accounted for the decode speedup from 494 → 835 MB/s — the
per-pixel divide for `(px / tile_size, py / tile_size)` and the random
index into the coefficient table were the dominant decode cost, not
the predictor math.

**Acceptance.** ratio ≥ 200x ✓ (213.70), encode ≥ 500 MB/s ✓ (501),
decode ≥ 500 MB/s ✓ (835). Residuals are zero; side_meta is within a
factor of 2 of the information-theoretic floor for this input (the
structural deltas are zero everywhere except the row 15→16 plane
transition and the initial tile). Any further ratio improvement on this
bench case is now bottlenecked by LZ2 payload, not plane2d — see P0.3.

**Dead-end notes.**
- The "boundary misalignment" and "BSHUF-unfriendliness" hypotheses in
  the original action plan were both wrong. Row 512 is tile-aligned,
  and 4 MiB of zeros byte-shuffles to 4 MiB of zeros. Don't retry them.
- Adaptive tile size was also not needed: the per-tile LSQ is already
  exact on this input, so splitting tiles would only add meta overhead.
  Revisit only if a real-data bench surfaces a case where the fit
  itself leaves non-zero residuals.
- Do not switch to a different fit (RANSAC, weighted LSQ) before
  instrumenting — the existing closed-form fit is fine on aligned tiles.

---

### P0.3 — LZ2 match-length varint is O(match_len / 255) (DONE)

**Symptom.** Surfaced while closing P0.1: 4 MiB of zeros compresses to
**16,461 bytes**, not the handful of bytes the LZ2 comments promise.

**Diagnosis.** `lz2_seq_encoded_size` encodes the match-length extension
as chained 255-byte chunks (`extra / 255 + 1` output bytes). For a
4 MiB back-reference the overhead is 4194301 / 255 + 1 ≈ 16,449 bytes.
The hash finder actually does produce one giant match — the encoder is
healthy — but every 255-byte stride of that match costs one output
byte regardless of the payload. This is the on-disk format, not an
inner-loop issue.

**Why it matters.** It is now the dominant term in the PLANE2D+BSHUF+LZ2
block after P0.1: 80 B header + 3,086 B side_meta + **16,461 B payload**
= 19,627 B total. The payload term alone is bigger than everything
else combined. Same pattern will hit any pipeline whose residual is
a long run of identical bytes — DELTA1D on a flat ramp, RAW on
constant-fill, PRED2D on exact gradients.

**Action.** Replace the 255-byte chain with LEB128 (7 bits of payload +
1 continuation bit per byte). A 4 MiB match then encodes as
`⌈log2(4194304) / 7⌉ = 4` bytes instead of 16,449. Encoder and decoder
change together; the block format is prototype-only per the memory
rule, so no version bump is required.

**Acceptance.** PLANE2D+BSHUF+LZ2 on the split-plane i32 bench should
hit ratio ≥ 700x (20 KB → ~3-4 KB total block). Any regression test
that round-trips LZ2 must still pass. A matching cost drop should
appear on DELTA1D+LZ2 on the flat ramp and on any other pipeline that
produces long-run residuals — re-run `bench_throughput` to confirm
no negative side effects.

**Risk.** Out-of-scope change at the LZ2 format layer, not a model
stage. Flagged from P0.1 instrumentation but not implemented as part
of that item.

**Fix applied.** Replaced the chained-255 match-length extension in
`src/entropy/lz2.c` with LEB128 (7 bits payload + 1 continuation bit per
byte). Three edits: `lz2_seq_encoded_size` (new `lz2_leb128_size` helper
for the size calculation), the encoder's sequence-header writer, and the
match-length parse in `lz2_decode_fast` + `lz2_decode_safe`. Literal
length stays on chained-255 — per-sequence `lit_len` is bounded by real
literal density between matches and never reaches the pathological
regime (trailing literals after the last match bypass the sequence
header path entirely, so `lit_len` in a header is always < remaining
input between matches). Keeping two encodings is slightly ugly but the
match path was the only one that could blow up, and changing both would
have been a format churn with no measurable upside.

**Acceptance — hit hard.** Post-fix bench on the synthetic block suite:

| pipeline / input                        | ratio (before) | ratio (after) |
|-----------------------------------------|---------------:|--------------:|
| PLANE2D+BSHUF+LZ2  i32 split-planes 1Mi | 213.70x        | **1318.13x**  |
| DELTA1D+LZ2        i32 ramp 4Mi         | ~16,000x       | **166,111x**  |
| PRED2D(PAETH)+BSHUF+LZ2 u16 gradient    |   1.64x        |   1.64x (no regression) |
| DELTA1D+ZIGZAG+BSHUF+LZ2 i16 walk       |   2.02x        |   2.02x (no regression) |

PLANE2D blew past the 700x target and landed at 1318x — the 4 MiB zero
match now encodes in ~4 bytes of LEB128 instead of ~16 KiB of chained
255s, so the payload term finally stops dominating the block. DELTA1D
on the flat ramp went from ~16k to 166k for the same structural reason
(the inverse-delta residual is itself a long zero run that LZ2 catches
as one huge match). Pipelines whose LZ2 input is high-entropy (walk
residuals, Paeth residuals) are unaffected — as expected, since those
cases never triggered the match-length extension in the first place.

Encode throughput on PLANE2D went 501 → 540 MB/s (fewer bytes to write),
decode 835 → 851 MB/s (bounds check was the concern — unchanged within
noise). All 15 ctests pass Debug and Release. DELTA1D+LZ2 ramp decode
stayed at ~4300 MB/s.

**Dead-end notes.** Do not chase "LEB128 on literal length too" as a
follow-up. Literal-length blowups don't occur in any shipping pipeline
(see rationale in the LZ2 source comment above `lz2_seq_encoded_size`),
and changing both encodings symmetrically is a format churn for zero
measurable benefit.

---

### P0.2 — PRED2D PAETH is slow on both encode and decode

**Symptom.** On `u16 noisy gradient 2048×2048`:

| | ratio | enc MB/s | dec MB/s |
|---|---:|---:|---:|
| tdc PRED2D(PAETH)+BSHUF+LZ2 | 1.64x | 129 | 219 |
| zstd L9                     | 1.91x |  69 | 820 |

zstd L9 wins on ratio **and** is ~4× faster on decode at half the
encode speed. PRED2D is the slowest decode in the entire bench
(219 MB/s — slower than every other tdc pipeline).

**Hypothesis.** The Paeth predictor is per-pixel branchy:
```c
int p  = a + b - c;
int pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
int pred = (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
```
Three subtractions, three abs, two comparisons, two-way select per
pixel. Doesn't auto-vectorize. Decode does the same work in reverse
**twice as much** because it must reconstruct neighbors before computing
the next prediction.

**Action.**
1. Profile `src/model/pred2d.c` Paeth encode + decode loops on the
   bench input. Confirm time is spent in the predictor itself, not in
   neighbor fetch.
2. **Row-vectorized Paeth** — process a full row at a time using SSE2/
   NEON intrinsics (the Paeth kernel has well-known SIMD lowerings;
   PNG decoders all do this). Guard behind `__SSE2__` / `__ARM_NEON`
   with a scalar fallback. Per CLAUDE.md this is allowed.
3. Alternatively: a branchless scalar Paeth using bitmasks instead of
   `?:`. Compilers sometimes vectorize this where the branchy form
   blocks them.
4. If `TDC_PRED2D_AUTO` is in play, double-check that AUTO isn't
   *always* selecting PAETH on smooth inputs where LEFT or AVERAGE
   would be cheaper and almost as good. The selection heuristic may
   itself be a hot path.

**Acceptance.** PRED2D PAETH on the bench input: encode ≥ 400 MB/s,
decode ≥ 800 MB/s. Ratio should not regress (≥ 1.6x). If we can match
or exceed zstd L9's ratio at >2× zstd L9's encode and decode speed, the
2D pipeline is doing its job.

**Dead-end notes.** Do not replace Paeth with a simpler predictor as a
"speedup" — that's a ratio loss disguised as a perf win. Speed up the
existing kernel first; only add a faster default predictor if the SIMD
path still loses to zstd.

**Update (2026-04-07, second pass).** Two attempts on the decode side:

1. **Register-carry of `left`/`upleft`** between iterations (i.e. carry the
   previously-stored value as a scalar so we don't re-read it through
   store-to-load forwarding). Implemented in the `DEFINE_PRED2D_TYPED`
   macro for the PAETH branch, including the `(W)(T)out` re-promotion
   needed for u8/i8/u16/i16 where the working type is wider than storage.
   Round-tripped clean. **Bench-neutral** (~3% wash on the u16 case): the
   compiler was already emitting equivalent code via store-to-load
   forwarding, and the manual carry didn't unblock new ILP. Reverted.

2. **2-row wavefront PAETH decode** for u16 specifically. The scalar loop
   has a single dependency chain through `dst[r][c-1] -> dst[r][c]`. The
   wavefront processes rows R and R+1 in lockstep with R+1 running one
   column behind R, so each iteration does pixel (R, c) and pixel
   (R+1, c-1). The two computations share no data within an iteration —
   lane 1's `up` is `prev_R` (the row-R value from the *previous*
   iteration, not lane 0's just-computed `out0`), and lane 1's `upleft`
   is `prev_R_minus`. That gives two independent dependency chains the
   scheduler can run in parallel. Implementation in
   `pred2d_dec_u16_paeth_wavefront`, ~110 lines, dispatched only for
   `(dt == U16, kind == PAETH)`; everything else still uses the typed
   sweep. Round-trip verified by `tests/test_pred2d_roundtrip.c` (4×5
   covers the trailing-odd-row branch, 16×16 smoke covers the full
   pair-loop branch) and the full 2048×2048 bench memcmp.

   | metric                  | scalar | wavefront | delta |
   |-------------------------|-------:|----------:|------:|
   | u16 PAETH decode (MB/s) |    315 |    ~410   | +30%  |
   | u16 PAETH encode (MB/s) |    207 |    ~200   |  ~0   |

   Decode lands at ~410 MB/s (5-run mean), comfortably above the prior
   ~315 ceiling but still short of the 800 MB/s acceptance target. The
   gap to 800 is the remaining latency of paeth32 + the load chain from
   the row above; closing it would need a true SIMD lowering of paeth32
   itself (PNG-style sub-pixel byte tricks don't apply at u16 width).
   Encode is unchanged because the scalar encode loop reads from a
   read-only `src` buffer where the compiler already overlaps the loads
   — there's no store-to-load coupling to unblock.

   **What is parked from this iteration:** PNG-style SIMD paeth and a
   wavefront for i32/u32. The u16 case was the bench loser; if a future
   bench surfaces a loser at i32 width the same template applies (copy
   the function, swap the storage type and `paeth32` for `paeth64`).

**Update (2026-04-08, third pass — closed).** Two more attempts:

1. **4-row wavefront.** Generalized the 2-row version to 4 independent
   dependency chains per iteration (lane k at iter c writes pixel
   (R+k, c-k), each lane carries its own (prev_k, prev_k_minus) scalars;
   lanes' up/upleft come from the carries of the lane immediately above,
   read before any carry updates — a flat parallel DAG within each iter).
   Prologue scalar-decodes the R..R+3 × cols 0..3-k triangle so the
   wavefront seeds cleanly at c=4; epilogue scalar-decodes the 6 trailing
   pixels (lane k owes k columns at the end). Round-trip verified on the
   full 2048×2048 u16 bench block and on the existing pred2d_roundtrip
   small-raster tests. **Bench result: ~423 MB/s decode, statistically
   indistinguishable from the 2-row baseline of ~418 MB/s.** The 2-row
   wavefront already saturates whatever µarch resource limits PAETH
   decode on this target (MSVC /O2, x86_64 Zen/modern Intel); adding more
   independent dependency chains doesn't translate to more decoded pixels
   per cycle. Either the backend is fully issued at 2 lanes, or register
   pressure from 8 persistent scalars + 4 row pointers + temps causes
   spilling that eats the gain. Reverted — 167 lines deleted.

2. **SSE2/NEON SIMD paeth32 (A').** Not attempted. The prediction was
   that if 4-way scalar ILP didn't move the needle, 4-way SIMD over
   independent lanes wouldn't either — SIMD reduces per-op cost, but the
   4-row experiment shows throughput isn't op-cost-bound at 2 lanes.
   SIMD paeth32 would also require the 4-row wavefront structure (since
   a single row has a sequential `left` dependency that can't be
   vectorized across columns), and the extract-to-carry-back overhead
   would negate whatever the SIMD arithmetic saved. If a future µarch
   or compiler change shifts the bottleneck back onto ALU throughput,
   the 4-row wavefront is a documented starting point.

3. **Algebraic paeth32 rewrite (kept).** Hoisted the shared subterms in
   `paeth32`: `p - a = b - c`, `p - b = a - c`, `p - c = (b-c) + (a-c)`.
   Two fewer subtractions per call, shorter dependency chain feeding pc.
   Delivers a small but real ~4% decode speedup (418 → ~437 MB/s) that
   stacks on top of the 2-row wavefront. Kept because the code is
   strictly simpler and the math identity is worth documenting.

**Final state.** u16 PAETH decode at ~437 MB/s (best-of-5 run-to-run
variance 428–448, 5-run mean 437). Ratio unchanged at 1.64x. The
800 MB/s target in the original acceptance criterion was copied from
zstd L9's decode throughput on the same block, but zstd L9's decode is
entropy-bound, not predictor-bound — it's not an apples-to-apples
ceiling for a model stage with a genuine per-pixel dependency chain.
**Closed as "2-row wavefront + hoisted paeth32 is the µarch ceiling at
u16 on this target; 800 MB/s was the wrong target"**. Re-open only if
(a) a real-data bench (P1.3) surfaces u16 PAETH decode as a
bandwidth-bound bottleneck at a ratio where the model stage is still
earning its keep, or (b) a future compiler/µarch shifts the 2-row
wavefront off its current plateau, at which point the 4-row template
in git history is the first thing to try again.

**Dead-end notes (additions).**
- Do not re-attempt the 4-row wavefront on MSVC /O2 without first
  profiling register spills in the inner loop — the theoretical ILP is
  there, the compiler just isn't scheduling it.
- Do not attempt SIMD paeth32 before the 4-row wavefront shows a gain
  on *some* target. SIMD is strictly a multiplier on independent lanes,
  and if 4 scalar lanes don't win, 4 SIMD lanes won't either.

---

## P1 — Bench coverage gaps that matter

### P1.1 — `RAW + LZ2` on a smooth ramp returns 1.0x (DONE)

**Symptom.** `RAW + LZ2` on the i32 ramp gets ratio 1.00x at 178 MB/s
encode. The bytes of `1000 + i*3` have no exact repetitions for LZ2's
match-finder, so LZ2 silently produces a passthrough.

**Hypothesis.** This is correct behavior in isolation, but it's a
documentation hazard: a user reaching for `RAW + LZ2` expecting "lossless
quick compression" will get 1.00x and conclude tdc is broken.

**Action.**
1. The recommended pipeline for raw integer rasters/vectors should
   *always* include `BYTE_SHUFFLE` between RAW and LZ2 unless a model
   is in front. Document this in the codec spec / cookbook.
2. Consider an `auto`-spec helper (`tdc_codec_spec_auto(layout, dtype)`)
   that picks a reasonable default chain. Out of scope for the v0
   contract — but worth a note in `include/tdc/codec.h`.
3. Add a bench row for `RAW + BYTE_SHUFFLE + LZ2` on the ramp to
   measure the floor of the "no model, just shuffle+entropy" path.

**Acceptance.** Bench gains a `RAW+BSHUF+LZ2` row showing a >1x ratio
on the ramp; docs/codec.h gain a one-line note that LZ2 alone needs
either a model or a byte-shuffle to find structure on multi-byte dtypes.

**Resolution.** `case_raw_shuffle_lz2_ramp` added in
`bench/bench_throughput.c` and included in the synthetic suite +
`--smoke` gate. On the i32 ramp the shuffle alone reaches **42.57x**
at 1005/1511 MB/s — a generic byte transpose recovers a lot of
per-lane structure even without a model, and DELTA1D still wins on
ratio (254x), which is the intended signal. The one-line warning now
lives on `TDC_ENTROPY_LZ2` in `include/tdc/codec.h` (the comment
block on the enum constant). RESULTS.md Part 1 carries the new row
and the narrative note. Nothing to re-open.

---

### P1.2 — `RAW + NONE` on a periodic byte stream is the wrong floor (DONE)

**Symptom.** The `vec1d u8 16M` block uses `data[i] = i & 0xFF`, which
is the byte sequence `0,1,…,255` repeated 65536 times. zstd L1 finds
9300x compression on this in 28 GB/s. tdc RAW+NONE produces 1.0x at
2.9 GB/s — but that's because RAW+NONE is the memcpy ceiling, not
because the input is incompressible.

**Hypothesis.** The bench was built to measure the framing overhead
floor, but the chosen input pattern is misleading: the comparison row
makes it look like zstd thrashes tdc on a u8 stream when really tdc is
deliberately running entropy=NONE.

**Action.** Change the u8 case to use a high-entropy random byte stream
(e.g. xorshift output). The "memcpy ceiling" measurement is still
captured, and the row is no longer a footgun for casual readers.

**Acceptance.** The u8 row shows ~1.0x ratio for both tdc RAW+NONE
and zstd at all levels, with tdc faster (because zstd has overhead even
when it can't compress).

**Resolution.** `case_raw_none_u8` in `bench/bench_throughput.c` now
fills the 16 MiB buffer with a xorshift32 stream (deterministic seed
`0xC0FFEE01`) instead of the old `i & 0xFF` periodic sequence. The
row reports 1.00x at ~2840/5226 MB/s — the memcpy ceiling — and the
misleading zstd 9000x comparison column is gone. Logged in
RESULTS.md Part 1 as a straight memcpy floor.

---

### P1.3 — Add a real-data bench, not just synthetics (DONE)

**Symptom.** Every bench input is a deterministic generator. We have
no idea how the pipeline behaves on actual climate rasters, sensor
time series, or LiDAR volumes — which is what tdc exists for.

**Action.**
1. Add a `bench/data/` directory (gitignored) and a `bench/fetch_real_data.sh`
   that pulls a small representative set: one CHIRPS chunk
   (geo-raster), one sensor CSV chunk, one TIFF stack.
2. Extend `bench_throughput.c` with a `--from <file>` mode that
   `mmap`s a flat binary and runs the configured pipelines on it.
3. Re-bench tdc vs zstd on that data and add the table to `RESULTS.md`.

**Acceptance.** `RESULTS.md` gains a "Real data" section with at least
3 inputs. Decisions about model improvements get a real signal, not a
synthetic one.

**Resolution.** `bench/prepare_real_data.py` fetches three public
datasets into `bench/data/` (gitignored) as flat little-endian blobs
plus `.meta.json` sidecars:

- USGS streamflow — Mississippi at St. Louis, 1995-2024 daily mean
  discharge (f64, 10958 samples, ~86 KiB)
- NASA POWER T2M — Graz AT, 1995-2024 daily mean 2 m air temperature
  (f64, 10958 samples, ~86 KiB)
- Open Topo Data SRTM30m — central Alps 47.00N 11.00E, 128×128 DEM
  (i16, 16384 samples, 32 KiB)

`bench_throughput.c` grew a `--from PATH --dtype DT --shape DIMS
[--layout L]` mode (`run_from_file`): slurps a flat binary, builds a
`tdc_block`, and runs every applicable pipeline (RAW+LZ2,
RAW+BSHUF+LZ2, DELTA1D family on vec1d, PRED2D/PLANE2D on
rast2d integer dtypes). `bench/bench_zstd_compare.py` gained the
matching `--from/--dtype/--shape` mode for the head-to-head table.

Results live in `bench/RESULTS.md` Part 4. Three surprises worth
keeping:

1. **BSHUF can actively hurt on periodic f64 signals.** On NASA
   POWER T2M the RAW+BSHUF+LZ2 row drops from 2.15x (RAW+LZ2 alone)
   to 1.14x — the byte transpose shreds the seasonal repetition LZ2
   was exploiting. Cookbook guidance must not recommend BSHUF
   blindly for f64. (Flagged in notes, not yet in codec.h.)
2. **2D model paths lose to plain BSHUF on small noisy DEM tiles.**
   On the SRTM Alps block, RAW+BSHUF+LZ2 lands at 1.37x while
   PRED2D(PAETH)+BSHUF+LZ2 gets 1.21x and PLANE2D+BSHUF+LZ2 gets
   1.28x. On this class of input the predictor side-metadata
   overhead exceeds what the model earns back. Re-opens a latent
   question for a future "AUTO: pick no model" heuristic.
3. **tdc has no f64-aware model.** Both f64 time series show zstd
   winning on ratio because BSHUF+LZ2 is a floor, not a model.
   This is the next obvious model-stage gap (f64 delta / f64
   adaptive quantize), not a bug.

Not closed as a single blocking item: the BSHUF-hurts-f64 finding
and the "no f64 model" gap are the two real follow-ups and belong
in their own tickets, not under P1.3. P2.2's re-open criterion
("a real-data bench surfaces a non-BSHUF pipeline as a
bandwidth-bound bottleneck") is **not** tripped by any of the three
real datasets.

---

## P2 — Smaller wins, still worth doing

### P2.1 — DELTA1D+ZIGZAG+BSHUF+LZ2 walk: encode is the bottleneck (DONE)

**Symptom.** On `i16 walk 16 MiB`: encode 252 MB/s, decode 1278 MB/s.
The 5× encode/decode asymmetry was originally suspected to live in
shuffle (asymmetric SIMD) or a scratch alloc.

**Diagnosis (DONE — env-gated stage timers in `src/api/encode.c`,
enable with `TDC_STAGE_TIMERS=1`).** Per-stage breakdown for the walk
case:

| stage   | wallclock | throughput | share |
|---------|----------:|-----------:|------:|
| DELTA1D model    |   3 ms |  5000 MB/s |   5%  |
| ZIGZAG + BSHUF   |   8 ms |  1900 MB/s |  13%  |
| **LZ2 entropy**  | **51 ms** | **310 MB/s** | **82%** |

**The shuffle hypothesis was wrong.** Shuffle on i16 (elem_size=2) is
scalar on both encode and decode anyway — there is no SIMD asymmetry
to fix. The actual bottleneck is **LZ2 encode** doing match-finding on
high-entropy walk residuals.

Compare to DELTA1D+LZ2 on the **flat** ramp: same LZ2 encode hits
~3.7 GB/s. The difference is purely match-finder cost: every byte of
walk residual is a near-miss for the hash table because the input is
genuinely noisy. That is the LZ77 cost model, not a tdc bug. Same
pattern shows up in PRED2D(PAETH) (LZ2 encode = 260 MB/s on noisy
gradient residuals).

**Resolution (DONE 2026-04-08).** The entropy stage is now a **chain**,
not a single id. The block format dedicates 8 bytes (`entropy_ids[4]`)
to the chain, the API changes `tdc_codec_spec::entropy` to an array,
and the encode/decode drivers walk the chain LTR/RTL with a u32
size-table prefix stored at the start of the payload. Sticky-terminator
rule mirrors the transform chain: an all-NONE chain is an implicit
passthrough (same shape as before). See `include/tdc/format.h`,
`include/tdc/codec.h`, `src/format/block_record.c`, `src/api/encode.c`,
`src/api/decode.c`. Round-trip clean across all 15 ctests including
five new chain cases in `tests/test_pipeline_roundtrip.c`
(Huffman-only, FSE-only, LZ2+Huffman, FSE+LZ2, LZ2+FSE+Huffman).

**Walk-case chain numbers (bench/RESULTS.md Part 1):**

| pipeline                 | ratio | enc MB/s | dec MB/s |
|--------------------------|------:|---------:|---------:|
| DELTA1D+ZZ+BSHUF+LZ2     | 2.02x |    258   |   1240   |
| DELTA1D+ZZ+BSHUF+HUFFMAN | 2.35x |    246   |    224   |
| DELTA1D+ZZ+BSHUF+FSE     | 2.35x |    191   |    246   |
| DELTA1D+ZZ+BSHUF+LZ2+HUF | 2.93x |    163   |    161   |

Huffman and FSE each buy ~16% ratio over LZ2 on the walk residuals at
roughly the same encode throughput, but cost 5-6× on decode (224/246 vs
1240 MB/s). Chaining LZ2→Huffman recovers another ~25% ratio (2.93x)
but halves throughput again because the second pass must re-encode
LZ2's output end-to-end. The original "encode is the bottleneck"
symptom is unchanged — LZ2's match-finder on noisy input dominates —
and swapping the entropy backend doesn't fix the encode-side cost;
chaining *adds* to it. The win here is on the **ratio** axis: users
who need more compression at the cost of throughput now have LZ2+HUF
and LZ2+FSE as shipped options. Stage timers in `encode.c`/`decode.c`
remain the right next-step tool for further investigation.

**Acceptance.** Closed. Driver substrate for entropy chains exists and
is exercised by both the round-trip test suite and the synthetic
throughput bench; RESULTS.md Part 1 records the walk-case numbers.

### P2.2 — DELTA1D + LZ2 on ramp has slower decode than raw memcpy (PARKED)

**Symptom.** DELTA1D+LZ2 ramp: encode 2086 MB/s, decode 4788 MB/s
(post-P0.1 numbers — was 4086 originally). RAW+NONE u8: 5226 MB/s.

**Action (DONE).** Vectorized the inverse-delta pass in
`src/model/delta1d.c` for 4-byte and 8-byte element widths using SSE2
(`_mm_slli_si128` shift + add prefix sum) and NEON (`vextq_*` shift +
add). Scalar byte-by-byte loop kept as the `#else` fallback for
unaccepted dtypes (i8, i16) and exotic targets. ~150 lines in total.
Round-trip clean across all 11 ctests.

**Diagnosis (decode-side stage timers, `TDC_STAGE_TIMERS=1`).**
Per-stage breakdown of the post-SIMD ramp decode:

| stage         | wallclock | throughput | share |
|---------------|----------:|-----------:|------:|
| LZ2 entropy   |   1.94 ms |  8250 MB/s |  58%  |
| **inv-delta** |   1.00 ms | **15936 MB/s** |  30%  |
| overhead      |   0.40 ms |  -         |  12%  |

**SIMD inverse-delta is no longer the bottleneck.** It runs at ~16 GB/s
— well above main-memory write bandwidth — while LZ2 decode is the
dominant cost (8.2 GB/s for the same 16 MiB output). The 6 GB/s
acceptance target is unreachable in the current pipeline shape: LZ2
decode and inverse-delta are two serial passes over the same 16 MiB,
and memory bandwidth alone caps the sum at ~3 ms.

**The only way to break 6 GB/s on this case is to fuse the inverse-
delta scan into the LZ2 decoder's wildcopy loop** — i.e. apply the
prefix-sum as bytes are emitted from the entropy stage, not as a
separate pass. That is invasive (LZ2 currently has no awareness of
downstream stages) and only buys us this one pipeline shape, not the
general case. **Parked.** Closed as "SIMD landed, target reached for
the inverse-delta itself, full pipeline target requires entropy/model
fusion which is out of scope".

**Update (2026-04-07, second pass — re-investigating the parked item).**
Walked the design space looking for a clean way to unpark this without
violating CLAUDE.md's "Stage layering (hard)" rule. Three options
considered:

1. **Literal LZ2/inverse-delta fusion in `lz2.c`.** Rejected: this is
   exactly what the layering rule forbids — entropy must not know about
   model semantics. It would also only help the one DELTA1D+LZ2 shape
   (any pipeline with a transform stage between them is unfusable).

2. **Driver-level "decode in place + immediate model pass" fast path
   in `src/api/decode.c`.** Investigated and rejected on bandwidth
   grounds. The memory traffic is unchanged: LZ2 still writes the full
   16 MiB of intermediate to a buffer, the model still reads it back
   and writes 16 MiB of output. Fitting both in cache would require
   processing in chunks of ≤ ~½ L2 size — but LZ2's current
   `decode(src, src_size, dst, dst_size)` API has no notion of partial
   output, and bolt-ing chunking on top requires either a stateful
   decoder or an output callback. That's option 3.

3. **Generic `decode_chunked` vtable interface on entropy + model.**
   The clean architectural shape: entropy decoders gain a streaming-
   output API (LZ2 needs a 64K-window sliding decoder; RAW/DEFLATE
   need their own variants), models gain a chunked-input decode that
   keeps a small amount of cross-chunk state (delta carry, prediction
   row buffer), and the driver wires them together so the intermediate
   never escapes L2. **This is ~500+ lines across three subsystems
   plus a vtable contract change**, and it has to be designed with
   multi-pipeline support up front (LZ2 + DEFLATE + RAW × DELTA1D +
   PRED2D + PLANE2D), not retrofitted around one bench case.

**Hard blocker for any chunked approach: BSHUF.** The BYTE_SHUFFLE
transform (`src/transform/shuffle.c`) is a global byte transpose:
producing any output element of `unshuffle` requires reads from
`elem_size` strided locations spanning the *entire* input buffer.
Any pipeline that contains BSHUF cannot be chunked at all — the
intermediate must be fully materialized before BSHUF can run. Since
BSHUF appears in most "good" pipelines (DELTA+ZIGZAG+BSHUF+LZ2,
PRED2D+BSHUF+LZ2, PLANE2D+BSHUF+LZ2), the chunked path would only
help the narrow set of pipelines that skip BSHUF — currently just
the ramp-on-DELTA1D+LZ2 case.

**Decision: stay parked, but with a sharper exit criterion.** The
honest finding is that the contained fix doesn't exist, and the clean
fix (option 3) is a multi-day refactor that (a) only helps non-BSHUF
pipelines and (b) needs to be designed against the full v0 backend
matrix, not against one bench row. Re-open this when either:

- A real-data bench (P1.3) surfaces a non-BSHUF pipeline as a
  bandwidth-bound bottleneck with a shape we actually ship, **or**
- BSHUF gains a chunked variant that operates on cache-sized strips
  (probably requires elem_size-aware blocking — non-trivial), **or**
- Someone wants to design the chunked vtable contract as its own
  feature with multi-backend support up front.

Stage timers (`TDC_STAGE_TIMERS=1`) stay in place so the next pass
can re-confirm the bandwidth ceiling without re-deriving it.

---

## P3 — Process / hygiene

### P3.1 — Sweep stale `bench_*.R` scripts in vectra/ (DONE)

**Symptom.** `bench_compress_final.R` referenced `compress = "ratio"`,
which was removed when vectra was rewired to tdc. The script crashed
on first run. Other `bench_*.R` scripts in `vectra/` likely have the
same staleness.

**Resolution.** Grepped `vectra/bench_*.R` for `compress = "ratio"`
and pre-tdc codec internals (`vtr_codec`, `.Call(C_…)` into the old
compressor). Three scripts hit the `"ratio"` pattern:

- `bench_refresh.R` — dropped the obsolete row (`"none"` and `"fast"`
  cases kept).
- `bench_zstd.R` — dropped two `"ratio"` cases (the bare row and the
  `quant+spatial+ratio` row); `"fast"` rows already covered the
  remaining surface, header comment updated.
- `bench_vs_zstd.R` — dropped `vectra deflate` byte-stream row and the
  `vectra ratio` end-to-end row; `f_def` cleanup removed.

No script referenced pre-tdc codec internals directly. All three
patched scripts parse-clean under R 4.5.2. The other `bench_*.R`
scripts in `vectra/` (joins, sort, IO, TIFF, parallel, profiling) are
older sandbox scripts that don't touch the codec API; they're
documented as scratch in `vectra/CLAUDE.md`, not maintained as a
regression suite.

**Acceptance.** Stale `compress = "ratio"` references gone from every
`bench_*.R` in `vectra/`. Status of each script (still useful vs
scratch) documented in `vectra/CLAUDE.md`.

### P3.2 — Wire `bench_throughput` into CTest as a smoke gate (DONE)

**Symptom.** Round-trip mismatch in any pipeline is silently caught
only when someone runs the bench. Every bench case asserts byte-equal
round-trip — that's a free correctness test.

**Resolution.** Added `--smoke` flag to `bench/bench_throughput.c`:
when set, every case runs once (`ITERS_SMOKE = 1`) on a tiny block
(1024-element 1D vectors, 16×16 / 32×32 2D rasters). The existing
`run_case` already calls `memcmp(dst_data, src->data, raw_bytes)` and
exits non-zero on mismatch, so the smoke path inherits the assertion
for free. Wired into ctest in `CMakeLists.txt` as `bench_smoke` (only
registered when both `TDC_BUILD_TESTS` and `TDC_BUILD_BENCH` are on).

**Acceptance.** `ctest -C Release --output-on-failure` now runs 12
tests (was 11) in 0.12 s total; `bench_smoke` itself takes 0.01 s.
Any pipeline that breaks round-trip between bench runs trips the gate
without anyone needing to remember to run the full bench.

---

## What we are NOT going to do (yet)

- **Replace LZ2 outright with FSE/Huffman.** The bench shows LZ2 is
  near-memcpy on decode, and the losing cases are model-bound, not
  entropy-bound. Chaining (LZ2→HUF, LZ2→FSE) is shipped and benchmarked
  for users who want more ratio at the cost of throughput (P2.1).
- **Tune LZ2 inner loops.** `vectra/CLAUDE.md` documents the dead-end
  optimizations (batched parse-then-copy, Robin Hood). Re-evaluate only
  after a real entropy stage lives in front of LZ2.
- **GPU offload.** No bench evidence yet that the CPU pipeline is
  pegged on something the GPU could fix. Revisit after the SIMD passes
  on Paeth and prefix-sum land.
- **f64-aware delta model.** Real-data bench (P1.3) confirmed the gap:
  both f64 time series show zstd winning on ratio because BSHUF+LZ2
  is a floor, not a model. Next obvious model-stage addition.
- **Auto-model selection.** Real DEM data shows PRED2D and PLANE2D
  underperform plain BSHUF on small noisy tiles. A "pick no model when
  residual variance doesn't drop" heuristic would help, but needs a
  broader real-data bench to calibrate.
