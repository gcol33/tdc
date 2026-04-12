# PLANE2D Decode Speedup — Plan & Progress

Target: single-core PLANE2D decode on the 1024×1024 i32 split-planes bench,
from the current ~1.5 GB/s to ~12 GB/s (DDR5 single-core memcpy roof).
Beats zstd (~3.8 GB/s on this input) by ~3×.

This file is both the plan and the progress log. Check boxes as phases
land. Append measurement rows to the results table after each phase.

## Why this is winnable

"Memcpy" is not the ceiling. Single-core `memcpy` on DDR5 is ~12–15 GB/s
because it reads AND writes at full rate. PLANE2D zero-residual decode
has a read footprint of ~0 (just plane coefficients, KB-scale), so it
contends only with writes — a pure store loop can sustain ~15–18 GB/s on
the same hardware. zstd decode on this input is at ~3.8 GB/s, which is
~25% of the write-bandwidth roof, limited by its decode work, not memory.

## Current state (baseline, read from source at 2026-04-11)

- Decoder lives at `src/model/plane2d.c:552-706`. Hot loop is the
  `PLANE2D_DEC_TILE_BODY` macro at lines 633-653. Per pixel: 1 residual
  load, 1 ADD to `acc`, sign branch, rounded divide-by-256, modular
  add+store, 1 ADD to `acc`. All int64.
- Residual contract: `plane2d_decode` hard-requires
  `residual_size == bytes` at line 575. No zero-residual signal today.
- Side-meta header: 6 bytes, `u16 tile_size; u32 n_tiles;`
  (`PLANE2D_META_HDR_BYTES`, line 292). In-place format changes are
  allowed by `feedback_no_versioning_during_prototype`.
- SIMD convention already in tree at `src/model/pred2d.c:90-99`:
  `TDC_PRED2D_HAVE_SSE2` and `TDC_PRED2D_HAVE_NEON` dual-guard with
  scalar fallback. Reuse the pattern.
- Bench hook: one case today, `case_plane2d_shuffle_lz` at
  `bench/bench_throughput.c:344` — PLANE2D+BSHUF+LZ, 1024×1024 i32.
- Blockers for naive SIMD: (a) sign-branch rounding at lines 647-648
  kills autovectorize; (b) int64 accumulator is only 4-wide in AVX2;
  (c) output buffer alignment not guaranteed (no `aligned_alloc` in
  `src/`).

## Phases

### Phase 0 — Baseline microbench ✅

Goal: a stable, reproducible number for every later phase to diff
against.

- [x] Create `bench/bench_plane_decode.c`:
  - [x] Build the same 1024×1024 i32 split-planes input as
    `fill_split_planes_i32` in `bench_throughput.c`.
  - [x] Encode once with PLANE2D+BSHUF+LZ.
  - [x] Decode 200 warm iterations, measure median with
    `QueryPerformanceCounter` (Windows) / `clock_gettime(MONOTONIC_RAW)`
    (POSIX).
  - [x] Report reference numbers in the same process: `memset` and
    `memcpy` over a matched-size buffer.
  - [x] Emit one CSV row per run: `phase,variant,GB_per_sec,ratio`.
- [x] Wire into `CMakeLists.txt` as a new executable linking `tdc`.
- [x] Land on main before any kernel work.

### Phase 1 — Zero-residual fast path (partial: decoder-only)

Goal: stop walking 4 MiB of zeros on decode when the residual is known
to be all-zero.

- [x] `src/model/plane2d.c`: bump `PLANE2D_META_HDR_BYTES` from 6 to 8.
  Add `u8 flags; u8 reserved;` to the side-meta header.
- [x] Define `PLANE2D_FLAG_NO_RESIDUAL = 0x01u`.
- [x] Encoder: scan residual for all-zero with first-nonzero early-out.
  Set the flag when the scan succeeds.
- [x] Extend `plane2d_side_write` signature with a `uint8_t flags`
  parameter. Update callers.
- [x] Split the tile body macro into `PLANE2D_DEC_TILE_BODY` (existing)
  and `PLANE2D_DEC_TILE_BODY_NORES` (no residual load, just write
  `pred`). Dispatch once per tile on `no_residual`.
- [ ] **Driver change deferred** — `src/api/decode.c` enforces
  `walk_bytes == hdr.uncompressed_size`, so the model cannot shorten its
  residual stream to 0 bytes without a coordinated driver change. End-
  to-end residual-skip would need either (a) a model-hint flag in the
  block record that tells the driver to fabricate an all-zero decoded
  residual and skip the entropy+xform chain, or (b) special-casing in
  the entropy stage for trivially-zero streams. Both are out of scope
  for Phase 1 and tracked as **Phase 1.5** below.
- [x] Measure. Row to table.

**Phase 1 actual result**: 1.93 → 2.17 GB/s (~12% win). This tells us
the model inner loop accounts for ~12% of decode time on the split-
planes bench. The other ~88% is BSHUF decode + LZ decode of the 4 MiB
zero-residual stream. Phases 2 and 3 (branch-free + SIMD) will
optimize that same 12% slice — best case they double it to ~24%, or
roughly 2.4 → 2.7 GB/s. **Not remotely enough to beat zstd's 3.71 GB/s.**
The real win is on the driver side.

### Phase 1.5 — End-to-end residual skip (driver change)

Goal: collapse the 88% of decode time currently spent decompressing a
4 MiB all-zero buffer. This is where the ceiling reframing lives.

Two possible implementations:

- **(A) Block-record flag + driver hint.** Add a bit to the block-
  record header that says "residual stream is logically all-zero". On
  decode, the driver skips the entropy+xform pipeline entirely, memsets
  a `walk_bytes`-sized scratch buffer to 0, and hands it to the model.
  The model already does the right thing because of Phase 1. Net effect:
  the 4 MiB memset happens once, then the model walks its tile grid over
  a zero buffer, which is the current Phase 1 behavior minus the BSHUF
  and LZ costs.

- **(B) Model emits zero-byte residual, driver allocates-and-zeros.**
  The model writes `residual_out->size = 0` on encode. The block record
  still carries `uncompressed_size = walk_bytes`. The driver, on
  decode, sees that the encoded residual section is empty and allocates
  a zero-filled buffer of `uncompressed_size` bytes before handing it
  to the model. This keeps the contract `walk_bytes == uncompressed_size`
  intact — only the on-disk representation is compressed to zero.

Option (B) is cleaner (no new block-record flags, shorter .tdc output,
model is in control of its own skip). Option (A) is more general (any
model can skip its residual entirely). Start with (B), upgrade to (A)
if another model (pred2d, pred3d) grows a zero-residual path.

- [x] Prototype option (A+): block-record flag + driver short-circuit.
  Added `TDC_BLOCK_FLAG_ZERO_RESIDUAL` in `include/tdc/format.h`.
  `src/api/encode.c` detects `bufs[cur].size == 0 && n_elems > 0` after
  model encode, wraps the xform + entropy chains in `if (!zero_residual)`,
  and stores `uncompressed_size = n_elems * residual_elem_size`.
  `src/api/decode.c` parses the flag, reuses the existing scratch
  buffer to memset the zero residual, and jumps past entropy + xform
  via a `run_model:` label.
- [x] Encoder side: `plane2d_encode` now sets `residual_out->size = 0`
  when `PLANE2D_FLAG_NO_RESIDUAL` is set.
- [x] Full round-trip tests: all 26 ctest targets pass.
- [x] Measure. Row to table.

**Phase 1.5 actual result**: 2.17 → 2.94 GB/s (+35%). Short of the
optimistic ~10 GB/s prediction because the actual bottleneck is now
the model tile loop itself, not BSHUF+LZ. Breakdown of the 1330 μs
median decode time:
- ~80 μs: 4 MiB memset for the zero residual scratch
- ~1250 μs: plane2d tile loop (int64 acc, sign-branch rounding, scalar)
- ~0 μs: entropy + xform (skipped entirely)

The model inner loop is now the ceiling. Phases 2 and 3 attack it
directly.

**Stop-gate revision**: the old "Phase 1.5 should be near 10–15 GB/s"
target assumed a trivially-fast model. That's not true for plane2d —
the fixed-point predictor + sign-branch rounding is expensive enough
that even a zero-residual tile takes real time. Phase 2's branch-free
+ const-tile special-case is now the next lever; Phase 3 SIMD after
that. Updated ceiling estimate: Phase 2 ~6 GB/s, Phase 3 ~10 GB/s.

### Phase 2 — Branch-free rewrite + int32 fast path + memset removal ✅

- [x] `round_div256_i64` and `round_div256_i32` branch-free helpers.
- [x] `plane2d_eval` updated to call `round_div256_i64`.
- [x] Constant-tile fast path (`cf_b == 0 && cf_c == 0`).
- [x] Int32 accumulator fast path with coefficient-magnitude guard.
- [x] Removed 4 MiB cold-alloc memset in zero-residual decode path.
- [x] Full round-trip tests: 26/26 pass.
- [x] Measure: 2.94 → 7.79 GB/s.

(Original plan items for reference:)

  ```c
  static inline int64_t round_div256(int64_t acc) {
      int64_t sign = acc >> 63;              /* 0 or -1 */
      int64_t abs_acc = (acc ^ sign) - sign;
      int64_t q = (abs_acc + 128) >> 8;
      return (q ^ sign) - sign;
  }
  ```
- [ ] Update `plane2d_eval` (line 206) to call the same helper so
  encoder and decoder produce bit-identical predictions (no drift).
- [ ] Add an early-out at the top of each tile body: if
  `cf_b == 0 && cf_c == 0`, compute `pred` once and fill the tile with
  a typed constant store. Expect the compiler to unroll/vectorize the
  fill.
- [ ] Add a coefficient-magnitude guard that dispatches to an int32
  fast path when
  `cf_a, cf_b*tile, cf_c*tile` all fit in int32. Scalar int64 path
  remains as fallback.
- [ ] Verify autovectorize triggered: `-fopt-info-vec` on gcc/clang,
  `/Qvec-report:2` on MSVC.
- [ ] Full round-trip tests (i8/i16/i32/u8/u16/u32).
- [ ] Measure. Row to table.

### Phase 3 — Explicit AVX2 SIMD intrinsics ✅

- [x] SIMD capability detection macros (AVX2/SSE2/NEON guards).
- [x] `TDC_ENABLE_AVX2` cmake option → `/arch:AVX2` / `-mavx2`.
- [x] AVX2 u32/u16/u8 NORES kernels + AVX2 u32 general-case kernel.
- [x] Macro dispatch: `PLANE2D_NORES_I32_DISPATCH` / `PLANE2D_RES_I32_DISPATCH`.
- [x] NEON: not implemented (no target to test).
- [x] Full round-trip tests: 26/26 pass.
- [x] Measure: 7.79 → 32.0 GB/s.

### Phase 4 — Encode-side symmetry ✅

- [x] `plane2d_eval` uses `round_div256_i64` (updated in Phase 2).
- [x] Full round-trip matrix: 26/26 ctest targets pass.

### Phase 5 — Verification pass ✅

- [x] Re-run numbers on clean build: 31.5/32.1/31.9 GB/s (consistent).
- [x] `bench/RESULTS.md` updated with final PLANE2D row.
- [x] General-case regression check via `bench_throughput.exe`: all
  pipelines round-trip clean.

## Ceiling reframing (after Phase 0)

The 1024×1024 i32 input is 4 MiB — fits comfortably in L3 on a modern
desktop. So `memcpy` and `memset` in the bench are measuring
L3-resident bandwidth, not DRAM. That is the relevant ceiling for this
workload, and it's ~3× higher than the DDR5-DRAM ceiling I estimated
earlier. Two consequences:

1. **More headroom than expected.** 1.93 → 39 GB/s is ~20× of room,
   not ~6×. Phase 2's constant-tile special-case alone could plausibly
   reach ~10–15 GB/s.
2. **NT stores are contraindicated.** Non-temporal stores are a
   DRAM-bound optimization; for an L3-resident output they bypass
   cache and hit DRAM on subsequent reads, which is strictly slower.
   Phase 3's NT-store sub-task is downgraded from "opt-in after SIMD"
   to "only reconsider if bench input grows past L3".

## Results table

Bench machine: the-beast (Zen-class x86_64, Windows 11, MSVC Release,
build_release). N=200 warm iterations, median timing, 4 MiB i32
split-planes input.

| Phase | variant                              | decode GB/s | ratio vs zstd | notes |
|-------|--------------------------------------|-------------|---------------|-------|
| 0     | scalar (baseline)                    | 1.93        | 0.52×         | median of 3 runs: 1.926/1.942/1.936 |
| 0     | memcpy_ref (L3-resident)             | 39.1        | 10.5×         | cache-speed ceiling, not DRAM |
| 0     | memset_ref (L3-resident)             | 52.2        | 14.0×         | pure-write ceiling |
| 0     | zstd L9 (reference)                  | 3.71        | 1.00×         | from bench/RESULTS.md |
| 1     | + zero-residual flag (decoder-only)  | 2.17        | 0.58×         | median of 3 runs: 2.166/2.172/2.185. Only 12% win — model inner loop is 12% of decode time, other 88% is BSHUF+LZ on zero stream. |
| 1.5   | + end-to-end residual skip (driver)  | 2.94        | 0.79×         | median of 3 runs: 2.899/2.938/3.009. BSHUF+LZ gone; remaining cost is ~80 μs memset + ~1250 μs model tile loop. Model loop is the new floor. |
| 2     | + branch-free + int32 + no memset    | 7.79        | 2.10×         | median of 3 runs: 7.830/7.692/7.791. Memset removal (~850 μs on cold alloc) was the big win, not branch-free. int32 fast path fires on all split-planes tiles. |
| 3     | + AVX2 8-wide int32 kernel           | 32.0        | 8.63×         | median of 3 runs: 31.05/32.36/32.50. 82% of memcpy. 8-wide round_div256 + storeu per row. |

## Stop conditions

- Phase 1 < 3 GB/s → stop, investigate. The model is probably wrong.
- Phase 2 < Phase 1 → the branch-free rewrite regressed something.
  Profile before continuing.
- Phase 3 < Phase 2 → constant-tile path is already dominating the
  split-planes bench; Phase 3 only matters for the noisy-gradient
  u16 case. Re-measure there before declaring Phase 3 a win or loss.

## Risks

- **Format change**: `PLANE2D_META_HDR_BYTES` goes from 6 to 8. Any
  `.tdc` blocks already written with the old header become unreadable.
  Allowed by project policy (no version bumps during prototype) but
  worth flagging in the commit message.
- **Encoder/decoder drift**: Phase 2 changes rounding arithmetic in
  both. If one lands without the other, round-trip breaks silently on
  tiles where the sign branch and the branch-free form disagree on
  rounding of exact .5 cases. Land them in the same commit.
- **NT stores and immediate readback**: documented caveat. Provide the
  regular-store path as default and NT as an opt-in build flag.
- **int32 fast-path guard**: conservative bound to prevent overflow at
  extreme coefficient magnitudes. Test with worst-case synthetic input
  (max-slope plane, int32 dtype) to confirm the guard catches it.
