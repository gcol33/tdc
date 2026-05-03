# SPEEDUP-TODO

Actionable items from the 2026-04-07 bench results (`bench/RESULTS.md`).
All prior P0–P3 items are resolved and removed — their dead-end notes
and design decisions live in git history (commit range up to 2026-04-09).
N0 is resolved (verified 2026-04-10) — see note below.

Items are ordered by impact-per-effort. Each item has:
- **Symptom** — what the bench shows
- **Hypothesis** — why
- **Action** — concrete next step
- **Acceptance** — what number qualifies as fixed
- **Dead-end notes** — what not to retry

---

## N0 — LZ mega-run overhead on constant-value residuals — RESOLVED

**Resolved 2026-04-10.** The combination of P0.3 (LEB128 match-length
encoding) + the structural predictor side-meta rewrite (2026-04-08)
already met and exceeded the acceptance target. Re-bench on 2026-04-10:

| codec                    | ratio      | enc MB/s | dec MB/s |
|--------------------------|----------:|---------:|---------:|
| tdc PLANE2D+BSHUF+LZ    | 1316.48x  |    392   |   1493   |
| tdc DELTA1D+LZ (ramp)   | 159783.01x|   2006   |   3871   |
| zstd L1 (split-planes)   |  327.25x  |  16779   |   5813   |

tdc now beats zstd L1 by 4× on ratio for the split-planes case.
Acceptance was ≥ 500x; actual is 1316x. No throughput regressions on
walk (dec 1652 MB/s, was 1240) or noisy-gradient (dec 313, was 315).
All 24 ctests pass.

**Root cause of the stale 145x number.** The N0 symptom was written
when the LZ payload on a 4 MiB zero stream was still 16,461 bytes
(chained-255 match-length encoding) and side_meta was 12,294 bytes
(flat i32 layout). Both bottlenecks were fixed in the P0.1–P0.3 range
and the side-meta rewrite, but SPEEDUP-TODO was not updated. The
acceptance criteria were already met at the time N0 was written — the
numbers just hadn't been re-measured.

No additional work (RLE escape, TDC_XFORM_RLE) was needed.

---

## N1 — f64 delta model: validate on real data — PARTIALLY ADDRESSED

**Implementation complete (2026-04-12).** DELTA_1D now handles f64/f32
via ordered-integer transform (tdc_f64_to_ordered / tdc_ordered_to_f64).
Synthetic bench (commit 91989ed) confirms DELTA1D+LZ is the winner for
smooth f64 (2.67x ratio, 1.6 GB/s decode on synthetic ramp); BSHUF
hurts because it fragments the zero-heavy high bytes that LZ exploits.

**Remaining.** Re-bench on the two real f64 datasets (USGS streamflow,
NASA POWER T2M) with DELTA1D+LZ and DELTA1D+BSHUF+LZ chains to verify
the acceptance targets:

| dataset              | tdc best (no model) | target (with DELTA1D) | zstd L19 |
|----------------------|--------------------:|----------------------:|---------:|
| USGS streamflow f64  |              3.00x  |             ≥ 4.0x    |    4.50x |
| NASA POWER T2M f64   |              2.15x  |             ≥ 3.0x    |    3.48x |

If targets are met, mark resolved. If not, investigate whether the
ordered-integer transform is suboptimal for periodic signals and
whether XOR-delta would be better.

**Dead-end notes.**
- Do NOT add lossy quantization as part of this item. tdc is lossless.
- BSHUF actively hurts on periodic f64 (NASA T2M drops 2.15x → 1.14x
  without a model). Must bench DELTA1D both with and without BSHUF.

---

## N2 — BSHUF-aware pipeline selection for f64

**Symptom.** On NASA POWER T2M (smooth periodic f64 signal):

| pipeline          | ratio |
|-------------------|------:|
| RAW + LZ         | 2.15x |
| RAW + BSHUF + LZ | 1.14x |

BSHUF shreds the seasonal byte-level repetition that LZ was finding.
The byte-lane transpose scatters correlated bytes across lanes, turning
a compressible pattern into noise for LZ.

**Hypothesis.** BSHUF helps when per-lane structure dominates (integer
ramps, integer rasters) but hurts when cross-byte repetition dominates
(periodic f64 where the same exponent+high-mantissa bytes recur at
seasonal intervals).

**Action.**
1. Document in `codec.h` that BSHUF is not recommended unconditionally
   for f64 dtypes.
2. If/when `tdc_codec_spec_auto()` is implemented, it must skip BSHUF
   for f64 unless a model (delta, quantize) has already transformed
   the stream into integer residuals.
3. Consider a lightweight probe: encode a small prefix with and without
   BSHUF, pick the winner. Only worth doing if auto-selection is in
   scope.

**Acceptance.** Codec.h carries a note. No f64 pipeline in the cookbook
or auto-spec includes BSHUF without a preceding model stage.

---

## N3 — Auto-model selection ("pick no model" heuristic)

**Symptom.** On real SRTM DEM i16 (128x128, noisy terrain):

| pipeline                    | ratio |
|-----------------------------|------:|
| RAW + BSHUF + LZ           | 1.37x |
| PRED2D(PAETH) + BSHUF + LZ | 1.21x |
| PLANE2D + BSHUF + LZ       | 1.28x |

Both 2D model paths score **worse** than plain BSHUF. The predictor
side-metadata overhead exceeds what the model earns back on small
noisy tiles.

**Hypothesis.** On small blocks with high spatial noise, the model's
residual variance doesn't drop enough to offset its overhead (side
metadata + prediction cost). A "trial encode" heuristic that compares
residual energy with and without the model could detect this.

**Action.**
1. Define `TDC_MODEL_AUTO` (or a spec-level flag) that runs a cheap
   probe: encode a small tile with the candidate model, compare
   residual entropy estimate (e.g. byte histogram entropy) to the raw
   stream. Pick "no model" if the model doesn't reduce entropy by a
   threshold (e.g. 10%).
2. Bench on the three real datasets + the synthetic suite. Auto must
   never pick a model that loses to no-model on ratio.
3. The probe cost must be < 5% of total encode time on the synthetic
   suite.

**Acceptance.** On SRTM DEM: auto selects "no model" and matches the
1.37x BSHUF+LZ floor. On the synthetic gradient: auto selects PRED2D
and matches the 1.64x result. No ratio regression on any existing
bench case.

**Dead-end notes.**
- Don't try to make PRED2D/PLANE2D "better" on noisy small tiles.
  The models are correct; the data just doesn't have enough spatial
  structure to exploit. The right answer is "don't use a model here."

---

## N4 — PRED2D PAETH decode throughput (parked, ceiling confirmed 2026-05-03)

**Symptom.** PRED2D PAETH decode on u16 noisy gradient 2048x2048:
~480 MB/s (after 4-row wavefront + hoisted paeth32). zstd L9 decodes
the same block at 820 MB/s.

**Prior work (all in git history).**
- Register-carry of left/upleft: bench-neutral, reverted.
- 2-row wavefront: +30% (315 → 410 MB/s), kept.
- 4-row staggered wavefront (commit 67f9f8c): +10% over 2-row, kept.
  Dispatched for rasters with nx ≥ 4 && ny ≥ 5.
- Hoisted paeth32 algebra: +4%, kept.
- 4-row SSE2 SIMD paeth32 (`paeth_4x_sse2`): kept, dispatched.
- 8-row AVX2 SIMD wavefront `pred2d_dec_u16_paeth_wf8`
  **implemented and benched 2026-05-03** — measured **0.42× of wf4**
  throughput on Raptor Lake (i9-14900K, gcc 14.2 `-mavx2 -O3`,
  `bench_throughput.exe` interleaved microbench, best of 50).
  The cross-128-bit-lane shift (`permute2x128 + alignr_epi8`) adds
  ~3 cycles to the per-iter critical path, and 8 `_mm256_extract_epi32`
  scatter-stores double the store-port pressure vs SSE2's 4 extracts —
  neither offset because the bottleneck is the row-above memory
  dependency, not vector ALU throughput. Kernel and helper remain
  compiled and round-trip-tested (`tests/test_pred2d_wf_consistency`)
  so a future re-bench on another uarch or fused pipeline can
  re-evaluate without redoing the work; dispatcher continues to use
  wf4. See `bench/RESULTS.md` for the table.
- UP predictor: matches PAETH on ratio while decoding 42% faster.
  For noisy data where PAETH's adaptive selection adds little, UP
  is the better default.

**Current ceiling analysis.** The 4-row wavefront is at the uarch
ceiling on x86_64. Empirically confirmed by the 2026-05-03 wf8 bench
above: doubling lane width nets **negative** speedup on Raptor Lake.
Further ILP does not help because entropy/BSHUF stages now dominate
the pipeline AND the kernel itself is memory-latency bound on the
row-above dependency.

**Re-open criteria:**
- A real-data bench surfaces u16 PAETH decode as the dominant stage
  in a pipeline where the model is earning its keep on ratio.
- BSHUF gains a chunked variant enabling fused decode (see parked
  chunked-decode design in git history under old P2.2).
- A future uarch (Zen 5+, AVX-512 with native scatter, MSVC's
  improved AVX2 codegen) where the wf8 kernel might re-cross the
  break-even point. The wf8 implementation is kept warm so this
  is a one-line dispatcher change rather than a re-implementation.

---

## What we are NOT going to do (yet)

- **Replace LZ with FSE/Huffman.** LZ is near-memcpy on decode. The
  losing cases are model-bound, not entropy-bound. Chaining (LZ+HUF,
  LZ+FSE) is shipped for users who want more ratio at throughput cost.
- **Tune LZ inner loops.** `vectra/CLAUDE.md` documents dead ends
  (batched parse-then-copy: -7%, Robin Hood hashing: -12-29%).
- **GPU offload.** No bench evidence the CPU pipeline is pegged on
  something the GPU could fix.
- **Fused entropy+model decode (chunked streaming).** Clean design
  exists (git history, old P2.2) but requires ~500 lines across three
  subsystems, only helps non-BSHUF pipelines, and BSHUF appears in most
  good pipelines. Blocked until BSHUF gains a chunked variant.
