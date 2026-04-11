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

## N0 — LZ2 mega-run overhead on constant-value residuals — RESOLVED

**Resolved 2026-04-10.** The combination of P0.3 (LEB128 match-length
encoding) + the structural predictor side-meta rewrite (2026-04-08)
already met and exceeded the acceptance target. Re-bench on 2026-04-10:

| codec                    | ratio      | enc MB/s | dec MB/s |
|--------------------------|----------:|---------:|---------:|
| tdc PLANE2D+BSHUF+LZ2    | 1316.48x  |    392   |   1493   |
| tdc DELTA1D+LZ2 (ramp)   | 159783.01x|   2006   |   3871   |
| zstd L1 (split-planes)   |  327.25x  |  16779   |   5813   |

tdc now beats zstd L1 by 4× on ratio for the split-planes case.
Acceptance was ≥ 500x; actual is 1316x. No throughput regressions on
walk (dec 1652 MB/s, was 1240) or noisy-gradient (dec 313, was 315).
All 24 ctests pass.

**Root cause of the stale 145x number.** The N0 symptom was written
when the LZ2 payload on a 4 MiB zero stream was still 16,461 bytes
(chained-255 match-length encoding) and side_meta was 12,294 bytes
(flat i32 layout). Both bottlenecks were fixed in the P0.1–P0.3 range
and the side-meta rewrite, but SPEEDUP-TODO was not updated. The
acceptance criteria were already met at the time N0 was written — the
numbers just hadn't been re-measured.

No additional work (RLE escape, TDC_XFORM_RLE) was needed.

---

## N1 — f64-aware delta/quantize model

**Symptom.** Both real f64 time series (USGS streamflow, NASA POWER T2M)
show zstd winning on ratio by 30–50% because tdc has no f64-aware model.
BSHUF+LZ2 is a floor, not a model — it can't exploit the smooth
temporal structure that zstd's match-finder picks up at byte level.

| dataset              | tdc best (BSHUF+LZ2) | zstd L19 |
|----------------------|----------------------:|---------:|
| USGS streamflow f64  |                 3.00x |    4.50x |
| NASA POWER T2M f64   |                 2.15x |    3.48x |

tdc decode is still 2–3x faster than zstd on both, so the gap is
purely on the ratio axis.

**Hypothesis.** A delta model on the f64 bit pattern (XOR-delta or
integer-reinterpret delta) would collapse the smooth temporal
correlation into a low-entropy residual stream. The mantissa bits
of adjacent samples share long common prefixes when the signal is
smooth; XOR-delta turns those into leading zeros that LZ2/Huffman
compress well.

**Action.**
1. Implement `TDC_MODEL_DELTA_F64` (or generalize DELTA_1D to handle
   float dtypes via reinterpret-cast to u64/u32 + XOR-delta). The
   model produces a u64 residual stream; downstream transforms and
   entropy see integer bytes as usual.
2. Bench on both real f64 datasets. Target: match or beat zstd L9 on
   ratio while keeping tdc's decode speed advantage.
3. Consider whether XOR-delta or subtract-delta (on the integer
   reinterpretation) works better. XOR-delta produces leading zeros;
   subtract-delta produces small magnitudes. Profile both.

**Acceptance.** On USGS streamflow f64: ratio ≥ 4.0x (vs current 3.0x)
at decode ≥ 2000 MB/s. On NASA POWER T2M: ratio ≥ 3.0x (vs current
2.15x).

**Dead-end notes.**
- Do NOT add lossy quantization as part of this item. tdc is lossless;
  if a quantize stage is added later it belongs in the transform chain
  as an explicit user opt-in, not baked into the model.
- BSHUF actively hurts on periodic f64 (NASA T2M drops 2.15x → 1.14x).
  The f64 delta model must be benchmarked both with and without BSHUF
  to find the right default chain. See N2.

---

## N2 — BSHUF-aware pipeline selection for f64

**Symptom.** On NASA POWER T2M (smooth periodic f64 signal):

| pipeline          | ratio |
|-------------------|------:|
| RAW + LZ2         | 2.15x |
| RAW + BSHUF + LZ2 | 1.14x |

BSHUF shreds the seasonal byte-level repetition that LZ2 was finding.
The byte-lane transpose scatters correlated bytes across lanes, turning
a compressible pattern into noise for LZ2.

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
| RAW + BSHUF + LZ2           | 1.37x |
| PRED2D(PAETH) + BSHUF + LZ2 | 1.21x |
| PLANE2D + BSHUF + LZ2       | 1.28x |

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
1.37x BSHUF+LZ2 floor. On the synthetic gradient: auto selects PRED2D
and matches the 1.64x result. No ratio regression on any existing
bench case.

**Dead-end notes.**
- Don't try to make PRED2D/PLANE2D "better" on noisy small tiles.
  The models are correct; the data just doesn't have enough spatial
  structure to exploit. The right answer is "don't use a model here."

---

## N4 — PRED2D PAETH decode throughput (parked, carried forward)

**Symptom.** PRED2D PAETH decode on u16 noisy gradient 2048x2048:
437 MB/s (after 2-row wavefront + hoisted paeth32). zstd L9 decodes
the same block at 820 MB/s.

**Prior work (all in git history).**
- Register-carry of left/upleft: bench-neutral, reverted.
- 2-row wavefront: +30% (315 → 410 MB/s), kept.
- 4-row wavefront: no gain over 2-row (+1% noise), reverted.
- Hoisted paeth32 algebra: +4% (418 → 437 MB/s), kept.
- SSE2/NEON SIMD paeth32: not attempted (predicted no gain given
  4-row scalar showed no gain — the bottleneck is memory latency
  from the row-above dependency, not ALU throughput).

**Current ceiling analysis.** The 2-row wavefront saturates whatever
uarch resource limits PAETH decode on x86_64 (MSVC /O2). The per-row
serial dependency (`left = dst[r][c-1]`) is the fundamental constraint;
more ILP doesn't help because the backend is already fully issued at
2 lanes.

**Re-open criteria:**
- A real-data bench surfaces u16 PAETH decode as a bandwidth-bound
  bottleneck at a ratio where the model stage is still earning its keep.
- A future compiler/uarch shifts the 2-row wavefront off its plateau
  (try the 4-row template from git history first).
- BSHUF gains a chunked variant enabling fused decode (see parked
  chunked-decode design in git history under old P2.2).

---

## What we are NOT going to do (yet)

- **Replace LZ2 with FSE/Huffman.** LZ2 is near-memcpy on decode. The
  losing cases are model-bound, not entropy-bound. Chaining (LZ2+HUF,
  LZ2+FSE) is shipped for users who want more ratio at throughput cost.
- **Tune LZ2 inner loops.** `vectra/CLAUDE.md` documents dead ends
  (batched parse-then-copy: -7%, Robin Hood hashing: -12-29%).
- **GPU offload.** No bench evidence the CPU pipeline is pegged on
  something the GPU could fix.
- **Fused entropy+model decode (chunked streaming).** Clean design
  exists (git history, old P2.2) but requires ~500 lines across three
  subsystems, only helps non-BSHUF pipelines, and BSHUF appears in most
  good pipelines. Blocked until BSHUF gains a chunked variant.
