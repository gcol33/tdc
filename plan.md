# Plan: beat zstd on f64 time series

## The gap

On real f64 time series, zstd wins on ratio while tdc wins on decode speed:

| dataset | tdc best | zstd L19 | gap |
|---|---|---|---|
| USGS streamflow (noisy, non-stationary) | 3.00x (RAW+BSHUF+LZ) | 4.50x | -33% |
| NASA T2M (smooth, periodic) | 2.15x (RAW+LZ) | 3.48x | -38% |

tdc's decode is 2-3x faster than zstd, but nobody cares about speed if
the file is 50% bigger. The goal is **ratio parity or better at equal
or better decode throughput**.

## Why tdc loses today

1. **No float-aware model in the `--from` pipeline.** The real-data
   benchmarks were collected before XOR-delta DELTA1D landed. The bench
   code does test DELTA1D on f64, but the RESULTS.md real-data section
   was never re-run with it. **Phase 0 is just re-benching.**

2. **BSHUF can hurt on periodic f64.** On NASA T2M, BSHUF drops ratio
   from 2.15x to 1.14x because the byte-lane transpose shreds the
   8-byte periodic repetitions LZ was exploiting. Any pipeline that
   uses BSHUF must be conditional on whether it actually helps.

3. **Flat entropy after BSHUF wastes bits.** After XOR-delta + BSHUF,
   the 8 byte lanes have wildly different entropy: lane 0 (MSB of XOR
   residual) is mostly 0x00, lane 7 (LSB) is near-random. A single
   Huffman/FSE/LZ pass on the concatenated stream averages them out.
   Per-lane entropy (LANE backend) should recover most of the gap.

4. **XOR-delta is a single predictor.** It predicts `x[i] = x[i-1]`.
   On smooth monotonic data that's excellent (4-5 leading zero bytes
   in the XOR). On non-stationary data (USGS flood spikes), the delta
   jumps are large and the leading-zero count drops. A second-order
   predictor (`x[i] = 2*x[i-1] - x[i-2]`, extrapolate the slope)
   would track trends much better.

## Phases

### Phase 0 — re-bench with existing tools (no code changes)

Re-run `bench_throughput --from` and `bench_zstd_compare.py --from` on
all three real datasets. This time DELTA1D is available for f64. Also
test pipelines that exist but were never tried on real data:

- `DELTA1D + LZ` (XOR-delta, no shuffle)
- `DELTA1D + BSHUF + LZ`
- `DELTA1D + BSHUF + HUF`
- `DELTA1D + BSHUF + FSE`
- `DELTA1D + BSHUF + LZ+HUF`
- `DELTA1D + BSHUF + LZ_STREAMS`
- `DELTA1D + LZ_OPT + HUF`
- `RAW + BSHUF + LZ_STREAMS`
- `RAW + LZ_OPT + HUF`
- `RAW + LZ_SPLIT`

Deliverable: updated Part 4 in RESULTS.md with best-of across all
pipelines. Establishes the true baseline before writing new code.

**Exit criterion:** know the best existing pipeline per dataset and the
exact gap to zstd L9/L19.

---

### Phase 1 — DELTA1D + BSHUF + LANE (per-lane entropy)

The LANE entropy backend already exists and auto-selects NONE/HUF/FSE
per byte lane. After XOR-delta + BSHUF:

- Lanes 0-4 (high bytes of XOR residual) are mostly zero -> HUFFMAN
  encodes them near-optimally (0.5-2.0 bits/byte)
- Lanes 5-6 are moderate entropy -> FSE
- Lane 7 (LSB) is near-random -> NONE (memcpy, don't waste header)

This is the lowest-hanging fruit: **zero new code**, just wire the
pipeline in the bench and measure.

Add to `bench_throughput.c` f64 section:
```c
tdc_codec_spec sl = {0};
sl.model    = TDC_MODEL_DELTA_1D;
sl.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
sl.entropy[0] = TDC_ENTROPY_LANE;
rc |= run_case("DELTA1D+BSHUF+LANE", block_desc, &b, &sl);
```

Also test `RAW + BSHUF + LANE` (no model) for comparison.

**Expected impact:** +20-40% ratio improvement over DELTA1D+BSHUF+LZ on
smooth data. The high-byte lanes are extremely compressible and LZ was
wasting bits on them. May close the gap to zstd L9 on USGS.

**Exit criterion:** DELTA1D+BSHUF+LANE ratio vs zstd L9 measured on all
three real datasets.

---

### Phase 2 — second-order XOR-delta (DELTA2)

Add a new model `TDC_MODEL_DELTA2_1D` (or a param on DELTA_1D) that
computes double-differencing on float bits:

```
d1[i] = bits[i] XOR bits[i-1]          // first-order XOR delta
d2[i] = d1[i]  XOR d1[i-1]            // second-order XOR delta
```

Why this helps on f64:
- A smooth ramp `y = a + b*x` has near-constant first-order XOR delta.
  First-order already wins here.
- A smooth curve `y = a + b*x + c*x^2` has near-constant **second**-order
  delta. Real signals (temperature, discharge) have trends — d2 tracks
  the curvature, producing more leading zeros than d1.
- XOR second-order is self-inverse: decode is the same XOR chain in reverse.
  No division, no overflow, no precision loss.

Side metadata: 16 bytes (first two raw f64 values).

Implementation: ~60 lines in `delta1d.c`, one new case in the float
encode/decode switch. Add `TDC_MODEL_DELTA2_1D = 0x0008` to codec.h,
register in `registry.c`.

**Expected impact:** +10-20% ratio on trended signals (USGS streamflow)
over first-order. Marginal on periodic signals (NASA T2M) where the
trend is weak.

**Exit criterion:** DELTA2+BSHUF+LANE ratio on all three real datasets.
If it beats DELTA1D+BSHUF+LANE on at least one, keep it.

---

### Phase 3 — FPC-style dual predictor (FCM + DFCM)

This is the heavy hitter. FPC (Burtscher & Ratanaworabhan, 2009) uses
two hash-table predictors:

- **FCM** (Finite Context Method): hash the last k values, predict
  `x[i] = table[hash]`. Good for periodic/repeating patterns.
- **DFCM** (Differential FCM): hash the last k *deltas*, predict
  `x[i] = x[i-1] + table[hash]`. Good for smooth trends.

For each element:
1. Compute both predictions.
2. XOR each with the actual value.
3. Count leading zero bytes in each XOR.
4. Pick the one with more leading zeros (1-bit selector).
5. Store: selector bit + leading-zero count (3 bits) + non-zero tail.

The selector bits pack into a bitstream; the non-zero tails go to a
byte stream. Both are highly compressible.

This is the algorithm that makes tools like `fpzip` and `ndzip`
competitive with zstd on scientific floats.

New model: `TDC_MODEL_FPC_1D = 0x0009`. Implementation:
- `src/model/fpc1d.c` (~250 lines)
- Hash table: 8 KiB-64 KiB (configurable via model_params)
- Context depth k=1 is usually sufficient for time series
- Side metadata: hash table size + first raw value

The residual stream from FPC is **not** a flat byte buffer of the same
size as the input. It's a variable-length encoding (selector bits +
packed tails). This means:
- The transform chain sees a smaller buffer (the packed output).
- BSHUF is irrelevant — the output is already lane-separated by
  construction.
- Entropy stage sees a byte stream that's mostly the non-zero tails.
  LZ or Huffman on this stream is the right choice.

Pipeline: `FPC1D + LZ` or `FPC1D + HUFFMAN`.

**Expected impact:** 4-6x ratio on smooth f64 time series (matching or
beating zstd L19). FPC is the algorithm that was designed specifically
for this problem class.

**Exit criterion:** FPC1D+LZ ratio >= zstd L9 on USGS streamflow. If
it also beats zstd L19, we've won the f64 column entirely.

---

### Phase 4 — bench sweep + RESULTS.md update

Full re-bench of all f64 pipelines (existing + Phase 1-3) on:
1. Synthetic f64 smooth (already in bench_throughput.c)
2. USGS streamflow
3. NASA T2M
4. Any new real f64 datasets worth adding (ERA5 reanalysis? MODIS LST?)

Update RESULTS.md Part 4 with the complete comparison table. Write the
"f64 pipeline recommendation" section: which pipeline to use when.

---

## Priority and dependencies

```
Phase 0 (re-bench)          no code     ~10 min
  |
Phase 1 (LANE wiring)      ~5 lines    ~15 min
  |
Phase 2 (DELTA2)           ~80 lines    ~1 hour
  |
Phase 3 (FPC)              ~300 lines   ~3 hours
  |
Phase 4 (final bench)      no code      ~30 min
```

Phases 0 and 1 should be done together — they require no new algorithms,
just wiring existing backends. If DELTA1D+BSHUF+LANE already matches
zstd L9, Phase 2 becomes optional and Phase 3 becomes the "beat L19"
stretch goal.

Phase 3 (FPC) is the only one that introduces a fundamentally new
prediction algorithm. It's also the one most likely to close the gap
entirely. If time is short, skip Phase 2 and go straight to Phase 3.

## Non-goals

- **Lossy compression.** QUANTIZE exists for that, but the goal here is
  lossless f64 round-trip.
- **f32-specific tuning.** f32 benefits from the same algorithms; once
  f64 works, f32 follows for free (same XOR/FPC logic, smaller element).
- **2D float models.** PRED2D already supports floats. The gap is in 1D
  time series, not rasters.
