# f64 entropy — closing the zstd gap

## Situation

On real f64 data, zstd L19 beats tdc's best pipeline (RAW+LZ_STREAMS) by
4-14% on ratio and 2-3x on decode speed. No prediction model helps — DELTA1D,
DELTA2, FPC all produce residuals with *more* byte-level entropy than raw bytes.
The win path is better entropy coding on raw f64 bytes, not better models.

Synthetic structured data (ramps, walks, gradients) is already dominated.
This plan targets the real-data gap.

---

## 1. Fix stream coder mis-selection (ratio, easy)

**Problem:** LZ_STREAMS picks FSE for the offset stream post-repcode because
Shannon entropy says so, but tdc's FSE compresses near-constant histograms
~5x worse than Huffman (comment at lz_streams.c:353). The repcode transform
collapses offsets into a histogram dominated by symbol 1 — Huffman's fixed
code-length floor is irrelevant when one symbol is >90%.

**Fix:** Force HUFFMAN when the top symbol exceeds ~85% frequency, or when
the alphabet after repcode is ≤4 effective symbols. Alternatively, just
always try both and pick the smaller output (the current "try all" loop may
be short-circuiting).

**Expected gain:** 0.5-1% ratio on f64 datasets where repcodes dominate.
Small but free.

---

## 2. Optimal parser for LZ_STREAMS (ratio, medium)

**Problem:** LZ_STREAMS uses the greedy parser (chain_depth=4, lazy_depth=1).
The optimal DP parser exists in lz_opt.c and already shows measurable gains
on periodic f64 (noted in the code), but it isn't wired into the STREAMS
backend. On the NASA regional grid, the gap between LZ_STREAMS (3.66x) and
zstd L19 (4.28x) is 14% — a better parser could recover a chunk of that.

**Approach:** Add an LZ_STREAMS level parameter. Level 0 = greedy (current),
level 1+ = optimal DP feeding into the same 4-stream layout. The stream
splitting and per-stream entropy coding stay identical; only the sequence
selection changes.

**Expected gain:** 3-6% ratio improvement on large f64 grids (based on
LZ_OPT vs LZ greedy gaps on similar data). Encode will be slower (expected,
acceptable for archival mode). Decode unchanged.

---

## 3. Larger LZ window (ratio, medium)

**Problem:** LZ_MAX_OFFSET is 1 MiB (lz_internal.h:28). The NASA regional
grid is 8.3 MiB — zstd's default window is 8 MiB at L19. Matches beyond
1 MiB are invisible to tdc. For f64 with 8-byte elements, a 1 MiB window
covers ~131K values. Seasonal data with period >131K samples, or spatial
grids with row strides >1 MiB, can't find long-range matches.

**Approach:** Bump to 4 MiB window (LZ_MAX_OFFSET = 1<<22). This requires:
- Wider offset encoding in LZ_STREAMS (2 extra bits per symbol tier)
- Larger hash table or chain array (4x memory)
- Benchmark to confirm the ratio gain justifies the memory cost

**Risk:** Memory pressure at encode time. Decode is fine (no hash table).
May need a level-gated approach: small window for fast levels, large for
high levels.

**Expected gain:** 2-5% on large grids. Diminishing returns on small
datasets (<1 MiB).

---

## 4. Interleaved Huffman decode (decode speed, medium-hard)

**Problem:** tdc Huffman decodes one symbol at a time from a single
bitstream. zstd interleaves 4 Huffman streams decoded in parallel, hiding
data dependency latency. On f64 literal streams (the largest stream in
LZ_STREAMS), this is the decode bottleneck.

**Approach:** Split the literal stream into 4 quarter-streams at encode time.
Decode with 4 independent bit-buffer states, round-robin or 4-wide unrolled.
The fast2 table already exists — extending to 4 independent cursors is
mechanical. Header cost: 4x2 bytes for stream boundaries.

**Expected gain:** 1.5-2x decode throughput on Huffman-heavy pipelines.
This directly attacks the 2-3x decode gap vs zstd.

---

## 5. 6-stream Huffman (decode speed, hard)

**Problem:** zstd uses 4 interleaved streams. Going to 6 could extract more
ILP on wide-issue CPUs (the-beast has Zen 5 / Raptor Cove class). But
diminishing returns vs header overhead.

**Gate:** Only pursue if item 4 shows <1.5x speedup. If 4-stream already
closes the gap to within 30% of zstd decode, stop here.

---

## 6. Tighter FSE table_log for small alphabets (ratio + speed, easy)

**Problem:** FSE uses TABLE_LOG=12 (4096 states) for the full 256-symbol
case. For f64 LZ_STREAMS, most sub-streams have effective alphabets of
8-32 symbols. The quantization overhead at 4096 states is fine, but the
decode table is 16 KB — cache pressure when 4 streams each have their own
FSE table.

**Fix:** Cap TABLE_LOG at ceil(log2(alphabet)) + 3, minimum 6. For a
16-symbol alphabet, that's TABLE_LOG=7 (128 states, 512-byte table).
Keeps the decode tables in L1.

**Expected gain:** 5-15% FSE decode speedup. Marginal ratio improvement
from reduced quantization waste on small alphabets.

---

## 7. Context-coded literals (ratio, hard, speculative)

**Problem:** f64 bytes have structure that flat Huffman/FSE misses. The
exponent field (bits 52-62) is highly predictable from recent values.
The high mantissa bytes correlate with the exponent. A simple order-1
context model on the literal stream could capture this.

**Approach:** For each literal byte, use the previous byte (or the byte
at the same position in the previous 8-byte element) as context. Build
per-context Huffman tables (like zstd's literal context modes). This is
essentially what zstd does with its "compressed literals with Huffman +
context" mode.

**Risk:** Complexity explosion. Context tables multiply header overhead.
May only help on large blocks where the per-context statistics are stable.

**Gate:** Only pursue after items 1-4 are done and measured. If the ratio
gap is <5%, skip this.

---

## 8. Exponent-mantissa split (ratio, speculative)

**Problem:** f64 has three semantic fields: sign (1 bit), exponent (11),
mantissa (52). Compressing them as a flat byte stream mixes three very
different distributions. An exponent-only stream would be tiny and
extremely compressible. A mantissa stream would be high-entropy but
might benefit from different coding.

**Approach:** New transform that splits f64 into exponent+sign (2 bytes)
and mantissa (6 bytes) sub-streams. Each gets its own entropy path.

**Risk:** Only helps if the per-field entropy sum is less than the mixed
entropy. For smooth data this is likely true; for noisy data the mantissa
dominates and the split adds overhead.

**Gate:** Prototype on NASA regional grid offline. If <2% ratio gain, drop.

---

## Priority order

```
[must-do]  1. Fix stream coder mis-selection     — free ratio, 1 hour
[must-do]  2. Optimal parser for LZ_STREAMS       — biggest ratio lever, 1-2 days
[must-do]  4. Interleaved Huffman decode           — biggest decode speed lever

[should]   3. Larger LZ window                     — ratio on large grids
[should]   6. Tighter FSE table_log                — easy decode speed win

[maybe]    7. Context-coded literals               — only if gap still >5%
[maybe]    8. Exponent-mantissa split              — prototype first
[skip]     5. 6-stream Huffman                     — only if 4-stream insufficient
```

## Success criteria

- RAW+LZ_STREAMS ratio within 5% of zstd L19 on NASA regional grid
- Decode speed within 1.5x of zstd L19 on f64 datasets
- No regression on i16/u16 pipelines
