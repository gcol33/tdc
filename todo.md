# tdc decode speed — SIMD & fused decode plan

**Goal:** Close the 2x decode gap vs zstd on regional NASA f64 data.

Current: LZ_STREAMS 269 MB/s decode, zstd L19 724 MB/s.
Target:  ≥600 MB/s decode on LZ_STREAMS (regional NASA, 8.3 MiB f64).

---

## Phase 0 — MEASUREMENT RESULTS (2026-04-14)

Instrumentation landed:
- `src/core/decode_profile.{h,c}` — shared TU rdtsc counters +
  offset histogram, flag-gated via `--profile` (CLI) or
  `TDC_DECODE_PROFILE=1` (env).
- `bench_throughput --decode-only [--decode-iters N]` — skips encode
  timing, runs N decode iters for stable numbers.
- Instrumented `lzs_decode_fused` (L1 path) and `lzs_reconstruct`
  (default L11 path, the 2-pass decoder).

### Regional NASA f64 (1.084 M samples, ~8.3 MiB), 5 iters

**RAW + LZ_STREAMS** (`lzs_reconstruct`, 269 MB/s, the target):

| section | % of cycles | cyc/seq |
|---|--:|--:|
| symbol (prefetch+addr walk) | 23.4% | 23.4 |
| literal copy                | 26.7% | 26.7 |
| match copy                  | 26.9% | 27.0 |
| other (mo bounds)           | 23.1% | 23.1 |

Full breakdown (after instrumenting SYMPRE + STREAMDEC, 150 cyc/seq):

| section   | %    | cyc/seq | what it is |
|-----------|-----:|--------:|------------|
| streamdec | 16.1% | 24.4 | entropy (HUF/FSE) decode of ll/ml/off streams |
| sympre    | 17.9% | 27.1 | symbol→value (extra bits + repcode) |
| symbol    | 15.4% | 23.3 | prefetch pipeline addr-walk |
| literal   | 17.7% | 26.8 | literal copy |
| match     | 17.5% | 26.5 | match copy (AVX2 wildcopy32 active) |
| other     | 15.4% | 23.4 | bounds checks |

**Profile is flat.** No stage dominates; every stage is ~25 cyc/seq.
This rules out single-stage SIMD wins as the path to 600 MB/s.

**RAW + LZ_STREAMS L1** (`lzs_decode_fused`, fused decoder):

| section | % of cycles | cyc/seq |
|---|--:|--:|
| symbol reconstruct | 34.2% | 52.3 |
| literal copy       | 16.5% | 25.2 |
| match copy         | 18.3% | 28.0 |
| other (bounds+rep) | 31.0% | 47.4 |

### Offset histogram (both decoders, same data)

| offset bucket | reconstruct % | fused % |
|---|--:|--:|
| 1-3   | 0.1% | 0.0% |
| 4-7   | 0.0% | 0.0% |
| 8-15  | 1.0% | 0.3% |
| 16-31 | 1.6% | 0.4% |
| 32-127 | 6.3% | 1.5% |
| 128-1K | 10.4% | 5.7% |
| 1K-16K | 29.3% | 31.0% |
| >16K   | 51.3% | 61.1% |

### What this changes about the plan

1. **Plan's 50-60% match-copy estimate was wrong.** Match copy is
   18-27% of cycles. The dominant costs are symbol reconstruct (fused
   path) and literal copy + bounds checks (reconstruct path).

2. **Item 1.4 (dec32/dec64 small-offset bootstrap) is moot here.**
   Offsets 1-15 are 0.3-1.1% of sequences. Not worth 2 hours of work
   for the regional NASA target.

3. **Items 1.1-1.3 and 1.5 are already implemented.** Read
   `src/core/simd.h`: `tdc_copy16` uses `_mm_storeu_si128`,
   `tdc_wildcopy32` is gated on `TDC_HAVE_AVX2` (enabled by
   `/arch:AVX2` in CMake), and `lzs_reconstruct` has an 8-ahead
   prefetch pipeline. The baseline numbers already include these.

4. **Real ROI order (revised after full breakdown):**

   The pipeline has six stages of ~25 cyc/seq each. To raise 269 MB/s
   toward zstd's 724, we need to *remove a stage*, not speed one up.

   - **A. Fuse sympre into streamdec (~27 cyc/seq reclaim).** The
     entropy decode iterates n_seqs producing bytes into ll_dec/
     ml_dec/off_dec; sympre then walks those arrays again producing
     uint32. If the Huffman/FSE decode emits the reconstructed value
     directly, one pass + its memory traffic disappears. Requires
     changing the sub-decoder signature or adding a "decode+expand"
     variant. Estimated: ~270 → ~320 MB/s.
   - **B. Merge execute into sympre (fused pipeline).** `lzs_decode
     _fused` already does this for L1 but lacks the 8-ahead prefetch
     pipeline that lzs_reconstruct has. Port the prefetch pipeline
     to fused and route default L11 through it. Eliminates the
     lit_lens/match_lens/match_offs scratch arrays entirely.
     Estimated: ~320 → ~420 MB/s.
   - **C. Reduce cyc/seq in fused sympre.** The plan's phase 2
     branchless-symbol changes apply here. Extra-bit reads dominate;
     merging the bit buffer into the stream decoder's state would
     save ~5-10 cyc/seq.
   - **D. Raise ml/seq ratio.** The parser chose ~4.3 M sequences
     for 8.3 MiB — ~2 bytes/seq. If we can improve the parser so
     fewer sequences cover the same bytes (LZ_OPT-style optimal
     parse but without its 1 MB/s encode speed), we amortize the
     150 cyc/seq over more bytes. This is an encode-side change
     that raises decode throughput for free.
   - **E. Architectural (phase 5 tANS / single-bitstream).** Last
     resort if A-D plateau before 600 MB/s.

5. **What NOT to do (confirmed moot for regional NASA):**
   - 1.4 small-offset bootstrap — offsets <16 are <2%.
   - 3.2 SSE2 bitshuffle — BSHUF is destructive on periodic f64.
   - 3.1 HUF4 more symbols/iter — relevant only to HUF pipelines,
     not LZ_STREAMS.

---

---

## Phase 6 — zstd source audit (2026-04-15) & revised plan

Sources read in full: `zstd-ref/lib/decompress/zstd_decompress_block.c`,
`huf_decompress.c`, `common/bitstream.h`, `common/fse.h`,
`common/fse_decompress.c`. Citations below are file:line in that tree.

Current state: 279 MB/s regional NASA f64 decode (post-fusion, 2026-04-14),
zstd L19 at 724 MB/s. Gap = 2.6×. Previous phase-0 measurements showed a
flat profile (~25 cyc/seq per stage, 6 stages). The zstd audit reveals
the flat profile is the *symptom* — zstd eliminates stages outright.

### Findings that change the plan

**F1. Single shared DStream across LL/ML/OF FSE states**
(`zstd_decompress_block.c:1738`, `bitstream.h:90-96`)
zstd decodes all three FSE states from *one* `BIT_DStream_t`. tdc has
separate `br_lm` and `br_off`. Merging them enables the next finding.

**F2. Bit-budget refill scheduling**
(`zstd_decompress_block.c:1328, 1390, 1419, 1430`)
zstd uses `STREAM_ACCUMULATOR_MIN_64 = 57` as a budget. On 64-bit, most
sequences need **zero mid-sequence refills** because ofBits+mlBits+llBits
< 57. When they overflow, zstd does exactly one split-refill. Our
per-sequence pre-refill is correct direction but we still refill twice
(once for br_lm, once for br_off) per sequence unconditionally.

**F3. 8-sequence prefetch ring (`STORED_SEQS=8`)**
(`zstd_decompress_block.c:1852-1876`)
`ZSTD_decompressSequencesLong_body` decodes 8 sequences into a circular
buffer *before* executing any. For each decoded sequence it calls
`ZSTD_prefetchMatch` which prefetches **two** cache lines: match start
and match+CACHELINE. Execute lags decode by 8 sequences, hiding ~200-400
cycles of match-source miss latency. We prefetch 1 ahead.

**F4. HUF4: 4 symbols per stream per refill, endSignal bitwise-AND**
(`huf_decompress.c:602-698`)
4 independent DStreams, 4 output quarters, unroll 4 symbols per stream
per iteration = 16 symbols/loop, **one** `BIT_reloadDStreamFast` per 4
symbols per stream. Uses `endSignal &= BIT_reloadDStreamFast(...)`
(line 671) — bitwise AND, not `&&` — to avoid branch misprediction.
Our HUF4 unroll is 2 symbols per stream per refill.

**F5. Three-tier match copy with `ZSTD_overlapCopy8`**
(`zstd_decompress_block.c:1046-1102`, `811-825`)
Three branches on offset: `>=16` unlimited wildcopy; `>=8` one 8B + wild;
`<8` → dec32/dec64 table spread to 8 valid bytes, then 8B wildcopy.
Literals: hardcoded 16B copy first, wildcopy only if `litLength > 16`
(line 1051). Our copy path is already AVX2-vectorized for `off>=32` but
the small-offset tier is still byte-wise.

**F6. `BIT_reloadDStreamFast` invariant** (`bitstream.h:400-405`)
Only checks `ptr < limitPtr`, no consumed-bits overflow check. The
caller guarantees `bitsConsumed <= containerBits`. Our `nbits>=56`
early-out is half of this; full adoption means dropping the consumed-
bits check on the hot path and doing it once per block.

**F7. Interleaved two-state FSE** (`fse_decompress.c:184-236`)
Two FSE states alternate in the same stream to hide each state's DTable
load latency. Applicable to LL+ML if we merge them behind one DStream.

**F8. BMI2 BZHI variant** (`zstd_decompress_block.c:2004-2034`,
`bitstream.h:166`)
`_body` function instantiated twice, once under `BMI2_TARGET_ATTRIBUTE`.
BZHI replaces a mask+shift with one instruction on Haswell+.

### Findings that confirm existing Phase-0 conclusions

- tdc already does AVX2 wildcopy32, 8-ahead prefetch in `lzs_reconstruct`,
  and fused decode in `lzs_decode_fused` — Phase-0 analysis was correct
  that the gap is no longer single-stage. But F3 shows zstd's "prefetch"
  is architecturally deeper than ours (decode/execute decoupled by a
  ring, not just an L1 hint fired 8 iters ahead of the same pointer).

### Revised execution order (supersedes the ROI table below)

| # | Item | Source-audit anchor | Est gain | Effort | Risk |
|---|------|--------------------|----------|--------|------|
| P1 | STORED_SEQS=8 ring: decode 8 seqs into buffer, execute with 2-line match prefetch per seq | F3 | +80-120 MB/s | 1 day | low — pure decode restructure, no format change |
| P2 | Merge `br_lm`+`br_off` into one DStream; schedule refills via 57-bit budget check | F1, F2 | +40-80 MB/s | 1 day | medium — touches core of `lzs_decode_fused` |
| P3 | Full `BIT_reloadDStreamFast` invariant: drop consumed-bits check on hot path, validate once per block | F6 | +15-30 MB/s | half day | medium — correctness-sensitive, needs fuzzing |
| P4 | HUF4 unroll from 2→4 symbols per stream per refill; switch `endSignal` to bitwise-AND | F4 | +50-100 MB/s on HUF-dominated pipelines (BSHUF+HUF) | half day | low — already-tested pattern, watch icache |
| P5 | Three-tier match copy: add `<8` dec32/dec64 path before existing `>=16` AVX2 path | F5 | +20-40 MB/s on small-offset data | 2 hours | low |
| P6 | Literal fast path: hardcoded 16B copy, wildcopy only if `ll > 16` | F5 | +10-20 MB/s | 1 hour | low |
| P7 | Two-state FSE interleave for LL+ML (requires P2 done) | F7 | +20-40 MB/s | 1 day | medium |
| P8 | BMI2 `_bmi2` variant via function-pointer dispatch | F8 | +5-15 MB/s | 2 hours | low |

Start with **P1** (biggest single lever, zero contract change).
Then **P2** (enables P3/P7). **P4** is independent and helps HUF
pipelines we'll need when LZ_STREAMS saturates. P5/P6 are cheap.
P7/P8 are polish once the architecture matches.

Cumulative estimate: 279 → ~600-700 MB/s if P1-P6 land cleanly.

---

### P1 — STORED_SEQS=8 decode/execute ring

**Goal.** Decouple symbol decode from match execution by 8 sequences so
the match-source prefetch has ~200-400 cycles to land before use.

**Anchor.** `zstd_decompress_block.c:1852-1876`,
`ZSTD_decompressSequencesLong_body`, `ZSTD_prefetchMatch` at `:1819-1827`.

**Target file.** `src/entropy/lz_streams.c`, function
`lzs_decode_fused_split_prefetch` (the default L11 path after recent
fusion work).

**Design.**
1. Struct: `typedef struct { uint32_t ll; uint32_t ml; uint32_t mo;
   const uint8_t *lit; /* literal src */ } lzs_seq_t;` sized to 8.
2. Allocation: 8 × sizeof(lzs_seq_t) = 128 B stack-local, **no**
   realloc_fn needed (sub-cacheline, automatic).
3. Loop structure:
   ```
   // Prefill ring
   for (k = 0; k < 8 && idx < n_seqs; k++, idx++) {
       decode_one_sequence(&ring[k & 7]);
       prefetch_match(ring[k & 7].mo, dp_running);
       dp_running += ring[k & 7].ll + ring[k & 7].ml;
   }
   // Steady state
   for (; idx < n_seqs; idx++) {
       uint32_t slot = idx & 7;
       execute_sequence(&ring[slot]);   // uses cache-warm match
       decode_one_sequence(&ring[slot]); // refill same slot
       prefetch_match(ring[slot].mo, dp_prefetch);
       dp_prefetch += ring[slot].ll + ring[slot].ml;
   }
   // Drain
   for (k = 0; k < 8; k++) execute_sequence(&ring[(idx+k) & 7]);
   ```
4. `prefetch_match(mo, dp)` issues *two* `_mm_prefetch(_MM_HINT_T0)`:
   at `dst + dp - mo` and `dst + dp - mo + 64`. Matches zstd's two-line
   prefetch.
5. Keep `dp_prefetch` separate from `dp_exec` — prefetch tracks the
   *future* dp, execute tracks the current.

**Correctness notes.**
- Repcodes (rep[0..2]) must be updated at *execute* time, not decode
  time. Decode stores the raw offset symbol + repcode index; execute
  resolves to absolute offset and rotates the rep table. This is a real
  change from the current code which resolves repcodes at decode.
- Near end-of-stream, ring may not fill: handle `n_seqs < 8` as "decode
  N, execute N in order, no prefetch pipeline".
- Fast/safe split: keep existing `dp + MAX_MATCH + 32 <= dst_size` guard
  but check against `dp_exec`, not `dp_prefetch`.

**Bench plan.**
- Baseline: `bench_real_nasa_regional.txt` (279 MB/s L11).
- Add microbench that forces a cache-cold match buffer (decode into a
  just-allocated 64 MiB region) — this is where prefetch should shine.
- Rollback criterion: < +30 MB/s on regional NASA → investigate before
  merging.

**Tests.** Existing 28 ctest tests cover round-trip. Add one
specifically sized to exercise ring prefill/drain (n_seqs = 3, 7, 8, 9,
16).

---

### P2 — Merge br_lm + br_off into one DStream, 57-bit refill budget

**Anchor.** `zstd_decompress_block.c:1328, 1390, 1419, 1430`,
`bitstream.h:43-45` (`STREAM_ACCUMULATOR_MIN_64 = 57`).

**Target files.** `src/entropy/lz_streams.c` (bit reader + decode
loops), `src/entropy/lzs_bitreader.h` if split.

**Why two readers exist today.** Historical: ll/ml used one stream, off
used another because off has a larger extra-bits footprint. The split
was never necessary — zstd shares one DStream across all three FSE
states.

**Steps.**
1. Audit all call sites of `br_lm` and `br_off` in `lz_streams.c`.
   Confirm they are never refilled concurrently from different threads
   (they aren't — single-threaded decode).
2. Encode side: merge the two bitstreams into one during sequence
   emission. This is a **format change at the block level** but *not*
   a container format change: the block record byte layout changes
   (one stream length instead of two), but no frozen header in
   `include/tdc/` moves. Per CLAUDE.md "byte-identical to vectra is
   not a goal" — this is allowed.
3. Refill site: replace the two `lzs_br_refill(&br_lm); lzs_br_refill(
   &br_off);` with one `lzs_br_refill(&br);` guarded by:
   ```c
   uint32_t need = ll_extra_bits + ml_extra_bits + of_bits;
   if (need > 57) {
       // split refill: read what fits, reload, read rest
   }
   // else single refill already has enough
   ```
4. Pre-compute `of_bits` from the offset code (already done via the
   split-extras decode in the last refactor). Extra bits for ll/ml
   come from the symbol base tables — look them up once.

**Risk.** Medium. This is the single biggest change; every sequence in
every block touches it. Fuzz with the existing fuzz harness (if any)
or add one. Roll out behind a `#define LZS_MERGED_BR 1` guard initially
so we can A/B bench.

**Expected gain.** +40-80 MB/s because most sequences become
refill-free (57-bit budget covers typical ll+ml+of extra bits ~20-40).

---

### P3 — Full BIT_reloadDStreamFast invariant

**Anchor.** `bitstream.h:400-405`, `bitstream.h:384-392`.

**Change.** Drop the consumed-bits overflow check on the per-sequence
refill path. Validate once per block that the compressed bitstream
size and sequence count cannot combine to exceed the container.

**Target.** `lzs_br_read_fast`, `lzs_br_refill` in `lz_streams.c`.

**Invariant to enforce at block entry.**
```c
// Upper bound: each sequence reads at most 24+16+32 = 72 bits
// (ll extra + ml extra + off extra + of code).
if (bitstream_bits < 72 * n_seqs + 8) return TDC_E_CORRUPT;
```
Then the hot loop assumes every refill can read 8 bytes without
overflowing consumed.

**Testing.** Deliberately craft a corrupt block where the compressed
bitstream is shorter than computed; confirm the upfront check catches
it. Fuzz with random truncation.

**Expected.** +15-30 MB/s from dropping ~2-3 instructions per refill.

---

### P4 — HUF4 unroll 2→4, bitwise-AND endSignal

**Anchor.** `huf_decompress.c:602-698`, especially `:653-676` and
`:671` (`endSignal &= BIT_reloadDStreamFast(...) == BIT_DStream_unfinished`).

**Target.** `src/entropy/huffman.c` (the HUF4 decode used by BSHUF+HUF
and HUF-literal paths).

**Steps.**
1. Current loop decodes 2 symbols per stream per refill = 8 symbols
   per iteration. Extend to 4 symbols per stream = 16 symbols/iter.
2. With `HUF_TABLELOG_MAX = 11`, 4 × 11 = 44 bits worst case per
   stream between refills. Safely < 57-bit budget.
3. Replace:
   ```c
   if (BIT_reloadDStreamFast(&bitD1) != BIT_DStream_unfinished) break;
   if (BIT_reloadDStreamFast(&bitD2) != BIT_DStream_unfinished) break;
   ...
   ```
   with:
   ```c
   BIT_DStream_status s = BIT_DStream_unfinished;
   s &= BIT_reloadDStreamFast(&bitD1);
   s &= BIT_reloadDStreamFast(&bitD2);
   s &= BIT_reloadDStreamFast(&bitD3);
   s &= BIT_reloadDStreamFast(&bitD4);
   if (s != BIT_DStream_unfinished) break;
   ```
   Bitwise-AND avoids branch mispredict on the common "all four OK" path.
4. Watch icache: previous 2× unroll regressed. Measure code size of
   the unrolled function; if > 4 KB, back off to 3 symbols/stream.

**Bench.** BSHUF+HUF pipeline (`bench_real_usgs.txt` has HUF-heavy
traces) is the test case. Regional NASA LZ_STREAMS path doesn't
exercise HUF4 directly.

**Expected.** +50-100 MB/s on HUF-dominated pipelines. Neutral on LZ.

---

### P5 — Three-tier match copy with overlap8

**Anchor.** `zstd_decompress_block.c:1082-1102`, dec tables at
`zstd_decompress_block.c:811-812`:
```c
static const U32 dec32table[] = {0, 1, 2, 1, 4, 4, 4, 4};
static const int dec64table[] = {8, 8, 8, 7, 8, 9, 10, 11};
```

**Target.** `lzs_match_copy` inline in `src/entropy/lz_streams.c`.

**Current tier.** Two tiers: `off >= 32` AVX2 wildcopy32, `off >= 16`
SSE2 wildcopy16, `off < 16` byte-by-byte + doubling.

**New tier.**
```c
if (off >= 32) {
    tdc_wildcopy32(op, match, mlen);
} else if (off >= 16) {
    tdc_wildcopy16(op, match, mlen);
} else if (off >= 8) {
    // single 8B copy + 8B wildcopy tail
    memcpy(op, match, 8);
    if (mlen > 8) tdc_wildcopy8(op+8, match+8, mlen-8);
} else {
    // off 1..7: bootstrap 8 valid bytes via dec32/dec64
    op[0]=match[0]; op[1]=match[1]; op[2]=match[2]; op[3]=match[3];
    match += dec32table[off];
    memcpy(op+4, match, 4);
    match -= dec64table[off];
    // now op - match == 8; 8B wildcopy for remainder
    if (mlen > 8) tdc_wildcopy8(op+8, match+8, mlen-8);
}
```

**Note on measured offset distribution.** Regional NASA shows
offsets <16 are ~2% of sequences — small absolute gain on that dataset,
but ubiquitous on other data (text, structured f32 with near repeats).
Bench across `bench_real_*.txt` suite.

**Expected.** +20-40 MB/s averaged; up to +100 MB/s on small-offset-
heavy data.

---

### P6 — Literal fast path: hardcoded 16B + conditional wildcopy

**Anchor.** `zstd_decompress_block.c:1046-1056`:
```c
ZSTD_copy16(op, *litPtr);
if (sequence.litLength > 16) { ZSTD_wildcopy(...); }
```

**Target.** Literal-emit path in `lzs_decode_fused_split_prefetch`.

**Change.**
```c
// Current:
memcpy(dst + dp, lit_raw + lp, ll);
// New:
tdc_copy16(dst + dp, lit_raw + lp);
if (ll > 16) tdc_wildcopy16(dst + dp + 16, lit_raw + lp + 16, ll - 16);
dp += ll; lp += ll;
```

**Correctness.** `tdc_copy16` writes 16 bytes unconditionally even if
`ll < 16` — overruns the current sequence's literal but is overwritten
by the subsequent match copy. zstd relies on this. Requires that
`dst + dp + 16 <= dst_end + OVERRUN_PAD`. Add 16 B pad to dst
allocation in the fast-region guard.

**Expected.** +10-20 MB/s. Very low effort.

---

### P7 — Two-state FSE interleave for LL+ML

**Anchor.** `fse_decompress.c:184-236`.

**Prereq.** P2 done (shared DStream).

**Target.** FSE decode path inside `lzs_decode_fused*` (if LL/ML are
FSE-coded) or the equivalent Huffman sub-decoder if they're HUF-coded.
Check current coding: `src/entropy/lz_streams.c` uses Huffman for
ll/ml symbols per recent commits.

**Decision gate.** If LL/ML are Huffman-coded, P7 does not apply as
written — it's an FSE-specific technique. In that case, substitute with
HUF4 interleave if the implementation doesn't already do it. Reassess
after P2 lands.

**If FSE.** Two state variables per stream, alternate updates so
state2's DTable load overlaps state1's bit-read. Pattern:
```c
sym1 = FSE_decodeSymbolFast(&state1, &DStream);
BIT_reloadDStreamFast(&DStream);
sym2 = FSE_decodeSymbolFast(&state2, &DStream);
BIT_reloadDStreamFast(&DStream);
```

**Expected.** +20-40 MB/s if applicable.

---

### P8 — BMI2 `_bmi2` variant via function-pointer dispatch

**Anchor.** `zstd_decompress_block.c:2004-2034` (BMI2 wrapper pattern),
`bitstream.h:166` (BZHI in BIT_readBits).

**Target.** `src/entropy/lz_streams.c`, `src/core/cpuid.c` (new).

**Steps.**
1. Split `lzs_decode_fused_split_prefetch` into `_body` (inline,
   templated on `bmi2` bool) and two wrappers: `_default` and `_bmi2`
   with `__attribute__((target("bmi2")))` (gcc/clang) or MSVC's BZHI
   intrinsic guarded on runtime detection.
2. Runtime dispatch: `cpuidex(7, 0)` → bit 8 of EBX is BMI2.
3. Function-pointer table populated once at library init.
4. MSVC has `_bzhi_u64`; gcc/clang have `__builtin_ia32_bzhi_di`.
5. `lzs_br_read_fast` uses `_bzhi_u64(acc, nbits)` instead of
   `acc & ((1ULL << nbits) - 1)`.

**Expected.** +5-15 MB/s on Haswell+ hosts. Zero impact on
non-BMI2 hosts (they get the default variant).

**Risk.** Low. Pure additive.

---

### Scratch rules (hard)
- No new bare malloc/free — reroute ring buffer allocation through
  `realloc_fn`. 8 sequences × {ll,ml,of,litSize,matchAddr} ≈ 256 B,
  but still goes through the allocator contract.
- No format change. All P1-P8 items are decode-side only.
- Bench after each item against `bench_real_nasa_regional.txt` baseline;
  regressions roll back that item.

---

## What zstd actually does (read from source, not guessed)

Sources read: `zstd_decompress_block.c`, `bitstream.h`,
`huf_decompress.c`, `huf_decompress_amd64.S`, `zstd_internal.h`.

### Copy primitives

```c
// zstd uses SSE2 intrinsic for 16-byte copy, not memcpy:
static void ZSTD_copy16(void* dst, const void* src) {
    _mm_storeu_si128((__m128i*)dst, _mm_loadu_si128((const __m128i*)src));
}
```

tdc's `lzs_wildcopy16` uses `memcpy(dst, src, 16)`. MSVC *may*
auto-vectorize this but it's not guaranteed — explicit SSE2 is a
guaranteed 1-instruction copy.

### Small-offset match (the `ZSTD_overlapCopy8` trick)

zstd bootstraps 8 valid bytes from any offset 1-7 using two static
lookup tables (`dec32table`, `dec64table`) and exactly 3 memory ops:

```c
// 4 scattered byte stores + 1 table-adjusted copy:
op[0]=ip[0]; op[1]=ip[1]; op[2]=ip[2]; op[3]=ip[3];
ip += dec32table[offset];      // {0,1,2,1,4,4,4,4}
ZSTD_copy4(op+4, ip);
ip -= dec64table[offset];      // {8,8,8,7,8,9,10,11}
```

After this, `op-ip == 8`, so the caller can use 8-byte wildcopy.
tdc's overlap path does byte-by-byte + doubling which is much slower
for offset 1-3 (the most common small offsets in structured data).

### Bitstream design

zstd reads bits **MSB-first from the end of the buffer backward**.
The sentinel bit (`| 1`) in the refill lets the asm path use `bsfq`
(bit scan forward = count trailing zeros) to find the sentinel and
compute consumed bytes in one instruction. Reload is:

```asm
bsfq %bits        // trailing zeros → consumed bit count
shrq $3, %bits    // bytes consumed
subq %bits, %ip   // back up input pointer
movq (%ip), %bits // load next 8 bytes
orq  $1, %bits    // set sentinel
shlxq %rax, %bits // align
```

tdc uses LSB-first forward. This is fine — the sentinel trick saves
~1 instruction per refill but doesn't change the algorithmic structure.
Not worth changing the bit order; focus on reducing refill frequency.

### Fused sequence decode

zstd's `ZSTD_decodeSequence` reads litlen, matchlen, offset from
**one** bitstream using FSE state machines. Each symbol decode is:

```c
baseValue + BIT_readBitsFast(&DStream, nbAdditionalBits)
```

Then immediately `ZSTD_execSequence` copies literals + match.
One pass, one bitstream, fused symbol+execute.

tdc's `lzs_decode_fused` reads 3 separate symbol streams (ll, ml, off)
+ 1 extra-bits stream. This is already fused (symbol + execute in one
loop) but the 4 separate input pointers are more cache-unfriendly than
zstd's single bitstream.

### Prefetch

zstd does `PREFETCH_L1(match)` before the literal copy, so the match
address is in L1 by the time the match copy starts. tdc has zero
prefetch hints anywhere in the decode path.

### Huffman decode

zstd's fast Huffman path decodes 5 symbols per stream per iteration
(20 symbols total across 4 streams). Refill uses the trailing-zero
sentinel trick. The asm version (`huf_decompress_amd64.S`) is 64-byte
aligned and uses dedicated registers for each stream's bit buffer
(rbp, rdx, r12, r13), input pointer (r8-r11), and output pointer
(rsi, rbx, rcx, rdi).

tdc's HUF4 decodes 2 symbols per stream per iteration (8 total) with
C-level bit buffer management. The 2.5x symbol throughput difference
explains why zstd's Huffman is faster.

---

## Phase 0: measurement infrastructure

Before writing SIMD, we need to know where cycles go.

### 0.1 Cycle-level profiling

Add `__rdtsc()` or `QueryPerformanceCounter` instrumentation to
`lzs_decode_fused` to measure time in each section:

```
[symbol reconstruct] → [literal copy] → [match copy]
```

On the regional NASA dataset, expected breakdown:
- match copy: ~50-60% (wildcopy dominates)
- symbol reconstruct: ~20-30% (3 table lookups + bit reads + repcode)
- literal copy: ~10-15%
- bit refill + bookkeeping: ~5-10%

### 0.2 Decode-only benchmark mode

Add a `--decode-only` flag to `bench_throughput.c` that pre-encodes
once, then benchmarks only the decode path. Removes encode noise from
timing.

### 0.3 Assembly inspection

Dump MSVC codegen for `lzs_decode_fused` and `lzs_match_copy`:

```bash
cl /O2 /FA /c src/entropy/lz_streams.c
```

Check whether MSVC auto-vectorizes `memcpy(dst, src, 16)` to
`movdqu` or generates a `rep movsb`. If the latter, SSE2 intrinsics
are mandatory.

---

## Phase 1: copy primitives (biggest ROI)

Match copy is ~50-60% of decode time. These are pure wins with no
algorithmic change.

### 1.1 SIMD header (`src/core/simd.h`)

```c
#ifndef TDC_SIMD_H
#define TDC_SIMD_H

/* SSE2 detection */
#if defined(__SSE2__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#  include <emmintrin.h>
#  define TDC_HAVE_SSE2 1
#endif

/* AVX2 detection (opt-in via cmake flag) */
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
#  include <immintrin.h>
#  define TDC_HAVE_AVX2 1
#endif

/* NEON detection */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#  define TDC_HAVE_NEON 1
#endif

/* Prefetch */
#if defined(__GNUC__) || defined(__clang__)
#  define TDC_PREFETCH_L1(p) __builtin_prefetch((p), 0, 3)
#elif defined(_MSC_VER)
#  include <intrin.h>
#  define TDC_PREFETCH_L1(p) _mm_prefetch((const char*)(p), _MM_HINT_T0)
#else
#  define TDC_PREFETCH_L1(p) ((void)(p))
#endif

/* 16-byte copy: SSE2 guaranteed on x86_64 */
static inline void tdc_copy16(void *dst, const void *src) {
#if TDC_HAVE_SSE2
    _mm_storeu_si128((__m128i*)dst,
                     _mm_loadu_si128((const __m128i*)src));
#elif TDC_HAVE_NEON
    vst1q_u8((uint8_t*)dst, vld1q_u8((const uint8_t*)src));
#else
    memcpy(dst, src, 16);
#endif
}

/* 32-byte copy: AVX2 only */
#if TDC_HAVE_AVX2
static inline void tdc_copy32(void *dst, const void *src) {
    _mm256_storeu_si256((__m256i*)dst,
                        _mm256_loadu_si256((const __m256i*)src));
}
#endif

#endif /* TDC_SIMD_H */
```

### 1.2 SSE2 wildcopy16 (guaranteed on x86_64)

Replace `memcpy(dst, src, 16)` in `lzs_wildcopy16` and
`lzs_match_copy` with `tdc_copy16`. This guarantees a single
`movdqu` store + load instead of whatever MSVC emits for memcpy.

Expected: free 5-15% on match-heavy blocks.

### 1.3 AVX2 wildcopy32

For `off >= 32`, use 32-byte copies:

```c
static inline void lzs_wildcopy32(uint8_t *dst, const uint8_t *src,
                                   uint32_t len) {
    const uint8_t *end = dst + len;
    do {
        tdc_copy32(dst, src);
        dst += 32; src += 32;
    } while (dst < end);
}
```

Match copy dispatch becomes:
```c
if (off >= 32 && TDC_HAVE_AVX2) lzs_wildcopy32(op, match, mlen);
else if (off >= 16)              lzs_wildcopy16(op, match, mlen);
```

Expected: additional 10-20% for long matches (common in LZ_OPT).

### 1.4 Small-offset bootstrap (port zstd's trick)

Replace the byte-by-byte + doubling overlap path with zstd's table
approach:

```c
static inline void lzs_overlap_copy8(uint8_t *op, const uint8_t *match,
                                      uint32_t offset) {
    static const uint32_t dec32[8] = {0,1,2,1,4,4,4,4};
    static const int      dec64[8] = {8,8,8,7,8,9,10,11};
    op[0] = match[0];
    op[1] = match[1];
    op[2] = match[2];
    op[3] = match[3];
    match += dec32[offset];
    memcpy(op + 4, match, 4);
    match -= dec64[offset];
    /* Now op - match == 8, use 8-byte copies for the rest */
}
```

Then `lzs_match_copy` becomes:

```c
if (off >= 16) {
    lzs_wildcopy16(op, match, mlen);
} else if (off >= 8) {
    /* 8-byte copy loop (existing) */
} else {
    lzs_overlap_copy8(op, match, off);
    op += 8; match += 8;
    if (mlen > 8) {
        /* 8-byte overlapping wildcopy for remainder */
        uint8_t *end = op + mlen - 8;
        do { memcpy(op, match, 8); op += 8; match += 8; }
        while (op < end);
    }
}
```

Expected: 15-30% speedup for small-offset matches (offset 1-3 are
very common in byte-shuffled data).

### 1.5 Prefetch match address

Add `TDC_PREFETCH_L1(dst + dp - mo)` before the literal copy in
`lzs_decode_fused`, so the match address is in L1 by the time we
reach the match copy. This is exactly what zstd does:

```c
// Before literal copy:
TDC_PREFETCH_L1(dst + dp - mo);  // match will be here
if (ll > 0) {
    memcpy(dst + dp, lit_raw + lp, ll);
    dp += ll; lp += ll;
}
// Match copy starts — match address already in L1
```

Expected: 5-10% on large blocks where match addresses are
cache-cold.

---

## Phase 2: batch symbol reconstruction

Symbol reconstruct is ~20-30% of decode time. The current code does
3 table lookups + 3 conditional bit reads + repcode per sequence.

### 2.1 Branchless extra-bits threshold

Replace:
```c
if (s >= 2u) { nb = s - 1u; ex = lzs_br_read(&br, nb); }
```
with:
```c
uint32_t nb = (s >= 2u) ? (uint32_t)(s - 1u) : 0u;
ex = lzs_br_read(&br, nb);  // lzs_br_read already handles nb==0
```

Already partially done (lzs_br_read returns 0 for nb==0), but the
branch on `s >= 2u` should become a conditional move. Check codegen.

### 2.2 Inline lzs_symbol_to_uint

`lzs_symbol_to_uint(s, ex)` is `(1u << s) - 2u + ex` for s >= 2,
or just s for s < 2. Make sure this is fully inlined and the branch
is a CMOV. If MSVC doesn't CMOV it, use:

```c
// Branchless: works for all s
uint32_t base = (1u << s) - 2u;     // garbage for s < 2 but...
uint32_t val = (s < 2u) ? s : base + ex;
```

### 2.3 Batched repcode (speculative)

If profiling shows repcode logic is hot (unlikely — it's 3-4 branches
per sequence), consider converting the `if-else` chain to a computed
swap:

```c
// Current: 4 branches per sequence
// Possible: table of {which offset to use, rotation pattern}
```

Low priority — benchmark before implementing.

---

## Phase 3: Huffman decode speed

Huffman decode matters for LZ+HUF and BSHUF+HUF pipelines. Current
tdc decodes 2 symbols per stream per iteration (HUF4) vs zstd's 5.

### 3.1 Increase symbols per iteration

Decode 4 or 5 symbols per stream between refills instead of 2.
With 11-bit table, each symbol consumes ≤11 bits. 5 symbols = 55
bits max. After a 56-64 bit refill, this is safe.

The current HUF4 loop does:
```c
HUF_DECODE_SYMBOLX1_2(op1, &bitD1); // round 1 all 4 streams
...
HUF_DECODE_SYMBOLX1_0(op1, &bitD1); // round 2 all 4 streams
// refill all 4
```

Change to 5 rounds × 4 streams = 20 symbols between refills.
This matches zstd's throughput.

**Caveat:** previous 2x unroll regressed due to icache pressure.
5 rounds may be fine if we keep the per-symbol macro small (table
lookup + shift + store = 3 instructions). Monitor code size.

### 3.2 SSE2 bitshuffle (`src/transform/bitshuffle.c`)

Current bitshuffle is fully scalar (bit-by-bit extract + pack).
The standard SIMD approach:

```
For each bit plane b (0..7):
  Load 8 bytes of input
  Use _mm_movemask_epi8 to extract bit b from each byte → 1 byte output
  Rotate/shift for next bit
```

This replaces 64 bit-extract operations with 8 `movemask` calls.
Expected: 4-8x speedup on bitshuffle, which helps all BSHUF+*
pipelines.

### 3.3 Loop alignment (low effort, measurable)

Add MSVC `__declspec(align(64))` or gcc `__attribute__((aligned(64)))`
to the Huffman and LZ_STREAMS decode functions. Or use:

```c
#if defined(__GNUC__) || defined(__clang__)
#  define TDC_ALIGN_LOOP __asm__(".p2align 6")
#elif defined(_MSC_VER)
#  define TDC_ALIGN_LOOP __declspec(align(64))
#endif
```

zstd uses `.p2align 6` extensively in the asm Huffman loop. Even in
C, hinting the compiler can help.

---

## Phase 4: fused decode micro-optimizations

These are small wins that compound.

### 4.1 Fast/safe region split

Instead of checking `dp + ml <= safe_end` per sequence, precompute
how many sequences can run in fast mode:

```c
// Fast loop: no bounds checks on wildcopy
uint32_t i = 0;
for (; i < n_seqs && dp + MAX_MATCH + 15 <= dst_size; i++) {
    // ... decode + execute, no safe_end check on match copy
}
// Safe loop: byte-by-byte for remaining sequences
for (; i < n_seqs; i++) {
    // ... bounds-checked path
}
```

This removes a branch from the fast path.

### 4.2 Remove per-sequence corruption checks from fast path

The 6 corruption checks per sequence (`s > LZS_MAX_LL_SYMBOL`,
`lp + ll > total_lit`, `dp + ll > dst_size`, etc.) can be relaxed
in the fast path if we validate the symbol streams upfront:

```c
// Before the loop: validate all symbols are in range
for (uint32_t i = 0; i < n_seqs; i++) {
    if (ll_dec[i] > LZS_MAX_LL_SYMBOL) return TDC_E_CORRUPT;
    if (ml_dec[i] > LZS_MAX_ML_SYMBOL) return TDC_E_CORRUPT;
    if (off_dec[i] > LZS_MAX_OFFSET_SYMBOL) return TDC_E_CORRUPT;
}
// Fast loop: no symbol range checks needed
```

This moves 3 branches out of the hot loop.

### 4.3 Literal copy with tdc_copy16

Short literals (≤16 bytes) are common. Use `tdc_copy16` for the
first 16 bytes, then wildcopy for the remainder:

```c
if (ll > 0) {
    tdc_copy16(dst + dp, lit_raw + lp);
    if (ll > 16)
        lzs_wildcopy16(dst + dp + 16, lit_raw + lp + 16, ll - 16);
    dp += ll; lp += ll;
}
```

This matches zstd's `ZSTD_copy16` + conditional `ZSTD_wildcopy`.

---

## Phase 5: architectural changes (longer term)

### 5.1 Single-bitstream sequence format

tdc's LZ_STREAMS uses 4 separate streams (ll, ml, off, extras).
zstd uses 1 interleaved bitstream with FSE state machines. The
single-stream approach:
- Better cache locality (1 hot pointer vs 4)
- Fewer refills (1 bitstream vs 4)
- FSE state update is 1 table lookup vs Huffman+bit-read

This is a format change (new entropy backend, not a SIMD bolt-on).
Consider for v2 if the copy-primitive + micro-opt phases don't close
the gap enough.

### 5.2 tANS for symbol streams

Replace Huffman on the 3 symbol streams with table-ANS. Each symbol
decode becomes a single table lookup (state → symbol + next_state +
nbits) instead of bit-peek + table lookup + bit-skip.

This is §4 from the original roadmap. Can be combined with 5.1
(single-stream tANS) or applied independently to the current
multi-stream layout.

---

## Execution order (by ROI)

| # | Item | Expected gain | Effort | Dependency |
|---|------|--------------|--------|------------|
| 1 | 1.2 SSE2 wildcopy16 | 5-15% | 1 hour | 1.1 simd.h |
| 2 | 1.4 Small-offset bootstrap | 15-30% | 2 hours | none |
| 3 | 1.5 Prefetch match | 5-10% | 15 min | none |
| 4 | 4.2 Upfront symbol validation | 3-5% | 1 hour | none |
| 5 | 4.3 Literal tdc_copy16 | 3-5% | 30 min | 1.1 |
| 6 | 1.3 AVX2 wildcopy32 | 10-20% | 2 hours | 1.1 |
| 7 | 3.2 SSE2 bitshuffle | 4-8x bshuf | 4 hours | 1.1 |
| 8 | 3.1 HUF4 5-sym/stream | 2x Huffman | 4 hours | none |
| 9 | 4.1 Fast/safe split | 2-3% | 2 hours | none |
| 10 | 2.1-2.2 Branchless symbols | 2-5% | 1 hour | none |
| 11 | 0.1-0.3 Profiling infra | measure | 2 hours | none |
| 12 | 5.2 tANS decoder | 2-3x entropy | 1-2 days | none |
| 13 | 5.1 Single-stream format | structural | 3-5 days | 5.2 |

Items 1-5 are do-first: high ROI, low risk, no format changes.
Items 6-10 are do-next: moderate effort, compound on items 1-5.
Items 11-13 are measure-then-decide: architectural changes.

Compound estimate: items 1-5 should take decode from 338 → ~450-500.
Items 6-10 should reach ~550-650. Items 12-13 could close the full
gap to zstd's ~724 if the copy primitives aren't enough.
