# tdc decode speed — SIMD & fused decode plan

**Goal:** Close the 2x decode gap vs zstd on regional NASA f64 data.

Current: LZ_STREAMS 338 MB/s decode, zstd L19 724 MB/s.
Target:  ≥600 MB/s decode on LZ_STREAMS (regional NASA, 8.3 MiB f64).

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
