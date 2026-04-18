# LZ inner loop

This page documents the native LZ77 stage, `TDC_ENTROPY_LZ`, at the level the
encoder and decoder are actually implemented. The enum id, the vtable, and
the `encode_bound / encode / decode` triple are described in the backends
entropy page and in `include/tdc/entropy.h`. What follows is the machinery
inside those three function pointers: the matcher, the wildcopy, the fast
and safe decode loops, the on-disk token layout, and the benchmarked dead
ends that shaped all of the above. The inner loops came verbatim from
vectra's `vtr_codec.c` and have been measured to the last percent against
their alternatives. Nothing in this vignette is theoretical.

`TDC_ENTROPY_LZ` sees a flat byte stream by the time it runs. A model has
already turned a typed block into residuals; a transform chain has turned
those residuals into the exact bytes we now compress. The matcher does not
know the dtype, the layout, or the predictor that produced its input, and
it does not need to. Every number quoted below was measured on that byte
stream, typically 1 MiB of DEM residuals, 1 MiB of `i32`-ramp residuals,
or 1 MiB of LCG noise, depending on the point being made.

## Problem statement

LZ77 is the oldest trick in the dictionary-compression book. A parser walks
the input left to right; at each position it asks whether the next few
bytes have appeared earlier within a bounded window, and if yes, emits a
back-reference (an (offset, length) pair) instead of the literal bytes.
The decoder replays the sequence of literal runs and matches to rebuild
the input. Everything expensive lives in the parser; the decoder is a
straight-line walk.

tdc's LZ stage exploits two facts the rest of the pipeline leaves behind.
First, residuals from a good model cluster around zero and share low
bytes, which means even short matches occur often; the matcher pays off
on inputs where the entropy-only ratio would be roughly 1.0. Second, after
a byte-shuffle transform the top byte of every element is packed into one
contiguous run; a 1 MiB `f32` raster becomes a 262 KiB run of high-order
bytes that the matcher collapses to a few long offsets, with the remaining
lanes contributing shorter matches as the exponent and mantissa planes
share less structure.

We reach for LZ when the caller wants a fast, self-describing entropy pass
and is willing to tolerate a ratio a few percent short of what FSE or a
split-stream coder would land. Everything below describes what that
greedy, separated-stream LZ does. The backends page compares it to the
alternatives; this page explains the machinery.

What ratio can LZ achieve on tdc's typical residuals? The histogram
example at the end of the page measures the extremes. A 1 MiB all-zero
input compresses to 99 bytes (ratio 10592×) through one literal byte
followed by a single match of 1048575 bytes at offset 1; a 1 MiB
`abcabcabc...` input compresses to 101 bytes (ratio 10382×) through one
literal run of 3 bytes followed by a single match of 1048573 bytes at
offset 3; a 1 MiB LCG-noise input grows from 1048576 to 1048668 bytes
(ratio 0.999×) because no match of `LZ_MIN_MATCH` bytes is ever found
and the parser falls back to a literal-only stream. These three points
bracket what the hot path is up against.

## Math

The greedy parse is a sequence of decisions. At position `sp` in the
input, the matcher either emits `sp` as a literal and advances one byte,
or emits a triple `(lit_len, match_len, match_off)` describing a back-
reference and advances by `match_len`. `match_off` is the distance back
to the earlier occurrence, `match_len` is how many bytes extend a match.
The constraints are simple: `match_len >= LZ_MIN_MATCH = 3`, `match_off >=
1`, `match_off <= LZ_MAX_OFFSET = 2^22` (4 MiB window). With two extra
tricks layered on top (lazy matching, repcode probing) we land in the
ratio range produced by the zstd greedy parser at level 1–3 on the same
inputs. Pseudocode for the matcher core:

```
for sp = 0, len(src) - 1:
    h = hash(src[sp..sp+4])       // Fibonacci multiplicative hash
    cand = htab[h]
    htab[h] = sp
    if cand valid and src[cand..cand+3] == src[sp..sp+3]:
        length = extend(src, sp, cand)
        if length >= LZ_MIN_MATCH:
            emit_match(lit_run, length, sp - cand)
            sp += length
            continue
    lit_run.append(src[sp])
    sp += step                    // step >= 1; accelerates on misses
```

The hash function is Knuth's 4-byte multiplicative Fibonacci hash:

$$
h(b_0 b_1 b_2 b_3) = \left\lfloor \frac{(b_0 + 256 b_1 + 65536 b_2 + 16777216 b_3) \cdot \varphi'}{2^{32 - B}} \right\rfloor
$$

where `B = LZ_HASH_BITS = 18` and `φ' = 2654435761` is the 32-bit
rounded value of the reciprocal of the golden ratio scaled to 2^32. The
Fibonacci constant distributes any 32-bit input uniformly into `2^B`
buckets with no observed bias, which is the only property the matcher
needs. A hash collision does not break correctness because the one-byte
and four-byte equality checks gate the candidate before the extension
loop.

Extension past the 4-byte prefix is an 8-byte-at-a-time memcmp with
trailing-zeros count on mismatch:

```
while len + 8 <= max_len:
    load a = src[pos + len..], b = src[cand + len..]
    if a != b:
        len += ctz(a ^ b) >> 3       // bytes from the first differing byte
        break
    len += 8
```

`__builtin_ctzll` on gcc/clang and `_BitScanForward64` on MSVC return the
position of the first set bit in `a XOR b`, divided by 8 to turn bit
position into byte position. The scalar tail handles the last 0–7 bytes
of input. Read `src/entropy/mf_hashchain.c::mf_extend` for the production
version; the fast path in `src/entropy/lz.c::lz_parse_fast` carries an
inlined copy so the single-probe hot path avoids the vtable.

The on-disk token layout packs a variable-length sequence descriptor per
match, then writes all literal bytes contiguously after the descriptor
table:

```
[ u32 n_seq        ]
[ u32 literals_sz  ]
[ seq_hdr_0        ]      packed per sequence: tag byte + extensions + offset
[ seq_hdr_1        ]
...
[ seq_hdr_{N-1}    ]
[ literal_bytes    ]      literals_sz bytes, concatenated
```

The per-sequence tag byte is `[LLLLMMMM]`: high nibble is the literal-run
length (0–14, or 15 meaning extended), low nibble is `match_len - 3` (0–14,
or 15 meaning extended). Extensions use chained-255 for literal length
(each extra byte adds 255 until a byte `< 255` terminates) and LEB128 for
match length (7 data bits + 1 continuation bit per byte). Match offsets
are encoded as a 2-byte `uint16` base with a LEB128 extension when the
raw offset exceeds 65535. The common-case cost is 3 bytes per sequence:
1 tag + 2 offset.

Why a split encoding for the two lengths. A 4 MiB run of identical bytes
(PLANE_2D on a flat interior, RAW on zero residuals) produces one match
spanning the whole run. Chained-255 would cost `4 * 2^20 / 255 ≈ 16 KiB`
just to name that one match length; LEB128 names the same length in 4
bytes. Literal runs never reach that regime because trailing literals
after the last match are written to the literal buffer directly (no
sequence header), so per-sequence `lit_len` is bounded by real literal
density between matches. Chained-255 on literal lengths saves a branch in
the hot decoder and doesn't cost us on any measured input.

## Hash table and matcher

The matcher uses a direct-addressed hash table: no linear probing, no
open addressing, no rehashing. Layout:

```
htab[1 << 18]          // one u32 per bucket; holds the most recent
                       // position that hashed to this bucket, or 0xFFFFFFFF
                       // when empty

chain_prev[src_size]   // allocated only when chain_depth > 0;
                       // chain_prev[pos] = previous position sharing the
                       // same bucket; 0xFFFFFFFF terminates the chain
```

The 2^18 bucket count is fixed at 1 MiB of table memory (262144 buckets ×
4 bytes) regardless of input size, which fits comfortably in L2 on every
current server CPU and avoids the L3-miss penalty deeper tables pay on
real workloads. Collisions are frequent by design: a 1 MiB input fills
every bucket in expectation. The matcher handles collisions either by
overwriting the bucket (`chain_depth == 0`, flat hash) or by threading the
displaced position into a singly-linked chain via `chain_prev[]`
(`chain_depth > 0`).

The flat-hash branch lives in `lz_parse_fast` and is the hottest code
path in the library. It probes exactly one candidate per position, uses
an accelerating step that grows the skip distance after consecutive misses
(capped at 8, reset to 1 on every hit), and inserts every fourth position
within a match into the table so future references can still find them.
On noisy inputs where matches are sparse, the accelerating step skips
roughly `sp += 1, 2, 3, ..., 8, 8, 8` after each miss, cutting the
per-byte hash cost by 3–5× against a step-1 probe.

The chain branch (`tdc_lz_parse_greedy` with `chain_depth > 0`) walks up
to `chain_depth + 1` candidates per position, quick-rejecting each by a
one-byte comparison at the current best length before the 8-byte
extension. The chain insert cost is one conditional load and two stores;
the walk itself prefetches the next candidate's position and chain link,
so the memory system keeps up on runs where every lookup descends
several links.

Hash table sizing deliberately runs over the conventional "70% load
factor" target that a dictionary hash set would aim for. The matcher is
not building a dictionary; it is building a multi-valued index over
positions, and over-loading the table is the whole point. A bucket that
holds `k` positions on a 1 MiB input walks `k` chain links, not `k` probe
slots; the average chain length at `chain_depth = 4` stays small because
the matcher uses the most-recent-first convention (the newest position
for a bucket is at the head) and the rep-code probe catches the periodic
cases where a few offsets dominate.

### No Robin Hood hashing

The same table design was tried in vectra with Robin Hood displacement on
the collision chain. The benchmarked regression was 12–29% on encode
throughput across the standard corpus. The reason is simple enough that
it is worth spelling out: Robin Hood hashing wins on open-addressed tables
where the average probe length would otherwise be large, because moving
existing entries closer to the cache line where a probe starts amortizes
across many future lookups. tdc's LZ table is not open-addressed. Probe
length on the flat-hash path is exactly 1; probe length on the chain path
is bounded by `chain_depth + 1` (typically 5 or 9). The per-probe work
Robin Hood adds (reading the displaced entry's home-bucket index,
comparing it to the current probe distance, swapping entries when the
invariant breaks) dominates the savings because there is no probe
distribution to reshape. The measurement is cited in
`vectra/CLAUDE.md` and carries forward into `CLAUDE.md` at the tdc root;
do not retry.

The matcher also runs a rep-code probe at each position
(`lz_rep_probe` in `src/entropy/lz.c`). The three most recent match
offsets are kept in an MRU array `rep[3]`; each rep slot is tried with a
3-byte prefix compare and then the same 8-byte extension as the hash
path. A rep hit wins ties against a hash match of the same length because
repcode offsets encode with a much shorter tag downstream in
`TDC_ENTROPY_LZ_STREAMS`. On periodic signals such as a daily temperature
series or a seasonally-striped raster, the rep-code probe catches the true
period on the first match and reuses it for every subsequent sequence,
bypassing the hash walk entirely.

## Wildcopy

The decoder copies both literal runs and match bodies with a "wildcopy":
a loop that writes in 16-byte blocks regardless of the requested length,
rounding the end of every copy up to the next 16-byte boundary. The
helper `tdc_wildcopy16` in `src/core/simd.h` uses one SSE2 `_mm_storeu`
per iteration; on AVX2 builds the equivalent `tdc_wildcopy32` writes
32 bytes per iteration. `tdc_match_copy` dispatches on the match offset:
offsets ≥ 32 go straight to the 32-byte wildcopy; 16 ≤ offset < 32 use
the 16-byte version; 8 ≤ offset < 16 use repeated 8-byte copies;
offset < 8 uses a doubling seed-and-fill (copy `off` bytes, then double
the filled region until it reaches 16 bytes of valid pattern, then
standard 16-byte copies from there).

Wildcopy over-reads by up to 31 bytes past the requested end on AVX2
builds (15 on SSE2 / NEON). That is safe because the caller reserves
`TDC_WILDCOPY_SLACK` bytes of slack past the end of the output buffer
before invoking the fast path. The slack constant is defined in
`src/core/simd.h` alongside the copy helpers. The reader can audit the
single cross-check that bounds it: in `lz_decode_fast`, before every copy,

```c
if (dp + lit_len + mlen + TDC_WILDCOPY_SLACK > uncompressed_size) {
    sp = sp_save;
    break;
}
```

When this bail fires, the fast loop rewinds the sequence pointer to the
last safe state and hands control to `lz_decode_safe`, which copies
exact byte counts through `memcpy` and handles the overlap-correct
doubling pattern for small offsets. The cost of the check is a single
predictable branch per sequence; `__builtin_expect` annotations mark the
bail as unlikely, which is the measured behavior on every input that
compresses at all.

The small-offset doubling pattern is worth understanding because it
shows up in both the fast and the safe decode. For `offset = 1` (a run
of identical bytes), naively copying `mlen` bytes from `op - 1` to `op`
would read bytes that haven't been written yet. Instead, the decoder
seeds `off` bytes into the output, then repeatedly copies the filled
region onto itself at twice the offset, doubling the valid pattern until
it covers `mlen`. After $O(\log(mlen/off))$ iterations each doing a
single `memcpy`, the match is done. On PLANE_2D residuals with a flat
interior this path decodes a 4 MiB zero run at roughly `memcpy` speed
instead of byte-at-a-time.

## Fast and safe decode

The decode entry point `lz_decode_core` runs the fast path first,
then the safe path. The split is not about correctness (both paths are
correct); it is about branch count per byte. The fast path assumes the
sequence stream and the output buffer both have enough slack that one
bounds check per sequence is enough to cover the copies the sequence will
do. That check is:

```c
if (dp + lit_len + mlen + TDC_WILDCOPY_SLACK > uncompressed_size) break;
if (lp + lit_len > literals_size)                                 break;
```

One compare, one predictable branch, per sequence. When either inequality
might fire (we are near the end of the output, or near the end of the
literal pool), the fast path bails. The safe path then walks every
remaining sequence with per-copy clamping, exact `memcpy` instead of
wildcopy, and the doubling pattern spelled out in the previous section.

The phase split is measured, not theorized. Merging the two into a
single loop forces every `memcpy` to pay the bounds check that the fast
path runs once per sequence; on the `abc` input above the safe-only
build ran 22% slower. Running only the fast path and trusting the input
to be well-formed would save the trailing-literal handling and the
corruption check on small offsets; on an adversarial corrupt input the
fast path walks off the end of the buffer and segfaults, which is the
wrong behavior for a library that accepts records from disk.

### No batched parse-then-copy

Once the fast path is written as above, the next obvious optimization is
to split the per-sequence loop into two phases: parse a batch of 4–16
sequences into a stack-local descriptor array first, then run a tight
copy loop over the parsed descriptors. The hypothesis is that the parse
phase touches few bytes per sequence and stays in L1, the copy phase
touches many bytes per sequence and spills cache, and separating the two
lets the CPU prefetch the copy destinations while the parse runs.

Measured in vectra, at every batch size from 4 to 16, the batched
variant is 6–7% slower than the interleaved loop. The mechanism is
structural: LZ's parse is just a tag byte plus occasional varint
extensions plus a 2–6 byte offset. It is cheap enough that the CPU's
out-of-order scheduler already overlaps the next sequence's parse with
the current sequence's wildcopy. Forcing a phase split serializes parse
and copy, loses the overlap, and recovers nothing because the parse
phase has nothing expensive enough to benefit from a dedicated decode
pipeline. The decoder as written is already the overlapped form; batching
opts out of it.

The note in the source (`src/entropy/lz.c` above `lz_decode_fast`) adds a
condition on revisiting: once a real entropy stage such as FSE or
Huffman sits in front of LZ, the parse phase becomes expensive enough
that batching may pay. Until then, the batched variant loses every time.

## Correctness argument

Round-trip correctness falls out of two invariants. First, the encoder
never emits a match whose offset `off` exceeds the number of bytes
already written to the output; equivalently, at the point a sequence is
emitted, `off <= sp`. Every extension in the matcher is bounded by the
window size (`LZ_MAX_OFFSET = 4 MiB`) and by the distance from the
current position to the candidate's position. The safe decoder checks
the invariant explicitly (`if (dp < off) return TDC_E_CORRUPT;`) and
returns `TDC_E_CORRUPT` when a malformed record violates it.

Second, the decoder consumes exactly as many bytes from the literal
pool as the encoder wrote there. The literal pool is stored contiguously
after the sequence descriptors; its total length is written in the 8-byte
LZ header (`literals_size`), so the decoder can split sequence-header
bytes from literal-pool bytes without traversing the sequence stream
first. The split computation in `lz_decode_core` is a single subtraction:

```c
uint32_t seq_hdr_size = src_size - LZ_HEADER_SIZE - literals_size;
```

Any record where the subtraction underflows, or where `literals_size`
exceeds the payload, is rejected with `TDC_E_CORRUPT` before any byte is
written to the output.

Empty inputs round-trip through the literal-only fallback path
(`lz_encode_literal_only`). A block shorter than `LZ_MIN_MATCH + 1` bytes
cannot have any match (the matcher needs a 4-byte hash key plus the
next byte), so the parser emits zero sequences and the serializer
writes an 8-byte header with `n_seq = 0` followed by the raw bytes.
The decoder recognizes `n_seq == 0` and takes the safe path's trailing-
literal branch, which `memcpy`s the literal pool directly into the
output. Empty blocks (0-byte inputs) produce a 0-byte payload after the
8-byte header.

## Complexity and hot path

The greedy parser is $O(N)$ time for flat hash (one probe per position)
and $O(N \cdot (d+1))$ for `chain_depth = d`; the chain walk is bounded by
`d + 1` candidates per lookup and never exceeds the window size. Memory
is $O(1)$ for flat hash (1 MiB table) and $O(N)$ for the chain
(`chain_prev` needs one `u32` per input byte). Worst-case encode time
on a 1 MiB input with `chain_depth = 64` and `lazy_depth = 2` is under
40 ms on the author's desktop, driven almost entirely by the hash-chain
probes.

The decoder is $O(N)$ in the output size, not the input size. A 100-byte
record that decodes to 4 MiB spends almost all of its time inside the
wildcopy; the parse loop executes once per match, and each match covers
many output bytes. The hot copy primitive is `tdc_wildcopy32` on AVX2,
which puts down 32 bytes per loop iteration at roughly 20 GiB/s on
current server CPUs. Decode throughput on the `abc` input above measures
in the same 8–12 GiB/s range as a straight `memcpy` of the same length.

The hottest instruction sequence in encode is `lz_parse_fast`'s inner
loop: a hash load, one bucket insert, one candidate equality check, the
8-byte extension compare, and the miss-path step increment. Every step
is a single load or a single arithmetic op; there are no branches the
CPU cannot predict on typical residuals, which is why the flat-hash
path sustains 300+ MB/s encode on a single core.

## Interactions between the knobs

The matcher exposes `LZ_MIN_MATCH`, `LZ_MAX_MATCH`, `LZ_MAX_OFFSET`, and
the hash table size through compile-time constants in
`src/entropy/lz.c` and `src/entropy/lz_internal.h`. These interact:

- `LZ_MIN_MATCH = 3` is the smallest match the matcher will emit. A match
  of length 2 costs 3 bytes to encode (tag + offset), the same as two
  literal bytes plus overhead. Lowering to 2 costs ratio. Raising to 4 or
  higher is exposed through the `TDC_ENTROPY_LZ_STREAMS` `min_match`
  parameter; at 4 the matcher fuses more short matches into literals,
  raising bytes/sequence and decode throughput at the cost of ratio.
  The on-disk format is unchanged.
- `LZ_MAX_MATCH = UINT32_MAX` is effectively unbounded. An earlier
  130-byte cap (inherited from vectra's original encoding) chopped
  PLANE_2D's 4 MiB zero-residual runs into 32k three-byte sequences,
  totalling roughly 100 KiB of payload where a single match spans the
  whole run at 4 bytes. Uncapping the match length gave the largest
  single ratio improvement in tdc's history (roughly 3× on flat
  residuals).
- `LZ_MAX_OFFSET = 2^22` (4 MiB) is the window size. Offsets up to 65535
  fit in 2 bytes; larger offsets pay a 1–3 byte LEB128 extension. The
  4 MiB cap is what the `uint16 + LEB128` encoding can carry without
  format churn, and it covers every block tdc currently encodes in a
  single record.
- The hash table holds `2^18` buckets (262144 entries, 1 MiB). Doubling
  it to `2^19` gives a measurable but small ratio improvement (1–2%) on
  highly structured inputs and a consistent 5–7% speed loss from the
  extra L2 pressure. At `2^18` the table fits in most server CPUs' L2,
  so the hash load itself is a 3–4-cycle hit instead of a 10–15-cycle
  miss.

## Deviations from textbook LZ77

Three deviations from a clean LZ77 implementation are worth naming:

1. **Separated literal stream.** Classical LZ77 writes a `(lit, match)`
   alternation where literal bytes are inline between sequence
   descriptors. tdc follows the zstd convention of writing all sequence
   headers first, then all literal bytes in a contiguous run. This
   enables the split decoder (fast / safe) to reason about literal-pool
   bounds separately from sequence-stream bounds, and it keeps the
   literal bytes aligned for bulk `memcpy` inside the wildcopy.

2. **Accelerating step on misses.** Classical greedy LZ advances by
   exactly 1 byte when no match is found. tdc's flat-hash path grows the
   step linearly after consecutive misses, capped at 8 and reset to 1 on
   any hit. This is the "step" knob that makes LZ usable on semi-
   incompressible data; without it a 1 MiB LCG-noise input takes 6× as
   long to parse with no ratio improvement.

3. **Rep-code probe.** Classical LZ77 does not carry state across
   sequences. tdc's matcher tracks the three most recent match offsets
   and probes them first at every position. On periodic signals the
   rep probe beats the hash search on every sequence after the first.
   The on-disk format does not expose rep-code tags (`TDC_ENTROPY_LZ_STREAMS`
   is the backend that does); the rep hits emit ordinary offset bytes
   that happen to match earlier offsets.

## Dead ends

Before leaving the matcher alone, these alternatives were tried. Each row
records the change, the measurement, and the reason it lost. The notes
live in `CLAUDE.md` at the tdc root and in `vectra/CLAUDE.md` upstream;
the table here is a quick reference when future work reaches for one of
them again.

| Idea | What was tried | Measured delta | Why it lost |
|---|---|---|---|
| Robin Hood hashing on the collision chain | Displacement-aware inserts into `htab + chain_prev`, swap-on-steal | 12–29% slower encode | No probe distribution to reshape: flat-hash probes exactly 1 entry, chain walks bounded by `chain_depth`. Per-probe Robin Hood overhead dominates. |
| Batched parse-then-copy in the decoder | Parse 4–16 sequences into a stack descriptor array, then a pure copy loop | 6–7% slower decode at every batch size | Out-of-order execution already overlaps the next parse with the current wildcopy. Phase split serializes them and recovers nothing because parse is too cheap to benefit from its own pipeline. |
| Doubling the hash table to `2^19` buckets | `LZ_HASH_BITS = 19`; table grows to 2 MiB | ~1.5% better ratio, 5–7% slower encode | Table no longer fits in L2 on most CPUs; every probe pays an L3 miss. Not worth it on average; revisit only when L2 grows. |
| Lowering `LZ_MIN_MATCH` to 2 | Accept 2-byte matches from the hash; extend encoding to support them | 3–6% worse ratio | 2-byte match costs 3 bytes to encode (tag + offset), same as two literals plus the tag overhead. The matcher also sees 2-byte coincidences constantly, so the extra sequences flood the stream. |
| Chained-255 on match length | Use the literal-length extension shape for match length too | Payload for 4 MiB zero-run grows from 4 bytes to ~16 KiB | The match-length extension needs to encode values into the millions. Chained-255 costs `extra / 255 + 1` bytes, quadratic in the payload. LEB128 stays logarithmic. |
| Move wildcopy slack into per-copy clamping | Drop the pre-check, clamp inside every literal and match copy | 18–24% slower decode | The one-compare-per-sequence gate is cheap; two clamps per copy are not. Moreover the clamp variants cannot use 32-byte wildcopy and drop to 16-byte or scalar copies. |
| 128-bit Fibonacci multiplier (wyhash-style) | Replace 32-bit multiplier with a 64-bit mix + shift | ~0.4% better ratio, 8–12% slower encode | The distribution improvement shows up only on highly-contrived inputs; on real residuals 2654435761u already fills buckets uniformly. Slower mul path costs throughput everywhere. |

Each of those was a plausible idea. The table is what remains of the
benchmark runs that ruled them out.

## Worked example: abcabcabcabc...

It is worth walking the matcher on a small input by hand. Take the 12-byte
input `abcabcabcabc`. `LZ_MIN_MATCH = 3`, the hash table starts empty, and
the parser uses the flat-hash fast path.

At `sp = 0`: hash the 4 bytes `abca`, probe the table, empty. Write
`sp = 0` into the bucket. No match. Append `'a'` to the pending
literal run. Advance `sp` by 1.

At `sp = 1`: hash `bcab`, probe, empty. Write `sp = 1`. No match.
Append `'b'`. Advance.

At `sp = 2`: hash `cabc`, probe, empty. Write `sp = 2`. No match.
Append `'c'`. Advance.

At `sp = 3`: hash `abca` (same 4 bytes as `sp = 0`), probe, hits
bucket, `cand = 0`. `src[0] == src[3] == 'a'`. Extend: `src[0..8] ==
src[3..11]`, i.e. 9 bytes match. Continue: `src[9..12]` is 3 more
`abc`-shifted bytes, still matching. Full extension length is `len = 9`.
Emit the sequence `(lit_len = 3, match_len = 9, match_off = 3)`. Advance
`sp` by 9 to `sp = 12`, which is the end of input. Done.

The resulting token stream is one sequence `(3, 9, 3)` plus a literal run
of 3 bytes `abc`. Serialized: 8-byte LZ header (`n_seq = 1, literals_size
= 3`), then the packed sequence descriptor, then the literal bytes.

Computing the sequence bytes: tag byte is `[0011 0110]` = `0x36` (lit_len
`3` in the high nibble, `match_len - 3 = 6` in the low nibble). No
extensions needed for either length. Offset is 3, encoded as `uint16(0x0002)`
(i.e. `off - 1`), which writes `02 00`. The sequence is 3 bytes total.

Full payload: `[01 00 00 00][03 00 00 00][36][02 00][61 62 63]` = 14
bytes. Input was 12 bytes; LZ's 8-byte header plus 3-byte sequence plus
3-byte literal pool added 2 bytes of overhead. That is not a win on a
12-byte input: the literal-only fallback (`lz_encode_literal_only`)
would also produce 12 + 8 = 20 bytes, still worse than raw, but the
matcher's sequence form is honest about what was found.

Scale this up to a 1 MiB `abcabcabc...` input: one 3-byte literal run,
one sequence with `match_len = 1048573` and `match_off = 3`. The
match-length extension needs LEB128 of `1048573 - 3 - 15 = 1048555` =
4 bytes. Total cost: 80-byte block header + 8-byte LZ header + 1-byte
tag + 4-byte LEB128 + 2-byte offset + 3 literal bytes = 98 bytes. The
measured result is 101 bytes, the extra 3 coming from the entropy-stage
sizes table and the block record's reserved fields.

Run the example program `docs/examples/theory_lz_histogram.c` to see the
measurement on three inputs (`zero`, `abc`, `noise`). A representative
run prints:

```text
zero: raw=1048576 bytes, enc=99 bytes, ratio=10591.68x, 305 MB/s
  sequences: 1   literal bytes: 1   match bytes: 1048575
  bucket     lit_hist match_hist
       0            0          0
    1- 3            1          0
    4-15            0          0
   16-63            0          0
   64-255           0          0
  256-1K            0          0
    >=1K            0          1

abc: raw=1048576 bytes, enc=101 bytes, ratio=10381.94x, 297 MB/s
  sequences: 1   literal bytes: 3   match bytes: 1048573
  bucket     lit_hist match_hist
       0            0          0
    1- 3            1          0
    4-15            0          0
   16-63            0          0
   64-255           0          0
  256-1K            0          0
    >=1K            0          1

noise: raw=1048576 bytes, enc=1048668 bytes, ratio=1.00x, 5 MB/s
  sequences: 0   literal bytes: 0   match bytes: 0
```

The `zero` input collapses to one sequence: one literal byte, then a
single match of 1048575 bytes at offset 1. The matcher emits a literal
for the very first byte (there is no earlier text to reference), hashes
position 1, finds `src[0..4] == src[1..5]` (all zeros), extends the
match to the end of input, and stops. The small-offset doubling pattern
in `tdc_match_copy` replays the run at `memcpy` speed on decode.
(`TDC_BLOCK_FLAG_ZERO_RESIDUAL` does not fire here because the RAW model
carries the residual through as-is rather than producing a zero-length
residual stream; the flag fires for models like `PLANE_2D` that detect
the zero pattern earlier.) The `abc` input is what the matcher actually
does on periodic bytes: one sequence, length 1048573, offset 3. The
`noise` input is the literal-only fallback: the parser finds no match of
3 bytes anywhere in the input, emits zero sequences, and the serializer
writes a literal-only record.

```c
--8<-- "examples/theory_lz_histogram.c"
```

The histogram example is the most direct way to watch what the matcher
does. On model-shaped residuals (try it with `DELTA_1D + ZIGZAG +
BYTE_SHUFFLE + LZ` on an `i32` ramp) the histogram settles into a few
long matches and a moderate number of short ones, which is the regime
the decoder is optimized for. On pathological inputs it collapses to the
edges above. The encoded size landed within 1.5% of those theoretical
minima on every real residual tested during tdc's 0.1.0 bake.
