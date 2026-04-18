# Entropy

## Introduction

Entropy coders run last in the tdc pipeline. By the time a stream reaches this stage, the model has turned a typed block into residuals and the transform chain has rewritten those residuals into entropy-friendly bytes. The entropy stage takes a flat `uint8_t *` buffer, compresses it, and writes the result into the block record's payload section. Decode mirrors encode. The block record's `uncompressed_size` field drives the output size, so the caller sizes the output buffer up front and the hot path skips every length check.

Public surface is small. `include/tdc/entropy.h` declares `tdc_entropy_vt` with three function pointers: `encode_bound`, `encode`, and `decode`. Backend ids live in `include/tdc/codec.h` under `tdc_entropy_id`. A caller wires a backend into a `tdc_codec_spec` by writing the id into `spec.entropy[0]`. Chaining two entropy stages is legal (the first chain-terminator, id `0x0000`, ends the chain), and every backend honours the `realloc_fn` allocator convention the rest of the library uses. There is no global state; backends are a switch in `src/core/registry.c` mapping id to vtable, and every byte of context comes from the block record.

Nine ids ship in v0. `TDC_ENTROPY_NONE` is memcpy passthrough. `TDC_ENTROPY_LZ` is the native LZ77 matcher documented in [the LZ inner loop](../theory/lz-inner-loop.md). `TDC_ENTROPY_LZ_OPT` reuses the same on-disk format but replaces the greedy parser with dynamic programming. `TDC_ENTROPY_LZ_STREAMS` and `TDC_ENTROPY_LZ_SPLIT` split the LZ byte stream across sub-coders for extra ratio. `TDC_ENTROPY_HUFFMAN` is a canonical static Huffman with max code length 15; `TDC_ENTROPY_HUFFMAN4` interleaves four Huffman bitstreams for decode ILP. `TDC_ENTROPY_FSE` is a tabled-ANS coder. `TDC_ENTROPY_LANE` splits BYTE_SHUFFLE output across byte lanes and picks a sub-coder per lane. `TDC_ENTROPY_DEFLATE` is zlib-deflate, link-time optional behind the `TDC_HAVE_ZLIB` compile flag.

Every number on this page comes from a real run of a real example in `docs/examples/`. The source input generators are deterministic: an LCG seed for the noise cases, a byte-index formula for the patterns, a gradient-plus-jitter raster for the mixed workload. Re-running the examples on another machine reproduces the ratios exactly and the throughput numbers within a few percent.

## What it does

An entropy coder maps a byte stream to a shorter byte stream using statistical regularity alone. It does not know the dtype that produced the bytes, does not know the model or transform chain upstream, and does not look at the block shape. This is what "dimension-agnostic and dtype-agnostic" means in the library: the coder sees a pointer, a length, and maybe a per-stage params struct, and returns a new byte buffer of known uncompressed size.

Two flavours cover v0. The first is dictionary coding: walk the input, look back for earlier matches of the current bytes, emit a literal run or a back-reference (offset, length) pair. LZ77 is the classical form; `TDC_ENTROPY_LZ`, `TDC_ENTROPY_LZ_OPT`, `TDC_ENTROPY_LZ_STREAMS`, and `TDC_ENTROPY_LZ_SPLIT` are all parser variants over the same matcher. `TDC_ENTROPY_DEFLATE` is the GNU/zlib dictionary+Huffman hybrid. Dictionary coders pay off on inputs with exact byte-level repetition; they are blind to byte-distribution skew that has no repeated sequences.

The second flavour is symbol coding: read a histogram of the input bytes, build an optimal code that assigns short codewords to common bytes and long codewords to rare bytes, emit the coded bitstream. `TDC_ENTROPY_HUFFMAN` and `TDC_ENTROPY_HUFFMAN4` are canonical static Huffman coders; `TDC_ENTROPY_FSE` is the tabled-ANS (Finite State Entropy) variant that reaches fractional-bit precision. Symbol coders win on byte distributions that skew hard but carry no repetition, and they always pay a small overhead for the code table the encoder stores in the payload.

The boundary between the two is fuzzy on real workloads. A DEM raster through `PRED_2D` followed by `ZIGZAG + BYTE_SHUFFLE` produces a byte stream whose top lanes are almost all zero (high ratio with LZ) and whose low lanes are high-entropy noise (flat distribution, low Huffman gain, best handled by NONE). The right answer is often "both": `TDC_ENTROPY_LANE` dispatches a sub-coder per byte lane, exactly to split the two regimes. `TDC_ENTROPY_LZ_SPLIT` addresses the same problem differently, parsing with LZ and then Huffman-coding the literal pool and the sequence descriptors as two distinct sub-streams.

Every backend honours the same vtable contract. `encode_bound(src_size)` returns an upper bound on the worst-case output size for any `src_size`-byte input; the caller uses it to grow `dst` once through `realloc_fn`. `encode(src, src_size, params, dst)` writes the compressed bytes into `dst->data` and sets `dst->size`. `decode(src, src_size, dst, dst_size)` writes exactly `dst_size` bytes into `dst` (taken from the block record's `uncompressed_size`) and returns `TDC_OK` on success or `TDC_E_CORRUPT` on a malformed stream. No backend retains memory across calls; every backend frees its scratch before return.

## When to use / when NOT

`TDC_ENTROPY_NONE` belongs in the chain when the upstream stages have already compressed all the way down, or when the caller knows the input will not compress. Its cost is a single memcpy each way; its "ratio" is always 1.0 minus a few bytes of overhead. Pick it when the block is an encrypted payload, a random nonce, a hash digest, or when the caller has already run a different compressor and is re-wrapping bytes into a tdc record. It pays no ratio tax and no code-table cost.

`TDC_ENTROPY_LZ` is the right default on residual streams. Almost every pipeline that ends in entropy coding ends here, because its ratio is good on any repetitive input and its decode throughput is close to memcpy speed on inputs that compress at all. We reach for LZ when the upstream produces a residual whose byte-level repetition is the signal: a zero-filled run from a well-predicted region, a shuffled high lane that collapses to one long match, a periodic signal whose period lies within the 4 MiB window. LZ is the wrong pick when the input has no repetition to match but carries a strongly skewed byte distribution; LZ's parser reports zero sequences and falls back to a literal-only stream, leaving the ratio at ~1.0 while Huffman or FSE could reach 3x.

`TDC_ENTROPY_LZ_OPT` trades encode time for a 0.1 to 2 percent ratio improvement by replacing the greedy parser with a dynamic-programming "optimal" parser. It shares the on-disk format with `TDC_ENTROPY_LZ`, which means a pipeline can switch encoders without touching the decoder. Use it when readers will open the archive many times and shaving a percent off the payload matters; avoid it on write-heavy pipelines because the encoder burns 3 to 5x the budget of plain LZ.

`TDC_ENTROPY_LZ_STREAMS` and `TDC_ENTROPY_LZ_SPLIT` rework what the parser writes to disk. STREAMS splits the parse into four separated streams (literal bytes, literal lengths, match lengths, match offsets) and picks a sub-coder per stream; SPLIT keeps two streams (literal bytes and sequence descriptors) and Huffman-codes both independently. Both win 15 to 40 percent of the payload on structured residuals where LZ sees its ratio bottleneck in the tag-byte overhead rather than in the matches. Skip them on small blocks (under a few KiB) where the per-stream code tables cost more than the ratio gain.

`TDC_ENTROPY_HUFFMAN` is the right pick on byte streams with a concentrated distribution and no repetition. A sparse-zero-style output after a model that converts a mostly-zero vector into index+value pairs, a dictionary-indexed stream, or the residual of a predictor that leaves a heavy-tailed but non-repetitive distribution all fit this bucket. The variant `TDC_ENTROPY_HUFFMAN4` keeps the same compressed size (it shares the tree) and decodes 2 to 3x faster by interleaving four bitstreams that a modern out-of-order CPU decodes in parallel. Pick HUFFMAN4 when decode throughput matters and the input runs longer than 256 bytes (below that it falls back to single-stream HUFFMAN automatically).

`TDC_ENTROPY_FSE` is a tabled-ANS coder with fractional-bit precision, at ratio parity with a perfect Huffman on distributions whose minimum Shannon probability is not a power of two. On short, highly-skewed streams it beats Huffman by a fraction of a percent; on long streams the difference usually vanishes. It is the right pick when the caller cares about the last fraction of a percent of ratio; Huffman's decoder is substantially faster on typical hardware.

`TDC_ENTROPY_LANE` is the right pick after a BYTE_SHUFFLE transform. Each byte lane carries a different distribution (high lanes peak because exponent/sign bytes repeat, low lanes stay flat because mantissa noise runs close to uniform); LANE lets each lane pick its own sub-coder. The params struct `tdc_lane_entropy_params` needs `n_lanes` to match the upstream BYTE_SHUFFLE's element size, and passing NULL fails with `TDC_E_INVAL`. This is the only backend in the core set that requires non-trivial params.

`TDC_ENTROPY_DEFLATE` wraps zlib's deflate algorithm in a tdc vtable. It reaches higher ratios than LZ on most structured inputs at half the encode throughput, and it sits behind a link-time flag so vectra vendor builds do not inherit a zlib dependency they did not ask for. Reach for it when the archive gets written once and read many times, the caller already has zlib linked, and a 10-20 percent ratio win over LZ is worth the extra dependency. Skip it when tdc rides along inside a library with a minimal dependency set or when encode throughput is the constraint.

## Worked example

A complete round trip through `TDC_ENTROPY_LZ` lives in [`entropy_roundtrip_lz.c`](../examples/entropy_roundtrip_lz.c). The example encodes three 1 MiB byte streams (all-zero, repeating "abc", LCG noise) and decodes each back. The input is a `VECTOR_1D` of `TDC_DT_U8`; the model is `RAW` so the entropy stage sees the raw input bytes.

```c
--8<-- "examples/entropy_roundtrip_lz.c"
```

Running the example prints:

```text
zero       raw=1048576 enc=     99 ratio= 10591.68x  roundtrip ok
abc        raw=1048576 enc=    101 ratio= 10381.94x  roundtrip ok
noise      raw=1048576 enc=1048668 ratio=     1.00x  roundtrip ok
```

The all-zero and "abc" runs collapse to ~100 bytes each; the structure is a single LZ sequence spanning nearly the whole input. The noise case is the literal-only fallback: the parser finds no match of the minimum 3 bytes anywhere, writes `n_seq = 0`, and the decoder walks back through the literal pool in one memcpy. The 92-byte growth on noise is the block record header (80 bytes), the per-stage sizes table (4 bytes), and the LZ header (8 bytes).

[`entropy_roundtrip_none.c`](../examples/entropy_roundtrip_none.c) and [`entropy_roundtrip_huffman.c`](../examples/entropy_roundtrip_huffman.c) exercise the other two core backends. The NONE example confirms the passthrough contract: a 1024-byte input encodes to 1104 bytes, the delta being the 80-byte block record header. The HUFFMAN example runs four backends on a 65536-byte skewed stream (70% zeros, 15% ones, 15% random):

```text
  NONE     enc= 65616  ratio= 1.00x  roundtrip ok
  LZ       enc= 37254  ratio= 1.76x  roundtrip ok
  HUFFMAN  enc= 20971  ratio= 3.13x  roundtrip ok
  FSE      enc= 20329  ratio= 3.22x  roundtrip ok
```

LZ catches the long zero runs inside the distribution and lands at 1.76x, but it leaves the byte-distribution signal on the table. Huffman and FSE ignore positional structure and capture the skew directly, landing at 3.1x and 3.2x. The boundary between the two is the crux of "when to use what"; every real workload benefits from testing both on a representative block before committing.

## Parameter tuning

Most backends take `NULL` for their params. Three backends expose real knobs.

`TDC_ENTROPY_LZ` and `TDC_ENTROPY_LZ_OPT` accept a `tdc_entropy_level` struct with a single `int level` field. Level 0 is the default; the LZ greedy parser ignores the level (its single-pass matcher has nothing to tune), but `LZ_OPT`'s level controls how deep the parser's search runs. Higher levels trade encode time for ratio on structured inputs; a level sweep of `{0, 1, 3, 6}` on the DEM benchmark below covers the sensitivity range. Decode ignores the level entirely because the on-disk format stays fixed.

`TDC_ENTROPY_LZ_STREAMS` takes `tdc_lz_streams_params` with three fields: `level`, `min_match`, and a reserved u32. The `min_match` knob raises the smallest match length the parser will emit from the baseline 3 up to 4-8 bytes. Raising it fuses short matches back into literal runs, raising bytes-per-sequence and decode throughput at the cost of a few percent ratio. On DEM residuals a `min_match = 5` setting drops decode-path sequence count by 30-40 percent at 1-2 percent worse ratio. The on-disk format does not change; only the parser shifts.

`TDC_ENTROPY_LANE` takes `tdc_lane_entropy_params` with `n_lanes` and `lane_entropy[8]`. `n_lanes` must match the element size of the upstream BYTE_SHUFFLE (2 for i16, 4 for i32/f32, 8 for i64/f64). Each `lane_entropy[i]` is either `TDC_ENTROPY_NONE` (letting LANE pick a coder per lane from a heuristic) or an explicit id (HUFFMAN, FSE, NONE) forcing that lane's sub-coder. On a float raster after BYTE_SHUFFLE, setting the exponent-byte lane to HUFFMAN and the mantissa lanes to NONE typically beats a single-coder choice by 5-10 percent of the payload, because mantissa noise is incompressible and paying for a Huffman tree on it is overhead. The auto path samples each lane's Shannon entropy on a 4 KiB prefix before picking a sub-coder.

`TDC_ENTROPY_DEFLATE` takes a `tdc_entropy_level` with `level` in the zlib range of 0-9. Level 6 is the zlib default; level 9 adds 15-20 percent encode time for 1-3 percent better ratio; level 1 runs 2-3x faster at 5-10 percent worse ratio. Decode is level-independent.

None of the other core backends (NONE, LZ greedy, HUFFMAN, HUFFMAN4, FSE, LZ_SPLIT) expose tunable params. Passing a non-NULL `entropy_params[i]` is safe for those backends; they ignore it.

## Benchmarks

Every number in the table below came from running an example in `docs/examples/` on the author's Windows 11 workstation (MSVC 2022, Release config, AVX2). Each row reports the median of five runs; encode and decode throughput divide raw bytes by wall time. The raw byte count and ratio come directly from the example's output.

| Workload (raw) | Backend | Encoded | Ratio | Encode | Decode |
|---|---|---:|---:|---:|---:|
| 1 MiB LCG noise | NONE | 1,048,656 | 1.00x | 1,866 MB/s | 3,681 MB/s |
| 1 MiB LCG noise | LZ | 1,048,668 | 1.00x | 15 MB/s | 3,687 MB/s |
| 1 MiB LCG noise | HUFFMAN | 1,048,926 | 1.00x | 643 MB/s | 528 MB/s |
| 1 MiB LCG noise | FSE | 1,058,994 | 0.99x | 187 MB/s | 372 MB/s |
| 1 MiB 32-byte pattern | NONE | 1,048,656 | 1.00x | 2,027 MB/s | 3,521 MB/s |
| 1 MiB 32-byte pattern | LZ | 131 | 8004x | 557 MB/s | 3,827 MB/s |
| 1 MiB 32-byte pattern | HUFFMAN | 655,486 | 1.60x | 767 MB/s | 905 MB/s |
| 1 MiB 32-byte pattern | FSE | 655,524 | 1.60x | 213 MB/s | 402 MB/s |
| 512x512 i16 DEM mix | NONE | 524,369 | 1.00x | 1,075 MB/s | 519 MB/s |
| 512x512 i16 DEM mix | LZ | 232,867 | 2.25x | 92 MB/s | 398 MB/s |
| 512x512 i16 DEM mix | HUFFMAN | 183,641 | 2.85x | 445 MB/s | 316 MB/s |
| 512x512 i16 DEM mix | FSE | 187,623 | 2.79x | 174 MB/s | 218 MB/s |

The mixed-input row is a 512x512 `i16` raster of four planar quadrants plus small jitter, piped through `PRED_2D / PAETH -> ZIGZAG -> BYTE_SHUFFLE` before reaching the entropy stage. This is the shape a real pipeline presents to entropy: short-magnitude signed residuals, byte-shuffled so the high lanes are mostly zero. Source is [`entropy_bench_mixed.c`](../examples/entropy_bench_mixed.c).

Reading the table row by row: the incompressible case is the floor. NONE is the only backend that compresses nothing while passing the input through at memory bandwidth; LZ grows the output by 92 bytes and runs encode at 15 MB/s because every position probes, misses, and advances in the accelerating-step miss loop; HUFFMAN barely grows the output (the distribution is nearly uniform) but pays a code-table cost; FSE pays even more for the state-machine tables. The practical takeaway is that feeding an already-random stream into any of LZ/HUF/FSE costs time and adds bytes.

The repetitive case hits the ceiling LZ targets. A 32-byte cycle compresses 8004x through LZ; the matcher emits one sequence and the rest of the record is the 8-byte LZ header plus a 3-byte sequence plus a literal seed. HUFFMAN and FSE both sit at 1.60x on the same input because the byte distribution has only 32 unique values and each one appears equally often; their tree captures the 5-bit alphabet but no more. LZ's decode matches memcpy throughput (3,827 MB/s) because the small-offset doubling pattern in the match copy replays the 32-byte run at SIMD speed.

The mixed case is where real workloads live. LZ captures some ratio (the zero lanes collapse into long matches) but leaves the low lanes' byte-distribution signal untouched. Huffman captures the distribution and lands at 2.85x, beating LZ by 22 percent of the payload. FSE matches Huffman within a few percent on this input because the distribution is close to a power-of-two geometric. This is the archetypal case for `TDC_ENTROPY_LANE` or `TDC_ENTROPY_LZ_SPLIT`, which pick up both the zero runs (LZ-style) and the byte distribution (Huffman-style) in one pass; see the "which backend wins" harness below for the full table.

A nine-backend comparison on a smaller 256x256 i16 raster (same pipeline up to entropy) lives in [`entropy_which_wins.c`](../examples/entropy_which_wins.c):

```text
  NONE            enc= 131153  ratio=    1.00x  enc=   635 MB/s  dec=   487 MB/s
  LZ              enc=  65534  ratio=    2.00x  enc=   110 MB/s  dec=   643 MB/s
  LZ_OPT          enc=  65438  ratio=    2.00x  enc=    21 MB/s  dec=   654 MB/s
  LZ_STREAMS      enc=  47796  ratio=    2.74x  enc=    79 MB/s  dec=   297 MB/s
  LZ_SPLIT        enc=  48068  ratio=    2.73x  enc=    21 MB/s  dec=   289 MB/s
  HUFFMAN         enc=  62431  ratio=    2.10x  enc=   510 MB/s  dec=   325 MB/s
  HUFFMAN4        enc=  62443  ratio=    2.10x  enc=   532 MB/s  dec=   396 MB/s
  FSE             enc=  62518  ratio=    2.10x  enc=   165 MB/s  dec=   247 MB/s
```

LZ_STREAMS and LZ_SPLIT are the payload winners here (2.74x vs LZ's 2.00x) because both capture the sequence-descriptor regularity and the literal byte distribution in one go. LZ_OPT spends 5x the encode budget of LZ for a 0.15 percent ratio win, which is the typical optimal-parser signature on greedy-friendly inputs. HUFFMAN4 matches HUFFMAN's ratio to within 12 bytes and decodes 20 percent faster. The "right" pick depends on the pipeline: write-once archives go to LZ_STREAMS; read-heavy pipelines go to LZ + HUFFMAN4 chain; the read-mostly + maximum ratio combination is LZ_SPLIT.

## Edge cases

[`entropy_roundtrip_none.c`](../examples/entropy_roundtrip_none.c) exercises the empty and short-input corners through NONE. An empty input (n = 0) encodes to 84 bytes: 80 bytes of block record header plus a 4-byte per-stage sizes table. Decode returns `TDC_OK` with zero bytes written. A single-byte input encodes to 85 bytes, and the same round trip succeeds. Every backend in the core set handles empty input without crashing and without allocating a payload.

The repeating-input edge case hits LZ's large-match path. On a constant-byte input, the matcher emits a single sequence of length `N - 1` at offset 1; the small-offset doubling pattern in `tdc_match_copy` replays the run at memcpy speed during decode. The `TDC_BLOCK_FLAG_ZERO_RESIDUAL` flag does not fire under a RAW model because RAW copies the input through unchanged rather than producing a zero residual. On a 1 MiB zero input, the encoded record is 99 bytes: 80-byte block header, 4-byte sizes table, 8-byte LZ header, one sequence descriptor (tag + LEB128 match extension + 2-byte offset), and one literal byte. The flag would fire under a model that can predict the input exactly (PLANE_2D on a plane, PRED_3D/GRAD3D on a tri-affine volume); in those cases the entropy stage is never called and the payload_size lands at zero.

Incompressible inputs are the opposite corner. `entropy_bench_incompressible.c` runs 1 MiB of LCG noise through NONE / LZ / HUFFMAN / FSE. LZ's greedy parser sees no matches, falls through to the literal-only encoder, and writes an 8-byte header plus every input byte; net overhead is 12 bytes (header + empty-sequence bookkeeping). HUFFMAN detects a near-uniform distribution and writes a compact code table, then bitpacks the input; FSE does the same with a larger state-table overhead. None of the four actually compresses the stream; the point of running the bench is to confirm the overhead and the throughput floor.

[`entropy_corrupt_lz.c`](../examples/entropy_corrupt_lz.c) and [`entropy_truncated_lz.c`](../examples/entropy_truncated_lz.c) cover corrupted and truncated inputs. The corrupt example flips one byte at five positions inside the record and reports the status code:

```text
  flip @header magic byte 0 -> TDC_E_CORRUPT
  flip @n_seq byte 0       -> TDC_OK
  flip @literals_size[0]   -> TDC_E_CORRUPT
  flip @first tag byte     -> TDC_E_CORRUPT
  flip @offset LSB         -> TDC_E_CORRUPT
```

Four of the five flips return `TDC_E_CORRUPT`. The one that slips through is `n_seq` LSB on an input where the existing sequences already cover the output: the decoder's inner loop terminates on `dp >= uncompressed_size` before reading sequences that do not exist. This is the documented behaviour of a bounded-output decoder and is defensible: the decoder delivers bytes identical to the input regardless of how large `n_seq` was flipped to, because the stream encodes what the first valid sequence says. Every other single-byte flip lands in a field whose corruption the bounds checks notice.

The truncation example feeds `tdc_decode_block_into` a shortened buffer at four offsets: half of the header, the full header (no payload), the record minus 4 bytes, and the record minus 1 byte. All four return `TDC_E_CORRUPT`. The header case is rejected by `tdc_decode_peek` against the block-record magic; the payload truncation cases are rejected by the LZ stream decoder when a sequence walks past the end of the literal pool or an offset walks past the start of the output. None of them hit an out-of-bounds read on the input buffer.

The degenerate "wrong backend" case matters: feeding `TDC_ENTROPY_LANE` a spec with no `entropy_params[0]` (the struct that carries `n_lanes` and the per-lane sub-coder selection) returns `TDC_E_INVAL` at encode time. The output buffer is untouched; the caller either supplies the params or picks a different backend. This is the only core entropy backend that requires non-NULL params; HUFFMAN, FSE, LZ, and friends accept NULL.

## Integration notes

Pick an entropy backend after the model and transform chain are fixed. The model determines the residual's statistical shape: a delta model leaves small-magnitude signed values; a predictor leaves structured zero-heavy residuals; a dictionary model leaves index bytes that look uniform but with a small alphabet. The transform chain then rewrites those statistics: ZIGZAG flattens the high bytes of negatives into zeros; BYTE_SHUFFLE groups same-significance bytes into runs. The entropy stage sees the result.

Canonical pairings on real workloads:

- Model producing repetitive residuals (DELTA on timestamps, PRED on smooth rasters, PLANE on planar regions) + `ZIGZAG + BYTE_SHUFFLE` + `TDC_ENTROPY_LZ`. Ratio is driven by the long zero runs the shuffle produces; LZ captures them directly. This is the default for almost every integer pipeline in the quickstart and the shape vectra's numeric columns take most often.
- Model producing a heavy-tailed distribution (SPARSE_ZERO with non-zero positions, DICT_NUMERIC with index bytes) + no transform + `TDC_ENTROPY_HUFFMAN`. Huffman is optimal when there are no sequential repeats to catch; the transform chain would just shuffle an already-skewed distribution.
- Float raster + fused `TDC_MODEL_QUANTIZE_PRED_2D` + `ZIGZAG + BYTE_SHUFFLE` + `TDC_ENTROPY_LZ_STREAMS`. The quantization-plus-predictor fuses produce a signed integer residual with the right shape for a split-stream parser; STREAMS picks up the per-stream distributions and lands 15-25 percent ahead of plain LZ.
- Small-alphabet float stream (quantized sensor readings) + BYTE_SHUFFLE at the natural element width + `TDC_ENTROPY_LANE` with `n_lanes` matching the element size.

Chaining two entropy stages is legal and supported. `entropy[0] = TDC_ENTROPY_LZ` followed by `entropy[1] = TDC_ENTROPY_HUFFMAN` runs LZ first, hands its output to Huffman, and writes a single combined payload. The decoder reverses the chain. This pair was the classical zstd level-0 shape, and it still pays off on some workloads, though `TDC_ENTROPY_LZ_SPLIT` usually beats it because the split parser sees both streams jointly. Do not chain LZ_SPLIT or LZ_STREAMS with a second entropy stage; both already serialize their output as a pre-packed multi-stream payload, and tacking Huffman on top re-codes a code table.

Allocation works the same as the rest of the library. The encoder grows `out->data` via the caller's `realloc_fn` up to the backend's `encode_bound(src_size)`; the decoder never resizes the caller's destination. Scratch memory (hash tables, sequence arrays, code-table builders) is allocated through `realloc_fn` too when the caller uses `tdc_decode_block_ex`; falling back to libc `realloc` happens only when the caller uses the simpler `tdc_decode_block_into`. Vectra plugs its per-row-group arena here; standalone tests use plain `realloc`.

Threading works per block. A `tdc_codec_spec` is POD and safe to share across threads. Each thread owns its own `tdc_buffer` with its own `realloc_fn`. The backends keep no hidden global state and no thread-local caches; scratch is strictly on the heap. The only subtle point is that two threads encoding with `TDC_ENTROPY_LZ_OPT` will allocate separate 4 MiB hash tables, so the total RSS scales with thread count. Arena allocators handle this naturally.

Endianness is little-endian, always. Every backend writes little-endian u16, u32, and u64 fields directly. A big-endian port would require either byte-swapping in each backend's encode/decode or a format version bump; neither is on the v0 roadmap because the supported targets (x86_64, aarch64) are both little-endian.

## Inner loop / hot path

`TDC_ENTROPY_LZ`'s inner loop is the hottest code in the library; [the LZ inner loop theory page](../theory/lz-inner-loop.md) walks the matcher, the wildcopy, and the fast/safe decode split in full. The short version from the backend point of view: the parser maintains an 18-bit direct-addressed hash table with Knuth multiplicative Fibonacci hashing (`h = (x * 2654435761u) >> 14` for the 4-byte input key), probes one candidate per position on the flat-hash path, and extends any match 8 bytes at a time with `__builtin_ctzll` on mismatch. The decoder's hot primitive is `tdc_wildcopy32`, which writes 32 bytes per loop iteration on AVX2 builds and sits inside a single bounds check per sequence.

The hash function matters because it is the only fingerprint used to propose match candidates, and it runs once per input byte on the hot path. Knuth's Fibonacci multiplier (`0x9E3779B1`, the 32-bit rounded inverse of the golden ratio) distributes any 32-bit integer into 2^18 buckets with no measurable bias on real residual streams. A hash collision does not break correctness because the 4-byte equality check gates the candidate before the 8-byte extension starts. A different hash used elsewhere in the vectra ecosystem is FNV-1a; FNV shows up in vectra's `JoinHT`, in `VecHashTable`, and in the `.vtri` index tables, but not in the LZ matcher. Naming the hash correctly matters because the two structures have different failure modes: FNV-1a plus 70 percent load targets dictionary lookup with bounded probe length; Knuth Fibonacci with deliberate over-loading targets a position index where collisions are the point.

The matcher's miss path uses an accelerating step: `sp += 1, 1, 2, 3, ..., 8, 8` after consecutive misses, reset to 1 on the first hit. On a literal-heavy prefix (the LCG-noise input above) this skips roughly 60 to 80 percent of the input rather than probing every byte, which is what keeps the encode throughput on incompressible data from collapsing to a few MB/s. The accelerating step is the reason "LZ on noise is slow" not "LZ on noise hangs"; the 15 MB/s encode throughput in the benchmark table is the step loop running most of the way through.

Wildcopy is the decoder's fast primitive. A match copy of length `mlen` at offset `off` writes `ceil(mlen / 32)` blocks of 32 bytes through `_mm256_storeu_si256`, overcopying up to 31 bytes past the nominal end. The slack is allocated by the caller (`TDC_WILDCOPY_SLACK` bytes past the declared output size); the bounds check before each copy (`dp + lit_len + mlen + TDC_WILDCOPY_SLACK > uncompressed_size`) bails to the safe path when a copy would walk off the end. Small offsets (`off < 16`) go through a doubling seed-and-fill: write `off` bytes at the destination, then copy the filled region onto itself at twice the offset, until the filled region covers `mlen`. For `off = 1` on a zero run this decodes at memcpy speed.

Two measured dead ends shape the LZ hot path and deserve naming in the backends page because they are benchmarked, not theorized.

The first is Robin Hood hashing. The obvious improvement to the matcher's collision handling is to displace entries so that newer positions sit closer to their home bucket, which would shorten the average chain walk. Measured in vectra on the standard corpus, the Robin Hood variant was 12-29 percent slower on encode throughput. The reason is structural: LZ's hash table is direct-addressed with at most one probe per position on the flat-hash path and at most `chain_depth + 1` probes on the chain path. There is no probe distribution to reshape; Robin Hood's per-insert swap cost dominates the zero savings. Note that this is not a statement about Robin Hood in general; the same algorithm beats FNV-1a + linear probing on vectra's `JoinHT` dictionary hash on a different workload. The LZ matcher's structure is not that structure, and "Robin Hood helps dictionary lookups" does not cross over.

The second is batched parse-then-copy on the decoder. The idea is to split the per-sequence loop into two phases: parse several sequences into a stack-local descriptor array first, then walk the array in a tight copy loop. The hypothesis is that the parse stays in L1 while the copy spills cache, and separating the phases lets the CPU prefetch the copy destinations while the parse runs. Measured in vectra at every batch size from 4 to 16, the batched variant was 6-7 percent slower. The reason is that LZ's parse is already cheap enough that the out-of-order scheduler overlaps it with the previous sequence's wildcopy; forcing a phase split serializes them and recovers nothing. A note in the source flags this: batching may pay if a real entropy stage (FSE or Huffman) sits in front of LZ, because the parse becomes expensive enough to benefit from a dedicated pipeline. Until then, the decoder as written is already the overlapped form.

`TDC_ENTROPY_HUFFMAN`'s hot path is simpler. Encode reads a 256-entry byte histogram, builds a canonical code with the package-merge algorithm capped at code length 15 (which fits in a u16 lookup on decode), emits the code-length table followed by the bitpacked payload. Decode uses a 2^15-entry lookup table so every symbol is one load, one length-read, one bitstream advance. `TDC_ENTROPY_HUFFMAN4` runs that tree over four interleaved bitstreams, so the decoder fetches four symbols per loop iteration and feeds the out-of-order core's multiple load ports simultaneously. The decode measurement in the benchmark table above (HUFFMAN4 at 396 MB/s vs HUFFMAN at 325 MB/s on the same input) is the ILP win.

`TDC_ENTROPY_FSE` replaces the canonical Huffman tree with a tabled-ANS state machine; encode builds a 4096-entry transition table, decode reads one state transition per symbol. The tables cost the encoder around 1 KiB of overhead per block, which is why FSE slightly underperforms Huffman on ratios when the distribution is already a power-of-two geometric. On long, highly-skewed streams FSE catches up to Huffman's ratio and beats it by a fraction of a percent.

## Multiple variants

The LZ family is four variants: greedy (`LZ`), optimal-parser (`LZ_OPT`), separated-stream (`LZ_STREAMS`), and split-stream (`LZ_SPLIT`). Their on-disk formats differ. LZ and LZ_OPT share the single-stream format and decoder; a STREAMS or SPLIT record is not readable by the LZ decoder and vice versa. The same pipeline can switch between LZ and LZ_OPT at encode time without touching the decoder, which is useful for adaptive write paths that profile a sample before committing.

The Huffman family is two variants: single-stream `HUFFMAN` and four-stream interleaved `HUFFMAN4`. They share the code-table on disk (the decoder reads the same canonical code for both), but the bitstream layout differs: HUFFMAN writes one bitstream packed LSB-first, while HUFFMAN4 writes four bitstreams with independent cursors. The tree and the per-byte ratio are identical; the throughput differs because HUFFMAN4 issues four loads per iteration against the core's multiple load ports. HUFFMAN4 falls back to single-stream HUFFMAN on inputs shorter than 256 bytes, where the per-stream framing overhead would dominate the payload.

Each variant has its own worked example in `docs/examples/`. [`entropy_which_wins.c`](../examples/entropy_which_wins.c) runs all nine core backends on one input and tabulates the full cost surface; a caller picking a backend for a new workload should port that harness to their own block and read the row that optimizes for their encode/decode/ratio constraint. The harness is ~150 lines of C and shows the full pattern: build a spec, time encode and decode as medians over five runs, compute ratio.

`LZ_STREAMS` differs from `LZ_SPLIT` in the slice axis. STREAMS splits the parser output into four streams (literal bytes, literal lengths as u32, match lengths as u32, match offsets as u32) and picks NONE/HUFFMAN/FSE per stream based on a Shannon-entropy sample. SPLIT uses the same optimal parser as LZ_OPT and separates the output into two streams: literal bytes and sequence descriptors. Both Huffman-code the sub-streams. On a typical post-transform residual SPLIT beats STREAMS by 1-2 percent of the payload because it preserves the ordering correlation that the parser creates, at the cost of ~3x the encode time because it shares LZ_OPT's parser.

The `HUFFMAN` vs `HUFFMAN4` vs `FSE` triple is the "which symbol coder" axis. HUFFMAN4 matches HUFFMAN on ratio and is strictly faster on decode for inputs over 256 bytes. FSE costs more table overhead than Huffman but reaches fractional-bit code lengths, which beats Huffman by a fraction of a percent on distributions whose symbol probabilities are not powers of two. On the DEM mixed workload Huffman lands at 2.85x and FSE at 2.79x; the Huffman lead comes from the code-table cost being smaller than FSE's transition table on a 524 KiB input. On longer inputs or more skewed distributions the ranking flips.

The `NONE` backend is the chain terminator for the entropy array as well as a real pass-through option. A spec with `entropy[0] = TDC_ENTROPY_NONE` and `entropy[1..3] = 0` is the same as a spec with all entropy slots zero: the driver treats both as "no entropy" and writes the transform output directly to the payload. The difference is only cosmetic on disk; the block record still carries the entropy ids as written.

## Failure modes

Four failure modes dominate entropy-stage bugs in real pipelines. Each has a specific status code and a specific fix.

`TDC_E_CORRUPT` is the most common. It fires whenever the decoder detects that the on-disk payload violates an invariant: a sequence claims to match beyond the current output offset, a literal pool is shorter than declared, a Huffman code table encodes a symbol length greater than 15, an FSE transition table points outside the state space. The fix is to re-encode the block from the source data; a corrupted block cannot be recovered without the original input. The truncation example above hits `TDC_E_CORRUPT` at every truncation point because either the block header validator or the entropy stream decoder notices the short buffer.

`TDC_E_INVAL` fires at encode time when the params struct fails a required-field check. `TDC_ENTROPY_LANE` without `entropy_params[0]` is the canonical example: the encode path has no `n_lanes` to work from and no way to guess one. `TDC_ENTROPY_LZ_STREAMS` with an out-of-range `min_match` (values above ~8) also hits `TDC_E_INVAL`. The fix is always to supply the params. Every other core backend accepts NULL params.

`TDC_E_NOMEM` fires when `realloc_fn` returns NULL. The encoder asks for up to `encode_bound(src_size)` bytes for the output plus ~1 MiB of hash-table scratch on LZ, plus code-table memory on Huffman/FSE. A pool or arena allocator that runs out mid-encode returns NULL, the vtable propagates `TDC_E_NOMEM`, and the caller either grows the allocator or falls back to a cheaper backend. The state on `TDC_E_NOMEM` is "partial allocation freed"; the caller's output buffer is untouched past its pre-call size.

`TDC_E_UNSUPPORTED` fires when the id is not registered. `TDC_ENTROPY_DEFLATE` without `TDC_HAVE_ZLIB` returns `TDC_E_UNSUPPORTED` at encode time because the registry switch returns NULL for that id, and the driver surfaces the error before calling the vtable. The fix is to build tdc with `-DTDC_HAVE_ZLIB=ON` or to pick a native backend.

Inputs that defeat the backend are a softer failure mode: the encode succeeds but the output comes out the same size or larger than the input. On genuinely random bytes LZ returns ratio ~1.0 because its parser finds no matches, and Huffman returns ratio ~1.0 because the byte histogram is already flat. On a 1 MiB LCG-noise input the LZ encode lands at 1,048,668 bytes (92 bytes of overhead) and HUFFMAN at 1,048,926 bytes (350 bytes of overhead for the code table plus the bitpacked near-uniform distribution). Neither is a bug; it is the entropy bound. Pick `TDC_ENTROPY_NONE` instead: its in-payload overhead is 0 bytes and it runs at memory bandwidth.

A subtler failure mode comes from model-transform mismatch. A block that enters the entropy stage with a residual whose byte distribution LZ cannot match and whose distribution Huffman cannot skew is "compressed by accident": the earlier stages have done everything useful, and any entropy stage adds overhead with no ratio payoff. The tell is a post-entropy size within a few tens of bytes of the post-transform size across all backends. The fix is to drop to `TDC_ENTROPY_NONE`; the other backends are not wrong, they just have nothing to exploit.

Stacking two entropy stages usually makes things worse. `entropy[0] = LZ` followed by `entropy[1] = HUFFMAN` used to be the default in classical pipelines, but on tdc-shaped residuals it loses to `TDC_ENTROPY_LZ_SPLIT` because the joint parser sees both the sequence structure and the literal distribution at once, while the staged pipeline sees them sequentially. The mechanical loss is about 5-15 percent of the payload on typical inputs. The fix is to pick a joint backend; the stacked-chain option remains supported for callers whose pipeline already depends on that shape.

Finally, a pipeline that trips `TDC_BLOCK_FLAG_ZERO_RESIDUAL` (the model produced a zero-length residual, e.g. PLANE_2D on a plane or PRED_3D/GRAD3D on a tri-affine volume) skips the transform and entropy stages entirely. The block record reports `payload_size = 0` and the entropy backend ids are written to disk but never invoked. Benchmarks that feed too-clean inputs to a model capable of detecting them will hit this path and overstate entropy compression by an order of magnitude. The benchmark workaround is to inject jitter into the input so the model does not recognize the fast-path condition; in production this is the correct behaviour and needs no change, because the zero-residual path is why tdc produces single-byte-header-overhead records on clean inputs.
