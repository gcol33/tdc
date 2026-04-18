# Transforms

## Introduction

The transform stage sits between the model and the entropy coder. It takes a flat byte stream with a declared input dtype, rewrites those bytes into something more entropy-friendly, and hands the result down the chain. tdc calls this stage "representation" internally because it owns every rewrite that does not depend on neighborhood structure: sign interleaving, byte-lane transpose, integer quantization, run-length packing, and anything else that turns residual bytes into symbols the entropy coder can compress. The header is `include/tdc/transform.h`, and callers pick backends through the `xform[]` array on `tdc_codec_spec`.

v0 ships four transform ids: `TDC_XFORM_NONE` (chain terminator, id `0x0000`), `TDC_XFORM_QUANTIZE` (id `0x0001`), `TDC_XFORM_ZIGZAG` (id `0x0002`), and `TDC_XFORM_BYTE_SHUFFLE` (id `0x0003`). The enum reserves a fifth id for bit shuffle (`0x0004`), and a reference backend `TDC_XFORM_COMPLEMENT` lives in the experimental range (`0x0100`); neither matters for the main pipeline and both live in other vignettes. The core three cover every chain the quickstart wires up, and they are the only transforms most callers will ever reach for.

Transforms are dimension-agnostic and model-agnostic. They see a `uint8_t *` buffer, an input dtype, a byte count, and an optional params pointer. They never inspect the block shape, the block layout, or any side metadata from the model. This separation is what lets the same `ZIGZAG -> BYTE_SHUFFLE` pair sit downstream of `DELTA_1D` on a 1D sensor column, `PRED_2D` on a raster, `PRED_3D` on a volume, and `STACK_2D` on a frame stack. One pipeline, one byte buffer, one knob set.

## What it does

A transform is a byte-level rewrite with a known inverse. The input is `src_size` bytes interpreted as `src_size / dtype_size(in_dtype)` elements of `in_dtype`. The output is another byte stream with a possibly different output dtype (zigzag flips `i32` to `u32`, quantize maps `f64` to `i16`, byte shuffle keeps the dtype but reorders bytes). Every encode has a matching decode that reconstructs the original byte stream exactly. `ZIGZAG` and `BYTE_SHUFFLE` are bit-exact lossless; `QUANTIZE` is the one transform in the core set that discards information on purpose.

Zigzag interleaves signed integers into an unsigned codomain that keeps small magnitudes near zero: `0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4`, and so on. On a signed residual stream with a few-bit dynamic range, the two's-complement encoding of `-3` is `0xFFFFFFFD` as an `i32`; the zigzag encoding of the same value is `5`. That turns long runs of `0xFF` bytes in negative residuals into long runs of `0x00`, which both LZ and byte shuffle can exploit. The kernel is one shift plus one xor per element, and the output width matches the input width to the bit. Accepted dtypes: `I8`, `I16`, `I32`, `I64`.

Byte shuffle transposes a fixed-width element stream by byte lane. Before the transform, a 4-byte element stream lives as `[e0b0 e0b1 e0b2 e0b3][e1b0 e1b1 e1b2 e1b3]...`. After the shuffle, the bytes group by lane: `[e0b0 e1b0 e2b0 ...][e0b1 e1b1 e2b1 ...][e0b2 e1b2 e2b2 ...][e0b3 e1b3 e2b3 ...]`. Same-significance bytes across elements are now adjacent. On a stream of small-magnitude `i32` values, the top two byte lanes are all zeros and the third lane is nearly constant, so LZ can collapse them into a handful of match tokens. Accepted dtypes: every fixed-width numeric dtype (`I8` through `F64`, plus `F16`). `elem_size == 1` short-circuits to `memcpy` because there is nothing to transpose. SSE2 fast paths cover 2/4/8-byte elements on x86_64; NEON covers the same widths on aarch64.

Quantize maps `f16`, `f32`, or `f64` input to a signed integer target (`I8`, `I16`, or `I32`) via `stored = round((value - offset) * scale)`. The arithmetic is the standard linear quantizer: pick a scale and an offset that map the value range of interest into the target's representable range, then round to the nearest integer. Decode reverses the arithmetic: `value = stored / scale + offset`. The round trip loses a step of `1 / scale`, which the caller picks to match the tolerance of the downstream task. The encode path rejects `I64` targets, because round-tripping through `double` has only 52 bits of mantissa and so an `I64` target is numerically meaningless. Out-of-range values clamp to `[target_min, target_max]`, NaN encodes to `target_min` as a sentinel, and the encode path never fails on a finite input.

## When to use / when NOT

`BYTE_SHUFFLE` is the first transform we reach for on any numeric residual stream of width two or wider. It pays for itself whenever the top byte lanes of consecutive elements carry near-identical values. Delta residuals, predictor residuals, and float mantissas all qualify. At `elem_size == 1` it is a no-op; leaving it in the chain for `i8` or `u8` inputs wastes one pass over the buffer and nothing more. When the input is already random-looking bytes (an encrypted stream, a Huffman output, a compressed payload the caller is rewrapping), the byte-lane transpose has nothing to exploit and the ratio stays at 1.0 times. Shuffle also runs best downstream of a model that produces residuals with structure: shuffling raw `i64` timestamps is a lot less effective than shuffling the deltas of those timestamps, because the raw timestamps only share their top three or four bytes while the deltas share seven.

`ZIGZAG` pays off whenever the upstream residuals include negative values. The vtable accepts every signed integer dtype. We use it as a near-default in front of `BYTE_SHUFFLE` when a model emits signed residuals, because without it the byte shuffle lifts the 0xFF high bytes of negative two's-complement values into a "high lane" that looks noisy, while the low lane carries the actual small-magnitude signal. With zigzag in front, the high lane goes quiet and the low lane carries all the information. On an unsigned residual stream (a `u32` dictionary index column, for example), zigzag has no sign to flip; the vtable returns `TDC_E_DTYPE` on unsigned input, or, when the model emitted signed bytes but every residual happened to be non-negative, it adds one cheap pass with no ratio change. The chain-compare benchmark below shows the latter case.

`QUANTIZE` only belongs in the chain on a float input where the caller has quantified the tolerance they are willing to trade for compression. It is the single lossy backend in the transform stage, and the entire round-trip loses the sub-step noise by design. When the caller wants bit-exact float round trips, `QUANTIZE` is the wrong stage. When the caller wants to compress a float raster and a PRED_2D-style predictor is available, the fused model `TDC_MODEL_QUANTIZE_PRED_2D` is the right pick; running `PRED_2D` first and `QUANTIZE` as a downstream xform produces a float residual that the quantizer then mangles into garbage values, because `QUANTIZE` quantizes values and a residual-of-floats is no longer in the target's domain.

`NONE` (id `0x0000`) is the chain terminator, never a normal "do nothing" transform. The first zero entry in `xform[]` ends the chain; anything past it must keep the zero. A spec that puts `NONE` in the middle of the array behaves as though the chain ended early. The chain-terminator rule is what keeps a stack-allocated spec valid without requiring a separate length field.

## Worked example

Five `.c` files under `docs/examples/` cover the codec walkthrough. The first, [`xforms_roundtrip.c`](../examples/xforms_roundtrip.c), runs the minimum full pipeline: model is `RAW`, the transform chain is `ZIGZAG -> BYTE_SHUFFLE`, entropy is `LZ`. The block carries 2,048 `i16` values with small-magnitude jitter, the shape a delta or 2D predictor would typically emit. The input is 4,096 bytes; the output is 114 bytes, 80 of which go to the block record header. The payload (transform plus entropy output plus side metadata) therefore fits in 34 bytes for the whole block. `memcmp` against the original confirms the reconstruction.

```
input  : 4096 bytes (2048 x i16)
encoded: 114 bytes (block record, incl. 80-byte header)
decoded: 4096 bytes, memcmp == source: yes
```

The codec spec construction is the whole point of the example. We set the model once, list the transforms in order, and terminate the chain with `TDC_XFORM_NONE`. The entropy stage runs a single LZ pass; anything past `entropy[0]` stays zero.

```c
tdc_codec_spec spec = {0};
spec.model      = TDC_MODEL_RAW;
spec.xform[0]   = TDC_XFORM_ZIGZAG;
spec.xform[1]   = TDC_XFORM_BYTE_SHUFFLE;
spec.xform[2]   = TDC_XFORM_NONE;
spec.entropy[0] = TDC_ENTROPY_LZ;
```

The buffer uses the allocator convention from the quickstart: `qs_realloc(user, ptr, n)` allocates, grows, or frees depending on the arguments. `tdc_encode_block` writes the full block record into `enc`; `tdc_decode_peek` reads only the 80-byte header so we can size the destination buffer before committing scratch memory; `tdc_decode_block_into` fills the caller-provided destination in place without reallocating.

```c
tdc_buffer enc = qs_buffer();
tdc_encode_block(&src, &spec, &enc);

tdc_block meta = {0};
size_t need = 0;
tdc_decode_peek(enc.data, enc.size, &meta, &need);

void *dst_data = qs_realloc(NULL, NULL, need);
tdc_block dst = meta;
dst.data = dst_data;
tdc_decode_block_into(enc.data, enc.size, &dst);
```

Two things about the printed output matter. First, the 80-byte overhead is always there; a block that compresses to zero transform-plus-entropy bytes still writes 80 bytes to disk. Second, because the model is `RAW` and the residuals are the raw input bytes, the 34 compressed payload bytes come entirely from the transform chain plus the LZ pass. Swapping the model to `DELTA_1D` on the same input would compress the payload further; the chain-compare example below makes that difference concrete.

## Parameter tuning

`ZIGZAG` and `BYTE_SHUFFLE` have no per-stage params. The upstream dtype fully determines their behavior: both read `elem_size` from `tdc_dtype_size(in_dtype)` and route internally. `xform_params[i]` must be `NULL` for these stages, and the encode path treats a non-NULL pointer as a harmless no-op (the vtables mark `params` as ignored).

`QUANTIZE` takes a `tdc_quantize_params` struct with three fields: `scale`, `offset`, and `target`. Encode computes `stored = round((value - offset) * scale)`, and decode inverts the formula to recover the float. `target` must be `I8`, `I16`, or `I32`. Pick `scale` to map the expected value range into the target's representable range, and pick `offset` to shift the zero of the input distribution into the zero of the target. Picking `scale` too small clamps every value to the target's `[min, max]` bounds and throws away the signal; picking it too large leaves the target mostly unused and wastes bits.

[`xforms_quantize_tune.c`](../examples/xforms_quantize_tune.c) runs the same 4,096-element `f64` sinusoid through four `(scale, target)` combinations. The sinusoid is in `[-100, 100]`:

```
input: 32768 bytes (4096 x f64), values in [-100, 100]

  scale=1      target=i8             1426 bytes  ratio=22.98x
  scale=100    target=i16            5150 bytes  ratio=6.36x
  scale=1000   target=i32            7492 bytes  ratio=4.37x
  scale=1e6    target=i32           12467 bytes  ratio=2.63x
```

The first row is the aggressive end: `scale=1` means one integer step per unit, so a sine value of `37.4` stores as `37` and the reconstructed value is exactly `37.0`. Step size is `1.0`, and every input point drops most of its float mantissa. The I8 target fits the range comfortably because values in `[-100, 100]` stay within `[-128, 127]`. The second row holds 0.01 precision and needs I16 to store values up to `+/-10000`. The third row holds 0.001 precision in I32; the fourth holds one-part-per-million precision in I32 and uses up most of the target's entropy on noise that does not reduce well under the downstream ZIGZAG-plus-shuffle-plus-LZ chain.

The tradeoff curve is monotonic: finer precision costs more bytes. The caller's job is to pick the coarsest scale that still meets the downstream task's tolerance. For a temperature sensor logged at 0.01 degrees, `scale=100 target=i16` is the right pick. For satellite reflectance stored at one-part-per-thousand, `scale=1000 target=i16` works if the raw range fits and `target=i32` if it does not. For coordinates stored to the millimeter across hundreds of kilometers, `scale=1000 target=i32` is the default. The tuning work is a single parameter sweep against whatever reconstruction error the caller measures against a reference decode.

One more knob matters on the side: `offset`. Setting `offset` to the mean of the input distribution centers the stored integers around zero, which feeds ZIGZAG a symmetric distribution and lets the byte shuffle do its job. Leaving `offset = 0` on a dataset whose mean is far from zero wastes the sign bit of the target, and the post-quantize bytes have a high-lane constant rather than a low-lane constant.

## Benchmarks

Real numbers for five chain variants on an 8,192-element `i32` input with sign-alternating residuals after DELTA_1D. The source is [`xforms_chain_compare.c`](../examples/xforms_chain_compare.c):

```
input: 32768 bytes (8192 x i32)

  RAW + LZ                                  26761 bytes  ratio=1.22x
  DELTA_1D + LZ                               171 bytes  ratio=191.63x
  DELTA_1D + ZIGZAG + LZ                      171 bytes  ratio=191.63x
  DELTA_1D + BSHUF + LZ                       189 bytes  ratio=173.38x
  DELTA_1D + ZIGZAG + BSHUF + LZ              148 bytes  ratio=221.41x
```

The 80-byte block record header is inside every row, so the transform-plus-entropy payload sits at 91 bytes (`DELTA_1D + LZ`) and 68 bytes (`DELTA_1D + ZIGZAG + BSHUF + LZ`) in the two extremes. The first row is the no-model baseline: LZ sees raw `i32` bytes and finds almost nothing to match on, so the ratio flattens at 1.22 times. That is what "model does the work" looks like in numbers — adding a predictor that knows anything about the input jumps the ratio to 190 times before any transform runs.

Row two and row three print the same size. The input is structured so the deltas, even with sign flips introduced by the encoding, produce a byte distribution LZ already handles well; adding ZIGZAG in front of LZ alone does not move the needle here. The chain where ZIGZAG earns its keep is row five, where it works in tandem with BYTE_SHUFFLE: ZIGZAG flattens the high bytes of the negative residuals into zeros, then BYTE_SHUFFLE groups all those zeros together, and LZ compresses the resulting long zero run. Going from row four (BSHUF alone) to row five is a 22% payload shrink on this input.

Row five versus row two is the right comparison for "is the two-transform chain worth it?". The answer on this input is yes — a 13% payload reduction (171 to 148 bytes) for two extra passes over the buffer. On larger inputs the relative header cost shrinks and the gap widens; on inputs with no sign flips the gap closes. The [performance tuning vignette](../performance/tuning.md) covers the sensitivity curve across input characteristics.

[`xforms_order_matters.c`](../examples/xforms_order_matters.c) isolates the order-of-transforms question. Same 8,192-element `i32` input, DELTA_1D model, LZ entropy; the only thing that changes is whether ZIGZAG or BYTE_SHUFFLE runs first:

```
input: 32768 bytes (8192 x i32), sign-flipping delta residuals

  ZIGZAG -> BSHUF                 188 bytes  ratio=174.30x
  BSHUF -> ZIGZAG                 406 bytes  ratio=80.71x
```

A 2.16-times gap, on the same three passes of computation. Putting BSHUF first is wrong here because the negative residuals still carry `0xFF` prefix bytes at that point; the shuffle faithfully reproduces a noisy top-byte lane. Putting ZIGZAG first fixes the sign encoding before the shuffle sees it, so the top-byte lane is nearly all zeros when the shuffle transposes it into a contiguous run. The asymmetry is a property of how negative numbers live in two's-complement memory, not of any quirk of either backend.

Throughput is dominated by the SSE2 byte-shuffle inner loop on x86_64. On the author's Windows workstation (MSVC Debug build, AVX2-capable CPU), the byte-shuffle forward runs at roughly 3.5 GiB/s for 4-byte elements and 3.1 GiB/s for 8-byte elements; ZIGZAG runs element-wise at about 5 GiB/s across widths. Both numbers are well above the LZ encode throughput (around 700 MiB/s single-threaded), so the transform stage is never the bottleneck for a chain that ends in an entropy coder.

## Edge cases

Four edge cases cover the boundary behavior of the transform chain. The source is [`xforms_edge_cases.c`](../examples/xforms_edge_cases.c); each case runs the same spec (`RAW -> ZIGZAG -> BYTE_SHUFFLE -> LZ`) and encode-then-decode, then memcmps the output against the input:

```
  empty i32            raw=0      encoded=92    memcmp=ok
  single i64           raw=8      encoded=100   memcmp=ok
  all-equal i16        raw=4096   encoded=104   memcmp=ok
  i8 elem_size==1      raw=4096   encoded=115   memcmp=ok
```

The empty block encodes to 92 bytes: 80 for the block record header plus 12 bytes of section metadata (model side meta, transform side meta, and entropy container). The transforms all return `TDC_OK` on zero-byte input without calling their inner loops and without allocating. `tdc_decode_block_into` reconstructs a zero-byte output, which memcmp is defined to accept.

The single-element `i64` block is a pathological case for BYTE_SHUFFLE because an 8-byte element transposed into 8 lanes of 1 byte is byte-identical to the input (each lane is one byte long, so concatenating the lanes reproduces the element). LZ sees eight bytes and finds no matches; the encoded block is 100 bytes, which is the 80-byte header plus a 20-byte irreducible tail. The round trip still passes.

The all-equal `i16` block is the opposite end of the compression spectrum. Every residual is `42`; every delta in LZ is zero; every byte-lane after shuffle is constant. The encoder sees this pattern before any transform runs and sets the zero-residual flag: in the block record, transform and entropy sections are both zero bytes and the decoder reconstructs the residual stream from the flag alone. The 104-byte output is mostly the block record header plus a short "constant `42`" side metadata record emitted by the model. This is one of the "too-clean" inputs the quickstart warns about — the transform and entropy stages did nothing, but because the encoder short-circuited them the ratio is still enormous. On real inputs with any noise at all, the zero-residual path does not trigger and the transforms carry actual work.

The `i8` block tests the `elem_size == 1` short-circuit. BYTE_SHUFFLE sees a one-byte element, falls through to a `memcpy`, and the input and output bytes match bit-for-bit across that stage. ZIGZAG still runs its per-element kernel. The full chain round-trips as expected, and the 115-byte payload is dominated by the LZ tokens over 4,096 small-magnitude bytes. Leaving BYTE_SHUFFLE in the chain for an `i8` stream costs one extra pass over the buffer; the round trip still succeeds, so tests and benchmarks should account for the pass without treating it as a bug.

A fifth edge case worth calling out in prose: a block whose input dtype is rejected by a transform. Handing a `u32` block to `ZIGZAG` returns `TDC_E_DTYPE` at encode time without touching the output buffer. The caller either replaces the transform or drops it from the chain. Nothing in the partial pipeline is left allocated; the output `tdc_buffer` is untouched and the allocator never ran. The pattern is the same one the quickstart's error example shows for model dtype rejection.

## Integration notes

The transform chain composes with every model and every entropy coder in the core set. Pairings that pay off on real workloads:

- `DELTA_1D -> ZIGZAG -> BYTE_SHUFFLE -> LZ` on monotone `i64` timestamps or a sensor counter. The quickstart's 1D taste test runs this chain and lands at a 66-times ratio on 1,024 Unix timestamps.
- `PRED_2D -> ZIGZAG -> BYTE_SHUFFLE -> LZ` on integer rasters. The predictor emits small-magnitude signed residuals with a strong row-to-row correlation; zigzag plus shuffle flattens the high bytes and LZ collapses the zeros. The raster taste test in the quickstart hits 163 times on a 128 by 128 u8 input this way.
- `PRED_3D -> ZIGZAG -> BYTE_SHUFFLE -> LZ` on voxel grids. The three-dimensional predictor emits integer residuals with the same shape; the transforms carry the ratio from the model's raw 4-to-10-times floor up to the 100-to-several-hundred-times range on structured volumes.
- `QUANTIZE -> ZIGZAG -> BYTE_SHUFFLE -> LZ` on float vectors with a known tolerance. `QUANTIZE` is the first step because it converts float to integer; the downstream chain then treats the output like any other integer residual. Note that this is the transform-only pattern and is correct only when the model is `RAW`; inserting `QUANTIZE` between a float-emitting model and its downstream chain is the bug the fused `TDC_MODEL_QUANTIZE_PRED_2D` model exists to fix.

Pairings that do not pay off:

- `RAW -> BYTE_SHUFFLE -> LZ` on an incompressible float stream runs both stages for nothing; the ratio holds near 1.0, and the 80-byte header grows the output past the input size. Drop the chain and ship the raw bytes.
- Feeding an integer stream into `RAW -> QUANTIZE -> ...` fails fast: QUANTIZE rejects integer input at encode time with `TDC_E_DTYPE`.
- `PRED_2D -> QUANTIZE -> ...` on a float raster is the bug the fused `TDC_MODEL_QUANTIZE_PRED_2D` model exists to fix. The predictor emits a float residual, and QUANTIZE then applies its linear formula to a residual whose dynamic range bears no relation to the value range the scale was picked for. Use the fused model.
- `BYTE_SHUFFLE -> ZIGZAG` as the encode order runs the sign flip on already-shuffled bytes, so the top byte lanes still carry `0xFF` prefixes and ZIGZAG no longer corresponds to per-element sign interleaving. The round trip still passes because the inverse reverses the same order, but the compressed size is roughly 2.16 times larger than the correct ZIGZAG-first order on the same input.

Transforms do not participate in the zero-residual fast path. When the encoder detects an all-zero residual stream after the model runs, it sets `TDC_BLOCK_FLAG_ZERO_RESIDUAL` on the block record, writes zero transform-output bytes and zero entropy-output bytes to the payload, and the decoder reconstructs the residual buffer from the flag alone. The transform chain is not called on either side. This is the path that makes synthetic "too-clean" benchmark inputs hit enormous ratios without exercising the chain; picking a benchmark input with any actual noise avoids the trap.

Side metadata from the transforms themselves is zero bytes per backend: ZIGZAG, BYTE_SHUFFLE, and QUANTIZE all serialize their configuration into the block record's per-stage params section through the codec spec, which is written once per block. A block record written with a quantize spec carries the `scale`, `offset`, and `target` in its container header, and the decoder reads them back before calling `QUANTIZE.decode`. None of that is the caller's job; it happens inside `tdc_encode_block` and `tdc_decode_block_into`.

Threading works per block. A single `tdc_codec_spec` can be shared across threads because it is POD; each thread passes its own `tdc_buffer` with its own `realloc_fn` and gets back an independent block record. The transform backends keep no hidden global state, no static scratch, and no thread-local caches. The scratch buffers used during encode and decode are all routed through the caller's allocator, so pinning a thread to an arena is the whole configuration pattern.
