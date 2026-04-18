# Quickstart

tdc is a typed, dimensional compression library. The public API is C11. This walkthrough starts from a raw buffer, builds a `tdc_block`, runs one round trip, then widens the aperture to show every layout and every decision point the caller has to make. Every code snippet on this page maps to a working program under `docs/examples/` that compiles against the public library and runs end to end; the byte counts printed below come from running those programs on the author's machine at tdc 0.2.0-dev.

Four orthogonal stages make up the library. A **model** reads a typed block and emits residuals plus side metadata. A **transform chain** rewrites those residuals into a flat byte stream friendlier to entropy coding (zigzag, byte shuffle, optional quantization). An **entropy coder** compresses the resulting bytes. A **block record** glues the three outputs together as a self-describing unit that survives being copied, concatenated, or extracted with `dd`. Each stage exposes one enum id and one optional params struct. Every layer stays in its lane: a transform sees only bytes and dtypes, a model sees only the input block, an entropy coder sees only a flat byte buffer. That separation is what lets a single pipeline encode a sensor timeseries, a satellite tile, a frame stack, and a voxel grid without special cases.

Where does compression come from, given this split? Almost always from the model. A well-chosen model turns correlated data into small, near-uniform residuals; the transform and entropy stages then carry those residuals to disk at near-information-theoretic cost. When the caller picks the wrong model (running `PRED_2D` on a 1D ramp, say), the transforms and entropy have nothing to exploit and the ratio collapses to roughly 1.0×. This vignette makes that tradeoff concrete by running the same block through several specs and watching the byte counts move.

This vignette covers the public API only. Everything we call lives under `include/tdc/`; the internal `src/` tree never appears here. The single umbrella header `tdc.h` pulls in every public sub-header; a caller embedding tdc into a consumer project only needs that one include.

## The block abstraction

The `tdc_block` struct is the universal carrier for input and output data. It describes a pointer, a dtype, a semantic layout, and a shape; nothing else. No compression policy fields, no scratch pointers, no codec hints. A caller constructs one, hands it to `tdc_encode_block`, and receives the same struct type back from the decoder with the `data` pointer filled in.

The interplay of three fields is what makes a block dimensional rather than merely N-D:

- `dtype` is a numeric type id (`TDC_DT_I8` through `TDC_DT_F64`, plus `TDC_DT_F16` and `TDC_DT_STRING`). It tells tdc how wide each element is and whether the residual path can use signed-integer arithmetic.
- `shape.rank` and `shape.dim[]` give the structural rank in memory. A 4×4 matrix has rank 2; an 8×64×64 frame stack has rank 3.
- `layout` tells the dispatcher *how* to traverse the block. A `RASTER_2D` is rank 2 and the model walks it as a single raster. A `STACK_2D` is rank 3 but the model walks each first-axis slice as a raster. A `VOLUME_3D` is also rank 3, and the model walks it with true 3D neighborhoods.

The critical rule, baked into `src/core/registry.c` and stated once in `types.h`, is that model dispatch branches on `layout`, never on `rank` alone. A frame stack (`STACK_2D`) and a voxel grid (`VOLUME_3D`) are structurally identical in memory; they differ in the neighborhood the model is allowed to consult. The caller owns this choice. tdc never tries to infer it.

Here is the block for a tiny 1D integer column:

```c
int64_t data[256];
/* fill data ... */

tdc_block src = {0};
src.data   = data;
src.dtype  = TDC_DT_I64;
src.layout = TDC_LAYOUT_VECTOR_1D;
src.shape.rank   = 1;
src.shape.dim[0] = 256;
tdc_shape_set_contiguous(&src.shape);
```

`tdc_shape_set_contiguous` fills `stride[]` assuming row-major contiguous memory. A caller with a pre-strided buffer can set `stride[]` directly and skip the helper. The block takes no ownership of `data`: the pointer must stay valid across the encode or decode call, and tdc never frees it on return.

Two fields on the block are left NULL in this example. The `validity` pointer optionally points at a packed NA bitmap (1 bit per element, LSB-first). The v0 decoder treats it as pass-through bytes, so setting it on encode preserves the mask on decode without interpreting it. The `offsets` pointer is required when `dtype == TDC_DT_STRING` (variable-width strings, where `offsets[i+1] - offsets[i]` gives the byte length of string `i`) and must be NULL for every other dtype. `tdc_block_validate` enforces the NULL-ness rule; a caller that sets `offsets` on an `I64` block gets `TDC_E_DTYPE` back immediately. Strings are out of scope for this quickstart; the dict1d and varlen decode paths are covered in their own backend walkthroughs.

One last rule about the shape struct: `rank` caps at `TDC_MAX_RANK = 3`. Everything tdc encodes in v0 is rank 1, 2, or 3. Higher-dimensional inputs are the consumer's responsibility to tile into one of the four layouts. Staying within that grid is what keeps the model dispatch table finite and the per-backend benchmarks tractable.

## Minimal encode/decode

The one-shot path is two functions: `tdc_encode_block` for the round trip in, `tdc_decode_block_into` (paired with `tdc_decode_peek`) for the round trip out. The full example lives at [`docs/examples/quickstart_roundtrip.c`](examples/quickstart_roundtrip.c). Running it:

```
input  : 2048 bytes (256 x i64)
encoded: 2128 bytes (block record, incl. 80-byte header)
decoded: 2048 bytes, memcmp == source: yes
```

The first surprise is that the encoded block is 80 bytes bigger than the raw input. We picked `tdc_codec_spec_raw()`, which is identity model plus passthrough entropy. The output here is *not* compressed; it is the input wrapped in a block record header. Every block, even a raw one, carries its own 80-byte header describing dtype, layout, shape, and section sizes, because that record is the one and only unit tdc writes to disk. We will compress in the next section.

Encoding starts with a spec and a growable output buffer. The buffer never allocates on its own; it calls through a function pointer the caller installed:

```c
static void *qs_realloc(void *user, void *ptr, size_t n) {
    (void)user;
    if (n == 0) { free(ptr); return NULL; }
    return realloc(ptr, n);
}

tdc_codec_spec spec = tdc_codec_spec_raw();
tdc_buffer enc = {0};
enc.realloc_fn = qs_realloc;

tdc_status st = tdc_encode_block(&src, &spec, &enc);
/* st is TDC_OK; enc.data / enc.size hold the block record */
```

The `(user, ptr, n)` convention is the same one the library uses internally. Three cases cover every call: a NULL pointer with nonzero size allocates, a non-NULL pointer with zero size frees, and a non-NULL pointer with a new size grows (possibly moving the buffer). Vectra, the primary consumer, plugs its own arena allocator here; stdlib `realloc`/`free` also works for tools and tests.

Decoding is symmetric. The caller first peeks the record to discover dtype, layout, shape, and the destination byte count, then allocates the output buffer and calls the zero-copy decode:

```c
tdc_block meta = {0};
size_t need = 0;
tdc_decode_peek(enc.data, enc.size, &meta, &need);

void *dst_data = qs_realloc(NULL, NULL, need);
tdc_block dst = meta;
dst.data = dst_data;
tdc_decode_block_into(enc.data, enc.size, &dst);
```

Peek reads only the 80-byte header, so the caller can size the output buffer before committing any scratch memory. `tdc_decode_block_into` then writes directly into `dst->data` without touching the pointer. It never reallocates, and it never frees. The caller owns every byte of `dst->data` before and after the call. That invariant is what makes tdc safe to embed in R (writing into an SEXP buffer) or behind mmap (writing into a mapped file region).

Internal scratch memory for entropy and transform stages still allocates during decode. By default it flows through libc `realloc`; callers who want every byte under their allocator can use `tdc_decode_block_ex` instead, which takes a `tdc_buffer` whose `realloc_fn` handles both the output and the scratch. On return (success or failure), every internal allocation has been freed via the same function pointer. No buffer leaks are possible without the caller's own allocator leaking them first.

The full round trip prints identical byte counts in and out, and `memcmp` against the original confirms the reconstruction is exact. On any failure the return status names the cause, the destination pointer is unchanged, and the caller can either retry with a different spec or propagate the error upstream. tdc never aborts, never logs, and never retains state between calls.

## Choosing a codec spec

A `tdc_codec_spec` is a POD bundle: one model id, up to four transforms, up to four entropy coders, plus optional pointers to per-stage params. Both the transform chain and the entropy chain run left to right on encode and right to left on decode, and a zero entry terminates each chain. That is the whole mental model.

The interesting question is what to put in the chain for a given block. [`docs/examples/quickstart_codec_compare.c`](examples/quickstart_codec_compare.c) encodes the same 4,096-element `i32` near-linear ramp three ways:

```
input: 16384 bytes (4096 x i32)

  RAW                               16464 bytes
  DELTA_1D + LZ                       111 bytes
  DELTA + ZIGZAG + BSHUF + LZ         118 bytes
```

`RAW` is the passthrough spec we just saw. It is 80 bytes over the raw size because of the block record header. Switching the model to `TDC_MODEL_DELTA_1D` replaces each element with `x[i] - x[i-1]`, which for this input is a near-constant stream of threes. Running `TDC_ENTROPY_LZ` across that residual stream collapses it to 111 bytes — a 148× ratio, almost entirely from the model.

Adding `TDC_XFORM_ZIGZAG` and `TDC_XFORM_BYTE_SHUFFLE` makes the output slightly larger here (111 → 118 bytes). The two transforms are designed for residual streams whose low-order bytes still carry noise after the model pass; on a deterministic ramp they introduce a small fixed overhead with nothing to exploit. We leave them in the example on purpose: they are the right default for residuals that *do* have per-byte structure (float deltas after DELTA2_1D, 16-bit sensor traces with jitter), and the measurement shows the gate at which the tradeoff flips.

The minimum decision the caller makes is the model. Picking `TDC_MODEL_RAW` means "the transforms and the entropy coder carry the whole load." Picking `TDC_MODEL_DELTA_1D` on a monotone integer column means "the entropy coder only has to do whatever the model didn't." Every subsequent choice is a refinement. The one-per-backend codec walkthroughs linked at the end of this page cover the accepted dtypes, accepted layouts, and the benchmarked tradeoff curves for each id.

Two details make the spec smaller in practice than the header suggests. First, params pointers are optional: most model ids accept NULL and use reasonable defaults (`PRED_2D` defaults to `TDC_PRED2D_AUTO`, which samples the block and picks the best kind; `ENTROPY_LZ` defaults to its documented level). A caller that just wants "compress this sensibly" can set the model id, leave everything else zero, and the chain-terminator convention does the rest. Second, the encoder has a whole-pipeline fast path: when the model emits an all-zero residual (the tri-affine `VOLUME_3D` example below hits this), the block record sets `TDC_BLOCK_FLAG_ZERO_RESIDUAL` and stores zero transform and entropy bytes. The decoder sees the flag and reconstructs a zero-filled residual of the declared size before handing control to the model's inverse. No transform or entropy code runs at all on that path.

## Layout taste tests

Each layout has a dedicated vignette; here we run one end-to-end example per layout so the four-way split feels concrete.

### VECTOR_1D

[`docs/examples/quickstart_vector1d.c`](examples/quickstart_vector1d.c) encodes 1,024 monotonically increasing Unix timestamps (1-second cadence with small jitter) using `DELTA_1D` plus `ZIGZAG`, `BYTE_SHUFFLE`, and `LZ`. The timestamps themselves are 8 bytes each; the deltas are all in the 1-to-5-second range and fit easily in one byte.

```
VECTOR_1D  raw=8192 encoded=124 ratio=66.06x
           memcmp == source: yes
```

This is the shape `VECTOR_1D` was built for: a single column of numeric values with strong along-axis correlation. `DELTA_1D` converts the column into small residuals, `ZIGZAG` maps signed deltas into unsigned byte-friendly integers (negative deltas would otherwise look like 0xFFFF… noise), `BYTE_SHUFFLE` groups the top seven bytes of every 8-byte element together (they are all zero here, post-zigzag) and puts the one interesting byte lane contiguously, and `LZ` eats the long runs of zeros. The full walkthrough of these stacking choices lives in [models / delta1d](backends/models.md).

### RASTER_2D

[`docs/examples/quickstart_raster2d.c`](examples/quickstart_raster2d.c) encodes a 128×128 `uint8` image whose value at `(x, y)` is the low byte of `x + y`. The sum-of-coordinates function is linear in both axes, so `TDC_PRED2D_PAETH` predicts each pixel exactly from its three causal neighbors. The residual is zero everywhere except the top row and the left column.

```
RASTER_2D  raw=16384 encoded=100 ratio=163.84x
           memcmp == source: yes
```

The model id here is `TDC_MODEL_PRED_2D` and the parameters go through a `tdc_pred2d_params` struct. Only one field is non-trivial: `kind`, which picks among `LEFT`, `UP`, `AVERAGE`, `PAETH`, or `AUTO`. `AUTO` samples the block and picks the kind with the smallest residual sum. On this input any kind would hit near-zero, but `PAETH` is the right default for most natural rasters because it reduces to `LEFT` or `UP` on the gradient axes automatically. The backend walkthrough at [models / pred2d](backends/models.md) shows how each kind responds to real-world image inputs.

A separate `PLANE_2D` model (`TDC_MODEL_PLANE_2D`) fits a least-squares plane per tile and stores the three plane coefficients as side metadata. It is the right pick for smooth topographic rasters like DEM tiles, where a single local plane explains 32×32 pixels at a time. Float rasters have their own fused model. `TDC_MODEL_QUANTIZE_PRED_2D` quantizes to a narrow integer target and runs the predictor on the integer raster in one step. Running `PRED_2D` first and `XFORM_QUANTIZE` afterward would be wrong for floats: the predictor emits a float residual, and the subsequent quantization of that residual produces garbage values the decoder cannot reverse.

### STACK_2D

[`docs/examples/quickstart_stack2d.c`](examples/quickstart_stack2d.c) encodes 8 frames of a 64×64 `uint8` raster, each frame a translated version of the same linear pattern. The model is `TDC_MODEL_STACK_2D` with `inter_slice = 1`, which tells the encoder to subtract the previous slice before running the in-plane predictor.

```
STACK_2D   raw=32768 encoded=120 ratio=273.07x
           memcmp == source: yes
```

A frame stack is rank 3 in memory but `STACK_2D` in semantics: tdc walks the first axis as "which frame", and inside each frame it runs the same 2D predictor it would run on a standalone raster. The `inter_slice` flag is the one knob that distinguishes `STACK_2D` from "a batch of independent `RASTER_2D` blocks". When frames are nearly identical (this example, medical imaging, video), subtracting the previous slice first collapses most of the signal into the first frame's residual and leaves essentially nothing for the others. Setting `inter_slice = 0` falls back to per-frame prediction; on independent frames that would be the right pick.

### VOLUME_3D

[`docs/examples/quickstart_volume3d.c`](examples/quickstart_volume3d.c) encodes a 12×16×20 `int32` voxel grid holding a tri-affine field `f(x, y, z) = 3x + 5y + 7z + 17`. The model is `TDC_MODEL_PRED_3D` with kind `GRAD3D`: a trilinear predictor exact on any function linear in all three axes.

```
VOLUME_3D  raw=15360 encoded=127 ratio=120.94x
           memcmp == source: yes
```

The `GRAD3D` kind evaluates `pred = a + b + c - ab - ac - bc + abc` where the letters name the seven causal neighbors (left, up, front, and the four diagonal seeds). On a tri-affine field every interior voxel matches exactly; the encoder picks up the zero-residual flag and stores only the seed voxels on the three boundary faces. `TDC_PRED3D_AUTO` on arbitrary volumes samples the prefix and chooses among `LEFT`, `UP`, `FRONT`, `AVG3`, `GRAD3D`, or `PAETH3D`. The deep walkthrough at [models / pred3d](backends/models.md) covers the variant tradeoffs and why `PAETH3D` is the safer default on natural volumes with sharp edges.

## Error handling

Every tdc entry point returns a `tdc_status`, and `tdc_strerror` turns it into a one-line string suitable for logging. Status codes are contiguous small integers (`TDC_OK = 0`, `TDC_E_INVAL = 1`, up through `TDC_E_IO = 10`), so a single switch statement in caller code covers the full surface. The full list lives in `types.h`.

Here is a deliberately wrong call: we hand a 2D raster block to `DELTA_1D`, which only accepts `VECTOR_1D`. See [`docs/examples/quickstart_error.c`](examples/quickstart_error.c):

```
status = 5 (layout not accepted by stage)
```

The encoder checked the block against the model's `accepted_layouts` bitmask and returned `TDC_E_LAYOUT` without touching the output buffer. The same pattern catches dtype mismatches (`TDC_E_DTYPE`), shape errors (`TDC_E_SHAPE`), buffer-too-small (`TDC_E_BUF_TOO_SMALL`), and decode-time corruption (`TDC_E_CORRUPT`). None of the errors leave partial output: the buffer pointer is unchanged, the caller is free to retry with a different spec, and there is nothing to clean up beyond the allocator calls the caller owns. Mapping a status code to an action belongs to the caller; tdc never prints, never `abort`s, and never retains state across return.

## What to read next

Three next stops, ordered by what most callers reach for first:

- The backend pages: [Models](backends/models.md), [Transforms](backends/transforms.md), and [Entropy](backends/entropy.md). One worked example per id, with accepted dtypes, accepted layouts, benchmark numbers on real inputs, edge cases (empty, single element, degenerate runs), integration notes, and one explicit "when NOT to reach for this backend" section per page.
- The [format spec](format/on-disk.md) annotates every field of the 80-byte block record against a hex dump captured from this quickstart.
- [Custom backends](extending/custom-backends.md): adding a new id to the static registry.

Callers planning to vendor tdc into an R package or a CMake subproject will want the [integration guide](integration.md), which covers the allocator passing pattern, Makevars-vs-CMake options, and the ABI promises across 0.x minor bumps.
