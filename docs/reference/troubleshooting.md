# Troubleshooting

Every tdc entry point returns a `tdc_status`. Eleven codes cover the surface (`TDC_OK` plus ten errors), and a single lookup in `tdc_strerror` maps each to a one-line string. This page walks through each code in turn, then traces two realistic failure patterns from the symptom the caller sees down to the line of C that fixes them.

The code used for every row in the status-code table lives at [`docs/examples/troubleshooting_statuses.c`](../examples/troubleshooting_statuses.c). Running that program on tdc 0.2.0-dev prints:

```
OK: raw encode             -> status=0 (ok)
INVAL: no realloc_fn       -> status=1 (invalid argument)
LAYOUT: 2D to DELTA_1D     -> status=5 (layout not accepted by stage)
DTYPE: I32 to DELTA2_1D    -> status=4 (dtype not accepted by stage)
SHAPE: negative dim        -> status=6 (shape rank or dims invalid)
UNSUPPORTED: bad model id  -> status=3 (unsupported model/transform/entropy id)
CORRUPT: short peek        -> status=8 (on-disk header or payload failed validation)
VERSION: future version    -> status=9 (container or block version not understood)
INVAL: NULL dst->data      -> status=1 (invalid argument)
```

Each entry below pins one status to a concrete return site inside `src/`, names the caller-visible symptom, gives the most common cause, and states the shortest fix. The codes are stable: future 0.x minor bumps keep the numbering intact.

## Status code reference

Every code a public entry point can return. The enum lives in `include/tdc/types.h`; the error strings come from `include/tdc/error.h` via `tdc_strerror`. One line per code names the return site; the body says how to unblock it.

### `TDC_OK = 0`

**Symptom.** The call succeeded. Output buffers hold the produced bytes. On a decode, `dst->data` holds the reconstructed values and `memcmp` against the original input is exact. Nothing to fix.

**Cause.** The encoder consumed the block and produced a full record; the round trip through decode reconstructed every element, and the result matched the header's `uncompressed_size`.

**Fix.** None. Note that `TDC_OK == 0`, so the common `if (st) { ... handle error ... }` pattern works without a named constant.

### `TDC_E_INVAL = 1`

**Symptom.** The call returns 1 immediately and produces no output. Output pointers and sizes stay at their entry values; the caller's buffer has not grown.

**Likely cause.** A required pointer is NULL, a buffer is missing its `realloc_fn`, or a sanity invariant failed. The two most common shapes are a `tdc_buffer` with a zero `realloc_fn` handed to `tdc_encode_block`, and a `tdc_decode_block_into` call where `dst->data` is NULL for a non-empty record. Both return `TDC_E_INVAL` before the pipeline runs (`src/api/decode_impl.c` line 100).

**Fix.** Install the allocator before passing the buffer.

```c
tdc_buffer enc = {0};
enc.realloc_fn = my_realloc;        /* never leave this NULL */
```

For `tdc_decode_block_into`, peek first, then allocate:

```c
tdc_block meta = {0}; size_t need = 0;
tdc_decode_peek(src, src_size, &meta, &need);
tdc_block dst = meta;
dst.data = my_realloc(NULL, NULL, need);
tdc_decode_block_into(src, src_size, &dst);
```

### `TDC_E_NOMEM = 2`

**Symptom.** The call returns 2 partway through the pipeline. The library has already freed its internal scratch through the same allocator, and no partial output reaches the caller.

**Likely cause.** The caller's `realloc_fn` returned NULL under memory pressure, or an arena allocator hit its capacity. Sites that can report this include `src/format/schema.c` (column descriptor arrays) and `src/api/decode_varlen.c` (offsets + byte-heap allocation for decoded strings).

**Fix.** Size the arena for the block's worst case before calling, or fall back to libc `realloc` for that call. A back-of-the-envelope upper bound is `2 * n_elems * dtype_size + 80` for the output record, plus `3 * n_elems * dtype_size` of scratch for the ping-pong buffers on decode.

### `TDC_E_UNSUPPORTED = 3`

**Symptom.** The encode or decode returns 3 before any data moves. On encode this usually means a `model`, `xform`, or `entropy` id in the `tdc_codec_spec` has no registered backend.

**Likely cause.** The id falls in the user-reserved range (`0xFF00..0xFFFF`) with no corresponding `case` in `src/core/registry.c`, or it is a v0 enum value whose backend the build left out (e.g. `TDC_ENTROPY_DEFLATE` without `-DTDC_HAVE_ZLIB=ON`). The decode-side site sits at `src/api/decode_impl.c` line 104: `if (!model_vt) return TDC_E_UNSUPPORTED;`.

**Fix.** Either switch to a shipped id (the core range `0x0001..0x00FF`), or rebuild with the optional backend enabled:

```bash
cmake -B build -DTDC_HAVE_ZLIB=ON
```

### `TDC_E_DTYPE = 4`

**Symptom.** Encode returns 4 before any residual leaves the model, or decode returns 4 when the pre-populated `dst->dtype` disagrees with the record header.

**Likely cause.** The spec picked a model or transform that does not accept the input dtype. `DELTA2_1D` and `FPC_1D` are float-only; `SPARSE_ZERO_1D` accepts numeric dtypes but not I8/U8; `ZIGZAG` only accepts signed integers. Each stage checks its `accepted_dtypes` bitmask and returns `TDC_E_DTYPE` on miss.

**Fix.** Match the model to the dtype. Signed integer vectors route through `DELTA_1D`; float vectors want `DELTA2_1D` or `FPC_1D` instead. The full acceptance table lives in [Models](../backends/models.md) and [Transforms](../backends/transforms.md).

### `TDC_E_LAYOUT = 5`

**Symptom.** Encode returns 5 before any residual leaves the model. Decode returns 5 when the pre-populated `dst->layout` disagrees with the record header (`src/api/decode_impl.c` line 90).

**Likely cause.** The spec's model targets a layout the block does not carry. Every model carries an accepted-layouts bitmask; `DELTA_1D` takes `VECTOR_1D` only, `PRED_2D` takes `RASTER_2D` only, `PRED_3D` takes `VOLUME_3D` only. A 2D raster handed to `DELTA_1D` hits `src/model/delta1d.c` line 718.

**Fix.** Set `src->layout` to one the model accepts, or pick a model that accepts the layout the block already has. The quickstart walks the four layouts end to end.

### `TDC_E_SHAPE = 6`

**Symptom.** The call returns 6 during validation. On encode, the block never reaches the model. On decode, the destination's shape did not match the header's `dim[]`.

**Likely cause.** A negative dimension, a rank that does not match the layout (e.g. `VECTOR_1D` with `rank = 2`), or `dst->shape.dim[i]` on decode that differs from the record's stored dim. The encode gate lives in `tdc_block_validate` (`src/core/block.c` lines 75-82). On the decode side, `src/api/decode_impl.c` lines 91-93 compare each `dst->shape.dim[i]` against the record and reject the first disagreement.

**Fix.** Fill the shape from the record on decode instead of hand-setting it:

```c
tdc_block dst = meta;           /* meta came from tdc_decode_peek */
dst.data = my_alloc(need);
tdc_decode_block_into(src, src_size, &dst);
```

On encode, call `tdc_block_validate(&src)` as a debug-build assertion before `tdc_encode_block`. A negative dim usually means an off-by-one in the caller's size math, not a tdc bug.

### `TDC_E_BUF_TOO_SMALL = 7`

**Symptom.** A parser returned 7. The destination buffer was not large enough to hold the parsed data.

**Likely cause.** The only return site in v0 is `tdc_stats_parse` (`src/format/stats.c` line 292), reached via the stream-decoder row-group stats path. It fires when the serialized stats section is shorter than `n_cols * TDC_STATS_ENTRY_SIZE`. The one-shot `tdc_encode_block` and `tdc_decode_block` entry points never return this code in v0.

**Fix.** Size the destination buffer from the header before the parse:

```c
size_t required = (size_t)n_cols * TDC_STATS_ENTRY_SIZE;
uint8_t *buf = my_alloc(required);
tdc_stats_parse(buf, required, n_cols, out);
```

If the error surfaces from a streaming read, the upstream index is likely truncated. Re-read the container from the beginning and confirm `index_size` matches the trailing-index entry count.

### `TDC_E_CORRUPT = 8`

**Symptom.** Decode returns 8 before any residual bytes are produced. The record's header failed structural validation.

**Likely cause.** The 80-byte header's magic does not match `TDC_BLOCK_MAGIC` (`BLKB`), the advertised section sizes do not fit in `src_size`, or `validity_size` disagrees with the flag bit. The container-header analog is `src/format/header.c` line 44: `if (h->magic != TDC_CONTAINER_MAGIC) return TDC_E_CORRUPT;`. Block-record cases are scattered through `src/api/decode_impl.c` lines 57-85.

**Fix.** Re-read the bytes from the source. If the corruption is reproducible, run the block through `tdc_inspect` (`tools/tdc_inspect.c`) to locate the first invalid field. The most common real-world trigger is a truncated download; checksum the bytes before retrying.

### `TDC_E_VERSION = 9`

**Symptom.** Decode returns 9 with the container or block record held out of reach. The header's magic was correct but the version was not one the linked library understands.

**Likely cause.** The record was produced by a future tdc that bumped `TDC_BLOCK_VERSION` or `TDC_CONTAINER_VERSION`. Current v0 values are `TDC_CONTAINER_VERSION = 1` and `TDC_BLOCK_VERSION = 2`. The two return sites are `src/format/header.c` line 45 (containers) and `src/format/block_record.c` line 96 (per-block).

**Fix.** Upgrade tdc to a library version that supports the record's version, or have the producer write a version this reader understands. There is no in-place migration; v0 does not read future versions.

### `TDC_E_IO = 10`

**Symptom.** A streaming read or write returned 10. The underlying file or socket failed.

**Likely cause.** The stream decoder's read hook returned a short count or a hard error. The two call sites are `src/api/stream_decode.c` lines 94 and 109, wrapping the caller's `read_fn` callback. One-shot `tdc_encode_block` / `tdc_decode_block` never touch I/O and never return this code.

**Fix.** Check the underlying descriptor, confirm the file length matches the container's advertised `index_offset + index_size`, and retry. If the read hook is buffered, flush it before the next call.

## Common failure patterns

Two concrete miswirings, from symptom to fix. Both are trivial to reproduce, both are common in first-week tdc integrations, and both are fully explained by the status the library returns.

### Pattern 1: layout mismatch on encode

A consumer pipelines 2D sensor frames through an encoder written initially for 1D timeseries. The first frame trips the following trace:

```
status = 5 (layout not accepted by stage)
```

That is `TDC_E_LAYOUT`, returned by `delta1d_encode` (`src/model/delta1d.c` line 718) the moment it sees a `tdc_block` whose `layout` is `TDC_LAYOUT_RASTER_2D` instead of `TDC_LAYOUT_VECTOR_1D`. The encoder never allocates, never writes a partial record, and leaves the output buffer pointer unchanged.

Two things went wrong at the call site. The `tdc_codec_spec` still carries `spec.model = TDC_MODEL_DELTA_1D`, copied from the 1D path. The block construction switched to `TDC_LAYOUT_RASTER_2D` when the shape grew to rank 2, but the spec did not. The library's dispatcher branches on layout, never on rank, so this mismatch is detected before any residuals are computed.

Fix. Pick a model whose `accepted_layouts` bitmask covers `RASTER_2D`:

```c
tdc_codec_spec spec = {0};
spec.model       = TDC_MODEL_PRED_2D;         /* accepts RASTER_2D */
spec.xform[0]    = TDC_XFORM_ZIGZAG;
spec.xform[1]    = TDC_XFORM_BYTE_SHUFFLE;
spec.entropy[0]  = TDC_ENTROPY_LZ;
```

A 64x64 `int32` frame that initially returned `TDC_E_LAYOUT` at 16 KiB of untouched input now round-trips through a 1.2 KiB record in [`integration_error.c`](../examples/integration_error.c). Between the two calls the only change is the model id; the block struct, the allocator, and the output buffer are identical.

A small library-design note about this failure mode: `tdc_block_validate` does not check the spec. It only checks whether the block is structurally coherent on its own (rank vs layout, dtype id in range, `offsets` NULL-ness). The mismatch between a valid block and a model that does not accept its layout is caught by the model dispatcher, not by block validation, because only the dispatcher knows which layouts each backend accepts. Running `tdc_block_validate` on every source block in debug builds is still defensible; it catches the off-by-one on `rank` that otherwise waits until the model itself fails with `TDC_E_SHAPE`.

### Pattern 2: wrong destination buffer size on decode

A consumer hand-rolls its decode path to avoid a spurious malloc. The destination dtype is hardcoded to `TDC_DT_F32`, which matched the producer when the pipeline was written. A later change migrated the producer to `TDC_DT_F16`. The first decode after the migration now prints:

```
status = 4 (dtype not accepted by stage)
```

That is `TDC_E_DTYPE`, returned at `src/api/decode_impl.c` line 89 when `dst->dtype != hdr.dtype`. No bytes have been written into `dst->data`; the caller's buffer is untouched. A step before that check, `tdc_decode_peek` would have reported the record's dtype as `TDC_DT_F16` and its required byte count as `n_elems * 2` rather than `n_elems * 4`.

The fix has two parts. Source the dtype from the record, not from caller assumptions:

```c
tdc_block meta = {0};
size_t need = 0;
tdc_status st = tdc_decode_peek(src, src_size, &meta, &need);
if (st != TDC_OK) return st;

tdc_block dst = meta;                /* dtype, layout, shape copied from the record */
dst.data = my_alloc(need);           /* sized from the peek */

st = tdc_decode_block_into(src, src_size, &dst);
```

The pre-populated fields are validated against the header, not overwritten. A caller that continues to pre-fill `dtype` manually can still do so, as long as it matches the record; the peek-then-copy idiom shown above sidesteps the problem entirely by filling the fields from the record in one step.

If the caller cannot use `tdc_decode_peek` (for example, they are reading a memory-mapped region and want one less helper call), the defensive move is to treat any of `TDC_E_DTYPE`, `TDC_E_LAYOUT`, or `TDC_E_SHAPE` on return as "destination pre-population is wrong; peek, reallocate if needed, and retry." The three-code fallback covers every pre-population bug without the caller needing to inspect the header by hand.

Both patterns share the same shape. The library detects the mismatch early, returns a code that names the stage it failed in, and leaves every caller-owned buffer in the state it had on entry. A call that returns `TDC_E_LAYOUT`, `TDC_E_DTYPE`, or `TDC_E_SHAPE` is trivially retryable: fix the parameter named by the code, call again.

## Debug build tips

A debug build of tdc is just the default CMake target with `-DCMAKE_BUILD_TYPE=Debug`. On MSVC this adds `/Zi` and `/Od`; on gcc or clang it adds `-g` and `-O0`. None of the warning flags change. The library runs with the same compile-time invariants either way, so a failure reproduced in debug is a failure in release.

Two non-default flags help when investigating a status code:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
```

AddressSanitizer catches out-of-bounds writes in caller-owned destinations (the most common way a user-supplied `realloc_fn` corrupts tdc's ping-pong buffers). UndefinedBehaviorSanitizer flags integer overflow in the shape product before it reaches `TDC_E_SHAPE`. Both are compatible with MSVC (AddressSanitizer ships with Visual Studio 17+).

To dump a block record without decoding, run `tdc_inspect` on the raw bytes. The tool prints every field of the 80-byte header, the side-metadata section, and the payload size. It reads only the header, so a record that returns `TDC_E_CORRUPT` during decode still prints cleanly if the corruption is in the payload:

```bash
build/Debug/tdc_inspect.exe path/to/block.bin
```

Three other cheap instrumentation steps help in the field. Log the return of every `tdc_encode_block` and `tdc_decode_block` with `tdc_strerror(st)` so logs carry the human-readable code next to the numeric one. Hash the source bytes before handing them to tdc and again after a decode round-trip; a `memcmp` mismatch reduces the search space to one stage. Capture the `tdc_codec_spec` alongside the first byte of each block: when reproducing a `TDC_E_CORRUPT` or `TDC_E_VERSION`, the spec plus the first 80 bytes are enough to reconstruct the failing call without the full container.

## Reporting a bug

A bug report against tdc is easiest to triage when it carries the five items below. Any one missing turns a five-minute fix into an hour of guessing.

**tdc version.** Either the release tag (`0.1.0`), the commit hash from `src/core/version.c`, or the output of `git describe --tags` from the build tree.

**The `tdc_codec_spec`.** Include the model id, the transform chain (all four slots), the entropy chain (all four slots), and the values of any params pointers (`tdc_pred2d_params::kind`, `tdc_quantize_params::scale`, etc.). A one-line `printf` at the top of the failing encode is enough.

**Input shape and dtype.** `dtype`, `layout`, `shape.rank`, and `shape.dim[]`. Include the byte count (`n_elems * dtype_size`) so the report is self-contained.

**The status code and its `tdc_strerror` string.** The numeric code alone is ambiguous across patch bumps; the string is stable.

**A minimal repro.** Ideally a single `.c` file under 80 lines that reproduces the failure with `cc -I include repro.c libtdc.a -o repro`. The examples under `docs/examples/` are a good template. If the input is a real file, attach or link the first 4 KiB as `xxd` output so the bytes survive copy/paste.

Open the issue at [github.com/gcol33/tdc/issues](https://github.com/gcol33/tdc/issues). The maintainer reads every new issue; a clean report with the five items above usually gets a reproduction within a day.
