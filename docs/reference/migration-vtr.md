# Migration from vectra's old vtr codec

This vignette walks through the migration from vectra's legacy C codec (`vtr_codec.c`, `vtr_encodings.c`, `vtr_compress.c`, and the v4 row-group paths in `vtr1.c`) to tdc's public API. Two audiences should find it useful: the vectra codebase itself, where the migration already shipped and these notes serve as the historical design record, and other R or C libraries that run a similar per-column encoding pipeline and want to adopt tdc as a drop-in backend.

## Status of migration

Vectra completed the rewire to tdc in its 0.5.0 release. Commit `bf4c8db` on the vectra `main` branch deleted the legacy v4 codec files (`vtr_codec.c`, `vtr_encodings.c`, `vtr_compress.c`) and the v4 row-group paths in `vtr1.c`; tdc is now the sole compression path behind every `C_write_vtr` / `C_scan_node` entry point. The rewire shipped as phases P1 through P5 described in `VECTRA_REWIRE.md`, and nothing from the old codec is reachable at runtime. Treat `VECTRA_REWIRE.md` as a historical design document, not a to-do list.

Two caveats matter for anyone reading this vignette in its own migration context. First, the tdc-backed `.vtr` writer intentionally breaks file-level backward compatibility with pre-0.5.0 vectra archives. No tdc reader exists for the old v4 layout; the old and new writers share only the `.vtr` file extension. Second, the rewire closed most of vectra's fast paths but left one open. The v4 writer carried a dict-defer CHARSXP optimization that skipped R's string-interning hash on repeated strings, and the tdc-backed writer currently hits the hash per string. Vectra plans to rebuild that path on top of tdc's `TDC_MODEL_DICT_1D` output once string throughput becomes the hot spot. The regression is flagged in vectra's own `CLAUDE.md` under "Key Design Decisions" and remains the last piece of v4-era behaviour with no direct tdc counterpart.

Full public tdc surface: `include/tdc/types.h` (block, buffer, status codes), `include/tdc/codec.h` (codec spec, one-shot encode/decode), `include/tdc/stream.h` (container-level row-group writer / reader), `include/tdc/format.h` (on-disk block record and container header), `include/tdc/error.h`. Every call swap below cites these headers; the internal tree under `src/` is not part of the contract and is not referenced here.

## Function mapping

The table below pairs every externally visible old-vectra codec entry point with the tdc call that replaced it in 0.5.0. Left column is the function or constant as it appeared in vectra 0.4 and earlier. Right column is the tdc equivalent as of tdc 0.2.x. A downstream project that followed the same four-function shape (encode-one-column, decode-one-column, write-rowgroup, read-rowgroup) can use this table as a direct recipe.

| Vectra v4 (pre-0.5.0)                             | tdc public API                                                         |
|---                                                |---                                                                     |
| `VecArray` (typed column + validity + length)     | `tdc_block` (data + dtype + layout + shape + optional validity)        |
| `VtrEncodedCol` (data, sizes, quantize meta, spatial meta, plane coeffs) | One `tdc_block_record` byte buffer carrying header + side meta + payload |
| `VTR_COMPRESS_NONE / _FAST / _SMALL` level knob   | Hand-picked `tdc_codec_spec` (model + xform chain + entropy chain)      |
| `VTR_ENC_PLAIN`                                   | `TDC_MODEL_RAW`                                                        |
| `VTR_ENC_DELTA` / `VTR_ENC_DIFF`                  | `TDC_MODEL_DELTA_1D`                                                   |
| `VTR_ENC_DICTIONARY`                              | `TDC_MODEL_DICT_1D`                                                    |
| `VTR_ENC_DICT_NUM`                                | `TDC_MODEL_DICT_NUMERIC_1D`                                            |
| `VTR_ENC_SPARSE_ZERO`                             | `TDC_MODEL_SPARSE_ZERO_1D`                                             |
| `VTR_ENC_QUANTIZE`                                | `TDC_XFORM_QUANTIZE` (transform stage; `tdc_quantize_params`)          |
| `VTR_ENC_SPATIAL` (`VtrSpatialSpec.coeffs[]`)     | `TDC_MODEL_PRED_2D` (AUTO/LEFT/UP/AVERAGE/PAETH) or `TDC_MODEL_PLANE_2D` |
| `VTR_COMP_SHUFFLE_LZ`                             | xform chain `[TDC_XFORM_BYTE_SHUFFLE]` + `TDC_ENTROPY_LZ`              |
| `VTR_COMP_SHUFFLE_LZ_HUFF`                        | `[TDC_XFORM_BYTE_SHUFFLE]` + `TDC_ENTROPY_LZ_OPT`                      |
| `VTR_COMP_SHUFFLE_LZ_STREAMS`                     | `[TDC_XFORM_BYTE_SHUFFLE]` + `TDC_ENTROPY_LZ_STREAMS`                  |
| `VTR_COMP_SHUFFLE_FSE`                            | `[TDC_XFORM_BYTE_SHUFFLE]` + `TDC_ENTROPY_FSE`                         |
| `VTR_COMP_SHUFFLE_HUFF`                           | `[TDC_XFORM_BYTE_SHUFFLE]` + `TDC_ENTROPY_HUFFMAN`                     |
| `VecType::VEC_INT64 / _INT32 / _INT16 / _INT8`    | `TDC_DT_I64 / _I32 / _I16 / _I8`                                       |
| `VecType::VEC_DOUBLE`                             | `TDC_DT_F64`                                                           |
| `VecType::VEC_BOOL`                               | `TDC_DT_U8` (one byte per boolean; vectra's convention)                |
| `VecType::VEC_STRING`                             | `TDC_DT_STRING` with a `uint32_t offsets[n+1]` buffer                  |
| `vtr_encode_column_qs(...)` (v4 encode entry)     | `tdc_encode_block(&src, &spec, &out)` (one-shot)                       |
| `vtr_decode_column_qs(...)` (v4 decode)           | `tdc_decode_peek` + `tdc_decode_block_into` (fixed-width)              |
| `vtr_decode_column_qs_string(...)`                | `tdc_decode_block_varlen` (varlen / strings)                           |
| `vtr1_write_rowgroup_qs(...)`                     | `tdc_stream_encoder_open` + N × `write_block` + `end_rowgroup` + `close` |
| `vtr1_read_rowgroup_v4(...)`                      | `tdc_stream_decoder_open` + N × `peek_block` + `read_block`            |
| v4 per-column stats sidecar                       | `tdc_stream_encoder_set_rowgroup_stats` + `tdc_stream_decoder_get_stats` |
| v4 column directory + row-group index             | Trailing row-group index carried in the tdc container header           |
| `VtrQuantizeSpec.scale / offset / target_type`    | `tdc_quantize_params.scale / offset / target`                          |
| `VtrSpatialSpec.predictor` (LEFT/UP/AVG/PAETH/PLANE) | `tdc_pred2d_kind` (AUTO/LEFT/UP/AVERAGE/PAETH); PLANE is its own model id |

The vectra source tree under `vectra/src/vtr_codec_tdc.c` and `vectra/src/vtr1_tdc.c` is the reference implementation of this table. Every row maps to a concrete transition visible in the diff from vectra 0.4 to 0.5.0.

## What the tdc call looks like

The worked example below encodes a 1024-element `i64` ramp column through the tdc public API and decodes it back. It is the smallest possible stand-in for what `vtr_encode_column_qs` / `vtr_decode_column_qs` did inside the old codec, minus every vectra-specific integration detail (R bridge, validity bitmap plumbing, `VecArray` allocation). The full runnable program lives at [`docs/examples/migration_one_column.c`](../examples/migration_one_column.c); the encode-side core is:

```c
tdc_block src = {0};
src.data = values;
src.dtype = TDC_DT_I64;
src.layout = TDC_LAYOUT_VECTOR_1D;
src.shape.rank = 1;
src.shape.dim[0] = (int64_t)N;
tdc_shape_set_contiguous(&src.shape);

tdc_codec_spec spec = {0};
spec.model = TDC_MODEL_DELTA_1D;
spec.xform[0] = TDC_XFORM_ZIGZAG;
spec.xform[1] = TDC_XFORM_BYTE_SHUFFLE;
spec.entropy[0] = TDC_ENTROPY_LZ;

tdc_buffer enc = {0};
enc.realloc_fn = mg_realloc;

tdc_status st = tdc_encode_block(&src, &spec, &enc);
```

Running it on the author's machine:

```
raw     : 8192 bytes (1024 x i64)
encoded : 106 bytes (one tdc_block_record)
decoded : 8192 bytes, match=yes
```

The encoded block is 106 bytes: 80 bytes of tdc block header plus 26 bytes for the residual payload. The residual collapses so hard because a monotone i64 ramp flows through DELTA_1D into a constant-valued residual, then through ZIGZAG and BYTE_SHUFFLE into a long run of zero bytes, then through LZ into a two-byte match sequence. This is the same flow the old `VTR_ENC_DELTA + VTR_COMP_SHUFFLE_LZ` path produced, except the result is one self-describing record instead of a struct of four sibling byte buffers.

Three structural things changed at the call site relative to the v4 codec. The column handle lost its dedicated type: `VecArray` became a generic `tdc_block` with dtype plus layout plus shape. The encoding level and compression level merged into one `tdc_codec_spec` value with four independent axes (model, transform chain, entropy chain, per-stage params). And the multi-field `VtrEncodedCol` collapsed into a single `uint8_t *` byte buffer carrying every field the decoder needs. The decoder has no sidecar it reads; `tdc_decode_peek` on the first 80 bytes of the block is enough to size and allocate the output, and `tdc_decode_block_into` writes directly into the caller-owned buffer without touching the pointer.

## Format compatibility

tdc does not read v4 `.vtr` files, and the tdc-backed vectra writer does not produce a v4-compatible archive. The file extension stays `.vtr` because it is the vectra product identity, but the byte-for-byte contents are a tdc container (`TDC1` magic) with an embedded schema section, per-block tdc block records, and a trailing row-group index. The old column directory, encoding-tag byte, sidecar coefficient table, and per-encoding metadata layout are gone. Vectra's maintainers spelled out the break in `VECTRA_REWIRE.md`, and the project memory note records the same decision: archives written before vectra 0.5.0 need to be re-written through a current version before they round-trip through the new reader. No conversion tool ships with the release and none is planned. P5 deleted the v4 reader alongside the writer so that the codec surface has a single source of truth.

Three design choices baked into tdc from day zero drive the format break. First, block records carry their own description: every field the decoder needs lives in the 80-byte record header, including dtype, layout, shape, model id, transform chain, entropy chain, and section sizes. Second, side metadata for predictor coefficients and dictionary tables lives inside the block record's side-metadata section, never in a sidecar. Third, the container metadata (schema, row-group index, statistics) duplicates into the block records themselves, so a single block record extracted with `dd` is still decodable on its own. None of these three properties held for the v4 layout, and layering them on top would have required a format-version bump plus a branch in every read path for every old encoding tag. P5 shipped precisely to delete that branching surface.

Evolution from here gets simpler. Block-level model, transform, and entropy ids are `uint16_t`, with id ranges reserved up front: `0x0001`–`0x00FF` for core (stable), `0x0100`–`0x01FF` for experimental (semantics may change without a format bump), and `0xFF00`–`0xFFFF` for user-defined backends once a runtime plugin API lands. A tdc 0.2 reader will not decode a file written against a tdc release that ships a new core backend at id `0x0010`, but shipping that backend does not itself bump the format version. tdc 0.x may still break on-disk layout between minor releases while the API settles; the first stable archive version will land when the downstream consumers cross their own 1.0 thresholds. Every current tdc-written archive is a working artifact, not an archival artifact. Treat stored `.vtr` files the same way you treat a `gzip -3` cache, not the way you treat a tar backup.

## When NOT to migrate

Three situations are clear holds.

A C consumer sitting on a large corpus of pre-0.5.0 `.vtr` archives cannot migrate by swapping the library alone: the tdc reader refuses to open those files. The migration path runs in three steps: read with an old vectra, re-write with a current vectra, then switch the reader dependency. The old reader has to stay in the build until the re-write covers every file in the corpus. The cost is transient rather than permanent, but it is the first line item in any migration plan.

The second hold concerns string-heavy write pipelines. A consumer that depends on vectra's old CHARSXP dict-defer fast path will hit the regression flagged in vectra's `CLAUDE.md`: strings still round-trip correctly through tdc's varlen decode, but the per-string intern-hash lookup is back on the write side. If repeated-string write throughput drives the format choice, wait for the dict-defer rebuild on top of `TDC_MODEL_DICT_1D` output. That rebuild is scoped as a vectra-side fast path, not a tdc-side feature. tdc already exposes everything the rebuild will need; the holdup is prioritization, not capability.

Adaptive `SMALL` mode is the third hold. Consumers leaning on `VTR_COMPRESS_SMALL`'s try-all-pick-smallest outer loop should know tdc carries no equivalent `tdc_encode_block_auto` call. Vectra's tdc bridge currently treats `SMALL` identically to `FAST`: one codec spec, no candidate menu. P2a deliberately left the outer loop on the vectra side rather than promoting it into tdc, because picking-smallest is a vectra-product choice rather than a codec-library primitive. If smallest-of-several-specs behaviour matters, either keep the outer loop inside the consumer (write to N buffers, pick the shortest, discard the rest), or wait until a second tdc consumer makes the concrete case for a core auto-spec call.

Beyond those three holds, the migration is ready. Vectra itself passed its full `testthat` suite across the cutover with benchmarks inside the 1-2% ratio band that the project memory notes permit. The codec surface is smaller, the on-disk record is self-describing, and the same `tdc_encode_block` / `tdc_decode_block_into` pair encodes a 1D column, a 2D raster, a 2D stack, and a 3D volume without any dispatch branching leaking into the caller.

## What you lose

The losses are finite and listed once above; they repeat here so the list is one scroll away. The v4 on-disk format is gone: every pre-0.5.0 `.vtr` archive has to be round-tripped through a current vectra before the tdc reader will touch it. The adaptive `SMALL` outer loop is not shipped on the tdc side and is currently treated as `FAST` in the vectra bridge. The CHARSXP dict-defer fast path is removed on the write side pending a rebuild on top of `TDC_MODEL_DICT_1D`. Beyond those three, no v4 feature is silently missing; every encoding tag, predictor selection, and compression level in the v4 knob set has a direct counterpart in the function-mapping table.

## What you gain

The gains fall into three groups. The first is dimensional generality: the same `tdc_encode_block` / `tdc_decode_block_into` pair handles `VECTOR_1D`, `RASTER_2D`, `STACK_2D`, and `VOLUME_3D` blocks, and the model dispatcher branches on semantic layout rather than structural rank. A consumer that previously had parallel code paths for 1D columns and 2D rasters can collapse them into one. The second is the self-describing block record: the decoder needs no schema file, no sidecar, and no version negotiation — a `tdc_decode_peek` on the first 80 bytes of any record returns the dtype, layout, and shape the caller needs to allocate its destination buffer. The third is the reserved id space. Model, transform, and entropy ids are `uint16_t` with the core range carved out up front, which leaves room for a future runtime plugin API to slot in without touching the format version or the call surface used by code written today.
