# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **New format generation: 'TDCF' container, 'TDBK' block record, both at
  version 1.** A block's side metadata now goes through the block's entropy
  chain, prefixed by its raw size, instead of being stored verbatim; a string
  dictionary (DICT_1D), which for a column of distinct values holds every byte
  of the column, is compressed like the payload. DICT_1D's dictionary is
  stored packed, a LEB128 length per entry in place of the u32 offsets table,
  and unpacked by the driver, so the model and every consumer of decoded side
  metadata keep the offsets form. A grown-in-place container is marked by the
  new `TDC_CONTAINER_FLAG_WIDENED` rather than by a second container version.
  Writers emit only the new generation. The legacy generation ('TDC1'
  container v1/v2, 'BLKB' block v2) is still read, decided per record through
  `tdc_container_header_is_legacy()` / `tdc_block_record_is_legacy()`;
  growing a legacy container rewrites its header in the new generation. New
  helpers: `tdc_container_header_check_generation()`,
  `tdc_container_header_is_widened()`, `tdc_block_magic_known()`.
- The entropy chain (stage-size table + final bytes) has one implementation,
  `driver_entropy_chain_encode` / `driver_entropy_chain_decode`, shared by the
  payload and the side metadata.

- **Widening and extending reuse the superseded index's space.** Both now
  write from the end of the container's live data (the last block or schema
  byte it refers to, which is where its trailing index starts unless a pass
  was interrupted) instead of past the index, so a container grown by any
  number of passes holds one index
  and is as long as one written in a single pass (byte-identical from the
  schema to the end of the index for row-group appends). Each pass used to
  leave its predecessor's index behind, so file size grew quadratically with
  the number of passes.

  The index the header names is still never overwritten: a byte-identical
  copy of the original index is kept ahead of the write cursor (moved with
  doubling room, O(log) times per pass) and the header is repointed at it, so
  a process that dies at any write leaves the container from before or after
  the pass, and the next pass writes over what an interrupted one left.
  `tdc_stream_encoder_abort` now writes the original index and header back. Both config structs gain an optional `relocated_fn` hook that
  reports the container end after each move, for callers that keep a record
  at the end of the file. The bytes past the index after close are the
  caller's to truncate.

### Added

- **Column widening**: `tdc_stream_encoder_open_widen` +
  `tdc_stream_encoder_widen_block` append new columns to an existing
  heterogeneous container without reading or rewriting its body. Cost is
  proportional to the appended columns plus the rebuilt schema and index,
  not to the container's size.

  This needs a new container version, `TDC_CONTAINER_VERSION_WIDENED` (2).
  The schema section of a v1 container sits at offset 64 immediately before
  the first block record and cannot grow in place, so a widen relocates the
  schema to the tail and records `schema_offset` / `blocks_start` in the
  header. Those two fields occupy the `global_dim[3]` slots, now a union
  discriminated by the `HETEROGENEOUS` flag: a heterogeneous container has
  no global shape, so the storage was dead. Containers that are never
  widened stay v1 and byte-identical.

  Everything a widen writes goes past the existing trailing index rather
  than over it, and the 64-byte header is patched last, so a crash before
  that patch leaves the pre-widen container fully readable. The superseded
  index remains as a gap in the blocks region, which makes a widened
  container random-access only -- `tdc_stream_decoder_peek_block` refuses a
  sequential walk on v2 rather than silently reporting end-of-blocks at the
  gap.

- `tdc_stream_encoder_write_record` writes a block record the caller
  already produced with `tdc_encode_block`. The container is byte-identical
  to the one `write_block` builds; splitting encode from write lets a
  caller encode the columns of a row group concurrently and write them in
  order (#1).

### Changed

- LZ encode at the default level (and levels 2-8, and LZ_STREAMS levels
  2-4, which share the greedy parser) is 7-9x faster on
  incompressible and weakly compressible input (#1). The hash-chain parser
  skips ahead on miss streaks instead of walking the chain at every byte,
  the match finder no longer clears its per-position chain array, and the
  single-stream LZ encoder drops matches that save no bytes. On 131,072-row
  f64 blocks through RAW + BYTE_SHUFFLE + LZ (`bench/bench_lz_levels.c`):
  standard normal 42 -> 387 MB/s, smooth surface 55 -> 424 MB/s, both at
  unchanged ratio; normal rounded to 2 decimals goes from ratio 1.000 to
  0.881 (level 1: 0.982). The bitstream is unchanged; only the parse
  differs, so existing records decode as before.

- The row-group index wire format now has a single implementation
  (`src/format/rowgroup.c`). The stream encoder's serializer, the stream
  decoder's parser, and `format/rowgroup.c` were three separate hand-
  maintained copies of one layout; the third could not parse any
  stats-bearing index (it required the byte the stats size lives in to be
  zero), so it would have rejected every container the library actually
  writes. Encoder and decoder both route through the shared codec, which
  also gained the stats block, `_pad` and `n_cols` validation, and an
  allocation bound taken before any allocation.

### Fixed

- `tdc_container_header_validate` accepted no real container: it required
  `index_size == n_blocks * TDC_INDEX_ENTRY_SIZE`, an invariant from the
  superseded flat index. A row-group index's size depends on per-group
  column counts and on whether stats are attached, so it is not derivable
  from `n_blocks`; only the presence relationship is checkable from the
  header alone. (The function has no callers, so nothing was failing in
  practice.)

## [0.1.0] - 2026-04-18

Initial public release. Core pipeline is complete for 1D/2D/stack/3D blocks;
public headers under `include/tdc/` are stable for the v0 contract.

### Added
- Four-layer codec pipeline — model, transform chain, entropy, storage — with
  independent structural rank and semantic layout (`VECTOR_1D`, `RASTER_2D`,
  `STACK_2D`, `VOLUME_3D`).
- Streaming encoder/decoder API (`tdc_stream_encoder_*`, `tdc_stream_decoder_*`)
  and push/pull variants for zero-copy pipelines.
- Self-describing block records — every field needed to decode a block lives
  in the record header; blocks survive copy and extraction.
- Per-row-group column statistics wired into the container index, enabling
  predicate pushdown at decode time.
- Model backends: `RAW`, `DELTA_1D`, `DELTA2_1D`, `DICT_1D`, `DICT_NUMERIC`,
  `SPARSE_ZERO_1D`, `FPC_1D`, `PRED_2D` (LEFT/UP/AVG/PAETH/PLANE), `PLANE_2D`,
  `STACK_2D`, `PRED_3D`, `PRED_3D_FLOAT`, `QUANTIZE_PRED_2D` composite with
  I64/U64 kernels.
- Transforms: `QUANTIZE`, `ZIGZAG`, `BYTE_SHUFFLE`, `BITSHUFFLE`.
- Entropy backends: `NONE`, `LZ` (native, separated streams), `HUFFMAN`,
  `HUFFMAN4`, `FSE`, `LANE`, and `DEFLATE` (zlib, optional behind
  `TDC_HAVE_ZLIB`).
- `tdc_decode_block_varlen` for variable-width string column decode.
- Experimental plugin ABI for user-defined stages under the reserved
  `0xFF00-0xFFFF` enum id range.

### Known limitations
- Public API is stable for 0.x but may break between minor versions until the
  first real consumer (vectra) has fully rewired onto tdc.
- Big-endian targets are not supported; a future port would require a format
  version bump.
- No public plugin registration API in 0.1.0 — adding a backend requires
  recompilation against the static registry.

[Unreleased]: https://github.com/gcol33/tdc/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gcol33/tdc/releases/tag/v0.1.0
