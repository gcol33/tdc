# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Changed

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
