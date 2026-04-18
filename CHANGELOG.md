# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
