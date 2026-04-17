# VECTRA_REWIRE.md

Integration plan for rewiring the vectra R package to use tdc as its
compression backend. Tracks task #17 ("Vectra rewire to tdc"). Written
2026-04-18 at the end of the session that completed P1.

## Status

- **P1 DONE** (2026-04-18): tdc re-vendored into `vectra/src/tdc/`.
  Snapshot commit = HEAD of tdc working tree at time of vendoring
  (includes uncommitted `schema_size` rename, Kraft fix, SPARSE_ZERO_1D,
  DICT_NUMERIC_1D, lz_streams repcodes).
  Full vectra testthat suite passed: 41 files, 0 failures, 0 errors.
  `vendor_tdc.sh` and the existing tdc-delegation in `vtr_compress.c` are
  both working.
- **P2–P5 PENDING**. Scope is bigger than initially estimated — see
  "Scope correction" below.

## What's already delegated to tdc

Per `vectra/CLAUDE.md` and `src/vtr_compress.c`:

- **Byte-shuffle**: `tdc_xform_get(TDC_XFORM_BYTE_SHUFFLE)` vtable
- **LZ entropy**: `tdc_entropy_lz_vt` (and the variants LZ_OPT, LZ_STREAMS,
  HUFFMAN, HUFFMAN4, FSE reachable via `tdc_entropy_get`)

These are called as low-level transform/entropy stages. Vectra still
owns:

- Model dispatch (which encoding to pick per column)
- Quantization logic (`VtrQuantizeSpec`)
- Spatial prediction (`VtrSpatialSpec`, plane coefficients)
- Compression-level menu (`VTR_COMPRESS_SMALL` = try several, pick smallest)
- Container format (`.vtr` v4: row-group header, column directory, block bytes)
- R-side fast paths (direct-write into pre-allocated R vectors, dict-blob
  STRSXP interning)

## Target after rewire

Vectra becomes a thin R shim over tdc's public API:

- `tdc_block` replaces `VecArray` views at the encode boundary
- `tdc_codec_spec` replaces per-column `(encoding, compression, quantize,
  spatial)` tuple
- `tdc_stream_encoder_*` / `tdc_stream_decoder_*` replaces
  `vtr1_write_rowgroup_qs` / `vtr1_read_rowgroup_v4`
- tdc's block record replaces `VtrEncodedCol` serialization
- `.vtr` file extension stays (product identity), format becomes tdc
  container

R-specific stays in vectra:

- `collect()` direct-write path (R `INTSXP`/`REALSXP` are pre-allocated by
  R's allocator; avoid malloc+memcpy)
- Dict-blob STRSXP interning (skip per-row `Rf_mkCharLenCE`)
- NA sentinel patching (R's NA conventions, not tdc's validity bitmap)
- Query-plan evaluator (everything under `filter`/`join`/`group_agg` etc.)

## Mapping table: vectra concept → tdc primitive

| Vectra                         | tdc equivalent                                |
|---                             |---                                            |
| `VTR_ENC_PLAIN`                | `TDC_MODEL_RAW`                               |
| `VTR_ENC_DICTIONARY`           | `TDC_MODEL_DICT_1D`                           |
| `VTR_ENC_DELTA`                | `TDC_MODEL_DELTA1D`                           |
| `VTR_ENC_DIFF`                 | `TDC_MODEL_DELTA1D` (same thing, rename only) |
| `VTR_ENC_QUANTIZE`             | `TDC_XFORM_QUANTIZE` (transform stage, not model) |
| `VTR_ENC_SPATIAL`              | `TDC_MODEL_PRED2D` / `TDC_MODEL_PLANE2D`      |
| `VTR_ENC_DICT_NUM`             | `TDC_MODEL_DICT_NUMERIC_1D`                   |
| `VTR_ENC_SPARSE_ZERO`          | `TDC_MODEL_SPARSE_ZERO_1D`                    |
| `VTR_COMP_NONE`                | `TDC_ENTROPY_NONE`                            |
| `VTR_COMP_SHUFFLE_LZ`          | xform chain `[BYTE_SHUFFLE]` + `TDC_ENTROPY_LZ` |
| `VTR_COMP_SHUFFLE_LZ_HUFF`     | xform `[BYTE_SHUFFLE]` + `TDC_ENTROPY_LZ_OPT` (Huffman-aware) |
| `VTR_COMP_SHUFFLE_LZ_STREAMS`  | xform `[BYTE_SHUFFLE]` + `TDC_ENTROPY_LZ_STREAMS` |
| `VTR_COMP_SHUFFLE_FSE`         | xform `[BYTE_SHUFFLE]` + `TDC_ENTROPY_FSE`    |
| `VTR_COMP_SHUFFLE_HUFF`        | xform `[BYTE_SHUFFLE]` + `TDC_ENTROPY_HUFFMAN` |
| `VTR_COMPRESS_FAST`            | single codec spec (default per dtype)         |
| `VTR_COMPRESS_SMALL`           | *no direct equivalent* — see "Open questions" |
| `VecType::VEC_INT64`           | `TDC_DT_I64`                                  |
| `VecType::VEC_DOUBLE`          | `TDC_DT_F64`                                  |
| `VecType::VEC_INT32`           | `TDC_DT_I32`                                  |
| `VecType::VEC_INT16`           | `TDC_DT_I16`                                  |
| `VecType::VEC_INT8`            | `TDC_DT_I8`                                   |
| `VecType::VEC_BOOL`            | `TDC_DT_U8` (byte per bool)                   |
| `VecType::VEC_STRING`          | `TDC_DT_STRING` (+ offsets[])                 |
| `VtrQuantizeSpec.scale/offset` | `tdc_xform_quantize_params` (already carried in side_meta) |
| `VtrSpatialSpec.coeffs[]`      | tdc pred2d side_meta (plane coefficients per tile) |
| row-group write (`vtr1.c`)     | `tdc_stream_encoder_open` + N × `write_block` + `close` |
| row-group read (`vtr1.c`)      | `tdc_stream_decoder_open` + N × `read_block` + `close` |

## Scope correction

Original phase estimate (before reading `vtr_codec.h` fully):
- P2 ≈ 1h to replace one encode function
- P4 ≈ 2h to replace container read/write

Reality after reading the header (337 lines of API surface):
- P2 is actually the biggest phase — vectra's encode produces a
  `VtrEncodedCol` struct with 10+ fields (data, sizes, quantize metadata,
  spatial metadata, coefficients). tdc emits block bytes + side_meta
  bytes. The "bridge" is a non-trivial reshape, not a drop-in.
- P2 and P4 are coupled: you cannot swap what encode returns without
  swapping what the writer consumes, and vice versa.
- `SMALL` candidate menu has no tdc-side equivalent yet (tdc picks one
  codec spec at encode time, doesn't write-and-keep-smallest). Either
  vectra keeps its outer loop, or tdc grows a `tdc_encode_block_auto`.
- Direct-write + dict-blob fast paths must be preserved across the
  rewrite or `collect()` performance regresses (bench_refresh.R would flag
  this immediately).

## Revised phase plan

Each phase is an independent commit gate. Go/no-go at each boundary.

### P2a — side-by-side encode bridge (~2h)

Deliverable: new function `vtr_encode_column_tdc` in a new file
`src/vtr_codec_tdc.c`. Signature:

```c
tdc_status vtr_encode_column_tdc(const VecArray *col,
                                 int64_t n_rows,
                                 int comp_level,
                                 const VtrQuantizeSpec *qspec,
                                 const VtrSpatialSpec  *sspec,
                                 tdc_buffer *block_out);
```

- Builds a `tdc_block` view over the `VecArray` (no copy)
- Picks `tdc_codec_spec` from qspec/sspec heuristics
- Calls `tdc_encode_block` → emits tdc block record bytes
- Returns the bytes; no separate `VtrEncodedCol` metadata struct

Old `vtr_encode_column_qs` stays intact. Nothing on the read or write
path calls the new function yet. Purely additive. Unit-test in isolation.

### P2b — side-by-side decode bridge (~1.5h)

Deliverable: `vtr_decode_column_tdc` + `vtr_decode_column_tdc_into`
matching the existing split. Calls `tdc_decode_block`, copies into
`VecArray` or caller-provided dst. Old `vtr_decode_column*` stays.

### P3 — new row-group writer/reader (~3h)

Deliverable: `src/vtr1_tdc.c` with `vtr1_write_rowgroup_tdc` and
`vtr1_read_rowgroup_tdc`. Uses `tdc_stream_encoder_*` / `tdc_stream_decoder_*`
for the container. Schema is written via `tdc_stream_encoder_write_schema`.

The `.vtr` file header gains a new magic tag so the reader can tell v4
vs tdc-backed. v4 reader stays functional during migration.

Gate: roundtrip a representative data.frame through the new path, verify
byte-exact equality on read-back.

### P4 — R bridge switch (~1h)

Deliverable: `C_write_vtr` routes to `vtr1_write_rowgroup_tdc`,
`C_scan_node` routes to `vtr1_read_rowgroup_tdc`. Keep v4 reader for
backward reads if opting in via an env var or argument (to be decided —
per tdc's "no format versioning" memory, probably NOT, just clean cut).

Gate: full testthat suite passes, benchmarks within 2% of baseline.

### P5 — delete old code (~1h)

Remove `vtr_codec.c`, `vtr_encodings.c`, `vtr_compress.c`, `vtr1.c`'s v4
paths. Delete `test-compression.R` expectations that tested v4 tags.
Update `CLAUDE.md`.

## Open questions (resolve before P2a)

1. **SMALL mode**: keep vectra's outer try-all-then-pick-smallest loop,
   or promote it into tdc as `tdc_encode_block_auto`? *Recommendation*:
   keep it vectra-side for now. It's a vectra-product feature. Revisit
   if a second tdc consumer wants the same behavior.

2. **Per-block codec spec on disk**: tdc's block record already carries
   model/xform/entropy ids. vectra's old encoding tag byte becomes
   redundant. Just drop it.

3. **Validity bitmap**: tdc has `tdc_block.validity`. vectra has its own
   separate NA bitmap write path. Do we pipe validity through tdc or
   keep it beside? *Recommendation*: pipe through. tdc already supports
   it as a pass-through field.

4. **String offsets**: tdc expects `uint32_t offsets[n+1]`; vectra's
   `VecArray` uses `int64_t offsets[n+1]`. Needs a narrow-cast at the
   bridge. Validate `max(offsets) <= UINT32_MAX` (tdc limit per block).

5. **Compression-level knob**: the `comp_level` int needs to translate
   to a tdc `codec_spec`. Proposed mapping:
   - `NONE` → RAW + NONE
   - `FAST` → auto-model + `[BYTE_SHUFFLE, ZIGZAG]` + LZ_STREAMS
   - `SMALL` → outer loop (see Q1), each candidate a different spec

## Benchmarks to protect

From `vectra/CLAUDE.md` and the bench_*.R scripts:

- `bench_refresh.R` — primary read/write throughput vs fst/parquet/zstd
- `bench_zstd.R` / `bench_vs_zstd.R` — compression ratio comparison
- `bench_compress_final.R` — per-encoding path coverage
- `bench_tiff.R` — spatial predictor on raster data (tests `VTR_ENC_SPATIAL`)

Run before P2a to establish baseline numbers; re-run at end of P4.
Acceptable regression: 1-2% (per user memory "Ratio primary, ~1-2%
tradeoff OK for speed").

## Files touched per phase (estimate)

| Phase | New files | Modified files | Deleted files |
|---    |---        |---             |---            |
| P2a   | 1 (vtr_codec_tdc.c) | 0 | 0 |
| P2b   | 0 (extend P2a file) | 0 | 0 |
| P3    | 1 (vtr1_tdc.c) | 0 | 0 |
| P4    | 0 | 2 (init.c, r_bridge_io.c) | 0 |
| P5    | 0 | 1 (CLAUDE.md) | 3 (vtr_codec.c, vtr_encodings.c, vtr_compress.c) + v4 code in vtr1.c |

## How to resume

New session should:

1. `cd vectra && git status` to confirm P1 vendoring is committed.
2. Read this file + `vectra/CLAUDE.md` + `vectra/src/vtr_codec.h`.
3. Resolve the 5 open questions above, then start P2a.
4. Each phase gets its own commit. Don't bundle.

tdc side is stable — nothing more to do in tdc for this task unless an
integration gap surfaces (e.g., if tdc is missing a model vectra needs).
