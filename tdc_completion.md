# tdc completion plan

This document lays out the path from tdc's current state (feature-complete
codec library) to its final role as the canonical compressed column format
for vectra.

Two phases:

    Phase 1  "C"   vectra calls tdc_encode_block / tdc_decode_block
                    per column chunk.  VTR1 keeps the container.

    Phase 2  "B"   tdc owns the container.  VTR1 disappears.

Phase 1 validates the codec in production.  Phase 2 is the format
migration.  Neither phase can be skipped safely.


---

## Phase 1 — tdc as codec backend  (Path C)

### Goal

Replace vectra's hand-wired byte-shuffle + LZ2 calls with a single
`tdc_encode_block` / `tdc_decode_block` call per column chunk.  VTR1
continues to own the container (schema, stats, row groups, column
chunk headers).

### What changes in vectra

**`vtr_codec.c` (the column encode/decode layer):**

Currently wires individual vtable calls:

```
byte_shuffle()   →  tdc_xform_get(TDC_XFORM_BYTE_SHUFFLE)->encode()
lz2_compress()   →  tdc_entropy_get(TDC_ENTROPY_LZ2)->encode()
```

Replace with one call per column:

```
tdc_codec_spec spec = { .model = TDC_MODEL_RAW,
                        .xform = { TDC_XFORM_BYTE_SHUFFLE },
                        .entropy = { TDC_ENTROPY_LZ2 } };
tdc_encode_block(&blk, &spec, &out);
```

For spatial columns: `.model = TDC_MODEL_PRED_2D` (or PLANE_2D).
For delta columns:  `.model = TDC_MODEL_DELTA_1D`.
For dict columns:   `.model = TDC_MODEL_DICT_1D`.
For quantized:      add `TDC_XFORM_QUANTIZE` to the chain.

**Decode path:**

Currently parses VTR1 encoding/compression tags and calls the
corresponding vtable stages manually.  Replace with:

```
tdc_decode_block(encoded_bytes, encoded_size, &dst_block);
```

The block record is self-describing, so the decoder doesn't need the
VTR1 encoding tag — it reads the model/xform/entropy ids from the
80-byte block header.

**VTR1 on-disk format change:**

The column chunk payload changes from raw bytes (byte-shuffled + LZ2)
to a tdc block record (80-byte header + side_meta + xform_params +
payload + validity).  This is a VTR format version bump (v5 → v6).

The VTR1 column chunk header still carries:
  - encoding tag (map to tdc model id)
  - compression tag (replaced by "TDC_BLOCK" sentinel, or removed)
  - data_size (now = total block record size)
  - uncompressed_size (now = tdc block's uncompressed_size)

Spatial metadata (predictor, nx, ny, tile coefficients) migrates INTO
the tdc block record's side_meta section — it's already there for
PRED_2D and PLANE_2D models.  VTR1 stops writing it separately.

### What does NOT change

- VTR1 container header (schema, column names, annotations)
- Row group index (offsets, stats)
- Column statistics (min/max)
- Sorted column detection
- Streaming write API (vtr_write.c)
- Parallel row group reader
- Validity bitmap write (stays before column chunk, same position)

### Migration checklist

- [ ] Map VTR1 encoding tags to tdc_codec_spec (RAW, DELTA, DICT,
      PRED_2D, PLANE_2D, QUANTIZE chains)
- [ ] Replace vtr_encode_column_qs() internals with tdc_encode_block
- [ ] Replace vtr_decode_column() internals with tdc_decode_block
- [ ] Bump VTR format version (v6)
- [ ] Preserve v4/v5 read path for backwards compatibility
- [ ] Run vectra's full test suite + round-trip checks
- [ ] Benchmark: confirm no regression vs direct vtable calls

### Risks

- tdc_decode_block currently uses a libc realloc shim (no caller-
  supplied allocator on decode).  This is fine for correctness but
  prevents zero-copy decode into R-allocated SEXP buffers.  Acceptable
  for Phase 1; must be fixed before Phase 2.

- Spatial metadata lives in tdc's side_meta now, not in VTR1's inline
  chunk metadata.  The VTR1 read path for v6 must know to look inside
  the block record, not after the chunk header.  Clean but requires
  careful version gating.


---

## Phase 2 — tdc as the storage layer  (Path B)

### Goal

Eliminate VTR1.  tdc owns the full on-disk format: container header,
column schema, statistics, row groups, streaming write, random access.
vectra becomes a pure consumer.

### What TDC1 must gain

The gap between TDC1 and VTR1 is well-defined.  Every feature below
exists in VTR1 today and must be added to TDC1 before the cutover.

#### 1. Column schema

VTR1 header carries a column schema: per-column name (string),
dtype (u8), and annotation (string).  TDC1 has no schema concept.

**Design:**  Add an optional schema section after the 64-byte container
header.  Proposal:

```
offset 64:  u16 n_columns
per column: u16 name_len, name_len bytes (UTF-8)
            u8  dtype
            u16 ann_len,  ann_len bytes (UTF-8)
```

The container header gains a `schema_size` field (u32) so the reader
can skip it or validate it.  Schema is written once at open time.
Heterogeneous containers (n_columns = 0) remain valid.

Container version bumps to 2.

#### 2. Row group concept

VTR1 row groups are batches of N rows, each containing one encoded
chunk per column.  TDC1 has flat blocks with no grouping.

**Design:**  A "row group" in TDC1 becomes a contiguous run of
n_columns blocks (one per column), written in column order.  The
trailing index gains a row-group-level structure:

```
per row group:
    u64 offset          (file offset of first block in this group)
    u64 n_rows          (rows in this group)
    u16 n_cols          (columns; must match schema unless heterogeneous)
    per column:
        u64 block_offset
        u64 block_total
```

This replaces the current flat `tdc_index_entry_v1` array.  The index
version bumps.  The block-level index is still derivable (it's the
union of all per-column entries).

#### 3. Column statistics (min/max)

VTR1 writes per-column, per-row-group statistics: min/max for integers,
floats, booleans, and 8-byte string prefixes.

**Design:**  Statistics become a section of each row-group index entry:

```
per column in row group:
    u8  has_stats
    16 bytes  min (type-dependent)
    16 bytes  max (type-dependent)
```

Statistics are computed at encode time and written to the index.
This enables zone-map filtering and sorted-column detection without
scanning block data.

#### 4. Decode allocator

tdc_decode_block currently wraps libc realloc internally (documented
limitation).  For zero-copy decode into R SEXP buffers, the decode
path must accept a caller-supplied allocator.

**Design:**  Add `tdc_decode_block_ex` that takes a `tdc_buffer *`
for scratch allocation, matching the encode side.  The original
`tdc_decode_block` becomes a thin wrapper.  This is a frozen header
change — requires approval.

#### 5. Parallel row-group reader

VTR1 opens independent FILE handles per thread for parallel row-group
reads.  TDC1's streaming decoder is single-threaded.

**Design:**  With the row-group index, each row group has a known
file offset.  The caller opens N file handles (or uses pread), creates
N independent `tdc_stream_decoder` instances (each seeked to a
different row group), and decodes in parallel.  tdc does NOT manage
threads — it provides the index and the seekable decoder.  vectra
(or any consumer) owns the thread pool.

This requires no new tdc API beyond what already exists: seek_block +
read_block on independent decoder instances.

#### 6. Streaming write with batching

VTR1's vtr_write_node_batched_qs accumulates rows in builders, flushes
when >= batch_size.

**Design:**  This is a vectra-level concern, not a tdc concern.  tdc
provides `tdc_stream_encoder_write_block`.  vectra accumulates rows
and calls write_block when a row group is ready.  The batching logic
stays in vectra (or moves to a thin helper library).

tdc should NOT absorb row-level buffering.  That couples tdc to a
specific data representation (R vectors, Arrow arrays, etc.).  The
boundary is: vectra hands tdc complete blocks, tdc writes them.

#### 7. Atomic file writes

VTR1 writes to a temp file (`.~writing`) and renames on success.

**Design:**  This is an I/O-layer concern.  tdc's callback-based I/O
already supports this: the caller wraps a temp file behind the write
callback and renames after `tdc_stream_encoder_close`.  No tdc change
needed.

### Container layout (Phase 2, TDC1 v2)

```
[container header]       64 bytes (v2: +schema_size field)
[schema section]         variable (column names, dtypes, annotations)
[row group 0]
    [block 0: column 0]  80-byte header + sections
    [block 1: column 1]  80-byte header + sections
    ...
[row group 1]
    ...
[trailing index]         row-group-level entries with per-column offsets
                         + optional statistics
```

### Migration sequence

1. Add `tdc_decode_block_ex` (allocator on decode)
2. Add schema section to container header (container v2)
3. Add row-group-level index with per-column entries
4. Add statistics to index entries
5. Update streaming encoder: accept schema at open, write schema
   section, accumulate row-group-level index
6. Update streaming decoder: parse schema, load row-group index,
   expose column-level seek
7. Add `tdc_stream_decoder_read_schema` and
   `tdc_stream_decoder_seek_rowgroup` to public API
8. Rewrite vectra's read/write paths to use tdc streaming API
9. Drop VTR1 code (vtr1.h, vtr1.c, vtr_codec.c column-level logic)
10. Deprecate .vtr format; new default extension .tdc

### What tdc explicitly does NOT absorb

- **Thread pool management.**  vectra owns OpenMP / thread dispatch.
  tdc provides thread-safe per-instance decoders.
- **Row-level buffering.**  vectra decides batch sizes and accumulates
  rows.  tdc sees complete blocks.
- **R type conversion.**  SEXP ↔ tdc_block mapping stays in vectra's
  r_bridge layer.
- **Query planning / predicate pushdown.**  vectra reads statistics
  from the tdc index and decides which row groups to skip.  tdc
  exposes stats, does not interpret them.


---

## Feature matrix: what exists, what's needed

| Feature                     | v0 (done) | Phase 1 | Phase 2 |
|-----------------------------|-----------|---------|---------|
| Block encode/decode         | done      | used    | used    |
| Streaming encode/decode     | done      | —       | used    |
| 7 models                   | done      | used    | used    |
| 4 transforms               | done      | used    | used    |
| 6 entropy coders            | done      | used    | used    |
| Self-describing blocks      | done      | used    | used    |
| Container header            | done      | —       | extended|
| Trailing block index        | done      | —       | replaced|
| Column schema               | —         | —       | new     |
| Row group index             | —         | —       | new     |
| Column statistics           | —         | —       | new     |
| Decode allocator            | —         | —       | new     |
| Plugin API                  | —         | —       | future  |


---

## Decision log

- 2026-04-09: v1 priorities set: streaming > container I/O > plugin API.
- 2026-04-09: Streaming encode/decode API implemented and tested.
- 2026-04-09: Integration path chosen: Phase 1 (C) then Phase 2 (B).
  Rationale: C validates the codec in production without format risk;
  B requires schema/stats/index extensions that should be informed by
  real usage.
