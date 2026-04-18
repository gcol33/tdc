# On-disk format

tdc writes two on-disk objects: a 64-byte **container header** that describes
a file of blocks, and an 80-byte **block record** header that precedes every
block's side metadata and payload. Both are fixed-width C structs with a
natural alignment that matches the on-disk byte order, and both live in the
frozen public header `include/tdc/format.h`. The structs are declared with
`_Static_assert` gates for `sizeof(tdc_container_header) == 64` and
`sizeof(tdc_block_record) == 80`, so a layout skew on any target fails the
compile rather than silently shipping a broken reader.

Every code sample on this page compiles against the public API. Real hex
output is captured from running `format_hexdump` and `format_peek_header`
under `docs/examples/`. Running `cmake -B build -DTDC_BUILD_EXAMPLES=ON`,
followed by `cmake --build build --target format_hexdump format_peek_header`,
produces the executables; the byte-for-byte tables below come from those runs.

## Scope and guarantees

tdc's on-disk contract covers three artefacts: the magic numbers at the top
of each record, the fixed 64-byte container header, and the fixed 80-byte
block record header plus its four trailing sections (side metadata, xform
params TLV, payload, validity). Everything a decoder needs to reconstruct
a block lives inside its 80-byte record: model id, full transform chain,
entropy chain, dtype, layout, rank, shape, every section size, and the
uncompressed residual byte count. A record is **self-describing** by design,
so it survives being copied between containers, extracted with `dd`, or
concatenated from multiple producers.

Fixed **little-endian** on disk, documented in `types.h` and reiterated
under the struct declarations in `format.h`. The supported targets
(x86_64, aarch64) are both little-endian. A big-endian port is the one
change that would force a format version bump; no endian branches live in
the encoder or decoder.

The record header is **fixed size**. Exactly 80 bytes, declared at compile
time, readable with a single `pread`/`memcpy` of `TDC_BLOCK_HEADER_SIZE`
with no growth logic. Adding a field requires a version bump; re-packing
within the existing 80 bytes (using the `_reserved0` / `_reserved_pad`
slots) does not. The container header follows the same discipline at 64
bytes.

The format carries two version numbers. `TDC_CONTAINER_VERSION = 1` stamps
the container; `TDC_BLOCK_VERSION = 2` stamps every block record. They are
independent by design so a later block-layout change does not force a
rewrite of existing container indexes. The version numbers are written by
the encoder and compared against the compile-time constants during
`tdc_container_header_validate` and `tdc_block_record_validate`; mismatches
return `TDC_E_VERSION`, distinct from `TDC_E_CORRUPT`.

tdc is pre-release. Until a real external consumer ships (vectra's rewire
lands, or another downstream integrates), the format may change in place
without bumping version. The rules in this file describe the current
layout, not a committed long-term ABI. The `project_no_versioning_during_prototype`
note in the project memory governs evolution policy until that first
downstream consumer exists.

## Container header

The container header is written once at the start of a container file,
followed by a sequence of block records, followed by a trailing index
table at `index_offset` of length `index_size`. The header carries global
shape / dtype / layout for homogeneous containers, or zeros with the
`TDC_CONTAINER_FLAG_HETEROGENEOUS` flag set when each block is independent
(vectra's row-group case). The `schema_size` field gives the byte count of
a serialized schema section that sits immediately after the 64-byte header
and before the first block.

Field order follows natural C alignment so the struct compiles to exactly
64 bytes on x86_64 and aarch64 without `#pragma pack`. `schema_size` fits
into the 4-byte gap before `global_dim[3]`, which keeps the int64 array at
an 8-aligned offset.

| Offset | Size | Field           | Type   | Meaning                                                    |
|-------:|-----:|-----------------|--------|------------------------------------------------------------|
|      0 |    4 | magic           | u32    | `TDC_CONTAINER_MAGIC` = `0x31434454` (`'TDC1'` LE)         |
|      4 |    2 | version         | u16    | `TDC_CONTAINER_VERSION` = 1                                |
|      6 |    2 | flags           | u16    | bit 0 `HETEROGENEOUS`, bit 1 `HAS_STATS`                   |
|      8 |    8 | n_blocks        | u64    | number of block records in the container                   |
|     16 |    8 | index_offset    | u64    | absolute file offset of trailing block-index table         |
|     24 |    8 | index_size      | u64    | byte length of the trailing block-index table              |
|     32 |    1 | global_dtype    | u8     | `tdc_dtype`, 0 if heterogeneous                            |
|     33 |    1 | global_layout   | u8     | `tdc_layout`, 0 if heterogeneous                           |
|     34 |    1 | global_rank     | u8     | 0..3, 0 if heterogeneous                                   |
|     35 |    1 | _reserved0      | u8     | must be 0                                                  |
|     36 |    4 | schema_size     | u32    | serialized schema bytes immediately after this header      |
|     40 |   24 | global_dim[3]   | i64[3] | zeros if heterogeneous                                     |

`TDC_CONTAINER_FLAG_HETEROGENEOUS` is bit 0. When set, per-block dtype and
layout vary; the global fields must read as zero. `TDC_CONTAINER_FLAG_HAS_STATS`
is bit 1 and signals that at least one row group's trailing index entry
carries per-column min/max statistics.

The trailing block-index is a contiguous array of 16-byte entries:

| Offset | Size | Field         | Meaning                                    |
|-------:|-----:|---------------|--------------------------------------------|
|      0 |    8 | block_offset  | absolute file offset of the block record   |
|      8 |    8 | block_total   | header + side + xform + payload + validity |

`TDC_INDEX_ENTRY_SIZE` is 16. A container with `n_blocks` records has an
index table of `n_blocks * 16` bytes; `index_size` equals that product.

`tdc_container_header_validate` checks the magic, the version, that the
dtype / layout / rank trio is either all-populated or all-zero with the
heterogeneous flag set, and that `index_offset + index_size` does not
overflow a `uint64_t`. It does not follow `index_offset` or decompress
anything; corruption inside a single block does not fail the container
validator. The block validators handle the per-record checks.

## Block record

The block record is the unit tdc encodes and decodes. Everything the
decoder needs for a single block, including its transform chain and
entropy chain, is inside the 80-byte header plus the four trailing
sections. The container header is informational; a block extracted with
`dd` and handed to `tdc_decode_block_into` reconstructs cleanly without
any reference to it.

Section order on disk, each section immediately following the previous:

```
  [ header          (80 bytes)                 ]
  [ side_meta       (side_meta_size bytes)     ]
  [ xform_params    (xform_params_size bytes)  ]
  [ payload         (payload_size bytes)       ]
  [ validity        (validity_size bytes)      ]
```

Every section but the header is optional. An all-RAW spec writes an 80-byte
header and a payload equal to `n_elems * dtype_size(dtype)`. A block whose
model predicts the data exactly sets `TDC_BLOCK_FLAG_ZERO_RESIDUAL` in the
flags, skips the transform and entropy chains, and stores zero bytes for
payload and xform params. The decoder reconstructs a zero-filled residual
of size `uncompressed_size` before handing it to the model.

| Offset | Size | Field             | Type   | Meaning                                                       |
|-------:|-----:|-------------------|--------|---------------------------------------------------------------|
|      0 |    4 | magic             | u32    | `TDC_BLOCK_MAGIC` = `0x424B4C42` (`'BLKB'` LE)                |
|      4 |    2 | version           | u16    | `TDC_BLOCK_VERSION` = 2                                       |
|      6 |    2 | flags             | u16    | `HAS_VALIDITY`, `LOSSY`, `ZERO_RESIDUAL`                      |
|      8 |    2 | model_id          | u16    | `tdc_model_id`                                                |
|     10 |    8 | xform_ids[4]      | u16[4] | `tdc_xform_id`, 0 terminates the chain                        |
|     18 |    8 | entropy_ids[4]    | u16[4] | `tdc_entropy_id`, 0 terminates the chain                      |
|     26 |    1 | dtype             | u8     | `tdc_dtype`                                                   |
|     27 |    1 | layout            | u8     | `tdc_layout`                                                  |
|     28 |    1 | rank              | u8     | 1..3                                                          |
|     29 |    1 | _reserved0        | u8     | must be 0                                                     |
|     30 |    2 | _reserved_pad     | u16    | alignment padding for dim[3]                                  |
|     32 |   24 | dim[3]            | i64[3] | block shape; trailing unused entries zero                     |
|     56 |    8 | uncompressed_size | u64    | residual bytes **before** entropy stage                       |
|     64 |    4 | side_meta_size    | u32    | model side metadata byte count                                |
|     68 |    4 | payload_size      | u32    | entropy-compressed payload bytes                              |
|     72 |    4 | xform_params_size | u32    | TLV section byte count (0 if no per-slot xform params)        |
|     76 |    4 | validity_size     | u32    | validity bitmap bytes (`ceil(n_elems / 8)` if `HAS_VALIDITY`) |

Three size fields distinguish the pipeline stages. `uncompressed_size` is
the byte length of the residual stream coming out of the model, *before*
the transform chain runs. `payload_size` is the byte length *after* the
transform chain and the entropy chain. On the decode side, the entropy
chain runs right-to-left and grows the payload back up to the
post-transform intermediate size; the transform chain then runs right-to-left
and grows that back up to `uncompressed_size`. Keeping both counts on disk
means the decode hot path never guesses at output sizes.

`side_meta_size` carries model-specific bytes that parametrize the
decoder. The contents vary by model id. `RAW` and `DELTA_1D` write nothing
into this section, so the size is zero. `PRED_2D` and `PRED_3D` store a
single byte holding the resolved predictor kind after `AUTO` is dispatched
at encode time, so their size is 1.
`TDC_MODEL_PLANE_2D` stores a u16 tile size, a u32 tile count, and three
i32 coefficients per tile. `TDC_MODEL_DICT_1D` and the numeric-dict variant
store a packed dictionary. The block header does not describe the internal
layout of this section; the model id implies the format.

`xform_params_size` is a TLV section placed between the side metadata and
the payload. Layout, repeating until the byte budget is consumed:

```
  u16 slot_index      (0 .. TDC_MAX_TRANSFORMS - 1)
  u16 blob_length
  blob_length bytes   (transform-specific payload, little-endian)
```

Transforms without on-disk parameters (`ZIGZAG`, `BYTE_SHUFFLE`) are simply
absent from the TLV stream. `QUANTIZE` stores its `scale`/`offset`/`target`
there. Slot order in the stream is unconstrained: the decoder gathers
entries into a per-slot pointer table keyed by `slot_index`.

`validity_size`, when nonzero, is exactly `ceil(n_elems / 8)` and the
bitmap is laid out with 1 bit per element, LSB-first within each byte. The
v0 decoder treats it as opaque pass-through and does not surface it back
to the caller; the field exists so the on-disk record stays honest about
the bytes it carries and a future API extension can hand them back.

Three flag bits are defined. `TDC_BLOCK_FLAG_HAS_VALIDITY` (bit 0) means a
validity bitmap follows the payload. `TDC_BLOCK_FLAG_LOSSY` (bit 1) is set
when any stage in the chain was lossy; `TDC_XFORM_QUANTIZE` sets it. The
flag is informational; the decoder will not refuse to decode a lossy
block, but a caller that wanted exact reconstruction can check it before
treating the output as authoritative. `TDC_BLOCK_FLAG_ZERO_RESIDUAL`
(bit 2) declares that the model emitted an all-zero residual and the
transform and entropy chains were skipped. A decoder observes `payload_size ==
0` and `xform_params_size == 0`, then reconstructs a zero-filled residual of
length `uncompressed_size` before handing control to the model's decode path.

`tdc_block_record_validate` is the cheap structural validator. It checks
the magic, the version, that `rank` is within `[1, TDC_MAX_RANK]`, that
`dim[i]` for `i < rank` is non-negative, that `model_id` is non-zero (a
zero id never appears on disk), that the transform and entropy chains are
sticky-terminated (everything after the first zero must also be zero), and
that section sizes do not overflow. It does not decompress, does not
allocate, and runs in constant time. A block that fails validation returns
`TDC_E_CORRUPT` or `TDC_E_VERSION` without touching the payload.

## On-disk example

Two worked examples live under `docs/examples/`. The first,
`format_hexdump.c`, encodes a known input and dumps the resulting bytes as
annotated hex. The second, `format_peek_header.c`, parses an existing
block record and prints the decoded header struct. Both route every
allocation through a `realloc_fn` that wraps libc `realloc`/`free`,
matching the allocator convention the library uses internally.

The hex-dump example encodes a 16-element `i16` ramp (`src[i] = 100 + i`)
with `model = DELTA_1D`, `xform[0] = ZIGZAG`, no entropy. Real output from
a run on x86_64 MSVC:

```
input   : 16 x i16 = 32 bytes
encoded : 112 bytes (80-byte header + side + payload)

block record header:
  0000  42 4c 4b 42               magic            ('BLKB')
  0004  02 00                     version
  0006  00 00                     flags
  0008  02 00                     model_id
  000a  02 00 00 00 00 00 00 00   xform_ids[4]    (u16 x 4)
  0012  00 00 00 00 00 00 00 00   entropy_ids[4]  (u16 x 4)
  001a  02                        dtype
  001b  01                        layout
  001c  01                        rank
  001d  00                        _reserved0
  001e  00 00                     _reserved_pad
  0020  10 00 00 00 00 00 00 00   dim[0]          (int64)
  0028  00 00 00 00 00 00 00 00   dim[1]          (int64)
  0030  00 00 00 00 00 00 00 00   dim[2]          (int64)
  0038  20 00 00 00 00 00 00 00   uncompressed_size (u64)
  0040  00 00 00 00               side_meta_size   (u32)
  0044  20 00 00 00               payload_size     (u32)
  0048  00 00 00 00               xform_params_size(u32)
  004c  00 00 00 00               validity_size    (u32)

payload (32 bytes):
  0050  c8 00 02 00 02 00 02 00 02 00 02 00 02 00 02 00   payload
  0060  02 00 02 00 02 00 02 00 02 00 02 00 02 00 02 00   payload

total on-disk bytes: 112
round trip memcmp == source: yes
```

Every byte of the 80-byte header lines up with the table above. The magic
is `42 4c 4b 42`, the ASCII for `'BLKB'` read little-endian as
`0x424B4C42`. Version 2 is `02 00`; the flags field is clear. The model id
`02 00` is `TDC_MODEL_DELTA_1D = 0x0002`. The transform array's first slot
`02 00` is `TDC_XFORM_ZIGZAG = 0x0002`; slots 1..3 are zero, terminating
the chain. The entropy array is all zeros, and an all-zero entropy chain is
valid and means passthrough, as `tdc_codec_spec_raw` documents.

The type triple `02 01 01` decodes as `dtype = 2 (I16)`,
`layout = 1 (VECTOR_1D)`, `rank = 1`. `dim[0] = 0x10 = 16` elements; the
unused `dim[1]` and `dim[2]` are zeroed. `uncompressed_size` is
`0x20 = 32` bytes (16 elements times 2 bytes each). `side_meta_size = 0`
because `DELTA_1D` stores no side metadata. `payload_size = 0x20 = 32`.
`xform_params_size = 0` because `ZIGZAG` is a fixed-behavior transform
with no per-slot TLV payload. The trailing `validity_size = 0` reports an
absent NA bitmap for this block.

The payload starts at offset 80. The first two bytes `c8 00` are `0x00C8
= 200` little-endian, the zigzagged delta of the first element (`x[0] = 100`
has no predecessor, so the delta is `100`, zigzag-encoded as `2 * 100 = 200`).
Every subsequent pair is `02 00`: the zigzag of the constant delta of 1
between consecutive ramp elements. That pattern is exactly what DELTA_1D
plus ZIGZAG is supposed to produce on a linear ramp; the hex output makes
the pipeline visible in 32 bytes.

The second example, `format_peek_header.c`, runs a heavier spec
(`PRED_2D` + `ZIGZAG` + `BYTE_SHUFFLE` + `LZ`) on a 4×6 `i32` raster and
prints the decoded header struct plus the section offsets. Real output:

```
block record header (80 bytes)
  magic              = 0x424b4c42  (TDC_BLOCK_MAGIC = 0x424b4c42)
  version            = 2
  flags              = 0x0000
  model_id           = 0x0004
  xform_ids[0..3]    = 0x0002 0x0003 0x0000 0x0000
  entropy_ids[0..3]  = 0x0001 0x0000 0x0000 0x0000
  dtype              = 3 (I32)
  layout             = 2 (RASTER_2D)
  rank               = 2
  dim[0..2]          = 4 6 0
  uncompressed_size  = 96 bytes
  side_meta_size     = 1
  xform_params_size  = 0
  payload_size       = 27
  validity_size      = 0

section offsets:
  header       : 0 .. 80
  side_meta    : 80 .. 81
  xform_params : 81 .. 81
  payload      : 81 .. 108
  validity     : 108 .. 108

peek bytes_required = 96 (matches uncompressed element bytes)
tdc_block_record_validate = 0 (ok)
```

Here the model id is `0x0004` (`PRED_2D`). The transform chain is
`ZIGZAG -> BYTE_SHUFFLE`, both terminated by zero. The entropy chain is
`LZ` then zero. `side_meta_size = 1` because `PRED_2D` stores one byte
containing the resolved predictor kind after `AUTO` resolution. The whole
block record is 108 bytes; the 24 i32 elements that would have been 96
bytes raw compressed down to a 27-byte payload plus 1 side byte, plus the
80-byte header. `tdc_block_record_validate` returns `TDC_OK`, and
`tdc_decode_peek`'s `bytes_required` output exactly matches
`n_elems * sizeof(i32) = 24 * 4 = 96`.

## Evolution policy

tdc is pre-release. The format will change in place while the design is
still settling: struct reshuffles, new flag bits, new enum id
allocations, new sections inside the existing 80-byte and 64-byte
envelopes. No migration code, no dual-version reader branches. The rule
we carry from the project memory is that format-version bumps only begin
after the first real external consumer ships. Until then, edits land as
"format change: `<what>`" in the commit log.

Once that first consumer ships, the policy tightens. Any of the following
would require a version bump: adding a field that enlarges the 80-byte
block record or 64-byte container header; reordering existing fields;
redefining an existing flag bit; changing the semantics of a size field;
changing endianness. The format_hexdump program is the backstop for this
rule: every named field in its output is an offset in `format.h`, and
every change to `format.h` either preserves the program's output or
invalidates the vignette along with the consumer. The compile-time
`_Static_assert` on the struct sizes ensures a careless edit breaks the
build before it breaks a reader.

Changes that do *not* require a bump: reusing a `_reserved` byte for a new
field with a zero default that decodes identically to its previous
behavior; adding a new model / transform / entropy enum id in the unused
id range; adding a new flag bit. Backends must read unknown flag bits as
zero and unknown enum ids as `TDC_E_UNSUPPORTED`; the validator handles
both cases explicitly.

The enum id ranges are partitioned in `include/tdc/codec.h` so the static
registry in v0 and a future plugin API share the same u16 namespace
without collision. `0x0000` is a sentinel (never written as an active id).
`0x0001` through `0x00FF` is the core range for the ids shipped with tdc.
`0x0100` through `0x01FF` is experimental: backends that may change
without a version bump. `0x0200` through `0xFEFF` is reserved for future
core allocations. `0xFF00` through `0xFFFF` is earmarked for user-defined
backends once the plugin API lands; a user backend today should pick from
this range to avoid future collisions. The ranges apply identically to
model, transform, and entropy ids, giving three separate u16 spaces,
each with the same partition.

The `TDC_CONTAINER_VERSION` and `TDC_BLOCK_VERSION` constants will tick
up together or separately as changes require. The two versions are
independent so a block-layout change does not force a re-indexing of an
existing container. A v0-aware reader that has been compiled against
older headers will refuse to decode a record with a higher version: the
validator returns `TDC_E_VERSION`, distinct from `TDC_E_CORRUPT`, which
gives callers a clear signal that the record is well-formed but from a
future format they do not understand yet.
