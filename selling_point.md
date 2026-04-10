# tdc — Selling Point

## The gap tdc fills

Most compression systems specialize in one side of a triangle:

```
        column compression
       (Parquet, Arrow, DuckDB)
              /\
             /  \
            /    \
           /  tdc \
          /________\
   tensor chunk        modular codec
   storage             pipelines
 (Zarr, TileDB,      (Blosc, ZFP, SZ)
  HDF5)
```

tdc sits at the intersection of all three.

## Landscape

### Scientific tensor compressors (ZFP, SZ/SZ3, MGARD)

Dominate HPC simulation data compression. They operate on blocks of
multidimensional arrays and exploit spatial correlation:

```
tensor block → prediction/transform → quantization → entropy coding
```

SZ compresses 2D blocks (often 16x16 tiles) using predictors and entropy
coding. ZFP handles 1D through 4D floating-point tensors. SZ3 is
explicitly modular and composable, but centered on scientific data
compression rather than a general storage format.

**Limitation.** Each implements a specialized compression algorithm.
They compress arrays but do not define how arrays are stored in datasets
— no container, no schema, no chunking strategy, no query metadata.

| What they do well         | What they lack                    |
|---------------------------|-----------------------------------|
| Spatial prediction        | Modular transform chains          |
| Block-level compression   | Pluggable entropy backends        |
| Dimensional awareness     | Self-describing container format  |
| N-D locality exploitation | Schema, statistics, selective I/O |

### Columnar systems (Parquet, ORC, DuckDB, ClickHouse)

Compress 1D columns using encoding transforms before entropy compression:

```
column → encoding (delta, dictionary, RLE, bitpacking) → block compression
```

The principle that "encoding transforms precede entropy compression" is
fundamental in columnar databases. These pipelines are sophisticated —
multiple transform stages, statistics per column group, rich metadata.
Their power comes from schema awareness and metadata-driven pruning.

**Limitation.** Columns are 1D. These systems do not naturally extend to
2D tiles or 3D volumes. The encoding transforms assume linear sequences,
not spatial neighborhoods.

### Chunk-based tensor stores (Zarr, TileDB, HDF5)

Handle N-dimensional arrays as chunked blocks with schema, indexing, and
metadata:

```
array → chunking → compression per chunk → container metadata
```

**Limitation.** Compression is shallow. Per-chunk compression is usually
just `chunk → zstd` or `chunk → gzip`. No transform pipelines, no
encoding stages, no dtype-aware prediction. The sophistication lives in
the container layer, not the codec layer.

### Blosc2 — the closest existing system

Blosc2 is the nearest practical system to tdc's architecture. It
explicitly positions itself as a compressor and data format for binary
and numerical arrays, with codecs and filters plus chunk and block
structure:

- Chunked containers
- Transform filters (shuffle, bitshuffle)
- Compression pipelines
- Multidimensional array support

```
block → filter (shuffle/bitshuffle) → compressor → container
```

**Gap.** Blosc2 gets closer to a general compressed-chunk format, but
does not carry the full columnar semantics: no per-column statistics, no
row-group-style query metadata, no schema in the Parquet sense. Its
transform system applies filters, not chained encoding stages. It does
not track semantic layout (raster vs. stack vs. volume) or branch
prediction strategy on data meaning.

### TensorStore (Google)

Handles multidimensional tensors with chunk storage, metadata, and
multiple backends. But compression is plugin-based and opaque — no
exposed transform chain, no codec-level awareness of data structure.

### Modular compression frameworks (research)

Research papers propose frameworks where different compressors (SZ, ZFP,
MGARD) can be swapped dynamically based on data characteristics. But
these operate at the level of **choosing an algorithm**, not designing a
**transform pipeline architecture**.

## Where each system sits

| System      | Tensor compression | Column encoding | Container |
|-------------|:------------------:|:---------------:|:---------:|
| ZFP         | yes                | no              | no        |
| SZ/SZ3      | yes                | no              | no        |
| MGARD       | yes                | no              | no        |
| Parquet     | no                 | yes             | yes       |
| Arrow IPC   | no                 | yes             | partial   |
| Zarr        | yes                | no              | yes       |
| TileDB      | yes                | no              | yes       |
| HDF5        | yes                | no              | yes       |
| Blosc2      | partial            | partial         | yes       |
| TensorStore | yes                | no              | yes       |
| **tdc**     | **yes**            | **yes**         | **yes**   |

## The hard problem

The hardest unsolved problem in this space is not "how do I compress 3D
data well?" — that has strong answers. The hard problem is:

> **How do you build one storage and compression architecture that works
> well for both columnar data and multidimensional chunked data, without
> forcing one model to pretend to be the other?**

This is hard because the two sides draw power from different things:

- **Column systems** derive their strength from schema awareness,
  per-column statistics, and metadata-driven pruning. Compression is
  only part of the story — selective reads, predicate pushdown, and
  row-group skipping are what make Parquet powerful.

- **Tensor compressors** derive their strength from exploiting
  neighborhood structure inside blocks — spatial prediction,
  blockwise transforms, local correlation. They do not need or use
  column schema or query planning.

- **Chunked array systems** and Blosc2 sit between: they know about
  chunks, blocks, filters, and multidimensional arrays. But they do not
  carry full columnar semantics — no per-column statistics, no
  row-group-style query metadata in the Parquet sense.

The tension:

- Optimize for **columns** → schema, stats, selective queryability.
- Optimize for **N-dimensional locality** → better blockwise compression.
- Try to unify them → risk building something elegant in theory but
  second-best for both.

## What tdc does differently

tdc's answer to the hard problem is a two-layer separation:

1. **Block layer (Phase C):** A universal block abstraction that carries
   enough structure to compress 1D columns, 2D tiles, 2D stacks, and 3D
   chunks through the same pipeline.

2. **Container layer (Phase B):** The metadata contract — schema,
   statistics, indexing, selective parallel reads — layered on top of
   blocks without leaking into the codec.

The block pipeline:

```
tdc_block (shape + dtype + semantic layout)
   ↓
model        — dimension-aware prediction (residual stream)
   ↓
transform[]  — dimension-agnostic chain (symbolization, delta, shuffle)
   ↓
entropy      — dimension-agnostic byte compression
   ↓
block record — self-describing, all decode metadata inline
```

The core abstraction:

```c
encode_block(data, codec_spec)
```

Where `data` could be a 1D column, a 2D tile, or a 3D tensor chunk —
and the same compression pipeline applies. Most systems assume one data
structure (Parquet assumes columns, Zarr assumes arrays, ZFP assumes
floating-point tensors). tdc unifies them.

The container layer then decides how much schema, statistics, and
indexing to expose — without the codec needing to change.

### Key properties

- **Structural rank and semantic layout are independent.** A 2D array
  can be a raster or a stack of 1D series — the model dispatcher
  branches on layout, not rank alone. This distinction does not exist
  in any system listed above.

- **Transform chains are first-class.** Four transform slots from day
  one, walked as an array. Not a single hardcoded filter (Blosc2) or
  a monolithic algorithm (ZFP). Columnar-style encoding depth applied
  to tensor data.

- **Self-describing block records.** Every field needed to decode lives
  in the record header. No external schema required, no sidecar files.
  Unlike Zarr or HDF5, the block is decodable in isolation.

- **No external dependencies in core.** Pure C11. zlib is opt-in at
  link time for a single entropy backend. Everything else is native.
  Embeddable in R packages, Python extensions, or any C-linkable host.

- **Caller owns allocation.** A single `realloc_fn` callback controls
  all memory. No hidden mallocs, no global state, thread-safe by
  construction.

## The deeper insight: compression is modeling

Shannon showed that the entropy coder cannot beat the entropy of the
modeled distribution. The only way to compress more is to improve the
model of the data distribution. Compression breakthroughs rarely come
from new entropy coders — they come from better models of structure.

The standard decomposition:

```
data → model/transform → residual → entropy coding
```

Entropy coding (Huffman, arithmetic, ANS) is already close to optimal.
The real gains come from the model that removes structure before the
entropy coder sees the data.

This means tdc is not really "a compression algorithm." It is a
**framework for structured data compression models.** The entropy coder
is stable; the models vary by data structure:

| Data structure | Good models                          |
|----------------|--------------------------------------|
| 1D column      | delta, dictionary, RLE               |
| time series    | predictive filters                   |
| 2D raster      | spatial predictors, wavelets         |
| 3D volume      | multigrid predictors, 3D wavelets    |

Most systems fix the model and vary the entropy coder. tdc inverts this:
the entropy layer is a stable backend, while the model layer is the
primary axis of variation — selected by semantic layout, not hardcoded.

The modularity test: can someone add a new model (e.g. 3D wavelet
predictor) without changing anything else in the system? If yes, the
abstraction boundary is right. If not, the model layer is not truly
decoupled.

## The architectural bet

Column stores (Parquet) give you rich per-column pipelines but only for
1D data. Tensor stores (Zarr) give you N-dimensional chunking but
shallow compression. tdc unifies both:

> Columnar compression pipelines applied to multidimensional chunked
> data inside a self-describing container.

That intersection is not heavily explored. Most projects choose columnar
**or** multidimensional, not both cleanly. tdc's block abstraction makes
adding the next dimensionality O(1) effort — write a layout iterator and
a model, the transform chain and entropy layer come for free.

## The clean test

> **Would another project be able to adopt tdc as its native storage
> layer without inheriting vectra's container logic?**

If yes, tdc has crossed from "compression library" into "storage and
compression architecture." That is the goal.

The compression algorithms are not new as a category. The storage
boundary tdc is trying to define — one codec and block model across
dimensionalities, with the container layer deciding how much metadata
to expose — is the genuinely interesting part.

## Why C first, B later

Phase C validates the universal block abstraction: can one pipeline
compress 1D, 2D, and 3D data without forcing one to pretend to be the
other?

Phase B proves whether that abstraction can carry a real storage model
— schema, statistics, indexing, selective parallel reads — without
collapsing under metadata complexity.

If tdc only compresses bytes, it is useful. If tdc can compress typed
blocks across dimensions *and* own the metadata contract needed to read
them intelligently, it becomes a serious architecture.

## What this is and is not

**Not a new algorithmic category.** The individual components — spatial
prediction, transform encoding, entropy compression, chunked containers
— all exist in isolation.

**A new system architecture.** The combination of columnar compression
pipelines with multidimensional tensor blocks inside a self-describing
format is not common. Many important systems innovations look exactly
like this: not a new algorithm, but a new way of composing known pieces.

## Who benefits

- **Scientific data pipelines** that handle mixed-dimensional outputs
  (time series, spatial grids, volumetric fields) and want one codec
  instead of three.

- **R and Python packages** (starting with vectra) that need embedded,
  zero-dependency compression aware of array semantics.

- **Anyone building a storage format** who wants a compression backend
  that understands typed, shaped data — not just flat byte buffers.
