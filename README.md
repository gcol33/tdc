# tdc

[![CI](https://github.com/gcol33/tdc/actions/workflows/ci.yml/badge.svg)](https://github.com/gcol33/tdc/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Typed, dimensional compression library in C.

`tdc` is a general-purpose multidimensional codec organized around four
orthogonal layers — **model**, **representation**, **entropy**, **storage** —
that share a single block abstraction across 1D, 2D, 2D-stack, and 3D data.

Originally extracted from the [vectra](https://github.com/gcol33/vectra) R
package, where it powers the `.vtr` columnar format.

## Status

**v0.1.0.** Public headers under `include/tdc/` are stable for the v0
contract. The API may still evolve between minor versions until the first
real consumer (vectra) has fully rewired onto tdc. See
[CHANGELOG.md](CHANGELOG.md) for release notes.

## Architecture

```
                    +------------------+
   raw buffer  -->  |   tdc_block_t    |   shape + dtype + layout (semantic)
                    +------------------+
                            |
                            v
                   [ Model / Predictor ]    dim-aware
                            |  residuals + side metadata
                            v
                   [ Transform chain  ]    dim-agnostic, owns symbolization
                            |  bytes
                            v
                   [ Entropy coder    ]    dim-agnostic
                            |  payload
                            v
                   [ Block record     ]    self-describing
```

The core insight: **structural rank** (rank-1, rank-2, rank-3) and
**semantic layout** (`VECTOR_1D`, `RASTER_2D`, `STACK_2D`, `VOLUME_3D`) are
independent. A 2D stack and a 3D volume are both rank-3 in memory; they
differ in how the model is allowed to traverse them. The model dispatcher
branches on layout, never on rank alone.

## v0 scope

Models (every entry has either a vectra extraction or a v0 implementation):

| ID | Source |
|---|---|
| `TDC_MODEL_RAW` | identity |
| `TDC_MODEL_DELTA_1D` | extract from vectra `VTR_ENC_DELTA` |
| `TDC_MODEL_DICT_1D` | extract from vectra `VTR_ENC_DICTIONARY` |
| `TDC_MODEL_PRED_2D` | extract from vectra `VTR_ENC_SPATIAL` (LEFT/UP/AVG/PAETH/PLANE) |
| `TDC_MODEL_STACK_2D` | new in tdc v0 |
| `TDC_MODEL_PRED_3D` | new in tdc v0 |

Transforms: `RAW`, `QUANTIZE`, `ZIGZAG`, `BYTE_SHUFFLE`.

Entropy: `NONE`, `LZ` (native, separated-stream), `DEFLATE` (zlib, optional).

## Source layout

```
include/tdc/
    types.h         status, dtype, layout, shape, block, buffer
    pipeline.h      tdc_codec_spec + stage ids
    format.h        container header + self-describing block record
    model.h         model vtable + lookup
    transform.h     transform vtable + lookup
    entropy.h       entropy vtable + lookup
    error.h         tdc_strerror

src/
    core/           block helpers, arena, stream, registry
    layout/         iteration, tiling, neighborhood (called by models)
    model/          predictors (dimension-aware)
    representation/ transforms + symbolization (dimension-agnostic)
    entropy/        compressors (dimension-agnostic)
    storage/        container reader/writer, header, block record
    api/            tdc_encode_block, tdc_decode_block
```

## Design rules

1. `tdc_block` describes memory only — no compression policy fields.
2. Endianness is fixed little-endian on disk and at runtime.
3. Block records are self-describing: every field needed to decode lives in
   the record header. Container metadata is duplicated, intentionally, so
   blocks survive being copied or extracted.
4. Side metadata (model params, dictionaries, plane coefficients) is a
   first-class section of the block record, not an afterthought.
5. The transform stage is a chain (`xform_ids[4]`) from day 0.
6. Symbolization belongs to the transform stage, not its own phase.
7. Static enum-driven registry in v0; no public plugin API yet, but enum id
   ranges are reserved so one can be added later without a format bump.
8. Hard layout/model boundary: `layout/` answers "how do I iterate?",
   `model/` answers "given accessible neighbors, how do I predict?".

## License

MIT — see [LICENSE](LICENSE).
