# tdc

Typed, dimensional compression library in C.

tdc is a general-purpose multidimensional codec organized around four orthogonal layers — **model**, **transform**, **entropy**, **storage** — that share a single block abstraction across 1D vectors, 2D rasters, 2D stacks, and 3D volumes.

The unit of work is a `tdc_block`: shape plus dtype plus semantic layout. The same pipeline encodes a seismic trace, a satellite tile, a stack of frames, and a voxel grid. Structural rank and semantic layout are tracked independently; the model dispatcher branches on layout, never on rank alone.

## Where to go next

- [Quickstart](quickstart.md) — encode and decode a block in a dozen lines of C.
- [Backends: models](backends/models.md), [transforms](backends/transforms.md), [entropy](backends/entropy.md) — one walkthrough per backend, with benchmarks and edge cases.
- [On-disk format](format/on-disk.md) — container header and self-describing block record, field by field.
- [Custom backends](extending/custom-backends.md) — add a model, transform, or entropy coder against the internal registry.
- [Migration from vectra](reference/migration-vtr.md) — if you are coming from `.vtr`.

## Status

v0.1.0. Public headers under `include/tdc/` are stable for the v0 contract. The API may still change between minor versions until the first real consumer (vectra) has fully rewired onto tdc.

Source: [gcol33/tdc](https://github.com/gcol33/tdc). License: MIT.
