# CLAUDE.md — tdc

This file is the project-level brief for Claude Code sessions working in
this repository. It supersedes the user's global CLAUDE.md only where it
disagrees; otherwise both apply.

## What tdc is

A typed, dimensional compression library in C11. Four orthogonal layers:

```
   tdc_block  -->  Model  -->  Transform chain  -->  Entropy  -->  Block record
                  (dim-aware) (dim-agnostic)       (dim-agnostic) (self-describing)
```

The unit of work is a `tdc_block`: shape + dtype + semantic layout. The
same pipeline encodes 1D vectors, 2D rasters, 2D stacks, and 3D volumes.
**Structural rank** (1/2/3) and **semantic layout** (`VECTOR_1D`,
`RASTER_2D`, `STACK_2D`, `VOLUME_3D`) are tracked independently. The model
dispatcher branches on layout, never on rank alone.

Originally extracted from the [vectra](https://github.com/gcol33/vectra)
R package, but tdc is **not** a vectra subset. See "No vectra back-compat"
below.

## Frozen contract: `include/tdc/`

Public headers are the contract. Treat them as locked unless the user
explicitly authorizes a header change.

| Header | Owns |
|---|---|
| `types.h`     | `tdc_dtype`, `tdc_layout`, `tdc_shape`, `tdc_block`, `tdc_buffer`, `tdc_status` |
| `codec.h`     | `tdc_codec_spec`, model/xform/entropy enum ids, per-stage params |
| `model.h`     | `tdc_model_vt` + `tdc_model_get` |
| `transform.h` | `tdc_xform_vt` + `tdc_xform_get` |
| `entropy.h`   | `tdc_entropy_vt` + `tdc_entropy_get` |
| `format.h`    | container header + self-describing block record |
| `error.h`     | `tdc_strerror` |

If a frozen header is genuinely broken (e.g. struct doesn't pack to its
declared size), **say so before patching it** and get explicit approval.
Format-level fixes are allowed but require a heads-up because they ripple.

## No vectra backwards compatibility

tdc is a **complete redesign** of vectra's compression backend, not a
drop-in replacement. The vectra `.vtr` on-disk format is being abandoned;
nothing in tdc needs to read or write existing `.vtr` files.

When extracting code from `vectra/src/vtr_codec.c` and friends:

- Preserve the **inner loops** (matcher, fast/safe decode, wildcopy,
  predictor kernels) verbatim — they are benchmarked and the dead-end
  optimizations are documented in `vectra/CLAUDE.md`. Don't re-tune.
- Feel free to **change everything else**: byte layout, header fields,
  field ordering, encode-time fallback policy, error handling, allocation
  path, public API. Anything that isn't the hot loop is fair game.
- "Byte-identical to vectra" is **not** a goal. Round-trip correctness
  inside tdc is the only requirement.
- If a vectra convention conflicts with the tdc headers, **tdc wins**.

Vectra will be rewired to call tdc once extraction is done; at that point
vectra's old codec disappears entirely.

## Allocation rule (hard)

`tdc_buffer::realloc_fn` is the **only** allocation path inside tdc. No
bare `malloc/free/calloc/realloc` in any `src/**.c`. Convention:

```c
realloc_fn(user, NULL, n)  /* allocate n bytes */
realloc_fn(user, p,    0)  /* free p */
realloc_fn(user, p,    n)  /* grow p to n bytes (may move) */
```

For scratch memory inside an encode/decode (hash tables, sequence arrays,
literal buffers): allocate via `realloc_fn`, free before returning. Do
not stash scratch in static globals — that breaks thread safety and the
"caller owns allocation" promise.

## Stage layering (hard)

The four-layer pipeline is the architecture. Don't blur the boundaries:

- **Model** is dimension-aware. It owns iteration order, neighborhood
  access, and prediction rules. It produces a residual stream + serialized
  side metadata. It does NOT do byte shuffling, quantization, or entropy
  coding.
- **Transform** is dimension-agnostic. It consumes bytes from the previous
  stage and produces bytes for the next. Symbolization (residual → byte
  stream) lives here, not in its own phase. The transform stage is a
  **chain** (`xform_ids[4]`) from day 0; even with one transform, the
  call site walks an array.
- **Entropy** is dimension-agnostic and dtype-agnostic. It sees a flat byte
  buffer. It does NOT need to grow its decode output buffer — block
  records carry exact uncompressed sizes. Decode hot path stays branch-
  free on size.
- **Storage** writes self-describing block records. Side metadata
  (model params, dictionaries, plane coeffs) is a **first-class** section
  of the record, not a sidecar.

`layout/` answers "how do I iterate?". `model/` answers "given accessible
neighbors, how do I predict?". Layout helpers must never call back into
models.

## Static registry, internal headers

v0 has no public plugin API. Each stage is dispatched by enum id through
a static switch in `src/core/registry.c`:

```
tdc_model_get   -> src/core/registry.c     -> entropy_internal.h, etc.
tdc_xform_get
tdc_entropy_get
```

Pattern for adding a backend:

1. Define `const tdc_<stage>_vt tdc_<stage>_<name>_vt = { ... };` in the
   backend's `.c` file (no `static`).
2. Add an `extern` declaration to the matching internal header
   (e.g. `src/entropy/entropy_internal.h`). These internal headers live
   under `src/`, not `include/`, on purpose — they are not part of the
   public ABI.
3. Add the case to the switch in `src/core/registry.c`.

A future plugin API can be bolted on without breaking the on-disk format
because the ids are u16 and id ranges are reserved (`0x0001-0x00FF` core,
`0x0100-0x01FF` experimental, `0xFF00-0xFFFF` user-defined).

## Build

```bash
cd tdc
cmake -B build
cmake --build build --config Debug
cd build && ctest -C Debug --output-on-failure
```

The repo is developed on Windows under MSVC + Visual Studio generator.
Compile flags include `/W4 /permissive-` (MSVC) or
`-Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes
 -Wmissing-prototypes -Wcast-align` (gcc/clang). All `.c` files must
build clean under both.

C standard: **C11**, no extensions (`CMAKE_C_EXTENSIONS OFF`).

Vectra (the primary consumer) does NOT use CMake — it will vendor the
tdc source tree under `vectra/src/tdc/` and compile via R's Makevars.
That means: no CMake-only features in `src/**.c`. Plain C11 + headers.
No generated files, no external link deps in core (`TDC_HAVE_ZLIB` is
opt-in for the deflate backend only).

## Endianness and ABI

Fixed **little-endian** on disk and at runtime. Documented in `types.h`.
A big-endian port would require a format version bump.

Supported targets: x86_64, aarch64. Both are little-endian. SIMD intrinsics
are allowed but must be guarded behind `__SSE2__` / `__ARM_NEON` and
have a scalar fallback.

## Testing

Tests live in `tests/` and are wired into ctest via the existing
`CMakeLists.txt`. Pattern:

1. Use a POSIX-style `realloc_fn` wrapper around stdlib `realloc/free`
   for the test harness — same convention as `realloc_fn`.
2. Resolve vtables by extern declaration when testing a stage in
   isolation (don't depend on `tdc_*_get` from registry).
3. Round-trip + edge cases (empty, single byte, sub-min-match input,
   highly compressible, incompressible) for every entropy/transform.
4. For models, test on a synthetic deterministic input (LCG, gradient,
   sinusoid) — not random data.

Tests are NOT part of the frozen contract; modify them freely.

## Deviations and "do not retry" notes

Compressed-loop optimization is benchmarked, not theorized. Before
"improving" any inner loop, check `vectra/CLAUDE.md` for documented dead
ends. As of now:

- **No batched parse-then-copy in LZ2 fast path** — measured 6-7% slower
  in vectra benchmarks. Re-evaluate only after a real entropy stage
  (FSE/Huffman) lives in front of LZ2.
- **No Robin Hood hashing** in `JoinHT`/`VecHashTable` style structures —
  measured 12-29% regression in vectra. FNV-1a + 70% load distributes
  well enough that average probe length doesn't justify Robin Hood's
  per-probe overhead.

When extracting from vectra, **read the relevant section of
`vectra/CLAUDE.md` first**. The benchmark notes there are the closest
thing this project has to institutional memory.

## Engineering rules in force

- C11. Warnings clean under MSVC `/W4 /permissive-` and gcc/clang
  `-Wall -Wextra -Wpedantic`.
- No external dependencies in core. zlib is optional, link-time, behind
  `TDC_HAVE_ZLIB`. No new dependencies without explicit approval.
- No bare malloc/free in `src/**.c`. Use `realloc_fn`.
- Little-endian assumed and documented.
- Self-describing block records: every field needed to decode lives in
  the record header. Container metadata is duplicated, intentionally.
- Symbolization belongs to the transform stage, not its own phase.
- The transform stage is a **chain** (`xform_ids[4]`), not a single id,
  even when only one transform is needed.
