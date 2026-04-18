# Custom backends

tdc's four-stage pipeline (model, transform chain, entropy, block record) runs every backend through the same vtable interface. Adding a new backend means writing one vtable, wiring it through the static registry, and shipping a round-trip test. This vignette walks through exactly that, using a reference transform (`TDC_XFORM_COMPLEMENT`) that lives in the tree at `src/transform/complement.c`. The backend is intentionally trivial so the prose can concentrate on the extension contract rather than the algorithm; a real backend swaps the kernel and keeps the rest of the scaffolding identical.

This is the only vignette that reaches into `src/` by design. Every other vignette stays on the public header surface under `include/tdc/`; plugin-API callers building against a consumer project can ignore `src/` entirely. The extension path described here targets tdc contributors and projects that vendor the tdc source tree and want a backend to ship alongside the core ones. A strictly runtime plugin path also exists under `include/tdc/plugin.h`, and covers the case where a consumer wants to inject a backend at load time without touching the library source. Both paths appear below; the static path comes first because it is the one a core contributor uses when the backend is meant to ship with tdc itself.

## Scope

A "custom backend" is one implementation of a `tdc_model_vt`, `tdc_xform_vt`, or `tdc_entropy_vt`. Each of those three structs is a small vtable: a few function pointers, an id, a name, and per-stage metadata (accepted dtypes, layouts, lossy flag). The library never sees the backend's source: it calls the function pointers through the vtable it gets back from `tdc_model_get`, `tdc_xform_get`, or `tdc_entropy_get`. Those lookup functions live in `src/core/registry.c` and run a static switch over the enum id. The default case of each switch falls through to a plugin registry array populated by `tdc_*_register()` calls from `include/tdc/plugin.h`. Two registration paths, one dispatch surface.

The reserved id ranges (declared in `include/tdc/codec.h` and enforced in `include/tdc/plugin.h`) partition the u16 id space:

| Range | Meaning | Registration |
|---|---|---|
| `0x0000` | Chain terminator (transforms, entropy) or sentinel (models) | never a real backend |
| `0x0001-0x00FF` | Core, shipped with tdc | enum entry + static registry switch case |
| `0x0100-0x01FF` | Experimental, statically compiled, may change without a format version bump | enum entry + static registry switch case |
| `0x0200-0xFEFF` | Reserved for future use | registration rejected |
| `0xFF00-0xFFFF` | User-defined, runtime plugin registration | no enum entry; call `tdc_*_register()` at startup |

The on-disk format never has to grow because every id slot is already a u16. A block encoded with `TDC_XFORM_COMPLEMENT` (id `0x0100`) carries the bare `0x0100` in its header, and any tdc build with a backend registered for that id reads the block back correctly.

The reference backend for this vignette sits in the experimental range because it is a teaching example. If we were writing a permanent backend we would give it a core-range id instead and document it alongside the rest of the core transforms. Everything below about the code is identical either way; the range affects validation policy, not the vtable.

## The vtable

`tdc_xform_vt` is the smallest of the three vtables, so we use it for the walkthrough. `include/tdc/transform.h` declares:

```c
typedef struct tdc_xform_vt {
    tdc_xform_id id;
    const char  *name;

    uint32_t accepted_dtypes;   /* bitmask over tdc_dtype (input dtype) */
    int      can_inplace;       /* 1 if encode/decode may share src and dst */
    int      is_lossy;          /* 1 if information is discarded */

    tdc_status (*encode)(const uint8_t *src, size_t src_size,
                         tdc_dtype      in_dtype,
                         const void    *params,
                         tdc_buffer    *dst,
                         tdc_dtype     *out_dtype);

    tdc_status (*decode)(const uint8_t *src, size_t src_size,
                         tdc_dtype      in_dtype,
                         const void    *params,
                         uint8_t       *dst, size_t dst_size,
                         tdc_dtype     *out_dtype);
} tdc_xform_vt;
```

Every transform sees a flat byte buffer on both sides of the call, plus the element dtype it came in as. Transforms are dimension-agnostic by design: they never look at layout, rank, or shape. The `can_inplace` flag lets the driver elide a scratch copy when the kernel is element-local. The `is_lossy` flag tells the encoder whether a spec like `QUANTIZE -> LZ` round-trips exactly or approximately.

The reference backend is a single-byte bitwise NOT:

```c
/* src/transform/complement.c */
static tdc_status complement_encode(const uint8_t *src, size_t src_size,
                                    tdc_dtype      in_dtype,
                                    const void    *params,
                                    tdc_buffer    *dst,
                                    tdc_dtype     *out_dtype) {
    (void)params;
    if (!dst || !dst->realloc_fn) return TDC_E_INVAL;
    if (src_size > 0 && !src)     return TDC_E_INVAL;
    if (!complement_dtype_ok(in_dtype)) return TDC_E_DTYPE;

    tdc_status st = tdc_buf_reserve(dst, src_size);
    if (st != TDC_OK) return st;

    if (out_dtype) *out_dtype = in_dtype;

    for (size_t i = 0; i < src_size; ++i) dst->data[i] = (uint8_t)~src[i];
    dst->size = src_size;
    return TDC_OK;
}
```

The decoder runs the same loop with opposite buffer conventions: caller-provided output, exact size known up front, no reallocation. `tdc_buf_reserve` is a header helper from `src/core/buffer.h` that grows the output buffer through the caller's `realloc_fn`. That is the only allocation path any backend may use. No bare `malloc`, no bare `free`, no thread-local scratch pool. The reason is not stylistic: vectra, the primary consumer, feeds tdc an arena allocator that reclaims every byte on block exit, and that only works if every scratch allocation flows through the supplied function pointer.

The vtable instance itself is a file-scope `const` with no `static` storage duration:

```c
const tdc_xform_vt tdc_xform_complement_vt = {
    .id              = TDC_XFORM_COMPLEMENT,
    .name            = "complement",
    .accepted_dtypes = COMPLEMENT_ACCEPTED_DTYPES,
    .can_inplace     = 1,
    .is_lossy        = 0,
    .encode          = complement_encode,
    .decode          = complement_decode,
};
```

The `const` discipline lets the compiler place the vtable in `.rodata`. The missing `static` matters: the registry switch in `src/core/registry.c` takes the address of this symbol from another translation unit, which requires external linkage. A predictable name (`tdc_xform_<name>_vt`) lets the internal header cross-declare it without collision. The encode and decode functions stay `static` because the vtable is their only caller; outside access goes through the function pointers.

Models and entropy backends follow the same pattern. `tdc_model_vt` adds `accepted_layouts` and a `side_out` buffer (so the model can serialize its state alongside the residual stream). `tdc_entropy_vt` adds an `encode_bound` function so the driver can size the output buffer before the coder starts filling it. Both keep the allocator rule and the `const` plus external-linkage convention.

## Registering

Registering a core or experimental backend takes three edits in three files:

1. A `.c` file under `src/transform/` (or `src/model/`, `src/entropy/`) exporting a single `const tdc_<stage>_vt tdc_<stage>_<name>_vt = { ... };`.
2. An `extern` declaration in the matching internal header (`src/transform/transform_internal.h`, `src/model/model_internal.h`, `src/entropy/entropy_internal.h`). Internal headers live under `src/`, not `include/`, so they stay out of the public ABI.
3. One new `case` in the static switch inside `src/core/registry.c`.

For the reference backend those three edits are:

```c
/* src/transform/transform_internal.h */
extern const tdc_xform_vt tdc_xform_byte_shuffle_vt;
extern const tdc_xform_vt tdc_xform_bit_shuffle_vt;
extern const tdc_xform_vt tdc_xform_quantize_vt;
extern const tdc_xform_vt tdc_xform_zigzag_vt;
extern const tdc_xform_vt tdc_xform_complement_vt;  /* new */
```

```c
/* src/core/registry.c */
const tdc_xform_vt *tdc_xform_get(tdc_xform_id id) {
    switch (id) {
        case TDC_XFORM_BYTE_SHUFFLE: return &tdc_xform_byte_shuffle_vt;
        case TDC_XFORM_NONE:         return NULL;
        case TDC_XFORM_QUANTIZE:     return &tdc_xform_quantize_vt;
        case TDC_XFORM_ZIGZAG:       return &tdc_xform_zigzag_vt;
        case TDC_XFORM_BIT_SHUFFLE:  return &tdc_xform_bit_shuffle_vt;
        case TDC_XFORM_COMPLEMENT:   return &tdc_xform_complement_vt;  /* new */
        default:
            return (const tdc_xform_vt *)plugin_lookup(
                (uint16_t)id, s_xform_slots, s_xform_count);
    }
}
```

The enum value itself lives in `include/tdc/codec.h`, which is a frozen public header:

```c
typedef enum {
    TDC_XFORM_NONE         = 0x0000,
    TDC_XFORM_QUANTIZE     = 0x0001,
    TDC_XFORM_ZIGZAG       = 0x0002,
    TDC_XFORM_BYTE_SHUFFLE = 0x0003,
    TDC_XFORM_BIT_SHUFFLE  = 0x0004,
    /* Experimental range (0x0100-0x01FF). */
    TDC_XFORM_COMPLEMENT   = 0x0100
} tdc_xform_id;
```

Adding a new enum value to `tdc_xform_id` preserves backward compatibility: old callers that do not know about the new constant simply never dispatch to it, and compiled binaries stay happy because the enum's underlying type does not change. The frozen-header rule keeps struct layouts and function signatures stable; id-space growth falls outside that rule because every id slot in `codec.h` is a u16 and the reserved ranges already accommodate it.

The last edit is the CMake source list. `CMakeLists.txt` has a `TDC_SOURCES` variable; append the new file so the static library rebuilds against it. Vectra's vendor build (its `Makevars` recursively compiles everything under `vectra/src/tdc/src/`) picks up the new file automatically, which saves a vendor-side edit. That split matters: tdc contributors edit the CMake list, vectra consumers re-sync the vendored tree, and neither side knows about the other's build system.

A new backend does not need a block-record format change. The block record already carries the 16-bit xform id verbatim. If the id is `0x0100`, the record stores `0x0100`, the decoder hands that id to `tdc_xform_get`, and `tdc_xform_get` returns the vtable. Self-describing block records carry every stage id with the block, and every id the encoder emitted maps to a concrete vtable the decoder can look up.

## Wiring tests

Every backend gets a round-trip test under `tests/`, wired into CTest by `CMakeLists.txt`. The test exists to detect exactly the class of bug that registry wiring breaks: a mis-edited switch, an `extern` declaration typoed, a vtable field left NULL, an allocator call that bypasses `realloc_fn`. The reference test covers six points:

1. The registry returns the expected vtable pointer when queried for the backend's id.
2. Vtable metadata (id, `accepted_dtypes`, `can_inplace`, `is_lossy`, `name`) is internally consistent.
3. A round trip on every supported dtype reproduces the input byte-for-byte.
4. The empty input does not allocate, does not crash, and reports `TDC_OK`.
5. A single-element input round-trips.
6. A degenerate all-equal block round-trips, with an assertion on the encoded pattern so the test fails loudly if the kernel ever regresses.

The skeleton follows the existing `tests/test_zigzag_roundtrip.c` convention: a POSIX-style `realloc_fn`, a `make_buffer` helper, a single `main` that calls each subtest and reports the first failure. Tests resolve the vtable via `extern`, not `tdc_xform_get`, so the backend itself is testable in isolation from the registry. A separate subtest then verifies the registry path returns the same address, which catches the "forgot to add the case" bug.

```c
/* tests/test_complement_roundtrip.c */
extern const tdc_xform_vt tdc_xform_complement_vt;

static int test_registry(void) {
    const tdc_xform_vt *vt = tdc_xform_get(TDC_XFORM_COMPLEMENT);
    CHECK(vt == &tdc_xform_complement_vt,
          "registry should return &tdc_xform_complement_vt");
    CHECK(vt->id == TDC_XFORM_COMPLEMENT, "vt->id mismatch");
    CHECK(vt->is_lossy == 0,  "complement is not lossy");
    CHECK(vt->can_inplace == 1, "complement is element-local");
    printf("  [registry + vtable] OK\n");
    return 0;
}
```

The round-trip pass runs through three dtypes (`u8`, `i32`, `f64`) chosen to exercise the three fixed-width families. For each, the test encodes into a `tdc_buffer` (with `realloc_fn` wired to a `free`/`realloc` wrapper), asserts the encoded size matches the input size, checks the encoded bytes against the expected `~src[i]` pattern, decodes into a caller-owned output, and compares with `memcmp`. The same helper pattern covers every new transform; only the expected-byte formula changes.

Edge cases matter more than throughput for a correctness test. Empty input is the most common cliff for a new backend: an encode loop that accidentally reads `src[0]` on zero input blows up at the first smoke test under AddressSanitizer. A single-element input catches off-by-one wrap-around and ensures the `accepted_dtypes` check comes before the element-count guard. The all-equal (`0xAA`) block exercises a predictable output (`0x55`); asserting that pattern directly turns future regressions into a content error instead of a memcmp mismatch. These three cases earn their keep; everything else is variations on round-trip equality.

The last subtest runs the backend through the full `tdc_encode_block` + `tdc_decode_block` pipeline, with the reference backend in the transform slot. This catches wiring bugs that the vtable-only tests miss: the block record validator has to accept the new id, the driver has to route the correct dtype through the transform chain, and the decoder has to pick up the new backend from the registry at decode time. On a 128-byte u8 input the output is 208 bytes (128-byte payload plus 80-byte block record header), and `memcmp` confirms the reconstruction is exact.

The test source goes in `tests/test_<name>_roundtrip.c`; the `CMakeLists.txt` additions are two lines:

```cmake
add_executable(test_complement_roundtrip tests/test_complement_roundtrip.c)
target_link_libraries(test_complement_roundtrip PRIVATE tdc)
add_test(NAME complement_roundtrip COMMAND test_complement_roundtrip)
```

Running `ctest -C Debug -R complement` selects the new test without running the rest of the suite, useful during development; `ctest -C Debug` runs everything, which is what we do before a commit. Under sanitizers the test catches the second-most-common backend bug after registry wiring, which is a signed-vs-unsigned dtype mismatch that causes the out_dtype field to diverge between encode and decode. The test pins `out_dtype == in_dtype` on both sides for the reference backend; a transform that actually changes dtype (like ZIGZAG) would assert the expected unsigned dtype instead.

## Performance checklist

Before claiming a new backend is ready, run each of these gates. Every item either measures something or rules something out; none is decorative.

**Allocation rule.** `src/**.c` is forbidden from calling `malloc`, `calloc`, `realloc`, or `free` directly. Every scratch buffer goes through `buf->realloc_fn`. Grep is cheap and the rule is hard: a backend that sneaks in a raw `malloc` breaks vectra's arena allocator and every downstream caller that supplied its own. The `can_inplace` flag is not a workaround for this; it lets the driver skip the round-trip allocation, nothing more.

**Ratio and throughput on two inputs.** Pick two inputs that stress the backend differently: one where the algorithm is expected to pay off, and one where it should no-op or slightly hurt. Run both through `bench/bench_throughput.c` (or a one-shot driver that wraps `tdc_encode_block` in a timed loop on a hot input) and record encode MiB/s, decode MiB/s, and ratio. Real numbers only; adjectives are noise. The goal is calibration, not a leaderboard. Ratio is the primary target, and ratio-neutral speed losses are allowed up to roughly 1-2%; an experimental transform that costs more than that without moving the ratio needle does not ship.

**No regression on existing backends.** Re-run every backend's `*_roundtrip` test after the new code lands. CTest handles this mechanically; the failure mode that matters is a switch-case that accidentally shadows a pre-existing id, which the plugin-capacity test and the per-backend round-trip tests both catch. Do not treat the new backend's pass as a green light; the registry is shared, and adding an experimental case in the wrong place can dispatch the old id to the new vtable.

**SIMD and fallbacks.** tdc targets x86_64 and aarch64, both little-endian, both potentially SIMD-capable. AVX2 intrinsics are allowed behind `#if defined(__AVX2__)`, NEON behind `#if defined(__ARM_NEON)`. Every SIMD path requires a scalar fallback that compiles when the intrinsic guard is false. CMake's `TDC_ENABLE_AVX2` option defaults to ON; a backend that assumes AVX2 without a fallback breaks when the option is flipped off. The reference backend has no SIMD: a per-byte kernel hits memory bandwidth on any half-modern CPU and a vectorized version saves nothing. Adding SIMD to a transform that does not need it is a speed regression waiting for a maintenance tax.

**"Do not retry" notes.** `tdc/CLAUDE.md` and `vectra/CLAUDE.md` document the hot-path experiments that have already been measured and rejected. The current list: no Robin Hood hashing in hash tables (12-29% regression in vectra benchmarks), no batched parse-then-copy in the LZ fast path (6-7% regression, may revisit after a real entropy stage lives in front of LZ), no thread-local scratch in `src/**.c` (breaks the allocator rule and thread safety). A new backend that rediscovers one of these dead ends wastes a review cycle; read the relevant section before touching a loop.

**Fuzz the round trip.** A backend that round-trips cleanly on a hand-written test suite can still crash on random input. Run a few million iterations of "random buffer -> encode -> decode -> compare" under `-fsanitize=address,undefined`. Pattern-generate the inputs so the shape of the input varies: power-of-two sizes, prime sizes, empty, single-byte, edges of the dtype range, runs of the same byte, interleaved noise. The reference test covers the categorical cases; a night of fuzzing catches the ones no reviewer thought to write down. The cost is trivial (the backends are fast, the comparison is fast), and it rules out a whole class of memory-safety and edge-case bugs that would otherwise surface as customer reports.

**Block-record validation.** The validator in `src/format/block_record.c` hard-rejects id values outside the reserved ranges. Confirm the appropriate range is accepted for the id you chose: `0x0001-0x00FF` for core, `0x0100-0x01FF` for experimental, `0xFF00-0xFFFF` for user-defined. The validator is the single defense against a stage id being corrupted on disk or injected by a hostile caller; the test suite exercises it via the full pipeline round trip.

## Plugin API forward compat

`include/tdc/plugin.h` exposes the runtime-registration path. Its three functions are `tdc_model_register`, `tdc_xform_register`, and `tdc_entropy_register`; each accepts an id in the user-defined range `0xFF00-0xFFFF` and a pointer to a user-owned vtable. A backend registered this way is visible to `tdc_*_get` immediately, and blocks encoded with its id decode without any library rebuild. The registry capacity is `TDC_PLUGIN_MAX_SLOTS` (16) per stage; registrations above that return `TDC_E_NOMEM`. All registration happens before any concurrent encode or decode call because the registry is not thread-safe; the expected pattern is "register once at startup, use forever".

The static path documented above and the runtime plugin path share one dispatch surface. `tdc_xform_get(id)` first walks the static switch for core and experimental ids, then falls through to the plugin array for user-defined ids. A backend written against this vignette's static pattern is visible through the same pointer type a plugin would return. Swapping the static registration for a plugin registration is mechanical: move the vtable into its own translation unit, drop the internal-header `extern`, drop the registry-switch case, and call `tdc_xform_register(id, &vt)` from the consumer's startup path. The on-disk format is identical either way because the block record only carries the u16 id, not a pointer and not a type.

The plugin API has three constraints that reshape how a backend should be written if a consumer ever wants to register it at runtime. First, the vtable pointer must remain valid for the lifetime of the process. In practice this means the vtable is a file-scope `static const`, or at least a global with no static destruction order. Second, the `id` field inside the vtable must agree with the id passed to `tdc_*_register`; the function cross-checks and returns `TDC_E_INVAL` on mismatch. Third, the same id cannot be registered twice, and an id in the core or experimental range cannot be registered at all; the function returns `TDC_E_INVAL` for both. The static path does not enforce these invariants because there is no runtime registration to validate; a consumer moving a backend across the line needs to add the check mentally.

The rest of the plugin path matches the static walkthrough. The vtable is the same struct. The encode and decode functions have the same signatures. The allocator rule is the same. The `can_inplace`, `is_lossy`, and `accepted_dtypes` fields carry the same meaning. A test covering a runtime-registered backend also looks like the one above, with an extra call to `tdc_*_register` at the top and `tdc_plugin_clear` at the bottom. `tests/test_plugin.c` is the canonical example and exercises the full registration lifecycle plus a round trip through a user-defined entropy coder.

One forward-compatibility guarantee is worth naming explicitly. The plugin API is additive. Adding a new field to one of the three vtables is allowed only if the zero value of the field is a safe default, because older callers zero-initialize the struct and the library has to accept both the old and the new shapes. This is the same forward-compat rule that applies to the per-stage params structs (`tdc_quantize_params`, `tdc_pred2d_params`, etc.) and is stated once in `include/tdc/codec.h`. A backend that follows the rule survives every minor-version bump; one that relies on an uninitialized field behaving a certain way does not.

The long-term trajectory is that the static path becomes thinner and the plugin path becomes a strict superset. Today a backend that ships with tdc goes through the static switch because the switch is trivially cheap, the vtable pointer stays in `.rodata`, and the binary ships with every backend pre-linked. A future release may switch core backends to run through the same plugin array and register them at library init time. The on-disk format does not care; both paths resolve through `tdc_*_get(id)`. Consumers that build against the v0 public headers and use either path keep working across that transition because the surface they depend on (`id -> vtable` lookup plus a stable vtable layout) is preserved on both sides.

## What to read next

A backend that passes the checklist above is ready for a code review. Two adjacent vignettes cover the pieces this one skips. The Performance tuning vignette walks through the bench harness and the knob-calibration methodology; Troubleshooting lists the `tdc_status` codes and the most common decode failures they point at. The [Integration vignette](../integration.md) is the companion document for consumers vendoring the tdc source tree into their own build. The [format spec](../format/on-disk.md) documents the block record layout the new id will appear in.
