# Integration

tdc is a C11 static library with a frozen public header set under `include/tdc/`. Consumers wire it in one of three ways: as a CMake subproject via `add_subdirectory`, as a vendored source tree compiled by the consumer's own build system, or, eventually, as an installed library via `find_package`. The first two paths are production-ready today and cover every current consumer. The installed-library path is not yet wired. This vignette walks through all three options, the allocator contract the caller has to honor, the threading guarantees the library makes, the ABI policy across `0.x` releases, and a minimal consumer template that rolls every piece into one translation unit.

The four-layer pipeline (model, transform, entropy, storage) never surfaces in the integration story. From the consumer's side, tdc is a single library target that exposes one umbrella header (`tdc.h`) and a set of one-shot entry points. The caller builds a `tdc_block`, picks a `tdc_codec_spec`, hands both to `tdc_encode_block`, and routes the result back through `tdc_decode_block_into` (fixed-width dtypes) or `tdc_decode_block_varlen` (variable-width). Everything in this vignette is scaffolding around those calls.

## Linking options

The three paths correspond to three different consumer situations. A C or C++ project that already uses CMake picks the subproject path and inherits tdc's warning flags, libm link, and platform defines automatically. A consumer whose build system is not CMake (R's Makevars, a custom Makefile, a vendored-source library release) picks the vendoring path and replicates the handful of flags tdc sets in its own CMakeLists. An end user who has tdc installed system-wide picks `find_package` once we ship install rules.

### CMake subproject

Drop the tdc tree into `third_party/tdc/` and reference it from the top-level `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.15)
project(my_app C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# tdc's own options; turn off everything the consumer does not need.
set(TDC_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(TDC_BUILD_BENCH    OFF CACHE BOOL "" FORCE)
set(TDC_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(TDC_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/tdc)

add_executable(my_app src/main.c)
target_link_libraries(my_app PRIVATE tdc)
```

The `tdc` target propagates its public include directory to every downstream target, so `#include "tdc.h"` resolves inside `my_app` without any further configuration. On non-Windows platforms tdc links `libm` publicly, and downstream executables pick up that link automatically. Disabling the test, bench, and tools options cuts configure time in half on fresh checkouts and keeps the downstream build graph small.

Enabling the optional deflate backend requires two extra lines:

```cmake
set(TDC_HAVE_ZLIB ON CACHE BOOL "" FORCE)
find_package(ZLIB REQUIRED)
```

With `TDC_HAVE_ZLIB=ON`, the library's `tdc_entropy_get(TDC_ENTROPY_DEFLATE)` returns a real vtable instead of NULL. Every other backend ships unconditionally.

### Vendoring the source tree

Non-CMake consumers copy `include/` and `src/` under their own tree (`third_party/tdc/` by convention) and compile the sources with their own rules. This is how the vectra R package consumes tdc: the Makevars file adds `third_party/tdc/include` to `PKG_CPPFLAGS` and globs `third_party/tdc/src/**/*.c` into the compilation unit list.

Two flags matter on Linux, and the consumer's build system has to set them explicitly when it is not CMake:

```makefile
PKG_CPPFLAGS += -I./tdc/include -D_POSIX_C_SOURCE=200809L
PKG_LIBS    += -lm
```

`-D_POSIX_C_SOURCE=200809L` exposes `clock_gettime` and `CLOCK_MONOTONIC` under glibc's strict `-std=c11` mode. tdc cannot set this in a header because sibling system headers pull in glibc's `<features.h>` first; the flag has to sit on the compile command line. `-lm` picks up `round`, `log2`, and a handful of other libm entries that the predictor and plane-fit models call through. Both flags are no-ops on macOS, where the headers stay permissive and the linker pulls in libm by default, and no-ops on Windows, where MSVC's CRT already carries them. The CMakeLists carries the same two lines for its own Linux builds, so the requirement is identical; the only difference is that the vendored consumer types them by hand.

A vendored R Makevars typically looks like this:

```makefile
PKG_CPPFLAGS = -I./tdc/include -D_POSIX_C_SOURCE=200809L
PKG_LIBS     = -lm

TDC_OBJECTS = $(patsubst %.c,%.o,$(wildcard tdc/src/*/*.c))
OBJECTS     = $(TDC_OBJECTS) pkg_glue.o
```

Per-translation-unit compilation matches how tdc builds inside CMake; there is no unity-build step and no generated source to run first. A consumer that wants AVX2 can append `-mavx2` on x86_64 and call it done; tdc's AVX2 code paths already sit behind `__AVX2__` guards with scalar fallbacks.

A pure-Makefile consumer on Linux looks nearly identical:

```makefile
CFLAGS  += -std=c11 -O2 -Wall -I./third_party/tdc/include \
           -D_POSIX_C_SOURCE=200809L
LDFLAGS += -lm
SRCS    := $(wildcard third_party/tdc/src/*/*.c) main.c
```

A Windows MSVC consumer replaces the POSIX define with nothing (MSVC's headers already expose `clock_gettime`-equivalents through `<timespec.h>` on recent SDKs, and tdc uses its own `_WIN32` branch when neither is available) and skips the `-lm` line.

### Installed library

`find_package(tdc)` is not yet shipped. We have not committed the install rules because the ABI is still moving between `0.x` minors, and publishing an install target would lock down a symbol set ahead of that stabilization. The plan is to add `install(TARGETS tdc EXPORT tdcTargets ...)` and a generated `tdcConfig.cmake` once vectra's rewire completes and the first `1.0` tag goes out. Until then, consumers use either of the two paths above; both produce the same static library and the same public symbols.

## Allocator wiring

tdc does not allocate memory behind the caller's back. The caller owns every growable buffer the library touches and grows it through a function pointer the caller supplies on the buffer. The contract is a single signature:

```c
void *realloc_fn(void *user, void *ptr, size_t new_size);
```

Three cases cover every call: `(user, NULL, n)` allocates `n` bytes, `(user, p, 0)` frees `p`, `(user, p, n)` grows `p` to `n` bytes and may return a different pointer. Returning NULL on a non-zero request surfaces as `TDC_E_NOMEM`. A stdlib-backed implementation is a dozen lines:

```c
static void *my_realloc(void *user, void *ptr, size_t n) {
    (void)user;
    if (n == 0) { free(ptr); return NULL; }
    return realloc(ptr, n);
}
```

That wiring works for tools, tests, and most first-draft integrations. Larger consumers usually want something with state: a pool allocator, a tracked arena, a debug allocator that records every call site. [`integration_arena.c`](examples/integration_arena.c) wires a bump-allocated arena with peak tracking into both the encode output buffer and the decode scratch buffer of `tdc_decode_block_ex`. Running it on a 16 KiB `i32` ramp:

```
encode arena: 8 allocs, 0 reallocs, 7 frees
encode arena: live=1147120 peak=1196080 out_size=106
decode arena: peak=1229040 memcmp=yes
```

Encoding made eight allocations through the arena, freed seven of them before return (the only live allocation after encode is the output block record), and peaked at roughly 1.17 MiB. The encode-time LZ scratch accounts for most of the peak, and all of it stays inside the one arena the caller passed in. Without a caller-owned allocator there would be no way to account for that memory against a per-request budget. The value of `realloc_fn` is that the peak is visible at all.

The caller frees the output buffer on every path. `tdc_encode_block` returns a grown `tdc_buffer`; the caller owns `buffer.data` until it calls `buffer.realloc_fn(buffer.user, buffer.data, 0)`. The same holds for the decode destination: `tdc_decode_block_into` writes into a buffer the caller allocated, and the caller frees it after use. `tdc_decode_block_ex` adds one more clause: the function borrows the `tdc_buffer *scratch` the caller passes in for internal ping-pong scratch, and on return (success or failure) every scratch allocation has already gone back through the same `realloc_fn`. Callers do not need to clean up anything extra; the function promises balance.

`tdc_decode_block_varlen` is the exception to the "caller allocates the destination" rule. Variable-width blocks cannot size their output heap from the 80-byte block header alone, so the function allocates `dst->data` and `dst->offsets` through the caller's `realloc_fn` during the decode. The caller must pass those in as NULL (the function rejects non-NULL with `TDC_E_INVAL` to prevent leaking the caller's buffer) and must free them with the same `realloc_fn` when done. The library never retains a pointer past return in any path.

## Threading

tdc has no thread-local state and no module-level locks. Every pipeline-stage vtable is a const struct with pure function pointers; one call site owns each `tdc_block`, `tdc_codec_spec`, and `tdc_buffer` for the duration of its encode or decode. The consequence is a simple rule: two threads can safely encode or decode two different blocks concurrently, with no coordination, as long as their `tdc_block`, `tdc_buffer`, and codec-spec pointers do not alias each other.

Sharing a single `tdc_buffer` across threads does not work. The library mutates the buffer's `data`, `size`, and `capacity` fields in place during encode and during every internal grow through `realloc_fn`. Two threads growing the same buffer simultaneously will tear the size field. The same rule applies to the decode destination: `tdc_decode_block_into` writes element-by-element into `dst->data`, and two threads pointing at overlapping destinations race. The fix is the normal parallel-pipelines pattern: one buffer per thread, one block per thread.

Two things survive sharing. The first is read-only source data. A worker pool encoding N blocks from a shared `const` input array can all point at the same source buffer; tdc never writes through `blk->data` during encode. The second is a `tdc_codec_spec`. Specs are POD and the library treats them as immutable; all workers can point at the same spec.

Plugin registration through `tdc_model_register` and its friends in `tdc/plugin.h` is explicitly NOT thread-safe, and every registration call must complete before any concurrent encode or decode starts. The plugin header documents this up front; registering during active workers triggers undefined behavior. The typical pattern is a call in the consumer's initialization path, well before any worker thread spawns.

Allocators passed through `realloc_fn` must themselves be thread-safe across the set of buffers they back. The stdlib `realloc`/`free` pair is thread-safe under every C runtime we target, so the one-liner wrapper is fine for a worker pool out of the box. A tracked arena or pool allocator usually is not, and the consumer either gives each worker its own arena or wraps the allocator in a mutex. tdc never calls the allocator recursively, so a plain pthread_mutex around the four arena entry points is enough; there is no reentrancy to design around.

## ABI and version policy

The public ABI lives under `include/tdc/`. We freeze those headers unless the user authorizes a change. The freeze applies to struct layouts (field order, padding, total size), enum values (numeric id of every `tdc_model_id`, `tdc_xform_id`, `tdc_entropy_id`, `tdc_status`, `tdc_dtype`, `tdc_layout`), function signatures on every public entry point, and the on-disk format described in `tdc/format.h` (container header, block record). Within a `0.x` cycle we may add fields to existing structs behind a zero-default rule: the new field must have a meaningful zero value so old call sites that `memset` the struct keep working. Anything that breaks that rule forces a format version bump and a minor-version tag.

Enum ranges are reserved for who can register which ids. The `0x0001-0x00FF` range is core (shipped with tdc), `0x0100-0x01FF` is experimental (may change without a version bump), `0x0200-0xFEFF` is reserved, and `0xFF00-0xFFFF` is user-defined. The plugin API in `tdc/plugin.h` enforces this at registration time; `tdc_model_register` rejects any id outside the user-defined range with `TDC_E_INVAL`. Consumers that want to ship their own backends today pick ids in the user range and know those ids will never collide with future core additions.

The v0 on-disk format is little-endian only. Supported targets are x86_64 and aarch64; both are little-endian. A big-endian port would require a format version bump and a full pass through every serializer. Cross-endian reads are not a goal and will not become one without an explicit decision. Inside a version, the self-describing block record carries every field needed to decode it, so a block copied out of one container and into another survives without reformatting. The container header is duplicated on purpose; a stranded block is still decodable.

tdc is pre-`1.0`. Between `0.x` minor bumps the API may break, and every break is called out in the release notes. The ABI becomes stable at `1.0`; that tag is gated on vectra's rewire completing and on at least one external consumer shipping against the library.

## Minimal consumer template

[`integration_consumer.c`](examples/integration_consumer.c) is the smallest useful wiring a downstream project usually copies first: one function that encodes an i16 column, one function that decodes it back, a stdlib allocator, and an error path that prints `tdc_strerror` on any non-`TDC_OK` status. Running it on 1,024 signed 16-bit values with mild jitter:

```
raw=2048 encoded=109 decoded=2048 match=yes
```

The encode function constructs a `tdc_block` over the caller's array, installs `cn_realloc` on a fresh `tdc_buffer`, picks a `DELTA_1D + ZIGZAG + BSHUF + LZ` spec, calls `tdc_encode_block`, and returns the grown buffer's `data` pointer as a consumer-owned byte blob. The decode function peeks the record, allocates the destination, and calls `tdc_decode_block_into`. Every failure path runs the buffer pointer through `cn_realloc` with size zero before returning, so there is no leak on any branch. The caller of either function owns the returned pointer and frees it through the same `cn_realloc`.

For consumers that want richer diagnostics than a raw status code, [`integration_error.c`](examples/integration_error.c) shows a pattern: wrap every `tdc` call with a helper that records the call site, the block index, and a pre-formatted message. Running it on a deliberately wrong spec (`DELTA_1D` against a 2D raster) surfaces the mismatch with full context:

```
tdc[encode_row_group_block] block=42 status=5 (layout not accepted by stage)
  fix: pick a model that accepts RASTER_2D (e.g. PRED_2D)
retry: TDC_OK
encoded size: 4177 bytes
```

`tdc_strerror` returns a static literal, so the wrapper can `snprintf` it into a caller-owned message buffer without lifetime concerns. The error-handling table in [the troubleshooting reference](reference/troubleshooting.md) covers every status code with one likely cause and one fix per entry. Consumers that want to dispatch on the error (retry with a different spec, escalate to the caller, propagate through their own error type) have the full enum to switch on; none of the codes require parsing a string.

The three examples cover the full wiring surface. Most real consumers need nothing beyond what they show: a `realloc_fn`, a codec spec picked from the backend walkthroughs, a peek-then-decode pair, and a consumer-side error type that wraps `tdc_status`. The [quickstart](quickstart.md) covers the block abstraction itself; the [backends](backends/models.md) pages cover which codec to pick for which layout; the [on-disk format](format/on-disk.md) spells out what is actually landing in the byte stream once the allocator gives those bytes somewhere to live.
