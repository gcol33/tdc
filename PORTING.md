# PORTING.md — vectra → tdc extraction log

Working document. Tracks what's done, what's in flight, and what's next
for the vectra → tdc compression-backend extraction. Update at the end of
each session so a `/clear`'d agent can pick up without rereading old turns.

**Repos**
- vectra: `C:\Users\Gilles Colling\Documents\dev\vectra`
- tdc:    `C:\Users\Gilles Colling\Documents\dev\tdc`

**Read first**: `tdc/CLAUDE.md` (frozen contract, allocation rule, stage
layering, no-vectra-back-compat policy, do-not-retry notes).

---

## Master extraction plan

The full vectra compression backend gets pulled out one stage at a time,
then vectra is rewired in a single sweep to call tdc. Each row is a
session boundary — keep them apart so byte-identical extraction doesn't
get tangled with call-site rewiring.

| # | Stage     | What                                              | Status |
|---|-----------|---------------------------------------------------|--------|
| 1 | entropy   | LZ → `tdc/src/entropy/lz.c`                     | DONE   |
| 2 | transform | byte-shuffle → `tdc/src/transform/shuffle.c`      | DONE   |
| 3 | transform | (optional) bit-shuffle → `tdc/src/transform/bitshuffle.c` | later  |
| 4 | transform | quantize → `tdc/src/transform/quantize.c`         | DONE   |
| 5 | transform | zigzag → `tdc/src/transform/zigzag.c`             | DONE   |
| 6 | model     | DELTA_1D → `tdc/src/model/delta1d.c`              | DONE   |
| 7 | model     | DICT_1D → `tdc/src/model/dict.c`                  | DONE   |
| 8 | model     | PRED_2D family → `tdc/src/model/pred2d.c` (+ `plane2d.c`) | DONE   |
| 9 | model     | RAW → `tdc/src/model/raw.c`                       | DONE   |
| 10| api       | `src/api/encode.c` + `src/api/decode.c` driver    | DONE (quantize chain decode landed via TLV in Session 8) |
| 11| format    | `src/format/header.c` + `block_record.c` impls    | DONE (`tdc_strerror` + `tdc_container_header_validate` landed in Session 10; `block_record_validate`, `block_validate`, entropy NONE, TLV xform_params, validity_size already in earlier sessions) |
| 12| vectra    | rip zlib + legacy LZ_VTR out of vectra `.vtr` codec | with #13 |
| 13| wiring    | replace vectra `vtr_codec.c` with tdc calls       | LAST   |

New-in-tdc (no vectra source — write from scratch):
- `model/stack2d.c` (per-slice 2D predictor over a `STACK_2D` block)
- `model/pred3d.c` (true 3D neighborhood predictor)
- `transform/bitshuffle.c` (post-v0; reserved id only)

---

## Conventions established

These are now baked in. Don't relitigate:

1. **No vectra back-compat.** tdc abandons the `.vtr` on-disk format.
   "Byte-identical to vectra" is NOT a goal. Preserve only the inner
   loops; change everything else freely.
2. **Allocation**: `tdc_buffer::realloc_fn` only. POSIX semantics —
   `(NULL,n)` allocs, `(p,0)` frees, `(p,n)` grows. No bare
   malloc/free in `src/**.c`.
3. **Internal headers**: each stage has an `entropy_internal.h` /
   (future) `transform_internal.h` / `model_internal.h` under `src/`,
   listing `extern const tdc_<stage>_vt tdc_<stage>_<name>_vt;` for
   every backend in that stage. registry.c includes them and dispatches
   via switch.
4. **Vtable definition pattern**: `const tdc_<stage>_vt
   tdc_<stage>_<name>_vt = { .id = ..., .name = ..., ... };` (no
   `static`; the extern is in the internal header).
5. **Static helpers** in backend `.c` files: `static`, snake_case,
   `<stage>_<purpose>` prefix (e.g. `lz_decode_fast`,
   `lz_encode_literal_only`).
6. **Encoder fallback policy**: every entropy/transform encode must
   ALWAYS produce a valid output stream of its own format, even on
   incompressible input. Vectra returned NULL and let the caller pick
   a different tag — tdc encodes a literal/passthrough record instead.
   `encode_bound` accounts for this worst case.
7. **Decode error handling**: replace vectra's `vectra_error` (which
   longjmps to R) with `return TDC_E_CORRUPT`. Decoders never abort.

---

## Done — Session 1: LZ

**Files created**
- `tdc/src/entropy/lz.c` — full encoder + decoder behind
  `tdc_entropy_lz_vt`.
- `tdc/src/entropy/entropy_internal.h` — internal extern list for
  entropy vtables.
- `tdc/tests/test_lz_roundtrip.c` — 1 MB mixed buffer round-trip,
  match-heavy compression check, edge cases (0/1/3 bytes, periodic).
- `tdc/CLAUDE.md` — project brief.
- `tdc/PORTING.md` — this file.

**Files modified**
- `tdc/src/core/registry.c` — switch dispatch for `tdc_entropy_get`;
  stub `tdc_model_get`/`tdc_xform_get` returning NULL.
- `tdc/CMakeLists.txt` — added `test_lz_roundtrip` ctest entry.
- `tdc/include/tdc/format.h` — **proper fix**, see "Format header fix"
  below.

**Vectra source extracted** (read-only, no edits to vectra)
- `vectra/src/vtr_codec.c` lines **411–916** — `LZ_*` macros,
  `lz_hash4`, `lz_wildcopy16`, `LZSeq`, `lz_seq_encoded_size`,
  `lz_vtr_compress`, `lz_decompress_fast`, `lz_decompress_safe`,
  `lz_vtr_decompress_core`. Inner matcher and decode loops are
  byte-identical; outer wrapping changed (allocation, fallback policy,
  error returns).

**Deviations from vectra**
1. All scratch (htab 256 KB, seqs, lit_buf) goes through `realloc_fn`.
2. Incompressible / sub-min-match input emits a literal-only LZ stream
   (`n_seq=0, literals_size=src_size, raw bytes`) instead of returning
   NULL. The on-disk LZ format is unchanged — vectra's safe decoder
   already handles that shape via the trailing-literal block.
3. `encode_bound = src_size + 8` (the literal-only fallback size).
4. `lz_decode_safe` returns `TDC_E_CORRUPT` instead of `vectra_error`
   on invalid back-references.

**Test results** (`ctest -C Debug --output-on-failure`)
```
1/2 smoke ............ Passed
2/2 lz_roundtrip .... Passed
[mixed 1MB]      1048576 -> 542499  (51.7%)
[match-heavy]     524288 ->  16157  ( 3.1%)
[literal-heavy]   524288 -> 524296  (100.0% — hits literal-only fallback)
[empty/1B/3B]    edge cases round-trip
[periodic 64B]        64 ->     29  (45.3%)
```

### Format header fix (collateral, not LZ)

`tdc_container_header` declared as 64 bytes in `format.h` but did
NOT pack to 64 bytes under MSVC default alignment — `int64_t
global_dim[3]` forced 4 bytes of pad before it, plus trailing pad after
`uint32_t _reserved1`, total 72 bytes. The `_Static_assert` in the same
header fired on every TU that included it, blocking the entire
`src/format/*.c` and `src/api/*.c` build.

Fix: rearranged the field order so `_reserved1` (4 bytes) sits in the
natural gap at offset 36, pushing `global_dim[3]` to offset 40
(8-aligned). Struct is now naturally 64 bytes on every supported
target. Block record was already correct.

tdc is in active prototype development with no external consumers, so
format changes land in place. Documented in the offset table comment
in `format.h`.

---

## Done — Session 2: byte-shuffle

**Files created**
- `tdc/src/transform/shuffle.c` — full encoder/decoder behind
  `tdc_xform_byte_shuffle_vt`. Scalar shuffle + scalar unshuffle, plus
  the SSE2 and NEON 8-byte fast paths preserved verbatim from vectra.
- `tdc/src/transform/transform_internal.h` — internal extern list for
  transform vtables (mirror of `entropy_internal.h`).
- `tdc/tests/test_byte_shuffle_roundtrip.c` — round-trip on f64/f32/i16/u8
  buffers; explicit spot-check of the forward layout for a 4-elem f64
  input; SIMD boundary tests at n=16 and n=17 (one full SIMD iter + one
  scalar tail); empty / single-element edges; rejection of misaligned
  src_size and `TDC_DT_STRING`.

**Files modified**
- `tdc/src/core/registry.c` — `tdc_xform_get` now switches on the xform
  id and returns `&tdc_xform_byte_shuffle_vt` for `TDC_XFORM_BYTE_SHUFFLE`.
- `tdc/CMakeLists.txt` — added `test_byte_shuffle_roundtrip` ctest entry.

**Vectra source extracted** (read-only, no edits to vectra)
- `vectra/src/vtr_codec.c` lines **220-409** — `byte_unshuffle_8_sse2`,
  `byte_unshuffle_8_neon`, `byte_shuffle`, `byte_unshuffle`,
  `vtr_byte_unshuffle`, `vtr_byte_unshuffle_to`. The SIMD inner loops
  (3-stage SSE2 unpack transpose, NEON vzip cascade) are byte-identical;
  outer wrapping is rewritten for tdc allocation and error conventions.

**Design choices**
1. **No params struct.** Element size is derived from `in_dtype` via
   `tdc_dtype_size()`. The transform takes no params and `codec.h` did
   not need a new `tdc_byte_shuffle_params`. PORTING.md authorized adding
   one but deriving from dtype is cleaner — there is one source of
   truth (the dtype threaded through the chain) instead of two.
2. **`accepted_dtypes` bitmask convention**: bit `(1u << dtype_id)`
   indicates that dtype is accepted. byte-shuffle accepts I8/I16/I32/I64,
   U8/U16/U32/U64, F32/F64; rejects `TDC_DT_STRING`. This is the first
   transform to populate `tdc_xform_vt::accepted_dtypes` so the bitmask
   convention is now baked in for every subsequent transform.
3. **`elem_size == 1` short-circuits to memcpy** in both directions. The
   transposed layout is identical to the input for single-byte elems.
4. **No on-disk header.** Output bytes are exactly `src_size` long; the
   inverse always knows `src_size` and `elem_size` and therefore
   `n_elems = src_size / elem_size`. No length field needed.
5. **SIMD detection inline.** SSE2 is detected via
   `__SSE2__ || _M_X64 || _M_IX86_FP >= 2` (MSVC x64 doesn't define
   `__SSE2__`); NEON via `__ARM_NEON || __ARM_NEON__`. Both fast paths
   are 8-byte-elem only — for f32/i32/i16, the scalar loop runs.
6. **Decode allocates nothing.** It writes directly into the
   caller-supplied `dst`, so there's no `realloc_fn` traffic on the
   decode side. Encode goes through a `shuffle_buf_reserve` helper that
   mirrors `lz_buf_reserve` exactly (will be a candidate for extraction
   into `core/` once a third stage needs it).

**Test results** (`ctest -C Debug --output-on-failure`)
```
1/3 smoke .................... Passed
2/3 lz_roundtrip ............ Passed
3/3 byte_shuffle_roundtrip ... Passed
  [spot-check] f64 layout verified (4 elems, 8-byte lanes)
  [f64 1024 elems]    8192 bytes round-trip OK   (SSE2 fast path)
  [f64 16 elems]       128 bytes round-trip OK   (one full SIMD iter)
  [f64 17 elems]       136 bytes round-trip OK   (SIMD + scalar tail)
  [f32 1024 elems]    4096 bytes round-trip OK   (scalar)
  [i16 512 elems]     1024 bytes round-trip OK   (scalar)
  [u8 256 elems]       256 bytes round-trip OK   (memcpy)
  [empty f64]            0 bytes round-trip OK
  [single f64]           8 bytes round-trip OK
  [reject misaligned] OK
  [reject string dtype] OK
  [encode reject misaligned] OK
```

---

## Done — Session 3: quantize

**Files created**
- `tdc/src/transform/quantize.c` — full encoder/decoder behind
  `tdc_xform_quantize_vt`. f32/f64 → i8/i16/i32 with clamp + NaN
  sentinel policy documented in the file header.
- `tdc/tests/test_quantize_roundtrip.c` — f64→i16 round-trip with
  max-abs-error bound, f64→i8 clamp spot-check, f32→i32 sinusoid
  round-trip, NaN→INT16_MIN sentinel, empty buffer, all rejection
  paths.

**Files modified**
- `tdc/src/transform/transform_internal.h` — added
  `extern const tdc_xform_vt tdc_xform_quantize_vt;`.
- `tdc/src/core/registry.c` — `TDC_XFORM_QUANTIZE` now returns
  `&tdc_xform_quantize_vt`.
- `tdc/CMakeLists.txt` — added `test_quantize_roundtrip` ctest entry,
  with conditional `-lm` link for non-MSVC builds (round/isfinite/isnan).

**Vectra source extracted** (read-only, no edits to vectra)
- `vectra/src/vtr_codec.c` lines **1826-1873** —
  `quantize_float_to_int`, `vtr_dequantize`. The arithmetic kernel
  (`round((v - offset) * scale)`, clamp to target range, divide-and-add
  inverse) is byte-identical; the outer wrapping is rewritten for tdc
  conventions.

**Design decisions**
1. **Targets supported: I8, I16, I32 only.** I64 is rejected because a
   double-precision round trip has only 52 mantissa bits, so quantizing
   into a 64-bit target is meaningless. Matches vectra.
2. **Encoder is infallible** per tdc convention 6: out-of-range values
   are silently clamped to the target's `[min, max]`. NaN encodes to
   the target's min as a sentinel. Inf clamps via the normal comparison
   (round(Inf) = Inf > tmax → tmax). No `overflow_count` return.
3. **Rounding mode**: `round()` (half away from zero). Same as vectra's
   `round()` call. Documented in the file header.
4. **Decode `in_dtype` convention**: `in_dtype` passed to decode is the
   *original* float dtype (the dtype the user wants back), mirroring
   encode. The encoded byte width comes from `params->target`. This
   matches a chain driver that tracks each transform's "user-side"
   dtype across encode and decode. Without this convention the decoder
   would have no way to know whether to write F32 or F64 output.
5. **Bounds stored as `double`**, not `int64_t`. INT8/16/32 min and max
   all fit exactly in IEEE-754 double, so the clamp is two double
   comparisons with no float-conversion warnings in the hot loop.
6. **Element I/O via `memcpy`**, not casts through `int16_t *`. Avoids
   `-Wcast-align` on strict-alignment targets and the `(int16_t *)
   uint8_t *` aliasing pattern. Compiler folds the 1/2/4-byte memcpys.
7. **Validity bitmap NOT handled here.** Vectra's quantize loop skipped
   NA elements; tdc transforms run on a flat byte buffer and the
   validity bitmap is a `tdc_block`-level concept that the model layer
   handles before/after the transform chain.
8. **`accepted_dtypes`**: F32 | F64 only. `is_lossy = 1`.

**Header policy reminder**
`codec.h` was NOT touched. `tdc_quantize_params { scale, offset, target }`
already had everything needed once the decode-`in_dtype` convention
above was settled.

**Test results** (`ctest -C Debug --output-on-failure`)
```
1/4 smoke ................................. Passed
2/4 lz_roundtrip ......................... Passed
3/4 byte_shuffle_roundtrip ................ Passed
4/4 quantize_roundtrip .................... Passed
  [f64->i16 1024 elems] max_err=0.000500 OK   (= 1/(2*scale), as expected)
  [f64->i8 clamp]                          OK
  [f32->i32 256 elems]  max_err=0.000000 OK
  [NaN sentinel]                           OK
  [empty]                                  OK
  [rejections]                             OK
```

**Notes for next session**
- `quantize_buf_reserve` is now the third copy of the same pattern
  (lz, shuffle, quantize). The next stage that needs it is the lift
  threshold — extract into `src/core/buffer.h` (or similar) instead of
  copy-pasting a fourth time.

---

## Done — Session 4: zigzag (+ buf_reserve lift)

**Files created**
- `tdc/src/core/buffer.h` — single source of truth for the
  output-buffer growth helper. `static inline tdc_status
  tdc_buf_reserve(tdc_buffer *, size_t)`. Header-only, lives under
  `src/core/`, not `include/` — internal building block, not part
  of the public ABI. No new `.c` file, so CMakeLists is untouched
  for the lift.
- `tdc/src/transform/zigzag.c` — full encoder/decoder behind
  `tdc_xform_zigzag_vt`. One per-width kernel pair (i8/i16/i32/i64)
  expressed in unsigned arithmetic to side-step both the C11 left-
  shift-of-negative UB and the implementation-defined arithmetic
  right shift on signed operands.
- `tdc/tests/test_zigzag_roundtrip.c` — small-magnitude spot check
  for all four widths, full-range i8 sweep, edge values for
  i16/i32/i64 (0, ±1, type min, type max), empty, rejections
  (float / unsigned / misaligned src / mismatched dst), and a
  registry-wiring assertion.

**Files modified**
- `tdc/src/entropy/lz.c` — `lz_buf_reserve` deleted; both call
  sites now go through `tdc_buf_reserve`. `#include "../core/buffer.h"`.
- `tdc/src/transform/shuffle.c` — `shuffle_buf_reserve` deleted;
  call site now uses `tdc_buf_reserve`. `#include "../core/buffer.h"`.
- `tdc/src/transform/quantize.c` — `quantize_buf_reserve` deleted;
  call site now uses `tdc_buf_reserve`. `#include "../core/buffer.h"`.
- `tdc/src/transform/transform_internal.h` — added
  `extern const tdc_xform_vt tdc_xform_zigzag_vt;`.
- `tdc/src/core/registry.c` — `TDC_XFORM_ZIGZAG` now returns
  `&tdc_xform_zigzag_vt`.
- `tdc/CMakeLists.txt` — added `test_zigzag_roundtrip` ctest entry.
  (`src/transform/zigzag.c` was already in the source list as a stub.)

**Vectra source extracted**
- None. Grep of `vectra/src/vtr_codec.c` for `zigzag` returned
  nothing — vectra never had an explicit zigzag step. tdc's
  implementation is fresh.

**Design decisions**
1. **`tdc_buf_reserve` is `static inline` in a header**, not a
   non-inline function in a new `src/core/buffer.c`. Each TU compiles
   its own copy from the same source line, so there is one definition
   to maintain (single source of truth) without needing a new
   translation unit, a new CMake source list entry, or a new public
   symbol. The `Always Clean, Always Scaling` rule is satisfied: a
   fifth call site is now zero work.
2. **No `tdc_zigzag_params` struct.** Element width is derived from
   `in_dtype` via `tdc_dtype_size()`, same convention as
   byte-shuffle. The output dtype is just the unsigned counterpart of
   the input dtype, computed by a small switch.
3. **Decode `in_dtype` convention**: `in_dtype` passed to decode is
   the *original signed* dtype (matching the encoder's input dtype),
   not the dtype of the on-the-wire bytes. Same rule as quantize.
   Documented at the top of `zigzag.c`.
4. **Unsigned-only kernels.** Forward written as
   `((uN)x << 1) ^ (uN)-(int_least_t)((uN)x >> (N-1))` so the only
   shift on a signed value is `>> (N-1)` of an *unsigned* operand
   (which is `0` or `1` for any input), avoiding both UB on
   `(signed) << 1` of a negative value and the implementation-defined
   arithmetic right shift. Inverse uses the analogous unsigned
   reformulation.
5. **`is_lossy = 0`, `can_inplace = 1`** — the per-element kernel
   reads then writes the same offset before moving on, so encode
   and decode tolerate `src == dst`.
6. **`accepted_dtypes`**: I8 | I16 | I32 | I64 only. Unsigned and
   float dtypes are rejected with `TDC_E_DTYPE` (zigzag of an
   unsigned value is meaningless; floats need quantize first).

**Test results** (`ctest -C Debug --output-on-failure`)
```
1/5 smoke ......................................... Passed
2/5 lz_roundtrip ................................. Passed
3/5 byte_shuffle_roundtrip ........................ Passed
4/5 quantize_roundtrip ............................ Passed
5/5 zigzag_roundtrip .............................. Passed
  [spot-check 0,-1,1,-2,2 across i8/i16/i32/i64] OK
  [i8 sweep 256 elems]      256 bytes round-trip OK
  [i16 edges 8 elems]        16 bytes round-trip OK
  [i32 sweep 1024 elems]   4096 bytes round-trip OK
  [i64 edges 6 elems]        48 bytes round-trip OK
  [empty]            OK
  [rejections]       OK
  [registry wiring]  OK
```

**Notes for next session**
- The buf_reserve lift is done; future encode-side stages just
  `#include "../core/buffer.h"` and call `tdc_buf_reserve`.
- Next stage is the first **model**: `TDC_MODEL_DELTA_1D`. That
  introduces `src/model/` extraction conventions, the model
  internal-header pattern, and the model-side dispatch in
  `registry.c` (currently still a stub returning `NULL`).

---

## Done — Session 5: DELTA_1D model

**Files created**
- `tdc/src/model/model_internal.h` — internal extern list for model
  vtables (mirror of `entropy_internal.h` / `transform_internal.h`).
  First entry is `tdc_model_delta1d_vt`.
- `tdc/src/model/delta1d.c` — full encoder/decoder behind
  `tdc_model_delta1d_vt`. Per-width unsigned-arithmetic kernels
  (1/2/4/8 bytes) covering every fixed-width integer dtype
  (i8/i16/i32/i64/u8/u16/u32/u64).
- `tdc/tests/test_delta1d_roundtrip.c` — round-trip on every accepted
  dtype with full-range edges (type min/max, ±1, 0), explicit residual
  spot-check on a monotonic i64 column, empty block, layout/dtype
  rejections, decoder rejections for non-empty side metadata and
  mismatched residual size, and a registry-wiring assertion.

**Files modified**
- `tdc/src/core/registry.c` — `tdc_model_get` is now a real switch.
  `TDC_MODEL_DELTA_1D` returns `&tdc_model_delta1d_vt`; every other
  id (RAW, DICT_1D, PRED_2D, STACK_2D, PRED_3D) explicitly returns
  NULL with a "not yet extracted" comment so adding the next model
  is one line. Includes `../model/model_internal.h`.
- `tdc/CMakeLists.txt` — added `test_delta1d_roundtrip` ctest entry.
  (`src/model/delta1d.c` was already in the source list as a stub.)

**Vectra source extracted** (read-only, no edits to vectra)
- `vectra/src/vtr_codec.c` lines **1380-1408** — `delta_encode` /
  `delta_decode`. The arithmetic kernel
  (`out[0] = src[0]; out[i] = src[i] - src[i-1]`) is preserved
  conceptually but is now per-width and unsigned-only; the outer
  wrapping is rewritten for tdc allocation, error returns, and the
  `tdc_model_vt::encode` signature.

**Design decisions**
1. **All integer widths, not just i64.** Vectra's delta path was
   `INT64`-only because vectra never had a column of any other
   integer type. tdc has no such restriction and the cost of
   generalizing is one switch with four cases. Same kernel shape,
   width chosen by `tdc_dtype_size(in->dtype)`.
2. **Unsigned arithmetic everywhere.** The kernel is written entirely
   in `uintN_t` so subtraction and addition wrap modulo `2^N`.
   Reinterpreting signed → unsigned is bit-preserving on every
   two's-complement target (which tdc requires). This sidesteps
   the C UB that would otherwise hit on `INT64_MIN - INT64_MAX`
   in a naïve signed implementation.
3. **No side metadata.** The seed value lives in `residual[0]`.
   On encode, `side_out->size = 0`; on decode, `side_size != 0`
   is rejected with `TDC_E_CORRUPT`. This is the cleanest possible
   shape for the model — there is genuinely nothing to carry — and
   it sets the precedent that "absent side metadata" is encoded as
   `size == 0`, not via a NULL pointer or a special tag.
4. **`residual_dtype == in->dtype`.** Delta is a relabelling, not a
   width change. The chain driver downstream (zigzag → byte-shuffle →
   LZ) gets the same width and signedness it would have seen on the
   raw input. This matches the existing convention in `quantize.c`
   and `zigzag.c` that every stage is explicit about its output dtype.
5. **Validity bitmap ignored.** vectra's caller carries validity
   separately and the model just round-trips bytes. NA-aware delta
   is a future model concern; documented in the file header so it's
   not forgotten.
6. **`accepted_layouts` bitmask convention** baked in for models the
   same way `accepted_dtypes` was for transforms — `1u << layout_id`,
   stored as a `uint32_t`. Every subsequent model should use it.
7. **Encoder is infallible on valid input.** Per tdc convention 6,
   delta-of-anything always produces a valid stream because the
   modular kernel never has a "doesn't compress" mode — there is no
   fallback path to think about.

**Test results** (`ctest -C Debug --output-on-failure`)
```
1/6 smoke ............................. Passed
2/6 lz_roundtrip ..................... Passed
3/6 byte_shuffle_roundtrip ............ Passed
4/6 quantize_roundtrip ................ Passed
5/6 zigzag_roundtrip .................. Passed
6/6 delta1d_roundtrip ................. Passed
  [i8 edges]   n=8  bytes=8  round-trip OK
  [i16 edges]  n=10 bytes=20 round-trip OK
  [i32 edges]  n=12 bytes=48 round-trip OK
  [i64 edges]  n=12 bytes=96 round-trip OK
  [u8 edges]   n=8  bytes=8  round-trip OK
  [u16 edges]  n=8  bytes=16 round-trip OK
  [u32 edges]  n=8  bytes=32 round-trip OK
  [u64 edges]  n=8  bytes=64 round-trip OK
  [monotonic i64] residuals match expected gaps OK
  [empty]            OK
  [rejections]       OK
  [registry wiring]  OK
```

**Notes for next session**
- Model internal-header pattern + `tdc_model_get` switch are now in
  place; the next model extraction is exactly the shape of this
  session — add a vtable, an extern in `model_internal.h`, one
  switch case in `registry.c`, a roundtrip test, a CMake entry.
- The unsigned-arithmetic-only convention should propagate to PRED_2D
  when it lands — same UB-avoidance rationale, same bit-preserving
  reinterpretation.

---

## Done — Session 6: RAW + PRED_2D models

This session bundled two model extractions because RAW is trivial and
PRED_2D is the natural next-up among the model stages.

### 6a — RAW

**Files created**
- `tdc/src/model/raw.c` — full encoder/decoder behind `tdc_model_raw_vt`.
  memcpy identity in both directions; accepts every fixed-width numeric
  dtype (i8/i16/i32/i64/u8/u16/u32/u64/f32/f64) and every layout
  (VECTOR_1D, RASTER_2D, STACK_2D, VOLUME_3D).
- `tdc/tests/test_raw_roundtrip.c` — round-trip on every accepted dtype
  (VECTOR_1D), then one example each of RASTER_2D, STACK_2D, VOLUME_3D
  to exercise the multi-rank product. Edge cases (empty, string
  rejection, rank/layout mismatch, decode side-size / residual-size /
  residual-dtype rejections) and a registry-wiring assertion.

**Files modified**
- `tdc/src/model/model_internal.h` — added `extern const tdc_model_vt
  tdc_model_raw_vt;`.
- `tdc/src/core/registry.c` — `TDC_MODEL_RAW` now returns
  `&tdc_model_raw_vt`.
- `tdc/CMakeLists.txt` — added `test_raw_roundtrip` ctest entry.
  (`src/model/raw.c` was already in the source list as a stub.)

**Vectra source extracted**
- None. Vectra had no explicit "raw" model — its codec branched on the
  compression-tag enum and the no-op tag was a literal write of the
  column buffer. tdc treats RAW as a first-class model so the api
  driver can use one dispatch path for everything.

**Design decisions**
1. **All four layouts accepted.** RAW is the only model that promises
   to round-trip *anything* the model dispatcher could feed it. The
   per-layout rank check (`raw_expected_rank`) catches callers who set
   up the block envelope inconsistently — VECTOR_1D with rank 2 is
   `TDC_E_SHAPE`, not silently mishandled.
2. **TDC_DT_STRING rejected.** Strings need `offsets[]` serialization
   through `side_meta`, and v0 is numeric-only (see `types.h` header).
   When DICT_1D lands it will own the string path; RAW stays
   fixed-width forever. Documented in the file header so a future
   reader doesn't try to "generalize" RAW.
3. **No side metadata.** Same convention as delta1d — `side_out->size
   = 0` on encode, non-zero `side_size` on decode is `TDC_E_CORRUPT`.
4. **`raw_n_elems` overflow check uses `INT64_MAX`, not
   `(int64_t)SIZE_MAX`.** The first version cast `SIZE_MAX` to
   `int64_t` which on 64-bit Windows wraps to -1, making the check
   trigger on every non-empty block. Caught by the first run of the
   round-trip test. Documented here so the next "guard against
   overflow" pattern doesn't repeat the bug.

**Test results**
```
1/8 smoke ............................. Passed
...
7/8 raw_roundtrip ..................... Passed
  [i8 vec1d]   bytes=8  round-trip OK
  [i16 vec1d]  bytes=8  round-trip OK
  [i32 vec1d]  bytes=16 round-trip OK
  [i64 vec1d]  bytes=32 round-trip OK
  [u8 vec1d]   bytes=8  round-trip OK
  [u16 vec1d]  bytes=8  round-trip OK
  [u32 vec1d]  bytes=16 round-trip OK
  [u64 vec1d]  bytes=32 round-trip OK
  [f32 vec1d]  bytes=16 round-trip OK
  [f64 vec1d]  bytes=32 round-trip OK
  [i32 raster2d 3x4]    bytes=48  round-trip OK
  [u16 stack2d 2x3x5]   bytes=60  round-trip OK
  [f64 volume3d 2x2x2]  bytes=64  round-trip OK
  [empty] OK
  [rejections] OK
  [registry wiring] OK
```

### 6b — PRED_2D

**Files created**
- `tdc/src/model/pred2d.c` — full encoder/decoder behind
  `tdc_model_pred2d_vt`. Implements LEFT / UP / AVERAGE / PAETH plus
  AUTO. PLANE is intentionally NOT in this file — see "PLANE deferred"
  below.
- `tdc/tests/test_pred2d_roundtrip.c` — round-trip on every supported
  dtype (i8/i16/i32/u8/u16/u32) for every fixed kind on a 4×5
  synthetic raster, edge-value rasters for i16/u8 across all kinds,
  AUTO mode with an in-test argmin verifier (so AUTO and the per-kind
  sweep can't drift apart silently), empty block, rejection paths,
  registry-wiring assertion.

**Files modified**
- `tdc/src/model/model_internal.h` — added `extern const tdc_model_vt
  tdc_model_pred2d_vt;`.
- `tdc/src/core/registry.c` — `TDC_MODEL_PRED_2D` now returns
  `&tdc_model_pred2d_vt`.
- `tdc/CMakeLists.txt` — added `test_pred2d_roundtrip` ctest entry.

**Vectra source extracted** (read-only)
- `vectra/src/vtr_codec.c` lines **1572-1813** — `paeth_predict`,
  `spatial_encode_int`, `spatial_decode_int`, `auto_select_predictor`.
  The four predictor kernels and the auto-select scoring loop are
  preserved conceptually; the outer wrapping is rewritten for tdc
  allocation, error returns, dtype generality (vectra was int64-only),
  and the side metadata convention.

**Design decisions**
1. **dtype set is i8/i16/i32/u8/u16/u32 — no 64-bit.** Vectra was
   int64-only in spirit (its `get_val_i64` widened anything to int64).
   tdc generalizes downward (8/16/32-bit raster imagery is the common
   case) but excludes 64-bit because the internal pred arithmetic is
   int64 and PAETH's `a + b - c` plus `|p - x|` distances would
   overflow at i64 inputs. 64-bit raster columns are vanishingly rare
   in practice. Documented in the file header.
2. **`residual_dtype == in->dtype`.** Vectra's path emitted i64
   residuals from any input width, padding small inputs needlessly.
   tdc keeps the residual stream at the input width and writes back
   modulo 2^N. The kernel does internal arithmetic in int64 (where the
   sums always fit) and the typed `pred2d_store` truncates to the low
   N bits via the unsigned counterpart of the dtype. Round-trips
   correctly because every operation involved is modular at width N
   once written through the typed store. PORTING.md previously said
   PRED_2D would "emit a residual dtype wider than the input"; that
   prediction was wrong — the modular-at-width-N convention from
   delta1d turned out to compose cleanly here too.
3. **AUTO records the resolved kind in side metadata.** 1 byte =
   resolved predictor kind (LEFT/UP/AVERAGE/PAETH, never AUTO). Even
   when the caller passes a non-AUTO kind, the resolved kind is
   recorded so the decoder dispatches identically regardless of how
   the encoder selected it. This is the simplest forward-compatible
   shape and matches the design rule that decoders never re-derive
   encoder choices. Side metadata size is exactly 1; non-1 sizes are
   `TDC_E_CORRUPT`.
4. **AUTO scoring loop is a faithful port of vectra's
   `auto_select_predictor`** — sample of up to 10000 elements, sum of
   absolute residuals per kind, pick argmin. The sample is a row-aligned
   prefix so the predictor sees the same neighborhood structure it
   will see at full size. Allocation-free (no scratch buffer — the
   loop reads `src` directly).
5. **PAETH `(left+up)/2` uses C truncation, not floor division.** Same
   form as vectra. Encode and decode use the *exact same expression*
   so they're bit-identical regardless of sign-handling subtleties.
6. **Type-generic load/store via `pred2d_load`/`pred2d_store`.** memcpy
   in/out of typed locals — no `(int16_t *)uint8_t *` casts, so it's
   `-Wcast-align` clean on strict-alignment targets. Store goes through
   the unsigned counterpart of the dtype (`uint8_t` for i8, etc.) so
   the int->unsigned narrowing is the well-defined modular truncation
   instead of C's implementation-defined signed narrowing.
7. **Rejected `params->kind` outside {AUTO, LEFT, UP, AVERAGE, PAETH}**
   with `TDC_E_INVAL` — that's where PLANE would land if it were
   added to the enum, and we want that to fail loudly until plane2d.c
   is wired up.

**PLANE deferred**

`tdc_pred2d_kind` does NOT currently include `TDC_PRED2D_PLANE`. The
plane predictor needs:
- a different params struct (`tdc_plane2d_params { tile_size }`)
- per-tile coefficient serialization in side_meta (12 bytes per tile)
- a different file (`src/model/plane2d.c`)
- either a new `TDC_PRED2D_PLANE` enum value or a new
  `TDC_MODEL_PLANE_2D` model id

Punted to a follow-up session that drops `plane_encode`/`plane_decode`
from vectra into `src/model/plane2d.c` and wires it through the
preferred dispatch path.

**Test results**
```
1/8 smoke ............................. Passed
...
8/8 pred2d_roundtrip .................. Passed
  [i8/i16/i32/u8/u16/u32 4x5 kind=LEFT/UP/AVG/PAETH]   24 round-trips OK
  [i16 edges 2x4]  4 kinds OK
  [u8 edges 2x4]   4 kinds OK
  [auto argmin]    resolved to kind=4 (PAETH, best_sum=145) as expected
  [empty]          OK
  [rejections]     OK
  [registry wiring] OK
```

**Notes for next session**
- The model internal-header pattern + `tdc_model_get` switch are now
  populated for RAW, DELTA_1D, PRED_2D. DICT_1D, STACK_2D, PRED_3D
  still return NULL.
- The `pred2d_load`/`pred2d_store` typed-load helpers in `pred2d.c`
  are a candidate for extraction into `src/core/typed_io.h` if a
  third site needs them (STACK_2D will). Don't extract preemptively.

---

## Done — Session 7: api/encode + api/decode driver

User picked the order **driver → PLANE_2D → DICT_1D**. This session
landed the driver and the supporting plumbing it needed.

**Files created**
- `tdc/src/api/driver_internal.h` — internal helpers shared between
  encode.c and decode.c. Two responsibilities:
  - Scratch `tdc_buffer` lifecycle (`driver_scratch_init`,
    `driver_scratch_free`) backed by the parent buffer's `realloc_fn`.
  - Static dtype walk for known transforms
    (`driver_xform_out_dtype`). Hardcoded mapping for BYTE_SHUFFLE,
    BIT_SHUFFLE (width-preserving), ZIGZAG (signed→unsigned), and a
    deliberate `0` return for QUANTIZE (params-dependent — see the
    "deferred" section below).
- `tdc/tests/test_pipeline_roundtrip.c` — 8 round-trip cases sweeping
  models × layouts × dtypes × chains × entropy + 3 rejection cases.

**Files modified**
- `tdc/src/api/encode.c` — full implementation of `tdc_encode_block`.
  Validates the block, calls model.encode into a scratch buffer, walks
  the transform chain ping-ponging between two scratch buffers,
  threads the dtype through each stage, calls entropy.encode into a
  payload buffer, then assembles `[ tdc_block_record_v1 |
  side_meta | payload | optional validity ]` into the caller's `out`
  buffer. Sets the LOSSY flag if any transform in the chain reports
  `is_lossy = 1`. Sets HAS_VALIDITY if `src->validity != NULL`.
- `tdc/src/api/decode.c` — full implementation of `tdc_decode_block`.
  Memcpys the 64-byte header (struct layout matches the documented
  little-endian on-disk byte order one-to-one on every supported
  target), validates it, cross-checks dst dtype/layout/shape, walks
  the transform chain forward to build the encoder-side `in_dtype`
  array, runs entropy.decode into scratch, walks the chain in reverse,
  and finally calls model.decode with `dst->dtype` as `residual_dtype`.
- `tdc/src/core/block.c` — implemented `tdc_shape_set_contiguous` and
  `tdc_block_validate`. Cheap structural validators required by the
  encode driver. Both were stubs before this session.
- `tdc/src/format/block_record.c` — implemented
  `tdc_block_record_validate`. Checks magic, version, id ranges,
  layout↔rank consistency, dim non-negativity and overflow, sticky
  chain terminator, and zero trailing dim slots beyond rank.
- `tdc/src/entropy/none.c` — implemented `tdc_entropy_none_vt`
  (memcpy passthrough). Was a stub before this session; needed so
  test cases without LZ can drive the pipeline.
- `tdc/src/entropy/entropy_internal.h` — added
  `extern const tdc_entropy_vt tdc_entropy_none_vt;`.
- `tdc/src/core/registry.c` — `TDC_ENTROPY_NONE` now returns
  `&tdc_entropy_none_vt` (was `NULL`).
- `tdc/CMakeLists.txt` — added `test_pipeline_roundtrip` ctest entry.

**Vectra source extracted** — none. This session is pure plumbing
over the stages already extracted in Sessions 1–6. No vectra inner
loops touched.

**Design decisions**

1. **Scratch buffer plumbing**: every internal allocation goes
   through caller-supplied `realloc_fn`/`user`. Two helper functions
   live in `driver_internal.h` so encode.c and decode.c share one
   pattern. Each encode call allocates four scratch buffers (two
   ping-pong, one side_meta, one payload); each decode call allocates
   two ping-pong buffers. All are freed before return. Zero bare
   `malloc`/`free` in encode.c.

2. **Decode allocator shim** (`driver_libc_realloc` in decode.c):
   `tdc_decode_block` has no `tdc_buffer` parameter and therefore no
   caller-supplied allocator. The driver wraps the C runtime in a
   single shim function so the rest of the pipeline still flows
   through `tdc_buf_reserve` like any other path. This is the **only**
   place in `src/**.c` that calls `realloc/free` directly. It exists
   because the current public decode API does not pass an allocator.
   Lifting this means adding a `tdc_buffer *scratch` arg to
   `tdc_decode_block` — straightforward signature change, not done in
   this session.

3. **Dtype tracking on decode** is the subtle bit. Every transform's
   `decode(in_dtype, ...)` expects the **encoder-side** input dtype
   (per the convention pinned in Sessions 3 and 4 — see the design
   notes for quantize and zigzag). The block record carries
   `dst->dtype` and the chain ids but not intermediate dtypes. The
   driver derives them by forward-walking the chain once via
   `driver_xform_out_dtype`, starting from `dst->dtype` (every v0
   model emits `residual_dtype == in->dtype`). The mapping is
   hardcoded — same approach as the rest of `src/core/registry.c`.
   Adding a new transform means adding a case to both. Acceptable for
   v0's static registry; revisit if/when a runtime plugin API lands.

4. **Quantize-in-chain decode is deferred.** `tdc_quantize_params`
   carries `scale`, `offset`, `target` — three pieces of state needed
   to invert the transform — and the current block record has
   nowhere to store them. The header carries `xform_ids[4]` but no
   `xform_params_size[4]`. Two ways to close this in a later session:

   - **(a)** Extend `side_meta` from a single model-owned blob to a
     multi-section blob `[ model_meta | xform_meta_0 | xform_meta_1 | ... ]`,
     with per-section sizes stored in a small index. The block record
     then needs a `xform_meta_offsets[]` or `xform_meta_size[]` array.
     Cleanest design; expands the record header.
   - **(b)** Add a `tdc_codec_spec *spec` argument to
     `tdc_decode_block` so the caller passes params on the side. Leaves
     the on-disk format alone but breaks the "self-describing block"
     property for any chain that uses parameterized transforms.

   For session 7 the encoder still calls quantize correctly via the
   vtable (it has the params from the spec), but the decoder rejects
   any chain whose forward dtype walk hits a transform with no
   statically-derivable mapping (currently only QUANTIZE) with
   `TDC_E_UNSUPPORTED`. The error surfaces *during* decode rather
   than at encode time so encoded payloads remain valid for whatever
   future decode mechanism lands.

5. **Validity bitmap on decode is deferred.** `tdc_block::validity`
   is `const uint8_t *`, so the driver cannot return reconstructed
   validity bytes through the destination block — the field is
   structurally read-only. The encoder still writes the bitmap to
   disk (so a future API extension can surface it), and
   `TDC_BLOCK_FLAG_HAS_VALIDITY` is honored end-to-end on the encode
   side. The decoder validates that `src_size` covers
   `header + side_meta + payload + validity_bytes` so a corrupted
   record is still caught, but it does not copy the bitmap anywhere.
   Closing this means either dropping the `const` on `tdc_block` or
   adding a separate `tdc_decode_block_validity()` helper.

6. **Header serialization via memcpy of the struct.** The block
   record header is laid out in `tdc/format.h` to pack to exactly 64
   bytes on every supported little-endian target without `#pragma
   pack`. The driver writes the header by `memcpy(out->data, &hdr,
   TDC_BLOCK_HEADER_SIZE)` and reads it the same way. No per-field
   byte twiddling. The `_Static_assert` in `format.h` guarantees the
   struct size at compile time.

7. **out->size after encode** is the total record length (header +
   side_meta + payload + validity), not appended. tdc_encode_block
   writes one block to one buffer; container-level concatenation is
   the caller's job at a higher level. Documented in the encode
   docstring.

8. **Entropy NONE was implemented opportunistically.** It was a
   30-line passthrough stub blocking the test matrix from including
   "no entropy" cases. Implementing it took three obvious functions
   plus an entry in `entropy_internal.h` and registry.c. Deferred
   work would have just re-built the same shim later.

9. **`tdc_block_validate` and `tdc_shape_set_contiguous` were also
   stubs**; the driver could not start without them. Both implemented
   here. The validator checks dtype id range, layout↔rank
   consistency, non-negative dims, n_elems overflow, and the
   STRING/offsets sidecar contract from `tdc/types.h`. The
   contiguous-strides helper fills row-major strides with overflow
   guards and zeros trailing slots beyond rank.

10. **Encoder accepts any registered chain; decoder rejects what it
    cannot statically reverse.** This split keeps the encoder simple
    and forward-compatible: a future decode mechanism (per (a) or (b)
    above) can decode payloads written by the current encoder without
    re-encoding. Encoded files for quantize chains are NOT
    permanently lost — they're just unreadable until the
    transform-params-on-disk decision lands.

**Header policy reminder**
No frozen header touched. The driver is pure implementation behind
existing declarations in `codec.h`, `format.h`, `model.h`,
`transform.h`, `entropy.h`, `types.h`. The only "new code" exposed
publicly is what those headers have always promised would exist.

**Test results** (`ctest -C Debug --output-on-failure`)
```
1/9 smoke ............................. Passed
2/9 lz_roundtrip ..................... Passed
3/9 byte_shuffle_roundtrip ............ Passed
4/9 quantize_roundtrip ................ Passed
5/9 zigzag_roundtrip .................. Passed
6/9 delta1d_roundtrip ................. Passed
7/9 raw_roundtrip ..................... Passed
8/9 pred2d_roundtrip .................. Passed
9/9 pipeline_roundtrip ................ Passed
```

`pipeline_roundtrip` per-case output:
```
tdc_encode_block / tdc_decode_block round-trips:
  RAW + NONE | vec1d i32 n=16                                 OK  enc=128 bytes
  RAW + LZ  | vec1d f64 n=256                                OK  enc=1166 bytes
  RAW + LZ  | vol3d u8 4x4x4                                 OK  enc=136 bytes
  RAW + LZ  | vec1d i16 n=0 (empty)                          OK  enc=72 bytes
  DELTA1D + LZ | vec1d i32 n=4096 ramp                       OK  enc=582 bytes
  DELTA1D + ZIGZAG + BYTE_SHUFFLE + LZ | vec1d i16 n=1024    OK  enc=139 bytes
  PRED2D(PAETH) + BYTE_SHUFFLE + LZ | rast2d u16 64x64       OK  enc=380 bytes
  PRED2D(AUTO) + LZ | rast2d i16 32x32 neg-grad              OK  enc=199 bytes
rejection tests:
  decode rejects dtype mismatch                              OK
  decode rejects truncated src                               OK
  encode rejects unregistered model                          OK
test_pipeline_roundtrip: OK
```

Sanity-check ratios from the working cases:
- `DELTA1D ramp 16384 → 582` bytes: constant deltas of 3 collapse
  through LZ to ~3.5% of raw.
- `DELTA1D + ZIGZAG + SHUFFLE 2048 → 139` bytes: alternating ±5/±3
  sawtooth → small zigzagged residuals → byte-shuffled high lanes
  collapse → 6.8% of raw. Confirms the chain is composing as
  expected.
- `PRED2D(PAETH) 8192 → 380` bytes on a smooth gradient: 4.6%. The
  predictor and the byte-shuffle stage are clearly working in series.

**Notes for next session**

- The two deferred items (transform params on disk; validity bitmap on
  decode) and PLANE_2D (`tdc_pred2d_kind` enum extension OR new
  `TDC_MODEL_PLANE_2D` model id) all touch the public headers. Batch
  them into one pass so the record layout only churns once.
- The driver currently never sees `TDC_XFORM_BIT_SHUFFLE` (reserved,
  unimplemented) or `TDC_ENTROPY_DEFLATE` (zlib link-time). The
  static dtype walk handles bitshuffle as width-preserving so the
  shape of decode is ready when bitshuffle lands.

---

## Done — Session 8: PLANE_2D + TLV xform_params + validity pass-through

Three things landed in one session because they all touched the block
record layout — better to churn the layout once than three times.

**Files created**
- `tdc/src/model/plane2d.c` — `TDC_MODEL_PLANE_2D` predictor. Per-tile
  closed-form 3×3 normal-equations LSQ fit (`pred(lx,ly) = a + b*lx
  + c*ly`), int32 fixed-point coefficients with implied 8-bit fractional
  scale (`round(coef*256)`). Encode and decode use the **same**
  fixed-point arithmetic so the round-trip is bit-exact under modular
  wrap at the input dtype width — same convention pinned in `pred2d.c`.
  Source: `vectra/src/vtr_codec.c:1666-1786` (math kernel preserved
  one-to-one; allocation, dispatch, side-meta layout rewritten for tdc).

**Files modified**
- `tdc/include/tdc/codec.h` — added `TDC_MODEL_PLANE_2D = 0x0007` and
  `tdc_plane2d_params { uint32_t tile_size }`. Picked the
  separate-model-id route over a `TDC_PRED2D_PLANE` enum extension
  because the side meta is structurally incompatible (PRED_2D = 1 byte
  kind tag; PLANE_2D = `u16 tile_size + u32 n_tiles + n_tiles*3*i32`).
- `tdc/include/tdc/format.h` — block record grew from 64 → 80 bytes:
  added `xform_params_size` (u32) and `validity_size` (u32) fields,
  plus a `_reserved1` (u64) tail. Section order on disk is now
  `[ header | side_meta | xform_params | payload | validity ]`. The
  `_Static_assert` at the bottom of the header pins the new size.
- `tdc/src/format/block_record.c` — validator updated for the new
  size, accepts `TDC_MODEL_PLANE_2D` in the model-id range, and
  enforces the flag↔size agreement: `HAS_VALIDITY` set iff
  `validity_size != 0`.
- `tdc/src/api/encode.c` — appends per-slot transform params blobs to
  a TLV section between `side_meta` and `payload`. TLV entry layout
  is `u16 slot_index + u16 blob_length + blob`. Slots whose params
  pointer is NULL contribute nothing. The encoder calls a new
  `driver_xform_params_blob_size()` helper to size each slot.
- `tdc/src/api/decode.c` — reads the TLV section, gathers entries
  into a per-slot pointer/length table, and threads the recovered
  params through `driver_xform_out_dtype()` and the transform decode
  calls. With this, `driver_xform_out_dtype()` can finally do the
  QUANTIZE branch (it now knows `qp->target`), and the decoder
  accepts chains containing QUANTIZE.
- `tdc/src/api/driver_internal.h` — `driver_xform_out_dtype()` now
  takes the per-slot params pointer. This is the change that closes
  the v0 quantize-in-chain gap.
- `tdc/tests/test_pipeline_roundtrip.c` — four new cases (see below).

**Design notes**

1. **TLV instead of a fixed `xform_params_size[4]` array.** Most
   transforms (BYTE_SHUFFLE, ZIGZAG, DELTA-style) carry no params, so
   a fixed `[4]` array would burn 16 bytes per record for the common
   case. TLV adds 4 bytes per *populated* slot (slot_index + length)
   and 0 bytes for empty slots. For chains with one quantize and three
   empty slots that's 4 + sizeof(qp) instead of 4*sizeof(qp). The TLV
   walker is six lines on each side.

2. **PLANE_2D as a separate model id, not a PRED_2D kind.** Considered
   `TDC_PRED2D_PLANE = 5` and dispatching inside `pred2d.c`, but the
   side-meta formats don't share a single layout — every `pred2d.c`
   helper would need a `kind == PLANE` branch on every read/write.
   Cleaner to give PLANE its own vtable that owns its own side-meta.
   Exact dispatch tax: one extra `case` in `registry.c`.

3. **Plane fit kernel: int64 accumulator, int32 stored coefficients.**
   The fit accumulates `s_v, s_vx, s_vy, s_xx, s_xy, s_yy` in int64
   inside the tile, solves the 3x3 system in double, then quantizes
   each coefficient back to `int32` via `round(coef * 256)`. The
   decoder evaluates `pred = (a + b*lx + c*ly + 128) >> 8` in int64
   then narrows to the input dtype with modular wrap. Both sides see
   the *same* int32 coefficients off the wire, so the residual added
   back at decode is the literal byte-wise inverse of what the
   encoder subtracted — bit-exact even when the LSQ fit itself is
   approximate.

4. **PLANE_2D dtype set: i8/i16/i32/u8/u16/u32.** Same width set as
   PRED_2D. Floats rejected (quantize first). 64-bit raster data is
   rare and the int64 accumulator inside the fit can't overflow-guard
   at that width.

5. **Validity bitmap pass-through.** The encoder writes
   `ceil(n_elems / 8)` bytes after the payload when `src->validity`
   is non-NULL. The block record stores `validity_size` explicitly
   so a corrupted record's bounds check still trips. The decoder
   does NOT surface those bytes back through `dst->validity` —
   that field is `const uint8_t *` so the driver structurally
   cannot. Closing that gap means either dropping the `const` on
   `tdc_block` or adding a separate `tdc_decode_block_validity()`
   helper. Punted for now; the on-disk side is honest.

6. **Quantize-in-chain decode gap is now closed.** Session 7 left
   QUANTIZE encoded but not decodable through the driver. With the
   TLV section, `driver_xform_out_dtype()` can read `qp->target`
   from the per-slot params blob and the decoder walks the chain
   correctly. The Session 7 "deferred" rejection branch in
   `driver_xform_out_dtype()` is gone.

**Test results** (`ctest -C Debug --output-on-failure`)
```
1/9 smoke ............................. Passed
2/9 lz_roundtrip ..................... Passed
3/9 byte_shuffle_roundtrip ............ Passed
4/9 quantize_roundtrip ................ Passed
5/9 zigzag_roundtrip .................. Passed
6/9 delta1d_roundtrip ................. Passed
7/9 raw_roundtrip ..................... Passed
8/9 pred2d_roundtrip .................. Passed
9/9 pipeline_roundtrip ................ Passed
```

`pipeline_roundtrip` per-case output (Session 8 additions in the
last four lines):
```
RAW + NONE | vec1d i32 n=16                                  OK  enc=144 bytes
RAW + LZ  | vec1d f64 n=256                                 OK  enc=1182 bytes
RAW + LZ  | vol3d u8 4x4x4                                  OK  enc=152 bytes
RAW + LZ  | vec1d i16 n=0 (empty)                           OK  enc=88 bytes
DELTA1D + LZ | vec1d i32 n=4096 ramp                        OK  enc=598 bytes
DELTA1D + ZIGZAG + BYTE_SHUFFLE + LZ | vec1d i16 n=1024     OK  enc=155 bytes
PRED2D(PAETH) + BYTE_SHUFFLE + LZ | rast2d u16 64x64        OK  enc=396 bytes
PRED2D(AUTO) + LZ | rast2d i16 32x32 neg-grad               OK  enc=215 bytes
PLANE2D + BYTE_SHUFFLE + LZ | rast2d u16 96x64 (split planes, ts=32) OK  enc=547 bytes
PLANE2D + LZ | rast2d i32 50x37 (unaligned, ts=16)          OK  enc=467 bytes
RAW + QUANTIZE(i16) + LZ | vec1d f32 n=256 (TLV)            OK
RAW + LZ | vec1d i32 n=64 + validity bitmap                 OK
```

Compression ratios from the new cases:
- `PLANE2D + BYTE_SHUFFLE + LZ u16 96x64 split planes ts=32 → 547`
  bytes: 4.5% of 12 288. The split-plane raster is constructed with
  two distinct linear gradients on left/right halves, exactly the
  shape PLANE_2D is designed to nail.
- `PLANE2D + LZ i32 50x37 unaligned ts=16 → 467` bytes: 6.3% of
  7 400. Unaligned dims (50 % 16 != 0, 37 % 16 != 0) exercise the
  partial-tile code path. Round-trip is exact.
- `QUANTIZE(i16) f32 n=256` round-trips end-to-end via the TLV
  section — validates the new params plumbing on both encode and
  decode.

**Notes for next session**

- The remaining `tdc_block::validity` `const` is the only structural
  thing still blocking validity bitmap surfacing. It's a one-line
  signature change to `tdc_block` whenever a use case shows up.
- `tdc_block_record` has 8 bytes of `_reserved1` for cheap field
  additions later.

---

## Done — Session 9: DICT_1D

`TDC_MODEL_DICT_1D` is now the first model that consumes `TDC_DT_STRING`
blocks end-to-end. The port from vectra's `try_dict_encode` /
`dict_decode` was straightforward once the open questions from the
"Next — Session 9" stub got resolved by inspection rather than fresh
design:

- STRING storage was already pinned in `include/tdc/types.h`:
  `data` is a packed UTF-8 heap, `offsets` is a `uint32_t [n+1]`
  sidecar (Arrow `Utf8` shape, option (a) from the prior stub). No
  header changes needed.
- `tdc_block_record_validate` already accepts `TDC_DT_STRING` and
  the `TDC_MODEL_DICT_1D` id falls inside its model-id range.
- DICT_1D's residual is **always** `TDC_DT_U32`, regardless of
  dictionary cardinality. Picking a data-dependent index width
  (u8/u16/u32) would force per-block model state on disk for no
  meaningful win — downstream BYTE_SHUFFLE + LZ collapses the high
  zero bytes of the u32 stream in the common low-cardinality case.

**Files written**

- `src/model/dict.c` — full implementation. Encode walks the input
  strings once, builds a unique-string dictionary via an
  open-addressing hash table (FNV-1a, 70% load, double on resize —
  the do-not-retry note in `tdc/CLAUDE.md` rules out Robin Hood),
  and writes a u32 indices residual plus a side-meta dictionary
  blob. Decode validates the side meta (offsets monotonic,
  `offsets[count] == dict_total`, indices in range) and reconstructs
  the heap + offsets. All scratch goes through the
  `side_out->realloc_fn` allocator. No malloc/free.

  Side meta layout (little-endian, fully self-describing):

      u32 dict_count
      u32 dict_total_bytes
      u32 dict_offsets[dict_count + 1]   (last entry == dict_total_bytes)
      u8  dict_data[dict_total_bytes]

  Even an empty block emits `dict_count = 0`, `dict_total = 0`, and
  the trailing `offsets[1] = 0` sentinel — 12 bytes — so the decode
  bounds check is uniform across the empty/non-empty paths.

  Vectra's RLE-on-indices step is intentionally NOT ported. Vectra
  needed it because no entropy stage followed the model; tdc lets
  LZ (or any future entropy backend) handle index runs. Vectra's
  cardinality cap and fallback path are also gone — tdc has no
  fallback path, so the "all unique" case is just the worst case
  of the single dictionary path (n entries in the dict, residual is
  the monotone sequence 0..n-1).

- `tests/test_dict1d_roundtrip.c` — standalone vtable round-trip
  with seven cases: low cardinality (12 rows / 4 unique), all
  unique 600 (forces a hash table resize), single row, empty
  strings interleaved with non-empty, fully empty block,
  dtype/layout rejections, and decoder rejections (truncated side
  meta, out-of-range index). Asserts `residual_dtype == U32` and
  the registry wiring lookup.

**Files modified**

- `src/model/model_internal.h` — added `extern const tdc_model_vt
  tdc_model_dict1d_vt;`.
- `src/core/registry.c` — `TDC_MODEL_DICT_1D` now returns
  `&tdc_model_dict1d_vt` instead of `NULL`.
- `src/api/driver_internal.h` — new `driver_model_residual_dtype()`
  helper. Returns `TDC_DT_U32` for `TDC_MODEL_DICT_1D` and
  `in_dtype` for everything else. Only the decoder needs this; the
  encoder learns the residual dtype directly from the model's
  out-parameter.
- `src/api/decode.c` — the static forward dtype walk now seeds
  `walk` from `driver_model_residual_dtype(hdr.model_id, dst->dtype)`
  instead of `dst->dtype`, and the final `model_vt->decode` call
  passes the same `residual_dtype`. For every model except
  DICT_1D this is identical to the previous behavior.
- `tests/test_pipeline_roundtrip.c` — new
  `case_dict1d_byte_shuffle_lz()` exercising the full driver
  pipeline (DICT_1D + BYTE_SHUFFLE + LZ) on a 16-row string block,
  and the existing `case_encode_rejects_unknown_model` retargeted
  from `TDC_MODEL_DICT_1D` (now registered) to `TDC_MODEL_STACK_2D`
  (still unregistered).
- `CMakeLists.txt` — registers `test_dict1d_roundtrip`.

**Test results**

10/10 ctest passes (was 9/9 + 1 new). The new pipeline case
encodes the 16-row string block in 154 bytes — for context, the
input heap is 87 bytes plus a 68-byte offsets sidecar — i.e.
DICT_1D buys nothing on a 16-row toy block; the win shows up on
larger low-cardinality columns where the dictionary heap stays
constant while the residual collapses. Compression behavior on
realistic inputs will be measured once the vectra wiring step is
in place.

**Notes for next session**

- DICT_1D is the only model that produces a residual whose dtype
  differs from the input block dtype. The
  `driver_model_residual_dtype()` hook will need a new entry for
  every future model with the same property (the `default:` arm
  passes the input dtype through, which is correct for every
  current and historical model except DICT_1D).
- DICT_1D treats the validity bitmap as opaque pass-through, same
  as every other v0 model. NA rows still contribute their (possibly
  garbage) string to the dictionary; the bitmap is what tells the
  caller which rows to ignore.
- The next porting target with substance is the vectra wiring
  itself (master plan rows 12 + 13). Sessions 1–9 cover every model
  / transform / entropy slot needed for vectra's existing column
  types.

---

## Done — Session 10: format/core finalization (master plan row 11)

Cleanup pass over the small surface area that the public headers
declared but earlier sessions had not yet implemented. No new
algorithms; the goal was to leave `format/` and `core/` in a state
where every prototype that ships in `include/tdc/` has a definition
in `src/`.

**Audit findings.** Most of the supposed row-11 work was already
done by sessions 1–9 and just hadn't been crossed off:

- `src/entropy/none.c` — full memcpy passthrough vtable, registered
  in `src/core/registry.c` as `TDC_ENTROPY_NONE`.
- `src/core/block.c` — `tdc_block_validate` and
  `tdc_shape_set_contiguous` fully implemented, including the
  STRING `offsets[]` contract from `types.h`.
- `src/format/block_record.c` — `tdc_block_record_validate` already
  covered magic, version, reserved bytes, flag/size coherence,
  sticky chain terminator, dim sanity, and rank-vs-layout
  consistency.

So row 11 collapsed to two genuinely missing functions plus a test.

**Files written**

- `src/core/error.c` — `tdc_strerror`. Static string lookup over
  every defined `tdc_status`. The default arm returns
  `"unknown tdc_status"` rather than asserting; tdc_strerror is the
  function callers reach for when something has already gone wrong
  and must not itself become a source of new failures.

- `src/format/header.c` — `tdc_container_header_validate`. Cheap
  structural checker mirroring the shape of
  `tdc_block_record_validate`. Validates magic, version, both
  reserved fields, the heterogeneous-flag/global-fields coherence
  rule (the flag and zero-ness of `global_dtype/layout/rank/dim`
  must agree), homogeneous rank-vs-layout match, dim sanity with
  no `n_elems` overflow, trailing dim slots zero, and
  `index_size = n_blocks * TDC_INDEX_ENTRY_SIZE` (or both zero
  for an empty container). Does NOT touch any bytes outside the
  64-byte struct — container read/write of full files is
  intentionally out of scope in v0 because vectra brings its own
  row-group container.

- `tests/test_format_validate.c` — direct coverage for
  `tdc_strerror` (every defined code returns a non-empty string,
  out-of-range falls through to "unknown"),
  `tdc_container_header_validate` (homogeneous + heterogeneous +
  empty happy paths, plus 11 reject cases — bad magic, bad
  version, both reserved fields, het-flag-with-globals,
  rank/layout disagreement, trailing-dim non-zero, negative dim,
  index_size not a multiple of entry size, index_size mismatched
  with n_blocks, and zero-block-with-non-zero-index), and a
  smoke pass over `tdc_block_record_validate` covering bad
  magic, validity-flag-without-size, and the sticky-terminator
  rule.

**Files modified**

- `CMakeLists.txt` — registers `test_format_validate`.

**What stayed as a deliberate stub**

- `src/format/metadata.c` — was originally meant to host
  byte-level packing helpers shared across model side-meta
  blobs. Every model that ships today (DICT_1D, PRED_2D,
  PLANE_2D, DELTA_1D, RAW) does its own serialization in its
  own `.c` file because the layouts are structurally
  incompatible. The shared-utilities abstraction never paid
  rent. Left as a no-op TU rather than deleted so future
  multi-model side-meta helpers (e.g. when STACK_2D and PRED_3D
  land) have an obvious home — but if those don't happen, this
  file should be removed in a later cleanup.
- `src/entropy/{deflate,huffman,fse}.c` — deliberate post-v0
  stubs documented in their own headers. Not part of row 11.

**Test results**

`ctest -C Debug --output-on-failure`: 11/11 pass (was 10/10 + 1
new). Build is clean under MSVC `/W4 /permissive-`.

**Notes for next session**

- Master plan rows 1–11 are now all DONE. The only remaining
  work is rows 12 + 13 — the vectra cleanup and the wiring,
  which the existing PORTING.md sections below already plan in
  detail.
- Vendoring decision: vendor at sync time via a
  `vectra/tools/vendor_tdc.sh` script that copies
  `tdc/include/` + `tdc/src/` into `vectra/src/tdc/`. Run by
  the developer before release commits, not at install time
  (CRAN tarballs must be self-contained). `vectra/src/tdc/`
  is committed to git and shipped in the source tarball.
  `tools/vendor_tdc.sh` itself is `.Rbuildignore`'d so it
  never makes it into the tarball. tdc remains the single
  source of truth in its own repo.

---

## Vectra cleanup: rip zlib + legacy LZ_VTR out of `.vtr`

This is a **vectra-side implementation change**, not a tdc extraction.
It removes dead/legacy codec paths from vectra's `.vtr` format so the
wiring session has a clean slate. Pair this with Session N (wiring)
as a single atomic transition — don't land it standalone, because it
temporarily leaves vectra without a "ratio" compression mode.

### Why

- **No external compression libs in `.vtr`** — `.vtr` codec must be
  native C. zlib stays as an interop dependency for CSV.gz and TIFF
  (separate file formats vectra reads/writes), but it has no business
  inside the columnar `.vtr` codec. This is an explicit user feedback
  rule (`memory/feedback_no_zstd.md`).
- **Legacy `LZ_VTR` (256-byte window LZ77) is dead** — kept only as
  read-side back-compat for old `.vtr` files. Since the wiring session
  abandons old `.vtr` files entirely, the legacy decoder has no
  remaining job.
- Removing both shrinks `vtr_codec.c` significantly and removes the
  "primary path + fallback path" anti-pattern (the `comp_level`
  branches in encode + the compression-tag dispatch in decode).

### What gets removed from vectra

**`vectra/src/vtr_codec.c`**

Encoder side (the `comp_level == VTR_COMPRESS_RATIO` branches):
- Lines ~1990-2003 (`vtr_encode_column` numeric path)
- Lines ~2087-2103 (`vtr_encode_column_qs` quantized/spatial path)
- Lines ~2400-2410 (string column path; check exact line numbers
  before editing — they will have shifted by then)

Decoder side:
- Lines ~2154-2163 — the `VTR_COMP_SHUFFLE_DEFLATE` branch in
  `vtr_decode_column` calling `uncompress()`.

Legacy LZ_VTR:
- Lines ~97-220 — `lz_vtr_compress` / `lz_vtr_decompress` /
  `vtr_lz_decompress_into` definitions. **Delete entirely.**
- Lines ~2138-2141 — the `VTR_COMP_LZ_VTR` branch in `vtr_decode_column`.
- Line 4 — `#include <zlib.h>` at the top of `vtr_codec.c`. The
  remaining file should not need it.

**`vectra/src/vtr_codec.h`**

- Remove `VTR_COMP_LZ_VTR` (0x03) and `VTR_COMP_SHUFFLE_DEFLATE` (0x05)
  defines.
- Remove `VTR_COMPRESS_RATIO` (=2) define from the compression-level enum.
- Remove `vtr_lz_decompress_into` declaration.
- Update the file-top comment block that lists the codec set.

**`vectra/src/Makevars` and `Makevars.win`**

- **Keep `-lz`**. zlib is still needed for `tiff_format.c`,
  `tiff_write.c`, `csv_reader.c` (gzip CSV streams). Only the `.vtr`
  codec stops using it.

**R layer (`vectra/R/write.R` and friends)**

- Find any user-facing `compress = "ratio"` or
  `compress = "deflate"` references. Reduce the accepted set to
  `c("none", "fast")` or remove the parameter if `fast` is the only
  remaining option. The user-visible default is currently `"fast"`
  per `vectra/CLAUDE.md`, so collapsing to a single mode is plausible
  — confirm with the user before doing it.

**Tests**

- `vectra/tests/testthat/test-compression.R` — remove any test cases
  that explicitly request `ratio` / `deflate`.
- Any benchmark scripts in the vectra repo root that compare LZ vs
  deflate become single-mode benchmarks.

### What stays

- `tiff_format.c`, `tiff_write.c`: TIFF deflate encoding via zlib —
  TIFF format requirement, not optional.
- `csv_reader.c`: gzip CSV streams via zlib — CSV interop.
- The `-lz` linker flag in both `Makevars` files.
- The `vtr_codec.h` `VTR_COMP_NONE` and `VTR_COMP_SHUFFLE_LZ` tags.

### Verification

After ripping:
1. `Rscript -e 'devtools::clean_dll(); devtools::load_all()'` —
   compiles with no `<zlib.h>` reference inside `vtr_codec.c`.
2. `grep -n zlib src/vtr_codec.c` — empty.
3. `grep -n 'LZ_VTR\|SHUFFLE_DEFLATE\|VTR_COMPRESS_RATIO' src/` —
   empty (or only in TIFF/CSV files where it's fine).
4. `devtools::test()` — full vectra test suite passes (after the
   compression test cases were updated).
5. `Rscript -e 'devtools::check(args = "--no-manual")'` clean.
6. A round-trip on a fresh `write_vtr() -> tbl() -> collect()` works
   end-to-end (this exercises the LZ path that LZ-in-tdc replaces
   in the wiring session).

### Order of operations with the wiring session

Recommended atomic sequence inside the wiring session:

1. Vendor `tdc/include/` + `tdc/src/` into `vectra/src/tdc/`.
2. Update Makevars to compile the vendored tdc sources.
3. Replace `vtr_lz_*` call sites in `vtr_codec.c` /
   `vtr1.c` / `scan.c` with `tdc_entropy_lz_vt.encode/decode`.
4. **Then** rip the deflate branches and the legacy LZ_VTR path
   from the same files. (Doing it in this order means the wiring
   change and the rip touch overlapping line ranges in one logical
   commit, instead of two passes that conflict.)
5. Update R-layer compression-arg validation.
6. Run vectra tests + benchmarks.
7. Commit.

---

## LAST — Session N: vectra wiring

When all the stages tdc needs are extracted:

1. Vendor `tdc/include/` and `tdc/src/` into `vectra/src/tdc/` (or
   add tdc as a git submodule — decide at that point).
2. Update `vectra/src/Makevars` and `Makevars.win` to compile the
   vendored tdc sources alongside vectra's own.
3. Replace `vtr_lz_compress` / `vtr_lz_decompress_into` /
   `vtr_byte_shuffle*` / etc. call sites in
   `vectra/src/vtr_codec.c`, `vectra/src/vtr1.c`, `vectra/src/scan.c`
   with `tdc_entropy_lz_vt.encode/decode` and friends (or with the
   higher-level `tdc_encode_block` / `tdc_decode_block` if the api
   driver is ready).
4. Old `.vtr` files won't be readable — that's the deal (no
   back-compat).
5. Delete the dead vectra codec code paths.
6. Run the full vectra test suite + benchmarks. The performance
   target is "no worse than the pre-tdc vectra reads". If something
   regresses, profile before changing inner loops — see do-not-retry
   notes in `tdc/CLAUDE.md`.

---

## Build / test loop

```bash
cd /c/Users/Gilles\ Colling/Documents/dev/tdc
cmake -B build                                    # one-time
cmake --build build --config Debug                # rebuild
cd build && ctest -C Debug --output-on-failure
```

Generator: Visual Studio (default on Windows). MSVC `cl.exe` from
VS2022/2026 BuildTools is on PATH via the VS env. Don't bother with
ninja unless gcc/clang shows up — `where gcc` returned nothing on this
box.

Compile flags: `/W4 /permissive-` (MSVC) or `-Wall -Wextra -Wpedantic
-Wshadow -Wstrict-prototypes -Wmissing-prototypes -Wcast-align`
(gcc/clang). Both must be clean before commit.

Test layout: each extracted stage gets `tests/test_<stage>_roundtrip.c`
+ a ctest entry. Tests resolve vtables by `extern` (don't depend on
registry.c so the test still works if registry isn't wired yet).

---

## Open questions / decisions to make later

- **`TDC_ENTROPY_NONE` backend**: declared in `codec.h` but not
  implemented. Trivial (memcpy passthrough). Land it the next time
  we touch entropy/.
- **`tdc_block_validate`** is declared in `types.h` but not defined
  anywhere. Add to `src/core/block.c` when api/encode.c lands and
  actually needs it.
- **`tdc_strerror`**: declared in `error.h`, undefined. Land when
  needed.
- **Dictionary model + `TDC_DT_STRING`**: types.h reserves the dtype
  but no model uses it yet. Decision deferred until DICT_1D
  extraction — the offsets/heap representation needs careful design
  so it composes with transforms that assume fixed-width elements.
- **Container reader/writer**: `src/format/*.c` are stubs. Vectra
  doesn't need them (vectra has its own row-group container); they'll
  matter for tdc-as-standalone-format.

---

## Quick orientation for a fresh agent

If you've just `/clear`'d and are picking this up:

1. Read `tdc/CLAUDE.md` first. It's the contract.
2. Read this file (PORTING.md) to see what's done and what's next.
3. Look at the "Status: NEXT" row in the master plan above. That's
   the session goal.
4. Read the corresponding "Next — Session N" section for the shape
   of the work and the gotchas.
5. The LZ extraction (`tdc/src/entropy/lz.c` +
   `tdc/src/entropy/entropy_internal.h` + the test) is the
   reference template for every subsequent extraction. Mirror its
   structure.
6. Do not touch vectra in extraction sessions. Vectra rewiring is
   its own session at the very end.
