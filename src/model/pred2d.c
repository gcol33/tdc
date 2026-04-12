/*
 * src/model/pred2d.c
 *
 * TDC_MODEL_PRED_2D — 2D spatial predictor family for RASTER_2D blocks.
 *
 * Predictor kinds (tdc_pred2d_kind):
 *   LEFT     — pred = val[r][c-1]
 *   UP       — pred = val[r-1][c]
 *   AVERAGE  — pred = (left + up) / 2     (C truncation, vectra-compatible)
 *   PAETH    — PNG-style: of {left, up, upleft}, pick the one closest to
 *              the linear predictor p = left + up - upleft
 *   AUTO     — encoder picks the LEFT/UP/AVERAGE/PAETH variant that
 *              minimizes sum of |residual| on a sample of up to 10000
 *              elements
 *
 * PLANE is intentionally NOT in this file. It needs side metadata
 * (per-tile coefficients), a different params struct (tdc_plane2d_params),
 * and is large enough to live in its own src/model/plane2d.c when it
 * lands.
 *
 * Accepted dtypes: i8, i16, i32, u8, u16, u32. (No 64-bit: 64-bit raster
 * imagery is vanishingly rare and the predictor's internal arithmetic
 * cannot guard against overflow at that width.) Floats are rejected —
 * quantize first.
 *
 * Accepted layouts: RASTER_2D only. shape.rank must be 2 with row-major
 * contiguous storage:
 *   ny = shape.dim[0]   (number of rows)
 *   nx = shape.dim[1]   (row length)
 *   idx = row * nx + col
 *
 * Residual dtype: same as input. The hot path uses typed kernels with a
 * signed wide working type (int32_t for 8/16-bit dtypes, int64_t for
 * 32-bit dtypes) — wide enough to keep `a + b - c` overflow-free in the
 * Paeth case. Residuals are written through the unsigned narrow
 * counterpart of the dtype (well-defined modular conversion in C),
 * matching the IDB-avoiding store convention of the original int64
 * reference path. Decode is the modular inverse and round-trips
 * exactly because every operation is modular at width N once written
 * back through the typed store.
 *
 * Side metadata: 1 byte = the resolved predictor kind (LEFT, UP, AVERAGE,
 * or PAETH — never AUTO). Even when the caller passes a non-AUTO kind,
 * the resolved kind is recorded so the decoder dispatches identically
 * regardless of how the encoder selected it. This is the simplest
 * forward-compatible shape and matches the design rule that decoders
 * never re-derive encoder choices.
 *
 * Validity bitmap: ignored, same convention as every other v0 model. The
 * encode driver carries the validity bitmap around the model stage; the
 * model itself only round-trips bytes.
 *
 * Source: vectra/src/vtr_codec.c lines 1572-1813 (paeth_predict,
 * spatial_encode_int, spatial_decode_int, auto_select_predictor). The
 * predictor kernels and the auto-select scoring loop are preserved
 * conceptually; the outer wrapping is rewritten for tdc allocation,
 * error returns, dtype generality, side metadata convention, and per-
 * dtype kernel specialization. Vectra's path was int64-only and
 * longjmp'd on alloc failure.
 */

#include "tdc/model.h"
#include "tdc/codec.h"
#include "model_internal.h"
#include "model_load_store.h"
#include "../core/buffer.h"
#include "../core/float_order.h"
#include "float_pred_helpers.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* ----- Optional SIMD: vectorized UP-predictor row operations -------------- *
 * The UP predictor has no left-pixel dependency — each output element only
 * depends on the element directly above. Both encode (sub) and decode (add)
 * are therefore fully data-parallel within a row and vectorize cleanly.
 *
 * We support four element widths (1/2/4/8 bytes). Byte-level subtraction is
 * NOT used for multi-byte elements: borrows cross byte boundaries, so we
 * dispatch on element width and use the matching integer intrinsic width.
 *
 * Tail handling is inside each helper: the bulk SIMD loop processes as many
 * full vectors as possible; the remainder falls through to the scalar tail.
 *
 * LEFT/AVERAGE/PAETH are not vectorized here because they have a left-pixel
 * read-after-write dependency on decode (each pixel reads dst[col-1] which
 * was just written). Only UP is dependency-free.
 */
#if defined(__SSE2__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#  include <emmintrin.h>
#  define TDC_PRED2D_HAVE_SSE2 1
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#  define TDC_PRED2D_HAVE_NEON 1
#endif

/* --- SSE2 UP helpers ------------------------------------------------------- */
#ifdef TDC_PRED2D_HAVE_SSE2

/* Encode: res[i] = cur[i] - prev[i] at element width `esz`. */
static void pred2d_up_enc_row_sse2(uint8_t *res, const uint8_t *cur,
                                   const uint8_t *prev,
                                   size_t nbytes, size_t esz) {
    size_t i = 0;
    if (esz == 1) {
        for (; i + 16 <= nbytes; i += 16) {
            __m128i a = _mm_loadu_si128((const __m128i *)(cur  + i));
            __m128i b = _mm_loadu_si128((const __m128i *)(prev + i));
            _mm_storeu_si128((__m128i *)(res + i), _mm_sub_epi8(a, b));
        }
        for (; i < nbytes; ++i) res[i] = (uint8_t)(cur[i] - prev[i]);
    } else if (esz == 2) {
        for (; i + 16 <= nbytes; i += 16) {
            __m128i a = _mm_loadu_si128((const __m128i *)(cur  + i));
            __m128i b = _mm_loadu_si128((const __m128i *)(prev + i));
            _mm_storeu_si128((__m128i *)(res + i), _mm_sub_epi16(a, b));
        }
        for (; i < nbytes; i += 2) {
            uint16_t a, b;
            memcpy(&a, cur  + i, 2);
            memcpy(&b, prev + i, 2);
            uint16_t r = (uint16_t)(a - b);
            memcpy(res + i, &r, 2);
        }
    } else if (esz == 4) {
        for (; i + 16 <= nbytes; i += 16) {
            __m128i a = _mm_loadu_si128((const __m128i *)(cur  + i));
            __m128i b = _mm_loadu_si128((const __m128i *)(prev + i));
            _mm_storeu_si128((__m128i *)(res + i), _mm_sub_epi32(a, b));
        }
        for (; i < nbytes; i += 4) {
            uint32_t a, b;
            memcpy(&a, cur  + i, 4);
            memcpy(&b, prev + i, 4);
            uint32_t r = a - b;
            memcpy(res + i, &r, 4);
        }
    } else { /* esz == 8 */
        for (; i + 16 <= nbytes; i += 16) {
            __m128i a = _mm_loadu_si128((const __m128i *)(cur  + i));
            __m128i b = _mm_loadu_si128((const __m128i *)(prev + i));
            _mm_storeu_si128((__m128i *)(res + i), _mm_sub_epi64(a, b));
        }
        for (; i < nbytes; i += 8) {
            uint64_t a, b;
            memcpy(&a, cur  + i, 8);
            memcpy(&b, prev + i, 8);
            uint64_t r = a - b;
            memcpy(res + i, &r, 8);
        }
    }
}

/* Decode: dst[i] = res[i] + prev[i] at element width `esz`. */
static void pred2d_up_dec_row_sse2(uint8_t *dst, const uint8_t *res,
                                   const uint8_t *prev,
                                   size_t nbytes, size_t esz) {
    size_t i = 0;
    if (esz == 1) {
        for (; i + 16 <= nbytes; i += 16) {
            __m128i a = _mm_loadu_si128((const __m128i *)(res  + i));
            __m128i b = _mm_loadu_si128((const __m128i *)(prev + i));
            _mm_storeu_si128((__m128i *)(dst + i), _mm_add_epi8(a, b));
        }
        for (; i < nbytes; ++i) dst[i] = (uint8_t)(res[i] + prev[i]);
    } else if (esz == 2) {
        for (; i + 16 <= nbytes; i += 16) {
            __m128i a = _mm_loadu_si128((const __m128i *)(res  + i));
            __m128i b = _mm_loadu_si128((const __m128i *)(prev + i));
            _mm_storeu_si128((__m128i *)(dst + i), _mm_add_epi16(a, b));
        }
        for (; i < nbytes; i += 2) {
            uint16_t a, b;
            memcpy(&a, res  + i, 2);
            memcpy(&b, prev + i, 2);
            uint16_t r = (uint16_t)(a + b);
            memcpy(dst + i, &r, 2);
        }
    } else if (esz == 4) {
        for (; i + 16 <= nbytes; i += 16) {
            __m128i a = _mm_loadu_si128((const __m128i *)(res  + i));
            __m128i b = _mm_loadu_si128((const __m128i *)(prev + i));
            _mm_storeu_si128((__m128i *)(dst + i), _mm_add_epi32(a, b));
        }
        for (; i < nbytes; i += 4) {
            uint32_t a, b;
            memcpy(&a, res  + i, 4);
            memcpy(&b, prev + i, 4);
            uint32_t r = a + b;
            memcpy(dst + i, &r, 4);
        }
    } else { /* esz == 8 */
        for (; i + 16 <= nbytes; i += 16) {
            __m128i a = _mm_loadu_si128((const __m128i *)(res  + i));
            __m128i b = _mm_loadu_si128((const __m128i *)(prev + i));
            _mm_storeu_si128((__m128i *)(dst + i), _mm_add_epi64(a, b));
        }
        for (; i < nbytes; i += 8) {
            uint64_t a, b;
            memcpy(&a, res  + i, 8);
            memcpy(&b, prev + i, 8);
            uint64_t r = a + b;
            memcpy(dst + i, &r, 8);
        }
    }
}
#endif /* TDC_PRED2D_HAVE_SSE2 */

/* --- NEON UP helpers ------------------------------------------------------- */
#ifdef TDC_PRED2D_HAVE_NEON

static void pred2d_up_enc_row_neon(uint8_t *res, const uint8_t *cur,
                                   const uint8_t *prev,
                                   size_t nbytes, size_t esz) {
    size_t i = 0;
    if (esz == 1) {
        for (; i + 16 <= nbytes; i += 16) {
            uint8x16_t a = vld1q_u8(cur  + i);
            uint8x16_t b = vld1q_u8(prev + i);
            vst1q_u8(res + i, vsubq_u8(a, b));
        }
        for (; i < nbytes; ++i) res[i] = (uint8_t)(cur[i] - prev[i]);
    } else if (esz == 2) {
        for (; i + 16 <= nbytes; i += 16) {
            uint16x8_t a = vld1q_u16((const uint16_t *)(cur  + i));
            uint16x8_t b = vld1q_u16((const uint16_t *)(prev + i));
            vst1q_u16((uint16_t *)(res + i), vsubq_u16(a, b));
        }
        for (; i < nbytes; i += 2) {
            uint16_t a, b;
            memcpy(&a, cur  + i, 2);
            memcpy(&b, prev + i, 2);
            uint16_t r = (uint16_t)(a - b);
            memcpy(res + i, &r, 2);
        }
    } else if (esz == 4) {
        for (; i + 16 <= nbytes; i += 16) {
            uint32x4_t a = vld1q_u32((const uint32_t *)(cur  + i));
            uint32x4_t b = vld1q_u32((const uint32_t *)(prev + i));
            vst1q_u32((uint32_t *)(res + i), vsubq_u32(a, b));
        }
        for (; i < nbytes; i += 4) {
            uint32_t a, b;
            memcpy(&a, cur  + i, 4);
            memcpy(&b, prev + i, 4);
            uint32_t r = a - b;
            memcpy(res + i, &r, 4);
        }
    } else { /* esz == 8 */
        for (; i + 16 <= nbytes; i += 16) {
            uint64x2_t a = vld1q_u64((const uint64_t *)(cur  + i));
            uint64x2_t b = vld1q_u64((const uint64_t *)(prev + i));
            vst1q_u64((uint64_t *)(res + i), vsubq_u64(a, b));
        }
        for (; i < nbytes; i += 8) {
            uint64_t a, b;
            memcpy(&a, cur  + i, 8);
            memcpy(&b, prev + i, 8);
            uint64_t r = a - b;
            memcpy(res + i, &r, 8);
        }
    }
}

static void pred2d_up_dec_row_neon(uint8_t *dst, const uint8_t *res,
                                   const uint8_t *prev,
                                   size_t nbytes, size_t esz) {
    size_t i = 0;
    if (esz == 1) {
        for (; i + 16 <= nbytes; i += 16) {
            uint8x16_t a = vld1q_u8(res  + i);
            uint8x16_t b = vld1q_u8(prev + i);
            vst1q_u8(dst + i, vaddq_u8(a, b));
        }
        for (; i < nbytes; ++i) dst[i] = (uint8_t)(res[i] + prev[i]);
    } else if (esz == 2) {
        for (; i + 16 <= nbytes; i += 16) {
            uint16x8_t a = vld1q_u16((const uint16_t *)(res  + i));
            uint16x8_t b = vld1q_u16((const uint16_t *)(prev + i));
            vst1q_u16((uint16_t *)(dst + i), vaddq_u16(a, b));
        }
        for (; i < nbytes; i += 2) {
            uint16_t a, b;
            memcpy(&a, res  + i, 2);
            memcpy(&b, prev + i, 2);
            uint16_t r = (uint16_t)(a + b);
            memcpy(dst + i, &r, 2);
        }
    } else if (esz == 4) {
        for (; i + 16 <= nbytes; i += 16) {
            uint32x4_t a = vld1q_u32((const uint32_t *)(res  + i));
            uint32x4_t b = vld1q_u32((const uint32_t *)(prev + i));
            vst1q_u32((uint32_t *)(dst + i), vaddq_u32(a, b));
        }
        for (; i < nbytes; i += 4) {
            uint32_t a, b;
            memcpy(&a, res  + i, 4);
            memcpy(&b, prev + i, 4);
            uint32_t r = a + b;
            memcpy(dst + i, &r, 4);
        }
    } else { /* esz == 8 */
        for (; i + 16 <= nbytes; i += 16) {
            uint64x2_t a = vld1q_u64((const uint64_t *)(res  + i));
            uint64x2_t b = vld1q_u64((const uint64_t *)(prev + i));
            vst1q_u64((uint64_t *)(dst + i), vaddq_u64(a, b));
        }
        for (; i < nbytes; i += 8) {
            uint64_t a, b;
            memcpy(&a, res  + i, 8);
            memcpy(&b, prev + i, 8);
            uint64_t r = a + b;
            memcpy(dst + i, &r, 8);
        }
    }
}
#endif /* TDC_PRED2D_HAVE_NEON */

/* ----- Acceptance bitmasks ----------------------------------------------- */

#define PRED2D_ACCEPTED_DTYPES (         \
    TDC_DT_BIT(TDC_DT_I8)  |             \
    TDC_DT_BIT(TDC_DT_I16) |             \
    TDC_DT_BIT(TDC_DT_I32) |             \
    TDC_DT_BIT(TDC_DT_U8)  |             \
    TDC_DT_BIT(TDC_DT_U16) |             \
    TDC_DT_BIT(TDC_DT_U32) |             \
    TDC_DT_BIT(TDC_DT_F16) |             \
    TDC_DT_BIT(TDC_DT_F32) |             \
    TDC_DT_BIT(TDC_DT_F64))

#define PRED2D_ACCEPTED_LAYOUTS TDC_LAYOUT_BIT(TDC_LAYOUT_RASTER_2D)

static int pred2d_dtype_accepted(tdc_dtype dt) {
    return tdc_model_dtype_accepted(PRED2D_ACCEPTED_DTYPES, dt);
}

/* ----- Paeth (typed) ----------------------------------------------------- */
/*
 * Two width-specialized Paeth helpers. Marked inline so the typed sweeps
 * below can fold the call away. The ternary chains compile to branch-
 * free cmov / csel sequences on x86_64 / aarch64 under -O2 and above.
 *
 * Tie-break order matches the original paeth_predict: pa<=pb && pa<=pc
 * → a, else pb<=pc → b, else c. This is the PNG convention and the
 * vectra reference path; preserving it keeps existing on-disk side
 * metadata round-tripping bit-identical regardless of which kernel
 * runs.
 */

static inline int32_t paeth32(int32_t a, int32_t b, int32_t c) {
    /* Algebraic identities for the abs-of-difference terms:
     *   p - a = (a+b-c) - a = b - c
     *   p - b = (a+b-c) - b = a - c
     *   p - c = (a+b-c) - c = (b-c) + (a-c)
     * Hoisting bc and ac as shared subexpressions saves two subtractions
     * and — more importantly — shortens the dependency chain feeding pc.
     */
    int32_t bc = b - c;
    int32_t ac = a - c;
    int32_t pa = bc < 0 ? -bc : bc;
    int32_t pb = ac < 0 ? -ac : ac;
    int32_t pcs = bc + ac;
    int32_t pc = pcs < 0 ? -pcs : pcs;
    int32_t r  = (pb <= pc) ? b : c;
    /* Bitwise & instead of && so neither branch nor short-circuit blocks
     * the compiler from emitting a flat cmov chain. The mask-arithmetic
     * form was tried and benchmarked ~3-5% slower under MSVC /O2 — the
     * compiler already lowers this ternary chain to cmov sequences. */
    return ((pa <= pb) & (pa <= pc)) ? a : r;
}

static inline int64_t paeth64(int64_t a, int64_t b, int64_t c) {
    int64_t p  = a + b - c;
    int64_t pa = p > a ? p - a : a - p;
    int64_t pb = p > b ? p - b : b - p;
    int64_t pc = p > c ? p - c : c - p;
    int64_t r  = (pb <= pc) ? b : c;
    return ((pa <= pb) & (pa <= pc)) ? a : r;
}

/* ----- Typed kernel template --------------------------------------------- */
/*
 * For each accepted dtype we generate a pair of fully-specialized typed
 * encode/decode sweeps. The kind dispatch is hoisted to the top of each
 * function so the inner loops are branch-free on (dtype, kind, boundary).
 * Boundary cases (pixel (0,0), row 0 col >= 1, col 0 row >= 1) are
 * handled in dedicated prologues so the main inner loop has no
 * `(col > 0)` / `(row > 0)` ternaries.
 *
 * Reads from typed pointers to the wide working type W use C's natural
 * integer promotion: signed narrow -> sign-extend, unsigned narrow ->
 * zero-extend. This matches the original int64-based reference path
 * bit-for-bit.
 *
 * Writes go through the unsigned narrow counterpart U of T to keep the
 * narrowing well-defined per C11. Aliasing T<->U through pointer cast
 * is permitted by C11 §6.5p7 (signed/unsigned of the same effective
 * type).
 *
 * Encode has no inter-pixel dependency (all neighbors come from the
 * read-only src buffer) and vectorizes for any predictor. Decode reads
 * `left` from the dst buffer being written, so LEFT/AVERAGE/PAETH
 * decode is inherently scalar; only the UP variant has no left-of
 * dependency and vectorizes.
 */

/* Dispatch shims for the UP SIMD fast path inside DEFINE_PRED2D_TYPED.
 * Exactly one of IF_SSE2/IF_NEON is active; IF_SCALAR is its complement. */
#if defined(TDC_PRED2D_HAVE_SSE2)
#  define IF_SSE2(code)   code
#  define IF_NEON(code)
#  define IF_SCALAR(code)
#elif defined(TDC_PRED2D_HAVE_NEON)
#  define IF_SSE2(code)
#  define IF_NEON(code)   code
#  define IF_SCALAR(code)
#else
#  define IF_SSE2(code)
#  define IF_NEON(code)
#  define IF_SCALAR(code) code
#endif

#define DEFINE_PRED2D_TYPED(SUFFIX, T, U, W, PAETH)                            \
                                                                               \
static void pred2d_enc_##SUFFIX(tdc_pred2d_kind kind,                          \
                                const T *src, T *res,                          \
                                int64_t nx, int64_t ny) {                      \
    if (nx <= 0 || ny <= 0) return;                                            \
                                                                               \
    /* (0,0): all neighbors are 0, residual = src[0]. */                       \
    *(U *)&res[0] = (U)(W)src[0];                                              \
                                                                               \
    /* row 0, col >= 1 — only `left` is in-bounds */                           \
    switch (kind) {                                                            \
        case TDC_PRED2D_LEFT:                                                  \
        case TDC_PRED2D_PAETH:                                                 \
            for (int64_t col = 1; col < nx; ++col) {                           \
                W val  = (W)src[col];                                          \
                W left = (W)src[col - 1];                                      \
                *(U *)&res[col] = (U)(val - left);                             \
            }                                                                  \
            break;                                                             \
        case TDC_PRED2D_UP:                                                    \
            for (int64_t col = 1; col < nx; ++col)                             \
                *(U *)&res[col] = (U)(W)src[col];                              \
            break;                                                             \
        case TDC_PRED2D_AVERAGE:                                               \
            for (int64_t col = 1; col < nx; ++col) {                           \
                W val  = (W)src[col];                                          \
                W left = (W)src[col - 1];                                      \
                *(U *)&res[col] = (U)(val - left / 2);                         \
            }                                                                  \
            break;                                                             \
        default: break;                                                        \
    }                                                                          \
                                                                               \
    /* rows >= 1 */                                                            \
    switch (kind) {                                                            \
        case TDC_PRED2D_LEFT: {                                                \
            for (int64_t row = 1; row < ny; ++row) {                           \
                const T *s0 = src + row * nx;                                  \
                T       *r0 = res + row * nx;                                  \
                *(U *)&r0[0] = (U)(W)s0[0]; /* col 0: left = 0 */              \
                for (int64_t col = 1; col < nx; ++col) {                       \
                    W val  = (W)s0[col];                                       \
                    W left = (W)s0[col - 1];                                   \
                    *(U *)&r0[col] = (U)(val - left);                          \
                }                                                              \
            }                                                                  \
            break;                                                             \
        }                                                                      \
        case TDC_PRED2D_UP: {                                                  \
            for (int64_t row = 1; row < ny; ++row) {                           \
                const T *s0 = src + row * nx;                                  \
                const T *s1 = src + (row - 1) * nx;                            \
                T       *r0 = res + row * nx;                                  \
                size_t   _nb = (size_t)nx * sizeof(T);                         \
                size_t   _esz = sizeof(T);                                     \
                (void)_esz; /* used only when SIMD is enabled */               \
                (void)_nb;                                                     \
IF_SSE2(        pred2d_up_enc_row_sse2((uint8_t *)r0,                          \
                    (const uint8_t *)s0, (const uint8_t *)s1, _nb, _esz);)    \
IF_NEON(        pred2d_up_enc_row_neon((uint8_t *)r0,                          \
                    (const uint8_t *)s0, (const uint8_t *)s1, _nb, _esz);)    \
IF_SCALAR(      for (int64_t col = 0; col < nx; ++col) {                       \
                    W val = (W)s0[col];                                        \
                    W up  = (W)s1[col];                                        \
                    *(U *)&r0[col] = (U)(val - up);                            \
                })                                                             \
            }                                                                  \
            break;                                                             \
        }                                                                      \
        case TDC_PRED2D_AVERAGE: {                                             \
            for (int64_t row = 1; row < ny; ++row) {                           \
                const T *s0 = src + row * nx;                                  \
                const T *s1 = src + (row - 1) * nx;                            \
                T       *r0 = res + row * nx;                                  \
                /* col 0: left = 0 -> pred = up / 2 */                         \
                {                                                              \
                    W val = (W)s0[0];                                          \
                    W up  = (W)s1[0];                                          \
                    *(U *)&r0[0] = (U)(val - up / 2);                          \
                }                                                              \
                for (int64_t col = 1; col < nx; ++col) {                       \
                    W val  = (W)s0[col];                                       \
                    W left = (W)s0[col - 1];                                   \
                    W up   = (W)s1[col];                                       \
                    *(U *)&r0[col] = (U)(val - (left + up) / 2);               \
                }                                                              \
            }                                                                  \
            break;                                                             \
        }                                                                      \
        case TDC_PRED2D_PAETH: {                                               \
            for (int64_t row = 1; row < ny; ++row) {                           \
                const T *s0 = src + row * nx;                                  \
                const T *s1 = src + (row - 1) * nx;                            \
                T       *r0 = res + row * nx;                                  \
                /* col 0: left = upleft = 0 -> paeth(0, up, 0) = up */         \
                {                                                              \
                    W val = (W)s0[0];                                          \
                    W up  = (W)s1[0];                                          \
                    *(U *)&r0[0] = (U)(val - up);                              \
                }                                                              \
                for (int64_t col = 1; col < nx; ++col) {                       \
                    W val    = (W)s0[col];                                     \
                    W left   = (W)s0[col - 1];                                 \
                    W up     = (W)s1[col];                                     \
                    W upleft = (W)s1[col - 1];                                 \
                    W pred   = PAETH(left, up, upleft);                        \
                    *(U *)&r0[col] = (U)(val - pred);                          \
                }                                                              \
            }                                                                  \
            break;                                                             \
        }                                                                      \
        default: break;                                                        \
    }                                                                          \
}                                                                              \
                                                                               \
static void pred2d_dec_##SUFFIX(tdc_pred2d_kind kind,                          \
                                const T *res, T *dst,                          \
                                int64_t nx, int64_t ny) {                      \
    if (nx <= 0 || ny <= 0) return;                                            \
                                                                               \
    *(U *)&dst[0] = (U)(W)res[0];                                              \
                                                                               \
    switch (kind) {                                                            \
        case TDC_PRED2D_LEFT:                                                  \
        case TDC_PRED2D_PAETH:                                                 \
            for (int64_t col = 1; col < nx; ++col) {                           \
                W rv   = (W)res[col];                                          \
                W left = (W)dst[col - 1];                                      \
                *(U *)&dst[col] = (U)(rv + left);                              \
            }                                                                  \
            break;                                                             \
        case TDC_PRED2D_UP:                                                    \
            for (int64_t col = 1; col < nx; ++col)                             \
                *(U *)&dst[col] = (U)(W)res[col];                              \
            break;                                                             \
        case TDC_PRED2D_AVERAGE:                                               \
            for (int64_t col = 1; col < nx; ++col) {                           \
                W rv   = (W)res[col];                                          \
                W left = (W)dst[col - 1];                                      \
                *(U *)&dst[col] = (U)(rv + left / 2);                          \
            }                                                                  \
            break;                                                             \
        default: break;                                                        \
    }                                                                          \
                                                                               \
    switch (kind) {                                                            \
        case TDC_PRED2D_LEFT: {                                                \
            for (int64_t row = 1; row < ny; ++row) {                           \
                const T *r0 = res + row * nx;                                  \
                T       *d0 = dst + row * nx;                                  \
                *(U *)&d0[0] = (U)(W)r0[0];                                    \
                for (int64_t col = 1; col < nx; ++col) {                       \
                    W rv   = (W)r0[col];                                       \
                    W left = (W)d0[col - 1];                                   \
                    *(U *)&d0[col] = (U)(rv + left);                           \
                }                                                              \
            }                                                                  \
            break;                                                             \
        }                                                                      \
        case TDC_PRED2D_UP: {                                                  \
            for (int64_t row = 1; row < ny; ++row) {                           \
                const T *r0 = res + row * nx;                                  \
                T       *d0 = dst + row * nx;                                  \
                const T *d1 = dst + (row - 1) * nx;                            \
                size_t   _nb = (size_t)nx * sizeof(T);                         \
                size_t   _esz = sizeof(T);                                     \
                (void)_esz;                                                    \
                (void)_nb;                                                     \
IF_SSE2(        pred2d_up_dec_row_sse2((uint8_t *)d0,                          \
                    (const uint8_t *)r0, (const uint8_t *)d1, _nb, _esz);)    \
IF_NEON(        pred2d_up_dec_row_neon((uint8_t *)d0,                          \
                    (const uint8_t *)r0, (const uint8_t *)d1, _nb, _esz);)    \
IF_SCALAR(      for (int64_t col = 0; col < nx; ++col) {                       \
                    W rv = (W)r0[col];                                          \
                    W up = (W)d1[col];                                          \
                    *(U *)&d0[col] = (U)(rv + up);                              \
                })                                                             \
            }                                                                  \
            break;                                                             \
        }                                                                      \
        case TDC_PRED2D_AVERAGE: {                                             \
            for (int64_t row = 1; row < ny; ++row) {                           \
                const T *r0 = res + row * nx;                                  \
                T       *d0 = dst + row * nx;                                  \
                const T *d1 = dst + (row - 1) * nx;                            \
                {                                                              \
                    W rv = (W)r0[0];                                           \
                    W up = (W)d1[0];                                           \
                    *(U *)&d0[0] = (U)(rv + up / 2);                           \
                }                                                              \
                for (int64_t col = 1; col < nx; ++col) {                       \
                    W rv   = (W)r0[col];                                       \
                    W left = (W)d0[col - 1];                                   \
                    W up   = (W)d1[col];                                       \
                    *(U *)&d0[col] = (U)(rv + (left + up) / 2);                \
                }                                                              \
            }                                                                  \
            break;                                                             \
        }                                                                      \
        case TDC_PRED2D_PAETH: {                                               \
            for (int64_t row = 1; row < ny; ++row) {                           \
                const T *r0 = res + row * nx;                                  \
                T       *d0 = dst + row * nx;                                  \
                const T *d1 = dst + (row - 1) * nx;                            \
                {                                                              \
                    W rv = (W)r0[0];                                           \
                    W up = (W)d1[0];                                           \
                    *(U *)&d0[0] = (U)(rv + up);                               \
                }                                                              \
                for (int64_t col = 1; col < nx; ++col) {                       \
                    W rv     = (W)r0[col];                                     \
                    W left   = (W)d0[col - 1];                                 \
                    W up     = (W)d1[col];                                     \
                    W upleft = (W)d1[col - 1];                                 \
                    W pred   = PAETH(left, up, upleft);                        \
                    *(U *)&d0[col] = (U)(rv + pred);                           \
                }                                                              \
            }                                                                  \
            break;                                                             \
        }                                                                      \
        default: break;                                                        \
    }                                                                          \
}

DEFINE_PRED2D_TYPED(i8,  int8_t,   uint8_t,  int32_t, paeth32)
DEFINE_PRED2D_TYPED(u8,  uint8_t,  uint8_t,  int32_t, paeth32)
DEFINE_PRED2D_TYPED(i16, int16_t,  uint16_t, int32_t, paeth32)
DEFINE_PRED2D_TYPED(u16, uint16_t, uint16_t, int32_t, paeth32)
DEFINE_PRED2D_TYPED(i32, int32_t,  uint32_t, int64_t, paeth64)
DEFINE_PRED2D_TYPED(u32, uint32_t, uint32_t, int64_t, paeth64)

#undef DEFINE_PRED2D_TYPED
#undef IF_SSE2
#undef IF_NEON
#undef IF_SCALAR

/* ----- 2-row wavefront PAETH decode (16-bit) ----------------------------- *
 * The scalar PAETH decode is dependency-bound: each pixel reads dst[r][c-1]
 * which was just stored, so the inner loop runs at the latency of the
 * paeth+add+store-to-load chain (~6 cycles/pixel measured). The CPU's OoO
 * cannot find independent work because the dependency chain is one row
 * deep.
 *
 * The wavefront breaks the bottleneck by processing TWO consecutive rows
 * in lockstep, with row R+1 running one column behind row R. At iteration
 * `c` we compute pixel (R, c) AND pixel (R+1, c-1). The two computations
 * are *independent within a single iteration*:
 *
 *   lane 0: pred0 = paeth(prev_R,  dRm[c],     dRm[c-1])
 *           out0  = res[R][c]   + pred0
 *           dst[R][c]   = out0
 *
 *   lane 1: pred1 = paeth(prev_R1, prev_R,     prev_R_minus)
 *           out1  = res[R+1][c-1] + pred1
 *           dst[R+1][c-1] = out1
 *
 * Lane 1's `up` = dst[R][c-1] = `prev_R` (carried from the *previous*
 * iteration, NOT lane 0's just-computed out0). Lane 1's `upleft` =
 * dst[R][c-2] = `prev_R_minus`. So the two predictor chains share no
 * data within an iteration — only across iterations via the carried
 * scalars. This unblocks two independent dependency chains for the
 * scheduler and roughly doubles useful IPC on the inner loop.
 *
 * Falls back to the scalar typed kernel when the raster is too small
 * (ny < 3 or nx < 2) or rows are odd (the trailing row is decoded by
 * the scalar kernel after the wavefront pair loop).
 *
 * Why 16-bit only: this is the dtype the bench loses on
 * (`PRED2D(PAETH)+BSHUF+LZ rast2d u16 2048x2048`). i32/u32 uses 64-bit
 * paeth and gains less from the trick because each lane is wider; the
 * 8-bit case is rare in real raster data. If a future bench shows a
 * loser at another width, the same template applies — copy and adjust
 * the storage type.
 */
static void pred2d_dec_u16_paeth_wavefront(const uint16_t *res, uint16_t *dst,
                                            int64_t nx, int64_t ny) {
    /* Row 0: degenerates to LEFT predictor (no row above). */
    dst[0] = res[0];
    for (int64_t c = 1; c < nx; ++c) {
        dst[c] = (uint16_t)(res[c] + dst[c - 1]);
    }

    int64_t row = 1;
    for (; row + 1 < ny; row += 2) {
        const uint16_t *rR  = res + row * nx;
        const uint16_t *rR1 = res + (row + 1) * nx;
        uint16_t       *dR  = dst + row * nx;
        uint16_t       *dR1 = dst + (row + 1) * nx;
        const uint16_t *dRm = dst + (row - 1) * nx;

        /* col 0 of both rows: paeth(0, up, 0) = up. */
        uint16_t out_R_0 = (uint16_t)(rR[0]  + dRm[0]);
        dR[0] = out_R_0;
        uint16_t out_R1_0 = (uint16_t)(rR1[0] + out_R_0);
        dR1[0] = out_R1_0;

        if (nx < 2) continue;

        /* col 1 of row R: standard PAETH (needed before wavefront). */
        uint16_t out_R_1;
        {
            int32_t left   = (int32_t)out_R_0;
            int32_t up     = (int32_t)dRm[1];
            int32_t upleft = (int32_t)dRm[0];
            int32_t pred   = paeth32(left, up, upleft);
            out_R_1 = (uint16_t)((int32_t)rR[1] + pred);
            dR[1]   = out_R_1;
        }

        /* Wavefront: at iter c, compute (R, c) and (R+1, c-1) in parallel. */
        int32_t prev_R       = (int32_t)(uint32_t)out_R_1;   /* dst[R][c-1]   */
        int32_t prev_R_minus = (int32_t)(uint32_t)out_R_0;   /* dst[R][c-2]   */
        int32_t prev_R1      = (int32_t)(uint32_t)out_R1_0;  /* dst[R+1][c-2] */

        for (int64_t c = 2; c < nx; ++c) {
            int32_t up0     = (int32_t)dRm[c];
            int32_t upleft0 = (int32_t)dRm[c - 1];
            int32_t rv0     = (int32_t)rR[c];
            int32_t pred0   = paeth32(prev_R, up0, upleft0);
            int32_t out0    = rv0 + pred0;
            dR[c] = (uint16_t)out0;

            int32_t rv1   = (int32_t)rR1[c - 1];
            int32_t pred1 = paeth32(prev_R1, prev_R, prev_R_minus);
            int32_t out1  = rv1 + pred1;
            dR1[c - 1] = (uint16_t)out1;

            prev_R_minus = prev_R;
            prev_R       = (int32_t)(uint32_t)(uint16_t)out0;
            prev_R1      = (int32_t)(uint32_t)(uint16_t)out1;
        }

        /* Epilogue: (R+1, nx-1) was deferred — left, up, upleft are all
         * carried in scalars from the last wavefront iter. */
        {
            int32_t pred = paeth32(prev_R1, prev_R, prev_R_minus);
            int32_t out  = (int32_t)rR1[nx - 1] + pred;
            dR1[nx - 1] = (uint16_t)out;
        }
    }

    /* Trailing odd row, if any. */
    if (row < ny) {
        const uint16_t *r0 = res + row * nx;
        uint16_t       *d0 = dst + row * nx;
        const uint16_t *d1 = dst + (row - 1) * nx;
        d0[0] = (uint16_t)((int32_t)r0[0] + (int32_t)d1[0]);
        for (int64_t c = 1; c < nx; ++c) {
            int32_t left   = (int32_t)d0[c - 1];
            int32_t up     = (int32_t)d1[c];
            int32_t upleft = (int32_t)d1[c - 1];
            int32_t pred   = paeth32(left, up, upleft);
            d0[c] = (uint16_t)((int32_t)r0[c] + pred);
        }
    }
}

/* ----- 4-row wavefront PAETH decode (16-bit) ----------------------------- *
 * Extends the 2-row wavefront to FOUR staggered rows, quadrupling the
 * independent work available to the OoO scheduler. At iteration `c` the
 * four lanes compute:
 *
 *   lane 0: (R,   c  )   left = L0,  up = dRm[c],   upleft = dRm[c-1]
 *   lane 1: (R+1, c-1)   left = L1,  up = L0_prev,  upleft = P0
 *   lane 2: (R+2, c-2)   left = L2,  up = L1_prev,  upleft = P1
 *   lane 3: (R+3, c-3)   left = L3,  up = L2_prev,  upleft = P2
 *
 * where L[i] is lane i's output from the PREVIOUS iteration (carried),
 * and P[i] is lane i's output from TWO iterations ago. Within a single
 * iteration all four paeth+add chains are independent — they share data
 * only across iterations via the 7 carried scalars (L0..L3, P0..P2).
 *
 * Requires nx >= 4 for the triangular prologue. The caller (dispatch)
 * gates this; rasters with nx < 4 use the 2-row wavefront instead.
 *
 * Trailing 0-3 rows after the last complete quad are decoded with scalar
 * paeth — at most 3 rows out of potentially thousands, so the overhead
 * is negligible.
 */
static void pred2d_dec_u16_paeth_wf4(const uint16_t *res, uint16_t *dst,
                                      int64_t nx, int64_t ny) {
    /* Row 0: LEFT predictor (no row above). */
    dst[0] = res[0];
    for (int64_t c = 1; c < nx; ++c)
        dst[c] = (uint16_t)(res[c] + dst[c - 1]);

    int64_t row = 1;

    /* ---------- 4-row wavefront groups ---------- */
    for (; row + 3 < ny; row += 4) {
        const uint16_t *rR  = res + row       * nx;
        const uint16_t *rR1 = res + (row + 1) * nx;
        const uint16_t *rR2 = res + (row + 2) * nx;
        const uint16_t *rR3 = res + (row + 3) * nx;
        uint16_t       *dR  = dst + row       * nx;
        uint16_t       *dR1 = dst + (row + 1) * nx;
        uint16_t       *dR2 = dst + (row + 2) * nx;
        uint16_t       *dR3 = dst + (row + 3) * nx;
        const uint16_t *dRm = dst + (row - 1) * nx;

        /* Col 0 of all 4 rows: paeth(0, up, 0) = up. */
        int32_t c0_R  = (int32_t)(uint16_t)((int32_t)rR [0] + (int32_t)dRm[0]);
        dR[0] = (uint16_t)c0_R;
        int32_t c0_R1 = (int32_t)(uint16_t)((int32_t)rR1[0] + c0_R);
        dR1[0] = (uint16_t)c0_R1;
        int32_t c0_R2 = (int32_t)(uint16_t)((int32_t)rR2[0] + c0_R1);
        dR2[0] = (uint16_t)c0_R2;
        int32_t c0_R3 = (int32_t)(uint16_t)((int32_t)rR3[0] + c0_R2);
        dR3[0] = (uint16_t)c0_R3;

        /* Carried state: L[i] = last output of lane i (left for next iter),
         * P[i] = output of lane i from one iter earlier (upleft for lane i+1). */
        int32_t L0 = c0_R, L1 = c0_R1, L2 = c0_R2, L3 = c0_R3;
        int32_t P0 = 0, P1 = 0, P2 = 0;

        /* Prologue step 1 (c=1): 1 lane — row R col 1. */
        {
            int32_t pred = paeth32(L0, (int32_t)dRm[1], (int32_t)dRm[0]);
            int32_t v = (int32_t)rR[1] + pred;
            dR[1] = (uint16_t)v;
            P0 = L0;
            L0 = (int32_t)(uint16_t)v;
        }

        /* Prologue step 2 (c=2): 2 lanes — R col 2, R+1 col 1. */
        {
            int32_t pred0 = paeth32(L0, (int32_t)dRm[2], (int32_t)dRm[1]);
            int32_t v0 = (int32_t)rR[2] + pred0;
            dR[2] = (uint16_t)v0;

            int32_t pred1 = paeth32(L1, L0, P0);
            int32_t v1 = (int32_t)rR1[1] + pred1;
            dR1[1] = (uint16_t)v1;

            P0 = L0; P1 = L1;
            L0 = (int32_t)(uint16_t)v0;
            L1 = (int32_t)(uint16_t)v1;
        }

        /* Prologue step 3 (c=3): 3 lanes — R col 3, R+1 col 2, R+2 col 1. */
        {
            int32_t pred0 = paeth32(L0, (int32_t)dRm[3], (int32_t)dRm[2]);
            int32_t v0 = (int32_t)rR[3] + pred0;
            dR[3] = (uint16_t)v0;

            int32_t pred1 = paeth32(L1, L0, P0);
            int32_t v1 = (int32_t)rR1[2] + pred1;
            dR1[2] = (uint16_t)v1;

            int32_t pred2 = paeth32(L2, L1, P1);
            int32_t v2 = (int32_t)rR2[1] + pred2;
            dR2[1] = (uint16_t)v2;

            P0 = L0; P1 = L1; P2 = L2;
            L0 = (int32_t)(uint16_t)v0;
            L1 = (int32_t)(uint16_t)v1;
            L2 = (int32_t)(uint16_t)v2;
        }

        /* Steady state: all 4 lanes, c = 4..nx-1. */
        for (int64_t c = 4; c < nx; ++c) {
            int32_t pred0 = paeth32(L0, (int32_t)dRm[c], (int32_t)dRm[c - 1]);
            int32_t v0 = (int32_t)rR[c] + pred0;
            dR[c] = (uint16_t)v0;

            int32_t pred1 = paeth32(L1, L0, P0);
            int32_t v1 = (int32_t)rR1[c - 1] + pred1;
            dR1[c - 1] = (uint16_t)v1;

            int32_t pred2 = paeth32(L2, L1, P1);
            int32_t v2 = (int32_t)rR2[c - 2] + pred2;
            dR2[c - 2] = (uint16_t)v2;

            int32_t pred3 = paeth32(L3, L2, P2);
            int32_t v3 = (int32_t)rR3[c - 3] + pred3;
            dR3[c - 3] = (uint16_t)v3;

            P0 = L0; P1 = L1; P2 = L2;
            L0 = (int32_t)(uint16_t)v0;
            L1 = (int32_t)(uint16_t)v1;
            L2 = (int32_t)(uint16_t)v2;
            L3 = (int32_t)(uint16_t)v3;
        }

        /* Epilogue: drain the staggered triangle (3 + 2 + 1 pixels). */

        /* 3-lane: (R+1, nx-1), (R+2, nx-2), (R+3, nx-3). */
        {
            int32_t pred1 = paeth32(L1, L0, P0);
            int32_t v1 = (int32_t)rR1[nx - 1] + pred1;
            dR1[nx - 1] = (uint16_t)v1;

            int32_t pred2 = paeth32(L2, L1, P1);
            int32_t v2 = (int32_t)rR2[nx - 2] + pred2;
            dR2[nx - 2] = (uint16_t)v2;

            int32_t pred3 = paeth32(L3, L2, P2);
            int32_t v3 = (int32_t)rR3[nx - 3] + pred3;
            dR3[nx - 3] = (uint16_t)v3;

            P1 = L1; P2 = L2;
            L1 = (int32_t)(uint16_t)v1;
            L2 = (int32_t)(uint16_t)v2;
            L3 = (int32_t)(uint16_t)v3;
        }
        /* 2-lane: (R+2, nx-1), (R+3, nx-2). */
        {
            int32_t pred2 = paeth32(L2, L1, P1);
            int32_t v2 = (int32_t)rR2[nx - 1] + pred2;
            dR2[nx - 1] = (uint16_t)v2;

            int32_t pred3 = paeth32(L3, L2, P2);
            int32_t v3 = (int32_t)rR3[nx - 2] + pred3;
            dR3[nx - 2] = (uint16_t)v3;

            P2 = L2;
            L2 = (int32_t)(uint16_t)v2;
            L3 = (int32_t)(uint16_t)v3;
        }
        /* 1-lane: (R+3, nx-1). */
        {
            int32_t pred3 = paeth32(L3, L2, P2);
            int32_t v3 = (int32_t)rR3[nx - 1] + pred3;
            dR3[nx - 1] = (uint16_t)v3;
        }
    }

    /* Trailing rows (at most 3): scalar PAETH — negligible cost. */
    for (; row < ny; ++row) {
        const uint16_t *rr = res + row * nx;
        uint16_t       *dr = dst + row * nx;
        const uint16_t *da = dst + (row - 1) * nx;
        dr[0] = (uint16_t)((int32_t)rr[0] + (int32_t)da[0]);
        for (int64_t c = 1; c < nx; ++c) {
            int32_t left   = (int32_t)dr[c - 1];
            int32_t up     = (int32_t)da[c];
            int32_t upleft = (int32_t)da[c - 1];
            dr[c] = (uint16_t)((int32_t)rr[c] + paeth32(left, up, upleft));
        }
    }
}

/* Forward declarations for float path (defined after auto-select). */
static void pred2d_encode_float(tdc_dtype dt, tdc_pred2d_kind kind,
                                const uint8_t *src, uint8_t *res,
                                int64_t nx, int64_t ny);
static void pred2d_decode_float(tdc_dtype dt, tdc_pred2d_kind kind,
                                const uint8_t *res, uint8_t *dst,
                                int64_t nx, int64_t ny);

/* ----- Sweep dispatchers ------------------------------------------------- */

static void pred2d_encode_sweep(tdc_dtype dt, tdc_pred2d_kind kind,
                                const uint8_t *src,
                                uint8_t *res,
                                int64_t nx, int64_t ny) {
    if (tdc_dtype_is_float(dt)) {
        pred2d_encode_float(dt, kind, src, res, nx, ny);
        return;
    }
    switch (dt) {
        case TDC_DT_I8:  pred2d_enc_i8 (kind, (const int8_t   *)src, (int8_t   *)res, nx, ny); break;
        case TDC_DT_U8:  pred2d_enc_u8 (kind, (const uint8_t  *)src, (uint8_t  *)res, nx, ny); break;
        case TDC_DT_I16: pred2d_enc_i16(kind, (const int16_t  *)src, (int16_t  *)res, nx, ny); break;
        case TDC_DT_U16: pred2d_enc_u16(kind, (const uint16_t *)src, (uint16_t *)res, nx, ny); break;
        case TDC_DT_I32: pred2d_enc_i32(kind, (const int32_t  *)src, (int32_t  *)res, nx, ny); break;
        case TDC_DT_U32: pred2d_enc_u32(kind, (const uint32_t *)src, (uint32_t *)res, nx, ny); break;
        default: break;
    }
}

static void pred2d_decode_sweep(tdc_dtype dt, tdc_pred2d_kind kind,
                                const uint8_t *res,
                                uint8_t *dst,
                                int64_t nx, int64_t ny) {
    /* u16 PAETH is the bench loser; route it through the wavefront kernel
     * which runs two independent dependency chains in parallel. The
     * fallback nx<2 case is handled by the wavefront itself; the ny<3
     * case still calls the wavefront which produces a one-row left-pred
     * row 0 and (if ny==2) a scalar trailing row. Round-trip is verified
     * by tests/test_pred2d_roundtrip.c on small rasters. */
    if (tdc_dtype_is_float(dt)) {
        pred2d_decode_float(dt, kind, res, dst, nx, ny);
        return;
    }
    if (dt == TDC_DT_U16 && kind == TDC_PRED2D_PAETH && nx > 0 && ny > 0) {
        /* 4-row wavefront for large rasters (4-way ILP); 2-row wavefront
         * for smaller ones. The 4-row kernel needs nx >= 4 for its
         * triangular prologue and at least one full quad after row 0
         * (ny >= 5). Below that threshold the 2-row version is still a
         * strict improvement over scalar. */
        if (nx >= 4 && ny >= 5) {
            pred2d_dec_u16_paeth_wf4((const uint16_t *)res,
                                      (uint16_t *)dst, nx, ny);
        } else {
            pred2d_dec_u16_paeth_wavefront((const uint16_t *)res,
                                            (uint16_t *)dst, nx, ny);
        }
        return;
    }
    switch (dt) {
        case TDC_DT_I8:  pred2d_dec_i8 (kind, (const int8_t   *)res, (int8_t   *)dst, nx, ny); break;
        case TDC_DT_U8:  pred2d_dec_u8 (kind, (const uint8_t  *)res, (uint8_t  *)dst, nx, ny); break;
        case TDC_DT_I16: pred2d_dec_i16(kind, (const int16_t  *)res, (int16_t  *)dst, nx, ny); break;
        case TDC_DT_U16: pred2d_dec_u16(kind, (const uint16_t *)res, (uint16_t *)dst, nx, ny); break;
        case TDC_DT_I32: pred2d_dec_i32(kind, (const int32_t  *)res, (int32_t  *)dst, nx, ny); break;
        case TDC_DT_U32: pred2d_dec_u32(kind, (const uint32_t *)res, (uint32_t *)dst, nx, ny); break;
        default: break;
    }
}

/* ----- Float predictor path ---------------------------------------------- */
/*
 * Float dtypes use ordered-integer mapping for lossless round-trip:
 *   1. Load float bits → ordered uint (preserves numerical ordering)
 *   2. Compute prediction in uint64 space (same predictor logic as integers)
 *   3. Store residual = value_ordered - pred_ordered (unsigned wrap)
 *
 * Uses uint64_t as the working type for all float widths. For f64, the
 * Paeth formula casts to int64 — the relative distances are preserved
 * because the ordered mapping is monotone. Overflow of p = a + b - c in
 * int64 is theoretically possible but requires neighboring pixels to span
 * the full float64 range, which never arises in real raster data.
 *
 * The switch-on-dtype load/store is slower than the typed macro kernels
 * used for integers but correct and maintainable. Optimization (per-width
 * typed float kernels) can be added later if profiling shows a need.
 */

/* Float prediction helpers: shared implementation in float_pred_helpers.h. */
#define pred2d_load_ordered        tdc_float_load_ordered
#define pred2d_store_ordered_residual tdc_float_store_ordered_residual
#define pred2d_load_residual       tdc_float_load_residual
#define pred2d_store_float         tdc_float_store_float
#define paeth_ordered              tdc_float_paeth_ordered

static inline uint64_t pred2d_float_predict(tdc_pred2d_kind kind,
                                            uint64_t left, uint64_t up,
                                            uint64_t upleft) {
    switch (kind) {
        case TDC_PRED2D_LEFT:    return left;
        case TDC_PRED2D_UP:      return up;
        case TDC_PRED2D_AVERAGE: return (left + up) / 2u;
        case TDC_PRED2D_PAETH:   return paeth_ordered(left, up, upleft);
        default:                 return 0;
    }
}

static void pred2d_encode_float(tdc_dtype dt, tdc_pred2d_kind kind,
                                const uint8_t *src, uint8_t *res,
                                int64_t nx, int64_t ny) {
    if (nx <= 0 || ny <= 0) return;

    /* (0,0): pred = 0. */
    uint64_t val00 = pred2d_load_ordered(dt, src, 0);
    pred2d_store_ordered_residual(dt, res, 0, val00);

    /* row 0, col >= 1: only `left` is in-bounds. */
    for (int64_t col = 1; col < nx; ++col) {
        uint64_t val  = pred2d_load_ordered(dt, src, col);
        uint64_t left = pred2d_load_ordered(dt, src, col - 1);
        uint64_t pred = (kind == TDC_PRED2D_UP) ? 0 :
                        (kind == TDC_PRED2D_AVERAGE) ? left / 2u : left;
        pred2d_store_ordered_residual(dt, res, col, val - pred);
    }

    /* rows >= 1. */
    for (int64_t row = 1; row < ny; ++row) {
        int64_t r0 = row * nx;
        /* col 0: only `up` is in-bounds. */
        {
            uint64_t val = pred2d_load_ordered(dt, src, r0);
            uint64_t up  = pred2d_load_ordered(dt, src, r0 - nx);
            uint64_t pred = (kind == TDC_PRED2D_LEFT) ? 0 :
                            (kind == TDC_PRED2D_AVERAGE) ? up / 2u : up;
            pred2d_store_ordered_residual(dt, res, r0, val - pred);
        }
        for (int64_t col = 1; col < nx; ++col) {
            int64_t i      = r0 + col;
            uint64_t val    = pred2d_load_ordered(dt, src, i);
            uint64_t left   = pred2d_load_ordered(dt, src, i - 1);
            uint64_t up     = pred2d_load_ordered(dt, src, i - nx);
            uint64_t upleft = pred2d_load_ordered(dt, src, i - nx - 1);
            uint64_t pred   = pred2d_float_predict(kind, left, up, upleft);
            pred2d_store_ordered_residual(dt, res, i, val - pred);
        }
    }
}

static void pred2d_decode_float(tdc_dtype dt, tdc_pred2d_kind kind,
                                const uint8_t *res, uint8_t *dst,
                                int64_t nx, int64_t ny) {
    if (nx <= 0 || ny <= 0) return;

    /* (0,0): pred = 0 → ordered = residual. */
    uint64_t val00 = pred2d_load_residual(dt, res, 0);
    pred2d_store_float(dt, dst, 0, val00);

    /* row 0, col >= 1. */
    for (int64_t col = 1; col < nx; ++col) {
        uint64_t rv   = pred2d_load_residual(dt, res, col);
        uint64_t left = pred2d_load_ordered(dt, dst, col - 1);
        uint64_t pred = (kind == TDC_PRED2D_UP) ? 0 :
                        (kind == TDC_PRED2D_AVERAGE) ? left / 2u : left;
        uint64_t ordered = rv + pred;
        pred2d_store_float(dt, dst, col, ordered);
    }

    /* rows >= 1. */
    for (int64_t row = 1; row < ny; ++row) {
        int64_t r0 = row * nx;
        /* col 0. */
        {
            uint64_t rv = pred2d_load_residual(dt, res, r0);
            uint64_t up = pred2d_load_ordered(dt, dst, r0 - nx);
            uint64_t pred = (kind == TDC_PRED2D_LEFT) ? 0 :
                            (kind == TDC_PRED2D_AVERAGE) ? up / 2u : up;
            pred2d_store_float(dt, dst, r0, rv + pred);
        }
        for (int64_t col = 1; col < nx; ++col) {
            int64_t i      = r0 + col;
            uint64_t rv     = pred2d_load_residual(dt, res, i);
            uint64_t left   = pred2d_load_ordered(dt, dst, i - 1);
            uint64_t up     = pred2d_load_ordered(dt, dst, i - nx);
            uint64_t upleft = pred2d_load_ordered(dt, dst, i - nx - 1);
            uint64_t pred   = pred2d_float_predict(kind, left, up, upleft);
            pred2d_store_float(dt, dst, i, rv + pred);
        }
    }
}

/* ----- Auto-select (cold path) ------------------------------------------- */
/*
 * Score each of LEFT/UP/AVERAGE/PAETH on a prefix of up to 10000 elements
 * and return the kind with the smallest sum of absolute residuals. Same
 * heuristic as vectra (vtr_codec.c:auto_select_predictor) — cheap, makes
 * no allocations, runs once per encode call. Kept on the int64 reference
 * path because the prefix is small enough that specialization wouldn't
 * change end-to-end throughput.
 */

static inline int64_t pred2d_load(tdc_dtype dt, const uint8_t *base, int64_t i) {
    return tdc_model_load(dt, base, i);
}

static int64_t pred2d_compute(tdc_pred2d_kind kind,
                              int64_t left, int64_t up, int64_t upleft) {
    switch (kind) {
        case TDC_PRED2D_LEFT:    return left;
        case TDC_PRED2D_UP:      return up;
        case TDC_PRED2D_AVERAGE: return (left + up) / 2;
        case TDC_PRED2D_PAETH:   return paeth64(left, up, upleft);
        default:                 return 0;
    }
}

#define PRED2D_AUTO_SAMPLE 10000

static tdc_pred2d_kind pred2d_auto_select(tdc_dtype dt, const uint8_t *src,
                                          int64_t nx, int64_t ny) {
    int64_t n = nx * ny;
    int64_t sample_n = n < PRED2D_AUTO_SAMPLE ? n : PRED2D_AUTO_SAMPLE;
    /* sample is a row-aligned prefix so the predictor sees the same
     * neighborhood structure it will see at full size. */
    int64_t sample_rows = sample_n / nx;
    if (sample_rows < 2) sample_rows = ny < 2 ? ny : 2; /* always score at least 2 rows when possible */
    if (sample_rows > ny) sample_rows = ny;

    tdc_pred2d_kind best_kind = TDC_PRED2D_AVERAGE;
    uint64_t        best_sum  = UINT64_MAX;

    static const tdc_pred2d_kind candidates[4] = {
        TDC_PRED2D_LEFT, TDC_PRED2D_UP, TDC_PRED2D_AVERAGE, TDC_PRED2D_PAETH
    };

    for (int k = 0; k < 4; ++k) {
        tdc_pred2d_kind kind = candidates[k];
        uint64_t sum = 0;
        for (int64_t row = 0; row < sample_rows; ++row) {
            for (int64_t col = 0; col < nx; ++col) {
                int64_t i      = row * nx + col;
                int64_t val    = pred2d_load(dt, src, i);
                int64_t left   = (col > 0)              ? pred2d_load(dt, src, i - 1)      : 0;
                int64_t up     = (row > 0)              ? pred2d_load(dt, src, i - nx)     : 0;
                int64_t upleft = (col > 0 && row > 0)   ? pred2d_load(dt, src, i - nx - 1) : 0;
                int64_t pred   = pred2d_compute(kind, left, up, upleft);
                int64_t res    = val - pred;
                sum += (uint64_t)(res < 0 ? -res : res);
            }
        }
        if (sum < best_sum) {
            best_sum  = sum;
            best_kind = kind;
        }
    }

    return best_kind;
}

/* ----- Encode ------------------------------------------------------------- */

static tdc_status pred2d_encode(const tdc_block *in,
                                const void      *params,
                                tdc_buffer      *residual_out,
                                tdc_dtype       *residual_dtype,
                                tdc_buffer      *side_out) {
    if (!in || !residual_out || !residual_out->realloc_fn) return TDC_E_INVAL;
    if (!side_out || !side_out->realloc_fn)                return TDC_E_INVAL;
    if (in->layout != TDC_LAYOUT_RASTER_2D) return TDC_E_LAYOUT;
    if (in->shape.rank != 2)                return TDC_E_SHAPE;
    if (!pred2d_dtype_accepted(in->dtype))  return TDC_E_DTYPE;

    int64_t ny = in->shape.dim[0];
    int64_t nx = in->shape.dim[1];
    if (nx < 0 || ny < 0)                                     return TDC_E_SHAPE;
    if (nx != 0 && ny != 0 && nx > INT64_MAX / ny)            return TDC_E_SHAPE;

    size_t  elem_size = tdc_dtype_size(in->dtype);
    if (elem_size == 0) return TDC_E_DTYPE;

    /* Resolve predictor kind. */
    tdc_pred2d_kind kind = TDC_PRED2D_AUTO;
    if (params) {
        const tdc_pred2d_params *p = (const tdc_pred2d_params *)params;
        kind = p->kind;
    }

    int64_t n = nx * ny;
    if (kind == TDC_PRED2D_AUTO) {
        if (n > 0) {
            if (!in->data) return TDC_E_INVAL;
            kind = pred2d_auto_select(in->dtype, (const uint8_t *)in->data, nx, ny);
        } else {
            kind = TDC_PRED2D_AVERAGE; /* arbitrary; nothing to encode */
        }
    } else if (kind != TDC_PRED2D_LEFT && kind != TDC_PRED2D_UP &&
               kind != TDC_PRED2D_AVERAGE && kind != TDC_PRED2D_PAETH) {
        return TDC_E_INVAL; /* PLANE / unknown — not handled by this file */
    }

    /* Side metadata: 1 byte = resolved kind. */
    tdc_status st = tdc_buf_reserve(side_out, 1u);
    if (st != TDC_OK) return st;
    side_out->data[0] = (uint8_t)kind;
    side_out->size    = 1u;

    /* Reserve residual output. */
    size_t bytes = (size_t)n * elem_size;
    st = tdc_buf_reserve(residual_out, bytes);
    if (st != TDC_OK) return st;

    if (residual_dtype) *residual_dtype = in->dtype;

    if (n == 0) {
        residual_out->size = 0;
        return TDC_OK;
    }

    if (!in->data) return TDC_E_INVAL;

    pred2d_encode_sweep(in->dtype, kind,
                        (const uint8_t *)in->data,
                        residual_out->data,
                        nx, ny);
    residual_out->size = bytes;
    return TDC_OK;
}

/* ----- Decode ------------------------------------------------------------- */

static tdc_status pred2d_decode(tdc_block      *out,
                                const void     *params,
                                tdc_dtype       residual_dtype,
                                const uint8_t  *residuals, size_t residual_size,
                                const uint8_t  *side_meta, size_t side_size) {
    (void)params;
    if (!out) return TDC_E_INVAL;
    if (out->layout != TDC_LAYOUT_RASTER_2D) return TDC_E_LAYOUT;
    if (out->shape.rank != 2)                return TDC_E_SHAPE;
    if (residual_dtype != out->dtype)        return TDC_E_DTYPE;
    if (!pred2d_dtype_accepted(out->dtype))  return TDC_E_DTYPE;

    int64_t ny = out->shape.dim[0];
    int64_t nx = out->shape.dim[1];
    if (nx < 0 || ny < 0)                                     return TDC_E_SHAPE;
    if (nx != 0 && ny != 0 && nx > INT64_MAX / ny)            return TDC_E_SHAPE;

    size_t elem_size = tdc_dtype_size(out->dtype);
    if (elem_size == 0) return TDC_E_DTYPE;

    int64_t n     = nx * ny;
    size_t  bytes = (size_t)n * elem_size;
    if (residual_size != bytes) return TDC_E_CORRUPT;

    /* Side metadata: exactly 1 byte = the resolved predictor kind. */
    if (side_size != 1u || side_meta == NULL) return TDC_E_CORRUPT;
    tdc_pred2d_kind kind = (tdc_pred2d_kind)side_meta[0];
    if (kind != TDC_PRED2D_LEFT && kind != TDC_PRED2D_UP &&
        kind != TDC_PRED2D_AVERAGE && kind != TDC_PRED2D_PAETH) {
        return TDC_E_CORRUPT;
    }

    if (n == 0) return TDC_OK;
    if (!out->data || !residuals) return TDC_E_INVAL;

    pred2d_decode_sweep(out->dtype, kind, residuals, (uint8_t *)out->data, nx, ny);
    return TDC_OK;
}

/* ----- Vtable ------------------------------------------------------------- */

const tdc_model_vt tdc_model_pred2d_vt = {
    .id               = TDC_MODEL_PRED_2D,
    .name             = "pred2d",
    .accepted_dtypes  = PRED2D_ACCEPTED_DTYPES,
    .accepted_layouts = PRED2D_ACCEPTED_LAYOUTS,
    .encode           = pred2d_encode,
    .decode           = pred2d_decode,
};
