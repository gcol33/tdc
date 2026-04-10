/*
 * tests/test_delta1d_roundtrip.c
 *
 * Round-trip test for the TDC_MODEL_DELTA_1D backend.
 *
 * Verifies:
 *   1. residual[0] == data[0] (the seed is the first residual itself).
 *   2. residual_dtype == in->dtype, side metadata is empty.
 *   3. Round-trip on every supported integer dtype
 *        (i8/i16/i32/i64/u8/u16/u32/u64).
 *   4. Modular wraparound: full-range edges (type min, type max, ±1) round-trip
 *      correctly even when (cur - prev) overflows the signed range.
 *   5. Monotonic int64 column compresses to small residuals (basic
 *      sanity check that the math is right, not a compression-ratio test).
 *   6. Empty block (n == 0) round-trips.
 *   7. Encoder rejects non-VECTOR_1D layouts (TDC_E_LAYOUT).
 *   8. Decoder rejects non-empty side_size (TDC_E_CORRUPT) and a residual_size
 *      that disagrees with the block shape (TDC_E_CORRUPT).
 *   9. Registry returns &tdc_model_delta1d_vt for TDC_MODEL_DELTA_1D.
 *  10. Float round-trip (f16/f32/f64) using ordered-integer delta mapping.
 *      Covers smooth ramps, NaN, ±Inf, ±0, subnormals, sign changes.
 */

#include "tdc/model.h"
#include "tdc/codec.h"
#include "tdc/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

extern const tdc_model_vt tdc_model_delta1d_vt;

static void *test_realloc(void *user, void *ptr, size_t new_size) {
    (void)user;
    if (new_size == 0) { free(ptr); return NULL; }
    return realloc(ptr, new_size);
}

static tdc_buffer make_buffer(void) {
    tdc_buffer b = {0};
    b.realloc_fn = test_realloc;
    return b;
}

static void free_buffer(tdc_buffer *b) {
    if (b->data) free(b->data);
    b->data = NULL;
    b->size = 0;
    b->capacity = 0;
}

#define ASSERT_OR_DIE(cond, msg) do {                                       \
    if (!(cond)) {                                                          \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);   \
        return 1;                                                           \
    }                                                                       \
} while (0)

/* Build a 1D block over `data` of `n` elements of dtype `dt`. */
static tdc_block make_block(void *data, int64_t n, tdc_dtype dt) {
    tdc_block b = {0};
    b.data = data;
    b.dtype = dt;
    b.layout = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank = 1;
    b.shape.dim[0] = n;
    b.shape.stride[0] = 1;
    return b;
}

/* ----- Round-trip helper for one dtype --------------------------------- */

static int rt_one(const char *label, tdc_dtype dt, size_t elem_size,
                  const void *src, int64_t n) {
    const tdc_model_vt *vt = &tdc_model_delta1d_vt;

    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;

    /* Encode requires a non-const block; copy the dtype/shape envelope. */
    tdc_block in = make_block((void *)src, n, dt);

    tdc_status st = vt->encode(&in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "encode returned non-OK");
    ASSERT_OR_DIE(rdt == dt, "residual_dtype != input dtype");
    ASSERT_OR_DIE(side.size == 0, "delta1d must not emit side metadata");
    ASSERT_OR_DIE(residual.size == (size_t)n * elem_size, "residual size mismatch");

    /* residual[0] == src[0] (seed) — only for integer dtypes.
     * Float dtypes store the ordered-integer mapping as the seed. */
    if (n > 0 && !tdc_dtype_is_float(dt)) {
        ASSERT_OR_DIE(memcmp(residual.data, src, elem_size) == 0,
                      "residual[0] should equal data[0]");
    }

    /* Decode */
    void *dst = malloc((size_t)n * elem_size > 0 ? (size_t)n * elem_size : 1u);
    ASSERT_OR_DIE(dst != NULL, "malloc dst");
    tdc_block out = make_block(dst, n, dt);

    st = vt->decode(&out, NULL, rdt, residual.data, residual.size, NULL, 0);
    ASSERT_OR_DIE(st == TDC_OK, "decode returned non-OK");
    ASSERT_OR_DIE(memcmp(dst, src, (size_t)n * elem_size) == 0,
                  "decoded data does not match source");

    printf("  [%s] n=%lld bytes=%zu round-trip OK\n",
           label, (long long)n, (size_t)n * elem_size);

    free(dst);
    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

/* ----- Per-dtype tests -------------------------------------------------- */

static int test_i8(void) {
    int8_t src[8] = { 0, 1, -1, 127, -128, 5, -5, 0 };
    return rt_one("i8 edges", TDC_DT_I8, 1, src, 8);
}

static int test_i16(void) {
    int16_t src[10] = { 0, 1, -1, 32767, -32768, 100, -100, 32767, -32768, 0 };
    return rt_one("i16 edges", TDC_DT_I16, 2, src, 10);
}

static int test_i32(void) {
    int32_t src[12] = { 0, 1, -1, INT32_MAX, INT32_MIN, INT32_MAX, INT32_MIN,
                        1000, -1000, 5, -5, 0 };
    return rt_one("i32 edges", TDC_DT_I32, 4, src, 12);
}

static int test_i64(void) {
    int64_t src[12] = { 0, 1, -1, INT64_MAX, INT64_MIN, INT64_MAX, INT64_MIN,
                        1000000, -1000000, 5, -5, 0 };
    return rt_one("i64 edges", TDC_DT_I64, 8, src, 12);
}

static int test_u8(void) {
    uint8_t src[8] = { 0, 1, 255, 254, 0, 128, 127, 255 };
    return rt_one("u8 edges", TDC_DT_U8, 1, src, 8);
}

static int test_u16(void) {
    uint16_t src[8] = { 0, 1, 65535, 65534, 0, 32768, 32767, 65535 };
    return rt_one("u16 edges", TDC_DT_U16, 2, src, 8);
}

static int test_u32(void) {
    uint32_t src[8] = { 0u, 1u, UINT32_MAX, UINT32_MAX - 1u,
                        0u, 0x80000000u, 0x7FFFFFFFu, UINT32_MAX };
    return rt_one("u32 edges", TDC_DT_U32, 4, src, 8);
}

static int test_u64(void) {
    uint64_t src[8] = { 0u, 1u, UINT64_MAX, UINT64_MAX - 1u,
                        0u, 0x8000000000000000ull, 0x7FFFFFFFFFFFFFFFull,
                        UINT64_MAX };
    return rt_one("u64 edges", TDC_DT_U64, 8, src, 8);
}

/* Monotonic int64 column: encoded residuals (after the seed) must equal
 * the gaps. Sanity check the math, not a compression ratio. */
static int test_monotonic_i64_residuals(void) {
    const tdc_model_vt *vt = &tdc_model_delta1d_vt;
    int64_t src[6] = { 1000, 1003, 1010, 1042, 1042, 99999 };
    int64_t expected[6] = { 1000, 3, 7, 32, 0, 98957 };

    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_block  in       = make_block(src, 6, TDC_DT_I64);

    tdc_status st = vt->encode(&in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "monotonic encode failed");
    ASSERT_OR_DIE(residual.size == 6u * 8u, "monotonic residual size");
    ASSERT_OR_DIE(memcmp(residual.data, expected, sizeof expected) == 0,
                  "monotonic residuals != expected gaps");

    printf("  [monotonic i64] residuals match expected gaps OK\n");

    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

static int test_empty(void) {
    const tdc_model_vt *vt = &tdc_model_delta1d_vt;
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;

    tdc_block in = make_block(NULL, 0, TDC_DT_I32);
    tdc_status st = vt->encode(&in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "empty encode");
    ASSERT_OR_DIE(residual.size == 0, "empty residual size");
    ASSERT_OR_DIE(side.size == 0, "empty side size");
    ASSERT_OR_DIE(rdt == TDC_DT_I32, "empty residual_dtype");

    tdc_block out = make_block(NULL, 0, TDC_DT_I32);
    st = vt->decode(&out, NULL, TDC_DT_I32, NULL, 0, NULL, 0);
    ASSERT_OR_DIE(st == TDC_OK, "empty decode");

    printf("  [empty] OK\n");

    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

static int test_rejections(void) {
    const tdc_model_vt *vt = &tdc_model_delta1d_vt;
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;

    /* non-VECTOR_1D layout rejected */
    {
        int32_t src[4] = { 0, 1, 2, 3 };
        tdc_block in = make_block(src, 4, TDC_DT_I32);
        in.layout = TDC_LAYOUT_RASTER_2D;
        in.shape.rank = 2;
        in.shape.dim[0] = 2;
        in.shape.dim[1] = 2;
        tdc_status st = vt->encode(&in, NULL, &residual, &rdt, &side);
        ASSERT_OR_DIE(st == TDC_E_LAYOUT, "RASTER_2D should be rejected");
    }

    /* decode rejects non-empty side metadata */
    {
        int32_t dst[4] = { 0 };
        tdc_block out = make_block(dst, 4, TDC_DT_I32);
        uint8_t residuals[16] = { 0 };
        uint8_t bogus_side[1] = { 0 };
        tdc_status st = vt->decode(&out, NULL, TDC_DT_I32,
                                   residuals, sizeof residuals,
                                   bogus_side, 1);
        ASSERT_OR_DIE(st == TDC_E_CORRUPT, "side_size != 0 should be rejected");
    }

    /* decode rejects mismatched residual_size */
    {
        int32_t dst[4] = { 0 };
        tdc_block out = make_block(dst, 4, TDC_DT_I32);
        uint8_t residuals[8] = { 0 };
        tdc_status st = vt->decode(&out, NULL, TDC_DT_I32,
                                   residuals, sizeof residuals,
                                   NULL, 0);
        ASSERT_OR_DIE(st == TDC_E_CORRUPT, "wrong residual_size should be rejected");
    }

    printf("  [rejections] OK\n");

    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

/* ----- Float round-trip tests -------------------------------------------- */

static int test_f32(void) {
    /* Smooth ramp, sign changes, specials */
    float src[12] = {
        -3.0f, -2.0f, -1.0f, -0.0f, 0.0f, 1.0f,
        2.5f, 100.0f, INFINITY, -INFINITY, NAN, 1.0e-38f
    };
    return rt_one("f32 mixed", TDC_DT_F32, 4, src, 12);
}

static int test_f64(void) {
    double src[12] = {
        -3.0, -2.0, -1.0, -0.0, 0.0, 1.0,
        2.5, 100.0, INFINITY, -INFINITY, NAN, 1.0e-300
    };
    return rt_one("f64 mixed", TDC_DT_F64, 8, src, 12);
}

static int test_f16(void) {
    /* f16 bit patterns: ±0, ±1.0, ±Inf, NaN, subnormal */
    uint16_t src[10] = {
        0x0000,  /* +0 */
        0x8000,  /* -0 */
        0x3C00,  /* +1.0 */
        0xBC00,  /* -1.0 */
        0x7C00,  /* +Inf */
        0xFC00,  /* -Inf */
        0x7E00,  /* NaN */
        0x0001,  /* smallest subnormal */
        0x7BFF,  /* largest finite f16 (65504) */
        0x3555   /* ~0.333 */
    };
    return rt_one("f16 mixed", TDC_DT_F16, 2, src, 10);
}

static int test_f32_smooth_ramp(void) {
    /* Slowly varying data: residuals should be small in ordered space.
     * Not a compression test, just verifies the ramp round-trips. */
    float src[256];
    for (int i = 0; i < 256; ++i)
        src[i] = 1000.0f + (float)i * 0.1f;
    return rt_one("f32 smooth ramp", TDC_DT_F32, 4, src, 256);
}

static int test_registry(void) {
    const tdc_model_vt *vt = tdc_model_get(TDC_MODEL_DELTA_1D);
    ASSERT_OR_DIE(vt == &tdc_model_delta1d_vt,
                  "tdc_model_get(TDC_MODEL_DELTA_1D) wiring");
    ASSERT_OR_DIE(vt->id == TDC_MODEL_DELTA_1D, "vt->id mismatch");
    printf("  [registry wiring] OK\n");
    return 0;
}

int main(void) {
    printf("test_delta1d_roundtrip\n");
    if (test_i8())                          return 1;
    if (test_i16())                         return 1;
    if (test_i32())                         return 1;
    if (test_i64())                         return 1;
    if (test_u8())                          return 1;
    if (test_u16())                         return 1;
    if (test_u32())                         return 1;
    if (test_u64())                         return 1;
    if (test_monotonic_i64_residuals())     return 1;
    if (test_f32())                         return 1;
    if (test_f64())                         return 1;
    if (test_f16())                         return 1;
    if (test_f32_smooth_ramp())             return 1;
    if (test_empty())                       return 1;
    if (test_rejections())                  return 1;
    if (test_registry())                    return 1;
    printf("ALL OK\n");
    return 0;
}
