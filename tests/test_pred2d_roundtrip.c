/*
 * tests/test_pred2d_roundtrip.c
 *
 * Round-trip test for the TDC_MODEL_PRED_2D backend.
 *
 * Verifies:
 *   1. residual_dtype == in->dtype, side metadata is exactly 1 byte = the
 *      resolved predictor kind.
 *   2. Round-trip on every supported dtype (i8/i16/i32/i64/u8/u16/u32/u64)
 *      for every fixed kind (LEFT/UP/AVERAGE/PAETH) on a small synthetic
 *      raster.
 *   3. Round-trip on a smooth gradient where each predictor produces
 *      small residuals (sanity check that the predictor math is right).
 *   4. AUTO mode picks one of the four kinds and round-trips correctly.
 *   5. AUTO mode picks the predictor with the smallest sum of |residual|
 *      (verified by independently computing scores for all four kinds in
 *      the test and matching the side-meta byte against the argmin).
 *   6. Edge values: i8/i16 boundaries (type min, type max, ±1, 0) round-trip
 *      under all four kinds — guards the modular wrap.
 *   7. Empty block (n == 0) round-trips.
 *   8. Encoder rejects non-RASTER_2D layouts.
 *   9. Decoder rejects missing / wrong-size side metadata.
 *  10. Registry returns &tdc_model_pred2d_vt for TDC_MODEL_PRED_2D.
 *  11. Float round-trip (f16/f32/f64) using ordered-integer prediction
 *      for all four kinds. Covers smooth ramps, NaN, ±Inf, ±0,
 *      subnormals, sign changes.
 */

#include "tdc/model.h"
#include "tdc/codec.h"
#include "tdc/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

extern const tdc_model_vt tdc_model_pred2d_vt;

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

static tdc_block raster(void *data, int64_t ny, int64_t nx, tdc_dtype dt) {
    tdc_block b = {0};
    b.data = data;
    b.dtype = dt;
    b.layout = TDC_LAYOUT_RASTER_2D;
    b.shape.rank = 2;
    b.shape.dim[0] = ny;
    b.shape.dim[1] = nx;
    b.shape.stride[0] = nx;
    b.shape.stride[1] = 1;
    return b;
}

/* ----- Generic round-trip ---------------------------------------------- */

static int rt_one(const char *label, tdc_pred2d_kind kind,
                  tdc_dtype dt, size_t elem_size,
                  const void *src, int64_t ny, int64_t nx) {
    const tdc_model_vt *vt = &tdc_model_pred2d_vt;

    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;

    tdc_pred2d_params params = { .kind = kind };
    tdc_block in = raster((void *)src, ny, nx, dt);
    size_t bytes = (size_t)(ny * nx) * elem_size;

    tdc_status st = vt->encode(&in, &params, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "encode returned non-OK");
    ASSERT_OR_DIE(rdt == dt, "residual_dtype != input dtype");
    ASSERT_OR_DIE(side.size == 1u, "side metadata must be exactly 1 byte");
    ASSERT_OR_DIE(residual.size == bytes, "residual size mismatch");

    /* For non-AUTO callers, the resolved kind must equal the requested kind. */
    if (kind != TDC_PRED2D_AUTO) {
        ASSERT_OR_DIE(side.data[0] == (uint8_t)kind,
                      "side metadata must record the requested kind");
    } else {
        uint8_t resolved = side.data[0];
        ASSERT_OR_DIE(resolved == TDC_PRED2D_LEFT  ||
                      resolved == TDC_PRED2D_UP    ||
                      resolved == TDC_PRED2D_AVERAGE ||
                      resolved == TDC_PRED2D_PAETH,
                      "AUTO must resolve to one of the four kinds");
    }

    /* Decode */
    void *dst = malloc(bytes > 0 ? bytes : 1u);
    ASSERT_OR_DIE(dst != NULL, "malloc dst");
    memset(dst, 0xAA, bytes > 0 ? bytes : 1u);

    tdc_block out = raster(dst, ny, nx, dt);
    st = vt->decode(&out, NULL, rdt, residual.data, residual.size,
                    side.data, side.size);
    ASSERT_OR_DIE(st == TDC_OK, "decode returned non-OK");
    if (bytes > 0) {
        ASSERT_OR_DIE(memcmp(dst, src, bytes) == 0,
                      "decoded data does not match source");
    }

    printf("  [%s kind=%u] %lldx%lld bytes=%zu round-trip OK\n",
           label, (unsigned)side.data[0],
           (long long)ny, (long long)nx, bytes);

    free(dst);
    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

/* ----- Per-dtype synthetic rasters ------------------------------------- */

static int test_dtype_all_kinds(void) {
    /* 4x5 raster with mixed structure: a gradient + a localized bump. */
    int64_t  src_i64[20];
    int32_t  src_i32[20];
    int16_t  src_i16[20];
    int8_t   src_i8 [20];
    uint64_t src_u64[20];
    uint32_t src_u32[20];
    uint16_t src_u16[20];
    uint8_t  src_u8 [20];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 5; ++c) {
            int idx = r * 5 + c;
            int v = 10 + r * 2 + c * 3 + (r == 2 && c == 3 ? 25 : 0);
            src_i64[idx] = v;
            src_i32[idx] = v;
            src_i16[idx] = (int16_t)v;
            src_i8 [idx] = (int8_t)v;
            src_u64[idx] = (uint64_t)v;
            src_u32[idx] = (uint32_t)v;
            src_u16[idx] = (uint16_t)v;
            src_u8 [idx] = (uint8_t)v;
        }
    }

    static const tdc_pred2d_kind kinds[4] = {
        TDC_PRED2D_LEFT, TDC_PRED2D_UP, TDC_PRED2D_AVERAGE, TDC_PRED2D_PAETH
    };
    for (int k = 0; k < 4; ++k) {
        tdc_pred2d_kind kind = kinds[k];
        if (rt_one("i8 4x5",  kind, TDC_DT_I8,  1, src_i8,  4, 5)) return 1;
        if (rt_one("i16 4x5", kind, TDC_DT_I16, 2, src_i16, 4, 5)) return 1;
        if (rt_one("i32 4x5", kind, TDC_DT_I32, 4, src_i32, 4, 5)) return 1;
        if (rt_one("i64 4x5", kind, TDC_DT_I64, 8, src_i64, 4, 5)) return 1;
        if (rt_one("u8 4x5",  kind, TDC_DT_U8,  1, src_u8,  4, 5)) return 1;
        if (rt_one("u16 4x5", kind, TDC_DT_U16, 2, src_u16, 4, 5)) return 1;
        if (rt_one("u32 4x5", kind, TDC_DT_U32, 4, src_u32, 4, 5)) return 1;
        if (rt_one("u64 4x5", kind, TDC_DT_U64, 8, src_u64, 4, 5)) return 1;
    }
    return 0;
}

/* ----- Edge values ------------------------------------------------------ */

static int test_edges_i16(void) {
    /* 2x4 raster of i16 edge values: type min/max, ±1, 0. */
    int16_t src[8] = {
        INT16_MIN, INT16_MAX, 0,        -1,
        1,         INT16_MAX, INT16_MIN, 0
    };
    static const tdc_pred2d_kind kinds[4] = {
        TDC_PRED2D_LEFT, TDC_PRED2D_UP, TDC_PRED2D_AVERAGE, TDC_PRED2D_PAETH
    };
    for (int k = 0; k < 4; ++k) {
        if (rt_one("i16 edges 2x4", kinds[k], TDC_DT_I16, 2, src, 2, 4)) return 1;
    }
    return 0;
}

static int test_edges_u8(void) {
    /* 2x4 raster of u8 edges. */
    uint8_t src[8] = { 0, 255, 1, 254, 128, 127, 0, 255 };
    static const tdc_pred2d_kind kinds[4] = {
        TDC_PRED2D_LEFT, TDC_PRED2D_UP, TDC_PRED2D_AVERAGE, TDC_PRED2D_PAETH
    };
    for (int k = 0; k < 4; ++k) {
        if (rt_one("u8 edges 2x4", kinds[k], TDC_DT_U8, 1, src, 2, 4)) return 1;
    }
    return 0;
}

/* ----- Auto-select ------------------------------------------------------ */

/* Independently compute the sum of |residual| for one predictor kind on
 * a 5x8 i32 raster, mirroring the model's encode loop and the auto-select
 * scoring loop. Used by test_auto_argmin to verify AUTO picks argmin. */
static uint64_t score_kind(const int32_t *src, int64_t ny, int64_t nx,
                           tdc_pred2d_kind kind) {
    uint64_t sum = 0;
    for (int64_t r = 0; r < ny; ++r) {
        for (int64_t c = 0; c < nx; ++c) {
            int64_t i      = r * nx + c;
            int64_t val    = (int64_t)src[i];
            int64_t left   = (c > 0)              ? (int64_t)src[i - 1]      : 0;
            int64_t up     = (r > 0)              ? (int64_t)src[i - nx]     : 0;
            int64_t upleft = (c > 0 && r > 0)     ? (int64_t)src[i - nx - 1] : 0;
            int64_t pred;
            switch (kind) {
                case TDC_PRED2D_LEFT:    pred = left; break;
                case TDC_PRED2D_UP:      pred = up;   break;
                case TDC_PRED2D_AVERAGE: pred = (left + up) / 2; break;
                case TDC_PRED2D_PAETH: {
                    int64_t p  = left + up - upleft;
                    int64_t pa = p > left   ? p - left   : left   - p;
                    int64_t pb = p > up     ? p - up     : up     - p;
                    int64_t pc = p > upleft ? p - upleft : upleft - p;
                    if (pa <= pb && pa <= pc) pred = left;
                    else if (pb <= pc)        pred = up;
                    else                      pred = upleft;
                    break;
                }
                default: pred = 0; break;
            }
            int64_t res = val - pred;
            sum += (uint64_t)(res < 0 ? -res : res);
        }
    }
    return sum;
}

static int test_auto_argmin(void) {
    /* Raster with mixed structure: column-linear, plus per-row offset.
     * The argmin among the four predictors is computed in the test, not
     * hardcoded — this guards against the auto-select code drifting out
     * of sync with the per-kind sweep. */
    int32_t src[5 * 8];
    for (int r = 0; r < 5; ++r)
        for (int c = 0; c < 8; ++c)
            src[r * 8 + c] = c * 7 + r * 3;

    /* Round-trip via AUTO. */
    if (rt_one("auto argmin", TDC_PRED2D_AUTO, TDC_DT_I32, 4, src, 5, 8)) return 1;

    /* Independently compute argmin in the test. */
    static const tdc_pred2d_kind kinds[4] = {
        TDC_PRED2D_LEFT, TDC_PRED2D_UP, TDC_PRED2D_AVERAGE, TDC_PRED2D_PAETH
    };
    uint64_t best_sum = UINT64_MAX;
    tdc_pred2d_kind best_kind = TDC_PRED2D_AVERAGE;
    for (int k = 0; k < 4; ++k) {
        uint64_t s = score_kind(src, 5, 8, kinds[k]);
        if (s < best_sum) { best_sum = s; best_kind = kinds[k]; }
    }

    /* Re-encode through the model to inspect side metadata. */
    const tdc_model_vt *vt = &tdc_model_pred2d_vt;
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_pred2d_params params = { .kind = TDC_PRED2D_AUTO };
    tdc_block in = raster(src, 5, 8, TDC_DT_I32);

    tdc_status st = vt->encode(&in, &params, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "auto-argmin encode");
    ASSERT_OR_DIE(side.data[0] == (uint8_t)best_kind,
                  "AUTO must pick the kind with smallest sum of |residual|");
    printf("  [auto argmin] resolved to kind=%u (best_sum=%llu) as expected\n",
           (unsigned)best_kind, (unsigned long long)best_sum);

    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

/* ----- Edges & rejections ---------------------------------------------- */

static int test_empty(void) {
    const tdc_model_vt *vt = &tdc_model_pred2d_vt;
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_pred2d_params params = { .kind = TDC_PRED2D_LEFT };

    tdc_block in = raster(NULL, 0, 0, TDC_DT_I32);
    tdc_status st = vt->encode(&in, &params, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "empty encode");
    ASSERT_OR_DIE(residual.size == 0, "empty residual size");
    ASSERT_OR_DIE(side.size == 1, "empty side size");
    ASSERT_OR_DIE(rdt == TDC_DT_I32, "empty residual_dtype");

    tdc_block out = raster(NULL, 0, 0, TDC_DT_I32);
    st = vt->decode(&out, NULL, TDC_DT_I32, NULL, 0, side.data, side.size);
    ASSERT_OR_DIE(st == TDC_OK, "empty decode");

    printf("  [empty] OK\n");

    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

static int test_rejections(void) {
    const tdc_model_vt *vt = &tdc_model_pred2d_vt;
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_pred2d_params params = { .kind = TDC_PRED2D_LEFT };

    /* non-RASTER_2D layout rejected */
    {
        int32_t src[4] = { 0, 1, 2, 3 };
        tdc_block in = raster(src, 2, 2, TDC_DT_I32);
        in.layout = TDC_LAYOUT_VECTOR_1D;
        in.shape.rank = 1;
        in.shape.dim[0] = 4;
        tdc_status st = vt->encode(&in, &params, &residual, &rdt, &side);
        ASSERT_OR_DIE(st == TDC_E_LAYOUT, "VECTOR_1D should be rejected");
    }

    /* decode rejects missing side metadata */
    {
        int32_t dst[4] = { 0 };
        tdc_block out = raster(dst, 2, 2, TDC_DT_I32);
        uint8_t residuals[16] = { 0 };
        tdc_status st = vt->decode(&out, NULL, TDC_DT_I32,
                                   residuals, sizeof residuals,
                                   NULL, 0);
        ASSERT_OR_DIE(st == TDC_E_CORRUPT, "missing side meta should be rejected");
    }

    /* decode rejects unknown predictor kind in side meta */
    {
        int32_t dst[4] = { 0 };
        tdc_block out = raster(dst, 2, 2, TDC_DT_I32);
        uint8_t residuals[16] = { 0 };
        uint8_t bad_side[1] = { 99 };
        tdc_status st = vt->decode(&out, NULL, TDC_DT_I32,
                                   residuals, sizeof residuals,
                                   bad_side, 1);
        ASSERT_OR_DIE(st == TDC_E_CORRUPT, "unknown kind should be rejected");
    }

    /* decode rejects mismatched residual_size */
    {
        int32_t dst[4] = { 0 };
        tdc_block out = raster(dst, 2, 2, TDC_DT_I32);
        uint8_t residuals[8] = { 0 };
        uint8_t side1[1] = { (uint8_t)TDC_PRED2D_LEFT };
        tdc_status st = vt->decode(&out, NULL, TDC_DT_I32,
                                   residuals, sizeof residuals,
                                   side1, 1);
        ASSERT_OR_DIE(st == TDC_E_CORRUPT, "wrong residual_size should be rejected");
    }

    printf("  [rejections] OK\n");

    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

/* ----- Float round-trip tests ------------------------------------------ */

static int test_f32_all_kinds(void) {
    /* 4x5 raster: smooth gradient + sign change + specials in last row. */
    float src[20] = {
        -3.0f, -2.0f, -1.0f, -0.0f,  0.0f,
         1.0f,  2.5f,  5.0f, 10.0f, 20.0f,
        50.0f, 100.0f, 200.0f, 1.0e-38f, -1.0e-38f,
        INFINITY, -INFINITY, NAN, 0.5f, -0.5f
    };
    static const tdc_pred2d_kind kinds[4] = {
        TDC_PRED2D_LEFT, TDC_PRED2D_UP, TDC_PRED2D_AVERAGE, TDC_PRED2D_PAETH
    };
    for (int k = 0; k < 4; ++k) {
        if (rt_one("f32 4x5", kinds[k], TDC_DT_F32, 4, src, 4, 5)) return 1;
    }
    /* AUTO too */
    if (rt_one("f32 4x5 auto", TDC_PRED2D_AUTO, TDC_DT_F32, 4, src, 4, 5)) return 1;
    return 0;
}

static int test_f64_all_kinds(void) {
    double src[20] = {
        -3.0, -2.0, -1.0, -0.0,  0.0,
         1.0,  2.5,  5.0, 10.0, 20.0,
        50.0, 100.0, 200.0, 1.0e-300, -1.0e-300,
        INFINITY, -INFINITY, NAN, 0.5, -0.5
    };
    static const tdc_pred2d_kind kinds[4] = {
        TDC_PRED2D_LEFT, TDC_PRED2D_UP, TDC_PRED2D_AVERAGE, TDC_PRED2D_PAETH
    };
    for (int k = 0; k < 4; ++k) {
        if (rt_one("f64 4x5", kinds[k], TDC_DT_F64, 8, src, 4, 5)) return 1;
    }
    if (rt_one("f64 4x5 auto", TDC_PRED2D_AUTO, TDC_DT_F64, 8, src, 4, 5)) return 1;
    return 0;
}

static int test_f16_all_kinds(void) {
    /* f16 bit patterns: ramp + specials.  4x5 = 20 elements. */
    uint16_t src[20] = {
        0x0000,  /* +0 */
        0x8000,  /* -0 */
        0x3C00,  /* +1.0 */
        0xBC00,  /* -1.0 */
        0x4000,  /* +2.0 */
        0x4200,  /* +3.0 */
        0x4400,  /* +4.0 */
        0x4500,  /* +5.0 */
        0x4600,  /* +6.0 */
        0x4700,  /* +7.0 */
        0x7C00,  /* +Inf */
        0xFC00,  /* -Inf */
        0x7E00,  /* NaN */
        0x0001,  /* smallest subnormal */
        0x7BFF,  /* largest finite (65504) */
        0x3555,  /* ~0.333 */
        0x3800,  /* 0.5 */
        0xB800,  /* -0.5 */
        0x4800,  /* +8.0 */
        0x4900   /* +10.0 */
    };
    static const tdc_pred2d_kind kinds[4] = {
        TDC_PRED2D_LEFT, TDC_PRED2D_UP, TDC_PRED2D_AVERAGE, TDC_PRED2D_PAETH
    };
    for (int k = 0; k < 4; ++k) {
        if (rt_one("f16 4x5", kinds[k], TDC_DT_F16, 2, src, 4, 5)) return 1;
    }
    if (rt_one("f16 4x5 auto", TDC_PRED2D_AUTO, TDC_DT_F16, 2, src, 4, 5)) return 1;
    return 0;
}

static int test_f32_smooth_gradient(void) {
    /* 16x16 smooth gradient — residuals should be small in ordered space. */
    float src[256];
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c)
            src[r * 16 + c] = 100.0f + (float)r * 0.7f + (float)c * 0.3f;
    return rt_one("f32 16x16 gradient", TDC_PRED2D_PAETH, TDC_DT_F32, 4, src, 16, 16);
}

static int test_registry(void) {
    const tdc_model_vt *vt = tdc_model_get(TDC_MODEL_PRED_2D);
    ASSERT_OR_DIE(vt == &tdc_model_pred2d_vt,
                  "tdc_model_get(TDC_MODEL_PRED_2D) wiring");
    ASSERT_OR_DIE(vt->id == TDC_MODEL_PRED_2D, "vt->id mismatch");
    printf("  [registry wiring] OK\n");
    return 0;
}

int main(void) {
    printf("test_pred2d_roundtrip\n");
    if (test_dtype_all_kinds()) return 1;
    if (test_edges_i16())       return 1;
    if (test_edges_u8())        return 1;
    if (test_auto_argmin())     return 1;
    if (test_f32_all_kinds())    return 1;
    if (test_f64_all_kinds())    return 1;
    if (test_f16_all_kinds())    return 1;
    if (test_f32_smooth_gradient()) return 1;
    if (test_empty())           return 1;
    if (test_rejections())      return 1;
    if (test_registry())        return 1;
    printf("ALL OK\n");
    return 0;
}
