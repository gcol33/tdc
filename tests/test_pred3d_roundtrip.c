/*
 * tests/test_pred3d_roundtrip.c
 *
 * Round-trip test for the TDC_MODEL_PRED_3D backend.
 *
 * Verifies:
 *   1. residual_dtype == in->dtype, side metadata is exactly 1 byte = the
 *      resolved predictor kind.
 *   2. Round-trip on every supported dtype (i8/i16/i32/u8/u16/u32) for
 *      every fixed kind on a small synthetic 3x4x5 volume.
 *   3. Edge values: i16 boundaries (type min, type max, ±1, 0) round-trip
 *      under all six kinds — guards the modular wrap.
 *   4. AUTO mode picks one of the six kinds and round-trips correctly.
 *   5. AUTO mode picks the predictor with the smallest sum of |residual|
 *      (verified by independently computing scores in the test).
 *   6. Empty volume (n == 0) round-trips.
 *   7. Encoder rejects 64-bit ints, non-VOLUME_3D layouts.
 *   8. Decoder rejects missing / wrong-size side metadata, unknown kinds.
 *   9. Registry returns &tdc_model_pred3d_vt for TDC_MODEL_PRED_3D.
 *  10. Float round-trip (f16/f32/f64) using ordered-integer prediction
 *      for all six kinds. Covers smooth ramps, NaN, ±Inf, ±0,
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

extern const tdc_model_vt tdc_model_pred3d_vt;

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

static tdc_block volume(void *data, int64_t nz, int64_t ny, int64_t nx,
                        tdc_dtype dt) {
    tdc_block b = {0};
    b.data = data;
    b.dtype = dt;
    b.layout = TDC_LAYOUT_VOLUME_3D;
    b.shape.rank = 3;
    b.shape.dim[0] = nz;
    b.shape.dim[1] = ny;
    b.shape.dim[2] = nx;
    b.shape.stride[0] = ny * nx;
    b.shape.stride[1] = nx;
    b.shape.stride[2] = 1;
    return b;
}

static int kind_is_resolved(uint8_t k) {
    return k == TDC_PRED3D_LEFT  || k == TDC_PRED3D_UP     ||
           k == TDC_PRED3D_FRONT || k == TDC_PRED3D_AVG3   ||
           k == TDC_PRED3D_GRAD3D || k == TDC_PRED3D_PAETH3D;
}

/* ----- Generic round-trip ---------------------------------------------- */

static int rt_one(const char *label, tdc_pred3d_kind kind,
                  tdc_dtype dt, size_t elem_size,
                  const void *src, int64_t nz, int64_t ny, int64_t nx) {
    const tdc_model_vt *vt = &tdc_model_pred3d_vt;

    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;

    tdc_pred3d_params params = { .kind = kind };
    tdc_block in = volume((void *)src, nz, ny, nx, dt);
    size_t bytes = (size_t)(nz * ny * nx) * elem_size;

    tdc_status st = vt->encode(&in, &params, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "encode returned non-OK");
    ASSERT_OR_DIE(rdt == dt, "residual_dtype != input dtype");
    ASSERT_OR_DIE(side.size == 1u, "side metadata must be exactly 1 byte");
    ASSERT_OR_DIE(residual.size == bytes, "residual size mismatch");

    if (kind != TDC_PRED3D_AUTO) {
        ASSERT_OR_DIE(side.data[0] == (uint8_t)kind,
                      "side metadata must record the requested kind");
    } else {
        ASSERT_OR_DIE(kind_is_resolved(side.data[0]),
                      "AUTO must resolve to one of the six kinds");
    }

    void *dst = malloc(bytes > 0 ? bytes : 1u);
    ASSERT_OR_DIE(dst != NULL, "malloc dst");
    memset(dst, 0xAA, bytes > 0 ? bytes : 1u);

    tdc_block out = volume(dst, nz, ny, nx, dt);
    st = vt->decode(&out, NULL, rdt, residual.data, residual.size,
                    side.data, side.size);
    ASSERT_OR_DIE(st == TDC_OK, "decode returned non-OK");
    if (bytes > 0) {
        ASSERT_OR_DIE(memcmp(dst, src, bytes) == 0,
                      "decoded data does not match source");
    }

    printf("  [%s kind=%u] %lldx%lldx%lld bytes=%zu round-trip OK\n",
           label, (unsigned)side.data[0],
           (long long)nz, (long long)ny, (long long)nx, bytes);

    free(dst);
    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

/* ----- Per-dtype synthetic volumes ------------------------------------- */

static int test_dtype_all_kinds(void) {
    /* 3x4x5 volume with mixed structure: trilinear gradient + a localized bump. */
    enum { NZ = 3, NY = 4, NX = 5, N = NZ * NY * NX };
    int32_t  src_i32[N];
    int16_t  src_i16[N];
    int8_t   src_i8 [N];
    uint32_t src_u32[N];
    uint16_t src_u16[N];
    uint8_t  src_u8 [N];
    for (int z = 0; z < NZ; ++z) {
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                int idx = (z * NY + y) * NX + x;
                int v = 10 + z * 5 + y * 2 + x * 3
                        + (z == 1 && y == 2 && x == 3 ? 25 : 0);
                src_i32[idx] = v;
                src_i16[idx] = (int16_t)v;
                src_i8 [idx] = (int8_t)v;
                src_u32[idx] = (uint32_t)v;
                src_u16[idx] = (uint16_t)v;
                src_u8 [idx] = (uint8_t)v;
            }
        }
    }

    static const tdc_pred3d_kind kinds[6] = {
        TDC_PRED3D_LEFT, TDC_PRED3D_UP, TDC_PRED3D_FRONT,
        TDC_PRED3D_AVG3, TDC_PRED3D_GRAD3D, TDC_PRED3D_PAETH3D
    };
    for (int k = 0; k < 6; ++k) {
        tdc_pred3d_kind kind = kinds[k];
        if (rt_one("i8  3x4x5", kind, TDC_DT_I8,  1, src_i8,  NZ, NY, NX)) return 1;
        if (rt_one("i16 3x4x5", kind, TDC_DT_I16, 2, src_i16, NZ, NY, NX)) return 1;
        if (rt_one("i32 3x4x5", kind, TDC_DT_I32, 4, src_i32, NZ, NY, NX)) return 1;
        if (rt_one("u8  3x4x5", kind, TDC_DT_U8,  1, src_u8,  NZ, NY, NX)) return 1;
        if (rt_one("u16 3x4x5", kind, TDC_DT_U16, 2, src_u16, NZ, NY, NX)) return 1;
        if (rt_one("u32 3x4x5", kind, TDC_DT_U32, 4, src_u32, NZ, NY, NX)) return 1;
    }
    return 0;
}

/* ----- Edge values ------------------------------------------------------ */

static int test_edges_i16(void) {
    /* 2x2x4 volume of i16 edge values. */
    int16_t src[16] = {
        INT16_MIN, INT16_MAX, 0,        -1,
        1,         INT16_MAX, INT16_MIN, 0,
        INT16_MIN, 0,         INT16_MAX, 1,
        -1,        INT16_MIN, INT16_MAX, 0
    };
    static const tdc_pred3d_kind kinds[6] = {
        TDC_PRED3D_LEFT, TDC_PRED3D_UP, TDC_PRED3D_FRONT,
        TDC_PRED3D_AVG3, TDC_PRED3D_GRAD3D, TDC_PRED3D_PAETH3D
    };
    for (int k = 0; k < 6; ++k) {
        if (rt_one("i16 edges 2x2x4", kinds[k], TDC_DT_I16, 2, src, 2, 2, 4)) return 1;
    }
    return 0;
}

/* ----- Auto-select ------------------------------------------------------ */

/* Independent reference scorer. Mirrors the model's pred3d_compute / auto-select
 * loop on a 3x3x4 i32 volume. */
static int64_t ipaeth(int64_t a, int64_t b, int64_t c, int64_t p) {
    int64_t pa = p > a ? p - a : a - p;
    int64_t pb = p > b ? p - b : b - p;
    int64_t pc = p > c ? p - c : c - p;
    int64_t r  = (pb <= pc) ? b : c;
    return (pa <= pb && pa <= pc) ? a : r;
}

static uint64_t score_kind(const int32_t *src,
                           int64_t nz, int64_t ny, int64_t nx,
                           tdc_pred3d_kind kind) {
    int64_t slab = nx * ny;
    uint64_t sum = 0;
    for (int64_t z = 0; z < nz; ++z) {
        for (int64_t y = 0; y < ny; ++y) {
            for (int64_t x = 0; x < nx; ++x) {
                int64_t i = z * slab + y * nx + x;
                int64_t v = (int64_t)src[i];
                int has_a = x > 0, has_b = y > 0, has_c = z > 0;
                int64_t a   = has_a ? (int64_t)src[i - 1]    : 0;
                int64_t b   = has_b ? (int64_t)src[i - nx]   : 0;
                int64_t c   = has_c ? (int64_t)src[i - slab] : 0;
                int64_t ab  = (has_a && has_b)            ? (int64_t)src[i - 1 - nx]        : 0;
                int64_t ac  = (has_a && has_c)            ? (int64_t)src[i - 1 - slab]      : 0;
                int64_t bc  = (has_b && has_c)            ? (int64_t)src[i - nx - slab]     : 0;
                int64_t abc = (has_a && has_b && has_c)   ? (int64_t)src[i - 1 - nx - slab] : 0;
                int64_t pred;
                switch (kind) {
                    case TDC_PRED3D_LEFT:  pred = a; break;
                    case TDC_PRED3D_UP:    pred = b; break;
                    case TDC_PRED3D_FRONT: pred = c; break;
                    case TDC_PRED3D_AVG3: {
                        int cnt = has_a + has_b + has_c;
                        pred = cnt ? (a + b + c) / (int64_t)cnt : 0;
                        break;
                    }
                    case TDC_PRED3D_GRAD3D:
                        pred = a + b + c - ab - ac - bc + abc;
                        break;
                    case TDC_PRED3D_PAETH3D: {
                        int64_t p = a + b + c - ab - ac - bc + abc;
                        pred = ipaeth(a, b, c, p);
                        break;
                    }
                    default: pred = 0; break;
                }
                int64_t res = v - pred;
                sum += (uint64_t)(res < 0 ? -res : res);
            }
        }
    }
    return sum;
}

static int test_auto_argmin(void) {
    /* Volume with mixed structure so the six predictors give different scores. */
    enum { NZ = 3, NY = 3, NX = 4, N = NZ * NY * NX };
    int32_t src[N];
    for (int z = 0; z < NZ; ++z)
        for (int y = 0; y < NY; ++y)
            for (int x = 0; x < NX; ++x)
                src[(z * NY + y) * NX + x] = z * 11 + y * 7 + x * 3;

    if (rt_one("auto argmin", TDC_PRED3D_AUTO, TDC_DT_I32, 4, src, NZ, NY, NX)) return 1;

    static const tdc_pred3d_kind kinds[6] = {
        TDC_PRED3D_LEFT, TDC_PRED3D_UP, TDC_PRED3D_FRONT,
        TDC_PRED3D_AVG3, TDC_PRED3D_GRAD3D, TDC_PRED3D_PAETH3D
    };
    uint64_t best_sum = UINT64_MAX;
    tdc_pred3d_kind best_kind = TDC_PRED3D_GRAD3D;
    for (int k = 0; k < 6; ++k) {
        uint64_t s = score_kind(src, NZ, NY, NX, kinds[k]);
        if (s < best_sum) { best_sum = s; best_kind = kinds[k]; }
    }

    const tdc_model_vt *vt = &tdc_model_pred3d_vt;
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_pred3d_params params = { .kind = TDC_PRED3D_AUTO };
    tdc_block in = volume(src, NZ, NY, NX, TDC_DT_I32);

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

/* ----- Empty + rejections ---------------------------------------------- */

static int test_empty(void) {
    const tdc_model_vt *vt = &tdc_model_pred3d_vt;
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_pred3d_params params = { .kind = TDC_PRED3D_LEFT };

    tdc_block in = volume(NULL, 0, 0, 0, TDC_DT_I32);
    tdc_status st = vt->encode(&in, &params, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "empty encode");
    ASSERT_OR_DIE(residual.size == 0, "empty residual size");
    ASSERT_OR_DIE(side.size == 1, "empty side size");
    ASSERT_OR_DIE(rdt == TDC_DT_I32, "empty residual_dtype");

    tdc_block out = volume(NULL, 0, 0, 0, TDC_DT_I32);
    st = vt->decode(&out, NULL, TDC_DT_I32, NULL, 0, side.data, side.size);
    ASSERT_OR_DIE(st == TDC_OK, "empty decode");

    printf("  [empty] OK\n");

    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

static int test_rejections(void) {
    const tdc_model_vt *vt = &tdc_model_pred3d_vt;
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_pred3d_params params = { .kind = TDC_PRED3D_LEFT };

    /* i64 rejected */
    {
        int64_t src[8] = { 0,1,2,3,4,5,6,7 };
        tdc_block in = volume(src, 2, 2, 2, TDC_DT_I64);
        tdc_status st = vt->encode(&in, &params, &residual, &rdt, &side);
        ASSERT_OR_DIE(st == TDC_E_DTYPE, "i64 should be rejected");
    }

    /* non-VOLUME_3D layout rejected */
    {
        int32_t src[8] = { 0,1,2,3,4,5,6,7 };
        tdc_block in = volume(src, 2, 2, 2, TDC_DT_I32);
        in.layout = TDC_LAYOUT_RASTER_2D;
        in.shape.rank = 2;
        tdc_status st = vt->encode(&in, &params, &residual, &rdt, &side);
        ASSERT_OR_DIE(st == TDC_E_LAYOUT, "RASTER_2D should be rejected");
    }

    /* decode rejects missing side metadata */
    {
        int32_t dst[8] = { 0 };
        tdc_block out = volume(dst, 2, 2, 2, TDC_DT_I32);
        uint8_t residuals[32] = { 0 };
        tdc_status st = vt->decode(&out, NULL, TDC_DT_I32,
                                   residuals, sizeof residuals, NULL, 0);
        ASSERT_OR_DIE(st == TDC_E_CORRUPT, "missing side meta should be rejected");
    }

    /* decode rejects unknown predictor kind in side meta */
    {
        int32_t dst[8] = { 0 };
        tdc_block out = volume(dst, 2, 2, 2, TDC_DT_I32);
        uint8_t residuals[32] = { 0 };
        uint8_t bad_side[1] = { 99 };
        tdc_status st = vt->decode(&out, NULL, TDC_DT_I32,
                                   residuals, sizeof residuals,
                                   bad_side, 1);
        ASSERT_OR_DIE(st == TDC_E_CORRUPT, "unknown kind should be rejected");
    }

    /* decode rejects mismatched residual_size */
    {
        int32_t dst[8] = { 0 };
        tdc_block out = volume(dst, 2, 2, 2, TDC_DT_I32);
        uint8_t residuals[16] = { 0 };
        uint8_t side1[1] = { (uint8_t)TDC_PRED3D_LEFT };
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
    /* 2x3x4 = 24 elements: gradient + sign changes + specials. */
    float src[24] = {
        -3.0f, -2.0f, -1.0f, -0.0f,
         0.0f,  1.0f,  2.5f,  5.0f,
        10.0f, 20.0f, 50.0f, 100.0f,
        200.0f, 1.0e-38f, -1.0e-38f, INFINITY,
        -INFINITY, NAN, 0.5f, -0.5f,
        3.0f, 7.0f, 11.0f, 15.0f
    };
    static const tdc_pred3d_kind kinds[6] = {
        TDC_PRED3D_LEFT, TDC_PRED3D_UP, TDC_PRED3D_FRONT,
        TDC_PRED3D_AVG3, TDC_PRED3D_GRAD3D, TDC_PRED3D_PAETH3D
    };
    for (int k = 0; k < 6; ++k) {
        if (rt_one("f32 2x3x4", kinds[k], TDC_DT_F32, 4, src, 2, 3, 4)) return 1;
    }
    if (rt_one("f32 2x3x4 auto", TDC_PRED3D_AUTO, TDC_DT_F32, 4, src, 2, 3, 4)) return 1;
    return 0;
}

static int test_f64_all_kinds(void) {
    double src[24] = {
        -3.0, -2.0, -1.0, -0.0,
         0.0,  1.0,  2.5,  5.0,
        10.0, 20.0, 50.0, 100.0,
        200.0, 1.0e-300, -1.0e-300, INFINITY,
        -INFINITY, NAN, 0.5, -0.5,
        3.0, 7.0, 11.0, 15.0
    };
    static const tdc_pred3d_kind kinds[6] = {
        TDC_PRED3D_LEFT, TDC_PRED3D_UP, TDC_PRED3D_FRONT,
        TDC_PRED3D_AVG3, TDC_PRED3D_GRAD3D, TDC_PRED3D_PAETH3D
    };
    for (int k = 0; k < 6; ++k) {
        if (rt_one("f64 2x3x4", kinds[k], TDC_DT_F64, 8, src, 2, 3, 4)) return 1;
    }
    if (rt_one("f64 2x3x4 auto", TDC_PRED3D_AUTO, TDC_DT_F64, 8, src, 2, 3, 4)) return 1;
    return 0;
}

static int test_f16_all_kinds(void) {
    /* f16 bit patterns: 2x3x4 = 24 elements. */
    uint16_t src[24] = {
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
        0x4800,  /* +8.0 */
        0x4900,  /* +10.0 */
        0x7C00,  /* +Inf */
        0xFC00,  /* -Inf */
        0x7E00,  /* NaN */
        0x0001,  /* smallest subnormal */
        0x7BFF,  /* largest finite (65504) */
        0x3555,  /* ~0.333 */
        0x3800,  /* 0.5 */
        0xB800,  /* -0.5 */
        0x4A00,  /* +12.0 */
        0x4B00,  /* +14.0 */
        0x4C00,  /* +16.0 */
        0x4D00   /* +20.0 */
    };
    static const tdc_pred3d_kind kinds[6] = {
        TDC_PRED3D_LEFT, TDC_PRED3D_UP, TDC_PRED3D_FRONT,
        TDC_PRED3D_AVG3, TDC_PRED3D_GRAD3D, TDC_PRED3D_PAETH3D
    };
    for (int k = 0; k < 6; ++k) {
        if (rt_one("f16 2x3x4", kinds[k], TDC_DT_F16, 2, src, 2, 3, 4)) return 1;
    }
    if (rt_one("f16 2x3x4 auto", TDC_PRED3D_AUTO, TDC_DT_F16, 2, src, 2, 3, 4)) return 1;
    return 0;
}

static int test_f32_smooth_gradient(void) {
    /* 4x4x8 smooth 3D gradient. */
    float src[128];
    for (int z = 0; z < 4; ++z)
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 8; ++x)
                src[(z * 4 + y) * 8 + x] = 100.0f +
                    (float)z * 3.0f + (float)y * 0.7f + (float)x * 0.3f;
    return rt_one("f32 4x4x8 gradient", TDC_PRED3D_GRAD3D,
                  TDC_DT_F32, 4, src, 4, 4, 8);
}

static int test_registry(void) {
    const tdc_model_vt *vt = tdc_model_get(TDC_MODEL_PRED_3D);
    ASSERT_OR_DIE(vt == &tdc_model_pred3d_vt,
                  "tdc_model_get(TDC_MODEL_PRED_3D) wiring");
    ASSERT_OR_DIE(vt->id == TDC_MODEL_PRED_3D, "vt->id mismatch");
    printf("  [registry wiring] OK\n");
    return 0;
}

int main(void) {
    printf("test_pred3d_roundtrip\n");
    if (test_dtype_all_kinds()) return 1;
    if (test_edges_i16())       return 1;
    if (test_auto_argmin())     return 1;
    if (test_f32_all_kinds())   return 1;
    if (test_f64_all_kinds())   return 1;
    if (test_f16_all_kinds())   return 1;
    if (test_f32_smooth_gradient()) return 1;
    if (test_empty())           return 1;
    if (test_rejections())      return 1;
    if (test_registry())        return 1;
    printf("ALL OK\n");
    return 0;
}
