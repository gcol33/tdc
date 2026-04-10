/*
 * tests/test_stack2d_roundtrip.c
 *
 * Round-trip test for the TDC_MODEL_STACK_2D backend.
 *
 * Verifies:
 *   1. Round-trip on i32/u8/f32 with all four pred2d kinds on a small
 *      synthetic STACK_2D block (3 slices x 4 rows x 5 cols).
 *   2. Inter-slice delta mode round-trips correctly.
 *   3. AUTO mode picks a kind and round-trips correctly.
 *   4. Empty block (n == 0) round-trips.
 *   5. Encoder rejects non-STACK_2D layouts.
 *   6. Decoder rejects missing / wrong-size side metadata.
 *   7. Registry returns &tdc_model_stack2d_vt for TDC_MODEL_STACK_2D.
 */

#include "tdc/model.h"
#include "tdc/codec.h"
#include "tdc/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern const tdc_model_vt tdc_model_stack2d_vt;

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

static tdc_block stack_block(void *data, int64_t nz, int64_t ny, int64_t nx,
                             tdc_dtype dt) {
    tdc_block b = {0};
    b.data = data;
    b.dtype = dt;
    b.layout = TDC_LAYOUT_STACK_2D;
    b.shape.rank = 3;
    b.shape.dim[0] = nz;
    b.shape.dim[1] = ny;
    b.shape.dim[2] = nx;
    b.shape.stride[0] = ny * nx;
    b.shape.stride[1] = nx;
    b.shape.stride[2] = 1;
    return b;
}

/* ----- Generic round-trip ---------------------------------------------- */

static int rt_one(const char *label, tdc_pred2d_kind kind, int inter_slice,
                  tdc_dtype dt, size_t elem_size,
                  const void *src, int64_t nz, int64_t ny, int64_t nx) {
    const tdc_model_vt *vt = &tdc_model_stack2d_vt;

    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;

    tdc_stack2d_params params = { .kind = kind, .inter_slice = inter_slice };
    tdc_block in = stack_block((void *)src, nz, ny, nx, dt);
    size_t bytes = (size_t)(nz * ny * nx) * elem_size;

    tdc_status st = vt->encode(&in, &params, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "encode returned non-OK");
    ASSERT_OR_DIE(rdt == dt, "residual_dtype != input dtype");
    ASSERT_OR_DIE(side.size == 2u, "side metadata must be exactly 2 bytes");
    ASSERT_OR_DIE(residual.size == bytes, "residual size mismatch");

    /* Verify side meta. */
    if (kind != TDC_PRED2D_AUTO) {
        ASSERT_OR_DIE(side.data[0] == (uint8_t)kind,
                      "side metadata must record the requested kind");
    } else {
        uint8_t resolved = side.data[0];
        ASSERT_OR_DIE(resolved >= TDC_PRED2D_LEFT && resolved <= TDC_PRED2D_PAETH,
                      "AUTO must resolve to one of the four kinds");
    }
    ASSERT_OR_DIE(side.data[1] == (uint8_t)inter_slice,
                  "side meta inter_slice flag mismatch");

    /* Decode */
    void *dst = malloc(bytes > 0 ? bytes : 1u);
    ASSERT_OR_DIE(dst != NULL, "malloc dst");
    memset(dst, 0xAA, bytes > 0 ? bytes : 1u);

    tdc_block out = stack_block(dst, nz, ny, nx, dt);
    st = vt->decode(&out, NULL, rdt, residual.data, residual.size,
                    side.data, side.size);
    ASSERT_OR_DIE(st == TDC_OK, "decode returned non-OK");
    if (bytes > 0) {
        ASSERT_OR_DIE(memcmp(dst, src, bytes) == 0,
                      "decoded data does not match source");
    }

    printf("  [%s kind=%u inter=%d] %lldx%lldx%lld round-trip OK\n",
           label, (unsigned)side.data[0], inter_slice,
           (long long)nz, (long long)ny, (long long)nx);

    free(dst);
    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

/* ----- Test: i32 all kinds ----------------------------------------------- */

static int test_i32_all_kinds(void) {
    /* 3 slices x 4 rows x 5 cols. Per-slice gradient + inter-slice offset. */
    int32_t src[60];
    for (int s = 0; s < 3; ++s)
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 5; ++c)
                src[s * 20 + r * 5 + c] = 100 + s * 50 + r * 3 + c * 7;

    static const tdc_pred2d_kind kinds[4] = {
        TDC_PRED2D_LEFT, TDC_PRED2D_UP, TDC_PRED2D_AVERAGE, TDC_PRED2D_PAETH
    };
    for (int k = 0; k < 4; ++k) {
        if (rt_one("i32 3x4x5", kinds[k], 0, TDC_DT_I32, 4, src, 3, 4, 5))
            return 1;
    }
    return 0;
}

/* ----- Test: inter-slice delta ------------------------------------------- */

static int test_inter_slice(void) {
    /* Slices with high inter-slice correlation. */
    int32_t src[60];
    for (int s = 0; s < 3; ++s)
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 5; ++c)
                src[s * 20 + r * 5 + c] = 1000 + s * 2 + r * 3 + c * 7;

    static const tdc_pred2d_kind kinds[4] = {
        TDC_PRED2D_LEFT, TDC_PRED2D_UP, TDC_PRED2D_AVERAGE, TDC_PRED2D_PAETH
    };
    for (int k = 0; k < 4; ++k) {
        if (rt_one("i32 inter-slice", kinds[k], 1, TDC_DT_I32, 4, src, 3, 4, 5))
            return 1;
    }
    return 0;
}

/* ----- Test: u8 round-trip ----------------------------------------------- */

static int test_u8(void) {
    uint8_t src[60];
    for (int i = 0; i < 60; ++i) src[i] = (uint8_t)((i * 7 + 13) & 0xFF);
    return rt_one("u8 3x4x5", TDC_PRED2D_LEFT, 0, TDC_DT_U8, 1, src, 3, 4, 5);
}

/* ----- Test: f32 round-trip ---------------------------------------------- */

static int test_f32(void) {
    float src[60];
    for (int s = 0; s < 3; ++s)
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 5; ++c)
                src[s * 20 + r * 5 + c] = 10.0f + (float)s * 5.0f +
                                           (float)r * 0.7f + (float)c * 0.3f;
    return rt_one("f32 3x4x5", TDC_PRED2D_PAETH, 0, TDC_DT_F32, 4, src, 3, 4, 5);
}

/* ----- Test: AUTO mode --------------------------------------------------- */

static int test_auto(void) {
    int32_t src[60];
    for (int s = 0; s < 3; ++s)
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 5; ++c)
                src[s * 20 + r * 5 + c] = s * 100 + r * 10 + c;

    if (rt_one("auto no-delta", TDC_PRED2D_AUTO, 0, TDC_DT_I32, 4, src, 3, 4, 5))
        return 1;
    if (rt_one("auto inter-slice", TDC_PRED2D_AUTO, 1, TDC_DT_I32, 4, src, 3, 4, 5))
        return 1;
    return 0;
}

/* ----- Test: empty ------------------------------------------------------- */

static int test_empty(void) {
    const tdc_model_vt *vt = &tdc_model_stack2d_vt;
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_stack2d_params params = { .kind = TDC_PRED2D_LEFT, .inter_slice = 0 };

    tdc_block in = stack_block(NULL, 0, 0, 0, TDC_DT_I32);
    tdc_status st = vt->encode(&in, &params, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "empty encode");
    ASSERT_OR_DIE(residual.size == 0, "empty residual size");
    ASSERT_OR_DIE(side.size == 2, "empty side size");

    tdc_block out = stack_block(NULL, 0, 0, 0, TDC_DT_I32);
    st = vt->decode(&out, NULL, TDC_DT_I32, NULL, 0, side.data, side.size);
    ASSERT_OR_DIE(st == TDC_OK, "empty decode");

    printf("  [empty] OK\n");
    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

/* ----- Test: rejections -------------------------------------------------- */

static int test_rejections(void) {
    const tdc_model_vt *vt = &tdc_model_stack2d_vt;
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_stack2d_params params = { .kind = TDC_PRED2D_LEFT, .inter_slice = 0 };

    /* non-STACK_2D layout rejected */
    {
        int32_t src[20] = {0};
        tdc_block in = {0};
        in.data = src;
        in.dtype = TDC_DT_I32;
        in.layout = TDC_LAYOUT_RASTER_2D;
        in.shape.rank = 2;
        in.shape.dim[0] = 4;
        in.shape.dim[1] = 5;
        tdc_status st = vt->encode(&in, &params, &residual, &rdt, &side);
        ASSERT_OR_DIE(st == TDC_E_LAYOUT, "RASTER_2D should be rejected");
    }

    /* decode rejects missing side metadata */
    {
        int32_t dst[60] = {0};
        tdc_block out = stack_block(dst, 3, 4, 5, TDC_DT_I32);
        uint8_t residuals[240] = {0};
        tdc_status st = vt->decode(&out, NULL, TDC_DT_I32,
                                   residuals, sizeof residuals,
                                   NULL, 0);
        ASSERT_OR_DIE(st == TDC_E_CORRUPT, "missing side meta should be rejected");
    }

    /* decode rejects wrong side size */
    {
        int32_t dst[60] = {0};
        tdc_block out = stack_block(dst, 3, 4, 5, TDC_DT_I32);
        uint8_t residuals[240] = {0};
        uint8_t bad_side[1] = {1};
        tdc_status st = vt->decode(&out, NULL, TDC_DT_I32,
                                   residuals, sizeof residuals,
                                   bad_side, 1);
        ASSERT_OR_DIE(st == TDC_E_CORRUPT, "wrong side size should be rejected");
    }

    printf("  [rejections] OK\n");
    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

/* ----- Test: single slice ------------------------------------------------ */

static int test_single_slice(void) {
    int32_t src[20];
    for (int i = 0; i < 20; ++i) src[i] = i * 3;
    return rt_one("1x4x5", TDC_PRED2D_PAETH, 0, TDC_DT_I32, 4, src, 1, 4, 5);
}

/* ----- Test: registry ---------------------------------------------------- */

static int test_registry(void) {
    const tdc_model_vt *vt = tdc_model_get(TDC_MODEL_STACK_2D);
    ASSERT_OR_DIE(vt == &tdc_model_stack2d_vt,
                  "tdc_model_get(TDC_MODEL_STACK_2D) wiring");
    ASSERT_OR_DIE(vt->id == TDC_MODEL_STACK_2D, "vt->id mismatch");
    printf("  [registry wiring] OK\n");
    return 0;
}

int main(void) {
    printf("test_stack2d_roundtrip\n");
    if (test_i32_all_kinds()) return 1;
    if (test_inter_slice())   return 1;
    if (test_u8())            return 1;
    if (test_f32())           return 1;
    if (test_auto())          return 1;
    if (test_empty())         return 1;
    if (test_rejections())    return 1;
    if (test_single_slice())  return 1;
    if (test_registry())      return 1;
    printf("ALL OK\n");
    return 0;
}
