/*
 * tests/test_raw_roundtrip.c
 *
 * Round-trip test for the TDC_MODEL_RAW backend.
 *
 * Verifies:
 *   1. residual_dtype == in->dtype, side metadata is empty.
 *   2. residual bytes == data bytes (memcpy identity).
 *   3. Round-trip across every supported numeric dtype
 *      (i8/i16/i32/i64/u8/u16/u32/u64/f32/f64).
 *   4. Round-trip across every supported layout
 *      (VECTOR_1D, RASTER_2D, STACK_2D, VOLUME_3D) — RAW must work
 *      for any layout the model dispatcher could feed it.
 *   5. Empty block (n == 0) round-trips.
 *   6. Encoder rejects TDC_DT_STRING.
 *   7. Encoder rejects rank that disagrees with layout.
 *   8. Decoder rejects non-empty side_size and mismatched residual_size.
 *   9. Registry returns &tdc_model_raw_vt for TDC_MODEL_RAW.
 */

#include "tdc/model.h"
#include "tdc/codec.h"
#include "tdc/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern const tdc_model_vt tdc_model_raw_vt;

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

/* ----- Generic round-trip helper --------------------------------------- */

static int rt_block(const char *label, tdc_block *in, size_t bytes) {
    const tdc_model_vt *vt = &tdc_model_raw_vt;

    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;

    tdc_status st = vt->encode(in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "encode returned non-OK");
    ASSERT_OR_DIE(rdt == in->dtype, "residual_dtype != input dtype");
    ASSERT_OR_DIE(side.size == 0, "raw must not emit side metadata");
    ASSERT_OR_DIE(residual.size == bytes, "residual size mismatch");
    if (bytes > 0) {
        ASSERT_OR_DIE(memcmp(residual.data, in->data, bytes) == 0,
                      "residual bytes must equal source bytes");
    }

    /* Decode into a fresh buffer using the same shape envelope. */
    void *dst = malloc(bytes > 0 ? bytes : 1u);
    ASSERT_OR_DIE(dst != NULL, "malloc dst");
    memset(dst, 0xAA, bytes > 0 ? bytes : 1u);

    tdc_block out = *in;
    out.data = dst;

    st = vt->decode(&out, NULL, rdt, residual.data, residual.size, NULL, 0);
    ASSERT_OR_DIE(st == TDC_OK, "decode returned non-OK");
    if (bytes > 0) {
        ASSERT_OR_DIE(memcmp(dst, in->data, bytes) == 0,
                      "decoded data does not match source");
    }

    printf("  [%s] bytes=%zu round-trip OK\n", label, bytes);

    free(dst);
    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

/* ----- Per-dtype VECTOR_1D tests --------------------------------------- */

static tdc_block vec1d(void *data, int64_t n, tdc_dtype dt) {
    tdc_block b = {0};
    b.data = data;
    b.dtype = dt;
    b.layout = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank = 1;
    b.shape.dim[0] = n;
    b.shape.stride[0] = 1;
    return b;
}

static int test_dtypes(void) {
    int8_t   i8s[8]  = { 0, 1, -1, 127, -128, 5, -5, 0 };
    int16_t  i16s[4] = { 0, -1, 32767, -32768 };
    int32_t  i32s[4] = { 0, -1, INT32_MAX, INT32_MIN };
    int64_t  i64s[4] = { 0, -1, INT64_MAX, INT64_MIN };
    uint8_t  u8s[8]  = { 0, 1, 255, 254, 0, 128, 127, 255 };
    uint16_t u16s[4] = { 0, 65535, 32768, 1 };
    uint32_t u32s[4] = { 0u, UINT32_MAX, 0x80000000u, 1u };
    uint64_t u64s[4] = { 0u, UINT64_MAX, 0x8000000000000000ull, 1u };
    float    f32s[4] = { 0.0f, -1.5f, 3.14f, 1e30f };
    double   f64s[4] = { 0.0, -1.5, 3.14159, 1e300 };

    tdc_block b;
    b = vec1d(i8s,  8, TDC_DT_I8);  if (rt_block("i8 vec1d",  &b, 8))  return 1;
    b = vec1d(i16s, 4, TDC_DT_I16); if (rt_block("i16 vec1d", &b, 8))  return 1;
    b = vec1d(i32s, 4, TDC_DT_I32); if (rt_block("i32 vec1d", &b, 16)) return 1;
    b = vec1d(i64s, 4, TDC_DT_I64); if (rt_block("i64 vec1d", &b, 32)) return 1;
    b = vec1d(u8s,  8, TDC_DT_U8);  if (rt_block("u8 vec1d",  &b, 8))  return 1;
    b = vec1d(u16s, 4, TDC_DT_U16); if (rt_block("u16 vec1d", &b, 8))  return 1;
    b = vec1d(u32s, 4, TDC_DT_U32); if (rt_block("u32 vec1d", &b, 16)) return 1;
    b = vec1d(u64s, 4, TDC_DT_U64); if (rt_block("u64 vec1d", &b, 32)) return 1;
    b = vec1d(f32s, 4, TDC_DT_F32); if (rt_block("f32 vec1d", &b, 16)) return 1;
    b = vec1d(f64s, 4, TDC_DT_F64); if (rt_block("f64 vec1d", &b, 32)) return 1;
    return 0;
}

/* ----- Layout tests ----------------------------------------------------- */

static int test_raster_2d(void) {
    /* 3 x 4 i32 raster */
    int32_t src[12];
    for (int i = 0; i < 12; ++i) src[i] = i * i - 7;

    tdc_block b = {0};
    b.data = src;
    b.dtype = TDC_DT_I32;
    b.layout = TDC_LAYOUT_RASTER_2D;
    b.shape.rank = 2;
    b.shape.dim[0] = 3;
    b.shape.dim[1] = 4;
    b.shape.stride[0] = 4;
    b.shape.stride[1] = 1;
    return rt_block("i32 raster2d 3x4", &b, sizeof src);
}

static int test_stack_2d(void) {
    /* 2 slices x 3 rows x 5 cols of u16 */
    uint16_t src[2 * 3 * 5];
    for (int i = 0; i < 30; ++i) src[i] = (uint16_t)(i * 1009u);

    tdc_block b = {0};
    b.data = src;
    b.dtype = TDC_DT_U16;
    b.layout = TDC_LAYOUT_STACK_2D;
    b.shape.rank = 3;
    b.shape.dim[0] = 2;
    b.shape.dim[1] = 3;
    b.shape.dim[2] = 5;
    b.shape.stride[0] = 15;
    b.shape.stride[1] = 5;
    b.shape.stride[2] = 1;
    return rt_block("u16 stack2d 2x3x5", &b, sizeof src);
}

static int test_volume_3d(void) {
    /* 2 x 2 x 2 f64 volume */
    double src[8] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

    tdc_block b = {0};
    b.data = src;
    b.dtype = TDC_DT_F64;
    b.layout = TDC_LAYOUT_VOLUME_3D;
    b.shape.rank = 3;
    b.shape.dim[0] = 2;
    b.shape.dim[1] = 2;
    b.shape.dim[2] = 2;
    b.shape.stride[0] = 4;
    b.shape.stride[1] = 2;
    b.shape.stride[2] = 1;
    return rt_block("f64 volume3d 2x2x2", &b, sizeof src);
}

/* ----- Edges & rejections ----------------------------------------------- */

static int test_empty(void) {
    const tdc_model_vt *vt = &tdc_model_raw_vt;
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;

    tdc_block in = vec1d(NULL, 0, TDC_DT_I32);
    tdc_status st = vt->encode(&in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "empty encode");
    ASSERT_OR_DIE(residual.size == 0, "empty residual size");
    ASSERT_OR_DIE(side.size == 0, "empty side size");
    ASSERT_OR_DIE(rdt == TDC_DT_I32, "empty residual_dtype");

    tdc_block out = vec1d(NULL, 0, TDC_DT_I32);
    st = vt->decode(&out, NULL, TDC_DT_I32, NULL, 0, NULL, 0);
    ASSERT_OR_DIE(st == TDC_OK, "empty decode");

    printf("  [empty] OK\n");

    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

static int test_rejections(void) {
    const tdc_model_vt *vt = &tdc_model_raw_vt;
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;

    /* string dtype rejected */
    {
        uint8_t heap[4] = { 'a','b','c','d' };
        tdc_block in = vec1d(heap, 2, TDC_DT_STRING);
        tdc_status st = vt->encode(&in, NULL, &residual, &rdt, &side);
        ASSERT_OR_DIE(st == TDC_E_DTYPE, "string should be rejected");
    }

    /* rank disagrees with layout: VECTOR_1D with rank 2 */
    {
        int32_t src[4] = { 0, 1, 2, 3 };
        tdc_block in = vec1d(src, 4, TDC_DT_I32);
        in.shape.rank = 2;
        tdc_status st = vt->encode(&in, NULL, &residual, &rdt, &side);
        ASSERT_OR_DIE(st == TDC_E_SHAPE, "rank/layout mismatch should be rejected");
    }

    /* decode rejects non-empty side metadata */
    {
        int32_t dst[4] = { 0 };
        tdc_block out = vec1d(dst, 4, TDC_DT_I32);
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
        tdc_block out = vec1d(dst, 4, TDC_DT_I32);
        uint8_t residuals[8] = { 0 };
        tdc_status st = vt->decode(&out, NULL, TDC_DT_I32,
                                   residuals, sizeof residuals,
                                   NULL, 0);
        ASSERT_OR_DIE(st == TDC_E_CORRUPT, "wrong residual_size should be rejected");
    }

    /* decode rejects residual_dtype != out->dtype */
    {
        int32_t dst[4] = { 0 };
        tdc_block out = vec1d(dst, 4, TDC_DT_I32);
        uint8_t residuals[16] = { 0 };
        tdc_status st = vt->decode(&out, NULL, TDC_DT_I64,
                                   residuals, sizeof residuals,
                                   NULL, 0);
        ASSERT_OR_DIE(st == TDC_E_DTYPE, "residual_dtype mismatch should be rejected");
    }

    printf("  [rejections] OK\n");

    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

static int test_registry(void) {
    const tdc_model_vt *vt = tdc_model_get(TDC_MODEL_RAW);
    ASSERT_OR_DIE(vt == &tdc_model_raw_vt,
                  "tdc_model_get(TDC_MODEL_RAW) wiring");
    ASSERT_OR_DIE(vt->id == TDC_MODEL_RAW, "vt->id mismatch");
    printf("  [registry wiring] OK\n");
    return 0;
}

int main(void) {
    printf("test_raw_roundtrip\n");
    if (test_dtypes())     return 1;
    if (test_raster_2d())  return 1;
    if (test_stack_2d())   return 1;
    if (test_volume_3d())  return 1;
    if (test_empty())      return 1;
    if (test_rejections()) return 1;
    if (test_registry())   return 1;
    printf("ALL OK\n");
    return 0;
}
