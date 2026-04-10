/*
 * tests/test_bitshuffle_roundtrip.c
 *
 * Round-trip test for the TDC_XFORM_BIT_SHUFFLE transform.
 *
 * Verifies:
 *   1. Round-trip on i8/i16/i32/i64/u8/u16/u32/u64/f32/f64 dtypes.
 *   2. Round-trip with n_elems that is not a multiple of 8.
 *   3. Empty input (n == 0) round-trips.
 *   4. Single element round-trips.
 *   5. Highly compressible data (all zeros) round-trips.
 *   6. Registry returns &tdc_xform_bit_shuffle_vt for TDC_XFORM_BIT_SHUFFLE.
 */

#include "tdc/transform.h"
#include "tdc/codec.h"
#include "tdc/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern const tdc_xform_vt tdc_xform_bit_shuffle_vt;

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

static int rt_one(const char *label, tdc_dtype dt, size_t elem_size,
                  const void *src, size_t n_elems) {
    const tdc_xform_vt *vt = &tdc_xform_bit_shuffle_vt;
    size_t src_size = n_elems * elem_size;

    tdc_buffer enc = make_buffer();
    tdc_dtype  out_dt = (tdc_dtype)0;

    tdc_status st = vt->encode((const uint8_t *)src, src_size, dt,
                               NULL, &enc, &out_dt);
    ASSERT_OR_DIE(st == TDC_OK, "encode returned non-OK");
    ASSERT_OR_DIE(out_dt == dt, "out_dtype != in_dtype");

    /* Validate encoded size. */
    uint32_t n_planes = (uint32_t)elem_size * 8u;
    uint32_t bpp = ((uint32_t)n_elems + 7u) / 8u;
    size_t expected_enc = (size_t)n_planes * bpp;
    ASSERT_OR_DIE(enc.size == expected_enc, "encoded size mismatch");

    /* Decode */
    uint8_t *dst = (uint8_t *)malloc(src_size > 0 ? src_size : 1u);
    ASSERT_OR_DIE(dst != NULL, "malloc dst");
    memset(dst, 0xAA, src_size > 0 ? src_size : 1u);

    tdc_dtype dec_dt = (tdc_dtype)0;
    st = vt->decode(enc.data, enc.size, dt, NULL, dst, src_size, &dec_dt);
    ASSERT_OR_DIE(st == TDC_OK, "decode returned non-OK");
    ASSERT_OR_DIE(dec_dt == dt, "decode out_dtype mismatch");

    if (src_size > 0) {
        ASSERT_OR_DIE(memcmp(dst, src, src_size) == 0,
                      "decoded data does not match source");
    }

    printf("  [%s] n=%zu elem=%zu enc=%zu round-trip OK\n",
           label, n_elems, elem_size, enc.size);

    free(dst);
    free_buffer(&enc);
    return 0;
}

/* ----- Test suites -------------------------------------------------------- */

static int test_dtypes(void) {
    /* 16 elements (multiple of 8) of each dtype. */
    int8_t   i8[16];
    int16_t  i16[16];
    int32_t  i32[16];
    int64_t  i64[16];
    uint8_t  u8[16];
    uint16_t u16[16];
    uint32_t u32[16];
    uint64_t u64[16];
    float    f32[16];
    double   f64[16];

    for (int i = 0; i < 16; ++i) {
        i8[i]  = (int8_t)(i * 7 - 50);
        i16[i] = (int16_t)(i * 137 - 1000);
        i32[i] = (int32_t)(i * 100003 - 800000);
        i64[i] = (int64_t)i * 1000000007LL;
        u8[i]  = (uint8_t)(i * 17);
        u16[i] = (uint16_t)(i * 4099);
        u32[i] = (uint32_t)(i * 100003u);
        u64[i] = (uint64_t)i * 1000000007ULL;
        f32[i] = (float)i * 1.5f - 10.0f;
        f64[i] = (double)i * 2.7 - 20.0;
    }

    if (rt_one("i8",  TDC_DT_I8,  1, i8,  16)) return 1;
    if (rt_one("i16", TDC_DT_I16, 2, i16, 16)) return 1;
    if (rt_one("i32", TDC_DT_I32, 4, i32, 16)) return 1;
    if (rt_one("i64", TDC_DT_I64, 8, i64, 16)) return 1;
    if (rt_one("u8",  TDC_DT_U8,  1, u8,  16)) return 1;
    if (rt_one("u16", TDC_DT_U16, 2, u16, 16)) return 1;
    if (rt_one("u32", TDC_DT_U32, 4, u32, 16)) return 1;
    if (rt_one("u64", TDC_DT_U64, 8, u64, 16)) return 1;
    if (rt_one("f32", TDC_DT_F32, 4, f32, 16)) return 1;
    if (rt_one("f64", TDC_DT_F64, 8, f64, 16)) return 1;
    return 0;
}

static int test_non_multiple_of_8(void) {
    /* 13 elements — not a multiple of 8. */
    int32_t src[13];
    for (int i = 0; i < 13; ++i) src[i] = i * 31 - 200;
    if (rt_one("i32 n=13", TDC_DT_I32, 4, src, 13)) return 1;

    /* 1 element. */
    int32_t one[1] = { 42 };
    if (rt_one("i32 n=1", TDC_DT_I32, 4, one, 1)) return 1;

    /* 7 elements of u16. */
    uint16_t u7[7] = { 0, 1, 2, 100, 200, 300, 65535 };
    if (rt_one("u16 n=7", TDC_DT_U16, 2, u7, 7)) return 1;

    return 0;
}

static int test_empty(void) {
    const tdc_xform_vt *vt = &tdc_xform_bit_shuffle_vt;
    tdc_buffer enc = make_buffer();
    tdc_dtype out_dt = (tdc_dtype)0;

    tdc_status st = vt->encode(NULL, 0, TDC_DT_I32, NULL, &enc, &out_dt);
    ASSERT_OR_DIE(st == TDC_OK, "empty encode");
    ASSERT_OR_DIE(enc.size == 0, "empty enc size");
    ASSERT_OR_DIE(out_dt == TDC_DT_I32, "empty out_dtype");

    printf("  [empty] OK\n");
    free_buffer(&enc);
    return 0;
}

static int test_all_zeros(void) {
    uint32_t src[32];
    memset(src, 0, sizeof src);
    return rt_one("u32 zeros n=32", TDC_DT_U32, 4, src, 32);
}

static int test_registry(void) {
    const tdc_xform_vt *vt = tdc_xform_get(TDC_XFORM_BIT_SHUFFLE);
    ASSERT_OR_DIE(vt == &tdc_xform_bit_shuffle_vt,
                  "tdc_xform_get(TDC_XFORM_BIT_SHUFFLE) wiring");
    ASSERT_OR_DIE(vt->id == TDC_XFORM_BIT_SHUFFLE, "vt->id mismatch");
    printf("  [registry wiring] OK\n");
    return 0;
}

int main(void) {
    printf("test_bitshuffle_roundtrip\n");
    if (test_dtypes())           return 1;
    if (test_non_multiple_of_8()) return 1;
    if (test_empty())            return 1;
    if (test_all_zeros())        return 1;
    if (test_registry())         return 1;
    printf("ALL OK\n");
    return 0;
}
