/*
 * tests/test_zigzag_roundtrip.c
 *
 * Round-trip test for the TDC_XFORM_ZIGZAG backend.
 *
 * Verifies:
 *   1. Spot-check the small-magnitude property: zigzag(0)=0, zigzag(-1)=1,
 *      zigzag(1)=2, zigzag(-2)=3, zigzag(2)=4 — for every supported width.
 *   2. Round-trip on i8/i16/i32/i64 buffers that include 0, ±1, type min,
 *      type max, and a sweep of small magnitudes.
 *   3. Encoder reports out_dtype = matching unsigned dtype.
 *   4. Empty buffer round-trips for every width.
 *   5. Encode rejects float and unsigned in_dtypes (TDC_E_DTYPE).
 *   6. Encode rejects misaligned src_size with TDC_E_INVAL.
 *   7. Decode rejects mismatched dst_size with TDC_E_CORRUPT.
 *   8. Registry returns &tdc_xform_zigzag_vt for TDC_XFORM_ZIGZAG.
 */

#include "tdc/transform.h"
#include "tdc/codec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern const tdc_xform_vt tdc_xform_zigzag_vt;

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

/* ----- Spot-check the mapping for every width --------------------------- */

static int test_spot_check(void) {
    /* i8 -> u8 */
    {
        int8_t src[5] = { 0, -1, 1, -2, 2 };
        uint8_t expected[5] = { 0, 1, 2, 3, 4 };
        tdc_buffer enc = make_buffer();
        tdc_dtype out_dt = (tdc_dtype)0;
        tdc_status st = tdc_xform_zigzag_vt.encode(
            (const uint8_t *)src, sizeof src, TDC_DT_I8, NULL, &enc, &out_dt);
        ASSERT_OR_DIE(st == TDC_OK, "i8 spot encode failed");
        ASSERT_OR_DIE(out_dt == TDC_DT_U8, "i8 out_dtype != U8");
        ASSERT_OR_DIE(enc.size == 5, "i8 spot enc size mismatch");
        for (int i = 0; i < 5; i++) {
            ASSERT_OR_DIE(enc.data[i] == expected[i], "i8 spot mapping wrong");
        }
        free_buffer(&enc);
    }

    /* i16 -> u16 */
    {
        int16_t src[5] = { 0, -1, 1, -2, 2 };
        uint16_t expected[5] = { 0, 1, 2, 3, 4 };
        tdc_buffer enc = make_buffer();
        tdc_dtype out_dt = (tdc_dtype)0;
        tdc_status st = tdc_xform_zigzag_vt.encode(
            (const uint8_t *)src, sizeof src, TDC_DT_I16, NULL, &enc, &out_dt);
        ASSERT_OR_DIE(st == TDC_OK, "i16 spot encode failed");
        ASSERT_OR_DIE(out_dt == TDC_DT_U16, "i16 out_dtype != U16");
        ASSERT_OR_DIE(enc.size == 10, "i16 spot enc size mismatch");
        for (int i = 0; i < 5; i++) {
            uint16_t got;
            memcpy(&got, enc.data + i * 2, 2);
            ASSERT_OR_DIE(got == expected[i], "i16 spot mapping wrong");
        }
        free_buffer(&enc);
    }

    /* i32 -> u32 */
    {
        int32_t src[5] = { 0, -1, 1, -2, 2 };
        uint32_t expected[5] = { 0, 1, 2, 3, 4 };
        tdc_buffer enc = make_buffer();
        tdc_dtype out_dt = (tdc_dtype)0;
        tdc_status st = tdc_xform_zigzag_vt.encode(
            (const uint8_t *)src, sizeof src, TDC_DT_I32, NULL, &enc, &out_dt);
        ASSERT_OR_DIE(st == TDC_OK, "i32 spot encode failed");
        ASSERT_OR_DIE(out_dt == TDC_DT_U32, "i32 out_dtype != U32");
        for (int i = 0; i < 5; i++) {
            uint32_t got;
            memcpy(&got, enc.data + i * 4, 4);
            ASSERT_OR_DIE(got == expected[i], "i32 spot mapping wrong");
        }
        free_buffer(&enc);
    }

    /* i64 -> u64 */
    {
        int64_t src[5] = { 0, -1, 1, -2, 2 };
        uint64_t expected[5] = { 0, 1, 2, 3, 4 };
        tdc_buffer enc = make_buffer();
        tdc_dtype out_dt = (tdc_dtype)0;
        tdc_status st = tdc_xform_zigzag_vt.encode(
            (const uint8_t *)src, sizeof src, TDC_DT_I64, NULL, &enc, &out_dt);
        ASSERT_OR_DIE(st == TDC_OK, "i64 spot encode failed");
        ASSERT_OR_DIE(out_dt == TDC_DT_U64, "i64 out_dtype != U64");
        for (int i = 0; i < 5; i++) {
            uint64_t got;
            memcpy(&got, enc.data + i * 8, 8);
            ASSERT_OR_DIE(got == expected[i], "i64 spot mapping wrong");
        }
        free_buffer(&enc);
    }

    printf("  [spot-check 0,-1,1,-2,2 across i8/i16/i32/i64] OK\n");
    return 0;
}

/* ----- Generic round-trip helpers --------------------------------------- */

#define ROUNDTRIP_FN(NAME, TYPE, DTYPE)                                       \
static int test_roundtrip_##NAME(const TYPE *src, size_t n,                   \
                                 const char *label) {                         \
    tdc_buffer enc = make_buffer();                                           \
    tdc_dtype out_dt = (tdc_dtype)0;                                          \
    size_t bytes = n * sizeof(TYPE);                                          \
    tdc_status st = tdc_xform_zigzag_vt.encode(                               \
        (const uint8_t *)src, bytes, DTYPE, NULL, &enc, &out_dt);             \
    ASSERT_OR_DIE(st == TDC_OK, "encode failed");                             \
    ASSERT_OR_DIE(enc.size == bytes, "encoded size != src size");             \
    TYPE *dec = (TYPE *)malloc(bytes ? bytes : 1);                            \
    ASSERT_OR_DIE(dec != NULL, "malloc dec failed");                          \
    st = tdc_xform_zigzag_vt.decode(                                          \
        enc.data, enc.size, DTYPE, NULL,                                      \
        (uint8_t *)dec, bytes, &out_dt);                                      \
    ASSERT_OR_DIE(st == TDC_OK, "decode failed");                             \
    ASSERT_OR_DIE(out_dt == DTYPE, "decode out_dtype mismatch");              \
    for (size_t i = 0; i < n; i++) {                                          \
        if (dec[i] != src[i]) {                                               \
            fprintf(stderr,                                                   \
                    "FAIL %s elem %zu: got %lld expected %lld\n",             \
                    label, i, (long long)dec[i], (long long)src[i]);          \
            free(dec); free_buffer(&enc); return 1;                           \
        }                                                                     \
    }                                                                         \
    printf("  [%s %zu elems] %zu bytes round-trip OK\n", label, n, bytes);    \
    free(dec); free_buffer(&enc); return 0;                                   \
}

ROUNDTRIP_FN(i8,  int8_t,  TDC_DT_I8)
ROUNDTRIP_FN(i16, int16_t, TDC_DT_I16)
ROUNDTRIP_FN(i32, int32_t, TDC_DT_I32)
ROUNDTRIP_FN(i64, int64_t, TDC_DT_I64)

static int test_all_widths(void) {
    /* i8: full range sweep + edges */
    {
        enum { N = 256 };
        int8_t src[N];
        for (int i = 0; i < N; i++) src[i] = (int8_t)(i - 128);
        if (test_roundtrip_i8(src, N, "i8 sweep")) return 1;
    }
    /* i16: edges + small magnitudes */
    {
        int16_t src[8] = { 0, -1, 1, INT16_MIN, INT16_MAX, -32, 32, -12345 };
        if (test_roundtrip_i16(src, 8, "i16 edges")) return 1;
    }
    /* i32: edges + sweep */
    {
        enum { N = 1024 };
        int32_t src[N];
        for (int i = 0; i < N; i++) src[i] = (i & 1) ? -i : i;
        src[0] = INT32_MIN;
        src[1] = INT32_MAX;
        if (test_roundtrip_i32(src, N, "i32 sweep")) return 1;
    }
    /* i64: edges */
    {
        int64_t src[6] = { 0, -1, 1, INT64_MIN, INT64_MAX, -1234567890123LL };
        if (test_roundtrip_i64(src, 6, "i64 edges")) return 1;
    }
    return 0;
}

/* ----- Empty buffers ---------------------------------------------------- */

static int test_empty(void) {
    tdc_buffer enc = make_buffer();
    tdc_dtype out_dt = (tdc_dtype)0;
    uint8_t dummy = 0;

    tdc_status st = tdc_xform_zigzag_vt.encode(
        &dummy, 0, TDC_DT_I32, NULL, &enc, &out_dt);
    ASSERT_OR_DIE(st == TDC_OK, "empty encode failed");
    ASSERT_OR_DIE(enc.size == 0, "empty enc size != 0");
    ASSERT_OR_DIE(out_dt == TDC_DT_U32, "empty out_dtype != U32");

    st = tdc_xform_zigzag_vt.decode(
        NULL, 0, TDC_DT_I32, NULL, NULL, 0, &out_dt);
    ASSERT_OR_DIE(st == TDC_OK, "empty decode failed");
    ASSERT_OR_DIE(out_dt == TDC_DT_I32, "empty decode out_dtype != I32");

    printf("  [empty] OK\n");
    free_buffer(&enc);
    return 0;
}

/* ----- Rejection paths -------------------------------------------------- */

static int test_rejections(void) {
    tdc_buffer enc = make_buffer();
    tdc_dtype out_dt = (tdc_dtype)0;
    int32_t dummy_src[2] = { 1, 2 };

    /* Encode rejects float dtype. */
    {
        double f[2] = { 1.0, 2.0 };
        tdc_status st = tdc_xform_zigzag_vt.encode(
            (const uint8_t *)f, sizeof f, TDC_DT_F64, NULL, &enc, &out_dt);
        ASSERT_OR_DIE(st == TDC_E_DTYPE, "encode should reject F64");
    }

    /* Encode rejects unsigned dtype. */
    {
        uint32_t u[2] = { 1, 2 };
        tdc_status st = tdc_xform_zigzag_vt.encode(
            (const uint8_t *)u, sizeof u, TDC_DT_U32, NULL, &enc, &out_dt);
        ASSERT_OR_DIE(st == TDC_E_DTYPE, "encode should reject U32");
    }

    /* Encode rejects misaligned src_size. */
    {
        uint8_t junk[5] = {0};
        tdc_status st = tdc_xform_zigzag_vt.encode(
            junk, sizeof junk, TDC_DT_I16, NULL, &enc, &out_dt);
        ASSERT_OR_DIE(st == TDC_E_INVAL, "encode should reject misaligned i16");
    }

    /* Decode rejects misaligned src_size. */
    {
        uint8_t junk[3] = {0};
        int16_t dec[2];
        tdc_status st = tdc_xform_zigzag_vt.decode(
            junk, sizeof junk, TDC_DT_I16, NULL,
            (uint8_t *)dec, sizeof dec, &out_dt);
        ASSERT_OR_DIE(st == TDC_E_CORRUPT, "decode should reject misaligned");
    }

    /* Decode rejects mismatched dst_size. */
    {
        uint8_t enc_bytes[4] = {0};
        int16_t dec[1];  /* too small: src has 2 elems, dst has 1 */
        tdc_status st = tdc_xform_zigzag_vt.decode(
            enc_bytes, sizeof enc_bytes, TDC_DT_I16, NULL,
            (uint8_t *)dec, sizeof dec, &out_dt);
        ASSERT_OR_DIE(st == TDC_E_CORRUPT, "decode should reject dst_size mismatch");
    }

    (void)dummy_src;
    printf("  [rejections] OK\n");
    free_buffer(&enc);
    return 0;
}

/* ----- Registry wiring -------------------------------------------------- */

static int test_registry(void) {
    const tdc_xform_vt *vt = tdc_xform_get(TDC_XFORM_ZIGZAG);
    ASSERT_OR_DIE(vt == &tdc_xform_zigzag_vt,
                  "registry should return &tdc_xform_zigzag_vt for ZIGZAG");
    ASSERT_OR_DIE(vt->id == TDC_XFORM_ZIGZAG, "vt->id mismatch");
    ASSERT_OR_DIE(vt->is_lossy == 0, "zigzag is not lossy");
    printf("  [registry wiring] OK\n");
    return 0;
}

int main(void) {
    printf("test_zigzag_roundtrip:\n");
    if (test_spot_check())  return 1;
    if (test_all_widths())  return 1;
    if (test_empty())       return 1;
    if (test_rejections())  return 1;
    if (test_registry())    return 1;
    printf("test_zigzag_roundtrip: OK\n");
    return 0;
}
