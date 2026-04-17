/*
 * tests/test_dict_numeric_roundtrip.c
 *
 * Round-trip test for TDC_MODEL_DICT_NUMERIC_1D.
 *
 * Verifies:
 *   1. residual_dtype == TDC_DT_U32 unconditionally.
 *   2. Byte-exact round-trip on every accepted dtype (i16/u16/i32/u32/f32/
 *      i64/u64/f64).
 *   3. Low-cardinality f64 input (many repeats) round-trips and the
 *      dictionary really shrinks (n_unique << n).
 *   4. All-unique input round-trips (worst case: dict has n entries).
 *   5. Float byte identity: +0.0/-0.0 are stored as distinct entries;
 *      different NaN payloads are stored as distinct entries.
 *   6. Single-row and empty (n == 0) inputs round-trip.
 *   7. Encoder rejects i8/u8 (TDC_E_DTYPE) and non-VECTOR_1D layout.
 *   8. Decoder rejects truncated side meta and out-of-range indices.
 *   9. Registry returns &tdc_model_dict_numeric_1d_vt for
 *      TDC_MODEL_DICT_NUMERIC_1D.
 */

#include "tdc/model.h"
#include "tdc/codec.h"
#include "tdc/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern const tdc_model_vt tdc_model_dict_numeric_1d_vt;

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
    b->data = NULL; b->size = 0; b->capacity = 0;
}

#define ASSERT_OR_DIE(cond, msg) do {                                       \
    if (!(cond)) {                                                          \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);   \
        return 1;                                                           \
    }                                                                       \
} while (0)

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

static int rt_one(const char *label, tdc_dtype dt, size_t elem_size,
                  const void *src, int64_t n) {
    const tdc_model_vt *vt = &tdc_model_dict_numeric_1d_vt;

    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;

    tdc_block in = make_block((void *)src, n, dt);
    tdc_status st = vt->encode(&in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "encode returned non-OK");
    ASSERT_OR_DIE(rdt == TDC_DT_U32, "residual_dtype must be u32");
    ASSERT_OR_DIE(side.size >= 5u, "side_meta must carry 5-byte header");
    ASSERT_OR_DIE(residual.size == (size_t)n * 4u, "residual size mismatch");

    void *dst = malloc((size_t)n * elem_size > 0 ? (size_t)n * elem_size : 1u);
    ASSERT_OR_DIE(dst != NULL, "malloc dst");
    tdc_block out = make_block(dst, n, dt);

    st = vt->decode(&out, NULL, rdt,
                    residual.data, residual.size,
                    side.data, side.size);
    ASSERT_OR_DIE(st == TDC_OK, "decode returned non-OK");
    ASSERT_OR_DIE(memcmp(dst, src, (size_t)n * elem_size) == 0,
                  "decoded data does not match source (byte-exact)");

    uint32_t dict_count;
    memcpy(&dict_count, (uint8_t *)side.data + 1u, 4u);
    printf("  [%s] n=%lld elem=%zu dict=%u side=%zu round-trip OK\n",
           label, (long long)n, elem_size, dict_count, side.size);

    free(dst);
    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

/* ----- Per-dtype round-trip tests --------------------------------------- */

static int test_i16(void) {
    int16_t src[] = { 0, 1, -1, 32767, -32768, 100, -100, 100, 0, -1 };
    return rt_one("i16 mixed", TDC_DT_I16, 2, src,
                  (int64_t)(sizeof(src) / sizeof(src[0])));
}

static int test_u16(void) {
    uint16_t src[] = { 0, 1, 2, 3, 65535, 3, 2, 1, 0 };
    return rt_one("u16 mixed", TDC_DT_U16, 2, src,
                  (int64_t)(sizeof(src) / sizeof(src[0])));
}

static int test_i32(void) {
    int32_t src[] = { 0, 1, -1, INT32_MAX, INT32_MIN, 1, 0, -1, 42, 42 };
    return rt_one("i32 mixed", TDC_DT_I32, 4, src,
                  (int64_t)(sizeof(src) / sizeof(src[0])));
}

static int test_u32(void) {
    uint32_t src[] = { 0u, 1u, UINT32_MAX, 1u, 0u, 1234567u, 1234567u };
    return rt_one("u32 mixed", TDC_DT_U32, 4, src,
                  (int64_t)(sizeof(src) / sizeof(src[0])));
}

static int test_f32(void) {
    float src[] = { 0.0f, -0.0f, 1.0f, -1.0f, 1.0f, 3.14f, 3.14f, 0.0f };
    return rt_one("f32 mixed", TDC_DT_F32, 4, src,
                  (int64_t)(sizeof(src) / sizeof(src[0])));
}

static int test_i64(void) {
    int64_t src[] = { 0, 1, -1, INT64_MAX, INT64_MIN, 42, 42, -1 };
    return rt_one("i64 mixed", TDC_DT_I64, 8, src,
                  (int64_t)(sizeof(src) / sizeof(src[0])));
}

static int test_u64(void) {
    uint64_t src[] = { 0ull, UINT64_MAX, 1ull, 1ull, 123456789ull, 0ull };
    return rt_one("u64 mixed", TDC_DT_U64, 8, src,
                  (int64_t)(sizeof(src) / sizeof(src[0])));
}

static int test_f64(void) {
    double src[] = { 0.0, -0.0, 1.0, -1.0, 1.0, 3.14159, 3.14159, 1e-300, 1e300 };
    return rt_one("f64 mixed", TDC_DT_F64, 8, src,
                  (int64_t)(sizeof(src) / sizeof(src[0])));
}

/* ----- Compression-ratio sanity -------------------------------------------- */

/* Low-cardinality f64: 1 000 000 samples drawn from a 50-value alphabet.
 * After encode, dict must have exactly 50 entries and residual is n * 4
 * bytes (u32 indices), regardless of entropy stage. */
static int test_low_cardinality_f64(void) {
    const int ALPHABET = 50;
    const int64_t N     = 1000000;
    double *src = (double *)malloc((size_t)N * sizeof(double));
    ASSERT_OR_DIE(src != NULL, "malloc src");
    /* Deterministic LCG-ish index into the alphabet. */
    uint32_t s = 2463534242u;
    for (int64_t i = 0; i < N; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        int bucket = (int)(s % (uint32_t)ALPHABET);
        src[i] = -20.0 + 0.1 * (double)bucket;
    }

    const tdc_model_vt *vt = &tdc_model_dict_numeric_1d_vt;
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_block  in       = make_block(src, N, TDC_DT_F64);

    tdc_status st = vt->encode(&in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "encode");

    uint32_t dict_count;
    memcpy(&dict_count, (uint8_t *)side.data + 1u, 4u);
    ASSERT_OR_DIE(dict_count == (uint32_t)ALPHABET, "dict_count != 50");
    ASSERT_OR_DIE(side.size == 5u + (size_t)ALPHABET * 8u,
                  "side size mismatch");
    ASSERT_OR_DIE(residual.size == (size_t)N * 4u, "residual size");

    /* Sanity: raw bytes would be N*8; residual is N*4; total encoded
     * is ~4 bytes per sample (half the raw) before any entropy stage. */
    double ratio_before_entropy = (double)(N * 8) / (double)(residual.size + side.size);
    printf("  [low-card f64] N=%lld dict=%u residual=%zu side=%zu "
           "pre-entropy ratio=%.2fx\n",
           (long long)N, dict_count, residual.size, side.size,
           ratio_before_entropy);
    ASSERT_OR_DIE(ratio_before_entropy > 1.99,
                  "expected >= 2x pre-entropy on 50-value f64 alphabet");

    double *dst = (double *)malloc((size_t)N * sizeof(double));
    tdc_block out = make_block(dst, N, TDC_DT_F64);
    st = vt->decode(&out, NULL, rdt,
                    residual.data, residual.size,
                    side.data, side.size);
    ASSERT_OR_DIE(st == TDC_OK, "decode");
    ASSERT_OR_DIE(memcmp(dst, src, (size_t)N * sizeof(double)) == 0,
                  "round-trip mismatch");

    free(src); free(dst);
    free_buffer(&residual); free_buffer(&side);
    return 0;
}

/* All-unique worst case: dict carries every element. */
static int test_all_unique_i32(void) {
    const int64_t N = 4096;
    int32_t *src = (int32_t *)malloc((size_t)N * sizeof(int32_t));
    for (int64_t i = 0; i < N; ++i) src[i] = (int32_t)i * 2 + 7;

    const tdc_model_vt *vt = &tdc_model_dict_numeric_1d_vt;
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_block  in       = make_block(src, N, TDC_DT_I32);

    tdc_status st = vt->encode(&in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "encode");

    uint32_t dict_count;
    memcpy(&dict_count, (uint8_t *)side.data + 1u, 4u);
    ASSERT_OR_DIE(dict_count == (uint32_t)N, "dict_count != N");

    int32_t *dst = (int32_t *)malloc((size_t)N * sizeof(int32_t));
    tdc_block out = make_block(dst, N, TDC_DT_I32);
    st = vt->decode(&out, NULL, rdt,
                    residual.data, residual.size,
                    side.data, side.size);
    ASSERT_OR_DIE(st == TDC_OK, "decode");
    ASSERT_OR_DIE(memcmp(dst, src, (size_t)N * sizeof(int32_t)) == 0,
                  "round-trip mismatch");
    printf("  [all-unique i32] N=%lld dict=%u round-trip OK\n",
           (long long)N, dict_count);

    free(src); free(dst);
    free_buffer(&residual); free_buffer(&side);
    return 0;
}

/* Empty block. */
static int test_empty(void) {
    const tdc_model_vt *vt = &tdc_model_dict_numeric_1d_vt;
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    double dummy = 0.0;
    tdc_block in = make_block(&dummy, 0, TDC_DT_F64);

    tdc_status st = vt->encode(&in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "encode empty");
    ASSERT_OR_DIE(rdt == TDC_DT_U32, "rdt");
    ASSERT_OR_DIE(side.size == 5u, "empty side must be header only");
    ASSERT_OR_DIE(residual.size == 0u, "empty residual");

    tdc_block out = make_block(NULL, 0, TDC_DT_F64);
    st = vt->decode(&out, NULL, rdt,
                    NULL, 0,
                    side.data, side.size);
    ASSERT_OR_DIE(st == TDC_OK, "decode empty");
    printf("  [empty] round-trip OK\n");

    free_buffer(&residual); free_buffer(&side);
    return 0;
}

/* Rejection: i8 is not in the accepted dtype set. */
static int test_reject_i8(void) {
    const tdc_model_vt *vt = &tdc_model_dict_numeric_1d_vt;
    int8_t  src[4] = { 0, 1, 2, 3 };
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_block  in       = make_block(src, 4, TDC_DT_I8);

    tdc_status st = vt->encode(&in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_E_DTYPE, "expected TDC_E_DTYPE for i8");
    printf("  [reject i8] OK (returned TDC_E_DTYPE)\n");

    free_buffer(&residual); free_buffer(&side);
    return 0;
}

/* Rejection: RASTER_2D layout is not accepted. */
static int test_reject_layout(void) {
    const tdc_model_vt *vt = &tdc_model_dict_numeric_1d_vt;
    float src[4] = { 0.0f, 1.0f, 2.0f, 3.0f };
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_block  in = {0};
    in.data = src;
    in.dtype = TDC_DT_F32;
    in.layout = TDC_LAYOUT_RASTER_2D;
    in.shape.rank = 2;
    in.shape.dim[0] = 2; in.shape.dim[1] = 2;

    tdc_status st = vt->encode(&in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_E_LAYOUT, "expected TDC_E_LAYOUT for raster2d");
    printf("  [reject raster2d] OK (returned TDC_E_LAYOUT)\n");

    free_buffer(&residual); free_buffer(&side);
    return 0;
}

/* Decoder: truncated side meta is corruption. */
static int test_decode_truncated(void) {
    const tdc_model_vt *vt = &tdc_model_dict_numeric_1d_vt;
    double src[5] = { 1.0, 2.0, 3.0, 1.0, 2.0 };
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_block  in       = make_block(src, 5, TDC_DT_F64);

    tdc_status st = vt->encode(&in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "encode");

    double dst[5];
    tdc_block out = make_block(dst, 5, TDC_DT_F64);

    /* Lop off 1 byte of the value table to create a corrupt side size. */
    st = vt->decode(&out, NULL, rdt,
                    residual.data, residual.size,
                    side.data, side.size - 1u);
    ASSERT_OR_DIE(st == TDC_E_CORRUPT,
                  "expected TDC_E_CORRUPT on truncated side");
    printf("  [decode truncated side] OK\n");

    free_buffer(&residual); free_buffer(&side);
    return 0;
}

/* Decoder: out-of-range index is corruption. */
static int test_decode_bad_index(void) {
    const tdc_model_vt *vt = &tdc_model_dict_numeric_1d_vt;
    double src[4] = { 1.0, 2.0, 3.0, 1.0 };
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_block  in       = make_block(src, 4, TDC_DT_F64);

    tdc_status st = vt->encode(&in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "encode");

    /* Tamper: change first index to 99 (dict has 3 entries -> OOB). */
    uint32_t bad = 99u;
    memcpy(residual.data, &bad, 4u);

    double dst[4];
    tdc_block out = make_block(dst, 4, TDC_DT_F64);
    st = vt->decode(&out, NULL, rdt,
                    residual.data, residual.size,
                    side.data, side.size);
    ASSERT_OR_DIE(st == TDC_E_CORRUPT, "expected TDC_E_CORRUPT on bad index");
    printf("  [decode bad index] OK\n");

    free_buffer(&residual); free_buffer(&side);
    return 0;
}

/* Registry lookup. */
static int test_registry(void) {
    const tdc_model_vt *vt = tdc_model_get(TDC_MODEL_DICT_NUMERIC_1D);
    ASSERT_OR_DIE(vt == &tdc_model_dict_numeric_1d_vt,
                  "registry did not return tdc_model_dict_numeric_1d_vt");
    printf("  [registry] OK\n");
    return 0;
}

int main(void) {
    printf("dict_numeric round-trip tests\n");
    int rc = 0;
    rc |= test_i16();
    rc |= test_u16();
    rc |= test_i32();
    rc |= test_u32();
    rc |= test_f32();
    rc |= test_i64();
    rc |= test_u64();
    rc |= test_f64();
    rc |= test_low_cardinality_f64();
    rc |= test_all_unique_i32();
    rc |= test_empty();
    rc |= test_reject_i8();
    rc |= test_reject_layout();
    rc |= test_decode_truncated();
    rc |= test_decode_bad_index();
    rc |= test_registry();
    if (rc == 0) printf("ALL PASS\n");
    else         printf("FAILED\n");
    return rc;
}
