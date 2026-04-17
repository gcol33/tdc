/*
 * tests/test_sparse_zero_roundtrip.c
 *
 * Round-trip test for TDC_MODEL_SPARSE_ZERO_1D.
 *
 * Verifies:
 *   1. residual_dtype == TDC_DT_U32 unconditionally.
 *   2. Byte-exact round-trip on every accepted dtype (i16/u16/i32/u32/f32/
 *      i64/u64/f64).
 *   3. High-sparsity inputs (95%+ zeros) round-trip and the encoded size
 *      is materially smaller than RAW.
 *   4. All-zero input round-trips (n_nonzero == 0, empty residual).
 *   5. Dense (no zeros) input round-trips — worst case, but still correct.
 *   6. Float byte identity: +0.0 counts as zero, -0.0 is a non-zero value;
 *      -0.0 survives the round-trip bit-exactly.
 *   7. Single-row and empty (n == 0) inputs round-trip.
 *   8. Encoder rejects i8/u8 and STRING (TDC_E_DTYPE) and non-VECTOR_1D
 *      layout (TDC_E_LAYOUT).
 *   9. Decoder rejects: truncated side meta, wrong stored dtype,
 *      out-of-range positions, duplicate/descending positions, residual
 *      size mismatch.
 *  10. Registry returns &tdc_model_sparse_zero_1d_vt for
 *      TDC_MODEL_SPARSE_ZERO_1D.
 */

#include "tdc/model.h"
#include "tdc/codec.h"
#include "tdc/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern const tdc_model_vt tdc_model_sparse_zero_1d_vt;

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
    const tdc_model_vt *vt = &tdc_model_sparse_zero_1d_vt;

    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;

    tdc_block in = make_block((void *)src, n, dt);
    tdc_status st = vt->encode(&in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "encode returned non-OK");
    ASSERT_OR_DIE(rdt == TDC_DT_U32, "residual_dtype must be u32");
    ASSERT_OR_DIE(side.size >= 5u, "side_meta must carry 5-byte header");

    uint32_t n_nz;
    memcpy(&n_nz, (uint8_t *)side.data + 1u, 4u);
    ASSERT_OR_DIE(residual.size == (size_t)n_nz * 4u,
                  "residual size must equal n_nonzero * 4");
    ASSERT_OR_DIE(side.size == 5u + (size_t)n_nz * elem_size,
                  "side_meta size must equal 5 + n_nonzero*elem");

    void *dst = malloc((size_t)n * elem_size > 0 ? (size_t)n * elem_size : 1u);
    ASSERT_OR_DIE(dst != NULL, "malloc dst");
    tdc_block out = make_block(dst, n, dt);

    st = vt->decode(&out, NULL, rdt,
                    residual.data, residual.size,
                    side.data, side.size);
    ASSERT_OR_DIE(st == TDC_OK, "decode returned non-OK");
    ASSERT_OR_DIE(memcmp(dst, src, (size_t)n * elem_size) == 0,
                  "decoded data does not match source (byte-exact)");

    printf("  [%s] n=%lld elem=%zu nz=%u side=%zu resid=%zu round-trip OK\n",
           label, (long long)n, elem_size, n_nz, side.size, residual.size);

    free(dst);
    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

/* ----- Per-dtype round-trip tests --------------------------------------- */

static int test_i16(void) {
    int16_t src[] = {0, 0, 0, 7, 0, 0, -1, 0, 0, 42};
    return rt_one("i16 sparse", TDC_DT_I16, 2, src,
                  (int64_t)(sizeof(src) / sizeof(src[0])));
}

static int test_u16(void) {
    uint16_t src[] = {0, 0, 0, 65535, 0, 0, 0, 0, 1, 0};
    return rt_one("u16 sparse", TDC_DT_U16, 2, src,
                  (int64_t)(sizeof(src) / sizeof(src[0])));
}

static int test_i32(void) {
    int32_t src[] = {0, 0, 0, INT32_MAX, 0, 0, INT32_MIN, 0, 0, 0};
    return rt_one("i32 sparse", TDC_DT_I32, 4, src,
                  (int64_t)(sizeof(src) / sizeof(src[0])));
}

static int test_u32(void) {
    uint32_t src[] = {0, 1, 0, 2, 0, 3, 0, 4, 0, 0};
    return rt_one("u32 mixed", TDC_DT_U32, 4, src,
                  (int64_t)(sizeof(src) / sizeof(src[0])));
}

static int test_f32(void) {
    float src[] = {0.0f, 0.0f, 0.0f, 1.5f, 0.0f, -2.25f, 0.0f, 0.0f, 0.0f, 3.5f};
    return rt_one("f32 sparse", TDC_DT_F32, 4, src,
                  (int64_t)(sizeof(src) / sizeof(src[0])));
}

static int test_i64(void) {
    int64_t src[] = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0};
    return rt_one("i64 sparse", TDC_DT_I64, 8, src,
                  (int64_t)(sizeof(src) / sizeof(src[0])));
}

static int test_u64(void) {
    uint64_t src[] = {0, 0, (uint64_t)-1, 0, 0, 0, 0, 42, 0, 0};
    return rt_one("u64 sparse", TDC_DT_U64, 8, src,
                  (int64_t)(sizeof(src) / sizeof(src[0])));
}

static int test_f64(void) {
    double src[] = {0.0, 0.0, 0.0, 3.14159, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0};
    return rt_one("f64 sparse", TDC_DT_F64, 8, src,
                  (int64_t)(sizeof(src) / sizeof(src[0])));
}

/* ----- Density edge cases ----------------------------------------------- */

static int test_all_zero(void) {
    int64_t src[128];
    memset(src, 0, sizeof(src));
    return rt_one("i64 all-zero", TDC_DT_I64, 8, src, 128);
}

static int test_dense(void) {
    /* No zeros — worst case. Encode still correct but larger than RAW. */
    int64_t src[16];
    for (int i = 0; i < 16; ++i) src[i] = i + 1;  /* 1..16, none zero */
    return rt_one("i64 dense", TDC_DT_I64, 8, src, 16);
}

static int test_high_sparsity_ratio(void) {
    /* 1000 elements, 10 non-zero. Verify the encoded size is materially
     * smaller than RAW (n*8 = 8000 bytes). */
    const int N = 1000;
    int64_t *src = calloc((size_t)N, sizeof(int64_t));
    ASSERT_OR_DIE(src != NULL, "calloc");
    for (int i = 0; i < 10; ++i) src[i * 100] = (int64_t)(i + 1);

    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_block in = make_block(src, N, TDC_DT_I64);

    tdc_status st = tdc_model_sparse_zero_1d_vt.encode(
        &in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "encode");

    /* Side = 5 + 10*8 = 85. Residual = 10*4 = 40. Total 125 vs RAW 8000. */
    size_t total = side.size + residual.size;
    ASSERT_OR_DIE(total < (size_t)N * 8u / 4u,
                  "sparse encoding should be <25% of RAW on this input");
    printf("  [sparsity 99%%] raw=%d encoded=%zu (%.1f%%)\n",
           N * 8, total, 100.0 * (double)total / (double)(N * 8));

    int64_t *dst = calloc((size_t)N, sizeof(int64_t));
    ASSERT_OR_DIE(dst != NULL, "calloc dst");
    /* Pre-fill with sentinel to verify memset-to-zero inside decode. */
    for (int i = 0; i < N; ++i) dst[i] = 0xBADF00Dll;
    tdc_block out = make_block(dst, N, TDC_DT_I64);
    st = tdc_model_sparse_zero_1d_vt.decode(&out, NULL, rdt,
        residual.data, residual.size, side.data, side.size);
    ASSERT_OR_DIE(st == TDC_OK, "decode");
    ASSERT_OR_DIE(memcmp(dst, src, (size_t)N * 8u) == 0,
                  "high-sparsity round-trip mismatch");

    free(src);
    free(dst);
    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

/* ----- Float byte identity --------------------------------------------- */

static int test_float_zero_identity(void) {
    /* +0.0 -> zero bytes; -0.0 -> 0x80000000; must survive as a non-zero
     * value and round-trip bit-exactly. */
    uint32_t neg_zero_bits = 0x80000000u;
    float neg_zero;
    memcpy(&neg_zero, &neg_zero_bits, 4u);

    float src[8] = {0.0f, 0.0f, neg_zero, 0.0f, 1.0f, 0.0f, neg_zero, 0.0f};

    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_block in = make_block(src, 8, TDC_DT_F32);
    tdc_status st = tdc_model_sparse_zero_1d_vt.encode(
        &in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "encode");

    uint32_t n_nz;
    memcpy(&n_nz, (uint8_t *)side.data + 1u, 4u);
    ASSERT_OR_DIE(n_nz == 3u, "expect -0.0 and 1.0 to count as non-zero");

    float dst[8];
    tdc_block out = make_block(dst, 8, TDC_DT_F32);
    st = tdc_model_sparse_zero_1d_vt.decode(&out, NULL, rdt,
        residual.data, residual.size, side.data, side.size);
    ASSERT_OR_DIE(st == TDC_OK, "decode");
    ASSERT_OR_DIE(memcmp(dst, src, sizeof(src)) == 0,
                  "byte-identity float round-trip failed (-0.0 preserved?)");
    printf("  [float zero identity] +0.0 == zero, -0.0 != zero, OK\n");

    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

/* ----- Empty + single-row ---------------------------------------------- */

static int test_empty(void) {
    int64_t src[1] = {0};  /* unused, n == 0 */
    return rt_one("empty", TDC_DT_I64, 8, src, 0);
}

static int test_single_zero(void) {
    int64_t src[1] = {0};
    return rt_one("single zero", TDC_DT_I64, 8, src, 1);
}

static int test_single_nonzero(void) {
    int64_t src[1] = {42};
    return rt_one("single nonzero", TDC_DT_I64, 8, src, 1);
}

/* ----- Rejection tests -------------------------------------------------- */

static int test_reject_i8(void) {
    int8_t src[4] = {0, 1, 0, -1};
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_block in = make_block(src, 4, TDC_DT_I8);
    tdc_status st = tdc_model_sparse_zero_1d_vt.encode(
        &in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_E_DTYPE, "i8 must be rejected");
    free_buffer(&residual);
    free_buffer(&side);
    printf("  [reject i8] OK\n");
    return 0;
}

static int test_reject_string(void) {
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_block in = make_block(NULL, 1, TDC_DT_STRING);
    tdc_status st = tdc_model_sparse_zero_1d_vt.encode(
        &in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_E_DTYPE, "STRING must be rejected");
    free_buffer(&residual);
    free_buffer(&side);
    printf("  [reject string] OK\n");
    return 0;
}

static int test_reject_layout(void) {
    int64_t src[4] = {0, 1, 0, 2};
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_block in = make_block(src, 4, TDC_DT_I64);
    in.layout = TDC_LAYOUT_RASTER_2D;  /* not accepted */
    tdc_status st = tdc_model_sparse_zero_1d_vt.encode(
        &in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_E_LAYOUT, "non-VECTOR_1D must be rejected");
    free_buffer(&residual);
    free_buffer(&side);
    printf("  [reject layout] OK\n");
    return 0;
}

/* ----- Decoder corruption detection ------------------------------------ */

static int test_decode_truncated_side(void) {
    uint8_t side[3] = {0, 0, 0};  /* 2 bytes short of header */
    int64_t dst[4] = {0};
    tdc_block out = make_block(dst, 4, TDC_DT_I64);
    tdc_status st = tdc_model_sparse_zero_1d_vt.decode(
        &out, NULL, TDC_DT_U32, NULL, 0, side, sizeof(side));
    ASSERT_OR_DIE(st == TDC_E_CORRUPT, "truncated side must be rejected");
    printf("  [decode truncated side] OK\n");
    return 0;
}

static int test_decode_wrong_dtype(void) {
    /* Encode as i64, attempt decode as f64. */
    int64_t src[4] = {0, 1, 0, 2};
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_block in = make_block(src, 4, TDC_DT_I64);
    tdc_status st = tdc_model_sparse_zero_1d_vt.encode(
        &in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "encode");
    double dst[4] = {0.0};
    tdc_block out = make_block(dst, 4, TDC_DT_F64);
    st = tdc_model_sparse_zero_1d_vt.decode(&out, NULL, rdt,
        residual.data, residual.size, side.data, side.size);
    ASSERT_OR_DIE(st == TDC_E_DTYPE, "mismatched dtype must be rejected");
    free_buffer(&residual);
    free_buffer(&side);
    printf("  [decode wrong dtype] OK\n");
    return 0;
}

static int test_decode_oor_position(void) {
    /* Hand-craft side + residual with one position == n (out of range). */
    const int n = 4;
    uint8_t side[5 + 8];  /* header + one i64 value */
    uint32_t n_nz = 1u;
    uint8_t dt8 = (uint8_t)TDC_DT_I64;
    memcpy(side + 0, &dt8, 1);
    memcpy(side + 1, &n_nz, 4);
    int64_t val = 42;
    memcpy(side + 5, &val, 8);
    uint32_t bad_pos = (uint32_t)n;  /* == n, invalid */
    uint8_t residuals[4];
    memcpy(residuals, &bad_pos, 4);

    int64_t dst[4] = {0};
    tdc_block out = make_block(dst, n, TDC_DT_I64);
    tdc_status st = tdc_model_sparse_zero_1d_vt.decode(
        &out, NULL, TDC_DT_U32, residuals, 4, side, sizeof(side));
    ASSERT_OR_DIE(st == TDC_E_CORRUPT, "OOR position must be rejected");
    printf("  [decode OOR position] OK\n");
    return 0;
}

static int test_decode_duplicate_position(void) {
    /* Two non-zero entries at the same position — ascending check catches it. */
    const int n = 4;
    uint8_t side[5 + 16];
    uint32_t n_nz = 2u;
    uint8_t dt8 = (uint8_t)TDC_DT_I64;
    memcpy(side + 0, &dt8, 1);
    memcpy(side + 1, &n_nz, 4);
    int64_t v1 = 1, v2 = 2;
    memcpy(side + 5,       &v1, 8);
    memcpy(side + 5 + 8,   &v2, 8);
    uint32_t p1 = 2u, p2 = 2u;
    uint8_t residuals[8];
    memcpy(residuals + 0, &p1, 4);
    memcpy(residuals + 4, &p2, 4);

    int64_t dst[4] = {0};
    tdc_block out = make_block(dst, n, TDC_DT_I64);
    tdc_status st = tdc_model_sparse_zero_1d_vt.decode(
        &out, NULL, TDC_DT_U32, residuals, 8, side, sizeof(side));
    ASSERT_OR_DIE(st == TDC_E_CORRUPT, "duplicate position must be rejected");
    printf("  [decode duplicate position] OK\n");
    return 0;
}

static int test_decode_descending_position(void) {
    const int n = 4;
    uint8_t side[5 + 16];
    uint32_t n_nz = 2u;
    uint8_t dt8 = (uint8_t)TDC_DT_I64;
    memcpy(side + 0, &dt8, 1);
    memcpy(side + 1, &n_nz, 4);
    int64_t v1 = 1, v2 = 2;
    memcpy(side + 5,       &v1, 8);
    memcpy(side + 5 + 8,   &v2, 8);
    uint32_t p1 = 3u, p2 = 1u;
    uint8_t residuals[8];
    memcpy(residuals + 0, &p1, 4);
    memcpy(residuals + 4, &p2, 4);

    int64_t dst[4] = {0};
    tdc_block out = make_block(dst, n, TDC_DT_I64);
    tdc_status st = tdc_model_sparse_zero_1d_vt.decode(
        &out, NULL, TDC_DT_U32, residuals, 8, side, sizeof(side));
    ASSERT_OR_DIE(st == TDC_E_CORRUPT, "descending position must be rejected");
    printf("  [decode descending position] OK\n");
    return 0;
}

static int test_decode_residual_size_mismatch(void) {
    int64_t src[4] = {0, 1, 0, 2};
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_block in = make_block(src, 4, TDC_DT_I64);
    tdc_status st = tdc_model_sparse_zero_1d_vt.encode(
        &in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "encode");

    int64_t dst[4] = {0};
    tdc_block out = make_block(dst, 4, TDC_DT_I64);
    /* Feed decoder the wrong residual_size (off by 4). */
    st = tdc_model_sparse_zero_1d_vt.decode(&out, NULL, rdt,
        residual.data, residual.size - 4u, side.data, side.size);
    ASSERT_OR_DIE(st == TDC_E_CORRUPT, "truncated residual must be rejected");
    free_buffer(&residual);
    free_buffer(&side);
    printf("  [decode residual mismatch] OK\n");
    return 0;
}

/* ----- Registry --------------------------------------------------------- */

static int test_registry(void) {
    const tdc_model_vt *vt = tdc_model_get(TDC_MODEL_SPARSE_ZERO_1D);
    ASSERT_OR_DIE(vt == &tdc_model_sparse_zero_1d_vt,
                  "registry must resolve to sparse_zero_1d vtable");
    ASSERT_OR_DIE(vt->id == TDC_MODEL_SPARSE_ZERO_1D, "vt id mismatch");
    printf("  [registry] OK\n");
    return 0;
}

/* ----- main ------------------------------------------------------------- */

int main(void) {
    int rc = 0;
    rc |= test_i16();
    rc |= test_u16();
    rc |= test_i32();
    rc |= test_u32();
    rc |= test_f32();
    rc |= test_i64();
    rc |= test_u64();
    rc |= test_f64();
    rc |= test_all_zero();
    rc |= test_dense();
    rc |= test_high_sparsity_ratio();
    rc |= test_float_zero_identity();
    rc |= test_empty();
    rc |= test_single_zero();
    rc |= test_single_nonzero();
    rc |= test_reject_i8();
    rc |= test_reject_string();
    rc |= test_reject_layout();
    rc |= test_decode_truncated_side();
    rc |= test_decode_wrong_dtype();
    rc |= test_decode_oor_position();
    rc |= test_decode_duplicate_position();
    rc |= test_decode_descending_position();
    rc |= test_decode_residual_size_mismatch();
    rc |= test_registry();
    if (rc == 0) {
        printf("test_sparse_zero_roundtrip: ALL OK\n");
    } else {
        fprintf(stderr, "test_sparse_zero_roundtrip: FAILURES (rc=%d)\n", rc);
    }
    return rc;
}
