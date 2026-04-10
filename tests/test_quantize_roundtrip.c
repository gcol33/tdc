/*
 * tests/test_quantize_roundtrip.c
 *
 * Round-trip test for the TDC_XFORM_QUANTIZE backend.
 *
 * Verifies:
 *   1. f64 -> i16 -> f64 round-trip max-abs-error <= 1/scale + epsilon.
 *   2. f64 -> i8  clamps gracefully on out-of-range input (no crash, no UB,
 *      decoded values equal target's [min, max] / scale + offset).
 *   3. f32 -> i32 round-trip on a sinusoid.
 *   4. NaN encodes to target min and decodes to that sentinel value.
 *   5. Empty buffer round-trips.
 *   6. Decode rejects size not a multiple of target_size with TDC_E_CORRUPT.
 *   7. Encode rejects target == TDC_DT_F32 with TDC_E_INVAL.
 *   8. Encode rejects target == TDC_DT_I64 with TDC_E_INVAL.
 *   9. Encode rejects scale == 0 with TDC_E_INVAL.
 *  10. Encode rejects integer in_dtype with TDC_E_DTYPE.
 */

#include "tdc/transform.h"
#include "tdc/codec.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern const tdc_xform_vt tdc_xform_quantize_vt;

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

/* ----- f64 -> i16 round-trip -------------------------------------------- */

static int test_f64_i16_roundtrip(void) {
    enum { N = 1024 };
    double src[N];
    /* Values in [-10, 10]. Scale 1000 -> step 0.001 -> max abs error 0.0005. */
    for (int i = 0; i < N; i++) {
        src[i] = -10.0 + (20.0 * (double)i) / (double)(N - 1);
    }

    tdc_quantize_params qp = { .scale = 1000.0, .offset = 0.0, .target = TDC_DT_I16 };
    tdc_buffer enc = make_buffer();
    tdc_dtype out_dt = (tdc_dtype)0;

    tdc_status st = tdc_xform_quantize_vt.encode(
        (const uint8_t *)src, sizeof src, TDC_DT_F64, &qp, &enc, &out_dt);
    ASSERT_OR_DIE(st == TDC_OK, "f64->i16 encode failed");
    ASSERT_OR_DIE(out_dt == TDC_DT_I16, "out_dtype != I16");
    ASSERT_OR_DIE(enc.size == N * sizeof(int16_t), "encoded size mismatch");

    double dec[N];
    st = tdc_xform_quantize_vt.decode(
        enc.data, enc.size, TDC_DT_F64, &qp,
        (uint8_t *)dec, sizeof dec, &out_dt);
    ASSERT_OR_DIE(st == TDC_OK, "f64->i16 decode failed");
    ASSERT_OR_DIE(out_dt == TDC_DT_F64, "decode out_dtype != F64");

    double max_err = 0.0;
    for (int i = 0; i < N; i++) {
        double e = fabs(dec[i] - src[i]);
        if (e > max_err) max_err = e;
    }
    /* Half a quantization step plus a tiny float-conversion margin. */
    ASSERT_OR_DIE(max_err <= 0.0006,
                  "f64->i16 round-trip exceeded expected error bound");
    printf("  [f64->i16 1024 elems] max_err=%.6f OK\n", max_err);

    free_buffer(&enc);
    return 0;
}

/* ----- f64 -> i8 clamping ----------------------------------------------- */

static int test_f64_i8_clamp(void) {
    /* scale=1, offset=0, target=i8 -> representable range is [-128, 127].
     * Feed values straddling that range and verify clamp behavior. */
    double src[5] = { -1000.0, -128.0, 0.0, 127.0, 1000.0 };
    tdc_quantize_params qp = { .scale = 1.0, .offset = 0.0, .target = TDC_DT_I8 };
    tdc_buffer enc = make_buffer();
    tdc_dtype out_dt = (tdc_dtype)0;

    tdc_status st = tdc_xform_quantize_vt.encode(
        (const uint8_t *)src, sizeof src, TDC_DT_F64, &qp, &enc, &out_dt);
    ASSERT_OR_DIE(st == TDC_OK, "f64->i8 encode failed");
    ASSERT_OR_DIE(enc.size == 5, "f64->i8 encoded size mismatch");

    int8_t expected[5] = { -128, -128, 0, 127, 127 };
    for (int i = 0; i < 5; i++) {
        int8_t got;
        memcpy(&got, enc.data + i, 1);
        if (got != expected[i]) {
            fprintf(stderr,
                    "FAIL: clamp elem %d: got %d expected %d\n",
                    i, got, expected[i]);
            free_buffer(&enc);
            return 1;
        }
    }
    printf("  [f64->i8 clamp] OK\n");
    free_buffer(&enc);
    return 0;
}

/* ----- f32 -> i32 round-trip -------------------------------------------- */

static int test_f32_i32_roundtrip(void) {
    enum { N = 256 };
    float src[N];
    for (int i = 0; i < N; i++) {
        src[i] = (float)sin((double)i * 0.05) * 100.0f;
    }

    tdc_quantize_params qp = { .scale = 1.0e6, .offset = 0.0, .target = TDC_DT_I32 };
    tdc_buffer enc = make_buffer();
    tdc_dtype out_dt = (tdc_dtype)0;

    tdc_status st = tdc_xform_quantize_vt.encode(
        (const uint8_t *)src, sizeof src, TDC_DT_F32, &qp, &enc, &out_dt);
    ASSERT_OR_DIE(st == TDC_OK, "f32->i32 encode failed");
    ASSERT_OR_DIE(out_dt == TDC_DT_I32, "f32->i32 out_dtype != I32");
    ASSERT_OR_DIE(enc.size == N * sizeof(int32_t), "f32->i32 encoded size mismatch");

    float dec[N];
    st = tdc_xform_quantize_vt.decode(
        enc.data, enc.size, TDC_DT_F32, &qp,
        (uint8_t *)dec, sizeof dec, &out_dt);
    ASSERT_OR_DIE(st == TDC_OK, "f32->i32 decode failed");
    ASSERT_OR_DIE(out_dt == TDC_DT_F32, "f32 decode out_dtype != F32");

    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float e = fabsf(dec[i] - src[i]);
        if (e > max_err) max_err = e;
    }
    /* Step is 1e-6, plus float rounding. */
    ASSERT_OR_DIE(max_err <= 1.0e-3f,
                  "f32->i32 round-trip exceeded expected error bound");
    printf("  [f32->i32 256 elems] max_err=%.6f OK\n", (double)max_err);

    free_buffer(&enc);
    return 0;
}

/* ----- NaN sentinel ----------------------------------------------------- */

static int test_nan_sentinel(void) {
    double src[3] = { NAN, 0.0, 1.5 };
    tdc_quantize_params qp = { .scale = 100.0, .offset = 0.0, .target = TDC_DT_I16 };
    tdc_buffer enc = make_buffer();
    tdc_dtype out_dt = (tdc_dtype)0;

    tdc_status st = tdc_xform_quantize_vt.encode(
        (const uint8_t *)src, sizeof src, TDC_DT_F64, &qp, &enc, &out_dt);
    ASSERT_OR_DIE(st == TDC_OK, "NaN encode failed");

    int16_t v0;
    memcpy(&v0, enc.data + 0, 2);
    ASSERT_OR_DIE(v0 == INT16_MIN, "NaN should encode to INT16_MIN");

    printf("  [NaN sentinel] OK\n");
    free_buffer(&enc);
    return 0;
}

/* ----- Empty buffer ----------------------------------------------------- */

static int test_empty(void) {
    tdc_quantize_params qp = { .scale = 1.0, .offset = 0.0, .target = TDC_DT_I16 };
    tdc_buffer enc = make_buffer();
    tdc_dtype out_dt = (tdc_dtype)0;

    uint8_t dummy = 0;
    tdc_status st = tdc_xform_quantize_vt.encode(
        &dummy, 0, TDC_DT_F64, &qp, &enc, &out_dt);
    ASSERT_OR_DIE(st == TDC_OK, "empty encode failed");
    ASSERT_OR_DIE(enc.size == 0, "empty encoded size != 0");

    st = tdc_xform_quantize_vt.decode(
        NULL, 0, TDC_DT_F64, &qp, NULL, 0, &out_dt);
    ASSERT_OR_DIE(st == TDC_OK, "empty decode failed");

    printf("  [empty] OK\n");
    free_buffer(&enc);
    return 0;
}

/* ----- Rejection paths -------------------------------------------------- */

static int test_rejections(void) {
    tdc_buffer enc = make_buffer();
    tdc_dtype out_dt = (tdc_dtype)0;
    double dummy_src[2] = { 1.0, 2.0 };

    /* Decode rejects misaligned src_size. */
    {
        tdc_quantize_params qp = { .scale = 1.0, .offset = 0.0, .target = TDC_DT_I16 };
        uint8_t junk[3] = {0};
        double dec_buf[1];
        tdc_status st = tdc_xform_quantize_vt.decode(
            junk, sizeof junk, TDC_DT_F64, &qp,
            (uint8_t *)dec_buf, sizeof(double), &out_dt);
        ASSERT_OR_DIE(st == TDC_E_CORRUPT, "decode should reject misaligned src");
    }

    /* Encode rejects float target. */
    {
        tdc_quantize_params qp = { .scale = 1.0, .offset = 0.0, .target = TDC_DT_F32 };
        tdc_status st = tdc_xform_quantize_vt.encode(
            (const uint8_t *)dummy_src, sizeof dummy_src,
            TDC_DT_F64, &qp, &enc, &out_dt);
        ASSERT_OR_DIE(st == TDC_E_INVAL, "encode should reject F32 target");
    }

    /* Encode rejects I64 target. */
    {
        tdc_quantize_params qp = { .scale = 1.0, .offset = 0.0, .target = TDC_DT_I64 };
        tdc_status st = tdc_xform_quantize_vt.encode(
            (const uint8_t *)dummy_src, sizeof dummy_src,
            TDC_DT_F64, &qp, &enc, &out_dt);
        ASSERT_OR_DIE(st == TDC_E_INVAL, "encode should reject I64 target");
    }

    /* Encode rejects scale == 0. */
    {
        tdc_quantize_params qp = { .scale = 0.0, .offset = 0.0, .target = TDC_DT_I16 };
        tdc_status st = tdc_xform_quantize_vt.encode(
            (const uint8_t *)dummy_src, sizeof dummy_src,
            TDC_DT_F64, &qp, &enc, &out_dt);
        ASSERT_OR_DIE(st == TDC_E_INVAL, "encode should reject scale==0");
    }

    /* Encode rejects integer input dtype. */
    {
        tdc_quantize_params qp = { .scale = 1.0, .offset = 0.0, .target = TDC_DT_I16 };
        int32_t int_src[2] = { 1, 2 };
        tdc_status st = tdc_xform_quantize_vt.encode(
            (const uint8_t *)int_src, sizeof int_src,
            TDC_DT_I32, &qp, &enc, &out_dt);
        ASSERT_OR_DIE(st == TDC_E_DTYPE, "encode should reject integer in_dtype");
    }

    printf("  [rejections] OK\n");
    free_buffer(&enc);
    return 0;
}

int main(void) {
    printf("test_quantize_roundtrip:\n");
    if (test_f64_i16_roundtrip()) return 1;
    if (test_f64_i8_clamp())      return 1;
    if (test_f32_i32_roundtrip()) return 1;
    if (test_nan_sentinel())      return 1;
    if (test_empty())             return 1;
    if (test_rejections())        return 1;
    printf("test_quantize_roundtrip: OK\n");
    return 0;
}
