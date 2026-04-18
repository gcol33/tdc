/*
 * tests/test_complement_roundtrip.c
 *
 * Round-trip test for the TDC_XFORM_COMPLEMENT reference backend.
 *
 * The complement transform is the worked example from the Extending
 * vignette: a bitwise-NOT of every input byte. Self-inverting, dtype-
 * agnostic, zero params. The test covers exactly the checklist the
 * vignette calls out:
 *
 *   1. Registry returns &tdc_xform_complement_vt for TDC_XFORM_COMPLEMENT.
 *   2. Vtable self-check: id, accepted_dtypes, can_inplace, is_lossy.
 *   3. Round-trip on u8 / i32 / f64 buffers.
 *   4. Empty buffer round-trips without allocating.
 *   5. Single-element buffer round-trips.
 *   6. Degenerate all-equal block round-trips (predictable output pattern).
 *   7. Decode rejects dst_size != src_size with TDC_E_CORRUPT.
 *   8. Encode rejects TDC_DT_STRING with TDC_E_DTYPE.
 *   9. Full pipeline round-trip via tdc_encode_block / tdc_decode_block.
 */

#include "tdc/transform.h"
#include "tdc/codec.h"
#include "tdc/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern const tdc_xform_vt tdc_xform_complement_vt;

static void *t_realloc(void *user, void *ptr, size_t n) {
    (void)user;
    if (n == 0) { free(ptr); return NULL; }
    return realloc(ptr, n);
}

static tdc_buffer t_buffer(void) {
    tdc_buffer b = {0};
    b.realloc_fn = t_realloc;
    return b;
}

static void t_buffer_free(tdc_buffer *b) {
    if (b->data) free(b->data);
    b->data = NULL;
    b->size = 0;
    b->capacity = 0;
}

#define CHECK(cond, msg) do {                                             \
    if (!(cond)) {                                                        \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        return 1;                                                         \
    }                                                                     \
} while (0)

/* ----- Registry + vtable self-check -------------------------------------- */

static int test_registry(void) {
    const tdc_xform_vt *vt = tdc_xform_get(TDC_XFORM_COMPLEMENT);
    CHECK(vt == &tdc_xform_complement_vt,
          "registry should return &tdc_xform_complement_vt");
    CHECK(vt->id == TDC_XFORM_COMPLEMENT, "vt->id mismatch");
    CHECK(vt->is_lossy == 0,  "complement is not lossy");
    CHECK(vt->can_inplace == 1, "complement is element-local");
    CHECK(vt->name != NULL, "vt->name must be set");
    CHECK((vt->accepted_dtypes & (1u << (uint32_t)TDC_DT_U8))  != 0u, "accept U8");
    CHECK((vt->accepted_dtypes & (1u << (uint32_t)TDC_DT_F64)) != 0u, "accept F64");
    CHECK((vt->accepted_dtypes & (1u << (uint32_t)TDC_DT_STRING)) == 0u, "reject STRING");
    printf("  [registry + vtable] OK\n");
    return 0;
}

/* ----- Round trips over several dtypes ----------------------------------- */

static int roundtrip_bytes(const uint8_t *src, size_t n,
                           tdc_dtype dt, const char *label) {
    tdc_buffer enc = t_buffer();
    tdc_dtype  out_dt = (tdc_dtype)0;
    tdc_status st = tdc_xform_complement_vt.encode(
        src, n, dt, NULL, &enc, &out_dt);
    CHECK(st == TDC_OK, "encode failed");
    CHECK(out_dt == dt, "encode out_dtype must match in_dtype");
    CHECK(enc.size == n, "encode size must equal input size");

    /* Encode output is ~src, byte-by-byte. */
    for (size_t i = 0; i < n; ++i) {
        if (enc.data[i] != (uint8_t)~src[i]) {
            fprintf(stderr, "FAIL %s: encoded byte %zu is 0x%02X, expected 0x%02X\n",
                    label, i, enc.data[i], (uint8_t)~src[i]);
            t_buffer_free(&enc);
            return 1;
        }
    }

    uint8_t *dec = (uint8_t *)malloc(n ? n : 1);
    CHECK(dec != NULL, "malloc dec failed");
    st = tdc_xform_complement_vt.decode(
        enc.data, enc.size, dt, NULL, dec, n, &out_dt);
    CHECK(st == TDC_OK, "decode failed");
    CHECK(out_dt == dt, "decode out_dtype must match in_dtype");
    CHECK(memcmp(dec, src, n) == 0, "round-trip mismatch");

    printf("  [%s %zu bytes] round-trip OK\n", label, n);
    free(dec);
    t_buffer_free(&enc);
    return 0;
}

static int test_dtypes(void) {
    /* u8: full byte range. */
    {
        uint8_t src[256];
        for (int i = 0; i < 256; ++i) src[i] = (uint8_t)i;
        if (roundtrip_bytes(src, 256, TDC_DT_U8, "u8 full range")) return 1;
    }
    /* i32: mixed sign values. */
    {
        int32_t vals[8] = { 0, -1, 1, INT32_MIN, INT32_MAX, -12345, 12345, -7 };
        if (roundtrip_bytes((const uint8_t *)vals, sizeof vals,
                            TDC_DT_I32, "i32 mixed")) return 1;
    }
    /* f64: IEEE-754 bit patterns. */
    {
        double vals[5] = { 0.0, -0.0, 1.0, -3.14159265358979, 1e300 };
        if (roundtrip_bytes((const uint8_t *)vals, sizeof vals,
                            TDC_DT_F64, "f64 mixed")) return 1;
    }
    return 0;
}

/* ----- Edge cases: empty, single, degenerate ----------------------------- */

static int test_empty(void) {
    tdc_buffer enc = t_buffer();
    tdc_dtype  out_dt = (tdc_dtype)0;
    uint8_t    dummy = 0;

    tdc_status st = tdc_xform_complement_vt.encode(
        &dummy, 0, TDC_DT_I32, NULL, &enc, &out_dt);
    CHECK(st == TDC_OK, "empty encode failed");
    CHECK(enc.size == 0, "empty encode size != 0");
    CHECK(out_dt == TDC_DT_I32, "empty out_dtype != I32");

    st = tdc_xform_complement_vt.decode(
        NULL, 0, TDC_DT_I32, NULL, NULL, 0, &out_dt);
    CHECK(st == TDC_OK, "empty decode failed");

    printf("  [empty] OK\n");
    t_buffer_free(&enc);
    return 0;
}

static int test_single(void) {
    uint8_t src = 0x42;
    uint8_t dec = 0;
    tdc_buffer enc = t_buffer();
    tdc_dtype  out_dt = (tdc_dtype)0;

    tdc_status st = tdc_xform_complement_vt.encode(
        &src, 1, TDC_DT_U8, NULL, &enc, &out_dt);
    CHECK(st == TDC_OK, "single encode failed");
    CHECK(enc.size == 1, "single encode size");
    CHECK(enc.data[0] == (uint8_t)~0x42, "single encoded byte");

    st = tdc_xform_complement_vt.decode(
        enc.data, enc.size, TDC_DT_U8, NULL, &dec, 1, &out_dt);
    CHECK(st == TDC_OK, "single decode failed");
    CHECK(dec == src, "single round-trip");

    printf("  [single byte] OK\n");
    t_buffer_free(&enc);
    return 0;
}

static int test_degenerate(void) {
    /* All-equal input: the complement is also all-equal, which lets us
     * assert the output pattern instead of only round-trip equality. */
    enum { N = 64 };
    uint8_t src[N];
    memset(src, 0xAA, N);

    tdc_buffer enc = t_buffer();
    tdc_dtype  out_dt = (tdc_dtype)0;
    tdc_status st = tdc_xform_complement_vt.encode(
        src, N, TDC_DT_U8, NULL, &enc, &out_dt);
    CHECK(st == TDC_OK, "degenerate encode failed");
    for (size_t i = 0; i < N; ++i) {
        CHECK(enc.data[i] == 0x55u, "all-equal encode pattern");
    }

    uint8_t dec[N];
    st = tdc_xform_complement_vt.decode(
        enc.data, enc.size, TDC_DT_U8, NULL, dec, N, &out_dt);
    CHECK(st == TDC_OK, "degenerate decode failed");
    CHECK(memcmp(dec, src, N) == 0, "degenerate round-trip");

    printf("  [degenerate 0xAA x%d] OK\n", N);
    t_buffer_free(&enc);
    return 0;
}

/* ----- Rejection paths --------------------------------------------------- */

static int test_rejections(void) {
    tdc_buffer enc = t_buffer();
    tdc_dtype  out_dt = (tdc_dtype)0;

    /* Variable-width string dtype must be rejected. */
    {
        uint8_t junk[4] = {0};
        tdc_status st = tdc_xform_complement_vt.encode(
            junk, sizeof junk, TDC_DT_STRING, NULL, &enc, &out_dt);
        CHECK(st == TDC_E_DTYPE, "encode should reject STRING");
    }

    /* Decode dst_size != src_size must report corruption. */
    {
        uint8_t enc_bytes[4] = {0};
        uint8_t dec_buf[3];
        tdc_status st = tdc_xform_complement_vt.decode(
            enc_bytes, sizeof enc_bytes, TDC_DT_U8, NULL,
            dec_buf, sizeof dec_buf, &out_dt);
        CHECK(st == TDC_E_CORRUPT, "decode should reject size mismatch");
    }

    printf("  [rejections] OK\n");
    t_buffer_free(&enc);
    return 0;
}

/* ----- Full pipeline via tdc_encode_block / tdc_decode_block ------------- */

static int test_pipeline(void) {
    /* Build a VECTOR_1D u8 block with a predictable pattern. */
    enum { N = 128 };
    uint8_t src[N];
    for (int i = 0; i < N; ++i) src[i] = (uint8_t)(i * 7u);

    tdc_block blk = {0};
    blk.data   = src;
    blk.dtype  = TDC_DT_U8;
    blk.layout = TDC_LAYOUT_VECTOR_1D;
    blk.shape.rank   = 1;
    blk.shape.dim[0] = N;
    tdc_shape_set_contiguous(&blk.shape);

    /* Spec: RAW model, complement transform, NONE entropy. */
    tdc_codec_spec spec = tdc_codec_spec_raw();
    spec.xform[0] = TDC_XFORM_COMPLEMENT;

    tdc_buffer out = t_buffer();
    tdc_status st = tdc_encode_block(&blk, &spec, &out);
    CHECK(st == TDC_OK, "pipeline encode failed");
    CHECK(out.size > 0, "pipeline produced empty output");

    uint8_t decoded[N];
    memset(decoded, 0xCC, sizeof decoded);

    tdc_block dst = {0};
    dst.data   = decoded;
    dst.dtype  = TDC_DT_U8;
    dst.layout = TDC_LAYOUT_VECTOR_1D;
    dst.shape.rank   = 1;
    dst.shape.dim[0] = N;
    tdc_shape_set_contiguous(&dst.shape);

    st = tdc_decode_block(out.data, out.size, &dst);
    CHECK(st == TDC_OK, "pipeline decode failed");
    CHECK(memcmp(decoded, src, N) == 0, "pipeline round-trip");

    printf("  [pipeline RAW + COMPLEMENT + NONE] %zu -> %zu bytes, OK\n",
           (size_t)N, out.size);
    t_buffer_free(&out);
    return 0;
}

int main(void) {
    printf("test_complement_roundtrip:\n");
    if (test_registry())   return 1;
    if (test_dtypes())     return 1;
    if (test_empty())      return 1;
    if (test_single())     return 1;
    if (test_degenerate()) return 1;
    if (test_rejections()) return 1;
    if (test_pipeline())   return 1;
    printf("test_complement_roundtrip: OK\n");
    return 0;
}
