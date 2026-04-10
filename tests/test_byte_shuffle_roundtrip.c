/*
 * tests/test_byte_shuffle_roundtrip.c
 *
 * Round-trip test for the TDC_XFORM_BYTE_SHUFFLE backend.
 *
 * Verifies:
 *   1. shuffle -> unshuffle is byte-identical for f64, f32, i16, u8.
 *   2. The forward shuffle actually transposes (spot-check the byte
 *      lanes for a tiny known input).
 *   3. The SSE2/NEON 8-byte fast path runs through n=16 (one full
 *      vectorized iteration) and n=17 (one iteration + scalar tail).
 *   4. Edge cases: empty buffer, single element.
 *   5. Decode rejects src_size that is not a multiple of elem_size with
 *      TDC_E_CORRUPT.
 *   6. Decode rejects TDC_DT_STRING with TDC_E_DTYPE.
 *
 * Allocation goes through a tiny realloc wrapper so the buffer code path
 * matches what vectra (and any other downstream caller) will use.
 */

#include "tdc/transform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Resolve the byte_shuffle vtable directly so this test does not depend
 * on registry.c. */
extern const tdc_xform_vt tdc_xform_byte_shuffle_vt;

/* POSIX-style realloc wrapper: (NULL, n) allocs, (p, 0) frees, (p, n) grows. */
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

#define ASSERT_OR_DIE(cond, msg) do {                                   \
    if (!(cond)) {                                                      \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);\
        return 1;                                                       \
    }                                                                   \
} while (0)

/* LCG byte fill (Numerical Recipes constants). */
static void fill_lcg(uint8_t *buf, size_t n, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        buf[i] = (uint8_t)(s >> 24);
    }
}

static int roundtrip_check(const uint8_t *src, size_t n, tdc_dtype dt,
                           const char *label) {
    tdc_buffer enc = make_buffer();
    tdc_dtype  out_dt = (tdc_dtype)0;

    tdc_status st = tdc_xform_byte_shuffle_vt.encode(
        src, n, dt, NULL, &enc, &out_dt);
    ASSERT_OR_DIE(st == TDC_OK, "encode failed");
    ASSERT_OR_DIE(enc.size == n, "encoded size != input size");
    ASSERT_OR_DIE(out_dt == dt, "out_dtype != in_dtype");

    uint8_t *dec = (uint8_t *)malloc(n ? n : 1);
    ASSERT_OR_DIE(dec != NULL, "decode buffer alloc failed");

    st = tdc_xform_byte_shuffle_vt.decode(
        enc.data, enc.size, dt, NULL, dec, n, &out_dt);
    ASSERT_OR_DIE(st == TDC_OK, "decode failed");
    ASSERT_OR_DIE(memcmp(dec, src, n) == 0, "round-trip mismatch");
    ASSERT_OR_DIE(out_dt == dt, "decode out_dtype != in_dtype");

    printf("  [%s] %zu bytes round-trip OK\n", label, n);

    free(dec);
    free_buffer(&enc);
    return 0;
}

/* Spot-check the forward transpose for a tiny known input.
 * 4 elements of 8 bytes each, lane bytes set to (elem_index, lane_index)
 * so we can verify the output layout exactly. */
static int spot_check_layout(void) {
    enum { N = 4, E = 8 };
    uint8_t src[N * E];
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t b = 0; b < E; b++) {
            src[i * E + b] = (uint8_t)((i << 4) | b);
        }
    }

    tdc_buffer enc = make_buffer();
    tdc_dtype out_dt = (tdc_dtype)0;
    tdc_status st = tdc_xform_byte_shuffle_vt.encode(
        src, sizeof src, TDC_DT_F64, NULL, &enc, &out_dt);
    ASSERT_OR_DIE(st == TDC_OK, "spot-check encode failed");
    ASSERT_OR_DIE(enc.size == sizeof src, "spot-check size mismatch");

    /* Expected layout: lane b at offset b*N, contains src[0*E+b], src[1*E+b],
     * src[2*E+b], src[3*E+b]. */
    for (uint32_t b = 0; b < E; b++) {
        for (uint32_t i = 0; i < N; i++) {
            uint8_t got = enc.data[b * N + i];
            uint8_t expected = (uint8_t)((i << 4) | b);
            if (got != expected) {
                fprintf(stderr,
                        "FAIL: spot_check lane %u elem %u: got 0x%02x expected 0x%02x\n",
                        b, i, got, expected);
                free_buffer(&enc);
                return 1;
            }
        }
    }

    free_buffer(&enc);
    printf("  [spot-check] f64 layout verified (4 elems, 8-byte lanes)\n");
    return 0;
}

int main(void) {
    printf("test_byte_shuffle_roundtrip:\n");

    if (spot_check_layout()) return 1;

    /* f64: must hit the SSE2/NEON fast path. n_elems = 1024 -> 8 KB. */
    {
        const size_t N_ELEMS = 1024;
        const size_t N = N_ELEMS * 8;
        uint8_t *src = (uint8_t *)malloc(N);
        ASSERT_OR_DIE(src, "alloc failed");
        fill_lcg(src, N, 0xDEADBEEFu);
        if (roundtrip_check(src, N, TDC_DT_F64, "f64 1024 elems")) {
            free(src); return 1;
        }
        free(src);
    }

    /* f64 fast path boundary: exactly 16 elems (one full SIMD iteration). */
    {
        const size_t N = 16 * 8;
        uint8_t src[16 * 8];
        fill_lcg(src, N, 0x12345678u);
        if (roundtrip_check(src, N, TDC_DT_F64, "f64 16 elems")) return 1;
    }

    /* f64 with scalar tail: 17 elems (one SIMD iter + one scalar). */
    {
        const size_t N = 17 * 8;
        uint8_t src[17 * 8];
        fill_lcg(src, N, 0xABCDEF01u);
        if (roundtrip_check(src, N, TDC_DT_F64, "f64 17 elems")) return 1;
    }

    /* f32: 4-byte elem, scalar path only. */
    {
        const size_t N = 1024 * 4;
        uint8_t *src = (uint8_t *)malloc(N);
        ASSERT_OR_DIE(src, "alloc failed");
        fill_lcg(src, N, 0xCAFEBABEu);
        if (roundtrip_check(src, N, TDC_DT_F32, "f32 1024 elems")) {
            free(src); return 1;
        }
        free(src);
    }

    /* i16: 2-byte elem. */
    {
        const size_t N = 512 * 2;
        uint8_t *src = (uint8_t *)malloc(N);
        ASSERT_OR_DIE(src, "alloc failed");
        fill_lcg(src, N, 0x5A5A5A5Au);
        if (roundtrip_check(src, N, TDC_DT_I16, "i16 512 elems")) {
            free(src); return 1;
        }
        free(src);
    }

    /* u8: identity (elem_size == 1, memcpy fast path). */
    {
        const size_t N = 256;
        uint8_t src[256];
        fill_lcg(src, N, 0x01020304u);
        if (roundtrip_check(src, N, TDC_DT_U8, "u8 256 elems")) return 1;
    }

    /* Edge cases: empty, single f64. */
    {
        uint8_t dummy = 0;
        if (roundtrip_check(&dummy, 0, TDC_DT_F64, "empty f64")) return 1;
    }
    {
        uint8_t src[8];
        fill_lcg(src, 8, 0x11223344u);
        if (roundtrip_check(src, 8, TDC_DT_F64, "single f64")) return 1;
    }

    /* Decode rejects size not a multiple of elem_size. */
    {
        uint8_t junk[7] = {0};
        uint8_t dec[7];
        tdc_dtype out_dt = (tdc_dtype)0;
        tdc_status st = tdc_xform_byte_shuffle_vt.decode(
            junk, sizeof junk, TDC_DT_F64, NULL, dec, sizeof dec, &out_dt);
        ASSERT_OR_DIE(st == TDC_E_CORRUPT,
                      "decode should reject non-multiple size with TDC_E_CORRUPT");
        printf("  [reject misaligned] OK\n");
    }

    /* Decode rejects TDC_DT_STRING (variable-length, elem_size == 0). */
    {
        uint8_t junk[16] = {0};
        uint8_t dec[16];
        tdc_dtype out_dt = (tdc_dtype)0;
        tdc_status st = tdc_xform_byte_shuffle_vt.decode(
            junk, sizeof junk, TDC_DT_STRING, NULL, dec, sizeof dec, &out_dt);
        ASSERT_OR_DIE(st == TDC_E_DTYPE,
                      "decode should reject TDC_DT_STRING with TDC_E_DTYPE");
        printf("  [reject string dtype] OK\n");
    }

    /* Encode rejects mismatched size too. */
    {
        uint8_t junk[7] = {0};
        tdc_buffer enc = make_buffer();
        tdc_dtype out_dt = (tdc_dtype)0;
        tdc_status st = tdc_xform_byte_shuffle_vt.encode(
            junk, sizeof junk, TDC_DT_F64, NULL, &enc, &out_dt);
        ASSERT_OR_DIE(st == TDC_E_INVAL,
                      "encode should reject non-multiple size with TDC_E_INVAL");
        free_buffer(&enc);
        printf("  [encode reject misaligned] OK\n");
    }

    printf("test_byte_shuffle_roundtrip: OK\n");
    return 0;
}
