/*
 * tests/test_deflate_roundtrip.c
 *
 * Round-trip test for the TDC_ENTROPY_DEFLATE backend.
 * Only compiled and run when TDC_HAVE_ZLIB is enabled.
 * When zlib is not available, the test prints a skip message and returns 0.
 *
 * Verifies:
 *   1. Round-trip on compressible data (repeated pattern).
 *   2. Round-trip on incompressible data (pseudo-random).
 *   3. Round-trip on empty input.
 *   4. Round-trip on single-byte input.
 *   5. Registry returns the vtable when TDC_HAVE_ZLIB is on.
 */

#include "tdc/entropy.h"
#include "tdc/codec.h"
#include "tdc/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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

#ifdef TDC_HAVE_ZLIB

extern const tdc_entropy_vt tdc_entropy_deflate_vt;

static int rt_one(const char *label, const uint8_t *src, size_t src_size) {
    const tdc_entropy_vt *vt = &tdc_entropy_deflate_vt;

    tdc_buffer enc = make_buffer();
    tdc_status st = vt->encode(src, src_size, NULL, &enc);
    ASSERT_OR_DIE(st == TDC_OK, "encode returned non-OK");

    uint8_t *dst = (uint8_t *)malloc(src_size > 0 ? src_size : 1u);
    ASSERT_OR_DIE(dst != NULL, "malloc dst");
    memset(dst, 0xBB, src_size > 0 ? src_size : 1u);

    st = vt->decode(enc.data, enc.size, dst, src_size);
    ASSERT_OR_DIE(st == TDC_OK, "decode returned non-OK");
    if (src_size > 0) {
        ASSERT_OR_DIE(memcmp(dst, src, src_size) == 0,
                      "decoded data does not match source");
    }

    printf("  [%s] src=%zu enc=%zu round-trip OK\n",
           label, src_size, enc.size);

    free(dst);
    free_buffer(&enc);
    return 0;
}

static int test_compressible(void) {
    uint8_t src[1024];
    for (int i = 0; i < 1024; ++i)
        src[i] = (uint8_t)(i % 16);
    return rt_one("compressible 1024", src, sizeof src);
}

static int test_incompressible(void) {
    uint8_t src[256];
    uint32_t lcg = 12345;
    for (int i = 0; i < 256; ++i) {
        lcg = lcg * 1103515245u + 12345u;
        src[i] = (uint8_t)(lcg >> 16);
    }
    return rt_one("pseudo-random 256", src, sizeof src);
}

static int test_empty(void) {
    return rt_one("empty", NULL, 0);
}

static int test_single_byte(void) {
    uint8_t src[1] = { 42 };
    return rt_one("single byte", src, 1);
}

static int test_with_level(void) {
    const tdc_entropy_vt *vt = &tdc_entropy_deflate_vt;
    uint8_t src[512];
    for (int i = 0; i < 512; ++i) src[i] = (uint8_t)(i % 8);

    tdc_entropy_level params = { .level = 9 }; /* max compression */
    tdc_buffer enc = make_buffer();
    tdc_status st = vt->encode(src, sizeof src, &params, &enc);
    ASSERT_OR_DIE(st == TDC_OK, "encode with level=9");

    uint8_t dst[512];
    st = vt->decode(enc.data, enc.size, dst, sizeof dst);
    ASSERT_OR_DIE(st == TDC_OK, "decode");
    ASSERT_OR_DIE(memcmp(dst, src, sizeof src) == 0, "data mismatch");

    printf("  [level=9] src=512 enc=%zu OK\n", enc.size);
    free_buffer(&enc);
    return 0;
}

static int test_registry(void) {
    const tdc_entropy_vt *vt = tdc_entropy_get(TDC_ENTROPY_DEFLATE);
    ASSERT_OR_DIE(vt == &tdc_entropy_deflate_vt,
                  "tdc_entropy_get(TDC_ENTROPY_DEFLATE) wiring");
    ASSERT_OR_DIE(vt->id == TDC_ENTROPY_DEFLATE, "vt->id mismatch");
    printf("  [registry wiring] OK\n");
    return 0;
}

int main(void) {
    printf("test_deflate_roundtrip\n");
    if (test_compressible())   return 1;
    if (test_incompressible()) return 1;
    if (test_empty())          return 1;
    if (test_single_byte())    return 1;
    if (test_with_level())     return 1;
    if (test_registry())       return 1;
    printf("ALL OK\n");
    return 0;
}

#else /* !TDC_HAVE_ZLIB */

int main(void) {
    printf("test_deflate_roundtrip: SKIPPED (TDC_HAVE_ZLIB not enabled)\n");
    return 0;
}

#endif /* TDC_HAVE_ZLIB */
