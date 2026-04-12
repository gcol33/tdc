/*
 * tests/test_lz_roundtrip.c
 *
 * Round-trip test for the TDC_ENTROPY_LZ backend.
 *
 * Generates a 1 MB deterministic byte buffer with two regions:
 *   - First half: an LCG-driven pseudo-random stream (literal-heavy;
 *     should not compress meaningfully).
 *   - Second half: a short repeating pattern (match-heavy; should
 *     compress aggressively).
 *
 * Verifies:
 *   1. encode -> decode round-trip is byte-identical.
 *   2. The match-heavy region alone compresses to noticeably less than
 *      its uncompressed size.
 *   3. The encode_bound is honored.
 *
 * Allocation goes through a tiny realloc wrapper so the buffer code path
 * matches what vectra (and any other downstream caller) will use.
 */

#include "tdc/entropy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Resolve the lz vtable directly (not via tdc_entropy_get) so this test
 * doesn't depend on registry.c. */
extern const tdc_entropy_vt tdc_entropy_lz_vt;

/* POSIX-style realloc wrapper: (NULL, n) allocs, (p, 0) frees, (p, n) grows. */
static void *test_realloc(void *user, void *ptr, size_t new_size) {
    (void)user;
    if (new_size == 0) {
        free(ptr);
        return NULL;
    }
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

/* Fill `buf` with deterministic LCG bytes (Numerical Recipes constants). */
static void fill_lcg(uint8_t *buf, size_t n, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        buf[i] = (uint8_t)(s >> 24);
    }
}

/* Fill `buf` with a short repeating pattern. */
static void fill_periodic(uint8_t *buf, size_t n) {
    static const uint8_t pattern[16] = {
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
        0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF
    };
    for (size_t i = 0; i < n; i++) buf[i] = pattern[i & 15];
}

static int roundtrip_check(const uint8_t *src, size_t n, const char *label) {
    tdc_buffer enc = make_buffer();

    tdc_status st = tdc_entropy_lz_vt.encode(src, n, NULL, &enc);
    ASSERT_OR_DIE(st == TDC_OK, "encode failed");
    ASSERT_OR_DIE(enc.size <= tdc_entropy_lz_vt.encode_bound(n),
                  "encode_bound exceeded");

    uint8_t *dec = (uint8_t *)malloc(n ? n : 1);
    ASSERT_OR_DIE(dec != NULL, "decode buffer alloc failed");

    st = tdc_entropy_lz_vt.decode(enc.data, enc.size, dec, n);
    ASSERT_OR_DIE(st == TDC_OK, "decode failed");
    ASSERT_OR_DIE(memcmp(dec, src, n) == 0, "round-trip mismatch");

    printf("  [%s] %zu bytes -> %zu bytes (%.1f%%)\n",
           label, n, enc.size,
           n ? (100.0 * (double)enc.size / (double)n) : 0.0);

    free(dec);
    free_buffer(&enc);
    return 0;
}

int main(void) {
    const size_t N = 1u << 20; /* 1 MB */
    uint8_t *src = (uint8_t *)malloc(N);
    if (!src) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* Region A: literal-heavy LCG bytes. Region B: periodic pattern. */
    fill_lcg(src, N / 2, 0xC0FFEEu);
    fill_periodic(src + N / 2, N / 2);

    printf("test_lz_roundtrip:\n");

    /* 1. Mixed buffer round-trip. */
    if (roundtrip_check(src, N, "mixed 1MB")) { free(src); return 1; }

    /* 2. Match-heavy region in isolation: must compress significantly. */
    {
        tdc_buffer enc = make_buffer();
        tdc_status st = tdc_entropy_lz_vt.encode(src + N / 2, N / 2, NULL, &enc);
        ASSERT_OR_DIE(st == TDC_OK, "encode (match-heavy) failed");
        ASSERT_OR_DIE(enc.size < (N / 2) / 4,
                      "match-heavy region failed to compress to <25% of input");
        printf("  [match-heavy] %zu bytes -> %zu bytes (%.1f%%)\n",
               N / 2, enc.size, 100.0 * (double)enc.size / (double)(N / 2));
        free_buffer(&enc);
    }

    /* 3. Literal-heavy region in isolation. Must round-trip; size is
     *    allowed to be up to encode_bound(n). */
    if (roundtrip_check(src, N / 2, "literal-heavy")) { free(src); return 1; }

    /* 4. Edge cases: empty, 1 byte, 3 bytes (below LZ_MIN_MATCH+1). */
    if (roundtrip_check(src, 0, "empty"))  { free(src); return 1; }
    if (roundtrip_check(src, 1, "1 byte")) { free(src); return 1; }
    if (roundtrip_check(src, 3, "3 bytes")) { free(src); return 1; }

    /* 5. Pure periodic, small (16 bytes). */
    {
        uint8_t small[64];
        fill_periodic(small, sizeof small);
        if (roundtrip_check(small, sizeof small, "periodic 64B")) {
            free(src); return 1;
        }
    }

    free(src);
    printf("test_lz_roundtrip: OK\n");
    return 0;
}
