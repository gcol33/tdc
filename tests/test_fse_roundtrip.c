/*
 * tests/test_fse_roundtrip.c
 *
 * Round-trip test for the TDC_ENTROPY_FSE backend (rANS).
 *
 * Verifies:
 *   1. Round-trip on empty, single byte, single-symbol streams,
 *      uniform-random, and skewed-frequency inputs.
 *   2. Skewed input compresses substantially below the input size
 *      (the whole point of an entropy coder).
 *   3. Uniform-random input does not expand by much (within header).
 *   4. encode_bound is honored on every shape.
 */

#include "tdc/entropy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern const tdc_entropy_vt tdc_entropy_fse_vt;

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

#define ASSERT_OR_DIE(cond, msg) do {                                    \
    if (!(cond)) {                                                       \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);\
        return 1;                                                        \
    }                                                                    \
} while (0)

static int roundtrip(const uint8_t *src, size_t n, const char *label) {
    tdc_buffer enc = make_buffer();
    tdc_status st = tdc_entropy_fse_vt.encode(src, n, NULL, &enc);
    ASSERT_OR_DIE(st == TDC_OK, "encode failed");
    ASSERT_OR_DIE(enc.size <= tdc_entropy_fse_vt.encode_bound(n),
                  "encode_bound exceeded");

    uint8_t *dec = (uint8_t *)malloc(n ? n : 1);
    ASSERT_OR_DIE(dec != NULL, "decode buffer alloc");
    st = tdc_entropy_fse_vt.decode(enc.data, enc.size, dec, n);
    ASSERT_OR_DIE(st == TDC_OK, "decode failed");
    ASSERT_OR_DIE(memcmp(dec, src, n) == 0, "round-trip mismatch");

    printf("  [%s] %zu -> %zu (%.1f%%)\n",
           label, n, enc.size, n ? 100.0 * (double)enc.size / (double)n : 0.0);
    free(dec);
    free_buffer(&enc);
    return 0;
}

static void fill_lcg(uint8_t *buf, size_t n, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        buf[i] = (uint8_t)(s >> 24);
    }
}

static void fill_skewed(uint8_t *buf, size_t n, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        uint32_t r = s >> 8;
        uint8_t v = 7;
        for (int k = 0; k < 7; ++k) {
            if (r & (1u << k)) { v = (uint8_t)k; break; }
        }
        buf[i] = v;
    }
}

int main(void) {
    printf("test_fse_roundtrip:\n");

    /* 1. empty */
    if (roundtrip(NULL, 0, "empty")) return 1;

    /* 2. single byte */
    {
        uint8_t one = 0x5A;
        if (roundtrip(&one, 1, "1 byte")) return 1;
    }

    /* 3. single distinct symbol, many copies */
    {
        size_t n = 1000;
        uint8_t *buf = (uint8_t *)malloc(n);
        memset(buf, 0x42, n);
        if (roundtrip(buf, n, "1-sym x1000")) { free(buf); return 1; }
        free(buf);
    }

    /* 4. uniform LCG, 64 KB. Should not expand much. */
    {
        size_t n = 64u * 1024u;
        uint8_t *buf = (uint8_t *)malloc(n);
        fill_lcg(buf, n, 0xDEADBEEFu);
        tdc_buffer enc = make_buffer();
        tdc_status st = tdc_entropy_fse_vt.encode(buf, n, NULL, &enc);
        ASSERT_OR_DIE(st == TDC_OK, "uniform encode");
        /* TABLE_LOG=11 for 256-symbol alphabets gives ~1% quantization
         * overhead on uniform data (see fse_pick_table_log). 64 KiB ×
         * 1.03 covers the observed overhead plus 528-byte header. */
        ASSERT_OR_DIE(enc.size <= n + n / 32u, "uniform expanded too much");
        uint8_t *dec = (uint8_t *)malloc(n);
        st = tdc_entropy_fse_vt.decode(enc.data, enc.size, dec, n);
        ASSERT_OR_DIE(st == TDC_OK, "uniform decode");
        ASSERT_OR_DIE(memcmp(dec, buf, n) == 0, "uniform round-trip mismatch");
        printf("  [uniform 64KB] %zu -> %zu (%.1f%%)\n",
               n, enc.size, 100.0 * (double)enc.size / (double)n);
        free(dec);
        free_buffer(&enc);
        free(buf);
    }

    /* 5. skewed distribution, 64 KB. Must compress to clearly < 50%. */
    {
        size_t n = 64u * 1024u;
        uint8_t *buf = (uint8_t *)malloc(n);
        fill_skewed(buf, n, 0xCAFEBABEu);
        tdc_buffer enc = make_buffer();
        tdc_status st = tdc_entropy_fse_vt.encode(buf, n, NULL, &enc);
        ASSERT_OR_DIE(st == TDC_OK, "skewed encode");
        ASSERT_OR_DIE(enc.size < n / 2u, "skewed should compress < 50%");
        uint8_t *dec = (uint8_t *)malloc(n);
        st = tdc_entropy_fse_vt.decode(enc.data, enc.size, dec, n);
        ASSERT_OR_DIE(st == TDC_OK, "skewed decode");
        ASSERT_OR_DIE(memcmp(dec, buf, n) == 0, "skewed round-trip mismatch");
        printf("  [skewed 64KB] %zu -> %zu (%.1f%%)\n",
               n, enc.size, 100.0 * (double)enc.size / (double)n);
        free(dec);
        free_buffer(&enc);
        free(buf);
    }

    printf("test_fse_roundtrip: OK\n");
    return 0;
}
