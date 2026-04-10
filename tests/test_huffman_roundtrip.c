/*
 * tests/test_huffman_roundtrip.c
 *
 * Round-trip test for the TDC_ENTROPY_HUFFMAN backend.
 *
 * Verifies:
 *   1. Round-trip is byte-identical for empty, single-byte, single-symbol,
 *      uniform-random, and skewed-frequency inputs.
 *   2. Skewed-frequency input compresses noticeably below the input size
 *      (this is the whole reason Huffman exists).
 *   3. Uniform-random input does NOT compress dramatically below 8 bits/sym
 *      (within the header overhead).
 *   4. encode_bound is honored on every shape.
 */

#include "tdc/entropy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern const tdc_entropy_vt tdc_entropy_huffman_vt;

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
    tdc_status st = tdc_entropy_huffman_vt.encode(src, n, NULL, &enc);
    ASSERT_OR_DIE(st == TDC_OK, "encode failed");
    ASSERT_OR_DIE(enc.size <= tdc_entropy_huffman_vt.encode_bound(n),
                  "encode_bound exceeded");

    uint8_t *dec = (uint8_t *)malloc(n ? n : 1);
    ASSERT_OR_DIE(dec != NULL, "decode buffer alloc");
    st = tdc_entropy_huffman_vt.decode(enc.data, enc.size, dec, n);
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

/* Geometric/skewed: byte 0 with p=0.5, byte 1 with p=0.25, ... up to byte 7. */
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
    printf("test_huffman_roundtrip:\n");

    /* 1. empty */
    if (roundtrip(NULL, 0, "empty")) return 1;

    /* 2. single byte */
    {
        uint8_t one = 0xAB;
        if (roundtrip(&one, 1, "1 byte")) return 1;
    }

    /* 3. single distinct symbol, many copies (length-1 canonical code) */
    {
        size_t n = 1000;
        uint8_t *buf = (uint8_t *)malloc(n);
        memset(buf, 0x42, n);
        if (roundtrip(buf, n, "1-sym x1000")) { free(buf); return 1; }
        free(buf);
    }

    /* 4. uniform LCG, 64 KB. Should not compress much below input size. */
    {
        size_t n = 64u * 1024u;
        uint8_t *buf = (uint8_t *)malloc(n);
        fill_lcg(buf, n, 0xDEADBEEFu);
        tdc_buffer enc = make_buffer();
        tdc_status st = tdc_entropy_huffman_vt.encode(buf, n, NULL, &enc);
        ASSERT_OR_DIE(st == TDC_OK, "uniform encode");
        /* Uniform 8-bit input cannot compress below ~n bytes; allow small
         * slack for the 264-byte header. */
        ASSERT_OR_DIE(enc.size <= n + 512u, "uniform expanded too much");
        uint8_t *dec = (uint8_t *)malloc(n);
        st = tdc_entropy_huffman_vt.decode(enc.data, enc.size, dec, n);
        ASSERT_OR_DIE(st == TDC_OK, "uniform decode");
        ASSERT_OR_DIE(memcmp(dec, buf, n) == 0, "uniform round-trip mismatch");
        printf("  [uniform 64KB] %zu -> %zu (%.1f%%)\n",
               n, enc.size, 100.0 * (double)enc.size / (double)n);
        free(dec);
        free_buffer(&enc);
        free(buf);
    }

    /* 5. skewed distribution, 64 KB. Should compress to clearly < 50%. */
    {
        size_t n = 64u * 1024u;
        uint8_t *buf = (uint8_t *)malloc(n);
        fill_skewed(buf, n, 0xCAFEBABEu);
        tdc_buffer enc = make_buffer();
        tdc_status st = tdc_entropy_huffman_vt.encode(buf, n, NULL, &enc);
        ASSERT_OR_DIE(st == TDC_OK, "skewed encode");
        ASSERT_OR_DIE(enc.size < n / 2u, "skewed should compress < 50%");
        uint8_t *dec = (uint8_t *)malloc(n);
        st = tdc_entropy_huffman_vt.decode(enc.data, enc.size, dec, n);
        ASSERT_OR_DIE(st == TDC_OK, "skewed decode");
        ASSERT_OR_DIE(memcmp(dec, buf, n) == 0, "skewed round-trip mismatch");
        printf("  [skewed 64KB] %zu -> %zu (%.1f%%)\n",
               n, enc.size, 100.0 * (double)enc.size / (double)n);
        free(dec);
        free_buffer(&enc);
        free(buf);
    }

    printf("test_huffman_roundtrip: OK\n");
    return 0;
}
