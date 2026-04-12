/*
 * tests/test_lz_streams_roundtrip.c
 *
 * Round-trip + ratio-sanity test for the TDC_ENTROPY_LZ_STREAMS backend.
 *
 * Invariants:
 *   1. decode(encode(x)) == x for every fixture.
 *   2. The encode_bound is honored.
 *   3. On the structured-numeric fixture (shuffled int64 ramp) the
 *      streams serializer must not produce a larger stream than the
 *      single-stream LZ. If it does, the stream layout is regressing
 *      against its design target — the whole point is to beat the
 *      packed single-stream format on structured data.
 *
 * Fixture shapes mirror test_lz_opt_roundtrip.c so any future
 * divergence between the two serializers is obvious at a glance.
 */

#include "tdc/entropy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

extern const tdc_entropy_vt tdc_entropy_lz_vt;
extern const tdc_entropy_vt tdc_entropy_lz_streams_vt;

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

#define ASSERT_OR_DIE(cond, msg) do {                                   \
    if (!(cond)) {                                                      \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);\
        return 1;                                                       \
    }                                                                   \
} while (0)

static void fill_lcg(uint8_t *buf, size_t n, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        buf[i] = (uint8_t)(s >> 24);
    }
}

static void fill_periodic(uint8_t *buf, size_t n) {
    static const uint8_t pattern[16] = {
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
        0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF
    };
    for (size_t i = 0; i < n; i++) buf[i] = pattern[i & 15];
}

static void fill_shuffled_i64_ramp(uint8_t *buf, size_t n) {
    size_t elems = n / 8;
    size_t lane_len = elems;
    for (size_t e = 0; e < elems; e++) {
        int64_t v = (int64_t)(1000 + (int64_t)e * 3);
        uint64_t u;
        memcpy(&u, &v, 8);
        for (int lane = 0; lane < 8; lane++) {
            buf[(size_t)lane * lane_len + e] = (uint8_t)(u >> (lane * 8));
        }
    }
    for (size_t i = elems * 8; i < n; i++) buf[i] = 0;
}

static void fill_zeros(uint8_t *buf, size_t n) { memset(buf, 0, n); }

static int check(const uint8_t *src, size_t n, const char *label,
                 int require_beat_greedy) {
    tdc_buffer greedy  = make_buffer();
    tdc_buffer streams = make_buffer();

    tdc_status st = tdc_entropy_lz_vt.encode(src, n, NULL, &greedy);
    ASSERT_OR_DIE(st == TDC_OK, "greedy encode failed");

    st = tdc_entropy_lz_streams_vt.encode(src, n, NULL, &streams);
    ASSERT_OR_DIE(st == TDC_OK, "streams encode failed");

    ASSERT_OR_DIE(streams.size <= tdc_entropy_lz_streams_vt.encode_bound(n),
                  "streams encode_bound exceeded");

    double ratio_g = n ? (100.0 * (double)greedy.size  / (double)n) : 0.0;
    double ratio_s = n ? (100.0 * (double)streams.size / (double)n) : 0.0;
    double saving  = greedy.size
                       ? (100.0 * (1.0 - (double)streams.size / (double)greedy.size))
                       : 0.0;
    printf("  [%-26s] %7zu B  lz=%7zu (%5.1f%%)  streams=%7zu (%5.1f%%)  delta=%+6.1f%%\n",
           label, n, greedy.size, ratio_g, streams.size, ratio_s, saving);

    int invariant_ok = 1;
    if (require_beat_greedy && n > 64 && streams.size > greedy.size) {
        fprintf(stderr, "    REGRESSION: streams (%zu) > greedy (%zu) on '%s'\n",
                streams.size, greedy.size, label);
        invariant_ok = 0;
    }

    /* Round-trip. */
    uint8_t *dec = (uint8_t *)malloc(n ? n : 1);
    ASSERT_OR_DIE(dec != NULL, "decode buffer alloc failed");
    memset(dec, 0xA5, n ? n : 1);

    st = tdc_entropy_lz_streams_vt.decode(streams.data, streams.size, dec, n);
    ASSERT_OR_DIE(st == TDC_OK, "streams decode failed");
    ASSERT_OR_DIE(memcmp(dec, src, n) == 0, "streams round-trip mismatch");

    free(dec);
    free_buffer(&streams);
    free_buffer(&greedy);
    return invariant_ok ? 0 : 2;
}

int main(void) {
    const size_t N = 1u << 20; /* 1 MB */
    uint8_t *src = (uint8_t *)malloc(N);
    if (!src) { fprintf(stderr, "alloc failed\n"); return 1; }

    printf("test_lz_streams_roundtrip:\n");
    int any_fail = 0;

    fill_lcg(src, N / 2, 0xC0FFEEu);
    fill_periodic(src + N / 2, N / 2);
    if (check(src, N, "mixed 1MB", 0)) any_fail = 1;

    fill_periodic(src, N);
    if (check(src, N, "periodic 1MB", 0)) any_fail = 1;

    fill_lcg(src, N / 2, 0xDEADBEEFu);
    if (check(src, N / 2, "literal-heavy 512KB", 0)) any_fail = 1;

    fill_shuffled_i64_ramp(src, N);
    /* Structured-numeric fixture: this is the workload the streams
     * layout is aimed at. It must not regress vs greedy single-stream. */
    if (check(src, N, "shuffled i64 ramp 1MB", 1)) any_fail = 1;

    fill_zeros(src, N);
    if (check(src, N, "all zeros 1MB", 0)) any_fail = 1;

    fill_lcg(src, 64, 0x12345678u);
    if (check(src,  0, "empty",   0)) any_fail = 1;
    if (check(src,  1, "1 byte",  0)) any_fail = 1;
    if (check(src,  3, "3 bytes", 0)) any_fail = 1;
    if (check(src,  4, "4 bytes", 0)) any_fail = 1;

    {
        uint8_t small[64];
        fill_periodic(small, sizeof small);
        if (check(small, sizeof small, "periodic 64B", 0)) any_fail = 1;
    }

    free(src);
    if (any_fail) {
        printf("test_lz_streams_roundtrip: FAIL\n");
        return 1;
    }
    printf("test_lz_streams_roundtrip: OK\n");
    return 0;
}
