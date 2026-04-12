/*
 * tests/test_lz_opt_roundtrip.c
 *
 * Round-trip + size-invariant test for the TDC_ENTROPY_LZ_OPT backend.
 *
 * Verifies three hard invariants:
 *   1. lz_opt encode -> lz decode is byte-identical (shared decoder).
 *   2. For every input, opt_size <= greedy_size. If this ever fails the
 *      DP cost model is wrong — the optimal parser must never produce a
 *      larger stream than the greedy parser.
 *   3. The encode_bound is honored.
 *
 * Uses the same fixtures as test_lz_roundtrip (LCG literal-heavy + short
 * repeating pattern) plus a structured-numeric fixture (byte-shuffled
 * int64 ramp) which is the workload optimal parsing is actually aimed at.
 */

#include "tdc/entropy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

extern const tdc_entropy_vt tdc_entropy_lz_vt;
extern const tdc_entropy_vt tdc_entropy_lz_opt_vt;

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

/* Byte-shuffled int64 ramp: 8 lanes, lane 0 = LSB (varies), lanes 1-7
 * mostly constant. This mimics what vectra's SHUFFLE_LZ pipeline feeds
 * the entropy stage on structured tabular data. */
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

static int check(const uint8_t *src, size_t n, const char *label) {
    tdc_buffer greedy = make_buffer();
    tdc_buffer opt    = make_buffer();

    tdc_status st = tdc_entropy_lz_vt.encode(src, n, NULL, &greedy);
    ASSERT_OR_DIE(st == TDC_OK, "greedy encode failed");

    st = tdc_entropy_lz_opt_vt.encode(src, n, NULL, &opt);
    ASSERT_OR_DIE(st == TDC_OK, "opt encode failed");

    ASSERT_OR_DIE(opt.size <= tdc_entropy_lz_opt_vt.encode_bound(n),
                  "opt encode_bound exceeded");

    double ratio_g = n ? (100.0 * (double)greedy.size / (double)n) : 0.0;
    double ratio_o = n ? (100.0 * (double)opt.size    / (double)n) : 0.0;
    double saving  = greedy.size ? (100.0 * (1.0 - (double)opt.size / (double)greedy.size)) : 0.0;
    printf("  [%-22s] %7zu B  greedy=%7zu (%5.1f%%)  opt=%7zu (%5.1f%%)  saving=%+5.1f%%\n",
           label, n, greedy.size, ratio_g, opt.size, ratio_o, saving);

    /* Invariant: opt must never produce a larger stream than greedy. */
    int invariant_ok = opt.size <= greedy.size;
    if (!invariant_ok) {
        fprintf(stderr, "    INVARIANT FAIL: opt (%zu) > greedy (%zu)\n",
                opt.size, greedy.size);
    }

    /* Round-trip through the shared decoder. */
    uint8_t *dec = (uint8_t *)malloc(n ? n : 1);
    ASSERT_OR_DIE(dec != NULL, "decode buffer alloc failed");

    st = tdc_entropy_lz_vt.decode(opt.data, opt.size, dec, n);
    ASSERT_OR_DIE(st == TDC_OK, "decode (opt stream) failed");
    ASSERT_OR_DIE(memcmp(dec, src, n) == 0, "opt round-trip mismatch");

    /* Also decode via the lz_opt vtable's decode pointer (forwards to
     * the shared decoder) to exercise that path. */
    st = tdc_entropy_lz_opt_vt.decode(opt.data, opt.size, dec, n);
    ASSERT_OR_DIE(st == TDC_OK, "decode (opt vtable) failed");
    ASSERT_OR_DIE(memcmp(dec, src, n) == 0, "opt vtable round-trip mismatch");

    free(dec);
    free_buffer(&opt);
    free_buffer(&greedy);
    return invariant_ok ? 0 : 2;
}

int main(void) {
    const size_t N = 1u << 20; /* 1 MB */
    uint8_t *src = (uint8_t *)malloc(N);
    if (!src) { fprintf(stderr, "alloc failed\n"); return 1; }

    printf("test_lz_opt_roundtrip:\n");
    int any_fail = 0;

    /* 1. Mixed: half LCG noise, half periodic. */
    fill_lcg(src, N / 2, 0xC0FFEEu);
    fill_periodic(src + N / 2, N / 2);
    if (check(src, N, "mixed 1MB"))                { any_fail = 1; }

    /* 2. Pure match-heavy region. */
    fill_periodic(src, N);
    if (check(src, N, "periodic 1MB"))             { any_fail = 1; }

    /* 3. Pure literal-heavy LCG (essentially incompressible). */
    fill_lcg(src, N / 2, 0xDEADBEEFu);
    if (check(src, N / 2, "literal-heavy 512KB"))  { any_fail = 1; }

    /* 4. Shuffled int64 ramp — the structured-numeric workload. */
    fill_shuffled_i64_ramp(src, N);
    if (check(src, N, "shuffled i64 ramp 1MB"))    { any_fail = 1; }

    /* 5. Edge cases. */
    fill_lcg(src, 64, 0x12345678u);
    if (check(src, 0, "empty"))                    { any_fail = 1; }
    if (check(src, 1, "1 byte"))                   { any_fail = 1; }
    if (check(src, 3, "3 bytes"))                  { any_fail = 1; }
    if (check(src, 4, "4 bytes"))                  { any_fail = 1; }

    /* 6. Small periodic. */
    {
        uint8_t small[64];
        fill_periodic(small, sizeof small);
        if (check(small, sizeof small, "periodic 64B")) { any_fail = 1; }
    }

    /* 7. Large: 8 MiB mixed (reproduces pipeline-scale inputs). */
    {
        const size_t BIG = 8u << 20; /* 8 MiB */
        uint8_t *big = (uint8_t *)malloc(BIG);
        if (big) {
            fill_lcg(big, BIG / 2, 0xBEEFCAFEu);
            fill_periodic(big + BIG / 2, BIG / 2);
            if (check(big, BIG, "mixed 8MB"))         { any_fail = 1; }
            free(big);
        }
    }

    /* 8. Byte-shuffled noisy u16 gradient — matches PRED2D+BSHUF pipeline.
     *    2048x2048 u16 = 8 MiB raw, byte-shuffled into 2 lanes. */
    {
        const int NY = 2048, NX = 2048;
        const size_t n_elems = (size_t)NY * (size_t)NX;
        const size_t raw_size = n_elems * 2;
        uint16_t *raw = (uint16_t *)malloc(raw_size);
        uint8_t *shuf = (uint8_t *)malloc(raw_size);
        if (raw && shuf) {
            uint32_t s = 0xC0FFEEu;
            for (int r = 0; r < NY; ++r)
                for (int c = 0; c < NX; ++c) {
                    s = s * 1103515245u + 12345u;
                    int noise = (int)((s >> 20) & 0x7) - 3;
                    raw[r * NX + c] = (uint16_t)(100 + r * 5 + c * 3 + noise);
                }
            /* Apply Paeth-like residual (just use raw as-is for repro). */
            /* Byte-shuffle: lane 0 = low bytes, lane 1 = high bytes. */
            for (size_t i = 0; i < n_elems; i++) {
                shuf[i]           = (uint8_t)(raw[i] & 0xFF);
                shuf[n_elems + i] = (uint8_t)(raw[i] >> 8);
            }
            if (check(shuf, raw_size, "bshuf noisy u16 8MB")) { any_fail = 1; }
        }
        free(raw);
        free(shuf);
    }

    free(src);
    if (any_fail) {
        printf("test_lz_opt_roundtrip: FAIL\n");
        return 1;
    }
    printf("test_lz_opt_roundtrip: OK\n");
    return 0;
}
