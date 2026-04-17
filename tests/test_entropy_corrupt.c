/*
 * tests/test_entropy_corrupt.c
 *
 * Corrupt-bitstream harness for every entropy decoder.
 *
 * For each backend in {LZ, LZ_OPT, LZ_STREAMS, HUFFMAN, HUFFMAN4, FSE,
 * LANE}: produce a valid encoded buffer, then systematically wreck it
 * in three ways:
 *
 *   1. Bit flips: flip one random bit at each of 100 random positions.
 *   2. Truncation: truncate at 10 random lengths.
 *   3. Windowed zero-out: zero an 8-byte window at a random offset.
 *
 * Expectation: the decoder must either succeed with identical bytes
 * (for near-no-op corruptions the data section may happen to still be
 * valid — that's fine) or return a non-crash error status. It must
 * NEVER segfault, out-of-bounds read, or infinite-loop.
 *
 * Infinite-loop protection isn't done with signals on Windows; the
 * decoders are expected to have internal bounds and return in finite
 * time. If a specific backend hangs, that IS a bug — the test will
 * time out in ctest and the responsible backend will be identified by
 * the last "TESTING <name>" line that was printed.
 *
 * Note: this test INTENTIONALLY does not validate the decode output
 * when the status is TDC_OK — a bit flip in an LZ match distance can
 * still yield a runnable stream that emits "valid-looking" garbage.
 * The deliverable is "no crashes" — wrong bytes on corrupt input are
 * acceptable as long as the program does not read/write out of bounds.
 */

#include "tdc/entropy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern const tdc_entropy_vt tdc_entropy_lz_vt;
extern const tdc_entropy_vt tdc_entropy_lz_opt_vt;
extern const tdc_entropy_vt tdc_entropy_lz_streams_vt;
extern const tdc_entropy_vt tdc_entropy_huffman_vt;
extern const tdc_entropy_vt tdc_entropy_huffman4_vt;
extern const tdc_entropy_vt tdc_entropy_fse_vt;
extern const tdc_entropy_vt tdc_entropy_lane_vt;

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

/* ---- Deterministic LCG -------------------------------------------------- */

static uint32_t lcg_next(uint32_t *s) {
    *s = (*s) * 1664525u + 1013904223u;
    return *s;
}

/* ---- Input generator ---------------------------------------------------- */

/* Mixed LCG + periodic regions so the encoded stream contains both
 * literal and match sections — exercises more decoder branches. */
static void fill_mixed(uint8_t *buf, size_t n, uint32_t seed) {
    uint32_t s = seed;
    static const uint8_t pattern[16] = {
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
        0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF
    };
    for (size_t i = 0; i < n; i++) {
        if ((i / 256) & 1) {
            buf[i] = pattern[i & 15];
        } else {
            s = s * 1664525u + 1013904223u;
            buf[i] = (uint8_t)(s >> 24);
        }
    }
}

/* ---- Per-backend descriptor --------------------------------------------- */

typedef struct {
    const char             *name;
    const tdc_entropy_vt   *vt;
    const void             *encode_params; /* usually NULL */
    size_t                  src_size;      /* input size to feed encoder */
} backend_t;

/* Backend-specific input size constraints:
 *  - LANE requires src_size % n_lanes == 0. We use 8 lanes x 8192 = 64 KiB.
 *  - HUFFMAN4 switches to single-stream < 256 bytes; use ~64 KiB to keep it
 *    in 4-stream mode.
 *  - All others accept any size. 64 KiB is a good default (large enough for
 *    meaningful bitstream, small enough to fuzz 100 positions quickly).
 */
#define DEFAULT_SIZE (64u * 1024u)

static tdc_lane_entropy_params g_lane_params = {
    .n_lanes = 8,
    .lane_entropy = { TDC_ENTROPY_NONE, TDC_ENTROPY_NONE, TDC_ENTROPY_NONE,
                      TDC_ENTROPY_NONE, TDC_ENTROPY_NONE, TDC_ENTROPY_NONE,
                      TDC_ENTROPY_NONE, TDC_ENTROPY_NONE }
    /* NONE in lane params = AUTO, resolved internally. */
};

static backend_t k_backends[] = {
    { "LZ",         NULL, NULL, DEFAULT_SIZE },
    { "LZ_OPT",     NULL, NULL, DEFAULT_SIZE },
    { "LZ_STREAMS", NULL, NULL, DEFAULT_SIZE },
    { "HUFFMAN",    NULL, NULL, DEFAULT_SIZE },
    { "HUFFMAN4",   NULL, NULL, DEFAULT_SIZE },
    { "FSE",        NULL, NULL, DEFAULT_SIZE },
    { "LANE",       NULL, &g_lane_params, DEFAULT_SIZE },
};
#define N_BACKENDS (sizeof(k_backends) / sizeof(k_backends[0]))

static void wire_backends(void) {
    k_backends[0].vt = &tdc_entropy_lz_vt;
    k_backends[1].vt = &tdc_entropy_lz_opt_vt;
    k_backends[2].vt = &tdc_entropy_lz_streams_vt;
    k_backends[3].vt = &tdc_entropy_huffman_vt;
    k_backends[4].vt = &tdc_entropy_huffman4_vt;
    k_backends[5].vt = &tdc_entropy_fse_vt;
    k_backends[6].vt = &tdc_entropy_lane_vt;
}

/* ---- Corruption strategies --------------------------------------------- */

/* Run decode on corrupt buffer. Expectation: no crash. Status can be
 * OK or any error. Returns 0 on "well-behaved" (no crash), 1 on a
 * status that is clearly wrong (e.g. the function walked past
 * boundaries in a detectable way — we can't detect that here without
 * ASAN, so we just check the return type is a valid tdc_status). */
static int try_decode(const tdc_entropy_vt *vt,
                      const uint8_t *corrupt_src, size_t corrupt_src_size,
                      uint8_t *dst, size_t dst_size,
                      const char *label, int *n_crashes, int *n_ok) {
    (void)label; (void)n_crashes;
    /* Scramble dst so any OOB write leaves a trace, though we can't
     * detect it without ASAN. */
    memset(dst, 0xE7, dst_size);
    tdc_status st = vt->decode(corrupt_src, corrupt_src_size, dst, dst_size);
    /* Accept ANY status as long as it's in the enum range. A status
     * outside the declared range suggests memory corruption. */
    if (st < 0 || st > TDC_E_IO) {
        fprintf(stderr, "FAIL: decoder returned invalid status %d\n", (int)st);
        return 1;
    }
    if (st == TDC_OK) (*n_ok)++;
    return 0;
}

/* Run the three corruption strategies against one backend.
 * Returns 0 on success, 1 if a decoder returned an out-of-range status. */
static int run_backend(backend_t *bk, uint32_t *lcg) {
    printf("  TESTING %-12s ", bk->name);
    fflush(stdout);

    /* Produce a clean encoded buffer. */
    uint8_t *src = (uint8_t *)malloc(bk->src_size);
    if (!src) { fprintf(stderr, "alloc failed\n"); return 1; }
    fill_mixed(src, bk->src_size, 0xC0FFEE00u + (uint32_t)(uintptr_t)bk);

    tdc_buffer enc = make_buffer();
    tdc_status st = bk->vt->encode(src, bk->src_size, bk->encode_params, &enc);
    if (st != TDC_OK) {
        fprintf(stderr, "encode(%s) -> %d\n", bk->name, (int)st);
        free(src); free_buffer(&enc); return 1;
    }
    if (enc.size == 0) {
        fprintf(stderr, "encode(%s) produced 0 bytes\n", bk->name);
        free(src); free_buffer(&enc); return 1;
    }

    /* Decode destination. The decoder is told dst_size == bk->src_size;
     * on corrupt input it must respect this bound. */
    uint8_t *dst = (uint8_t *)malloc(bk->src_size);
    if (!dst) { free(src); free_buffer(&enc); return 1; }

    /* Working copy of the encoded buffer so we can mutate and restore. */
    uint8_t *work = (uint8_t *)malloc(enc.size);
    if (!work) { free(dst); free(src); free_buffer(&enc); return 1; }

    int failures = 0;
    int n_ok = 0, n_crashes = 0;

    /* 1. Bit flips — 100 positions. */
    for (int it = 0; it < 100; ++it) {
        memcpy(work, enc.data, enc.size);
        uint32_t r = lcg_next(lcg);
        size_t pos = (size_t)(r % enc.size);
        uint8_t bit = (uint8_t)(lcg_next(lcg) & 7);
        work[pos] ^= (uint8_t)(1u << bit);
        if (try_decode(bk->vt, work, enc.size, dst, bk->src_size,
                       bk->name, &n_crashes, &n_ok)) {
            failures++;
        }
    }

    /* 2. Truncation — 10 random lengths in [1, enc.size-1]. */
    for (int it = 0; it < 10; ++it) {
        uint32_t r = lcg_next(lcg);
        size_t trunc_len = (size_t)(r % enc.size);
        if (trunc_len == 0) trunc_len = 1;
        if (try_decode(bk->vt, enc.data, trunc_len, dst, bk->src_size,
                       bk->name, &n_crashes, &n_ok)) {
            failures++;
        }
    }

    /* 3. Windowed zero-out — 10 random 8-byte windows. */
    for (int it = 0; it < 10; ++it) {
        memcpy(work, enc.data, enc.size);
        uint32_t r = lcg_next(lcg);
        size_t pos = (size_t)(r % enc.size);
        size_t win = 8;
        if (pos + win > enc.size) win = enc.size - pos;
        memset(work + pos, 0, win);
        if (try_decode(bk->vt, work, enc.size, dst, bk->src_size,
                       bk->name, &n_crashes, &n_ok)) {
            failures++;
        }
    }

    printf("(%3d corrupt trials; %2d decoded OK; %2d invalid-status)\n",
           100 + 10 + 10, n_ok, failures);

    free(work); free(dst); free(src); free_buffer(&enc);
    return (failures == 0) ? 0 : 1;
}

int main(void) {
    wire_backends();

    printf("test_entropy_corrupt: %zu backends\n", N_BACKENDS);

    uint32_t lcg = 0xBADF00D0u; /* reproducible */
    int any_fail = 0;
    for (size_t i = 0; i < N_BACKENDS; ++i) {
        if (run_backend(&k_backends[i], &lcg)) any_fail = 1;
    }

    if (any_fail) {
        printf("test_entropy_corrupt: FAIL\n");
        return 1;
    }
    printf("test_entropy_corrupt: OK\n");
    return 0;
}
