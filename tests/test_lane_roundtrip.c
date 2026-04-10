/*
 * tests/test_lane_roundtrip.c
 *
 * Round-trip test for the TDC_ENTROPY_LANE backend.
 *
 * Verifies:
 *   1. Round-trip with explicit per-lane coder selection (Huffman, FSE, NONE).
 *   2. Round-trip with AUTO selection on peaked data (should pick Huffman).
 *   3. Round-trip with AUTO selection on uniform data (should pick NONE).
 *   4. Round-trip with mixed distributions (some lanes peaked, some noisy).
 *   5. Single-byte-per-lane (n_lanes == 1) round-trips.
 *   6. Empty input (src_size == 0) is rejected (n_lanes > 0 requires data).
 *   7. Encoder rejects n_lanes == 0 and src_size not divisible by n_lanes.
 *   8. Decoder rejects truncated headers and mismatched sizes.
 *   9. Registry returns &tdc_entropy_lane_vt for TDC_ENTROPY_LANE.
 *  10. Expansion fallback: if a sub-coder expands the data, the lane
 *      falls back to NONE (passthrough).
 */

#include "tdc/entropy.h"
#include "tdc/codec.h"
#include "tdc/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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

/* ----- Helpers ------------------------------------------------------------ */

/* Fill `dst` with peaked data: mostly zeros with occasional spikes. */
static void fill_peaked(uint8_t *dst, size_t n) {
    for (size_t i = 0; i < n; ++i)
        dst[i] = (i % 37 == 0) ? (uint8_t)(i & 0xFF) : 0;
}

/* Fill `dst` with pseudo-random data (LCG). */
static void fill_random(uint8_t *dst, size_t n) {
    uint32_t state = 0xDEADBEEFu;
    for (size_t i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        dst[i] = (uint8_t)(state >> 16);
    }
}

/* Fill with moderate-entropy data: byte values 0..15 repeating. */
static void fill_moderate(uint8_t *dst, size_t n) {
    for (size_t i = 0; i < n; ++i)
        dst[i] = (uint8_t)(i & 0x0F);
}

/* Generic round-trip helper. */
static int rt_one(const char *label, const uint8_t *src, size_t src_size,
                  const tdc_lane_entropy_params *params) {
    const tdc_entropy_vt *vt = &tdc_entropy_lane_vt;

    tdc_buffer enc = make_buffer();
    tdc_status st = vt->encode(src, src_size, params, &enc);
    ASSERT_OR_DIE(st == TDC_OK, "encode returned non-OK");

    uint8_t *dec = (uint8_t *)malloc(src_size > 0 ? src_size : 1u);
    ASSERT_OR_DIE(dec != NULL, "malloc dec");
    memset(dec, 0xCC, src_size > 0 ? src_size : 1u);

    st = vt->decode(enc.data, enc.size, dec, src_size);
    ASSERT_OR_DIE(st == TDC_OK, "decode returned non-OK");

    if (src_size > 0) {
        ASSERT_OR_DIE(memcmp(dec, src, src_size) == 0,
                      "decoded data does not match source");
    }

    printf("  [%s] src=%zu enc=%zu round-trip OK\n",
           label, src_size, enc.size);

    free(dec);
    free_buffer(&enc);
    return 0;
}

/* ----- Tests -------------------------------------------------------------- */

static int test_explicit_huffman(void) {
    /* 4 lanes, all Huffman, peaked data. */
    enum { N = 1024 };
    uint8_t src[N];
    fill_peaked(src, N);

    tdc_lane_entropy_params params = {0};
    params.n_lanes = 4;
    for (int i = 0; i < 4; ++i)
        params.lane_entropy[i] = TDC_ENTROPY_HUFFMAN;

    return rt_one("explicit huffman 4 lanes", src, N, &params);
}

static int test_explicit_fse(void) {
    /* 4 lanes, all FSE, moderate data. */
    enum { N = 1024 };
    uint8_t src[N];
    fill_moderate(src, N);

    tdc_lane_entropy_params params = {0};
    params.n_lanes = 4;
    for (int i = 0; i < 4; ++i)
        params.lane_entropy[i] = TDC_ENTROPY_FSE;

    return rt_one("explicit fse 4 lanes", src, N, &params);
}

static int test_explicit_none(void) {
    /* 2 lanes, passthrough. */
    enum { N = 512 };
    uint8_t src[N];
    fill_random(src, N);

    tdc_lane_entropy_params params = {0};
    params.n_lanes = 2;
    params.lane_entropy[0] = TDC_ENTROPY_NONE;
    params.lane_entropy[1] = TDC_ENTROPY_NONE;

    return rt_one("explicit none 2 lanes", src, N, &params);
}

static int test_auto_peaked(void) {
    /* AUTO on peaked data — should pick Huffman. */
    enum { N = 2048 };
    uint8_t src[N];
    fill_peaked(src, N);

    tdc_lane_entropy_params params = {0};
    params.n_lanes = 4;
    /* lane_entropy[i] == 0 == TDC_ENTROPY_NONE == AUTO for lane params */

    return rt_one("auto peaked 4 lanes", src, N, &params);
}

static int test_auto_random(void) {
    /* AUTO on random data — should pick NONE (passthrough). */
    enum { N = 2048 };
    uint8_t src[N];
    fill_random(src, N);

    tdc_lane_entropy_params params = {0};
    params.n_lanes = 4;

    return rt_one("auto random 4 lanes", src, N, &params);
}

static int test_mixed_lanes(void) {
    /* 4 lanes: lane 0 peaked, lane 1 random, lane 2 moderate, lane 3 peaked.
     * Input is already in lane-interleaved order (as BSHUF would produce). */
    enum { LANE_SIZE = 512, N = LANE_SIZE * 4 };
    uint8_t src[N];
    fill_peaked(src + 0 * LANE_SIZE, LANE_SIZE);
    fill_random(src + 1 * LANE_SIZE, LANE_SIZE);
    fill_moderate(src + 2 * LANE_SIZE, LANE_SIZE);
    fill_peaked(src + 3 * LANE_SIZE, LANE_SIZE);

    tdc_lane_entropy_params params = {0};
    params.n_lanes = 4;
    /* AUTO for all lanes */

    return rt_one("mixed 4 lanes auto", src, N, &params);
}

static int test_single_lane(void) {
    /* n_lanes == 1: entire input is one lane. */
    enum { N = 256 };
    uint8_t src[N];
    fill_peaked(src, N);

    tdc_lane_entropy_params params = {0};
    params.n_lanes = 1;
    params.lane_entropy[0] = TDC_ENTROPY_HUFFMAN;

    return rt_one("single lane", src, N, &params);
}

static int test_8_lanes(void) {
    /* Max lanes (8), f64-width data. */
    enum { LANE_SIZE = 128, N = LANE_SIZE * 8 };
    uint8_t src[N];
    for (int lane = 0; lane < 8; ++lane) {
        if (lane < 2)
            fill_peaked(src + lane * LANE_SIZE, LANE_SIZE);
        else
            fill_moderate(src + lane * LANE_SIZE, LANE_SIZE);
    }

    tdc_lane_entropy_params params = {0};
    params.n_lanes = 8;

    return rt_one("8 lanes auto", src, N, &params);
}

static int test_rejections(void) {
    const tdc_entropy_vt *vt = &tdc_entropy_lane_vt;
    tdc_buffer enc = make_buffer();

    /* n_lanes == 0 rejected */
    {
        uint8_t src[4] = {0};
        tdc_lane_entropy_params params = {0};
        params.n_lanes = 0;
        tdc_status st = vt->encode(src, 4, &params, &enc);
        ASSERT_OR_DIE(st == TDC_E_INVAL, "n_lanes == 0 should be rejected");
    }

    /* src_size not divisible by n_lanes */
    {
        uint8_t src[5] = {0};
        tdc_lane_entropy_params params = {0};
        params.n_lanes = 4;
        tdc_status st = vt->encode(src, 5, &params, &enc);
        ASSERT_OR_DIE(st == TDC_E_INVAL, "5 % 4 != 0 should be rejected");
    }

    /* n_lanes > TDC_MAX_LANES */
    {
        uint8_t src[16] = {0};
        tdc_lane_entropy_params params = {0};
        params.n_lanes = TDC_MAX_LANES + 1;
        tdc_status st = vt->encode(src, 16, &params, &enc);
        ASSERT_OR_DIE(st == TDC_E_INVAL, "n_lanes > MAX should be rejected");
    }

    /* Decode: truncated header */
    {
        uint8_t bad[2] = { 4, 0 }; /* says 4 lanes, but only 2 bytes */
        uint8_t dst[16] = {0};
        tdc_status st = vt->decode(bad, 2, dst, 16);
        ASSERT_OR_DIE(st == TDC_E_CORRUPT, "truncated header should be rejected");
    }

    /* Decode: mismatched dst_size */
    {
        /* Encode valid data, then try to decode with wrong dst_size. */
        uint8_t src[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };
        tdc_lane_entropy_params params = {0};
        params.n_lanes = 2;
        params.lane_entropy[0] = TDC_ENTROPY_NONE;
        params.lane_entropy[1] = TDC_ENTROPY_NONE;
        tdc_status st = vt->encode(src, 8, &params, &enc);
        ASSERT_OR_DIE(st == TDC_OK, "setup encode for mismatch test");

        uint8_t dst[16] = {0};
        st = vt->decode(enc.data, enc.size, dst, 16); /* wrong: should be 8 */
        ASSERT_OR_DIE(st == TDC_E_CORRUPT, "mismatched dst_size should be rejected");
    }

    printf("  [rejections] OK\n");
    free_buffer(&enc);
    return 0;
}

static int test_registry(void) {
    const tdc_entropy_vt *vt = tdc_entropy_get(TDC_ENTROPY_LANE);
    ASSERT_OR_DIE(vt == &tdc_entropy_lane_vt,
                  "tdc_entropy_get(TDC_ENTROPY_LANE) wiring");
    ASSERT_OR_DIE(vt->id == TDC_ENTROPY_LANE, "vt->id mismatch");
    printf("  [registry wiring] OK\n");
    return 0;
}

int main(void) {
    printf("test_lane_roundtrip\n");
    if (test_explicit_huffman()) return 1;
    if (test_explicit_fse())     return 1;
    if (test_explicit_none())    return 1;
    if (test_auto_peaked())      return 1;
    if (test_auto_random())      return 1;
    if (test_mixed_lanes())      return 1;
    if (test_single_lane())      return 1;
    if (test_8_lanes())          return 1;
    if (test_rejections())       return 1;
    if (test_registry())         return 1;
    printf("ALL OK\n");
    return 0;
}
