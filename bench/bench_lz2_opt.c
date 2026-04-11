/*
 * bench/bench_lz2_opt.c
 *
 * Side-by-side throughput + ratio comparison for TDC_ENTROPY_LZ2 (greedy)
 * vs TDC_ENTROPY_LZ2_OPT (forward-DP optimal parser) on the entropy stage
 * in isolation. Drives the entropy vtables directly with raw byte buffers
 * — no model or transform in front — so the numbers reflect pure encode
 * algorithm differences on content that mimics what the tdc pipeline
 * actually feeds the entropy stage.
 *
 * Fixtures:
 *   - Byte-shuffled int64 ramp     (the structured-tabular target workload)
 *   - Byte-shuffled double ramp    (structured float workload)
 *   - LCG literal-heavy noise      (incompressible worst case)
 *   - Short periodic pattern       (match-heavy best case)
 *   - Mixed (LCG + periodic halves)
 *
 * Invariant: opt_size <= greedy_size on every fixture. Printed as a
 * "saving" column (positive = opt wins).
 */

#include "tdc/entropy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../src/core/timer.h"

extern const tdc_entropy_vt tdc_entropy_lz2_vt;
extern const tdc_entropy_vt tdc_entropy_lz2_opt_vt;

#define ITERS_ENC_GREEDY 5
#define ITERS_ENC_OPT    3   /* opt is ~5-10x slower; fewer iters keeps runtime sane */
#define ITERS_DEC        5

static void *bench_realloc(void *user, void *ptr, size_t new_size) {
    (void)user;
    if (new_size == 0) { free(ptr); return NULL; }
    return realloc(ptr, new_size);
}

static tdc_buffer make_buffer(void) {
    tdc_buffer b = {0};
    b.realloc_fn = bench_realloc;
    return b;
}

static void free_buffer(tdc_buffer *b) {
    if (b->data) free(b->data);
    b->data = NULL; b->size = 0; b->capacity = 0;
}

/* ----- Fixtures ---------------------------------------------------------- */

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

/* Byte-shuffled int64 ramp: 8 lanes, lane 0 = LSB. Mimics what vectra's
 * SHUFFLE_LZ2 pipeline feeds the entropy stage on structured tabular data. */
static void fill_shuffled_i64_ramp(uint8_t *buf, size_t n) {
    size_t elems = n / 8;
    for (size_t e = 0; e < elems; e++) {
        int64_t v = (int64_t)(1000 + (int64_t)e * 3);
        uint64_t u;
        memcpy(&u, &v, 8);
        for (int lane = 0; lane < 8; lane++) {
            buf[(size_t)lane * elems + e] = (uint8_t)(u >> (lane * 8));
        }
    }
    for (size_t i = elems * 8; i < n; i++) buf[i] = 0;
}

/* Byte-shuffled double ramp: 8 lanes, low mantissa bits vary, high bits
 * near-constant. Structurally similar to the i64 ramp but with a
 * different lane-occupancy profile. */
static void fill_shuffled_f64_ramp(uint8_t *buf, size_t n) {
    size_t elems = n / 8;
    for (size_t e = 0; e < elems; e++) {
        double v = 1000.0 + (double)e * 0.25;
        uint64_t u;
        memcpy(&u, &v, 8);
        for (int lane = 0; lane < 8; lane++) {
            buf[(size_t)lane * elems + e] = (uint8_t)(u >> (lane * 8));
        }
    }
    for (size_t i = elems * 8; i < n; i++) buf[i] = 0;
}

/* ----- Timed encode / decode helpers ------------------------------------- */

static int bench_encode(const tdc_entropy_vt *vt, const uint8_t *src, size_t n,
                        int iters, tdc_buffer *out, double *best_secs) {
    double best = 1e18;
    for (int i = 0; i < iters; i++) {
        out->size = 0;
        double t0 = tdc_now_secs();
        tdc_status st = vt->encode(src, n, NULL, out);
        double t1 = tdc_now_secs();
        if (st != TDC_OK) return (int)st;
        double dt = t1 - t0;
        if (dt < best) best = dt;
    }
    *best_secs = best;
    return 0;
}

static int bench_decode(const tdc_entropy_vt *vt, const uint8_t *enc_data, size_t enc_size,
                        uint8_t *dst, size_t dst_size, int iters, double *best_secs) {
    double best = 1e18;
    for (int i = 0; i < iters; i++) {
        double t0 = tdc_now_secs();
        tdc_status st = vt->decode(enc_data, enc_size, dst, dst_size);
        double t1 = tdc_now_secs();
        if (st != TDC_OK) return (int)st;
        double dt = t1 - t0;
        if (dt < best) best = dt;
    }
    *best_secs = best;
    return 0;
}

/* ----- Case runner ------------------------------------------------------- */

static int run_case(const char *label, const uint8_t *src, size_t n) {
    tdc_buffer g = make_buffer();
    tdc_buffer o = make_buffer();

    double g_enc_secs = 0.0, o_enc_secs = 0.0;
    double g_dec_secs = 0.0, o_dec_secs = 0.0;

    int rc;
    rc = bench_encode(&tdc_entropy_lz2_vt,     src, n, ITERS_ENC_GREEDY, &g, &g_enc_secs);
    if (rc) { fprintf(stderr, "FAIL [%s]: greedy encode -> %d\n", label, rc); free_buffer(&g); free_buffer(&o); return 1; }
    rc = bench_encode(&tdc_entropy_lz2_opt_vt, src, n, ITERS_ENC_OPT,    &o, &o_enc_secs);
    if (rc) { fprintf(stderr, "FAIL [%s]: opt encode -> %d\n", label, rc); free_buffer(&g); free_buffer(&o); return 1; }

    /* Round-trip through the shared decoder. Also times decode for each
     * stream (same decoder, but the stream shape may affect cache behaviour). */
    uint8_t *dec = (uint8_t *)malloc(n ? n : 1);
    if (!dec) { free_buffer(&g); free_buffer(&o); return 1; }

    rc = bench_decode(&tdc_entropy_lz2_vt, g.data, g.size, dec, n, ITERS_DEC, &g_dec_secs);
    if (rc || memcmp(dec, src, n) != 0) {
        fprintf(stderr, "FAIL [%s]: greedy decode/round-trip\n", label);
        free(dec); free_buffer(&g); free_buffer(&o); return 1;
    }
    rc = bench_decode(&tdc_entropy_lz2_vt, o.data, o.size, dec, n, ITERS_DEC, &o_dec_secs);
    if (rc || memcmp(dec, src, n) != 0) {
        fprintf(stderr, "FAIL [%s]: opt decode/round-trip\n", label);
        free(dec); free_buffer(&g); free_buffer(&o); return 1;
    }

    double mib = (double)n / (1024.0 * 1024.0);
    double g_ratio = n ? (double)n / (double)g.size : 0.0;
    double o_ratio = n ? (double)n / (double)o.size : 0.0;
    double saving  = g.size ? 100.0 * (1.0 - (double)o.size / (double)g.size) : 0.0;
    double g_enc_mbps = mib / g_enc_secs;
    double o_enc_mbps = mib / o_enc_secs;
    double g_dec_mbps = mib / g_dec_secs;
    double o_dec_mbps = mib / o_dec_secs;
    double enc_slowdown = o_enc_secs / g_enc_secs;

    printf("%-28s %7.2f  %8zu %6.2fx  %8zu %6.2fx  %+6.1f%%  %8.1f %8.1f %6.1fx  %8.1f %8.1f\n",
           label, mib,
           g.size, g_ratio,
           o.size, o_ratio,
           saving,
           g_enc_mbps, o_enc_mbps, enc_slowdown,
           g_dec_mbps, o_dec_mbps);

    int invariant_ok = o.size <= g.size;
    if (!invariant_ok) {
        fprintf(stderr, "  INVARIANT FAIL: opt (%zu) > greedy (%zu)\n", o.size, g.size);
    }

    free(dec);
    free_buffer(&g);
    free_buffer(&o);
    return invariant_ok ? 0 : 2;
}

/* ----- Main -------------------------------------------------------------- */

int main(void) {
    const size_t N = 1u << 20; /* 1 MiB per fixture */
    uint8_t *src = (uint8_t *)malloc(N);
    if (!src) { fprintf(stderr, "alloc failed\n"); return 1; }

    printf("\n");
    printf("bench_lz2_opt: greedy vs forward-DP optimal LZ2 (entropy stage, isolated)\n");
    printf("               1 MiB fixtures, encode best-of-%d/%d, decode best-of-%d\n\n",
           ITERS_ENC_GREEDY, ITERS_ENC_OPT, ITERS_DEC);

    printf("%-28s %7s  %8s %7s  %8s %7s   %6s  %8s %8s %6s  %8s %8s\n",
           "fixture", "MiB",
           "g_size", "g_ratio",
           "o_size", "o_ratio",
           "saving",
           "g_enc",  "o_enc",  "slow",
           "g_dec",  "o_dec");
    printf("%-28s %7s  %8s %7s  %8s %7s   %6s  %8s %8s %6s  %8s %8s\n",
           "----------------------------", "-------",
           "--------", "-------",
           "--------", "-------",
           "------",
           "--------", "--------", "------",
           "--------", "--------");

    int any_fail = 0;

    /* 1. Byte-shuffled int64 ramp — the structured-tabular target. */
    fill_shuffled_i64_ramp(src, N);
    if (run_case("shuffled i64 ramp 1MiB", src, N)) any_fail = 1;

    /* 2. Byte-shuffled double ramp. */
    fill_shuffled_f64_ramp(src, N);
    if (run_case("shuffled f64 ramp 1MiB", src, N)) any_fail = 1;

    /* 3. Periodic — pure match-heavy. */
    fill_periodic(src, N);
    if (run_case("periodic 1MiB",          src, N)) any_fail = 1;

    /* 4. LCG noise — pure literal-heavy (essentially incompressible). */
    fill_lcg(src, N, 0xDEADBEEFu);
    if (run_case("LCG literal 1MiB",       src, N)) any_fail = 1;

    /* 5. Mixed — half LCG, half periodic. */
    fill_lcg(src, N / 2, 0xC0FFEEu);
    fill_periodic(src + N / 2, N / 2);
    if (run_case("mixed LCG+periodic 1MiB", src, N)) any_fail = 1;

    free(src);
    printf("\n");
    if (any_fail) {
        printf("bench_lz2_opt: FAIL (invariant or round-trip)\n");
        return 1;
    }
    printf("bench_lz2_opt: OK\n");
    return 0;
}
