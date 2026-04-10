/*
 * bench/bench_pred3d_avg3.c
 *
 * One-off comparison: AVG3 with division-by-3-always vs division-by-
 * in-bounds-count, using the integrated branchy loop shape (i.e. before
 * Tier 3 octant prologues are applied). Picks the faster variant for
 * pred3d.c and feeds the numbers into notes.md.
 *
 * NOT part of the regular bench. Compile manually:
 *   cl /O2 /std:c11 bench/bench_pred3d_avg3.c
 * or build via the temporary CMake target if added.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../src/core/timer.h"
#define now_seconds tdc_now_secs

/* Variant A: divide by in-bounds neighbor count (current pred3d.c). */
static void avg3_count_u16(const uint16_t *src, uint16_t *res,
                           int64_t nx, int64_t ny, int64_t nz) {
    const int64_t slab = nx * ny;
    for (int64_t z = 0; z < nz; ++z) {
        for (int64_t y = 0; y < ny; ++y) {
            const uint16_t *s = src + z * slab + y * nx;
            uint16_t       *r = res + z * slab + y * nx;
            for (int64_t x = 0; x < nx; ++x) {
                int32_t val = (int32_t)s[x];
                int32_t sum = 0; int cnt = 0;
                if (x > 0) { sum += (int32_t)s[x - 1];    cnt++; }
                if (y > 0) { sum += (int32_t)s[x - nx];   cnt++; }
                if (z > 0) { sum += (int32_t)s[x - slab]; cnt++; }
                int32_t pred = cnt ? sum / (int32_t)cnt : 0;
                r[x] = (uint16_t)(val - pred);
            }
        }
    }
}

/* Variant B: always divide by 3 (treats out-of-bounds as 0). */
static void avg3_three_u16(const uint16_t *src, uint16_t *res,
                           int64_t nx, int64_t ny, int64_t nz) {
    const int64_t slab = nx * ny;
    for (int64_t z = 0; z < nz; ++z) {
        for (int64_t y = 0; y < ny; ++y) {
            const uint16_t *s = src + z * slab + y * nx;
            uint16_t       *r = res + z * slab + y * nx;
            for (int64_t x = 0; x < nx; ++x) {
                int32_t val = (int32_t)s[x];
                int32_t sum = 0;
                if (x > 0) sum += (int32_t)s[x - 1];
                if (y > 0) sum += (int32_t)s[x - nx];
                if (z > 0) sum += (int32_t)s[x - slab];
                int32_t pred = sum / 3;
                r[x] = (uint16_t)(val - pred);
            }
        }
    }
}

/* Variant C: per-octant prologues (Tier 3 inner box only — for the
 * O8 hot loop, cnt is the compile-time constant 3 and the divide is
 * folded by the compiler into a multiply-and-shift). This is the
 * version Tier 3 will install. Approximated here by running the
 * inner box only (the boundary is O(n^2), negligible). */
static void avg3_inner_u16(const uint16_t *src, uint16_t *res,
                           int64_t nx, int64_t ny, int64_t nz) {
    const int64_t slab = nx * ny;
    /* Skip the boundary entirely — this is a microbench of the inner
     * box hot loop only. Real Tier 3 also handles boundaries; those
     * are O(n^2) and don't affect throughput at 256^3. */
    for (int64_t z = 1; z < nz; ++z) {
        for (int64_t y = 1; y < ny; ++y) {
            const uint16_t *s  = src + z * slab + y       * nx;
            const uint16_t *sy = src + z * slab + (y - 1) * nx;
            const uint16_t *sz = src + (z - 1) * slab + y * nx;
            uint16_t       *r  = res + z * slab + y       * nx;
            for (int64_t x = 1; x < nx; ++x) {
                int32_t val = (int32_t)s[x];
                int32_t a   = (int32_t)s[x - 1];
                int32_t b   = (int32_t)sy[x];
                int32_t c   = (int32_t)sz[x];
                int32_t pred = (a + b + c) / 3;
                r[x] = (uint16_t)(val - pred);
            }
        }
    }
}

#define ITERS 5

int main(void) {
    const int64_t nx = 256, ny = 256, nz = 256;
    const int64_t n = nx * ny * nz;
    const size_t bytes = (size_t)n * sizeof(uint16_t);

    uint16_t *src = (uint16_t *)malloc(bytes);
    uint16_t *res = (uint16_t *)malloc(bytes);
    if (!src || !res) { fprintf(stderr, "alloc fail\n"); return 1; }

    /* Smooth gradient + small noise — representative of the kind of
     * data AVG3 is good for. */
    for (int64_t z = 0; z < nz; ++z)
        for (int64_t y = 0; y < ny; ++y)
            for (int64_t x = 0; x < nx; ++x)
                src[(z * ny + y) * nx + x] = (uint16_t)(z * 5 + y * 3 + x * 2 + ((x * y * z) & 7));

    /* Warm up. */
    avg3_count_u16(src, res, nx, ny, nz);
    avg3_three_u16(src, res, nx, ny, nz);
    avg3_inner_u16(src, res, nx, ny, nz);

    double best_count = 1e9, best_three = 1e9, best_inner = 1e9;
    for (int i = 0; i < ITERS; ++i) {
        double t0 = now_seconds();
        avg3_count_u16(src, res, nx, ny, nz);
        double dt = now_seconds() - t0;
        if (dt < best_count) best_count = dt;
    }
    for (int i = 0; i < ITERS; ++i) {
        double t0 = now_seconds();
        avg3_three_u16(src, res, nx, ny, nz);
        double dt = now_seconds() - t0;
        if (dt < best_three) best_three = dt;
    }
    for (int i = 0; i < ITERS; ++i) {
        double t0 = now_seconds();
        avg3_inner_u16(src, res, nx, ny, nz);
        double dt = now_seconds() - t0;
        if (dt < best_inner) best_inner = dt;
    }

    double mb = (double)bytes / (1024.0 * 1024.0);
    printf("AVG3 u16 256x256x256 = %.1f MiB raw\n", mb);
    printf("  /count (branchy):     %7.2f ms  %8.1f MB/s\n",
           best_count * 1000.0, mb / best_count);
    printf("  /3     (branchy):     %7.2f ms  %8.1f MB/s\n",
           best_three * 1000.0, mb / best_three);
    printf("  Tier3 inner box only: %7.2f ms  %8.1f MB/s\n",
           best_inner * 1000.0, mb / best_inner);

    free(src);
    free(res);
    return 0;
}
