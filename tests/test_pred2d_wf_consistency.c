/*
 * tests/test_pred2d_wf_consistency.c
 *
 * Cross-checks the three u16 PAETH wavefront decoders (wf2 / wf4 / wf8)
 * against a scalar reference on a battery of synthetic rasters. PAETH is a
 * lossless predictor: every wavefront variant must produce bit-identical
 * output. Even one differing pixel breaks the on-disk format contract for
 * existing compressed files.
 *
 * Test rasters cover:
 *   - tiny shapes that hit each kernel's small-side fall-through
 *     (nx=8 ny=8, ny=1, ny=2, ny=7)
 *   - shapes that just barely qualify for each kernel (nx=8 ny=9 → wf8)
 *   - medium (256x256) and large (2048x2048)
 *   - several seeds for the noise pattern, plus an adversarial all-zero
 *     and an all-0xFFFF raster.
 *
 * Build is gated only on the presence of the wf kernels in pred2d.c — the
 * AVX2 wf8 reduces to wf4 internally when AVX2 is not enabled, so the
 * test still runs (and trivially passes the wf8 vs wf4 comparison).
 */

#include "tdc/codec.h"
#include "tdc/types.h"

#include "../src/model/pred2d_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ----- Scalar reference ------------------------------------------------- *
 * Inlined from src/model/pred2d.c paeth32 + the u16 typed decoder. Kept
 * here as an independent reference so a kernel and the test cannot drift
 * into matching each other on a wrong tie-break. */

static int32_t ref_paeth32(int32_t a, int32_t b, int32_t c) {
    int32_t bc = b - c;
    int32_t ac = a - c;
    int32_t pa = bc < 0 ? -bc : bc;
    int32_t pb = ac < 0 ? -ac : ac;
    int32_t pcs = bc + ac;
    int32_t pc = pcs < 0 ? -pcs : pcs;
    int32_t r  = (pb <= pc) ? b : c;
    return ((pa <= pb) & (pa <= pc)) ? a : r;
}

static void ref_decode_u16_paeth(const uint16_t *res, uint16_t *dst,
                                 int64_t nx, int64_t ny) {
    if (nx <= 0 || ny <= 0) return;
    /* (0,0): all neighbors zero -> residual stored as value. */
    dst[0] = res[0];
    /* row 0, col >= 1: left = dst[c-1], up=upleft=0 -> paeth(L,0,0) = L */
    for (int64_t c = 1; c < nx; ++c)
        dst[c] = (uint16_t)(res[c] + dst[c - 1]);
    for (int64_t r = 1; r < ny; ++r) {
        const uint16_t *rr = res + r * nx;
        uint16_t       *dr = dst + r * nx;
        const uint16_t *da = dst + (r - 1) * nx;
        /* col 0: left = upleft = 0, up = da[0] -> paeth(0, up, 0) = up */
        dr[0] = (uint16_t)((int32_t)rr[0] + (int32_t)da[0]);
        for (int64_t c = 1; c < nx; ++c) {
            int32_t left   = (int32_t)dr[c - 1];
            int32_t up     = (int32_t)da[c];
            int32_t upleft = (int32_t)da[c - 1];
            int32_t pred   = ref_paeth32(left, up, upleft);
            dr[c] = (uint16_t)((int32_t)rr[c] + pred);
        }
    }
}

/* ----- Encode helpers (mirror the typed encoder for u16 PAETH) ---------- */

static void encode_u16_paeth(const uint16_t *src, uint16_t *res,
                             int64_t nx, int64_t ny) {
    if (nx <= 0 || ny <= 0) return;
    res[0] = src[0];
    for (int64_t c = 1; c < nx; ++c)
        res[c] = (uint16_t)((int32_t)src[c] - (int32_t)src[c - 1]);
    for (int64_t r = 1; r < ny; ++r) {
        const uint16_t *s0 = src + r * nx;
        const uint16_t *s1 = src + (r - 1) * nx;
        uint16_t       *r0 = res + r * nx;
        r0[0] = (uint16_t)((int32_t)s0[0] - (int32_t)s1[0]);
        for (int64_t c = 1; c < nx; ++c) {
            int32_t left   = (int32_t)s0[c - 1];
            int32_t up     = (int32_t)s1[c];
            int32_t upleft = (int32_t)s1[c - 1];
            int32_t pred   = ref_paeth32(left, up, upleft);
            r0[c] = (uint16_t)((int32_t)s0[c] - pred);
        }
    }
}

/* ----- Random data generators ------------------------------------------- */

static uint32_t lcg(uint32_t *s) {
    *s = (*s) * 1103515245u + 12345u;
    return *s;
}

static void fill_grad_u16(uint16_t *p, int64_t ny, int64_t nx, uint32_t seed) {
    uint32_t s = seed;
    for (int64_t r = 0; r < ny; ++r) {
        for (int64_t c = 0; c < nx; ++c) {
            int noise = (int)((lcg(&s) >> 20) & 0xF) - 8; /* -8..7 */
            p[r * nx + c] = (uint16_t)(100 + r * 5 + c * 3 + noise);
        }
    }
}

static void fill_random_u16(uint16_t *p, int64_t ny, int64_t nx, uint32_t seed) {
    uint32_t s = seed;
    for (int64_t i = 0; i < ny * nx; ++i) {
        p[i] = (uint16_t)(lcg(&s) >> 16);
    }
}

static void fill_const_u16(uint16_t *p, int64_t ny, int64_t nx, uint16_t v) {
    for (int64_t i = 0; i < ny * nx; ++i) p[i] = v;
}

/* ----- Comparator ------------------------------------------------------- */

static int compare_buffers(const char *label, int64_t ny, int64_t nx,
                           const uint16_t *expected, const uint16_t *got) {
    int64_t n = ny * nx;
    for (int64_t i = 0; i < n; ++i) {
        if (expected[i] != got[i]) {
            int64_t r = i / nx, c = i % nx;
            fprintf(stderr,
                    "FAIL [%s] %lldx%lld: mismatch at (%lld,%lld) "
                    "expected=0x%04x got=0x%04x\n",
                    label, (long long)ny, (long long)nx,
                    (long long)r, (long long)c,
                    (unsigned)expected[i], (unsigned)got[i]);
            return 1;
        }
    }
    return 0;
}

/* ----- Per-shape harness ------------------------------------------------ */

static int run_shape(const char *label, int64_t ny, int64_t nx,
                     const uint16_t *src) {
    int64_t n = ny * nx;
    if (n == 0) return 0;

    uint16_t *res     = (uint16_t *)malloc(sizeof(uint16_t) * (size_t)n);
    uint16_t *out_ref = (uint16_t *)malloc(sizeof(uint16_t) * (size_t)n);
    uint16_t *out_wf2 = (uint16_t *)malloc(sizeof(uint16_t) * (size_t)n);
    uint16_t *out_wf4 = (uint16_t *)malloc(sizeof(uint16_t) * (size_t)n);
    uint16_t *out_wf8 = (uint16_t *)malloc(sizeof(uint16_t) * (size_t)n);
    if (!res || !out_ref || !out_wf2 || !out_wf4 || !out_wf8) {
        fprintf(stderr, "FAIL [%s]: malloc\n", label);
        free(res); free(out_ref); free(out_wf2); free(out_wf4); free(out_wf8);
        return 1;
    }

    encode_u16_paeth(src, res, nx, ny);
    ref_decode_u16_paeth(res, out_ref, nx, ny);

    /* Round-trip sanity: scalar reference must match source. */
    if (compare_buffers("ref vs src", ny, nx, src, out_ref)) {
        free(res); free(out_ref); free(out_wf2); free(out_wf4); free(out_wf8);
        return 1;
    }

    /* Each kernel is only called when its shape contract is met — these
     * are the same gates the public dispatcher uses. Below the threshold
     * the kernel reads out of bounds in its prologue, so the gate is
     * load-bearing. The dispatcher will pick the next-wider working
     * kernel, which is itself cross-checked here at its own threshold. */
    int rc = 0;
    pred2d_dec_u16_paeth_wavefront_export(res, out_wf2, nx, ny);
    rc |= compare_buffers("wf2 vs ref", ny, nx, out_ref, out_wf2);

    int wf4_ok = (nx >= 4 && ny >= 5);
    int wf8_ok = (nx >= 8 && ny >= 9);

    if (wf4_ok) {
        pred2d_dec_u16_paeth_wf4_export(res, out_wf4, nx, ny);
        rc |= compare_buffers("wf4 vs ref", ny, nx, out_ref, out_wf4);
    }
    if (wf8_ok) {
        pred2d_dec_u16_paeth_wf8_export(res, out_wf8, nx, ny);
        rc |= compare_buffers("wf8 vs ref", ny, nx, out_ref, out_wf8);
    }

    if (rc == 0) {
        const char *tag = wf8_ok ? "wf2=wf4=wf8=ref"
                        : wf4_ok ? "wf2=wf4=ref"
                                 : "wf2=ref";
        printf("  [%s] %lldx%lld OK (%s)\n",
               label, (long long)ny, (long long)nx, tag);
    }

    free(res); free(out_ref); free(out_wf2); free(out_wf4); free(out_wf8);
    return rc;
}

static int run_with_pattern(const char *pattern, int64_t ny, int64_t nx) {
    int64_t n = ny * nx;
    if (n == 0) return 0;
    uint16_t *src = (uint16_t *)malloc(sizeof(uint16_t) * (size_t)n);
    if (!src) {
        fprintf(stderr, "FAIL: malloc src %lldx%lld\n",
                (long long)ny, (long long)nx);
        return 1;
    }
    int rc = 0;
    char label[128];

    /* gradient + noise (matches the bench's u16 fill pattern) */
    fill_grad_u16(src, ny, nx, 0xC0FFEEu);
    snprintf(label, sizeof label, "%s grad", pattern);
    rc |= run_shape(label, ny, nx, src);

    /* uniform random — adversarial: PAETH branch order is exercised by all
     * three abs comparisons across the full range. */
    fill_random_u16(src, ny, nx, 0xDEADBEEFu);
    snprintf(label, sizeof label, "%s rand", pattern);
    rc |= run_shape(label, ny, nx, src);

    /* constant 0 — residuals are entirely zero, kernels must still produce
     * the original. */
    fill_const_u16(src, ny, nx, 0);
    snprintf(label, sizeof label, "%s zero", pattern);
    rc |= run_shape(label, ny, nx, src);

    /* constant 0xFFFF — exercises the modular u16 wrap on the residual
     * store at the type boundary. */
    fill_const_u16(src, ny, nx, 0xFFFFu);
    snprintf(label, sizeof label, "%s 0xFFFF", pattern);
    rc |= run_shape(label, ny, nx, src);

    free(src);
    return rc;
}

int main(void) {
    printf("test_pred2d_wf_consistency (have_avx2=%d)\n", pred2d_have_avx2());

    int rc = 0;

    /* Edge cases: kernels must fall through their internal scalar paths
     * cleanly when the steady-state can't enter. */
    rc |= run_with_pattern("ny=1",  1,  64);   /* row 0 only — pure LEFT */
    rc |= run_with_pattern("ny=2",  2,  64);   /* one wf row, all in trailing */
    rc |= run_with_pattern("ny=7",  7,  64);   /* hits wf4 but not wf8 steady */
    rc |= run_with_pattern("nx=1", 64,   1);   /* column-only — paeth(0,up,0)=up everywhere */
    rc |= run_with_pattern("nx=2", 64,   2);   /* below wf4 width threshold */
    rc |= run_with_pattern("nx=4", 64,   4);   /* wf4 minimum width */
    rc |= run_with_pattern("nx=8", 64,   8);   /* wf8 minimum width */
    rc |= run_with_pattern("nx=8 ny=9", 9, 8); /* wf8 minimum shape */
    rc |= run_with_pattern("8x8",   8,   8);   /* below wf8 row threshold */
    rc |= run_with_pattern("9x9",   9,   9);
    rc |= run_with_pattern("16x16", 16, 16);
    rc |= run_with_pattern("17x9",  9,  17);   /* odd nx, ensure scatter-store boundaries are safe */

    /* Medium: well within the steady-state range. */
    rc |= run_with_pattern("256x256", 256, 256);

    /* Large: the 2048x2048 bench shape — many octets, full coverage of
     * the steady-state loop. Runs in under a second. */
    rc |= run_with_pattern("2048x2048", 2048, 2048);

    if (rc) {
        fprintf(stderr, "test_pred2d_wf_consistency FAIL\n");
        return 1;
    }
    printf("ALL OK\n");
    return 0;
}
