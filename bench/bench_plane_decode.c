/*
 * bench/bench_plane_decode.c
 *
 * Dedicated PLANE2D decode microbench. Measures only the decode side of
 * PLANE2D+BSHUF+LZ on a fixed 1024x1024 i32 split-planes input — the
 * same workload as case_plane2d_shuffle_lz in bench_throughput.c, but
 * with two important differences:
 *
 *   1. N=200 warm iterations, median timing (bench_throughput.c uses
 *      best-of-5 which is too noisy for the fine-grained comparisons
 *      we need while optimizing the kernel).
 *   2. Reference numbers (memset, memcpy) measured in the same process
 *      so we can compare tdc decode against the memory-bandwidth roof
 *      on the same machine, same cache state, same compiler flags.
 *
 * Output is one CSV row per variant to stdout:
 *
 *     phase,variant,GB_per_sec,ratio_vs_zstd
 *
 * `ratio_vs_zstd` is filled in as a constant (the zstd L9 number from
 * bench/RESULTS.md) so every row is self-describing. When the plan
 * lands a new phase, append the row to PLANE2D-DECODE-SPEEDUP.md.
 *
 * Phase 0 scope: scalar-only. Later phases add -Dxxx build flags or
 * new variants to select between scalar / constant-tile / SIMD
 * kernels, and this same bench gets re-run to populate the table.
 */

#ifdef _MSC_VER
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "tdc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../src/core/timer.h"

#define N_ITERS 200

/* Reference ratio from bench/RESULTS.md — zstd L9 decode MB/s on the
 * same split-planes input. Used only for the ratio column; does not
 * participate in timing. Update when we re-measure zstd. */
#define ZSTD_DECODE_MB_PER_SEC 3800.0

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

/* Split-plane raster generator: identical to fill_split_planes_i32 in
 * bench_throughput.c. Reproduced here so bench_plane_decode can be
 * built standalone without pulling in the bigger harness. */
static void fill_split_planes_i32(int32_t *p, int ny, int nx) {
    for (int r = 0; r < ny; ++r) {
        for (int c = 0; c < nx; ++c) {
            int v = (r < ny / 2) ? (200 + r * 4 + c * 2)
                                 : (1000 - r * 3 + c);
            p[r * nx + c] = v;
        }
    }
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return  1;
    return 0;
}

static double median_of(double *xs, int n) {
    qsort(xs, (size_t)n, sizeof(double), cmp_double);
    return (n & 1) ? xs[n / 2]
                   : 0.5 * (xs[n / 2 - 1] + xs[n / 2]);
}

static double gb_per_sec(size_t bytes, double secs) {
    return (double)bytes / secs / (1024.0 * 1024.0 * 1024.0);
}

static void emit_row(const char *phase, const char *variant,
                     size_t raw_bytes, double median_secs) {
    double gbps = gb_per_sec(raw_bytes, median_secs);
    double zstd_gbps = ZSTD_DECODE_MB_PER_SEC / 1024.0;
    double ratio = gbps / zstd_gbps;
    printf("%s,%s,%.3f,%.3fx\n", phase, variant, gbps, ratio);
}

int main(int argc, char **argv) {
    const char *phase = (argc > 1) ? argv[1] : "phase0";

    const int NY = 1024;
    const int NX = 1024;
    const size_t n_elems = (size_t)NY * (size_t)NX;
    const size_t raw_bytes = n_elems * sizeof(int32_t);

    int32_t *src_data = (int32_t *)malloc(raw_bytes);
    if (!src_data) { fprintf(stderr, "alloc src\n"); return 1; }
    fill_split_planes_i32(src_data, NY, NX);

    tdc_block src = {0};
    src.data        = src_data;
    src.dtype       = TDC_DT_I32;
    src.layout      = TDC_LAYOUT_RASTER_2D;
    src.shape.rank  = 2;
    src.shape.dim[0] = NY;
    src.shape.dim[1] = NX;
    tdc_shape_set_contiguous(&src.shape);

    tdc_plane2d_params params = {0};
    params.tile_size = 32;

    tdc_codec_spec spec = {0};
    spec.model        = TDC_MODEL_PLANE_2D;
    spec.model_params = &params;
    spec.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0]   = TDC_ENTROPY_LZ;

    /* Encode once. We only care about decode throughput here. */
    tdc_buffer enc = make_buffer();
    tdc_status st = tdc_encode_block(&src, &spec, &enc);
    if (st != TDC_OK) {
        fprintf(stderr, "encode failed: %d\n", (int)st);
        free(src_data); free_buffer(&enc); return 1;
    }

    /* Decode destination buffer (reused across iterations). */
    void *dst_data = malloc(raw_bytes);
    if (!dst_data) { free(src_data); free_buffer(&enc); return 1; }

    tdc_block dst = {0};
    dst.data   = dst_data;
    dst.dtype  = src.dtype;
    dst.layout = src.layout;
    dst.shape  = src.shape;

    /* Correctness gate before timing: if decode is wrong, no point
     * reporting a number. */
    st = tdc_decode_block(enc.data, enc.size, &dst);
    if (st != TDC_OK || memcmp(dst_data, src_data, raw_bytes) != 0) {
        fprintf(stderr, "decode correctness gate failed (st=%d)\n", (int)st);
        free(src_data); free(dst_data); free_buffer(&enc); return 1;
    }

    double samples[N_ITERS];

    /* --- tdc decode ------------------------------------------------------ */
    for (int i = 0; i < N_ITERS; ++i) {
        double t0 = tdc_now_secs();
        st = tdc_decode_block(enc.data, enc.size, &dst);
        double t1 = tdc_now_secs();
        if (st != TDC_OK) {
            fprintf(stderr, "decode failed at iter %d: %d\n", i, (int)st);
            free(src_data); free(dst_data); free_buffer(&enc); return 1;
        }
        samples[i] = t1 - t0;
    }
    double med_decode = median_of(samples, N_ITERS);

    /* --- memset reference: pure write bandwidth -------------------------- */
    for (int i = 0; i < N_ITERS; ++i) {
        double t0 = tdc_now_secs();
        memset(dst_data, (int)(i & 0xFF), raw_bytes);
        double t1 = tdc_now_secs();
        samples[i] = t1 - t0;
    }
    double med_memset = median_of(samples, N_ITERS);

    /* --- memcpy reference: read + write bandwidth ------------------------ */
    void *src_copy = malloc(raw_bytes);
    if (!src_copy) { free(src_data); free(dst_data); free_buffer(&enc); return 1; }
    memcpy(src_copy, src_data, raw_bytes); /* warm */
    for (int i = 0; i < N_ITERS; ++i) {
        double t0 = tdc_now_secs();
        memcpy(dst_data, src_copy, raw_bytes);
        double t1 = tdc_now_secs();
        samples[i] = t1 - t0;
    }
    double med_memcpy = median_of(samples, N_ITERS);
    free(src_copy);

    /* CSV header on first run — detected by TDC_BENCH_NO_HEADER env var
     * so CI can suppress it when appending rows. */
    if (getenv("TDC_BENCH_NO_HEADER") == NULL) {
        printf("phase,variant,GB_per_sec,ratio_vs_zstd\n");
    }
    emit_row(phase, "plane2d_decode", raw_bytes, med_decode);
    emit_row(phase, "memcpy_ref",     raw_bytes, med_memcpy);
    emit_row(phase, "memset_ref",     raw_bytes, med_memset);

    /* Sanity annotation to stderr: compression ratio and encoded size,
     * so the reader can cross-check that the codec is doing what we
     * think it's doing. */
    fprintf(stderr, "[info] raw=%zu B enc=%zu B ratio=%.1fx "
                    "n_iters=%d decode_median_us=%.1f\n",
            raw_bytes, enc.size,
            (double)raw_bytes / (double)enc.size,
            N_ITERS, med_decode * 1e6);

    free(src_data);
    free(dst_data);
    free_buffer(&enc);
    return 0;
}
