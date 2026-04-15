/*
 * bench/bench_lz_decode_micro.c
 *
 * A/B microbench for LZ_STREAMS decode. Generates the same smooth f64
 * 2M input used by bench_throughput, encodes once with a selectable
 * level, then decodes N iterations back-to-back. Reports best, median,
 * and mean ns/byte so small changes are visible above noise.
 *
 * Usage: bench_lz_decode_micro [iters=200] [level=19]
 */

#include "tdc.h"
#include "../src/core/timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *ba(void *u, void *p, size_t n) {
    (void)u;
    if (n == 0) { free(p); return NULL; }
    return realloc(p, n);
}

static void fill_smooth_f64(double *p, size_t n) {
    uint32_t s = 0xBEEF1234u;
    double v = 1000.0;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1103515245u + 12345u;
        double noise = ((double)((int)((s >> 16) & 0xFF) - 128)) * 1e-6;
        v += 0.001 + noise;
        p[i] = v;
    }
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

int main(int argc, char **argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 200;
    int level = (argc > 2) ? atoi(argv[2]) : 19;
    if (iters < 3) iters = 3;

    const size_t N = 2u * 1024u * 1024u;
    double *src64 = (double *)malloc(N * sizeof *src64);
    fill_smooth_f64(src64, N);
    size_t src_bytes = N * sizeof *src64;

    const tdc_entropy_vt *vt = tdc_entropy_get(TDC_ENTROPY_LZ_STREAMS);
    if (!vt) { fprintf(stderr, "no vt\n"); return 1; }

    tdc_buffer enc = { 0 };
    enc.realloc_fn = ba;

    /* Use entropy params to select level if supported. For now encode with
     * defaults; level arg is ignored unless wiring exists. */
    (void)level;

    tdc_status st = vt->encode((const uint8_t *)src64, src_bytes, NULL, &enc);
    if (st != TDC_OK) { fprintf(stderr, "encode failed: %d\n", st); return 1; }

    printf("encoded %zu -> %zu bytes (%.3fx)\n",
           src_bytes, enc.size, (double)src_bytes / (double)enc.size);

    uint8_t *dec = (uint8_t *)malloc(src_bytes);
    double *times = (double *)malloc((size_t)iters * sizeof *times);

    /* Warmup: 3 untimed decodes. */
    for (int i = 0; i < 3; ++i) {
        st = vt->decode(enc.data, enc.size, dec, src_bytes);
        if (st != TDC_OK) { fprintf(stderr, "decode failed: %d\n", st); return 1; }
    }

    if (memcmp(src64, dec, src_bytes) != 0) {
        fprintf(stderr, "mismatch after warmup\n"); return 1;
    }

    for (int i = 0; i < iters; ++i) {
        double t0 = tdc_now_secs();
        st = vt->decode(enc.data, enc.size, dec, src_bytes);
        double dt = tdc_now_secs() - t0;
        if (st != TDC_OK) { fprintf(stderr, "decode failed: %d\n", st); return 1; }
        times[i] = dt;
    }

    qsort(times, (size_t)iters, sizeof *times, cmp_double);
    double best   = times[0];
    double median = times[iters / 2];
    double p90    = times[(int)(iters * 0.9)];

    double mb_per_s_best = ((double)src_bytes / 1e6) / best;
    double mb_per_s_med  = ((double)src_bytes / 1e6) / median;
    double mb_per_s_p90  = ((double)src_bytes / 1e6) / p90;

    printf("iters=%d\n", iters);
    printf("  best   %.2f MB/s  (%.2f ns/byte)\n",
           mb_per_s_best, best * 1e9 / (double)src_bytes);
    printf("  median %.2f MB/s  (%.2f ns/byte)\n",
           mb_per_s_med,  median * 1e9 / (double)src_bytes);
    printf("  p90    %.2f MB/s  (%.2f ns/byte)\n",
           mb_per_s_p90,  p90    * 1e9 / (double)src_bytes);

    free(times);
    free(dec);
    free(enc.data);
    free(src64);
    return 0;
}
