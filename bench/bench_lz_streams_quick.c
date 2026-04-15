/*
 * bench/bench_lz_streams_quick.c
 *
 * Targeted LZ_STREAMS encode-once, decode-many benchmark for bisecting
 * decode-speed regressions. Takes a raw f64 file on argv[1], encodes
 * once, then decodes 5x to report min decode throughput.
 */

#include "tdc.h"
#include "../src/core/timer.h"
#include "../src/core/decode_profile.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *bench_realloc(void *user, void *ptr, size_t new_size) {
    (void)user;
    if (new_size == 0) { free(ptr); return NULL; }
    return realloc(ptr, new_size);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s PATH [min_match] [level]\n", argv[0]); return 2; }
    uint32_t min_match = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 0u;
    int level = (argc >= 4) ? atoi(argv[3]) : 0;

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *src = (uint8_t *)malloc((size_t)fsize);
    if (fread(src, 1, (size_t)fsize, f) != (size_t)fsize) {
        perror("fread"); return 1;
    }
    fclose(f);

    const tdc_entropy_vt *vt = tdc_entropy_get(TDC_ENTROPY_LZ_STREAMS);
    if (!vt) { fprintf(stderr, "no LZ_STREAMS vt\n"); return 1; }

    tdc_buffer enc = {0};
    enc.realloc_fn = bench_realloc;

    tdc_lz_streams_params p = { .level = level, .min_match = min_match };
    double t0 = tdc_now_secs();
    tdc_status st = vt->encode(src, (size_t)fsize, &p, &enc);
    double t_enc = tdc_now_secs() - t0;
    if (st != TDC_OK) { fprintf(stderr, "encode failed: %d\n", st); return 1; }

    printf("min_match=%u level=%d  encoded %ld -> %zu bytes (%.3fx) in %.2fs (%.1f MB/s)\n",
           min_match, level, fsize, enc.size, (double)fsize / (double)enc.size, t_enc,
           ((double)fsize / 1e6) / t_enc);

    uint8_t *dec = (uint8_t *)malloc((size_t)fsize);
    double best = 1e18;
    tdc_dp_reset();
    for (int i = 0; i < 5; ++i) {
        double t1 = tdc_now_secs();
        st = vt->decode(enc.data, enc.size, dec, (size_t)fsize);
        double dt = tdc_now_secs() - t1;
        if (st != TDC_OK) { fprintf(stderr, "decode failed: %d\n", st); return 1; }
        if (dt < best) best = dt;
    }
    tdc_dp_dump("LZ_STREAMS");
    if (memcmp(src, dec, (size_t)fsize) != 0) {
        fprintf(stderr, "mismatch\n"); return 1;
    }

    printf("decode best-of-5: %.3fs -> %.1f MB/s\n",
           best, ((double)fsize / 1e6) / best);

    free(enc.data);
    free(dec);
    free(src);
    return 0;
}
