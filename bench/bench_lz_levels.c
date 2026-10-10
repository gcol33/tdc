/*
 * bench/bench_lz_levels.c
 *
 * LZ level throughput on double columns (gcol33/tdc#1).
 *
 * Encodes 131,072-row f64 blocks through RAW + BYTE_SHUFFLE + LZ, the
 * pipeline vectra uses for a plain double column, at several LZ levels,
 * and reports encode MB/s, decode MB/s and ratio (encoded / raw bytes).
 * A second table times the LZ entropy stage alone on the shuffled bytes,
 * split into parse (tdc_lz_parse_greedy) and full stage encode, so the
 * parser's share of the time is visible.
 *
 * Data shapes:
 *   rnorm      standard normal doubles (Box-Muller over a fixed-seed PRNG)
 *   raster     smooth surface sin(x/40) cos(y/55) on a 512 x 256 grid,
 *              row-major, encoded as a 1D vector
 *   distinct5  doubles drawn uniformly from {1, 2, 3, 4, 5}
 *   round2     rnorm rounded to 2 decimals
 *
 * Every encode is decoded and compared bit-for-bit against the input.
 *
 * Usage: bench_lz_levels [--iters N] [--rows N] [--dump DIR] [--check DIR]
 *   --dump DIR   write each encoded record to DIR/<shape>_l<level>.tdc
 *   --check DIR  decode the records in DIR and compare against the input
 *                (verifies that records written by another build decode)
 * Throughput is uncompressed bytes / best-of-N wallclock, MB = 1e6 bytes.
 */

#include "tdc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "../src/core/timer.h"
#include "../src/entropy/lz_internal.h"

#define N_ROWS_DEFAULT 131072u
#define ITERS_DEFAULT  7
#define PARSE_MIN_GAIN 1u  /* min_gain lz_encode_core passes at levels >= 2 and 0 */

static const int k_levels[] = { 0, 1, 2 };
#define N_LEVELS ((int)(sizeof(k_levels) / sizeof(k_levels[0])))

static void *bench_realloc(void *user, void *ptr, size_t n) {
    (void)user;
    if (n == 0) { free(ptr); return NULL; }
    return realloc(ptr, n);
}

static tdc_buffer make_buffer(void) {
    tdc_buffer b;
    memset(&b, 0, sizeof(b));
    b.realloc_fn = bench_realloc;
    return b;
}

static void free_buffer(tdc_buffer *b) {
    free(b->data);
    b->data = NULL;
    b->size = 0;
    b->capacity = 0;
}

/* splitmix64: fixed-seed, platform-independent PRNG. */
static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint64_t rng_next(void) {
    uint64_t z = (g_rng += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static double rng_unif(void) {
    return ((double)(rng_next() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
static double rng_norm(void) {
    double u1 = rng_unif(), u2 = rng_unif();
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

typedef enum { SHAPE_RNORM, SHAPE_RASTER, SHAPE_DISTINCT5, SHAPE_ROUND2, N_SHAPES } shape_id;
static const char *k_shape_names[N_SHAPES] = { "rnorm", "raster", "distinct5", "round2" };

static void fill_shape(shape_id s, double *x, uint32_t n) {
    g_rng = 0x9E3779B97F4A7C15ull + (uint64_t)s;
    switch (s) {
    case SHAPE_RNORM:
        for (uint32_t i = 0; i < n; i++) x[i] = rng_norm();
        break;
    case SHAPE_RASTER: {
        const uint32_t nx = 512u;
        for (uint32_t i = 0; i < n; i++) {
            double cx = (double)(i % nx), cy = (double)(i / nx);
            x[i] = 100.0 * sin(cx / 40.0) * cos(cy / 55.0) + cy * 0.01;
        }
        break;
    }
    case SHAPE_DISTINCT5: {
        for (uint32_t i = 0; i < n; i++) x[i] = (double)(1u + rng_next() % 5u);
        break;
    }
    case SHAPE_ROUND2:
        for (uint32_t i = 0; i < n; i++) x[i] = round(rng_norm() * 100.0) / 100.0;
        break;
    default:
        break;
    }
}

static tdc_block make_block(double *x, uint32_t n) {
    tdc_block b;
    memset(&b, 0, sizeof(b));
    b.dtype = TDC_DT_F64;
    b.layout = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank = 1;
    b.shape.dim[0] = (int64_t)n;
    tdc_shape_set_contiguous(&b.shape);
    b.data = x;
    return b;
}

static tdc_codec_spec make_spec(tdc_entropy_level *lv) {
    tdc_codec_spec spec = tdc_codec_spec_raw();
    spec.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0] = TDC_ENTROPY_LZ;
    spec.entropy_params[0] = lv;
    return spec;
}

static int decode_and_compare(const uint8_t *rec, size_t rec_size,
                              const double *x, uint32_t n, double *scratch) {
    tdc_block out = make_block(scratch, n);
    memset(scratch, 0, (size_t)n * sizeof(double));
    tdc_status st = tdc_decode_block(rec, rec_size, &out);
    if (st != TDC_OK) {
        fprintf(stderr, "decode failed: %s\n", tdc_strerror(st));
        return 1;
    }
    if (memcmp(scratch, x, (size_t)n * sizeof(double)) != 0) {
        fprintf(stderr, "round-trip mismatch\n");
        return 1;
    }
    return 0;
}

static int write_file(const char *path, const uint8_t *p, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) return 1;
    size_t w = fwrite(p, 1, n, f);
    fclose(f);
    return w != n;
}

static uint8_t *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *p = (uint8_t *)malloc(sz > 0 ? (size_t)sz : 1u);
    if (p && fread(p, 1, (size_t)sz, f) != (size_t)sz) { free(p); p = NULL; }
    fclose(f);
    *n = (size_t)sz;
    return p;
}

int main(int argc, char **argv) {
    int iters = ITERS_DEFAULT;
    uint32_t n = N_ROWS_DEFAULT;
    const char *dump_dir = NULL;
    const char *check_dir = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rows") && i + 1 < argc) n = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump_dir = argv[++i];
        else if (!strcmp(argv[i], "--check") && i + 1 < argc) check_dir = argv[++i];
        else { fprintf(stderr, "unknown argument: %s\n", argv[i]); return 2; }
    }
    if (iters < 1) iters = 1;

    const size_t raw_bytes = (size_t)n * sizeof(double);
    double *x = (double *)malloc(raw_bytes);
    double *scratch = (double *)malloc(raw_bytes);
    uint8_t *shuffled = (uint8_t *)malloc(raw_bytes);
    if (!x || !scratch || !shuffled) return 1;
    int fails = 0;

    printf("rows %u (%.2f MB per block), best of %d\n\n", n, raw_bytes / 1e6, iters);
    printf("%-10s  %5s  %10s  %10s  %7s\n", "data", "level", "enc MB/s", "dec MB/s", "ratio");
    printf("%-10s  %5s  %10s  %10s  %7s\n", "----------", "-----", "----------", "----------", "-------");

    for (int s = 0; s < N_SHAPES; s++) {
        fill_shape((shape_id)s, x, n);
        for (int li = 0; li < N_LEVELS; li++) {
            tdc_entropy_level lv;
            lv.level = k_levels[li];
            tdc_codec_spec spec = make_spec(&lv);
            tdc_block blk = make_block(x, n);

            if (check_dir) {
                char path[1024];
                snprintf(path, sizeof(path), "%s/%s_l%d.tdc", check_dir, k_shape_names[s], lv.level);
                size_t sz = 0;
                uint8_t *rec = read_file(path, &sz);
                if (!rec) { fprintf(stderr, "missing %s\n", path); fails++; continue; }
                if (decode_and_compare(rec, sz, x, n, scratch)) {
                    fprintf(stderr, "  stored record %s failed\n", path);
                    fails++;
                }
                free(rec);
            }

            double best_enc = 1e30, best_dec = 1e30;
            size_t enc_size = 0;
            for (int it = 0; it < iters; it++) {
                tdc_buffer out = make_buffer();
                double t0 = tdc_now_secs();
                tdc_status st = tdc_encode_block(&blk, &spec, &out);
                double t1 = tdc_now_secs();
                if (st != TDC_OK) {
                    fprintf(stderr, "encode failed: %s\n", tdc_strerror(st));
                    return 1;
                }
                if (t1 - t0 < best_enc) best_enc = t1 - t0;
                enc_size = out.size;

                tdc_block dec = make_block(scratch, n);
                double t2 = tdc_now_secs();
                st = tdc_decode_block(out.data, out.size, &dec);
                double t3 = tdc_now_secs();
                if (st != TDC_OK) {
                    fprintf(stderr, "decode failed: %s\n", tdc_strerror(st));
                    return 1;
                }
                if (t3 - t2 < best_dec) best_dec = t3 - t2;
                if (it == 0) {
                    if (decode_and_compare(out.data, out.size, x, n, scratch)) {
                        fprintf(stderr, "  shape %s level %d\n", k_shape_names[s], lv.level);
                        fails++;
                    }
                    if (dump_dir) {
                        char path[1024];
                        snprintf(path, sizeof(path), "%s/%s_l%d.tdc", dump_dir, k_shape_names[s], lv.level);
                        if (write_file(path, out.data, out.size)) {
                            fprintf(stderr, "cannot write %s\n", path);
                            fails++;
                        }
                    }
                }
                free_buffer(&out);
            }
            printf("%-10s  %5d  %10.1f  %10.1f  %7.3f\n", k_shape_names[s], lv.level,
                   raw_bytes / best_enc / 1e6, raw_bytes / best_dec / 1e6,
                   (double)enc_size / (double)raw_bytes);
        }
    }

    /* LZ stage alone on the byte-shuffled input. */
    const tdc_xform_vt *shuf = tdc_xform_get(TDC_XFORM_BYTE_SHUFFLE);
    const tdc_entropy_vt *lz = tdc_entropy_get(TDC_ENTROPY_LZ);
    printf("\nLZ stage only (input = byte-shuffled block)\n\n");
    printf("%-10s  %5s  %10s  %10s  %7s\n", "data", "level", "parse MB/s", "lz MB/s", "seqs");
    printf("%-10s  %5s  %10s  %10s  %7s\n", "----------", "-----", "----------", "----------", "-------");
    for (int s = 0; s < N_SHAPES; s++) {
        fill_shape((shape_id)s, x, n);
        tdc_buffer sb = make_buffer();
        tdc_dtype out_dt;
        if (shuf->encode((const uint8_t *)x, raw_bytes, TDC_DT_F64, NULL, &sb, &out_dt) != TDC_OK) {
            fprintf(stderr, "shuffle failed\n");
            return 1;
        }
        memcpy(shuffled, sb.data, raw_bytes);
        free_buffer(&sb);

        for (int li = 0; li < N_LEVELS; li++) {
            tdc_entropy_level lv;
            lv.level = k_levels[li];
            uint32_t chain = lv.level == 1 ? 0u : 4u;
            uint32_t lazy = lv.level == 0 ? 1u : 0u;
            double best_parse = 1e30, best_lz = 1e30;
            uint32_t nseq = 0;
            for (int it = 0; it < iters; it++) {
                tdc_buffer tmp = make_buffer();
                LZSeq *seqs = NULL;
                double t0 = tdc_now_secs();
                tdc_status st = tdc_lz_parse_greedy(shuffled, (uint32_t)raw_bytes, &tmp,
                                                    chain, lazy, PARSE_MIN_GAIN, &seqs, &nseq);
                double t1 = tdc_now_secs();
                if (st != TDC_OK) return 1;
                free(seqs);
                if (t1 - t0 < best_parse) best_parse = t1 - t0;

                tdc_buffer out = make_buffer();
                double t2 = tdc_now_secs();
                st = lz->encode(shuffled, raw_bytes, &lv, &out);
                double t3 = tdc_now_secs();
                if (st != TDC_OK) return 1;
                free_buffer(&out);
                if (t3 - t2 < best_lz) best_lz = t3 - t2;
            }
            printf("%-10s  %5d  %10.1f  %10.1f  %7u\n", k_shape_names[s], lv.level,
                   raw_bytes / best_parse / 1e6, raw_bytes / best_lz / 1e6, nseq);
        }
    }

    free(x);
    free(scratch);
    free(shuffled);
    if (fails) {
        fprintf(stderr, "%d failure(s)\n", fails);
        return 1;
    }
    return 0;
}
