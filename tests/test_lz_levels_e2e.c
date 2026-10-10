/*
 * tests/test_lz_levels_e2e.c
 *
 * Full-pipeline round trip of f64 columns through RAW + BYTE_SHUFFLE + LZ
 * at every LZ level, plus the LZ stage alone on inputs that mix
 * high-entropy and matchable regions.
 *
 * What is asserted:
 *   - every level round-trips every data shape bit-for-bit;
 *   - the default level is never larger than level 1 on the double shapes
 *     (standard normal, smooth surface, 5 small integers, normal rounded to
 *     2 decimals), i.e. the hash-chain parser keeps its ratio edge;
 *   - a matchable region that follows a long high-entropy run still
 *     compresses: the miss-streak skip over the random bytes may delay the
 *     parser's re-entry into matchable data by at most a bounded step;
 *   - a parse of incompressible bytes stays a valid stream no larger than
 *     the literal-only bound.
 */

#include "tdc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define N_ROWS 131072u

static void *test_realloc(void *user, void *ptr, size_t n) {
    (void)user;
    if (n == 0) { free(ptr); return NULL; }
    return realloc(ptr, n);
}

static tdc_buffer make_buffer(void) {
    tdc_buffer b;
    memset(&b, 0, sizeof(b));
    b.realloc_fn = test_realloc;
    return b;
}

#define ASSERT_OR_DIE(cond, msg) do {                                       \
    if (!(cond)) {                                                          \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);   \
        return 1;                                                           \
    }                                                                       \
} while (0)

static uint64_t g_rng;
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

enum { SHAPE_RNORM, SHAPE_RASTER, SHAPE_DISTINCT5, SHAPE_ROUND2, N_SHAPES };
static const char *k_shape_names[N_SHAPES] = { "rnorm", "raster", "distinct5", "round2" };

static void fill_shape(int s, double *x, uint32_t n) {
    g_rng = 0x1234567ull + (uint64_t)s;
    for (uint32_t i = 0; i < n; i++) {
        switch (s) {
        case SHAPE_RNORM:     x[i] = rng_norm(); break;
        case SHAPE_RASTER:    x[i] = 100.0 * sin((double)(i % 512u) / 40.0)
                                           * cos((double)(i / 512u) / 55.0); break;
        case SHAPE_DISTINCT5: x[i] = (double)(1u + rng_next() % 5u); break;
        default:              x[i] = round(rng_norm() * 100.0) / 100.0; break;
        }
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

/* Encode x at `level`, decode, compare. Returns the record size, 0 on failure. */
static size_t pipeline_roundtrip(double *x, double *scratch, uint32_t n, int level) {
    tdc_entropy_level lv;
    lv.level = level;
    tdc_codec_spec spec = tdc_codec_spec_raw();
    spec.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0] = TDC_ENTROPY_LZ;
    spec.entropy_params[0] = &lv;

    tdc_block blk = make_block(x, n);
    tdc_buffer out = make_buffer();
    if (tdc_encode_block(&blk, &spec, &out) != TDC_OK) { free(out.data); return 0; }

    memset(scratch, 0, (size_t)n * sizeof(double));
    tdc_block dec = make_block(scratch, n);
    tdc_status st = tdc_decode_block(out.data, out.size, &dec);
    size_t size = out.size;
    free(out.data);
    if (st != TDC_OK) return 0;
    if (memcmp(scratch, x, (size_t)n * sizeof(double)) != 0) return 0;
    return size;
}

/* LZ stage alone. Returns the encoded size, 0 on failure. */
static size_t lz_roundtrip(const uint8_t *src, size_t n, int level) {
    const tdc_entropy_vt *lz = tdc_entropy_get(TDC_ENTROPY_LZ);
    tdc_entropy_level lv;
    lv.level = level;
    tdc_buffer out = make_buffer();
    if (lz->encode(src, n, &lv, &out) != TDC_OK) { free(out.data); return 0; }
    if (out.size > lz->encode_bound(n)) { free(out.data); return 0; }
    uint8_t *back = (uint8_t *)malloc(n ? n : 1u);
    if (!back) { free(out.data); return 0; }
    tdc_status st = lz->decode(out.data, out.size, back, n);
    int same = st == TDC_OK && memcmp(back, src, n) == 0;
    size_t size = out.size;
    free(back);
    free(out.data);
    return same ? size : 0u;
}

static void fill_random(uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)(rng_next() >> 56);
}

/* Matchable text-like bytes: words from a small vocabulary. */
static void fill_words(uint8_t *p, size_t n) {
    static const char *words[] = { "alpha ", "beta ", "gamma ", "delta ",
                                   "epsilon ", "zeta ", "eta ", "theta " };
    size_t i = 0;
    while (i < n) {
        const char *w = words[rng_next() % 8u];
        for (size_t k = 0; w[k] && i < n; k++) p[i++] = (uint8_t)w[k];
    }
}

int main(void) {
    double *x = (double *)malloc((size_t)N_ROWS * sizeof(double));
    double *scratch = (double *)malloc((size_t)N_ROWS * sizeof(double));
    ASSERT_OR_DIE(x && scratch, "alloc");

    for (int s = 0; s < N_SHAPES; s++) {
        fill_shape(s, x, N_ROWS);
        size_t size_default = 0, size_l1 = 0;
        for (int level = 0; level <= 8; level++) {
            size_t sz = pipeline_roundtrip(x, scratch, N_ROWS, level);
            if (sz == 0) {
                fprintf(stderr, "FAIL: round trip %s level %d\n", k_shape_names[s], level);
                return 1;
            }
            if (level == 0) size_default = sz;
            if (level == 1) size_l1 = sz;
        }
        printf("%-10s default %zu  level1 %zu bytes\n", k_shape_names[s], size_default, size_l1);
        if (size_default > size_l1) {
            fprintf(stderr, "FAIL: %s default level larger than level 1\n", k_shape_names[s]);
            return 1;
        }
    }

    /* random | words: the words region must still compress after the long
     * random run, close to how it compresses on its own. */
    {
        const size_t half = 256u * 1024u;
        uint8_t *buf = (uint8_t *)malloc(2u * half);
        ASSERT_OR_DIE(buf != NULL, "alloc");
        g_rng = 77u;
        fill_random(buf, half);
        fill_words(buf + half, half);
        for (int level = 0; level <= 8; level++) {
            size_t mixed = lz_roundtrip(buf, 2u * half, level);
            size_t words = lz_roundtrip(buf + half, half, level);
            ASSERT_OR_DIE(mixed != 0 && words != 0, "mixed round trip");
            /* Random half stays literal (half bytes); the words half may
             * lose at most a few KiB to the delayed re-entry. */
            ASSERT_OR_DIE(mixed <= half + words + 4096u, "words region after random run");
        }

        /* words | random | words */
        const size_t third = 128u * 1024u;
        fill_words(buf, third);
        fill_random(buf + third, third);
        fill_words(buf + 2u * third, third);
        for (int level = 0; level <= 8; level++)
            ASSERT_OR_DIE(lz_roundtrip(buf, 3u * third, level) != 0, "sandwich round trip");

        /* All random: literal-only bound and round trip. */
        fill_random(buf, 2u * half);
        for (int level = 0; level <= 8; level++)
            ASSERT_OR_DIE(lz_roundtrip(buf, 2u * half, level) != 0, "random round trip");

        /* Short inputs around LZ_MIN_MATCH. */
        for (size_t n = 0; n < 16u; n++)
            for (int level = 0; level <= 8; level++)
                ASSERT_OR_DIE(n == 0 || lz_roundtrip(buf, n, level) != 0, "short round trip");
        free(buf);
    }

    free(x);
    free(scratch);
    printf("PASS: test_lz_levels_e2e\n");
    return 0;
}
