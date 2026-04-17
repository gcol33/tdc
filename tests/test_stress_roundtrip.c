/*
 * tests/test_stress_roundtrip.c
 *
 * Property-style stress harness: iterate random block sizes on a log
 * scale from 64 B to 32 MiB across a set of real encode pipelines, run
 * encode -> decode -> memcmp for each, and report pass/fail counts.
 *
 * The goal is to catch size-dependent latent bugs like the LL/ML symbol
 * cap issue that fired only on blocks > 2 MiB (raised 21 -> 25 in
 * src/entropy/lz_streams.c). A fixed-size test suite missed it because
 * every fixture was 1 MiB. This harness sweeps sizes deterministically:
 *
 *   - Per-iteration size is drawn from a log-uniform LCG, in bytes.
 *   - Size classes: small  (64 B ..  1 MiB)
 *                    medium ( 1 MiB .. 4 MiB)
 *                    large  ( 4 MiB .. 32 MiB)
 *   - At least 30 iterations per (pipeline, class) — enough to hit the
 *     match-length-cap regime multiple times on each pipeline.
 *
 * Determinism: the outer LCG is seeded with a fixed seed so a failure
 * prints the failing seed/size and reproduces on re-run.
 */

#include "tdc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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

/* ---- Deterministic LCG (Numerical Recipes). Returns 32 random bits. ---- */
static uint32_t lcg_next(uint32_t *s) {
    *s = (*s) * 1664525u + 1013904223u;
    return *s;
}

/* ---- Fill helpers -------------------------------------------------------- */

/* Mixed LCG: literal-heavy pseudo-random bytes. */
static void fill_bytes_lcg(uint8_t *buf, size_t n, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        buf[i] = (uint8_t)(s >> 24);
    }
}

/* Fill a dtype=I32 ramp. Elements indexed as int32_t *. */
static void fill_i32_ramp(int32_t *p, size_t n_elems, uint32_t seed) {
    uint32_t s = seed;
    int32_t v = 1000;
    for (size_t i = 0; i < n_elems; i++) {
        s = s * 1664525u + 1013904223u;
        int step = (int)((s >> 20) & 0xF); /* small positive step */
        v += step;
        p[i] = v;
    }
}

/* Fill f64 low-cardinality grid — DICT_NUMERIC target workload. */
static void fill_f64_lowcard(double *p, size_t n_elems, uint32_t seed) {
    /* Draw from a pool of ~64 values. LCG picks which one per element. */
    double pool[64];
    for (int k = 0; k < 64; k++) pool[k] = 0.5 + 0.25 * (double)k;
    uint32_t s = seed;
    for (size_t i = 0; i < n_elems; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = pool[(s >> 24) & 63];
    }
}

/* Fill f64 smooth field: delta1d+bshuf friendly. */
static void fill_f64_smooth(double *p, size_t n_elems, uint32_t seed) {
    uint32_t s = seed;
    double v = 1000.0;
    for (size_t i = 0; i < n_elems; i++) {
        s = s * 1664525u + 1013904223u;
        double noise = ((double)((int)((s >> 16) & 0xFF) - 128)) * 1e-6;
        v += 0.001 + noise;
        p[i] = v;
    }
}

/* ---- Pipeline descriptor table ------------------------------------------ */

typedef enum {
    DT_U8,
    DT_I32,
    DT_F64_LOWCARD,
    DT_F64_SMOOTH
} fill_kind;

typedef struct {
    const char      *name;
    tdc_model_id     model;
    tdc_xform_id     xform[TDC_MAX_TRANSFORMS];
    tdc_entropy_id   entropy[TDC_MAX_ENTROPY];
    fill_kind        filler;
    tdc_dtype        dtype;
} pipeline_t;

/* The four pipelines called out in the task. One pipeline per line;
 * trailing zeros in xform[]/entropy[] terminate the chain. */
static const pipeline_t k_pipelines[] = {
    { "RAW+LZ",
      TDC_MODEL_RAW,
      { TDC_XFORM_NONE, 0, 0, 0 },
      { TDC_ENTROPY_LZ, 0, 0, 0 },
      DT_U8, TDC_DT_U8 },

    { "RAW+LZ_STREAMS",
      TDC_MODEL_RAW,
      { TDC_XFORM_NONE, 0, 0, 0 },
      { TDC_ENTROPY_LZ_STREAMS, 0, 0, 0 },
      DT_U8, TDC_DT_U8 },

    { "DELTA1D+BSHUF+LZ",
      TDC_MODEL_DELTA_1D,
      { TDC_XFORM_BYTE_SHUFFLE, 0, 0, 0 },
      { TDC_ENTROPY_LZ, 0, 0, 0 },
      DT_I32, TDC_DT_I32 },

    { "DICT_NUMERIC+BSHUF+LZ_STREAMS",
      TDC_MODEL_DICT_NUMERIC_1D,
      { TDC_XFORM_BYTE_SHUFFLE, 0, 0, 0 },
      { TDC_ENTROPY_LZ_STREAMS, 0, 0, 0 },
      DT_F64_LOWCARD, TDC_DT_F64 },

    /* Bonus coverage: a smooth f64 workload through the same pipeline to
     * ensure DICT_NUMERIC also round-trips when the value cardinality is
     * near the element count (degenerate dict case). */
    { "DICT_NUMERIC+BSHUF+LZ_STREAMS/smooth",
      TDC_MODEL_DICT_NUMERIC_1D,
      { TDC_XFORM_BYTE_SHUFFLE, 0, 0, 0 },
      { TDC_ENTROPY_LZ_STREAMS, 0, 0, 0 },
      DT_F64_SMOOTH, TDC_DT_F64 },
};
#define N_PIPELINES (sizeof(k_pipelines) / sizeof(k_pipelines[0]))

/* ---- Size class definitions --------------------------------------------- */

typedef struct {
    const char *name;
    size_t      min_bytes;
    size_t      max_bytes;
    int         iters;
} size_class_t;

static const size_class_t k_classes[] = {
    { "small",  (size_t)64,                (size_t)(1u * 1024u * 1024u),         30 },
    { "medium", (size_t)(1u * 1024u * 1024u), (size_t)(4u * 1024u * 1024u),      30 },
    { "large",  (size_t)(4u * 1024u * 1024u), (size_t)(32u * 1024u * 1024u),     30 },
};
#define N_CLASSES (sizeof(k_classes) / sizeof(k_classes[0]))

/* Draw a log-uniform size in [lo, hi] from an LCG state. */
static size_t draw_size(uint32_t *s, size_t lo, size_t hi) {
    /* Log-uniform: exponent is uniform in [log2(lo), log2(hi)]. */
    double ll = 0.0, hh = 0.0;
    {
        size_t x = lo ? lo : 1;
        while (x > 1) { x >>= 1; ll += 1.0; }
    }
    {
        size_t x = hi ? hi : 1;
        while (x > 1) { x >>= 1; hh += 1.0; }
    }
    uint32_t r = lcg_next(s);
    double u = ((double)r) / 4294967296.0;
    double e = ll + (hh - ll) * u;
    /* Convert back from log2 via pow of 2 by repeated multiplication. */
    double val = 1.0;
    int k = (int)e;
    for (int i = 0; i < k; ++i) val *= 2.0;
    /* Fractional mantissa: uniform in [1, 2). */
    double frac = e - (double)k;
    double mant = 1.0;
    for (int i = 0; i < 3; ++i) {
        /* Piecewise-linear approximation of 2^frac — we don't care
         * about smoothness, only that the size distribution covers the
         * range. 3 halvings give a factor of at most 2 error. */
        if (frac >= 0.5) { mant *= 1.4142135; frac -= 0.5; }
        else             { mant *= 1.0;        frac *= 2.0; }
    }
    double out = val * mant;
    size_t sz = (size_t)out;
    if (sz < lo) sz = lo;
    if (sz > hi) sz = hi;
    return sz;
}

/* ---- Block construction ------------------------------------------------- */

/* Returns the number of raw bytes allocated at *pbytes. Caller frees *pdata. */
static int build_block(const pipeline_t *pl, size_t target_bytes,
                       uint32_t seed,
                       tdc_block *blk, void **pdata, size_t *pbytes) {
    size_t elem = tdc_dtype_size(pl->dtype);
    if (elem == 0) return 1;
    size_t n_elems = target_bytes / elem;
    if (n_elems == 0) n_elems = 1;
    size_t bytes = n_elems * elem;
    void *data = malloc(bytes);
    if (!data) return 1;

    switch (pl->filler) {
        case DT_U8:
            fill_bytes_lcg((uint8_t *)data, bytes, seed);
            break;
        case DT_I32:
            fill_i32_ramp((int32_t *)data, n_elems, seed);
            break;
        case DT_F64_LOWCARD:
            fill_f64_lowcard((double *)data, n_elems, seed);
            break;
        case DT_F64_SMOOTH:
            fill_f64_smooth((double *)data, n_elems, seed);
            break;
    }

    memset(blk, 0, sizeof *blk);
    blk->data       = data;
    blk->dtype      = pl->dtype;
    blk->layout     = TDC_LAYOUT_VECTOR_1D;
    blk->shape.rank = 1;
    blk->shape.dim[0] = (int64_t)n_elems;
    tdc_shape_set_contiguous(&blk->shape);

    *pdata  = data;
    *pbytes = bytes;
    return 0;
}

static void build_spec(const pipeline_t *pl, tdc_codec_spec *s) {
    memset(s, 0, sizeof *s);
    s->model = pl->model;
    for (int i = 0; i < TDC_MAX_TRANSFORMS; ++i) s->xform[i]   = pl->xform[i];
    for (int i = 0; i < TDC_MAX_ENTROPY;    ++i) s->entropy[i] = pl->entropy[i];
}

/* One encode/decode/memcmp iteration. Returns 0 on success, 1 on hard
 * failure, prints a FAIL line on mismatch. Tracks suspicious ratios. */
static int one_trip(const pipeline_t *pl, size_t target_bytes,
                    uint32_t seed, int *suspicious_count) {
    tdc_block src;
    void     *src_data = NULL;
    size_t    raw_bytes = 0;
    if (build_block(pl, target_bytes, seed, &src, &src_data, &raw_bytes)) {
        fprintf(stderr, "FAIL [%s seed=0x%08x sz=%zu]: build_block\n",
                pl->name, seed, target_bytes);
        return 1;
    }

    tdc_codec_spec spec;
    build_spec(pl, &spec);

    tdc_buffer enc = make_buffer();
    tdc_status st = tdc_encode_block(&src, &spec, &enc);
    if (st != TDC_OK) {
        fprintf(stderr,
                "FAIL [%s seed=0x%08x sz=%zu raw=%zu]: encode -> %d\n",
                pl->name, seed, target_bytes, raw_bytes, (int)st);
        free(src_data); free_buffer(&enc);
        return 1;
    }

    void *dst_data = malloc(raw_bytes ? raw_bytes : 1);
    if (!dst_data) {
        fprintf(stderr, "FAIL [%s]: dst alloc\n", pl->name);
        free(src_data); free_buffer(&enc); return 1;
    }
    memset(dst_data, 0xA5, raw_bytes ? raw_bytes : 1);

    tdc_block dst = {0};
    dst.data       = dst_data;
    dst.dtype      = src.dtype;
    dst.layout     = src.layout;
    dst.shape      = src.shape;
    dst.offsets    = NULL;

    st = tdc_decode_block(enc.data, enc.size, &dst);
    if (st != TDC_OK) {
        fprintf(stderr,
                "FAIL [%s seed=0x%08x sz=%zu raw=%zu enc=%zu]: decode -> %d\n",
                pl->name, seed, target_bytes, raw_bytes, enc.size, (int)st);
        free(dst_data); free(src_data); free_buffer(&enc);
        return 1;
    }

    if (memcmp(dst_data, src_data, raw_bytes) != 0) {
        /* Find first diverging byte. */
        size_t first = 0;
        for (size_t k = 0; k < raw_bytes; ++k) {
            if (((uint8_t *)dst_data)[k] != ((uint8_t *)src_data)[k]) {
                first = k; break;
            }
        }
        fprintf(stderr,
                "FAIL [%s seed=0x%08x sz=%zu raw=%zu]:"
                " bytes differ at offset %zu (got 0x%02x, want 0x%02x)\n",
                pl->name, seed, target_bytes, raw_bytes, first,
                ((uint8_t *)dst_data)[first],
                ((uint8_t *)src_data)[first]);
        free(dst_data); free(src_data); free_buffer(&enc);
        return 1;
    }

    /* Ratio sanity: complain if encoded is dramatically larger than
     * uncompressed on an input that has any structure. We only flag
     * cases where encoded > 2x raw, since RAW+LZ on high-entropy u8
     * should legitimately produce ~1.01x (LZ header+overhead). */
    if (raw_bytes > 1024 && enc.size > 2 * raw_bytes) {
        fprintf(stderr,
                "SUSPICIOUS [%s seed=0x%08x sz=%zu]: enc=%zu > 2x raw=%zu\n",
                pl->name, seed, target_bytes, enc.size, raw_bytes);
        if (suspicious_count) (*suspicious_count)++;
    }

    free(dst_data);
    free(src_data);
    free_buffer(&enc);
    return 0;
}

/* ---- Main -------------------------------------------------------------- */

int main(void) {
    printf("test_stress_roundtrip: iterating %zu pipelines x %zu classes\n",
           N_PIPELINES, N_CLASSES);

    uint32_t lcg_state = 0xF00DC0DEu; /* fixed seed; reproducible */

    int total = 0, passed = 0, failed = 0, suspicious = 0;

    for (size_t p = 0; p < N_PIPELINES; ++p) {
        for (size_t c = 0; c < N_CLASSES; ++c) {
            const pipeline_t   *pl  = &k_pipelines[p];
            const size_class_t *sc  = &k_classes[c];
            int class_pass = 0, class_fail = 0;
            for (int it = 0; it < sc->iters; ++it) {
                total++;
                uint32_t seed = lcg_next(&lcg_state);
                size_t   sz   = draw_size(&lcg_state, sc->min_bytes, sc->max_bytes);
                int rc = one_trip(pl, sz, seed, &suspicious);
                if (rc == 0) { class_pass++; passed++; }
                else         { class_fail++; failed++; }
            }
            printf("  [%-38s %-6s] %2d/%2d OK\n",
                   pl->name, sc->name, class_pass, class_pass + class_fail);
        }
    }

    printf("test_stress_roundtrip: total=%d pass=%d fail=%d suspicious=%d\n",
           total, passed, failed, suspicious);
    return (failed == 0) ? 0 : 1;
}
