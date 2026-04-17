/*
 * tests/test_adversarial_inputs.c
 *
 * Per-model adversarial fixtures. Constructs nine pathological input
 * shapes (all-same, all-unique, alternating, sorted, reverse-sorted,
 * random, high-cardinality f64, low-cardinality f64, sparse-mostly-zero)
 * and runs each one through every model whose accepted_dtypes / layout
 * admits it. Every call is encode -> decode -> memcmp.
 *
 * Explicit priority: DICT_NUMERIC on low-cardinality f64 (the 4.79x
 * winner and the trigger for the LL/ML cap bug on > 2 MiB blocks).
 *
 * Size is fixed at 2 MiB per fixture where possible so we stay inside
 * the large-block regime where size-dependent bugs surface.
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

/* ---- Fixture kinds ------------------------------------------------------ */

typedef enum {
    FX_ALL_SAME,
    FX_ALL_UNIQUE,
    FX_ALTERNATING,
    FX_SORTED,
    FX_REVERSE_SORTED,
    FX_RANDOM,
    FX_HIGHCARD_F64,
    FX_LOWCARD_F64,
    FX_SPARSE_ZEROS
} fixture_kind;

static const char *fixture_name(fixture_kind k) {
    switch (k) {
        case FX_ALL_SAME:       return "all-same";
        case FX_ALL_UNIQUE:     return "all-unique";
        case FX_ALTERNATING:    return "alternating";
        case FX_SORTED:         return "sorted";
        case FX_REVERSE_SORTED: return "reverse-sorted";
        case FX_RANDOM:         return "random";
        case FX_HIGHCARD_F64:   return "highcard-f64";
        case FX_LOWCARD_F64:    return "lowcard-f64";
        case FX_SPARSE_ZEROS:   return "sparse-zeros";
    }
    return "?";
}

/* Fill fixture into buf (n_elems of size elem bytes, treated as little-
 * endian integers for integer fixtures). Works for dtype sizes 1/2/4/8. */
static void build_fixture_bytes(fixture_kind kind, tdc_dtype dt,
                                void *buf, size_t n_elems, uint32_t seed) {
    size_t elem = tdc_dtype_size(dt);
    uint8_t *p = (uint8_t *)buf;
    uint32_t s = seed;

    /* Helper: write `val` at element index i in little-endian. */
    #define WRITE_ELEM(i, val) do {                                       \
        uint64_t _v = (uint64_t)(val);                                    \
        for (size_t _b = 0; _b < elem; ++_b) {                            \
            p[(i) * elem + _b] = (uint8_t)(_v >> (_b * 8));               \
        }                                                                 \
    } while (0)

    switch (kind) {
        case FX_ALL_SAME:
            for (size_t i = 0; i < n_elems; ++i) WRITE_ELEM(i, 42);
            break;
        case FX_ALL_UNIQUE:
            for (size_t i = 0; i < n_elems; ++i) WRITE_ELEM(i, i);
            break;
        case FX_ALTERNATING:
            for (size_t i = 0; i < n_elems; ++i) WRITE_ELEM(i, (i & 1) ? 0xAAu : 0x55u);
            break;
        case FX_SORTED:
            for (size_t i = 0; i < n_elems; ++i) WRITE_ELEM(i, 1000 + i * 3);
            break;
        case FX_REVERSE_SORTED:
            for (size_t i = 0; i < n_elems; ++i)
                WRITE_ELEM(i, (uint64_t)(n_elems * 3 + 1000 - i * 3));
            break;
        case FX_RANDOM:
            for (size_t i = 0; i < n_elems; ++i) {
                s = s * 1664525u + 1013904223u;
                WRITE_ELEM(i, s);
            }
            break;
        case FX_HIGHCARD_F64:
            if (dt == TDC_DT_F64) {
                double *dp = (double *)buf;
                for (size_t i = 0; i < n_elems; ++i) {
                    s = s * 1664525u + 1013904223u;
                    /* Two calls to fill 52 mantissa bits densely — nearly
                     * all-unique f64 values. */
                    uint64_t a = s;
                    s = s * 1664525u + 1013904223u;
                    uint64_t b = s;
                    uint64_t u = (a << 32) ^ b;
                    /* Force a safe exponent range (avoid NaN/Inf). */
                    u = (u & 0x000FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL;
                    memcpy(&dp[i], &u, 8);
                }
            } else {
                /* Fallback: plain random for non-f64. */
                for (size_t i = 0; i < n_elems; ++i) {
                    s = s * 1664525u + 1013904223u;
                    WRITE_ELEM(i, s);
                }
            }
            break;
        case FX_LOWCARD_F64:
            if (dt == TDC_DT_F64) {
                double *dp = (double *)buf;
                /* Pool of 32 values. DICT_NUMERIC sweet spot. */
                double pool[32];
                for (int k = 0; k < 32; ++k) pool[k] = 0.5 + 0.25 * (double)k;
                for (size_t i = 0; i < n_elems; ++i) {
                    s = s * 1664525u + 1013904223u;
                    dp[i] = pool[(s >> 24) & 31];
                }
            } else {
                for (size_t i = 0; i < n_elems; ++i) {
                    s = s * 1664525u + 1013904223u;
                    WRITE_ELEM(i, (s >> 24) & 31);
                }
            }
            break;
        case FX_SPARSE_ZEROS:
            memset(p, 0, n_elems * elem);
            for (size_t i = 0; i < n_elems; ++i) {
                s = s * 1664525u + 1013904223u;
                if (((s >> 24) & 0x3F) == 0) WRITE_ELEM(i, s);
            }
            break;
    }
    #undef WRITE_ELEM
}

/* ---- Pipeline table ----------------------------------------------------- */

typedef struct {
    const char     *name;
    tdc_model_id    model;
    tdc_xform_id    xform[TDC_MAX_TRANSFORMS];
    tdc_entropy_id  entropy[TDC_MAX_ENTROPY];
    /* Bitmask of dtypes this pipeline accepts (test side). */
    uint32_t        accepts_dtypes;
    tdc_layout      layout;
} pipe_t;

#define MASK_I32   TDC_DT_MASK(TDC_DT_I32)
#define MASK_I16   TDC_DT_MASK(TDC_DT_I16)
#define MASK_U8    TDC_DT_MASK(TDC_DT_U8)
#define MASK_F32   TDC_DT_MASK(TDC_DT_F32)
#define MASK_F64   TDC_DT_MASK(TDC_DT_F64)

static const pipe_t k_pipes[] = {
    /* RAW works on any dtype; run on u8 and i32 for coverage. */
    { "RAW+LZ", TDC_MODEL_RAW,
      { TDC_XFORM_NONE, 0, 0, 0 },
      { TDC_ENTROPY_LZ, 0, 0, 0 },
      MASK_U8 | MASK_I32 | MASK_F64,
      TDC_LAYOUT_VECTOR_1D },

    { "RAW+LZ_STREAMS", TDC_MODEL_RAW,
      { TDC_XFORM_NONE, 0, 0, 0 },
      { TDC_ENTROPY_LZ_STREAMS, 0, 0, 0 },
      MASK_U8 | MASK_I32 | MASK_F64,
      TDC_LAYOUT_VECTOR_1D },

    /* DELTA_1D is integer-only. */
    { "DELTA1D+ZIGZAG+BSHUF+LZ", TDC_MODEL_DELTA_1D,
      { TDC_XFORM_ZIGZAG, TDC_XFORM_BYTE_SHUFFLE, 0, 0 },
      { TDC_ENTROPY_LZ, 0, 0, 0 },
      MASK_I32 | MASK_I16,
      TDC_LAYOUT_VECTOR_1D },

    { "DELTA1D+BSHUF+LZ_STREAMS", TDC_MODEL_DELTA_1D,
      { TDC_XFORM_BYTE_SHUFFLE, 0, 0, 0 },
      { TDC_ENTROPY_LZ_STREAMS, 0, 0, 0 },
      MASK_I32 | MASK_I16,
      TDC_LAYOUT_VECTOR_1D },

    /* DICT_NUMERIC on multi-byte numerics. THE bug-triggering path. */
    { "DICT_NUMERIC+BSHUF+LZ", TDC_MODEL_DICT_NUMERIC_1D,
      { TDC_XFORM_BYTE_SHUFFLE, 0, 0, 0 },
      { TDC_ENTROPY_LZ, 0, 0, 0 },
      MASK_F64 | MASK_F32 | MASK_I32,
      TDC_LAYOUT_VECTOR_1D },

    { "DICT_NUMERIC+BSHUF+LZ_STREAMS", TDC_MODEL_DICT_NUMERIC_1D,
      { TDC_XFORM_BYTE_SHUFFLE, 0, 0, 0 },
      { TDC_ENTROPY_LZ_STREAMS, 0, 0, 0 },
      MASK_F64 | MASK_F32 | MASK_I32,
      TDC_LAYOUT_VECTOR_1D },

    { "DICT_NUMERIC+BSHUF+HUFFMAN", TDC_MODEL_DICT_NUMERIC_1D,
      { TDC_XFORM_BYTE_SHUFFLE, 0, 0, 0 },
      { TDC_ENTROPY_HUFFMAN, 0, 0, 0 },
      MASK_F64 | MASK_F32 | MASK_I32,
      TDC_LAYOUT_VECTOR_1D },

    { "DICT_NUMERIC+BSHUF+FSE", TDC_MODEL_DICT_NUMERIC_1D,
      { TDC_XFORM_BYTE_SHUFFLE, 0, 0, 0 },
      { TDC_ENTROPY_FSE, 0, 0, 0 },
      MASK_F64 | MASK_F32 | MASK_I32,
      TDC_LAYOUT_VECTOR_1D },
};
#define N_PIPES (sizeof(k_pipes) / sizeof(k_pipes[0]))

/* ---- Dtype list for iteration ------------------------------------------ */

static const tdc_dtype k_dtypes[] = {
    TDC_DT_U8, TDC_DT_I16, TDC_DT_I32, TDC_DT_F32, TDC_DT_F64
};
#define N_DTYPES (sizeof(k_dtypes) / sizeof(k_dtypes[0]))

static const fixture_kind k_fixtures[] = {
    FX_ALL_SAME, FX_ALL_UNIQUE, FX_ALTERNATING, FX_SORTED,
    FX_REVERSE_SORTED, FX_RANDOM, FX_HIGHCARD_F64, FX_LOWCARD_F64,
    FX_SPARSE_ZEROS
};
#define N_FIXTURES (sizeof(k_fixtures) / sizeof(k_fixtures[0]))

static const char *dtype_name(tdc_dtype dt) {
    switch (dt) {
        case TDC_DT_U8:  return "u8";
        case TDC_DT_I16: return "i16";
        case TDC_DT_I32: return "i32";
        case TDC_DT_F32: return "f32";
        case TDC_DT_F64: return "f64";
        default: return "?";
    }
}

static int is_fixture_applicable(fixture_kind k, tdc_dtype dt) {
    /* The f64-specific fixtures only fire on F64. */
    if ((k == FX_HIGHCARD_F64 || k == FX_LOWCARD_F64) && dt != TDC_DT_F64)
        return 0;
    return 1;
}

/* ---- Single encode/decode/memcmp ---------------------------------------- */

static int run_one(const pipe_t *pl, tdc_dtype dt, fixture_kind kind,
                   size_t n_elems, uint32_t seed) {
    size_t elem = tdc_dtype_size(dt);
    size_t bytes = n_elems * elem;
    void *src_data = malloc(bytes);
    if (!src_data) return 1;

    build_fixture_bytes(kind, dt, src_data, n_elems, seed);

    tdc_block src = {0};
    src.data       = src_data;
    src.dtype      = dt;
    src.layout     = pl->layout;
    src.shape.rank = 1;
    src.shape.dim[0] = (int64_t)n_elems;
    tdc_shape_set_contiguous(&src.shape);

    tdc_codec_spec spec = {0};
    spec.model = pl->model;
    for (int i = 0; i < TDC_MAX_TRANSFORMS; ++i) spec.xform[i]   = pl->xform[i];
    for (int i = 0; i < TDC_MAX_ENTROPY;    ++i) spec.entropy[i] = pl->entropy[i];

    tdc_buffer enc = make_buffer();
    tdc_status st = tdc_encode_block(&src, &spec, &enc);
    if (st != TDC_OK) {
        fprintf(stderr,
                "FAIL [%s dt=%s fx=%s n=%zu]: encode -> %d\n",
                pl->name, dtype_name(dt), fixture_name(kind), n_elems,
                (int)st);
        free(src_data); free_buffer(&enc);
        return 1;
    }

    void *dst_data = malloc(bytes ? bytes : 1);
    if (!dst_data) { free(src_data); free_buffer(&enc); return 1; }
    memset(dst_data, 0xCD, bytes ? bytes : 1);

    tdc_block dst = {0};
    dst.data    = dst_data;
    dst.dtype   = dt;
    dst.layout  = src.layout;
    dst.shape   = src.shape;
    dst.offsets = NULL;

    st = tdc_decode_block(enc.data, enc.size, &dst);
    if (st != TDC_OK) {
        fprintf(stderr,
                "FAIL [%s dt=%s fx=%s n=%zu]: decode -> %d\n",
                pl->name, dtype_name(dt), fixture_name(kind), n_elems,
                (int)st);
        free(dst_data); free(src_data); free_buffer(&enc);
        return 1;
    }

    if (memcmp(dst_data, src_data, bytes) != 0) {
        size_t first = 0;
        for (size_t k = 0; k < bytes; ++k) {
            if (((uint8_t *)dst_data)[k] != ((uint8_t *)src_data)[k]) {
                first = k; break;
            }
        }
        fprintf(stderr,
                "FAIL [%s dt=%s fx=%s n=%zu]:"
                " bytes differ at offset %zu (got 0x%02x, want 0x%02x)\n",
                pl->name, dtype_name(dt), fixture_name(kind), n_elems, first,
                ((uint8_t *)dst_data)[first],
                ((uint8_t *)src_data)[first]);
        free(dst_data); free(src_data); free_buffer(&enc);
        return 1;
    }

    free(dst_data); free(src_data); free_buffer(&enc);
    return 0;
}

/* ---- Main --------------------------------------------------------------- */

int main(void) {
    /* 2 MiB per fixture when possible. Computed per dtype since elem
     * sizes differ. Large enough to push lz_streams into the > 2 MiB
     * match-length regime that was buggy. */
    const size_t target_bytes = 2u * 1024u * 1024u;

    printf("test_adversarial_inputs: %zu pipelines x %zu dtypes x %zu fixtures\n",
           N_PIPES, N_DTYPES, N_FIXTURES);

    int total = 0, passed = 0, failed = 0;
    uint32_t seed = 0xDEFACED0u;

    for (size_t p = 0; p < N_PIPES; ++p) {
        const pipe_t *pl = &k_pipes[p];
        for (size_t d = 0; d < N_DTYPES; ++d) {
            tdc_dtype dt = k_dtypes[d];
            if (!(pl->accepts_dtypes & TDC_DT_MASK(dt))) continue;
            size_t elem = tdc_dtype_size(dt);
            size_t n_elems = target_bytes / elem;

            for (size_t f = 0; f < N_FIXTURES; ++f) {
                fixture_kind fk = k_fixtures[f];
                if (!is_fixture_applicable(fk, dt)) continue;
                total++;
                int rc = run_one(pl, dt, fk, n_elems, seed + (uint32_t)total);
                if (rc == 0) {
                    passed++;
                } else {
                    failed++;
                }
            }
        }
        printf("  [%-34s] cumulative: pass=%d fail=%d of %d\n",
               pl->name, passed, failed, total);
    }

    /* Explicit priority case: DICT_NUMERIC on low-cardinality f64 at
     * 4 MiB, the exact shape that triggered the LL/ML cap bug. Larger
     * than the standard 2 MiB sweep above. */
    {
        const pipe_t dn = { "DICT_NUMERIC+BSHUF+LZ_STREAMS/4MiB",
            TDC_MODEL_DICT_NUMERIC_1D,
            { TDC_XFORM_BYTE_SHUFFLE, 0, 0, 0 },
            { TDC_ENTROPY_LZ_STREAMS, 0, 0, 0 },
            MASK_F64, TDC_LAYOUT_VECTOR_1D };
        size_t n = (4u * 1024u * 1024u) / 8u;
        total++;
        int rc = run_one(&dn, TDC_DT_F64, FX_LOWCARD_F64, n, 0xFEEDFACEu);
        if (rc == 0) { passed++; printf("  [dict_numeric lowcard f64 4MiB] OK\n"); }
        else         { failed++; printf("  [dict_numeric lowcard f64 4MiB] FAIL\n"); }
    }

    printf("test_adversarial_inputs: total=%d pass=%d fail=%d\n",
           total, passed, failed);
    return (failed == 0) ? 0 : 1;
}
