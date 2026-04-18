/*
 * tests/test_decode_varlen.c
 *
 * Round-trip tests for tdc_decode_block_varlen — the public entry point
 * for variable-width (TDC_DT_STRING) decode that allocates dst->data and
 * dst->offsets via the caller-supplied realloc_fn.
 *
 * Strategy:
 *   1. Build a TDC_DT_STRING / VECTOR_1D block on the stack/heap.
 *   2. Encode with TDC_MODEL_DICT_1D (+ optional xform/entropy chain).
 *   3. Discover dtype/layout/shape via tdc_decode_peek (peek must
 *      tolerate variable-width dtypes by returning TDC_E_UNSUPPORTED for
 *      out_bytes_required while still populating out_meta).
 *   4. Hand a freshly-zeroed dst plus a realloc-tracking allocator to
 *      tdc_decode_block_varlen and confirm:
 *        - dst->data and dst->offsets are non-NULL and freshly allocated;
 *        - the decoded strings match the source byte-for-byte;
 *        - every allocation reaches realloc_fn (no bare malloc);
 *        - freeing both buffers via the same realloc_fn balances the
 *          allocator (no leaks, no double-frees).
 *
 * Covers:
 *   - low-cardinality strings (the common case);
 *   - all-unique strings (worst case for the dictionary);
 *   - strings with embedded zeros (heap is opaque bytes);
 *   - empty block (n == 0): offsets[1] sentinel + dst->data == NULL;
 *   - empty strings interleaved with non-empty ones;
 *   - rejections: NULL src/dst/alloc, missing realloc_fn, fixed-width
 *     dtype, pre-populated dst->data or dst->offsets.
 */

#include "tdc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ----- Allocator with a live counter ------------------------------------- */
/*
 * Wraps libc realloc/free so we can assert no leaks (live == 0 after
 * every test). Stored in tdc_buffer::user.
 */
typedef struct {
    int64_t live_bytes;
    int64_t live_blocks;
    int64_t total_allocs;
} alloc_stats;

static void *tracking_realloc(void *user, void *ptr, size_t new_size) {
    alloc_stats *s = (alloc_stats *)user;
    if (new_size == 0) {
        if (ptr) {
            /* libc realloc has no portable size_of_block — we don't
             * track per-pointer sizes here, so live_bytes is approximate
             * and the test only checks live_blocks for leaks. */
            s->live_blocks--;
            free(ptr);
        }
        return NULL;
    }
    if (!ptr) {
        s->total_allocs++;
        s->live_blocks++;
    }
    return realloc(ptr, new_size);
}

static tdc_buffer make_alloc(alloc_stats *s) {
    tdc_buffer b = {0};
    b.realloc_fn = tracking_realloc;
    b.user       = s;
    return b;
}

/* Simple buffer helper for the encoded record bytes. */
static void *plain_realloc(void *user, void *ptr, size_t new_size) {
    (void)user;
    if (new_size == 0) { free(ptr); return NULL; }
    return realloc(ptr, new_size);
}

static tdc_buffer make_plain_buffer(void) {
    tdc_buffer b = {0};
    b.realloc_fn = plain_realloc;
    return b;
}

static void free_plain_buffer(tdc_buffer *b) {
    if (b->data) free(b->data);
    b->data = NULL; b->size = 0; b->capacity = 0;
}

#define ASSERT_OR_DIE(cond, msg) do {                                       \
    if (!(cond)) {                                                          \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);   \
        return 1;                                                           \
    }                                                                       \
} while (0)

/* ----- String block helpers ---------------------------------------------- */

typedef struct {
    tdc_block block;
    uint8_t  *heap;
    uint32_t *offsets;
} string_block;

static string_block make_string_block_lens(const uint8_t *const *rows,
                                           const uint32_t       *lens,
                                           int64_t               n) {
    string_block sb = {0};
    sb.offsets = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)(n + 1));
    uint32_t total = 0;
    for (int64_t i = 0; i < n; ++i) {
        sb.offsets[i] = total;
        total += lens[i];
    }
    sb.offsets[n] = total;

    sb.heap = (uint8_t *)malloc(total > 0 ? total : 1u);
    for (int64_t i = 0; i < n; ++i) {
        if (lens[i] > 0) memcpy(sb.heap + sb.offsets[i], rows[i], lens[i]);
    }

    sb.block.data        = sb.heap;
    sb.block.offsets     = sb.offsets;
    sb.block.dtype       = TDC_DT_STRING;
    sb.block.layout      = TDC_LAYOUT_VECTOR_1D;
    sb.block.shape.rank  = 1;
    sb.block.shape.dim[0] = n;
    sb.block.shape.stride[0] = 1;
    return sb;
}

static string_block make_string_block(const char *const *rows, int64_t n) {
    /* Trampoline so tests can pass plain C-strings. */
    const uint8_t **rb = (const uint8_t **)malloc(sizeof(*rb) * (size_t)(n > 0 ? n : 1));
    uint32_t       *ln = (uint32_t *)malloc(sizeof(*ln) * (size_t)(n > 0 ? n : 1));
    for (int64_t i = 0; i < n; ++i) {
        rb[i] = (const uint8_t *)rows[i];
        ln[i] = (uint32_t)strlen(rows[i]);
    }
    string_block sb = make_string_block_lens(rb, ln, n);
    free(rb);
    free(ln);
    return sb;
}

static void free_string_block(string_block *sb) {
    if (sb->heap)    free(sb->heap);
    if (sb->offsets) free(sb->offsets);
    sb->heap = NULL; sb->offsets = NULL;
}

/* Compare two STRING blocks for byte-for-byte equality. */
static int string_blocks_equal(const tdc_block *a, const tdc_block *b) {
    if (a->shape.dim[0] != b->shape.dim[0]) return 0;
    int64_t n = a->shape.dim[0];
    if (n == 0) return 1;
    for (int64_t i = 0; i < n; ++i) {
        uint32_t la = a->offsets[i + 1] - a->offsets[i];
        uint32_t lb = b->offsets[i + 1] - b->offsets[i];
        if (la != lb) return 0;
        if (la > 0 && memcmp((const uint8_t *)a->data + a->offsets[i],
                             (const uint8_t *)b->data + b->offsets[i],
                             la) != 0) return 0;
    }
    return 1;
}

/* ----- Generic round-trip via decode_varlen ------------------------------ */

static int rt_varlen(const char         *label,
                     const string_block *src,
                     const tdc_codec_spec *spec) {
    tdc_buffer enc = make_plain_buffer();
    tdc_status st  = tdc_encode_block(&src->block, spec, &enc);
    if (st != TDC_OK) {
        fprintf(stderr, "FAIL [%s]: encode -> %d\n", label, (int)st);
        free_plain_buffer(&enc);
        return 1;
    }

    /* Peek populates dtype/layout/shape; for STRING the bytes_required
     * out-param is unsupported. We only consume the meta fields. */
    tdc_block meta = {0};
    size_t    need = 0xDEADBEEFu;
    st = tdc_decode_peek(enc.data, enc.size, &meta, &need);
    ASSERT_OR_DIE(st == TDC_E_UNSUPPORTED,
                  "peek should be TDC_E_UNSUPPORTED for variable-width");
    ASSERT_OR_DIE(meta.dtype  == TDC_DT_STRING,        "peek dtype");
    ASSERT_OR_DIE(meta.layout == TDC_LAYOUT_VECTOR_1D, "peek layout");
    ASSERT_OR_DIE(meta.shape.rank   == 1,              "peek rank");
    ASSERT_OR_DIE(meta.shape.dim[0] == src->block.shape.dim[0],
                  "peek n_elems");

    alloc_stats stats = {0};
    tdc_buffer  alloc = make_alloc(&stats);

    tdc_block dst = meta;   /* dtype/layout/shape from peek */
    dst.data    = NULL;
    dst.offsets = NULL;

    st = tdc_decode_block_varlen(enc.data, enc.size, &dst, &alloc);
    if (st != TDC_OK) {
        fprintf(stderr, "FAIL [%s]: decode_varlen -> %d\n", label, (int)st);
        free_plain_buffer(&enc);
        return 1;
    }

    int64_t n = dst.shape.dim[0];
    /* offsets is always allocated (sentinel even for n==0). */
    ASSERT_OR_DIE(dst.offsets != NULL, "offsets must be allocated");
    if (n > 0) {
        ASSERT_OR_DIE(string_blocks_equal(&src->block, &dst),
                      "decoded != source");
    } else {
        ASSERT_OR_DIE(dst.offsets[0] == 0u, "empty: offsets[0] must be 0");
        ASSERT_OR_DIE(dst.data == NULL,
                      "empty: dst.data must remain NULL");
    }

    /* Free both buffers via the same allocator and confirm the live
     * block count returns to zero — proves the wrapper allocates
     * exclusively through realloc_fn. */
    if (dst.data)    alloc.realloc_fn(alloc.user, dst.data,    0);
    if (dst.offsets) alloc.realloc_fn(alloc.user, dst.offsets, 0);
    ASSERT_OR_DIE(stats.live_blocks == 0,
                  "live blocks must be zero after free");
    ASSERT_OR_DIE(stats.total_allocs >= 1,
                  "expected at least one realloc_fn allocation");

    fprintf(stdout, "  %-55s OK  n=%lld allocs=%lld\n",
            label, (long long)n, (long long)stats.total_allocs);
    free_plain_buffer(&enc);
    return 0;
}

/* ----- Test cases -------------------------------------------------------- */

static int test_low_cardinality(void) {
    static const char *rows[12] = {
        "alpha", "beta", "alpha", "gamma",
        "beta",  "alpha", "alpha", "delta",
        "delta", "beta",  "gamma", "alpha"
    };
    string_block sb = make_string_block(rows, 12);
    tdc_codec_spec spec = {0};
    spec.model      = TDC_MODEL_DICT_1D;
    spec.xform[0]   = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0] = TDC_ENTROPY_LZ;
    int rc = rt_varlen("DICT_1D + BSHUF + LZ (low cardinality strings)",
                       &sb, &spec);
    free_string_block(&sb);
    return rc;
}

static int test_all_unique(void) {
    enum { N = 200 };
    char  storage[N][16];
    const char *rows[N];
    for (int i = 0; i < N; ++i) {
        snprintf(storage[i], sizeof storage[i], "row_%05d", i);
        rows[i] = storage[i];
    }
    string_block sb = make_string_block(rows, N);
    tdc_codec_spec spec = {0};
    spec.model = TDC_MODEL_DICT_1D;
    int rc = rt_varlen("DICT_1D bare (all unique 200)", &sb, &spec);
    free_string_block(&sb);
    return rc;
}

static int test_embedded_zeros(void) {
    /* Heap is opaque bytes — embedded NULs must round-trip. */
    static const uint8_t a[] = { 'a', 0, 'b' };
    static const uint8_t b[] = { 0, 0, 0, 1 };
    static const uint8_t c[] = { 'x' };
    const uint8_t *rows[6] = { a, b, a, c, b, a };
    uint32_t       lens[6] = { sizeof a, sizeof b, sizeof a, sizeof c, sizeof b, sizeof a };
    string_block sb = make_string_block_lens(rows, lens, 6);
    tdc_codec_spec spec = {0};
    spec.model      = TDC_MODEL_DICT_1D;
    spec.xform[0]   = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0] = TDC_ENTROPY_LZ;
    int rc = rt_varlen("DICT_1D + BSHUF + LZ (embedded NULs)", &sb, &spec);
    free_string_block(&sb);
    return rc;
}

static int test_empty_strings_mixed(void) {
    static const char *rows[7] = { "", "x", "", "longer", "x", "", "x" };
    string_block sb = make_string_block(rows, 7);
    tdc_codec_spec spec = {0};
    spec.model      = TDC_MODEL_DICT_1D;
    spec.xform[0]   = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0] = TDC_ENTROPY_LZ;
    int rc = rt_varlen("DICT_1D + BSHUF + LZ (empty strings interleaved)",
                       &sb, &spec);
    free_string_block(&sb);
    return rc;
}

static int test_all_empty_strings(void) {
    /* All rows are zero-length. The encoded heap is 0 bytes, so the
     * varlen hook leaves dst.data == NULL; the decode must tolerate that
     * (regression for dict1d_decode rejecting NULL data even when no
     * memcpy actually runs — uncovered by vectra rg=1 with an empty
     * string in its own rowgroup). */
    static const char *rows[3] = { "", "", "" };
    string_block sb = make_string_block(rows, 3);
    tdc_codec_spec spec = {0};
    spec.model = TDC_MODEL_DICT_1D;
    int rc = rt_varlen("DICT_1D all-empty strings (n>0, heap=0)", &sb, &spec);
    free_string_block(&sb);
    return rc;
}

static int test_single_empty_string(void) {
    /* The minimal trigger: n=1, heap_bytes=0. */
    static const char *rows[1] = { "" };
    string_block sb = make_string_block(rows, 1);
    tdc_codec_spec spec = {0};
    spec.model = TDC_MODEL_DICT_1D;
    int rc = rt_varlen("DICT_1D single empty string (n=1, heap=0)", &sb, &spec);
    free_string_block(&sb);
    return rc;
}

static int test_empty_block(void) {
    string_block sb = {0};
    sb.block.dtype       = TDC_DT_STRING;
    sb.block.layout      = TDC_LAYOUT_VECTOR_1D;
    sb.block.shape.rank  = 1;
    sb.block.shape.dim[0] = 0;
    sb.block.shape.stride[0] = 1;
    /* data and offsets stay NULL — block_validate allows this for n==0. */
    tdc_codec_spec spec = {0};
    spec.model = TDC_MODEL_DICT_1D;
    return rt_varlen("DICT_1D empty block (n=0)", &sb, &spec);
}

/* ----- Negative tests ---------------------------------------------------- */

static int test_negatives(void) {
    /* Encode a small valid string block to drive the rejection paths. */
    static const char *rows[3] = { "a", "b", "a" };
    string_block sb = make_string_block(rows, 3);
    tdc_codec_spec spec = {0};
    spec.model = TDC_MODEL_DICT_1D;

    tdc_buffer enc = make_plain_buffer();
    tdc_status st  = tdc_encode_block(&sb.block, &spec, &enc);
    ASSERT_OR_DIE(st == TDC_OK, "setup encode");

    alloc_stats stats = {0};
    tdc_buffer  alloc = make_alloc(&stats);

    /* Build a meta dst from the encoded record. */
    tdc_block ok = {0};
    ok.dtype       = TDC_DT_STRING;
    ok.layout      = TDC_LAYOUT_VECTOR_1D;
    ok.shape.rank  = 1;
    ok.shape.dim[0] = 3;
    ok.shape.stride[0] = 1;

    /* 1. NULL src. */
    {
        tdc_block dst = ok;
        st = tdc_decode_block_varlen(NULL, enc.size, &dst, &alloc);
        ASSERT_OR_DIE(st == TDC_E_INVAL, "NULL src");
    }
    /* 2. NULL dst. */
    {
        st = tdc_decode_block_varlen(enc.data, enc.size, NULL, &alloc);
        ASSERT_OR_DIE(st == TDC_E_INVAL, "NULL dst");
    }
    /* 3. NULL alloc. */
    {
        tdc_block dst = ok;
        st = tdc_decode_block_varlen(enc.data, enc.size, &dst, NULL);
        ASSERT_OR_DIE(st == TDC_E_INVAL, "NULL alloc");
    }
    /* 4. Missing realloc_fn. */
    {
        tdc_buffer bad = {0};   /* realloc_fn == NULL */
        tdc_block  dst = ok;
        st = tdc_decode_block_varlen(enc.data, enc.size, &dst, &bad);
        ASSERT_OR_DIE(st == TDC_E_INVAL, "missing realloc_fn");
    }
    /* 5. Fixed-width dtype rejected. */
    {
        tdc_block dst = ok;
        dst.dtype = TDC_DT_I32;
        st = tdc_decode_block_varlen(enc.data, enc.size, &dst, &alloc);
        ASSERT_OR_DIE(st == TDC_E_DTYPE,
                      "fixed-width dtype must be TDC_E_DTYPE");
    }
    /* 6. Pre-populated dst->data rejected. */
    {
        uint8_t  fake_heap[4] = { 0 };
        tdc_block dst = ok;
        dst.data = fake_heap;
        st = tdc_decode_block_varlen(enc.data, enc.size, &dst, &alloc);
        ASSERT_OR_DIE(st == TDC_E_INVAL, "non-NULL dst->data");
    }
    /* 7. Pre-populated dst->offsets rejected. */
    {
        uint32_t fake_offs[4] = { 0 };
        tdc_block dst = ok;
        dst.offsets = fake_offs;
        st = tdc_decode_block_varlen(enc.data, enc.size, &dst, &alloc);
        ASSERT_OR_DIE(st == TDC_E_INVAL, "non-NULL dst->offsets");
    }
    /* 8. Header dtype mismatch (record says STRING, dst says STRING but
     *    layout disagrees). */
    {
        tdc_block dst = ok;
        dst.layout = TDC_LAYOUT_RASTER_2D;
        st = tdc_decode_block_varlen(enc.data, enc.size, &dst, &alloc);
        ASSERT_OR_DIE(st == TDC_E_LAYOUT, "layout mismatch");
    }

    /* All rejection paths should leave the allocator empty. */
    ASSERT_OR_DIE(stats.live_blocks == 0,
                  "rejections must not leak allocations");

    printf("  [negatives] OK\n");
    free_string_block(&sb);
    free_plain_buffer(&enc);
    return 0;
}

/* ----- main -------------------------------------------------------------- */

int main(void) {
    printf("test_decode_varlen\n");
    if (test_low_cardinality())     return 1;
    if (test_all_unique())          return 1;
    if (test_embedded_zeros())      return 1;
    if (test_empty_strings_mixed()) return 1;
    if (test_all_empty_strings())   return 1;
    if (test_single_empty_string()) return 1;
    if (test_empty_block())         return 1;
    if (test_negatives())           return 1;
    printf("ALL OK\n");
    return 0;
}
