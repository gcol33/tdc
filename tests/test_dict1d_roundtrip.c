/*
 * tests/test_dict1d_roundtrip.c
 *
 * Round-trip test for TDC_MODEL_DICT_1D.
 *
 * Verifies:
 *   1. Low-cardinality input round-trips bit-exact and the dictionary
 *      really shrinks (n_unique << n).
 *   2. All-unique input round-trips correctly (worst case: indices are
 *      the monotone sequence 0..n-1, dict has n entries).
 *   3. Single-row input round-trips.
 *   4. Empty input (n == 0) round-trips and emits an 8-byte side meta
 *      header with both fields zero.
 *   5. Strings of length 0 are allowed and survive a round trip
 *      (offsets[i] == offsets[i+1] for that row).
 *   6. residual_dtype == TDC_DT_U32 unconditionally.
 *   7. Encoder rejects non-STRING dtype, non-VECTOR_1D layout.
 *   8. Decoder rejects truncated side meta and out-of-range indices.
 *   9. Registry returns &tdc_model_dict1d_vt for TDC_MODEL_DICT_1D.
 */

#include "tdc/model.h"
#include "tdc/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern const tdc_model_vt tdc_model_dict1d_vt;

/* ----- Test plumbing ------------------------------------------------------ */

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
    b->data = NULL;
    b->size = 0;
    b->capacity = 0;
}

#define ASSERT_OR_DIE(cond, msg) do {                                       \
    if (!(cond)) {                                                          \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);   \
        return 1;                                                           \
    }                                                                       \
} while (0)

/* Build a TDC_DT_STRING / VECTOR_1D block from an array of C strings.
 * Allocates a packed heap and offsets table that the caller must free
 * via free_string_block(). */
typedef struct {
    tdc_block block;
    uint8_t  *heap;
    uint32_t *offsets;
} string_block;

static string_block make_string_block(const char *const *rows, int64_t n) {
    string_block sb = {0};
    sb.offsets = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)(n + 1));
    uint32_t total = 0;
    for (int64_t i = 0; i < n; ++i) {
        sb.offsets[i] = total;
        total += (uint32_t)strlen(rows[i]);
    }
    sb.offsets[n] = total;

    sb.heap = (uint8_t *)malloc(total > 0 ? total : 1u);
    for (int64_t i = 0; i < n; ++i) {
        size_t len = (size_t)(sb.offsets[i + 1] - sb.offsets[i]);
        if (len > 0) memcpy(sb.heap + sb.offsets[i], rows[i], len);
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

static void free_string_block(string_block *sb) {
    if (sb->heap)    free(sb->heap);
    if (sb->offsets) free(sb->offsets);
    sb->heap = NULL;
    sb->offsets = NULL;
}

/* Compare two STRING blocks for value equality. */
static int string_blocks_equal(const tdc_block *a, const tdc_block *b) {
    if (a->shape.dim[0] != b->shape.dim[0]) return 0;
    int64_t n = a->shape.dim[0];
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

/* Encode-then-decode helper. Allocates a generously-sized output heap
 * (sum of input string lengths is the upper bound). */
static int rt_strings(const char *label, const char *const *rows, int64_t n) {
    const tdc_model_vt *vt = &tdc_model_dict1d_vt;

    string_block in = make_string_block(rows, n);

    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;

    tdc_status st = vt->encode(&in.block, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "encode returned non-OK");
    ASSERT_OR_DIE(rdt == TDC_DT_U32, "residual_dtype must be u32");
    if (n > 0) {
        ASSERT_OR_DIE(residual.size == (size_t)n * 4u,
                      "residual size must be n*4");
    }
    /* Side meta header is at minimum 8 bytes (count + total). */
    ASSERT_OR_DIE(side.size >= 8u, "side meta too small");

    /* Allocate decode buffers. Sized as the worst case (input heap
     * total bytes) — the model writes <= that many bytes. */
    uint32_t in_total = in.offsets[n];
    string_block out = {0};
    out.offsets = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)(n + 1));
    out.heap    = (uint8_t *)malloc(in_total > 0 ? in_total : 1u);
    out.block.data        = out.heap;
    out.block.offsets     = out.offsets;
    out.block.dtype       = TDC_DT_STRING;
    out.block.layout      = TDC_LAYOUT_VECTOR_1D;
    out.block.shape.rank  = 1;
    out.block.shape.dim[0] = n;
    out.block.shape.stride[0] = 1;

    st = vt->decode(&out.block, NULL, rdt,
                    residual.data, residual.size,
                    side.data, side.size);
    ASSERT_OR_DIE(st == TDC_OK, "decode returned non-OK");

    ASSERT_OR_DIE(string_blocks_equal(&in.block, &out.block),
                  "decoded strings != source");

    /* Pull dict_count from side meta for the report. */
    uint32_t dict_count = 0u;
    memcpy(&dict_count, side.data, 4u);
    printf("  [%s] n=%lld unique=%u side=%zu residual=%zu OK\n",
           label, (long long)n, dict_count, side.size, residual.size);

    free_string_block(&in);
    free_string_block(&out);
    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

/* ----- Cases -------------------------------------------------------------- */

static int test_low_cardinality(void) {
    static const char *rows[12] = {
        "alpha", "beta", "alpha", "gamma",
        "beta",  "alpha", "alpha", "delta",
        "delta", "beta",  "gamma", "alpha"
    };
    /* Sanity: with 4 unique strings the dictionary should have 4 entries. */
    const tdc_model_vt *vt = &tdc_model_dict1d_vt;
    string_block in = make_string_block(rows, 12);
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_status st = vt->encode(&in.block, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "encode");
    uint32_t dict_count = 0;
    memcpy(&dict_count, side.data, 4u);
    ASSERT_OR_DIE(dict_count == 4u, "expected 4 unique strings");
    free_string_block(&in);
    free_buffer(&residual);
    free_buffer(&side);

    return rt_strings("low cardinality", rows, 12);
}

static int test_all_unique(void) {
    /* 600 unique rows: forces at least one hash table resize (initial
     * capacity 256, doubles at 70% load). */
    enum { N = 600 };
    char  storage[N][16];
    const char *rows[N];
    for (int i = 0; i < N; ++i) {
        snprintf(storage[i], sizeof storage[i], "row_%05d", i);
        rows[i] = storage[i];
    }
    return rt_strings("all unique 600", rows, N);
}

static int test_single_row(void) {
    static const char *rows[1] = { "only" };
    return rt_strings("single row", rows, 1);
}

static int test_empty_strings(void) {
    static const char *rows[5] = { "", "x", "", "x", "" };
    return rt_strings("empty strings", rows, 5);
}

static int test_empty_block(void) {
    const tdc_model_vt *vt = &tdc_model_dict1d_vt;
    /* n == 0: data and offsets may be NULL per the block contract. */
    tdc_block in = {0};
    in.dtype       = TDC_DT_STRING;
    in.layout      = TDC_LAYOUT_VECTOR_1D;
    in.shape.rank  = 1;
    in.shape.dim[0] = 0;
    in.shape.stride[0] = 1;

    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_status st = vt->encode(&in, NULL, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "empty encode");
    ASSERT_OR_DIE(rdt == TDC_DT_U32, "empty residual_dtype");
    ASSERT_OR_DIE(side.size == 12u,
                  "empty side: 8-byte header + offsets[1] sentinel");
    uint32_t count = 0xDEADBEEFu, total = 0xDEADBEEFu;
    memcpy(&count, side.data + 0, 4u);
    memcpy(&total, side.data + 4, 4u);
    ASSERT_OR_DIE(count == 0u && total == 0u, "empty header zero fields");
    ASSERT_OR_DIE(residual.size == 0u, "empty residual");

    /* Decode the empty block back. */
    uint32_t out_offsets[1] = { 0xFFFFFFFFu };
    tdc_block out = {0};
    out.dtype = TDC_DT_STRING;
    out.layout = TDC_LAYOUT_VECTOR_1D;
    out.shape.rank = 1;
    out.shape.dim[0] = 0;
    out.shape.stride[0] = 1;
    out.offsets = out_offsets;
    st = vt->decode(&out, NULL, TDC_DT_U32, NULL, 0, side.data, side.size);
    ASSERT_OR_DIE(st == TDC_OK, "empty decode");
    ASSERT_OR_DIE(out_offsets[0] == 0u, "empty decode offsets[0] == 0");

    printf("  [empty block] OK\n");
    free_buffer(&residual);
    free_buffer(&side);
    return 0;
}

static int test_rejections(void) {
    const tdc_model_vt *vt = &tdc_model_dict1d_vt;

    /* Wrong dtype */
    {
        int32_t data[3] = { 1, 2, 3 };
        tdc_block in = {0};
        in.data = data;
        in.dtype = TDC_DT_I32;
        in.layout = TDC_LAYOUT_VECTOR_1D;
        in.shape.rank = 1;
        in.shape.dim[0] = 3;
        in.shape.stride[0] = 1;

        tdc_buffer residual = make_buffer();
        tdc_buffer side     = make_buffer();
        tdc_dtype  rdt      = (tdc_dtype)0;
        tdc_status st = vt->encode(&in, NULL, &residual, &rdt, &side);
        ASSERT_OR_DIE(st == TDC_E_DTYPE, "non-STRING must be TDC_E_DTYPE");
        free_buffer(&residual);
        free_buffer(&side);
    }

    /* Wrong layout */
    {
        static const char *rows[4] = { "a", "b", "c", "d" };
        string_block sb = make_string_block(rows, 4);
        sb.block.layout = TDC_LAYOUT_RASTER_2D;
        sb.block.shape.rank = 2;
        sb.block.shape.dim[0] = 2;
        sb.block.shape.dim[1] = 2;

        tdc_buffer residual = make_buffer();
        tdc_buffer side     = make_buffer();
        tdc_dtype  rdt      = (tdc_dtype)0;
        tdc_status st = vt->encode(&sb.block, NULL, &residual, &rdt, &side);
        ASSERT_OR_DIE(st == TDC_E_LAYOUT, "RASTER_2D must be TDC_E_LAYOUT");
        free_string_block(&sb);
        free_buffer(&residual);
        free_buffer(&side);
    }

    /* Decode rejects truncated side meta. */
    {
        tdc_block out = {0};
        out.dtype = TDC_DT_STRING;
        out.layout = TDC_LAYOUT_VECTOR_1D;
        out.shape.rank = 1;
        out.shape.dim[0] = 1;
        out.shape.stride[0] = 1;
        uint8_t side_short[4] = { 0 };
        tdc_status st = vt->decode(&out, NULL, TDC_DT_U32,
                                   NULL, 0, side_short, 4);
        ASSERT_OR_DIE(st == TDC_E_CORRUPT,
                      "decode must reject side_size < 8");
    }

    /* Decode rejects an out-of-range index. */
    {
        /* Side meta: count=2, total=2, offsets=[0,1,2], heap="ab". */
        uint8_t side_meta[8 + 12 + 2];
        uint32_t cnt = 2u, tot = 2u;
        memcpy(side_meta + 0, &cnt, 4u);
        memcpy(side_meta + 4, &tot, 4u);
        uint32_t offs[3] = { 0u, 1u, 2u };
        memcpy(side_meta + 8, offs, sizeof offs);
        side_meta[20] = 'a';
        side_meta[21] = 'b';

        /* Residual asks for index 5 — past the end. */
        uint32_t bad_idx = 5u;
        uint8_t residual[4];
        memcpy(residual, &bad_idx, 4u);

        uint32_t out_offsets[2] = { 0u };
        uint8_t  out_data[16] = { 0 };
        tdc_block out = {0};
        out.data = out_data;
        out.offsets = out_offsets;
        out.dtype = TDC_DT_STRING;
        out.layout = TDC_LAYOUT_VECTOR_1D;
        out.shape.rank = 1;
        out.shape.dim[0] = 1;
        out.shape.stride[0] = 1;
        tdc_status st = vt->decode(&out, NULL, TDC_DT_U32,
                                   residual, sizeof residual,
                                   side_meta, sizeof side_meta);
        ASSERT_OR_DIE(st == TDC_E_CORRUPT,
                      "decode must reject out-of-range index");
    }

    printf("  [rejections] OK\n");
    return 0;
}

static int test_registry(void) {
    const tdc_model_vt *vt = tdc_model_get(TDC_MODEL_DICT_1D);
    ASSERT_OR_DIE(vt == &tdc_model_dict1d_vt,
                  "tdc_model_get(TDC_MODEL_DICT_1D) wiring");
    ASSERT_OR_DIE(vt->id == TDC_MODEL_DICT_1D, "vt->id mismatch");
    printf("  [registry wiring] OK\n");
    return 0;
}

int main(void) {
    printf("test_dict1d_roundtrip\n");
    if (test_low_cardinality()) return 1;
    if (test_all_unique())      return 1;
    if (test_single_row())      return 1;
    if (test_empty_strings())   return 1;
    if (test_empty_block())     return 1;
    if (test_rejections())      return 1;
    if (test_registry())        return 1;
    printf("ALL OK\n");
    return 0;
}
