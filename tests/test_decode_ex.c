/*
 * tests/test_decode_ex.c
 *
 * Round-trip tests for tdc_decode_block_ex — the allocator-aware decode
 * entry point.
 *
 * Strategy:
 *   1. Encode a block with tdc_encode_block (which uses its own allocator).
 *   2. Decode with tdc_decode_block_ex using a custom allocator that
 *      tracks allocation counts.
 *   3. Verify decoded data matches the source byte-for-byte.
 *   4. Verify the custom allocator was actually invoked (not bypassed).
 *
 * Covers:
 *   - RAW model, no entropy (simplest pipeline)
 *   - RAW model + LZ2 entropy
 *   - DELTA_1D + ZIGZAG + LZ2 (multi-stage pipeline)
 *   - Empty block (n == 0)
 *   - Rejection: NULL scratch, NULL realloc_fn
 */

#include "tdc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ----- Custom allocator with tracking ----------------------------------- */

typedef struct {
    int alloc_count;
    int free_count;
} alloc_tracker;

static void *tracking_realloc(void *user, void *ptr, size_t new_size) {
    alloc_tracker *t = (alloc_tracker *)user;
    if (new_size == 0) {
        t->free_count++;
        free(ptr);
        return NULL;
    }
    if (ptr == NULL) t->alloc_count++;
    return realloc(ptr, new_size);
}

/* ----- Standard allocator for encode side ------------------------------- */

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

/* ----- Extern for the _ex function -------------------------------------- */

/* Declared here rather than including the integration header, since that
 * header is an internal reference doc, not a build dependency. */
tdc_status tdc_decode_block_ex(const uint8_t *src, size_t src_size,
                               tdc_block *dst, tdc_buffer *scratch);

#define ASSERT_OR_DIE(cond, msg) do {                                       \
    if (!(cond)) {                                                          \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);   \
        return 1;                                                           \
    }                                                                       \
} while (0)

/* ----- Generic round-trip helper via decode_ex -------------------------- */

static int rt_ex(const char           *label,
                 const tdc_block      *src,
                 const tdc_codec_spec *spec) {
    tdc_buffer enc = make_buffer();

    tdc_status st = tdc_encode_block(src, spec, &enc);
    if (st != TDC_OK) {
        fprintf(stderr, "FAIL [%s]: encode -> %d\n", label, (int)st);
        free_buffer(&enc);
        return 1;
    }

    ASSERT_OR_DIE(enc.size >= TDC_BLOCK_HEADER_SIZE,
                  "encoded size < header");

    /* Build destination block: same envelope, fresh data. */
    int64_t n = 1;
    for (uint8_t i = 0; i < src->shape.rank; ++i) n *= src->shape.dim[i];
    size_t elem  = tdc_dtype_size(src->dtype);
    size_t bytes = (size_t)n * elem;

    void *dst_data = (bytes > 0) ? malloc(bytes) : NULL;
    if (bytes > 0) {
        ASSERT_OR_DIE(dst_data != NULL, "malloc dst");
        memset(dst_data, 0xCD, bytes);
    }

    tdc_block dst = {0};
    dst.data    = dst_data;
    dst.dtype   = src->dtype;
    dst.layout  = src->layout;
    dst.shape   = src->shape;
    dst.offsets = NULL;

    /* Decode with tracking allocator. */
    alloc_tracker tracker = {0, 0};
    tdc_buffer scratch = {0};
    scratch.realloc_fn = tracking_realloc;
    scratch.user       = &tracker;

    st = tdc_decode_block_ex(enc.data, enc.size, &dst, &scratch);
    if (st != TDC_OK) {
        fprintf(stderr, "FAIL [%s]: decode_ex -> %d\n", label, (int)st);
        free(dst_data);
        free_buffer(&enc);
        return 1;
    }

    /* Verify allocator was used (at least 2 ping-pong buffers allocated
     * and freed — unless the block is empty, in which case we still
     * allocate at least the 1-byte sentinel). */
    ASSERT_OR_DIE(tracker.alloc_count >= 1,
                  "custom allocator should have been called");
    ASSERT_OR_DIE(tracker.free_count >= 1,
                  "custom allocator should have freed scratch");

    /* Verify data. */
    if (bytes > 0) {
        ASSERT_OR_DIE(memcmp(dst_data, src->data, bytes) == 0,
                      "decoded data does not match source");
    }

    fprintf(stdout, "  %-55s OK  allocs=%d frees=%d\n",
            label, tracker.alloc_count, tracker.free_count);
    free(dst_data);
    free_buffer(&enc);
    return 0;
}

/* ----- Test cases ------------------------------------------------------- */

static int test_raw_passthrough(void) {
    int32_t data[16];
    for (int i = 0; i < 16; ++i) data[i] = i * 1000 - 7000;

    tdc_block blk = {0};
    blk.data   = data;
    blk.dtype  = TDC_DT_I32;
    blk.layout = TDC_LAYOUT_VECTOR_1D;
    blk.shape.rank   = 1;
    blk.shape.dim[0] = 16;
    blk.shape.stride[0] = 1;

    tdc_codec_spec spec = tdc_codec_spec_raw();
    return rt_ex("RAW passthrough (no entropy)", &blk, &spec);
}

static int test_raw_lz2(void) {
    /* Highly compressible: repeated pattern. */
    int32_t data[256];
    for (int i = 0; i < 256; ++i) data[i] = i % 4;

    tdc_block blk = {0};
    blk.data   = data;
    blk.dtype  = TDC_DT_I32;
    blk.layout = TDC_LAYOUT_VECTOR_1D;
    blk.shape.rank   = 1;
    blk.shape.dim[0] = 256;
    blk.shape.stride[0] = 1;

    tdc_codec_spec spec = tdc_codec_spec_raw();
    spec.entropy[0] = TDC_ENTROPY_LZ2;
    return rt_ex("RAW + LZ2", &blk, &spec);
}

static int test_delta_zigzag_lz2(void) {
    /* Linear ramp: DELTA_1D produces near-constant residuals, ZIGZAG maps
     * them to small unsigned values, LZ2 compresses the repetition. */
    int32_t data[128];
    for (int i = 0; i < 128; ++i) data[i] = i * 7 + 3;

    tdc_block blk = {0};
    blk.data   = data;
    blk.dtype  = TDC_DT_I32;
    blk.layout = TDC_LAYOUT_VECTOR_1D;
    blk.shape.rank   = 1;
    blk.shape.dim[0] = 128;
    blk.shape.stride[0] = 1;

    tdc_codec_spec spec = {0};
    spec.model      = TDC_MODEL_DELTA_1D;
    spec.xform[0]   = TDC_XFORM_ZIGZAG;
    spec.xform[1]   = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0] = TDC_ENTROPY_LZ2;
    return rt_ex("DELTA_1D + ZIGZAG + BSHUF + LZ2", &blk, &spec);
}

static int test_empty_block(void) {
    tdc_block blk = {0};
    blk.data   = NULL;
    blk.dtype  = TDC_DT_I32;
    blk.layout = TDC_LAYOUT_VECTOR_1D;
    blk.shape.rank   = 1;
    blk.shape.dim[0] = 0;
    blk.shape.stride[0] = 1;

    tdc_codec_spec spec = tdc_codec_spec_raw();
    return rt_ex("empty block (n=0)", &blk, &spec);
}

static int test_f64_raw_lz2(void) {
    /* Float data with byte shuffle to expose lane structure. */
    double data[64];
    for (int i = 0; i < 64; ++i) data[i] = (double)i * 0.5;

    tdc_block blk = {0};
    blk.data   = data;
    blk.dtype  = TDC_DT_F64;
    blk.layout = TDC_LAYOUT_VECTOR_1D;
    blk.shape.rank   = 1;
    blk.shape.dim[0] = 64;
    blk.shape.stride[0] = 1;

    tdc_codec_spec spec = tdc_codec_spec_raw();
    spec.xform[0]   = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0] = TDC_ENTROPY_LZ2;
    return rt_ex("f64 RAW + BSHUF + LZ2", &blk, &spec);
}

/* ----- Rejection tests -------------------------------------------------- */

static int test_rejections(void) {
    /* Encode something valid first. */
    int32_t data[4] = { 1, 2, 3, 4 };
    tdc_block blk = {0};
    blk.data   = data;
    blk.dtype  = TDC_DT_I32;
    blk.layout = TDC_LAYOUT_VECTOR_1D;
    blk.shape.rank   = 1;
    blk.shape.dim[0] = 4;
    blk.shape.stride[0] = 1;

    tdc_buffer enc = make_buffer();
    tdc_codec_spec spec = tdc_codec_spec_raw();
    tdc_status st = tdc_encode_block(&blk, &spec, &enc);
    ASSERT_OR_DIE(st == TDC_OK, "setup encode");

    int32_t dst_data[4] = {0};
    tdc_block dst = blk;
    dst.data = dst_data;

    /* NULL scratch */
    st = tdc_decode_block_ex(enc.data, enc.size, &dst, NULL);
    ASSERT_OR_DIE(st == TDC_E_INVAL, "NULL scratch should be rejected");

    /* scratch with NULL realloc_fn */
    {
        tdc_buffer bad_scratch = {0};
        st = tdc_decode_block_ex(enc.data, enc.size, &dst, &bad_scratch);
        ASSERT_OR_DIE(st == TDC_E_INVAL, "NULL realloc_fn should be rejected");
    }

    /* NULL src */
    {
        alloc_tracker tracker = {0, 0};
        tdc_buffer scratch = {0};
        scratch.realloc_fn = tracking_realloc;
        scratch.user       = &tracker;
        st = tdc_decode_block_ex(NULL, enc.size, &dst, &scratch);
        ASSERT_OR_DIE(st == TDC_E_INVAL, "NULL src should be rejected");
    }

    /* NULL dst */
    {
        alloc_tracker tracker = {0, 0};
        tdc_buffer scratch = {0};
        scratch.realloc_fn = tracking_realloc;
        scratch.user       = &tracker;
        st = tdc_decode_block_ex(enc.data, enc.size, NULL, &scratch);
        ASSERT_OR_DIE(st == TDC_E_INVAL, "NULL dst should be rejected");
    }

    printf("  [rejections] OK\n");
    free_buffer(&enc);
    return 0;
}

/* ----- main ------------------------------------------------------------- */

int main(void) {
    printf("test_decode_ex\n");
    if (test_raw_passthrough())    return 1;
    if (test_raw_lz2())           return 1;
    if (test_delta_zigzag_lz2())  return 1;
    if (test_empty_block())       return 1;
    if (test_f64_raw_lz2())       return 1;
    if (test_rejections())        return 1;
    printf("ALL OK\n");
    return 0;
}
