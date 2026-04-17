/*
 * tests/test_decode_into.c
 *
 * Round-trip tests for tdc_decode_block_into — the zero-allocation
 * decode entry point that writes into caller-provided dst->data.
 *
 * Strategy for each case:
 *   1. Encode a block with tdc_encode_block.
 *   2. Call tdc_decode_peek() on the bytes to discover dtype / layout /
 *      shape / required byte count.
 *   3. Allocate dst->data via plain malloc at the required size.
 *   4. Populate the rest of dst from peek, call tdc_decode_block_into,
 *      and memcmp the result against the original.
 *
 * Covers:
 *   - RAW passthrough, f64 VECTOR_1D
 *   - DELTA_1D + ZIGZAG + BSHUF + LZ, i64 VECTOR_1D
 *   - 3D volume with PRED_3D
 *   - Empty block
 *   - Negative: NULL src, NULL dst, NULL dst->data on non-empty block,
 *     dtype mismatch, shape mismatch
 */

#include "tdc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ----- Helpers ------------------------------------------------------------ */

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

/* ----- Generic round-trip via peek + decode_into ------------------------- */

static int rt_into(const char           *label,
                   const tdc_block      *src,
                   const tdc_codec_spec *spec) {
    tdc_buffer enc = make_buffer();
    tdc_status st = tdc_encode_block(src, spec, &enc);
    if (st != TDC_OK) {
        fprintf(stderr, "FAIL [%s]: encode -> %d\n", label, (int)st);
        free_buffer(&enc);
        return 1;
    }
    ASSERT_OR_DIE(enc.size >= TDC_BLOCK_HEADER_SIZE, "encoded size < header");

    /* Step 1: peek. */
    tdc_block meta = {0};
    size_t    need = 0;
    st = tdc_decode_peek(enc.data, enc.size, &meta, &need);
    if (st != TDC_OK) {
        fprintf(stderr, "FAIL [%s]: peek -> %d\n", label, (int)st);
        free_buffer(&enc);
        return 1;
    }
    ASSERT_OR_DIE(meta.dtype  == src->dtype,  "peek dtype mismatch");
    ASSERT_OR_DIE(meta.layout == src->layout, "peek layout mismatch");
    ASSERT_OR_DIE(meta.shape.rank == src->shape.rank, "peek rank mismatch");
    for (uint8_t i = 0; i < src->shape.rank; ++i) {
        ASSERT_OR_DIE(meta.shape.dim[i] == src->shape.dim[i],
                      "peek dim mismatch");
    }

    int64_t n = 1;
    for (uint8_t i = 0; i < src->shape.rank; ++i) n *= src->shape.dim[i];
    size_t elem     = tdc_dtype_size(src->dtype);
    size_t expected = (n > 0) ? (size_t)n * elem : 0u;
    ASSERT_OR_DIE(need == expected, "peek bytes_required mismatch");

    /* Step 2: allocate caller buffer at exactly the required size. */
    void *dst_data = (need > 0) ? malloc(need) : NULL;
    if (need > 0) {
        ASSERT_OR_DIE(dst_data != NULL, "malloc dst");
        memset(dst_data, 0xCD, need);
    }

    /* Step 3: build dst from peek result and decode in place. */
    tdc_block dst = meta;
    dst.data = dst_data;

    st = tdc_decode_block_into(enc.data, enc.size, &dst);
    if (st != TDC_OK) {
        fprintf(stderr, "FAIL [%s]: decode_into -> %d\n", label, (int)st);
        free(dst_data);
        free_buffer(&enc);
        return 1;
    }

    /* dst.data pointer must be unchanged — the "in place" contract. */
    ASSERT_OR_DIE(dst.data == dst_data, "decode_into rewrote dst->data");

    if (need > 0) {
        ASSERT_OR_DIE(memcmp(dst_data, src->data, need) == 0,
                      "decoded data does not match source");
    }

    fprintf(stdout, "  %-55s OK  bytes=%zu\n", label, need);
    free(dst_data);
    free_buffer(&enc);
    return 0;
}

/* ----- Test cases -------------------------------------------------------- */

static int test_raw_f64(void) {
    double data[32];
    for (int i = 0; i < 32; ++i) data[i] = (double)i * 0.25 - 4.0;

    tdc_block blk = {0};
    blk.data   = data;
    blk.dtype  = TDC_DT_F64;
    blk.layout = TDC_LAYOUT_VECTOR_1D;
    blk.shape.rank   = 1;
    blk.shape.dim[0] = 32;
    blk.shape.stride[0] = 1;

    tdc_codec_spec spec = tdc_codec_spec_raw();
    return rt_into("RAW f64 VECTOR_1D (passthrough)", &blk, &spec);
}

static int test_delta_zigzag_bshuf_lz_i64(void) {
    /* Near-linear i64 ramp: DELTA_1D -> near-constant residuals. */
    int64_t data[200];
    for (int i = 0; i < 200; ++i) data[i] = (int64_t)i * 131 - 500000;

    tdc_block blk = {0};
    blk.data   = data;
    blk.dtype  = TDC_DT_I64;
    blk.layout = TDC_LAYOUT_VECTOR_1D;
    blk.shape.rank   = 1;
    blk.shape.dim[0] = 200;
    blk.shape.stride[0] = 1;

    tdc_codec_spec spec = {0};
    spec.model      = TDC_MODEL_DELTA_1D;
    spec.xform[0]   = TDC_XFORM_ZIGZAG;
    spec.xform[1]   = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0] = TDC_ENTROPY_LZ;
    return rt_into("DELTA_1D + ZIGZAG + BSHUF + LZ (i64)", &blk, &spec);
}

static int test_pred3d_volume(void) {
    /* Tri-affine 3D field; PRED_3D GRAD3D should reduce to near-zero
     * residuals. */
    enum { D = 6, H = 8, W = 10 };
    int32_t data[D * H * W];
    for (int z = 0; z < D; ++z) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                data[z * H * W + y * W + x] = 3 * x + 5 * y + 7 * z + 17;
            }
        }
    }

    tdc_block blk = {0};
    blk.data   = data;
    blk.dtype  = TDC_DT_I32;
    blk.layout = TDC_LAYOUT_VOLUME_3D;
    blk.shape.rank   = 3;
    blk.shape.dim[0] = D;
    blk.shape.dim[1] = H;
    blk.shape.dim[2] = W;
    tdc_shape_set_contiguous(&blk.shape);

    tdc_pred3d_params pp = { .kind = TDC_PRED3D_GRAD3D };
    tdc_codec_spec spec = {0};
    spec.model        = TDC_MODEL_PRED_3D;
    spec.model_params = &pp;
    spec.xform[0]     = TDC_XFORM_ZIGZAG;
    spec.xform[1]     = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0]   = TDC_ENTROPY_LZ;
    return rt_into("PRED_3D + ZIGZAG + BSHUF + LZ (i32 volume)", &blk, &spec);
}

static int test_empty(void) {
    tdc_block blk = {0};
    blk.data   = NULL;
    blk.dtype  = TDC_DT_I32;
    blk.layout = TDC_LAYOUT_VECTOR_1D;
    blk.shape.rank   = 1;
    blk.shape.dim[0] = 0;
    blk.shape.stride[0] = 1;

    tdc_codec_spec spec = tdc_codec_spec_raw();
    return rt_into("empty block (n=0)", &blk, &spec);
}

/* ----- Negative tests ---------------------------------------------------- */

static int test_negatives(void) {
    /* Encode a small valid block to exercise the rejection paths. */
    int32_t data[4] = { 10, 20, 30, 40 };
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
    tdc_block dst_ok = blk;
    dst_ok.data = dst_data;

    /* 1. NULL src. */
    st = tdc_decode_block_into(NULL, enc.size, &dst_ok);
    ASSERT_OR_DIE(st == TDC_E_INVAL, "NULL src should be TDC_E_INVAL");

    /* 2. NULL dst. */
    st = tdc_decode_block_into(enc.data, enc.size, NULL);
    ASSERT_OR_DIE(st == TDC_E_INVAL, "NULL dst should be TDC_E_INVAL");

    /* 3. NULL dst->data on non-empty block. */
    {
        tdc_block dst = blk;
        dst.data = NULL;
        st = tdc_decode_block_into(enc.data, enc.size, &dst);
        ASSERT_OR_DIE(st == TDC_E_INVAL,
                      "NULL dst->data should be TDC_E_INVAL");
    }

    /* 4. dtype mismatch. */
    {
        tdc_block dst = blk;
        dst.data  = dst_data;
        dst.dtype = TDC_DT_I16;  /* record says I32 */
        st = tdc_decode_block_into(enc.data, enc.size, &dst);
        ASSERT_OR_DIE(st == TDC_E_DTYPE,
                      "dtype mismatch should be TDC_E_DTYPE");
    }

    /* 5. layout mismatch. */
    {
        tdc_block dst = blk;
        dst.data   = dst_data;
        dst.layout = TDC_LAYOUT_RASTER_2D;  /* record says VECTOR_1D */
        st = tdc_decode_block_into(enc.data, enc.size, &dst);
        ASSERT_OR_DIE(st == TDC_E_LAYOUT,
                      "layout mismatch should be TDC_E_LAYOUT");
    }

    /* 6. shape mismatch. */
    {
        tdc_block dst = blk;
        dst.data = dst_data;
        dst.shape.dim[0] = 8;  /* record says 4 */
        st = tdc_decode_block_into(enc.data, enc.size, &dst);
        ASSERT_OR_DIE(st == TDC_E_SHAPE,
                      "shape mismatch should be TDC_E_SHAPE");
    }

    /* 7. Truncated src (below header size). */
    {
        st = tdc_decode_block_into(enc.data, 16u, &dst_ok);
        ASSERT_OR_DIE(st == TDC_E_CORRUPT,
                      "truncated src should be TDC_E_CORRUPT");
    }

    /* 8. tdc_decode_peek: NULL out_meta. */
    {
        st = tdc_decode_peek(enc.data, enc.size, NULL, NULL);
        ASSERT_OR_DIE(st == TDC_E_INVAL,
                      "peek NULL out_meta should be TDC_E_INVAL");
    }

    /* 9. tdc_decode_peek: truncated src. */
    {
        tdc_block meta;
        st = tdc_decode_peek(enc.data, 16u, &meta, NULL);
        ASSERT_OR_DIE(st == TDC_E_CORRUPT,
                      "peek truncated src should be TDC_E_CORRUPT");
    }

    printf("  [negatives] OK\n");
    free_buffer(&enc);
    return 0;
}

/* ----- main -------------------------------------------------------------- */

int main(void) {
    printf("test_decode_into\n");
    if (test_raw_f64())                  return 1;
    if (test_delta_zigzag_bshuf_lz_i64()) return 1;
    if (test_pred3d_volume())            return 1;
    if (test_empty())                    return 1;
    if (test_negatives())                return 1;
    printf("ALL OK\n");
    return 0;
}
