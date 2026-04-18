/* docs/examples/models_edge_cases.c
 *
 * Model edge cases: empty block, single element, all-equal block, and
 * layout mismatch. Each case exercises a corner the main backend
 * walkthrough does not: the encoder still emits an 80-byte block record
 * for an empty block, the zero-residual fast path fires on all-equal
 * inputs, and the dispatcher rejects a layout mismatch with
 * TDC_E_LAYOUT before ever calling the model.
 */

#include "quickstart_common.h"
#include <string.h>

static size_t encode_or_die(const tdc_block *src,
                            const tdc_codec_spec *spec,
                            const char *label) {
    tdc_buffer enc = qs_buffer();
    tdc_status st = tdc_encode_block(src, spec, &enc);
    if (st != TDC_OK) {
        printf("  %-28s error: %s\n", label, tdc_strerror(st));
        qs_buffer_free(&enc);
        return 0;
    }
    size_t n = enc.size;
    printf("  %-28s encoded=%zu bytes\n", label, n);
    qs_buffer_free(&enc);
    return n;
}

static void roundtrip_i32_1d(const char *label, int32_t *data, size_t n,
                             const tdc_codec_spec *spec) {
    tdc_block src = {0};
    src.data = data;
    src.dtype = TDC_DT_I32;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank = 1;
    src.shape.dim[0] = (int64_t)n;
    tdc_shape_set_contiguous(&src.shape);

    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(&src, spec, &enc), label)) {
        qs_buffer_free(&enc);
        return;
    }
    size_t raw = n * sizeof(int32_t);
    printf("  %-28s raw=%4zu encoded=%4zu",
           label, raw, enc.size);

    /* Decode path: honour the n=0 case where tdc_decode_block_into
     * expects dst->data to be non-NULL only when n_elems > 0. For an
     * empty block we still peek the header and skip the copy. */
    tdc_block meta; size_t need = 0;
    tdc_decode_peek(enc.data, enc.size, &meta, &need);
    if (n == 0) {
        printf("  (empty block, no payload to decode)\n");
        qs_buffer_free(&enc);
        return;
    }
    void *dst_data = qs_realloc(NULL, NULL, need);
    tdc_block dst = meta; dst.data = dst_data;
    int ok = tdc_decode_block_into(enc.data, enc.size, &dst) == TDC_OK
          && memcmp(dst_data, data, raw) == 0;
    printf("  memcmp=%s\n", ok ? "ok" : "FAIL");
    qs_realloc(NULL, dst_data, 0);
    qs_buffer_free(&enc);
}

int main(void) {
    tdc_codec_spec delta_lz = {0};
    delta_lz.model = TDC_MODEL_DELTA_1D;
    delta_lz.xform[0] = TDC_XFORM_ZIGZAG;
    delta_lz.xform[1] = TDC_XFORM_BYTE_SHUFFLE;
    delta_lz.entropy[0] = TDC_ENTROPY_LZ;

    /* Case 1: empty block. */
    printf("-- empty block (n=0, i32, VECTOR_1D): --\n");
    roundtrip_i32_1d("DELTA_1D + LZ      ", NULL, 0, &delta_lz);

    /* Case 2: single-element block. */
    int32_t one[1] = { 42 };
    printf("-- single element (n=1): --\n");
    roundtrip_i32_1d("DELTA_1D + LZ      ", one, 1, &delta_lz);

    /* Case 3: all-equal block (zero-residual fast path). */
    enum { N3 = 4096 };
    static int32_t flat[N3];
    for (int i = 0; i < N3; ++i) flat[i] = 12345;
    printf("-- all-equal block (n=%d, zero-residual fast path): --\n", N3);
    roundtrip_i32_1d("DELTA_1D + LZ      ", flat, N3, &delta_lz);

    /* Case 4: layout mismatch. PRED_2D rejects a VECTOR_1D block with
     * TDC_E_LAYOUT before any model code runs. */
    printf("-- layout mismatch (PRED_2D on VECTOR_1D): --\n");
    tdc_block v = {0};
    v.data = flat;
    v.dtype = TDC_DT_I32;
    v.layout = TDC_LAYOUT_VECTOR_1D;
    v.shape.rank = 1;
    v.shape.dim[0] = N3;
    tdc_shape_set_contiguous(&v.shape);
    tdc_pred2d_params p = { .kind = TDC_PRED2D_PAETH };
    tdc_codec_spec pred = {0};
    pred.model = TDC_MODEL_PRED_2D;
    pred.model_params = &p;
    pred.entropy[0] = TDC_ENTROPY_LZ;
    (void)encode_or_die(&v, &pred, "PRED_2D on VECTOR_1D");

    /* Case 5: a 1-row raster — shape is 2D but ny=1, so every pixel's
     * UP neighbor is out of bounds. PRED_2D is still valid; the model
     * degenerates to LEFT internally at the boundary. */
    enum { RW = 256 };
    static uint8_t row[1 * RW];
    for (int i = 0; i < RW; ++i) row[i] = (uint8_t)(i * 3);
    tdc_block r = {0};
    r.data = row;
    r.dtype = TDC_DT_U8;
    r.layout = TDC_LAYOUT_RASTER_2D;
    r.shape.rank = 2;
    r.shape.dim[0] = 1;
    r.shape.dim[1] = RW;
    tdc_shape_set_contiguous(&r.shape);
    printf("-- degenerate raster 1x%d (PRED_2D/PAETH): --\n", RW);
    (void)encode_or_die(&r, &pred, "PAETH on 1xW raster");

    return 0;
}
