/* docs/examples/xforms_chain_compare.c
 *
 * Same i32 input, five chain variants. Measures the byte cost of each
 * transform in the xform[4] array. The model is DELTA_1D so the residual
 * stream has the shape the transform stage is designed for: signed
 * small-magnitude integers with trailing zero bytes.
 *
 * Build:
 *   cc -I include docs/examples/xforms_chain_compare.c \
 *      build/libtdc.a -lm -o /tmp/xf_compare
 */

#include "quickstart_common.h"
#include <stdlib.h>

static void run(const char *label, const tdc_block *src,
                const tdc_codec_spec *spec, size_t raw) {
    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(src, spec, &enc), label)) exit(1);
    double ratio = (double)raw / (double)enc.size;
    printf("  %-40s %6zu bytes  ratio=%.2fx\n", label, enc.size, ratio);
    qs_buffer_free(&enc);
}

int main(void) {
    enum { N = 8192 };
    static int32_t data[N];
    /* i32 values with sign-alternating residuals. Every third pair steps
     * back by a few units, so the DELTA_1D residual stream carries both
     * positive and negative small integers. Positive deltas pass through
     * LZ as short runs; negative deltas are 0xFFFFFF-prefixed in the raw
     * i32 bytes, which ZIGZAG flattens back into small-magnitude bytes. */
    for (int i = 0; i < N; ++i) {
        int step = ((i % 5) < 3) ? 4 : -3;
        data[i] = 500000 + i + step * (i % 7);
    }

    tdc_block src = {0};
    src.data   = data;
    src.dtype  = TDC_DT_I32;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank   = 1;
    src.shape.dim[0] = N;
    tdc_shape_set_contiguous(&src.shape);

    size_t raw = N * sizeof(int32_t);
    printf("input: %zu bytes (%d x i32)\n\n", raw, N);

    /* 1: RAW + LZ. No model, no transforms. LZ sees the raw i32 bytes. */
    tdc_codec_spec s1 = tdc_codec_spec_raw();
    s1.entropy[0] = TDC_ENTROPY_LZ;
    run("RAW + LZ", &src, &s1, raw);

    /* 2: DELTA_1D + LZ. Residuals, LZ. */
    tdc_codec_spec s2 = {0};
    s2.model      = TDC_MODEL_DELTA_1D;
    s2.entropy[0] = TDC_ENTROPY_LZ;
    run("DELTA_1D + LZ", &src, &s2, raw);

    /* 3: DELTA_1D + ZIGZAG + LZ. Fixes the 0xFF-prefix problem. */
    tdc_codec_spec s3 = {0};
    s3.model      = TDC_MODEL_DELTA_1D;
    s3.xform[0]   = TDC_XFORM_ZIGZAG;
    s3.entropy[0] = TDC_ENTROPY_LZ;
    run("DELTA_1D + ZIGZAG + LZ", &src, &s3, raw);

    /* 4: DELTA_1D + BYTE_SHUFFLE + LZ. Groups top bytes together. */
    tdc_codec_spec s4 = {0};
    s4.model      = TDC_MODEL_DELTA_1D;
    s4.xform[0]   = TDC_XFORM_BYTE_SHUFFLE;
    s4.entropy[0] = TDC_ENTROPY_LZ;
    run("DELTA_1D + BSHUF + LZ", &src, &s4, raw);

    /* 5: Full chain. ZIGZAG then BYTE_SHUFFLE. */
    tdc_codec_spec s5 = {0};
    s5.model      = TDC_MODEL_DELTA_1D;
    s5.xform[0]   = TDC_XFORM_ZIGZAG;
    s5.xform[1]   = TDC_XFORM_BYTE_SHUFFLE;
    s5.entropy[0] = TDC_ENTROPY_LZ;
    run("DELTA_1D + ZIGZAG + BSHUF + LZ", &src, &s5, raw);

    return 0;
}
