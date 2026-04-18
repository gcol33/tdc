/* docs/examples/quickstart_codec_compare.c
 *
 * Encode the same block with three codec specs and report payload sizes.
 * Input: 4,096 i32 values of a near-linear ramp with small noise — the
 * shape DELTA_1D was built for.
 *
 * Build:
 *   cc -I include docs/examples/quickstart_codec_compare.c \
 *      build/libtdc.a -lm -o /tmp/qs_compare
 */

#include "quickstart_common.h"

static int run(const char *label, const tdc_block *src, const tdc_codec_spec *spec) {
    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(src, spec, &enc), label)) return 1;
    printf("  %-32s %6zu bytes\n", label, enc.size);
    qs_buffer_free(&enc);
    return 0;
}

int main(void) {
    enum { N = 4096 };
    static int32_t data[N];
    for (int i = 0; i < N; ++i) data[i] = 42000 + i * 3 + (i % 7);

    tdc_block src = {0};
    src.data   = data;
    src.dtype  = TDC_DT_I32;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank   = 1;
    src.shape.dim[0] = N;
    tdc_shape_set_contiguous(&src.shape);

    printf("input: %zu bytes (%d x i32)\n\n", N * sizeof(int32_t), N);

    tdc_codec_spec raw = tdc_codec_spec_raw();
    run("RAW", &src, &raw);

    tdc_codec_spec delta_lz = {0};
    delta_lz.model      = TDC_MODEL_DELTA_1D;
    delta_lz.entropy[0] = TDC_ENTROPY_LZ;
    run("DELTA_1D + LZ", &src, &delta_lz);

    tdc_codec_spec full = {0};
    full.model      = TDC_MODEL_DELTA_1D;
    full.xform[0]   = TDC_XFORM_ZIGZAG;
    full.xform[1]   = TDC_XFORM_BYTE_SHUFFLE;
    full.entropy[0] = TDC_ENTROPY_LZ;
    run("DELTA + ZIGZAG + BSHUF + LZ", &src, &full);

    return 0;
}
