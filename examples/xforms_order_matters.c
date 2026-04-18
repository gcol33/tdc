/* docs/examples/xforms_order_matters.c
 *
 * The same two transforms in two orders. DELTA_1D emits small signed
 * residuals with 0xFF-prefixed high bytes for negatives. If BYTE_SHUFFLE
 * runs before ZIGZAG, the high-byte lane contains 0xFF / 0x00 noise
 * instead of a near-constant 0x00 lane. If ZIGZAG runs first, the
 * high-byte lane is almost entirely zero and LZ collapses it.
 *
 * This example shows the byte-count gap between the two orders.
 *
 * Build:
 *   cc -I include docs/examples/xforms_order_matters.c \
 *      build/libtdc.a -lm -o /tmp/xf_order
 */

#include "quickstart_common.h"
#include <stdlib.h>

static void run(const char *label, const tdc_block *src,
                tdc_xform_id x0, tdc_xform_id x1, size_t raw) {
    tdc_codec_spec spec = {0};
    spec.model      = TDC_MODEL_DELTA_1D;
    spec.xform[0]   = x0;
    spec.xform[1]   = x1;
    spec.entropy[0] = TDC_ENTROPY_LZ;

    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(src, &spec, &enc), label)) exit(1);
    double ratio = (double)raw / (double)enc.size;
    printf("  %-28s %6zu bytes  ratio=%.2fx\n", label, enc.size, ratio);
    qs_buffer_free(&enc);
}

int main(void) {
    enum { N = 8192 };
    static int32_t data[N];
    /* Oscillating signal so the DELTA residual sign flips often. */
    for (int i = 0; i < N; ++i) {
        int sign = (i % 7 < 3) ? 1 : -1;
        data[i] = 200000 + i * 2 + sign * (i % 11);
    }

    tdc_block src = {0};
    src.data   = data;
    src.dtype  = TDC_DT_I32;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank   = 1;
    src.shape.dim[0] = N;
    tdc_shape_set_contiguous(&src.shape);

    size_t raw = N * sizeof(int32_t);
    printf("input: %zu bytes (%d x i32), sign-flipping delta residuals\n\n",
           raw, N);

    run("ZIGZAG -> BSHUF", &src, TDC_XFORM_ZIGZAG, TDC_XFORM_BYTE_SHUFFLE, raw);
    run("BSHUF -> ZIGZAG", &src, TDC_XFORM_BYTE_SHUFFLE, TDC_XFORM_ZIGZAG, raw);

    return 0;
}
