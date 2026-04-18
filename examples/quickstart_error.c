/* docs/examples/quickstart_error.c
 *
 * Deliberately wrong call: ask DELTA_1D to encode a 2D raster. DELTA_1D
 * only accepts VECTOR_1D. The dispatcher rejects the mismatch with
 * TDC_E_LAYOUT; we print the status code's string.
 *
 * Build:
 *   cc -I include docs/examples/quickstart_error.c \
 *      build/libtdc.a -lm -o /tmp/qs_err
 */

#include "quickstart_common.h"

int main(void) {
    int32_t img[4 * 4];
    for (int i = 0; i < 16; ++i) img[i] = i;

    tdc_block src = {0};
    src.data   = img;
    src.dtype  = TDC_DT_I32;
    src.layout = TDC_LAYOUT_RASTER_2D;      /* 2D raster */
    src.shape.rank   = 2;
    src.shape.dim[0] = 4;
    src.shape.dim[1] = 4;
    tdc_shape_set_contiguous(&src.shape);

    tdc_codec_spec spec = {0};
    spec.model = TDC_MODEL_DELTA_1D;        /* accepts VECTOR_1D only */

    tdc_buffer enc = qs_buffer();
    tdc_status st = tdc_encode_block(&src, &spec, &enc);

    printf("status = %d (%s)\n", (int)st, tdc_strerror(st));
    /* Expected: status = 5 (layout not accepted by this stage) */

    qs_buffer_free(&enc);
    return st == TDC_E_LAYOUT ? 0 : 1;
}
