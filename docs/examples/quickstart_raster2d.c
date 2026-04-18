/* docs/examples/quickstart_raster2d.c
 *
 * RASTER_2D taste test: a 128x128 u8 "image" where each pixel is the
 * sum of its (x, y) coordinates. PRED_2D with PAETH captures the
 * linear structure so residuals are near zero everywhere.
 *
 * Build:
 *   cc -I include docs/examples/quickstart_raster2d.c \
 *      build/libtdc.a -lm -o /tmp/qs_r2d
 */

#include "quickstart_common.h"
#include <string.h>

int main(void) {
    enum { H = 128, W = 128 };
    static uint8_t img[H * W];
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            img[y * W + x] = (uint8_t)(x + y);

    tdc_block src = {0};
    src.data   = img;
    src.dtype  = TDC_DT_U8;
    src.layout = TDC_LAYOUT_RASTER_2D;
    src.shape.rank   = 2;
    src.shape.dim[0] = H;
    src.shape.dim[1] = W;
    tdc_shape_set_contiguous(&src.shape);

    tdc_pred2d_params p = { .kind = TDC_PRED2D_PAETH };
    tdc_codec_spec spec = {0};
    spec.model        = TDC_MODEL_PRED_2D;
    spec.model_params = &p;
    spec.entropy[0]   = TDC_ENTROPY_LZ;

    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(&src, &spec, &enc), "encode")) return 1;

    size_t raw = H * W;
    printf("RASTER_2D  raw=%zu encoded=%zu ratio=%.2fx\n",
           raw, enc.size, (double)raw / (double)enc.size);

    tdc_block meta; size_t need;
    tdc_decode_peek(enc.data, enc.size, &meta, &need);
    void *dst_data = qs_realloc(NULL, NULL, need);
    tdc_block dst = meta; dst.data = dst_data;
    if (qs_check(tdc_decode_block_into(enc.data, enc.size, &dst), "decode")) return 1;
    int ok = memcmp(dst_data, img, raw) == 0;
    printf("           memcmp == source: %s\n", ok ? "yes" : "NO");

    qs_realloc(NULL, dst_data, 0);
    qs_buffer_free(&enc);
    return ok ? 0 : 1;
}
