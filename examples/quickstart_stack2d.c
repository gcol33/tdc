/* docs/examples/quickstart_stack2d.c
 *
 * STACK_2D taste test: 8 frames of a 64x64 u8 raster. The "stack" layout
 * tells tdc to treat the first axis as a slice index, running the 2D
 * predictor inside each slice independently (and optionally subtracting
 * the prior slice first).
 *
 * Build:
 *   cc -I include docs/examples/quickstart_stack2d.c \
 *      build/libtdc.a -lm -o /tmp/qs_s2d
 */

#include "quickstart_common.h"
#include <string.h>

int main(void) {
    enum { S = 8, H = 64, W = 64 };
    static uint8_t stack[S * H * W];
    for (int s = 0; s < S; ++s)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                stack[(s * H + y) * W + x] = (uint8_t)((x + y + s) & 0xFF);

    tdc_block src = {0};
    src.data   = stack;
    src.dtype  = TDC_DT_U8;
    src.layout = TDC_LAYOUT_STACK_2D;
    src.shape.rank   = 3;
    src.shape.dim[0] = S;
    src.shape.dim[1] = H;
    src.shape.dim[2] = W;
    tdc_shape_set_contiguous(&src.shape);

    tdc_stack2d_params p = { .kind = TDC_PRED2D_PAETH, .inter_slice = 1 };
    tdc_codec_spec spec = {0};
    spec.model        = TDC_MODEL_STACK_2D;
    spec.model_params = &p;
    spec.entropy[0]   = TDC_ENTROPY_LZ;

    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(&src, &spec, &enc), "encode")) return 1;

    size_t raw = (size_t)S * H * W;
    printf("STACK_2D   raw=%zu encoded=%zu ratio=%.2fx\n",
           raw, enc.size, (double)raw / (double)enc.size);

    tdc_block meta; size_t need;
    tdc_decode_peek(enc.data, enc.size, &meta, &need);
    void *dst_data = qs_realloc(NULL, NULL, need);
    tdc_block dst = meta; dst.data = dst_data;
    if (qs_check(tdc_decode_block_into(enc.data, enc.size, &dst), "decode")) return 1;
    int ok = memcmp(dst_data, stack, raw) == 0;
    printf("           memcmp == source: %s\n", ok ? "yes" : "NO");

    qs_realloc(NULL, dst_data, 0);
    qs_buffer_free(&enc);
    return ok ? 0 : 1;
}
