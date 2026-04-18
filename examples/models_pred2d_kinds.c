/* docs/examples/models_pred2d_kinds.c
 *
 * TDC_MODEL_PRED_2D walkthrough: compares LEFT, UP, AVERAGE, PAETH, and
 * AUTO on a synthetic raster that has a gradient in both axes plus a
 * diagonal ripple — the combination PAETH was designed for.
 */

#include "quickstart_common.h"
#include <string.h>

static uint8_t clip8(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

static void run_kind(const char *label,
                     const tdc_block *src,
                     tdc_pred2d_kind kind,
                     size_t raw_bytes) {
    tdc_pred2d_params p = { .kind = kind };
    tdc_codec_spec spec = {0};
    spec.model = TDC_MODEL_PRED_2D;
    spec.model_params = &p;
    /* u8 residuals wrap through the unsigned narrow counterpart already,
     * so zigzag would only add overhead. BYTE_SHUFFLE on a 1-byte dtype
     * is a no-op too — LZ sees the pred2d residual stream directly. */
    spec.entropy[0] = TDC_ENTROPY_LZ;

    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(src, &spec, &enc), label)) {
        qs_buffer_free(&enc);
        return;
    }
    printf("  %-10s enc=%6zu ratio=%6.2fx\n",
           label, enc.size, (double)raw_bytes / (double)enc.size);

    /* Verify round trip for one of the kinds so the example is a real
     * worked example, not a size-only comparison. */
    if (kind == TDC_PRED2D_PAETH) {
        tdc_block meta; size_t need;
        tdc_decode_peek(enc.data, enc.size, &meta, &need);
        void *dst_data = qs_realloc(NULL, NULL, need);
        tdc_block dst = meta; dst.data = dst_data;
        tdc_status st = tdc_decode_block_into(enc.data, enc.size, &dst);
        int ok = st == TDC_OK && memcmp(dst_data, src->data, raw_bytes) == 0;
        printf("  PAETH roundtrip: %s\n", ok ? "ok" : "FAIL");
        qs_realloc(NULL, dst_data, 0);
    }
    qs_buffer_free(&enc);
}

int main(void) {
    /* Synthetic 256x256 u8 raster: a bi-axial gradient plus a
     * low-amplitude diagonal ripple. The gradient means LEFT and UP
     * each leave half the signal in the residual; PAETH captures the
     * plane and drives the residual down to the ripple alone. */
    enum { H = 256, W = 256 };
    static uint8_t img[H * W];
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int ripple = ((x + y) & 7) - 3;
            int v = (x + y) / 2 + ripple;
            img[y * W + x] = clip8(v);
        }
    }

    tdc_block src = {0};
    src.data = img;
    src.dtype = TDC_DT_U8;
    src.layout = TDC_LAYOUT_RASTER_2D;
    src.shape.rank = 2;
    src.shape.dim[0] = H;
    src.shape.dim[1] = W;
    tdc_shape_set_contiguous(&src.shape);

    size_t raw = H * W;
    printf("-- 256x256 u8, bi-axial gradient + diagonal ripple (raw=%zu): --\n", raw);
    run_kind("LEFT    ", &src, TDC_PRED2D_LEFT,    raw);
    run_kind("UP      ", &src, TDC_PRED2D_UP,      raw);
    run_kind("AVERAGE ", &src, TDC_PRED2D_AVERAGE, raw);
    run_kind("PAETH   ", &src, TDC_PRED2D_PAETH,   raw);
    run_kind("AUTO    ", &src, TDC_PRED2D_AUTO,    raw);

    /* Second input: a horizontal-only gradient (every row identical).
     * UP predicts every interior row exactly (zero residuals there)
     * but pays for the full 0..255 seed row. LEFT leaves a constant-1
     * residual everywhere, which LZ collapses into one long match. */
    static uint8_t hgrad[H * W];
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            hgrad[y * W + x] = (uint8_t)x;

    tdc_block src2 = {0};
    src2.data = hgrad;
    src2.dtype = TDC_DT_U8;
    src2.layout = TDC_LAYOUT_RASTER_2D;
    src2.shape.rank = 2;
    src2.shape.dim[0] = H;
    src2.shape.dim[1] = W;
    tdc_shape_set_contiguous(&src2.shape);

    printf("-- 256x256 u8, horizontal gradient (UP is exact): --\n");
    run_kind("LEFT    ", &src2, TDC_PRED2D_LEFT, raw);
    run_kind("UP      ", &src2, TDC_PRED2D_UP,   raw);
    run_kind("PAETH   ", &src2, TDC_PRED2D_PAETH, raw);
    run_kind("AUTO    ", &src2, TDC_PRED2D_AUTO, raw);

    return 0;
}
