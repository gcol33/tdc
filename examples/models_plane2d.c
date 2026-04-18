/* docs/examples/models_plane2d.c
 *
 * TDC_MODEL_PLANE_2D walkthrough: least-squares plane fit per tile.
 * Compares PLANE_2D against PRED_2D/PAETH on a piecewise-planar raster
 * that imitates a tiled digital-elevation model (a DEM).
 */

#include "quickstart_common.h"
#include <string.h>

static void run(const char *label,
                const tdc_block *src,
                const tdc_codec_spec *spec,
                size_t raw_bytes) {
    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(src, spec, &enc), label)) {
        qs_buffer_free(&enc);
        return;
    }
    printf("  %-32s enc=%6zu ratio=%6.2fx\n",
           label, enc.size, (double)raw_bytes / (double)enc.size);

    tdc_block meta; size_t need;
    tdc_decode_peek(enc.data, enc.size, &meta, &need);
    void *dst_data = qs_realloc(NULL, NULL, need);
    tdc_block dst = meta; dst.data = dst_data;
    tdc_status st = tdc_decode_block_into(enc.data, enc.size, &dst);
    int ok = st == TDC_OK && memcmp(dst_data, src->data, raw_bytes) == 0;
    if (!ok) printf("    roundtrip FAIL: %s\n", tdc_strerror(st));
    qs_realloc(NULL, dst_data, 0);
    qs_buffer_free(&enc);
}

int main(void) {
    /* 512x512 i16 DEM: split into 4 quadrants, each a distinct plane
     * a + b*x + c*y with integer coefficients. Each 32x32 tile (the
     * PLANE_2D default tile_size) sits entirely inside one quadrant
     * except for the two seam rows and two seam columns. */
    enum { H = 512, W = 512 };
    static int16_t dem[H * W];
    for (int y = 0; y < H; ++y) {
        int qy = (y >= H/2);
        for (int x = 0; x < W; ++x) {
            int qx = (x >= W/2);
            int v;
            switch ((qy << 1) | qx) {
                case 0: v = 1000 + x + 2*y;          break;
                case 1: v = 2000 - x + 3*y;          break;
                case 2: v = 1500 + 2*x - y;          break;
                default: v = 2500 - 2*x - 2*y;       break;
            }
            dem[y * W + x] = (int16_t)v;
        }
    }

    tdc_block src = {0};
    src.data = dem;
    src.dtype = TDC_DT_I16;
    src.layout = TDC_LAYOUT_RASTER_2D;
    src.shape.rank = 2;
    src.shape.dim[0] = H;
    src.shape.dim[1] = W;
    tdc_shape_set_contiguous(&src.shape);

    size_t raw = H * W * sizeof(int16_t);

    tdc_pred2d_params paeth = { .kind = TDC_PRED2D_PAETH };
    tdc_codec_spec pred = {0};
    pred.model = TDC_MODEL_PRED_2D;
    pred.model_params = &paeth;
    pred.xform[0] = TDC_XFORM_ZIGZAG;
    pred.xform[1] = TDC_XFORM_BYTE_SHUFFLE;
    pred.entropy[0] = TDC_ENTROPY_LZ;

    tdc_plane2d_params pp32 = { .tile_size = 32 };
    tdc_codec_spec plane32 = {0};
    plane32.model = TDC_MODEL_PLANE_2D;
    plane32.model_params = &pp32;
    plane32.xform[0] = TDC_XFORM_ZIGZAG;
    plane32.xform[1] = TDC_XFORM_BYTE_SHUFFLE;
    plane32.entropy[0] = TDC_ENTROPY_LZ;

    tdc_plane2d_params pp16 = { .tile_size = 16 };
    tdc_codec_spec plane16 = plane32;
    plane16.model_params = &pp16;

    tdc_plane2d_params pp128 = { .tile_size = 128 };
    tdc_codec_spec plane128 = plane32;
    plane128.model_params = &pp128;

    printf("-- 512x512 i16 piecewise-planar DEM (raw=%zu): --\n", raw);
    run("PRED_2D / PAETH                 ", &src, &pred,    raw);
    run("PLANE_2D, tile_size=16          ", &src, &plane16, raw);
    run("PLANE_2D, tile_size=32 (default)", &src, &plane32, raw);
    run("PLANE_2D, tile_size=128         ", &src, &plane128, raw);

    /* Second input: a raster with noise that does NOT fit a per-tile
     * plane. PLANE_2D pays a side-metadata cost with no residual
     * benefit; PRED_2D wins. */
    static int16_t noisy[H * W];
    uint32_t s = 0x13371337u;
    for (int i = 0; i < H * W; ++i) {
        s = s * 1664525u + 1013904223u;
        noisy[i] = (int16_t)(s >> 16);
    }
    src.data = noisy;

    printf("-- 512x512 i16 incompressible noise: --\n");
    run("PRED_2D / PAETH                 ", &src, &pred,    raw);
    run("PLANE_2D, tile_size=32          ", &src, &plane32, raw);

    return 0;
}
