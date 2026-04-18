/* docs/examples/models_pred3d.c
 *
 * TDC_MODEL_PRED_3D walkthrough: compares LEFT / UP / FRONT / AVG3 /
 * GRAD3D / PAETH3D on a non-trivial volume that has structure in all
 * three axes plus a step edge (so the exact-on-trilinear GRAD3D
 * predictor is not the automatic winner).
 */

#include "quickstart_common.h"
#include <string.h>

static void run_kind(const char *label,
                     const tdc_block *src,
                     tdc_pred3d_kind kind,
                     size_t raw_bytes) {
    tdc_pred3d_params p = { .kind = kind };
    tdc_codec_spec spec = {0};
    spec.model = TDC_MODEL_PRED_3D;
    spec.model_params = &p;
    spec.xform[0] = TDC_XFORM_ZIGZAG;
    spec.xform[1] = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0] = TDC_ENTROPY_LZ;

    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(src, &spec, &enc), label)) {
        qs_buffer_free(&enc);
        return;
    }
    printf("  %-10s enc=%6zu ratio=%7.2fx\n",
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
    /* Volume 1: a pure tri-affine field. GRAD3D is exact, everything
     * else leaves residuals along one or more axes. The encoder's
     * zero-residual fast path kicks in for GRAD3D. */
    enum { D1 = 24, H1 = 32, W1 = 40 };
    static int32_t tri[D1 * H1 * W1];
    for (int z = 0; z < D1; ++z)
        for (int y = 0; y < H1; ++y)
            for (int x = 0; x < W1; ++x)
                tri[(z * H1 + y) * W1 + x] = 3*x + 5*y + 7*z + 17;

    tdc_block v1 = {0};
    v1.data = tri;
    v1.dtype = TDC_DT_I32;
    v1.layout = TDC_LAYOUT_VOLUME_3D;
    v1.shape.rank = 3;
    v1.shape.dim[0] = D1;
    v1.shape.dim[1] = H1;
    v1.shape.dim[2] = W1;
    tdc_shape_set_contiguous(&v1.shape);

    size_t raw1 = sizeof(tri);
    printf("-- 24x32x40 i32 tri-affine volume (raw=%zu): --\n", raw1);
    run_kind("LEFT   ", &v1, TDC_PRED3D_LEFT,    raw1);
    run_kind("UP     ", &v1, TDC_PRED3D_UP,      raw1);
    run_kind("FRONT  ", &v1, TDC_PRED3D_FRONT,   raw1);
    run_kind("AVG3   ", &v1, TDC_PRED3D_AVG3,    raw1);
    run_kind("GRAD3D ", &v1, TDC_PRED3D_GRAD3D,  raw1);
    run_kind("PAETH3D", &v1, TDC_PRED3D_PAETH3D, raw1);
    run_kind("AUTO   ", &v1, TDC_PRED3D_AUTO,    raw1);

    /* Volume 2: a smooth Gaussian blob plus a sharp step edge at the
     * slab midpoint. The step breaks GRAD3D's trilinear assumption
     * everywhere it crosses; PAETH3D adapts to the edge by picking the
     * neighbor closest to the linear predictor on each side. */
    enum { D2 = 16, H2 = 32, W2 = 32 };
    static int32_t mix[D2 * H2 * W2];
    for (int z = 0; z < D2; ++z)
        for (int y = 0; y < H2; ++y)
            for (int x = 0; x < W2; ++x) {
                int dx = x - W2/2, dy = y - H2/2, dz = z - D2/2;
                int bell = (3000 * 100) / (100 + dx*dx + dy*dy + dz*dz);
                int step = (x >= W2/2) ? 500 : 0;
                mix[(z * H2 + y) * W2 + x] = bell + step;
            }

    tdc_block v2 = {0};
    v2.data = mix;
    v2.dtype = TDC_DT_I32;
    v2.layout = TDC_LAYOUT_VOLUME_3D;
    v2.shape.rank = 3;
    v2.shape.dim[0] = D2;
    v2.shape.dim[1] = H2;
    v2.shape.dim[2] = W2;
    tdc_shape_set_contiguous(&v2.shape);

    size_t raw2 = sizeof(mix);
    printf("-- 16x32x32 i32 smooth bell + step edge (raw=%zu): --\n", raw2);
    run_kind("LEFT   ", &v2, TDC_PRED3D_LEFT,    raw2);
    run_kind("AVG3   ", &v2, TDC_PRED3D_AVG3,    raw2);
    run_kind("GRAD3D ", &v2, TDC_PRED3D_GRAD3D,  raw2);
    run_kind("PAETH3D", &v2, TDC_PRED3D_PAETH3D, raw2);
    run_kind("AUTO   ", &v2, TDC_PRED3D_AUTO,    raw2);

    return 0;
}
