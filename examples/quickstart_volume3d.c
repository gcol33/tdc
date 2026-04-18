/* docs/examples/quickstart_volume3d.c
 *
 * VOLUME_3D taste test: a 12x16x20 i32 voxel grid holding a tri-affine
 * field. PRED_3D GRAD3D is exact on any linear-in-(x,y,z) function, so
 * the residual is zero everywhere except the seed voxels at the volume
 * boundary. The encoder picks up the zero-residual fast path.
 *
 * Build:
 *   cc -I include docs/examples/quickstart_volume3d.c \
 *      build/libtdc.a -lm -o /tmp/qs_v3d
 */

#include "quickstart_common.h"
#include <string.h>

int main(void) {
    enum { D = 12, H = 16, W = 20 };
    static int32_t vol[D * H * W];
    for (int z = 0; z < D; ++z)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                vol[(z * H + y) * W + x] = 3 * x + 5 * y + 7 * z + 17;

    tdc_block src = {0};
    src.data   = vol;
    src.dtype  = TDC_DT_I32;
    src.layout = TDC_LAYOUT_VOLUME_3D;
    src.shape.rank   = 3;
    src.shape.dim[0] = D;
    src.shape.dim[1] = H;
    src.shape.dim[2] = W;
    tdc_shape_set_contiguous(&src.shape);

    tdc_pred3d_params p = { .kind = TDC_PRED3D_GRAD3D };
    tdc_codec_spec spec = {0};
    spec.model        = TDC_MODEL_PRED_3D;
    spec.model_params = &p;
    spec.xform[0]     = TDC_XFORM_ZIGZAG;
    spec.xform[1]     = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0]   = TDC_ENTROPY_LZ;

    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(&src, &spec, &enc), "encode")) return 1;

    size_t raw = (size_t)D * H * W * sizeof(int32_t);
    printf("VOLUME_3D  raw=%zu encoded=%zu ratio=%.2fx\n",
           raw, enc.size, (double)raw / (double)enc.size);

    tdc_block meta; size_t need;
    tdc_decode_peek(enc.data, enc.size, &meta, &need);
    void *dst_data = qs_realloc(NULL, NULL, need);
    tdc_block dst = meta; dst.data = dst_data;
    if (qs_check(tdc_decode_block_into(enc.data, enc.size, &dst), "decode")) return 1;
    int ok = memcmp(dst_data, vol, raw) == 0;
    printf("           memcmp == source: %s\n", ok ? "yes" : "NO");

    qs_realloc(NULL, dst_data, 0);
    qs_buffer_free(&enc);
    return ok ? 0 : 1;
}
