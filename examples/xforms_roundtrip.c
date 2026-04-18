/* docs/examples/xforms_roundtrip.c
 *
 * Walks a single i16 block through the xform_ids[4] chain and back.
 * Model is RAW (no prediction), so the chain is the only compression
 * source. The spec is:
 *
 *   xform[0] = ZIGZAG         (signed -> unsigned, same width)
 *   xform[1] = BYTE_SHUFFLE   (group same-significance bytes)
 *   entropy  = LZ             (runs of repeated bytes)
 *
 * Round-trip uses the public tdc_encode_block / tdc_decode_block_into
 * pair. The allocator is wired through qs_realloc.
 *
 * Build:
 *   cc -I include docs/examples/xforms_roundtrip.c \
 *      build/libtdc.a -lm -o /tmp/xf_roundtrip
 */

#include "quickstart_common.h"
#include <string.h>

int main(void) {
    enum { N = 2048 };
    static int16_t src_data[N];
    /* Small-magnitude signed residuals, of the shape a delta or 2D
     * predictor would typically emit: zero-centred with light jitter. */
    for (int i = 0; i < N; ++i) {
        int v = ((i * 37) % 11) - 5;
        src_data[i] = (int16_t)v;
    }

    tdc_block src = {0};
    src.data   = src_data;
    src.dtype  = TDC_DT_I16;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank   = 1;
    src.shape.dim[0] = N;
    tdc_shape_set_contiguous(&src.shape);

    tdc_codec_spec spec = {0};
    spec.model      = TDC_MODEL_RAW;
    spec.xform[0]   = TDC_XFORM_ZIGZAG;
    spec.xform[1]   = TDC_XFORM_BYTE_SHUFFLE;
    spec.xform[2]   = TDC_XFORM_NONE;          /* terminates the chain */
    spec.entropy[0] = TDC_ENTROPY_LZ;

    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(&src, &spec, &enc), "encode")) return 1;

    size_t raw = N * sizeof(int16_t);
    printf("input  : %zu bytes (%d x i16)\n", raw, N);
    printf("encoded: %zu bytes (block record, incl. 80-byte header)\n", enc.size);

    tdc_block meta = {0};
    size_t need = 0;
    if (qs_check(tdc_decode_peek(enc.data, enc.size, &meta, &need), "peek")) return 1;

    void *dst_data = qs_realloc(NULL, NULL, need);
    tdc_block dst = meta;
    dst.data = dst_data;
    if (qs_check(tdc_decode_block_into(enc.data, enc.size, &dst), "decode")) return 1;

    int ok = memcmp(dst_data, src_data, raw) == 0;
    printf("decoded: %zu bytes, memcmp == source: %s\n", need, ok ? "yes" : "NO");

    qs_realloc(NULL, dst_data, 0);
    qs_buffer_free(&enc);
    return ok ? 0 : 1;
}
