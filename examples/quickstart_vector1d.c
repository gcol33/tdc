/* docs/examples/quickstart_vector1d.c
 *
 * VECTOR_1D taste test: an i64 column of timestamps compresses well with
 * DELTA_1D because consecutive values differ by a small, almost-constant
 * increment. Zigzag + byte shuffle + LZ cleans up the residual bytes.
 *
 * Build:
 *   cc -I include docs/examples/quickstart_vector1d.c \
 *      build/libtdc.a -lm -o /tmp/qs_v1d
 */

#include "quickstart_common.h"
#include <string.h>

int main(void) {
    enum { N = 1024 };
    static int64_t ts[N];
    int64_t t = 1700000000LL;
    for (int i = 0; i < N; ++i) { ts[i] = t; t += 1000 + (i % 5); }

    tdc_block src = {0};
    src.data   = ts;
    src.dtype  = TDC_DT_I64;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank   = 1;
    src.shape.dim[0] = N;
    tdc_shape_set_contiguous(&src.shape);

    tdc_codec_spec spec = {0};
    spec.model      = TDC_MODEL_DELTA_1D;
    spec.xform[0]   = TDC_XFORM_ZIGZAG;
    spec.xform[1]   = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0] = TDC_ENTROPY_LZ;

    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(&src, &spec, &enc), "encode")) return 1;

    size_t raw = N * sizeof(int64_t);
    printf("VECTOR_1D  raw=%zu encoded=%zu ratio=%.2fx\n",
           raw, enc.size, (double)raw / (double)enc.size);

    tdc_block meta; size_t need;
    tdc_decode_peek(enc.data, enc.size, &meta, &need);
    void *dst_data = qs_realloc(NULL, NULL, need);
    tdc_block dst = meta; dst.data = dst_data;
    if (qs_check(tdc_decode_block_into(enc.data, enc.size, &dst), "decode")) return 1;
    int ok = memcmp(dst_data, ts, raw) == 0;
    printf("           memcmp == source: %s\n", ok ? "yes" : "NO");

    qs_realloc(NULL, dst_data, 0);
    qs_buffer_free(&enc);
    return ok ? 0 : 1;
}
