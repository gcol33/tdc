/* docs/examples/quickstart_roundtrip.c
 *
 * Minimal round trip: encode a 1D i64 ramp with RAW, decode it back,
 * print input/output byte counts. Verifies the result matches the source.
 *
 * Build (from repo root, after `cmake --build build`):
 *   cc -I include docs/examples/quickstart_roundtrip.c \
 *      build/libtdc.a -lm -o /tmp/qs_roundtrip
 *   /tmp/qs_roundtrip
 */

#include "quickstart_common.h"
#include <string.h>

int main(void) {
    int64_t src_data[256];
    for (int i = 0; i < 256; ++i) src_data[i] = 1000 + (int64_t)i * 3;

    tdc_block src = {0};
    src.data   = src_data;
    src.dtype  = TDC_DT_I64;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank   = 1;
    src.shape.dim[0] = 256;
    tdc_shape_set_contiguous(&src.shape);

    tdc_codec_spec spec = tdc_codec_spec_raw();
    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(&src, &spec, &enc), "encode")) return 1;

    size_t raw_bytes = 256 * sizeof(int64_t);
    printf("input  : %zu bytes (256 x i64)\n", raw_bytes);
    printf("encoded: %zu bytes (block record, incl. 80-byte header)\n", enc.size);

    tdc_block meta = {0};
    size_t need = 0;
    if (qs_check(tdc_decode_peek(enc.data, enc.size, &meta, &need), "peek")) return 1;

    void *dst_data = qs_realloc(NULL, NULL, need);
    tdc_block dst = meta;
    dst.data = dst_data;
    if (qs_check(tdc_decode_block_into(enc.data, enc.size, &dst), "decode")) return 1;

    int ok = memcmp(dst_data, src_data, raw_bytes) == 0;
    printf("decoded: %zu bytes, memcmp == source: %s\n", need, ok ? "yes" : "NO");

    qs_realloc(NULL, dst_data, 0);
    qs_buffer_free(&enc);
    return ok ? 0 : 1;
}
