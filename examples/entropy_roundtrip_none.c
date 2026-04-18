/* docs/examples/entropy_roundtrip_none.c
 *
 * TDC_ENTROPY_NONE round-trip. Model RAW, no transforms, entropy
 * passthrough. The encoded payload is the input bytes verbatim plus
 * an 80-byte block record and a 4-byte per-stage sizes table.
 *
 * Build: part of TDC_DOC_EXAMPLES in the top-level CMakeLists.txt.
 */

#include "quickstart_common.h"
#include <string.h>

int main(void) {
    enum { N = 1024 };
    static uint8_t src_data[N];
    for (int i = 0; i < N; ++i) src_data[i] = (uint8_t)(i & 0xFF);

    tdc_block src = {0};
    src.data = src_data;
    src.dtype = TDC_DT_U8;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank = 1;
    src.shape.dim[0] = N;
    tdc_shape_set_contiguous(&src.shape);

    tdc_codec_spec spec = tdc_codec_spec_raw();
    /* Entropy chain left at zero = NONE = passthrough. */

    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(&src, &spec, &enc), "encode")) return 1;

    printf("input   : %d bytes\n", N);
    printf("encoded : %zu bytes (block record, incl. %d-byte header)\n",
           enc.size, TDC_BLOCK_HEADER_SIZE);

    tdc_block meta = {0};
    size_t need = 0;
    if (qs_check(tdc_decode_peek(enc.data, enc.size, &meta, &need), "peek")) return 1;

    void *dst_data = qs_realloc(NULL, NULL, need);
    tdc_block dst = meta;
    dst.data = dst_data;
    if (qs_check(tdc_decode_block_into(enc.data, enc.size, &dst), "decode")) return 1;

    int ok = memcmp(dst_data, src_data, N) == 0;
    printf("decoded : memcmp %s\n", ok ? "ok" : "MISMATCH");

    qs_realloc(NULL, dst_data, 0);
    qs_buffer_free(&enc);
    return ok ? 0 : 1;
}
