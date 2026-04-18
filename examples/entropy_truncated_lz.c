/* docs/examples/entropy_truncated_lz.c
 *
 * Feed tdc_decode_block_into a shortened buffer. Three truncation points:
 *
 *   1. Inside the 80-byte block record header -> TDC_E_CORRUPT from
 *      tdc_decode_peek before any payload is touched.
 *   2. Just after the header but before the payload -> payload_size
 *      exceeds the remaining bytes, caught by decode_block.
 *   3. Inside the LZ payload, past the sequence headers -> the stream
 *      decoder runs off the end of the literal pool, returns
 *      TDC_E_CORRUPT.
 *
 * Build: part of TDC_DOC_EXAMPLES in the top-level CMakeLists.txt.
 */

#include "quickstart_common.h"
#include <stdint.h>
#include <string.h>

static const char *status_name(tdc_status s) {
    switch (s) {
        case TDC_OK:            return "TDC_OK";
        case TDC_E_INVAL:       return "TDC_E_INVAL";
        case TDC_E_CORRUPT:     return "TDC_E_CORRUPT";
        case TDC_E_VERSION:     return "TDC_E_VERSION";
        case TDC_E_UNSUPPORTED: return "TDC_E_UNSUPPORTED";
        default:                return "(other)";
    }
}

static void try_truncate(const uint8_t *rec, size_t truncated_size,
                         size_t need, tdc_block meta, const char *where) {
    void *dst = qs_realloc(NULL, NULL, need);
    tdc_block out = meta;
    out.data = dst;
    tdc_status s = tdc_decode_block_into(rec, truncated_size, &out);
    printf("  truncated to %-24s -> %s\n", where, status_name(s));
    qs_realloc(NULL, dst, 0);
}

int main(void) {
    enum { N = 4096 };
    static uint8_t data[N];
    for (size_t i = 0; i < N; ++i) data[i] = (uint8_t)(i * 31);

    tdc_block src = {0};
    src.data = data;
    src.dtype = TDC_DT_U8;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank = 1;
    src.shape.dim[0] = N;
    tdc_shape_set_contiguous(&src.shape);

    tdc_codec_spec spec = tdc_codec_spec_raw();
    spec.entropy[0] = TDC_ENTROPY_LZ;

    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(&src, &spec, &enc), "encode")) return 1;
    printf("record: %zu bytes\n", enc.size);

    tdc_block meta = {0};
    size_t need = 0;
    tdc_decode_peek(enc.data, enc.size, &meta, &need);

    try_truncate(enc.data, TDC_BLOCK_HEADER_SIZE / 2, need, meta, "half header");
    try_truncate(enc.data, TDC_BLOCK_HEADER_SIZE, need, meta, "header only");
    try_truncate(enc.data, enc.size - 4, need, meta, "record minus 4 bytes");
    try_truncate(enc.data, enc.size - 1, need, meta, "record minus 1 byte");

    qs_buffer_free(&enc);
    return 0;
}
