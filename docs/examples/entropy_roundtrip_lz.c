/* docs/examples/entropy_roundtrip_lz.c
 *
 * TDC_ENTROPY_LZ round-trip on three shapes the LZ matcher sees in the
 * wild: a zero run (hits offset-1 doubling in the decoder), a periodic
 * byte run (one match spans almost the whole input), and LCG noise
 * (parser emits zero sequences, literal-only fallback).
 *
 * Build: part of TDC_DOC_EXAMPLES in the top-level CMakeLists.txt.
 */

#include "quickstart_common.h"
#include <stdint.h>
#include <string.h>

static void run(const char *label, uint8_t *data, size_t n) {
    tdc_block src = {0};
    src.data = data;
    src.dtype = TDC_DT_U8;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank = 1;
    src.shape.dim[0] = (int64_t)n;
    tdc_shape_set_contiguous(&src.shape);

    tdc_codec_spec spec = tdc_codec_spec_raw();
    spec.entropy[0] = TDC_ENTROPY_LZ;

    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(&src, &spec, &enc), label)) return;

    tdc_block meta = {0};
    size_t need = 0;
    if (qs_check(tdc_decode_peek(enc.data, enc.size, &meta, &need), label)) return;

    void *dst_data = qs_realloc(NULL, NULL, need);
    tdc_block dst = meta;
    dst.data = dst_data;
    if (qs_check(tdc_decode_block_into(enc.data, enc.size, &dst), label)) return;

    int ok = memcmp(dst_data, data, n) == 0;
    printf("%-10s raw=%7zu enc=%7zu ratio=%9.2fx  roundtrip %s\n",
           label, n, enc.size, (double)n / (double)enc.size,
           ok ? "ok" : "MISMATCH");

    qs_realloc(NULL, dst_data, 0);
    qs_buffer_free(&enc);
}

int main(void) {
    enum { N = 1u << 20 };   /* 1 MiB per input */
    static uint8_t buf[N];

    memset(buf, 0, N);
    run("zero", buf, N);

    for (size_t i = 0; i < N; ++i) buf[i] = "abc"[i % 3];
    run("abc", buf, N);

    uint64_t s = 0x243F6A8885A308D3ull;
    for (size_t i = 0; i < N; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        buf[i] = (uint8_t)(s >> 56);
    }
    run("noise", buf, N);

    return 0;
}
