/* docs/examples/entropy_roundtrip_huffman.c
 *
 * TDC_ENTROPY_HUFFMAN round-trip on a skewed byte distribution. Huffman
 * is a statistical coder with no memory; it wins when the symbol
 * distribution is non-uniform but shows no repeats LZ could exploit.
 *
 * Build: part of TDC_DOC_EXAMPLES in the top-level CMakeLists.txt.
 */

#include "quickstart_common.h"
#include <stdint.h>
#include <string.h>

int main(void) {
    /* A skewed byte stream: 70% zeros, 15% ones, 15% one random byte. An
     * entropy-free baseline would need 8 bits/byte; Shannon's limit here
     * is ~1.4 bits/byte. LZ has no repeats to match and returns ~1.0x;
     * Huffman captures the skew directly. */
    enum { N = 1u << 16 };   /* 64 KiB */
    static uint8_t buf[N];

    uint64_t s = 0xC0FFEEDEADBEEFull;
    for (size_t i = 0; i < N; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        uint32_t r = (uint32_t)(s >> 33) % 100;
        if (r < 70)      buf[i] = 0;
        else if (r < 85) buf[i] = 1;
        else             buf[i] = (uint8_t)(s >> 56);
    }

    tdc_block src = {0};
    src.data = buf;
    src.dtype = TDC_DT_U8;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank = 1;
    src.shape.dim[0] = N;
    tdc_shape_set_contiguous(&src.shape);

    struct { const char *label; tdc_entropy_id id; } backends[] = {
        { "NONE   ",  TDC_ENTROPY_NONE    },
        { "LZ     ",  TDC_ENTROPY_LZ      },
        { "HUFFMAN",  TDC_ENTROPY_HUFFMAN },
        { "FSE    ",  TDC_ENTROPY_FSE     },
    };

    printf("input: %d bytes, byte distribution skewed (70%%/15%%/15%%)\n\n", N);
    for (size_t i = 0; i < sizeof(backends)/sizeof(backends[0]); ++i) {
        tdc_codec_spec spec = tdc_codec_spec_raw();
        spec.entropy[0] = backends[i].id;

        tdc_buffer enc = qs_buffer();
        if (qs_check(tdc_encode_block(&src, &spec, &enc), backends[i].label)) continue;

        tdc_block meta = {0};
        size_t need = 0;
        tdc_decode_peek(enc.data, enc.size, &meta, &need);
        void *dst_data = qs_realloc(NULL, NULL, need);
        tdc_block dst = meta;
        dst.data = dst_data;
        tdc_status rs = tdc_decode_block_into(enc.data, enc.size, &dst);
        int ok = rs == TDC_OK && memcmp(dst_data, buf, N) == 0;

        printf("  %s  enc=%6zu  ratio=%5.2fx  %s\n",
               backends[i].label, enc.size,
               (double)N / (double)enc.size,
               ok ? "roundtrip ok" : "ROUNDTRIP FAIL");

        qs_realloc(NULL, dst_data, 0);
        qs_buffer_free(&enc);
    }
    return 0;
}
