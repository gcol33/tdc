/* docs/examples/models_raw.c
 *
 * TDC_MODEL_RAW walkthrough: identity model across several layouts. Shows
 * what "no model" does, how the 80-byte block record header dominates
 * small blocks, and how RAW + LZ behaves on inputs with byte-level
 * repetition versus inputs with multi-byte structure the byte-level LZ
 * matcher cannot see.
 *
 * Build: part of TDC_DOC_EXAMPLES in the top-level CMakeLists.txt.
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
    printf("%-26s raw=%5zu encoded=%5zu ratio=%5.2fx\n",
           label, raw_bytes, enc.size,
           (double)raw_bytes / (double)enc.size);
    qs_buffer_free(&enc);
}

int main(void) {
    /* Input 1: a 256-element i32 ramp. No byte-level repetition at the
     * per-element width because the stride is 1; the low byte cycles
     * 0..255 exactly once and the top three bytes are all zero. */
    enum { N1 = 256 };
    static int32_t ramp[N1];
    for (int i = 0; i < N1; ++i) ramp[i] = i;

    tdc_block b1 = {0};
    b1.data = ramp;
    b1.dtype = TDC_DT_I32;
    b1.layout = TDC_LAYOUT_VECTOR_1D;
    b1.shape.rank = 1;
    b1.shape.dim[0] = N1;
    tdc_shape_set_contiguous(&b1.shape);

    /* Input 2: a 128x128 u8 raster of pure noise (LCG-modulated). The
     * byte-level LZ has nothing to match here; RAW + LZ returns the
     * input almost verbatim. */
    enum { H = 128, W = 128 };
    static uint8_t noise[H * W];
    uint32_t s = 0xdeadbeefu;
    for (int i = 0; i < H * W; ++i) {
        s = s * 1664525u + 1013904223u;
        noise[i] = (uint8_t)(s >> 24);
    }

    tdc_block b2 = {0};
    b2.data = noise;
    b2.dtype = TDC_DT_U8;
    b2.layout = TDC_LAYOUT_RASTER_2D;
    b2.shape.rank = 2;
    b2.shape.dim[0] = H;
    b2.shape.dim[1] = W;
    tdc_shape_set_contiguous(&b2.shape);

    /* Input 3: a 4096-element u8 buffer of a short repeating pattern.
     * The byte matcher finds an enormous single match. */
    enum { N3 = 4096 };
    static uint8_t pattern[N3];
    static const uint8_t seed[16] = {
        0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x74, 0x64,
        0x63, 0x21, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05
    };
    for (int i = 0; i < N3; ++i) pattern[i] = seed[i & 15];

    tdc_block b3 = {0};
    b3.data = pattern;
    b3.dtype = TDC_DT_U8;
    b3.layout = TDC_LAYOUT_VECTOR_1D;
    b3.shape.rank = 1;
    b3.shape.dim[0] = N3;
    tdc_shape_set_contiguous(&b3.shape);

    tdc_codec_spec raw = tdc_codec_spec_raw();

    tdc_codec_spec raw_lz = {0};
    raw_lz.model = TDC_MODEL_RAW;
    raw_lz.entropy[0] = TDC_ENTROPY_LZ;

    printf("-- i32 ramp (N=256): --\n");
    run("RAW                    ", &b1, &raw,    N1 * sizeof(int32_t));
    run("RAW + LZ               ", &b1, &raw_lz, N1 * sizeof(int32_t));

    printf("-- u8 noise (128x128): --\n");
    run("RAW                    ", &b2, &raw,    H * W);
    run("RAW + LZ               ", &b2, &raw_lz, H * W);

    printf("-- u8 repeating pattern (N=4096, period 16): --\n");
    run("RAW                    ", &b3, &raw,    N3);
    run("RAW + LZ               ", &b3, &raw_lz, N3);

    return 0;
}
