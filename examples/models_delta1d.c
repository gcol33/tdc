/* docs/examples/models_delta1d.c
 *
 * TDC_MODEL_DELTA_1D walkthrough: first-order differencing on integer
 * columns, plus the XOR-delta variant for floats. Confirms round trip
 * and prints compression ratios for three representative inputs.
 */

#include "quickstart_common.h"
#include <string.h>

static int roundtrip_i64(const char *label,
                         const int64_t *src_data, size_t n,
                         const tdc_codec_spec *spec,
                         size_t *out_encoded) {
    tdc_block src = {0};
    src.data = (void*)src_data;
    src.dtype = TDC_DT_I64;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank = 1;
    src.shape.dim[0] = (int64_t)n;
    tdc_shape_set_contiguous(&src.shape);

    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(&src, spec, &enc), label)) return 1;
    *out_encoded = enc.size;

    tdc_block meta; size_t need;
    if (qs_check(tdc_decode_peek(enc.data, enc.size, &meta, &need), "peek")) return 1;
    void *dst_data = qs_realloc(NULL, NULL, need);
    tdc_block dst = meta; dst.data = dst_data;
    int ok = tdc_decode_block_into(enc.data, enc.size, &dst) == TDC_OK
          && memcmp(dst_data, src_data, n * sizeof(int64_t)) == 0;
    qs_realloc(NULL, dst_data, 0);
    qs_buffer_free(&enc);
    return ok ? 0 : 1;
}

static int roundtrip_f64(const char *label,
                         const double *src_data, size_t n,
                         const tdc_codec_spec *spec,
                         size_t *out_encoded) {
    tdc_block src = {0};
    src.data = (void*)src_data;
    src.dtype = TDC_DT_F64;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank = 1;
    src.shape.dim[0] = (int64_t)n;
    tdc_shape_set_contiguous(&src.shape);

    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(&src, spec, &enc), label)) return 1;
    *out_encoded = enc.size;

    tdc_block meta; size_t need;
    if (qs_check(tdc_decode_peek(enc.data, enc.size, &meta, &need), "peek")) return 1;
    void *dst_data = qs_realloc(NULL, NULL, need);
    tdc_block dst = meta; dst.data = dst_data;
    int ok = tdc_decode_block_into(enc.data, enc.size, &dst) == TDC_OK
          && memcmp(dst_data, src_data, n * sizeof(double)) == 0;
    qs_realloc(NULL, dst_data, 0);
    qs_buffer_free(&enc);
    return ok ? 0 : 1;
}

int main(void) {
    /* Integer path: residuals are signed, so ZIGZAG maps them to
     * unsigned small-magnitude bytes, and BYTE_SHUFFLE groups byte
     * lanes before LZ. */
    tdc_codec_spec delta_lz = {0};
    delta_lz.model = TDC_MODEL_DELTA_1D;
    delta_lz.xform[0] = TDC_XFORM_ZIGZAG;
    delta_lz.xform[1] = TDC_XFORM_BYTE_SHUFFLE;
    delta_lz.entropy[0] = TDC_ENTROPY_LZ;

    /* Float path: DELTA_1D emits an XOR residual (unsigned byte pattern
     * already). ZIGZAG rejects float dtypes, so we drop it and go
     * straight to BYTE_SHUFFLE + LZ. */
    tdc_codec_spec delta_lz_f = {0};
    delta_lz_f.model = TDC_MODEL_DELTA_1D;
    delta_lz_f.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
    delta_lz_f.entropy[0] = TDC_ENTROPY_LZ;

    tdc_codec_spec raw_lz = {0};
    raw_lz.model = TDC_MODEL_RAW;
    raw_lz.entropy[0] = TDC_ENTROPY_LZ;

    /* Input 1: monotonic i64 timestamps with small jitter. This is the
     * DELTA_1D sweet spot — the column is near-linear, so deltas are
     * small constants and zigzag maps them into a tiny byte range. */
    enum { N1 = 4096 };
    static int64_t ts[N1];
    int64_t t = 1700000000LL;
    for (int i = 0; i < N1; ++i) { ts[i] = t; t += 1000 + (i % 7); }

    /* Input 2: a jagged i64 walk (LCG-driven steps). Deltas are in a
     * wider range but still much narrower than the absolute values. */
    enum { N2 = 4096 };
    static int64_t walk[N2];
    int64_t v = 0;
    uint32_t s = 0xa5a5a5a5u;
    for (int i = 0; i < N2; ++i) {
        s = s * 1103515245u + 12345u;
        int32_t step = (int32_t)(s >> 20) - (1 << 11);
        v += step;
        walk[i] = v;
    }

    /* Input 3: an f64 sinusoid. Integer subtraction would mangle the bit
     * pattern; the XOR-delta path XORs raw bits so consecutive samples
     * share their exponent and the high mantissa bits. */
    enum { N3 = 4096 };
    static double sine[N3];
    for (int i = 0; i < N3; ++i) {
        /* cheap polynomial wave — no libm dependency */
        double x = (double)i * 0.01;
        double y = x - (x*x*x)/6.0 + (x*x*x*x*x)/120.0;
        sine[i] = y;
    }

    size_t raw = 0, enc = 0;

    printf("-- i64 timestamp column (N=4096): --\n");
    roundtrip_i64("RAW + LZ       ", ts, N1, &raw_lz, &enc);
    printf("  RAW + LZ          raw=%5zu enc=%5zu ratio=%6.2fx\n",
           N1 * sizeof(int64_t), enc, (double)(N1 * sizeof(int64_t)) / enc);
    roundtrip_i64("DELTA + LZ     ", ts, N1, &delta_lz, &enc);
    printf("  DELTA+ZZ+BSHUF+LZ raw=%5zu enc=%5zu ratio=%6.2fx\n",
           N1 * sizeof(int64_t), enc, (double)(N1 * sizeof(int64_t)) / enc);

    raw = N2 * sizeof(int64_t);
    printf("-- i64 random walk (N=4096): --\n");
    roundtrip_i64("RAW + LZ       ", walk, N2, &raw_lz, &enc);
    printf("  RAW + LZ          raw=%5zu enc=%5zu ratio=%6.2fx\n",
           raw, enc, (double)raw / enc);
    roundtrip_i64("DELTA + LZ     ", walk, N2, &delta_lz, &enc);
    printf("  DELTA+ZZ+BSHUF+LZ raw=%5zu enc=%5zu ratio=%6.2fx\n",
           raw, enc, (double)raw / enc);

    raw = N3 * sizeof(double);
    printf("-- f64 smooth sinusoid (XOR-delta path, N=4096): --\n");
    roundtrip_f64("RAW + LZ       ", sine, N3, &raw_lz, &enc);
    printf("  RAW + LZ          raw=%5zu enc=%5zu ratio=%6.2fx\n",
           raw, enc, (double)raw / enc);
    roundtrip_f64("DELTA + LZ     ", sine, N3, &delta_lz_f, &enc);
    printf("  DELTA + BSHUF + LZ raw=%5zu enc=%5zu ratio=%6.2fx\n",
           raw, enc, (double)raw / enc);

    return 0;
}
