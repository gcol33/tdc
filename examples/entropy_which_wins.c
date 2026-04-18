/* docs/examples/entropy_which_wins.c
 *
 * One input, every entropy backend. Prints a table of encoded bytes,
 * ratio, encode throughput, and decode throughput so the caller can
 * pick the backend that wins on their own workload. The input is the
 * post-shuffle byte stream of an i16 raster after PRED_2D + ZIGZAG +
 * BYTE_SHUFFLE; it resembles what real pipelines hand to entropy.
 *
 * Build: part of TDC_DOC_EXAMPLES in the top-level CMakeLists.txt.
 */

#include "quickstart_common.h"
#include <stdint.h>
#include <string.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
static double bench_now(void) {
    static LARGE_INTEGER freq;
    static int inited = 0;
    LARGE_INTEGER t;
    if (!inited) { QueryPerformanceFrequency(&freq); inited = 1; }
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
}
#else
#  include <time.h>
static double bench_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

#define ITERS 5

static double median5(double xs[ITERS]) {
    for (int i = 1; i < ITERS; ++i) {
        double v = xs[i]; int j = i;
        while (j > 0 && xs[j-1] > v) { xs[j] = xs[j-1]; --j; }
        xs[j] = v;
    }
    return xs[ITERS / 2];
}

static void bench(const char *label, tdc_entropy_id id,
                  const tdc_block *src, size_t raw) {
    tdc_pred2d_params paeth = { .kind = TDC_PRED2D_PAETH };
    tdc_codec_spec spec = {0};
    spec.model        = TDC_MODEL_PRED_2D;
    spec.model_params = &paeth;
    spec.xform[0]     = TDC_XFORM_ZIGZAG;
    spec.xform[1]     = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0]   = id;

    tdc_buffer enc = qs_buffer();
    tdc_status es = tdc_encode_block(src, &spec, &enc);
    if (es != TDC_OK) {
        printf("  %-14s  (encode error: %s)\n", label, tdc_strerror(es));
        qs_buffer_free(&enc);
        return;
    }

    double enc_times[ITERS], dec_times[ITERS];
    for (int i = 0; i < ITERS; ++i) {
        enc.size = 0;
        double t0 = bench_now();
        tdc_encode_block(src, &spec, &enc);
        enc_times[i] = bench_now() - t0;
    }

    tdc_block meta = {0};
    size_t need = 0;
    tdc_decode_peek(enc.data, enc.size, &meta, &need);
    void *dst = qs_realloc(NULL, NULL, need);
    tdc_block out = meta;
    out.data = dst;

    for (int i = 0; i < ITERS; ++i) {
        double t0 = bench_now();
        tdc_decode_block_into(enc.data, enc.size, &out);
        dec_times[i] = bench_now() - t0;
    }

    double mb = (double)raw / 1.0e6;
    printf("  %-14s  enc=%7zu  ratio=%8.2fx  enc=%6.0f MB/s  dec=%6.0f MB/s\n",
           label, enc.size, (double)raw / (double)enc.size,
           mb / median5(enc_times), mb / median5(dec_times));

    qs_realloc(NULL, dst, 0);
    qs_buffer_free(&enc);
}

int main(void) {
    enum { H = 256, W = 256, N = H * W };
    static int16_t raster[N];
    uint32_t lcg = 1u;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            lcg = lcg * 1103515245u + 12345u;
            int jitter = (int)((lcg >> 16) & 0x1F) - 16;
            raster[y * W + x] = (int16_t)(3 * x + 5 * y + jitter);
        }
    }

    tdc_block src = {0};
    src.data = raster;
    src.dtype = TDC_DT_I16;
    src.layout = TDC_LAYOUT_RASTER_2D;
    src.shape.rank = 2;
    src.shape.dim[0] = H;
    src.shape.dim[1] = W;
    tdc_shape_set_contiguous(&src.shape);

    size_t raw = (size_t)N * sizeof(int16_t);
    printf("input: %zu bytes (%dx%d i16 smooth raster + jitter)\n", raw, H, W);
    printf("spec:  PRED_2D/PAETH + ZIGZAG + BYTE_SHUFFLE + <entropy>\n\n");

    bench("NONE",         TDC_ENTROPY_NONE,       &src, raw);
    bench("LZ",           TDC_ENTROPY_LZ,         &src, raw);
    bench("LZ_OPT",       TDC_ENTROPY_LZ_OPT,     &src, raw);
    bench("LZ_STREAMS",   TDC_ENTROPY_LZ_STREAMS, &src, raw);
    bench("LZ_SPLIT",     TDC_ENTROPY_LZ_SPLIT,   &src, raw);
    bench("HUFFMAN",      TDC_ENTROPY_HUFFMAN,    &src, raw);
    bench("HUFFMAN4",     TDC_ENTROPY_HUFFMAN4,   &src, raw);
    bench("FSE",          TDC_ENTROPY_FSE,        &src, raw);
    /* LANE requires tdc_lane_entropy_params with n_lanes matching the
     * upstream BYTE_SHUFFLE elem_size; omitted here to keep the harness
     * parameter-free. See backends/entropy.md for the per-lane walkthrough. */
    return 0;
}
