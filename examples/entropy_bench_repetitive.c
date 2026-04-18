/* docs/examples/entropy_bench_repetitive.c
 *
 * Four entropy backends on 1 MiB of a 32-byte repeating pattern. LZ
 * collapses this to a handful of bytes; Huffman and FSE still pay their
 * alphabet cost because the symbol distribution looks near-uniform to a
 * memoryless coder; NONE is the passthrough reference.
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
    tdc_codec_spec spec = tdc_codec_spec_raw();
    spec.entropy[0] = id;

    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(src, &spec, &enc), label)) return;

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
    printf("  %-10s enc=%8zu  ratio=%9.2fx  enc=%6.0f MB/s  dec=%6.0f MB/s\n",
           label, enc.size, (double)raw / (double)enc.size,
           mb / median5(enc_times), mb / median5(dec_times));

    qs_realloc(NULL, dst, 0);
    qs_buffer_free(&enc);
}

int main(void) {
    enum { N = 1u << 20 };
    static uint8_t buf[N];
    /* 32-byte pattern repeated; every match of 4 bytes or more finds a
     * candidate four bytes earlier in the window. */
    for (size_t i = 0; i < N; ++i) buf[i] = (uint8_t)(i % 32);

    tdc_block src = {0};
    src.data = buf;
    src.dtype = TDC_DT_U8;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank = 1;
    src.shape.dim[0] = N;
    tdc_shape_set_contiguous(&src.shape);

    printf("repetitive input: %d bytes, 32-byte cycle\n", N);
    bench("NONE     ", TDC_ENTROPY_NONE,    &src, N);
    bench("LZ       ", TDC_ENTROPY_LZ,      &src, N);
    bench("HUFFMAN  ", TDC_ENTROPY_HUFFMAN, &src, N);
    bench("FSE      ", TDC_ENTROPY_FSE,     &src, N);
    return 0;
}
