/* docs/examples/tuning_pipeline_compare.c
 *
 * Performance-tuning harness: same 128x256 i16 raster, five codec specs.
 * Reports encoded bytes, ratio, encode MB/s, and decode MB/s for each
 * configuration. Numbers come from a warm-cache median over ITERS runs.
 *
 * The input is a smooth bi-axial gradient with small deterministic jitter
 * on top. Smooth enough that a 2D predictor beats RAW by a wide margin,
 * noisy enough that the zero-residual fast path never fires and the
 * transform and entropy stages have real work to do.
 *
 * Build:
 *   cc -I include docs/examples/tuning_pipeline_compare.c \
 *      build/libtdc.a -lm -o /tmp/tune_pipe
 */

#include "quickstart_common.h"
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
static double tune_now(void) {
    static LARGE_INTEGER freq;
    static int inited = 0;
    LARGE_INTEGER t;
    if (!inited) { QueryPerformanceFrequency(&freq); inited = 1; }
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
}
#else
#  include <time.h>
static double tune_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

#define ITERS 9

/* Reported throughput is uncompressed-bytes / wallclock — the usual way to
 * compare codec speeds. Decode MB/s measured over the encoded bytes going
 * into the pipeline would be misleading when ratios vary 100x across rows. */
static double mb_per_sec(size_t raw_bytes, double secs) {
    if (secs <= 0.0) return 0.0;
    return (double)raw_bytes / secs / 1e6;
}

/* Insertion sort tiny array for median — keep the harness dependency-free. */
static double median(double *xs, int n) {
    for (int i = 1; i < n; ++i) {
        double v = xs[i];
        int j = i;
        while (j > 0 && xs[j - 1] > v) { xs[j] = xs[j - 1]; --j; }
        xs[j] = v;
    }
    return xs[n / 2];
}

typedef struct {
    const char           *label;
    const tdc_codec_spec *spec;
} tune_case;

static void run_case(const char *label, const tdc_block *src,
                     const tdc_codec_spec *spec, size_t raw_bytes) {
    /* Encode once to size the output, then time N more iterations into a
     * re-used buffer so the allocator path is warm. */
    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(src, spec, &enc), label)) exit(1);
    size_t enc_bytes = enc.size;

    double enc_times[ITERS];
    for (int i = 0; i < ITERS; ++i) {
        enc.size = 0;  /* re-use the capacity, no realloc churn */
        double t0 = tune_now();
        tdc_status s = tdc_encode_block(src, spec, &enc);
        double t1 = tune_now();
        if (qs_check(s, label)) exit(1);
        enc_times[i] = t1 - t0;
    }

    /* Decode setup: peek the header, allocate the destination once, run
     * tdc_decode_block_into N times against the same encoded bytes. */
    tdc_block meta = {0};
    size_t need = 0;
    if (qs_check(tdc_decode_peek(enc.data, enc.size, &meta, &need), label)) exit(1);
    void *dst_data = qs_realloc(NULL, NULL, need);
    tdc_block dst = meta;
    dst.data = dst_data;

    double dec_times[ITERS];
    for (int i = 0; i < ITERS; ++i) {
        double t0 = tune_now();
        tdc_status s = tdc_decode_block_into(enc.data, enc.size, &dst);
        double t1 = tune_now();
        if (qs_check(s, label)) exit(1);
        dec_times[i] = t1 - t0;
    }

    /* Sanity: last decode matches source byte-for-byte. */
    if (memcmp(dst_data, src->data, raw_bytes) != 0) {
        fprintf(stderr, "%s: roundtrip mismatch\n", label);
        exit(1);
    }

    double enc_med = median(enc_times, ITERS);
    double dec_med = median(dec_times, ITERS);
    double ratio   = (double)raw_bytes / (double)enc_bytes;

    printf("  %-40s %6zu B  %6.2fx  %8.0f enc MB/s  %8.0f dec MB/s\n",
           label, enc_bytes, ratio,
           mb_per_sec(raw_bytes, enc_med),
           mb_per_sec(raw_bytes, dec_med));

    qs_realloc(NULL, dst_data, 0);
    qs_buffer_free(&enc);
}

int main(void) {
    enum { NY = 128, NX = 256, N = NY * NX };
    static int16_t raster[N];

    /* Smooth bi-axial gradient + small pseudo-random jitter. The LCG has a
     * short enough period that reproducing the numbers on another machine
     * yields the same block. Jitter keeps the residual nonzero so the
     * transform + entropy stages do real work (zero-residual fast path
     * stays off). */
    uint32_t lcg = 1u;
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            lcg = lcg * 1103515245u + 12345u;
            int jitter = (int)((lcg >> 16) & 0x1F) - 16;  /* [-16, 15] */
            int v = 3 * x + 5 * y + jitter;
            raster[y * NX + x] = (int16_t)v;
        }
    }

    tdc_block src = {0};
    src.data   = raster;
    src.dtype  = TDC_DT_I16;
    src.layout = TDC_LAYOUT_RASTER_2D;
    src.shape.rank   = 2;
    src.shape.dim[0] = NY;
    src.shape.dim[1] = NX;
    tdc_shape_set_contiguous(&src.shape);

    size_t raw = N * sizeof(int16_t);
    printf("input: %zu bytes (%dx%d i16, gradient + jitter)\n\n",
           raw, NY, NX);

    /* Case 1: RAW + LZ. Baseline. No model, no transforms. LZ sees raw
     * i16 bytes directly; it finds a few matches inside the low-byte
     * lane but the high-byte lane is near-random. */
    tdc_codec_spec s_raw_lz = tdc_codec_spec_raw();
    s_raw_lz.entropy[0] = TDC_ENTROPY_LZ;

    /* Case 2: PRED_2D / PAETH + LZ. Predictor residual is small and
     * signed; LZ compresses the resulting byte stream directly. */
    tdc_pred2d_params paeth = { .kind = TDC_PRED2D_PAETH };
    tdc_codec_spec s_pred_lz = {0};
    s_pred_lz.model        = TDC_MODEL_PRED_2D;
    s_pred_lz.model_params = &paeth;
    s_pred_lz.entropy[0]   = TDC_ENTROPY_LZ;

    /* Case 3: PRED_2D / PAETH + ZIGZAG + LZ. ZIGZAG turns the negative
     * residuals' 0xFFFF high bytes into 0x0000, which LZ matches readily. */
    tdc_codec_spec s_pred_zz_lz = {0};
    s_pred_zz_lz.model        = TDC_MODEL_PRED_2D;
    s_pred_zz_lz.model_params = &paeth;
    s_pred_zz_lz.xform[0]     = TDC_XFORM_ZIGZAG;
    s_pred_zz_lz.entropy[0]   = TDC_ENTROPY_LZ;

    /* Case 4: PRED_2D / PAETH + ZIGZAG + BYTE_SHUFFLE + LZ. Shuffle
     * groups the high-byte lane (mostly zeros after zigzag) together so
     * LZ collapses it into a single long run. */
    tdc_codec_spec s_full = {0};
    s_full.model        = TDC_MODEL_PRED_2D;
    s_full.model_params = &paeth;
    s_full.xform[0]     = TDC_XFORM_ZIGZAG;
    s_full.xform[1]     = TDC_XFORM_BYTE_SHUFFLE;
    s_full.entropy[0]   = TDC_ENTROPY_LZ;

    /* Case 5: PLANE_2D at the default 32x32 tile, same downstream chain.
     * Different model family: fits a plane per tile, stores three i32
     * coefficients per tile as side metadata, emits residual against
     * the fitted plane. */
    tdc_plane2d_params plane = { .tile_size = 32 };
    tdc_codec_spec s_plane = {0};
    s_plane.model        = TDC_MODEL_PLANE_2D;
    s_plane.model_params = &plane;
    s_plane.xform[0]     = TDC_XFORM_ZIGZAG;
    s_plane.xform[1]     = TDC_XFORM_BYTE_SHUFFLE;
    s_plane.entropy[0]   = TDC_ENTROPY_LZ;

    printf("%-40s %7s  %7s  %13s  %13s\n",
           "  spec", "enc", "ratio", "encode MB/s", "decode MB/s");
    printf("  ----------------------------------------------------"
           "---------------------------------------\n");

    run_case("RAW + LZ",                               &src, &s_raw_lz,     raw);
    run_case("PRED_2D / PAETH + LZ",                   &src, &s_pred_lz,    raw);
    run_case("PRED_2D / PAETH + ZZ + LZ",              &src, &s_pred_zz_lz, raw);
    run_case("PRED_2D / PAETH + ZZ + BSHUF + LZ",      &src, &s_full,       raw);
    run_case("PLANE_2D tile=32 + ZZ + BSHUF + LZ",     &src, &s_plane,      raw);

    return 0;
}
