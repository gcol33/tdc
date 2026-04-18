/* docs/examples/tuning_knob_walk.c
 *
 * Knob-walk harness for the three tunables that matter most in v0:
 *
 *   1. QUANTIZE step size   (scale * target combination)
 *   2. LZ level              (tdc_entropy_level.level, 1..7)
 *   3. BYTE_SHUFFLE grouping (element width driven by dtype, walked by
 *      running the same i64/i32/i16/i8 payload through the chain)
 *
 * Each sub-walk shares the same helper and prints encoded bytes, ratio,
 * encode MB/s, decode MB/s. The input for every sub-walk is sized so the
 * 80-byte block header is a small fraction of the output (no header
 * distortion).
 *
 * Build:
 *   cc -I include docs/examples/tuning_knob_walk.c \
 *      build/libtdc.a -lm -o /tmp/tune_knobs
 */

#include "quickstart_common.h"
#include <math.h>
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

static double mb_per_sec(size_t raw_bytes, double secs) {
    if (secs <= 0.0) return 0.0;
    return (double)raw_bytes / secs / 1e6;
}

static double median(double *xs, int n) {
    for (int i = 1; i < n; ++i) {
        double v = xs[i];
        int j = i;
        while (j > 0 && xs[j - 1] > v) { xs[j] = xs[j - 1]; --j; }
        xs[j] = v;
    }
    return xs[n / 2];
}

/* Single round-trip timing run. Caller provides the block and spec; we
 * encode once to size, ITERS more times for encode-median, then ITERS
 * decodes against the stable encoded bytes. */
static void bench_one(const char *label,
                      const tdc_block    *src,
                      const tdc_codec_spec *spec,
                      size_t              raw_bytes) {
    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(src, spec, &enc), label)) exit(1);
    size_t enc_bytes = enc.size;

    double enc_times[ITERS];
    for (int i = 0; i < ITERS; ++i) {
        enc.size = 0;
        double t0 = tune_now();
        tdc_status s = tdc_encode_block(src, spec, &enc);
        double t1 = tune_now();
        if (qs_check(s, label)) exit(1);
        enc_times[i] = t1 - t0;
    }

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

    double enc_med = median(enc_times, ITERS);
    double dec_med = median(dec_times, ITERS);
    double ratio   = (double)raw_bytes / (double)enc_bytes;

    printf("  %-32s %7zu B  %7.2fx  %8.0f enc MB/s  %8.0f dec MB/s\n",
           label, enc_bytes, ratio,
           mb_per_sec(raw_bytes, enc_med),
           mb_per_sec(raw_bytes, dec_med));

    qs_realloc(NULL, dst_data, 0);
    qs_buffer_free(&enc);
}

/* ----- 1. QUANTIZE step sweep --------------------------------------------- */

static void walk_quantize(void) {
    enum { N = 16384 };
    static double data[N];
    /* Sinusoid in [-100, 100] plus small deterministic jitter. The
     * jitter is sub-unit so coarse scales (step = 1) still lose real
     * information rather than round-tripping exactly. */
    for (int i = 0; i < N; ++i) {
        double base = 100.0 * sin((double)i * 0.01);
        double noise = 0.37 * cos((double)i * 0.73);
        data[i] = base + noise;
    }

    tdc_block src = {0};
    src.data   = data;
    src.dtype  = TDC_DT_F64;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank   = 1;
    src.shape.dim[0] = N;
    tdc_shape_set_contiguous(&src.shape);

    size_t raw = N * sizeof(double);
    printf("1. QUANTIZE step sweep (N=%d f64 sinusoid in [-100, 100]):\n", N);
    printf("   input: %zu bytes\n\n", raw);

    struct { const char *name; double scale; tdc_dtype target; } rows[] = {
        { "scale=1       step=1      i8 ",    1.0,    TDC_DT_I8  },
        { "scale=10      step=0.1    i16",    10.0,   TDC_DT_I16 },
        { "scale=100     step=0.01   i16",    100.0,  TDC_DT_I16 },
        { "scale=1000    step=0.001  i32",    1000.0, TDC_DT_I32 },
    };

    for (size_t k = 0; k < sizeof(rows) / sizeof(rows[0]); ++k) {
        tdc_quantize_params qp = {
            .scale = rows[k].scale, .offset = 0.0, .target = rows[k].target
        };
        tdc_codec_spec spec = {0};
        spec.model        = TDC_MODEL_RAW;
        spec.xform[0]     = TDC_XFORM_QUANTIZE;
        spec.xform[1]     = TDC_XFORM_ZIGZAG;
        spec.xform[2]     = TDC_XFORM_BYTE_SHUFFLE;
        spec.xform_params[0] = &qp;
        spec.entropy[0]   = TDC_ENTROPY_LZ;
        bench_one(rows[k].name, &src, &spec, raw);
    }
    printf("\n");
}

/* ----- 2. LZ level sweep -------------------------------------------------- */

static void walk_lz_level(void) {
    enum { NY = 256, NX = 256, N = NY * NX };
    static int16_t raster[N];

    /* Structured raster: smooth gradient + small jitter, identical idea
     * to the pipeline_compare input but larger so the LZ parser actually
     * finds matches for higher levels to deepen on. */
    uint32_t lcg = 1u;
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            lcg = lcg * 1103515245u + 12345u;
            int jitter = (int)((lcg >> 16) & 0x1F) - 16;
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
    printf("2. LZ level sweep (%dx%d i16 gradient raster):\n", NY, NX);
    printf("   input: %zu bytes  model=PRED_2D/PAETH  xform=ZIGZAG+BSHUF\n\n",
           raw);

    tdc_pred2d_params paeth = { .kind = TDC_PRED2D_PAETH };
    int levels[] = { 1, 2, 3, 5, 7 };

    for (size_t k = 0; k < sizeof(levels) / sizeof(levels[0]); ++k) {
        tdc_entropy_level lvl = { .level = levels[k] };
        tdc_codec_spec spec = {0};
        spec.model           = TDC_MODEL_PRED_2D;
        spec.model_params    = &paeth;
        spec.xform[0]        = TDC_XFORM_ZIGZAG;
        spec.xform[1]        = TDC_XFORM_BYTE_SHUFFLE;
        spec.entropy[0]      = TDC_ENTROPY_LZ;
        spec.entropy_params[0] = &lvl;
        char label[48];
        snprintf(label, sizeof(label), "LZ level=%d", levels[k]);
        bench_one(label, &src, &spec, raw);
    }
    printf("\n");
}

/* ----- 3. BYTE_SHUFFLE grouping sweep ------------------------------------ */
/*
 * BYTE_SHUFFLE has no params — it reads elem_size from the input dtype.
 * Walking the knob therefore means running the same payload shape at
 * four dtype widths (i8, i16, i32, i64). The input is a 1D delta stream
 * with identical per-element structure; only the storage width changes.
 *
 * Absolute ratios are not comparable across widths because the raw byte
 * count differs. The interesting signal is how the *payload shrink*
 * from adding BSHUF varies with elem_size: at elem_size=1 BSHUF is a
 * no-op memcpy and adds zero ratio; at elem_size=8 BSHUF unlocks the
 * most work because seven of the eight lanes are near-constant.
 */

static void walk_shuffle(void) {
    enum { N = 16384 };
    static int8_t  v_i8[N];
    static int16_t v_i16[N];
    static int32_t v_i32[N];
    static int64_t v_i64[N];

    /* Same near-linear monotonic sequence, stored at four widths. The
     * DELTA_1D residual is a single small-magnitude integer stream; what
     * changes with the dtype is how many high-order zero bytes each
     * element carries. */
    for (int i = 0; i < N; ++i) {
        int v = i * 3 + ((i * 17) % 7);
        v_i8[i]  = (int8_t)(v & 0x7F);
        v_i16[i] = (int16_t)v;
        v_i32[i] = v;
        v_i64[i] = v;
    }

    printf("3. BYTE_SHUFFLE grouping sweep (N=%d DELTA_1D integer streams):\n",
           N);
    printf("   model=DELTA_1D  entropy=LZ\n\n");

    struct {
        const char *name;
        void       *data;
        tdc_dtype   dtype;
        size_t      elem;
    } rows[] = {
        { "i8  elem=1 (BSHUF no-op)", v_i8,  TDC_DT_I8,  1 },
        { "i16 elem=2",               v_i16, TDC_DT_I16, 2 },
        { "i32 elem=4",               v_i32, TDC_DT_I32, 4 },
        { "i64 elem=8",               v_i64, TDC_DT_I64, 8 },
    };

    for (size_t k = 0; k < sizeof(rows) / sizeof(rows[0]); ++k) {
        tdc_block src = {0};
        src.data   = rows[k].data;
        src.dtype  = rows[k].dtype;
        src.layout = TDC_LAYOUT_VECTOR_1D;
        src.shape.rank   = 1;
        src.shape.dim[0] = N;
        tdc_shape_set_contiguous(&src.shape);
        size_t raw = (size_t)N * rows[k].elem;

        /* No-BSHUF line: isolates what the shuffle contributes on top of
         * the DELTA_1D + ZIGZAG + LZ baseline. */
        tdc_codec_spec no_bshuf = {0};
        no_bshuf.model      = TDC_MODEL_DELTA_1D;
        no_bshuf.xform[0]   = TDC_XFORM_ZIGZAG;
        no_bshuf.entropy[0] = TDC_ENTROPY_LZ;
        char label_no[64];
        snprintf(label_no, sizeof(label_no),
                 "%s  (no BSHUF)", rows[k].name);
        bench_one(label_no, &src, &no_bshuf, raw);

        /* With BSHUF: the shuffle groups the top (elem-1) byte lanes (all
         * zeros after DELTA+ZIGZAG on this input) into one long run. */
        tdc_codec_spec with_bshuf = {0};
        with_bshuf.model      = TDC_MODEL_DELTA_1D;
        with_bshuf.xform[0]   = TDC_XFORM_ZIGZAG;
        with_bshuf.xform[1]   = TDC_XFORM_BYTE_SHUFFLE;
        with_bshuf.entropy[0] = TDC_ENTROPY_LZ;
        char label_with[64];
        snprintf(label_with, sizeof(label_with),
                 "%s  + BSHUF", rows[k].name);
        bench_one(label_with, &src, &with_bshuf, raw);
    }
    printf("\n");
}

int main(void) {
    printf("%-32s %7s  %7s  %13s  %13s\n",
           "  knob / setting", "enc", "ratio", "encode MB/s", "decode MB/s");
    printf("  -------------------------------------------------------"
           "---------------------------------\n");
    walk_quantize();
    walk_lz_level();
    walk_shuffle();
    return 0;
}
