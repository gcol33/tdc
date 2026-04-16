/*
 * bench/bench_throughput.c
 *
 * Encode / decode throughput + compression ratio for representative
 * tdc pipelines on synthetic blocks.
 *
 * Each case allocates a ~16 MiB block of synthetic data with the right
 * structure for its model (smooth ramp for delta, gradient raster for
 * pred2d, split-plane raster for plane2d, etc.), runs the encode N times
 * to get a stable wallclock minimum, then runs decode N times against
 * the same encoded buffer.
 *
 * Output columns: pipeline | dtype/layout/shape | raw MiB | enc MiB |
 *                 ratio | enc MB/s | dec MB/s
 *
 * Throughput is reported on UNCOMPRESSED bytes (raw_bytes / time), which
 * is the conventional way to compare codec speeds.
 */

#include "tdc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "../src/core/timer.h"
#include "../src/core/decode_profile.h"
#define now_seconds tdc_now_secs

#define ITERS_FULL  5
#define ITERS_SMOKE 1

/* --decode-only: skip encode timing, run more decode iters for stable
 * numbers (task 0.2). Still encodes once for correctness. */
static int g_decode_only = 0;
static int g_decode_iters = 0;  /* 0 => use g_iters */

/* Set by --smoke. When non-zero, every case runs once with a tiny block
 * (~1 KiB), asserting round-trip cleanly and exiting non-zero on any
 * mismatch. Wired into ctest as a sub-second correctness gate so a
 * pipeline that breaks round-trip cannot land silently between bench
 * runs. SPEEDUP-TODO P3.2. */
static int g_smoke = 0;
static int g_iters = ITERS_FULL;

static void *bench_realloc(void *user, void *ptr, size_t new_size) {
    (void)user;
    if (new_size == 0) { free(ptr); return NULL; }
    return realloc(ptr, new_size);
}

static tdc_buffer make_buffer(void) {
    tdc_buffer b = {0};
    b.realloc_fn = bench_realloc;
    return b;
}
static void free_buffer(tdc_buffer *b) {
    if (b->data) free(b->data);
    b->data = NULL; b->size = 0; b->capacity = 0;
}

/* ----- Result row -------------------------------------------------------- */

static void print_header(void) {
    printf("\n");
    printf("%-50s  %12s  %10s  %10s  %6s  %10s  %10s\n",
           "pipeline", "block",
           "raw MiB", "enc MiB", "ratio", "enc MB/s", "dec MB/s");
    printf("%-50s  %12s  %10s  %10s  %6s  %10s  %10s\n",
           "--------------------------------------------------",
           "------------",
           "----------", "----------",
           "------", "----------", "----------");
}

static void print_row(const char *label, const char *block_desc,
                      size_t raw_bytes, size_t enc_bytes,
                      double enc_secs, double dec_secs) {
    double raw_mib   = (double)raw_bytes / (1024.0 * 1024.0);
    double enc_mib   = (double)enc_bytes / (1024.0 * 1024.0);
    double ratio     = (enc_bytes > 0) ? (double)raw_bytes / (double)enc_bytes : 0.0;
    double enc_mbps  = (double)raw_bytes / (1024.0 * 1024.0) / enc_secs;
    double dec_mbps  = (double)raw_bytes / (1024.0 * 1024.0) / dec_secs;
    printf("%-50s  %12s  %10.2f  %10.2f  %6.2fx  %10.1f  %10.1f\n",
           label, block_desc, raw_mib, enc_mib, ratio, enc_mbps, dec_mbps);
}

/* ----- Bench driver: encode N times, decode N times --------------------- */

static int run_case(const char *label, const char *block_desc,
                    const tdc_block *src, const tdc_codec_spec *spec) {
    int64_t n_elems = tdc_shape_n_elems(&src->shape);
    size_t  elem    = tdc_dtype_size(src->dtype);
    size_t  raw_bytes = (size_t)n_elems * elem;

    /* Encode timing: best of ITERS. Reuse the same buffer; the realloc_fn
     * will keep capacity stable after the first iteration.
     * In --decode-only mode we still encode once (for correctness) but
     * skip the timing loop. */
    tdc_buffer enc = make_buffer();
    double best_enc = 1e18;
    int enc_iters = g_decode_only ? 1 : g_iters;
    for (int i = 0; i < enc_iters; ++i) {
        enc.size = 0;  /* reset, retain capacity */
        double t0 = now_seconds();
        tdc_status st = tdc_encode_block(src, spec, &enc);
        double t1 = now_seconds();
        if (st != TDC_OK) {
            fprintf(stderr, "FAIL [%s]: encode -> %d\n", label, (int)st);
            free_buffer(&enc); return 1;
        }
        double dt = t1 - t0;
        if (dt < best_enc) best_enc = dt;
    }
    size_t enc_bytes = enc.size;
    if (g_decode_only) best_enc = 0.0;  /* sentinel: not measured */

    /* Decode timing: best of ITERS into a reusable destination. */
    void *dst_data = malloc(raw_bytes ? raw_bytes : 1);
    if (!dst_data) { free_buffer(&enc); return 1; }

    tdc_block dst = {0};
    dst.data    = dst_data;
    dst.dtype   = src->dtype;
    dst.layout  = src->layout;
    dst.shape   = src->shape;
    dst.offsets = NULL;

    int dec_iters = g_decode_iters > 0 ? g_decode_iters : g_iters;
    double best_dec = 1e18;
    tdc_dp_reset();
    for (int i = 0; i < dec_iters; ++i) {
        double t0 = now_seconds();
        tdc_status st = tdc_decode_block(enc.data, enc.size, &dst);
        double t1 = now_seconds();
        if (st != TDC_OK) {
            fprintf(stderr, "FAIL [%s]: decode -> %d\n", label, (int)st);
            free(dst_data); free_buffer(&enc); return 1;
        }
        double dt = t1 - t0;
        if (dt < best_dec) best_dec = dt;
    }
    tdc_dp_dump(label);

    /* Round-trip sanity: bytes must match. */
    if (memcmp(dst_data, src->data, raw_bytes) != 0) {
        size_t first_off = 0;
        for (size_t k = 0; k < raw_bytes; ++k) {
            if (((uint8_t*)dst_data)[k] != ((uint8_t*)src->data)[k]) {
                first_off = k;
                break;
            }
        }
        fprintf(stderr, "FAIL [%s]: round-trip mismatch at byte %zu / %zu"
                " (got 0x%02x, want 0x%02x)\n",
                label, first_off, raw_bytes,
                ((uint8_t*)dst_data)[first_off],
                ((uint8_t*)src->data)[first_off]);
        free(dst_data); free_buffer(&enc); return 1;
    }

    print_row(label, block_desc, raw_bytes, enc_bytes, best_enc, best_dec);
    free(dst_data);
    free_buffer(&enc);
    return 0;
}

/* ----- Synthetic data generators ---------------------------------------- */

/* Smooth integer ramp: ideal for delta1d. */
static void fill_ramp_i32(int32_t *p, size_t n) {
    for (size_t i = 0; i < n; ++i) p[i] = (int32_t)(1000 + (int64_t)i * 3);
}

/* Mixed-sign random walk: i16, exercises zigzag. */
static void fill_walk_i16(int16_t *p, size_t n) {
    int16_t v = 0;
    uint32_t s = 0xDEADBEEFu;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1103515245u + 12345u;
        int step = (int)((s >> 16) & 0x1F) - 16; /* -16..15 */
        v = (int16_t)(v + step);
        p[i] = v;
    }
}

/* 2D smooth gradient with mild noise: pred2d / plane2d friendly. */
static void fill_grad_u16(uint16_t *p, int ny, int nx) {
    uint32_t s = 0xC0FFEEu;
    for (int r = 0; r < ny; ++r) {
        for (int c = 0; c < nx; ++c) {
            s = s * 1103515245u + 12345u;
            int noise = (int)((s >> 20) & 0x7) - 3; /* -3..4 */
            p[r * nx + c] = (uint16_t)(100 + r * 5 + c * 3 + noise);
        }
    }
}

/* Smooth monotonic f64 with small noise — simulates USGS/NASA lat/lon or
 * elevation rows where consecutive values differ by a small, nearly
 * constant increment. delta1d + ordered-integer mapping produces tiny
 * uint64 residuals; byte shuffle groups the sparse high bytes together
 * for entropy coding. */
static void fill_smooth_f64(double *p, size_t n) {
    uint32_t s = 0xBEEF1234u;
    double v = 1000.0;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1103515245u + 12345u;
        double noise = ((double)((int)((s >> 16) & 0xFF) - 128)) * 1e-6;
        v += 0.001 + noise;
        p[i] = v;
    }
}

/* Two-plane raster: PLANE2D shines here. */
static void fill_split_planes_i32(int32_t *p, int ny, int nx) {
    for (int r = 0; r < ny; ++r) {
        for (int c = 0; c < nx; ++c) {
            int v = (r < ny / 2) ? (200 + r * 4 + c * 2)
                                 : (1000 - r * 3 + c);
            p[r * nx + c] = v;
        }
    }
}

/* 3D smooth scalar field + mild noise: exercise PRED3D's neighborhood. */
static void fill_smooth_f32_vol(float *p, int nz, int ny, int nx) {
    uint32_t s = 0xD0F3C0DEu;
    for (int k = 0; k < nz; ++k) {
        for (int r = 0; r < ny; ++r) {
            for (int c = 0; c < nx; ++c) {
                s = s * 1103515245u + 12345u;
                float noise = (float)((int)((s >> 20) & 0xFF) - 128) * 1e-4f;
                p[(k * ny + r) * nx + c] =
                    0.1f * (float)k + 0.03f * (float)r + 0.02f * (float)c
                    + noise;
            }
        }
    }
}

/* 3D integer gradient + noise: exercise PRED3D integer kernels. */
static void fill_grad_i16_vol(int16_t *p, int nz, int ny, int nx) {
    uint32_t s = 0x0001BEEFu;
    for (int k = 0; k < nz; ++k) {
        for (int r = 0; r < ny; ++r) {
            for (int c = 0; c < nx; ++c) {
                s = s * 1103515245u + 12345u;
                int noise = (int)((s >> 20) & 0x7) - 3;
                p[(k * ny + r) * nx + c] =
                    (int16_t)(k * 5 + r * 3 + c * 2 + noise);
            }
        }
    }
}

/* ----- Cases ------------------------------------------------------------- */

static int case_raw_none_u8(void) {
    /* memcpy passthrough — pure baseline. */
    const size_t N = g_smoke ? 1024u : (size_t)(16 * 1024 * 1024);
    uint8_t *data = (uint8_t *)malloc(N);
    if (!data) return 1;
    /* High-entropy xorshift32 fill — the previous `i & 0xFF` was the
     * sequence 0..255 repeated, which a real entropy coder finds
     * trivially compressible (zstd L1 hits ~9000x on it). The point of
     * RAW+NONE is the memcpy ceiling, not a compressibility benchmark;
     * a random byte stream makes that intent explicit. SPEEDUP-TODO P1.2. */
    {
        uint32_t x = 0xC0FFEE01u;
        for (size_t i = 0; i < N; ++i) {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            data[i] = (uint8_t)x;
        }
    }

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_U8;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = (int64_t)N;
    tdc_shape_set_contiguous(&b.shape);

    tdc_codec_spec s = tdc_codec_spec_raw();
    int rc = run_case("RAW + NONE", "vec1d u8 16M", &b, &s);
    free(data);
    return rc;
}

static int case_raw_lz_ramp(void) {
    const size_t N = g_smoke ? 256u : (size_t)(4 * 1024 * 1024); /* 16 MiB i32 */
    int32_t *data = (int32_t *)malloc(sizeof(int32_t) * N);
    if (!data) return 1;
    fill_ramp_i32(data, N);

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_I32;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = (int64_t)N;
    tdc_shape_set_contiguous(&b.shape);

    tdc_codec_spec s = tdc_codec_spec_raw();
    s.entropy[0] = TDC_ENTROPY_LZ;
    int rc = run_case("RAW + LZ", "vec1d i32 4M (ramp)", &b, &s);
    free(data);
    return rc;
}

/* Isolated Huffman decode: RAW model + no transforms + Huffman.
 * Measures pure entropy throughput without model/xform overhead.
 * Uses u8 random-walk data to get realistic symbol distribution. */
static int case_raw_huffman_u8(void) {
    const size_t N = g_smoke ? 512u : (size_t)(16 * 1024 * 1024);
    uint8_t *data = (uint8_t *)malloc(N);
    if (!data) return 1;
    /* Low-entropy data: biased distribution (mostly low bytes). */
    {
        uint32_t rng = 0xDEADBEEF;
        for (size_t j = 0; j < N; ++j) {
            rng = rng * 1103515245u + 12345u;
            /* Geometric-ish distribution: more short codes, some long. */
            uint32_t v = rng >> 24;
            data[j] = (uint8_t)(v < 128 ? (v >> 4) : v);
        }
    }

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_U8;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = (int64_t)N;
    tdc_shape_set_contiguous(&b.shape);

    int rc = 0;
    {
        tdc_codec_spec s = tdc_codec_spec_raw();
        s.entropy[0] = TDC_ENTROPY_HUFFMAN;
        rc |= run_case("RAW+HUF (isolated)", "vec1d u8 16M", &b, &s);
    }
    {
        tdc_codec_spec s = tdc_codec_spec_raw();
        s.entropy[0] = TDC_ENTROPY_HUFFMAN4;
        rc |= run_case("RAW+HUF4 (isolated)", "vec1d u8 16M", &b, &s);
    }
    free(data);
    return rc;
}

static int case_raw_shuffle_lz_ramp(void) {
    const size_t N = g_smoke ? 256u : (size_t)(4 * 1024 * 1024); /* 16 MiB i32 */
    int32_t *data = (int32_t *)malloc(sizeof(int32_t) * N);
    if (!data) return 1;
    fill_ramp_i32(data, N);

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_I32;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = (int64_t)N;
    tdc_shape_set_contiguous(&b.shape);

    /* The "no model, just shuffle+entropy" floor — exposes how much
     * structure a generic byte-shuffle + LZ can find on a multi-byte
     * dtype without any model in front. SPEEDUP-TODO P1.1. */
    tdc_codec_spec s = tdc_codec_spec_raw();
    s.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
    s.entropy[0]  = TDC_ENTROPY_LZ;
    int rc = run_case("RAW + BSHUF + LZ", "vec1d i32 4M (ramp)", &b, &s);
    free(data);
    return rc;
}

static int case_delta_lz_ramp(void) {
    const size_t N = g_smoke ? 256u : (size_t)(4 * 1024 * 1024);
    int32_t *data = (int32_t *)malloc(sizeof(int32_t) * N);
    if (!data) return 1;
    fill_ramp_i32(data, N);

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_I32;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = (int64_t)N;
    tdc_shape_set_contiguous(&b.shape);

    tdc_codec_spec s = {0};
    s.model   = TDC_MODEL_DELTA_1D;
    s.entropy[0] = TDC_ENTROPY_LZ;
    int rc = run_case("DELTA1D + LZ", "vec1d i32 4M (ramp)", &b, &s);
    free(data);
    return rc;
}

/* Shared harness for DELTA1D+ZIGZAG+BSHUF+<entropy> walk benchmarks. */
static int case_delta_zigzag_shuffle_walk(const char *label,
                                          tdc_entropy_id e0,
                                          tdc_entropy_id e1) {
    const size_t N = g_smoke ? 512u : (size_t)(8 * 1024 * 1024); /* 16 MiB i16 */
    int16_t *data = (int16_t *)malloc(sizeof(int16_t) * N);
    if (!data) return 1;
    fill_walk_i16(data, N);

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_I16;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = (int64_t)N;
    tdc_shape_set_contiguous(&b.shape);

    tdc_codec_spec s = {0};
    s.model    = TDC_MODEL_DELTA_1D;
    s.xform[0] = TDC_XFORM_ZIGZAG;
    s.xform[1] = TDC_XFORM_BYTE_SHUFFLE;
    s.entropy[0] = e0;
    s.entropy[1] = e1;
    int rc = run_case(label, "vec1d i16 8M (walk)", &b, &s);
    free(data);
    return rc;
}

static int case_pred2d_noisy_u16(void) {
    const int NY = g_smoke ? 16 : 2048;
    const int NX = g_smoke ? 16 : 2048; /* 8 MiB u16 (or 512 B smoke) */
    uint16_t *data = (uint16_t *)malloc(sizeof(uint16_t) * (size_t)NY * (size_t)NX);
    if (!data) return 1;
    fill_grad_u16(data, NY, NX);

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_U16;
    b.layout      = TDC_LAYOUT_RASTER_2D;
    b.shape.rank  = 2;
    b.shape.dim[0] = NY; b.shape.dim[1] = NX;
    tdc_shape_set_contiguous(&b.shape);

    const char *desc = "rast2d u16 2048x2048";
    tdc_pred2d_params params; params.kind = TDC_PRED2D_PAETH;
    int rc = 0;

    /* PRED2D(PAETH) + BSHUF + LZ */
    {
        tdc_codec_spec s = {0};
        s.model        = TDC_MODEL_PRED_2D;
        s.model_params = &params;
        s.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0]   = TDC_ENTROPY_LZ;
        rc |= run_case("PRED2D+BSHUF+LZ", desc, &b, &s);
    }
    /* PRED2D(PAETH) + BSHUF + LZ + HUFFMAN */
    {
        tdc_codec_spec s = {0};
        s.model        = TDC_MODEL_PRED_2D;
        s.model_params = &params;
        s.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0]   = TDC_ENTROPY_LZ;
        s.entropy[1]   = TDC_ENTROPY_HUFFMAN;
        rc |= run_case("PRED2D+BSHUF+LZ+HUF", desc, &b, &s);
    }
    /* PRED2D(PAETH) + BSHUF + LZ_OPT */
    {
        tdc_codec_spec s = {0};
        s.model        = TDC_MODEL_PRED_2D;
        s.model_params = &params;
        s.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0]   = TDC_ENTROPY_LZ_OPT;
        rc |= run_case("PRED2D+BSHUF+LZ_OPT", desc, &b, &s);
    }
    /* PRED2D(PAETH) + BSHUF + LZ_OPT + HUFFMAN */
    {
        tdc_codec_spec s = {0};
        s.model        = TDC_MODEL_PRED_2D;
        s.model_params = &params;
        s.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0]   = TDC_ENTROPY_LZ_OPT;
        s.entropy[1]   = TDC_ENTROPY_HUFFMAN;
        rc |= run_case("PRED2D+BSHUF+LZ_OPT+HUF", desc, &b, &s);
    }
    /* PRED2D(PAETH) + BSHUF + FSE */
    {
        tdc_codec_spec s = {0};
        s.model        = TDC_MODEL_PRED_2D;
        s.model_params = &params;
        s.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0]   = TDC_ENTROPY_FSE;
        rc |= run_case("PRED2D+BSHUF+FSE", desc, &b, &s);
    }
    /* PRED2D(PAETH) + BSHUF + HUFFMAN */
    {
        tdc_codec_spec s = {0};
        s.model        = TDC_MODEL_PRED_2D;
        s.model_params = &params;
        s.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0]   = TDC_ENTROPY_HUFFMAN;
        rc |= run_case("PRED2D+BSHUF+HUF", desc, &b, &s);
    }
    /* PRED2D(PAETH) + BSHUF + LZ + FSE */
    {
        tdc_codec_spec s = {0};
        s.model        = TDC_MODEL_PRED_2D;
        s.model_params = &params;
        s.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0]   = TDC_ENTROPY_LZ;
        s.entropy[1]   = TDC_ENTROPY_FSE;
        rc |= run_case("PRED2D+BSHUF+LZ+FSE", desc, &b, &s);
    }

    /* --- UP predictor variants: SIMD-friendly, no left-pixel dependency --- */
    tdc_pred2d_params up_params; up_params.kind = TDC_PRED2D_UP;

    /* PRED2D(UP) + BSHUF + HUF */
    {
        tdc_codec_spec s = {0};
        s.model        = TDC_MODEL_PRED_2D;
        s.model_params = &up_params;
        s.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0]   = TDC_ENTROPY_HUFFMAN;
        rc |= run_case("PRED2D(UP)+BSHUF+HUF", desc, &b, &s);
    }
    /* PRED2D(UP) + BSHUF + HUF4 (4-stream) */
    {
        tdc_codec_spec s = {0};
        s.model        = TDC_MODEL_PRED_2D;
        s.model_params = &up_params;
        s.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0]   = TDC_ENTROPY_HUFFMAN4;
        rc |= run_case("PRED2D(UP)+BSHUF+HUF4", desc, &b, &s);
    }
    /* PRED2D(UP) + BSHUF + FSE */
    {
        tdc_codec_spec s = {0};
        s.model        = TDC_MODEL_PRED_2D;
        s.model_params = &up_params;
        s.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0]   = TDC_ENTROPY_FSE;
        rc |= run_case("PRED2D(UP)+BSHUF+FSE", desc, &b, &s);
    }
    /* PRED2D(UP) + BSHUF + LZ */
    {
        tdc_codec_spec s = {0};
        s.model        = TDC_MODEL_PRED_2D;
        s.model_params = &up_params;
        s.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0]   = TDC_ENTROPY_LZ;
        rc |= run_case("PRED2D(UP)+BSHUF+LZ", desc, &b, &s);
    }

    free(data);
    return rc;
}

static int case_delta1d_f64(void) {
    const size_t N = g_smoke ? 256u : (size_t)(2 * 1024 * 1024); /* 16 MiB f64 */
    double *data = (double *)malloc(sizeof(double) * N);
    if (!data) return 1;
    fill_smooth_f64(data, N);

    const char *desc = g_smoke ? "vec1d f64 256" : "vec1d f64 2M (smooth)";
    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_F64;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = (int64_t)N;
    tdc_shape_set_contiguous(&b.shape);

    int rc = 0;

    /* DELTA1D + BSHUF + LZ — baseline for f64. */
    {
        tdc_codec_spec s = {0};
        s.model    = TDC_MODEL_DELTA_1D;
        s.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0] = TDC_ENTROPY_LZ;
        rc |= run_case("DELTA1D+BSHUF+LZ", desc, &b, &s);
    }
    /* DELTA1D + BSHUF + HUF */
    {
        tdc_codec_spec s = {0};
        s.model    = TDC_MODEL_DELTA_1D;
        s.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0] = TDC_ENTROPY_HUFFMAN;
        rc |= run_case("DELTA1D+BSHUF+HUF", desc, &b, &s);
    }
    /* DELTA1D + BSHUF + FSE */
    {
        tdc_codec_spec s = {0};
        s.model    = TDC_MODEL_DELTA_1D;
        s.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0] = TDC_ENTROPY_FSE;
        rc |= run_case("DELTA1D+BSHUF+FSE", desc, &b, &s);
    }
    /* DELTA1D + BSHUF + LZ + HUF */
    {
        tdc_codec_spec s = {0};
        s.model    = TDC_MODEL_DELTA_1D;
        s.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0] = TDC_ENTROPY_LZ;
        s.entropy[1] = TDC_ENTROPY_HUFFMAN;
        rc |= run_case("DELTA1D+BSHUF+LZ+HUF", desc, &b, &s);
    }
    /* DELTA1D + LZ (no shuffle) — reference. */
    {
        tdc_codec_spec s = {0};
        s.model    = TDC_MODEL_DELTA_1D;
        s.entropy[0] = TDC_ENTROPY_LZ;
        rc |= run_case("DELTA1D+LZ", desc, &b, &s);
    }
    /* RAW + BSHUF + LZ — no model, just shuffle. */
    {
        tdc_codec_spec s = tdc_codec_spec_raw();
        s.xform[0]   = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0] = TDC_ENTROPY_LZ;
        rc |= run_case("RAW+BSHUF+LZ", desc, &b, &s);
    }
    /* RAW + LZ_STREAMS — per-stream entropy coding. */
    {
        tdc_codec_spec s = tdc_codec_spec_raw();
        s.entropy[0] = TDC_ENTROPY_LZ_STREAMS;
        rc |= run_case("RAW+LZ_STREAMS", desc, &b, &s);
    }
    /* RAW + LZ_STREAMS L1 — flat hash + accel step, fast encode. */
    {
        tdc_codec_spec s = tdc_codec_spec_raw();
        s.entropy[0] = TDC_ENTROPY_LZ_STREAMS;
        static const tdc_lz_streams_params lzs_l1 = { .level = 1 };
        s.entropy_params[0] = &lzs_l1;
        rc |= run_case("RAW+LZ_STREAMS L1", desc, &b, &s);
    }
    /* FPC1D + BSHUF + LZ — head-to-head with DELTA1D on identical input. */
    {
        tdc_codec_spec s = {0};
        s.model      = TDC_MODEL_FPC_1D;
        s.xform[0]   = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0] = TDC_ENTROPY_LZ;
        rc |= run_case("FPC+BSHUF+LZ", desc, &b, &s);
    }
    /* FPC1D + BSHUF + HUF */
    {
        tdc_codec_spec s = {0};
        s.model      = TDC_MODEL_FPC_1D;
        s.xform[0]   = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0] = TDC_ENTROPY_HUFFMAN;
        rc |= run_case("FPC+BSHUF+HUF", desc, &b, &s);
    }

    free(data);
    return rc;
}

/* 3D f32 volume: exercises pred3d_float.c. Size chosen to match the 2D
 * bench (~8 MiB): 128³ f32 = 8 MiB. Smoke uses 16³. */
static int case_pred3d_f32(void) {
    const int N = g_smoke ? 16 : 128;
    float *data = (float *)malloc(sizeof(float) * (size_t)N * N * N);
    if (!data) return 1;
    fill_smooth_f32_vol(data, N, N, N);

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_F32;
    b.layout      = TDC_LAYOUT_VOLUME_3D;
    b.shape.rank  = 3;
    b.shape.dim[0] = N; b.shape.dim[1] = N; b.shape.dim[2] = N;
    tdc_shape_set_contiguous(&b.shape);

    const char *desc = g_smoke ? "vol3d f32 16^3" : "vol3d f32 128^3";
    tdc_pred3d_params params; params.kind = TDC_PRED3D_GRAD3D;
    int rc = 0;

    /* PRED3D(GRAD3D) + LZ */
    {
        tdc_codec_spec s = {0};
        s.model        = TDC_MODEL_PRED_3D;
        s.model_params = &params;
        s.entropy[0]   = TDC_ENTROPY_LZ;
        rc |= run_case("PRED3D(GRAD)+LZ", desc, &b, &s);
    }
    /* PRED3D(GRAD3D) + BSHUF + LZ */
    {
        tdc_codec_spec s = {0};
        s.model        = TDC_MODEL_PRED_3D;
        s.model_params = &params;
        s.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0]   = TDC_ENTROPY_LZ;
        rc |= run_case("PRED3D(GRAD)+BSHUF+LZ", desc, &b, &s);
    }
    /* PRED3D(AUTO) + BSHUF + HUF — let encoder pick predictor. */
    {
        tdc_pred3d_params auto_p = { .kind = TDC_PRED3D_AUTO };
        tdc_codec_spec s = {0};
        s.model        = TDC_MODEL_PRED_3D;
        s.model_params = &auto_p;
        s.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0]   = TDC_ENTROPY_HUFFMAN;
        rc |= run_case("PRED3D(AUTO)+BSHUF+HUF", desc, &b, &s);
    }
    free(data);
    return rc;
}

/* 3D i16 volume: exercises pred3d.c integer kernels. 128³ = 4 MiB. */
static int case_pred3d_i16(void) {
    const int N = g_smoke ? 16 : 128;
    int16_t *data = (int16_t *)malloc(sizeof(int16_t) * (size_t)N * N * N);
    if (!data) return 1;
    fill_grad_i16_vol(data, N, N, N);

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_I16;
    b.layout      = TDC_LAYOUT_VOLUME_3D;
    b.shape.rank  = 3;
    b.shape.dim[0] = N; b.shape.dim[1] = N; b.shape.dim[2] = N;
    tdc_shape_set_contiguous(&b.shape);

    const char *desc = g_smoke ? "vol3d i16 16^3" : "vol3d i16 128^3";
    int rc = 0;

    /* PRED3D(AUTO) + LZ — baseline. */
    {
        tdc_pred3d_params params = { .kind = TDC_PRED3D_AUTO };
        tdc_codec_spec s = {0};
        s.model        = TDC_MODEL_PRED_3D;
        s.model_params = &params;
        s.entropy[0]   = TDC_ENTROPY_LZ;
        rc |= run_case("PRED3D(AUTO)+LZ", desc, &b, &s);
    }
    /* PRED3D(AUTO) + ZIGZAG + BSHUF + LZ — exercises residual pipeline. */
    {
        tdc_pred3d_params params = { .kind = TDC_PRED3D_AUTO };
        tdc_codec_spec s = {0};
        s.model        = TDC_MODEL_PRED_3D;
        s.model_params = &params;
        s.xform[0]     = TDC_XFORM_ZIGZAG;
        s.xform[1]     = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0]   = TDC_ENTROPY_LZ;
        rc |= run_case("PRED3D(AUTO)+ZZ+BSHUF+LZ", desc, &b, &s);
    }
    /* PRED3D(GRAD3D) + ZIGZAG + BSHUF + HUFFMAN */
    {
        tdc_pred3d_params params = { .kind = TDC_PRED3D_GRAD3D };
        tdc_codec_spec s = {0};
        s.model        = TDC_MODEL_PRED_3D;
        s.model_params = &params;
        s.xform[0]     = TDC_XFORM_ZIGZAG;
        s.xform[1]     = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0]   = TDC_ENTROPY_HUFFMAN;
        rc |= run_case("PRED3D(GRAD)+ZZ+BSHUF+HUF", desc, &b, &s);
    }
    free(data);
    return rc;
}

static int case_plane2d_shuffle_lz(void) {
    const int NY = g_smoke ? 32 : 1024;
    const int NX = g_smoke ? 32 : 1024; /* 4 MiB i32 (or 4 KiB smoke) */
    int32_t *data = (int32_t *)malloc(sizeof(int32_t) * (size_t)NY * (size_t)NX);
    if (!data) return 1;
    fill_split_planes_i32(data, NY, NX);

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_I32;
    b.layout      = TDC_LAYOUT_RASTER_2D;
    b.shape.rank  = 2;
    b.shape.dim[0] = NY; b.shape.dim[1] = NX;
    tdc_shape_set_contiguous(&b.shape);

    tdc_plane2d_params params = {0}; params.tile_size = 32;
    tdc_codec_spec s = {0};
    s.model        = TDC_MODEL_PLANE_2D;
    s.model_params = &params;
    s.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
    s.entropy[0]      = TDC_ENTROPY_LZ;
    int rc = run_case("PLANE2D+BSHUF+LZ", "rast2d i32 1024x1024", &b, &s);
    free(data);
    return rc;
}

/* ----- Entropy chain bench cases ----------------------------------------- *
 * Same input as the DELTA+ZZ+BSHUF walk above, but swap LZ for
 * HUFFMAN / FSE / chained combinations. Lets RESULTS.md compare
 * per-stage throughput and ratio for the new entropy backends against
 * the LZ baseline on identical residual streams. */

/* Entropy chain bench cases removed — now use case_delta_zigzag_shuffle_walk. */

/* ----- --from <file> mode (real data) ------------------------------------- *
 * Reads a flat binary blob, interprets it as `dtype` with the given shape,
 * and runs every applicable pipeline against it. Gives the same numbers
 * as the synthetic cases but on data the user actually cares about.
 * SPEEDUP-TODO P1.3.
 */

static int parse_dtype(const char *s, tdc_dtype *out) {
    if (!strcmp(s, "i8"))  { *out = TDC_DT_I8;  return 0; }
    if (!strcmp(s, "u8"))  { *out = TDC_DT_U8;  return 0; }
    if (!strcmp(s, "i16")) { *out = TDC_DT_I16; return 0; }
    if (!strcmp(s, "u16")) { *out = TDC_DT_U16; return 0; }
    if (!strcmp(s, "i32")) { *out = TDC_DT_I32; return 0; }
    if (!strcmp(s, "u32")) { *out = TDC_DT_U32; return 0; }
    if (!strcmp(s, "i64")) { *out = TDC_DT_I64; return 0; }
    if (!strcmp(s, "u64")) { *out = TDC_DT_U64; return 0; }
    if (!strcmp(s, "f32")) { *out = TDC_DT_F32; return 0; }
    if (!strcmp(s, "f64")) { *out = TDC_DT_F64; return 0; }
    return 1;
}

/* "1024x1024" -> rank=2 dim={1024,1024}; "16777216" -> rank=1. */
static int parse_shape(const char *s, tdc_shape *out) {
    memset(out, 0, sizeof(*out));
    int rank = 0;
    const char *p = s;
    while (*p && rank < 4) {
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p || v <= 0) return 1;
        out->dim[rank++] = v;
        p = end;
        if (*p == 'x' || *p == 'X' || *p == ',') ++p;
        else if (*p) return 1;
    }
    if (rank == 0) return 1;
    out->rank = (uint8_t)rank;
    tdc_shape_set_contiguous(out);
    return 0;
}

static void *slurp_file(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long len = ftell(f);
    if (len < 0) { fclose(f); return NULL; }
    rewind(f);
    void *buf = malloc((size_t)len ? (size_t)len : 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)len, f);
    fclose(f);
    if (got != (size_t)len) { free(buf); return NULL; }
    *out_size = (size_t)len;
    return buf;
}

static int run_from_file(const char *path, const char *dtype_s,
                         const char *shape_s, const char *layout_s) {
    tdc_dtype dtype;
    if (parse_dtype(dtype_s, &dtype) != 0) {
        fprintf(stderr, "bad --dtype: %s\n", dtype_s); return 1;
    }
    tdc_shape shape;
    if (parse_shape(shape_s, &shape) != 0) {
        fprintf(stderr, "bad --shape: %s\n", shape_s); return 1;
    }
    tdc_layout layout = TDC_LAYOUT_VECTOR_1D;
    if (layout_s) {
        if      (!strcmp(layout_s, "vec1d"))  layout = TDC_LAYOUT_VECTOR_1D;
        else if (!strcmp(layout_s, "rast2d")) layout = TDC_LAYOUT_RASTER_2D;
        else if (!strcmp(layout_s, "vol3d"))  layout = TDC_LAYOUT_VOLUME_3D;
        else { fprintf(stderr, "bad --layout: %s\n", layout_s); return 1; }
    } else if (shape.rank == 2) {
        layout = TDC_LAYOUT_RASTER_2D;
    } else if (shape.rank == 3) {
        layout = TDC_LAYOUT_VOLUME_3D;
    }

    size_t file_bytes = 0;
    void *data = slurp_file(path, &file_bytes);
    if (!data) { fprintf(stderr, "cannot read: %s\n", path); return 1; }

    int64_t n_elems = tdc_shape_n_elems(&shape);
    size_t expect = (size_t)n_elems * tdc_dtype_size(dtype);
    if (expect != file_bytes) {
        fprintf(stderr,
                "size mismatch: file=%zu B, dtype*shape=%zu B (n=%lld * %zu)\n",
                file_bytes, expect, (long long)n_elems,
                tdc_dtype_size(dtype));
        free(data); return 1;
    }

    tdc_block b = {0};
    b.data    = data;
    b.dtype   = dtype;
    b.layout  = layout;
    b.shape   = shape;

    char block_desc[24];
    snprintf(block_desc, sizeof(block_desc), "%s %s", dtype_s, shape_s);

    int rc = 0;

    /* RAW + LZ — generic floor with no model. */
    {
        tdc_codec_spec s = tdc_codec_spec_raw();
        s.entropy[0] = TDC_ENTROPY_LZ;
        rc |= run_case("RAW + LZ", block_desc, &b, &s);
    }
    /* RAW + LZ+HUF / LZ+FSE — entropy-coded LZ without byte shuffle. */
    {
        tdc_codec_spec slh = tdc_codec_spec_raw();
        slh.entropy[0] = TDC_ENTROPY_LZ;
        slh.entropy[1] = TDC_ENTROPY_HUFFMAN;
        rc |= run_case("RAW + LZ+HUF", block_desc, &b, &slh);
    }
    {
        tdc_codec_spec slf = tdc_codec_spec_raw();
        slf.entropy[0] = TDC_ENTROPY_LZ;
        slf.entropy[1] = TDC_ENTROPY_FSE;
        rc |= run_case("RAW + LZ+FSE", block_desc, &b, &slf);
    }
    /* RAW + LZ_OPT — optimal parser, better match quality. */
    {
        tdc_codec_spec so = tdc_codec_spec_raw();
        so.entropy[0] = TDC_ENTROPY_LZ_OPT;
        rc |= run_case("RAW + LZ_OPT", block_desc, &b, &so);
    }
    {
        tdc_codec_spec soh = tdc_codec_spec_raw();
        soh.entropy[0] = TDC_ENTROPY_LZ_OPT;
        soh.entropy[1] = TDC_ENTROPY_HUFFMAN;
        rc |= run_case("RAW + LZ_OPT+HUF", block_desc, &b, &soh);
    }
    {
        tdc_codec_spec sof = tdc_codec_spec_raw();
        sof.entropy[0] = TDC_ENTROPY_LZ_OPT;
        sof.entropy[1] = TDC_ENTROPY_FSE;
        rc |= run_case("RAW + LZ_OPT+FSE", block_desc, &b, &sof);
    }
    /* RAW + LZ_SPLIT — optimal parser + split descriptor/literal Huffman. */
    {
        tdc_codec_spec ssp = tdc_codec_spec_raw();
        ssp.entropy[0] = TDC_ENTROPY_LZ_SPLIT;
        rc |= run_case("RAW + LZ_SPLIT", block_desc, &b, &ssp);
    }
    /* RAW + LZ_STREAMS — per-stream entropy, streams-aware optimal parser. */
    {
        tdc_codec_spec ss = tdc_codec_spec_raw();
        ss.entropy[0] = TDC_ENTROPY_LZ_STREAMS;
        rc |= run_case("RAW + LZ_STREAMS", block_desc, &b, &ss);
    }
    /* RAW + LZ_STREAMS L1 — flat hash, no lazy, fast STREAMS encode. */
    {
        tdc_codec_spec ss = tdc_codec_spec_raw();
        ss.entropy[0] = TDC_ENTROPY_LZ_STREAMS;
        static const tdc_lz_streams_params lzs_l1 = { .level = 1 };
        ss.entropy_params[0] = &lzs_l1;
        rc |= run_case("RAW + LZ_STREAMS L1", block_desc, &b, &ss);
    }
    /* RAW + BSHUF + LZ — exposes per-lane structure on multi-byte dtypes. */
    if (tdc_dtype_size(dtype) > 1) {
        tdc_codec_spec s = tdc_codec_spec_raw();
        s.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0]  = TDC_ENTROPY_LZ;
        rc |= run_case("RAW + BSHUF + LZ", block_desc, &b, &s);
    }
    /* DELTA1D family — integer dtypes use zigzag+bshuf; float dtypes use
     * ordered-integer residuals (uint64/32/16) with bshuf only (no zigzag). */
    if (layout == TDC_LAYOUT_VECTOR_1D &&
        dtype != TDC_DT_F32 && dtype != TDC_DT_F64) {
        tdc_codec_spec s = {0};
        s.model   = TDC_MODEL_DELTA_1D;
        s.entropy[0] = TDC_ENTROPY_LZ;
        rc |= run_case("DELTA1D + LZ", block_desc, &b, &s);

        tdc_codec_spec s2 = {0};
        s2.model    = TDC_MODEL_DELTA_1D;
        s2.xform[0] = TDC_XFORM_ZIGZAG;
        s2.xform[1] = TDC_XFORM_BYTE_SHUFFLE;
        s2.entropy[0]  = TDC_ENTROPY_LZ;
        rc |= run_case("DELTA1D+ZIGZAG+BSHUF+LZ", block_desc, &b, &s2);

        /* Same residual stream, different entropy backends. */
        tdc_codec_spec sh = {0};
        sh.model    = TDC_MODEL_DELTA_1D;
        sh.xform[0] = TDC_XFORM_ZIGZAG;
        sh.xform[1] = TDC_XFORM_BYTE_SHUFFLE;
        sh.entropy[0] = TDC_ENTROPY_HUFFMAN;
        rc |= run_case("DELTA1D+ZZ+BSHUF+HUFFMAN", block_desc, &b, &sh);

        tdc_codec_spec sf = {0};
        sf.model    = TDC_MODEL_DELTA_1D;
        sf.xform[0] = TDC_XFORM_ZIGZAG;
        sf.xform[1] = TDC_XFORM_BYTE_SHUFFLE;
        sf.entropy[0] = TDC_ENTROPY_FSE;
        rc |= run_case("DELTA1D+ZZ+BSHUF+FSE", block_desc, &b, &sf);

        tdc_codec_spec slh = {0};
        slh.model    = TDC_MODEL_DELTA_1D;
        slh.xform[0] = TDC_XFORM_ZIGZAG;
        slh.xform[1] = TDC_XFORM_BYTE_SHUFFLE;
        slh.entropy[0] = TDC_ENTROPY_LZ;
        slh.entropy[1] = TDC_ENTROPY_HUFFMAN;
        rc |= run_case("DELTA1D+ZZ+BSHUF+LZ+HUF", block_desc, &b, &slh);
    }
    /* DELTA1D on float dtypes — ordered-integer residuals, no zigzag. */
    if (layout == TDC_LAYOUT_VECTOR_1D &&
        (dtype == TDC_DT_F32 || dtype == TDC_DT_F64)) {
        tdc_codec_spec s = {0};
        s.model    = TDC_MODEL_DELTA_1D;
        s.entropy[0] = TDC_ENTROPY_LZ;
        rc |= run_case("DELTA1D+LZ", block_desc, &b, &s);

        tdc_codec_spec sb = {0};
        sb.model    = TDC_MODEL_DELTA_1D;
        sb.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
        sb.entropy[0] = TDC_ENTROPY_LZ;
        rc |= run_case("DELTA1D+BSHUF+LZ", block_desc, &b, &sb);

        tdc_codec_spec sh = {0};
        sh.model    = TDC_MODEL_DELTA_1D;
        sh.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
        sh.entropy[0] = TDC_ENTROPY_HUFFMAN;
        rc |= run_case("DELTA1D+BSHUF+HUF", block_desc, &b, &sh);

        tdc_codec_spec sh4 = {0};
        sh4.model    = TDC_MODEL_DELTA_1D;
        sh4.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
        sh4.entropy[0] = TDC_ENTROPY_HUFFMAN4;
        rc |= run_case("DELTA1D+BSHUF+HUF4", block_desc, &b, &sh4);

        tdc_codec_spec sf = {0};
        sf.model    = TDC_MODEL_DELTA_1D;
        sf.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
        sf.entropy[0] = TDC_ENTROPY_FSE;
        rc |= run_case("DELTA1D+BSHUF+FSE", block_desc, &b, &sf);

        /* LZ chained with entropy back-end. */
        tdc_codec_spec slh = {0};
        slh.model    = TDC_MODEL_DELTA_1D;
        slh.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
        slh.entropy[0] = TDC_ENTROPY_LZ;
        slh.entropy[1] = TDC_ENTROPY_HUFFMAN;
        rc |= run_case("DELTA1D+BSHUF+LZ+HUF", block_desc, &b, &slh);

        tdc_codec_spec slf = {0};
        slf.model    = TDC_MODEL_DELTA_1D;
        slf.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
        slf.entropy[0] = TDC_ENTROPY_LZ;
        slf.entropy[1] = TDC_ENTROPY_FSE;
        rc |= run_case("DELTA1D+BSHUF+LZ+FSE", block_desc, &b, &slf);

        /* LZ_OPT — optimal parser, no/with second-stage entropy. */
        {
            tdc_codec_spec so = {0};
            so.model    = TDC_MODEL_DELTA_1D;
            so.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
            so.entropy[0] = TDC_ENTROPY_LZ_OPT;
            so.entropy[1] = TDC_ENTROPY_HUFFMAN;
            rc |= run_case("DELTA1D+BSHUF+LZ_OPT+HUF", block_desc, &b, &so);
        }
        {
            tdc_codec_spec so2 = {0};
            so2.model    = TDC_MODEL_DELTA_1D;
            so2.entropy[0] = TDC_ENTROPY_LZ_OPT;
            so2.entropy[1] = TDC_ENTROPY_HUFFMAN;
            rc |= run_case("DELTA1D+LZ_OPT+HUF", block_desc, &b, &so2);
        }

        /* LZ_STREAMS — per-stream entropy coding. */
        tdc_codec_spec sls = {0};
        sls.model    = TDC_MODEL_DELTA_1D;
        sls.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
        sls.entropy[0] = TDC_ENTROPY_LZ_STREAMS;
        rc |= run_case("DELTA1D+BSHUF+LZ_STREAMS", block_desc, &b, &sls);

        /* No model: BSHUF + alternative entropy. */
        tdc_codec_spec bslh = tdc_codec_spec_raw();
        bslh.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
        bslh.entropy[0] = TDC_ENTROPY_LZ;
        bslh.entropy[1] = TDC_ENTROPY_HUFFMAN;
        rc |= run_case("RAW+BSHUF+LZ+HUF", block_desc, &b, &bslh);

        tdc_codec_spec bslf = tdc_codec_spec_raw();
        bslf.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
        bslf.entropy[0] = TDC_ENTROPY_LZ;
        bslf.entropy[1] = TDC_ENTROPY_FSE;
        rc |= run_case("RAW+BSHUF+LZ+FSE", block_desc, &b, &bslf);

        tdc_codec_spec bsls = tdc_codec_spec_raw();
        bsls.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
        bsls.entropy[0] = TDC_ENTROPY_LZ_STREAMS;
        rc |= run_case("RAW+BSHUF+LZ_STREAMS", block_desc, &b, &bsls);

        /* LANE — per-lane entropy after byte shuffle. */
        {
            tdc_lane_entropy_params lp = {0};
            lp.n_lanes = (uint8_t)tdc_dtype_size(dtype);
            tdc_codec_spec bsla = tdc_codec_spec_raw();
            bsla.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
            bsla.entropy[0] = TDC_ENTROPY_LANE;
            bsla.entropy_params[0] = &lp;
            rc |= run_case("RAW+BSHUF+LANE", block_desc, &b, &bsla);
        }

        /* DELTA1D + BSHUF + LANE — XOR-delta separates byte lanes. */
        {
            tdc_lane_entropy_params lp = {0};
            lp.n_lanes = (uint8_t)tdc_dtype_size(dtype);
            tdc_codec_spec sl = {0};
            sl.model    = TDC_MODEL_DELTA_1D;
            sl.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
            sl.entropy[0] = TDC_ENTROPY_LANE;
            sl.entropy_params[0] = &lp;
            rc |= run_case("DELTA1D+BSHUF+LANE", block_desc, &b, &sl);
        }

        /* DELTA2 — second-order XOR-delta. */
        {
            tdc_codec_spec s = {0};
            s.model    = TDC_MODEL_DELTA2_1D;
            s.entropy[0] = TDC_ENTROPY_LZ;
            rc |= run_case("DELTA2+LZ", block_desc, &b, &s);
        }
        {
            tdc_codec_spec s = {0};
            s.model    = TDC_MODEL_DELTA2_1D;
            s.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
            s.entropy[0] = TDC_ENTROPY_LZ;
            rc |= run_case("DELTA2+BSHUF+LZ", block_desc, &b, &s);
        }
        {
            tdc_codec_spec s = {0};
            s.model    = TDC_MODEL_DELTA2_1D;
            s.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
            s.entropy[0] = TDC_ENTROPY_LZ_STREAMS;
            rc |= run_case("DELTA2+BSHUF+LZ_STREAMS", block_desc, &b, &s);
        }
        {
            tdc_lane_entropy_params lp = {0};
            lp.n_lanes = (uint8_t)tdc_dtype_size(dtype);
            tdc_codec_spec s = {0};
            s.model    = TDC_MODEL_DELTA2_1D;
            s.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
            s.entropy[0] = TDC_ENTROPY_LANE;
            s.entropy_params[0] = &lp;
            rc |= run_case("DELTA2+BSHUF+LANE", block_desc, &b, &s);
        }
        {
            tdc_codec_spec s = {0};
            s.model    = TDC_MODEL_DELTA2_1D;
            s.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
            s.entropy[0] = TDC_ENTROPY_LZ;
            s.entropy[1] = TDC_ENTROPY_HUFFMAN;
            rc |= run_case("DELTA2+BSHUF+LZ+HUF", block_desc, &b, &s);
        }

        /* FPC — dual-predictor FCM+DFCM. */
        {
            tdc_codec_spec s = {0};
            s.model    = TDC_MODEL_FPC_1D;
            s.entropy[0] = TDC_ENTROPY_LZ;
            rc |= run_case("FPC+LZ", block_desc, &b, &s);
        }
        {
            tdc_codec_spec s = {0};
            s.model    = TDC_MODEL_FPC_1D;
            s.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
            s.entropy[0] = TDC_ENTROPY_LZ;
            rc |= run_case("FPC+BSHUF+LZ", block_desc, &b, &s);
        }
        {
            tdc_codec_spec s = {0};
            s.model    = TDC_MODEL_FPC_1D;
            s.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
            s.entropy[0] = TDC_ENTROPY_LZ_STREAMS;
            rc |= run_case("FPC+BSHUF+LZ_STREAMS", block_desc, &b, &s);
        }
        {
            tdc_lane_entropy_params lp = {0};
            lp.n_lanes = (uint8_t)tdc_dtype_size(dtype);
            tdc_codec_spec s = {0};
            s.model    = TDC_MODEL_FPC_1D;
            s.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
            s.entropy[0] = TDC_ENTROPY_LANE;
            s.entropy_params[0] = &lp;
            rc |= run_case("FPC+BSHUF+LANE", block_desc, &b, &s);
        }
        {
            tdc_codec_spec s = {0};
            s.model    = TDC_MODEL_FPC_1D;
            s.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
            s.entropy[0] = TDC_ENTROPY_HUFFMAN;
            rc |= run_case("FPC+BSHUF+HUF", block_desc, &b, &s);
        }
        {
            tdc_codec_spec s = {0};
            s.model    = TDC_MODEL_FPC_1D;
            s.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
            s.entropy[0] = TDC_ENTROPY_LZ;
            s.entropy[1] = TDC_ENTROPY_HUFFMAN;
            rc |= run_case("FPC+BSHUF+LZ+HUF", block_desc, &b, &s);
        }
    }
    /* PRED2D + PLANE2D — only on 2D integer rasters. */
    if (layout == TDC_LAYOUT_RASTER_2D &&
        dtype != TDC_DT_F32 && dtype != TDC_DT_F64) {
        tdc_pred2d_params pp; pp.kind = TDC_PRED2D_PAETH;
        tdc_codec_spec s = {0};
        s.model        = TDC_MODEL_PRED_2D;
        s.model_params = &pp;
        s.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0]      = TDC_ENTROPY_LZ;
        rc |= run_case("PRED2D(PAETH)+BSHUF+LZ", block_desc, &b, &s);

        tdc_codec_spec sph = {0};
        sph.model        = TDC_MODEL_PRED_2D;
        sph.model_params = &pp;
        sph.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
        sph.entropy[0]   = TDC_ENTROPY_LZ;
        sph.entropy[1]   = TDC_ENTROPY_HUFFMAN;
        rc |= run_case("PRED2D+BSHUF+LZ+HUF", block_desc, &b, &sph);

        if (dtype == TDC_DT_I32 || dtype == TDC_DT_U32 ||
            dtype == TDC_DT_I16 || dtype == TDC_DT_U16) {
            tdc_plane2d_params plp = {0}; plp.tile_size = 32;
            tdc_codec_spec s2 = {0};
            s2.model        = TDC_MODEL_PLANE_2D;
            s2.model_params = &plp;
            s2.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
            s2.entropy[0]      = TDC_ENTROPY_LZ;
            rc |= run_case("PLANE2D+BSHUF+LZ", block_desc, &b, &s2);
        }
    }

    free(data);
    return rc;
}

/* ----- main -------------------------------------------------------------- */

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s                                  # synthetic suite\n"
        "       %s --smoke                          # round-trip smoke (~ms)\n"
        "       %s --decode-only [--decode-iters N] # skip encode timing\n"
        "       %s --from PATH --dtype NAME --shape DIMS [--layout L]\n"
        "  NAME  in {i8,u8,i16,u16,i32,u32,i64,u64,f32,f64}\n"
        "  DIMS  e.g. 16777216 or 2048x2048\n"
        "  L     in {vec1d,rast2d,vol3d} (defaults from rank)\n",
        argv0, argv0, argv0, argv0);
}

int main(int argc, char **argv) {
    const char *from_path = NULL;
    const char *dtype_s   = NULL;
    const char *shape_s   = NULL;
    const char *layout_s  = NULL;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--from")   && i + 1 < argc) from_path = argv[++i];
        else if (!strcmp(argv[i], "--dtype")  && i + 1 < argc) dtype_s  = argv[++i];
        else if (!strcmp(argv[i], "--shape")  && i + 1 < argc) shape_s  = argv[++i];
        else if (!strcmp(argv[i], "--layout") && i + 1 < argc) layout_s = argv[++i];
        else if (!strcmp(argv[i], "--smoke")) g_smoke = 1;
        else if (!strcmp(argv[i], "--decode-only")) g_decode_only = 1;
        else if (!strcmp(argv[i], "--profile")) tdc_dp_force_enable();
        else if (!strcmp(argv[i], "--decode-iters") && i + 1 < argc) {
            g_decode_iters = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]); return 0;
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            usage(argv[0]); return 1;
        }
    }

    g_iters = g_smoke ? ITERS_SMOKE : ITERS_FULL;

    if (from_path) {
        if (!dtype_s || !shape_s) { usage(argv[0]); return 1; }
        printf("tdc throughput bench --from %s (best of %d)\n", from_path, g_iters);
        print_header();
        int rc = run_from_file(from_path, dtype_s, shape_s, layout_s);
        if (rc) { fprintf(stderr, "\nbench_throughput: FAIL\n"); return 1; }
        return 0;
    }

    if (g_smoke) {
        printf("tdc throughput bench --smoke (round-trip gate, tiny inputs)\n");
    } else {
        printf("tdc throughput bench (best of %d, sizes are uncompressed)\n", g_iters);
    }
    print_header();

    int rc = 0;
    rc |= case_raw_none_u8();
    rc |= case_raw_huffman_u8();
    rc |= case_raw_lz_ramp();
    rc |= case_raw_shuffle_lz_ramp();
    rc |= case_delta_lz_ramp();
    rc |= case_delta_zigzag_shuffle_walk("DELTA1D+ZIGZAG+BSHUF+LZ",     TDC_ENTROPY_LZ,     TDC_ENTROPY_NONE);
    rc |= case_delta_zigzag_shuffle_walk("DELTA1D+ZZ+BSHUF+HUFFMAN",    TDC_ENTROPY_HUFFMAN,  TDC_ENTROPY_NONE);
    rc |= case_delta_zigzag_shuffle_walk("DELTA1D+ZZ+BSHUF+HUF4",      TDC_ENTROPY_HUFFMAN4, TDC_ENTROPY_NONE);
    rc |= case_delta_zigzag_shuffle_walk("DELTA1D+ZZ+BSHUF+FSE",        TDC_ENTROPY_FSE,      TDC_ENTROPY_NONE);
    rc |= case_delta_zigzag_shuffle_walk("DELTA1D+ZZ+BSHUF+LZ+HUF",   TDC_ENTROPY_LZ,      TDC_ENTROPY_HUFFMAN);
    rc |= case_delta1d_f64();
    rc |= case_pred2d_noisy_u16();
    rc |= case_plane2d_shuffle_lz();
    rc |= case_pred3d_f32();
    rc |= case_pred3d_i16();

    if (rc) {
        fprintf(stderr, "\nbench_throughput: FAIL\n");
        return 1;
    }
    return 0;
}
