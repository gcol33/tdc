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
#define now_seconds tdc_now_secs

#define ITERS_FULL  5
#define ITERS_SMOKE 1

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
     * will keep capacity stable after the first iteration. */
    tdc_buffer enc = make_buffer();
    double best_enc = 1e18;
    for (int i = 0; i < g_iters; ++i) {
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

    /* Decode timing: best of ITERS into a reusable destination. */
    void *dst_data = malloc(raw_bytes ? raw_bytes : 1);
    if (!dst_data) { free_buffer(&enc); return 1; }

    tdc_block dst = {0};
    dst.data    = dst_data;
    dst.dtype   = src->dtype;
    dst.layout  = src->layout;
    dst.shape   = src->shape;
    dst.offsets = NULL;

    double best_dec = 1e18;
    for (int i = 0; i < g_iters; ++i) {
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

    /* Round-trip sanity: bytes must match. */
    if (memcmp(dst_data, src->data, raw_bytes) != 0) {
        fprintf(stderr, "FAIL [%s]: round-trip mismatch\n", label);
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
    /* RAW + BSHUF + LZ — exposes per-lane structure on multi-byte dtypes. */
    if (tdc_dtype_size(dtype) > 1) {
        tdc_codec_spec s = tdc_codec_spec_raw();
        s.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0]  = TDC_ENTROPY_LZ;
        rc |= run_case("RAW + BSHUF + LZ", block_desc, &b, &s);
    }
    /* DELTA1D family — only on 1D integer dtypes. */
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
        "       %s --from PATH --dtype NAME --shape DIMS [--layout L]\n"
        "  NAME  in {i8,u8,i16,u16,i32,u32,i64,u64,f32,f64}\n"
        "  DIMS  e.g. 16777216 or 2048x2048\n"
        "  L     in {vec1d,rast2d,vol3d} (defaults from rank)\n",
        argv0, argv0, argv0);
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
    rc |= case_raw_lz_ramp();
    rc |= case_raw_shuffle_lz_ramp();
    rc |= case_delta_lz_ramp();
    rc |= case_delta_zigzag_shuffle_walk("DELTA1D+ZIGZAG+BSHUF+LZ",     TDC_ENTROPY_LZ,     TDC_ENTROPY_NONE);
    rc |= case_delta_zigzag_shuffle_walk("DELTA1D+ZZ+BSHUF+HUFFMAN",    TDC_ENTROPY_HUFFMAN,  TDC_ENTROPY_NONE);
    rc |= case_delta_zigzag_shuffle_walk("DELTA1D+ZZ+BSHUF+FSE",        TDC_ENTROPY_FSE,      TDC_ENTROPY_NONE);
    rc |= case_delta_zigzag_shuffle_walk("DELTA1D+ZZ+BSHUF+LZ+HUF",   TDC_ENTROPY_LZ,      TDC_ENTROPY_HUFFMAN);
    rc |= case_pred2d_noisy_u16();
    rc |= case_plane2d_shuffle_lz();

    if (rc) {
        fprintf(stderr, "\nbench_throughput: FAIL\n");
        return 1;
    }
    return 0;
}
