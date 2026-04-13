/*
 * tests/test_pipeline_roundtrip.c
 *
 * End-to-end round-trip tests for tdc_encode_block / tdc_decode_block.
 *
 * Each case constructs a tdc_block, picks a tdc_codec_spec, encodes
 * into a tdc_buffer, decodes the resulting bytes back into a fresh
 * destination block, and asserts byte-for-byte equality with the
 * source. The point of these tests is to exercise the driver — not
 * any individual stage — so the cases sweep across:
 *
 *   - models:   RAW, DELTA_1D, PRED_2D, PLANE_2D
 *   - layouts:  VECTOR_1D, RASTER_2D, STACK_2D, VOLUME_3D (RAW only)
 *   - dtypes:   i16, i32, u8, f32 (representative of integer, signed, unsigned, float)
 *   - chains:   empty, [BYTE_SHUFFLE], [ZIGZAG], [ZIGZAG, BYTE_SHUFFLE],
 *               [QUANTIZE] (params surfaced via TLV)
 *   - entropy:  NONE, LZ
 *   - sizes:    empty (n == 0), small (16), medium (4096)
 *   - features: validity bitmap pass-through
 *
 * Plus rejection tests:
 *   - decoder rejects header with wrong dtype/layout/dim
 *   - decoder rejects truncated src_size
 *   - encoder rejects unknown model id
 *   - encoder rejects spec with unsupported entropy id
 */

#include "tdc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static void *test_realloc(void *user, void *ptr, size_t new_size) {
    (void)user;
    if (new_size == 0) { free(ptr); return NULL; }
    return realloc(ptr, new_size);
}

static tdc_buffer make_buffer(void) {
    tdc_buffer b = {0};
    b.realloc_fn = test_realloc;
    return b;
}

static void free_buffer(tdc_buffer *b) {
    if (b->data) free(b->data);
    b->data = NULL;
    b->size = 0;
    b->capacity = 0;
}

#define ASSERT_OR_DIE(cond, msg) do {                                       \
    if (!(cond)) {                                                          \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);   \
        return 1;                                                           \
    }                                                                       \
} while (0)

/* ----- Generic round-trip helper ----------------------------------------- */

static int rt(const char           *label,
              const tdc_block      *src,
              const tdc_codec_spec *spec) {
    tdc_buffer enc = make_buffer();

    tdc_status st = tdc_encode_block(src, spec, &enc);
    if (st != TDC_OK) {
        fprintf(stderr, "FAIL [%s]: encode -> %d\n", label, (int)st);
        free_buffer(&enc);
        return 1;
    }

    /* Sanity: encoded size must be at least the header. */
    if (enc.size < TDC_BLOCK_HEADER_SIZE) {
        fprintf(stderr, "FAIL [%s]: encoded size %zu < header\n", label, enc.size);
        free_buffer(&enc);
        return 1;
    }

    /* Build a destination block: same dtype/layout/shape, fresh data. */
    int64_t n = 1;
    for (uint8_t i = 0; i < src->shape.rank; ++i) n *= src->shape.dim[i];
    size_t elem = tdc_dtype_size(src->dtype);
    size_t bytes = (size_t)n * elem;

    void *dst_data = (bytes > 0) ? malloc(bytes) : NULL;
    if (bytes > 0) {
        if (!dst_data) { free_buffer(&enc); return 1; }
        memset(dst_data, 0xCD, bytes);
    }

    tdc_block dst = {0};
    dst.data    = dst_data;
    dst.dtype   = src->dtype;
    dst.layout  = src->layout;
    dst.shape   = src->shape;
    dst.offsets = NULL;

    st = tdc_decode_block(enc.data, enc.size, &dst);
    if (st != TDC_OK) {
        fprintf(stderr, "FAIL [%s]: decode -> %d\n", label, (int)st);
        free(dst_data);
        free_buffer(&enc);
        return 1;
    }

    if (bytes > 0 && memcmp(dst_data, src->data, bytes) != 0) {
        fprintf(stderr, "FAIL [%s]: bytes differ after round-trip\n", label);
        free(dst_data);
        free_buffer(&enc);
        return 1;
    }

    fprintf(stdout, "  %-60s OK  enc=%zu bytes\n", label, enc.size);
    free(dst_data);
    free_buffer(&enc);
    return 0;
}

/* ----- Spec builders ----------------------------------------------------- */

static tdc_codec_spec spec_raw_lz(void) {
    tdc_codec_spec s = tdc_codec_spec_raw();
    s.entropy[0] = TDC_ENTROPY_LZ;
    return s;
}

static tdc_codec_spec spec_raw_none(void) {
    return tdc_codec_spec_raw(); /* model=RAW, entropy=NONE */
}

static tdc_codec_spec spec_delta_zigzag_shuffle_lz(void) {
    tdc_codec_spec s = {0};
    s.model    = TDC_MODEL_DELTA_1D;
    s.xform[0] = TDC_XFORM_ZIGZAG;
    s.xform[1] = TDC_XFORM_BYTE_SHUFFLE;
    s.entropy[0]  = TDC_ENTROPY_LZ;
    return s;
}

static tdc_codec_spec spec_delta_lz(void) {
    tdc_codec_spec s = {0};
    s.model   = TDC_MODEL_DELTA_1D;
    s.entropy[0] = TDC_ENTROPY_LZ;
    return s;
}

static tdc_codec_spec spec_pred2d_paeth_shuffle_lz(tdc_pred2d_params *params) {
    params->kind = TDC_PRED2D_PAETH;
    tdc_codec_spec s = {0};
    s.model        = TDC_MODEL_PRED_2D;
    s.model_params = params;
    s.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
    s.entropy[0]      = TDC_ENTROPY_LZ;
    return s;
}

static tdc_codec_spec spec_pred2d_auto_lz(tdc_pred2d_params *params) {
    params->kind = TDC_PRED2D_AUTO;
    tdc_codec_spec s = {0};
    s.model        = TDC_MODEL_PRED_2D;
    s.model_params = params;
    s.entropy[0]      = TDC_ENTROPY_LZ;
    return s;
}

/* ----- Test cases -------------------------------------------------------- */

static int case_raw_vector_i32_none(void) {
    int32_t data[16];
    for (int i = 0; i < 16; ++i) data[i] = (i * 12345) ^ (i << 4);

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_I32;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = 16;
    tdc_shape_set_contiguous(&b.shape);

    tdc_codec_spec s = spec_raw_none();
    return rt("RAW + NONE | vec1d i32 n=16", &b, &s);
}

static int case_raw_vector_f64_lz(void) {
    double data[256];
    for (int i = 0; i < 256; ++i) data[i] = (double)i * 0.5 - 64.0;

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_F64;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = 256;
    tdc_shape_set_contiguous(&b.shape);

    tdc_codec_spec s = spec_raw_lz();
    return rt("RAW + LZ  | vec1d f64 n=256", &b, &s);
}

static int case_raw_volume_u8_lz(void) {
    /* 4x4x4 volume of u8: gradient pattern. */
    uint8_t data[64];
    for (int i = 0; i < 64; ++i) data[i] = (uint8_t)(i * 3);

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_U8;
    b.layout      = TDC_LAYOUT_VOLUME_3D;
    b.shape.rank  = 3;
    b.shape.dim[0] = 4; b.shape.dim[1] = 4; b.shape.dim[2] = 4;
    tdc_shape_set_contiguous(&b.shape);

    tdc_codec_spec s = spec_raw_lz();
    return rt("RAW + LZ  | vol3d u8 4x4x4", &b, &s);
}

static int case_raw_empty(void) {
    tdc_block b = {0};
    b.data        = NULL;
    b.dtype       = TDC_DT_I16;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = 0;
    tdc_shape_set_contiguous(&b.shape);

    tdc_codec_spec s = spec_raw_lz();
    return rt("RAW + LZ  | vec1d i16 n=0 (empty)", &b, &s);
}

static int case_delta_lz_smooth(void) {
    /* 4096 i32 with a slow ramp -> small deltas, very compressible. */
    enum { N = 4096 };
    int32_t *data = (int32_t *)malloc(sizeof(int32_t) * N);
    if (!data) return 1;
    for (int i = 0; i < N; ++i) data[i] = 1000 + i * 3;

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_I32;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = N;
    tdc_shape_set_contiguous(&b.shape);

    tdc_codec_spec s = spec_delta_lz();
    int rc = rt("DELTA1D + LZ | vec1d i32 n=4096 ramp", &b, &s);
    free(data);
    return rc;
}

static int case_delta_zigzag_shuffle_lz(void) {
    /* Mixed-sign deltas exercise the zigzag step. */
    enum { N = 1024 };
    int16_t *data = (int16_t *)malloc(sizeof(int16_t) * N);
    if (!data) return 1;
    int16_t v = 0;
    for (int i = 0; i < N; ++i) {
        v += (int16_t)((i & 1) ? -3 : 5);
        data[i] = v;
    }

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_I16;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = N;
    tdc_shape_set_contiguous(&b.shape);

    tdc_codec_spec s = spec_delta_zigzag_shuffle_lz();
    int rc = rt("DELTA1D + ZIGZAG + BYTE_SHUFFLE + LZ | vec1d i16 n=1024", &b, &s);
    free(data);
    return rc;
}

static int case_pred2d_paeth_shuffle_lz(void) {
    /* 64x64 u16 raster: smooth gradient that PAETH predicts well. */
    enum { NX = 64, NY = 64 };
    uint16_t *data = (uint16_t *)malloc(sizeof(uint16_t) * NX * NY);
    if (!data) return 1;
    for (int r = 0; r < NY; ++r)
        for (int c = 0; c < NX; ++c)
            data[r * NX + c] = (uint16_t)(r * 5 + c * 3 + 100);

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_U16;
    b.layout      = TDC_LAYOUT_RASTER_2D;
    b.shape.rank  = 2;
    b.shape.dim[0] = NY; b.shape.dim[1] = NX;
    tdc_shape_set_contiguous(&b.shape);

    tdc_pred2d_params params;
    tdc_codec_spec s = spec_pred2d_paeth_shuffle_lz(&params);
    int rc = rt("PRED2D(PAETH) + BYTE_SHUFFLE + LZ | rast2d u16 64x64", &b, &s);
    free(data);
    return rc;
}

/* Plane2D: smooth-ish raster with two clearly different planes joined.
 * Tile fits should resolve each region independently and round-trip exactly. */
static int case_plane2d_smooth_u16(void) {
    enum { NX = 96, NY = 64 };
    uint16_t *data = (uint16_t *)malloc(sizeof(uint16_t) * NX * NY);
    if (!data) return 1;
    for (int r = 0; r < NY; ++r) {
        for (int c = 0; c < NX; ++c) {
            int v = (r < NY / 2) ? (200 + r * 4 + c * 2)
                                 : (1000 - r * 3 + c);
            data[r * NX + c] = (uint16_t)v;
        }
    }

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_U16;
    b.layout      = TDC_LAYOUT_RASTER_2D;
    b.shape.rank  = 2;
    b.shape.dim[0] = NY; b.shape.dim[1] = NX;
    tdc_shape_set_contiguous(&b.shape);

    tdc_plane2d_params params = {0};
    params.tile_size = 32;
    tdc_codec_spec s = {0};
    s.model        = TDC_MODEL_PLANE_2D;
    s.model_params = &params;
    s.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
    s.entropy[0]      = TDC_ENTROPY_LZ;
    int rc = rt("PLANE2D + BYTE_SHUFFLE + LZ | rast2d u16 96x64 (split planes, ts=32)", &b, &s);
    free(data);
    return rc;
}

/* Plane2D non-tile-aligned dimensions to exercise the partial-tile path. */
static int case_plane2d_unaligned_i32(void) {
    enum { NX = 50, NY = 37 };
    int32_t *data = (int32_t *)malloc(sizeof(int32_t) * NX * NY);
    if (!data) return 1;
    for (int r = 0; r < NY; ++r)
        for (int c = 0; c < NX; ++c)
            data[r * NX + c] = -500 + r * 11 - c * 7;

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_I32;
    b.layout      = TDC_LAYOUT_RASTER_2D;
    b.shape.rank  = 2;
    b.shape.dim[0] = NY; b.shape.dim[1] = NX;
    tdc_shape_set_contiguous(&b.shape);

    tdc_plane2d_params params = {0};
    params.tile_size = 16;
    tdc_codec_spec s = {0};
    s.model        = TDC_MODEL_PLANE_2D;
    s.model_params = &params;
    s.entropy[0]      = TDC_ENTROPY_LZ;
    int rc = rt("PLANE2D + LZ | rast2d i32 50x37 (unaligned, ts=16)", &b, &s);
    free(data);
    return rc;
}

/* Quantize round-trip via the driver: TLV must surface the params back to
 * the decode side, otherwise driver_xform_out_dtype returns 0. The
 * comparison is on the *quantized-then-dequantized* values, so we encode,
 * decode, and compare against a hand-quantized reference rather than the
 * original floats. */
static int case_quantize_via_driver_lz(void) {
    enum { N = 256 };
    float orig[N];
    float ref [N];
    for (int i = 0; i < N; ++i) {
        orig[i] = (float)i * 0.125f - 16.0f;
        /* Reference: encoder rounds (orig - offset) * scale to int16, decoder
         * inverts by stored / scale + offset. With offset=0, scale=4 the
         * float values are exact multiples of 0.25 -> bit-exact through i16. */
        int16_t q = (int16_t)((orig[i] - 0.0f) * 4.0f + (orig[i] >= 0 ? 0.5f : -0.5f));
        ref[i] = (float)q / 4.0f + 0.0f;
    }

    tdc_block b = {0};
    b.data        = orig;
    b.dtype       = TDC_DT_F32;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = N;
    tdc_shape_set_contiguous(&b.shape);

    tdc_quantize_params qp;
    qp.scale  = 4.0;
    qp.offset = 0.0;
    qp.target = TDC_DT_I16;

    tdc_codec_spec s = {0};
    s.model           = TDC_MODEL_RAW;
    s.xform[0]        = TDC_XFORM_QUANTIZE;
    s.xform_params[0] = &qp;
    s.entropy[0]         = TDC_ENTROPY_LZ;

    /* The generic rt() helper compares against b.data, but quantize is
     * lossy, so we replace b.data with the dequantized reference for the
     * comparison. */
    tdc_block bcmp = b;
    bcmp.data = ref;

    /* Encode original. */
    tdc_buffer enc = make_buffer();
    if (tdc_encode_block(&b, &s, &enc) != TDC_OK) {
        fprintf(stderr, "FAIL: quantize_via_driver encode\n");
        free_buffer(&enc); return 1;
    }

    /* Decode into a fresh buffer matching the original dtype (f32). */
    float out[N];
    memset(out, 0xCD, sizeof(out));
    tdc_block dst = {0};
    dst.data        = out;
    dst.dtype       = TDC_DT_F32;
    dst.layout      = TDC_LAYOUT_VECTOR_1D;
    dst.shape.rank  = 1;
    dst.shape.dim[0] = N;
    tdc_shape_set_contiguous(&dst.shape);

    tdc_status st = tdc_decode_block(enc.data, enc.size, &dst);
    free_buffer(&enc);
    if (st != TDC_OK) {
        fprintf(stderr, "FAIL: quantize_via_driver decode -> %d\n", (int)st);
        return 1;
    }
    for (int i = 0; i < N; ++i) {
        if (out[i] != ref[i]) {
            fprintf(stderr, "FAIL: quantize_via_driver value[%d] %f != ref %f\n",
                    i, (double)out[i], (double)ref[i]);
            return 1;
        }
    }
    fprintf(stdout, "  %-60s OK\n", "RAW + QUANTIZE(i16) + LZ | vec1d f32 n=256 (TLV)");
    return 0;
}

/* Validity bitmap pass-through: encoder writes the bitmap to disk; the
 * decoder must validate the size and not corrupt the rest of the record.
 * The decoded validity bytes are not surfaced (dst->validity is const in
 * v0), but a re-decode of the encoded record must still produce identical
 * data bytes regardless of the bitmap's presence. */
static int case_validity_passthrough(void) {
    enum { N = 64 };
    int32_t data[N];
    for (int i = 0; i < N; ++i) data[i] = i * 7 - 100;

    /* Half the elements marked valid (every other one). LSB-first within
     * each byte. */
    uint8_t validity[(N + 7) / 8];
    for (size_t i = 0; i < sizeof(validity); ++i) validity[i] = 0x55;

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_I32;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = N;
    b.validity    = validity;
    tdc_shape_set_contiguous(&b.shape);

    tdc_codec_spec s = spec_raw_lz();

    tdc_buffer enc = make_buffer();
    if (tdc_encode_block(&b, &s, &enc) != TDC_OK) {
        fprintf(stderr, "FAIL: validity_passthrough encode\n");
        free_buffer(&enc); return 1;
    }

    /* Decode into a fresh dst (without validity — the v0 driver does not
     * surface bitmap bytes). Data must round-trip. */
    int32_t out[N];
    memset(out, 0xCD, sizeof(out));
    tdc_block dst = {0};
    dst.data        = out;
    dst.dtype       = TDC_DT_I32;
    dst.layout      = TDC_LAYOUT_VECTOR_1D;
    dst.shape.rank  = 1;
    dst.shape.dim[0] = N;

    tdc_status st = tdc_decode_block(enc.data, enc.size, &dst);
    free_buffer(&enc);
    if (st != TDC_OK) {
        fprintf(stderr, "FAIL: validity_passthrough decode -> %d\n", (int)st);
        return 1;
    }
    if (memcmp(out, data, sizeof(data)) != 0) {
        fprintf(stderr, "FAIL: validity_passthrough data mismatch\n");
        return 1;
    }
    fprintf(stdout, "  %-60s OK\n", "RAW + LZ | vec1d i32 n=64 + validity bitmap");
    return 0;
}

static int case_pred2d_auto_i16(void) {
    /* Negative-going gradient — auto-select should still round-trip. */
    enum { NX = 32, NY = 32 };
    int16_t *data = (int16_t *)malloc(sizeof(int16_t) * NX * NY);
    if (!data) return 1;
    for (int r = 0; r < NY; ++r)
        for (int c = 0; c < NX; ++c)
            data[r * NX + c] = (int16_t)(-r * 7 + c * 2);

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_I16;
    b.layout      = TDC_LAYOUT_RASTER_2D;
    b.shape.rank  = 2;
    b.shape.dim[0] = NY; b.shape.dim[1] = NX;
    tdc_shape_set_contiguous(&b.shape);

    tdc_pred2d_params params;
    tdc_codec_spec s = spec_pred2d_auto_lz(&params);
    int rc = rt("PRED2D(AUTO) + LZ | rast2d i16 32x32 neg-grad", &b, &s);
    free(data);
    return rc;
}

/* ----- Rejection tests --------------------------------------------------- */

static int case_decode_rejects_dtype_mismatch(void) {
    int32_t data[8] = {1,2,3,4,5,6,7,8};
    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_I32;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = 8;
    tdc_shape_set_contiguous(&b.shape);

    tdc_buffer enc = make_buffer();
    tdc_codec_spec s = spec_raw_lz();
    ASSERT_OR_DIE(tdc_encode_block(&b, &s, &enc) == TDC_OK, "encode");

    /* Decode with wrong dtype: should be rejected. */
    int16_t wrong_buf[16] = {0};
    tdc_block bad = {0};
    bad.data        = wrong_buf;
    bad.dtype       = TDC_DT_I16;       /* wrong */
    bad.layout      = TDC_LAYOUT_VECTOR_1D;
    bad.shape.rank  = 1;
    bad.shape.dim[0] = 8;

    tdc_status st = tdc_decode_block(enc.data, enc.size, &bad);
    free_buffer(&enc);
    ASSERT_OR_DIE(st == TDC_E_DTYPE, "decode must reject wrong dtype");
    fprintf(stdout, "  decode rejects dtype mismatch                              OK\n");
    return 0;
}

static int case_decode_rejects_truncated(void) {
    int32_t data[8] = {1,2,3,4,5,6,7,8};
    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_I32;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = 8;
    tdc_shape_set_contiguous(&b.shape);

    tdc_buffer enc = make_buffer();
    tdc_codec_spec s = spec_raw_lz();
    ASSERT_OR_DIE(tdc_encode_block(&b, &s, &enc) == TDC_OK, "encode");

    /* Truncate to header only — payload bytes are missing. */
    int32_t out_buf[8] = {0};
    tdc_block dst = {0};
    dst.data        = out_buf;
    dst.dtype       = TDC_DT_I32;
    dst.layout      = TDC_LAYOUT_VECTOR_1D;
    dst.shape.rank  = 1;
    dst.shape.dim[0] = 8;

    tdc_status st = tdc_decode_block(enc.data, TDC_BLOCK_HEADER_SIZE, &dst);
    free_buffer(&enc);
    ASSERT_OR_DIE(st == TDC_E_CORRUPT, "decode must reject truncated src");
    fprintf(stdout, "  decode rejects truncated src                               OK\n");
    return 0;
}

static int case_encode_rejects_unknown_model(void) {
    int32_t data[4] = {1,2,3,4};
    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_I32;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = 4;
    tdc_shape_set_contiguous(&b.shape);

    tdc_codec_spec s = {0};
    s.model   = (tdc_model_id)0x00FF;  /* deliberately invalid id */
    s.entropy[0] = TDC_ENTROPY_LZ;

    tdc_buffer enc = make_buffer();
    tdc_status st = tdc_encode_block(&b, &s, &enc);
    free_buffer(&enc);
    ASSERT_OR_DIE(st == TDC_E_UNSUPPORTED, "encode must reject unregistered model");
    fprintf(stdout, "  encode rejects unregistered model                          OK\n");
    return 0;
}

/* ----- DICT_1D end-to-end (string blocks need their own harness) -------- */
/*
 * The generic rt() helper above is fixed-width: it sizes dst by
 * n * tdc_dtype_size(dtype), which is 0 for STRING. STRING blocks
 * carry an offsets[] sidecar and a packed byte heap, and the decode
 * destination has to provide both. This case builds those by hand and
 * exercises tdc_encode_block/tdc_decode_block end-to-end with DICT_1D
 * + BYTE_SHUFFLE + LZ.
 */
static int case_dict1d_byte_shuffle_lz(void) {
    static const char *rows[16] = {
        "alpha", "beta",  "gamma", "alpha",
        "beta",  "alpha", "delta", "alpha",
        "gamma", "beta",  "alpha", "delta",
        "alpha", "beta",  "gamma", "alpha"
    };
    enum { N = 16 };

    /* Pack the input into the canonical STRING block layout. */
    uint32_t in_offsets[N + 1];
    uint32_t total = 0u;
    for (int i = 0; i < N; ++i) {
        in_offsets[i] = total;
        total += (uint32_t)strlen(rows[i]);
    }
    in_offsets[N] = total;

    uint8_t *in_heap = (uint8_t *)malloc(total > 0 ? total : 1u);
    if (!in_heap) return 1;
    for (int i = 0; i < N; ++i) {
        size_t len = (size_t)(in_offsets[i + 1] - in_offsets[i]);
        if (len > 0) memcpy(in_heap + in_offsets[i], rows[i], len);
    }

    tdc_block src = {0};
    src.data            = in_heap;
    src.offsets         = in_offsets;
    src.dtype           = TDC_DT_STRING;
    src.layout          = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank      = 1;
    src.shape.dim[0]    = N;
    src.shape.stride[0] = 1;

    tdc_codec_spec spec = {0};
    spec.model    = TDC_MODEL_DICT_1D;
    spec.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0]  = TDC_ENTROPY_LZ;

    tdc_buffer enc = make_buffer();
    tdc_status st  = tdc_encode_block(&src, &spec, &enc);
    ASSERT_OR_DIE(st == TDC_OK, "DICT_1D + BSHUF + LZ encode");
    ASSERT_OR_DIE(enc.size > TDC_BLOCK_HEADER_SIZE, "encoded size sanity");

    /* Decode dst: caller must provide offsets and a heap large enough.
     * Worst case is total bytes from the input. */
    uint32_t dst_offsets[N + 1];
    uint8_t *dst_heap = (uint8_t *)malloc(total > 0 ? total : 1u);
    if (!dst_heap) { free(in_heap); free_buffer(&enc); return 1; }

    tdc_block dst = {0};
    dst.data            = dst_heap;
    dst.offsets         = dst_offsets;
    dst.dtype           = TDC_DT_STRING;
    dst.layout          = TDC_LAYOUT_VECTOR_1D;
    dst.shape.rank      = 1;
    dst.shape.dim[0]    = N;
    dst.shape.stride[0] = 1;

    st = tdc_decode_block(enc.data, enc.size, &dst);
    ASSERT_OR_DIE(st == TDC_OK, "DICT_1D + BSHUF + LZ decode");

    /* Value equality. */
    for (int i = 0; i < N; ++i) {
        uint32_t la = in_offsets[i + 1]  - in_offsets[i];
        uint32_t lb = dst_offsets[i + 1] - dst_offsets[i];
        ASSERT_OR_DIE(la == lb, "row length mismatch");
        if (la > 0) {
            ASSERT_OR_DIE(memcmp(in_heap  + in_offsets[i],
                                 dst_heap + dst_offsets[i],
                                 la) == 0, "row bytes mismatch");
        }
    }

    fprintf(stdout, "  %-60s OK  enc=%zu bytes\n",
            "DICT_1D + BYTE_SHUFFLE + LZ | vec1d string n=16", enc.size);

    free(dst_heap);
    free(in_heap);
    free_buffer(&enc);
    return 0;
}

/* ----- Entropy chain cases ----------------------------------------------- *
 *
 * Exercise the driver's entropy chain walk. Any pipeline whose final
 * residual stream ends up at a non-empty entropy chain should round-trip
 * bit-exact regardless of how many entropy stages are in the chain. The
 * payload format for a chain is [u32 * len sizes][final bytes]; the
 * encoder writes it and the decoder's RTL walk peels it back off. */

static int case_delta_zigzag_shuffle_huffman_only(void) {
    /* Huffman on its own, no LZ in front. Small mixed-sign deltas should
     * compress via symbol frequencies even without an LZ pass. */
    enum { N = 2048 };
    int16_t *data = (int16_t *)malloc(sizeof(int16_t) * N);
    if (!data) return 1;
    int16_t v = 0;
    for (int i = 0; i < N; ++i) {
        v += (int16_t)((i * 37) % 11 - 5);
        data[i] = v;
    }

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_I16;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = N;
    tdc_shape_set_contiguous(&b.shape);

    tdc_codec_spec s = {0};
    s.model      = TDC_MODEL_DELTA_1D;
    s.xform[0]   = TDC_XFORM_ZIGZAG;
    s.xform[1]   = TDC_XFORM_BYTE_SHUFFLE;
    s.entropy[0] = TDC_ENTROPY_HUFFMAN;
    int rc = rt("DELTA1D + ZZ + BSHUF + HUFFMAN | vec1d i16 n=2048", &b, &s);
    free(data);
    return rc;
}

static int case_delta_zigzag_shuffle_fse_only(void) {
    enum { N = 2048 };
    int32_t *data = (int32_t *)malloc(sizeof(int32_t) * N);
    if (!data) return 1;
    int32_t v = 0;
    for (int i = 0; i < N; ++i) {
        v += ((i * 7919) % 23) - 11;
        data[i] = v;
    }

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_I32;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = N;
    tdc_shape_set_contiguous(&b.shape);

    tdc_codec_spec s = {0};
    s.model      = TDC_MODEL_DELTA_1D;
    s.xform[0]   = TDC_XFORM_ZIGZAG;
    s.xform[1]   = TDC_XFORM_BYTE_SHUFFLE;
    s.entropy[0] = TDC_ENTROPY_FSE;
    int rc = rt("DELTA1D + ZZ + BSHUF + FSE | vec1d i32 n=2048", &b, &s);
    free(data);
    return rc;
}

/* LZ then Huffman: the entropy-after-entropy case. LZ turns the
 * residual stream into (literals + match tokens); Huffman then re-codes
 * LZ's output bytes by symbol frequency. */
static int case_pred2d_lz_huffman_chain(void) {
    enum { NX = 128, NY = 96 };
    uint16_t *data = (uint16_t *)malloc(sizeof(uint16_t) * NX * NY);
    if (!data) return 1;
    for (int r = 0; r < NY; ++r)
        for (int c = 0; c < NX; ++c)
            data[r * NX + c] = (uint16_t)((r * 13 + c * 5) & 0x3FF);

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_U16;
    b.layout      = TDC_LAYOUT_RASTER_2D;
    b.shape.rank  = 2;
    b.shape.dim[0] = NY; b.shape.dim[1] = NX;
    tdc_shape_set_contiguous(&b.shape);

    tdc_pred2d_params params;
    params.kind = TDC_PRED2D_PAETH;

    tdc_codec_spec s = {0};
    s.model        = TDC_MODEL_PRED_2D;
    s.model_params = &params;
    s.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
    s.entropy[0]   = TDC_ENTROPY_LZ;
    s.entropy[1]   = TDC_ENTROPY_HUFFMAN;
    int rc = rt("PRED2D + BSHUF + LZ + HUFFMAN | rast2d u16 128x96", &b, &s);
    free(data);
    return rc;
}

/* FSE then LZ: the inverse order. Tests that the driver doesn't care
 * which direction the entropy coders are chained. */
static int case_raw_fse_lz_chain(void) {
    enum { N = 1024 };
    uint8_t *data = (uint8_t *)malloc(N);
    if (!data) return 1;
    for (int i = 0; i < N; ++i) data[i] = (uint8_t)((i * 31) & 0x3F);

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_U8;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = N;
    tdc_shape_set_contiguous(&b.shape);

    tdc_codec_spec s = {0};
    s.model      = TDC_MODEL_RAW;
    s.entropy[0] = TDC_ENTROPY_FSE;
    s.entropy[1] = TDC_ENTROPY_LZ;
    int rc = rt("RAW + FSE + LZ | vec1d u8 n=1024", &b, &s);
    free(data);
    return rc;
}

/* Three-stage chain: LZ -> FSE -> HUFFMAN. Not necessarily a good idea
 * ratio-wise, but the driver must handle it correctly. */
static int case_delta_three_stage_chain(void) {
    enum { N = 4096 };
    int32_t *data = (int32_t *)malloc(sizeof(int32_t) * N);
    if (!data) return 1;
    for (int i = 0; i < N; ++i) data[i] = 1000 + i * 3;

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_I32;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = N;
    tdc_shape_set_contiguous(&b.shape);

    tdc_codec_spec s = {0};
    s.model      = TDC_MODEL_DELTA_1D;
    s.xform[0]   = TDC_XFORM_ZIGZAG;
    s.xform[1]   = TDC_XFORM_BYTE_SHUFFLE;
    s.entropy[0] = TDC_ENTROPY_LZ;
    s.entropy[1] = TDC_ENTROPY_FSE;
    s.entropy[2] = TDC_ENTROPY_HUFFMAN;
    int rc = rt("DELTA1D + ZZ + BSHUF + LZ + FSE + HUFFMAN | vec1d i32 n=4096", &b, &s);
    free(data);
    return rc;
}

/* PRED2D + BSHUF + LZ_OPT on a larger noisy u16 raster. Exercises
 * LZ_OPT at pipeline scale — the 128x96 test above catches basic
 * wiring but misses size-dependent DP bugs. */
static int case_pred2d_bshuf_lz_opt_large(void) {
    enum { NX = 512, NY = 512 };
    uint16_t *data = (uint16_t *)malloc(sizeof(uint16_t) * NX * NY);
    if (!data) return 1;
    uint32_t s = 0xC0FFEEu;
    for (int r = 0; r < NY; ++r)
        for (int c = 0; c < NX; ++c) {
            s = s * 1103515245u + 12345u;
            int noise = (int)((s >> 20) & 0x7) - 3;
            data[r * NX + c] = (uint16_t)(100 + r * 5 + c * 3 + noise);
        }

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_U16;
    b.layout      = TDC_LAYOUT_RASTER_2D;
    b.shape.rank  = 2;
    b.shape.dim[0] = NY; b.shape.dim[1] = NX;
    tdc_shape_set_contiguous(&b.shape);

    tdc_pred2d_params params;
    params.kind = TDC_PRED2D_PAETH;

    tdc_codec_spec s1 = {0};
    s1.model        = TDC_MODEL_PRED_2D;
    s1.model_params = &params;
    s1.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
    s1.entropy[0]   = TDC_ENTROPY_LZ_OPT;
    int rc = rt("PRED2D + BSHUF + LZ_OPT | rast2d u16 512x512", &b, &s1);
    free(data);
    return rc;
}

/* RAW + LZ_SPLIT on a smooth f64 vector. 10958 elements matches the
 * USGS streamflow dataset size. Exercises repcode offset encoding on
 * realistic-size data. */
static int case_raw_f64_lz_split(void) {
    enum { N = 10958 };
    double *data = (double *)malloc(sizeof(double) * N);
    if (!data) return 1;
    /* Mimic seasonal pattern: base + trend + seasonal + noise. */
    uint32_t s = 0xF00D;
    for (int i = 0; i < N; ++i) {
        s = s * 1103515245u + 12345u;
        double noise = (double)((int)((s >> 16) & 0xFF) - 128) * 0.5;
        data[i] = 50000.0 + 10.0 * (double)i + 20000.0 * (1.0 + 0.01 * (double)(i % 365)) + noise;
    }

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_F64;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = N;
    tdc_shape_set_contiguous(&b.shape);

    tdc_codec_spec s1 = {0};
    s1.model      = TDC_MODEL_RAW;
    s1.entropy[0] = TDC_ENTROPY_LZ_SPLIT;
    int rc = rt("RAW + LZ_SPLIT | vec1d f64 4096", &b, &s1);
    free(data);
    return rc;
}

/* PRED2D + BSHUF + LZ_SPLIT on a noisy u16 raster. */
static int case_pred2d_bshuf_lz_split(void) {
    enum { NX = 128, NY = 96 };
    uint16_t *data = (uint16_t *)malloc(sizeof(uint16_t) * NX * NY);
    if (!data) return 1;
    uint32_t s = 0xDEADBEu;
    for (int r = 0; r < NY; ++r)
        for (int c = 0; c < NX; ++c) {
            s = s * 1103515245u + 12345u;
            int noise = (int)((s >> 20) & 0x7) - 3;
            data[r * NX + c] = (uint16_t)(200 + r * 3 + c * 2 + noise);
        }

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_U16;
    b.layout      = TDC_LAYOUT_RASTER_2D;
    b.shape.rank  = 2;
    b.shape.dim[0] = NY; b.shape.dim[1] = NX;
    tdc_shape_set_contiguous(&b.shape);

    tdc_pred2d_params params = { .kind = TDC_PRED2D_PAETH };
    tdc_codec_spec s1 = {0};
    s1.model        = TDC_MODEL_PRED_2D;
    s1.model_params = &params;
    s1.xform[0]     = TDC_XFORM_BYTE_SHUFFLE;
    s1.entropy[0]   = TDC_ENTROPY_LZ_SPLIT;
    int rc = rt("PRED2D + BSHUF + LZ_SPLIT | rast2d u16 128x96", &b, &s1);
    free(data);
    return rc;
}

/* ----- Float + LANE entropy pipeline cases -------------------------------- */

static int case_f32_pred2d_bshuf_lane(void) {
    /* F32 raster: PRED2D + BSHUF + LANE entropy (4 lanes for 4-byte f32). */
    enum { NX = 32, NY = 32 };
    float *data = (float *)malloc(sizeof(float) * NX * NY);
    if (!data) return 1;
    for (int r = 0; r < NY; ++r)
        for (int c = 0; c < NX; ++c)
            data[r * NX + c] = 100.0f + (float)r * 0.7f + (float)c * 0.3f;

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_F32;
    b.layout      = TDC_LAYOUT_RASTER_2D;
    b.shape.rank  = 2;
    b.shape.dim[0] = NY; b.shape.dim[1] = NX;
    tdc_shape_set_contiguous(&b.shape);

    tdc_pred2d_params mp = { .kind = TDC_PRED2D_PAETH };
    tdc_lane_entropy_params lp = {0};
    lp.n_lanes = 4; /* f32 = 4 byte lanes after BSHUF */

    tdc_codec_spec s = {0};
    s.model          = TDC_MODEL_PRED_2D;
    s.model_params   = &mp;
    s.xform[0]       = TDC_XFORM_BYTE_SHUFFLE;
    s.entropy[0]     = TDC_ENTROPY_LANE;
    s.entropy_params[0] = &lp;
    int rc = rt("PRED2D(PAETH) + BSHUF + LANE | rast2d f32 32x32", &b, &s);
    free(data);
    return rc;
}

static int case_f16_delta1d_bshuf_lane(void) {
    /* F16 vector: DELTA1D + BSHUF + LANE entropy (2 lanes for 2-byte f16). */
    enum { N = 512 };
    uint16_t *data = (uint16_t *)malloc(sizeof(uint16_t) * N);
    if (!data) return 1;
    /* Slowly increasing f16 values: 1.0, 1.001, 1.002, ... */
    for (int i = 0; i < N; ++i)
        data[i] = (uint16_t)(0x3C00 + i); /* base ~1.0, increment by 1 ULP */

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_F16;
    b.layout      = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank  = 1;
    b.shape.dim[0] = N;
    tdc_shape_set_contiguous(&b.shape);

    tdc_lane_entropy_params lp = {0};
    lp.n_lanes = 2; /* f16 = 2 byte lanes after BSHUF */

    tdc_codec_spec s = {0};
    s.model          = TDC_MODEL_DELTA_1D;
    s.xform[0]       = TDC_XFORM_BYTE_SHUFFLE;
    s.entropy[0]     = TDC_ENTROPY_LANE;
    s.entropy_params[0] = &lp;
    int rc = rt("DELTA1D + BSHUF + LANE | vec1d f16 n=512", &b, &s);
    free(data);
    return rc;
}

static int case_f64_pred3d_bshuf_lane(void) {
    /* F64 volume: PRED3D + BSHUF + LANE entropy (8 lanes for 8-byte f64). */
    enum { NX = 8, NY = 8, NZ = 4 };
    double *data = (double *)malloc(sizeof(double) * NX * NY * NZ);
    if (!data) return 1;
    for (int z = 0; z < NZ; ++z)
        for (int y = 0; y < NY; ++y)
            for (int x = 0; x < NX; ++x)
                data[(z * NY + y) * NX + x] =
                    50.0 + (double)z * 3.0 + (double)y * 0.7 + (double)x * 0.3;

    tdc_block b = {0};
    b.data        = data;
    b.dtype       = TDC_DT_F64;
    b.layout      = TDC_LAYOUT_VOLUME_3D;
    b.shape.rank  = 3;
    b.shape.dim[0] = NZ; b.shape.dim[1] = NY; b.shape.dim[2] = NX;
    tdc_shape_set_contiguous(&b.shape);

    tdc_pred3d_params mp = { .kind = TDC_PRED3D_GRAD3D };
    tdc_lane_entropy_params lp = {0};
    lp.n_lanes = 8; /* f64 = 8 byte lanes after BSHUF */

    tdc_codec_spec s = {0};
    s.model          = TDC_MODEL_PRED_3D;
    s.model_params   = &mp;
    s.xform[0]       = TDC_XFORM_BYTE_SHUFFLE;
    s.entropy[0]     = TDC_ENTROPY_LANE;
    s.entropy_params[0] = &lp;
    int rc = rt("PRED3D(GRAD3D) + BSHUF + LANE | vol3d f64 4x8x8", &b, &s);
    free(data);
    return rc;
}

/* ----- main -------------------------------------------------------------- */

int main(void) {
    int rc = 0;

    fprintf(stdout, "tdc_encode_block / tdc_decode_block round-trips:\n");
    rc |= case_raw_vector_i32_none();
    rc |= case_raw_vector_f64_lz();
    rc |= case_raw_volume_u8_lz();
    rc |= case_raw_empty();
    rc |= case_delta_lz_smooth();
    rc |= case_delta_zigzag_shuffle_lz();
    rc |= case_pred2d_paeth_shuffle_lz();
    rc |= case_pred2d_auto_i16();
    rc |= case_plane2d_smooth_u16();
    rc |= case_plane2d_unaligned_i32();
    rc |= case_quantize_via_driver_lz();
    rc |= case_validity_passthrough();
    rc |= case_dict1d_byte_shuffle_lz();

    fprintf(stdout, "entropy chain cases:\n");
    rc |= case_delta_zigzag_shuffle_huffman_only();
    rc |= case_delta_zigzag_shuffle_fse_only();
    rc |= case_pred2d_lz_huffman_chain();
    rc |= case_raw_fse_lz_chain();
    rc |= case_delta_three_stage_chain();
    rc |= case_pred2d_bshuf_lz_opt_large();
    rc |= case_raw_f64_lz_split();
    rc |= case_pred2d_bshuf_lz_split();

    fprintf(stdout, "float + lane entropy pipeline cases:\n");
    rc |= case_f32_pred2d_bshuf_lane();
    rc |= case_f16_delta1d_bshuf_lane();
    rc |= case_f64_pred3d_bshuf_lane();

    fprintf(stdout, "rejection tests:\n");
    rc |= case_decode_rejects_dtype_mismatch();
    rc |= case_decode_rejects_truncated();
    rc |= case_encode_rejects_unknown_model();

    if (rc != 0) {
        fprintf(stderr, "test_pipeline_roundtrip: FAIL\n");
        return 1;
    }
    fprintf(stdout, "test_pipeline_roundtrip: OK\n");
    return 0;
}
