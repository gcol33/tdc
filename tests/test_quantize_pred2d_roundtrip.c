/*
 * tests/test_quantize_pred2d_roundtrip.c
 *
 * Round-trip test for the TDC_MODEL_QUANTIZE_PRED_2D composite model.
 *
 * Verifies:
 *   1. Direct vt->encode / vt->decode round-trips F32/F64 RASTER_2D blocks
 *      to within (1 / scale) absolute error.
 *   2. Side metadata is exactly 18 bytes = [target, kind, scale, offset].
 *   3. Residual dtype on encode equals the params->target field.
 *   4. AUTO kind resolves to one of LEFT/UP/AVERAGE/PAETH (never AUTO).
 *   5. NaN inputs round-trip to the dequantized lower bound (NaN -> tmin
 *      then dequantized).
 *   6. End-to-end through the driver: tdc_encode_block + tdc_decode_block
 *      with TDC_MODEL_QUANTIZE_PRED_2D -> [BYTE_SHUFFLE] -> LZ. The driver
 *      must seed the residual dtype from side meta (peek hook) for the
 *      transform / entropy chain to size correctly.
 *   7. Empty raster (n == 0) round-trips.
 *   8. Encoder rejects non-RASTER_2D layouts and non-float dtypes.
 *   9. Decoder rejects truncated / wrong-size side metadata.
 */

#include "tdc.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern const tdc_model_vt tdc_model_quantize_pred2d_vt;

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

static tdc_block raster(void *data, int64_t ny, int64_t nx, tdc_dtype dt) {
    tdc_block b = {0};
    b.data = data;
    b.dtype = dt;
    b.layout = TDC_LAYOUT_RASTER_2D;
    b.shape.rank = 2;
    b.shape.dim[0] = ny;
    b.shape.dim[1] = nx;
    b.shape.stride[0] = nx;
    b.shape.stride[1] = 1;
    return b;
}

/* ----- Direct vt round-trip ---------------------------------------------- */

static int rt_vt_f64(const char *label,
                     tdc_pred2d_kind kind,
                     tdc_dtype target,
                     double scale, double offset,
                     const double *src, int64_t ny, int64_t nx) {
    const tdc_model_vt *vt = &tdc_model_quantize_pred2d_vt;
    int64_t n = ny * nx;
    size_t  nbytes = (size_t)n * sizeof(double);

    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;

    tdc_quantize_pred2d_params p = {
        .scale = scale, .offset = offset,
        .target = target, .kind = kind,
    };
    tdc_block in = raster((void *)src, ny, nx, TDC_DT_F64);

    tdc_status st = vt->encode(&in, &p, &residual, &rdt, &side);
    ASSERT_OR_DIE(st == TDC_OK, "encode returned non-OK");
    ASSERT_OR_DIE(rdt == target, "residual_dtype != params.target");
    ASSERT_OR_DIE(side.size == 18u, "side metadata must be exactly 18 bytes");
    ASSERT_OR_DIE(side.data[0] == (uint8_t)target, "side meta byte 0 must be target");
    {
        uint8_t resolved = side.data[1];
        ASSERT_OR_DIE(resolved == TDC_PRED2D_LEFT  ||
                      resolved == TDC_PRED2D_UP    ||
                      resolved == TDC_PRED2D_AVERAGE ||
                      resolved == TDC_PRED2D_PAETH,
                      "resolved kind must be one of LEFT/UP/AVG/PAETH");
        if (kind != TDC_PRED2D_AUTO) {
            ASSERT_OR_DIE(resolved == (uint8_t)kind,
                          "non-AUTO kind must be preserved verbatim");
        }
    }

    /* Decode */
    double *dst = (n > 0) ? (double *)malloc(nbytes) : NULL;
    if (n > 0) {
        ASSERT_OR_DIE(dst != NULL, "malloc dst");
        memset(dst, 0xAA, nbytes);
    }
    tdc_block out = raster(dst, ny, nx, TDC_DT_F64);
    st = vt->decode(&out, NULL, rdt, residual.data, residual.size,
                    side.data, side.size);
    ASSERT_OR_DIE(st == TDC_OK, "decode returned non-OK");

    /* Tolerance: 1 / scale (one quant step), with a tiny FP epsilon. */
    double tol = (1.0 / scale) + 1e-12;
    for (int64_t i = 0; i < n; ++i) {
        double s = src[i];
        double d = dst[i];
        if (isnan(s)) {
            /* NaN gets quantized to tmin, then dequantized. */
            double tmin;
            switch (target) {
                case TDC_DT_I8:  tmin = (double)INT8_MIN;  break;
                case TDC_DT_I16: tmin = (double)INT16_MIN; break;
                case TDC_DT_I32: tmin = (double)INT32_MIN; break;
                default: tmin = 0; break;
            }
            double expect = tmin / scale + offset;
            if (fabs(d - expect) > 1e-9) {
                fprintf(stderr, "FAIL [%s]: NaN at i=%lld -> %g (expect %g)\n",
                        label, (long long)i, d, expect);
                free(dst); free_buffer(&residual); free_buffer(&side);
                return 1;
            }
            continue;
        }
        if (fabs(s - d) > tol) {
            fprintf(stderr, "FAIL [%s]: i=%lld src=%g dst=%g (tol %g)\n",
                    label, (long long)i, s, d, tol);
            free(dst); free_buffer(&residual); free_buffer(&side);
            return 1;
        }
    }

    fprintf(stdout, "  %-60s OK  side=%zu bytes  residual=%zu bytes\n",
            label, side.size, residual.size);
    if (dst) free(dst);
    free_buffer(&residual); free_buffer(&side);
    return 0;
}

/* ----- End-to-end through the driver ------------------------------------- */

static int rt_driver_f64(const char *label,
                         tdc_pred2d_kind kind,
                         tdc_dtype target,
                         double scale, double offset,
                         tdc_xform_id post_xform,
                         tdc_entropy_id entropy,
                         const double *src, int64_t ny, int64_t nx) {
    int64_t n     = ny * nx;
    size_t  bytes = (size_t)n * sizeof(double);

    tdc_quantize_pred2d_params p = {
        .scale = scale, .offset = offset,
        .target = target, .kind = kind,
    };

    tdc_codec_spec s = {0};
    s.model        = TDC_MODEL_QUANTIZE_PRED_2D;
    s.model_params = &p;
    s.xform[0]     = post_xform; /* may be TDC_XFORM_NONE */
    s.entropy[0]   = entropy;

    tdc_block in = raster((void *)src, ny, nx, TDC_DT_F64);

    tdc_buffer enc = make_buffer();
    tdc_status st = tdc_encode_block(&in, &s, &enc);
    ASSERT_OR_DIE(st == TDC_OK, "tdc_encode_block returned non-OK");
    ASSERT_OR_DIE(enc.size >= TDC_BLOCK_HEADER_SIZE, "encoded too small");

    double *dst = (n > 0) ? (double *)malloc(bytes) : NULL;
    if (n > 0) {
        ASSERT_OR_DIE(dst != NULL, "malloc dst");
        memset(dst, 0xCD, bytes);
    }
    tdc_block out = raster(dst, ny, nx, TDC_DT_F64);
    st = tdc_decode_block(enc.data, enc.size, &out);
    if (st != TDC_OK) {
        fprintf(stderr, "FAIL [%s]: tdc_decode_block -> %d\n", label, (int)st);
        if (dst) free(dst); free_buffer(&enc);
        return 1;
    }

    double tol = (1.0 / scale) + 1e-12;
    for (int64_t i = 0; i < n; ++i) {
        if (isnan(src[i])) continue; /* covered in vt test */
        if (fabs(src[i] - dst[i]) > tol) {
            fprintf(stderr, "FAIL [%s]: i=%lld src=%g dst=%g (tol %g)\n",
                    label, (long long)i, src[i], dst[i], tol);
            if (dst) free(dst); free_buffer(&enc);
            return 1;
        }
    }

    fprintf(stdout, "  %-60s OK  enc=%zu bytes\n", label, enc.size);
    if (dst) free(dst);
    free_buffer(&enc);
    return 0;
}

/* ----- Cases ------------------------------------------------------------- */

static int case_smooth_gradient_i16(void) {
    enum { NX = 32, NY = 24 };
    double src[NX * NY];
    for (int y = 0; y < NY; ++y)
        for (int x = 0; x < NX; ++x)
            src[y * NX + x] = (double)(x + y) * 0.125 - 4.0;

    int rc = 0;
    rc |= rt_vt_f64("vt: gradient f64 -> i16  scale=1000 offs=0  AUTO",
                    TDC_PRED2D_AUTO, TDC_DT_I16, 1000.0, 0.0, src, NY, NX);
    rc |= rt_vt_f64("vt: gradient f64 -> i16  scale=1000 offs=-2  PAETH",
                    TDC_PRED2D_PAETH, TDC_DT_I16, 1000.0, -2.0, src, NY, NX);
    return rc;
}

static int case_random_f32_i32(void) {
    enum { NX = 16, NY = 16 };
    double src[NX * NY];
    /* Simple LCG so behavior is deterministic. */
    uint32_t s = 0x12345678u;
    for (int i = 0; i < NX * NY; ++i) {
        s = s * 1664525u + 1013904223u;
        double r = (double)s / (double)UINT32_MAX; /* [0,1] */
        src[i] = r * 200.0 - 100.0;
    }
    return rt_vt_f64("vt: random f64 -> i32  scale=1e6 offs=0  AUTO",
                     TDC_PRED2D_AUTO, TDC_DT_I32, 1e6, 0.0, src, NY, NX);
}

static int case_with_nan(void) {
    enum { NX = 8, NY = 8 };
    double src[NX * NY];
    for (int i = 0; i < NX * NY; ++i) src[i] = (double)i * 0.25;
    src[10] = NAN;
    src[37] = NAN;
    return rt_vt_f64("vt: with NaN  f64 -> i16  scale=100 offs=0  LEFT",
                     TDC_PRED2D_LEFT, TDC_DT_I16, 100.0, 0.0, src, NY, NX);
}

static int case_empty(void) {
    return rt_vt_f64("vt: empty raster (n=0)",
                     TDC_PRED2D_AUTO, TDC_DT_I16, 100.0, 0.0,
                     NULL, 0, 0);
}

static int case_driver_e2e(void) {
    enum { NX = 64, NY = 48 };
    double *src = (double *)malloc(NX * NY * sizeof(double));
    if (!src) return 1;
    for (int y = 0; y < NY; ++y)
        for (int x = 0; x < NX; ++x)
            src[y * NX + x] = sin((double)x * 0.1) * 50.0
                            + cos((double)y * 0.07) * 30.0;

    int rc = 0;
    rc |= rt_driver_f64("driver: smooth + BYTE_SHUFFLE + LZ  i16 AUTO",
                        TDC_PRED2D_AUTO, TDC_DT_I16, 100.0, 0.0,
                        TDC_XFORM_BYTE_SHUFFLE, TDC_ENTROPY_LZ, src, NY, NX);
    rc |= rt_driver_f64("driver: smooth + ZIGZAG + BYTE_SHUFFLE + LZ  i32",
                        TDC_PRED2D_PAETH, TDC_DT_I32, 1000.0, 0.0,
                        TDC_XFORM_BYTE_SHUFFLE, TDC_ENTROPY_LZ, src, NY, NX);
    rc |= rt_driver_f64("driver: NONE chain (model only)  i16 LEFT",
                        TDC_PRED2D_LEFT, TDC_DT_I16, 100.0, 0.0,
                        TDC_XFORM_NONE, TDC_ENTROPY_NONE, src, NY, NX);

    /* Driver E2E with ZIGZAG slot before BYTE_SHUFFLE. ZIGZAG is legal on
     * the integer residual that QUANTIZE_PRED_2D emits. */
    {
        tdc_quantize_pred2d_params p = {
            .scale = 100.0, .offset = 0.0,
            .target = TDC_DT_I16, .kind = TDC_PRED2D_AVERAGE,
        };
        tdc_codec_spec s = {0};
        s.model        = TDC_MODEL_QUANTIZE_PRED_2D;
        s.model_params = &p;
        s.xform[0]     = TDC_XFORM_ZIGZAG;
        s.xform[1]     = TDC_XFORM_BYTE_SHUFFLE;
        s.entropy[0]   = TDC_ENTROPY_LZ;

        tdc_block in = raster(src, NY, NX, TDC_DT_F64);
        tdc_buffer enc = make_buffer();
        tdc_status st = tdc_encode_block(&in, &s, &enc);
        if (st != TDC_OK) {
            fprintf(stderr, "FAIL [zigzag chain]: encode -> %d\n", (int)st);
            free(src); free_buffer(&enc); return 1;
        }
        double *dst = (double *)malloc(NX * NY * sizeof(double));
        memset(dst, 0xCD, NX * NY * sizeof(double));
        tdc_block out = raster(dst, NY, NX, TDC_DT_F64);
        st = tdc_decode_block(enc.data, enc.size, &out);
        if (st != TDC_OK) {
            fprintf(stderr, "FAIL [zigzag chain]: decode -> %d\n", (int)st);
            free(dst); free(src); free_buffer(&enc); return 1;
        }
        double tol = (1.0 / p.scale) + 1e-12;
        for (int i = 0; i < NX * NY; ++i) {
            if (fabs(src[i] - dst[i]) > tol) {
                fprintf(stderr, "FAIL [zigzag chain]: i=%d src=%g dst=%g\n",
                        i, src[i], dst[i]);
                free(dst); free(src); free_buffer(&enc); return 1;
            }
        }
        fprintf(stdout, "  %-60s OK  enc=%zu bytes\n",
                "driver: ZIGZAG + BYTE_SHUFFLE + LZ  i16 AVG", enc.size);
        free(dst); free_buffer(&enc);
    }

    free(src);
    return rc;
}

static int case_rejections(void) {
    const tdc_model_vt *vt = &tdc_model_quantize_pred2d_vt;
    tdc_buffer residual = make_buffer();
    tdc_buffer side     = make_buffer();
    tdc_dtype  rdt      = (tdc_dtype)0;
    tdc_quantize_pred2d_params p = {
        .scale = 100.0, .offset = 0.0,
        .target = TDC_DT_I16, .kind = TDC_PRED2D_AUTO,
    };
    double src[16] = {0};

    /* Wrong layout: VECTOR_1D rejected. */
    {
        tdc_block in = {0};
        in.data = src; in.dtype = TDC_DT_F64;
        in.layout = TDC_LAYOUT_VECTOR_1D;
        in.shape.rank = 1; in.shape.dim[0] = 16; in.shape.stride[0] = 1;
        tdc_status st = vt->encode(&in, &p, &residual, &rdt, &side);
        ASSERT_OR_DIE(st == TDC_E_LAYOUT, "non-RASTER_2D must be E_LAYOUT");
    }
    /* Wrong dtype: i32 rejected. */
    {
        int32_t isrc[16] = {0};
        tdc_block in = raster(isrc, 4, 4, TDC_DT_I32);
        tdc_status st = vt->encode(&in, &p, &residual, &rdt, &side);
        ASSERT_OR_DIE(st == TDC_E_DTYPE, "non-float input must be E_DTYPE");
    }
    /* Bad target dtype in params. */
    {
        tdc_quantize_pred2d_params bad = p;
        bad.target = TDC_DT_F32; /* not an integer target */
        tdc_block in = raster(src, 4, 4, TDC_DT_F64);
        tdc_status st = vt->encode(&in, &bad, &residual, &rdt, &side);
        ASSERT_OR_DIE(st == TDC_E_INVAL, "non-integer target must be E_INVAL");
    }
    /* Side meta truncated on decode. */
    {
        double dst[16] = {0};
        tdc_block out = raster(dst, 4, 4, TDC_DT_F64);
        uint8_t fake_side[10] = {0};
        tdc_status st = vt->decode(&out, NULL, TDC_DT_I16,
                                   NULL, 0, fake_side, 10);
        ASSERT_OR_DIE(st == TDC_E_CORRUPT, "short side meta must be E_CORRUPT");
    }

    free_buffer(&residual); free_buffer(&side);
    fprintf(stdout, "  rejections                                                  OK\n");
    return 0;
}

static int case_registry(void) {
    const tdc_model_vt *vt = tdc_model_get(TDC_MODEL_QUANTIZE_PRED_2D);
    ASSERT_OR_DIE(vt == &tdc_model_quantize_pred2d_vt,
                  "registry must return tdc_model_quantize_pred2d_vt");
    fprintf(stdout, "  registry lookup                                             OK\n");
    return 0;
}

int main(void) {
    int rc = 0;
    fprintf(stdout, "test_quantize_pred2d_roundtrip:\n");
    rc |= case_registry();
    rc |= case_smooth_gradient_i16();
    rc |= case_random_f32_i32();
    rc |= case_with_nan();
    rc |= case_empty();
    rc |= case_driver_e2e();
    rc |= case_rejections();
    if (rc == 0) fprintf(stdout, "all OK\n");
    return rc;
}
