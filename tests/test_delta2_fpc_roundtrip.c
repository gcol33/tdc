/*
 * tests/test_delta2_fpc_roundtrip.c
 *
 * Round-trip correctness for DELTA2_1D and FPC_1D on f64 VECTOR_1D blocks.
 * Tests: smooth ramp, periodic signal, single element, two elements, empty.
 */

#include "tdc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *test_realloc(void *user, void *ptr, size_t n) {
    (void)user;
    if (n == 0) { free(ptr); return NULL; }
    return realloc(ptr, n);
}

static int roundtrip_f64(const char *label, tdc_model_id model,
                         const double *data, int64_t n) {
    tdc_block src = {0};
    src.data   = (void *)data;
    src.dtype  = TDC_DT_F64;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank = 1;
    src.shape.dim[0] = n;
    tdc_shape_set_contiguous(&src.shape);

    tdc_codec_spec spec = {0};
    spec.model = model;
    spec.entropy[0] = TDC_ENTROPY_LZ;

    tdc_buffer enc = {0};
    enc.realloc_fn = test_realloc;

    tdc_status st = tdc_encode_block(&src, &spec, &enc);
    if (st != TDC_OK) {
        fprintf(stderr, "FAIL [%s n=%lld]: encode -> %d\n",
                label, (long long)n, (int)st);
        if (enc.data) free(enc.data);
        return 1;
    }

    double *out = (double *)malloc(sizeof(double) * (size_t)(n ? n : 1));
    if (!out) { free(enc.data); return 1; }

    tdc_block dst = {0};
    dst.data   = out;
    dst.dtype  = TDC_DT_F64;
    dst.layout = TDC_LAYOUT_VECTOR_1D;
    dst.shape  = src.shape;

    st = tdc_decode_block(enc.data, enc.size, &dst);
    if (st != TDC_OK) {
        fprintf(stderr, "FAIL [%s n=%lld]: decode -> %d\n",
                label, (long long)n, (int)st);
        free(out); free(enc.data);
        return 1;
    }

    if (n > 0 && memcmp(data, out, sizeof(double) * (size_t)n) != 0) {
        fprintf(stderr, "FAIL [%s n=%lld]: mismatch\n",
                label, (long long)n);
        for (int64_t i = 0; i < n && i < 10; ++i) {
            if (memcmp(&data[i], &out[i], sizeof(double)) != 0)
                fprintf(stderr, "  [%lld] %.17g != %.17g\n",
                        (long long)i, data[i], out[i]);
        }
        free(out); free(enc.data);
        return 1;
    }

    double raw_mib = (double)(n * 8) / (1024.0 * 1024.0);
    double enc_mib = (double)enc.size / (1024.0 * 1024.0);
    double ratio = enc.size > 0 ? (double)(n * 8) / (double)enc.size : 0.0;
    printf("  OK  %-30s  n=%-6lld  raw=%.3f MiB  enc=%.3f MiB  ratio=%.2fx\n",
           label, (long long)n, raw_mib, enc_mib, ratio);

    free(out);
    free(enc.data);
    return 0;
}

int main(void) {
    int rc = 0;

    /* Smooth quadratic ramp: ideal for DELTA2. */
    {
        const int64_t N = 10000;
        double *data = (double *)malloc(sizeof(double) * (size_t)N);
        for (int64_t i = 0; i < N; ++i)
            data[i] = 100.0 + 0.01 * (double)i + 0.0001 * (double)i * (double)i;

        rc |= roundtrip_f64("DELTA2 smooth ramp", TDC_MODEL_DELTA2_1D, data, N);
        rc |= roundtrip_f64("FPC smooth ramp",    TDC_MODEL_FPC_1D,    data, N);
        free(data);
    }

    /* Periodic sinusoidal: climate-like. */
    {
        const int64_t N = 10958;
        double *data = (double *)malloc(sizeof(double) * (size_t)N);
        for (int64_t i = 0; i < N; ++i)
            data[i] = 15.0 + 12.0 * sin(2.0 * 3.14159265358979 * (double)i / 365.25);

        rc |= roundtrip_f64("DELTA2 periodic", TDC_MODEL_DELTA2_1D, data, N);
        rc |= roundtrip_f64("FPC periodic",    TDC_MODEL_FPC_1D,    data, N);
        free(data);
    }

    /* Noisy walk: USGS-like. */
    {
        const int64_t N = 10000;
        double *data = (double *)malloc(sizeof(double) * (size_t)N);
        uint32_t s = 0xDEADBEEFu;
        data[0] = 150000.0;
        for (int64_t i = 1; i < N; ++i) {
            s = s * 1103515245u + 12345u;
            double noise = ((double)((int)((s >> 16) & 0xFFFF) - 32768)) * 0.1;
            data[i] = data[i - 1] + noise;
        }

        rc |= roundtrip_f64("DELTA2 noisy walk", TDC_MODEL_DELTA2_1D, data, N);
        rc |= roundtrip_f64("FPC noisy walk",    TDC_MODEL_FPC_1D,    data, N);
        free(data);
    }

    /* Edge: single element. */
    {
        double v = 42.0;
        rc |= roundtrip_f64("DELTA2 n=1", TDC_MODEL_DELTA2_1D, &v, 1);
        rc |= roundtrip_f64("FPC n=1",    TDC_MODEL_FPC_1D,    &v, 1);
    }

    /* Edge: two elements. */
    {
        double v[2] = {1.0, 2.0};
        rc |= roundtrip_f64("DELTA2 n=2", TDC_MODEL_DELTA2_1D, v, 2);
        rc |= roundtrip_f64("FPC n=2",    TDC_MODEL_FPC_1D,    v, 2);
    }

    /* Edge: empty. */
    rc |= roundtrip_f64("DELTA2 n=0", TDC_MODEL_DELTA2_1D, NULL, 0);
    rc |= roundtrip_f64("FPC n=0",    TDC_MODEL_FPC_1D,    NULL, 0);

    if (rc) fprintf(stderr, "\nFAILED\n");
    else    printf("\nAll DELTA2/FPC round-trip tests passed.\n");
    return rc;
}
