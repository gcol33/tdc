/*
 * tests/test_dnum_e2e.c
 *
 * End-to-end test for DICT_NUMERIC_1D through the full block-record path
 * (tdc_encode_block / tdc_decode_block). The standalone roundtrip test
 * calls the model's encode/decode directly; this one exercises the
 * container + transform + entropy chain.
 */

#include "tdc/codec.h"
#include "tdc/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int main(void) {
    const int64_t N = 1024;
    double *src = (double *)malloc((size_t)N * sizeof(double));
    for (int64_t i = 0; i < N; ++i) {
        src[i] = -20.0 + 0.1 * (double)(i % 50);
    }

    tdc_block in = {0};
    in.data = src;
    in.dtype = TDC_DT_F64;
    in.layout = TDC_LAYOUT_VECTOR_1D;
    in.shape.rank = 1;
    in.shape.dim[0] = N;
    in.shape.stride[0] = 1;

    tdc_codec_spec spec = {0};
    spec.model = TDC_MODEL_DICT_NUMERIC_1D;
    spec.entropy[0] = TDC_ENTROPY_LZ;

    tdc_buffer enc = make_buffer();
    tdc_status st = tdc_encode_block(&in, &spec, &enc);
    printf("encode -> %d, size=%zu\n", (int)st, enc.size);
    if (st != TDC_OK) return 1;

    double *dst = (double *)malloc((size_t)N * sizeof(double));
    tdc_block out = {0};
    out.data = dst;
    out.dtype = TDC_DT_F64;
    out.layout = TDC_LAYOUT_VECTOR_1D;
    out.shape.rank = 1;
    out.shape.dim[0] = N;
    out.shape.stride[0] = 1;

    st = tdc_decode_block(enc.data, enc.size, &out);
    printf("decode -> %d\n", (int)st);
    if (st != TDC_OK) { free(src); free(dst); free(enc.data); return 1; }

    if (memcmp(dst, src, (size_t)N * sizeof(double)) != 0) {
        printf("MISMATCH\n");
        free(src); free(dst); free(enc.data);
        return 1;
    }
    printf("OK\n");
    free(src); free(dst); free(enc.data);
    return 0;
}
