/*
 * tests/test_sparse_zero_e2e.c
 *
 * End-to-end test for SPARSE_ZERO_1D through the full block-record path
 * (tdc_encode_block / tdc_decode_block). The standalone roundtrip test
 * calls the model's encode/decode directly; that path never exercises the
 * decode driver's forward byte-size walk, which assumed every model's
 * residual is exactly n_elems elements. SPARSE_ZERO's residual is
 * n_nonzero u32 positions (n_nonzero < n_elems), so the walk mis-sized the
 * residual and the driver's walk_bytes == uncompressed_size cross-check
 * falsely reported TDC_E_CORRUPT. This test locks the full pipeline.
 *
 * Swept across every entropy coder (with and without a leading byte-shuffle)
 * because the residual length feeds the transform-chain byte walk too, not
 * just the entropy stage.
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

/* One full encode -> decode -> compare round trip for a given spec. */
static int roundtrip(const char *label, const double *src, int64_t N,
                     const tdc_codec_spec *spec) {
    tdc_block in = {0};
    in.data = (void *)src;
    in.dtype = TDC_DT_F64;
    in.layout = TDC_LAYOUT_VECTOR_1D;
    in.shape.rank = 1;
    in.shape.dim[0] = N;
    in.shape.stride[0] = 1;

    tdc_buffer enc = make_buffer();
    tdc_status st = tdc_encode_block(&in, spec, &enc);
    if (st != TDC_OK) {
        printf("  %-28s encode -> %d FAIL\n", label, (int)st);
        if (enc.data) test_realloc(NULL, enc.data, 0);
        return 1;
    }

    double *dst = (double *)malloc((size_t)N * sizeof(double));
    tdc_block out = {0};
    out.data = dst;
    out.dtype = TDC_DT_F64;
    out.layout = TDC_LAYOUT_VECTOR_1D;
    out.shape.rank = 1;
    out.shape.dim[0] = N;
    out.shape.stride[0] = 1;

    st = tdc_decode_block(enc.data, enc.size, &out);
    int rc = 0;
    if (st != TDC_OK) {
        printf("  %-28s decode -> %d FAIL\n", label, (int)st);
        rc = 1;
    } else if (memcmp(dst, src, (size_t)N * sizeof(double)) != 0) {
        printf("  %-28s MISMATCH\n", label);
        rc = 1;
    } else {
        printf("  %-28s OK size=%zu\n", label, enc.size);
    }
    free(dst);
    test_realloc(NULL, enc.data, 0);
    return rc;
}

int main(void) {
    const int64_t N = 4096;
    double *src = (double *)malloc((size_t)N * sizeof(double));
    /* ~10% non-zero, +0.0 elsewhere (all-zero bytes = the model's zero). */
    unsigned seed = 12345u;
    for (int64_t i = 0; i < N; ++i) {
        seed = seed * 1103515245u + 12345u;
        int nz = ((seed >> 16) % 10u) == 0u;
        src[i] = nz ? (1.0 + (double)((seed >> 8) % 997u)) : 0.0;
    }

    const tdc_entropy_id ents[] = {
        TDC_ENTROPY_NONE, TDC_ENTROPY_LZ, TDC_ENTROPY_LZ_OPT,
        TDC_ENTROPY_LZ_SPLIT, TDC_ENTROPY_FSE, TDC_ENTROPY_HUFFMAN4,
    };
    const int n_ents = (int)(sizeof(ents) / sizeof(ents[0]));

    int rc = 0;
    char label[64];
    for (int e = 0; e < n_ents; ++e) {
        /* No transform. */
        tdc_codec_spec s = {0};
        s.model = TDC_MODEL_SPARSE_ZERO_1D;
        s.entropy[0] = ents[e];
        snprintf(label, sizeof label, "spz ent=0x%04X", (unsigned)ents[e]);
        rc |= roundtrip(label, src, N, &s);

        /* With a leading byte-shuffle over the u32 positions. */
        tdc_codec_spec sb = {0};
        sb.model = TDC_MODEL_SPARSE_ZERO_1D;
        sb.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
        sb.entropy[0] = ents[e];
        snprintf(label, sizeof label, "spz bshuf ent=0x%04X", (unsigned)ents[e]);
        rc |= roundtrip(label, src, N, &sb);
    }

    /* Edge cases: all-zero (n_nonzero == 0) and fully-dense (n_nonzero == N). */
    {
        double *z = (double *)calloc((size_t)N, sizeof(double));
        tdc_codec_spec s = {0};
        s.model = TDC_MODEL_SPARSE_ZERO_1D;
        s.entropy[0] = TDC_ENTROPY_LZ;
        rc |= roundtrip("spz all-zero", z, N, &s);
        for (int64_t i = 0; i < N; ++i) z[i] = (double)(i + 1);
        rc |= roundtrip("spz fully-dense", z, N, &s);
        free(z);
    }

    /* Tiny-n edges. A single all-zero element yields n_nonzero == 0, an
     * empty residual, and the ZERO_RESIDUAL short-circuit; its logical
     * residual byte count is 0 (not n_elems * 4), which is where the
     * encode/decode length accounting used to disagree. */
    {
        double one_zero = 0.0, one_nz = 7.0;
        double two_mixed[2] = {0.0, 5.0};
        tdc_codec_spec s = {0};
        s.model = TDC_MODEL_SPARSE_ZERO_1D;
        s.entropy[0] = TDC_ENTROPY_LZ;
        rc |= roundtrip("spz n=1 zero",  &one_zero, 1, &s);
        rc |= roundtrip("spz n=1 nonzero", &one_nz, 1, &s);
        rc |= roundtrip("spz n=2 mixed", two_mixed, 2, &s);
    }

    free(src);
    printf(rc ? "FAILED\n" : "ALL OK\n");
    return rc;
}
