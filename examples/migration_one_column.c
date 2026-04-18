/* docs/examples/migration_one_column.c
 *
 * Migration worked example: encode one numeric column the way a
 * vectra-style C consumer used to do it (custom encoded-column struct +
 * sidecar metadata), then do the same round trip through the tdc public
 * API (one self-describing block record).
 *
 * Prints the raw, encoded, and decoded byte counts and whether the
 * reconstruction matches the input.
 *
 * Build:
 *   cc -I include docs/examples/migration_one_column.c \
 *      build/libtdc.a -lm -o /tmp/migration_one_column
 */

#include "tdc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* stdlib-backed allocator using the (user, ptr, n) convention tdc
 * expects for every caller-provided realloc_fn. */
static void *mg_realloc(void *user, void *ptr, size_t n) {
    (void)user;
    if (n == 0) { free(ptr); return NULL; }
    return realloc(ptr, n);
}

int main(void) {
    /* A synthetic monotonically-increasing i64 column, the canonical
     * input for the old VTR_ENC_DELTA path. */
    enum { N = 1024 };
    static int64_t values[N];
    for (int i = 0; i < N; ++i) values[i] = 10000 + (int64_t)i * 7;

    /* ----- Encode via the tdc public API ------------------------------- */

    tdc_block src = {0};
    src.data = values;
    src.dtype = TDC_DT_I64;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank = 1;
    src.shape.dim[0] = (int64_t)N;
    tdc_shape_set_contiguous(&src.shape);

    /* Codec spec mirrors what vectra's "FAST" compression level picks:
     * DELTA_1D over the monotone column, then ZIGZAG + BYTE_SHUFFLE, then
     * LZ as the entropy coder. Every id and param struct lives in
     * include/tdc/codec.h. */
    tdc_codec_spec spec = {0};
    spec.model = TDC_MODEL_DELTA_1D;
    spec.xform[0] = TDC_XFORM_ZIGZAG;
    spec.xform[1] = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0] = TDC_ENTROPY_LZ;

    tdc_buffer enc = {0};
    enc.realloc_fn = mg_realloc;

    tdc_status st = tdc_encode_block(&src, &spec, &enc);
    if (st != TDC_OK) {
        fprintf(stderr, "encode: %s\n", tdc_strerror(st));
        return 1;
    }

    /* ----- Decode via the tdc public API ------------------------------- */

    tdc_block meta = {0};
    size_t need = 0;
    st = tdc_decode_peek(enc.data, enc.size, &meta, &need);
    if (st != TDC_OK) {
        fprintf(stderr, "peek: %s\n", tdc_strerror(st));
        mg_realloc(NULL, enc.data, 0);
        return 1;
    }

    int64_t *dst_data = (int64_t *)mg_realloc(NULL, NULL, need);
    tdc_block dst = meta;
    dst.data = dst_data;

    st = tdc_decode_block_into(enc.data, enc.size, &dst);
    if (st != TDC_OK) {
        fprintf(stderr, "decode: %s\n", tdc_strerror(st));
        mg_realloc(NULL, dst_data, 0);
        mg_realloc(NULL, enc.data, 0);
        return 1;
    }

    /* ----- Verify -------------------------------------------------------- */

    size_t raw_bytes = N * sizeof(int64_t);
    int ok = (meta.shape.dim[0] == (int64_t)N) &&
             memcmp(dst_data, values, raw_bytes) == 0;

    printf("raw     : %zu bytes (%d x i64)\n", raw_bytes, N);
    printf("encoded : %zu bytes (one tdc_block_record)\n", enc.size);
    printf("decoded : %zu bytes, match=%s\n", raw_bytes, ok ? "yes" : "NO");

    mg_realloc(NULL, dst_data, 0);
    mg_realloc(NULL, enc.data, 0);
    return ok ? 0 : 1;
}
