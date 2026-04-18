/* docs/examples/integration_consumer.c
 *
 * Minimal consumer template: one block encoded, one block decoded, a
 * single function wrapping the round trip with a consumer-side allocator
 * and an error-path. This is the smallest useful wiring — roughly the
 * shape of the function a downstream project copies first.
 *
 * Build:
 *   cc -I include docs/examples/integration_consumer.c \
 *      build/libtdc.a -lm -o /tmp/integration_consumer
 */

#include "tdc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* stdlib-backed allocator, (user, ptr, n) convention. */
static void *cn_realloc(void *user, void *ptr, size_t n) {
    (void)user;
    if (n == 0) { free(ptr); return NULL; }
    return realloc(ptr, n);
}

/*
 * Encode one contiguous i16 column with a DELTA_1D + ZIGZAG + BSHUF + LZ
 * pipeline. Returns a freshly-allocated byte buffer owned by the caller
 * (free via cn_realloc). On failure writes a diagnostic to stderr and
 * returns NULL with *out_size == 0.
 */
static uint8_t *consumer_encode_i16(const int16_t *values, size_t n,
                                    size_t *out_size) {
    *out_size = 0;

    tdc_block src = {0};
    src.data = (void *)values;
    src.dtype = TDC_DT_I16;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank = 1;
    src.shape.dim[0] = (int64_t)n;
    tdc_shape_set_contiguous(&src.shape);

    tdc_codec_spec spec = {0};
    spec.model = TDC_MODEL_DELTA_1D;
    spec.xform[0] = TDC_XFORM_ZIGZAG;
    spec.xform[1] = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0] = TDC_ENTROPY_LZ;

    tdc_buffer enc = {0};
    enc.realloc_fn = cn_realloc;

    tdc_status st = tdc_encode_block(&src, &spec, &enc);
    if (st != TDC_OK) {
        fprintf(stderr, "consumer_encode_i16: %s\n", tdc_strerror(st));
        cn_realloc(NULL, enc.data, 0);
        return NULL;
    }
    *out_size = enc.size;
    return enc.data;  /* caller frees */
}

/*
 * Decode one block into a consumer-owned destination. Returns 0 on
 * success, non-zero on any error (error written to stderr).
 */
static int consumer_decode_i16(const uint8_t *src, size_t src_size,
                               int16_t **out_values, size_t *out_n) {
    *out_values = NULL;
    *out_n = 0;

    tdc_block meta = {0};
    size_t need = 0;
    tdc_status st = tdc_decode_peek(src, src_size, &meta, &need);
    if (st != TDC_OK) {
        fprintf(stderr, "consumer_decode_i16: peek: %s\n", tdc_strerror(st));
        return 1;
    }
    if (meta.dtype != TDC_DT_I16 || meta.layout != TDC_LAYOUT_VECTOR_1D) {
        fprintf(stderr, "consumer_decode_i16: unexpected dtype/layout\n");
        return 2;
    }

    int16_t *dst_data = (int16_t *)cn_realloc(NULL, NULL, need);
    tdc_block dst = meta;
    dst.data = dst_data;

    st = tdc_decode_block_into(src, src_size, &dst);
    if (st != TDC_OK) {
        fprintf(stderr, "consumer_decode_i16: decode: %s\n", tdc_strerror(st));
        cn_realloc(NULL, dst_data, 0);
        return 3;
    }
    *out_values = dst_data;
    *out_n = (size_t)meta.shape.dim[0];
    return 0;
}

int main(void) {
    enum { N = 1024 };
    static int16_t values[N];
    int16_t v = -4096;
    for (int i = 0; i < N; ++i) { values[i] = v; v += 1 + (i & 3); }

    size_t enc_size = 0;
    uint8_t *encoded = consumer_encode_i16(values, N, &enc_size);
    if (!encoded) return 1;

    int16_t *decoded = NULL;
    size_t decoded_n = 0;
    if (consumer_decode_i16(encoded, enc_size, &decoded, &decoded_n) != 0) {
        cn_realloc(NULL, encoded, 0);
        return 1;
    }

    int ok = (decoded_n == N) &&
             memcmp(decoded, values, N * sizeof(int16_t)) == 0;
    printf("raw=%zu encoded=%zu decoded=%zu match=%s\n",
           N * sizeof(int16_t), enc_size, decoded_n * sizeof(int16_t),
           ok ? "yes" : "NO");

    cn_realloc(NULL, encoded, 0);
    cn_realloc(NULL, decoded, 0);
    return ok ? 0 : 1;
}
