/*
 * tests/test_dict_defer_e2e.c
 *
 * End-to-end test for tdc_decode_block_dict: encode a string column as a
 * DICT_1D block, decode it back as (dictionary + indices) WITHOUT flattening,
 * then rebuild the column by indexing the dictionary and compare against the
 * input. Also checks that tdc_decode_block_dict rejects a non-DICT block.
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

/* Encode `n` strings (given as concatenated data + n+1 offsets) as a DICT_1D
 * block, decode via tdc_decode_block_dict, rebuild, and compare. */
static int roundtrip(const char *label, const char *data,
                     const uint32_t *offsets, int64_t n) {
    tdc_block in = {0};
    in.data    = (void *)data;
    in.offsets = (uint32_t *)offsets;
    in.dtype   = TDC_DT_STRING;
    in.layout  = TDC_LAYOUT_VECTOR_1D;
    in.shape.rank   = 1;
    in.shape.dim[0] = n;
    in.shape.stride[0] = 1;

    tdc_codec_spec spec = {0};
    spec.model      = TDC_MODEL_DICT_1D;
    spec.entropy[0] = TDC_ENTROPY_LZ;

    tdc_buffer enc = make_buffer();
    tdc_status st = tdc_encode_block(&in, &spec, &enc);
    if (st != TDC_OK) {
        printf("  %-20s encode -> %d FAIL\n", label, (int)st);
        if (enc.data) test_realloc(NULL, enc.data, 0);
        return 1;
    }

    tdc_dict_block dict = {0};
    tdc_buffer alloc = make_buffer();
    st = tdc_decode_block_dict(enc.data, enc.size, &dict, &alloc);
    if (st != TDC_OK) {
        printf("  %-20s decode_dict -> %d FAIL\n", label, (int)st);
        test_realloc(NULL, enc.data, 0);
        return 1;
    }

    int rc = 0;
    if (dict.n != n) { printf("  %-20s n mismatch\n", label); rc = 1; }
    for (int64_t i = 0; i < n && !rc; ++i) {
        uint32_t idx = dict.indices[i];
        if (idx >= dict.dict_count) { printf("  %-20s idx oob\n", label); rc = 1; break; }
        uint32_t ds = dict.dict_offsets[idx];
        uint32_t de = dict.dict_offsets[idx + 1];
        uint32_t es = offsets[i];
        uint32_t ee = offsets[i + 1];
        if ((de - ds) != (ee - es) ||
            (de > ds && memcmp(dict.dict_data + ds, data + es, de - ds) != 0)) {
            printf("  %-20s value mismatch at %lld\n", label, (long long)i);
            rc = 1;
        }
    }
    if (!rc) printf("  %-20s OK (n=%lld dict=%u)\n", label,
                    (long long)n, dict.dict_count);

    if (dict.indices)      test_realloc(NULL, dict.indices, 0);
    if (dict.dict_offsets) test_realloc(NULL, dict.dict_offsets, 0);
    if (dict.dict_data)    test_realloc(NULL, dict.dict_data, 0);
    test_realloc(NULL, enc.data, 0);
    return rc;
}

int main(void) {
    int rc = 0;

    /* Duplicated values: "a","bb","a","ccc","bb","a". */
    {
        const char *data = "abbacccbba";
        const uint32_t off[] = {0, 1, 3, 4, 7, 9, 10};
        rc |= roundtrip("dup", data, off, 6);
    }
    /* All distinct. */
    {
        const char *data = "onetwothree";
        const uint32_t off[] = {0, 3, 6, 11};
        rc |= roundtrip("distinct", data, off, 3);
    }
    /* Empty strings mixed in: "", "x", "", "x". */
    {
        const char *data = "xx";
        const uint32_t off[] = {0, 0, 1, 1, 2};
        rc |= roundtrip("empty-mixed", data, off, 4);
    }

    /* Non-DICT block must be rejected with TDC_E_UNSUPPORTED. */
    {
        double v[4] = {1, 2, 3, 4};
        tdc_block in = {0};
        in.data = v; in.dtype = TDC_DT_F64; in.layout = TDC_LAYOUT_VECTOR_1D;
        in.shape.rank = 1; in.shape.dim[0] = 4; in.shape.stride[0] = 1;
        tdc_codec_spec spec = {0};
        spec.model = TDC_MODEL_RAW; spec.entropy[0] = TDC_ENTROPY_LZ;
        tdc_buffer enc = make_buffer();
        if (tdc_encode_block(&in, &spec, &enc) == TDC_OK) {
            tdc_dict_block d = {0};
            tdc_buffer alloc = make_buffer();
            tdc_status s = tdc_decode_block_dict(enc.data, enc.size, &d, &alloc);
            if (s != TDC_E_UNSUPPORTED) {
                printf("  non-dict reject   -> %d FAIL (want UNSUPPORTED)\n", (int)s);
                rc = 1;
            } else {
                printf("  non-dict reject   OK\n");
            }
        }
        if (enc.data) test_realloc(NULL, enc.data, 0);
    }

    printf(rc ? "FAILED\n" : "ALL OK\n");
    return rc;
}
