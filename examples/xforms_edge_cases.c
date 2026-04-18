/* docs/examples/xforms_edge_cases.c
 *
 * Four edge cases for the transform chain:
 *   - empty block         (dim[0] = 0)
 *   - single element      (dim[0] = 1)
 *   - all-equal block     (residual after a constant is zero)
 *   - i8 block            (byte shuffle is a no-op at elem_size == 1)
 *
 * Each case is encoded and decoded through the full public pipeline and
 * the decoded bytes are memcmp'd against the source.
 *
 * Build:
 *   cc -I include docs/examples/xforms_edge_cases.c \
 *      build/libtdc.a -lm -o /tmp/xf_edge
 */

#include "quickstart_common.h"
#include <stdlib.h>
#include <string.h>

static int run_case(const char *label, const tdc_block *src, size_t raw) {
    tdc_codec_spec spec = {0};
    spec.model      = TDC_MODEL_RAW;
    spec.xform[0]   = TDC_XFORM_ZIGZAG;
    spec.xform[1]   = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0] = TDC_ENTROPY_LZ;

    tdc_buffer enc = qs_buffer();
    tdc_status st = tdc_encode_block(src, &spec, &enc);
    if (st != TDC_OK) {
        printf("  %-20s encode failed: %s\n", label, tdc_strerror(st));
        qs_buffer_free(&enc);
        return 1;
    }

    tdc_block meta = {0};
    size_t need = 0;
    if (qs_check(tdc_decode_peek(enc.data, enc.size, &meta, &need), "peek")) {
        qs_buffer_free(&enc);
        return 1;
    }

    void *dst_data = need ? qs_realloc(NULL, NULL, need) : NULL;
    tdc_block dst = meta;
    dst.data = dst_data;
    st = tdc_decode_block_into(enc.data, enc.size, &dst);
    if (st != TDC_OK) {
        printf("  %-20s decode failed: %s\n", label, tdc_strerror(st));
        qs_realloc(NULL, dst_data, 0);
        qs_buffer_free(&enc);
        return 1;
    }

    int ok = raw == 0 || memcmp(dst_data, src->data, raw) == 0;
    printf("  %-20s raw=%-6zu encoded=%-5zu memcmp=%s\n",
           label, raw, enc.size, ok ? "ok" : "FAIL");

    qs_realloc(NULL, dst_data, 0);
    qs_buffer_free(&enc);
    return ok ? 0 : 1;
}

int main(void) {
    int rc = 0;

    /* Case 1: empty i32 block. */
    {
        tdc_block src = {0};
        src.data = NULL;
        src.dtype  = TDC_DT_I32;
        src.layout = TDC_LAYOUT_VECTOR_1D;
        src.shape.rank   = 1;
        src.shape.dim[0] = 0;
        tdc_shape_set_contiguous(&src.shape);
        rc |= run_case("empty i32", &src, 0);
    }

    /* Case 2: single i64 element. */
    {
        int64_t v = 12345678;
        tdc_block src = {0};
        src.data   = &v;
        src.dtype  = TDC_DT_I64;
        src.layout = TDC_LAYOUT_VECTOR_1D;
        src.shape.rank   = 1;
        src.shape.dim[0] = 1;
        tdc_shape_set_contiguous(&src.shape);
        rc |= run_case("single i64", &src, sizeof v);
    }

    /* Case 3: all-equal i16 block. 2048 elements, all 42. */
    {
        enum { N = 2048 };
        static int16_t data[N];
        for (int i = 0; i < N; ++i) data[i] = 42;
        tdc_block src = {0};
        src.data   = data;
        src.dtype  = TDC_DT_I16;
        src.layout = TDC_LAYOUT_VECTOR_1D;
        src.shape.rank   = 1;
        src.shape.dim[0] = N;
        tdc_shape_set_contiguous(&src.shape);
        rc |= run_case("all-equal i16", &src, N * sizeof(int16_t));
    }

    /* Case 4: i8 block. BYTE_SHUFFLE with elem_size 1 is a memcpy.
     * ZIGZAG runs the per-byte kernel. Chain is still exercised end to end. */
    {
        enum { N = 4096 };
        static int8_t data[N];
        for (int i = 0; i < N; ++i) data[i] = (int8_t)((i * 3) % 17 - 8);
        tdc_block src = {0};
        src.data   = data;
        src.dtype  = TDC_DT_I8;
        src.layout = TDC_LAYOUT_VECTOR_1D;
        src.shape.rank   = 1;
        src.shape.dim[0] = N;
        tdc_shape_set_contiguous(&src.shape);
        rc |= run_case("i8 elem_size==1", &src, N);
    }

    return rc;
}
