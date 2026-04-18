/* docs/examples/troubleshooting_statuses.c
 *
 * One call site per easily-triggered tdc_status. The program exercises
 * each failure path in turn and prints the returned code plus the string
 * from tdc_strerror. The output maps 1:1 to the status-code reference
 * table in the troubleshooting vignette.
 *
 * Build:
 *   cc -I include docs/examples/troubleshooting_statuses.c \
 *      build/libtdc.a -lm -o /tmp/troubleshooting_statuses
 */

#include "tdc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *ts_realloc(void *user, void *ptr, size_t n) {
    (void)user;
    if (n == 0) { free(ptr); return NULL; }
    return realloc(ptr, n);
}

static void report(const char *label, tdc_status st) {
    printf("%-26s -> status=%d (%s)\n",
           label, (int)st, tdc_strerror(st));
}

/* Make a minimal I32 VECTOR_1D block referencing caller-owned storage. */
static void make_i32_vec(tdc_block *b, int32_t *data, int64_t n) {
    memset(b, 0, sizeof(*b));
    b->data  = data;
    b->dtype = TDC_DT_I32;
    b->layout = TDC_LAYOUT_VECTOR_1D;
    b->shape.rank = 1;
    b->shape.dim[0] = n;
    tdc_shape_set_contiguous(&b->shape);
}

int main(void) {
    int32_t data[16];
    for (int i = 0; i < 16; ++i) data[i] = i;

    /* --- TDC_OK: reference baseline --------------------------------- */
    {
        tdc_block src; make_i32_vec(&src, data, 16);
        tdc_codec_spec spec = tdc_codec_spec_raw();
        tdc_buffer out = {0}; out.realloc_fn = ts_realloc;
        tdc_status st = tdc_encode_block(&src, &spec, &out);
        report("OK: raw encode", st);
        ts_realloc(NULL, out.data, 0);
    }

    /* --- TDC_E_INVAL: buffer without a realloc_fn ------------------- */
    {
        tdc_block src; make_i32_vec(&src, data, 16);
        tdc_codec_spec spec = tdc_codec_spec_raw();
        tdc_buffer out = {0};           /* realloc_fn left NULL on purpose */
        tdc_status st = tdc_encode_block(&src, &spec, &out);
        report("INVAL: no realloc_fn", st);
    }

    /* --- TDC_E_LAYOUT: 2D raster handed to DELTA_1D ----------------- */
    {
        tdc_block src = {0};
        src.data = data;
        src.dtype = TDC_DT_I32;
        src.layout = TDC_LAYOUT_RASTER_2D;
        src.shape.rank = 2;
        src.shape.dim[0] = 4;
        src.shape.dim[1] = 4;
        tdc_shape_set_contiguous(&src.shape);

        tdc_codec_spec spec = {0};
        spec.model = TDC_MODEL_DELTA_1D;       /* accepts VECTOR_1D only */
        tdc_buffer out = {0}; out.realloc_fn = ts_realloc;

        tdc_status st = tdc_encode_block(&src, &spec, &out);
        report("LAYOUT: 2D to DELTA_1D", st);
        ts_realloc(NULL, out.data, 0);
    }

    /* --- TDC_E_DTYPE: DELTA2_1D rejects integer vectors ------------
     * DELTA2_1D is a float-only XOR-delta model; handing it an I32
     * vector exercises the stage's accepted_dtypes gate. */
    {
        tdc_block src; make_i32_vec(&src, data, 16);
        tdc_codec_spec spec = {0};
        spec.model = TDC_MODEL_DELTA2_1D;
        tdc_buffer out = {0}; out.realloc_fn = ts_realloc;

        tdc_status st = tdc_encode_block(&src, &spec, &out);
        report("DTYPE: I32 to DELTA2_1D", st);
        ts_realloc(NULL, out.data, 0);
    }

    /* --- TDC_E_SHAPE: negative dimension ---------------------------- */
    {
        tdc_block src; make_i32_vec(&src, data, 16);
        src.shape.dim[0] = -1;                  /* rejected by validator */
        tdc_status st = tdc_block_validate(&src);
        report("SHAPE: negative dim", st);
    }

    /* --- TDC_E_UNSUPPORTED: no vtable for the model id -------------- */
    {
        tdc_block src; make_i32_vec(&src, data, 16);
        tdc_codec_spec spec = {0};
        spec.model = (tdc_model_id)0xFF01;      /* unassigned user slot */
        tdc_buffer out = {0}; out.realloc_fn = ts_realloc;
        tdc_status st = tdc_encode_block(&src, &spec, &out);
        report("UNSUPPORTED: bad model id", st);
        ts_realloc(NULL, out.data, 0);
    }

    /* --- TDC_E_CORRUPT: peek on a too-short buffer ------------------ */
    {
        uint8_t short_buf[10] = {0};            /* < TDC_BLOCK_HEADER_SIZE */
        tdc_block meta = {0};
        size_t need = 0;
        tdc_status st = tdc_decode_peek(short_buf, sizeof short_buf,
                                        &meta, &need);
        report("CORRUPT: short peek", st);
    }

    /* --- TDC_E_VERSION: peek on a record with a bad version --------- */
    {
        tdc_block src; make_i32_vec(&src, data, 16);
        tdc_codec_spec spec = tdc_codec_spec_raw();
        tdc_buffer out = {0}; out.realloc_fn = ts_realloc;
        (void)tdc_encode_block(&src, &spec, &out);
        /* version at offset 4 (little-endian u16); bump it to something
         * the current library cannot parse. */
        out.data[4] = 0xFE;
        out.data[5] = 0xCA;
        tdc_block meta = {0};
        size_t need = 0;
        tdc_status st = tdc_decode_peek(out.data, out.size, &meta, &need);
        report("VERSION: future version", st);
        ts_realloc(NULL, out.data, 0);
    }

    /* --- TDC_E_INVAL: NULL dst->data on decode_into ----------------
     * The canonical 'forgot to pre-allocate' case. decode_into refuses
     * to allocate on the caller's behalf; BUF_TOO_SMALL itself is
     * reached from internal stats parsers and cannot be triggered via
     * the one-shot encode/decode API. */
    {
        tdc_block src; make_i32_vec(&src, data, 16);
        tdc_codec_spec spec = tdc_codec_spec_raw();
        tdc_buffer enc = {0}; enc.realloc_fn = ts_realloc;
        (void)tdc_encode_block(&src, &spec, &enc);

        tdc_block dst; make_i32_vec(&dst, data, 16);
        dst.data = NULL;
        tdc_status st = tdc_decode_block_into(enc.data, enc.size, &dst);
        report("INVAL: NULL dst->data", st);
        ts_realloc(NULL, enc.data, 0);
    }

    return 0;
}
