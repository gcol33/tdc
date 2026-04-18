/* docs/examples/integration_error.c
 *
 * Shows how a consumer can wrap tdc_status into its own error type with
 * source context added. The library itself returns only a status code;
 * a consumer that wants richer diagnostics (file path, block index,
 * codec spec) attaches them around the call.
 *
 * Build:
 *   cc -I include docs/examples/integration_error.c \
 *      build/libtdc.a -lm -o /tmp/integration_error
 */

#include "tdc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ----- Consumer-side error type --------------------------------------- */

typedef enum {
    APP_OK = 0,
    APP_ERR_TDC,      /* wraps a tdc_status */
    APP_ERR_IO,
    APP_ERR_INPUT
} app_status;

typedef struct {
    app_status code;
    tdc_status tdc_code;        /* meaningful only when code == APP_ERR_TDC */
    const char *where;          /* static string: which call site */
    size_t      block_index;    /* row group / block in the container */
    char        message[128];   /* pre-formatted one-line message */
} app_error;

static void app_error_clear(app_error *e) {
    memset(e, 0, sizeof(*e));
}

static void app_wrap_tdc(app_error *e, tdc_status st,
                         const char *where, size_t block_index) {
    e->code = (st == TDC_OK) ? APP_OK : APP_ERR_TDC;
    e->tdc_code = st;
    e->where = where;
    e->block_index = block_index;
    /* Compose a one-line log message. tdc_strerror returns a static
     * literal; we can copy it without worrying about lifetime. */
    snprintf(e->message, sizeof(e->message),
             "tdc[%s] block=%zu status=%d (%s)",
             where ? where : "?", block_index, (int)st, tdc_strerror(st));
}

/* ----- Realloc wrapper ------------------------------------------------ */

static void *app_realloc(void *user, void *ptr, size_t n) {
    (void)user;
    if (n == 0) { free(ptr); return NULL; }
    return realloc(ptr, n);
}

/* ----- Workload: deliberately wrong codec spec ------------------------ */
/*
 * Ask DELTA_1D to compress a 2D raster. The dispatcher rejects the
 * layout mismatch with TDC_E_LAYOUT. We surface that into the consumer
 * error channel with full context.
 */
int main(void) {
    enum { W = 32, H = 32 };
    static int32_t img[W * H];
    for (int i = 0; i < W * H; ++i) img[i] = i;

    tdc_block src = {0};
    src.data = img;
    src.dtype = TDC_DT_I32;
    src.layout = TDC_LAYOUT_RASTER_2D;
    src.shape.rank = 2;
    src.shape.dim[0] = H;
    src.shape.dim[1] = W;
    tdc_shape_set_contiguous(&src.shape);

    tdc_codec_spec spec = {0};
    spec.model = TDC_MODEL_DELTA_1D;  /* wrong model for a 2D raster */

    tdc_buffer enc = {0};
    enc.realloc_fn = app_realloc;

    app_error err;
    app_error_clear(&err);

    tdc_status st = tdc_encode_block(&src, &spec, &enc);
    app_wrap_tdc(&err, st, "encode_row_group_block", /*block_index*/ 42);

    if (err.code != APP_OK) {
        fprintf(stderr, "%s\n", err.message);
        /* The caller dispatches on err.tdc_code to decide what to do. */
        if (err.tdc_code == TDC_E_LAYOUT) {
            fprintf(stderr, "  fix: pick a model that accepts RASTER_2D (e.g. PRED_2D)\n");
        } else if (err.tdc_code == TDC_E_DTYPE) {
            fprintf(stderr, "  fix: pick a dtype-compatible model for this column\n");
        } else if (err.tdc_code == TDC_E_CORRUPT) {
            fprintf(stderr, "  fix: re-read the source block; header failed validation\n");
        }
    }

    /* Retry with a model that accepts RASTER_2D. */
    spec.model = TDC_MODEL_PRED_2D;
    st = tdc_encode_block(&src, &spec, &enc);
    app_wrap_tdc(&err, st, "encode_row_group_block", /*block_index*/ 42);
    printf("retry: %s\n", err.code == APP_OK ? "TDC_OK" : err.message);
    printf("encoded size: %zu bytes\n", enc.size);

    app_realloc(NULL, enc.data, 0);
    return (err.code == APP_OK) ? 0 : 1;
}
