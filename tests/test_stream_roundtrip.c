/*
 * tests/test_stream_roundtrip.c
 *
 * Comprehensive round-trip test for the streaming container with schema,
 * row-group index, and random-access by (row_group, column).
 *
 * Test scenario:
 *   - Schema: 3 columns ("temperature" f64, "pressure" i32, "label" u8)
 *   - 2 row groups of 100 rows each
 *   - 3 blocks per row group (one per column)
 *   - RAW model + no transforms + NONE entropy (container test, not compression)
 *   - Sequential decode: all 6 blocks verified
 *   - Random access: seek to (rg=1, col=1), decode and verify "pressure"
 */

#include "tdc.h"
#include "tdc/stream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ===== In-memory I/O adapter ============================================= */

typedef struct {
    uint8_t *data;
    size_t   size;     /* bytes written */
    size_t   capacity;
    size_t   cursor;   /* read/seek position */
} mem_io;

static mem_io mem_io_new(void) {
    mem_io m = {0};
    return m;
}

static void mem_io_free(mem_io *m) {
    free(m->data);
    memset(m, 0, sizeof(*m));
}

static tdc_status mem_write(void *ctx, const void *data, size_t size) {
    mem_io *m = (mem_io *)ctx;
    size_t need = m->cursor + size;
    if (need > m->capacity) {
        size_t new_cap = m->capacity ? m->capacity : 64;
        while (new_cap < need) new_cap *= 2;
        void *p = realloc(m->data, new_cap);
        if (!p) return TDC_E_NOMEM;
        m->data     = (uint8_t *)p;
        m->capacity = new_cap;
    }
    memcpy(m->data + m->cursor, data, size);
    m->cursor += size;
    if (m->cursor > m->size) m->size = m->cursor;
    return TDC_OK;
}

static tdc_status mem_read(void *ctx, void *buf, size_t size,
                           size_t *bytes_read) {
    mem_io *m = (mem_io *)ctx;
    size_t avail = (m->cursor < m->size) ? m->size - m->cursor : 0;
    size_t n = (size < avail) ? size : avail;
    if (n > 0) memcpy(buf, m->data + m->cursor, n);
    m->cursor += n;
    *bytes_read = n;
    return TDC_OK;
}

static tdc_status mem_seek(void *ctx, int64_t offset, int whence) {
    mem_io *m = (mem_io *)ctx;
    int64_t base = 0;
    switch (whence) {
        case TDC_SEEK_SET: base = 0; break;
        case TDC_SEEK_CUR: base = (int64_t)m->cursor; break;
        case TDC_SEEK_END: base = (int64_t)m->size; break;
        default: return TDC_E_INVAL;
    }
    int64_t pos = base + offset;
    if (pos < 0) return TDC_E_INVAL;
    m->cursor = (size_t)pos;
    return TDC_OK;
}

/* ===== Test allocator ==================================================== */

static void *test_realloc(void *user, void *ptr, size_t new_size) {
    (void)user;
    if (new_size == 0) { free(ptr); return NULL; }
    return realloc(ptr, new_size);
}

/* ===== Assertion macro =================================================== */

#define ASSERT_OR_DIE(cond, msg) do {                                       \
    if (!(cond)) {                                                          \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);   \
        return 1;                                                           \
    }                                                                       \
} while (0)

/* ===== Constants ========================================================= */

#define N_ROWS       100
#define N_ROWGROUPS  2
#define N_COLUMNS    3

/* ===== Test data generation ============================================== */

/*
 * Deterministic test data. Each column uses a simple formula so round-trip
 * correctness is trivially verifiable:
 *
 *   temperature (f64):  100.0 + row * 0.5
 *   pressure    (i32):  1000  + row * 3
 *   label       (u8):   row % 256
 *
 * `row` is the absolute row index across the full stream (0..199), but each
 * row group holds N_ROWS elements. Within row group `rg`, the local index
 * `i` maps to absolute row `rg * N_ROWS + i`.
 */

static void fill_temperature(double *buf, int rg) {
    for (int i = 0; i < N_ROWS; ++i) {
        int row = rg * N_ROWS + i;
        buf[i] = 100.0 + (double)row * 0.5;
    }
}

static void fill_pressure(int32_t *buf, int rg) {
    for (int i = 0; i < N_ROWS; ++i) {
        int row = rg * N_ROWS + i;
        buf[i] = 1000 + row * 3;
    }
}

static void fill_label(uint8_t *buf, int rg) {
    for (int i = 0; i < N_ROWS; ++i) {
        int row = rg * N_ROWS + i;
        buf[i] = (uint8_t)(row % 256);
    }
}

/* ===== Block builders ==================================================== */

static tdc_block make_block_f64(double *data, int64_t n) {
    tdc_block blk;
    memset(&blk, 0, sizeof(blk));
    blk.data   = data;
    blk.dtype  = TDC_DT_F64;
    blk.layout = TDC_LAYOUT_VECTOR_1D;
    blk.shape.rank   = 1;
    blk.shape.dim[0] = n;
    blk.shape.dim[1] = 0;
    blk.shape.dim[2] = 0;
    tdc_shape_set_contiguous(&blk.shape);
    return blk;
}

static tdc_block make_block_i32(int32_t *data, int64_t n) {
    tdc_block blk;
    memset(&blk, 0, sizeof(blk));
    blk.data   = data;
    blk.dtype  = TDC_DT_I32;
    blk.layout = TDC_LAYOUT_VECTOR_1D;
    blk.shape.rank   = 1;
    blk.shape.dim[0] = n;
    blk.shape.dim[1] = 0;
    blk.shape.dim[2] = 0;
    tdc_shape_set_contiguous(&blk.shape);
    return blk;
}

static tdc_block make_block_u8(uint8_t *data, int64_t n) {
    tdc_block blk;
    memset(&blk, 0, sizeof(blk));
    blk.data   = data;
    blk.dtype  = TDC_DT_U8;
    blk.layout = TDC_LAYOUT_VECTOR_1D;
    blk.shape.rank   = 1;
    blk.shape.dim[0] = n;
    blk.shape.dim[1] = 0;
    blk.shape.dim[2] = 0;
    tdc_shape_set_contiguous(&blk.shape);
    return blk;
}

/* ===== Codec spec: RAW + no transforms + NONE entropy ==================== */

static tdc_codec_spec make_raw_spec(void) {
    tdc_codec_spec spec;
    memset(&spec, 0, sizeof(spec));
    spec.model = TDC_MODEL_RAW;
    /* xform[] and entropy[] are all zero = NONE (passthrough) */
    return spec;
}

/* ===== Schema definition ================================================= */

static const tdc_column_desc test_columns[N_COLUMNS] = {
    { "temperature", 11, TDC_DT_F64, "degrees_celsius", 15 },
    { "pressure",     8, TDC_DT_I32, "pascal",           6 },
    { "label",        5, TDC_DT_U8,  NULL,               0 }
};

static const tdc_schema test_schema = {
    N_COLUMNS,
    test_columns
};

/* ===== The round-trip test =============================================== */

static int test_stream_roundtrip(void) {
    mem_io mio = mem_io_new();
    tdc_status st;

    /* ------------------------------------------------------------------ */
    /* Phase 1: Encode                                                     */
    /* ------------------------------------------------------------------ */

    tdc_stream_encoder_config ecfg;
    memset(&ecfg, 0, sizeof(ecfg));
    ecfg.io.write_fn    = mem_write;
    ecfg.io.read_fn     = NULL;
    ecfg.io.seek_fn     = mem_seek;
    ecfg.io.ctx          = &mio;
    ecfg.flags           = TDC_CONTAINER_FLAG_HETEROGENEOUS;
    ecfg.schema          = &test_schema;
    ecfg.realloc_fn      = test_realloc;
    ecfg.alloc_user      = NULL;

    tdc_stream_encoder *enc = NULL;
    st = tdc_stream_encoder_open(&ecfg, &enc);
    ASSERT_OR_DIE(st == TDC_OK, "encoder open");
    ASSERT_OR_DIE(enc != NULL,  "encoder non-null");

    tdc_codec_spec raw_spec = make_raw_spec();

    /* Source data buffers — kept alive for verification later. */
    double   temp_src[N_ROWGROUPS][N_ROWS];
    int32_t  pres_src[N_ROWGROUPS][N_ROWS];
    uint8_t  labl_src[N_ROWGROUPS][N_ROWS];

    for (int rg = 0; rg < N_ROWGROUPS; ++rg) {
        /* Generate deterministic data for this row group. */
        fill_temperature(temp_src[rg], rg);
        fill_pressure(pres_src[rg], rg);
        fill_label(labl_src[rg], rg);

        /* Write 3 blocks: temperature, pressure, label. */
        tdc_block blk_temp = make_block_f64(temp_src[rg], N_ROWS);
        st = tdc_stream_encoder_write_block(enc, &blk_temp, &raw_spec);
        ASSERT_OR_DIE(st == TDC_OK, "write temperature block");

        tdc_block blk_pres = make_block_i32(pres_src[rg], N_ROWS);
        st = tdc_stream_encoder_write_block(enc, &blk_pres, &raw_spec);
        ASSERT_OR_DIE(st == TDC_OK, "write pressure block");

        tdc_block blk_labl = make_block_u8(labl_src[rg], N_ROWS);
        st = tdc_stream_encoder_write_block(enc, &blk_labl, &raw_spec);
        ASSERT_OR_DIE(st == TDC_OK, "write label block");

        /* Close out the row group. */
        st = tdc_stream_encoder_end_rowgroup(enc, (uint64_t)N_ROWS);
        ASSERT_OR_DIE(st == TDC_OK, "end rowgroup");
    }

    /* Close encoder: writes trailing row-group index, patches header. */
    st = tdc_stream_encoder_close(&enc);
    ASSERT_OR_DIE(st == TDC_OK, "encoder close");
    ASSERT_OR_DIE(enc == NULL,  "enc nulled after close");

    /* Sanity: the output buffer should have content. */
    ASSERT_OR_DIE(mio.size > TDC_CONTAINER_HEADER_SIZE,
                  "output larger than bare header");

    /* ------------------------------------------------------------------ */
    /* Phase 2: Decode — schema validation                                 */
    /* ------------------------------------------------------------------ */

    mio.cursor = 0;

    tdc_stream_decoder_config dcfg;
    memset(&dcfg, 0, sizeof(dcfg));
    dcfg.io.write_fn = NULL;
    dcfg.io.read_fn  = mem_read;
    dcfg.io.seek_fn  = mem_seek;
    dcfg.io.ctx      = &mio;
    dcfg.realloc_fn  = test_realloc;
    dcfg.alloc_user  = NULL;

    tdc_stream_decoder *dec = NULL;
    st = tdc_stream_decoder_open(&dcfg, &dec);
    ASSERT_OR_DIE(st == TDC_OK, "decoder open");
    ASSERT_OR_DIE(dec != NULL,  "decoder non-null");

    /* Read schema and verify columns. */
    const tdc_schema *schema = tdc_stream_decoder_read_schema(dec);
    ASSERT_OR_DIE(schema != NULL, "schema non-null");
    ASSERT_OR_DIE(schema->n_columns == N_COLUMNS, "schema n_columns == 3");

    /* Column 0: temperature, f64, "degrees_celsius" */
    ASSERT_OR_DIE(schema->columns[0].name_len == 11,
                  "col0 name_len");
    ASSERT_OR_DIE(memcmp(schema->columns[0].name, "temperature", 11) == 0,
                  "col0 name == temperature");
    ASSERT_OR_DIE(schema->columns[0].dtype == TDC_DT_F64,
                  "col0 dtype == F64");
    ASSERT_OR_DIE(schema->columns[0].ann_len == 15,
                  "col0 ann_len");
    ASSERT_OR_DIE(memcmp(schema->columns[0].annotation,
                         "degrees_celsius", 15) == 0,
                  "col0 annotation == degrees_celsius");

    /* Column 1: pressure, i32, "pascal" */
    ASSERT_OR_DIE(schema->columns[1].name_len == 8,
                  "col1 name_len");
    ASSERT_OR_DIE(memcmp(schema->columns[1].name, "pressure", 8) == 0,
                  "col1 name == pressure");
    ASSERT_OR_DIE(schema->columns[1].dtype == TDC_DT_I32,
                  "col1 dtype == I32");
    ASSERT_OR_DIE(schema->columns[1].ann_len == 6,
                  "col1 ann_len");
    ASSERT_OR_DIE(memcmp(schema->columns[1].annotation, "pascal", 6) == 0,
                  "col1 annotation == pascal");

    /* Column 2: label, u8, no annotation */
    ASSERT_OR_DIE(schema->columns[2].name_len == 5,
                  "col2 name_len");
    ASSERT_OR_DIE(memcmp(schema->columns[2].name, "label", 5) == 0,
                  "col2 name == label");
    ASSERT_OR_DIE(schema->columns[2].dtype == TDC_DT_U8,
                  "col2 dtype == U8");
    ASSERT_OR_DIE(schema->columns[2].ann_len == 0,
                  "col2 ann_len == 0 (no annotation)");

    /* ------------------------------------------------------------------ */
    /* Phase 3: Row-group index validation                                 */
    /* ------------------------------------------------------------------ */

    ASSERT_OR_DIE(tdc_stream_decoder_has_rowgroup_index(dec),
                  "has rowgroup index");
    ASSERT_OR_DIE(tdc_stream_decoder_rowgroup_count(dec) == N_ROWGROUPS,
                  "rowgroup count == 2");

    for (int rg = 0; rg < N_ROWGROUPS; ++rg) {
        const tdc_rowgroup_entry *rge =
            tdc_stream_decoder_get_rowgroup(dec, (uint64_t)rg);
        ASSERT_OR_DIE(rge != NULL, "rowgroup entry non-null");
        ASSERT_OR_DIE(rge->n_rows == (uint64_t)N_ROWS,
                      "rowgroup n_rows == 100");
        ASSERT_OR_DIE(rge->n_cols == N_COLUMNS,
                      "rowgroup n_cols == 3");

        /* Each column entry should have nonzero block_total (at minimum
         * the 80-byte block record header). */
        for (int c = 0; c < N_COLUMNS; ++c) {
            ASSERT_OR_DIE(rge->columns[c].block_total >= TDC_BLOCK_HEADER_SIZE,
                          "column block_total >= 80");
        }
    }

    /* Row group offsets must be strictly increasing. */
    {
        const tdc_rowgroup_entry *rg0 =
            tdc_stream_decoder_get_rowgroup(dec, 0);
        const tdc_rowgroup_entry *rg1 =
            tdc_stream_decoder_get_rowgroup(dec, 1);
        ASSERT_OR_DIE(rg1->offset > rg0->offset,
                      "rg1 offset > rg0 offset");
    }

    /* ------------------------------------------------------------------ */
    /* Phase 4: Sequential decode — read all 6 blocks, verify data         */
    /* ------------------------------------------------------------------ */

    for (int rg = 0; rg < N_ROWGROUPS; ++rg) {
        /* Column 0: temperature (f64) */
        {
            tdc_block_record rec;
            st = tdc_stream_decoder_peek_block(dec, &rec);
            ASSERT_OR_DIE(st == TDC_OK, "peek temperature block");
            ASSERT_OR_DIE(rec.magic == TDC_BLOCK_MAGIC,
                          "temperature block magic");
            ASSERT_OR_DIE(rec.dtype == TDC_DT_F64,
                          "temperature block dtype");
            ASSERT_OR_DIE(rec.dim[0] == N_ROWS,
                          "temperature block dim[0]");

            double *dst = (double *)test_realloc(
                NULL, NULL, (size_t)N_ROWS * sizeof(double));
            ASSERT_OR_DIE(dst != NULL, "alloc temperature dst");

            tdc_block dblk = make_block_f64(dst, N_ROWS);
            st = tdc_stream_decoder_read_block(dec, &dblk);
            ASSERT_OR_DIE(st == TDC_OK, "read temperature block");

            for (int i = 0; i < N_ROWS; ++i) {
                ASSERT_OR_DIE(dst[i] == temp_src[rg][i],
                              "temperature data mismatch");
            }
            test_realloc(NULL, dst, 0);
        }

        /* Column 1: pressure (i32) */
        {
            tdc_block_record rec;
            st = tdc_stream_decoder_peek_block(dec, &rec);
            ASSERT_OR_DIE(st == TDC_OK, "peek pressure block");
            ASSERT_OR_DIE(rec.magic == TDC_BLOCK_MAGIC,
                          "pressure block magic");
            ASSERT_OR_DIE(rec.dtype == TDC_DT_I32,
                          "pressure block dtype");

            int32_t *dst = (int32_t *)test_realloc(
                NULL, NULL, (size_t)N_ROWS * sizeof(int32_t));
            ASSERT_OR_DIE(dst != NULL, "alloc pressure dst");

            tdc_block dblk = make_block_i32(dst, N_ROWS);
            st = tdc_stream_decoder_read_block(dec, &dblk);
            ASSERT_OR_DIE(st == TDC_OK, "read pressure block");

            for (int i = 0; i < N_ROWS; ++i) {
                ASSERT_OR_DIE(dst[i] == pres_src[rg][i],
                              "pressure data mismatch");
            }
            test_realloc(NULL, dst, 0);
        }

        /* Column 2: label (u8) */
        {
            tdc_block_record rec;
            st = tdc_stream_decoder_peek_block(dec, &rec);
            ASSERT_OR_DIE(st == TDC_OK, "peek label block");
            ASSERT_OR_DIE(rec.magic == TDC_BLOCK_MAGIC,
                          "label block magic");
            ASSERT_OR_DIE(rec.dtype == TDC_DT_U8,
                          "label block dtype");

            uint8_t *dst = (uint8_t *)test_realloc(
                NULL, NULL, (size_t)N_ROWS * sizeof(uint8_t));
            ASSERT_OR_DIE(dst != NULL, "alloc label dst");

            tdc_block dblk = make_block_u8(dst, N_ROWS);
            st = tdc_stream_decoder_read_block(dec, &dblk);
            ASSERT_OR_DIE(st == TDC_OK, "read label block");

            for (int i = 0; i < N_ROWS; ++i) {
                ASSERT_OR_DIE(dst[i] == labl_src[rg][i],
                              "label data mismatch");
            }
            test_realloc(NULL, dst, 0);
        }
    }

    /* ------------------------------------------------------------------ */
    /* Phase 5: Random access — seek to row group 1, column 1 (pressure)   */
    /* ------------------------------------------------------------------ */

    {
        tdc_block_record rec;
        st = tdc_stream_decoder_seek_rowgroup(dec, 1, 1, &rec);
        ASSERT_OR_DIE(st == TDC_OK, "seek to rg=1 col=1");
        ASSERT_OR_DIE(rec.magic == TDC_BLOCK_MAGIC,
                      "seeked block magic");
        ASSERT_OR_DIE(rec.dtype == TDC_DT_I32,
                      "seeked block dtype == I32");
        ASSERT_OR_DIE(rec.dim[0] == N_ROWS,
                      "seeked block dim[0] == 100");

        int32_t *dst = (int32_t *)test_realloc(
            NULL, NULL, (size_t)N_ROWS * sizeof(int32_t));
        ASSERT_OR_DIE(dst != NULL, "alloc seek dst");

        tdc_block dblk = make_block_i32(dst, N_ROWS);
        st = tdc_stream_decoder_read_block(dec, &dblk);
        ASSERT_OR_DIE(st == TDC_OK, "read seeked pressure block");

        /* Verify against row group 1 pressure data. */
        for (int i = 0; i < N_ROWS; ++i) {
            ASSERT_OR_DIE(dst[i] == pres_src[1][i],
                          "seeked pressure data mismatch");
        }
        test_realloc(NULL, dst, 0);
    }

    /* ------------------------------------------------------------------ */
    /* Phase 6: Random access — seek to row group 0, column 2 (label)      */
    /* ------------------------------------------------------------------ */

    {
        tdc_block_record rec;
        st = tdc_stream_decoder_seek_rowgroup(dec, 0, 2, &rec);
        ASSERT_OR_DIE(st == TDC_OK, "seek to rg=0 col=2");
        ASSERT_OR_DIE(rec.dtype == TDC_DT_U8,
                      "seeked label dtype == U8");

        uint8_t *dst = (uint8_t *)test_realloc(
            NULL, NULL, (size_t)N_ROWS * sizeof(uint8_t));
        ASSERT_OR_DIE(dst != NULL, "alloc seek label dst");

        tdc_block dblk = make_block_u8(dst, N_ROWS);
        st = tdc_stream_decoder_read_block(dec, &dblk);
        ASSERT_OR_DIE(st == TDC_OK, "read seeked label block");

        for (int i = 0; i < N_ROWS; ++i) {
            ASSERT_OR_DIE(dst[i] == labl_src[0][i],
                          "seeked label data mismatch");
        }
        test_realloc(NULL, dst, 0);
    }

    /* ------------------------------------------------------------------ */
    /* Phase 7: Random access — seek to row group 1, column 0 (temperature)*/
    /* ------------------------------------------------------------------ */

    {
        tdc_block_record rec;
        st = tdc_stream_decoder_seek_rowgroup(dec, 1, 0, &rec);
        ASSERT_OR_DIE(st == TDC_OK, "seek to rg=1 col=0");
        ASSERT_OR_DIE(rec.dtype == TDC_DT_F64,
                      "seeked temp dtype == F64");

        double *dst = (double *)test_realloc(
            NULL, NULL, (size_t)N_ROWS * sizeof(double));
        ASSERT_OR_DIE(dst != NULL, "alloc seek temp dst");

        tdc_block dblk = make_block_f64(dst, N_ROWS);
        st = tdc_stream_decoder_read_block(dec, &dblk);
        ASSERT_OR_DIE(st == TDC_OK, "read seeked temperature block");

        for (int i = 0; i < N_ROWS; ++i) {
            ASSERT_OR_DIE(dst[i] == temp_src[1][i],
                          "seeked temperature data mismatch");
        }
        test_realloc(NULL, dst, 0);
    }

    /* ------------------------------------------------------------------ */
    /* Phase 8: Close decoder                                              */
    /* ------------------------------------------------------------------ */

    st = tdc_stream_decoder_close(&dec);
    ASSERT_OR_DIE(st == TDC_OK, "decoder close");
    ASSERT_OR_DIE(dec == NULL,  "dec nulled after close");

    mem_io_free(&mio);
    return 0;
}

/* ===== Main ============================================================== */

int main(void) {
    printf("test_stream_roundtrip\n");
    int fail = 0;

    fail += test_stream_roundtrip();

    printf("\n  %d test(s) failed\n", fail);
    return fail ? 1 : 0;
}
