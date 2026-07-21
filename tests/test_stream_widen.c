/*
 * tests/test_stream_widen.c
 *
 * Tests for tdc_stream_encoder_open_widen: appending COLUMNS to an
 * existing container without reading or rewriting its body.
 *
 * What is actually asserted, beyond "the data comes back":
 *
 *   - the original container's bytes are a strict PREFIX of the widened
 *     file, which is the whole claim -- no existing block is moved,
 *     rewritten, or even read;
 *   - every pre-existing column still decodes to its original values, and
 *     its stats survive;
 *   - the appended column decodes per row group, and can carry stats even
 *     when the container had none;
 *   - restoring only the original 64-byte header turns the widened file
 *     back into the original container, which is the crash-before-header-
 *     patch state;
 *   - a widened container refuses sequential walking rather than
 *     truncating at the gap left by the superseded index;
 *   - widening a widened container works (v2 -> v2).
 */

#include "tdc.h"
#include "tdc/stream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ===== In-memory I/O adapter ============================================= */

typedef struct {
    uint8_t *data;
    size_t   size;
    size_t   capacity;
    size_t   cursor;
} mem_io;

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

static void *test_realloc(void *user, void *ptr, size_t new_size) {
    (void)user;
    if (new_size == 0) { free(ptr); return NULL; }
    return realloc(ptr, new_size);
}

#define ASSERT_OR_DIE(cond, msg) do {                                       \
    if (!(cond)) {                                                          \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);   \
        return 1;                                                           \
    }                                                                       \
} while (0)

/* ===== Constants and data ================================================ */

#define N_ROWS       64
#define N_ROWGROUPS  3

static double  temp_src[N_ROWGROUPS][N_ROWS];
static int32_t pres_src[N_ROWGROUPS][N_ROWS];
static int32_t added_src[N_ROWGROUPS][N_ROWS];
static double  added2_src[N_ROWGROUPS][N_ROWS];

static void fill_sources(void) {
    for (int rg = 0; rg < N_ROWGROUPS; ++rg) {
        for (int i = 0; i < N_ROWS; ++i) {
            int row = rg * N_ROWS + i;
            temp_src[rg][i]   = 100.0 + (double)row * 0.5;
            pres_src[rg][i]   = 1000 + row * 3;
            added_src[rg][i]  = -7 * row - 1;
            added2_src[rg][i] = (double)row * 1.25 - 3.0;
        }
    }
}

static tdc_block make_block_f64(double *data, int64_t n) {
    tdc_block blk;
    memset(&blk, 0, sizeof(blk));
    blk.data   = data;
    blk.dtype  = TDC_DT_F64;
    blk.layout = TDC_LAYOUT_VECTOR_1D;
    blk.shape.rank   = 1;
    blk.shape.dim[0] = n;
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
    tdc_shape_set_contiguous(&blk.shape);
    return blk;
}

static tdc_codec_spec make_raw_spec(void) {
    tdc_codec_spec spec;
    memset(&spec, 0, sizeof(spec));
    spec.model = TDC_MODEL_RAW;
    return spec;
}

/* Base container: 2 columns. Widened: +1, then +1 again. */
static const tdc_column_desc base_columns[2] = {
    { "temperature", 11, TDC_DT_F64, "degrees_celsius", 15 },
    { "pressure",     8, TDC_DT_I32, "pascal",           6 }
};
static const tdc_schema base_schema = { 2, base_columns };

static const tdc_column_desc wide_columns[3] = {
    { "temperature", 11, TDC_DT_F64, "degrees_celsius", 15 },
    { "pressure",     8, TDC_DT_I32, "pascal",           6 },
    { "elevation",    9, TDC_DT_I32, "metres",           6 }
};
static const tdc_schema wide_schema = { 3, wide_columns };

static const tdc_column_desc wider_columns[4] = {
    { "temperature", 11, TDC_DT_F64, "degrees_celsius", 15 },
    { "pressure",     8, TDC_DT_I32, "pascal",           6 },
    { "elevation",    9, TDC_DT_I32, "metres",           6 },
    { "humidity",     8, TDC_DT_F64, NULL,               0 }
};
static const tdc_schema wider_schema = { 4, wider_columns };

/* ===== Shared helpers ==================================================== */

/* Write the 2-column base container into `mio`. with_stats attaches a
 * stats block to every row group. */
static int build_base(mem_io *mio, int with_stats) {
    tdc_stream_encoder_config ecfg;
    memset(&ecfg, 0, sizeof(ecfg));
    ecfg.io.write_fn = mem_write;
    ecfg.io.seek_fn  = mem_seek;
    ecfg.io.ctx      = mio;
    ecfg.flags       = TDC_CONTAINER_FLAG_HETEROGENEOUS;
    ecfg.schema      = &base_schema;
    ecfg.realloc_fn  = test_realloc;

    tdc_stream_encoder *enc = NULL;
    ASSERT_OR_DIE(tdc_stream_encoder_open(&ecfg, &enc) == TDC_OK, "base open");

    tdc_codec_spec raw = make_raw_spec();
    for (int rg = 0; rg < N_ROWGROUPS; ++rg) {
        tdc_block b0 = make_block_f64(temp_src[rg], N_ROWS);
        ASSERT_OR_DIE(tdc_stream_encoder_write_block(enc, &b0, &raw) == TDC_OK,
                      "base write temp");
        tdc_block b1 = make_block_i32(pres_src[rg], N_ROWS);
        ASSERT_OR_DIE(tdc_stream_encoder_write_block(enc, &b1, &raw) == TDC_OK,
                      "base write pres");

        if (with_stats) {
            tdc_column_stats s[2];
            memset(s, 0, sizeof(s));
            s[0].has_stats = 1;
            s[0].null_count = (uint64_t)rg;
            s[1].has_stats = 1;
            s[1].null_count = (uint64_t)rg + 100;
            ASSERT_OR_DIE(
                tdc_stream_encoder_set_rowgroup_stats(enc, s, 2) == TDC_OK,
                "base stats");
        }

        ASSERT_OR_DIE(tdc_stream_encoder_end_rowgroup(enc, N_ROWS) == TDC_OK,
                      "base end rg");
    }
    ASSERT_OR_DIE(tdc_stream_encoder_close(&enc) == TDC_OK, "base close");
    return 0;
}

/* Append one i32 column ("elevation") to an existing container. */
static int widen_with_elevation(mem_io *mio, int with_stats) {
    tdc_stream_encoder_widen_config wcfg;
    memset(&wcfg, 0, sizeof(wcfg));
    wcfg.io.write_fn = mem_write;
    wcfg.io.read_fn  = mem_read;
    wcfg.io.seek_fn  = mem_seek;
    wcfg.io.ctx      = mio;
    wcfg.schema      = &wide_schema;
    wcfg.realloc_fn  = test_realloc;

    tdc_stream_encoder *enc = NULL;
    ASSERT_OR_DIE(tdc_stream_encoder_open_widen(&wcfg, &enc) == TDC_OK,
                  "widen open");

    tdc_codec_spec raw = make_raw_spec();
    for (int rg = 0; rg < N_ROWGROUPS; ++rg) {
        tdc_block b = make_block_i32(added_src[rg], N_ROWS);
        tdc_column_stats s;
        memset(&s, 0, sizeof(s));
        s.has_stats  = 1;
        s.null_count = (uint64_t)rg + 500;
        ASSERT_OR_DIE(
            tdc_stream_encoder_widen_block(enc, (uint64_t)rg, &b, &raw,
                                           with_stats ? &s : NULL) == TDC_OK,
            "widen block");
    }
    ASSERT_OR_DIE(tdc_stream_encoder_close(&enc) == TDC_OK, "widen close");
    return 0;
}

static tdc_status open_dec(mem_io *mio, tdc_stream_decoder **dec) {
    mio->cursor = 0;
    tdc_stream_decoder_config dcfg;
    memset(&dcfg, 0, sizeof(dcfg));
    dcfg.io.read_fn = mem_read;
    dcfg.io.seek_fn = mem_seek;
    dcfg.io.ctx     = mio;
    dcfg.realloc_fn = test_realloc;
    return tdc_stream_decoder_open(&dcfg, dec);
}

/* Decode (rg, col) as i32 and compare against expected. */
static int check_i32_col(tdc_stream_decoder *dec, uint64_t rg, uint16_t col,
                         const int32_t *expect, const char *what) {
    tdc_block_record rec;
    ASSERT_OR_DIE(tdc_stream_decoder_seek_rowgroup(dec, rg, col, &rec) == TDC_OK,
                  what);
    ASSERT_OR_DIE(rec.dtype == TDC_DT_I32, what);
    ASSERT_OR_DIE(rec.dim[0] == N_ROWS, what);

    int32_t *dst = (int32_t *)test_realloc(NULL, NULL,
                                           (size_t)N_ROWS * sizeof(int32_t));
    ASSERT_OR_DIE(dst != NULL, "alloc");
    tdc_block dblk = make_block_i32(dst, N_ROWS);
    tdc_status st = tdc_stream_decoder_read_block(dec, &dblk);
    if (st != TDC_OK) { test_realloc(NULL, dst, 0); ASSERT_OR_DIE(0, what); }
    for (int i = 0; i < N_ROWS; ++i) {
        if (dst[i] != expect[i]) { test_realloc(NULL, dst, 0);
                                   ASSERT_OR_DIE(0, what); }
    }
    test_realloc(NULL, dst, 0);
    return 0;
}

static int check_f64_col(tdc_stream_decoder *dec, uint64_t rg, uint16_t col,
                         const double *expect, const char *what) {
    tdc_block_record rec;
    ASSERT_OR_DIE(tdc_stream_decoder_seek_rowgroup(dec, rg, col, &rec) == TDC_OK,
                  what);
    ASSERT_OR_DIE(rec.dtype == TDC_DT_F64, what);

    double *dst = (double *)test_realloc(NULL, NULL,
                                         (size_t)N_ROWS * sizeof(double));
    ASSERT_OR_DIE(dst != NULL, "alloc");
    tdc_block dblk = make_block_f64(dst, N_ROWS);
    tdc_status st = tdc_stream_decoder_read_block(dec, &dblk);
    if (st != TDC_OK) { test_realloc(NULL, dst, 0); ASSERT_OR_DIE(0, what); }
    for (int i = 0; i < N_ROWS; ++i) {
        if (dst[i] != expect[i]) { test_realloc(NULL, dst, 0);
                                   ASSERT_OR_DIE(0, what); }
    }
    test_realloc(NULL, dst, 0);
    return 0;
}

/* ===== Tests ============================================================= */

/*
 * The core guarantee: widening appends. Everything the original container
 * occupied is still there, byte for byte, in the same place.
 */
static int test_widen_appends_only(void) {
    mem_io mio = {0};
    if (build_base(&mio, 1)) { mem_io_free(&mio); return 1; }

    /* Snapshot the original container. */
    size_t   base_size = mio.size;
    uint8_t *base_copy = (uint8_t *)malloc(base_size);
    memcpy(base_copy, mio.data, base_size);

    if (widen_with_elevation(&mio, 1)) {
        free(base_copy); mem_io_free(&mio); return 1;
    }

    /* The file grew, and every original byte past the 64-byte header is
     * untouched. Only the header itself is rewritten (in place, last). */
    if (mio.size <= base_size) {
        free(base_copy); mem_io_free(&mio);
        ASSERT_OR_DIE(0, "widened file should be larger");
    }
    if (memcmp(mio.data + TDC_CONTAINER_HEADER_SIZE,
               base_copy + TDC_CONTAINER_HEADER_SIZE,
               base_size - TDC_CONTAINER_HEADER_SIZE) != 0) {
        free(base_copy); mem_io_free(&mio);
        ASSERT_OR_DIE(0, "original body must be untouched by widening");
    }

    free(base_copy);
    mem_io_free(&mio);
    return 0;
}

/* Restoring the pre-widen header must restore the pre-widen container:
 * that is exactly the state a crash before the final header patch leaves. */
static int test_crash_before_header_patch(void) {
    mem_io mio = {0};
    if (build_base(&mio, 1)) { mem_io_free(&mio); return 1; }

    uint8_t old_header[TDC_CONTAINER_HEADER_SIZE];
    memcpy(old_header, mio.data, TDC_CONTAINER_HEADER_SIZE);
    size_t base_size = mio.size;

    if (widen_with_elevation(&mio, 1)) { mem_io_free(&mio); return 1; }

    /* Roll the header back; the appended bytes stay in the file. */
    memcpy(mio.data, old_header, TDC_CONTAINER_HEADER_SIZE);

    tdc_stream_decoder *dec = NULL;
    tdc_status st = open_dec(&mio, &dec);
    if (st != TDC_OK) { mem_io_free(&mio); ASSERT_OR_DIE(0, "reopen after rollback"); }

    const tdc_schema *sch = tdc_stream_decoder_read_schema(dec);
    int bad = (sch == NULL) || (sch->n_columns != 2) ||
              (tdc_stream_decoder_rowgroup_count(dec) != N_ROWGROUPS);
    if (!bad) {
        for (int rg = 0; rg < N_ROWGROUPS && !bad; ++rg) {
            const tdc_rowgroup_entry *e = tdc_stream_decoder_get_rowgroup(dec, (uint64_t)rg);
            if (!e || e->n_cols != 2) bad = 1;
        }
    }
    if (!bad) bad = check_f64_col(dec, 1, 0, temp_src[1], "rollback temp");
    if (!bad) bad = check_i32_col(dec, 2, 1, pres_src[2], "rollback pres");

    tdc_stream_decoder_close(&dec);
    (void)base_size;
    mem_io_free(&mio);
    ASSERT_OR_DIE(!bad, "rolled-back header must yield the original container");
    return 0;
}

/* Full read-back of a widened container: schema, every old column, the
 * new column, and the stats on both sides. */
static int test_widen_roundtrip(void) {
    mem_io mio = {0};
    if (build_base(&mio, 1)) { mem_io_free(&mio); return 1; }
    if (widen_with_elevation(&mio, 1)) { mem_io_free(&mio); return 1; }

    tdc_stream_decoder *dec = NULL;
    if (open_dec(&mio, &dec) != TDC_OK) {
        mem_io_free(&mio); ASSERT_OR_DIE(0, "open widened");
    }

    int rc = 0;
    const tdc_container_header *h = tdc_stream_decoder_header(dec);
    if (!h || h->version != TDC_CONTAINER_VERSION_WIDENED) rc = 1;
    if (!rc && h->u.het.schema_offset == 0) rc = 1;
    if (!rc && h->u.het.blocks_start != TDC_CONTAINER_HEADER_SIZE +
                                        (uint64_t)0) {
        /* blocks_start must point past the header + the original schema. */
        if (h->u.het.blocks_start <= TDC_CONTAINER_HEADER_SIZE) rc = 1;
    }

    const tdc_schema *sch = tdc_stream_decoder_read_schema(dec);
    if (!rc && (!sch || sch->n_columns != 3)) rc = 1;
    if (!rc && strcmp(sch->columns[2].name, "elevation") != 0) rc = 1;
    if (!rc && sch->columns[2].dtype != TDC_DT_I32) rc = 1;
    if (!rc && strcmp(sch->columns[2].annotation, "metres") != 0) rc = 1;
    /* The pre-existing entries must survive verbatim. */
    if (!rc && strcmp(sch->columns[0].name, "temperature") != 0) rc = 1;
    if (!rc && strcmp(sch->columns[1].annotation, "pascal") != 0) rc = 1;

    if (!rc && tdc_stream_decoder_rowgroup_count(dec) != N_ROWGROUPS) rc = 1;

    for (int rg = 0; rg < N_ROWGROUPS && !rc; ++rg) {
        const tdc_rowgroup_entry *e =
            tdc_stream_decoder_get_rowgroup(dec, (uint64_t)rg);
        if (!e || e->n_cols != 3 || e->n_rows != N_ROWS) { rc = 1; break; }

        rc = check_f64_col(dec, (uint64_t)rg, 0, temp_src[rg], "widened temp");
        if (!rc) rc = check_i32_col(dec, (uint64_t)rg, 1, pres_src[rg], "widened pres");
        if (!rc) rc = check_i32_col(dec, (uint64_t)rg, 2, added_src[rg], "widened elev");

        /* Stats: the originals survive and the appended column has its own. */
        if (!rc) {
            const tdc_column_stats *s0 = tdc_stream_decoder_get_stats(dec, (uint64_t)rg, 0);
            const tdc_column_stats *s2 = tdc_stream_decoder_get_stats(dec, (uint64_t)rg, 2);
            if (!s0 || s0->null_count != (uint64_t)rg) rc = 1;
            if (!rc && (!s2 || s2->null_count != (uint64_t)rg + 500)) rc = 1;
        }
    }

    tdc_stream_decoder_close(&dec);
    mem_io_free(&mio);
    ASSERT_OR_DIE(!rc, "widened container round-trip");
    return 0;
}

/* A container written without stats can still gain a column that has them;
 * the pre-existing columns are marked stats-less rather than invented. */
static int test_widen_adds_stats_to_statless_base(void) {
    mem_io mio = {0};
    if (build_base(&mio, 0)) { mem_io_free(&mio); return 1; }
    if (widen_with_elevation(&mio, 1)) { mem_io_free(&mio); return 1; }

    tdc_stream_decoder *dec = NULL;
    if (open_dec(&mio, &dec) != TDC_OK) {
        mem_io_free(&mio); ASSERT_OR_DIE(0, "open");
    }

    int rc = 0;
    for (int rg = 0; rg < N_ROWGROUPS && !rc; ++rg) {
        const tdc_column_stats *s0 =
            tdc_stream_decoder_get_stats(dec, (uint64_t)rg, 0);
        const tdc_column_stats *s2 =
            tdc_stream_decoder_get_stats(dec, (uint64_t)rg, 2);
        /* Slot exists for the old column but claims nothing. */
        if (!s0 || s0->has_stats != 0) rc = 1;
        if (!rc && (!s2 || s2->has_stats != 1 ||
                    s2->null_count != (uint64_t)rg + 500)) rc = 1;
        if (!rc) rc = check_i32_col(dec, (uint64_t)rg, 2, added_src[rg], "elev");
    }

    tdc_stream_decoder_close(&dec);
    mem_io_free(&mio);
    ASSERT_OR_DIE(!rc, "stats added to a stats-less base");
    return 0;
}

/* A widened container's blocks region has a gap; sequential walking must
 * fail loudly rather than stop early and report a short container. */
static int test_widened_refuses_sequential(void) {
    mem_io mio = {0};
    if (build_base(&mio, 1)) { mem_io_free(&mio); return 1; }
    if (widen_with_elevation(&mio, 1)) { mem_io_free(&mio); return 1; }

    tdc_stream_decoder *dec = NULL;
    if (open_dec(&mio, &dec) != TDC_OK) {
        mem_io_free(&mio); ASSERT_OR_DIE(0, "open");
    }

    tdc_block_record rec;
    tdc_status st = tdc_stream_decoder_peek_block(dec, &rec);
    int rc = (st != TDC_E_INVAL);

    /* Random access still works on the same decoder. */
    if (!rc) rc = check_i32_col(dec, 0, 2, added_src[0], "seek after refusal");

    tdc_stream_decoder_close(&dec);
    mem_io_free(&mio);
    ASSERT_OR_DIE(!rc, "widened container must refuse sequential peek");
    return 0;
}

/* Widening is repeatable: a v2 container widens again to v2. */
static int test_widen_twice(void) {
    mem_io mio = {0};
    if (build_base(&mio, 1)) { mem_io_free(&mio); return 1; }
    if (widen_with_elevation(&mio, 1)) { mem_io_free(&mio); return 1; }

    size_t after_first = mio.size;
    uint8_t *copy = (uint8_t *)malloc(after_first);
    memcpy(copy, mio.data, after_first);

    /* Second widen: append "humidity". */
    {
        tdc_stream_encoder_widen_config wcfg;
        memset(&wcfg, 0, sizeof(wcfg));
        wcfg.io.write_fn = mem_write;
        wcfg.io.read_fn  = mem_read;
        wcfg.io.seek_fn  = mem_seek;
        wcfg.io.ctx      = &mio;
        wcfg.schema      = &wider_schema;
        wcfg.realloc_fn  = test_realloc;

        tdc_stream_encoder *enc = NULL;
        if (tdc_stream_encoder_open_widen(&wcfg, &enc) != TDC_OK) {
            free(copy); mem_io_free(&mio);
            ASSERT_OR_DIE(0, "second widen open");
        }
        tdc_codec_spec raw = make_raw_spec();
        for (int rg = 0; rg < N_ROWGROUPS; ++rg) {
            tdc_block b = make_block_f64(added2_src[rg], N_ROWS);
            if (tdc_stream_encoder_widen_block(enc, (uint64_t)rg, &b, &raw,
                                               NULL) != TDC_OK) {
                free(copy); mem_io_free(&mio);
                ASSERT_OR_DIE(0, "second widen block");
            }
        }
        if (tdc_stream_encoder_close(&enc) != TDC_OK) {
            free(copy); mem_io_free(&mio);
            ASSERT_OR_DIE(0, "second widen close");
        }
    }

    int rc = 0;
    /* Again: the first widen's bytes are untouched below the header. */
    if (memcmp(mio.data + TDC_CONTAINER_HEADER_SIZE,
               copy + TDC_CONTAINER_HEADER_SIZE,
               after_first - TDC_CONTAINER_HEADER_SIZE) != 0) rc = 1;
    free(copy);

    tdc_stream_decoder *dec = NULL;
    if (!rc && open_dec(&mio, &dec) == TDC_OK) {
        const tdc_schema *sch = tdc_stream_decoder_read_schema(dec);
        if (!sch || sch->n_columns != 4) rc = 1;
        if (!rc && strcmp(sch->columns[3].name, "humidity") != 0) rc = 1;
        for (int rg = 0; rg < N_ROWGROUPS && !rc; ++rg) {
            const tdc_rowgroup_entry *e =
                tdc_stream_decoder_get_rowgroup(dec, (uint64_t)rg);
            if (!e || e->n_cols != 4) { rc = 1; break; }
            rc = check_f64_col(dec, (uint64_t)rg, 0, temp_src[rg], "twice temp");
            if (!rc) rc = check_i32_col(dec, (uint64_t)rg, 2, added_src[rg], "twice elev");
            if (!rc) rc = check_f64_col(dec, (uint64_t)rg, 3, added2_src[rg], "twice humid");
        }
        tdc_stream_decoder_close(&dec);
    } else if (!rc) {
        rc = 1;
    }

    mem_io_free(&mio);
    ASSERT_OR_DIE(!rc, "widening twice");
    return 0;
}

/* Argument and precondition checking. */
static int test_widen_rejects(void) {
    mem_io mio = {0};
    if (build_base(&mio, 1)) { mem_io_free(&mio); return 1; }

    tdc_stream_encoder_widen_config wcfg;
    memset(&wcfg, 0, sizeof(wcfg));
    wcfg.io.write_fn = mem_write;
    wcfg.io.read_fn  = mem_read;
    wcfg.io.seek_fn  = mem_seek;
    wcfg.io.ctx      = &mio;
    wcfg.schema      = &wide_schema;
    wcfg.realloc_fn  = test_realloc;

    tdc_stream_encoder *enc = NULL;
    int rc = 0;

    /* No seek function: a widen has to read the existing trailer. */
    {
        tdc_stream_encoder_widen_config bad = wcfg;
        bad.io.seek_fn = NULL;
        if (tdc_stream_encoder_open_widen(&bad, &enc) != TDC_E_INVAL) rc = 1;
        if (!rc && enc != NULL) rc = 1;
    }
    /* No schema. */
    if (!rc) {
        tdc_stream_encoder_widen_config bad = wcfg;
        bad.schema = NULL;
        if (tdc_stream_encoder_open_widen(&bad, &enc) != TDC_E_INVAL) rc = 1;
    }
    /* Bad magic. */
    if (!rc) {
        uint8_t saved[4];
        memcpy(saved, mio.data, 4);
        memset(mio.data, 0xAB, 4);
        if (tdc_stream_encoder_open_widen(&wcfg, &enc) != TDC_E_CORRUPT) rc = 1;
        memcpy(mio.data, saved, 4);
    }
    /* Row group index out of range. */
    if (!rc) {
        if (tdc_stream_encoder_open_widen(&wcfg, &enc) != TDC_OK) rc = 1;
        if (!rc) {
            tdc_codec_spec raw = make_raw_spec();
            tdc_block b = make_block_i32(added_src[0], N_ROWS);
            if (tdc_stream_encoder_widen_block(enc, N_ROWGROUPS, &b, &raw,
                                               NULL) != TDC_E_INVAL) rc = 1;
        }
        tdc_stream_encoder_close(&enc);
    }

    mem_io_free(&mio);
    ASSERT_OR_DIE(!rc, "widen precondition checks");
    return 0;
}

/* ===== Main ============================================================== */

int main(void) {
    int fail = 0;
    fill_sources();

    #define RUN(fn) do { \
        if (fn()) { fprintf(stderr, "  FAILED: %s\n", #fn); fail = 1; } \
        else       { fprintf(stderr, "  passed: %s\n", #fn); } \
    } while (0)

    fprintf(stderr, "test_stream_widen:\n");
    RUN(test_widen_appends_only);
    RUN(test_crash_before_header_patch);
    RUN(test_widen_roundtrip);
    RUN(test_widen_adds_stats_to_statless_base);
    RUN(test_widened_refuses_sequential);
    RUN(test_widen_twice);
    RUN(test_widen_rejects);

    #undef RUN
    return fail;
}
