/*
 * tests/test_stream_extend.c
 *
 * Tests for tdc_stream_encoder_open_extend: appending ROW GROUPS to an
 * existing container without reading or rewriting its body.
 *
 * What is actually asserted, beyond "the data comes back":
 *
 *   - the original container's bytes are a strict PREFIX of the extended
 *     file, which is the whole claim -- no existing block is moved,
 *     rewritten, or even read;
 *   - the index entries describing the pre-existing row groups are carried
 *     over verbatim: same offsets, same row counts, same stats;
 *   - the appended row groups decode to their own values;
 *   - restoring only the original 64-byte header turns the extended file
 *     back into the original container, which is the crash-before-header-
 *     patch state;
 *   - an extended container refuses sequential walking rather than reading
 *     the superseded index sitting in the gap as a block record;
 *   - extending an extended container works, repeatedly, and each pass
 *     leaves the earlier row groups where they were -- which is what makes
 *     building a store by appending linear rather than quadratic;
 *   - extending a WIDENED container preserves its relocated schema;
 *   - a row group whose column count disagrees with the schema is refused,
 *     rather than written to misalign every later column read.
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
#define BASE_RGS     2    /* row groups the container starts with */
#define ADDED_RGS    2    /* row groups each extend pass appends */
#define MAX_RGS      8

static double  temp_src[MAX_RGS][N_ROWS];
static int32_t pres_src[MAX_RGS][N_ROWS];
static int32_t elev_src[MAX_RGS][N_ROWS];

static void fill_sources(void) {
    for (int rg = 0; rg < MAX_RGS; ++rg) {
        for (int i = 0; i < N_ROWS; ++i) {
            int row = rg * N_ROWS + i;
            temp_src[rg][i] = 100.0 + (double)row * 0.5;
            pres_src[rg][i] = 1000 + row * 3;
            elev_src[rg][i] = -7 * row - 1;
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

/* ===== Shared helpers ==================================================== */

/* Stats are derived from the absolute row-group number, so a group keeps
 * identifiable stats wherever it ends up in the container. */
static void fill_stats(tdc_column_stats *s, int n_cols, int rg) {
    memset(s, 0, (size_t)n_cols * sizeof(tdc_column_stats));
    for (int c = 0; c < n_cols; ++c) {
        s[c].has_stats  = 1;
        s[c].null_count = (uint64_t)(rg * 10 + c);
    }
}

/* Write the 2-column container holding row groups [0, BASE_RGS). */
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
    for (int rg = 0; rg < BASE_RGS; ++rg) {
        tdc_block b0 = make_block_f64(temp_src[rg], N_ROWS);
        ASSERT_OR_DIE(tdc_stream_encoder_write_block(enc, &b0, &raw) == TDC_OK,
                      "base write temp");
        tdc_block b1 = make_block_i32(pres_src[rg], N_ROWS);
        ASSERT_OR_DIE(tdc_stream_encoder_write_block(enc, &b1, &raw) == TDC_OK,
                      "base write pres");

        if (with_stats) {
            tdc_column_stats s[2];
            fill_stats(s, 2, rg);
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

/* Append row groups [first_rg, first_rg + ADDED_RGS) to an existing
 * 2-column container. */
static int extend_by(mem_io *mio, int first_rg, int with_stats) {
    tdc_stream_encoder_extend_config xcfg;
    memset(&xcfg, 0, sizeof(xcfg));
    xcfg.io.write_fn = mem_write;
    xcfg.io.read_fn  = mem_read;
    xcfg.io.seek_fn  = mem_seek;
    xcfg.io.ctx      = mio;
    xcfg.realloc_fn  = test_realloc;

    tdc_stream_encoder *enc = NULL;
    ASSERT_OR_DIE(tdc_stream_encoder_open_extend(&xcfg, &enc) == TDC_OK,
                  "extend open");

    tdc_codec_spec raw = make_raw_spec();
    for (int k = 0; k < ADDED_RGS; ++k) {
        int rg = first_rg + k;
        tdc_block b0 = make_block_f64(temp_src[rg], N_ROWS);
        ASSERT_OR_DIE(tdc_stream_encoder_write_block(enc, &b0, &raw) == TDC_OK,
                      "extend write temp");
        tdc_block b1 = make_block_i32(pres_src[rg], N_ROWS);
        ASSERT_OR_DIE(tdc_stream_encoder_write_block(enc, &b1, &raw) == TDC_OK,
                      "extend write pres");

        if (with_stats) {
            tdc_column_stats s[2];
            fill_stats(s, 2, rg);
            ASSERT_OR_DIE(
                tdc_stream_encoder_set_rowgroup_stats(enc, s, 2) == TDC_OK,
                "extend stats");
        }
        ASSERT_OR_DIE(tdc_stream_encoder_end_rowgroup(enc, N_ROWS) == TDC_OK,
                      "extend end rg");
    }
    ASSERT_OR_DIE(tdc_stream_encoder_close(&enc) == TDC_OK, "extend close");
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

static int check_f64_col(tdc_stream_decoder *dec, uint64_t rg, uint16_t col,
                         const double *expect, const char *what) {
    tdc_block_record rec;
    ASSERT_OR_DIE(tdc_stream_decoder_seek_rowgroup(dec, rg, col, &rec) == TDC_OK,
                  what);
    ASSERT_OR_DIE(rec.dtype == TDC_DT_F64, what);
    ASSERT_OR_DIE(rec.dim[0] == N_ROWS, what);

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

/* Every row group in [0, n_rgs) holds the values it was written with. */
static int check_all_rowgroups(mem_io *mio, int n_rgs, int with_stats,
                               const char *what) {
    tdc_stream_decoder *dec = NULL;
    if (open_dec(mio, &dec) != TDC_OK) ASSERT_OR_DIE(0, what);

    int rc = 0;
    if (tdc_stream_decoder_rowgroup_count(dec) != (uint64_t)n_rgs) rc = 1;

    for (int rg = 0; rg < n_rgs && !rc; ++rg) {
        const tdc_rowgroup_entry *e =
            tdc_stream_decoder_get_rowgroup(dec, (uint64_t)rg);
        if (!e || e->n_cols != 2 || e->n_rows != N_ROWS) { rc = 1; break; }
        rc = check_f64_col(dec, (uint64_t)rg, 0, temp_src[rg], what);
        if (!rc) rc = check_i32_col(dec, (uint64_t)rg, 1, pres_src[rg], what);

        if (!rc && with_stats) {
            for (int c = 0; c < 2; ++c) {
                const tdc_column_stats *s = tdc_stream_decoder_get_stats(
                    dec, (uint64_t)rg, (uint16_t)c);
                if (!s || !s->has_stats ||
                    s->null_count != (uint64_t)(rg * 10 + c)) { rc = 1; break; }
            }
        }
    }

    tdc_stream_decoder_close(&dec);
    ASSERT_OR_DIE(!rc, what);
    return 0;
}

/* ===== Tests ============================================================= */

/*
 * The core guarantee: extending appends. Everything the original container
 * occupied is still there, byte for byte, in the same place.
 */
static int test_extend_appends_only(void) {
    mem_io mio = {0};
    if (build_base(&mio, 1)) { mem_io_free(&mio); return 1; }

    size_t   base_size = mio.size;
    uint8_t *base_copy = (uint8_t *)malloc(base_size);
    memcpy(base_copy, mio.data, base_size);

    if (extend_by(&mio, BASE_RGS, 1)) {
        free(base_copy); mem_io_free(&mio); return 1;
    }

    /* The file grew, and every original byte past the 64-byte header is
     * untouched. Only the header itself is rewritten (in place, last). */
    if (mio.size <= base_size) {
        free(base_copy); mem_io_free(&mio);
        ASSERT_OR_DIE(0, "extended file should be larger");
    }
    if (memcmp(mio.data + TDC_CONTAINER_HEADER_SIZE,
               base_copy + TDC_CONTAINER_HEADER_SIZE,
               base_size - TDC_CONTAINER_HEADER_SIZE) != 0) {
        free(base_copy); mem_io_free(&mio);
        ASSERT_OR_DIE(0, "original body must be untouched by extending");
    }

    free(base_copy);
    mem_io_free(&mio);
    return 0;
}

/*
 * The pre-existing row groups must be carried into the rebuilt index
 * unchanged -- same block offsets, same row count, same stats. That is the
 * property that lets the old blocks stay where they are.
 */
static int test_existing_index_entries_survive(void) {
    mem_io mio = {0};
    if (build_base(&mio, 1)) { mem_io_free(&mio); return 1; }

    /* Record what the base container's index says about its row groups. */
    uint64_t old_off[BASE_RGS][2];
    uint64_t old_tot[BASE_RGS][2];
    {
        tdc_stream_decoder *dec = NULL;
        if (open_dec(&mio, &dec) != TDC_OK) { mem_io_free(&mio); ASSERT_OR_DIE(0, "open base"); }
        for (int rg = 0; rg < BASE_RGS; ++rg) {
            const tdc_rowgroup_entry *e =
                tdc_stream_decoder_get_rowgroup(dec, (uint64_t)rg);
            if (!e) { tdc_stream_decoder_close(&dec); mem_io_free(&mio);
                      ASSERT_OR_DIE(0, "base rowgroup"); }
            for (int c = 0; c < 2; ++c) {
                old_off[rg][c] = e->columns[c].block_offset;
                old_tot[rg][c] = e->columns[c].block_total;
            }
        }
        tdc_stream_decoder_close(&dec);
    }

    if (extend_by(&mio, BASE_RGS, 1)) { mem_io_free(&mio); return 1; }

    int rc = 0;
    tdc_stream_decoder *dec = NULL;
    if (open_dec(&mio, &dec) != TDC_OK) { mem_io_free(&mio); ASSERT_OR_DIE(0, "open extended"); }
    for (int rg = 0; rg < BASE_RGS && !rc; ++rg) {
        const tdc_rowgroup_entry *e =
            tdc_stream_decoder_get_rowgroup(dec, (uint64_t)rg);
        if (!e || e->n_rows != N_ROWS || e->n_cols != 2) { rc = 1; break; }
        for (int c = 0; c < 2; ++c) {
            if (e->columns[c].block_offset != old_off[rg][c] ||
                e->columns[c].block_total  != old_tot[rg][c]) { rc = 1; break; }
        }
    }
    tdc_stream_decoder_close(&dec);
    mem_io_free(&mio);
    ASSERT_OR_DIE(!rc, "pre-existing index entries must be carried over verbatim");
    return 0;
}

/* Restoring the pre-extend header must restore the pre-extend container:
 * that is exactly the state a crash before the final header patch leaves. */
static int test_crash_before_header_patch(void) {
    mem_io mio = {0};
    if (build_base(&mio, 1)) { mem_io_free(&mio); return 1; }

    uint8_t old_header[TDC_CONTAINER_HEADER_SIZE];
    memcpy(old_header, mio.data, TDC_CONTAINER_HEADER_SIZE);

    if (extend_by(&mio, BASE_RGS, 1)) { mem_io_free(&mio); return 1; }

    /* Roll the header back; the appended bytes stay in the file. */
    memcpy(mio.data, old_header, TDC_CONTAINER_HEADER_SIZE);

    /* The container reads as it did before the extend: BASE_RGS row groups,
     * original schema, original values. */
    if (check_all_rowgroups(&mio, BASE_RGS, 1, "rolled-back container")) {
        mem_io_free(&mio); return 1;
    }

    tdc_stream_decoder *dec = NULL;
    if (open_dec(&mio, &dec) != TDC_OK) { mem_io_free(&mio); ASSERT_OR_DIE(0, "reopen"); }
    const tdc_schema *sch = tdc_stream_decoder_read_schema(dec);
    int bad = (sch == NULL) || (sch->n_columns != 2);
    const tdc_container_header *h = tdc_stream_decoder_header(dec);
    if (!bad && (!h || h->version != TDC_CONTAINER_VERSION)) bad = 1;
    tdc_stream_decoder_close(&dec);

    mem_io_free(&mio);
    ASSERT_OR_DIE(!bad, "rolled-back header must yield the original container");
    return 0;
}

/* Full read-back: every row group, old and new, plus the header fields the
 * extend is responsible for. */
static int test_extend_roundtrip(void) {
    mem_io mio = {0};
    if (build_base(&mio, 1)) { mem_io_free(&mio); return 1; }

    /* The base container's schema sits at offset 64 and is not moved, so the
     * blocks region starts right after it. */
    uint32_t base_schema_size;
    {
        tdc_stream_decoder *dec = NULL;
        if (open_dec(&mio, &dec) != TDC_OK) { mem_io_free(&mio); ASSERT_OR_DIE(0, "open base"); }
        base_schema_size = tdc_stream_decoder_header(dec)->schema_size;
        tdc_stream_decoder_close(&dec);
    }

    if (extend_by(&mio, BASE_RGS, 1)) { mem_io_free(&mio); return 1; }

    int rc = 0;
    tdc_stream_decoder *dec = NULL;
    if (open_dec(&mio, &dec) != TDC_OK) { mem_io_free(&mio); ASSERT_OR_DIE(0, "open extended"); }

    const tdc_container_header *h = tdc_stream_decoder_header(dec);
    /* Stamped as grown-in-place: the superseded index now sits in the blocks
     * region as a gap. */
    if (!h || h->version != TDC_CONTAINER_VERSION_WIDENED) rc = 1;
    /* The schema was NOT relocated: it is still where it was written. */
    if (!rc && h->u.het.schema_offset != TDC_CONTAINER_HEADER_SIZE) rc = 1;
    if (!rc && h->u.het.blocks_start !=
               TDC_CONTAINER_HEADER_SIZE + (uint64_t)base_schema_size) rc = 1;
    if (!rc && h->schema_size != base_schema_size) rc = 1;

    const tdc_schema *sch = tdc_stream_decoder_read_schema(dec);
    if (!rc && (!sch || sch->n_columns != 2)) rc = 1;
    if (!rc && strcmp(sch->columns[0].name, "temperature") != 0) rc = 1;
    if (!rc && strcmp(sch->columns[1].annotation, "pascal") != 0) rc = 1;

    tdc_stream_decoder_close(&dec);
    if (rc) { mem_io_free(&mio); ASSERT_OR_DIE(0, "extended header/schema"); }

    if (check_all_rowgroups(&mio, BASE_RGS + ADDED_RGS, 1, "extended values")) {
        mem_io_free(&mio); return 1;
    }
    mem_io_free(&mio);
    return 0;
}

/* A container with no stats stays statless, and its appended row groups can
 * introduce stats without the older ones inventing any. */
static int test_extend_statless_base(void) {
    mem_io mio = {0};
    if (build_base(&mio, 0)) { mem_io_free(&mio); return 1; }
    if (extend_by(&mio, BASE_RGS, 1)) { mem_io_free(&mio); return 1; }

    int rc = 0;
    tdc_stream_decoder *dec = NULL;
    if (open_dec(&mio, &dec) != TDC_OK) { mem_io_free(&mio); ASSERT_OR_DIE(0, "open"); }

    for (int rg = 0; rg < BASE_RGS && !rc; ++rg) {
        if (tdc_stream_decoder_get_stats(dec, (uint64_t)rg, 0) != NULL) rc = 1;
    }
    for (int rg = BASE_RGS; rg < BASE_RGS + ADDED_RGS && !rc; ++rg) {
        const tdc_column_stats *s =
            tdc_stream_decoder_get_stats(dec, (uint64_t)rg, 0);
        if (!s || !s->has_stats ||
            s->null_count != (uint64_t)(rg * 10)) rc = 1;
    }
    tdc_stream_decoder_close(&dec);
    if (!rc) rc = check_all_rowgroups(&mio, BASE_RGS + ADDED_RGS, 0,
                                      "statless base values");
    mem_io_free(&mio);
    ASSERT_OR_DIE(!rc, "appended row groups may add stats to a statless base");
    return 0;
}

static int test_extended_refuses_sequential(void) {
    mem_io mio = {0};
    if (build_base(&mio, 1)) { mem_io_free(&mio); return 1; }
    if (extend_by(&mio, BASE_RGS, 1)) { mem_io_free(&mio); return 1; }

    tdc_stream_decoder *dec = NULL;
    if (open_dec(&mio, &dec) != TDC_OK) { mem_io_free(&mio); ASSERT_OR_DIE(0, "open"); }

    tdc_block_record rec;
    tdc_status st = tdc_stream_decoder_peek_block(dec, &rec);
    int rc = (st != TDC_E_INVAL);

    /* Random access still works on the same decoder. */
    if (!rc) rc = check_f64_col(dec, BASE_RGS, 0, temp_src[BASE_RGS],
                                "seek after refusal");

    tdc_stream_decoder_close(&dec);
    mem_io_free(&mio);
    ASSERT_OR_DIE(!rc, "extended container must refuse sequential peek");
    return 0;
}

/*
 * The property the whole change exists for: appending again costs another
 * append, not another rewrite. Three passes, and every row group from every
 * pass still reads correctly -- including the ones written before the
 * container was ever reopened.
 */
static int test_extend_repeatedly(void) {
    mem_io mio = {0};
    if (build_base(&mio, 1)) { mem_io_free(&mio); return 1; }

    size_t   after_base = mio.size;
    uint8_t *snapshot   = (uint8_t *)malloc(after_base);
    memcpy(snapshot, mio.data, after_base);

    for (int pass = 0; pass < 3; ++pass) {
        if (extend_by(&mio, BASE_RGS + pass * ADDED_RGS, 1)) {
            free(snapshot); mem_io_free(&mio); return 1;
        }
    }

    /* Still a strict append: the bytes the base container wrote have never
     * moved, three reopens later. */
    if (memcmp(mio.data + TDC_CONTAINER_HEADER_SIZE,
               snapshot + TDC_CONTAINER_HEADER_SIZE,
               after_base - TDC_CONTAINER_HEADER_SIZE) != 0) {
        free(snapshot); mem_io_free(&mio);
        ASSERT_OR_DIE(0, "repeated extends must not move the original body");
    }
    free(snapshot);

    int rc = check_all_rowgroups(&mio, BASE_RGS + 3 * ADDED_RGS, 1,
                                 "repeatedly extended");
    mem_io_free(&mio);
    return rc;
}

/* Extending a WIDENED container keeps the schema at the tail, where widening
 * put it, rather than assuming offset 64. */
static int test_extend_widened_container(void) {
    mem_io mio = {0};
    if (build_base(&mio, 1)) { mem_io_free(&mio); return 1; }

    /* Widen to 3 columns first. */
    {
        tdc_stream_encoder_widen_config wcfg;
        memset(&wcfg, 0, sizeof(wcfg));
        wcfg.io.write_fn = mem_write;
        wcfg.io.read_fn  = mem_read;
        wcfg.io.seek_fn  = mem_seek;
        wcfg.io.ctx      = &mio;
        wcfg.schema      = &wide_schema;
        wcfg.realloc_fn  = test_realloc;

        tdc_stream_encoder *enc = NULL;
        if (tdc_stream_encoder_open_widen(&wcfg, &enc) != TDC_OK) {
            mem_io_free(&mio); ASSERT_OR_DIE(0, "widen open");
        }
        tdc_codec_spec raw = make_raw_spec();
        for (int rg = 0; rg < BASE_RGS; ++rg) {
            tdc_block b = make_block_i32(elev_src[rg], N_ROWS);
            if (tdc_stream_encoder_widen_block(enc, (uint64_t)rg, &b, &raw,
                                               NULL) != TDC_OK) {
                tdc_stream_encoder_close(&enc);
                mem_io_free(&mio); ASSERT_OR_DIE(0, "widen block");
            }
        }
        if (tdc_stream_encoder_close(&enc) != TDC_OK) {
            mem_io_free(&mio); ASSERT_OR_DIE(0, "widen close");
        }
    }

    uint64_t widened_schema_offset;
    {
        tdc_stream_decoder *dec = NULL;
        if (open_dec(&mio, &dec) != TDC_OK) { mem_io_free(&mio); ASSERT_OR_DIE(0, "open widened"); }
        widened_schema_offset = tdc_stream_decoder_header(dec)->u.het.schema_offset;
        tdc_stream_decoder_close(&dec);
    }
    /* Widening relocated the schema past the header, which is the case this
     * test exists to cover. */
    if (widened_schema_offset == TDC_CONTAINER_HEADER_SIZE) {
        mem_io_free(&mio);
        ASSERT_OR_DIE(0, "widening should have relocated the schema");
    }

    /* Now append 3-column row groups. */
    {
        tdc_stream_encoder_extend_config xcfg;
        memset(&xcfg, 0, sizeof(xcfg));
        xcfg.io.write_fn = mem_write;
        xcfg.io.read_fn  = mem_read;
        xcfg.io.seek_fn  = mem_seek;
        xcfg.io.ctx      = &mio;
        xcfg.realloc_fn  = test_realloc;

        tdc_stream_encoder *enc = NULL;
        if (tdc_stream_encoder_open_extend(&xcfg, &enc) != TDC_OK) {
            mem_io_free(&mio); ASSERT_OR_DIE(0, "extend widened open");
        }
        tdc_codec_spec raw = make_raw_spec();
        int rc = 0;
        for (int k = 0; k < ADDED_RGS && !rc; ++k) {
            int rg = BASE_RGS + k;
            tdc_block b0 = make_block_f64(temp_src[rg], N_ROWS);
            tdc_block b1 = make_block_i32(pres_src[rg], N_ROWS);
            tdc_block b2 = make_block_i32(elev_src[rg], N_ROWS);
            if (tdc_stream_encoder_write_block(enc, &b0, &raw) != TDC_OK) rc = 1;
            if (!rc && tdc_stream_encoder_write_block(enc, &b1, &raw) != TDC_OK) rc = 1;
            if (!rc && tdc_stream_encoder_write_block(enc, &b2, &raw) != TDC_OK) rc = 1;
            if (!rc && tdc_stream_encoder_end_rowgroup(enc, N_ROWS) != TDC_OK) rc = 1;
        }
        if (rc) { tdc_stream_encoder_close(&enc); mem_io_free(&mio);
                  ASSERT_OR_DIE(0, "extend widened write"); }
        if (tdc_stream_encoder_close(&enc) != TDC_OK) {
            mem_io_free(&mio); ASSERT_OR_DIE(0, "extend widened close");
        }
    }

    int rc = 0;
    tdc_stream_decoder *dec = NULL;
    if (open_dec(&mio, &dec) != TDC_OK) { mem_io_free(&mio); ASSERT_OR_DIE(0, "reopen"); }

    const tdc_container_header *h = tdc_stream_decoder_header(dec);
    if (!h || h->version != TDC_CONTAINER_VERSION_WIDENED) rc = 1;
    /* The relocated schema is still the one the header points at. */
    if (!rc && h->u.het.schema_offset != widened_schema_offset) rc = 1;

    const tdc_schema *sch = tdc_stream_decoder_read_schema(dec);
    if (!rc && (!sch || sch->n_columns != 3)) rc = 1;
    if (!rc && strcmp(sch->columns[2].name, "elevation") != 0) rc = 1;
    if (!rc && tdc_stream_decoder_rowgroup_count(dec) !=
               (uint64_t)(BASE_RGS + ADDED_RGS)) rc = 1;

    for (int rg = 0; rg < BASE_RGS + ADDED_RGS && !rc; ++rg) {
        rc = check_f64_col(dec, (uint64_t)rg, 0, temp_src[rg], "wide temp");
        if (!rc) rc = check_i32_col(dec, (uint64_t)rg, 1, pres_src[rg], "wide pres");
        if (!rc) rc = check_i32_col(dec, (uint64_t)rg, 2, elev_src[rg], "wide elev");
    }

    tdc_stream_decoder_close(&dec);
    mem_io_free(&mio);
    ASSERT_OR_DIE(!rc, "extending a widened container");
    return 0;
}

/* Aborting leaves the container exactly as it was found. */
static int test_extend_abort(void) {
    mem_io mio = {0};
    if (build_base(&mio, 1)) { mem_io_free(&mio); return 1; }

    size_t   base_size = mio.size;
    uint8_t *base_copy = (uint8_t *)malloc(base_size);
    memcpy(base_copy, mio.data, base_size);

    tdc_stream_encoder_extend_config xcfg;
    memset(&xcfg, 0, sizeof(xcfg));
    xcfg.io.write_fn = mem_write;
    xcfg.io.read_fn  = mem_read;
    xcfg.io.seek_fn  = mem_seek;
    xcfg.io.ctx      = &mio;
    xcfg.realloc_fn  = test_realloc;

    tdc_stream_encoder *enc = NULL;
    int rc = (tdc_stream_encoder_open_extend(&xcfg, &enc) != TDC_OK);
    if (!rc) {
        tdc_codec_spec raw = make_raw_spec();
        tdc_block b0 = make_block_f64(temp_src[BASE_RGS], N_ROWS);
        tdc_block b1 = make_block_i32(pres_src[BASE_RGS], N_ROWS);
        if (tdc_stream_encoder_write_block(enc, &b0, &raw) != TDC_OK) rc = 1;
        if (!rc && tdc_stream_encoder_write_block(enc, &b1, &raw) != TDC_OK) rc = 1;
        if (!rc && tdc_stream_encoder_end_rowgroup(enc, N_ROWS) != TDC_OK) rc = 1;
        if (!rc && tdc_stream_encoder_abort(&enc) != TDC_OK) rc = 1;
        if (enc) { tdc_stream_encoder_close(&enc); rc = 1; }
    }

    /* The header was never patched, so the container still reads as it did. */
    if (!rc && memcmp(mio.data, base_copy, base_size) != 0) rc = 1;
    free(base_copy);
    if (!rc) rc = check_all_rowgroups(&mio, BASE_RGS, 1, "after abort");

    mem_io_free(&mio);
    ASSERT_OR_DIE(!rc, "aborted extend must leave the container as found");
    return 0;
}

/* Preconditions and mode guards. */
static int test_extend_rejects(void) {
    mem_io mio = {0};
    if (build_base(&mio, 1)) { mem_io_free(&mio); return 1; }

    tdc_stream_encoder_extend_config xcfg;
    memset(&xcfg, 0, sizeof(xcfg));
    xcfg.io.write_fn = mem_write;
    xcfg.io.read_fn  = mem_read;
    xcfg.io.seek_fn  = mem_seek;
    xcfg.io.ctx      = &mio;
    xcfg.realloc_fn  = test_realloc;

    tdc_stream_encoder *enc = NULL;
    int rc = 0;

    /* No read_fn: a container cannot be reopened without reading it. */
    {
        tdc_stream_encoder_extend_config bad = xcfg;
        bad.io.read_fn = NULL;
        if (tdc_stream_encoder_open_extend(&bad, &enc) != TDC_E_INVAL) rc = 1;
    }

    /* A row group must carry exactly the schema's columns: one short is
     * refused at end_rowgroup rather than written misaligned. */
    if (!rc) {
        if (tdc_stream_encoder_open_extend(&xcfg, &enc) != TDC_OK) rc = 1;
        if (!rc) {
            tdc_codec_spec raw = make_raw_spec();
            tdc_block b0 = make_block_f64(temp_src[BASE_RGS], N_ROWS);
            if (tdc_stream_encoder_write_block(enc, &b0, &raw) != TDC_OK) rc = 1;
            if (!rc && tdc_stream_encoder_end_rowgroup(enc, N_ROWS) != TDC_E_INVAL)
                rc = 1;
            /* And one too many. */
            if (!rc) {
                tdc_block b1 = make_block_i32(pres_src[BASE_RGS], N_ROWS);
                tdc_block b2 = make_block_i32(elev_src[BASE_RGS], N_ROWS);
                if (tdc_stream_encoder_write_block(enc, &b1, &raw) != TDC_OK) rc = 1;
                if (!rc && tdc_stream_encoder_write_block(enc, &b2, &raw) != TDC_OK) rc = 1;
                if (!rc && tdc_stream_encoder_end_rowgroup(enc, N_ROWS) != TDC_E_INVAL)
                    rc = 1;
            }
            /* widen_block belongs to a widening encoder, not this one. */
            if (!rc) {
                tdc_block b = make_block_i32(elev_src[0], N_ROWS);
                if (tdc_stream_encoder_widen_block(enc, 0, &b, &raw, NULL)
                    != TDC_E_INVAL) rc = 1;
            }
        }
        tdc_stream_encoder_abort(&enc);
    }

    /* The mirror guard: write_block/end_rowgroup on a widening encoder. */
    if (!rc) {
        tdc_stream_encoder_widen_config wcfg;
        memset(&wcfg, 0, sizeof(wcfg));
        wcfg.io.write_fn = mem_write;
        wcfg.io.read_fn  = mem_read;
        wcfg.io.seek_fn  = mem_seek;
        wcfg.io.ctx      = &mio;
        wcfg.schema      = &wide_schema;
        wcfg.realloc_fn  = test_realloc;

        tdc_stream_encoder *wenc = NULL;
        if (tdc_stream_encoder_open_widen(&wcfg, &wenc) != TDC_OK) rc = 1;
        if (!rc) {
            tdc_codec_spec raw = make_raw_spec();
            tdc_block b = make_block_i32(elev_src[0], N_ROWS);
            if (tdc_stream_encoder_write_block(wenc, &b, &raw) != TDC_E_INVAL) rc = 1;
            if (!rc && tdc_stream_encoder_end_rowgroup(wenc, N_ROWS) != TDC_E_INVAL)
                rc = 1;
        }
        tdc_stream_encoder_abort(&wenc);
    }

    /* Nothing above committed, so the container is untouched. */
    if (!rc) rc = check_all_rowgroups(&mio, BASE_RGS, 1, "after rejections");

    mem_io_free(&mio);
    ASSERT_OR_DIE(!rc, "extend precondition checks");
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

    fprintf(stderr, "test_stream_extend:\n");
    RUN(test_extend_appends_only);
    RUN(test_existing_index_entries_survive);
    RUN(test_crash_before_header_patch);
    RUN(test_extend_roundtrip);
    RUN(test_extend_statless_base);
    RUN(test_extended_refuses_sequential);
    RUN(test_extend_repeatedly);
    RUN(test_extend_widened_container);
    RUN(test_extend_abort);
    RUN(test_extend_rejects);

    #undef RUN
    return fail;
}
