/*
 * tests/test_stream_extend.c
 *
 * Tests for tdc_stream_encoder_open_extend: appending ROW GROUPS to an
 * existing container without reading or rewriting its body.
 *
 * What is actually asserted, beyond "the data comes back":
 *
 *   - the original container's bytes up to its index are a strict PREFIX of
 *     the extended file -- no existing block is moved, rewritten, or even
 *     read -- and the first appended block starts where that index started;
 *   - the index entries describing the pre-existing row groups are carried
 *     over verbatim: same offsets, same row counts, same stats;
 *   - the appended row groups decode to their own values;
 *   - a process that dies at any write of an extend pass leaves a file that
 *     reads either as the container before the pass or as the one after it;
 *   - any number of extend passes yields the same container bytes as one pass
 *     writing the same row groups: the superseded indexes are reused, not
 *     left behind;
 *   - an extended container refuses sequential walking rather than reading
 *     the superseded index sitting in the gap as a block record;
 *   - extending an extended container works, repeatedly, and each pass
 *     leaves the earlier row groups where they were;
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
    /* When armed, writes past the first `budget` fail without touching the
     * buffer: the state a process that died at that write leaves behind. */
    int      armed;
    size_t   budget;
} mem_io;

static void mem_io_free(mem_io *m) {
    free(m->data);
    memset(m, 0, sizeof(*m));
}

static tdc_status mem_write(void *ctx, const void *data, size_t size) {
    mem_io *m = (mem_io *)ctx;
    if (m->armed) {
        if (m->budget == 0) return TDC_E_IO;
        m->budget--;
    }
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
#define MAX_RGS      16

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

/* Write the 2-column container holding row groups [0, n_rgs) in one pass. */
static int build_n(mem_io *mio, int n_rgs, int with_stats) {
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
    for (int rg = 0; rg < n_rgs; ++rg) {
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

/* Write the 2-column container holding row groups [0, BASE_RGS). */
static int build_base(mem_io *mio, int with_stats) {
    return build_n(mio, BASE_RGS, with_stats);
}

static tdc_container_header read_header(const mem_io *mio) {
    tdc_container_header h;
    memcpy(&h, mio->data, TDC_CONTAINER_HEADER_SIZE);
    return h;
}

/* Append row groups [first_rg, first_rg + count) to an existing 2-column
 * container. */
static int extend_n(mem_io *mio, int first_rg, int count, int with_stats) {
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
    for (int k = 0; k < count; ++k) {
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

/* Append row groups [first_rg, first_rg + ADDED_RGS). */
static int extend_by(mem_io *mio, int first_rg, int with_stats) {
    return extend_n(mio, first_rg, ADDED_RGS, with_stats);
}

/* Whether the container in a holds the same bytes as the one in b from the
 * end of the header to the end of the index, and b ends there. */
static int same_container_body(const mem_io *a, const mem_io *b) {
    tdc_container_header ha = read_header(a);
    tdc_container_header hb = read_header(b);
    uint64_t end_a = ha.index_offset + ha.index_size;
    uint64_t end_b = hb.index_offset + hb.index_size;
    return end_a == end_b && end_b == b->size &&
           memcmp(a->data + TDC_CONTAINER_HEADER_SIZE,
                  b->data + TDC_CONTAINER_HEADER_SIZE,
                  (size_t)end_b - TDC_CONTAINER_HEADER_SIZE) == 0;
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
 * The core guarantee: extending leaves every existing block where it is.
 * Everything before the original index is still there, byte for byte, and
 * the appended blocks start where that index started.
 */
static int test_extend_keeps_body(void) {
    mem_io mio = {0};
    if (build_base(&mio, 1)) { mem_io_free(&mio); return 1; }

    uint64_t base_index = read_header(&mio).index_offset;
    uint8_t *base_copy  = (uint8_t *)malloc((size_t)base_index);
    memcpy(base_copy, mio.data, (size_t)base_index);

    if (extend_by(&mio, BASE_RGS, 1)) {
        free(base_copy); mem_io_free(&mio); return 1;
    }

    int rc = memcmp(mio.data + TDC_CONTAINER_HEADER_SIZE,
                    base_copy + TDC_CONTAINER_HEADER_SIZE,
                    (size_t)base_index - TDC_CONTAINER_HEADER_SIZE) != 0;
    free(base_copy);

    tdc_stream_decoder *dec = NULL;
    if (!rc && open_dec(&mio, &dec) == TDC_OK) {
        const tdc_rowgroup_entry *e =
            tdc_stream_decoder_get_rowgroup(dec, BASE_RGS);
        if (!e || e->columns[0].block_offset != base_index) rc = 1;
        tdc_stream_decoder_close(&dec);
    } else {
        rc = 1;
    }

    mem_io_free(&mio);
    ASSERT_OR_DIE(!rc, "body kept, appended blocks start at the old index");
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

/*
 * A process can die at any write. Run an extend pass that stops at write k,
 * for every k until the pass completes, and require the file each time to
 * read as either the container before the pass or the one after it. The
 * pass appends more than the base index holds, so the original index is
 * moved ahead of the write cursor several times along the way. A pass run
 * again over the crashed file writes over what the crashed one left: the
 * result matches a single-pass container byte for byte.
 */
static int test_crash_at_every_write(void) {
    mem_io base = {0};
    if (build_base(&base, 1)) { mem_io_free(&base); return 1; }
    const int added = 8;
    mem_io once = {0};
    if (build_n(&once, BASE_RGS + added, 1)) {
        mem_io_free(&base); mem_io_free(&once); return 1;
    }

    int completed = 0;
    for (size_t k = 0; !completed && k < 10000; ++k) {
        mem_io mio = {0};
        mio.data     = (uint8_t *)malloc(base.size);
        mio.size     = base.size;
        mio.capacity = base.size;
        memcpy(mio.data, base.data, base.size);
        mio.armed  = 1;
        mio.budget = k;

        tdc_stream_encoder_extend_config xcfg;
        memset(&xcfg, 0, sizeof(xcfg));
        xcfg.io.write_fn = mem_write;
        xcfg.io.read_fn  = mem_read;
        xcfg.io.seek_fn  = mem_seek;
        xcfg.io.ctx      = &mio;
        xcfg.realloc_fn  = test_realloc;

        tdc_stream_encoder *enc = NULL;
        tdc_status st = tdc_stream_encoder_open_extend(&xcfg, &enc);
        tdc_codec_spec raw = make_raw_spec();
        for (int rg = BASE_RGS; st == TDC_OK && rg < BASE_RGS + added; ++rg) {
            tdc_block b0 = make_block_f64(temp_src[rg], N_ROWS);
            tdc_block b1 = make_block_i32(pres_src[rg], N_ROWS);
            tdc_column_stats s[2];
            fill_stats(s, 2, rg);
            st = tdc_stream_encoder_write_block(enc, &b0, &raw);
            if (st == TDC_OK) st = tdc_stream_encoder_write_block(enc, &b1, &raw);
            if (st == TDC_OK) st = tdc_stream_encoder_set_rowgroup_stats(enc, s, 2);
            if (st == TDC_OK) st = tdc_stream_encoder_end_rowgroup(enc, N_ROWS);
        }
        /* After a failed write every later write fails too, so tearing the
         * encoder down leaves the file as a dead process would. */
        if (st == TDC_OK) st = tdc_stream_encoder_close(&enc);
        else              tdc_stream_encoder_abort(&enc);
        completed = (st == TDC_OK);
        mio.armed = 0;

        tdc_stream_decoder *dec = NULL;
        uint64_t n = 0;
        if (open_dec(&mio, &dec) == TDC_OK) {
            n = tdc_stream_decoder_rowgroup_count(dec);
            tdc_stream_decoder_close(&dec);
        }
        int rc = (n != (uint64_t)BASE_RGS && n != (uint64_t)(BASE_RGS + added));
        if (!rc && completed && n != (uint64_t)(BASE_RGS + added)) rc = 1;
        if (!rc) rc = check_all_rowgroups(&mio, (int)n, 1, "state after a crash");
        if (!rc && n == (uint64_t)BASE_RGS) {
            rc = extend_n(&mio, BASE_RGS, added, 1);
            if (!rc) rc = !same_container_body(&mio, &once);
        }
        mem_io_free(&mio);
        if (rc) {
            fprintf(stderr, "  crash after %d writes\n", (int)k);
            mem_io_free(&base);
            mem_io_free(&once);
            ASSERT_OR_DIE(0, "a crash must leave the old or the new container");
        }
    }

    mem_io_free(&base);
    mem_io_free(&once);
    ASSERT_OR_DIE(completed, "the pass completes once writes stop failing");
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

    size_t   after_base = (size_t)read_header(&mio).index_offset;
    uint8_t *snapshot   = (uint8_t *)malloc(after_base);
    memcpy(snapshot, mio.data, after_base);

    for (int pass = 0; pass < 3; ++pass) {
        if (extend_by(&mio, BASE_RGS + pass * ADDED_RGS, 1)) {
            free(snapshot); mem_io_free(&mio); return 1;
        }
    }

    /* The base container's blocks have never moved, three reopens later. */
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

/*
 * The size claim: a container grown by many extend passes holds one index,
 * not one per pass. Its bytes from the schema to the end of the index are
 * identical to those of a container that wrote the same row groups in one
 * pass; only the header differs, by the grown-in-place stamp.
 */
static int test_extend_matches_single_pass(void) {
    const int passes = (MAX_RGS - BASE_RGS) / ADDED_RGS;
    const int total  = BASE_RGS + passes * ADDED_RGS;

    mem_io grown = {0};
    if (build_base(&grown, 1)) { mem_io_free(&grown); return 1; }
    for (int p = 0; p < passes; ++p) {
        if (extend_by(&grown, BASE_RGS + p * ADDED_RGS, 1)) {
            mem_io_free(&grown); return 1;
        }
    }

    mem_io once = {0};
    if (build_n(&once, total, 1)) {
        mem_io_free(&grown); mem_io_free(&once); return 1;
    }

    int rc = !same_container_body(&grown, &once);
    if (!rc) rc = check_all_rowgroups(&grown, total, 1, "grown container");

    mem_io_free(&grown);
    mem_io_free(&once);
    ASSERT_OR_DIE(!rc, "grown container must match a single-pass container");
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
    RUN(test_extend_keeps_body);
    RUN(test_existing_index_entries_survive);
    RUN(test_crash_at_every_write);
    RUN(test_extend_roundtrip);
    RUN(test_extend_statless_base);
    RUN(test_extended_refuses_sequential);
    RUN(test_extend_repeatedly);
    RUN(test_extend_widened_container);
    RUN(test_extend_matches_single_pass);
    RUN(test_extend_abort);
    RUN(test_extend_rejects);

    #undef RUN
    return fail;
}
