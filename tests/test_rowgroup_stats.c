/*
 * tests/test_rowgroup_stats.c
 *
 * Round-trip test for per-row-group column stats.
 *
 * Scenario:
 *   - 2 row groups x 3 columns (f64, i32, u8), RAW + no transforms + NONE
 *   - Row group 0: supplies stats for all 3 columns (null_count > 0 on col 2)
 *   - Row group 1: supplies NO stats (set_rowgroup_stats never called)
 *   - Verify TDC_CONTAINER_FLAG_HAS_STATS is set on the container header
 *   - Verify tdc_stream_decoder_get_stats returns the right values for
 *     (rg=0, col=*) and NULL for (rg=1, col=*)
 *   - Additional negative: out-of-range rg / col return NULL
 *
 * Then a second pass:
 *   - Encode a container where NO row group supplies stats
 *   - Verify the HAS_STATS flag is NOT set and get_stats returns NULL
 *     for every (rg, col) query
 */

#include "tdc.h"
#include "tdc/stream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ===== in-memory I/O (same shape as test_stream_roundtrip) =============== */

typedef struct {
    uint8_t *data;
    size_t   size;
    size_t   capacity;
    size_t   cursor;
} mem_io;

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

static void mem_io_free(mem_io *m) {
    free(m->data);
    memset(m, 0, sizeof(*m));
}

/* ===== allocator ========================================================= */

static void *test_realloc(void *user, void *ptr, size_t new_size) {
    (void)user;
    if (new_size == 0) { free(ptr); return NULL; }
    return realloc(ptr, new_size);
}

/* ===== check macro ======================================================= */

#define CHECK(cond, msg) do {                                               \
    if (!(cond)) {                                                          \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);   \
        return 1;                                                           \
    }                                                                       \
} while (0)

/* ===== block builders ==================================================== */

#define N_ROWS 32

static tdc_block make_block_f64(double *data, int64_t n) {
    tdc_block b;
    memset(&b, 0, sizeof(b));
    b.data   = data;
    b.dtype  = TDC_DT_F64;
    b.layout = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank   = 1;
    b.shape.dim[0] = n;
    tdc_shape_set_contiguous(&b.shape);
    return b;
}

static tdc_block make_block_i32(int32_t *data, int64_t n) {
    tdc_block b;
    memset(&b, 0, sizeof(b));
    b.data   = data;
    b.dtype  = TDC_DT_I32;
    b.layout = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank   = 1;
    b.shape.dim[0] = n;
    tdc_shape_set_contiguous(&b.shape);
    return b;
}

static tdc_block make_block_u8(uint8_t *data, int64_t n) {
    tdc_block b;
    memset(&b, 0, sizeof(b));
    b.data   = data;
    b.dtype  = TDC_DT_U8;
    b.layout = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank   = 1;
    b.shape.dim[0] = n;
    tdc_shape_set_contiguous(&b.shape);
    return b;
}

/* ===== typed helpers to build stats slots =============================== */
/*
 * The stats min/max slots are 16 bytes, dtype-sized value at the start,
 * zero-padded. We build them by hand here to avoid depending on the
 * internal tdc_stats_store_value helper — the caller of the public API
 * owns the stats payload and is responsible for its encoding.
 */

static void put_f64(uint8_t slot[TDC_STATS_VALUE_SIZE], double v) {
    memset(slot, 0, TDC_STATS_VALUE_SIZE);
    memcpy(slot, &v, sizeof(v));
}

static void put_i32(uint8_t slot[TDC_STATS_VALUE_SIZE], int32_t v) {
    memset(slot, 0, TDC_STATS_VALUE_SIZE);
    memcpy(slot, &v, sizeof(v));
}

static void put_u8(uint8_t slot[TDC_STATS_VALUE_SIZE], uint8_t v) {
    memset(slot, 0, TDC_STATS_VALUE_SIZE);
    slot[0] = v;
}

/* ===== scenario A: mixed (rg 0 has stats, rg 1 does not) ================= */

static int test_mixed_stats(void) {
    mem_io mio = {0};
    tdc_status st;

    /* ---- encode ---- */
    tdc_stream_encoder_config ecfg;
    memset(&ecfg, 0, sizeof(ecfg));
    ecfg.io.write_fn = mem_write;
    ecfg.io.seek_fn  = mem_seek;
    ecfg.io.ctx      = &mio;
    ecfg.flags       = TDC_CONTAINER_FLAG_HETEROGENEOUS;
    ecfg.realloc_fn  = test_realloc;

    tdc_stream_encoder *enc = NULL;
    st = tdc_stream_encoder_open(&ecfg, &enc);
    CHECK(st == TDC_OK, "encoder open");

    tdc_codec_spec raw;
    memset(&raw, 0, sizeof(raw));
    raw.model = TDC_MODEL_RAW;

    double  col0_rg0[N_ROWS]; for (int i = 0; i < N_ROWS; i++) col0_rg0[i] = 0.5 * i;
    int32_t col1_rg0[N_ROWS]; for (int i = 0; i < N_ROWS; i++) col1_rg0[i] = 100 + i;
    uint8_t col2_rg0[N_ROWS]; for (int i = 0; i < N_ROWS; i++) col2_rg0[i] = (uint8_t)i;

    double  col0_rg1[N_ROWS]; for (int i = 0; i < N_ROWS; i++) col0_rg1[i] = 1000.0 + i;
    int32_t col1_rg1[N_ROWS]; for (int i = 0; i < N_ROWS; i++) col1_rg1[i] = -i;
    uint8_t col2_rg1[N_ROWS]; for (int i = 0; i < N_ROWS; i++) col2_rg1[i] = (uint8_t)(i * 2);

    /* -- row group 0: write blocks, then set stats, then end -- */
    {
        tdc_block b0 = make_block_f64(col0_rg0, N_ROWS);
        tdc_block b1 = make_block_i32(col1_rg0, N_ROWS);
        tdc_block b2 = make_block_u8 (col2_rg0, N_ROWS);
        CHECK(tdc_stream_encoder_write_block(enc, &b0, &raw) == TDC_OK, "rg0 col0 write");
        CHECK(tdc_stream_encoder_write_block(enc, &b1, &raw) == TDC_OK, "rg0 col1 write");
        CHECK(tdc_stream_encoder_write_block(enc, &b2, &raw) == TDC_OK, "rg0 col2 write");

        tdc_column_stats stats[3];
        memset(stats, 0, sizeof(stats));

        stats[0].has_stats  = 1;
        put_f64(stats[0].min, 0.0);
        put_f64(stats[0].max, 0.5 * (N_ROWS - 1));
        stats[0].null_count = 0;

        stats[1].has_stats  = 1;
        put_i32(stats[1].min, 100);
        put_i32(stats[1].max, 100 + (N_ROWS - 1));
        stats[1].null_count = 3;  /* pretend 3 nulls observed on col 1 */

        stats[2].has_stats  = 1;
        put_u8(stats[2].min, 0);
        put_u8(stats[2].max, (uint8_t)(N_ROWS - 1));
        stats[2].null_count = 7;  /* pretend 7 nulls on col 2 */

        CHECK(tdc_stream_encoder_set_rowgroup_stats(enc, stats, 3) == TDC_OK,
              "set rg0 stats");
        CHECK(tdc_stream_encoder_end_rowgroup(enc, (uint64_t)N_ROWS) == TDC_OK,
              "end rg0");
    }

    /* -- row group 1: write blocks, NO set_rowgroup_stats, end -- */
    {
        tdc_block b0 = make_block_f64(col0_rg1, N_ROWS);
        tdc_block b1 = make_block_i32(col1_rg1, N_ROWS);
        tdc_block b2 = make_block_u8 (col2_rg1, N_ROWS);
        CHECK(tdc_stream_encoder_write_block(enc, &b0, &raw) == TDC_OK, "rg1 col0 write");
        CHECK(tdc_stream_encoder_write_block(enc, &b1, &raw) == TDC_OK, "rg1 col1 write");
        CHECK(tdc_stream_encoder_write_block(enc, &b2, &raw) == TDC_OK, "rg1 col2 write");
        CHECK(tdc_stream_encoder_end_rowgroup(enc, (uint64_t)N_ROWS) == TDC_OK,
              "end rg1");
    }

    CHECK(tdc_stream_encoder_close(&enc) == TDC_OK, "encoder close");

    /* ---- decode ---- */
    mio.cursor = 0;

    tdc_stream_decoder_config dcfg;
    memset(&dcfg, 0, sizeof(dcfg));
    dcfg.io.read_fn = mem_read;
    dcfg.io.seek_fn = mem_seek;
    dcfg.io.ctx     = &mio;
    dcfg.realloc_fn = test_realloc;

    tdc_stream_decoder *dec = NULL;
    CHECK(tdc_stream_decoder_open(&dcfg, &dec) == TDC_OK, "decoder open");

    /* Header flag must be set: at least one row group had stats. */
    const tdc_container_header *hdr = tdc_stream_decoder_header(dec);
    CHECK(hdr != NULL, "header non-null");
    CHECK((hdr->flags & TDC_CONTAINER_FLAG_HAS_STATS) != 0,
          "HAS_STATS flag set");

    /* rg 0: all three columns should have stats. */
    {
        const tdc_column_stats *s0 = tdc_stream_decoder_get_stats(dec, 0, 0);
        CHECK(s0 != NULL, "rg0 col0 stats present");
        CHECK(s0->has_stats == 1, "rg0 col0 has_stats");
        double min_f64, max_f64;
        memcpy(&min_f64, s0->min, sizeof(min_f64));
        memcpy(&max_f64, s0->max, sizeof(max_f64));
        CHECK(min_f64 == 0.0, "rg0 col0 min");
        CHECK(max_f64 == 0.5 * (N_ROWS - 1), "rg0 col0 max");
        CHECK(s0->null_count == 0, "rg0 col0 null_count");

        const tdc_column_stats *s1 = tdc_stream_decoder_get_stats(dec, 0, 1);
        CHECK(s1 != NULL, "rg0 col1 stats present");
        int32_t min_i32, max_i32;
        memcpy(&min_i32, s1->min, sizeof(min_i32));
        memcpy(&max_i32, s1->max, sizeof(max_i32));
        CHECK(min_i32 == 100, "rg0 col1 min");
        CHECK(max_i32 == 100 + (N_ROWS - 1), "rg0 col1 max");
        CHECK(s1->null_count == 3, "rg0 col1 null_count");

        const tdc_column_stats *s2 = tdc_stream_decoder_get_stats(dec, 0, 2);
        CHECK(s2 != NULL, "rg0 col2 stats present");
        CHECK(s2->min[0] == 0, "rg0 col2 min");
        CHECK(s2->max[0] == (uint8_t)(N_ROWS - 1), "rg0 col2 max");
        CHECK(s2->null_count == 7, "rg0 col2 null_count");
    }

    /* rg 1: no column should have stats. */
    for (uint16_t c = 0; c < 3; c++) {
        const tdc_column_stats *s = tdc_stream_decoder_get_stats(dec, 1, c);
        CHECK(s == NULL, "rg1 stats absent");
    }

    /* Out-of-range rg / col must return NULL, not crash. */
    CHECK(tdc_stream_decoder_get_stats(dec, 99, 0) == NULL,
          "oor rg returns NULL");
    CHECK(tdc_stream_decoder_get_stats(dec, 0, 99) == NULL,
          "oor col returns NULL");

    CHECK(tdc_stream_decoder_close(&dec) == TDC_OK, "decoder close");
    mem_io_free(&mio);
    return 0;
}

/* ===== scenario B: NO row group supplies stats ========================== */

static int test_no_stats(void) {
    mem_io mio = {0};
    tdc_status st;

    tdc_stream_encoder_config ecfg;
    memset(&ecfg, 0, sizeof(ecfg));
    ecfg.io.write_fn = mem_write;
    ecfg.io.seek_fn  = mem_seek;
    ecfg.io.ctx      = &mio;
    ecfg.flags       = TDC_CONTAINER_FLAG_HETEROGENEOUS;
    ecfg.realloc_fn  = test_realloc;

    tdc_stream_encoder *enc = NULL;
    st = tdc_stream_encoder_open(&ecfg, &enc);
    CHECK(st == TDC_OK, "encoder open (no-stats scenario)");

    tdc_codec_spec raw;
    memset(&raw, 0, sizeof(raw));
    raw.model = TDC_MODEL_RAW;

    int32_t col[N_ROWS]; for (int i = 0; i < N_ROWS; i++) col[i] = i;
    tdc_block b = make_block_i32(col, N_ROWS);
    CHECK(tdc_stream_encoder_write_block(enc, &b, &raw) == TDC_OK,
          "write (no-stats) col");
    CHECK(tdc_stream_encoder_end_rowgroup(enc, (uint64_t)N_ROWS) == TDC_OK,
          "end rg (no stats)");
    CHECK(tdc_stream_encoder_close(&enc) == TDC_OK, "close (no stats)");

    mio.cursor = 0;

    tdc_stream_decoder_config dcfg;
    memset(&dcfg, 0, sizeof(dcfg));
    dcfg.io.read_fn = mem_read;
    dcfg.io.seek_fn = mem_seek;
    dcfg.io.ctx     = &mio;
    dcfg.realloc_fn = test_realloc;

    tdc_stream_decoder *dec = NULL;
    CHECK(tdc_stream_decoder_open(&dcfg, &dec) == TDC_OK,
          "decoder open (no stats)");

    const tdc_container_header *hdr = tdc_stream_decoder_header(dec);
    CHECK((hdr->flags & TDC_CONTAINER_FLAG_HAS_STATS) == 0,
          "HAS_STATS flag NOT set");

    CHECK(tdc_stream_decoder_get_stats(dec, 0, 0) == NULL,
          "get_stats returns NULL when none");

    CHECK(tdc_stream_decoder_close(&dec) == TDC_OK, "close decoder (no stats)");
    mem_io_free(&mio);
    return 0;
}

/* ===== main ============================================================= */

int main(void) {
    int fail = 0;
    if (test_mixed_stats()) { fprintf(stderr, "FAILED: test_mixed_stats\n"); fail = 1; }
    else fprintf(stderr, "passed: test_mixed_stats\n");

    if (test_no_stats())    { fprintf(stderr, "FAILED: test_no_stats\n");    fail = 1; }
    else fprintf(stderr, "passed: test_no_stats\n");

    return fail;
}
