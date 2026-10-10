/*
 * tests/test_stream_write_record.c
 *
 * Tests for tdc_stream_encoder_write_record: writing block records that
 * the caller encoded with tdc_encode_block.
 *
 * What is asserted:
 *   - a container built from pre-encoded records is byte-identical to the
 *     one write_block builds from the same blocks and specs, including
 *     when the two calls are mixed within a row group;
 *   - that container decodes to the source values;
 *   - a malformed record (too short, size disagreeing with the header's
 *     section sizes, bad magic) is refused and leaves the encoder's block
 *     count and output untouched;
 *   - NULL arguments are refused.
 */

#include "tdc.h"
#include "tdc/stream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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

static tdc_status mem_read(void *ctx, void *buf, size_t size, size_t *bytes_read) {
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

#define N_ROWS 4096
#define N_RGS  3
#define N_COLS 3

static double  col_a[N_RGS][N_ROWS];   /* smooth f64 */
static double  col_b[N_RGS][N_ROWS];   /* pseudo-random f64 */
static int32_t col_c[N_RGS][N_ROWS];   /* ramp i32 */

static void fill_sources(void) {
    uint64_t s = 42u;
    for (int rg = 0; rg < N_RGS; ++rg) {
        for (int i = 0; i < N_ROWS; ++i) {
            int row = rg * N_ROWS + i;
            col_a[rg][i] = 20.0 + 0.01 * (double)row;
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            col_b[rg][i] = (double)(s >> 11) * (1.0 / 9007199254740992.0);
            col_c[rg][i] = 3 * row - 1000;
        }
    }
}

static tdc_block make_block(void *data, tdc_dtype dt) {
    tdc_block b;
    memset(&b, 0, sizeof(b));
    b.data = data;
    b.dtype = dt;
    b.layout = TDC_LAYOUT_VECTOR_1D;
    b.shape.rank = 1;
    b.shape.dim[0] = N_ROWS;
    tdc_shape_set_contiguous(&b.shape);
    return b;
}

static tdc_block col_block(int rg, int c) {
    switch (c) {
    case 0:  return make_block(col_a[rg], TDC_DT_F64);
    case 1:  return make_block(col_b[rg], TDC_DT_F64);
    default: return make_block(col_c[rg], TDC_DT_I32);
    }
}

static tdc_codec_spec col_spec(int c) {
    tdc_codec_spec spec = tdc_codec_spec_raw();
    spec.xform[0] = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0] = TDC_ENTROPY_LZ;
    if (c == 2) spec.model = TDC_MODEL_DELTA_1D;
    return spec;
}

static const tdc_column_desc k_columns[N_COLS] = {
    { "a", 1, TDC_DT_F64, "u", 1 },
    { "b", 1, TDC_DT_F64, "u", 1 },
    { "c", 1, TDC_DT_I32, "u", 1 },
};
static const tdc_schema k_schema = { N_COLS, k_columns };

static tdc_status open_enc(mem_io *mio, tdc_stream_encoder **enc) {
    tdc_stream_encoder_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.io.write_fn = mem_write;
    cfg.io.seek_fn  = mem_seek;
    cfg.io.ctx      = mio;
    cfg.flags       = TDC_CONTAINER_FLAG_HETEROGENEOUS;
    cfg.schema      = &k_schema;
    cfg.realloc_fn  = test_realloc;
    return tdc_stream_encoder_open(&cfg, enc);
}

/* mode 0: write_block only; 1: write_record only; 2: alternate per column. */
static int build(mem_io *mio, int mode) {
    tdc_stream_encoder *enc = NULL;
    ASSERT_OR_DIE(open_enc(mio, &enc) == TDC_OK, "open");
    for (int rg = 0; rg < N_RGS; ++rg) {
        /* Encode the whole row group first, as a parallel caller would. */
        tdc_buffer recs[N_COLS];
        for (int c = 0; c < N_COLS; ++c) {
            memset(&recs[c], 0, sizeof(recs[c]));
            recs[c].realloc_fn = test_realloc;
            if (mode == 0) continue;
            tdc_block b = col_block(rg, c);
            tdc_codec_spec spec = col_spec(c);
            ASSERT_OR_DIE(tdc_encode_block(&b, &spec, &recs[c]) == TDC_OK, "encode");
        }
        for (int c = 0; c < N_COLS; ++c) {
            int use_record = mode == 1 || (mode == 2 && ((rg + c) & 1));
            if (use_record) {
                ASSERT_OR_DIE(tdc_stream_encoder_write_record(enc, recs[c].data, recs[c].size)
                              == TDC_OK, "write_record");
            } else {
                tdc_block b = col_block(rg, c);
                tdc_codec_spec spec = col_spec(c);
                ASSERT_OR_DIE(tdc_stream_encoder_write_block(enc, &b, &spec) == TDC_OK,
                              "write_block");
            }
            free(recs[c].data);
        }
        ASSERT_OR_DIE(tdc_stream_encoder_end_rowgroup(enc, N_ROWS) == TDC_OK, "end rg");
    }
    ASSERT_OR_DIE(tdc_stream_encoder_close(&enc) == TDC_OK, "close");
    return 0;
}

static int check_values(mem_io *mio) {
    mio->cursor = 0;
    tdc_stream_decoder_config dcfg;
    memset(&dcfg, 0, sizeof(dcfg));
    dcfg.io.read_fn = mem_read;
    dcfg.io.seek_fn = mem_seek;
    dcfg.io.ctx     = mio;
    dcfg.realloc_fn = test_realloc;
    tdc_stream_decoder *dec = NULL;
    ASSERT_OR_DIE(tdc_stream_decoder_open(&dcfg, &dec) == TDC_OK, "decoder open");

    static double out[N_ROWS];
    for (int rg = 0; rg < N_RGS; ++rg) {
        for (int c = 0; c < N_COLS; ++c) {
            tdc_block_record rec;
            ASSERT_OR_DIE(tdc_stream_decoder_seek_rowgroup(dec, (uint64_t)rg, (uint16_t)c, &rec)
                          == TDC_OK, "seek");
            tdc_block src = col_block(rg, c);
            tdc_block dst = make_block(out, src.dtype);
            ASSERT_OR_DIE(tdc_stream_decoder_read_block(dec, &dst) == TDC_OK, "read");
            size_t bytes = (size_t)N_ROWS * tdc_dtype_size(src.dtype);
            ASSERT_OR_DIE(memcmp(out, src.data, bytes) == 0, "values");
        }
    }
    tdc_stream_decoder_close(&dec);
    return 0;
}

static int check_rejects(void) {
    mem_io mio;
    memset(&mio, 0, sizeof(mio));
    tdc_stream_encoder *enc = NULL;
    ASSERT_OR_DIE(open_enc(&mio, &enc) == TDC_OK, "open");

    tdc_buffer rec;
    memset(&rec, 0, sizeof(rec));
    rec.realloc_fn = test_realloc;
    tdc_block b = col_block(0, 0);
    tdc_codec_spec spec = col_spec(0);
    ASSERT_OR_DIE(tdc_encode_block(&b, &spec, &rec) == TDC_OK, "encode");

    size_t size_before = mio.size;
    ASSERT_OR_DIE(tdc_stream_encoder_write_record(NULL, rec.data, rec.size) == TDC_E_INVAL,
                  "NULL encoder");
    ASSERT_OR_DIE(tdc_stream_encoder_write_record(enc, NULL, rec.size) == TDC_E_INVAL,
                  "NULL record");
    ASSERT_OR_DIE(tdc_stream_encoder_write_record(enc, rec.data, 40) == TDC_E_CORRUPT,
                  "short record");
    ASSERT_OR_DIE(tdc_stream_encoder_write_record(enc, rec.data, rec.size - 1) == TDC_E_CORRUPT,
                  "truncated record");

    uint8_t *longer = (uint8_t *)malloc(rec.size + 1);
    ASSERT_OR_DIE(longer != NULL, "alloc");
    memcpy(longer, rec.data, rec.size);
    longer[rec.size] = 0;
    ASSERT_OR_DIE(tdc_stream_encoder_write_record(enc, longer, rec.size + 1) == TDC_E_CORRUPT,
                  "record with trailing byte");
    longer[0] ^= 0xFFu;
    ASSERT_OR_DIE(tdc_stream_encoder_write_record(enc, longer, rec.size) != TDC_OK,
                  "bad magic");
    free(longer);

    ASSERT_OR_DIE(tdc_stream_encoder_block_count(enc) == 0, "no block counted");
    ASSERT_OR_DIE(mio.size == size_before, "nothing written");

    ASSERT_OR_DIE(tdc_stream_encoder_write_record(enc, rec.data, rec.size) == TDC_OK,
                  "valid record after rejects");
    ASSERT_OR_DIE(tdc_stream_encoder_block_count(enc) == 1, "one block counted");
    free(rec.data);
    tdc_stream_encoder_abort(&enc);
    free(mio.data);
    return 0;
}

int main(void) {
    fill_sources();

    mem_io by_block, by_record, mixed;
    memset(&by_block, 0, sizeof(by_block));
    memset(&by_record, 0, sizeof(by_record));
    memset(&mixed, 0, sizeof(mixed));
    if (build(&by_block, 0)) return 1;
    if (build(&by_record, 1)) return 1;
    if (build(&mixed, 2)) return 1;

    ASSERT_OR_DIE(by_block.size == by_record.size &&
                  memcmp(by_block.data, by_record.data, by_block.size) == 0,
                  "write_record container identical to write_block container");
    ASSERT_OR_DIE(by_block.size == mixed.size &&
                  memcmp(by_block.data, mixed.data, by_block.size) == 0,
                  "mixed container identical to write_block container");

    if (check_values(&by_record)) return 1;
    if (check_rejects()) return 1;

    free(by_block.data);
    free(by_record.data);
    free(mixed.data);
    printf("PASS: test_stream_write_record\n");
    return 0;
}
