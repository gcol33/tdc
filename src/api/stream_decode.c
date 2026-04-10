/*
 * src/api/stream_decode.c
 *
 * Streaming decoder: schema + row-group index.
 *
 * Accepts both version 1 and version 2 containers.
 * On version 1 containers, schema is NULL and the row-group index is
 * absent; the decoder falls back to sequential-only block reading.
 *
 * Container layout:
 *
 *   [64-byte header]                (version == 2, _reserved1 = schema_size)
 *   [schema_size bytes of schema]   (column descriptors)
 *   [block records ...]             (self-describing, same as v1)
 *   [trailing row-group index]      (at index_offset, if present)
 *
 * Schema wire format:
 *   u16 n_columns
 *   per column: u16 name_len, name_len bytes UTF-8, u8 dtype,
 *               u16 ann_len, ann_len bytes UTF-8
 *
 * Row-group index wire format:
 *   u64 n_rowgroups
 *   per row group:
 *       u64 offset, u64 n_rows, u16 n_cols, u16 _pad, u32 _reserved
 *       per column: u64 block_offset, u64 block_total
 */

#ifdef _MSC_VER
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "tdc/stream.h"
#include "tdc/codec.h"
#include "tdc/format.h"
#include "tdc/types.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* ----- Internal state ---------------------------------------------------- */

struct tdc_stream_decoder {
    tdc_io      io;

    void *(*realloc_fn)(void *user, void *ptr, size_t new_size);
    void   *alloc_user;

    tdc_container_header header;

    /* Parsed schema (owned; NULL if v1 or schema_size == 0). */
    tdc_schema           schema;
    tdc_column_desc     *schema_columns;   /* owned array */
    char                *schema_strings;   /* owned: all names + annotations */
    int                  has_schema;

    /* Row-group index (owned; NULL if not seekable or no index). */
    tdc_rowgroup_entry  *rg_entries;
    uint64_t             rg_count;
    int                  has_rg_index;

    /* File offset where blocks begin (after header + schema). */
    uint64_t    blocks_start;

    /* Sequential reading state. */
    uint64_t    read_pos;
    uint64_t    blocks_read;

    /* Peek state. */
    int                  peeked;
    tdc_block_record     peeked_record;
    uint64_t             peeked_remaining;

    /* Scratch buffer for assembling full block records. */
    tdc_buffer  scratch;
};

/* ----- I/O helpers ------------------------------------------------------- */

static tdc_status v2_io_read_exact(const tdc_io *io, void *buf, size_t size) {
    size_t total = 0;
    while (total < size) {
        size_t got = 0;
        tdc_status st = io->read_fn(io->ctx,
                                     (uint8_t *)buf + total,
                                     size - total, &got);
        if (st != TDC_OK) return TDC_E_IO;
        if (got == 0) return TDC_E_CORRUPT;
        total += got;
    }
    return TDC_OK;
}

static tdc_status v2_io_read_maybe(const tdc_io *io, void *buf, size_t size,
                                   size_t *actually_read) {
    size_t total = 0;
    while (total < size) {
        size_t got = 0;
        tdc_status st = io->read_fn(io->ctx,
                                     (uint8_t *)buf + total,
                                     size - total, &got);
        if (st != TDC_OK) return TDC_E_IO;
        if (got == 0) {
            if (total == 0) { *actually_read = 0; return TDC_OK; }
            return TDC_E_CORRUPT;
        }
        total += got;
    }
    *actually_read = total;
    return TDC_OK;
}

static tdc_status v2_io_skip_bytes(const tdc_io *io, size_t n) {
    uint8_t tmp[4096];
    while (n > 0) {
        size_t chunk = n < sizeof(tmp) ? n : sizeof(tmp);
        tdc_status st = v2_io_read_exact(io, tmp, chunk);
        if (st != TDC_OK) return st;
        n -= chunk;
    }
    return TDC_OK;
}

/* ----- Allocator helpers ------------------------------------------------- */

static void *v2_alloc(struct tdc_stream_decoder *d, void *ptr, size_t sz) {
    return d->realloc_fn(d->alloc_user, ptr, sz);
}

static void v2_free(struct tdc_stream_decoder *d, void *ptr) {
    if (ptr) d->realloc_fn(d->alloc_user, ptr, 0);
}

/* ----- Little-endian read helpers ---------------------------------------- */
/*
 * The tdc on-disk format is fixed little-endian. On supported targets
 * (x86_64, aarch64 — both LE), memcpy is a no-op at the optimizer level
 * and is strictly-aliasing safe. No byte-swap needed.
 */

static uint16_t read_u16_le(const uint8_t *p) {
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
}

static uint32_t read_u32_le(const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static uint64_t read_u64_le(const uint8_t *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

/* ----- Schema parsing ---------------------------------------------------- */

/*
 * Parse the schema section from a raw byte buffer.
 *
 * Wire format:
 *   u16 n_columns
 *   per column: u16 name_len, name_len bytes, u8 dtype, u16 ann_len, ann_len bytes
 *
 * We make two passes:
 *   1. Validate and compute total string bytes needed (names + annotations
 *      + null terminators).
 *   2. Allocate and fill the column descriptor array + string buffer.
 *
 * All allocations go through d->realloc_fn. On error, nothing is allocated.
 */
static tdc_status parse_schema(struct tdc_stream_decoder *d,
                               const uint8_t *buf, size_t buf_size) {
    if (buf_size < 2) return TDC_E_CORRUPT;

    uint16_t n_cols = read_u16_le(buf);
    if (n_cols == 0) return TDC_OK; /* valid: zero-column schema */

    /* Pass 1: validate and sum string sizes. */
    size_t pos = 2;
    size_t total_str_bytes = 0;

    for (uint16_t i = 0; i < n_cols; i++) {
        /* name_len (2) + at least dtype (1) + ann_len (2) */
        if (pos + 2 > buf_size) return TDC_E_CORRUPT;
        uint16_t name_len = read_u16_le(buf + pos);
        pos += 2;

        if (pos + (size_t)name_len > buf_size) return TDC_E_CORRUPT;
        pos += name_len;
        total_str_bytes += (size_t)name_len + 1; /* +1 for null terminator */

        /* dtype */
        if (pos + 1 > buf_size) return TDC_E_CORRUPT;
        pos += 1;

        /* ann_len */
        if (pos + 2 > buf_size) return TDC_E_CORRUPT;
        uint16_t ann_len = read_u16_le(buf + pos);
        pos += 2;

        if (pos + (size_t)ann_len > buf_size) return TDC_E_CORRUPT;
        pos += ann_len;
        total_str_bytes += (size_t)ann_len + 1;
    }

    /* Allocate column descriptors. */
    tdc_column_desc *cols = (tdc_column_desc *)v2_alloc(
        d, NULL, (size_t)n_cols * sizeof(tdc_column_desc));
    if (!cols) return TDC_E_NOMEM;

    /* Allocate a single string buffer for all names + annotations. */
    char *strbuf = (char *)v2_alloc(d, NULL, total_str_bytes);
    if (!strbuf) {
        v2_free(d, cols);
        return TDC_E_NOMEM;
    }

    /* Pass 2: fill descriptors. */
    pos = 2;
    size_t str_pos = 0;

    for (uint16_t i = 0; i < n_cols; i++) {
        uint16_t name_len = read_u16_le(buf + pos);
        pos += 2;

        cols[i].name = strbuf + str_pos;
        cols[i].name_len = name_len;
        memcpy(strbuf + str_pos, buf + pos, name_len);
        str_pos += name_len;
        strbuf[str_pos++] = '\0';
        pos += name_len;

        cols[i].dtype = buf[pos];
        pos += 1;

        uint16_t ann_len = read_u16_le(buf + pos);
        pos += 2;

        cols[i].annotation = strbuf + str_pos;
        cols[i].ann_len = ann_len;
        if (ann_len > 0) {
            memcpy(strbuf + str_pos, buf + pos, ann_len);
        }
        str_pos += ann_len;
        strbuf[str_pos++] = '\0';
        pos += ann_len;
    }

    d->schema_columns  = cols;
    d->schema_strings  = strbuf;
    d->schema.n_columns = n_cols;
    d->schema.columns   = cols;
    d->has_schema       = 1;

    return TDC_OK;
}

/* ----- Row-group index parsing ------------------------------------------- */

/*
 * Row-group index wire format (at index_offset):
 *   u64 n_rowgroups
 *   per row group (24 + 16*n_cols bytes):
 *       u64 offset
 *       u64 n_rows
 *       u16 n_cols
 *       u16 _pad
 *       u32 _reserved
 *       per column:
 *           u64 block_offset
 *           u64 block_total
 */

#define RG_HEADER_SIZE 24  /* offset(8) + n_rows(8) + n_cols(2) + pad(2) + reserved(4) */
#define RG_COL_SIZE    16  /* block_offset(8) + block_total(8) */

static tdc_status parse_rowgroup_index(struct tdc_stream_decoder *d,
                                       const uint8_t *buf,
                                       size_t buf_size) {
    if (buf_size < 8) return TDC_E_CORRUPT;

    uint64_t n_rg = read_u64_le(buf);
    if (n_rg == 0) return TDC_OK;

    /* Validate that the buffer is large enough. We need to parse
     * incrementally because n_cols varies per row group. */
    size_t pos = 8;

    /* First pass: validate sizes. */
    for (uint64_t i = 0; i < n_rg; i++) {
        if (pos + RG_HEADER_SIZE > buf_size) return TDC_E_CORRUPT;
        uint16_t nc = read_u16_le(buf + pos + 16); /* n_cols at offset 16 within RG header */
        pos += RG_HEADER_SIZE + (size_t)nc * RG_COL_SIZE;
        if (pos > buf_size) return TDC_E_CORRUPT;
    }

    /* Allocate the entry array. */
    tdc_rowgroup_entry *entries = (tdc_rowgroup_entry *)v2_alloc(
        d, NULL, (size_t)n_rg * sizeof(tdc_rowgroup_entry));
    if (!entries) return TDC_E_NOMEM;
    memset(entries, 0, (size_t)n_rg * sizeof(tdc_rowgroup_entry));

    /* Second pass: fill entries. Allocate per-entry column arrays. */
    pos = 8;
    for (uint64_t i = 0; i < n_rg; i++) {
        entries[i].offset = read_u64_le(buf + pos);
        entries[i].n_rows = read_u64_le(buf + pos + 8);
        entries[i].n_cols = read_u16_le(buf + pos + 16);
        /* _pad at pos+18, _reserved at pos+20 — skip */
        pos += RG_HEADER_SIZE;

        uint16_t nc = entries[i].n_cols;
        if (nc > 0) {
            tdc_rg_col_entry *ce = (tdc_rg_col_entry *)v2_alloc(
                d, NULL, (size_t)nc * sizeof(tdc_rg_col_entry));
            if (!ce) {
                /* Free everything allocated so far. */
                for (uint64_t j = 0; j < i; j++) {
                    v2_free(d, entries[j].columns);
                }
                v2_free(d, entries);
                return TDC_E_NOMEM;
            }
            for (uint16_t c = 0; c < nc; c++) {
                ce[c].block_offset = read_u64_le(buf + pos);
                ce[c].block_total  = read_u64_le(buf + pos + 8);
                pos += RG_COL_SIZE;
            }
            entries[i].columns = ce;
        } else {
            entries[i].columns = NULL;
        }
    }

    d->rg_entries   = entries;
    d->rg_count     = n_rg;
    d->has_rg_index = 1;

    return TDC_OK;
}

/* ----- Free helpers for owned sub-allocations ---------------------------- */

static void free_schema(struct tdc_stream_decoder *d) {
    v2_free(d, d->schema_columns);
    v2_free(d, d->schema_strings);
    d->schema_columns = NULL;
    d->schema_strings = NULL;
    d->schema.columns = NULL;
    d->schema.n_columns = 0;
    d->has_schema = 0;
}

static void free_rg_index(struct tdc_stream_decoder *d) {
    if (d->rg_entries) {
        for (uint64_t i = 0; i < d->rg_count; i++) {
            v2_free(d, d->rg_entries[i].columns);
        }
        v2_free(d, d->rg_entries);
        d->rg_entries = NULL;
    }
    d->rg_count = 0;
    d->has_rg_index = 0;
}

/* ----- Public API -------------------------------------------------------- */

tdc_status tdc_stream_decoder_open(const tdc_stream_decoder_config *cfg,
                                      tdc_stream_decoder **dec) {
    if (!cfg || !dec) return TDC_E_INVAL;
    *dec = NULL;

    if (!cfg->io.read_fn) return TDC_E_INVAL;
    if (!cfg->realloc_fn) return TDC_E_INVAL;

    tdc_stream_decoder *d = (tdc_stream_decoder *)cfg->realloc_fn(
        cfg->alloc_user, NULL, sizeof(tdc_stream_decoder));
    if (!d) return TDC_E_NOMEM;
    memset(d, 0, sizeof(*d));

    d->io         = cfg->io;
    d->realloc_fn = cfg->realloc_fn;
    d->alloc_user = cfg->alloc_user;

    d->scratch.data       = NULL;
    d->scratch.size       = 0;
    d->scratch.capacity   = 0;
    d->scratch.realloc_fn = cfg->realloc_fn;
    d->scratch.user       = cfg->alloc_user;

    /* ---- Step 1: Read and validate the 64-byte container header. ---- */

    tdc_status st = v2_io_read_exact(&d->io, &d->header,
                                     TDC_CONTAINER_HEADER_SIZE);
    if (st != TDC_OK) { v2_free(d, d); return st; }

    /* Validate, but allow version 2 in addition to 1. The standard
     * validator only accepts TDC_CONTAINER_VERSION (== 1). For version 2
     * we do our own checks. */
    if (d->header.magic != TDC_CONTAINER_MAGIC) {
        v2_free(d, d);
        return TDC_E_CORRUPT;
    }
    if (d->header.version != 1 && d->header.version != 2) {
        v2_free(d, d);
        return TDC_E_VERSION;
    }

    uint64_t pos = TDC_CONTAINER_HEADER_SIZE;

    /* ---- Step 2: Parse schema if v2 and schema_size > 0. ---- */

    uint32_t schema_size = 0;
    if (d->header.version == 2) {
        /* In v2, _reserved1 (u32 at offset 36) is reinterpreted as
         * schema_size. We already read the full header into the struct;
         * _reserved1 holds the value. */
        schema_size = d->header._reserved1;
    }

    if (schema_size > 0) {
        uint8_t *schema_buf = (uint8_t *)v2_alloc(d, NULL, schema_size);
        if (!schema_buf) {
            v2_free(d, d);
            return TDC_E_NOMEM;
        }

        st = v2_io_read_exact(&d->io, schema_buf, schema_size);
        if (st != TDC_OK) {
            v2_free(d, schema_buf);
            v2_free(d, d);
            return st;
        }

        st = parse_schema(d, schema_buf, schema_size);
        v2_free(d, schema_buf); /* raw buffer no longer needed */
        if (st != TDC_OK) {
            v2_free(d, d);
            return st;
        }

        pos += schema_size;
    }

    d->blocks_start = pos;
    d->read_pos     = pos;

    /* ---- Step 3: Load row-group index if seekable + present. ---- */

    if (cfg->io.seek_fn &&
        d->header.index_offset > 0 && d->header.index_size > 0) {

        size_t idx_bytes = (size_t)d->header.index_size;
        uint8_t *idx_buf = (uint8_t *)v2_alloc(d, NULL, idx_bytes);
        if (!idx_buf) {
            free_schema(d);
            v2_free(d, d);
            return TDC_E_NOMEM;
        }

        st = cfg->io.seek_fn(cfg->io.ctx,
                              (int64_t)d->header.index_offset,
                              TDC_SEEK_SET);
        if (st != TDC_OK) {
            v2_free(d, idx_buf);
            free_schema(d);
            v2_free(d, d);
            return st;
        }

        st = v2_io_read_exact(&d->io, idx_buf, idx_bytes);
        if (st != TDC_OK) {
            v2_free(d, idx_buf);
            free_schema(d);
            v2_free(d, d);
            return st;
        }

        st = parse_rowgroup_index(d, idx_buf, idx_bytes);
        v2_free(d, idx_buf);
        if (st != TDC_OK) {
            free_schema(d);
            v2_free(d, d);
            return st;
        }

        /* Seek back to first block position. */
        st = cfg->io.seek_fn(cfg->io.ctx,
                              (int64_t)d->blocks_start,
                              TDC_SEEK_SET);
        if (st != TDC_OK) {
            free_rg_index(d);
            free_schema(d);
            v2_free(d, d);
            return st;
        }
        d->read_pos = d->blocks_start;
    }

    *dec = d;
    return TDC_OK;
}

const tdc_container_header *tdc_stream_decoder_header(
    const tdc_stream_decoder *dec) {
    return dec ? &dec->header : NULL;
}

const tdc_schema *tdc_stream_decoder_read_schema(
    const tdc_stream_decoder *dec) {
    if (!dec || !dec->has_schema) return NULL;
    return &dec->schema;
}

int tdc_stream_decoder_has_rowgroup_index(
    const tdc_stream_decoder *dec) {
    return dec ? dec->has_rg_index : 0;
}

uint64_t tdc_stream_decoder_rowgroup_count(
    const tdc_stream_decoder *dec) {
    if (!dec) return 0;
    return dec->has_rg_index ? dec->rg_count : 0;
}

const tdc_rowgroup_entry *tdc_stream_decoder_get_rowgroup(
    const tdc_stream_decoder *dec, uint64_t rg_index) {
    if (!dec || !dec->has_rg_index) return NULL;
    if (rg_index >= dec->rg_count) return NULL;
    return &dec->rg_entries[rg_index];
}

tdc_status tdc_stream_decoder_seek_rowgroup(tdc_stream_decoder *dec,
                                                uint64_t rg_index,
                                                uint16_t col_index,
                                                tdc_block_record *rec) {
    if (!dec || !rec) return TDC_E_INVAL;
    if (!dec->has_rg_index) return TDC_E_INVAL;
    if (!dec->io.seek_fn) return TDC_E_INVAL;
    if (rg_index >= dec->rg_count) return TDC_E_INVAL;

    const tdc_rowgroup_entry *rg = &dec->rg_entries[rg_index];
    if (col_index >= rg->n_cols) return TDC_E_INVAL;

    /* Cancel any outstanding peek and reset the sequential block
     * counter — seek_rowgroup repositions the stream, so the
     * blocks_read guard in peek_block must not prevent reading. */
    dec->peeked      = 0;
    dec->blocks_read = 0;

    /* Seek to the column's block offset. */
    uint64_t target = rg->columns[col_index].block_offset;
    tdc_status st = dec->io.seek_fn(dec->io.ctx,
                                     (int64_t)target,
                                     TDC_SEEK_SET);
    if (st != TDC_OK) return st;

    dec->read_pos = target;

    /* Peek the block header at the new position. */
    return tdc_stream_decoder_peek_block(dec, rec);
}

/* ----- Block-level operations (same protocol as v1) ---------------------- */

tdc_status tdc_stream_decoder_peek_block(tdc_stream_decoder *dec,
                                            tdc_block_record      *rec) {
    if (!dec || !rec) return TDC_E_INVAL;
    if (dec->peeked) return TDC_E_INVAL;

    /* Check if we've consumed all blocks (known count). */
    if (dec->header.n_blocks > 0 &&
        dec->blocks_read >= dec->header.n_blocks) {
        memset(rec, 0, sizeof(*rec));
        return TDC_OK;
    }

    /* Read the 4-byte magic to detect end-of-blocks. */
    uint32_t magic = 0;
    size_t got = 0;
    tdc_status st = v2_io_read_maybe(&dec->io, &magic, sizeof(magic), &got);
    if (st != TDC_OK) return st;

    if (got == 0) {
        memset(rec, 0, sizeof(*rec));
        return TDC_OK;
    }

    if (magic != TDC_BLOCK_MAGIC) {
        memset(rec, 0, sizeof(*rec));
        return TDC_OK;
    }

    /* Magic matched — read the remaining 76 bytes. */
    tdc_block_record hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = magic;
    st = v2_io_read_exact(&dec->io,
                           (uint8_t *)&hdr + sizeof(magic),
                           TDC_BLOCK_HEADER_SIZE - sizeof(magic));
    if (st != TDC_OK) return st;

    st = tdc_block_record_validate(&hdr);
    if (st != TDC_OK) return st;

    uint64_t remaining = (uint64_t)hdr.side_meta_size
                       + (uint64_t)hdr.xform_params_size
                       + (uint64_t)hdr.payload_size
                       + (uint64_t)hdr.validity_size;

    dec->peeked           = 1;
    dec->peeked_record    = hdr;
    dec->peeked_remaining = remaining;
    dec->read_pos        += TDC_BLOCK_HEADER_SIZE;

    *rec = hdr;
    return TDC_OK;
}

tdc_status tdc_stream_decoder_read_block(tdc_stream_decoder *dec,
                                            tdc_block             *dst) {
    if (!dec || !dst) return TDC_E_INVAL;
    if (!dec->peeked) return TDC_E_INVAL;

    size_t remaining = (size_t)dec->peeked_remaining;
    size_t total     = TDC_BLOCK_HEADER_SIZE + remaining;

    /* Grow scratch to hold the full block record. */
    size_t need = total > 0 ? total : 1;
    if (dec->scratch.capacity < need) {
        size_t new_cap = dec->scratch.capacity ? dec->scratch.capacity : 64;
        while (new_cap < need) new_cap *= 2;
        void *p = v2_alloc(dec, dec->scratch.data, new_cap);
        if (!p) return TDC_E_NOMEM;
        dec->scratch.data     = (uint8_t *)p;
        dec->scratch.capacity = new_cap;
    }

    /* Copy the already-read 80-byte header into scratch. */
    memcpy(dec->scratch.data, &dec->peeked_record, TDC_BLOCK_HEADER_SIZE);

    /* Read the remaining sections. */
    if (remaining > 0) {
        tdc_status st = v2_io_read_exact(&dec->io,
                                          dec->scratch.data + TDC_BLOCK_HEADER_SIZE,
                                          remaining);
        if (st != TDC_OK) return st;
    }

    dec->read_pos += remaining;
    dec->peeked    = 0;
    dec->blocks_read++;

    return tdc_decode_block(dec->scratch.data, total, dst);
}

tdc_status tdc_stream_decoder_skip_block(tdc_stream_decoder *dec) {
    if (!dec)         return TDC_E_INVAL;
    if (!dec->peeked) return TDC_E_INVAL;

    size_t remaining = (size_t)dec->peeked_remaining;

    tdc_status st;
    if (dec->io.seek_fn && remaining > 0) {
        st = dec->io.seek_fn(dec->io.ctx, (int64_t)remaining, TDC_SEEK_CUR);
    } else if (remaining > 0) {
        st = v2_io_skip_bytes(&dec->io, remaining);
    } else {
        st = TDC_OK;
    }

    dec->read_pos += remaining;
    dec->peeked    = 0;
    dec->blocks_read++;

    return st;
}

uint64_t tdc_stream_decoder_block_count(
    const tdc_stream_decoder *dec) {
    if (!dec) return 0;
    if (dec->header.n_blocks > 0) return dec->header.n_blocks;
    return dec->blocks_read;
}

tdc_status tdc_stream_decoder_close(tdc_stream_decoder **dec) {
    if (!dec || !*dec) return TDC_OK;

    tdc_stream_decoder *d = *dec;

    free_rg_index(d);
    free_schema(d);

    if (d->scratch.data) {
        d->realloc_fn(d->alloc_user, d->scratch.data, 0);
    }

    d->realloc_fn(d->alloc_user, d, 0);

    *dec = NULL;
    return TDC_OK;
}
