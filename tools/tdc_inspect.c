/*
 * tools/tdc_inspect.c
 *
 * Minimal inspector for .tdc container files.
 *
 *   tdc_inspect <file.tdc>
 *
 * Opens the file via tdc_stream_decoder_open and dumps:
 *   - container header (magic, version, n_blocks, flags, global dtype/layout/shape)
 *   - schema (if present)
 *   - row-group index (if present)
 *   - per-column block offsets for row group 0
 *
 * Does NOT decode block payload — header/schema/index only.
 */

#include "tdc.h"
#include "tdc/stream.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ----- I/O adapter: FILE* ------------------------------------------------- */

typedef struct {
    FILE *fp;
} file_io;

static tdc_status file_read(void *ctx, void *buf, size_t size,
                            size_t *bytes_read) {
    file_io *io = (file_io *)ctx;
    size_t n = fread(buf, 1, size, io->fp);
    if (n < size && ferror(io->fp)) {
        *bytes_read = n;
        return TDC_E_IO;
    }
    *bytes_read = n;
    return TDC_OK;
}

static tdc_status file_seek(void *ctx, int64_t offset, int whence) {
    file_io *io = (file_io *)ctx;
    int w;
    switch (whence) {
        case TDC_SEEK_SET: w = SEEK_SET; break;
        case TDC_SEEK_CUR: w = SEEK_CUR; break;
        case TDC_SEEK_END: w = SEEK_END; break;
        default: return TDC_E_INVAL;
    }
#if defined(_WIN32)
    if (_fseeki64(io->fp, (long long)offset, w) != 0) return TDC_E_IO;
#else
    if (fseeko(io->fp, (off_t)offset, w) != 0) return TDC_E_IO;
#endif
    return TDC_OK;
}

static void *stdlib_realloc(void *user, void *ptr, size_t new_size) {
    (void)user;
    if (new_size == 0) { free(ptr); return NULL; }
    return realloc(ptr, new_size);
}

/* ----- Enum-to-name helpers ---------------------------------------------- */

static const char *dtype_name(tdc_dtype d) {
    switch (d) {
        case TDC_DT_I8:     return "i8";
        case TDC_DT_I16:    return "i16";
        case TDC_DT_I32:    return "i32";
        case TDC_DT_I64:    return "i64";
        case TDC_DT_U8:     return "u8";
        case TDC_DT_U16:    return "u16";
        case TDC_DT_U32:    return "u32";
        case TDC_DT_U64:    return "u64";
        case TDC_DT_F16:    return "f16";
        case TDC_DT_F32:    return "f32";
        case TDC_DT_F64:    return "f64";
        case TDC_DT_STRING: return "string";
        default:            return "unknown";
    }
}

static const char *layout_name(tdc_layout l) {
    switch (l) {
        case TDC_LAYOUT_VECTOR_1D: return "VECTOR_1D";
        case TDC_LAYOUT_RASTER_2D: return "RASTER_2D";
        case TDC_LAYOUT_STACK_2D:  return "STACK_2D";
        case TDC_LAYOUT_VOLUME_3D: return "VOLUME_3D";
        default:                   return "none";
    }
}

/* ----- Pretty printer ---------------------------------------------------- */

static void print_shape(const int64_t *dim, uint8_t rank) {
    printf("[");
    for (uint8_t i = 0; i < rank; ++i) {
        if (i) printf(", ");
        printf("%" PRId64, dim[i]);
    }
    printf("]");
}

static int64_t file_size_bytes(FILE *fp) {
#if defined(_WIN32)
    if (_fseeki64(fp, 0, SEEK_END) != 0) return -1;
    long long n = _ftelli64(fp);
    if (_fseeki64(fp, 0, SEEK_SET) != 0) return -1;
    return (int64_t)n;
#else
    if (fseeko(fp, 0, SEEK_END) != 0) return -1;
    off_t n = ftello(fp);
    if (fseeko(fp, 0, SEEK_SET) != 0) return -1;
    return (int64_t)n;
#endif
}

/* ----- main -------------------------------------------------------------- */

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: tdc_inspect <file.tdc>\n");
        return 1;
    }
    const char *path = argv[1];

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "tdc_inspect: cannot open %s\n", path);
        return 1;
    }

    int64_t fsize = file_size_bytes(fp);
    if (fsize < 0) {
        fprintf(stderr, "tdc_inspect: cannot stat %s\n", path);
        fclose(fp);
        return 1;
    }

    file_io fio = { fp };

    tdc_stream_decoder_config dcfg;
    memset(&dcfg, 0, sizeof(dcfg));
    dcfg.io.write_fn = NULL;
    dcfg.io.read_fn  = file_read;
    dcfg.io.seek_fn  = file_seek;
    dcfg.io.ctx     = &fio;
    dcfg.realloc_fn  = stdlib_realloc;
    dcfg.alloc_user  = NULL;

    tdc_stream_decoder *dec = NULL;
    tdc_status st = tdc_stream_decoder_open(&dcfg, &dec);
    if (st != TDC_OK) {
        fprintf(stderr, "tdc_inspect: decoder open failed: %s\n",
                tdc_strerror(st));
        fclose(fp);
        return 1;
    }

    printf("=== tdc file: %s (%" PRId64 " bytes) ===\n", path, fsize);

    /* --- Container header ------------------------------------------------ */
    const tdc_container_header *h = tdc_stream_decoder_header(dec);
    if (!h) {
        fprintf(stderr, "tdc_inspect: header view unavailable\n");
        tdc_stream_decoder_close(&dec);
        fclose(fp);
        return 1;
    }

    printf("header          magic=0x%08" PRIx32 " version=%" PRIu16
           " n_blocks=%" PRIu64 " flags=0x%04" PRIx16 "\n",
           h->magic, h->version, h->n_blocks, h->flags);
    printf("                global_dtype=%s global_layout=%s\n",
           dtype_name((tdc_dtype)h->global_dtype),
           layout_name((tdc_layout)h->global_layout));
    printf("                global_shape=");
    print_shape(h->global_dim, h->global_rank);
    printf("\n");

    /* --- Schema ---------------------------------------------------------- */
    const tdc_schema *schema = tdc_stream_decoder_read_schema(dec);
    if (!schema) {
        printf("schema          <none>\n");
    } else {
        printf("schema          %" PRIu16 " columns\n", schema->n_columns);
        printf("  %-3s  %-18s  %-9s  %s\n",
               "idx", "name", "dtype", "annotation");
        for (uint16_t i = 0; i < schema->n_columns; ++i) {
            const tdc_column_desc *c = &schema->columns[i];
            const char *ann = (c->ann_len > 0 && c->annotation) ? c->annotation : "";
            printf("  %3u  %-18s  %-9s  \"%s\"\n",
                   (unsigned)i,
                   c->name ? c->name : "",
                   dtype_name((tdc_dtype)c->dtype),
                   ann);
        }
    }

    /* --- Row-group index ------------------------------------------------- */
    int has_rg = tdc_stream_decoder_has_rowgroup_index(dec);
    if (!has_rg) {
        printf("row groups      <none>\n");
    } else {
        uint64_t n_rg = tdc_stream_decoder_rowgroup_count(dec);
        printf("row groups      %" PRIu64 " groups (index_offset=0x%" PRIx64 ")\n",
               n_rg, h->index_offset);
        printf("  %-3s  %-8s  %-14s  %s\n",
               "rg", "n_rows", "offset", "n_cols");
        for (uint64_t i = 0; i < n_rg; ++i) {
            const tdc_rowgroup_entry *rge =
                tdc_stream_decoder_get_rowgroup(dec, i);
            if (!rge) continue;
            printf("  %3" PRIu64 "  %-8" PRIu64 "  0x%012" PRIx64 "  %" PRIu16 "\n",
                   i, rge->n_rows, rge->offset, rge->n_cols);
        }

        if (n_rg > 0) {
            const tdc_rowgroup_entry *rg0 =
                tdc_stream_decoder_get_rowgroup(dec, 0);
            if (rg0) {
                printf("\nblocks (rg=0)\n");
                printf("  %-3s  %-14s  %s\n", "col", "offset", "total");
                for (uint16_t c = 0; c < rg0->n_cols; ++c) {
                    printf("  %3u  0x%012" PRIx64 "  %" PRIu64 "\n",
                           (unsigned)c,
                           rg0->columns[c].block_offset,
                           rg0->columns[c].block_total);
                }
            }
        }
    }

    printf("\nstatus          OK\n");

    tdc_stream_decoder_close(&dec);
    fclose(fp);
    return 0;
}
