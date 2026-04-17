/*
 * tests/test_format_validate.c
 *
 * Direct coverage for the cheap structural validators that the rest of
 * the pipeline depends on:
 *
 *   tdc_strerror
 *   tdc_container_header_validate
 *   tdc_block_record_validate (smoke — already exercised by api/decode.c)
 *
 * No buffers, no allocators, no compression. Just struct -> status.
 */

#include "tdc/error.h"
#include "tdc/format.h"
#include "tdc/types.h"

#include <stdio.h>
#include <string.h>

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while (0)

/* ----- tdc_strerror ------------------------------------------------------- */

static int test_strerror(void) {
    /* Every defined code returns a non-NULL non-empty string. */
    const tdc_status codes[] = {
        TDC_OK, TDC_E_INVAL, TDC_E_NOMEM, TDC_E_UNSUPPORTED,
        TDC_E_DTYPE, TDC_E_LAYOUT, TDC_E_SHAPE, TDC_E_BUF_TOO_SMALL,
        TDC_E_CORRUPT, TDC_E_VERSION, TDC_E_IO,
    };
    for (size_t i = 0; i < sizeof(codes) / sizeof(codes[0]); ++i) {
        const char *s = tdc_strerror(codes[i]);
        CHECK(s != NULL);
        CHECK(s[0] != '\0');
    }
    /* Out-of-range falls through to the unknown branch — never NULL. */
    const char *u = tdc_strerror((tdc_status)9999);
    CHECK(u != NULL);
    CHECK(strstr(u, "unknown") != NULL);
    /* Sanity: TDC_OK should not say "ok" by accident — verify it does. */
    CHECK(strcmp(tdc_strerror(TDC_OK), "ok") == 0);
    return 0;
}

/* ----- tdc_container_header_validate -------------------------------------- */

static tdc_container_header make_homogeneous_header(void) {
    tdc_container_header h = {0};
    h.magic         = TDC_CONTAINER_MAGIC;
    h.version       = TDC_CONTAINER_VERSION;
    h.flags         = 0;
    h.n_blocks      = 4;
    h.index_offset  = 1024;
    h.index_size    = 4 * TDC_INDEX_ENTRY_SIZE;
    h.global_dtype  = (uint8_t)TDC_DT_F32;
    h.global_layout = (uint8_t)TDC_LAYOUT_RASTER_2D;
    h.global_rank   = 2;
    h.global_dim[0] = 256;
    h.global_dim[1] = 256;
    h.global_dim[2] = 0;
    return h;
}

static tdc_container_header make_heterogeneous_header(void) {
    tdc_container_header h = {0};
    h.magic        = TDC_CONTAINER_MAGIC;
    h.version      = TDC_CONTAINER_VERSION;
    h.flags        = TDC_CONTAINER_FLAG_HETEROGENEOUS;
    h.n_blocks     = 2;
    h.index_offset = 512;
    h.index_size   = 2 * TDC_INDEX_ENTRY_SIZE;
    /* All global_* fields stay zero. */
    return h;
}

static int test_container_header_happy(void) {
    tdc_container_header h = make_homogeneous_header();
    CHECK(tdc_container_header_validate(&h) == TDC_OK);

    tdc_container_header het = make_heterogeneous_header();
    CHECK(tdc_container_header_validate(&het) == TDC_OK);

    /* Empty container: n_blocks == 0, no index. */
    tdc_container_header empty = make_heterogeneous_header();
    empty.n_blocks     = 0;
    empty.index_offset = 0;
    empty.index_size   = 0;
    CHECK(tdc_container_header_validate(&empty) == TDC_OK);

    return 0;
}

static int test_container_header_rejects(void) {
    /* NULL */
    CHECK(tdc_container_header_validate(NULL) == TDC_E_INVAL);

    /* Bad magic */
    {
        tdc_container_header h = make_homogeneous_header();
        h.magic = 0xDEADBEEFu;
        CHECK(tdc_container_header_validate(&h) == TDC_E_CORRUPT);
    }

    /* Bad version */
    {
        tdc_container_header h = make_homogeneous_header();
        h.version = (uint16_t)(TDC_CONTAINER_VERSION + 99);
        CHECK(tdc_container_header_validate(&h) == TDC_E_VERSION);
    }

    /* Reserved bytes must be zero */
    {
        tdc_container_header h = make_homogeneous_header();
        h._reserved0 = 1;
        CHECK(tdc_container_header_validate(&h) == TDC_E_CORRUPT);
    }
    /* schema_size (formerly _reserved1) is no longer a reserved field —
     * any value is structurally valid at the header-validate layer. */

    /* Heterogeneous flag set but global_dtype non-zero */
    {
        tdc_container_header h = make_heterogeneous_header();
        h.global_dtype = (uint8_t)TDC_DT_I32;
        CHECK(tdc_container_header_validate(&h) == TDC_E_CORRUPT);
    }
    {
        tdc_container_header h = make_heterogeneous_header();
        h.global_dim[0] = 1;
        CHECK(tdc_container_header_validate(&h) == TDC_E_CORRUPT);
    }

    /* Homogeneous: rank disagrees with layout */
    {
        tdc_container_header h = make_homogeneous_header();
        h.global_rank = 3;  /* RASTER_2D demands rank 2 */
        CHECK(tdc_container_header_validate(&h) == TDC_E_CORRUPT);
    }

    /* Homogeneous: trailing dim slot non-zero */
    {
        tdc_container_header h = make_homogeneous_header();
        h.global_dim[2] = 7;  /* rank 2, slot 2 must be 0 */
        CHECK(tdc_container_header_validate(&h) == TDC_E_CORRUPT);
    }

    /* Homogeneous: negative dim */
    {
        tdc_container_header h = make_homogeneous_header();
        h.global_dim[0] = -1;
        CHECK(tdc_container_header_validate(&h) == TDC_E_CORRUPT);
    }

    /* Index size not a multiple of entry size */
    {
        tdc_container_header h = make_homogeneous_header();
        h.index_size = 4 * TDC_INDEX_ENTRY_SIZE + 3;
        CHECK(tdc_container_header_validate(&h) == TDC_E_CORRUPT);
    }

    /* Index size doesn't match n_blocks */
    {
        tdc_container_header h = make_homogeneous_header();
        h.index_size = 5 * TDC_INDEX_ENTRY_SIZE;  /* h.n_blocks == 4 */
        CHECK(tdc_container_header_validate(&h) == TDC_E_CORRUPT);
    }

    /* n_blocks == 0 must mean zero index */
    {
        tdc_container_header h = make_heterogeneous_header();
        h.n_blocks    = 0;
        h.index_size  = TDC_INDEX_ENTRY_SIZE;  /* should be 0 */
        CHECK(tdc_container_header_validate(&h) == TDC_E_CORRUPT);
    }

    return 0;
}

/* ----- tdc_block_record_validate (smoke) ---------------------------------- */

static int test_block_record_smoke(void) {
    /* Build a minimal valid record: RAW + NONE chain + LZ, RASTER_2D
     * 16x16 of i32. Mirrors what api/encode.c writes for the simplest
     * happy case. */
    tdc_block_record r;
    memset(&r, 0, sizeof(r));
    r.magic             = TDC_BLOCK_MAGIC;
    r.version           = TDC_BLOCK_VERSION;
    r.flags             = 0;
    r.model_id          = (uint16_t)TDC_MODEL_RAW;
    r.entropy_ids[0]    = (uint16_t)TDC_ENTROPY_LZ;
    r.dtype             = (uint8_t)TDC_DT_I32;
    r.layout            = (uint8_t)TDC_LAYOUT_RASTER_2D;
    r.rank              = 2;
    r.dim[0]            = 16;
    r.dim[1]            = 16;
    r.uncompressed_size = 16 * 16 * 4;
    r.payload_size      = 64;  /* arbitrary */
    CHECK(tdc_block_record_validate(&r) == TDC_OK);

    /* Bad magic */
    {
        tdc_block_record bad = r;
        bad.magic = 0;
        CHECK(tdc_block_record_validate(&bad) == TDC_E_CORRUPT);
    }
    /* Validity flag without validity_size */
    {
        tdc_block_record bad = r;
        bad.flags = TDC_BLOCK_FLAG_HAS_VALIDITY;
        CHECK(tdc_block_record_validate(&bad) == TDC_E_CORRUPT);
    }
    /* Sticky chain terminator */
    {
        tdc_block_record bad = r;
        bad.xform_ids[0] = (uint16_t)TDC_XFORM_NONE;
        bad.xform_ids[1] = (uint16_t)TDC_XFORM_ZIGZAG;  /* after NONE — illegal */
        CHECK(tdc_block_record_validate(&bad) == TDC_E_CORRUPT);
    }
    return 0;
}

int main(void) {
    if (test_strerror())                return 1;
    if (test_container_header_happy())  return 1;
    if (test_container_header_rejects()) return 1;
    if (test_block_record_smoke())      return 1;
    printf("test_format_validate: ok\n");
    return 0;
}
