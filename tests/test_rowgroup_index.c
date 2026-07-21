/*
 * tests/test_rowgroup_index.c
 *
 * Round-trip and edge-case tests for the row-group index
 * (serialize -> parse -> verify all fields), including the per-row-group
 * stats block and the ragged column counts a widened container produces.
 */

#include "../src/format/rowgroup_internal.h"
#include "../src/format/stats_internal.h"
#include "tdc/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while (0)

/* ---- realloc_fn wrapper -------------------------------------------------- */

static void *test_realloc(void *user, void *ptr, size_t new_size) {
    (void)user;
    if (new_size == 0) { free(ptr); return NULL; }
    return realloc(ptr, new_size);
}

/* ---- Helpers ------------------------------------------------------------- */

/* Build a test index: n_rg row groups, each with n_cols columns.
 * Fields are filled with deterministic values derived from indices.
 * With with_stats != 0 every group also gets a stats block. */
static tdc_rowgroup_index make_test_index_ex(uint64_t n_rg, uint16_t n_cols,
                                             int with_stats) {
    tdc_rowgroup_index idx;
    idx.n_rowgroups = n_rg;

    if (n_rg == 0) {
        idx.groups = NULL;
        return idx;
    }

    idx.groups = (tdc_rg_group *)test_realloc(
        NULL, NULL, (size_t)n_rg * sizeof(tdc_rg_group));
    memset(idx.groups, 0, (size_t)n_rg * sizeof(tdc_rg_group));

    for (uint64_t i = 0; i < n_rg; ++i) {
        tdc_rg_group *g = &idx.groups[i];
        g->entry.offset = 1000 + i * 500;
        g->entry.n_rows = 100 + i * 10;
        g->entry.n_cols = n_cols;

        g->entry.columns = (tdc_rg_col_entry *)test_realloc(
            NULL, NULL, (size_t)n_cols * sizeof(tdc_rg_col_entry));

        for (uint16_t c = 0; c < n_cols; ++c) {
            g->entry.columns[c].block_offset = i * 10000 + c * 80;
            g->entry.columns[c].block_total  = 80 + c * 16;
        }

        if (with_stats) {
            g->stats = (tdc_column_stats *)test_realloc(
                NULL, NULL, (size_t)n_cols * sizeof(tdc_column_stats));
            memset(g->stats, 0, (size_t)n_cols * sizeof(tdc_column_stats));
            for (uint16_t c = 0; c < n_cols; ++c) {
                /* Leave one column stats-less so the per-column has_stats
                 * byte is exercised in both states. */
                g->stats[c].has_stats  = (c == 0) ? 0 : 1;
                g->stats[c].null_count = i * 100 + c;
                g->stats[c].min[0]     = (uint8_t)(c + 1);
                g->stats[c].max[15]    = (uint8_t)(c + 2);
            }
            g->has_stats = 1;
        }
    }
    return idx;
}

static tdc_rowgroup_index make_test_index(uint64_t n_rg, uint16_t n_cols) {
    return make_test_index_ex(n_rg, n_cols, 0);
}

static void free_test_index(tdc_rowgroup_index *idx) {
    tdc_rowgroup_index_free(idx, test_realloc, NULL);
}

/* ---- Tests --------------------------------------------------------------- */

/* 3 row groups, 4 columns each: full round-trip. */
static int test_roundtrip_3rg_4col(void) {
    tdc_rowgroup_index src = make_test_index(3, 4);

    /* Serialize. */
    size_t sz = tdc_rowgroup_index_serialized_size(&src);
    CHECK(sz > 0);
    /* Expected: 8 + 3*(24 + 4*16) = 8 + 3*88 = 272 */
    CHECK(sz == 272);

    uint8_t *buf = (uint8_t *)test_realloc(NULL, NULL, sz);
    CHECK(buf != NULL);

    size_t written = tdc_rowgroup_index_serialize(&src, buf);
    CHECK(written == sz);

    /* Parse back. */
    tdc_rowgroup_index dst;
    tdc_status st = tdc_rowgroup_index_parse(buf, sz, 0, test_realloc, NULL, &dst);
    CHECK(st == TDC_OK);

    /* Verify. */
    CHECK(dst.n_rowgroups == 3);
    for (uint64_t i = 0; i < 3; ++i) {
        CHECK(dst.groups[i].entry.offset == src.groups[i].entry.offset);
        CHECK(dst.groups[i].entry.n_rows == src.groups[i].entry.n_rows);
        CHECK(dst.groups[i].entry.n_cols == 4);
        CHECK(dst.groups[i].has_stats == 0);
        for (uint16_t c = 0; c < 4; ++c) {
            CHECK(dst.groups[i].entry.columns[c].block_offset ==
                  src.groups[i].entry.columns[c].block_offset);
            CHECK(dst.groups[i].entry.columns[c].block_total ==
                  src.groups[i].entry.columns[c].block_total);
        }
    }

    /* Cleanup. */
    free_test_index(&dst);
    free_test_index(&src);
    test_realloc(NULL, buf, 0);
    return 0;
}

/* Stats blocks survive the round-trip, per column and per row group. */
static int test_roundtrip_with_stats(void) {
    tdc_rowgroup_index src = make_test_index_ex(3, 4, 1);

    size_t sz = tdc_rowgroup_index_serialized_size(&src);
    /* 8 + 3*(24 + 4*16 + 4*41) = 8 + 3*252 = 764 */
    CHECK(sz == 764);

    uint8_t *buf = (uint8_t *)test_realloc(NULL, NULL, sz);
    CHECK(buf != NULL);
    CHECK(tdc_rowgroup_index_serialize(&src, buf) == sz);

    tdc_rowgroup_index dst;
    tdc_status st = tdc_rowgroup_index_parse(buf, sz, 1, test_realloc, NULL, &dst);
    CHECK(st == TDC_OK);
    CHECK(dst.n_rowgroups == 3);

    for (uint64_t i = 0; i < 3; ++i) {
        CHECK(dst.groups[i].has_stats == 1);
        for (uint16_t c = 0; c < 4; ++c) {
            const tdc_column_stats *a = &src.groups[i].stats[c];
            const tdc_column_stats *b = &dst.groups[i].stats[c];
            CHECK(b->has_stats  == a->has_stats);
            CHECK(b->null_count == a->null_count);
            CHECK(memcmp(b->min, a->min, TDC_STATS_VALUE_SIZE) == 0);
            CHECK(memcmp(b->max, a->max, TDC_STATS_VALUE_SIZE) == 0);
        }
    }

    free_test_index(&dst);
    free_test_index(&src);
    test_realloc(NULL, buf, 0);
    return 0;
}

/* A stats block may not appear when the container flag is clear -- that
 * combination means the header and the index disagree. */
static int test_stats_without_container_flag(void) {
    tdc_rowgroup_index src = make_test_index_ex(1, 2, 1);
    size_t sz = tdc_rowgroup_index_serialized_size(&src);
    uint8_t *buf = (uint8_t *)test_realloc(NULL, NULL, sz);
    tdc_rowgroup_index_serialize(&src, buf);

    tdc_rowgroup_index out;
    tdc_status st = tdc_rowgroup_index_parse(buf, sz, 0, test_realloc, NULL, &out);
    CHECK(st == TDC_E_CORRUPT);
    CHECK(out.groups == NULL);

    free_test_index(&src);
    test_realloc(NULL, buf, 0);
    return 0;
}

/* A stats block whose size disagrees with n_cols is rejected. */
static int test_stats_size_mismatch(void) {
    tdc_rowgroup_index src = make_test_index_ex(1, 2, 1);
    size_t sz = tdc_rowgroup_index_serialized_size(&src);
    uint8_t *buf = (uint8_t *)test_realloc(NULL, NULL, sz);
    tdc_rowgroup_index_serialize(&src, buf);

    /* stats_size lives at 8 + 20 = 28; claim one entry too many. */
    uint32_t bad = 3 * TDC_STATS_ENTRY_SIZE;
    memcpy(buf + 28, &bad, 4);

    tdc_rowgroup_index out;
    tdc_status st = tdc_rowgroup_index_parse(buf, sz, 1, test_realloc, NULL, &out);
    CHECK(st == TDC_E_CORRUPT);

    free_test_index(&src);
    test_realloc(NULL, buf, 0);
    return 0;
}

/* Row groups may carry different column counts -- which is what a widened
 * container looks like mid-flight, and what the parser must not assume away. */
static int test_ragged_column_counts(void) {
    tdc_rowgroup_index src = make_test_index(3, 2);
    /* Drop the last group to a single column. */
    src.groups[2].entry.n_cols = 1;

    size_t sz = tdc_rowgroup_index_serialized_size(&src);
    /* 8 + 2*(24 + 2*16) + 1*(24 + 1*16) = 8 + 112 + 40 = 160 */
    CHECK(sz == 160);

    uint8_t *buf = (uint8_t *)test_realloc(NULL, NULL, sz);
    CHECK(tdc_rowgroup_index_serialize(&src, buf) == sz);

    tdc_rowgroup_index dst;
    CHECK(tdc_rowgroup_index_parse(buf, sz, 0, test_realloc, NULL, &dst) == TDC_OK);
    CHECK(dst.groups[0].entry.n_cols == 2);
    CHECK(dst.groups[1].entry.n_cols == 2);
    CHECK(dst.groups[2].entry.n_cols == 1);
    CHECK(dst.groups[2].entry.columns[0].block_offset ==
          src.groups[2].entry.columns[0].block_offset);

    free_test_index(&dst);
    free_test_index(&src);
    test_realloc(NULL, buf, 0);
    return 0;
}

/* 0 row groups: empty index round-trip. */
static int test_roundtrip_empty(void) {
    tdc_rowgroup_index src = make_test_index(0, 0);

    size_t sz = tdc_rowgroup_index_serialized_size(&src);
    CHECK(sz == 8);  /* just the u64 n_rowgroups */

    uint8_t *buf = (uint8_t *)test_realloc(NULL, NULL, sz);
    CHECK(buf != NULL);
    size_t written = tdc_rowgroup_index_serialize(&src, buf);
    CHECK(written == 8);

    tdc_rowgroup_index dst;
    tdc_status st = tdc_rowgroup_index_parse(buf, sz, 0, test_realloc, NULL, &dst);
    CHECK(st == TDC_OK);
    CHECK(dst.n_rowgroups == 0);
    CHECK(dst.groups == NULL);

    free_test_index(&dst);
    test_realloc(NULL, buf, 0);
    return 0;
}

/* 1 row group, 1 column: minimal non-empty case. */
static int test_roundtrip_1rg_1col(void) {
    tdc_rowgroup_index src = make_test_index(1, 1);

    size_t sz = tdc_rowgroup_index_serialized_size(&src);
    /* 8 + 1*(24 + 1*16) = 48 */
    CHECK(sz == 48);

    uint8_t *buf = (uint8_t *)test_realloc(NULL, NULL, sz);
    CHECK(buf != NULL);
    size_t written = tdc_rowgroup_index_serialize(&src, buf);
    CHECK(written == sz);

    tdc_rowgroup_index dst;
    tdc_status st = tdc_rowgroup_index_parse(buf, sz, 0, test_realloc, NULL, &dst);
    CHECK(st == TDC_OK);
    CHECK(dst.n_rowgroups == 1);
    CHECK(dst.groups[0].entry.n_cols == 1);
    CHECK(dst.groups[0].entry.offset == src.groups[0].entry.offset);
    CHECK(dst.groups[0].entry.n_rows == src.groups[0].entry.n_rows);
    CHECK(dst.groups[0].entry.columns[0].block_offset ==
          src.groups[0].entry.columns[0].block_offset);
    CHECK(dst.groups[0].entry.columns[0].block_total ==
          src.groups[0].entry.columns[0].block_total);

    free_test_index(&dst);
    free_test_index(&src);
    test_realloc(NULL, buf, 0);
    return 0;
}

/* Truncated buffer: too small to hold even n_rowgroups. */
static int test_parse_truncated_header(void) {
    uint8_t buf[4] = {1, 0, 0, 0};
    tdc_rowgroup_index out;
    tdc_status st = tdc_rowgroup_index_parse(buf, 4, 0, test_realloc, NULL, &out);
    CHECK(st == TDC_E_CORRUPT);
    CHECK(out.n_rowgroups == 0);
    CHECK(out.groups == NULL);
    return 0;
}

/* n_rowgroups claims more groups than buf can possibly hold. */
static int test_parse_n_rowgroups_overflow(void) {
    uint8_t buf[8];
    /* Claim 1000 row groups but only provide 8 bytes total. */
    uint64_t big = 1000;
    memcpy(buf, &big, 8);

    tdc_rowgroup_index out;
    tdc_status st = tdc_rowgroup_index_parse(buf, 8, 0, test_realloc, NULL, &out);
    CHECK(st == TDC_E_CORRUPT);
    return 0;
}

/* Buffer truncated mid-rowgroup (header present but column data missing). */
static int test_parse_truncated_columns(void) {
    /* Build a valid 1-rg, 4-col index, then truncate the buffer. */
    tdc_rowgroup_index src = make_test_index(1, 4);
    size_t full_sz = tdc_rowgroup_index_serialized_size(&src);
    uint8_t *buf = (uint8_t *)test_realloc(NULL, NULL, full_sz);
    tdc_rowgroup_index_serialize(&src, buf);

    /* Give only the first row group header + 1 column (need 4). */
    size_t short_sz = 8 + TDC_RG_HEADER_SIZE + 1 * TDC_RG_COL_SIZE;
    CHECK(short_sz < full_sz);

    tdc_rowgroup_index out;
    tdc_status st = tdc_rowgroup_index_parse(buf, short_sz, 0, test_realloc, NULL, &out);
    CHECK(st == TDC_E_CORRUPT);
    CHECK(out.groups == NULL);

    free_test_index(&src);
    test_realloc(NULL, buf, 0);
    return 0;
}

/* A stats block truncated by the buffer end is rejected, not read past. */
static int test_parse_truncated_stats(void) {
    tdc_rowgroup_index src = make_test_index_ex(1, 2, 1);
    size_t full_sz = tdc_rowgroup_index_serialized_size(&src);
    uint8_t *buf = (uint8_t *)test_realloc(NULL, NULL, full_sz);
    tdc_rowgroup_index_serialize(&src, buf);

    tdc_rowgroup_index out;
    tdc_status st = tdc_rowgroup_index_parse(buf, full_sz - 1, 1,
                                             test_realloc, NULL, &out);
    CHECK(st == TDC_E_CORRUPT);
    CHECK(out.groups == NULL);

    free_test_index(&src);
    test_realloc(NULL, buf, 0);
    return 0;
}

/* n_cols == 0 is invalid (must have at least one column). */
static int test_parse_zero_cols(void) {
    /* Serialize a 1-rg, 1-col index, then patch n_cols to 0. */
    tdc_rowgroup_index src = make_test_index(1, 1);
    size_t sz = tdc_rowgroup_index_serialized_size(&src);
    uint8_t *buf = (uint8_t *)test_realloc(NULL, NULL, sz);
    tdc_rowgroup_index_serialize(&src, buf);

    /* n_cols is at offset 8 (n_rowgroups) + 16 (offset+n_rows) = 24. */
    uint16_t zero = 0;
    memcpy(buf + 24, &zero, 2);

    tdc_rowgroup_index out;
    tdc_status st = tdc_rowgroup_index_parse(buf, sz, 0, test_realloc, NULL, &out);
    CHECK(st == TDC_E_CORRUPT);

    free_test_index(&src);
    test_realloc(NULL, buf, 0);
    return 0;
}

/* Non-zero padding, or a stats_size the container never advertised, is
 * rejected. */
static int test_parse_nonzero_reserved(void) {
    tdc_rowgroup_index src = make_test_index(1, 1);
    size_t sz = tdc_rowgroup_index_serialized_size(&src);
    uint8_t *buf = (uint8_t *)test_realloc(NULL, NULL, sz);
    tdc_rowgroup_index_serialize(&src, buf);

    /* _pad is at offset 8 + 18 = 26. Poke a non-zero byte. */
    buf[26] = 0xFF;

    tdc_rowgroup_index out;
    tdc_status st = tdc_rowgroup_index_parse(buf, sz, 0, test_realloc, NULL, &out);
    CHECK(st == TDC_E_CORRUPT);

    /* Restore pad, break stats_size (offset 8 + 20 = 28). */
    tdc_rowgroup_index_serialize(&src, buf);  /* re-serialize clean */
    buf[28] = 0x01;

    st = tdc_rowgroup_index_parse(buf, sz, 0, test_realloc, NULL, &out);
    CHECK(st == TDC_E_CORRUPT);

    free_test_index(&src);
    test_realloc(NULL, buf, 0);
    return 0;
}

/* NULL arguments to parse/serialize/free are handled gracefully. */
static int test_null_args(void) {
    CHECK(tdc_rowgroup_index_serialized_size(NULL) == 0);
    CHECK(tdc_rowgroup_index_serialize(NULL, NULL) == 0);

    tdc_rowgroup_index out;
    CHECK(tdc_rowgroup_index_parse(NULL, 0, 0, test_realloc, NULL, &out) == TDC_E_INVAL);
    CHECK(tdc_rowgroup_index_parse((const uint8_t *)"", 0, 0, NULL, NULL, &out) == TDC_E_INVAL);

    /* free with NULL idx or NULL realloc_fn should not crash. */
    tdc_rowgroup_index_free(NULL, test_realloc, NULL);
    tdc_rowgroup_index_free(&out, NULL, NULL);

    return 0;
}

/* ---- Main ---------------------------------------------------------------- */

int main(void) {
    int fail = 0;

    #define RUN(fn) do { \
        if (fn()) { fprintf(stderr, "  FAILED: %s\n", #fn); fail = 1; } \
        else       { fprintf(stderr, "  passed: %s\n", #fn); } \
    } while (0)

    fprintf(stderr, "test_rowgroup_index:\n");
    RUN(test_roundtrip_3rg_4col);
    RUN(test_roundtrip_with_stats);
    RUN(test_stats_without_container_flag);
    RUN(test_stats_size_mismatch);
    RUN(test_ragged_column_counts);
    RUN(test_roundtrip_empty);
    RUN(test_roundtrip_1rg_1col);
    RUN(test_parse_truncated_header);
    RUN(test_parse_n_rowgroups_overflow);
    RUN(test_parse_truncated_columns);
    RUN(test_parse_truncated_stats);
    RUN(test_parse_zero_cols);
    RUN(test_parse_nonzero_reserved);
    RUN(test_null_args);

    #undef RUN

    return fail;
}
