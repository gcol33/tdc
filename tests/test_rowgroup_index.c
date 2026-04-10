/*
 * tests/test_rowgroup_index.c
 *
 * Round-trip and edge-case tests for the row-group index
 * (serialize -> parse -> verify all fields).
 */

#include "../src/format/rowgroup_internal.h"
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
 * Fields are filled with deterministic values derived from indices. */
static tdc_rowgroup_index make_test_index(uint64_t n_rg, uint16_t n_cols) {
    tdc_rowgroup_index idx;
    idx.n_rowgroups = n_rg;

    if (n_rg == 0) {
        idx.entries = NULL;
        return idx;
    }

    idx.entries = (tdc_rowgroup_entry *)test_realloc(
        NULL, NULL, (size_t)n_rg * sizeof(tdc_rowgroup_entry));
    memset(idx.entries, 0, (size_t)n_rg * sizeof(tdc_rowgroup_entry));

    for (uint64_t i = 0; i < n_rg; ++i) {
        idx.entries[i].offset = 1000 + i * 500;
        idx.entries[i].n_rows = 100 + i * 10;
        idx.entries[i].n_cols = n_cols;

        idx.entries[i].columns = (tdc_rg_col_entry *)test_realloc(
            NULL, NULL, (size_t)n_cols * sizeof(tdc_rg_col_entry));

        for (uint16_t c = 0; c < n_cols; ++c) {
            idx.entries[i].columns[c].block_offset = i * 10000 + c * 80;
            idx.entries[i].columns[c].block_total  = 80 + c * 16;
        }
    }
    return idx;
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
    tdc_status st = tdc_rowgroup_index_parse(buf, sz, test_realloc, NULL, &dst);
    CHECK(st == TDC_OK);

    /* Verify. */
    CHECK(dst.n_rowgroups == 3);
    for (uint64_t i = 0; i < 3; ++i) {
        CHECK(dst.entries[i].offset == src.entries[i].offset);
        CHECK(dst.entries[i].n_rows == src.entries[i].n_rows);
        CHECK(dst.entries[i].n_cols == 4);
        for (uint16_t c = 0; c < 4; ++c) {
            CHECK(dst.entries[i].columns[c].block_offset ==
                  src.entries[i].columns[c].block_offset);
            CHECK(dst.entries[i].columns[c].block_total ==
                  src.entries[i].columns[c].block_total);
        }
    }

    /* Cleanup. */
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
    tdc_status st = tdc_rowgroup_index_parse(buf, sz, test_realloc, NULL, &dst);
    CHECK(st == TDC_OK);
    CHECK(dst.n_rowgroups == 0);
    CHECK(dst.entries == NULL);

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
    tdc_status st = tdc_rowgroup_index_parse(buf, sz, test_realloc, NULL, &dst);
    CHECK(st == TDC_OK);
    CHECK(dst.n_rowgroups == 1);
    CHECK(dst.entries[0].n_cols == 1);
    CHECK(dst.entries[0].offset == src.entries[0].offset);
    CHECK(dst.entries[0].n_rows == src.entries[0].n_rows);
    CHECK(dst.entries[0].columns[0].block_offset ==
          src.entries[0].columns[0].block_offset);
    CHECK(dst.entries[0].columns[0].block_total ==
          src.entries[0].columns[0].block_total);

    free_test_index(&dst);
    free_test_index(&src);
    test_realloc(NULL, buf, 0);
    return 0;
}

/* Truncated buffer: too small to hold even n_rowgroups. */
static int test_parse_truncated_header(void) {
    uint8_t buf[4] = {1, 0, 0, 0};
    tdc_rowgroup_index out;
    tdc_status st = tdc_rowgroup_index_parse(buf, 4, test_realloc, NULL, &out);
    CHECK(st == TDC_E_CORRUPT);
    CHECK(out.n_rowgroups == 0);
    CHECK(out.entries == NULL);
    return 0;
}

/* n_rowgroups claims more groups than buf can possibly hold. */
static int test_parse_n_rowgroups_overflow(void) {
    uint8_t buf[8];
    /* Claim 1000 row groups but only provide 8 bytes total. */
    uint64_t big = 1000;
    memcpy(buf, &big, 8);

    tdc_rowgroup_index out;
    tdc_status st = tdc_rowgroup_index_parse(buf, 8, test_realloc, NULL, &out);
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
    tdc_status st = tdc_rowgroup_index_parse(buf, short_sz, test_realloc, NULL, &out);
    CHECK(st == TDC_E_CORRUPT);
    CHECK(out.entries == NULL);

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
    tdc_status st = tdc_rowgroup_index_parse(buf, sz, test_realloc, NULL, &out);
    CHECK(st == TDC_E_CORRUPT);

    free_test_index(&src);
    test_realloc(NULL, buf, 0);
    return 0;
}

/* Non-zero padding or reserved field is rejected. */
static int test_parse_nonzero_reserved(void) {
    tdc_rowgroup_index src = make_test_index(1, 1);
    size_t sz = tdc_rowgroup_index_serialized_size(&src);
    uint8_t *buf = (uint8_t *)test_realloc(NULL, NULL, sz);
    tdc_rowgroup_index_serialize(&src, buf);

    /* _pad is at offset 8 + 18 = 26. Poke a non-zero byte. */
    buf[26] = 0xFF;

    tdc_rowgroup_index out;
    tdc_status st = tdc_rowgroup_index_parse(buf, sz, test_realloc, NULL, &out);
    CHECK(st == TDC_E_CORRUPT);

    /* Restore pad, break _reserved (offset 8 + 20 = 28). */
    tdc_rowgroup_index_serialize(&src, buf);  /* re-serialize clean */
    buf[28] = 0x01;

    st = tdc_rowgroup_index_parse(buf, sz, test_realloc, NULL, &out);
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
    CHECK(tdc_rowgroup_index_parse(NULL, 0, test_realloc, NULL, &out) == TDC_E_INVAL);
    CHECK(tdc_rowgroup_index_parse((const uint8_t *)"", 0, NULL, NULL, &out) == TDC_E_INVAL);

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
    RUN(test_roundtrip_empty);
    RUN(test_roundtrip_1rg_1col);
    RUN(test_parse_truncated_header);
    RUN(test_parse_n_rowgroups_overflow);
    RUN(test_parse_truncated_columns);
    RUN(test_parse_zero_cols);
    RUN(test_parse_nonzero_reserved);
    RUN(test_null_args);

    #undef RUN

    return fail;
}
