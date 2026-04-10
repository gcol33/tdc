/*
 * tests/test_schema.c
 *
 * Round-trip and edge-case tests for the column schema section.
 *
 * Coverage:
 *   1. 3-column schema: serialize -> parse -> field-by-field verify.
 *   2. 0-column schema (empty, valid).
 *   3. Column with maximum-length name (UINT16_MAX).
 *   4. Column without annotation (ann_len == 0).
 *   5. Truncated buffer rejected as corrupt.
 *   6. Unknown dtype rejected as corrupt.
 *   7. Zero-length name rejected as corrupt.
 */

#include "../src/format/schema_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while (0)

/* POSIX-style realloc wrapper matching the tdc allocation convention. */
static void *test_realloc(void *user, void *ptr, size_t new_size) {
    (void)user;
    if (new_size == 0) {
        free(ptr);
        return NULL;
    }
    return realloc(ptr, new_size);
}

/* ----- 3-column round-trip ------------------------------------------------ */

static int test_roundtrip_3cols(void) {
    tdc_column_desc cols[3];

    cols[0].name       = "temperature";
    cols[0].name_len   = 11;
    cols[0].dtype      = TDC_DT_F64;
    cols[0].annotation = "degrees Celsius";
    cols[0].ann_len    = 15;

    cols[1].name       = "id";
    cols[1].name_len   = 2;
    cols[1].dtype      = TDC_DT_I32;
    cols[1].annotation = NULL;
    cols[1].ann_len    = 0;

    cols[2].name       = "label";
    cols[2].name_len   = 5;
    cols[2].dtype      = TDC_DT_STRING;
    cols[2].annotation = "UTF-8 category";
    cols[2].ann_len    = 14;

    tdc_schema schema;
    schema.n_columns = 3;
    schema.columns   = cols;

    /* Compute size. */
    size_t sz = tdc_schema_serialized_size(&schema);
    /* 2 (n_columns) + 3 * (2 + name + 1 + 2 + ann) */
    size_t expected = 2
        + (2 + 11 + 1 + 2 + 15)
        + (2 +  2 + 1 + 2 +  0)
        + (2 +  5 + 1 + 2 + 14);
    CHECK(sz == expected);

    /* Serialize. */
    uint8_t *buf = (uint8_t *)malloc(sz);
    CHECK(buf != NULL);
    size_t written = tdc_schema_serialize(&schema, buf);
    CHECK(written == sz);

    /* Parse back. */
    uint16_t out_n = 0;
    tdc_column_desc *out_cols = NULL;
    tdc_status st = tdc_schema_parse(buf, sz, test_realloc, NULL,
                                     &out_n, &out_cols);
    CHECK(st == TDC_OK);
    CHECK(out_n == 3);
    CHECK(out_cols != NULL);

    /* Verify column 0. */
    CHECK(out_cols[0].name_len == 11);
    CHECK(memcmp(out_cols[0].name, "temperature", 11) == 0);
    CHECK(out_cols[0].name[11] == '\0');
    CHECK(out_cols[0].dtype == TDC_DT_F64);
    CHECK(out_cols[0].ann_len == 15);
    CHECK(memcmp(out_cols[0].annotation, "degrees Celsius", 15) == 0);
    CHECK(out_cols[0].annotation[15] == '\0');

    /* Verify column 1. */
    CHECK(out_cols[1].name_len == 2);
    CHECK(memcmp(out_cols[1].name, "id", 2) == 0);
    CHECK(out_cols[1].dtype == TDC_DT_I32);
    CHECK(out_cols[1].ann_len == 0);
    CHECK(out_cols[1].annotation == NULL);

    /* Verify column 2. */
    CHECK(out_cols[2].name_len == 5);
    CHECK(memcmp(out_cols[2].name, "label", 5) == 0);
    CHECK(out_cols[2].dtype == TDC_DT_STRING);
    CHECK(out_cols[2].ann_len == 14);
    CHECK(memcmp(out_cols[2].annotation, "UTF-8 category", 14) == 0);

    tdc_schema_free(out_cols, out_n, test_realloc, NULL);
    free(buf);
    return 0;
}

/* ----- 0-column schema ---------------------------------------------------- */

static int test_zero_columns(void) {
    tdc_schema schema;
    schema.n_columns = 0;
    schema.columns   = NULL;

    /* Serialized size of an empty schema is 0. */
    CHECK(tdc_schema_serialized_size(&schema) == 0);

    /* Serialize returns 0 written. */
    uint8_t dummy[4];
    CHECK(tdc_schema_serialize(&schema, dummy) == 0);

    /* Parse a 2-byte buffer with n_columns == 0. */
    uint8_t buf[2] = {0, 0};
    uint16_t out_n = 99;
    tdc_column_desc *out_cols = (tdc_column_desc *)(uintptr_t)1;
    tdc_status st = tdc_schema_parse(buf, 2, test_realloc, NULL,
                                     &out_n, &out_cols);
    CHECK(st == TDC_OK);
    CHECK(out_n == 0);
    CHECK(out_cols == NULL);

    return 0;
}

/* ----- NULL schema -------------------------------------------------------- */

static int test_null_schema(void) {
    CHECK(tdc_schema_serialized_size(NULL) == 0);
    CHECK(tdc_schema_serialize(NULL, NULL) == 0);
    return 0;
}

/* ----- single column, no annotation --------------------------------------- */

static int test_single_no_annotation(void) {
    tdc_column_desc col;
    col.name       = "x";
    col.name_len   = 1;
    col.dtype      = TDC_DT_U8;
    col.annotation = NULL;
    col.ann_len    = 0;

    tdc_schema schema;
    schema.n_columns = 1;
    schema.columns   = &col;

    size_t sz = tdc_schema_serialized_size(&schema);
    /* 2 + (2 + 1 + 1 + 2 + 0) = 8 */
    CHECK(sz == 8);

    uint8_t buf[8];
    size_t written = tdc_schema_serialize(&schema, buf);
    CHECK(written == 8);

    uint16_t out_n = 0;
    tdc_column_desc *out_cols = NULL;
    CHECK(tdc_schema_parse(buf, sz, test_realloc, NULL,
                           &out_n, &out_cols) == TDC_OK);
    CHECK(out_n == 1);
    CHECK(out_cols[0].name_len == 1);
    CHECK(out_cols[0].name[0] == 'x');
    CHECK(out_cols[0].dtype == TDC_DT_U8);
    CHECK(out_cols[0].ann_len == 0);
    CHECK(out_cols[0].annotation == NULL);

    tdc_schema_free(out_cols, out_n, test_realloc, NULL);
    return 0;
}

/* ----- truncated buffer rejected ------------------------------------------ */

static int test_truncated_corrupt(void) {
    /* Build a valid 1-column buffer, then try parsing with too few bytes. */
    tdc_column_desc col;
    col.name       = "abc";
    col.name_len   = 3;
    col.dtype      = TDC_DT_F32;
    col.annotation = NULL;
    col.ann_len    = 0;

    tdc_schema schema;
    schema.n_columns = 1;
    schema.columns   = &col;

    size_t sz = tdc_schema_serialized_size(&schema);
    uint8_t *buf = (uint8_t *)malloc(sz);
    CHECK(buf != NULL);
    tdc_schema_serialize(&schema, buf);

    uint16_t out_n;
    tdc_column_desc *out_cols;

    /* Truncate at every possible length < full. */
    for (size_t trunc = 0; trunc < sz; ++trunc) {
        out_n    = 99;
        out_cols = NULL;
        tdc_status st = tdc_schema_parse(buf, trunc, test_realloc, NULL,
                                         &out_n, &out_cols);
        CHECK(st == TDC_E_CORRUPT);
        CHECK(out_cols == NULL);
    }

    /* Full buffer must succeed. */
    CHECK(tdc_schema_parse(buf, sz, test_realloc, NULL,
                           &out_n, &out_cols) == TDC_OK);
    tdc_schema_free(out_cols, out_n, test_realloc, NULL);

    free(buf);
    return 0;
}

/* ----- unknown dtype rejected --------------------------------------------- */

static int test_unknown_dtype(void) {
    /* Manually craft a buffer: 1 column, name "x", dtype 0 (invalid). */
    uint8_t buf[] = {
        0x01, 0x00,   /* n_columns = 1 */
        0x01, 0x00,   /* name_len = 1 */
        'x',          /* name */
        0x00,         /* dtype = 0 (unknown) */
        0x00, 0x00    /* ann_len = 0 */
    };
    uint16_t out_n;
    tdc_column_desc *out_cols;
    CHECK(tdc_schema_parse(buf, sizeof(buf), test_realloc, NULL,
                           &out_n, &out_cols) == TDC_E_CORRUPT);

    /* dtype 13 is also unknown. */
    buf[5] = 13;
    CHECK(tdc_schema_parse(buf, sizeof(buf), test_realloc, NULL,
                           &out_n, &out_cols) == TDC_E_CORRUPT);

    /* dtype 255 is also unknown. */
    buf[5] = 255;
    CHECK(tdc_schema_parse(buf, sizeof(buf), test_realloc, NULL,
                           &out_n, &out_cols) == TDC_E_CORRUPT);

    return 0;
}

/* ----- zero-length name rejected ------------------------------------------ */

static int test_zero_name_corrupt(void) {
    /* 1 column with name_len = 0. */
    uint8_t buf[] = {
        0x01, 0x00,   /* n_columns = 1 */
        0x00, 0x00,   /* name_len = 0 (invalid) */
        0x01,         /* dtype = TDC_DT_I8 */
        0x00, 0x00    /* ann_len = 0 */
    };
    uint16_t out_n;
    tdc_column_desc *out_cols;
    CHECK(tdc_schema_parse(buf, sizeof(buf), test_realloc, NULL,
                           &out_n, &out_cols) == TDC_E_CORRUPT);
    return 0;
}

/* ----- trailing garbage rejected ------------------------------------------ */

static int test_trailing_garbage(void) {
    /* Valid 1-column buffer + 1 extra byte. Parser must reject because
     * pos != buf_size at the end. */
    tdc_column_desc col;
    col.name       = "y";
    col.name_len   = 1;
    col.dtype      = TDC_DT_I16;
    col.annotation = NULL;
    col.ann_len    = 0;

    tdc_schema schema;
    schema.n_columns = 1;
    schema.columns   = &col;

    size_t sz = tdc_schema_serialized_size(&schema);
    uint8_t *buf = (uint8_t *)malloc(sz + 1);
    CHECK(buf != NULL);
    tdc_schema_serialize(&schema, buf);
    buf[sz] = 0xFF;  /* trailing garbage */

    uint16_t out_n;
    tdc_column_desc *out_cols;
    CHECK(tdc_schema_parse(buf, sz + 1, test_realloc, NULL,
                           &out_n, &out_cols) == TDC_E_CORRUPT);

    free(buf);
    return 0;
}

/* ----- max-length name ---------------------------------------------------- */

static int test_max_name_length(void) {
    /* Use a 65535-byte name (UINT16_MAX). This is the largest representable
     * name_len in the wire format. We do NOT allocate 64 KB on the stack;
     * heap-allocate and fill with 'A'. */
    uint16_t max_len = UINT16_MAX;
    char *big_name = (char *)malloc((size_t)max_len);
    CHECK(big_name != NULL);
    memset(big_name, 'A', max_len);

    tdc_column_desc col;
    col.name       = big_name;
    col.name_len   = max_len;
    col.dtype      = TDC_DT_F32;
    col.annotation = NULL;
    col.ann_len    = 0;

    tdc_schema schema;
    schema.n_columns = 1;
    schema.columns   = &col;

    size_t sz = tdc_schema_serialized_size(&schema);
    /* 2 + (2 + 65535 + 1 + 2 + 0) = 65542 */
    CHECK(sz == 2 + 2 + (size_t)max_len + 1 + 2 + 0);

    uint8_t *buf = (uint8_t *)malloc(sz);
    CHECK(buf != NULL);
    size_t written = tdc_schema_serialize(&schema, buf);
    CHECK(written == sz);

    uint16_t out_n = 0;
    tdc_column_desc *out_cols = NULL;
    CHECK(tdc_schema_parse(buf, sz, test_realloc, NULL,
                           &out_n, &out_cols) == TDC_OK);
    CHECK(out_n == 1);
    CHECK(out_cols[0].name_len == max_len);
    CHECK(memcmp(out_cols[0].name, big_name, max_len) == 0);
    CHECK(out_cols[0].name[max_len] == '\0');

    tdc_schema_free(out_cols, out_n, test_realloc, NULL);
    free(buf);
    free(big_name);
    return 0;
}

/* ----- all valid dtypes accepted ------------------------------------------ */

static int test_all_valid_dtypes(void) {
    uint8_t valid_dtypes[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    for (size_t d = 0; d < sizeof(valid_dtypes); ++d) {
        /* Manually craft a 1-column buffer with this dtype. */
        uint8_t buf[] = {
            0x01, 0x00,          /* n_columns = 1 */
            0x01, 0x00,          /* name_len = 1 */
            'z',                 /* name */
            valid_dtypes[d],     /* dtype */
            0x00, 0x00           /* ann_len = 0 */
        };
        uint16_t out_n;
        tdc_column_desc *out_cols;
        tdc_status st = tdc_schema_parse(buf, sizeof(buf), test_realloc,
                                         NULL, &out_n, &out_cols);
        CHECK(st == TDC_OK);
        CHECK(out_cols[0].dtype == valid_dtypes[d]);
        tdc_schema_free(out_cols, out_n, test_realloc, NULL);
    }
    return 0;
}

/* ----- main --------------------------------------------------------------- */

int main(void) {
    if (test_roundtrip_3cols())     return 1;
    if (test_zero_columns())        return 1;
    if (test_null_schema())         return 1;
    if (test_single_no_annotation()) return 1;
    if (test_truncated_corrupt())   return 1;
    if (test_unknown_dtype())       return 1;
    if (test_zero_name_corrupt())   return 1;
    if (test_trailing_garbage())    return 1;
    if (test_max_name_length())     return 1;
    if (test_all_valid_dtypes())    return 1;
    printf("test_schema: ok\n");
    return 0;
}
