/*
 * tests/test_stats.c — column statistics: compute, serialize, parse
 *
 * Coverage:
 *   - tdc_stats_compute on i32, f64, u8, i8 (negative values)
 *   - tdc_stats_store_value / tdc_stats_load_value round-trip
 *   - tdc_stats_serialize + tdc_stats_parse round-trip
 *   - Edge cases: single element, all same values, empty block
 */

#include "tdc/types.h"
#include "../src/format/stats_internal.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while (0)

/* ---- helpers: build a 1D block from a typed array ----------------------- */

static tdc_block make_block_1d(void *data, tdc_dtype dtype, int64_t n) {
    tdc_block blk;
    memset(&blk, 0, sizeof(blk));
    blk.data    = data;
    blk.offsets = NULL;
    blk.dtype   = dtype;
    blk.layout  = TDC_LAYOUT_VECTOR_1D;
    blk.shape.rank   = 1;
    blk.shape.dim[0] = n;
    blk.shape.dim[1] = 1;
    blk.shape.dim[2] = 1;
    tdc_shape_set_contiguous(&blk.shape);
    blk.validity = NULL;
    return blk;
}

/* ---- test: i32 basic ---------------------------------------------------- */

static int test_i32_basic(void) {
    int32_t data[] = { 10, -5, 42, 0, -100, 7 };
    tdc_block blk = make_block_1d(data, TDC_DT_I32, 6);

    tdc_column_stats st;
    CHECK(tdc_stats_compute(&blk, &st) == TDC_OK);
    CHECK(st.has_stats == 1);

    int32_t got_min, got_max;
    tdc_stats_load_value(st.min, &got_min, TDC_DT_I32);
    tdc_stats_load_value(st.max, &got_max, TDC_DT_I32);
    CHECK(got_min == -100);
    CHECK(got_max == 42);

    return 0;
}

/* ---- test: f64 basic ---------------------------------------------------- */

static int test_f64_basic(void) {
    double data[] = { 3.14, -2.71, 0.0, 100.5, -0.001 };
    tdc_block blk = make_block_1d(data, TDC_DT_F64, 5);

    tdc_column_stats st;
    CHECK(tdc_stats_compute(&blk, &st) == TDC_OK);
    CHECK(st.has_stats == 1);

    double got_min, got_max;
    tdc_stats_load_value(st.min, &got_min, TDC_DT_F64);
    tdc_stats_load_value(st.max, &got_max, TDC_DT_F64);
    CHECK(got_min == -2.71);
    CHECK(got_max == 100.5);

    return 0;
}

/* ---- test: u8 basic ----------------------------------------------------- */

static int test_u8_basic(void) {
    uint8_t data[] = { 255, 0, 128, 1, 254 };
    tdc_block blk = make_block_1d(data, TDC_DT_U8, 5);

    tdc_column_stats st;
    CHECK(tdc_stats_compute(&blk, &st) == TDC_OK);
    CHECK(st.has_stats == 1);

    uint8_t got_min, got_max;
    tdc_stats_load_value(st.min, &got_min, TDC_DT_U8);
    tdc_stats_load_value(st.max, &got_max, TDC_DT_U8);
    CHECK(got_min == 0);
    CHECK(got_max == 255);

    return 0;
}

/* ---- test: i8 negative values ------------------------------------------- */

static int test_i8_negatives(void) {
    int8_t data[] = { -128, -1, 0, 1, 127 };
    tdc_block blk = make_block_1d(data, TDC_DT_I8, 5);

    tdc_column_stats st;
    CHECK(tdc_stats_compute(&blk, &st) == TDC_OK);
    CHECK(st.has_stats == 1);

    int8_t got_min, got_max;
    tdc_stats_load_value(st.min, &got_min, TDC_DT_I8);
    tdc_stats_load_value(st.max, &got_max, TDC_DT_I8);
    CHECK(got_min == -128);
    CHECK(got_max == 127);

    return 0;
}

/* ---- test: single element ----------------------------------------------- */

static int test_single_element(void) {
    int32_t data[] = { 42 };
    tdc_block blk = make_block_1d(data, TDC_DT_I32, 1);

    tdc_column_stats st;
    CHECK(tdc_stats_compute(&blk, &st) == TDC_OK);
    CHECK(st.has_stats == 1);

    int32_t got_min, got_max;
    tdc_stats_load_value(st.min, &got_min, TDC_DT_I32);
    tdc_stats_load_value(st.max, &got_max, TDC_DT_I32);
    CHECK(got_min == 42);
    CHECK(got_max == 42);

    return 0;
}

/* ---- test: all same values ---------------------------------------------- */

static int test_all_same(void) {
    int32_t data[] = { 7, 7, 7, 7 };
    tdc_block blk = make_block_1d(data, TDC_DT_I32, 4);

    tdc_column_stats st;
    CHECK(tdc_stats_compute(&blk, &st) == TDC_OK);
    CHECK(st.has_stats == 1);

    int32_t got_min, got_max;
    tdc_stats_load_value(st.min, &got_min, TDC_DT_I32);
    tdc_stats_load_value(st.max, &got_max, TDC_DT_I32);
    CHECK(got_min == 7);
    CHECK(got_max == 7);

    return 0;
}

/* ---- test: empty block -------------------------------------------------- */

static int test_empty_block(void) {
    tdc_block blk = make_block_1d(NULL, TDC_DT_I32, 0);

    tdc_column_stats st;
    memset(&st, 0xFF, sizeof(st)); /* fill with garbage to verify reset */
    CHECK(tdc_stats_compute(&blk, &st) == TDC_OK);
    CHECK(st.has_stats == 0);

    return 0;
}

/* ---- test: serialize + parse round-trip --------------------------------- */

static int test_serialize_roundtrip(void) {
    /* Build stats for three "columns" with different dtypes. */
    int32_t i32_data[] = { 10, -5, 42 };
    double  f64_data[] = { 3.14, -2.71, 100.5 };
    uint8_t u8_data[]  = { 255, 0, 128 };

    tdc_block blk_i32 = make_block_1d(i32_data, TDC_DT_I32, 3);
    tdc_block blk_f64 = make_block_1d(f64_data, TDC_DT_F64, 3);
    tdc_block blk_u8  = make_block_1d(u8_data,  TDC_DT_U8,  3);

    tdc_column_stats stats[3];
    CHECK(tdc_stats_compute(&blk_i32, &stats[0]) == TDC_OK);
    CHECK(tdc_stats_compute(&blk_f64, &stats[1]) == TDC_OK);
    CHECK(tdc_stats_compute(&blk_u8,  &stats[2]) == TDC_OK);

    /* Serialize. */
    uint8_t buf[3 * TDC_STATS_ENTRY_SIZE];
    size_t written = tdc_stats_serialize(stats, 3, buf);
    CHECK(written == 3 * TDC_STATS_ENTRY_SIZE);

    /* Parse back. */
    tdc_column_stats parsed[3];
    CHECK(tdc_stats_parse(buf, written, 3, parsed) == TDC_OK);

    /* Verify byte-exact match. */
    for (int i = 0; i < 3; ++i) {
        CHECK(parsed[i].has_stats == stats[i].has_stats);
        CHECK(memcmp(parsed[i].min, stats[i].min, TDC_STATS_VALUE_SIZE) == 0);
        CHECK(memcmp(parsed[i].max, stats[i].max, TDC_STATS_VALUE_SIZE) == 0);
    }

    /* Verify parsed values are correct. */
    int32_t i32_min, i32_max;
    tdc_stats_load_value(parsed[0].min, &i32_min, TDC_DT_I32);
    tdc_stats_load_value(parsed[0].max, &i32_max, TDC_DT_I32);
    CHECK(i32_min == -5);
    CHECK(i32_max == 42);

    double f64_min, f64_max;
    tdc_stats_load_value(parsed[1].min, &f64_min, TDC_DT_F64);
    tdc_stats_load_value(parsed[1].max, &f64_max, TDC_DT_F64);
    CHECK(f64_min == -2.71);
    CHECK(f64_max == 100.5);

    uint8_t u8_min, u8_max;
    tdc_stats_load_value(parsed[2].min, &u8_min, TDC_DT_U8);
    tdc_stats_load_value(parsed[2].max, &u8_max, TDC_DT_U8);
    CHECK(u8_min == 0);
    CHECK(u8_max == 255);

    return 0;
}

/* ---- test: parse rejects short buffer ----------------------------------- */

static int test_parse_short_buffer(void) {
    uint8_t buf[TDC_STATS_ENTRY_SIZE]; /* only 1 entry worth */
    tdc_column_stats out[2];
    CHECK(tdc_stats_parse(buf, sizeof(buf), 2, out) == TDC_E_BUF_TOO_SMALL);
    return 0;
}

/* ---- test: store/load round-trip for remaining dtypes ------------------- */

static int test_store_load_all_dtypes(void) {
    /* i16 */
    {
        int16_t v = -1234;
        uint8_t slot[TDC_STATS_VALUE_SIZE];
        tdc_stats_store_value(slot, &v, TDC_DT_I16);
        int16_t got;
        tdc_stats_load_value(slot, &got, TDC_DT_I16);
        CHECK(got == v);
        /* Trailing bytes must be zero. */
        for (int i = 2; i < TDC_STATS_VALUE_SIZE; ++i) CHECK(slot[i] == 0);
    }
    /* u16 */
    {
        uint16_t v = 60000;
        uint8_t slot[TDC_STATS_VALUE_SIZE];
        tdc_stats_store_value(slot, &v, TDC_DT_U16);
        uint16_t got;
        tdc_stats_load_value(slot, &got, TDC_DT_U16);
        CHECK(got == v);
    }
    /* u32 */
    {
        uint32_t v = 3000000000u;
        uint8_t slot[TDC_STATS_VALUE_SIZE];
        tdc_stats_store_value(slot, &v, TDC_DT_U32);
        uint32_t got;
        tdc_stats_load_value(slot, &got, TDC_DT_U32);
        CHECK(got == v);
    }
    /* i64 */
    {
        int64_t v = -9000000000LL;
        uint8_t slot[TDC_STATS_VALUE_SIZE];
        tdc_stats_store_value(slot, &v, TDC_DT_I64);
        int64_t got;
        tdc_stats_load_value(slot, &got, TDC_DT_I64);
        CHECK(got == v);
    }
    /* u64 */
    {
        uint64_t v = 18000000000000000000ULL;
        uint8_t slot[TDC_STATS_VALUE_SIZE];
        tdc_stats_store_value(slot, &v, TDC_DT_U64);
        uint64_t got;
        tdc_stats_load_value(slot, &got, TDC_DT_U64);
        CHECK(got == v);
    }
    /* f32 */
    {
        float v = -3.14f;
        uint8_t slot[TDC_STATS_VALUE_SIZE];
        tdc_stats_store_value(slot, &v, TDC_DT_F32);
        float got;
        tdc_stats_load_value(slot, &got, TDC_DT_F32);
        CHECK(got == v);
    }
    return 0;
}

/* ---- test: f64 with negative zero --------------------------------------- */

static int test_f64_negative_zero(void) {
    double data[] = { -0.0, 0.0 };
    tdc_block blk = make_block_1d(data, TDC_DT_F64, 2);

    tdc_column_stats st;
    CHECK(tdc_stats_compute(&blk, &st) == TDC_OK);
    CHECK(st.has_stats == 1);

    double got_min, got_max;
    tdc_stats_load_value(st.min, &got_min, TDC_DT_F64);
    tdc_stats_load_value(st.max, &got_max, TDC_DT_F64);

    /* Ordered mapping: -0 < +0, so min should be -0.0 and max should be +0.0.
     * Verify via signbit since -0.0 == +0.0 numerically. */
    CHECK(signbit(got_min) != 0);
    CHECK(signbit(got_max) == 0);

    return 0;
}

/* ---- test: NULL arguments ----------------------------------------------- */

static int test_null_args(void) {
    CHECK(tdc_stats_compute(NULL, NULL) == TDC_E_INVAL);

    tdc_column_stats st;
    CHECK(tdc_stats_compute(NULL, &st) == TDC_E_INVAL);

    int32_t data[] = { 1 };
    tdc_block blk = make_block_1d(data, TDC_DT_I32, 1);
    CHECK(tdc_stats_compute(&blk, NULL) == TDC_E_INVAL);

    CHECK(tdc_stats_parse(NULL, 33, 1, &st) == TDC_E_INVAL);
    uint8_t buf[33];
    CHECK(tdc_stats_parse(buf, 33, 1, NULL) == TDC_E_INVAL);

    CHECK(tdc_stats_serialize(NULL, 1, buf) == 0);
    CHECK(tdc_stats_serialize(&st, 1, NULL) == 0);

    return 0;
}

int main(void) {
    if (test_i32_basic())          return 1;
    if (test_f64_basic())          return 1;
    if (test_u8_basic())           return 1;
    if (test_i8_negatives())       return 1;
    if (test_single_element())     return 1;
    if (test_all_same())           return 1;
    if (test_empty_block())        return 1;
    if (test_serialize_roundtrip()) return 1;
    if (test_parse_short_buffer()) return 1;
    if (test_store_load_all_dtypes()) return 1;
    if (test_f64_negative_zero())  return 1;
    if (test_null_args())          return 1;
    printf("test_stats: ok\n");
    return 0;
}
