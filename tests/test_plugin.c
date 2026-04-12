/*
 * tests/test_plugin.c
 *
 * Tests for the plugin API (tdc/plugin.h).
 *
 * Verifies:
 *   1. Registration of user-defined model/transform/entropy backends.
 *   2. Lookup via tdc_model_get / tdc_xform_get / tdc_entropy_get.
 *   3. Rejection of core-range ids (< 0xFF00).
 *   4. Rejection of NULL vtable.
 *   5. Rejection of duplicate id.
 *   6. Rejection of vtable whose .id disagrees with the id argument.
 *   7. tdc_plugin_clear removes all registrations.
 *   8. Full encode/decode round-trip through a user-defined entropy backend.
 */

#include "tdc/plugin.h"
#include "tdc/model.h"
#include "tdc/transform.h"
#include "tdc/entropy.h"
#include "tdc/codec.h"
#include "tdc/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int n_pass;
static int n_fail;

#define ASSERT(cond, msg) do {                                                \
    if (!(cond)) {                                                            \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);     \
        ++n_fail;                                                             \
        return;                                                               \
    }                                                                         \
} while (0)

#define PASS(name) do { ++n_pass; printf("  ok  %s\n", (name)); } while (0)

/* ----- Dummy vtables for registration tests ------------------------------- */

#define USER_MODEL_ID   ((tdc_model_id)0xFF01)
#define USER_XFORM_ID   ((tdc_xform_id)0xFF02)
#define USER_ENTROPY_ID ((tdc_entropy_id)0xFF03)

static tdc_status dummy_model_encode(const tdc_block *in, const void *params,
                                     tdc_buffer *residual_out,
                                     tdc_dtype *residual_dtype,
                                     tdc_buffer *side_out) {
    (void)params; (void)side_out;
    *residual_dtype = in->dtype;
    size_t bytes = (size_t)tdc_shape_n_elems(&in->shape) * tdc_dtype_size(in->dtype);
    void *p = residual_out->realloc_fn(residual_out->user, residual_out->data, bytes);
    if (!p && bytes > 0) return TDC_E_NOMEM;
    residual_out->data = (uint8_t *)p;
    residual_out->size = bytes;
    residual_out->capacity = bytes;
    if (bytes > 0) memcpy(residual_out->data, in->data, bytes);
    side_out->size = 0;
    return TDC_OK;
}

static tdc_status dummy_model_decode(tdc_block *out, const void *params,
                                     tdc_dtype residual_dtype,
                                     const uint8_t *residuals, size_t residual_size,
                                     const uint8_t *side_meta, size_t side_size) {
    (void)params; (void)residual_dtype; (void)side_meta; (void)side_size;
    size_t bytes = (size_t)tdc_shape_n_elems(&out->shape) * tdc_dtype_size(out->dtype);
    if (residual_size != bytes) return TDC_E_CORRUPT;
    memcpy(out->data, residuals, bytes);
    return TDC_OK;
}

static const tdc_model_vt dummy_model_vt = {
    .id               = USER_MODEL_ID,
    .name             = "user_dummy",
    .accepted_dtypes  = TDC_DT_NUMERIC_MASK,
    .accepted_layouts = TDC_LAYOUT_MASK(TDC_LAYOUT_VECTOR_1D),
    .encode           = dummy_model_encode,
    .decode           = dummy_model_decode,
};

/* A trivial XOR-with-0x55 transform for testing. */
static tdc_status xor_xform_encode(const uint8_t *src, size_t src_size,
                                   tdc_dtype in_dtype, const void *params,
                                   tdc_buffer *dst, tdc_dtype *out_dtype) {
    (void)params;
    *out_dtype = in_dtype;
    void *p = dst->realloc_fn(dst->user, dst->data, src_size);
    if (!p && src_size > 0) return TDC_E_NOMEM;
    dst->data = (uint8_t *)p;
    dst->size = src_size;
    dst->capacity = src_size;
    for (size_t i = 0; i < src_size; ++i)
        dst->data[i] = src[i] ^ 0x55;
    return TDC_OK;
}

static tdc_status xor_xform_decode(const uint8_t *src, size_t src_size,
                                   tdc_dtype in_dtype, const void *params,
                                   uint8_t *dst, size_t dst_size,
                                   tdc_dtype *out_dtype) {
    (void)params;
    *out_dtype = in_dtype;
    if (src_size != dst_size) return TDC_E_CORRUPT;
    for (size_t i = 0; i < src_size; ++i)
        dst[i] = src[i] ^ 0x55;
    return TDC_OK;
}

static const tdc_xform_vt dummy_xform_vt = {
    .id              = USER_XFORM_ID,
    .name            = "user_xor55",
    .accepted_dtypes = TDC_DT_NUMERIC_MASK,
    .can_inplace     = 0,
    .is_lossy        = 0,
    .encode          = xor_xform_encode,
    .decode          = xor_xform_decode,
};

/* A memcpy passthrough entropy for testing. */
static size_t copy_entropy_bound(size_t n) { return n; }

static tdc_status copy_entropy_encode(const uint8_t *src, size_t src_size,
                                      const void *params, tdc_buffer *dst) {
    (void)params;
    void *p = dst->realloc_fn(dst->user, dst->data, src_size);
    if (!p && src_size > 0) return TDC_E_NOMEM;
    dst->data = (uint8_t *)p;
    dst->size = src_size;
    dst->capacity = src_size;
    if (src_size > 0) memcpy(dst->data, src, src_size);
    return TDC_OK;
}

static tdc_status copy_entropy_decode(const uint8_t *src, size_t src_size,
                                      uint8_t *dst, size_t dst_size) {
    if (src_size != dst_size) return TDC_E_CORRUPT;
    if (src_size > 0) memcpy(dst, src, src_size);
    return TDC_OK;
}

static const tdc_entropy_vt dummy_entropy_vt = {
    .id           = USER_ENTROPY_ID,
    .name         = "user_copy",
    .encode_bound = copy_entropy_bound,
    .encode       = copy_entropy_encode,
    .decode       = copy_entropy_decode,
};

/* ----- Test cases --------------------------------------------------------- */

static void test_register_and_lookup(void) {
    tdc_plugin_clear();

    tdc_status st;

    st = tdc_model_register(USER_MODEL_ID, &dummy_model_vt);
    ASSERT(st == TDC_OK, "model register");

    st = tdc_xform_register(USER_XFORM_ID, &dummy_xform_vt);
    ASSERT(st == TDC_OK, "xform register");

    st = tdc_entropy_register(USER_ENTROPY_ID, &dummy_entropy_vt);
    ASSERT(st == TDC_OK, "entropy register");

    /* Lookup must return the registered vtables. */
    ASSERT(tdc_model_get(USER_MODEL_ID) == &dummy_model_vt,
           "model lookup");
    ASSERT(tdc_xform_get(USER_XFORM_ID) == &dummy_xform_vt,
           "xform lookup");
    ASSERT(tdc_entropy_get(USER_ENTROPY_ID) == &dummy_entropy_vt,
           "entropy lookup");

    /* Core ids must still work. */
    ASSERT(tdc_model_get(TDC_MODEL_RAW) != NULL, "core model still works");
    ASSERT(tdc_entropy_get(TDC_ENTROPY_LZ) != NULL, "core entropy still works");

    tdc_plugin_clear();
    PASS("register_and_lookup");
}

static void test_reject_core_range(void) {
    tdc_plugin_clear();

    /* Core id 0x0001 must be rejected. */
    static const tdc_entropy_vt bad_vt = {
        .id = (tdc_entropy_id)0x0001, .name = "bad",
        .encode_bound = copy_entropy_bound,
        .encode = copy_entropy_encode,
        .decode = copy_entropy_decode,
    };
    tdc_status st = tdc_entropy_register((tdc_entropy_id)0x0001, &bad_vt);
    ASSERT(st == TDC_E_INVAL, "core-range id rejected");

    /* Experimental id 0x0100 must be rejected. */
    static const tdc_entropy_vt exp_vt = {
        .id = (tdc_entropy_id)0x0100, .name = "exp",
        .encode_bound = copy_entropy_bound,
        .encode = copy_entropy_encode,
        .decode = copy_entropy_decode,
    };
    st = tdc_entropy_register((tdc_entropy_id)0x0100, &exp_vt);
    ASSERT(st == TDC_E_INVAL, "experimental-range id rejected");

    /* Reserved id 0x0200 must be rejected. */
    static const tdc_entropy_vt rsv_vt = {
        .id = (tdc_entropy_id)0x0200, .name = "rsv",
        .encode_bound = copy_entropy_bound,
        .encode = copy_entropy_encode,
        .decode = copy_entropy_decode,
    };
    st = tdc_entropy_register((tdc_entropy_id)0x0200, &rsv_vt);
    ASSERT(st == TDC_E_INVAL, "reserved-range id rejected");

    tdc_plugin_clear();
    PASS("reject_core_range");
}

static void test_reject_null_vt(void) {
    tdc_plugin_clear();
    tdc_status st = tdc_model_register(USER_MODEL_ID, NULL);
    ASSERT(st == TDC_E_INVAL, "null vtable rejected");
    tdc_plugin_clear();
    PASS("reject_null_vt");
}

static void test_reject_duplicate(void) {
    tdc_plugin_clear();
    tdc_status st = tdc_entropy_register(USER_ENTROPY_ID, &dummy_entropy_vt);
    ASSERT(st == TDC_OK, "first register");
    st = tdc_entropy_register(USER_ENTROPY_ID, &dummy_entropy_vt);
    ASSERT(st == TDC_E_INVAL, "duplicate rejected");
    tdc_plugin_clear();
    PASS("reject_duplicate");
}

static void test_reject_id_mismatch(void) {
    tdc_plugin_clear();
    /* Register with id 0xFF10 but vtable says 0xFF03. */
    tdc_status st = tdc_entropy_register((tdc_entropy_id)0xFF10, &dummy_entropy_vt);
    ASSERT(st == TDC_E_INVAL, "id mismatch rejected");
    tdc_plugin_clear();
    PASS("reject_id_mismatch");
}

static void test_clear(void) {
    tdc_plugin_clear();
    tdc_model_register(USER_MODEL_ID, &dummy_model_vt);
    ASSERT(tdc_model_get(USER_MODEL_ID) != NULL, "registered");
    tdc_plugin_clear();
    ASSERT(tdc_model_get(USER_MODEL_ID) == NULL, "cleared");
    PASS("clear");
}

static void test_capacity_limit(void) {
    tdc_plugin_clear();

    /* Fill all slots with distinct ids. */
    tdc_entropy_vt vts[TDC_PLUGIN_MAX_SLOTS];
    for (int i = 0; i < TDC_PLUGIN_MAX_SLOTS; ++i) {
        memset(&vts[i], 0, sizeof(vts[i]));
        vts[i].id = (tdc_entropy_id)(0xFF00 + i);
        vts[i].name = "slot";
        vts[i].encode_bound = copy_entropy_bound;
        vts[i].encode = copy_entropy_encode;
        vts[i].decode = copy_entropy_decode;
        tdc_status st = tdc_entropy_register(vts[i].id, &vts[i]);
        ASSERT(st == TDC_OK, "fill slots");
    }

    /* One more must fail with TDC_E_NOMEM. */
    static const tdc_entropy_vt overflow_vt = {
        .id = (tdc_entropy_id)0xFFF0, .name = "overflow",
        .encode_bound = copy_entropy_bound,
        .encode = copy_entropy_encode,
        .decode = copy_entropy_decode,
    };
    tdc_status st = tdc_entropy_register((tdc_entropy_id)0xFFF0, &overflow_vt);
    ASSERT(st == TDC_E_NOMEM, "capacity exceeded");

    tdc_plugin_clear();
    PASS("capacity_limit");
}

/* ----- Full round-trip through user-defined entropy ----------------------- */

static void *test_realloc(void *user, void *ptr, size_t new_size) {
    (void)user;
    if (new_size == 0) { free(ptr); return NULL; }
    return realloc(ptr, new_size);
}

static void test_roundtrip_user_entropy(void) {
    tdc_plugin_clear();

    tdc_status st = tdc_entropy_register(USER_ENTROPY_ID, &dummy_entropy_vt);
    ASSERT(st == TDC_OK, "register user entropy");

    /* Build a VECTOR_1D i32 block with a simple ramp. */
    int32_t data[64];
    for (int i = 0; i < 64; ++i) data[i] = i * 7 + 3;

    tdc_block blk = {0};
    blk.data = data;
    blk.dtype = TDC_DT_I32;
    blk.layout = TDC_LAYOUT_VECTOR_1D;
    blk.shape.rank = 1;
    blk.shape.dim[0] = 64;
    tdc_shape_set_contiguous(&blk.shape);

    /* Encode with RAW model + user entropy. */
    tdc_codec_spec spec = {0};
    spec.model = TDC_MODEL_RAW;
    spec.entropy[0] = USER_ENTROPY_ID;

    tdc_buffer out = {0};
    out.realloc_fn = test_realloc;
    st = tdc_encode_block(&blk, &spec, &out);
    ASSERT(st == TDC_OK, "encode via user entropy");
    ASSERT(out.size > 0, "encoded output non-empty");

    /* Decode back. */
    int32_t decoded[64];
    memset(decoded, 0xCC, sizeof(decoded));

    tdc_block dst = {0};
    dst.data = decoded;
    dst.dtype = TDC_DT_I32;
    dst.layout = TDC_LAYOUT_VECTOR_1D;
    dst.shape.rank = 1;
    dst.shape.dim[0] = 64;
    tdc_shape_set_contiguous(&dst.shape);

    st = tdc_decode_block(out.data, out.size, &dst);
    ASSERT(st == TDC_OK, "decode via user entropy");
    ASSERT(memcmp(data, decoded, sizeof(data)) == 0, "round-trip data match");

    if (out.data) free(out.data);
    tdc_plugin_clear();
    PASS("roundtrip_user_entropy");
}

/* ----- main --------------------------------------------------------------- */

int main(void) {
    printf("test_plugin\n");

    test_register_and_lookup();
    test_reject_core_range();
    test_reject_null_vt();
    test_reject_duplicate();
    test_reject_id_mismatch();
    test_clear();
    test_capacity_limit();
    test_roundtrip_user_entropy();

    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
