/*
 * examples/plugin/xor_entropy.c
 *
 * Minimal demonstration of tdc_entropy_register: a user-defined entropy
 * backend that XORs every input byte with 0xA5. Decode is the same
 * operation (XOR is its own inverse). Not a compressor -- the point is
 * the plugin lifecycle, not entropy coding.
 *
 * Flow:
 *   1. Define a tdc_entropy_vt whose .id matches the chosen plugin id.
 *   2. Call tdc_entropy_register(id, &vt). The vtable pointer must
 *      outlive the registration, so we make it a static const.
 *   3. Resolve the same id through tdc_entropy_get() to prove the
 *      registry returns what we registered.
 *   4. Round-trip a small buffer through encode then decode.
 *
 * Plugin id is 0xFF01 (first slot of the user-defined range).
 */

#include "tdc/plugin.h"
#include "tdc/entropy.h"
#include "tdc/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define USER_XOR_ID ((tdc_entropy_id)0xFF01)
#define XOR_MASK    0xA5u

/* ----- Allocator: stdlib realloc wrapped in tdc's three-arg convention. */

static void *ex_realloc(void *user, void *ptr, size_t new_size) {
    (void)user;
    if (new_size == 0) { free(ptr); return NULL; }
    return realloc(ptr, new_size);
}

/* ----- Entropy vtable implementation. */

static size_t xor_encode_bound(size_t src_size) { return src_size; }

static tdc_status xor_encode(const uint8_t *src, size_t src_size,
                             const void *params, tdc_buffer *dst) {
    (void)params;
    void *p = dst->realloc_fn(dst->user, dst->data, src_size);
    if (!p && src_size > 0) return TDC_E_NOMEM;
    dst->data = (uint8_t *)p;
    dst->size = src_size;
    dst->capacity = src_size;
    for (size_t i = 0; i < src_size; ++i) dst->data[i] = src[i] ^ XOR_MASK;
    return TDC_OK;
}

static tdc_status xor_decode(const uint8_t *src, size_t src_size,
                             uint8_t *dst, size_t dst_size) {
    if (src_size != dst_size) return TDC_E_CORRUPT;
    for (size_t i = 0; i < src_size; ++i) dst[i] = src[i] ^ XOR_MASK;
    return TDC_OK;
}

static const tdc_entropy_vt xor_vt = {
    .id           = USER_XOR_ID,
    .name         = "user_xor_a5",
    .encode_bound = xor_encode_bound,
    .encode       = xor_encode,
    .decode       = xor_decode,
};

/* ----- Driver. */

int main(void) {
    if (tdc_entropy_register(USER_XOR_ID, &xor_vt) != TDC_OK) {
        fprintf(stderr, "register failed\n");
        return 1;
    }

    const tdc_entropy_vt *vt = tdc_entropy_get(USER_XOR_ID);
    if (vt != &xor_vt) {
        fprintf(stderr, "lookup mismatch\n");
        return 1;
    }

    static const uint8_t plain[] = { 0x00, 0x01, 0x7F, 0x80, 0xA5, 0xFF, 0x42, 0x13 };
    const size_t n = sizeof(plain);

    tdc_buffer enc = { NULL, 0, 0, ex_realloc, NULL };
    if (vt->encode(plain, n, NULL, &enc) != TDC_OK) {
        fprintf(stderr, "encode failed\n");
        return 1;
    }

    uint8_t decoded[sizeof(plain)];
    if (vt->decode(enc.data, enc.size, decoded, n) != TDC_OK) {
        fprintf(stderr, "decode failed\n");
        ex_realloc(NULL, enc.data, 0);
        return 1;
    }

    int ok = memcmp(plain, decoded, n) == 0;
    ex_realloc(NULL, enc.data, 0);
    tdc_plugin_clear();

    if (!ok) {
        fprintf(stderr, "round-trip mismatch\n");
        return 1;
    }
    printf("OK\n");
    return 0;
}
