/*
 * src/transform/bitshuffle.c
 *
 * TDC_XFORM_BIT_SHUFFLE — transpose by bit lane instead of byte lane.
 *
 * For N elements of elem_size bytes each, the output is organized as
 * (elem_size * 8) bit-planes, each plane holding ceil(N/8) packed bytes.
 * This groups same-significance bits together: after quantization to a
 * narrow integer type, the high-order bits are mostly zero and bit-
 * shuffling makes them collapse into long runs of 0s, dramatically
 * improving subsequent entropy coding.
 *
 * Output size: n_planes * bytes_per_plane, where
 *   n_planes      = elem_size * 8
 *   bytes_per_plane = (n_elems + 7) / 8
 * This may be slightly larger than src_size when n_elems is not a
 * multiple of 8. The excess bits in the last byte of each plane are
 * zeroed on encode and ignored on decode.
 *
 * Scalar implementation. No SIMD in v0 — the scalar path is
 * straightforward and correct; SIMD bit-transpose (e.g. via
 * movemask) is a future optimization.
 */

#include "tdc/transform.h"
#include "transform_internal.h"
#include "../core/buffer.h"

#include <string.h>
#include <stdint.h>
#include <stddef.h>

/* ----- Scalar bit-shuffle / unshuffle ------------------------------------ */

/*
 * Bit-shuffle: scatter element bits into bit-planes.
 *
 * For each element, for each bit position in the element, set the
 * corresponding bit in the output plane. Output is organized as:
 *   plane[p] occupies dst[p * bpp .. (p+1) * bpp), where bpp = bytes_per_plane
 *   Within a plane, bit i of element e is stored at
 *   byte offset (e / 8), bit offset (e % 8).
 */
static void bitshuffle_scatter(uint8_t *dst, const uint8_t *src,
                               uint32_t n_elems, uint32_t elem_size) {
    uint32_t n_planes = elem_size * 8u;
    uint32_t bpp = (n_elems + 7u) / 8u; /* bytes per plane */
    size_t out_size = (size_t)n_planes * bpp;

    /* Zero output: tail bits in partial final bytes must be 0. */
    memset(dst, 0, out_size);

    for (uint32_t e = 0; e < n_elems; ++e) {
        uint32_t out_byte_in_plane = e / 8u;
        uint32_t out_bit           = e % 8u;

        for (uint32_t b = 0; b < elem_size; ++b) {
            uint8_t byte_val = src[(size_t)e * elem_size + b];
            for (uint32_t bit = 0; bit < 8u; ++bit) {
                if (byte_val & (1u << bit)) {
                    uint32_t plane = b * 8u + bit;
                    dst[(size_t)plane * bpp + out_byte_in_plane] |=
                        (uint8_t)(1u << out_bit);
                }
            }
        }
    }
}

/*
 * Bit-unshuffle: gather element bits from bit-planes.
 */
static void bitshuffle_gather(uint8_t *dst, const uint8_t *src,
                              uint32_t n_elems, uint32_t elem_size) {
    uint32_t n_planes = elem_size * 8u;
    uint32_t bpp = (n_elems + 7u) / 8u;

    /* Zero output in case of partial reads. */
    memset(dst, 0, (size_t)n_elems * elem_size);

    for (uint32_t e = 0; e < n_elems; ++e) {
        uint32_t in_byte_in_plane = e / 8u;
        uint32_t in_bit           = e % 8u;

        for (uint32_t plane = 0; plane < n_planes; ++plane) {
            if (src[(size_t)plane * bpp + in_byte_in_plane] & (1u << in_bit)) {
                uint32_t b   = plane / 8u;
                uint32_t bit = plane % 8u;
                dst[(size_t)e * elem_size + b] |= (uint8_t)(1u << bit);
            }
        }
    }
}

/* ----- Vtable hooks ------------------------------------------------------- */

#define BITSHUFFLE_DTYPE_BIT(dt) (1u << (uint32_t)(dt))
#define BITSHUFFLE_ACCEPTED_DTYPES (                    \
    BITSHUFFLE_DTYPE_BIT(TDC_DT_I8)  |                  \
    BITSHUFFLE_DTYPE_BIT(TDC_DT_I16) |                  \
    BITSHUFFLE_DTYPE_BIT(TDC_DT_I32) |                  \
    BITSHUFFLE_DTYPE_BIT(TDC_DT_I64) |                  \
    BITSHUFFLE_DTYPE_BIT(TDC_DT_U8)  |                  \
    BITSHUFFLE_DTYPE_BIT(TDC_DT_U16) |                  \
    BITSHUFFLE_DTYPE_BIT(TDC_DT_U32) |                  \
    BITSHUFFLE_DTYPE_BIT(TDC_DT_U64) |                  \
    BITSHUFFLE_DTYPE_BIT(TDC_DT_F16) |                  \
    BITSHUFFLE_DTYPE_BIT(TDC_DT_F32) |                  \
    BITSHUFFLE_DTYPE_BIT(TDC_DT_F64))

static tdc_status bitshuffle_encode(const uint8_t *src, size_t src_size,
                                    tdc_dtype      in_dtype,
                                    const void    *params,
                                    tdc_buffer    *dst,
                                    tdc_dtype     *out_dtype) {
    (void)params;
    if (!dst || !dst->realloc_fn) return TDC_E_INVAL;
    if (src_size > 0 && !src) return TDC_E_INVAL;

    size_t elem_size = tdc_dtype_size(in_dtype);
    if (elem_size == 0) return TDC_E_DTYPE;
    if (src_size % elem_size != 0) return TDC_E_INVAL;

    if (out_dtype) *out_dtype = in_dtype;

    if (src_size == 0) {
        dst->size = 0;
        return TDC_OK;
    }

    size_t n_elems_sz = src_size / elem_size;
    if (n_elems_sz > UINT32_MAX) return TDC_E_INVAL;
    uint32_t n_elems = (uint32_t)n_elems_sz;

    /* elem_size 1: bit-shuffle of single-byte elements. */
    uint32_t n_planes = (uint32_t)elem_size * 8u;
    uint32_t bpp = (n_elems + 7u) / 8u;
    size_t out_size = (size_t)n_planes * bpp;

    tdc_status st = tdc_buf_reserve(dst, out_size);
    if (st != TDC_OK) return st;

    bitshuffle_scatter(dst->data, src, n_elems, (uint32_t)elem_size);
    dst->size = out_size;
    return TDC_OK;
}

static tdc_status bitshuffle_decode(const uint8_t *src, size_t src_size,
                                    tdc_dtype      in_dtype,
                                    const void    *params,
                                    uint8_t       *dst, size_t dst_size,
                                    tdc_dtype     *out_dtype) {
    (void)params;
    if (src_size > 0 && !src) return TDC_E_INVAL;
    if (dst_size > 0 && !dst) return TDC_E_INVAL;

    size_t elem_size = tdc_dtype_size(in_dtype);
    if (elem_size == 0) return TDC_E_DTYPE;
    if (dst_size % elem_size != 0) return TDC_E_CORRUPT;

    if (out_dtype) *out_dtype = in_dtype;

    if (dst_size == 0) return TDC_OK;

    size_t n_elems_sz = dst_size / elem_size;
    if (n_elems_sz > UINT32_MAX) return TDC_E_INVAL;
    uint32_t n_elems = (uint32_t)n_elems_sz;

    /* Validate that src_size matches the expected bit-shuffled size. */
    uint32_t n_planes = (uint32_t)elem_size * 8u;
    uint32_t bpp = (n_elems + 7u) / 8u;
    size_t expected_src = (size_t)n_planes * bpp;
    if (src_size != expected_src) return TDC_E_CORRUPT;

    bitshuffle_gather(dst, src, n_elems, (uint32_t)elem_size);
    return TDC_OK;
}

const tdc_xform_vt tdc_xform_bit_shuffle_vt = {
    .id              = TDC_XFORM_BIT_SHUFFLE,
    .name            = "bit_shuffle",
    .accepted_dtypes = BITSHUFFLE_ACCEPTED_DTYPES,
    .can_inplace     = 0,
    .is_lossy        = 0,
    .encode          = bitshuffle_encode,
    .decode          = bitshuffle_decode,
};
