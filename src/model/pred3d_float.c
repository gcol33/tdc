/*
 * src/model/pred3d_float.c
 *
 * Float path for the 3D neighborhood predictor (TDC_MODEL_PRED_3D).
 *
 * Why this is separate from pred3d.c: the integer path is a set of
 * typed, per-dtype macro-expanded kernels with per-octant prologues
 * that the compiler vectorizes. The float path is a single triple-
 * nested loop with per-element octant classification — correct and
 * maintainable but slower, and conceptually disjoint from the integer
 * hot loops. Splitting it out keeps pred3d.c focused on the integer
 * fast path.
 *
 * Shared with pred3d.c via pred3d_internal.h: nothing — the only link
 * is the two entry points pred3d_{encode,decode}_float, forward-
 * declared in pred3d.c and called from the sweep dispatchers there.
 */

#include "tdc/codec.h"
#include "../core/float_order.h"
#include "float_pred_helpers.h"

#include <stddef.h>
#include <stdint.h>

/* Float prediction helpers: shared implementation in float_pred_helpers.h. */
#define pred3d_load_ordered           tdc_float_load_ordered
#define pred3d_store_ordered_residual tdc_float_store_ordered_residual
#define pred3d_load_residual          tdc_float_load_residual
#define pred3d_store_float            tdc_float_store_float
#define paeth_ordered_3d              tdc_float_paeth_ordered

/* Per-octant prediction in ordered uint64 space. oct is the bitmask
 * (has_c << 2) | (has_b << 1) | has_a. For GRAD3D and PAETH3D inner
 * box, casts to int64 for the trilinear sum — overflow is theoretical
 * only (would require neighboring voxels spanning the full float64 range). */
static inline uint64_t pred3d_float_compute(tdc_pred3d_kind kind,
                                            uint64_t a, uint64_t b, uint64_t c,
                                            uint64_t ab, uint64_t ac,
                                            uint64_t bc, uint64_t abc,
                                            int oct) {
    switch (kind) {
        case TDC_PRED3D_LEFT:  return a;
        case TDC_PRED3D_UP:    return b;
        case TDC_PRED3D_FRONT: return c;
        case TDC_PRED3D_AVG3:
            switch (oct) {
                case 0: return 0;
                case 1: return a;
                case 2: return b;
                case 4: return c;
                case 3: return (a + b) / 2u;
                case 5: return (a + c) / 2u;
                case 6: return (b + c) / 2u;
                case 7: return (a + b + c) / 3u;
                default: return 0;
            }
        case TDC_PRED3D_GRAD3D: {
            int64_t ia = (int64_t)a, ib = (int64_t)b, ic = (int64_t)c;
            int64_t iab = (int64_t)ab, iac = (int64_t)ac;
            int64_t ibc = (int64_t)bc, iabc = (int64_t)abc;
            return (uint64_t)(ia + ib + ic - iab - iac - ibc + iabc);
        }
        case TDC_PRED3D_PAETH3D:
            switch (oct) {
                case 0: return 0;
                case 1: return a;
                case 2: return b;
                case 4: return c;
                case 3: return paeth_ordered_3d(a, b, ab);
                case 5: return paeth_ordered_3d(a, c, ac);
                case 6: return paeth_ordered_3d(b, c, bc);
                case 7: {
                    int64_t ia = (int64_t)a, ib = (int64_t)b, ic = (int64_t)c;
                    int64_t iab = (int64_t)ab, iac = (int64_t)ac;
                    int64_t ibc = (int64_t)bc, iabc = (int64_t)abc;
                    int64_t p = ia + ib + ic - iab - iac - ibc + iabc;
                    int64_t pa = p >= ia ? p - ia : ia - p;
                    int64_t pb = p >= ib ? p - ib : ib - p;
                    int64_t pc = p >= ic ? p - ic : ic - p;
                    if (pa <= pb && pa <= pc) return a;
                    if (pb <= pc) return b;
                    return c;
                }
                default: return 0;
            }
        default: return 0;
    }
}

void pred3d_encode_float(tdc_dtype dt, tdc_pred3d_kind kind,
                         const uint8_t *src, uint8_t *res,
                         int64_t nx, int64_t ny, int64_t nz) {
    if (nx <= 0 || ny <= 0 || nz <= 0) return;
    const int64_t slab = nx * ny;

    for (int64_t z = 0; z < nz; ++z) {
        for (int64_t y = 0; y < ny; ++y) {
            for (int64_t x = 0; x < nx; ++x) {
                int64_t i = z * slab + y * nx + x;
                uint64_t val = pred3d_load_ordered(dt, src, i);

                int has_a = x > 0, has_b = y > 0, has_c = z > 0;
                int oct = (has_c << 2) | (has_b << 1) | has_a;
                uint64_t a   = has_a               ? pred3d_load_ordered(dt, src, i - 1)             : 0;
                uint64_t b   = has_b               ? pred3d_load_ordered(dt, src, i - nx)            : 0;
                uint64_t c   = has_c               ? pred3d_load_ordered(dt, src, i - slab)          : 0;
                uint64_t ab  = (has_a && has_b)    ? pred3d_load_ordered(dt, src, i - 1 - nx)        : 0;
                uint64_t ac  = (has_a && has_c)    ? pred3d_load_ordered(dt, src, i - 1 - slab)      : 0;
                uint64_t bc  = (has_b && has_c)    ? pred3d_load_ordered(dt, src, i - nx - slab)     : 0;
                uint64_t abc = (has_a && has_b && has_c)
                                                   ? pred3d_load_ordered(dt, src, i - 1 - nx - slab) : 0;

                uint64_t pred = pred3d_float_compute(kind, a, b, c,
                                                     ab, ac, bc, abc, oct);
                pred3d_store_ordered_residual(dt, res, i, val - pred);
            }
        }
    }
}

void pred3d_decode_float(tdc_dtype dt, tdc_pred3d_kind kind,
                         const uint8_t *res, uint8_t *dst,
                         int64_t nx, int64_t ny, int64_t nz) {
    if (nx <= 0 || ny <= 0 || nz <= 0) return;
    const int64_t slab = nx * ny;

    for (int64_t z = 0; z < nz; ++z) {
        for (int64_t y = 0; y < ny; ++y) {
            for (int64_t x = 0; x < nx; ++x) {
                int64_t i = z * slab + y * nx + x;
                uint64_t rv = pred3d_load_residual(dt, res, i);

                int has_a = x > 0, has_b = y > 0, has_c = z > 0;
                int oct = (has_c << 2) | (has_b << 1) | has_a;
                uint64_t a   = has_a               ? pred3d_load_ordered(dt, dst, i - 1)             : 0;
                uint64_t b   = has_b               ? pred3d_load_ordered(dt, dst, i - nx)            : 0;
                uint64_t c   = has_c               ? pred3d_load_ordered(dt, dst, i - slab)          : 0;
                uint64_t ab  = (has_a && has_b)    ? pred3d_load_ordered(dt, dst, i - 1 - nx)        : 0;
                uint64_t ac  = (has_a && has_c)    ? pred3d_load_ordered(dt, dst, i - 1 - slab)      : 0;
                uint64_t bc  = (has_b && has_c)    ? pred3d_load_ordered(dt, dst, i - nx - slab)     : 0;
                uint64_t abc = (has_a && has_b && has_c)
                                                   ? pred3d_load_ordered(dt, dst, i - 1 - nx - slab) : 0;

                uint64_t pred = pred3d_float_compute(kind, a, b, c,
                                                     ab, ac, bc, abc, oct);
                pred3d_store_float(dt, dst, i, rv + pred);
            }
        }
    }
}
