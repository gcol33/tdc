/*
 * src/model/pred2d_internal.h
 *
 * Cross-TU access to the PRED_2D sweep dispatchers, used by the
 * TDC_MODEL_QUANTIZE_PRED_2D composite model in src/model/quantize_pred2d.c.
 *
 * Not part of the public include/ tree. Other models that want to fuse
 * PRED_2D's predictor logic into their own pipeline call these directly
 * on a typed integer raster — bypassing the model.encode boilerplate
 * (side-meta serialization, bounds validation, vtable indirection) so
 * the composite remains a single self-contained model record on disk.
 */

#ifndef TDC_PRED2D_INTERNAL_H
#define TDC_PRED2D_INTERNAL_H

#include "tdc/codec.h"
#include "tdc/types.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void pred2d_encode_sweep(tdc_dtype dt, tdc_pred2d_kind kind,
                         const uint8_t *src,
                         uint8_t *res,
                         int64_t nx, int64_t ny);

void pred2d_decode_sweep(tdc_dtype dt, tdc_pred2d_kind kind,
                         const uint8_t *res,
                         uint8_t *dst,
                         int64_t nx, int64_t ny);

tdc_pred2d_kind pred2d_auto_select(tdc_dtype dt, const uint8_t *src,
                                   int64_t nx, int64_t ny);

#ifdef __cplusplus
}
#endif

#endif /* TDC_PRED2D_INTERNAL_H */
