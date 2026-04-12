/*
 * src/core/registry.c
 *
 * Static id -> vtable lookup tables for the three pluggable stages:
 *   tdc_model_get   (tdc/model.h)
 *   tdc_xform_get   (tdc/transform.h)
 *   tdc_entropy_get (tdc/entropy.h)
 *
 * v0 has no public registration API. Adding a model/transform/entropy
 * means adding a vtable to the appropriate src/{model,transform,entropy}/
 * file, declaring it in the matching internal header, and listing it in
 * the switch below.
 *
 * Stages with no implementations yet (model, transform) return NULL for
 * every id; the encode/decode driver translates that into TDC_E_UNSUPPORTED.
 */

#include "tdc/model.h"
#include "tdc/transform.h"
#include "tdc/entropy.h"

#include "../entropy/entropy_internal.h"
#include "../model/model_internal.h"
#include "../transform/transform_internal.h"

const tdc_model_vt *tdc_model_get(tdc_model_id id) {
    switch (id) {
        case TDC_MODEL_DELTA_1D: return &tdc_model_delta1d_vt;
        case TDC_MODEL_NONE:     return NULL;
        case TDC_MODEL_RAW:      return &tdc_model_raw_vt;
        case TDC_MODEL_DICT_1D:  return &tdc_model_dict1d_vt;
        case TDC_MODEL_PRED_2D:  return &tdc_model_pred2d_vt;
        case TDC_MODEL_STACK_2D: return &tdc_model_stack2d_vt;
        case TDC_MODEL_PRED_3D:  return &tdc_model_pred3d_vt;
        case TDC_MODEL_PLANE_2D: return &tdc_model_plane2d_vt;
        default:                 return NULL;
    }
}

const tdc_xform_vt *tdc_xform_get(tdc_xform_id id) {
    switch (id) {
        case TDC_XFORM_BYTE_SHUFFLE: return &tdc_xform_byte_shuffle_vt;
        case TDC_XFORM_NONE:         return NULL;
        case TDC_XFORM_QUANTIZE:     return &tdc_xform_quantize_vt;
        case TDC_XFORM_ZIGZAG:       return &tdc_xform_zigzag_vt;
        case TDC_XFORM_BIT_SHUFFLE:  return &tdc_xform_bit_shuffle_vt;
        default:                     return NULL;
    }
}

const tdc_entropy_vt *tdc_entropy_get(tdc_entropy_id id) {
    switch (id) {
        case TDC_ENTROPY_LZ:         return &tdc_entropy_lz_vt;
        case TDC_ENTROPY_LZ_OPT:     return &tdc_entropy_lz_opt_vt;
        case TDC_ENTROPY_LZ_STREAMS: return &tdc_entropy_lz_streams_vt;
        case TDC_ENTROPY_NONE:    return &tdc_entropy_none_vt;
#ifdef TDC_HAVE_ZLIB
        case TDC_ENTROPY_DEFLATE: return &tdc_entropy_deflate_vt;
#else
        case TDC_ENTROPY_DEFLATE: return NULL;
#endif
        case TDC_ENTROPY_HUFFMAN: return &tdc_entropy_huffman_vt;
        case TDC_ENTROPY_FSE:     return &tdc_entropy_fse_vt;
        case TDC_ENTROPY_LANE:    return &tdc_entropy_lane_vt;
        default:                  return NULL;
    }
}
