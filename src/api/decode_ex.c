/*
 * src/api/decode_ex.c
 *
 * Implements: tdc_decode_block_ex
 *
 * Identical pipeline to tdc_decode_block (see decode.c) but accepts a
 * caller-supplied tdc_buffer *scratch whose realloc_fn is used for all
 * internal allocations. This enables zero-copy decode into R-allocated
 * SEXP buffers (or any other caller-managed arena) by letting the caller
 * control the allocator.
 *
 * The existing tdc_decode_block wraps libc realloc internally. Once
 * integration is done, tdc_decode_block becomes a thin wrapper:
 *
 *   tdc_status tdc_decode_block(...) {
 *       tdc_buffer scratch = {0};
 *       scratch.realloc_fn = driver_libc_realloc;
 *       return tdc_decode_block_ex(src, src_size, dst, &scratch);
 *   }
 *
 * See decode_ex_integration.h for the step-by-step refactoring guide.
 */

#ifdef _MSC_VER
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "tdc/codec.h"
#include "tdc/format.h"
#include "tdc/model.h"
#include "tdc/transform.h"
#include "tdc/entropy.h"
#include "tdc/types.h"

#include "../core/buffer.h"
#include "driver_internal.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Decode-side stage timers — shared implementation in core/timer.h. */
#include "../core/timer.h"

/* ----- Entry point -------------------------------------------------------- */

tdc_status tdc_decode_block_ex(const uint8_t *src, size_t src_size,
                               tdc_block     *dst, tdc_buffer *scratch) {
    if (!src || !dst || !scratch) return TDC_E_INVAL;
    if (!scratch->realloc_fn)     return TDC_E_INVAL;
    if (src_size < TDC_BLOCK_HEADER_SIZE) return TDC_E_CORRUPT;

    /* Pull the header out by memcpy (no aliasing assumption on src
     * alignment). The struct layout matches the on-disk byte order
     * one-to-one on every supported little-endian target. */
    tdc_block_record hdr;
    memcpy(&hdr, src, TDC_BLOCK_HEADER_SIZE);

    tdc_status st = tdc_block_record_validate(&hdr);
    if (st != TDC_OK) return st;

    /* Bounds: header + side_meta + xform_params + payload + validity
     * must fit inside src_size. */
    int64_t n_elems = 1;
    for (uint8_t i = 0; i < hdr.rank; ++i) n_elems *= hdr.dim[i];

    /* Validity byte count must match the header field exactly. */
    size_t expected_vbytes = (n_elems > 0) ? (size_t)((n_elems + 7) / 8) : 0u;
    if (hdr.flags & TDC_BLOCK_FLAG_HAS_VALIDITY) {
        if ((size_t)hdr.validity_size != expected_vbytes) return TDC_E_CORRUPT;
    } else {
        if (hdr.validity_size != 0u) return TDC_E_CORRUPT;
    }
    size_t vbytes = (size_t)hdr.validity_size;

    size_t total = (size_t)TDC_BLOCK_HEADER_SIZE
                 + hdr.side_meta_size + hdr.xform_params_size
                 + hdr.payload_size + vbytes;
    if (total > src_size) return TDC_E_CORRUPT;

    /* Cross-check dst against header. */
    if (dst->dtype  != (tdc_dtype)hdr.dtype)   return TDC_E_DTYPE;
    if (dst->layout != (tdc_layout)hdr.layout) return TDC_E_LAYOUT;
    if (dst->shape.rank != hdr.rank)           return TDC_E_SHAPE;
    for (uint8_t i = 0; i < hdr.rank; ++i) {
        if (dst->shape.dim[i] != hdr.dim[i]) return TDC_E_SHAPE;
    }

    /* dst->data may be NULL only if the block is empty. */
    if (n_elems > 0 && dst->data == NULL) return TDC_E_INVAL;

    /* Resolve vtables. */
    const tdc_model_vt *model_vt = tdc_model_get((tdc_model_id)hdr.model_id);
    if (!model_vt) return TDC_E_UNSUPPORTED;

    /* Count the entropy chain length from the header. */
    int entropy_len = 0;
    for (int i = 0; i < TDC_MAX_ENTROPY; ++i) {
        tdc_entropy_id eid = (tdc_entropy_id)hdr.entropy_ids[i];
        if (eid == TDC_ENTROPY_NONE) break;
        if (!tdc_entropy_get(eid)) return TDC_E_UNSUPPORTED;
        entropy_len = i + 1;
    }

    /* Source pointers into the record. */
    const uint8_t *side_meta_p    = src + TDC_BLOCK_HEADER_SIZE;
    const uint8_t *xform_params_p = side_meta_p + hdr.side_meta_size;
    const uint8_t *payload_p      = xform_params_p + hdr.xform_params_size;

    /* ----- Stage 1a: parse TLV xform params ---------------------------- */

    driver_xform_params_table xparams;
    st = driver_xform_params_parse(&xparams, hdr.xform_ids,
                                   xform_params_p,
                                   (size_t)hdr.xform_params_size);
    if (st != TDC_OK) return st;

    /* ----- Stage 1b: forward dtype walk for the transform chain -------- */

    tdc_dtype residual_dtype = driver_model_residual_dtype(
        (tdc_model_id)hdr.model_id, dst->dtype);
    size_t    residual_elem_size = tdc_dtype_size(residual_dtype);
    if (residual_elem_size == 0) return TDC_E_UNSUPPORTED;

    int       chain_len = 0;
    tdc_dtype xform_in[TDC_MAX_TRANSFORMS];
    size_t    xform_in_bytes[TDC_MAX_TRANSFORMS];
    tdc_dtype walk = residual_dtype;
    size_t    walk_bytes = (size_t)n_elems * residual_elem_size;
    for (int i = 0; i < TDC_MAX_TRANSFORMS; ++i) {
        tdc_xform_id xid = (tdc_xform_id)hdr.xform_ids[i];
        if (xid == TDC_XFORM_NONE) break;
        if (!tdc_xform_get(xid)) return TDC_E_UNSUPPORTED;
        xform_in[i]       = walk;
        xform_in_bytes[i] = walk_bytes;
        tdc_dtype next = driver_xform_out_dtype(xid, walk, xparams.xform_params[i]);
        if (next == (tdc_dtype)0) return TDC_E_UNSUPPORTED;
        size_t next_elem_size = tdc_dtype_size(next);
        if (next_elem_size == 0) return TDC_E_UNSUPPORTED;
        walk       = next;
        walk_bytes = (size_t)n_elems * next_elem_size;
        chain_len  = i + 1;
    }

    /* The post-chain byte count must match the header's uncompressed_size. */
    if ((uint64_t)walk_bytes != hdr.uncompressed_size) return TDC_E_CORRUPT;

    /* ----- Stage 2: scratch buffers + entropy decode ------------------ */

    /* Use the caller-supplied scratch as the parent for ping-pong buffers. */
    tdc_buffer bufs[2];
    driver_scratch_init(&bufs[0], scratch);
    driver_scratch_init(&bufs[1], scratch);

    int cur = 0;

    if (hdr.uncompressed_size > SIZE_MAX) { st = TDC_E_INVAL; goto cleanup; }
    size_t entropy_out_size = (size_t)hdr.uncompressed_size;

    st = tdc_buf_reserve(&bufs[cur], entropy_out_size > 0 ? entropy_out_size : 1u);
    if (st != TDC_OK) goto cleanup;

    const int dec_timing_on = tdc_stage_timers_enabled();
    double t_entropy = 0.0, t_xform = 0.0, t_model = 0.0;
    double dt0 = dec_timing_on ? tdc_now_secs() : 0.0;

    /* Entropy decode: walk the chain RTL. */
    if (entropy_len == 0) {
        if ((size_t)hdr.payload_size != entropy_out_size) {
            st = TDC_E_CORRUPT; goto cleanup;
        }
        if (entropy_out_size > 0) {
            memcpy(bufs[cur].data, payload_p, entropy_out_size);
        }
        bufs[cur].size = entropy_out_size;
    } else {
        size_t sizes_bytes = (size_t)entropy_len * sizeof(uint32_t);
        if ((size_t)hdr.payload_size < sizes_bytes) {
            st = TDC_E_CORRUPT; goto cleanup;
        }
        uint32_t stage_in_sizes[TDC_MAX_ENTROPY];
        for (int i = 0; i < entropy_len; ++i) {
            uint32_t le;
            memcpy(&le, payload_p + (size_t)i * sizeof(uint32_t), sizeof(le));
            stage_in_sizes[i] = le;
        }
        if ((uint64_t)stage_in_sizes[0] != hdr.uncompressed_size) {
            st = TDC_E_CORRUPT; goto cleanup;
        }

        const uint8_t *stage_src      = payload_p      + sizes_bytes;
        size_t         stage_src_size = (size_t)hdr.payload_size - sizes_bytes;

        for (int i = entropy_len - 1; i >= 0; --i) {
            tdc_entropy_id eid = (tdc_entropy_id)hdr.entropy_ids[i];
            const tdc_entropy_vt *evt = tdc_entropy_get(eid);
            if (!evt) { st = TDC_E_UNSUPPORTED; goto cleanup; }

            size_t dst_size = (size_t)stage_in_sizes[i];

            st = tdc_buf_reserve(&bufs[1 - cur], dst_size > 0 ? dst_size : 1u);
            if (st != TDC_OK) goto cleanup;

            st = evt->decode(stage_src, stage_src_size,
                             bufs[1 - cur].data, dst_size);
            if (st != TDC_OK) goto cleanup;
            bufs[1 - cur].size = dst_size;

            stage_src      = bufs[1 - cur].data;
            stage_src_size = dst_size;
            cur = 1 - cur;
        }
        if (bufs[cur].size != entropy_out_size) {
            st = TDC_E_CORRUPT; goto cleanup;
        }
    }
    if (dec_timing_on) { double t1 = tdc_now_secs(); t_entropy = t1 - dt0; dt0 = t1; }

    /* ----- Stage 3: reverse the transform chain ----------------------- */

    for (int i = chain_len - 1; i >= 0; --i) {
        tdc_xform_id xid = (tdc_xform_id)hdr.xform_ids[i];
        const tdc_xform_vt *xv = tdc_xform_get(xid);
        if (!xv) { st = TDC_E_UNSUPPORTED; goto cleanup; }

        size_t dst_bytes = xform_in_bytes[i];

        st = tdc_buf_reserve(&bufs[1 - cur], dst_bytes > 0 ? dst_bytes : 1u);
        if (st != TDC_OK) goto cleanup;

        tdc_dtype out_dtype = (tdc_dtype)0;
        st = xv->decode(bufs[cur].data, bufs[cur].size,
                        xform_in[i], xparams.xform_params[i],
                        bufs[1 - cur].data, dst_bytes,
                        &out_dtype);
        if (st != TDC_OK) goto cleanup;

        bufs[1 - cur].size = dst_bytes;
        cur = 1 - cur;
    }

    if (dec_timing_on) { double t1 = tdc_now_secs(); t_xform = t1 - dt0; dt0 = t1; }

    /* ----- Stage 4: model decode -------------------------------------- */

    st = model_vt->decode(dst, NULL, residual_dtype,
                          bufs[cur].data, bufs[cur].size,
                          (hdr.side_meta_size > 0) ? side_meta_p : NULL,
                          hdr.side_meta_size);
    if (st != TDC_OK) goto cleanup;
    if (dec_timing_on) { double t1 = tdc_now_secs(); t_model = t1 - dt0; }

    if (dec_timing_on) {
        size_t raw = (size_t)n_elems * tdc_dtype_size(dst->dtype);
        double raw_mib = (double)raw / (1024.0 * 1024.0);
        fprintf(stderr,
                "[decex] entropy=%6.2f ms (%7.1f MB/s) "
                "xform=%6.2f ms (%7.1f MB/s) "
                "model=%6.2f ms (%7.1f MB/s) "
                "raw=%.2f MiB enc=%llu B\n",
                t_entropy * 1000.0,
                t_entropy > 0 ? raw_mib / t_entropy : 0.0,
                t_xform   * 1000.0,
                t_xform   > 0 ? raw_mib / t_xform   : 0.0,
                t_model   * 1000.0,
                t_model   > 0 ? raw_mib / t_model   : 0.0,
                raw_mib, (unsigned long long)hdr.payload_size);
    }

    st = TDC_OK;

cleanup:
    driver_scratch_free(&bufs[0]);
    driver_scratch_free(&bufs[1]);
    return st;
}
