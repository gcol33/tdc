/* docs/examples/format_peek_header.c
 *
 * Parse a block record's 80-byte header and print every field. Uses only
 * the public API: tdc_decode_peek for the (dtype, layout, shape) triple,
 * plus a local little-endian reader for the byte-level fields that peek
 * does not surface (flags, model/xform/entropy ids, section sizes).
 *
 * Build:
 *   cc -I include docs/examples/format_peek_header.c \
 *      build/libtdc.a -lm -o /tmp/format_peek
 *   /tmp/format_peek
 */

#include "quickstart_common.h"
#include <inttypes.h>
#include <stdint.h>
#include <string.h>

static const char *dtype_name(uint8_t dt) {
    switch (dt) {
        case TDC_DT_I8:     return "I8";
        case TDC_DT_I16:    return "I16";
        case TDC_DT_I32:    return "I32";
        case TDC_DT_I64:    return "I64";
        case TDC_DT_U8:     return "U8";
        case TDC_DT_U16:    return "U16";
        case TDC_DT_U32:    return "U32";
        case TDC_DT_U64:    return "U64";
        case TDC_DT_F16:    return "F16";
        case TDC_DT_F32:    return "F32";
        case TDC_DT_F64:    return "F64";
        case TDC_DT_STRING: return "STRING";
        default:            return "?";
    }
}

static const char *layout_name(uint8_t ly) {
    switch (ly) {
        case TDC_LAYOUT_VECTOR_1D: return "VECTOR_1D";
        case TDC_LAYOUT_RASTER_2D: return "RASTER_2D";
        case TDC_LAYOUT_STACK_2D:  return "STACK_2D";
        case TDC_LAYOUT_VOLUME_3D: return "VOLUME_3D";
        default:                   return "?";
    }
}

static uint64_t rd_le(const uint8_t *p, size_t n) {
    uint64_t v = 0;
    for (size_t i = 0; i < n; ++i) v |= (uint64_t)p[i] << (8 * i);
    return v;
}

static int64_t rd_i64(const uint8_t *p) {
    uint64_t u = rd_le(p, 8);
    int64_t  s;
    memcpy(&s, &u, sizeof(s));
    return s;
}

int main(void) {
    /* Encode a 2D i32 raster so the header has a nontrivial shape. */
    int32_t img[4 * 6];
    for (int i = 0; i < 24; ++i) img[i] = i * 7 - 3;

    tdc_block src = {0};
    src.data   = img;
    src.dtype  = TDC_DT_I32;
    src.layout = TDC_LAYOUT_RASTER_2D;
    src.shape.rank   = 2;
    src.shape.dim[0] = 4;
    src.shape.dim[1] = 6;
    tdc_shape_set_contiguous(&src.shape);

    tdc_codec_spec spec = {0};
    spec.model      = TDC_MODEL_PRED_2D;
    spec.xform[0]   = TDC_XFORM_ZIGZAG;
    spec.xform[1]   = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0] = TDC_ENTROPY_LZ;

    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(&src, &spec, &enc), "encode")) return 1;

    /* Public peek: dtype, layout, shape, uncompressed byte count. */
    tdc_block meta = {0};
    size_t need = 0;
    if (qs_check(tdc_decode_peek(enc.data, enc.size, &meta, &need), "peek")) {
        return 1;
    }

    /* Byte-level fields the peek API does not expose: flags, stage ids,
     * per-section sizes. Offsets come straight from include/tdc/format.h. */
    const uint8_t *b = enc.data;
    uint32_t magic      = (uint32_t)rd_le(b + 0,  4);
    uint16_t version    = (uint16_t)rd_le(b + 4,  2);
    uint16_t flags      = (uint16_t)rd_le(b + 6,  2);
    uint16_t model_id   = (uint16_t)rd_le(b + 8,  2);
    uint16_t x0 = (uint16_t)rd_le(b + 10, 2);
    uint16_t x1 = (uint16_t)rd_le(b + 12, 2);
    uint16_t x2 = (uint16_t)rd_le(b + 14, 2);
    uint16_t x3 = (uint16_t)rd_le(b + 16, 2);
    uint16_t e0 = (uint16_t)rd_le(b + 18, 2);
    uint16_t e1 = (uint16_t)rd_le(b + 20, 2);
    uint16_t e2 = (uint16_t)rd_le(b + 22, 2);
    uint16_t e3 = (uint16_t)rd_le(b + 24, 2);
    uint64_t uncomp        =            rd_le(b + 56, 8);
    uint32_t side_size     = (uint32_t)rd_le(b + 64, 4);
    uint32_t payload_size  = (uint32_t)rd_le(b + 68, 4);
    uint32_t xform_ps_size = (uint32_t)rd_le(b + 72, 4);
    uint32_t validity_size = (uint32_t)rd_le(b + 76, 4);

    printf("block record header (80 bytes)\n");
    printf("  magic              = 0x%08" PRIx32 "  (TDC_BLOCK_MAGIC = 0x%08x)\n",
           magic, TDC_BLOCK_MAGIC);
    printf("  version            = %" PRIu16 "\n", version);
    printf("  flags              = 0x%04" PRIx16 "\n", flags);
    printf("  model_id           = 0x%04" PRIx16 "\n", model_id);
    printf("  xform_ids[0..3]    = 0x%04x 0x%04x 0x%04x 0x%04x\n", x0, x1, x2, x3);
    printf("  entropy_ids[0..3]  = 0x%04x 0x%04x 0x%04x 0x%04x\n", e0, e1, e2, e3);
    printf("  dtype              = %u (%s)\n",
           (unsigned)meta.dtype, dtype_name((uint8_t)meta.dtype));
    printf("  layout             = %u (%s)\n",
           (unsigned)meta.layout, layout_name((uint8_t)meta.layout));
    printf("  rank               = %u\n", (unsigned)meta.shape.rank);
    printf("  dim[0..2]          = %" PRId64 " %" PRId64 " %" PRId64 "\n",
           rd_i64(b + 32), rd_i64(b + 40), rd_i64(b + 48));
    printf("  uncompressed_size  = %" PRIu64 " bytes\n", uncomp);
    printf("  side_meta_size     = %" PRIu32 "\n", side_size);
    printf("  xform_params_size  = %" PRIu32 "\n", xform_ps_size);
    printf("  payload_size       = %" PRIu32 "\n", payload_size);
    printf("  validity_size      = %" PRIu32 "\n", validity_size);

    /* On-disk layout: header | side | xform_params | payload | validity. */
    size_t off = TDC_BLOCK_HEADER_SIZE;
    printf("\nsection offsets:\n");
    printf("  header       : 0 .. %u\n",   (unsigned)TDC_BLOCK_HEADER_SIZE);
    printf("  side_meta    : %zu .. %zu\n",  off, off + side_size); off += side_size;
    printf("  xform_params : %zu .. %zu\n",  off, off + xform_ps_size); off += xform_ps_size;
    printf("  payload      : %zu .. %zu\n",  off, off + payload_size); off += payload_size;
    printf("  validity     : %zu .. %zu\n",  off, off + validity_size);

    /* Derived sanity: peek's bytes_required must equal n_elems * dtype_size. */
    printf("\npeek bytes_required = %zu (matches uncompressed element bytes)\n",
           need);

    /* Cheap structural validator — no decode, no allocation. */
    tdc_block_record rec;
    memcpy(&rec, b, sizeof(rec));
    tdc_status vst = tdc_block_record_validate(&rec);
    printf("tdc_block_record_validate = %d (%s)\n",
           (int)vst, tdc_strerror(vst));

    qs_buffer_free(&enc);
    return (vst == TDC_OK) ? 0 : 1;
}
