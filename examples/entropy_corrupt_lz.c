/* docs/examples/entropy_corrupt_lz.c
 *
 * Flip one byte inside the LZ payload of an encoded block and decode.
 * The LZ decoder runs a bounds check per sequence (dp + lit_len + mlen
 * + TDC_WILDCOPY_SLACK > uncompressed_size) and a match-offset check
 * (dp < off) on the safe path; both return TDC_E_CORRUPT when the
 * invariants break. Flips in different parts of the stream exercise
 * different branches.
 *
 * Build: part of TDC_DOC_EXAMPLES in the top-level CMakeLists.txt.
 */

#include "quickstart_common.h"
#include <stdint.h>
#include <string.h>

/* Locate the first byte of the LZ payload inside a freshly-encoded block.
 * See docs/examples/theory_lz_histogram.c for the long-form derivation. */
static size_t lz_payload_offset(const uint8_t *rec) {
    const tdc_block_record *hdr = (const tdc_block_record *)rec;
    size_t off = TDC_BLOCK_HEADER_SIZE
               + (size_t)hdr->side_meta_size
               + (size_t)hdr->xform_params_size;
    /* Skip the per-stage sizes table: one u32 per active entropy slot. */
    size_t slots = 0;
    for (int i = 0; i < TDC_MAX_ENTROPY; ++i) {
        if (hdr->entropy_ids[i] == 0) break;
        ++slots;
    }
    return off + slots * sizeof(uint32_t);
}

static const char *status_name(tdc_status s) {
    switch (s) {
        case TDC_OK:               return "TDC_OK";
        case TDC_E_INVAL:          return "TDC_E_INVAL";
        case TDC_E_NOMEM:          return "TDC_E_NOMEM";
        case TDC_E_UNSUPPORTED:    return "TDC_E_UNSUPPORTED";
        case TDC_E_DTYPE:          return "TDC_E_DTYPE";
        case TDC_E_LAYOUT:         return "TDC_E_LAYOUT";
        case TDC_E_SHAPE:          return "TDC_E_SHAPE";
        case TDC_E_BUF_TOO_SMALL:  return "TDC_E_BUF_TOO_SMALL";
        case TDC_E_CORRUPT:        return "TDC_E_CORRUPT";
        case TDC_E_VERSION:        return "TDC_E_VERSION";
        case TDC_E_IO:             return "TDC_E_IO";
    }
    return "<unknown>";
}

static void try_flip(uint8_t *rec_copy, size_t rec_size, size_t pos, const char *where) {
    tdc_block_record *hdr = (tdc_block_record *)rec_copy;
    size_t need = (size_t)hdr->uncompressed_size;
    tdc_block meta = {0};
    size_t peek_need = 0;
    if (qs_check(tdc_decode_peek(rec_copy, rec_size, &meta, &peek_need),
                 "peek")) return;
    void *dst = qs_realloc(NULL, NULL, need);
    tdc_block out = meta;
    out.data = dst;

    uint8_t saved = rec_copy[pos];
    rec_copy[pos] ^= 0xA5;                       /* flip several bits */
    tdc_status s = tdc_decode_block_into(rec_copy, rec_size, &out);
    rec_copy[pos] = saved;                       /* restore */
    printf("  flip @%-18s -> %s\n", where, status_name(s));
    qs_realloc(NULL, dst, 0);
}

int main(void) {
    /* 8 KiB of alternating 16-byte patterns (A, B, A, B, ...). LZ sees
     * repeated matches of length 16 at offset 32, so the payload carries
     * many sequence descriptors with real tag bytes and offsets for
     * corruption to hit. */
    enum { N = 8192 };
    static uint8_t data[N];
    for (int i = 0; i < N; ++i) {
        int block = i / 16;
        int off = i % 16;
        data[i] = (uint8_t)(((block & 1) ? 0x80 : 0x40) + off);
    }

    tdc_block src = {0};
    src.data = data;
    src.dtype = TDC_DT_U8;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank = 1;
    src.shape.dim[0] = N;
    tdc_shape_set_contiguous(&src.shape);

    tdc_codec_spec spec = tdc_codec_spec_raw();
    spec.entropy[0] = TDC_ENTROPY_LZ;

    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(&src, &spec, &enc), "encode")) return 1;

    size_t lz_off = lz_payload_offset(enc.data);
    printf("record: %zu bytes; LZ payload at offset %zu\n", enc.size, lz_off);

    /* Five flip sites. literals_size corruption and sequence-byte
     * corruption are always caught by the LZ decoder's bounds checks.
     * n_seq corruption sometimes slips through on inputs where the
     * existing sequences already fill the output: the si < n_seq && dp <
     * uncompressed_size loop terminates on the second inequality before
     * reading sequences that do not exist. A flipped byte deeper in the
     * record also flips one bit of the magic value in the block header,
     * which tdc_decode_peek rejects before the entropy stage even runs. */
    uint8_t *copy = (uint8_t *)qs_realloc(NULL, NULL, enc.size);
    memcpy(copy, enc.data, enc.size);
    try_flip(copy, enc.size, 0,               "header magic byte 0");
    try_flip(copy, enc.size, lz_off + 0,      "n_seq byte 0");
    try_flip(copy, enc.size, lz_off + 4,      "literals_size[0]");
    try_flip(copy, enc.size, lz_off + 8,      "first tag byte");
    try_flip(copy, enc.size, lz_off + 10,     "offset LSB");

    qs_realloc(NULL, copy, 0);
    qs_buffer_free(&enc);
    return 0;
}
