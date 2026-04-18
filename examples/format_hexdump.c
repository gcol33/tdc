/* docs/examples/format_hexdump.c
 *
 * Encode a small deterministic block, then dump the resulting bytes to
 * stdout as annotated hex. Every byte of the block-record header is
 * labelled with the field name it belongs to, followed by the side
 * metadata, transform-params TLV, payload, and validity sections.
 *
 * The program uses the public tdc API only (include/tdc) and the
 * realloc_fn allocator convention from quickstart_common.h.
 *
 * Build (repo root, after `cmake --build build`):
 *   cc -I include docs/examples/format_hexdump.c \
 *      build/libtdc.a -lm -o /tmp/format_hexdump
 *   /tmp/format_hexdump
 */

#include "quickstart_common.h"
#include <inttypes.h>
#include <stdint.h>
#include <string.h>

/* Read a little-endian integer of n bytes from a pointer into uint64_t. */
static uint64_t rd_le(const uint8_t *p, size_t n) {
    uint64_t v = 0;
    for (size_t i = 0; i < n; ++i) v |= (uint64_t)p[i] << (8 * i);
    return v;
}

/* Print one annotated row: offset, bytes, and a field label. */
static void row(const uint8_t *base, size_t off, size_t n, const char *label) {
    printf("  %04zx  ", off);
    for (size_t i = 0; i < n; ++i) printf("%02x ", base[off + i]);
    for (size_t i = n; i < 8; ++i) printf("   ");
    printf("  %s\n", label);
}

/* Dump a contiguous byte range as offset+hex rows with no per-field label. */
static void raw_rows(const uint8_t *base, size_t start, size_t len,
                     const char *prefix) {
    const size_t width = 16;
    for (size_t off = start; off < start + len; off += width) {
        size_t remain = (start + len) - off;
        size_t take = remain < width ? remain : width;
        printf("  %04zx  ", off);
        for (size_t i = 0; i < take; ++i) printf("%02x ", base[off + i]);
        for (size_t i = take; i < width; ++i) printf("   ");
        printf("  %s\n", prefix);
    }
}

int main(void) {
    /* Deterministic input: 16-element i16 ramp. Small enough that every
     * section fits on one screen of hex output. */
    int16_t src_data[16];
    for (int i = 0; i < 16; ++i) src_data[i] = (int16_t)(100 + i);

    tdc_block src = {0};
    src.data   = src_data;
    src.dtype  = TDC_DT_I16;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank   = 1;
    src.shape.dim[0] = 16;
    tdc_shape_set_contiguous(&src.shape);

    /* DELTA_1D + ZIGZAG, no entropy: exercises model side metadata (1 byte
     * for delta order) and a transform with no TLV params, keeping the
     * payload tiny and easy to read in hex. */
    tdc_codec_spec spec = {0};
    spec.model    = TDC_MODEL_DELTA_1D;
    spec.xform[0] = TDC_XFORM_ZIGZAG;

    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(&src, &spec, &enc), "encode")) return 1;

    const uint8_t *b = enc.data;
    const size_t   n = enc.size;

    printf("input   : 16 x i16 = 32 bytes\n");
    printf("encoded : %zu bytes (80-byte header + side + payload)\n\n", n);

    /* ---- Block-record header (80 bytes) -------------------------------- */
    printf("block record header:\n");
    row(b,  0, 4, "magic            ('BLKB')");
    row(b,  4, 2, "version");
    row(b,  6, 2, "flags");
    row(b,  8, 2, "model_id");
    row(b, 10, 8, "xform_ids[4]    (u16 x 4)");
    row(b, 18, 8, "entropy_ids[4]  (u16 x 4)");
    row(b, 26, 1, "dtype");
    row(b, 27, 1, "layout");
    row(b, 28, 1, "rank");
    row(b, 29, 1, "_reserved0");
    row(b, 30, 2, "_reserved_pad");
    row(b, 32, 8, "dim[0]          (int64)");
    row(b, 40, 8, "dim[1]          (int64)");
    row(b, 48, 8, "dim[2]          (int64)");
    row(b, 56, 8, "uncompressed_size (u64)");
    row(b, 64, 4, "side_meta_size   (u32)");
    row(b, 68, 4, "payload_size     (u32)");
    row(b, 72, 4, "xform_params_size(u32)");
    row(b, 76, 4, "validity_size    (u32)");

    /* ---- Decoded field values ----------------------------------------- */
    uint32_t magic         = (uint32_t)rd_le(b + 0,  4);
    uint16_t version       = (uint16_t)rd_le(b + 4,  2);
    uint16_t flags         = (uint16_t)rd_le(b + 6,  2);
    uint16_t model_id      = (uint16_t)rd_le(b + 8,  2);
    uint64_t uncomp        =            rd_le(b + 56, 8);
    uint32_t side_size     = (uint32_t)rd_le(b + 64, 4);
    uint32_t payload_size  = (uint32_t)rd_le(b + 68, 4);
    uint32_t xform_ps_size = (uint32_t)rd_le(b + 72, 4);
    uint32_t validity_size = (uint32_t)rd_le(b + 76, 4);

    printf("\ndecoded header fields:\n");
    printf("  magic             = 0x%08" PRIx32 " (expect 0x%08x)\n",
           magic, TDC_BLOCK_MAGIC);
    printf("  version           = %" PRIu16 "\n", version);
    printf("  flags             = 0x%04" PRIx16 "\n", flags);
    printf("  model_id          = 0x%04" PRIx16 "\n", model_id);
    printf("  uncompressed_size = %" PRIu64 " bytes\n", uncomp);
    printf("  side_meta_size    = %" PRIu32 "\n", side_size);
    printf("  xform_params_size = %" PRIu32 "\n", xform_ps_size);
    printf("  payload_size      = %" PRIu32 "\n", payload_size);
    printf("  validity_size     = %" PRIu32 "\n", validity_size);

    /* ---- Side metadata, xform params TLV, payload, validity ----------- */
    size_t off = TDC_BLOCK_HEADER_SIZE;
    if (side_size) {
        printf("\nside metadata (%zu bytes):\n", (size_t)side_size);
        raw_rows(b, off, side_size, "side_meta");
        off += side_size;
    }
    if (xform_ps_size) {
        printf("\nxform params TLV (%zu bytes):\n", (size_t)xform_ps_size);
        raw_rows(b, off, xform_ps_size, "xform_params");
        off += xform_ps_size;
    }
    if (payload_size) {
        printf("\npayload (%zu bytes):\n", (size_t)payload_size);
        raw_rows(b, off, payload_size, "payload");
        off += payload_size;
    }
    if (validity_size) {
        printf("\nvalidity (%zu bytes):\n", (size_t)validity_size);
        raw_rows(b, off, validity_size, "validity");
        off += validity_size;
    }
    printf("\ntotal on-disk bytes: %zu\n", off);

    /* ---- Round trip to confirm the bytes round-trip cleanly ----------- */
    tdc_block meta = {0};
    size_t need = 0;
    if (qs_check(tdc_decode_peek(b, n, &meta, &need), "peek")) return 1;

    void *dst_data = qs_realloc(NULL, NULL, need);
    tdc_block dst = meta;
    dst.data = dst_data;
    if (qs_check(tdc_decode_block_into(b, n, &dst), "decode")) return 1;
    int ok = memcmp(dst_data, src_data, sizeof(src_data)) == 0;
    printf("\nround trip memcmp == source: %s\n", ok ? "yes" : "NO");

    qs_realloc(NULL, dst_data, 0);
    qs_buffer_free(&enc);
    return ok ? 0 : 1;
}
