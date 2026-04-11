/*
 * src/entropy/lz2_internal.h
 *
 * Private header shared between lz2.c (greedy encoder + decoder) and
 * lz2_opt.c (optimal parser). Neither end is part of the public ABI.
 *
 * The two encoders emit the SAME on-disk LZ2 stream — optimal parsing is
 * a pure encode-side optimization. Decoding is handled by lz2.c's
 * lz2_decode_core regardless of which parser produced the stream, so
 * existing round-trip tests gate both.
 */

#ifndef TDC_ENTROPY_LZ2_INTERNAL_H
#define TDC_ENTROPY_LZ2_INTERNAL_H

#include "tdc/entropy.h"

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* On-disk constants (frozen). */
#define LZ2_MIN_MATCH   3u
#define LZ2_MAX_OFFSET  65536u  /* 16 bits + 1 */
#define LZ2_HEADER_SIZE 8u      /* uint32 n_sequences + uint32 literals_size */

/* Encoder-side sequence descriptor.
 *   lit_len   — literal bytes emitted before this match (0 .. N)
 *   match_len — actual match length, >= LZ2_MIN_MATCH
 *   match_off — back-reference offset, 1 .. LZ2_MAX_OFFSET
 */
typedef struct {
    uint32_t lit_len;
    uint32_t match_len;
    uint32_t match_off;
} LZ2Seq;

/* Bytes charged to the packed sequence header for a given literal-run
 * length and (match_len - LZ2_MIN_MATCH). Common case is 3 bytes: tag
 * byte + uint16 offset. Literal runs >= 15 add a chained-255 varint;
 * matches with match_len_m3 >= 15 add a LEB128 varint.
 *
 * Used by both the greedy encoder (serializer size check) and the
 * optimal parser (static cost model).
 */
static inline uint32_t lz2_leb128_size(uint32_t v) {
    uint32_t n = 1u;
    while (v >= 128u) { v >>= 7; n++; }
    return n;
}

static inline uint32_t lz2_seq_encoded_size(uint32_t lit_len,
                                            uint32_t match_len_m3) {
    uint32_t size = 1u + 2u; /* tag byte + uint16 offset */
    if (lit_len >= 15u) {
        uint32_t extra = lit_len - 15u;
        size += extra / 255u + 1u;
    }
    if (match_len_m3 >= 15u) {
        size += lz2_leb128_size(match_len_m3 - 15u);
    }
    return size;
}

/* Shared on-disk serializer (defined in lz2.c). Takes a pre-built LZ2Seq
 * array (from greedy or optimal parsing) and writes the on-disk LZ2
 * stream into dst. Literals are read directly from src between match
 * ends. Falls back to a literal-only stream if the parse would not
 * compress. */
tdc_status tdc_lz2_serialize_sequences(const uint8_t *src, uint32_t src_size,
                                       const LZ2Seq *seqs, uint32_t seq_count,
                                       tdc_buffer *dst);

#ifdef __cplusplus
}
#endif

#endif /* TDC_ENTROPY_LZ2_INTERNAL_H */
