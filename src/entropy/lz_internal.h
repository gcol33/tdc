/*
 * src/entropy/lz_internal.h
 *
 * Private header shared between lz.c (greedy encoder + decoder) and
 * lz_opt.c (optimal parser). Neither end is part of the public ABI.
 *
 * The two encoders emit the SAME on-disk LZ stream — optimal parsing is
 * a pure encode-side optimization. Decoding is handled by lz.c's
 * lz_decode_core regardless of which parser produced the stream, so
 * existing round-trip tests gate both.
 */

#ifndef TDC_ENTROPY_LZ_INTERNAL_H
#define TDC_ENTROPY_LZ_INTERNAL_H

#include "tdc/entropy.h"

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* On-disk constants (frozen). */
#define LZ_MIN_MATCH   3u
#define LZ_MAX_OFFSET  65536u  /* 16 bits + 1 */
#define LZ_HEADER_SIZE 8u      /* uint32 n_sequences + uint32 literals_size */

/* Encoder-side sequence descriptor.
 *   lit_len   — literal bytes emitted before this match (0 .. N)
 *   match_len — actual match length, >= LZ_MIN_MATCH
 *   match_off — back-reference offset, 1 .. LZ_MAX_OFFSET
 */
typedef struct {
    uint32_t lit_len;
    uint32_t match_len;
    uint32_t match_off;
} LZSeq;

/* Bytes charged to the packed sequence header for a given literal-run
 * length and (match_len - LZ_MIN_MATCH). Common case is 3 bytes: tag
 * byte + uint16 offset. Literal runs >= 15 add a chained-255 varint;
 * matches with match_len_m3 >= 15 add a LEB128 varint.
 *
 * Used by both the greedy encoder (serializer size check) and the
 * optimal parser (static cost model).
 */
static inline uint32_t lz_leb128_size(uint32_t v) {
    uint32_t n = 1u;
    while (v >= 128u) { v >>= 7; n++; }
    return n;
}

static inline uint32_t lz_seq_encoded_size(uint32_t lit_len,
                                            uint32_t match_len_m3) {
    uint32_t size = 1u + 2u; /* tag byte + uint16 offset */
    if (lit_len >= 15u) {
        uint32_t extra = lit_len - 15u;
        size += extra / 255u + 1u;
    }
    if (match_len_m3 >= 15u) {
        size += lz_leb128_size(match_len_m3 - 15u);
    }
    return size;
}

/* Shared on-disk serializer (defined in lz.c). Takes a pre-built LZSeq
 * array (from greedy or optimal parsing) and writes the on-disk LZ
 * stream into dst. Literals are read directly from src between match
 * ends. Falls back to a literal-only stream if the parse would not
 * compress. */
tdc_status tdc_lz_serialize_sequences(const uint8_t *src, uint32_t src_size,
                                       const LZSeq *seqs, uint32_t seq_count,
                                       tdc_buffer *dst);

/* Shared greedy match-finder (defined in lz.c). Walks src once with the
 * 64K-window flat hash table and emits an LZSeq array. The caller frees
 * *out_seqs via dst->realloc_fn (allocated through it). Used by the
 * single-stream serializer (TDC_ENTROPY_LZ) and the multi-stream
 * serializer (TDC_ENTROPY_LZ_STREAMS). Both share this parser; the only
 * difference is what they do with the resulting LZSeq array. */
tdc_status tdc_lz_parse_greedy(const uint8_t *src, uint32_t src_size,
                                tdc_buffer *dst,
                                LZSeq **out_seqs, uint32_t *out_seq_count);

/* Shared optimal-parser match-finder (defined in lz_opt.c). Walks src
 * with a hash-chain match finder (depth ≤ 128) and runs a forward DP
 * over the single-stream cost model (literals = 8 bits, matches = 24 bits
 * + match_ext + bracket penalty for long literal runs). Emits the
 * minimum-cost LZSeq array. Same output contract as tdc_lz_parse_greedy:
 * caller frees *out_seqs via dst->realloc_fn. Used by TDC_ENTROPY_LZ_OPT
 * for single-stream encoding. */
tdc_status tdc_lz_parse_optimal(const uint8_t *src, uint32_t src_size,
                                 tdc_buffer *dst,
                                 LZSeq **out_seqs, uint32_t *out_seq_count);

/* STREAMS-cost optimal parser (defined in lz_opt.c). Same match-finder
 * as tdc_lz_parse_optimal but runs the DP with the multi-stream cost
 * model: literals ≈ 6 bits (entropy-coded), match header ≈ 31 bits
 * (ll + ml + mo combined), no match_ext, no lit-run bracket penalty.
 * The resulting parse is optimal for the LZ_STREAMS on-disk format, not
 * the single-stream format. Same output contract as the legacy variant.
 * Used by TDC_ENTROPY_LZ_STREAMS. */
tdc_status tdc_lz_parse_optimal_streams(const uint8_t *src, uint32_t src_size,
                                         tdc_buffer *dst,
                                         LZSeq **out_seqs,
                                         uint32_t *out_seq_count);

#ifdef __cplusplus
}
#endif

#endif /* TDC_ENTROPY_LZ_INTERNAL_H */
