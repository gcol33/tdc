/*
 * src/entropy/lz_streams.c
 *
 * TDC_ENTROPY_LZ_STREAMS — LZ parse with four separated, entropy-coded
 * streams.
 *
 * Shares the greedy parser (tdc_lz_parse_greedy) with TDC_ENTROPY_LZ.
 * Where the single-stream serializer interleaves sequence descriptors and
 * literals into one packed byte stream, this backend splits the parser
 * output into four streams and applies an independent entropy coder to
 * each:
 *
 *   literal_stream   — all literal bytes concatenated (includes trailing)
 *   lit_len_stream   — n_seqs × u32 lit_len, byte-shuffled
 *   match_len_stream — n_seqs × u32 match_len, byte-shuffled
 *   offset_sym_stream — n_seqs × u8 log2-prefix symbol (v2)
 *   offset_extra      — packed extra bits for novel offsets (v2, raw)
 *
 * Format v2 (current) uses log2-prefix offset encoding: after repcodes,
 * each offset is split into a symbol (0-23, entropy-coded) and extra
 * bits (packed raw). Symbols 0-2 are repcodes (0 extra bits), symbols
 * 3-23 encode floor(log2(offset))+3 with the remainder as extra bits.
 * This compresses the offset stream far better than v1's byte-shuffled
 * u32, especially for structured data with dominant strides.
 *
 * Each stream picks between NONE / HUFFMAN / FSE by trying all coders
 * and keeping the smallest. A whole-block fallback also exists: if the
 * full encoded size meets or exceeds src_size, the header is written
 * with n_seqs=0 and the lit stream carries src as passthrough NONE.
 *
 * On-disk header v2 (44 bytes, fixed):
 *
 *   offset  size  field
 *     0      1    format_version         (2)
 *     1      1    flags                  (bit 0: LZS_FLAG_REPCODES)
 *     2      2    reserved               (zero)
 *     4      4    n_seqs                 (0 => passthrough fallback)
 *     8      4    total_lit_size         (bytes in literal_stream)
 *    12      4    trailing_lit_len       (literals after last match)
 *    16      4    uncompressed_size      (== dst_size on decode)
 *    20      4    entropy_id[4]          (lit, lit_len, match_len, off_sym)
 *    24      4    stream_size_lit
 *    28      4    stream_size_lit_len
 *    32      4    stream_size_match_len
 *    36      4    stream_size_off_sym
 *    40      4    stream_size_off_extra
 *
 * followed by the five streams: lit, ll, ml, off_sym, off_extra.
 *
 * Repcode transform (when LZS_FLAG_REPCODES is set):
 *
 * Instead of the raw match_off (1..65536), the match_off stream carries an
 * offset_code that exploits offset-reuse locality. Three recent offsets
 * (rep1, rep2, rep3, initialised to 1, 4, 8 — the zstd defaults) are
 * maintained as an MRU list during parsing. On each sequence:
 *
 *   match_off == rep1  ->  code = 1           (state unchanged)
 *   match_off == rep2  ->  code = 2           (rep1,rep2 swap)
 *   match_off == rep3  ->  code = 3           (rep3 promoted, others shift)
 *   otherwise          ->  code = off + 3     (new offset becomes rep1)
 *
 * Structured tabular data hammers a few strides (8 for f64 columns, 1 for
 * RLE, the row width for per-row repetition). Under repcodes the match_off
 * stream collapses to a histogram dominated by the single symbol "1",
 * which the downstream entropy coder (Huffman or FSE) compresses to
 * near-zero bits/symbol. The decoder reverses the transform with the same
 * rep-state walk before reconstructing the output.
 */

#include "tdc/entropy.h"
#include "tdc/codec.h"
#include "entropy_internal.h"
#include "lz_internal.h"
#include "../core/buffer.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Diagnostic dump (enabled at runtime via TDC_LZS_DUMP=1). Prints, per
 * encode call: src_size, n_seqs, per-stream raw and compressed sizes, and
 * the coder id each stream ended up with. Use this to see where the bytes
 * are going on a given workload before chasing micro-optimizations. */
static int lzs_dump_enabled(void) {
    static int checked = 0;
    static int enabled = 0;
    if (!checked) {
        const char *v = getenv("TDC_LZS_DUMP");
        enabled = (v && v[0] && v[0] != '0');
        checked = 1;
    }
    return enabled;
}

#define LZS_HEADER_SIZE_V1   40u
#define LZS_HEADER_SIZE_V2   44u
#define LZS_FORMAT_VERSION   2u
#define LZS_FLAG_REPCODES    0x01u
#define LZS_FLAG_OFFSET_SHIFT 0x02u  /* offsets right-shifted by header[2] bits */

/* Initial repcode values (match zstd's rep-init). */
#define LZS_REP_INIT_1       1u
#define LZS_REP_INIT_2       4u
#define LZS_REP_INIT_3       8u

/* ----- Log2-prefix offset encoding (format v2) --------------------------- */
/*
 * After repcode transform, offset codes are:
 *   1-3:  repcodes  → symbol 0-2, 0 extra bits
 *   >= 4: novel     → actual_off = code-3, symbol = floor_log2(off)+3,
 *                      extra = off - (1 << floor_log2(off)), nbits extra bits
 *
 * Symbol alphabet: 0..LZS_MAX_OFFSET_SYMBOL (24 symbols for 20-bit window).
 * The entropy coder sees only this small alphabet; extra bits are packed
 * separately as a raw bitstream.
 */
#define LZS_MAX_OFFSET_SYMBOL 23u  /* floor_log2(LZ_MAX_OFFSET) + 3 */

static inline uint32_t lzs_floor_log2(uint32_t v) {
    uint32_t r = 0;
    while (v > 1u) { v >>= 1; r++; }
    return r;
}

/* --- Generic log2-prefix for non-negative integers ----------------------- *
 *
 * Encodes v >= 0 as (symbol, extra_bits):
 *   v = 0  → sym 0, 0 extra bits
 *   v = 1  → sym 1, 0 extra bits
 *   v >= 2 → sym = floor_log2(v) + 1, nbits = floor_log2(v),
 *             extra = v - (1 << nbits)
 *
 * Used for lit_len (v = lit_len) and match_len (v = match_len - 3).
 */
static inline void lzs_uint_to_symbol(uint32_t v,
                                        uint8_t *sym,
                                        uint32_t *extra,
                                        uint32_t *nbits) {
    if (v <= 1u) {
        *sym = (uint8_t)v;
        *extra = 0;
        *nbits = 0;
    } else {
        uint32_t lg = lzs_floor_log2(v);
        *sym   = (uint8_t)(lg + 1u);
        *extra = v - (1u << lg);
        *nbits = lg;
    }
}

static inline uint32_t lzs_symbol_to_uint(uint8_t sym, uint32_t extra) {
    if (sym <= 1u) return (uint32_t)sym;
    uint32_t lg = (uint32_t)sym - 1u;
    return (1u << lg) + extra;
}

/* Max symbol for lit_len: lit_len can be up to src_size (~1 MiB).
 * floor_log2(1048576) + 1 = 21. */
#define LZS_MAX_LL_SYMBOL  21u
/* Max symbol for match_len_m3: up to ~1 MiB - 3. Same range. */
#define LZS_MAX_ML_SYMBOL  21u

/* --- Offset log2-prefix with repcode symbols ----------------------------- */

static inline void lzs_offset_to_symbol(uint32_t code,
                                          uint8_t *sym,
                                          uint32_t *extra,
                                          uint32_t *nbits) {
    if (code <= 3u) {
        *sym = (uint8_t)(code - 1u);
        *extra = 0;
        *nbits = 0;
    } else {
        uint32_t off = code - 3u;
        uint32_t lg  = lzs_floor_log2(off);
        *sym   = (uint8_t)(lg + 3u);
        *extra = off - (1u << lg);
        *nbits = lg;
    }
}

static inline uint32_t lzs_symbol_to_code(uint8_t sym, uint32_t extra) {
    if (sym <= 2u) return (uint32_t)sym + 1u;
    uint32_t lg = (uint32_t)sym - 3u;
    return (1u << lg) + extra + 3u;
}

/* Bit packing — caller must memset buffer to zero before writing. */
static inline void lzs_bits_write(uint8_t *buf, uint32_t *bit_pos,
                                    uint32_t val, uint32_t nbits) {
    uint32_t pos = *bit_pos;
    for (uint32_t i = 0; i < nbits; i++) {
        buf[pos >> 3] |= (uint8_t)(((val >> i) & 1u) << (pos & 7u));
        pos++;
    }
    *bit_pos = pos;
}

static inline uint32_t lzs_bits_read(const uint8_t *buf, uint32_t *bit_pos,
                                       uint32_t nbits) {
    uint32_t pos = *bit_pos;
    uint32_t val = 0;
    for (uint32_t i = 0; i < nbits; i++) {
        val |= (uint32_t)((buf[pos >> 3] >> (pos & 7u)) & 1u) << i;
        pos++;
    }
    *bit_pos = pos;
    return val;
}

/* ----- Allocation helpers ------------------------------------------------- */

static void *lzs_alloc(tdc_buffer *buf, size_t n) {
    return buf->realloc_fn(buf->user, NULL, n);
}

static void lzs_free(tdc_buffer *buf, void *p) {
    if (p) (void)buf->realloc_fn(buf->user, p, 0);
}

/* ----- Sub-coder vtable lookup ------------------------------------------- */

static const tdc_entropy_vt *lzs_sub_vt(tdc_entropy_id id) {
    switch (id) {
        case TDC_ENTROPY_NONE:    return &tdc_entropy_none_vt;
        case TDC_ENTROPY_HUFFMAN: return &tdc_entropy_huffman_vt;
        case TDC_ENTROPY_FSE:     return &tdc_entropy_fse_vt;
        default:                  return NULL;
    }
}

/* Stream coder selection: try all three (NONE / HUFFMAN / FSE) and keep
 * the smallest output. The previous Shannon-entropy heuristic misclassified
 * the match-offset stream after repcodes — Shannon put it in the "medium
 * entropy, use FSE" bucket, but tdc's FSE compresses a near-constant
 * histogram ~5× worse than Huffman does. Measuring both directly removes
 * the heuristic error at the cost of 2-3× encode time, which is explicitly
 * acceptable in SMALL mode.
 *
 * NONE is included as a floor so the "coder expanded the stream" case is
 * handled in the same comparison instead of as a separate fallback. */

/* ----- Repcode transform -------------------------------------------------- */
/*
 * Encode: walk the LZSeq array, replacing each .match_off with an
 * offset_code in [1, LZ_MAX_OFFSET + 3]. Codes 1/2/3 encode rep1/rep2/rep3;
 * any other code is (actual_offset + 3). Rep state is maintained MRU.
 *
 * Decode: walk the unshuffled match_off stream, replacing each offset_code
 * with its actual_offset. Same rep state walk as encode — the two ends
 * stay in lockstep by construction.
 */
/* Compute the largest power-of-2 that divides all match offsets.
 * Returns the shift (trailing zeros), clamped to 0-7. */
static uint32_t lzs_detect_offset_shift(const LZSeq *seqs, uint32_t n_seqs) {
    if (n_seqs == 0) return 0;
    uint32_t all_or = 0;
    for (uint32_t i = 0; i < n_seqs; i++) {
        all_or |= seqs[i].match_off;
    }
    if (all_or == 0) return 0;
    uint32_t shift = 0;
    while (shift < 7u && (all_or & (1u << shift)) == 0) shift++;
    return shift;
}

static void lzs_repcode_encode(LZSeq *seqs, uint32_t n_seqs) {
    uint32_t r1 = LZS_REP_INIT_1;
    uint32_t r2 = LZS_REP_INIT_2;
    uint32_t r3 = LZS_REP_INIT_3;
    for (uint32_t i = 0; i < n_seqs; i++) {
        uint32_t off = seqs[i].match_off;
        uint32_t code;
        if (off == r1) {
            code = 1u;
            /* state unchanged */
        } else if (off == r2) {
            code = 2u;
            uint32_t t = r1; r1 = r2; r2 = t;
        } else if (off == r3) {
            code = 3u;
            uint32_t t = r3; r3 = r2; r2 = r1; r1 = t;
        } else {
            code = off + 3u;
            r3 = r2; r2 = r1; r1 = off;
        }
        seqs[i].match_off = code;
    }
}

static tdc_status lzs_repcode_decode(uint32_t *match_offs, uint32_t n_seqs) {
    uint32_t r1 = LZS_REP_INIT_1;
    uint32_t r2 = LZS_REP_INIT_2;
    uint32_t r3 = LZS_REP_INIT_3;
    for (uint32_t i = 0; i < n_seqs; i++) {
        uint32_t code = match_offs[i];
        uint32_t off;
        if (code == 0u) return TDC_E_CORRUPT;
        if (code == 1u) {
            off = r1;
        } else if (code == 2u) {
            off = r2;
            uint32_t t = r1; r1 = r2; r2 = t;
        } else if (code == 3u) {
            off = r3;
            uint32_t t = r3; r3 = r2; r2 = r1; r1 = t;
        } else {
            off = code - 3u;
            if (off == 0u || off > LZ_MAX_OFFSET) return TDC_E_CORRUPT;
            r3 = r2; r2 = r1; r1 = off;
        }
        match_offs[i] = off;
    }
    return TDC_OK;
}

/* ----- Byte-shuffle for u32 arrays --------------------------------------- */
/*
 * Transpose n u32 values into 4 contiguous byte lanes of n bytes each:
 *   lane0[i] = v[i]      & 0xFF
 *   lane1[i] = v[i] >> 8 & 0xFF
 *   lane2[i] = v[i] >>16 & 0xFF
 *   lane3[i] = v[i] >>24 & 0xFF
 *
 * Separating the lanes exposes per-lane structure: lane 3 (high byte) is
 * almost always zero for lit/match lengths and for most match offsets,
 * which is exactly the shape HUFFMAN and FSE compress well.
 */
static void lzs_shuffle_u32(const uint32_t *src, uint32_t n, uint8_t *dst) {
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t v = src[i];
        dst[i]                = (uint8_t)(v);
        dst[(size_t)n + i]    = (uint8_t)(v >> 8);
        dst[2u * (size_t)n + i] = (uint8_t)(v >> 16);
        dst[3u * (size_t)n + i] = (uint8_t)(v >> 24);
    }
}

static void lzs_unshuffle_u32(const uint8_t *src, uint32_t n, uint32_t *dst) {
    for (uint32_t i = 0; i < n; ++i) {
        dst[i] = (uint32_t)src[i]
               | ((uint32_t)src[(size_t)n + i]          << 8)
               | ((uint32_t)src[2u * (size_t)n + i]     << 16)
               | ((uint32_t)src[3u * (size_t)n + i]     << 24);
    }
}

/* ----- Encode one stream, try all coders, keep the smallest -------------- */
/*
 * Runs HUFFMAN and FSE against the input and compares their encoded sizes
 * against the NONE floor (n bytes). Whichever is smallest is written into
 * `final` (a per-stream destination buffer owned by the caller) and its
 * coder id is returned via *out_id. `scratch` is a tdc_buffer reused
 * across candidate coders; `final` receives a copy of the winning output.
 *
 * `final` is allocated here with dst->realloc_fn and transferred to the
 * caller, who is responsible for freeing via lzs_free(dst, *final).
 */
static tdc_status lzs_encode_stream(const uint8_t *src, size_t n,
                                     tdc_buffer    *scratch,
                                     tdc_buffer    *owner,   /* for realloc_fn */
                                     uint8_t       **final,
                                     uint32_t       *final_size,
                                     tdc_entropy_id *out_id) {
    *final = NULL;
    *final_size = 0;
    *out_id = TDC_ENTROPY_NONE;

    if (n == 0) {
        return TDC_OK;
    }

    /* Best-so-far is "NONE passthrough" at exactly n bytes. Only a coder
     * that strictly beats n wins. */
    size_t         best_size = n;
    tdc_entropy_id best_id   = TDC_ENTROPY_NONE;
    int            best_is_encoded = 0;

    /* Keep a stash of the winning encoded bytes across candidate coders.
     * Each candidate writes into scratch; on a win we snapshot into stash
     * so the next candidate's encode doesn't clobber it. */
    uint8_t *stash = NULL;
    size_t   stash_size = 0;

    static const tdc_entropy_id candidates[] = {
        TDC_ENTROPY_HUFFMAN,
        TDC_ENTROPY_FSE,
    };
    for (size_t c = 0; c < sizeof(candidates) / sizeof(candidates[0]); c++) {
        tdc_entropy_id id = candidates[c];
        const tdc_entropy_vt *sub = lzs_sub_vt(id);
        if (!sub) continue;

        scratch->size = 0;
        tdc_status st = sub->encode(src, n, NULL, scratch);
        if (st != TDC_OK) {
            /* Coder refused this input — skip, don't abort the whole block. */
            continue;
        }
        if (scratch->size >= best_size) continue;

        /* New champion — snapshot into stash. */
        uint8_t *new_stash = (uint8_t *)owner->realloc_fn(
            owner->user, stash, scratch->size);
        if (!new_stash) {
            if (stash) (void)owner->realloc_fn(owner->user, stash, 0);
            return TDC_E_NOMEM;
        }
        stash = new_stash;
        stash_size = scratch->size;
        memcpy(stash, scratch->data, stash_size);
        best_size = stash_size;
        best_id   = id;
        best_is_encoded = 1;
    }

    if (best_is_encoded) {
        *final      = stash;
        *final_size = (uint32_t)best_size;
        *out_id     = best_id;
        return TDC_OK;
    }

    /* No coder beat NONE. Write NONE passthrough into a fresh buffer.
     * (We can't reuse stash because it's NULL in this branch.) */
    uint8_t *none_buf = (uint8_t *)owner->realloc_fn(owner->user, NULL, n);
    if (!none_buf) return TDC_E_NOMEM;
    memcpy(none_buf, src, n);
    *final      = none_buf;
    *final_size = (uint32_t)n;
    *out_id     = TDC_ENTROPY_NONE;
    return TDC_OK;
}

/* ----- Encode bound ------------------------------------------------------ */

static size_t lzs_encode_bound(size_t src_size) {
    /* Worst case: literal stream = src_size, two u32 streams (ll, ml)
     * at most 4 * src_size each, symbol stream at most src_size, extra
     * bits stream at most 20 * src_size / 8. Each sub-coder adds at
     * most ~1 KiB of headers. Plus the 44-byte v2 header. */
    return LZS_HEADER_SIZE_V2 + src_size + 2u * src_size
         + src_size + 3u * src_size + 5u * 1024u;
}

/* ----- Passthrough fallback --------------------------------------------- */
/*
 * Writes an LZ_STREAMS record that carries src byte-for-byte in the
 * literal stream with NONE coding. Used when the full encoded form
 * (parse + four entropy-coded streams) would match or exceed src_size.
 */
static tdc_status lzs_encode_passthrough(const uint8_t *src, uint32_t src_size,
                                          tdc_buffer *dst) {
    size_t need = (size_t)LZS_HEADER_SIZE_V2 + src_size;
    tdc_status st = tdc_buf_reserve(dst, need);
    if (st != TDC_OK) return st;

    uint8_t *p = dst->data;
    uint32_t zero = 0;
    p[0] = (uint8_t)LZS_FORMAT_VERSION;
    p[1] = 0;                              /* flags — no repcodes in passthrough */
    p[2] = 0;
    p[3] = 0;
    memcpy(p +  4, &zero,     4); /* n_seqs */
    memcpy(p +  8, &src_size, 4); /* total_lit_size */
    memcpy(p + 12, &src_size, 4); /* trailing_lit_len */
    memcpy(p + 16, &src_size, 4); /* uncompressed_size */
    p[20] = (uint8_t)TDC_ENTROPY_NONE; /* id_lit */
    p[21] = (uint8_t)TDC_ENTROPY_NONE; /* id_ll  */
    p[22] = (uint8_t)TDC_ENTROPY_NONE; /* id_ml  */
    p[23] = (uint8_t)TDC_ENTROPY_NONE; /* id_mo_sym */
    memcpy(p + 24, &src_size, 4);      /* sz_lit */
    memset(p + 28, 0, 16);             /* sz_ll, sz_ml, sz_mo_sym, sz_extra = 0 */

    if (src_size > 0) memcpy(p + LZS_HEADER_SIZE_V2, src, src_size);
    dst->size = need;
    return TDC_OK;
}

/* ----- Price computation from first-pass frequencies ---------------------- *
 *
 * After the first (flat-cost) parse, we have the three symbol streams and
 * the literal stream. From these we compute per-symbol costs via Shannon
 * entropy: cost(s) = ceil(-log2(freq/total)) in 1/8-bit fixed-point.
 * Extra-bit costs are added on top: symbols with wider extra-bit fields
 * carry those bits at 1 bit each (= 8 in 1/8-bit units).
 *
 * These costs feed the second (priced) parse, which can make better
 * decisions about match vs. literal trade-offs because it knows the
 * actual entropy of each symbol rather than using flat estimates.
 */

/* Fixed-point scale: costs are in 1/8-bit units (multiply real bits × 8). */
#define LZS_PRICE_SCALE  8

/* Shannon cost for a symbol with frequency `freq` in a stream of `total`
 * symbols. Returns cost in 1/8-bit units. Clamps to [8, 120] (1-15 bits). */
static inline int32_t lzs_shannon_cost(uint32_t freq, uint32_t total) {
    if (freq == 0 || total == 0) return 15 * LZS_PRICE_SCALE; /* penalty */
    double p = (double)freq / (double)total;
    double bits = -log2(p);
    int32_t cost = (int32_t)(bits * LZS_PRICE_SCALE + 0.5);
    if (cost < LZS_PRICE_SCALE) cost = LZS_PRICE_SCALE;       /* min 1 bit */
    if (cost > 15 * LZS_PRICE_SCALE) cost = 15 * LZS_PRICE_SCALE;
    return cost;
}

/* Build per-symbol price tables from first-pass symbol frequencies. */
static void lzs_compute_prices(const uint8_t *lit_raw, uint32_t total_lit,
                                const uint8_t *ll_sym, const uint8_t *ml_sym,
                                const uint8_t *off_sym, uint32_t n_seqs,
                                LzsStreamsPrices *prices) {
    /* --- Literal byte frequencies --- */
    uint32_t lit_freq[256];
    memset(lit_freq, 0, sizeof(lit_freq));
    for (uint32_t i = 0; i < total_lit; i++)
        lit_freq[lit_raw[i]]++;
    for (int i = 0; i < 256; i++)
        prices->lit[i] = lzs_shannon_cost(lit_freq[i], total_lit);

    /* --- Lit-len symbol frequencies --- */
    uint32_t ll_freq[LZS_PRICE_MAX_SYM];
    memset(ll_freq, 0, sizeof(ll_freq));
    for (uint32_t i = 0; i < n_seqs; i++) {
        uint8_t s = ll_sym[i];
        if (s < LZS_PRICE_MAX_SYM) ll_freq[s]++;
    }

    /* ll_avg: average ll symbol cost + average extra bits.
     * The ll cost enters once per sequence in the DP; we use a flat
     * average rather than per-symbol because the DP decides lit_len
     * implicitly via the prefix-min trick — it controls where matches
     * start, and the literal run between matches falls out. */
    {
        int64_t total_ll_cost = 0;
        for (uint32_t s = 0; s < LZS_PRICE_MAX_SYM; s++) {
            if (ll_freq[s] == 0) continue;
            int32_t sym_cost = lzs_shannon_cost(ll_freq[s], n_seqs);
            uint32_t extra = (s >= 2u) ? (s - 1u) : 0u;
            total_ll_cost += (int64_t)(sym_cost + (int32_t)(extra * LZS_PRICE_SCALE))
                           * ll_freq[s];
        }
        prices->ll_avg = (n_seqs > 0)
            ? (int32_t)(total_ll_cost / (int64_t)n_seqs)
            : 10 * LZS_PRICE_SCALE;
    }

    /* --- Match-len symbol frequencies + extra bits --- */
    uint32_t ml_freq[LZS_PRICE_MAX_SYM];
    memset(ml_freq, 0, sizeof(ml_freq));
    for (uint32_t i = 0; i < n_seqs; i++) {
        uint8_t s = ml_sym[i];
        if (s < LZS_PRICE_MAX_SYM) ml_freq[s]++;
    }
    for (uint32_t s = 0; s < LZS_PRICE_MAX_SYM; s++) {
        int32_t sym_cost = lzs_shannon_cost(ml_freq[s], n_seqs);
        uint32_t extra = (s >= 2u) ? (s - 1u) : 0u;
        prices->ml[s] = sym_cost + (int32_t)(extra * LZS_PRICE_SCALE);
    }

    /* --- Offset symbol frequencies + extra bits --- */
    uint32_t off_freq[LZS_PRICE_MAX_SYM];
    memset(off_freq, 0, sizeof(off_freq));
    for (uint32_t i = 0; i < n_seqs; i++) {
        uint8_t s = off_sym[i];
        if (s < LZS_PRICE_MAX_SYM) off_freq[s]++;
    }
    for (uint32_t s = 0; s < LZS_PRICE_MAX_SYM; s++) {
        int32_t sym_cost = lzs_shannon_cost(off_freq[s], n_seqs);
        uint32_t extra = (s >= 3u) ? (s - 3u) : 0u;
        prices->off[s] = sym_cost + (int32_t)(extra * LZS_PRICE_SCALE);
    }
}

/* ----- Encode ------------------------------------------------------------- */

static tdc_status lzs_encode(const uint8_t *src, size_t src_size,
                              const void    *params,
                              tdc_buffer    *dst) {
    (void)params;
    if (!dst || !dst->realloc_fn) return TDC_E_INVAL;
    if (src_size > 0 && !src)     return TDC_E_INVAL;
    if (src_size > UINT32_MAX)    return TDC_E_INVAL;

    /* Empty input: emit an empty passthrough record. */
    if (src_size == 0) {
        return lzs_encode_passthrough(src, 0u, dst);
    }

    /* --- Multi-pass encoding ---
     * Pass 1: parse with approximate cost model (offset-aware flat costs)
     * Pass 2..N: extract per-symbol prices from previous parse, re-parse
     *
     * Each additional pass refines the price model toward convergence.
     * Diminishing returns after 3 passes on typical data. */
#define LZS_N_PASSES 3

    LZSeq *seqs = NULL;
    uint32_t seq_count = 0;
    tdc_status st = tdc_lz_parse_optimal_streams(src, (uint32_t)src_size, dst,
                                                  &seqs, &seq_count);
    if (st != TDC_OK) return st;

    if (seq_count == 0) {
        return lzs_encode_passthrough(src, (uint32_t)src_size, dst);
    }

    /* Detect offset stride from the first pass. The parser emits raw byte
     * offsets; for typed data these are always multiples of the element
     * width (e.g. 8 for f64). Shifting offsets right by this factor
     * before repcode/symbol encoding removes redundant low bits. */
    uint32_t off_shift = lzs_detect_offset_shift(seqs, seq_count);

    for (int pass = 1; pass < LZS_N_PASSES; pass++) {
        /* Shift offsets (parser produces unshifted; we shift for encoding). */
        if (off_shift > 0) {
            for (uint32_t i = 0; i < seq_count; i++)
                seqs[i].match_off >>= off_shift;
        }

        /* Apply repcodes, build symbol streams, compute prices. */
        lzs_repcode_encode(seqs, seq_count);

        uint64_t tl64 = 0, con64 = 0;
        for (uint32_t i = 0; i < seq_count; i++) {
            tl64  += seqs[i].lit_len;
            con64 += (uint64_t)seqs[i].lit_len + seqs[i].match_len;
        }
        uint32_t trail_p = (uint32_t)((uint64_t)src_size - con64);
        uint32_t tlit_p  = (uint32_t)(tl64 + trail_p);

        uint8_t *tlit_raw = NULL;
        uint8_t *tll  = NULL;
        uint8_t *tml  = NULL;
        uint8_t *toff = NULL;

        if (tlit_p > 0) {
            tlit_raw = (uint8_t *)lzs_alloc(dst, tlit_p);
            if (!tlit_raw) { lzs_free(dst, seqs); return TDC_E_NOMEM; }
            uint32_t sp = 0, lp = 0;
            for (uint32_t i = 0; i < seq_count; i++) {
                uint32_t ll = seqs[i].lit_len;
                if (ll > 0) { memcpy(tlit_raw + lp, src + sp, ll); lp += ll; }
                sp += ll + seqs[i].match_len;
            }
            if (trail_p > 0) memcpy(tlit_raw + lp, src + sp, trail_p);
        }

        tll  = (uint8_t *)lzs_alloc(dst, seq_count);
        tml  = (uint8_t *)lzs_alloc(dst, seq_count);
        toff = (uint8_t *)lzs_alloc(dst, seq_count);
        if (!tll || !tml || !toff) {
            lzs_free(dst, tlit_raw); lzs_free(dst, tll);
            lzs_free(dst, tml); lzs_free(dst, toff);
            lzs_free(dst, seqs);
            return TDC_E_NOMEM;
        }

        for (uint32_t i = 0; i < seq_count; i++) {
            uint8_t s; uint32_t ex, nb;
            lzs_uint_to_symbol(seqs[i].lit_len, &s, &ex, &nb);
            tll[i] = s;
            lzs_uint_to_symbol(seqs[i].match_len - LZ_MIN_MATCH, &s, &ex, &nb);
            tml[i] = s;
            lzs_offset_to_symbol(seqs[i].match_off, &s, &ex, &nb);
            toff[i] = s;
        }

        LzsStreamsPrices prices;
        lzs_compute_prices(tlit_raw, tlit_p, tll, tml, toff, seq_count, &prices);
        prices.offset_shift = off_shift;

        if (lzs_dump_enabled()) {
            uint32_t p_extra = 0;
            for (uint32_t i = 0; i < seq_count; i++) {
                uint8_t s; uint32_t ex, nb;
                lzs_uint_to_symbol(seqs[i].lit_len, &s, &ex, &nb); p_extra += nb;
                lzs_uint_to_symbol(seqs[i].match_len-LZ_MIN_MATCH, &s, &ex, &nb); p_extra += nb;
                lzs_offset_to_symbol(seqs[i].match_off, &s, &ex, &nb); p_extra += nb;
            }
            fprintf(stderr,
                    "[lzs-p%d] src=%zu n_seqs=%u  lit=%u  extra=%u bytes\n",
                    pass, src_size, seq_count, tlit_p, (p_extra + 7u) / 8u);
        }

        lzs_free(dst, tlit_raw);
        lzs_free(dst, tll);
        lzs_free(dst, tml);
        lzs_free(dst, toff);
        lzs_free(dst, seqs);
        seqs = NULL;
        seq_count = 0;

        st = tdc_lz_parse_optimal_streams_priced(
            src, (uint32_t)src_size, dst, &prices, &seqs, &seq_count);
        if (st != TDC_OK) return st;

        if (seq_count == 0) {
            return lzs_encode_passthrough(src, (uint32_t)src_size, dst);
        }
    }

    /* Detect and apply offset stride shift.
     * For typed data (e.g. f64, stride=8), all LZ offsets are multiples of
     * the element width. Right-shifting by the common factor removes
     * redundant low bits from the offset extras stream. */
    uint32_t offset_shift = lzs_detect_offset_shift(seqs, seq_count);
    if (offset_shift > 0) {
        for (uint32_t i = 0; i < seq_count; i++)
            seqs[i].match_off >>= offset_shift;
    }

    if (lzs_dump_enabled() && offset_shift > 0) {
        fprintf(stderr, "  [lzs-shift] offset_shift=%u (stride=%u)\n",
                offset_shift, 1u << offset_shift);
    }

    /* Repcode transform on the shifted offsets. */
    lzs_repcode_encode(seqs, seq_count);

    /* Compute stream sizes from the parse. */
    uint64_t total_lit64 = 0;
    uint64_t consumed64 = 0;
    for (uint32_t i = 0; i < seq_count; i++) {
        total_lit64 += seqs[i].lit_len;
        consumed64  += (uint64_t)seqs[i].lit_len + seqs[i].match_len;
    }
    uint32_t trailing = (uint32_t)((uint64_t)src_size - consumed64);
    uint32_t total_lit = (uint32_t)(total_lit64 + trailing);

    tdc_buffer scratch = {0};
    scratch.realloc_fn = dst->realloc_fn;
    scratch.user       = dst->user;

    uint8_t  *lit_raw    = NULL;
    uint8_t  *ll_sym    = NULL;   /* lit_len symbol stream (1 byte/seq) */
    uint8_t  *ml_sym    = NULL;   /* match_len symbol stream (1 byte/seq) */
    uint8_t  *off_sym   = NULL;   /* offset symbol stream (1 byte/seq) */
    uint8_t  *extra_raw = NULL;   /* combined packed extra bits */
    uint32_t  extra_bytes = 0;

    uint8_t *enc_lit = NULL, *enc_ll = NULL, *enc_ml = NULL, *enc_off = NULL;
    uint32_t sz_lit = 0, sz_ll = 0, sz_ml = 0, sz_off = 0;
    tdc_entropy_id id_lit = TDC_ENTROPY_NONE;
    tdc_entropy_id id_ll  = TDC_ENTROPY_NONE;
    tdc_entropy_id id_ml  = TDC_ENTROPY_NONE;
    tdc_entropy_id id_off = TDC_ENTROPY_NONE;

    /* Build literal stream. */
    if (total_lit > 0) {
        lit_raw = (uint8_t *)lzs_alloc(dst, total_lit);
        if (!lit_raw) { st = TDC_E_NOMEM; goto cleanup; }

        uint32_t src_pos = 0;
        uint32_t lit_pos = 0;
        for (uint32_t i = 0; i < seq_count; i++) {
            uint32_t ll = seqs[i].lit_len;
            if (ll > 0) {
                memcpy(lit_raw + lit_pos, src + src_pos, ll);
                lit_pos += ll;
            }
            src_pos += ll + seqs[i].match_len;
        }
        if (trailing > 0) memcpy(lit_raw + lit_pos, src + src_pos, trailing);
    }

    /* Build log2-prefix symbol streams + combined packed extra bits for
     * all three integer fields (lit_len, match_len, offset). Each value
     * is split into a small-alphabet symbol (entropy-coded) and extra
     * bits (packed raw). All extra bits go into a single combined
     * bitstream, interleaved per-sequence: ll_extra, ml_extra, off_extra. */
    {
        ll_sym  = (uint8_t *)lzs_alloc(dst, seq_count);
        ml_sym  = (uint8_t *)lzs_alloc(dst, seq_count);
        off_sym = (uint8_t *)lzs_alloc(dst, seq_count);
        if (!ll_sym || !ml_sym || !off_sym) {
            st = TDC_E_NOMEM; goto cleanup;
        }

        /* Compute symbols and total extra bits. */
        uint32_t total_extra_bits = 0;
        uint32_t ll_extra_bits = 0, ml_extra_bits = 0, off_extra_bits = 0;
        uint32_t n_repcodes = 0;
        for (uint32_t i = 0; i < seq_count; i++) {
            uint8_t s; uint32_t ex, nb;

            lzs_uint_to_symbol(seqs[i].lit_len, &s, &ex, &nb);
            ll_sym[i] = s;
            total_extra_bits += nb;
            ll_extra_bits += nb;

            lzs_uint_to_symbol(seqs[i].match_len - LZ_MIN_MATCH, &s, &ex, &nb);
            ml_sym[i] = s;
            total_extra_bits += nb;
            ml_extra_bits += nb;

            lzs_offset_to_symbol(seqs[i].match_off, &s, &ex, &nb);
            off_sym[i] = s;
            total_extra_bits += nb;
            off_extra_bits += nb;
            if (seqs[i].match_off <= 3u) n_repcodes++;
        }
        if (lzs_dump_enabled()) {
            uint32_t off_hist[24] = {0};
            for (uint32_t i = 0; i < seq_count; i++) {
                if (off_sym[i] < 24) off_hist[off_sym[i]]++;
            }
            fprintf(stderr,
                    "  [lzs-extra] ll=%u ml=%u off=%u total=%u bits "
                    "(%u bytes) repcodes=%u/%u (%.0f%%)\n",
                    ll_extra_bits, ml_extra_bits, off_extra_bits,
                    total_extra_bits, (total_extra_bits + 7u) / 8u,
                    n_repcodes, seq_count,
                    seq_count > 0 ? 100.0 * n_repcodes / seq_count : 0.0);
            fprintf(stderr, "  [lzs-off-hist]");
            for (int j = 0; j < 24; j++) {
                if (off_hist[j] > 0) fprintf(stderr, " s%d=%u", j, off_hist[j]);
            }
            fprintf(stderr, "\n");
        }

        /* Pack extra bits into combined buffer. */
        extra_bytes = (total_extra_bits + 7u) / 8u;
        if (extra_bytes > 0) {
            extra_raw = (uint8_t *)lzs_alloc(dst, extra_bytes);
            if (!extra_raw) { st = TDC_E_NOMEM; goto cleanup; }
            memset(extra_raw, 0, extra_bytes);

            uint32_t bit_pos = 0;
            for (uint32_t i = 0; i < seq_count; i++) {
                uint8_t s; uint32_t ex, nb;

                lzs_uint_to_symbol(seqs[i].lit_len, &s, &ex, &nb);
                if (nb > 0) lzs_bits_write(extra_raw, &bit_pos, ex, nb);

                lzs_uint_to_symbol(seqs[i].match_len - LZ_MIN_MATCH, &s, &ex, &nb);
                if (nb > 0) lzs_bits_write(extra_raw, &bit_pos, ex, nb);

                lzs_offset_to_symbol(seqs[i].match_off, &s, &ex, &nb);
                if (nb > 0) lzs_bits_write(extra_raw, &bit_pos, ex, nb);
            }
        }
    }

    /* Encode each symbol stream (Huffman/FSE best-of). */
    st = lzs_encode_stream(lit_raw, total_lit, &scratch, dst,
                            &enc_lit, &sz_lit, &id_lit);
    if (st != TDC_OK) goto cleanup;

    st = lzs_encode_stream(ll_sym, seq_count, &scratch, dst,
                            &enc_ll, &sz_ll, &id_ll);
    if (st != TDC_OK) goto cleanup;
    st = lzs_encode_stream(ml_sym, seq_count, &scratch, dst,
                            &enc_ml, &sz_ml, &id_ml);
    if (st != TDC_OK) goto cleanup;
    st = lzs_encode_stream(off_sym, seq_count, &scratch, dst,
                            &enc_off, &sz_off, &id_off);
    if (st != TDC_OK) goto cleanup;

    if (lzs_dump_enabled()) {
        static const char *id_name[] = {
            [TDC_ENTROPY_NONE]    = "NONE",
            [TDC_ENTROPY_HUFFMAN] = "HUF",
            [TDC_ENTROPY_FSE]     = "FSE",
        };
        uint32_t total_enc = (uint32_t)LZS_HEADER_SIZE_V2
                           + sz_lit + sz_ll + sz_ml + sz_off + extra_bytes;
        fprintf(stderr,
                "[lzs-p2] src=%zu n_seqs=%u  "
                "lit=%u(%u/%s) ll=%u/%s ml=%u/%s off=%u/%s extra=%u  "
                "total=%u (%.2fx)\n",
                src_size, seq_count,
                total_lit, sz_lit, id_name[id_lit],
                sz_ll, id_name[id_ll],
                sz_ml, id_name[id_ml],
                sz_off, id_name[id_off],
                extra_bytes, total_enc,
                total_enc > 0 ? (double)src_size / total_enc : 0.0);
    }

    /* Whole-block fallback: if we didn't beat passthrough, write one. */
    size_t final_size = (size_t)LZS_HEADER_SIZE_V2
                      + sz_lit + sz_ll + sz_ml + sz_off + extra_bytes;
    if (final_size >= src_size) {
        st = lzs_encode_passthrough(src, (uint32_t)src_size, dst);
        goto cleanup;
    }

    /* Write final output (v2 header, 44 bytes). */
    st = tdc_buf_reserve(dst, final_size);
    if (st != TDC_OK) goto cleanup;

    {
        uint8_t *p = dst->data;
        uint32_t us = (uint32_t)src_size;
        p[0] = (uint8_t)LZS_FORMAT_VERSION;
        p[1] = (uint8_t)(LZS_FLAG_REPCODES
                         | (offset_shift > 0 ? LZS_FLAG_OFFSET_SHIFT : 0));
        p[2] = (uint8_t)offset_shift;
        p[3] = 0;
        memcpy(p +  4, &seq_count,   4);
        memcpy(p +  8, &total_lit,   4);
        memcpy(p + 12, &trailing,    4);
        memcpy(p + 16, &us,          4);
        p[20] = (uint8_t)id_lit;
        p[21] = (uint8_t)id_ll;
        p[22] = (uint8_t)id_ml;
        p[23] = (uint8_t)id_off;
        memcpy(p + 24, &sz_lit,       4);
        memcpy(p + 28, &sz_ll,        4);
        memcpy(p + 32, &sz_ml,        4);
        memcpy(p + 36, &sz_off,       4);
        memcpy(p + 40, &extra_bytes,  4);

        uint8_t *body = p + LZS_HEADER_SIZE_V2;
        if (sz_lit)      { memcpy(body, enc_lit,   sz_lit);      body += sz_lit;      }
        if (sz_ll)       { memcpy(body, enc_ll,    sz_ll);       body += sz_ll;       }
        if (sz_ml)       { memcpy(body, enc_ml,    sz_ml);       body += sz_ml;       }
        if (sz_off)      { memcpy(body, enc_off,   sz_off);      body += sz_off;      }
        if (extra_bytes) { memcpy(body, extra_raw, extra_bytes);                      }
    }
    dst->size = final_size;

cleanup:
    if (scratch.data) scratch.realloc_fn(scratch.user, scratch.data, 0);
    lzs_free(dst, enc_lit);
    lzs_free(dst, enc_ll);
    lzs_free(dst, enc_ml);
    lzs_free(dst, enc_off);
    lzs_free(dst, lit_raw);
    lzs_free(dst, ll_sym);
    lzs_free(dst, ml_sym);
    lzs_free(dst, off_sym);
    lzs_free(dst, extra_raw);
    lzs_free(dst, seqs);
    return st;
}

/* ----- Decode ------------------------------------------------------------- */
/*
 * Scratch strategy: a single libc malloc for all decode-side buffers.
 * Using libc malloc keeps the entropy vtable's decode signature unchanged.
 *
 * Supports format v1 (byte-shuffled u32 offsets, header 40) and v2
 * (log2-prefix symbols + packed extra bits, header 44).
 */

/* Shared output reconstruction — used by both v1 and v2 decoders. */
static tdc_status lzs_reconstruct(uint8_t *dst, size_t dst_size,
                                    const uint8_t *lit_raw, uint32_t total_lit,
                                    uint32_t trailing,
                                    const uint32_t *lit_lens,
                                    const uint32_t *match_lens,
                                    const uint32_t *match_offs,
                                    uint32_t n_seqs) {
    size_t dp = 0, lp = 0;
    for (uint32_t i = 0; i < n_seqs; i++) {
        uint32_t ll = lit_lens[i];
        uint32_t ml = match_lens[i];
        uint32_t mo = match_offs[i];

        if ((uint64_t)lp + ll > total_lit) return TDC_E_CORRUPT;
        if ((uint64_t)dp + ll > dst_size)  return TDC_E_CORRUPT;
        if (ll > 0) {
            memcpy(dst + dp, lit_raw + lp, ll);
            dp += ll;
            lp += ll;
        }

        if (ml < LZ_MIN_MATCH)            return TDC_E_CORRUPT;
        if (mo == 0 || mo > dp)            return TDC_E_CORRUPT;
        if ((uint64_t)dp + ml > dst_size)  return TDC_E_CORRUPT;

        size_t from = dp - mo;
        for (uint32_t k = 0; k < ml; k++)
            dst[dp + k] = dst[from + k];
        dp += ml;
    }

    if ((uint64_t)lp + trailing != total_lit) return TDC_E_CORRUPT;
    if (dp + trailing != dst_size)            return TDC_E_CORRUPT;
    if (trailing > 0) memcpy(dst + dp, lit_raw + lp, trailing);
    return TDC_OK;
}

/* Shared symbol→value reconstruction for v2 and v3 decoders. Reads
 * symbols from ll_dec/ml_dec/off_dec (v2 separate, or de-interleaved
 * from v3 combined) and extra bits from extra_ptr. Fills lit_lens,
 * match_lens, match_offs. */
static tdc_status lzs_reconstruct_symbols(const uint8_t *ll_dec,
                                            const uint8_t *ml_dec,
                                            const uint8_t *off_dec,
                                            const uint8_t *extra_ptr,
                                            uint32_t sz_extra,
                                            uint32_t n_seqs,
                                            uint32_t *lit_lens,
                                            uint32_t *match_lens,
                                            uint32_t *match_offs) {
    uint32_t bit_pos = 0;
    for (uint32_t i = 0; i < n_seqs; i++) {
        uint8_t s; uint32_t ex, nb;

        /* Reconstruct lit_len. */
        s = ll_dec[i];
        if (s > LZS_MAX_LL_SYMBOL) return TDC_E_CORRUPT;
        ex = 0;
        if (s >= 2u) {
            nb = (uint32_t)s - 1u;
            if ((bit_pos + nb + 7u) / 8u > sz_extra) return TDC_E_CORRUPT;
            ex = lzs_bits_read(extra_ptr, &bit_pos, nb);
        }
        lit_lens[i] = lzs_symbol_to_uint(s, ex);

        /* Reconstruct match_len. */
        s = ml_dec[i];
        if (s > LZS_MAX_ML_SYMBOL) return TDC_E_CORRUPT;
        ex = 0;
        if (s >= 2u) {
            nb = (uint32_t)s - 1u;
            if ((bit_pos + nb + 7u) / 8u > sz_extra) return TDC_E_CORRUPT;
            ex = lzs_bits_read(extra_ptr, &bit_pos, nb);
        }
        match_lens[i] = lzs_symbol_to_uint(s, ex) + LZ_MIN_MATCH;

        /* Reconstruct offset code. */
        s = off_dec[i];
        if (s > LZS_MAX_OFFSET_SYMBOL) return TDC_E_CORRUPT;
        ex = 0;
        if (s >= 3u) {
            nb = (uint32_t)s - 3u;
            if (nb > 0) {
                if ((bit_pos + nb + 7u) / 8u > sz_extra) return TDC_E_CORRUPT;
                ex = lzs_bits_read(extra_ptr, &bit_pos, nb);
            }
        }
        match_offs[i] = lzs_symbol_to_code(s, ex);
    }
    return TDC_OK;
}

static tdc_status lzs_decode(const uint8_t *src, size_t src_size,
                              uint8_t       *dst, size_t dst_size) {
    if (src_size < LZS_HEADER_SIZE_V1) return TDC_E_CORRUPT;
    if (dst_size > 0 && !dst)           return TDC_E_INVAL;

    uint8_t version = src[0];
    uint8_t flags   = src[1];
    if (version != 1u && version != 2u) return TDC_E_CORRUPT;
    if (flags & ~(uint8_t)(LZS_FLAG_REPCODES | LZS_FLAG_OFFSET_SHIFT))
        return TDC_E_CORRUPT;
    int has_repcodes = (flags & LZS_FLAG_REPCODES) != 0;
    uint32_t offset_shift = (flags & LZS_FLAG_OFFSET_SHIFT) ? src[2] : 0;
    if (offset_shift > 7u) return TDC_E_CORRUPT;

    uint32_t hdr_size = (version == 1u) ? LZS_HEADER_SIZE_V1
                                        : LZS_HEADER_SIZE_V2;
    if (src_size < hdr_size) return TDC_E_CORRUPT;

    uint32_t n_seqs, total_lit, trailing, uncompressed;
    memcpy(&n_seqs,       src +  4, 4);
    memcpy(&total_lit,    src +  8, 4);
    memcpy(&trailing,     src + 12, 4);
    memcpy(&uncompressed, src + 16, 4);
    if ((size_t)uncompressed != dst_size) return TDC_E_CORRUPT;
    if (trailing > total_lit)             return TDC_E_CORRUPT;

    tdc_entropy_id id_lit = (tdc_entropy_id)src[20];
    tdc_entropy_id id_ll  = (tdc_entropy_id)src[21];
    tdc_entropy_id id_ml  = (tdc_entropy_id)src[22];
    tdc_entropy_id id_mo  = (tdc_entropy_id)src[23];

    uint32_t sz_lit, sz_ll, sz_ml, sz_mo;
    memcpy(&sz_lit, src + 24, 4);
    memcpy(&sz_ll,  src + 28, 4);
    memcpy(&sz_ml,  src + 32, 4);
    memcpy(&sz_mo,  src + 36, 4);

    uint32_t sz_extra = 0;
    if (version == 2u) memcpy(&sz_extra, src + 40, 4);

    /* Bounds check the body. */
    {
        uint64_t need = (uint64_t)hdr_size
                      + sz_lit + sz_ll + sz_ml + sz_mo + sz_extra;
        if (need > src_size) return TDC_E_CORRUPT;
    }

    /* Empty record. */
    if (dst_size == 0) {
        if (n_seqs != 0 || total_lit != 0) return TDC_E_CORRUPT;
        return TDC_OK;
    }

    size_t stream_bytes = (size_t)n_seqs * 4u;

    /* v2: three symbol streams (1 byte each) + u32 output arrays.
     * v1: three byte-shuffled streams (4 bytes each) + u32 output arrays. */
    size_t v2_sym_size  = 3u * (size_t)n_seqs;
    size_t v1_raw_size  = 3u * stream_bytes;
    size_t per_version  = (version == 2u) ? v2_sym_size : v1_raw_size;
    size_t scratch_bytes = (size_t)total_lit
                         + per_version
                         + 3u * (size_t)n_seqs * sizeof(uint32_t);
    uint8_t *scratch = scratch_bytes ? (uint8_t *)malloc(scratch_bytes) : NULL;
    if (scratch_bytes > 0 && !scratch) return TDC_E_NOMEM;

    uint8_t  *lit_raw    = scratch;
    uint32_t *lit_lens, *match_lens, *match_offs;

    uint8_t *region2 = lit_raw + total_lit;

    if (version == 1u) {
        lit_lens   = (uint32_t *)(region2 + 3u * stream_bytes);
    } else {
        lit_lens   = (uint32_t *)(region2 + 3u * (size_t)n_seqs);
    }
    match_lens = lit_lens   + n_seqs;
    match_offs = match_lens + n_seqs;

    tdc_status st = TDC_OK;
    const uint8_t *p = src + hdr_size;

    /* Decode literal stream. */
    if (total_lit > 0) {
        const tdc_entropy_vt *sub = lzs_sub_vt(id_lit);
        if (!sub) { st = TDC_E_CORRUPT; goto done; }
        st = sub->decode(p, sz_lit, lit_raw, total_lit);
        if (st != TDC_OK) goto done;
    }
    p += sz_lit;

    if (n_seqs > 0) {
        if (version == 1u) {
            /* v1: three byte-shuffled u32 streams. */
            uint8_t *ll_raw = region2;
            uint8_t *ml_raw = ll_raw + stream_bytes;
            uint8_t *mo_raw = ml_raw + stream_bytes;

            const tdc_entropy_vt *sub_ll = lzs_sub_vt(id_ll);
            const tdc_entropy_vt *sub_ml = lzs_sub_vt(id_ml);
            const tdc_entropy_vt *sub_mo = lzs_sub_vt(id_mo);
            if (!sub_ll || !sub_ml || !sub_mo) { st = TDC_E_CORRUPT; goto done; }

            st = sub_ll->decode(p, sz_ll, ll_raw, stream_bytes);
            if (st != TDC_OK) goto done;
            p += sz_ll;
            st = sub_ml->decode(p, sz_ml, ml_raw, stream_bytes);
            if (st != TDC_OK) goto done;
            p += sz_ml;
            st = sub_mo->decode(p, sz_mo, mo_raw, stream_bytes);
            if (st != TDC_OK) goto done;

            lzs_unshuffle_u32(ll_raw, n_seqs, lit_lens);
            lzs_unshuffle_u32(ml_raw, n_seqs, match_lens);
            lzs_unshuffle_u32(mo_raw, n_seqs, match_offs);
        } else {
            /* v2: three symbol streams + combined extra bits. */
            uint8_t *ll_dec  = region2;
            uint8_t *ml_dec  = ll_dec + n_seqs;
            uint8_t *off_dec = ml_dec + n_seqs;

            const tdc_entropy_vt *sub_ll  = lzs_sub_vt(id_ll);
            const tdc_entropy_vt *sub_ml  = lzs_sub_vt(id_ml);
            const tdc_entropy_vt *sub_off = lzs_sub_vt(id_mo);
            if (!sub_ll || !sub_ml || !sub_off) { st = TDC_E_CORRUPT; goto done; }

            st = sub_ll->decode(p, sz_ll, ll_dec, n_seqs);
            if (st != TDC_OK) goto done;
            p += sz_ll;
            st = sub_ml->decode(p, sz_ml, ml_dec, n_seqs);
            if (st != TDC_OK) goto done;
            p += sz_ml;
            st = sub_off->decode(p, sz_mo, off_dec, n_seqs);
            if (st != TDC_OK) goto done;
            p += sz_mo;

            st = lzs_reconstruct_symbols(ll_dec, ml_dec, off_dec,
                                          p, sz_extra, n_seqs,
                                          lit_lens, match_lens, match_offs);
            if (st != TDC_OK) goto done;
        }

        if (has_repcodes) {
            st = lzs_repcode_decode(match_offs, n_seqs);
            if (st != TDC_OK) goto done;
        }

        /* Undo offset stride shift. */
        if (offset_shift > 0) {
            for (uint32_t i = 0; i < n_seqs; i++)
                match_offs[i] <<= offset_shift;
        }
    }

    st = lzs_reconstruct(dst, dst_size, lit_raw, total_lit, trailing,
                           lit_lens, match_lens, match_offs, n_seqs);

done:
    free(scratch);
    return st;
}

/* ----- Vtable ------------------------------------------------------------- */

const tdc_entropy_vt tdc_entropy_lz_streams_vt = {
    .id           = TDC_ENTROPY_LZ_STREAMS,
    .name         = "lz_streams",
    .encode_bound = lzs_encode_bound,
    .encode       = lzs_encode,
    .decode       = lzs_decode,
};
