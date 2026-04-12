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
 *   match_off_stream — n_seqs × u32 match_off, byte-shuffled
 *
 * Each stream picks between NONE / HUFFMAN / FSE via order-0 Shannon
 * entropy (same heuristic as LANE). A stream that would expand under its
 * chosen coder falls back to NONE. A whole-block fallback also exists:
 * if the full encoded size meets or exceeds src_size, the header is
 * written with n_seqs=0 and the lit stream carries src as passthrough
 * NONE, so every input yields a valid LZ_STREAMS record regardless of
 * parse quality.
 *
 * Rationale: zstd's compression advantage over single-stream LZ comes
 * primarily from (a) separating the integer streams and entropy-coding
 * each with its own model, and (b) repcode offsets (last-N match offsets
 * reused for "same offset as before"). This file addresses (a); (b) is
 * a planned follow-up that will reuse the parser and the stream layout.
 *
 * On-disk header (40 bytes, fixed):
 *
 *   offset  size  field
 *     0      1    format_version         (LZS_FORMAT_VERSION, currently 1)
 *     1      1    flags                  (bit 0: LZS_FLAG_REPCODES)
 *     2      2    reserved               (zero)
 *     4      4    n_seqs                 (0 => passthrough fallback)
 *     8      4    total_lit_size         (bytes in literal_stream)
 *    12      4    trailing_lit_len       (literals after last match)
 *    16      4    uncompressed_size      (== dst_size on decode)
 *    20      4    entropy_id[4]          (lit, lit_len, match_len, match_off)
 *    24      4    stream_size_lit
 *    28      4    stream_size_lit_len
 *    32      4    stream_size_match_len
 *    36      4    stream_size_match_off
 *
 * followed by the four encoded streams in that order.
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

#define LZS_HEADER_SIZE      40u
#define LZS_FORMAT_VERSION   1u
#define LZS_FLAG_REPCODES    0x01u

/* Initial repcode values (match zstd's rep-init). */
#define LZS_REP_INIT_1       1u
#define LZS_REP_INIT_2       4u
#define LZS_REP_INIT_3       8u

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
    /* Worst case: literal stream = src_size (every byte is a literal),
     * and three integer streams each at most 4 * (src_size/4) = src_size
     * (one sequence per byte is impossible since LZ_MIN_MATCH=3, but
     * over-estimating is cheap). Each sub-coder adds at most ~1 KiB of
     * headers. Plus the 36-byte LZ_STREAMS header. */
    return LZS_HEADER_SIZE + src_size + 3u * src_size + 4u * 1024u;
}

/* ----- Passthrough fallback --------------------------------------------- */
/*
 * Writes an LZ_STREAMS record that carries src byte-for-byte in the
 * literal stream with NONE coding. Used when the full encoded form
 * (parse + four entropy-coded streams) would match or exceed src_size.
 */
static tdc_status lzs_encode_passthrough(const uint8_t *src, uint32_t src_size,
                                          tdc_buffer *dst) {
    size_t need = (size_t)LZS_HEADER_SIZE + src_size;
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
    p[23] = (uint8_t)TDC_ENTROPY_NONE; /* id_mo  */
    memcpy(p + 24, &src_size, 4);      /* sz_lit */
    memset(p + 28, 0, 12);             /* sz_ll, sz_ml, sz_mo = 0 */

    if (src_size > 0) memcpy(p + LZS_HEADER_SIZE, src, src_size);
    dst->size = need;
    return TDC_OK;
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

    /* Parse with the STREAMS-aware cost model. Unlike tdc_lz_parse_optimal
     * (which costs literals and matches as the single-stream serializer
     * would charge them), this variant charges literals at ~6 bits and
     * match headers at ~31 bits — the entropy-coded costs measured from
     * the multi-stream pipeline. That eliminates the systematic short-
     * match bias the single-stream DP induces on STREAMS output. */
    LZSeq *seqs = NULL;
    uint32_t seq_count = 0;
    tdc_status st = tdc_lz_parse_optimal_streams(src, (uint32_t)src_size, dst,
                                                  &seqs, &seq_count);
    if (st != TDC_OK) return st;

    if (seq_count == 0) {
        /* No matches found (src too short or trivial). Passthrough. */
        return lzs_encode_passthrough(src, (uint32_t)src_size, dst);
    }

    /* Repcode transform: rewrite each .match_off as an offset_code in
     * [1, LZ_MAX_OFFSET + 3]. Dominant stride offsets collapse to the
     * symbol "1" so Huffman/FSE can entropy-code the stream at a fraction
     * of its raw bits. The transform runs in place on the LZSeq array. */
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

    uint8_t  *lit_raw = NULL;
    uint8_t  *ll_raw  = NULL;
    uint8_t  *ml_raw  = NULL;
    uint8_t  *mo_raw  = NULL;
    uint32_t *tmp_u32 = NULL;

    uint8_t *enc_lit = NULL, *enc_ll = NULL, *enc_ml = NULL, *enc_mo = NULL;
    uint32_t sz_lit = 0, sz_ll = 0, sz_ml = 0, sz_mo = 0;
    tdc_entropy_id id_lit = TDC_ENTROPY_NONE;
    tdc_entropy_id id_ll  = TDC_ENTROPY_NONE;
    tdc_entropy_id id_ml  = TDC_ENTROPY_NONE;
    tdc_entropy_id id_mo  = TDC_ENTROPY_NONE;

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

    /* Build and byte-shuffle the three integer streams. */
    {
        size_t stream_bytes = (size_t)seq_count * 4u;
        ll_raw  = (uint8_t  *)lzs_alloc(dst, stream_bytes);
        ml_raw  = (uint8_t  *)lzs_alloc(dst, stream_bytes);
        mo_raw  = (uint8_t  *)lzs_alloc(dst, stream_bytes);
        tmp_u32 = (uint32_t *)lzs_alloc(dst, stream_bytes);
        if (!ll_raw || !ml_raw || !mo_raw || !tmp_u32) {
            st = TDC_E_NOMEM; goto cleanup;
        }

        for (uint32_t i = 0; i < seq_count; i++) tmp_u32[i] = seqs[i].lit_len;
        lzs_shuffle_u32(tmp_u32, seq_count, ll_raw);
        for (uint32_t i = 0; i < seq_count; i++) tmp_u32[i] = seqs[i].match_len;
        lzs_shuffle_u32(tmp_u32, seq_count, ml_raw);
        for (uint32_t i = 0; i < seq_count; i++) tmp_u32[i] = seqs[i].match_off;
        lzs_shuffle_u32(tmp_u32, seq_count, mo_raw);
    }

    /* Encode each stream. The encode helper tries all candidate coders
     * (HUFFMAN, FSE) and returns the smallest as a freshly allocated
     * per-stream buffer, falling back to NONE passthrough if nothing
     * beats it. That gives each stream a dedicated lifetime independent
     * of the others, which lets us keep all four in memory, sum the
     * sizes, decide between the normal layout and the passthrough
     * fallback, and only then copy into the final dst. */
    st = lzs_encode_stream(lit_raw, total_lit, &scratch, dst,
                            &enc_lit, &sz_lit, &id_lit);
    if (st != TDC_OK) goto cleanup;

    {
        size_t stream_bytes = (size_t)seq_count * 4u;
        st = lzs_encode_stream(ll_raw, stream_bytes, &scratch, dst,
                                &enc_ll, &sz_ll, &id_ll);
        if (st != TDC_OK) goto cleanup;
        st = lzs_encode_stream(ml_raw, stream_bytes, &scratch, dst,
                                &enc_ml, &sz_ml, &id_ml);
        if (st != TDC_OK) goto cleanup;
        st = lzs_encode_stream(mo_raw, stream_bytes, &scratch, dst,
                                &enc_mo, &sz_mo, &id_mo);
        if (st != TDC_OK) goto cleanup;
    }

    if (lzs_dump_enabled()) {
        size_t raw_lit = total_lit;
        size_t raw_u32 = (size_t)seq_count * 4u;
        fprintf(stderr,
                "[lzs] src=%zu n_seqs=%u  "
                "lit raw=%zu coder=%d enc=%u (%.1f%%)  "
                "ll  raw=%zu coder=%d enc=%u (%.1f%%)  "
                "ml  raw=%zu coder=%d enc=%u (%.1f%%)  "
                "mo  raw=%zu coder=%d enc=%u (%.1f%%)\n",
                src_size, seq_count,
                raw_lit, (int)id_lit, sz_lit,
                raw_lit ? 100.0 * sz_lit / raw_lit : 0.0,
                raw_u32, (int)id_ll, sz_ll,
                raw_u32 ? 100.0 * sz_ll / raw_u32 : 0.0,
                raw_u32, (int)id_ml, sz_ml,
                raw_u32 ? 100.0 * sz_ml / raw_u32 : 0.0,
                raw_u32, (int)id_mo, sz_mo,
                raw_u32 ? 100.0 * sz_mo / raw_u32 : 0.0);
    }

    /* Whole-block fallback: if we didn't beat passthrough, write one. */
    size_t final_size = (size_t)LZS_HEADER_SIZE
                      + sz_lit + sz_ll + sz_ml + sz_mo;
    if (final_size >= src_size) {
        st = lzs_encode_passthrough(src, (uint32_t)src_size, dst);
        goto cleanup;
    }

    /* Write final output. */
    st = tdc_buf_reserve(dst, final_size);
    if (st != TDC_OK) goto cleanup;

    {
        uint8_t *p = dst->data;
        uint32_t us = (uint32_t)src_size;
        p[0] = (uint8_t)LZS_FORMAT_VERSION;
        p[1] = (uint8_t)LZS_FLAG_REPCODES;
        p[2] = 0;
        p[3] = 0;
        memcpy(p +  4, &seq_count, 4);
        memcpy(p +  8, &total_lit, 4);
        memcpy(p + 12, &trailing,  4);
        memcpy(p + 16, &us,        4);
        p[20] = (uint8_t)id_lit;
        p[21] = (uint8_t)id_ll;
        p[22] = (uint8_t)id_ml;
        p[23] = (uint8_t)id_mo;
        memcpy(p + 24, &sz_lit, 4);
        memcpy(p + 28, &sz_ll,  4);
        memcpy(p + 32, &sz_ml,  4);
        memcpy(p + 36, &sz_mo,  4);

        uint8_t *body = p + LZS_HEADER_SIZE;
        if (sz_lit) { memcpy(body, enc_lit, sz_lit); body += sz_lit; }
        if (sz_ll)  { memcpy(body, enc_ll,  sz_ll);  body += sz_ll;  }
        if (sz_ml)  { memcpy(body, enc_ml,  sz_ml);  body += sz_ml;  }
        if (sz_mo)  { memcpy(body, enc_mo,  sz_mo);                  }
    }
    dst->size = final_size;

cleanup:
    if (scratch.data) scratch.realloc_fn(scratch.user, scratch.data, 0);
    lzs_free(dst, enc_lit);
    lzs_free(dst, enc_ll);
    lzs_free(dst, enc_ml);
    lzs_free(dst, enc_mo);
    lzs_free(dst, lit_raw);
    lzs_free(dst, ll_raw);
    lzs_free(dst, ml_raw);
    lzs_free(dst, mo_raw);
    lzs_free(dst, tmp_u32);
    lzs_free(dst, seqs);
    return st;
}

/* ----- Decode ------------------------------------------------------------- */
/*
 * Scratch strategy: a single libc malloc for all decode-side buffers
 * (raw byte-lane buffers + unshuffled u32 arrays). Using libc malloc
 * keeps the entropy vtable's decode signature unchanged — decode() has
 * no scratch buffer parameter to thread a tdc_buffer through, and
 * passing one would break the shared vtable shape. The one allocation
 * is freed on every exit path.
 */
static tdc_status lzs_decode(const uint8_t *src, size_t src_size,
                              uint8_t       *dst, size_t dst_size) {
    if (src_size < LZS_HEADER_SIZE) return TDC_E_CORRUPT;
    if (dst_size > 0 && !dst)        return TDC_E_INVAL;

    uint8_t  version = src[0];
    uint8_t  flags   = src[1];
    if (version != LZS_FORMAT_VERSION) return TDC_E_CORRUPT;
    /* Unknown flag bits are rejected so a future bit cannot be silently
     * ignored by a stale decoder. */
    if (flags & ~(uint8_t)LZS_FLAG_REPCODES) return TDC_E_CORRUPT;
    int has_repcodes = (flags & LZS_FLAG_REPCODES) != 0;

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

    /* Bounds check the body. */
    {
        uint64_t need = (uint64_t)LZS_HEADER_SIZE
                      + sz_lit + sz_ll + sz_ml + sz_mo;
        if (need > src_size) return TDC_E_CORRUPT;
    }

    /* Empty record. */
    if (dst_size == 0) {
        if (n_seqs != 0 || total_lit != 0) return TDC_E_CORRUPT;
        return TDC_OK;
    }

    /* Integer streams must each be exactly 4 * n_seqs bytes when decoded. */
    size_t stream_bytes = (size_t)n_seqs * 4u;

    /* Single scratch alloc for all decode-side buffers:
     *   [lit_raw      ] total_lit
     *   [ll_raw       ] stream_bytes
     *   [ml_raw       ] stream_bytes
     *   [mo_raw       ] stream_bytes
     *   [lit_lens     ] n_seqs * sizeof(uint32_t)
     *   [match_lens   ] n_seqs * sizeof(uint32_t)
     *   [match_offs   ] n_seqs * sizeof(uint32_t)
     */
    size_t scratch_bytes = (size_t)total_lit
                         + 3u * stream_bytes
                         + 3u * (size_t)n_seqs * sizeof(uint32_t);
    uint8_t *scratch = scratch_bytes ? (uint8_t *)malloc(scratch_bytes) : NULL;
    if (scratch_bytes > 0 && !scratch) return TDC_E_NOMEM;

    uint8_t  *lit_raw    = scratch;
    uint8_t  *ll_raw     = lit_raw + total_lit;
    uint8_t  *ml_raw     = ll_raw  + stream_bytes;
    uint8_t  *mo_raw     = ml_raw  + stream_bytes;
    uint32_t *lit_lens   = (uint32_t *)(mo_raw + stream_bytes);
    uint32_t *match_lens = lit_lens   + n_seqs;
    uint32_t *match_offs = match_lens + n_seqs;

    tdc_status st = TDC_OK;
    const uint8_t *p = src + LZS_HEADER_SIZE;

    /* Decode literal stream. */
    if (total_lit > 0) {
        const tdc_entropy_vt *sub = lzs_sub_vt(id_lit);
        if (!sub) { st = TDC_E_CORRUPT; goto done; }
        st = sub->decode(p, sz_lit, lit_raw, total_lit);
        if (st != TDC_OK) goto done;
    }
    p += sz_lit;

    /* Decode the three integer streams. */
    if (n_seqs > 0) {
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

        if (has_repcodes) {
            st = lzs_repcode_decode(match_offs, n_seqs);
            if (st != TDC_OK) goto done;
        }
    }

    /* Reconstruct the output. */
    {
        size_t dp = 0;
        size_t lp = 0;

        for (uint32_t i = 0; i < n_seqs; i++) {
            uint32_t ll = lit_lens[i];
            uint32_t ml = match_lens[i];
            uint32_t mo = match_offs[i];

            /* Literals. */
            if ((uint64_t)lp + ll > total_lit) { st = TDC_E_CORRUPT; goto done; }
            if ((uint64_t)dp + ll > dst_size)  { st = TDC_E_CORRUPT; goto done; }
            if (ll > 0) {
                memcpy(dst + dp, lit_raw + lp, ll);
                dp += ll;
                lp += ll;
            }

            /* Match. */
            if (ml < LZ_MIN_MATCH)            { st = TDC_E_CORRUPT; goto done; }
            if (mo == 0 || mo > dp)            { st = TDC_E_CORRUPT; goto done; }
            if ((uint64_t)dp + ml > dst_size)  { st = TDC_E_CORRUPT; goto done; }

            /* Byte-by-byte copy to handle RLE-style overlap (mo < ml). */
            {
                size_t from = dp - mo;
                for (uint32_t k = 0; k < ml; k++) {
                    dst[dp + k] = dst[from + k];
                }
                dp += ml;
            }
        }

        /* Trailing literals. */
        if ((uint64_t)lp + trailing != total_lit) { st = TDC_E_CORRUPT; goto done; }
        if (dp + trailing != dst_size)            { st = TDC_E_CORRUPT; goto done; }
        if (trailing > 0) memcpy(dst + dp, lit_raw + lp, trailing);
    }

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
