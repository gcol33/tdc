/*
 * src/entropy/fse.c
 *
 * TDC_ENTROPY_FSE — native rANS entropy coder.
 *
 * "FSE" in tdc names the family, not the specific table-coded variant
 * Yann Collet popularized. The coder shipped here is rANS (range
 * Asymmetric Numeral Systems), the range variant of Jarek Duda's ANS
 * family that tANS is the table variant of. Both target the same
 * compression ratio (within 0.1% of Shannon entropy on iid sources)
 * and are typically used in the same downstream slot. rANS was chosen
 * for v0 because:
 *
 *   - the encode/decode loops are short and easy to verify against the
 *     algorithm definition (one mod, one div, one shift per symbol),
 *   - there is no spread-table construction or per-symbol state-bit
 *     bookkeeping (which is the part of tANS that historically hides
 *     bugs),
 *   - the runtime ratio is identical to tANS for the residual streams
 *     tdc cares about.
 *
 * A future optimization can swap the inner loop for tANS without any
 * format change; the on-disk header (normalized counts + final state +
 * stream length) is the same shape both algorithms need.
 *
 * Algorithm:
 *
 *   Constants
 *     TABLE_LOG  = 12
 *     TABLE_SIZE = 1 << TABLE_LOG  = 4096   (precision of normalized PMF)
 *     L_MIN      = 1 << 16         = 65536  (renorm boundary; state stays
 *                                            in [L_MIN, L_MIN << 8))
 *
 *   Per-symbol state update (encode, run on input from end to start):
 *     while state >= ((L_MIN >> TABLE_LOG) << 8) * norm[s]:
 *         emit  state & 0xFF
 *         state >>= 8
 *     state = (state / norm[s]) << TABLE_LOG
 *           + (state % norm[s])
 *           + cum[s]
 *
 *   Per-symbol state update (decode, run on output from start to end):
 *     slot   = state & (TABLE_SIZE - 1)
 *     symbol = slot_to_sym[slot]
 *     state  = norm[s] * (state >> TABLE_LOG) + slot - cum[s]
 *     while state < L_MIN:
 *         state = (state << 8) | next_byte
 *
 *   The encoder pushes bytes "low byte first" while reducing state, so
 *   the on-disk byte stream IS reversed before being written. The
 *   decoder then reads the stream forward and pulls bytes high-byte-
 *   first via the standard rANS refill loop.
 *
 * On-disk payload (single self-contained block):
 *
 *     u32   src_size            (uncompressed bytes; cross-check vs caller)
 *     u32   payload_size        (number of bytes in the byte stream)
 *     u32   final_state         (encoder state at end of input)
 *     u16   norm[256]           (normalized PMF; sum exactly TABLE_SIZE)
 *     u8[]  payload             payload_size bytes, in decode (read) order
 *
 *   Header size: 4 + 4 + 4 + 512 = 524 bytes.
 *
 * Edge cases:
 *   - Empty input: src_size == 0, payload_size == 0, final_state == L_MIN,
 *     all norm[] zero. The decoder returns immediately.
 *   - Single distinct symbol s: norm[s] == TABLE_SIZE, all others zero.
 *     The encode update is the identity (state unchanged), no bytes are
 *     emitted, payload_size == 0, final_state == L_MIN. The decoder
 *     reads slot = state & (TABLE_SIZE - 1) → s every iteration.
 */

#include "tdc/entropy.h"
#include "entropy_internal.h"
#include "../core/buffer.h"
#include "../format/metadata_internal.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define FSE_NSYMS      256
#define FSE_TABLE_LOG  12
#define FSE_TABLE_SIZE (1u << FSE_TABLE_LOG)
#define FSE_TABLE_MASK (FSE_TABLE_SIZE - 1u)
#define FSE_L_MIN      (1u << 16)
#define FSE_HDR_SIZE   (4u + 4u + 4u + 2u * (uint32_t)FSE_NSYMS)

/* ----- Frequency normalization -------------------------------------------- */
/*
 * Quantize a histogram of total = sum(freq) into a normalized PMF whose
 * entries sum to exactly FSE_TABLE_SIZE. Used symbols receive at least
 * one count; unused symbols receive zero. The "Hamilton apportionment"
 * fix-up at the end never reduces a count below 1 (so the inverse
 * mapping stays well-defined for every symbol that ever appears in the
 * input).
 *
 * Edge case: total == 0 returns an all-zero PMF (caller treats this as
 * an empty input and short-circuits).
 */
static void fse_normalize(const uint32_t freq[FSE_NSYMS], uint32_t total,
                          uint16_t norm[FSE_NSYMS]) {
    for (int i = 0; i < FSE_NSYMS; ++i) norm[i] = 0;
    if (total == 0u) return;

    int distributed = 0;
    for (int i = 0; i < FSE_NSYMS; ++i) {
        if (freq[i] == 0u) continue;
        uint64_t prop = ((uint64_t)freq[i] * (uint64_t)FSE_TABLE_SIZE) / (uint64_t)total;
        if (prop == 0u) prop = 1u;
        if (prop > (uint64_t)FSE_TABLE_SIZE) prop = FSE_TABLE_SIZE;
        norm[i] = (uint16_t)prop;
        distributed += (int)prop;
    }

    int diff = (int)FSE_TABLE_SIZE - distributed;
    if (diff > 0) {
        /* Hand the slack to the largest-count symbol. With nonzero
         * `total`, at least one symbol has freq > 0. */
        int idx = 0;
        uint32_t best = 0;
        for (int i = 0; i < FSE_NSYMS; ++i) {
            if (freq[i] > best) { best = freq[i]; idx = i; }
        }
        norm[idx] = (uint16_t)((int)norm[idx] + diff);
    } else {
        /* Pull from symbols with norm > 1, biggest first, until balanced.
         * The loop is bounded by the magnitude of `diff`, which itself is
         * bounded by FSE_NSYMS (each over-allocation is at most 1 per
         * symbol from the floor()→1 minimum bump). */
        while (diff < 0) {
            int idx = -1;
            uint16_t best = 1;
            for (int i = 0; i < FSE_NSYMS; ++i) {
                if (norm[i] > best) { best = norm[i]; idx = i; }
            }
            if (idx < 0) break;  /* unreachable on valid inputs */
            norm[idx]--;
            diff++;
        }
    }
}

/* ----- Cumulative table + slot→symbol map --------------------------------- */
/*
 * cum[s]            = sum of norm[0..s-1]
 * cum[FSE_NSYMS]    = FSE_TABLE_SIZE   (sentinel)
 * slot_to_sym[i]    = symbol s such that cum[s] <= i < cum[s] + norm[s]
 *
 * The slot table makes the decoder's symbol lookup O(1) per output byte.
 * Sequential layout (no FSE-style spread) is the right choice for rANS:
 * each contiguous run [cum[s], cum[s] + norm[s]) corresponds to symbol
 * s, exactly matching the slot = state & MASK lookup.
 */
typedef struct {
    uint16_t cum[FSE_NSYMS + 1];
    uint16_t slot_to_sym[FSE_TABLE_SIZE];
} fse_decode_tables;

static tdc_status fse_build_tables(const uint16_t norm[FSE_NSYMS],
                                   fse_decode_tables *t) {
    uint32_t acc = 0;
    for (int s = 0; s < FSE_NSYMS; ++s) {
        t->cum[s] = (uint16_t)acc;
        acc += norm[s];
    }
    t->cum[FSE_NSYMS] = (uint16_t)acc;
    if (acc != FSE_TABLE_SIZE) return TDC_E_CORRUPT;

    uint32_t p = 0;
    for (int s = 0; s < FSE_NSYMS; ++s) {
        for (uint32_t j = 0; j < norm[s]; ++j) {
            t->slot_to_sym[p++] = (uint16_t)s;
        }
    }
    return TDC_OK;
}

/* ----- Encode ------------------------------------------------------------- */

static size_t fse_encode_bound(size_t src_size) {
    /* Header (524) + worst-case ~1 byte/symbol stream + 8 bytes flush slack.
     * On real distributions the stream shrinks toward H(X) bytes/symbol;
     * the bound is for the random/incompressible case. */
    return (size_t)FSE_HDR_SIZE + src_size + 16u;
}

/*
 * Push the encoder byte stream into `scratch` in encoder-emit order
 * (low byte first within a state, last symbol's bytes first overall).
 * Returns the final encoder state via `*out_state`.
 */
static tdc_status fse_encode_stream(const uint8_t *src, size_t src_size,
                                    const uint16_t norm[FSE_NSYMS],
                                    const uint16_t cum [FSE_NSYMS + 1],
                                    tdc_buffer *scratch,
                                    uint32_t   *out_state) {
    uint32_t state = FSE_L_MIN;

    /* x_max factor: state must satisfy state < x_max[s] = ((L_MIN >> TABLE_LOG) << 8) * norm[s]
     * for the encode update to keep state in [L_MIN, L_MIN << 8). With
     * L_MIN = 1<<16 and TABLE_LOG = 12, the factor is (16 << 8) = 4096. */
    const uint32_t x_max_factor = (FSE_L_MIN >> FSE_TABLE_LOG) << 8;

    for (size_t k = src_size; k-- > 0; ) {
        uint8_t  s    = src[k];
        uint32_t freq = norm[s];
        /* If `s` actually occurs in `src`, the normalizer guarantees freq >= 1.
         * Defensive check anyway: a freq of 0 here would be a build bug. */
        if (freq == 0u) return TDC_E_INVAL;

        uint32_t x_max = x_max_factor * freq;
        while (state >= x_max) {
            tdc_status st = tdc_meta_write_u8(scratch, (uint8_t)(state & 0xFFu));
            if (st != TDC_OK) return st;
            state >>= 8;
        }
        state = (state / freq) * FSE_TABLE_SIZE + (state % freq) + (uint32_t)cum[s];
    }

    *out_state = state;
    return TDC_OK;
}

static tdc_status fse_encode(const uint8_t *src, size_t src_size,
                             const void    *params,
                             tdc_buffer    *dst) {
    (void)params;
    if (!dst || !dst->realloc_fn) return TDC_E_INVAL;
    if (src_size > 0 && !src) return TDC_E_INVAL;
    if (src_size > 0xFFFFFFFFu) return TDC_E_INVAL;

    /* Frequencies and normalization. */
    uint32_t freq[FSE_NSYMS] = {0};
    for (size_t i = 0; i < src_size; ++i) freq[src[i]]++;

    uint16_t norm[FSE_NSYMS];
    fse_normalize(freq, (uint32_t)src_size, norm);

    /* Cumulative table (encode side only needs cum[]; decode rebuilds
     * the slot map from norm[]). */
    uint16_t cum[FSE_NSYMS + 1];
    {
        fse_decode_tables t;
        tdc_status st = fse_build_tables(norm, &t);
        /* Empty input is the only legitimate sum != TABLE_SIZE; handle
         * it by skipping the table check below. */
        if (st != TDC_OK && src_size != 0u) return st;
        memcpy(cum, t.cum, sizeof cum);
    }

    /* Encode bytes into a scratch buffer in emit order. */
    tdc_buffer scratch = {0};
    scratch.realloc_fn = dst->realloc_fn;
    scratch.user       = dst->user;

    uint32_t final_state = FSE_L_MIN;
    if (src_size > 0) {
        tdc_status st = fse_encode_stream(src, src_size, norm, cum, &scratch, &final_state);
        if (st != TDC_OK) {
            if (scratch.data) scratch.realloc_fn(scratch.user, scratch.data, 0);
            return st;
        }
    }

    /* Reserve dst and emit header + reversed payload. */
    size_t total = (size_t)FSE_HDR_SIZE + scratch.size;
    tdc_status st = tdc_buf_reserve(dst, total);
    if (st != TDC_OK) {
        if (scratch.data) scratch.realloc_fn(scratch.user, scratch.data, 0);
        return st;
    }

    uint8_t *out = dst->data;
    tdc_le_store_u32(out + 0, (uint32_t)src_size);
    tdc_le_store_u32(out + 4, (uint32_t)scratch.size);
    tdc_le_store_u32(out + 8, final_state);
    memcpy(out + 12, norm, sizeof norm);

    /* Reverse the scratch payload into the on-disk slot. The encoder
     * pushed bytes low-byte-first; the decoder pulls them high-byte-first,
     * so the in-order stream needs to be flipped exactly once. */
    uint8_t *payload_dst = out + FSE_HDR_SIZE;
    for (size_t i = 0; i < scratch.size; ++i) {
        payload_dst[i] = scratch.data[scratch.size - 1 - i];
    }

    if (scratch.data) scratch.realloc_fn(scratch.user, scratch.data, 0);
    dst->size = total;
    return TDC_OK;
}

/* ----- Decode ------------------------------------------------------------- */

static tdc_status fse_decode(const uint8_t *src, size_t src_size,
                             uint8_t       *dst, size_t dst_size) {
    if (src_size < FSE_HDR_SIZE) return TDC_E_CORRUPT;
    if (dst_size > 0 && !dst) return TDC_E_INVAL;

    uint32_t hdr_src_size    = tdc_le_load_u32(src + 0);
    uint32_t hdr_payload_size = tdc_le_load_u32(src + 4);
    uint32_t hdr_final_state  = tdc_le_load_u32(src + 8);
    if ((size_t)hdr_src_size != dst_size) return TDC_E_CORRUPT;
    if ((size_t)hdr_payload_size + (size_t)FSE_HDR_SIZE != src_size) return TDC_E_CORRUPT;

    uint16_t norm[FSE_NSYMS];
    memcpy(norm, src + 12, sizeof norm);

    if (dst_size == 0u) {
        /* Empty input: every norm[] entry must be zero and there must
         * be no payload bytes. final_state is conventionally L_MIN. */
        if (hdr_payload_size != 0u) return TDC_E_CORRUPT;
        for (int s = 0; s < FSE_NSYMS; ++s) {
            if (norm[s] != 0u) return TDC_E_CORRUPT;
        }
        if (hdr_final_state != FSE_L_MIN) return TDC_E_CORRUPT;
        return TDC_OK;
    }

    fse_decode_tables t;
    tdc_status st = fse_build_tables(norm, &t);
    if (st != TDC_OK) return st;

    /* Check that final_state is well-formed: it must be in
     * [L_MIN, L_MIN << 8). Anything outside means a corrupted header. */
    if (hdr_final_state < FSE_L_MIN || hdr_final_state >= (FSE_L_MIN << 8)) {
        return TDC_E_CORRUPT;
    }

    const uint8_t *payload     = src + FSE_HDR_SIZE;
    size_t         payload_off = 0;
    uint32_t       state       = hdr_final_state;

    for (size_t i = 0; i < dst_size; ++i) {
        uint32_t slot = state & FSE_TABLE_MASK;
        uint16_t s    = t.slot_to_sym[slot];
        dst[i] = (uint8_t)s;

        uint32_t freq = norm[s];
        /* slot_to_sym is built from norm[], so freq > 0 here always. */
        state = freq * (state >> FSE_TABLE_LOG) + slot - (uint32_t)t.cum[s];

        while (state < FSE_L_MIN) {
            if (payload_off >= hdr_payload_size) return TDC_E_CORRUPT;
            state = (state << 8) | (uint32_t)payload[payload_off++];
        }
    }

    /* Strict consumption: every payload byte must have been pulled by
     * the decoder. A leftover means encoder/decoder drift, which is
     * always a bug worth surfacing. */
    if (payload_off != hdr_payload_size) return TDC_E_CORRUPT;
    return TDC_OK;
}

const tdc_entropy_vt tdc_entropy_fse_vt = {
    .id           = TDC_ENTROPY_FSE,
    .name         = "fse",
    .encode_bound = fse_encode_bound,
    .encode       = fse_encode,
    .decode       = fse_decode,
};
