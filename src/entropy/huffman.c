/*
 * src/entropy/huffman.c
 *
 * TDC_ENTROPY_HUFFMAN — native static canonical Huffman entropy coder.
 *
 * Why static (not adaptive): the model + transform stages have already
 * de-correlated the byte stream by the time the entropy stage sees it.
 * For typical residual streams the per-symbol distribution is stable
 * over the whole block, so a single tree at the front is within a
 * fraction of a percent of the optimal adaptive coder while costing one
 * pass instead of N.
 *
 * Why canonical: storing only the per-symbol code lengths (256 bytes)
 * is dramatically cheaper than storing the tree itself, and canonical
 * decoders need only two small length-indexed arrays (firstcode[],
 * firstsym[]) plus the symbol-sort permutation.
 *
 * Length-limited to HUFFMAN_MAX_LEN = 15 so:
 *   - the per-symbol length fits in a u8 (we store the full 256-entry
 *     table verbatim, no run-length tricks)
 *   - the canonical decoder's per-length tables are tiny
 *
 * Length-limiting algorithm: heap-based optimal Huffman, then iterative
 * frequency rescaling (>>= 1, floor at 1) until max length <= 15. This
 * is suboptimal compared to package-merge but always converges in a few
 * iterations and is dramatically simpler. The cost is a fraction of a
 * percent in compression ratio on inputs that already need the rescale,
 * which on a 256-symbol alphabet is almost never.
 *
 * On-disk payload (single self-contained block):
 *
 *     u32  src_size            (uncompressed bytes; redundant with the
 *                               caller-provided dst_size, written for
 *                               self-description / corruption detection)
 *     u32  payload_bits        (number of valid bits in the payload)
 *     u16  n_lengths           (number of code length entries, 0..256;
 *                               symbols >= n_lengths are implicitly unused)
 *     u8[n_lengths] code_lens  (0 means symbol unused; otherwise 1..15)
 *     u8[]    payload          ceil(payload_bits / 8) bytes; bits packed
 *                              MSB-first within each byte
 *
 * The n_lengths field stores only as many code lengths as needed —
 * (max_active_symbol + 1) for streams with symbols present, 0 for empty.
 * For full 256-symbol alphabets (byte streams) n_lengths = 256 (header
 * = 266 bytes, 2 more than before). For small alphabets like the LZ
 * symbol streams (22-24 symbols) the header shrinks to ~34 bytes.
 *
 * Edge cases handled by the format:
 *   - Empty input:  src_size == 0, payload_bits == 0, n_lengths == 0.
 *   - Single distinct symbol s in src: code_lengths[s] = 1, codeword 0;
 *     payload is src_size zero bits. The "wasteful" 1 bit per symbol is
 *     intentional — keeping the canonical-Huffman invariant lets the
 *     decoder use the same code path with no special case.
 */

#include "tdc/entropy.h"
#include "entropy_internal.h"
#include "../core/buffer.h"
#include "../format/metadata_internal.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#define HUFFMAN_NSYMS    256
#define HUFFMAN_MAX_LEN  15
/* Fixed header prefix: src_size(4) + payload_bits(4) + n_lengths(2) = 10.
 * Full header size = 10 + n_lengths.  Worst case: 10 + 256 = 266. */
#define HUFFMAN_HDR_PREFIX 10u
#define HUFFMAN_HDR_MAX    (HUFFMAN_HDR_PREFIX + (uint32_t)HUFFMAN_NSYMS)
#define HUFFMAN_FAST_BITS  11
#define HUFFMAN_FAST_SIZE  (1u << HUFFMAN_FAST_BITS)

/* ----- Heap-based optimal Huffman length computation ---------------------- */
/*
 * Standard textbook Huffman: each leaf is a symbol with weight = freq;
 * pop the two smallest, push their sum, repeat. We track parent pointers
 * in a flat node array so we can compute per-leaf depth in a final pass.
 *
 * Node layout in the array:
 *   indices [0, n_leaves)              -> leaves (in input order)
 *   indices [n_leaves, 2*n_leaves - 1) -> internal nodes
 *
 * Min-heap holds node indices, ordered by their weight. With at most 256
 * leaves, the worst-case heap size is 256, internal-node count is 255,
 * total nodes 511. Sized at 512 for alignment slack.
 */

typedef struct {
    uint64_t weight;   /* u64 to avoid overflow on degenerate freq sums */
    int32_t  parent;   /* index of parent in the same array, or -1 */
} huff_node;

typedef struct {
    int32_t  idx[HUFFMAN_NSYMS];     /* heap indices into nodes[] */
    uint64_t key[HUFFMAN_NSYMS];     /* duplicated weight for fast compare */
    int      size;
} huff_heap;

static void heap_swap(huff_heap *h, int a, int b) {
    int32_t  ti = h->idx[a]; h->idx[a] = h->idx[b]; h->idx[b] = ti;
    uint64_t tk = h->key[a]; h->key[a] = h->key[b]; h->key[b] = tk;
}

static void heap_push(huff_heap *h, int32_t node, uint64_t weight) {
    int i = h->size++;
    h->idx[i] = node;
    h->key[i] = weight;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h->key[p] <= h->key[i]) break;
        heap_swap(h, p, i);
        i = p;
    }
}

static int32_t heap_pop(huff_heap *h) {
    int32_t top = h->idx[0];
    --h->size;
    if (h->size > 0) {
        h->idx[0] = h->idx[h->size];
        h->key[0] = h->key[h->size];
        int i = 0;
        for (;;) {
            int l = 2 * i + 1;
            int r = l + 1;
            int s = i;
            if (l < h->size && h->key[l] < h->key[s]) s = l;
            if (r < h->size && h->key[r] < h->key[s]) s = r;
            if (s == i) break;
            heap_swap(h, i, s);
            i = s;
        }
    }
    return top;
}

/*
 * Compute optimal (unlimited-length) code lengths for `freq[]`. Symbols
 * with freq == 0 receive length 0. Symbols with freq > 0 receive their
 * Huffman code length (1..ceil(log2(n_used)) typically, but can be much
 * larger on degenerate inputs).
 *
 * Returns the number of distinct symbols (n_used).
 */
static int huffman_compute_lengths(const uint32_t freq[HUFFMAN_NSYMS],
                                   uint8_t        lens[HUFFMAN_NSYMS]) {
    huff_node nodes[2 * HUFFMAN_NSYMS];
    huff_heap heap = { .size = 0 };
    int n_used = 0;

    for (int i = 0; i < HUFFMAN_NSYMS; ++i) lens[i] = 0;

    for (int s = 0; s < HUFFMAN_NSYMS; ++s) {
        if (freq[s] == 0u) continue;
        nodes[n_used].weight = (uint64_t)freq[s];
        nodes[n_used].parent = -1;
        heap_push(&heap, n_used, nodes[n_used].weight);
        /* Remember which leaf index corresponds to which symbol via a
         * separate pass below — we'll re-walk freq[] to map back. */
        ++n_used;
    }

    if (n_used == 0) return 0;
    if (n_used == 1) {
        /* Single distinct symbol: assign length 1 by convention so the
         * canonical encoder still emits one bit per occurrence. */
        for (int s = 0; s < HUFFMAN_NSYMS; ++s) {
            if (freq[s] != 0u) { lens[s] = 1; break; }
        }
        return 1;
    }

    int n_total = n_used;
    while (heap.size >= 2) {
        int32_t a = heap_pop(&heap);
        int32_t b = heap_pop(&heap);
        int32_t p = n_total++;
        nodes[p].weight = nodes[a].weight + nodes[b].weight;
        nodes[p].parent = -1;
        nodes[a].parent = p;
        nodes[b].parent = p;
        heap_push(&heap, p, nodes[p].weight);
    }

    /* Walk parent pointers from each leaf to compute depth. Leaves are
     * the first n_used nodes, in input-symbol order. */
    int leaf = 0;
    for (int s = 0; s < HUFFMAN_NSYMS; ++s) {
        if (freq[s] == 0u) continue;
        int depth = 0;
        for (int i = leaf; nodes[i].parent != -1; i = nodes[i].parent) {
            ++depth;
            if (depth > 255) break; /* hard guard against malformed parent chains */
        }
        lens[s] = (uint8_t)depth;
        ++leaf;
    }
    return n_used;
}

/*
 * Length-limited driver: compute optimal lengths, then iteratively
 * rescale frequencies (halving while keeping nonzero symbols >= 1)
 * until every length fits in HUFFMAN_MAX_LEN. Always converges:
 * each rescale strictly shrinks the sum of frequencies, which lowers
 * the maximum tree depth.
 */
static int huffman_compute_lengths_limited(const uint32_t freq_in[HUFFMAN_NSYMS],
                                           uint8_t        lens[HUFFMAN_NSYMS]) {
    uint32_t freq[HUFFMAN_NSYMS];
    memcpy(freq, freq_in, sizeof freq);

    int n_used = 0;
    for (int iter = 0; iter < 64; ++iter) {
        n_used = huffman_compute_lengths(freq, lens);
        int max_len = 0;
        for (int s = 0; s < HUFFMAN_NSYMS; ++s) {
            if (lens[s] > max_len) max_len = lens[s];
        }
        if (max_len <= HUFFMAN_MAX_LEN) return n_used;

        for (int s = 0; s < HUFFMAN_NSYMS; ++s) {
            if (freq[s] > 1u) freq[s] >>= 1;
        }
    }
    /* Should be unreachable on any sane 256-symbol input. */
    return n_used;
}

/* ----- Canonical code assignment ----------------------------------------- */
/*
 * Given per-symbol lengths, assign canonical codes:
 *   - sort symbols by (length, value) ascending
 *   - the first symbol at length L gets code "first[L]"
 *   - subsequent symbols at length L get first[L]+1, first[L]+2, ...
 *   - first[L+1] = (first[L] + count[L]) << 1
 *
 * Both the encoder and decoder build the same first[] table from the
 * same length array, so the decoder doesn't need to read codes from
 * disk — only the lengths.
 */

static void huffman_build_canonical(const uint8_t lens[HUFFMAN_NSYMS],
                                    uint16_t      codes[HUFFMAN_NSYMS]) {
    uint16_t bl_count[HUFFMAN_MAX_LEN + 2] = {0};
    uint16_t next_code[HUFFMAN_MAX_LEN + 2] = {0};

    for (int s = 0; s < HUFFMAN_NSYMS; ++s) {
        if (lens[s] != 0) bl_count[lens[s]]++;
    }

    uint16_t code = 0;
    for (int bits = 1; bits <= HUFFMAN_MAX_LEN; ++bits) {
        code = (uint16_t)((code + bl_count[bits - 1]) << 1);
        next_code[bits] = code;
    }

    for (int s = 0; s < HUFFMAN_NSYMS; ++s) {
        codes[s] = 0;
        if (lens[s] != 0) {
            codes[s] = next_code[lens[s]]++;
        }
    }
}

/* ----- 4-way unrolled byte histogram ------------------------------------- */
/*
 * Standard ILP technique for fast byte histograms: four independent
 * frequency tables, each processing every 4th byte. This avoids store-
 * forwarding stalls (RAW hazards) when consecutive bytes map to the same
 * bin — the CPU can keep all four increment chains in flight. Same
 * technique used by zstd, libdeflate, and lz4.
 *
 * Stack cost: 4 × 256 × 4 = 4 KB, well within L1d.
 */
static void huffman_count_freq(const uint8_t *src, size_t src_size,
                               uint32_t freq[HUFFMAN_NSYMS]) {
    memset(freq, 0, HUFFMAN_NSYMS * sizeof(uint32_t));

    /* Small inputs: simple loop, no overhead. */
    if (src_size < 32) {
        for (size_t i = 0; i < src_size; ++i)
            freq[src[i]]++;
        return;
    }

    uint32_t f1[HUFFMAN_NSYMS] = {0};
    uint32_t f2[HUFFMAN_NSYMS] = {0};
    uint32_t f3[HUFFMAN_NSYMS] = {0};
    /* freq itself serves as f0. */

    size_t n4 = src_size & ~(size_t)3;
    size_t i = 0;
    for (; i < n4; i += 4) {
        freq[src[i + 0]]++;
        f1  [src[i + 1]]++;
        f2  [src[i + 2]]++;
        f3  [src[i + 3]]++;
    }
    for (; i < src_size; ++i)
        freq[src[i]]++;

    /* Merge the three auxiliary tables into freq[]. */
    for (int s = 0; s < HUFFMAN_NSYMS; ++s)
        freq[s] += f1[s] + f2[s] + f3[s];
}

/* ----- Bit writer (64-bit accumulator) ----------------------------------- */
/*
 * MSB-first packer using a 64-bit accumulator.  Bits are inserted at
 * position (63 - nbits) downward; when the accumulator holds >= 32 valid
 * bits the top 4 bytes are flushed to the output buffer in big-endian
 * order (which *is* MSB-first byte order).
 *
 * One shift + OR per symbol, one conditional 4-byte flush roughly every
 * 4 symbols (at average code length ~8).  Replaces the previous per-bit
 * loop that ran 1-15 iterations per symbol.
 *
 * Overflow safety: max nbits before a write = 31 (flush threshold − 1).
 * Max code length = HUFFMAN_MAX_LEN = 15.  31 + 15 = 46 < 64.
 */

typedef struct {
    uint8_t *p;
    size_t   used;
    uint64_t buf;        /* MSB-aligned: valid bits sit at 63 .. 64-nbits */
    int      nbits;      /* number of valid bits in buf, 0..46 */
    uint64_t total_bits;
} bit_writer;

static void bw_init(bit_writer *bw, uint8_t *out) {
    bw->p          = out;
    bw->used       = 0;
    bw->buf        = 0;
    bw->nbits      = 0;
    bw->total_bits = 0;
}

static inline void bw_write_bits(bit_writer *bw, uint16_t code, int nbits) {
    bw->total_bits += (uint64_t)nbits;
    bw->buf   |= (uint64_t)code << (64 - bw->nbits - nbits);
    bw->nbits += nbits;
    if (bw->nbits >= 32) {
        uint32_t top = (uint32_t)(bw->buf >> 32);
        bw->p[bw->used + 0] = (uint8_t)(top >> 24);
        bw->p[bw->used + 1] = (uint8_t)(top >> 16);
        bw->p[bw->used + 2] = (uint8_t)(top >>  8);
        bw->p[bw->used + 3] = (uint8_t)(top);
        bw->used  += 4;
        bw->buf  <<= 32;
        bw->nbits -= 32;
    }
}

static void bw_flush(bit_writer *bw) {
    while (bw->nbits >= 8) {
        bw->p[bw->used++] = (uint8_t)(bw->buf >> 56);
        bw->buf  <<= 8;
        bw->nbits -= 8;
    }
    if (bw->nbits > 0) {
        /* Partial byte: valid bits are MSB-aligned, low bits already 0. */
        bw->p[bw->used++] = (uint8_t)(bw->buf >> 56);
        bw->buf   = 0;
        bw->nbits = 0;
    }
}

/* ----- 64-bit bit reader (decode) ---------------------------------------- */
/*
 * MSB-first reader with a 64-bit accumulator and bulk refill.  The
 * accumulator holds up to 64 valid bits packed at the MSB end; a bulk
 * refill loads up to 7 bytes in a single 8-byte memcpy + byte-swap,
 * filling the accumulator to >= 56 bits.
 *
 * Compared to the previous 32-bit byte-at-a-time reader:
 *   - Refills happen every ~7 symbols instead of every ~2
 *   - Each refill is one 8-byte load + bswap, not a while loop
 *   - The accumulator holds enough bits for 3+ symbols at max code
 *     length (15), eliminating most refill-check branches
 */

static inline uint64_t huf_bswap64(uint64_t v) {
#if defined(_MSC_VER)
    return _byteswap_uint64(v);          /* single bswap instruction */
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap64(v);
#else
    v = ((v & 0x00FF00FF00FF00FFULL) <<  8) | ((v >>  8) & 0x00FF00FF00FF00FFULL);
    v = ((v & 0x0000FFFF0000FFFFULL) << 16) | ((v >> 16) & 0x0000FFFF0000FFFFULL);
    return (v << 32) | (v >> 32);
#endif
}

typedef struct {
    uint64_t       bits;   /* MSB-aligned: valid bits at 63..64-nbits */
    int            nbits;  /* 0..64 */
    const uint8_t *ptr;    /* next unread byte */
    const uint8_t *safe;   /* ptr < safe => safe to do 8-byte memcpy */
    const uint8_t *end;    /* one past last payload byte */
} huf_br;

static inline void huf_br_init(huf_br *br, const uint8_t *src, size_t n) {
    br->bits  = 0;
    br->nbits = 0;
    br->ptr   = src;
    br->end   = src + n;
    br->safe  = (n >= 8) ? src + n - 7 : src;
}

/* Bulk refill: one 8-byte load + bswap, consumes up to 7 new bytes.
 * Caller must ensure ptr < safe. */
static inline void huf_br_refill_fast(huf_br *br) {
    uint64_t next;
    memcpy(&next, br->ptr, 8);
    next = huf_bswap64(next);
    br->bits |= next >> br->nbits;
    unsigned fresh = (unsigned)(64 - br->nbits) >> 3;
    br->ptr   += fresh;
    br->nbits += (int)(fresh << 3);
}

/* Byte-at-a-time refill for end-of-stream safety. */
static inline void huf_br_refill_safe(huf_br *br) {
    while (br->nbits <= 56 && br->ptr < br->end) {
        br->bits |= (uint64_t)(*br->ptr++) << (56 - br->nbits);
        br->nbits += 8;
    }
}

/* Refill dispatcher. */
static inline void huf_br_refill(huf_br *br) {
    if (br->ptr < br->safe) huf_br_refill_fast(br);
    else                     huf_br_refill_safe(br);
}

/* ----- Encode / decode entry points -------------------------------------- */

static size_t huffman_encode_bound(size_t src_size) {
    /* Header (up to 266 bytes) + worst-case payload (15 bits/symbol) +
     * 1 byte flush slack. 15/8 = 1.875 bytes/byte; round up via /4 + src. */
    return (size_t)HUFFMAN_HDR_MAX + src_size + (src_size / 2u) + 16u;
}

static tdc_status huffman_encode(const uint8_t *src, size_t src_size,
                                 const void    *params,
                                 tdc_buffer    *dst) {
    (void)params;
    if (!dst || !dst->realloc_fn) return TDC_E_INVAL;
    if (src_size > 0 && !src) return TDC_E_INVAL;
    if (src_size > 0xFFFFFFFFu) return TDC_E_INVAL;

    /* Reserve worst-case output up front; the bit writer assumes its
     * destination pointer is stable for the duration of the encode. */
    size_t bound = huffman_encode_bound(src_size);
    tdc_status st = tdc_buf_reserve(dst, bound);
    if (st != TDC_OK) return st;

    /* Frequencies (4-way unrolled for ILP). */
    uint32_t freq[HUFFMAN_NSYMS];
    huffman_count_freq(src, src_size, freq);

    /* Lengths (length-limited) and canonical codes. */
    uint8_t  lens[HUFFMAN_NSYMS];
    uint16_t codes[HUFFMAN_NSYMS];
    (void)huffman_compute_lengths_limited(freq, lens);
    huffman_build_canonical(lens, codes);

    /* Compute n_lengths: highest active symbol + 1 (0 for empty). */
    uint16_t n_lengths = 0;
    for (int s = HUFFMAN_NSYMS - 1; s >= 0; --s) {
        if (lens[s] != 0) { n_lengths = (uint16_t)(s + 1); break; }
    }
    uint32_t hdr_size = HUFFMAN_HDR_PREFIX + (uint32_t)n_lengths;

    /* Write header: src_size, payload_bits placeholder, n_lengths, code lengths. */
    uint8_t *out = dst->data;
    tdc_le_store_u32(out + 0, (uint32_t)src_size);
    /* payload_bits is patched after the bit writer reports its total. */
    tdc_le_store_u32(out + 4, 0u);
    tdc_le_store_u16(out + 8, n_lengths);
    if (n_lengths > 0) memcpy(out + HUFFMAN_HDR_PREFIX, lens, n_lengths);

    /* Pack code + length into a single table: enc[s] = (code << 8) | len.
     * One load per symbol instead of two, 1 KB instead of 2 KB in L1d. */
    uint32_t enc[HUFFMAN_NSYMS];
    for (int s = 0; s < HUFFMAN_NSYMS; ++s)
        enc[s] = ((uint32_t)codes[s] << 8) | (uint32_t)lens[s];

    /* Bit-pack payload. */
    bit_writer bw;
    bw_init(&bw, out + hdr_size);

    size_t i = 0;
    size_t n4 = src_size & ~(size_t)3;
    for (; i < n4; i += 4) {
        uint32_t e0 = enc[src[i + 0]];
        uint32_t e1 = enc[src[i + 1]];
        uint32_t e2 = enc[src[i + 2]];
        uint32_t e3 = enc[src[i + 3]];
        bw_write_bits(&bw, (uint16_t)(e0 >> 8), (int)(e0 & 0xFF));
        bw_write_bits(&bw, (uint16_t)(e1 >> 8), (int)(e1 & 0xFF));
        bw_write_bits(&bw, (uint16_t)(e2 >> 8), (int)(e2 & 0xFF));
        bw_write_bits(&bw, (uint16_t)(e3 >> 8), (int)(e3 & 0xFF));
    }
    for (; i < src_size; ++i) {
        uint32_t e = enc[src[i]];
        bw_write_bits(&bw, (uint16_t)(e >> 8), (int)(e & 0xFF));
    }
    uint64_t total_bits = bw.total_bits;
    bw_flush(&bw);

    /* Patch payload_bits and finalize size. */
    if (total_bits > 0xFFFFFFFFu) return TDC_E_INVAL;
    tdc_le_store_u32(out + 4, (uint32_t)total_bits);
    dst->size = (size_t)hdr_size + bw.used;
    return TDC_OK;
}

/* Huffman encode with pre-computed frequency table. Avoids the freq-counting
 * pass when the caller already has a histogram (e.g. lzs_encode_stream_fast
 * computes one for the Shannon entropy check). */
tdc_status tdc_huffman_encode_prefreq(const uint8_t *src, size_t src_size,
                                       const uint32_t freq[256],
                                       tdc_buffer *dst) {
    if (!dst || !dst->realloc_fn) return TDC_E_INVAL;
    if (src_size > 0 && !src) return TDC_E_INVAL;
    if (src_size > 0xFFFFFFFFu) return TDC_E_INVAL;

    size_t bound = huffman_encode_bound(src_size);
    tdc_status st = tdc_buf_reserve(dst, bound);
    if (st != TDC_OK) return st;

    uint8_t  lens[HUFFMAN_NSYMS];
    uint16_t codes[HUFFMAN_NSYMS];
    (void)huffman_compute_lengths_limited(freq, lens);
    huffman_build_canonical(lens, codes);

    uint16_t n_lengths = 0;
    for (int s = HUFFMAN_NSYMS - 1; s >= 0; --s) {
        if (lens[s] != 0) { n_lengths = (uint16_t)(s + 1); break; }
    }
    uint32_t hdr_size = HUFFMAN_HDR_PREFIX + (uint32_t)n_lengths;

    uint8_t *out = dst->data;
    tdc_le_store_u32(out + 0, (uint32_t)src_size);
    tdc_le_store_u32(out + 4, 0u);
    tdc_le_store_u16(out + 8, n_lengths);
    if (n_lengths > 0) memcpy(out + HUFFMAN_HDR_PREFIX, lens, n_lengths);

    uint32_t enc[HUFFMAN_NSYMS];
    for (int s = 0; s < HUFFMAN_NSYMS; ++s)
        enc[s] = ((uint32_t)codes[s] << 8) | (uint32_t)lens[s];

    bit_writer bw;
    bw_init(&bw, out + hdr_size);

    size_t i = 0;
    size_t n4 = src_size & ~(size_t)3;
    for (; i < n4; i += 4) {
        uint32_t e0 = enc[src[i + 0]];
        uint32_t e1 = enc[src[i + 1]];
        uint32_t e2 = enc[src[i + 2]];
        uint32_t e3 = enc[src[i + 3]];
        bw_write_bits(&bw, (uint16_t)(e0 >> 8), (int)(e0 & 0xFF));
        bw_write_bits(&bw, (uint16_t)(e1 >> 8), (int)(e1 & 0xFF));
        bw_write_bits(&bw, (uint16_t)(e2 >> 8), (int)(e2 & 0xFF));
        bw_write_bits(&bw, (uint16_t)(e3 >> 8), (int)(e3 & 0xFF));
    }
    for (; i < src_size; ++i) {
        uint32_t e = enc[src[i]];
        bw_write_bits(&bw, (uint16_t)(e >> 8), (int)(e & 0xFF));
    }
    uint64_t total_bits = bw.total_bits;
    bw_flush(&bw);

    if (total_bits > 0xFFFFFFFFu) return TDC_E_INVAL;
    tdc_le_store_u32(out + 4, (uint32_t)total_bits);
    dst->size = (size_t)hdr_size + bw.used;
    return TDC_OK;
}

static tdc_status huffman_decode(const uint8_t *src, size_t src_size,
                                 uint8_t       *dst, size_t dst_size) {
    if (src_size < HUFFMAN_HDR_PREFIX) return TDC_E_CORRUPT;
    if (dst_size > 0 && !dst) return TDC_E_INVAL;

    uint32_t hdr_src_size     = tdc_le_load_u32(src + 0);
    uint32_t hdr_payload_bits = tdc_le_load_u32(src + 4);
    uint16_t n_lengths        = tdc_le_load_u16(src + 8);
    if ((size_t)hdr_src_size != dst_size) return TDC_E_CORRUPT;
    if (n_lengths > HUFFMAN_NSYMS) return TDC_E_CORRUPT;

    uint32_t hdr_size = HUFFMAN_HDR_PREFIX + (uint32_t)n_lengths;
    if (src_size < hdr_size) return TDC_E_CORRUPT;

    /* Build full-size lens array from compact header. */
    uint8_t lens_buf[HUFFMAN_NSYMS];
    memset(lens_buf, 0, sizeof lens_buf);
    if (n_lengths > 0) memcpy(lens_buf, src + HUFFMAN_HDR_PREFIX, n_lengths);
    const uint8_t *lens = lens_buf;

    const uint8_t *payload = src + hdr_size;
    size_t         payload_bytes_avail = src_size - hdr_size;
    size_t         payload_bytes_need  = (size_t)((hdr_payload_bits + 7u) / 8u);
    if (payload_bytes_need > payload_bytes_avail) return TDC_E_CORRUPT;

    /* Empty input. */
    if (dst_size == 0u) {
        if (hdr_payload_bits != 0u) return TDC_E_CORRUPT;
        if (n_lengths != 0u) return TDC_E_CORRUPT;
        return TDC_OK;
    }

    /* Build canonical decode tables: bl_count, firstcode, firstsym,
     * sorted_syms (symbols sorted by (length, value)). All small. */
    uint16_t bl_count[HUFFMAN_MAX_LEN + 2] = {0};
    for (int s = 0; s < HUFFMAN_NSYMS; ++s) {
        if (lens[s] > HUFFMAN_MAX_LEN) return TDC_E_CORRUPT;
        if (lens[s] != 0) bl_count[lens[s]]++;
    }

    uint16_t firstcode[HUFFMAN_MAX_LEN + 2] = {0};
    uint16_t firstsym [HUFFMAN_MAX_LEN + 2] = {0};
    {
        uint16_t code = 0;
        uint16_t idx  = 0;
        for (int bits = 1; bits <= HUFFMAN_MAX_LEN; ++bits) {
            code = (uint16_t)((code + bl_count[bits - 1]) << 1);
            firstcode[bits] = code;
            firstsym[bits]  = idx;
            idx = (uint16_t)(idx + bl_count[bits]);
        }
    }

    uint8_t sorted_syms[HUFFMAN_NSYMS];
    {
        uint16_t cursor[HUFFMAN_MAX_LEN + 2];
        memcpy(cursor, firstsym, sizeof cursor);
        for (int s = 0; s < HUFFMAN_NSYMS; ++s) {
            uint8_t L = lens[s];
            if (L != 0) sorted_syms[cursor[L]++] = (uint8_t)s;
        }
    }

    /* Sanity: total symbol count must be > 0 (we already handled the
     * dst_size == 0 path above). */
    int n_used = 0;
    for (int bits = 1; bits <= HUFFMAN_MAX_LEN; ++bits) n_used += bl_count[bits];
    if (n_used == 0) return TDC_E_CORRUPT;

    /* Build a primary lookup table for codes of length <= HUFFMAN_FAST_BITS.
     * Each entry packs (sym << 8) | code_length; entries with length 0
     * indicate "code is longer than HUFFMAN_FAST_BITS, use the slow walk".
     * 11 bits -> 2 KiB table on the stack — well within any thread's stack
     * margin and small enough to live in L1d. Codes of length 12..15 fall
     * through to the canonical bit-walk (rare on real residual streams). */
    uint16_t fast[HUFFMAN_FAST_SIZE];
    memset(fast, 0, sizeof fast);
    {
        uint16_t code = 0;
        for (int L = 1; L <= HUFFMAN_FAST_BITS; ++L) {
            uint32_t shift = (uint32_t)(HUFFMAN_FAST_BITS - L);
            uint32_t span  = 1u << shift;
            for (uint16_t i = 0; i < bl_count[L]; ++i) {
                uint8_t  sym   = sorted_syms[firstsym[L] + i];
                uint16_t entry = (uint16_t)(((uint16_t)sym << 8) | (uint16_t)L);
                uint32_t start = (uint32_t)code << shift;
                uint32_t end   = start + span;
                for (uint32_t j = start; j < end; ++j) fast[j] = entry;
                code++;
            }
            code = (uint16_t)(code << 1);
        }
    }

    /* Build 2-symbol fast table.  Each 32-bit entry:
     *   bits 31..24 = sym0     bits 23..16 = sym1 (garbage when L1==0)
     *   bits 15..8  = total_bits (L0 for 1-sym, L0+L1 for 2-sym, 0=slow)
     *   bits  7..0  = L1       (>0 iff two symbols were decoded)
     * Pre-computing total_bits saves an add per symbol in every hot path.
     * 2048 × 4 = 8 KiB on stack — fits in L1d alongside fast[]. */
    uint32_t fast2[HUFFMAN_FAST_SIZE];
    {
        for (uint32_t idx = 0; idx < HUFFMAN_FAST_SIZE; ++idx) {
            uint16_t e0 = fast[idx];
            int L0 = e0 & 0xFF;
            if (L0 == 0) { fast2[idx] = 0; continue; }
            uint8_t s0 = (uint8_t)(e0 >> 8);
            int rem = HUFFMAN_FAST_BITS - L0;
            if (rem > 0) {
                uint32_t idx2 = (idx << L0) & (HUFFMAN_FAST_SIZE - 1);
                uint16_t e1 = fast[idx2];
                int L1 = e1 & 0xFF;
                if (L1 != 0 && L1 <= rem) {
                    uint8_t s1 = (uint8_t)(e1 >> 8);
                    fast2[idx] = ((uint32_t)s0 << 24) | ((uint32_t)s1 << 16) |
                                 ((uint32_t)(L0 + L1) << 8) | (uint32_t)L1;
                    continue;
                }
            }
            /* 1-symbol entry: total_bits = L0, L1 = 0 */
            fast2[idx] = ((uint32_t)s0 << 24) | ((uint32_t)L0 << 8);
        }
    }

    /* Decode loop — 2-symbol fast path, 1-symbol fallback, slow tail.
     *
     * Entry layout lets us read total_bits = (entry >> 8) & 0xFF for BOTH
     * the 1-sym and 2-sym paths — no add needed, just one extract.  */
    huf_br br;
    huf_br_init(&br, payload, payload_bytes_need);

    size_t i = 0;
    size_t safe_end = (dst_size > 0) ? dst_size - 1 : 0;
    for (; i < safe_end; ) {
        /* Guard: skip 8-byte load + bswap when the bit buffer still has
         * enough for a 2-sym fast2 entry (max 22 bits).  After a refill
         * to ~56 bits, this skips ~2 iterations before re-triggering. */
        if (br.nbits < 2 * HUFFMAN_FAST_BITS) huf_br_refill(&br);
        {
            uint32_t idx   = (uint32_t)(br.bits >> (64 - HUFFMAN_FAST_BITS));
            uint32_t entry = fast2[idx];
            int      tb    = (int)((entry >> 8) & 0xFF);
            if (entry & 0xFF) {           /* 2-symbol path */
                dst[i]     = (uint8_t)(entry >> 24);
                dst[i + 1] = (uint8_t)(entry >> 16);
                br.bits  <<= tb;
                br.nbits  -= tb;
                i += 2;
                continue;
            }
            if (tb) {                     /* 1-symbol path */
                dst[i]     = (uint8_t)(entry >> 24);
                br.bits  <<= tb;
                br.nbits  -= tb;
                i += 1;
                continue;
            }
        }
        /* Slow path: per-bit canonical walk for this one symbol. */
        {
            uint16_t code = 0;
            int L;
            for (L = 1; L <= HUFFMAN_MAX_LEN; ++L) {
                if (br.nbits < 1) huf_br_refill(&br);
                code = (uint16_t)((code << 1) | (uint16_t)(br.bits >> 63));
                br.bits <<= 1;
                br.nbits -= 1;
                if (bl_count[L] != 0u && code < (uint16_t)(firstcode[L] + bl_count[L])) {
                    uint16_t off = (uint16_t)(code - firstcode[L]);
                    dst[i] = sorted_syms[firstsym[L] + off];
                    break;
                }
            }
            if (L > HUFFMAN_MAX_LEN) return TDC_E_CORRUPT;
            i += 1;
        }
    }
    /* Final symbol (at most 1 remaining). */
    for (; i < dst_size; ++i) {
        if (br.nbits < HUFFMAN_FAST_BITS) huf_br_refill(&br);
        if (br.nbits >= HUFFMAN_FAST_BITS) {
            uint32_t idx   = (uint32_t)(br.bits >> (64 - HUFFMAN_FAST_BITS));
            uint16_t entry = fast[idx];
            int      L     = entry & 0xFF;
            if (L != 0) {
                dst[i]     = (uint8_t)(entry >> 8);
                br.bits  <<= L;
                br.nbits  -= L;
                continue;
            }
        }
        uint16_t code = 0;
        int L;
        for (L = 1; L <= HUFFMAN_MAX_LEN; ++L) {
            if (br.nbits < 1) huf_br_refill(&br);
            code = (uint16_t)((code << 1) | (uint16_t)(br.bits >> 63));
            br.bits <<= 1;
            br.nbits -= 1;
            if (bl_count[L] != 0u && code < (uint16_t)(firstcode[L] + bl_count[L])) {
                dst[i] = sorted_syms[firstsym[L] + (code - firstcode[L])];
                break;
            }
        }
        if (L > HUFFMAN_MAX_LEN) return TDC_E_CORRUPT;
    }

    return TDC_OK;
}

const tdc_entropy_vt tdc_entropy_huffman_vt = {
    .id           = TDC_ENTROPY_HUFFMAN,
    .name         = "huffman",
    .encode_bound = huffman_encode_bound,
    .encode       = huffman_encode,
    .decode       = huffman_decode,
};

/* ======================================================================== */
/* 4-stream Huffman (TDC_ENTROPY_HUFFMAN4)                                  */
/*                                                                          */
/* Same canonical Huffman tree but the payload is split into 4 independent  */
/* bitstreams.  The decoder interleaves 4 huf_br readers, giving the CPU   */
/* 4 independent decode chains to schedule in parallel — the same trick    */
/* that makes zstd's Huffman decoder fast.                                  */
/*                                                                          */
/* On-disk layout (after the shared code-lengths header):                   */
/*   u32 stream_bytes[3]   compressed size of streams 0-2 (stream 3 is     */
/*                         the remainder of the payload)                    */
/*   u8[] stream0 .. stream3   MSB-first bitstreams                        */
/*                                                                          */
/* For src_size < 256 the encoder/decoder delegate to single-stream         */
/* HUFFMAN (overhead of 4 streams not worth it on tiny blocks).            */
/* ======================================================================== */

#define HUF4_STREAM_HDR 12u   /* 3 x u32 stream sizes */
#define HUF4_SMALL_THRESH 256u

static size_t huffman4_encode_bound(size_t src_size) {
    return (size_t)HUFFMAN_HDR_MAX + HUF4_STREAM_HDR +
           src_size + (src_size / 2u) + 32u;
}

static tdc_status huffman4_encode(const uint8_t *src, size_t src_size,
                                  const void    *params,
                                  tdc_buffer    *dst) {
    (void)params;
    if (!dst || !dst->realloc_fn) return TDC_E_INVAL;
    if (src_size > 0 && !src) return TDC_E_INVAL;
    if (src_size > 0xFFFFFFFFu) return TDC_E_INVAL;

    /* Small input: delegate to single-stream. */
    if (src_size < HUF4_SMALL_THRESH)
        return huffman_encode(src, src_size, params, dst);

    size_t bound = huffman4_encode_bound(src_size);
    tdc_status st = tdc_buf_reserve(dst, bound);
    if (st != TDC_OK) return st;

    /* Shared Huffman tree across all 4 streams. */
    uint32_t freq[HUFFMAN_NSYMS];
    huffman_count_freq(src, src_size, freq);

    uint8_t  lens[HUFFMAN_NSYMS];
    uint16_t codes[HUFFMAN_NSYMS];
    (void)huffman_compute_lengths_limited(freq, lens);
    huffman_build_canonical(lens, codes);

    uint16_t n_lengths = 0;
    for (int s = HUFFMAN_NSYMS - 1; s >= 0; --s) {
        if (lens[s] != 0) { n_lengths = (uint16_t)(s + 1); break; }
    }
    uint32_t hdr_size = HUFFMAN_HDR_PREFIX + (uint32_t)n_lengths;

    uint8_t *out = dst->data;
    tdc_le_store_u32(out + 0, (uint32_t)src_size);
    tdc_le_store_u32(out + 4, 0u);  /* unused for 4-stream */
    tdc_le_store_u16(out + 8, n_lengths);
    if (n_lengths > 0) memcpy(out + HUFFMAN_HDR_PREFIX, lens, n_lengths);

    uint32_t enc[HUFFMAN_NSYMS];
    for (int s = 0; s < HUFFMAN_NSYMS; ++s)
        enc[s] = ((uint32_t)codes[s] << 8) | (uint32_t)lens[s];

    /* Split source into 4 chunks. */
    size_t chunk = (src_size + 3u) / 4u;
    size_t off[5];
    off[0] = 0;
    off[1] = chunk;
    off[2] = 2u * chunk;
    off[3] = 3u * chunk;
    off[4] = src_size;
    for (int j = 1; j < 4; ++j)
        if (off[j] > src_size) off[j] = src_size;

    uint8_t *ss_ptr      = out + hdr_size;
    uint8_t *payload_ptr = ss_ptr + HUF4_STREAM_HDR;

    /* Encode each chunk with its own bit_writer. */
    uint32_t stream_bytes[4];
    size_t write_pos = 0;
    for (int si = 0; si < 4; ++si) {
        size_t begin = off[si], end_off = off[si + 1];
        size_t count = end_off - begin;
        if (count == 0) { stream_bytes[si] = 0; continue; }

        bit_writer bw;
        bw_init(&bw, payload_ptr + write_pos);

        size_t k = begin;
        size_t n4 = begin + (count & ~(size_t)3);
        for (; k < n4; k += 4) {
            uint32_t e0 = enc[src[k + 0]];
            uint32_t e1 = enc[src[k + 1]];
            uint32_t e2 = enc[src[k + 2]];
            uint32_t e3 = enc[src[k + 3]];
            bw_write_bits(&bw, (uint16_t)(e0 >> 8), (int)(e0 & 0xFF));
            bw_write_bits(&bw, (uint16_t)(e1 >> 8), (int)(e1 & 0xFF));
            bw_write_bits(&bw, (uint16_t)(e2 >> 8), (int)(e2 & 0xFF));
            bw_write_bits(&bw, (uint16_t)(e3 >> 8), (int)(e3 & 0xFF));
        }
        for (; k < end_off; ++k) {
            uint32_t e = enc[src[k]];
            bw_write_bits(&bw, (uint16_t)(e >> 8), (int)(e & 0xFF));
        }
        bw_flush(&bw);
        stream_bytes[si] = (uint32_t)bw.used;
        write_pos += bw.used;
    }

    tdc_le_store_u32(ss_ptr + 0, stream_bytes[0]);
    tdc_le_store_u32(ss_ptr + 4, stream_bytes[1]);
    tdc_le_store_u32(ss_ptr + 8, stream_bytes[2]);

    dst->size = (size_t)(payload_ptr - out) + write_pos;
    return TDC_OK;
}

/* ----- 4-stream decode helper: build fast table into caller's array ------ */

static void huf4_build_fast_table(const uint8_t  *lens,
                                  const uint16_t *bl_count,
                                  const uint16_t *firstsym,
                                  const uint8_t  *sorted_syms,
                                  uint16_t        fast[HUFFMAN_FAST_SIZE]) {
    memset(fast, 0, HUFFMAN_FAST_SIZE * sizeof(uint16_t));
    uint16_t code = 0;
    for (int L = 1; L <= HUFFMAN_FAST_BITS; ++L) {
        uint32_t shift = (uint32_t)(HUFFMAN_FAST_BITS - L);
        uint32_t span  = 1u << shift;
        for (uint16_t i = 0; i < bl_count[L]; ++i) {
            uint8_t  sym   = sorted_syms[firstsym[L] + i];
            uint16_t entry = (uint16_t)(((uint16_t)sym << 8) | (uint16_t)L);
            uint32_t start = (uint32_t)code << shift;
            uint32_t end   = start + span;
            for (uint32_t j = start; j < end; ++j) fast[j] = entry;
            code++;
        }
        code = (uint16_t)(code << 1);
    }
    (void)lens;
}

static tdc_status huffman4_decode(const uint8_t *src, size_t src_size,
                                  uint8_t       *dst, size_t dst_size) {
    if (src_size < HUFFMAN_HDR_PREFIX) return TDC_E_CORRUPT;
    if (dst_size > 0 && !dst) return TDC_E_INVAL;

    uint32_t hdr_src_size = tdc_le_load_u32(src + 0);
    uint16_t n_lengths    = tdc_le_load_u16(src + 8);
    if ((size_t)hdr_src_size != dst_size) return TDC_E_CORRUPT;
    if (n_lengths > HUFFMAN_NSYMS) return TDC_E_CORRUPT;

    /* Small input: single-stream fallback (encoder wrote single-stream). */
    if (dst_size < HUF4_SMALL_THRESH)
        return huffman_decode(src, src_size, dst, dst_size);

    uint32_t hdr_size = HUFFMAN_HDR_PREFIX + (uint32_t)n_lengths;
    if (src_size < (size_t)hdr_size + HUF4_STREAM_HDR) return TDC_E_CORRUPT;

    uint8_t lens_buf[HUFFMAN_NSYMS];
    memset(lens_buf, 0, sizeof lens_buf);
    if (n_lengths > 0) memcpy(lens_buf, src + HUFFMAN_HDR_PREFIX, n_lengths);

    if (dst_size == 0u) return TDC_OK;

    /* Build canonical decode tables. */
    uint16_t bl_count[HUFFMAN_MAX_LEN + 2] = {0};
    for (int s = 0; s < HUFFMAN_NSYMS; ++s) {
        if (lens_buf[s] > HUFFMAN_MAX_LEN) return TDC_E_CORRUPT;
        if (lens_buf[s] != 0) bl_count[lens_buf[s]]++;
    }

    uint16_t firstcode[HUFFMAN_MAX_LEN + 2] = {0};
    uint16_t firstsym [HUFFMAN_MAX_LEN + 2] = {0};
    {
        uint16_t code = 0, idx = 0;
        for (int bits = 1; bits <= HUFFMAN_MAX_LEN; ++bits) {
            code = (uint16_t)((code + bl_count[bits - 1]) << 1);
            firstcode[bits] = code;
            firstsym[bits]  = idx;
            idx = (uint16_t)(idx + bl_count[bits]);
        }
    }

    uint8_t sorted_syms[HUFFMAN_NSYMS];
    {
        uint16_t cursor[HUFFMAN_MAX_LEN + 2];
        memcpy(cursor, firstsym, sizeof cursor);
        for (int s = 0; s < HUFFMAN_NSYMS; ++s) {
            uint8_t L = lens_buf[s];
            if (L != 0) sorted_syms[cursor[L]++] = (uint8_t)s;
        }
    }

    int n_used = 0;
    for (int bits = 1; bits <= HUFFMAN_MAX_LEN; ++bits) n_used += bl_count[bits];
    if (n_used == 0) return TDC_E_CORRUPT;

    uint16_t fast[HUFFMAN_FAST_SIZE];
    huf4_build_fast_table(lens_buf, bl_count, firstsym, sorted_syms, fast);

    /* Build 2-symbol fast table: (s0 << 24) | (s1 << 16) | (total << 8) | L1.
     * total = pre-computed L0+L1 (or L0 when L1=0). Saves an add per symbol. */
    uint32_t fast2[HUFFMAN_FAST_SIZE];
    {
        for (uint32_t idx = 0; idx < HUFFMAN_FAST_SIZE; ++idx) {
            uint16_t e0 = fast[idx];
            int L0 = e0 & 0xFF;
            if (L0 == 0) { fast2[idx] = 0; continue; }
            uint8_t s0 = (uint8_t)(e0 >> 8);
            int rem = HUFFMAN_FAST_BITS - L0;
            if (rem > 0) {
                uint32_t idx2 = (idx << L0) & (HUFFMAN_FAST_SIZE - 1);
                uint16_t e1 = fast[idx2];
                int L1 = e1 & 0xFF;
                if (L1 != 0 && L1 <= rem) {
                    uint8_t s1 = (uint8_t)(e1 >> 8);
                    fast2[idx] = ((uint32_t)s0 << 24) | ((uint32_t)s1 << 16) |
                                 ((uint32_t)(L0 + L1) << 8) | (uint32_t)L1;
                    continue;
                }
            }
            fast2[idx] = ((uint32_t)s0 << 24) | ((uint32_t)L0 << 8);
        }
    }

    /* Read stream sizes. */
    const uint8_t *ss_ptr = src + hdr_size;
    uint32_t ss0 = tdc_le_load_u32(ss_ptr + 0);
    uint32_t ss1 = tdc_le_load_u32(ss_ptr + 4);
    uint32_t ss2 = tdc_le_load_u32(ss_ptr + 8);

    const uint8_t *payload = ss_ptr + HUF4_STREAM_HDR;
    size_t payload_avail = src_size - (size_t)(payload - src);
    if ((size_t)ss0 + ss1 + ss2 > payload_avail) return TDC_E_CORRUPT;
    uint32_t ss3 = (uint32_t)(payload_avail - ss0 - ss1 - ss2);

    /* Output chunk boundaries. */
    size_t chunk = (dst_size + 3u) / 4u;
    size_t n0 = (chunk < dst_size) ? chunk : dst_size;
    size_t n1 = (dst_size > chunk)   ? ((2*chunk < dst_size) ? chunk : dst_size - chunk) : 0;
    size_t n2 = (dst_size > 2*chunk) ? ((3*chunk < dst_size) ? chunk : dst_size - 2*chunk) : 0;
    size_t n3 = (dst_size > 3*chunk) ? dst_size - 3*chunk : 0;

    uint8_t *d0 = dst;
    uint8_t *d1 = dst + n0;
    uint8_t *d2 = dst + n0 + n1;
    uint8_t *d3 = dst + n0 + n1 + n2;

    /* Initialize 4 bit readers. */
    huf_br br0, br1, br2, br3;
    huf_br_init(&br0, payload,             ss0);
    huf_br_init(&br1, payload + ss0,       ss1);
    huf_br_init(&br2, payload + ss0 + ss1, ss2);
    huf_br_init(&br3, payload + ss0 + ss1 + ss2, ss3);

    /* Interleaved 4-stream decode with per-stream position tracking.
     *
     * Each stream has its own output cursor (j0..j3) so it can independently
     * advance by 1 or 2 symbols per fast2 lookup.  Handles both 1-sym and
     * 2-sym fast2 entries inline.  Only bails to sequential tail on slow-path
     * entries (code > HUFFMAN_FAST_BITS, rare on residual streams).
     *
     * Two rounds between refills: max 22 bits per round × 2 = 44 < 56
     * (guaranteed bits after refill). */
    size_t j0 = 0, j1 = 0, j2 = 0, j3 = 0;
    {
        /* Safe margin: 2 fast2 rounds can each advance by 2, plus 1 fast
         * round advances by 1 = max 5 symbols past the loop-check position.
         * Need j + 5 <= n, i.e. j < n - 5, to avoid overwriting. */
        size_t safe0 = (n0 > 5) ? n0 - 5 : 0;
        size_t safe1 = (n1 > 5) ? n1 - 5 : 0;
        size_t safe2 = (n2 > 5) ? n2 - 5 : 0;
        size_t safe3 = (n3 > 5) ? n3 - 5 : 0;

        while (j0 < safe0 && j1 < safe1 && j2 < safe2 && j3 < safe3) {
            huf_br_refill(&br0);
            huf_br_refill(&br1);
            huf_br_refill(&br2);
            huf_br_refill(&br3);

            /* Round 1: peek all 4, lookup, decode. */
            {
                uint32_t pk0 = (uint32_t)(br0.bits >> (64 - HUFFMAN_FAST_BITS));
                uint32_t pk1 = (uint32_t)(br1.bits >> (64 - HUFFMAN_FAST_BITS));
                uint32_t pk2 = (uint32_t)(br2.bits >> (64 - HUFFMAN_FAST_BITS));
                uint32_t pk3 = (uint32_t)(br3.bits >> (64 - HUFFMAN_FAST_BITS));
                uint32_t e0 = fast2[pk0], e1 = fast2[pk1];
                uint32_t e2 = fast2[pk2], e3 = fast2[pk3];
                int tb0 = (int)((e0 >> 8) & 0xFF);
                int tb1 = (int)((e1 >> 8) & 0xFF);
                int tb2 = (int)((e2 >> 8) & 0xFF);
                int tb3 = (int)((e3 >> 8) & 0xFF);
                if (tb0 == 0 || tb1 == 0 || tb2 == 0 || tb3 == 0) break;
                d0[j0] = (uint8_t)(e0 >> 24); d0[j0 + 1] = (uint8_t)(e0 >> 16);
                j0 += 1u + ((e0 & 0xFFu) != 0);
                br0.bits <<= tb0; br0.nbits -= tb0;
                d1[j1] = (uint8_t)(e1 >> 24); d1[j1 + 1] = (uint8_t)(e1 >> 16);
                j1 += 1u + ((e1 & 0xFFu) != 0);
                br1.bits <<= tb1; br1.nbits -= tb1;
                d2[j2] = (uint8_t)(e2 >> 24); d2[j2 + 1] = (uint8_t)(e2 >> 16);
                j2 += 1u + ((e2 & 0xFFu) != 0);
                br2.bits <<= tb2; br2.nbits -= tb2;
                d3[j3] = (uint8_t)(e3 >> 24); d3[j3 + 1] = (uint8_t)(e3 >> 16);
                j3 += 1u + ((e3 & 0xFFu) != 0);
                br3.bits <<= tb3; br3.nbits -= tb3;
            }
            /* Round 2 (fast2: up to 2 symbols, up to 22 bits per stream). */
            {
                uint32_t pk0 = (uint32_t)(br0.bits >> (64 - HUFFMAN_FAST_BITS));
                uint32_t pk1 = (uint32_t)(br1.bits >> (64 - HUFFMAN_FAST_BITS));
                uint32_t pk2 = (uint32_t)(br2.bits >> (64 - HUFFMAN_FAST_BITS));
                uint32_t pk3 = (uint32_t)(br3.bits >> (64 - HUFFMAN_FAST_BITS));
                uint32_t e0 = fast2[pk0], e1 = fast2[pk1];
                uint32_t e2 = fast2[pk2], e3 = fast2[pk3];
                int tb0 = (int)((e0 >> 8) & 0xFF);
                int tb1 = (int)((e1 >> 8) & 0xFF);
                int tb2 = (int)((e2 >> 8) & 0xFF);
                int tb3 = (int)((e3 >> 8) & 0xFF);
                if (tb0 == 0 || tb1 == 0 || tb2 == 0 || tb3 == 0) break;
                d0[j0] = (uint8_t)(e0 >> 24); d0[j0 + 1] = (uint8_t)(e0 >> 16);
                j0 += 1u + ((e0 & 0xFFu) != 0);
                br0.bits <<= tb0; br0.nbits -= tb0;
                d1[j1] = (uint8_t)(e1 >> 24); d1[j1 + 1] = (uint8_t)(e1 >> 16);
                j1 += 1u + ((e1 & 0xFFu) != 0);
                br1.bits <<= tb1; br1.nbits -= tb1;
                d2[j2] = (uint8_t)(e2 >> 24); d2[j2 + 1] = (uint8_t)(e2 >> 16);
                j2 += 1u + ((e2 & 0xFFu) != 0);
                br2.bits <<= tb2; br2.nbits -= tb2;
                d3[j3] = (uint8_t)(e3 >> 24); d3[j3 + 1] = (uint8_t)(e3 >> 16);
                j3 += 1u + ((e3 & 0xFFu) != 0);
                br3.bits <<= tb3; br3.nbits -= tb3;
            }
            /* Round 3 (fast: single symbol, up to 11 bits per stream).
             * Bit budget: rounds 1+2 consumed up to 44 bits; this round
             * consumes up to 11 bits = 55 total, within the 56-bit post-
             * refill guarantee. Gets us to 5-6 symbols/stream/iteration. */
            {
                uint32_t pk0 = (uint32_t)(br0.bits >> (64 - HUFFMAN_FAST_BITS));
                uint32_t pk1 = (uint32_t)(br1.bits >> (64 - HUFFMAN_FAST_BITS));
                uint32_t pk2 = (uint32_t)(br2.bits >> (64 - HUFFMAN_FAST_BITS));
                uint32_t pk3 = (uint32_t)(br3.bits >> (64 - HUFFMAN_FAST_BITS));
                uint16_t fe0 = fast[pk0], fe1 = fast[pk1];
                uint16_t fe2 = fast[pk2], fe3 = fast[pk3];
                int fl0 = fe0 & 0xFF, fl1 = fe1 & 0xFF;
                int fl2 = fe2 & 0xFF, fl3 = fe3 & 0xFF;
                if (fl0 == 0 || fl1 == 0 || fl2 == 0 || fl3 == 0) break;
                d0[j0++] = (uint8_t)(fe0 >> 8);
                br0.bits <<= fl0; br0.nbits -= fl0;
                d1[j1++] = (uint8_t)(fe1 >> 8);
                br1.bits <<= fl1; br1.nbits -= fl1;
                d2[j2++] = (uint8_t)(fe2 >> 8);
                br2.bits <<= fl2; br2.nbits -= fl2;
                d3[j3++] = (uint8_t)(fe3 >> 8);
                br3.bits <<= fl3; br3.nbits -= fl3;
            }
        }
    }

    /* Per-stream sequential tail: finish remaining symbols from each
     * stream's current position (j0..j3).  Uses fast2 + slow path. */
    {
        huf_br  *brs[4] = { &br0, &br1, &br2, &br3 };
        uint8_t *ds[4]  = { d0, d1, d2, d3 };
        size_t   ns[4]  = { n0, n1, n2, n3 };
        size_t   js[4]  = { j0, j1, j2, j3 };

        for (int si = 0; si < 4; ++si) {
            huf_br  *b = brs[si];
            uint8_t *d = ds[si];
            size_t   n = ns[si];
            size_t   j = js[si];
            size_t   safe_end = (n > 0) ? n - 1 : 0;
            for (; j < safe_end; ) {
                if (b->nbits < HUFFMAN_FAST_BITS) huf_br_refill(b);
                if (b->nbits >= HUFFMAN_FAST_BITS) {
                    uint32_t idx   = (uint32_t)(b->bits >> (64 - HUFFMAN_FAST_BITS));
                    uint32_t entry = fast2[idx];
                    int      tb    = (int)((entry >> 8) & 0xFF);
                    if (entry & 0xFF) {
                        d[j]     = (uint8_t)(entry >> 24);
                        d[j + 1] = (uint8_t)(entry >> 16);
                        b->bits <<= tb;
                        b->nbits -= tb;
                        j += 2;
                        continue;
                    }
                    if (tb) {
                        d[j]     = (uint8_t)(entry >> 24);
                        b->bits <<= tb;
                        b->nbits -= tb;
                        j += 1;
                        continue;
                    }
                }
                {
                    uint16_t code = 0;
                    int L;
                    for (L = 1; L <= HUFFMAN_MAX_LEN; ++L) {
                        if (b->nbits < 1) huf_br_refill(b);
                        code = (uint16_t)((code << 1) | (uint16_t)(b->bits >> 63));
                        b->bits <<= 1;
                        b->nbits -= 1;
                        if (bl_count[L] != 0u &&
                            code < (uint16_t)(firstcode[L] + bl_count[L])) {
                            d[j] = sorted_syms[firstsym[L] + (code - firstcode[L])];
                            break;
                        }
                    }
                    if (L > HUFFMAN_MAX_LEN) return TDC_E_CORRUPT;
                    j += 1;
                }
            }
            for (; j < n; ++j) {
                if (b->nbits < HUFFMAN_FAST_BITS) huf_br_refill(b);
                if (b->nbits >= HUFFMAN_FAST_BITS) {
                    uint32_t idx   = (uint32_t)(b->bits >> (64 - HUFFMAN_FAST_BITS));
                    uint16_t entry = fast[idx];
                    int      L     = entry & 0xFF;
                    if (L != 0) {
                        d[j]      = (uint8_t)(entry >> 8);
                        b->bits <<= L;
                        b->nbits -= L;
                        continue;
                    }
                }
                uint16_t code = 0;
                int L;
                for (L = 1; L <= HUFFMAN_MAX_LEN; ++L) {
                    if (b->nbits < 1) huf_br_refill(b);
                    code = (uint16_t)((code << 1) | (uint16_t)(b->bits >> 63));
                    b->bits <<= 1;
                    b->nbits -= 1;
                    if (bl_count[L] != 0u &&
                        code < (uint16_t)(firstcode[L] + bl_count[L])) {
                        d[j] = sorted_syms[firstsym[L] + (code - firstcode[L])];
                        break;
                    }
                }
                if (L > HUFFMAN_MAX_LEN) return TDC_E_CORRUPT;
            }
        }
    }

    return TDC_OK;
}

const tdc_entropy_vt tdc_entropy_huffman4_vt = {
    .id           = TDC_ENTROPY_HUFFMAN4,
    .name         = "huffman4",
    .encode_bound = huffman4_encode_bound,
    .encode       = huffman4_encode,
    .decode       = huffman4_decode,
};
