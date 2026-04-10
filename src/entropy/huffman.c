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
 *     u8[256] code_lengths     (0 means symbol unused; otherwise 1..15)
 *     u8[]    payload          ceil(payload_bits / 8) bytes; bits packed
 *                              MSB-first within each byte
 *
 * Edge cases handled by the format:
 *   - Empty input:  src_size == 0, payload_bits == 0, all lengths zero.
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

#define HUFFMAN_NSYMS    256
#define HUFFMAN_MAX_LEN  15
#define HUFFMAN_HDR_SIZE (4u + 4u + (uint32_t)HUFFMAN_NSYMS)

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

/* ----- Bit writer -------------------------------------------------------- */
/*
 * MSB-first packer: bit i (i = 0..7) within a byte represents 2^(7-i),
 * which matches the canonical-Huffman convention where shorter codes are
 * lexicographically less than longer ones. Writing MSB-first lets the
 * decoder treat each byte as a 0..255 numeric value and shift in.
 *
 * Holds an in-flight byte plus a bit position 0..7. Flushes by writing
 * the partial byte (with trailing zeros) and advancing.
 */

typedef struct {
    uint8_t *p;
    size_t   used;
    uint8_t  cur;
    int      bit;       /* 0 = byte fresh, 8 = byte full */
    uint64_t total_bits;
} bit_writer;

static void bw_init(bit_writer *bw, uint8_t *out) {
    bw->p          = out;
    bw->used       = 0;
    bw->cur        = 0;
    bw->bit        = 0;
    bw->total_bits = 0;
}

static void bw_write_bits(bit_writer *bw, uint16_t code, int nbits) {
    /* Code is right-aligned in `code` (low `nbits` bits). Emit MSB first. */
    bw->total_bits += (uint64_t)nbits;
    for (int i = nbits - 1; i >= 0; --i) {
        bw->cur = (uint8_t)((bw->cur << 1) | ((code >> i) & 1u));
        ++bw->bit;
        if (bw->bit == 8) {
            bw->p[bw->used++] = bw->cur;
            bw->cur = 0;
            bw->bit = 0;
        }
    }
}

static void bw_flush(bit_writer *bw) {
    if (bw->bit > 0) {
        bw->cur = (uint8_t)(bw->cur << (8 - bw->bit));
        bw->p[bw->used++] = bw->cur;
        bw->cur = 0;
        bw->bit = 0;
    }
}

/* ----- Bit reader -------------------------------------------------------- */

typedef struct {
    const uint8_t *p;
    size_t         size;
    size_t         off;
    uint32_t       cur;     /* up to 24 valid bits, MSB-aligned */
    int            nbits;
} bit_reader;

static void br_init(bit_reader *br, const uint8_t *src, size_t size) {
    br->p     = src;
    br->size  = size;
    br->off   = 0;
    br->cur   = 0;
    br->nbits = 0;
}

/* Refill so that at least 16 bits are available (or end-of-stream). */
static inline void br_refill(bit_reader *br) {
    while (br->nbits <= 16 && br->off < br->size) {
        br->cur |= (uint32_t)br->p[br->off++] << (24 - br->nbits);
        br->nbits += 8;
    }
}

/* Pop the top `n` bits (1..16) as an unsigned integer. */
static inline uint16_t br_pop_bits(bit_reader *br, int n) {
    if (br->nbits < n) br_refill(br);
    uint16_t v = (uint16_t)(br->cur >> (32 - n));
    br->cur <<= n;
    br->nbits -= n;
    return v;
}

/* Pop a single bit. Faster path for the per-bit canonical decode loop. */
static inline int br_pop_bit(bit_reader *br) {
    if (br->nbits == 0) br_refill(br);
    int v = (int)(br->cur >> 31);
    br->cur <<= 1;
    --br->nbits;
    return v;
}

/* ----- Encode / decode entry points -------------------------------------- */

static size_t huffman_encode_bound(size_t src_size) {
    /* Header (264 bytes) + worst-case payload (15 bits/symbol) +
     * 1 byte flush slack. 15/8 = 1.875 bytes/byte; round up via /4 + src. */
    return (size_t)HUFFMAN_HDR_SIZE + src_size + (src_size / 2u) + 16u;
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

    /* Write fixed header: src_size, payload_bits placeholder, lengths. */
    uint8_t *out = dst->data;
    tdc_le_store_u32(out + 0, (uint32_t)src_size);
    /* payload_bits is patched after the bit writer reports its total. */
    tdc_le_store_u32(out + 4, 0u);
    memcpy(out + 8, lens, HUFFMAN_NSYMS);

    /* Bit-pack payload. */
    bit_writer bw;
    bw_init(&bw, out + HUFFMAN_HDR_SIZE);
    for (size_t i = 0; i < src_size; ++i) {
        uint8_t s = src[i];
        bw_write_bits(&bw, codes[s], lens[s]);
    }
    uint64_t total_bits = bw.total_bits;
    bw_flush(&bw);

    /* Patch payload_bits and finalize size. */
    if (total_bits > 0xFFFFFFFFu) return TDC_E_INVAL;
    tdc_le_store_u32(out + 4, (uint32_t)total_bits);
    dst->size = (size_t)HUFFMAN_HDR_SIZE + bw.used;
    return TDC_OK;
}

static tdc_status huffman_decode(const uint8_t *src, size_t src_size,
                                 uint8_t       *dst, size_t dst_size) {
    if (src_size < HUFFMAN_HDR_SIZE) return TDC_E_CORRUPT;
    if (dst_size > 0 && !dst) return TDC_E_INVAL;

    uint32_t hdr_src_size   = tdc_le_load_u32(src + 0);
    uint32_t hdr_payload_bits = tdc_le_load_u32(src + 4);
    if ((size_t)hdr_src_size != dst_size) return TDC_E_CORRUPT;

    const uint8_t *lens = src + 8;
    const uint8_t *payload = src + HUFFMAN_HDR_SIZE;
    size_t         payload_bytes_avail = src_size - HUFFMAN_HDR_SIZE;
    size_t         payload_bytes_need  = (size_t)((hdr_payload_bits + 7u) / 8u);
    if (payload_bytes_need > payload_bytes_avail) return TDC_E_CORRUPT;

    /* Empty input. */
    if (dst_size == 0u) {
        if (hdr_payload_bits != 0u) return TDC_E_CORRUPT;
        for (int s = 0; s < HUFFMAN_NSYMS; ++s) {
            if (lens[s] != 0u) return TDC_E_CORRUPT;
        }
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
     * 11 bits → 2 KiB table on the stack — well within any thread's stack
     * margin and small enough to live in L1d. Codes of length 12..15 fall
     * through to the canonical bit-walk (rare on real residual streams). */
    #define HUFFMAN_FAST_BITS 11
    #define HUFFMAN_FAST_SIZE (1u << HUFFMAN_FAST_BITS)
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

    /* Decode loop. Fast path: ensure ≥ FAST_BITS in the accumulator, peek
     * FAST_BITS, look up the symbol/length, advance. Slow path (only hit
     * for codes > FAST_BITS or near end-of-stream when the accumulator
     * can't be refilled to FAST_BITS): the original canonical bit-walk. */
    bit_reader br;
    br_init(&br, payload, payload_bytes_need);

    for (size_t i = 0; i < dst_size; ++i) {
        if (br.nbits < HUFFMAN_FAST_BITS) br_refill(&br);
        if (br.nbits >= HUFFMAN_FAST_BITS) {
            uint32_t idx   = br.cur >> (32 - HUFFMAN_FAST_BITS);
            uint16_t entry = fast[idx];
            int      L     = entry & 0xFF;
            if (L != 0) {
                dst[i]    = (uint8_t)(entry >> 8);
                br.cur  <<= L;
                br.nbits -= L;
                continue;
            }
            /* len 0 ⇒ code is > FAST_BITS, fall through to slow path. */
        }
        /* Slow path: per-bit canonical walk. Reads from the same bit
         * reader, so any bits already consumed by the fast-path peek
         * (none, since peek doesn't advance) are not double-counted. */
        {
            uint16_t code = 0;
            int L;
            for (L = 1; L <= HUFFMAN_MAX_LEN; ++L) {
                code = (uint16_t)((code << 1) | (uint16_t)br_pop_bit(&br));
                if (bl_count[L] != 0u && code < (uint16_t)(firstcode[L] + bl_count[L])) {
                    uint16_t off = (uint16_t)(code - firstcode[L]);
                    dst[i] = sorted_syms[firstsym[L] + off];
                    break;
                }
            }
            if (L > HUFFMAN_MAX_LEN) return TDC_E_CORRUPT;
        }
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
