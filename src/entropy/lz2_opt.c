/*
 * src/entropy/lz2_opt.c
 *
 * TDC_ENTROPY_LZ2_OPT — optimal-parsing LZ77 encoder.
 *
 * Emits the SAME on-disk format as TDC_ENTROPY_LZ2 (greedy). The decoder
 * (lz2_decode_core in lz2.c) round-trips both without change — optimal
 * parsing is a pure encode-side optimization.
 *
 * Greedy picks the longest match at each position (locally optimal, but
 * globally suboptimal: a shorter match now can let a much cheaper match
 * start one byte later). Optimal parsing computes the globally minimum
 * cost parse via forward DP over byte positions. Expected gain on
 * structured tabular data: 5–15% smaller output than greedy, at 5–10×
 * the encode time.
 *
 * Phase 1 — static cost model. The literal cost is fixed at 8 bits/byte
 * and the match cost is 24 bits + 8*match_ext(L) bits, matching
 * tdc_lz2_serialize_sequences exactly. Literal-run extension bytes
 * (chained-255 varint, one byte per bracket crossing) are folded into
 * a per-bracket penalty so the sliding-window DP remains additive.
 *
 * Phase 2 (separate commit) will plumb entropy-aware literal costs
 * (per-symbol bit lengths from Huffman/FSE tables) into cost_literal().
 *
 * Structure:
 *   1. Hash-chain match finder (deflate-style, 64K window, depth ≤ 128).
 *   2. Forward DP with sliding-window minima per literal-run bracket.
 *   3. Backtrack → LZ2Seq array, serialized via the shared writer in
 *      lz2.c (tdc_lz2_serialize_sequences).
 */

#include "tdc/entropy.h"
#include "entropy_internal.h"
#include "lz2_internal.h"
#include "../core/buffer.h"

#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

/* ----- Tunable constants ------------------------------------------------- */

#define LZ2_OPT_HASH_BITS   16
#define LZ2_OPT_HASH_SIZE   (1u << LZ2_OPT_HASH_BITS)

/* Chain walk depth limit. 32 matches deflate level 6; it bounds per-position
 * chain-walk cost without losing measurable compression on structured data
 * (long matches surface at the chain front, so deeper walks mostly find
 * dominated alternatives). */
#define LZ2_OPT_MAX_CHAIN   32u

/* Per-candidate extension cap. Each chain candidate is extended by at most
 * this many bytes during the walk. Keeps chain-walk work bounded at
 * O(LZ2_OPT_MAX_CHAIN × LZ2_OPT_EXTEND_CAP) per position regardless of
 * input entropy. On periodic inputs the longest candidate would otherwise
 * extend to end-of-input, giving an O(N²) walk. */
#define LZ2_OPT_EXTEND_CAP  64u

/* Commit-and-skip threshold. If the longest match at position r (after
 * greedy-extending past the cap) reaches this length, we emit the full
 * match as a single DP transition and fast-forward r past the match,
 * skipping the inner L loop and further match-finding inside the match.
 * This matches greedy parsing in long-match regions (preserving the
 * "opt ≤ greedy" size invariant) and eliminates the O(N²) behaviour on
 * repetitive inputs. 32 is comfortably above the leb128 first breakpoint
 * (match_len = 17) so typical structured matches still go through the
 * full DP inner loop. */
#define LZ2_OPT_COMMIT_LEN  256u

/* Literal-run brackets. Bracket b covers lit_run ∈ [start[b], end[b]]
 * inclusive on both ends. Penalty charged at match emission: b bytes
 * (one byte per chained-255 varint position). Covers lit_run up to 1799;
 * longer runs are handled by a "tail" Pareto set that tracks non-dominated
 * (post_match[p], p) pairs among positions aged out of bracket 7, and
 * computes exact cost at use time. Simple-domination test: (pm_a, p_a)
 * dominates (pm_b, p_b) iff pm_a ≤ pm_b AND p_a ≥ p_b. A single-slot
 * tail (min pm only) is insufficient when a later position with slightly
 * larger pm but much larger p gives a cheaper run cost at query time —
 * this is exactly what the "opt ≤ greedy" invariant needs on mixed
 * inputs where greedy finds lucky late matches. */
#define LZ2_OPT_NUM_BRACKETS 8
#define LZ2_OPT_TAIL_CAP     16

static const uint32_t lz2_opt_bracket_start[LZ2_OPT_NUM_BRACKETS] = {
    0u,   15u,  270u, 525u, 780u, 1035u, 1290u, 1545u
};
static const uint32_t lz2_opt_bracket_end[LZ2_OPT_NUM_BRACKETS] = {
    14u,  269u, 524u, 779u, 1034u, 1289u, 1544u, 1799u
};

/* Monotonic deque capacity (per bracket). Max window width is 255
 * (brackets 1..7). 512 gives plenty of headroom and a power-of-two mask. */
#define LZ2_OPT_DEQUE_CAP   512u
#define LZ2_OPT_DEQUE_MASK  (LZ2_OPT_DEQUE_CAP - 1u)

/* "Infinity" marker for unreachable DP states. Costs are in bits and
 * bounded by 8*N where N ≤ 2^32, so INT32_MAX is safe. */
#define LZ2_OPT_INF INT32_MAX

/* ----- Allocation helpers (realloc_fn passthrough) ----------------------- */

static void *lz2_opt_alloc(tdc_buffer *buf, size_t n) {
    return buf->realloc_fn(buf->user, NULL, n);
}

static void lz2_opt_free(tdc_buffer *buf, void *p) {
    if (p) (void)buf->realloc_fn(buf->user, p, 0);
}

/* ----- 4-byte hash (same Fibonacci constant as greedy) ------------------- */

static inline uint32_t lz2_opt_hash4(const uint8_t *p) {
    uint32_t h = ((uint32_t)p[0]) |
                 ((uint32_t)p[1] << 8) |
                 ((uint32_t)p[2] << 16) |
                 ((uint32_t)p[3] << 24);
    return (h * 2654435761u) >> (32u - LZ2_OPT_HASH_BITS);
}

/* ----- Match-length extension (bytes) ------------------------------------ *
 * Mirrors the ml_m3-branch in lz2_seq_encoded_size: ml_m3 >= 15 iff
 * L >= LZ2_MIN_MATCH + 15 = 18. The cost model must match the serializer
 * exactly or the "optimal ≤ greedy" invariant can break. */

static inline uint32_t lz2_opt_match_ext(uint32_t L) {
    if (L < LZ2_MIN_MATCH + 15u) return 0u;
    return lz2_leb128_size(L - LZ2_MIN_MATCH - 15u);
}

/* Literal-run extension bytes, exact. Mirrors the lit_len branch of
 * lz2_seq_encoded_size: 0 bytes for run < 15, else (run-15)/255 + 1. */
static inline uint32_t lz2_opt_lit_ext(uint32_t run) {
    if (run < 15u) return 0u;
    return (run - 15u) / 255u + 1u;
}

/* Extend a candidate at (pos, cand) up to `max_len` bytes. Returns the
 * match length (may be 0). Uses 8-byte compare + CTZ for speed. */
static inline uint32_t lz2_opt_extend(const uint8_t *src, uint32_t pos,
                                      uint32_t cand, uint32_t max_len) {
    uint32_t len = 0u;
    while (len + 8u <= max_len) {
        uint64_t a, b;
        memcpy(&a, src + pos + len, 8);
        memcpy(&b, src + cand + len, 8);
        if (a != b) {
#if defined(__GNUC__) || defined(__clang__)
            len += (uint32_t)(__builtin_ctzll(a ^ b) >> 3);
#elif defined(_MSC_VER)
            unsigned long idx;
            _BitScanForward64(&idx, a ^ b);
            len += (uint32_t)(idx >> 3);
#else
            {
                uint64_t diff = a ^ b;
                while (!(diff & 0xFFu)) { diff >>= 8; len++; }
            }
#endif
            return len;
        }
        len += 8u;
    }
    while (len < max_len && src[pos + len] == src[cand + len]) len++;
    return len;
}

/* ----- Hash-chain match finder ------------------------------------------- *
 *
 * For position `pos`, walk back from chain_head[h(pos)] through
 * chain_prev[] up to LZ2_OPT_MAX_CHAIN candidates, staying within the
 * 64K back-reference window. Each candidate is extended by at most
 * LZ2_OPT_EXTEND_CAP bytes during the walk so per-position work is
 * bounded regardless of input repetitiveness.
 *
 * Returns the longest match found plus the candidate position it came
 * from (so the caller can greedy-extend past the cap when needed to
 * preserve optimality on long matches).
 *
 * The caller inserts `pos` into the chain AFTER the lookup so the chain
 * at `pos` reflects only strictly-earlier positions.
 */
static void lz2_opt_find_longest(const uint8_t *src, uint32_t src_size,
                                 uint32_t pos,
                                 const uint32_t *chain_head,
                                 const uint32_t *chain_prev,
                                 uint32_t *out_len, uint32_t *out_off,
                                 uint32_t *out_cand_pos) {
    *out_len = 0u;
    *out_off = 0u;
    *out_cand_pos = 0u;

    if (pos + LZ2_MIN_MATCH + 1u > src_size) return;

    uint32_t h = lz2_opt_hash4(src + pos);
    uint32_t cand = chain_head[h];
    uint32_t depth = 0u;
    uint32_t min_pos = (pos > LZ2_MAX_OFFSET) ? (pos - LZ2_MAX_OFFSET) : 0u;
    uint32_t best_len = 0u;
    uint32_t best_off = 0u;
    uint32_t best_cand = 0u;
    uint32_t max_remain = src_size - pos;
    uint32_t cap = (max_remain < LZ2_OPT_EXTEND_CAP) ? max_remain : LZ2_OPT_EXTEND_CAP;

    while (cand != 0xFFFFFFFFu && cand < pos && cand >= min_pos &&
           depth < LZ2_OPT_MAX_CHAIN) {

        if (src[cand] == src[pos]) {
            uint32_t len = lz2_opt_extend(src, pos, cand, cap);
            /* If this candidate hit the cap and could exceed the current
             * best, do a full extension. This guarantees we find the true
             * longest match for any cap-hitting candidate (required for
             * the "opt ≤ greedy" invariant — greedy uses uncapped
             * extensions, so we must not under-report match length). */
            if (len == cap && cap < max_remain && len >= best_len) {
                len += lz2_opt_extend(src, pos + cap, cand + cap,
                                      max_remain - cap);
            }
            if (len >= LZ2_MIN_MATCH && len > best_len) {
                best_len = len;
                best_off = pos - cand;
                best_cand = cand;
            }
        }

        cand = chain_prev[cand];
        depth++;
    }

    *out_len = best_len;
    *out_off = best_off;
    *out_cand_pos = best_cand;
}

/* ----- Monotonic deque (sliding-window min) ------------------------------ *
 *
 * Each bracket owns one deque tracking (position, g) pairs where
 * g(p) = post_match[p] - 8*p. The deque is kept monotonically
 * increasing in g: front holds the minimum in the current window.
 *
 * Circular buffer with power-of-two capacity. head == tail means empty.
 */
typedef struct {
    uint32_t pos[LZ2_OPT_DEQUE_CAP];
    int32_t  g[LZ2_OPT_DEQUE_CAP];
    uint32_t head;
    uint32_t tail;
} Lz2OptDeque;

static inline void deque_init(Lz2OptDeque *q) {
    q->head = 0u;
    q->tail = 0u;
}

static inline int deque_empty(const Lz2OptDeque *q) {
    return q->head == q->tail;
}

static inline int32_t deque_front_g(const Lz2OptDeque *q) {
    return q->g[q->head];
}

static inline uint32_t deque_front_pos(const Lz2OptDeque *q) {
    return q->pos[q->head];
}

/* Push (pos, g) at the back. Drops any tail entries with g >= new g
 * (those positions are dominated and will never be the window min). */
static inline void deque_push_back(Lz2OptDeque *q, uint32_t pos, int32_t g) {
    while (q->head != q->tail) {
        uint32_t t = (q->tail - 1u) & LZ2_OPT_DEQUE_MASK;
        if (q->g[t] < g) break;
        q->tail = t;
    }
    q->pos[q->tail] = pos;
    q->g[q->tail] = g;
    q->tail = (q->tail + 1u) & LZ2_OPT_DEQUE_MASK;
}

/* If the front position equals `pos`, pop it. Called when a position
 * slides out of a bracket's window on the left. No-op if the front
 * entry was already dominated out by a later push_back. */
static inline void deque_pop_front_if(Lz2OptDeque *q, uint32_t pos) {
    if (q->head != q->tail && q->pos[q->head] == pos) {
        q->head = (q->head + 1u) & LZ2_OPT_DEQUE_MASK;
    }
}

/* ----- Backtrack entry --------------------------------------------------- */

typedef struct {
    uint32_t prev_p;   /* source post-match state for this transition */
    uint32_t r_start;  /* match START position (prev_p + lit_run) */
    uint32_t L;        /* match length */
    uint32_t off;      /* back-reference offset */
} Lz2OptBacktrack;

/* ----- Forward DP -------------------------------------------------------- *
 *
 * State:
 *   post_match[p] = min cost (in bits) of encoding src[0..p) such that
 *                   position p is either 0 or immediately after a
 *                   completed match.
 *   best_start[r] = min cost of reaching r as a match-start position,
 *                   = min over p of (post_match[p] + 8*(r-p) +
 *                                     8*bracket(r-p))
 *                   Computed via sliding-window minima of (post_match[p]
 *                   - 8*p) per bracket.
 *
 * Transitions:
 *   For each r with matches[r].len >= LZ2_MIN_MATCH:
 *     For each L in [LZ2_MIN_MATCH, matches[r].len]:
 *       candidate = best_start[r] + 24 + 8*match_ext(L)
 *       post_match[r + L] = min(., candidate)
 *
 * Final:
 *   answer = min over p of (post_match[p] + 8*(N - p))
 *            -- trailing literals, no header.
 */
static tdc_status lz2_opt_parse_and_serialize(
    const uint8_t *src, uint32_t src_size, tdc_buffer *dst)
{
    if (src_size < LZ2_MIN_MATCH + 1u) {
        /* Below minimum: serializer falls back to literal-only stream. */
        return tdc_lz2_serialize_sequences(src, src_size, NULL, 0u, dst);
    }

    uint32_t N = src_size;

    /* ---- Allocate DP state and match-finder scratch ---- */
    int32_t *post_match = (int32_t *)lz2_opt_alloc(dst, (size_t)(N + 1u) * sizeof(int32_t));
    Lz2OptBacktrack *bt = (Lz2OptBacktrack *)lz2_opt_alloc(dst,
        (size_t)(N + 1u) * sizeof(Lz2OptBacktrack));
    uint32_t *chain_head = (uint32_t *)lz2_opt_alloc(dst, LZ2_OPT_HASH_SIZE * sizeof(uint32_t));
    uint32_t *chain_prev = (uint32_t *)lz2_opt_alloc(dst, (size_t)N * sizeof(uint32_t));

    if (!post_match || !bt || !chain_head || !chain_prev) {
        lz2_opt_free(dst, post_match);
        lz2_opt_free(dst, bt);
        lz2_opt_free(dst, chain_head);
        lz2_opt_free(dst, chain_prev);
        return TDC_E_NOMEM;
    }

    for (uint32_t i = 0u; i <= N; i++) post_match[i] = LZ2_OPT_INF;
    post_match[0] = 0;
    memset(bt, 0, (size_t)(N + 1u) * sizeof(Lz2OptBacktrack));
    memset(chain_head, 0xFF, LZ2_OPT_HASH_SIZE * sizeof(uint32_t));
    memset(chain_prev, 0xFF, (size_t)N * sizeof(uint32_t));

    /* Sliding-window deques, one per bracket. Stack-allocated (~32 KB). */
    Lz2OptDeque deques[LZ2_OPT_NUM_BRACKETS];
    for (int i = 0; i < LZ2_OPT_NUM_BRACKETS; i++) deque_init(&deques[i]);

    /* Tail Pareto set: non-dominated (pm, p) pairs among positions aged
     * out of bracket 7. Ordered by p (ascending = age-out order).
     * Invariant: pm is strictly decreasing from front to back, so no
     * entry is simply-dominated by any other. At query time, evaluate
     * all entries and take the min cost. */
    int32_t  tail_pm[LZ2_OPT_TAIL_CAP];
    uint32_t tail_p[LZ2_OPT_TAIL_CAP];
    uint32_t tail_size = 0u;

    /* Skip-ahead counter: when a long match is committed at position r,
     * positions (r, r + match_len) are fast-forwarded through the DP —
     * deque slide and chain insertion still run, but match finding and
     * the inner L loop are bypassed. */
    uint32_t skip_until = 0u;

    /* ---- Main DP loop ---- */
    for (uint32_t r = 0u; r < N; r++) {
        /* Step A: slide each bracket's window by one. For bracket b,
         * the window covers lit_run ∈ [bracket_start[b], bracket_end[b]],
         * i.e. p ∈ [r - bracket_end[b], r - bracket_start[b]]. As r
         * advances by 1, we add p = r - bracket_start[b] on the right
         * and pop p = r - bracket_end[b] - 1 from the left. The position
         * popped from the oldest bracket becomes a tail candidate. */
        for (int b = 0; b < LZ2_OPT_NUM_BRACKETS; b++) {
            uint32_t bs = lz2_opt_bracket_start[b];
            uint32_t be = lz2_opt_bracket_end[b];

            /* Enter: new right edge. */
            if (r >= bs) {
                uint32_t p_enter = r - bs;
                if (post_match[p_enter] != LZ2_OPT_INF) {
                    int32_t g = post_match[p_enter] - (int32_t)(8u * p_enter);
                    deque_push_back(&deques[b], p_enter, g);
                }
            }
            /* Exit: old left edge leaves the window. */
            if (r > be) {
                uint32_t p_exit = r - be - 1u;
                deque_pop_front_if(&deques[b], p_exit);
                /* Position p_exit just aged out of the oldest bracket
                 * into the tail Pareto set. p_exit is strictly larger
                 * than any previously-aged position (FIFO). Simple
                 * domination: new (pm, p_exit) with p_exit > all existing
                 * p's dominates any existing entry whose pm ≥ new pm.
                 * Pop back those, then push new if not dominated by the
                 * remaining front (which has smaller p; it dominates new
                 * only if front pm ≤ new pm, but by invariant front pm
                 * is the LARGEST in the set, so we only skip the push
                 * if all existing pm ≤ new pm, i.e. back pm ≤ new pm). */
                if (b == LZ2_OPT_NUM_BRACKETS - 1 &&
                    post_match[p_exit] != LZ2_OPT_INF) {
                    int32_t new_pm = post_match[p_exit];
                    while (tail_size > 0u &&
                           tail_pm[tail_size - 1u] >= new_pm) {
                        tail_size--;
                    }
                    /* After pop-back, surviving entries all have pm < new_pm.
                     * Append new_pm unless the capacity is exhausted (drop
                     * the oldest front entry to make room — it has the
                     * largest pm and is least likely to win). */
                    if (tail_size == LZ2_OPT_TAIL_CAP) {
                        for (uint32_t i = 1u; i < tail_size; i++) {
                            tail_pm[i - 1u] = tail_pm[i];
                            tail_p[i - 1u]  = tail_p[i];
                        }
                        tail_size--;
                    }
                    tail_pm[tail_size] = new_pm;
                    tail_p[tail_size]  = p_exit;
                    tail_size++;
                }
            }
        }

        /* Step B: compute best_start[r] = min over brackets of
         *   (front g + 8*r + 8*b)  where b is the bracket index. */
        int32_t best_start = LZ2_OPT_INF;
        uint32_t best_start_prev_p = 0u;
        for (int b = 0; b < LZ2_OPT_NUM_BRACKETS; b++) {
            if (deque_empty(&deques[b])) continue;
            int32_t g_min = deque_front_g(&deques[b]);
            int32_t cand = g_min + (int32_t)(8u * r) + (int32_t)(8u * (uint32_t)b);
            if (cand < best_start) {
                best_start = cand;
                best_start_prev_p = deque_front_pos(&deques[b]);
            }
        }

        /* Tail candidates: positions aged out of bracket 7 can still reach r
         * through a long literal run. Cost = post_match[p] + 8*(r-p) +
         * 8*lit_ext(r-p), computed exactly at use time. Iterate the full
         * Pareto set — a single-slot tail (min pm only) misses the case
         * where a larger-p entry with slightly larger pm gives cheaper
         * total cost at specific r values. The tail is capped at
         * LZ2_OPT_TAIL_CAP entries so this loop is O(1) per r. */
        for (uint32_t ti = 0u; ti < tail_size; ti++) {
            if (tail_p[ti] >= r) continue;
            uint32_t run = r - tail_p[ti];
            int32_t cand = tail_pm[ti] + (int32_t)(8u * run) +
                           (int32_t)(8u * lz2_opt_lit_ext(run));
            if (cand < best_start) {
                best_start = cand;
                best_start_prev_p = tail_p[ti];
            }
        }

        /* Fast-forward through positions inside a previously committed
         * long match — still insert into the chain so later positions can
         * find matches into the skipped range, but skip match finding and
         * the inner L loop. */
        if (r < skip_until) {
            if (r + LZ2_MIN_MATCH + 1u <= src_size) {
                uint32_t hs = lz2_opt_hash4(src + r);
                chain_prev[r] = chain_head[hs];
                chain_head[hs] = r;
            }
            continue;
        }

        if (best_start >= LZ2_OPT_INF) {
            /* No reachable predecessor; still insert into the chain. */
            if (r + LZ2_MIN_MATCH + 1u <= src_size) {
                uint32_t hs = lz2_opt_hash4(src + r);
                chain_prev[r] = chain_head[hs];
                chain_head[hs] = r;
            }
            continue;
        }

        /* Step C: find matches at r (before inserting r itself into the
         * chain, so the walk starts from strictly-earlier positions). */
        uint32_t match_len = 0u, match_off = 0u, cand_pos = 0u;
        lz2_opt_find_longest(src, src_size, r, chain_head, chain_prev,
                             &match_len, &match_off, &cand_pos);

        /* Insert r into the chain after the lookup. */
        if (r + LZ2_MIN_MATCH + 1u <= src_size) {
            uint32_t hs = lz2_opt_hash4(src + r);
            chain_prev[r] = chain_head[hs];
            chain_head[hs] = r;
        }

        if (match_len < LZ2_MIN_MATCH) continue;
        (void)cand_pos; /* full-extension handled inside find_longest */

        /* Commit-and-skip path: on long matches, emit a single transition
         * (matching greedy's behaviour) and fast-forward past the match.
         * The next non-skipped r is at r + match_len. */
        if (match_len >= LZ2_OPT_COMMIT_LEN) {
            uint32_t ext = lz2_opt_match_ext(match_len);
            int32_t cand = best_start + (int32_t)24u + (int32_t)(8u * ext);
            uint32_t end = r + match_len;
            if (cand < post_match[end]) {
                post_match[end] = cand;
                bt[end].prev_p = best_start_prev_p;
                bt[end].r_start = r;
                bt[end].L = match_len;
                bt[end].off = match_off;
            }
            skip_until = end;
            continue;
        }

        /* Step D: explore every L in [LZ2_MIN_MATCH, match_len]. For
         * short-to-moderate matches this is bounded by LZ2_OPT_COMMIT_LEN
         * so the inner loop is O(1) per r. */
        for (uint32_t L = LZ2_MIN_MATCH; L <= match_len; L++) {
            uint32_t ext = lz2_opt_match_ext(L);
            int32_t cand = best_start + (int32_t)24u + (int32_t)(8u * ext);
            uint32_t end = r + L;
            if (cand < post_match[end]) {
                post_match[end] = cand;
                bt[end].prev_p = best_start_prev_p;
                bt[end].r_start = r;
                bt[end].L = L;
                bt[end].off = match_off;
            }
        }
    }

    /* ---- Final: best terminal (trailing literals, no header) ---- */
    uint32_t best_final_p = 0u;
    int32_t  best_final_cost = (int32_t)(8u * N); /* all-literal path */
    for (uint32_t p = 1u; p <= N; p++) {
        if (post_match[p] == LZ2_OPT_INF) continue;
        int32_t c = post_match[p] + (int32_t)(8u * (N - p));
        if (c < best_final_cost) {
            best_final_cost = c;
            best_final_p = p;
        }
    }

    /* ---- Backtrack: reconstruct LZ2Seq array ---- */
    uint32_t seq_count = 0u;
    {
        uint32_t cur = best_final_p;
        while (cur > 0u) {
            seq_count++;
            cur = bt[cur].prev_p;
        }
    }

    LZ2Seq *seqs = NULL;
    if (seq_count > 0u) {
        seqs = (LZ2Seq *)lz2_opt_alloc(dst, (size_t)seq_count * sizeof(LZ2Seq));
        if (!seqs) {
            lz2_opt_free(dst, post_match);
            lz2_opt_free(dst, bt);
            lz2_opt_free(dst, chain_head);
            lz2_opt_free(dst, chain_prev);
            return TDC_E_NOMEM;
        }
        uint32_t cur = best_final_p;
        uint32_t idx = seq_count;
        while (cur > 0u) {
            idx--;
            uint32_t prev_p = bt[cur].prev_p;
            seqs[idx].lit_len   = bt[cur].r_start - prev_p;
            seqs[idx].match_len = bt[cur].L;
            seqs[idx].match_off = bt[cur].off;
            cur = prev_p;
        }
    }

    /* ---- Free DP scratch before serializing ---- */
    lz2_opt_free(dst, post_match);
    lz2_opt_free(dst, bt);
    lz2_opt_free(dst, chain_head);
    lz2_opt_free(dst, chain_prev);

    /* ---- Serialize via shared writer in lz2.c ---- */
    tdc_status st = tdc_lz2_serialize_sequences(src, src_size,
                                                seqs, seq_count, dst);
    lz2_opt_free(dst, seqs);
    return st;
}

/* ----- vtable wiring ----------------------------------------------------- */

static size_t lz2_opt_encode_bound(size_t src_size) {
    /* Worst case is the literal-only fallback stream. */
    return src_size + LZ2_HEADER_SIZE;
}

static tdc_status lz2_opt_encode(const uint8_t *src, size_t src_size,
                                 const void *params, tdc_buffer *dst) {
    (void)params;
    if (!dst || !dst->realloc_fn) return TDC_E_INVAL;
    if (src_size > UINT32_MAX) return TDC_E_INVAL;
    if (src_size > 0 && !src) return TDC_E_INVAL;
    return lz2_opt_parse_and_serialize(src, (uint32_t)src_size, dst);
}

/* Decode is shared with TDC_ENTROPY_LZ2 — lz2.c's lz2_decode_core handles
 * both greedy- and optimal-parsed streams identically (same on-disk
 * format). We re-use the greedy vtable's decode function pointer via
 * registry.c's dispatch, so we forward to the same symbol here. */
extern const tdc_entropy_vt tdc_entropy_lz2_vt;

static tdc_status lz2_opt_decode(const uint8_t *src, size_t src_size,
                                 uint8_t *dst, size_t dst_size) {
    return tdc_entropy_lz2_vt.decode(src, src_size, dst, dst_size);
}

const tdc_entropy_vt tdc_entropy_lz2_opt_vt = {
    .id           = TDC_ENTROPY_LZ2_OPT,
    .name         = "lz2_opt",
    .encode_bound = lz2_opt_encode_bound,
    .encode       = lz2_opt_encode,
    .decode       = lz2_opt_decode,
};
