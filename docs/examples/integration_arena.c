/* docs/examples/integration_arena.c
 *
 * Consumer-side tracking allocator threaded through tdc_buffer::realloc_fn.
 * The arena owns a bump region; tdc_buffer uses it both for the encoded
 * output AND for internal scratch via tdc_decode_block_ex. The allocator
 * records live bytes and peak bytes, mirroring what an embedding consumer
 * (vectra's R arena, a C++ memory pool) usually wants from tdc.
 *
 * Build:
 *   cc -I include docs/examples/integration_arena.c \
 *      build/libtdc.a -lm -o /tmp/integration_arena
 */

#include "tdc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ----- Tracking arena -------------------------------------------------- */
/*
 * Minimal arena: one contiguous slab, bump-allocated. Free is a no-op
 * except for the most recent allocation (classic scratch pattern). Peak
 * bytes and live bytes are tracked so the consumer can report back.
 */

typedef struct {
    uint8_t *slab;
    size_t   capacity;
    size_t   used;
    size_t   peak;
    size_t   live;          /* allocations still pointed at by a user */
    size_t   n_allocs;
    size_t   n_frees;
    size_t   n_reallocs;
    size_t   last_offset;   /* offset of last allocation (for realloc-in-place) */
    size_t   last_size;
} arena_t;

static void arena_init(arena_t *a, size_t capacity) {
    memset(a, 0, sizeof(*a));
    a->slab = (uint8_t *)malloc(capacity);
    a->capacity = capacity;
    a->last_offset = (size_t)-1;
}

static void arena_destroy(arena_t *a) {
    free(a->slab);
    memset(a, 0, sizeof(*a));
}

/*
 * tdc_buffer::realloc_fn contract:
 *   (user, NULL, n)  -> allocate n bytes
 *   (user, p,    0)  -> free p
 *   (user, p,    n)  -> grow p to n bytes; may move
 * Returning NULL signals TDC_E_NOMEM to the library.
 */
static void *arena_realloc(void *user, void *ptr, size_t n) {
    arena_t *a = (arena_t *)user;
    if (n == 0) {
        a->n_frees++;
        /* We can only truly reclaim the most recent allocation. Older
         * allocations become garbage until arena_reset. This mirrors how
         * most downstream consumers implement arenas. */
        if (ptr && ptr == (void *)(a->slab + a->last_offset)) {
            a->used = a->last_offset;
            a->live -= a->last_size;
            a->last_offset = (size_t)-1;
        }
        return NULL;
    }

    if (ptr == NULL) {
        /* Fresh allocation. */
        if (a->used + n > a->capacity) return NULL;
        void *p = a->slab + a->used;
        a->last_offset = a->used;
        a->last_size = n;
        a->used += n;
        a->live += n;
        if (a->used > a->peak) a->peak = a->used;
        a->n_allocs++;
        return p;
    }

    /* Grow-in-place when possible (typical during tdc_buffer growth). */
    a->n_reallocs++;
    if (ptr == (void *)(a->slab + a->last_offset)) {
        if (a->last_offset + n > a->capacity) return NULL;
        a->live += (n > a->last_size) ? (n - a->last_size) : 0;
        a->live -= (n < a->last_size) ? (a->last_size - n) : 0;
        a->used = a->last_offset + n;
        a->last_size = n;
        if (a->used > a->peak) a->peak = a->used;
        return ptr;
    }

    /* Older allocation: copy forward and leak the old slot until reset. */
    if (a->used + n > a->capacity) return NULL;
    void *dst = a->slab + a->used;
    size_t copy_bytes = n; /* we don't know the old size; tdc always shrinks or grows knowing this */
    memcpy(dst, ptr, copy_bytes); /* caller guarantees valid source bytes */
    a->last_offset = a->used;
    a->last_size = n;
    a->used += n;
    a->live += n;
    if (a->used > a->peak) a->peak = a->used;
    return dst;
}

/* ----- Sample workload ------------------------------------------------- */

int main(void) {
    enum { N = 4096 };
    static int32_t src_data[N];
    for (int i = 0; i < N; ++i) src_data[i] = 1000 + i * 3;

    tdc_block src = {0};
    src.data = src_data;
    src.dtype = TDC_DT_I32;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank = 1;
    src.shape.dim[0] = N;
    tdc_shape_set_contiguous(&src.shape);

    tdc_codec_spec spec = {0};
    spec.model = TDC_MODEL_DELTA_1D;
    spec.xform[0] = TDC_XFORM_ZIGZAG;
    spec.xform[1] = TDC_XFORM_BYTE_SHUFFLE;
    spec.entropy[0] = TDC_ENTROPY_LZ;

    arena_t arena;
    arena_init(&arena, 4 * 1024 * 1024); /* 4 MiB slab: encode uses LZ hash table */

    tdc_buffer enc = {0};
    enc.realloc_fn = arena_realloc;
    enc.user = &arena;

    tdc_status st = tdc_encode_block(&src, &spec, &enc);
    if (st != TDC_OK) {
        fprintf(stderr, "encode failed: %s\n", tdc_strerror(st));
        arena_destroy(&arena);
        return 1;
    }

    printf("encode arena: %zu allocs, %zu reallocs, %zu frees\n",
           arena.n_allocs, arena.n_reallocs, arena.n_frees);
    printf("encode arena: live=%zu peak=%zu out_size=%zu\n",
           arena.live, arena.peak, enc.size);

    /* Decode into a freshly-sized destination. */
    tdc_block meta = {0};
    size_t need = 0;
    tdc_decode_peek(enc.data, enc.size, &meta, &need);

    void *dst_data = arena_realloc(&arena, NULL, need);
    tdc_block dst = meta;
    dst.data = dst_data;

    /* Route decode scratch through the same arena. */
    tdc_buffer scratch = {0};
    scratch.realloc_fn = arena_realloc;
    scratch.user = &arena;
    st = tdc_decode_block_ex(enc.data, enc.size, &dst, &scratch);
    if (st != TDC_OK) {
        fprintf(stderr, "decode failed: %s\n", tdc_strerror(st));
        arena_destroy(&arena);
        return 1;
    }

    int ok = memcmp(dst_data, src_data, N * sizeof(int32_t)) == 0;
    printf("decode arena: peak=%zu memcmp=%s\n",
           arena.peak, ok ? "yes" : "NO");

    arena_destroy(&arena);
    return ok ? 0 : 1;
}
