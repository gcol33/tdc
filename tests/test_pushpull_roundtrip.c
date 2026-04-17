/*
 * tests/test_pushpull_roundtrip.c
 *
 * Round-trip tests for the byte-stream push/pull API (tdc/pushpull.h).
 *
 * Covers:
 *   1. Big f64 vector (~10 MiB) pushed in 17 jittery chunks, decoded via
 *      23 jittery record-chunk pushes; verified memcmp. Run across 3
 *      pipelines (RAW+LZ, RAW+BSHUF+LZ, DICT_NUMERIC+BSHUF+LZ_STREAMS).
 *   2. Empty stream: open, finish, decode, expect *done = 1 first call.
 *   3. One-byte-at-a-time push and pull on both sides.
 *   4. Mid-stream flush: verify that a flush produces a short block that
 *      round-trips correctly.
 *   5. Checkpoint parity: encoder checkpoint after N blocks matches
 *      decoder checkpoint after pulling N blocks.
 */

#ifdef _MSC_VER
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "tdc/pushpull.h"
#include "tdc/codec.h"
#include "tdc/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ===== Test allocator ==================================================== */

static void *test_realloc(void *user, void *ptr, size_t new_size) {
    (void)user;
    if (new_size == 0) { free(ptr); return NULL; }
    return realloc(ptr, new_size);
}

/* ===== Assertion macro =================================================== */

#define ASSERT_OR_DIE(cond, msg) do {                                       \
    if (!(cond)) {                                                          \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);   \
        return 1;                                                           \
    }                                                                       \
} while (0)

/* ===== Byte sink that grows via realloc =================================== */

typedef struct {
    uint8_t *data;
    size_t   size;
    size_t   cap;
} byte_sink;

static int sink_append(byte_sink *s, const uint8_t *p, size_t n) {
    if (s->size + n > s->cap) {
        size_t nc = s->cap ? s->cap : 64;
        while (nc < s->size + n) nc *= 2;
        void *q = realloc(s->data, nc);
        if (!q) return 0;
        s->data = (uint8_t *)q;
        s->cap  = nc;
    }
    memcpy(s->data + s->size, p, n);
    s->size += n;
    return 1;
}

static void sink_free(byte_sink *s) {
    free(s->data);
    s->data = NULL;
    s->size = 0;
    s->cap  = 0;
}

/* ===== Drain helper ======================================================= */

/* Drain all currently-available encoder output into the sink. */
static int drain_encoder(tdc_pushpull_encoder *enc, byte_sink *sink) {
    uint8_t buf[4096];
    for (;;) {
        size_t got = 0;
        int more = 0;
        tdc_status st = tdc_pushpull_encoder_pull(enc, buf, sizeof(buf),
                                                  &got, &more);
        if (st != TDC_OK) return 0;
        if (got > 0) {
            if (!sink_append(sink, buf, got)) return 0;
        }
        if (!more) break;
    }
    return 1;
}

/* ===== Deterministic "jitter" chunk builder ============================== */

static size_t jitter_sized(uint64_t *s, size_t avg) {
    /* LCG step. */
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    uint64_t r = (*s >> 32);
    if (avg < 2) return avg;
    uint64_t jit = r % (avg / 2 + 1);
    int sign = (int)((r >> 3) & 1) ? 1 : -1;
    long adjusted = (long)avg + sign * (long)jit;
    if (adjusted < 1) adjusted = 1;
    return (size_t)adjusted;
}

/* Split [total] into [n_chunks] deterministically with jittered sizes.
 * Last chunk absorbs the remainder. */
static void build_chunks(size_t total, size_t n_chunks, uint64_t seed,
                         size_t *out) {
    size_t remaining = total;
    size_t avg = total / n_chunks;
    for (size_t i = 0; i + 1 < n_chunks; ++i) {
        size_t want = jitter_sized(&seed, avg);
        size_t max_take = remaining - (n_chunks - 1 - i);
        if (want > max_take) want = max_take ? max_take : 1;
        if (want == 0) want = 1;
        out[i] = want;
        remaining -= want;
    }
    out[n_chunks - 1] = remaining;
}

/* ===== Element-alignment shim ============================================ */
/*
 * jitter chunks may not be multiples of elem_size. The encoder requires
 * whole elements per push. We stash a tiny carry and only push multiples
 * of elem_size.
 */

typedef struct {
    uint8_t hold[8];
    size_t  hold_n;
    size_t  elem_size;
} push_aligner;

static tdc_status aligned_push(push_aligner *a,
                               tdc_pushpull_encoder *enc,
                               const uint8_t *data, size_t n) {
    size_t total_avail = a->hold_n + n;
    size_t whole = total_avail - (total_avail % a->elem_size);

    if (whole == 0) {
        if (a->hold_n + n > sizeof(a->hold)) return TDC_E_INVAL;
        memcpy(a->hold + a->hold_n, data, n);
        a->hold_n += n;
        return TDC_OK;
    }

    /* Push the hold prefix + enough of data to reach the next element
     * boundary, then push the rest of the whole chunk directly. */
    if (a->hold_n > 0) {
        size_t first_extra = a->elem_size - a->hold_n;
        uint8_t tmp[16];
        if (a->hold_n + first_extra > sizeof(tmp)) return TDC_E_INVAL;
        memcpy(tmp, a->hold, a->hold_n);
        memcpy(tmp + a->hold_n, data, first_extra);
        tdc_status st = tdc_pushpull_encoder_push(enc, tmp,
                                                   a->hold_n + first_extra);
        if (st != TDC_OK) return st;
        data += first_extra;
        n    -= first_extra;
        a->hold_n = 0;
        whole -= a->elem_size;
    }

    /* Whole bytes left to push from data. */
    size_t tail = n % a->elem_size;
    size_t push_n = n - tail;
    if (push_n > 0) {
        tdc_status st = tdc_pushpull_encoder_push(enc, data, push_n);
        if (st != TDC_OK) return st;
    }
    if (tail > 0) {
        if (tail > sizeof(a->hold)) return TDC_E_INVAL;
        memcpy(a->hold, data + push_n, tail);
        a->hold_n = tail;
    }
    return TDC_OK;
}

/* ===== Roundtrip for a given spec ========================================= */

typedef struct {
    const char    *name;
    tdc_codec_spec spec;
} pipeline;

static int run_big_roundtrip(const pipeline *p,
                             const uint8_t *orig, size_t n_bytes,
                             uint64_t seed_push, uint64_t seed_record,
                             size_t n_push_chunks, size_t n_record_chunks,
                             uint64_t target_block_elems) {
    printf("  pipeline=%s target_elems=%llu\n",
           p->name, (unsigned long long)target_block_elems);

    byte_sink encoded = {0};
    tdc_status st;

    /* ---- Encode ---------------------------------------------------- */
    {
        tdc_pushpull_encoder_config cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.dtype  = TDC_DT_F64;
        cfg.layout = TDC_LAYOUT_VECTOR_1D;
        cfg.shape_template.rank   = 1;
        cfg.shape_template.dim[0] = 0;
        cfg.spec                  = p->spec;
        cfg.target_block_elems    = target_block_elems;
        cfg.realloc_fn            = test_realloc;
        cfg.alloc_user            = NULL;

        tdc_pushpull_encoder *enc = NULL;
        st = tdc_pushpull_encoder_open(&cfg, &enc);
        ASSERT_OR_DIE(st == TDC_OK, "encoder open");

        size_t *chunks = (size_t *)malloc(n_push_chunks * sizeof(size_t));
        ASSERT_OR_DIE(chunks, "alloc push chunks");
        build_chunks(n_bytes, n_push_chunks, seed_push, chunks);

        push_aligner aligner = {0};
        aligner.elem_size = sizeof(double);

        size_t offset = 0;
        for (size_t i = 0; i < n_push_chunks; ++i) {
            st = aligned_push(&aligner, enc, orig + offset, chunks[i]);
            ASSERT_OR_DIE(st == TDC_OK, "encoder push (jittered)");
            offset += chunks[i];
            /* Intermittently drain so out buffer doesn't grow huge. */
            if ((i & 3) == 0) {
                ASSERT_OR_DIE(drain_encoder(enc, &encoded), "drain mid");
            }
        }
        ASSERT_OR_DIE(aligner.hold_n == 0, "push_aligner clean at finish");

        st = tdc_pushpull_encoder_finish(enc);
        ASSERT_OR_DIE(st == TDC_OK, "encoder finish");

        ASSERT_OR_DIE(drain_encoder(enc, &encoded), "drain final");

        double junk = 1.0;
        st = tdc_pushpull_encoder_push(enc, &junk, sizeof(junk));
        ASSERT_OR_DIE(st == TDC_E_INVAL, "push after finish rejected");

        tdc_pushpull_encoder_close(&enc);
        ASSERT_OR_DIE(enc == NULL, "encoder nulled after close");
        free(chunks);
    }

    ASSERT_OR_DIE(encoded.size > 0, "encoded non-empty");

    /* ---- Decode ---------------------------------------------------- */
    uint8_t *recon = (uint8_t *)malloc(n_bytes);
    ASSERT_OR_DIE(recon, "alloc recon");
    size_t recon_off = 0;

    {
        tdc_pushpull_decoder_config cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.realloc_fn = test_realloc;
        cfg.alloc_user = NULL;

        tdc_pushpull_decoder *dec = NULL;
        st = tdc_pushpull_decoder_open(&cfg, &dec);
        ASSERT_OR_DIE(st == TDC_OK, "decoder open");

        size_t *rchunks = (size_t *)malloc(n_record_chunks * sizeof(size_t));
        ASSERT_OR_DIE(rchunks, "alloc record chunks");
        build_chunks(encoded.size, n_record_chunks, seed_record, rchunks);

        size_t off = 0;
        for (size_t i = 0; i < n_record_chunks; ++i) {
            size_t consumed = 0;
            st = tdc_pushpull_decoder_push(dec, encoded.data + off,
                                           rchunks[i], &consumed);
            ASSERT_OR_DIE(st == TDC_OK, "decoder push");
            ASSERT_OR_DIE(consumed == rchunks[i], "decoder consumed all");
            off += rchunks[i];

            for (;;) {
                tdc_block blk;
                int have = 0, done = 0;
                st = tdc_pushpull_decoder_next(dec, &blk, &have, &done);
                ASSERT_OR_DIE(st == TDC_OK, "decoder next (mid)");
                if (!have) break;
                size_t nelems = (size_t)blk.shape.dim[0];
                size_t nb = nelems * sizeof(double);
                ASSERT_OR_DIE(recon_off + nb <= n_bytes, "recon capacity");
                memcpy(recon + recon_off, blk.data, nb);
                recon_off += nb;
            }
        }

        st = tdc_pushpull_decoder_finish(dec);
        ASSERT_OR_DIE(st == TDC_OK, "decoder finish");

        for (;;) {
            tdc_block blk;
            int have = 0, done = 0;
            st = tdc_pushpull_decoder_next(dec, &blk, &have, &done);
            ASSERT_OR_DIE(st == TDC_OK, "decoder next (final)");
            if (have) {
                size_t nelems = (size_t)blk.shape.dim[0];
                size_t nb = nelems * sizeof(double);
                ASSERT_OR_DIE(recon_off + nb <= n_bytes, "recon capacity tail");
                memcpy(recon + recon_off, blk.data, nb);
                recon_off += nb;
                continue;
            }
            if (done) break;
            ASSERT_OR_DIE(0, "decoder starved after finish");
        }

        tdc_pushpull_decoder_close(&dec);
        ASSERT_OR_DIE(dec == NULL, "decoder nulled after close");
        free(rchunks);
    }

    ASSERT_OR_DIE(recon_off == n_bytes, "recon size matches");
    ASSERT_OR_DIE(memcmp(recon, orig, n_bytes) == 0, "memcmp recon vs orig");

    free(recon);
    sink_free(&encoded);
    return 0;
}

/* ===== Pipeline specs ===================================================== */

static tdc_codec_spec spec_raw_lz(void) {
    tdc_codec_spec s = {0};
    s.model      = TDC_MODEL_RAW;
    s.entropy[0] = TDC_ENTROPY_LZ;
    return s;
}

/* RAW model + BYTE_SHUFFLE + LZ. DELTA1D is integer-only, so we substitute
 * a representative multi-transform pipeline that works on f64. */
static tdc_codec_spec spec_bshuf_lz(void) {
    tdc_codec_spec s = {0};
    s.model      = TDC_MODEL_RAW;
    s.xform[0]   = TDC_XFORM_BYTE_SHUFFLE;
    s.entropy[0] = TDC_ENTROPY_LZ;
    return s;
}

static tdc_codec_spec spec_dict_bshuf_lz_streams(void) {
    tdc_codec_spec s = {0};
    s.model      = TDC_MODEL_DICT_NUMERIC_1D;
    s.xform[0]   = TDC_XFORM_BYTE_SHUFFLE;
    s.entropy[0] = TDC_ENTROPY_LZ_STREAMS;
    return s;
}

/* ===== Test data ========================================================== */

static double *alloc_test_vector_f64(size_t n, unsigned kind) {
    double *v = (double *)malloc(n * sizeof(double));
    if (!v) return NULL;
    if (kind == 0) {
        for (size_t i = 0; i < n; ++i) {
            v[i] = 100.0 + (double)i * 0.01 + sin((double)i * 0.001) * 3.0;
        }
    } else if (kind == 1) {
        static const double vals[32] = {
            1.0, 2.0, 3.5, 4.25, 5.5, 7.0, 11.0, 13.0,
            17.0, 19.0, 23.0, 29.0, 31.0, 37.0, 41.0, 43.0,
            47.0, 53.0, 59.0, 61.0, 67.0, 71.0, 73.0, 79.0,
            83.0, 89.0, 97.0, 101.0, 103.0, 107.0, 109.0, 113.0
        };
        for (size_t i = 0; i < n; ++i) {
            v[i] = vals[(i * 7 + 3) & 31];
        }
    } else {
        uint64_t s = 0xc0ffee123456789full;
        for (size_t i = 0; i < n; ++i) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            union { uint64_t u; double d; } x;
            x.u = s;
            v[i] = x.d;
        }
    }
    return v;
}

/* ===== Edge-case tests ==================================================== */

static int test_empty_stream(void) {
    printf("test_empty_stream\n");
    tdc_pushpull_encoder_config ecfg;
    memset(&ecfg, 0, sizeof(ecfg));
    ecfg.dtype  = TDC_DT_F64;
    ecfg.layout = TDC_LAYOUT_VECTOR_1D;
    ecfg.shape_template.rank   = 1;
    ecfg.spec.model            = TDC_MODEL_RAW;
    ecfg.spec.entropy[0]       = TDC_ENTROPY_LZ;
    ecfg.target_block_elems    = 1024;
    ecfg.realloc_fn            = test_realloc;

    tdc_pushpull_encoder *enc = NULL;
    ASSERT_OR_DIE(tdc_pushpull_encoder_open(&ecfg, &enc) == TDC_OK, "enc open");

    byte_sink sink = {0};
    ASSERT_OR_DIE(tdc_pushpull_encoder_finish(enc) == TDC_OK, "enc finish");
    ASSERT_OR_DIE(drain_encoder(enc, &sink), "drain empty");

    ASSERT_OR_DIE(sink.size == 0, "empty stream emits 0 bytes");
    tdc_pushpull_encoder_close(&enc);

    tdc_pushpull_decoder_config dcfg;
    memset(&dcfg, 0, sizeof(dcfg));
    dcfg.realloc_fn = test_realloc;

    tdc_pushpull_decoder *dec = NULL;
    ASSERT_OR_DIE(tdc_pushpull_decoder_open(&dcfg, &dec) == TDC_OK, "dec open");
    ASSERT_OR_DIE(tdc_pushpull_decoder_finish(dec) == TDC_OK, "dec finish");

    tdc_block blk;
    int have = 0, done = 0;
    ASSERT_OR_DIE(tdc_pushpull_decoder_next(dec, &blk, &have, &done) == TDC_OK,
                  "dec next empty");
    ASSERT_OR_DIE(!have, "no block on empty");
    ASSERT_OR_DIE(done,  "done set on empty");

    tdc_pushpull_decoder_close(&dec);
    sink_free(&sink);
    return 0;
}

static int test_byte_at_a_time(void) {
    printf("test_byte_at_a_time\n");
    size_t n = 1024;
    double *src = alloc_test_vector_f64(n, 0);
    ASSERT_OR_DIE(src, "alloc");
    size_t n_bytes = n * sizeof(double);

    tdc_pushpull_encoder_config ecfg;
    memset(&ecfg, 0, sizeof(ecfg));
    ecfg.dtype  = TDC_DT_F64;
    ecfg.layout = TDC_LAYOUT_VECTOR_1D;
    ecfg.shape_template.rank   = 1;
    ecfg.spec.model            = TDC_MODEL_RAW;
    ecfg.spec.entropy[0]       = TDC_ENTROPY_LZ;
    ecfg.target_block_elems    = 128;
    ecfg.realloc_fn            = test_realloc;

    tdc_pushpull_encoder *enc = NULL;
    ASSERT_OR_DIE(tdc_pushpull_encoder_open(&ecfg, &enc) == TDC_OK, "enc open");

    /* Push 8 bytes at a time (a single f64). */
    const uint8_t *bp = (const uint8_t *)src;
    for (size_t i = 0; i < n_bytes; i += sizeof(double)) {
        ASSERT_OR_DIE(tdc_pushpull_encoder_push(enc, bp + i, sizeof(double))
                          == TDC_OK, "per-elem push");
    }
    ASSERT_OR_DIE(tdc_pushpull_encoder_finish(enc) == TDC_OK, "enc finish");

    byte_sink sink = {0};
    for (;;) {
        uint8_t b;
        size_t got = 0;
        int more = 0;
        tdc_status st = tdc_pushpull_encoder_pull(enc, &b, 1, &got, &more);
        ASSERT_OR_DIE(st == TDC_OK, "1-byte pull");
        if (got == 0 && !more) break;
        if (got) ASSERT_OR_DIE(sink_append(&sink, &b, 1), "sink append");
    }
    tdc_pushpull_encoder_close(&enc);

    tdc_pushpull_decoder_config dcfg;
    memset(&dcfg, 0, sizeof(dcfg));
    dcfg.realloc_fn = test_realloc;

    tdc_pushpull_decoder *dec = NULL;
    ASSERT_OR_DIE(tdc_pushpull_decoder_open(&dcfg, &dec) == TDC_OK, "dec open");

    uint8_t *recon = (uint8_t *)malloc(n_bytes);
    ASSERT_OR_DIE(recon, "alloc recon");
    size_t off = 0;
    for (size_t i = 0; i < sink.size; ++i) {
        ASSERT_OR_DIE(tdc_pushpull_decoder_push(dec, sink.data + i, 1, NULL)
                          == TDC_OK, "1-byte dec push");
        for (;;) {
            tdc_block blk;
            int have = 0, done = 0;
            tdc_status st = tdc_pushpull_decoder_next(dec, &blk, &have, &done);
            ASSERT_OR_DIE(st == TDC_OK, "dec next 1-byte");
            if (!have) break;
            size_t nb = (size_t)blk.shape.dim[0] * sizeof(double);
            ASSERT_OR_DIE(off + nb <= n_bytes, "recon cap");
            memcpy(recon + off, blk.data, nb);
            off += nb;
        }
    }
    ASSERT_OR_DIE(tdc_pushpull_decoder_finish(dec) == TDC_OK, "dec finish");
    for (;;) {
        tdc_block blk;
        int have = 0, done = 0;
        ASSERT_OR_DIE(tdc_pushpull_decoder_next(dec, &blk, &have, &done)
                          == TDC_OK, "dec next final");
        if (have) {
            size_t nb = (size_t)blk.shape.dim[0] * sizeof(double);
            memcpy(recon + off, blk.data, nb);
            off += nb;
            continue;
        }
        if (done) break;
        ASSERT_OR_DIE(0, "dec starved after finish");
    }
    tdc_pushpull_decoder_close(&dec);

    ASSERT_OR_DIE(off == n_bytes, "recon size");
    ASSERT_OR_DIE(memcmp(recon, src, n_bytes) == 0, "byte-at-a-time memcmp");

    free(recon);
    free(src);
    sink_free(&sink);
    return 0;
}

static int test_mid_flush_and_checkpoint(void) {
    printf("test_mid_flush_and_checkpoint\n");
    size_t n = 4000;
    double *src = alloc_test_vector_f64(n, 0);
    ASSERT_OR_DIE(src, "alloc");

    tdc_pushpull_encoder_config ecfg;
    memset(&ecfg, 0, sizeof(ecfg));
    ecfg.dtype  = TDC_DT_F64;
    ecfg.layout = TDC_LAYOUT_VECTOR_1D;
    ecfg.shape_template.rank   = 1;
    ecfg.spec.model            = TDC_MODEL_RAW;
    ecfg.spec.entropy[0]       = TDC_ENTROPY_LZ;
    ecfg.target_block_elems    = 1000;
    ecfg.realloc_fn            = test_realloc;

    tdc_pushpull_encoder *enc = NULL;
    ASSERT_OR_DIE(tdc_pushpull_encoder_open(&ecfg, &enc) == TDC_OK, "enc open");

    byte_sink sink = {0};

    ASSERT_OR_DIE(tdc_pushpull_encoder_push(enc, src,
                                            500 * sizeof(double)) == TDC_OK,
                  "push pre-flush");
    ASSERT_OR_DIE(tdc_pushpull_encoder_flush(enc) == TDC_OK, "flush");

    tdc_pushpull_ckpt ck1;
    ASSERT_OR_DIE(tdc_pushpull_encoder_checkpoint(enc, &ck1) == TDC_OK, "ck1");
    ASSERT_OR_DIE(ck1.blocks == 1, "1 block after flush");
    ASSERT_OR_DIE(ck1.bytes == 500 * sizeof(double), "bytes after flush");

    ASSERT_OR_DIE(tdc_pushpull_encoder_push(enc, src + 500,
                                            3500 * sizeof(double)) == TDC_OK,
                  "push post-flush");
    ASSERT_OR_DIE(tdc_pushpull_encoder_finish(enc) == TDC_OK, "finish");
    ASSERT_OR_DIE(drain_encoder(enc, &sink), "drain");

    tdc_pushpull_ckpt ck_final;
    ASSERT_OR_DIE(tdc_pushpull_encoder_checkpoint(enc, &ck_final) == TDC_OK,
                  "ck final");
    /* 500 elems (flush) + 3500 elems (3x1000 + trailing 500) = 5 blocks. */
    ASSERT_OR_DIE(ck_final.blocks == 5, "5 blocks at end");
    ASSERT_OR_DIE(ck_final.bytes == 4000 * sizeof(double),
                  "4000 elems bytes at end");

    tdc_pushpull_encoder_close(&enc);

    tdc_pushpull_decoder_config dcfg;
    memset(&dcfg, 0, sizeof(dcfg));
    dcfg.realloc_fn = test_realloc;
    tdc_pushpull_decoder *dec = NULL;
    ASSERT_OR_DIE(tdc_pushpull_decoder_open(&dcfg, &dec) == TDC_OK, "dec open");

    ASSERT_OR_DIE(tdc_pushpull_decoder_push(dec, sink.data, sink.size, NULL)
                      == TDC_OK, "dec push all");
    ASSERT_OR_DIE(tdc_pushpull_decoder_finish(dec) == TDC_OK, "dec finish");

    double *recon = (double *)malloc(n * sizeof(double));
    ASSERT_OR_DIE(recon, "alloc recon");
    size_t off = 0;
    uint64_t blocks_seen = 0;
    for (;;) {
        tdc_block blk;
        int have = 0, done = 0;
        ASSERT_OR_DIE(tdc_pushpull_decoder_next(dec, &blk, &have, &done)
                          == TDC_OK, "dec next");
        if (have) {
            size_t nb = (size_t)blk.shape.dim[0] * sizeof(double);
            memcpy((uint8_t *)recon + off, blk.data, nb);
            off += nb;
            blocks_seen++;

            if (blocks_seen == 1) {
                tdc_pushpull_ckpt dck;
                ASSERT_OR_DIE(tdc_pushpull_decoder_checkpoint(dec, &dck)
                                  == TDC_OK, "dec ck");
                ASSERT_OR_DIE(dck.blocks == 1, "dec 1 block");
                ASSERT_OR_DIE(dck.bytes == 500 * sizeof(double),
                              "dec bytes match ck1");
            }
            continue;
        }
        if (done) break;
        ASSERT_OR_DIE(0, "dec starved unexpectedly");
    }

    ASSERT_OR_DIE(blocks_seen == 5, "dec pulled 5 blocks");
    ASSERT_OR_DIE(off == n * sizeof(double), "dec output bytes");
    ASSERT_OR_DIE(memcmp(recon, src, n * sizeof(double)) == 0, "flush recon");

    tdc_pushpull_decoder_close(&dec);
    free(recon);
    free(src);
    sink_free(&sink);
    return 0;
}

/* ===== Main ============================================================== */

int main(void) {
    printf("test_pushpull_roundtrip\n");
    int fail = 0;

    size_t n_elems = 1310720;  /* 10 MiB exactly at 8 B/elem */
    size_t n_bytes = n_elems * sizeof(double);

    /* (1) Smooth ramp — RAW+LZ. */
    {
        double *src = alloc_test_vector_f64(n_elems, 0);
        ASSERT_OR_DIE(src != NULL, "alloc ramp");

        pipeline p_raw = { "RAW+LZ", spec_raw_lz() };
        fail += run_big_roundtrip(&p_raw, (uint8_t *)src, n_bytes,
                                  0x1111ull, 0x2222ull, 17, 23,
                                  0 /* default block size */);

        pipeline p_bs = { "RAW+BSHUF+LZ", spec_bshuf_lz() };
        fail += run_big_roundtrip(&p_bs, (uint8_t *)src, n_bytes,
                                  0x3333ull, 0x4444ull, 17, 23,
                                  200000);

        free(src);
    }

    /* (2) Low-cardinality for DICT_NUMERIC. */
    {
        double *src = alloc_test_vector_f64(n_elems, 1);
        ASSERT_OR_DIE(src != NULL, "alloc low-card");

        pipeline p_dn = { "DICT_NUMERIC+BSHUF+LZ_STREAMS",
                          spec_dict_bshuf_lz_streams() };
        fail += run_big_roundtrip(&p_dn, (uint8_t *)src, n_bytes,
                                  0x5555ull, 0x6666ull, 17, 23,
                                  150000);

        free(src);
    }

    fail += test_empty_stream();
    fail += test_byte_at_a_time();
    fail += test_mid_flush_and_checkpoint();

    printf("\n  %d test(s) failed\n", fail);
    return fail ? 1 : 0;
}
