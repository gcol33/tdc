/* docs/examples/theory_lz_histogram.c
 *
 * Encodes three synthetic inputs with TDC_ENTROPY_LZ and prints, for each:
 *   - raw / encoded byte counts and compression ratio
 *   - a literal-length histogram and match-length histogram over the LZ
 *     sequence stream carried inside the block record
 *   - encode throughput in MB/s
 *
 * The three inputs exercise the three regimes the greedy parser cares about:
 *
 *   1. "zero":    all-zero bytes. After one literal seeding byte, one long
 *                 match at offset 1 extends to end-of-input; the literal
 *                 histogram collapses to the "1-3" bucket and the match
 *                 histogram to the ">=1024" bucket. (TDC_BLOCK_FLAG_ZERO_RESIDUAL
 *                 would short-circuit this case entirely under a model that
 *                 detects zero residuals, e.g. PLANE_2D; under RAW the LZ
 *                 stage runs as above.)
 *   2. "abc":     the repeating string "abcabcabc..." (classic LZ teaching
 *                 input). After the first three literals the parser emits
 *                 one match that extends to end-of-input.
 *   3. "noise":   LCG bytes with no exploitable structure. The parser emits
 *                 no matches at all; the record is the 8-byte LZ header plus
 *                 every input byte as a trailing literal.
 *
 * Build:
 *   cc -I include docs/examples/theory_lz_histogram.c build/libtdc.a -lm \
 *      -o /tmp/lz_hist
 */

#include "quickstart_common.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/* ----- LZ stream decoder (decode-only, for inspection) -------------------- */
/*
 * Walks the sequence descriptors inside a TDC_ENTROPY_LZ payload and counts
 * literal-run and match-length occurrences into caller-provided histograms.
 * Matches the on-disk layout documented in src/entropy/lz_internal.h:
 *   [u32 n_seq][u32 literals_size][packed sequence headers][literal bytes]
 * The packed header per sequence is:
 *   1-byte tag   high nibble = lit_len (15 = extended chained-255),
 *                low nibble  = match_len - 3 (15 = extended LEB128)
 *   [lit ext]    chained-255 varint when lit_len == 15
 *   [match ext]  LEB128 when match_len_m3 == 15
 *   [offset]     uint16_le base, optional LEB128 extension when base == 0xFFFF
 */
enum { LZ_HEADER_SIZE = 8u };

static uint32_t bucket_index(uint32_t v) {
    /* Power-of-two buckets: 0, 1-3, 4-15, 16-63, 64-255, 256-1023, >=1024. */
    if (v == 0) return 0;
    if (v <= 3) return 1;
    if (v <= 15) return 2;
    if (v <= 63) return 3;
    if (v <= 255) return 4;
    if (v <= 1023) return 5;
    return 6;
}

static const char *BUCKET_LABELS[7] = {
    "     0", "  1- 3", "  4-15", " 16-63", " 64-255", "256-1K", "  >=1K"
};

static void histogram_scan(const uint8_t *payload, size_t payload_size,
                           uint32_t lit_hist[7], uint32_t match_hist[7],
                           uint32_t *n_seq_out, uint32_t *lit_bytes_out,
                           uint32_t *match_bytes_out) {
    memset(lit_hist,   0, 7 * sizeof(uint32_t));
    memset(match_hist, 0, 7 * sizeof(uint32_t));
    *n_seq_out = 0;
    *lit_bytes_out = 0;
    *match_bytes_out = 0;

    if (payload_size < LZ_HEADER_SIZE) return;

    uint32_t n_seq, literals_size;
    memcpy(&n_seq,         payload,     4);
    memcpy(&literals_size, payload + 4, 4);
    *n_seq_out = n_seq;

    const uint8_t *p = payload + LZ_HEADER_SIZE;
    const uint8_t *end = payload + payload_size;

    for (uint32_t i = 0; i < n_seq && p < end; i++) {
        uint8_t tag = *p++;
        uint32_t lit_len = tag >> 4;
        uint32_t match_m3 = tag & 0x0F;

        if (lit_len == 15) {
            uint8_t extra;
            do { extra = *p++; lit_len += extra; } while (extra == 255 && p < end);
        }
        if (match_m3 == 15) {
            uint32_t extra = 0, shift = 0;
            uint8_t b;
            do {
                b = *p++;
                extra |= ((uint32_t)(b & 0x7F)) << shift;
                shift += 7;
            } while ((b & 0x80) && p < end);
            match_m3 += extra;
        }
        uint32_t match_len = match_m3 + 3;

        /* Skip the offset field. 2 bytes base, plus LEB128 extension when
         * base == 0xFFFF. We do not need the value for the histogram. */
        if (p + 2 > end) return;
        uint16_t off_base;
        memcpy(&off_base, p, 2);
        p += 2;
        if (off_base == 0xFFFF) {
            uint8_t b;
            do { b = *p++; } while ((b & 0x80) && p < end);
        }

        lit_hist[bucket_index(lit_len)]++;
        match_hist[bucket_index(match_len)]++;
        *lit_bytes_out += lit_len;
        *match_bytes_out += match_len;
    }
}

/* ----- Locate the LZ payload inside a tdc block record -------------------- */
/*
 * The on-disk layout (see tdc/format.h) is:
 *
 *     0                                                        header  (80)
 *     80                                                       side_meta     (N)
 *     80 + N                                                   xform_params  (M)
 *     80 + N + M                                               entropy payload (P)
 *
 * The entropy payload itself opens with a per-stage sizes table
 * (one u32 per entropy slot recording the bytes FED INTO that stage). For a
 * single-stage LZ chain that is 4 bytes. The actual LZ stream follows
 * immediately after. Skipping those 4 bytes lands us at n_seq.
 */
static void find_entropy_payload(const uint8_t *rec, size_t rec_size,
                                  const uint8_t **out_payload,
                                  size_t *out_payload_size) {
    if (rec_size < TDC_BLOCK_HEADER_SIZE) {
        *out_payload = NULL; *out_payload_size = 0; return;
    }
    const tdc_block_record *hdr = (const tdc_block_record *)rec;
    size_t off = TDC_BLOCK_HEADER_SIZE
               + (size_t)hdr->side_meta_size
               + (size_t)hdr->xform_params_size;
    if (off + (size_t)hdr->payload_size > rec_size) {
        *out_payload = NULL; *out_payload_size = 0; return;
    }
    /* Count active entropy slots to find the sizes-table prefix length. */
    size_t entropy_slots = 0;
    for (int i = 0; i < TDC_MAX_ENTROPY; i++) {
        if (hdr->entropy_ids[i] == 0) break;
        entropy_slots++;
    }
    size_t prefix = entropy_slots * sizeof(uint32_t);
    if (prefix > (size_t)hdr->payload_size) {
        *out_payload = NULL; *out_payload_size = 0; return;
    }
    *out_payload = rec + off + prefix;
    *out_payload_size = (size_t)hdr->payload_size - prefix;
}

/* ----- Throughput timer --------------------------------------------------- */

static double ms_since(struct timespec t0) {
    struct timespec t1;
    timespec_get(&t1, TIME_UTC);
    return (double)(t1.tv_sec - t0.tv_sec) * 1000.0
         + (double)(t1.tv_nsec - t0.tv_nsec) / 1.0e6;
}

/* ----- Input generators --------------------------------------------------- */

static void fill_zero(uint8_t *dst, size_t n)   { memset(dst, 0, n); }

static void fill_abc(uint8_t *dst, size_t n) {
    static const char pat[3] = { 'a', 'b', 'c' };
    for (size_t i = 0; i < n; i++) dst[i] = (uint8_t)pat[i % 3];
}

static void fill_noise(uint8_t *dst, size_t n) {
    /* Minimal LCG — good enough to defeat LZ matching on 4-byte windows. */
    uint64_t s = 0x243F6A8885A308D3ull;
    for (size_t i = 0; i < n; i++) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        dst[i] = (uint8_t)(s >> 56);
    }
}

/* ----- Per-input driver --------------------------------------------------- */

static void run(const char *label, uint8_t *buf, size_t n) {
    tdc_block src = {0};
    src.data   = buf;
    src.dtype  = TDC_DT_U8;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank   = 1;
    src.shape.dim[0] = (int64_t)n;
    tdc_shape_set_contiguous(&src.shape);

    tdc_codec_spec spec = tdc_codec_spec_raw();
    spec.entropy[0] = TDC_ENTROPY_LZ;

    tdc_buffer enc = qs_buffer();

    struct timespec t0; timespec_get(&t0, TIME_UTC);
    if (qs_check(tdc_encode_block(&src, &spec, &enc), "encode")) exit(1);
    double elapsed_ms = ms_since(t0);
    double throughput = ((double)n / (1024.0 * 1024.0)) / (elapsed_ms / 1000.0);
    double ratio = (double)n / (double)enc.size;

    const tdc_block_record *hdr = (const tdc_block_record *)enc.data;
    int zero_residual = (hdr->flags & TDC_BLOCK_FLAG_ZERO_RESIDUAL) != 0;

    const uint8_t *payload; size_t payload_size;
    find_entropy_payload(enc.data, enc.size, &payload, &payload_size);

    uint32_t lit_hist[7], match_hist[7];
    uint32_t n_seq = 0, lit_bytes = 0, match_bytes = 0;
    if (!zero_residual) {
        histogram_scan(payload, payload_size, lit_hist, match_hist,
                       &n_seq, &lit_bytes, &match_bytes);
    } else {
        memset(lit_hist,   0, sizeof(lit_hist));
        memset(match_hist, 0, sizeof(match_hist));
    }

    printf("%s: raw=%zu bytes, enc=%zu bytes, ratio=%.2fx, %.1f MB/s\n",
           label, n, enc.size, ratio, throughput);
    if (zero_residual) {
        printf("  (zero-residual fast path: xform + entropy stages skipped)\n\n");
        qs_buffer_free(&enc);
        return;
    }
    printf("  sequences: %u   literal bytes: %u   match bytes: %u\n",
           n_seq, lit_bytes, match_bytes);
    printf("  %-8s %10s %10s\n", "bucket", "lit_hist", "match_hist");
    for (int k = 0; k < 7; k++) {
        printf("  %-8s %10u %10u\n",
               BUCKET_LABELS[k], lit_hist[k], match_hist[k]);
    }
    printf("\n");

    qs_buffer_free(&enc);
}

int main(void) {
    enum { N = 1u << 20 };   /* 1 MiB per input */
    static uint8_t buf[N];

    fill_zero(buf, N);   run("zero",  buf, N);
    fill_abc(buf,  N);   run("abc",   buf, N);
    fill_noise(buf, N);  run("noise", buf, N);
    return 0;
}
