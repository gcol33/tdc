/*
 * tests/test_side_meta_e2e.c
 *
 * Side metadata through the entropy chain, end to end through the driver.
 *
 * A column of distinct names puts every byte of the column in the DICT_1D
 * dictionary, which is side metadata. The test encodes such a column and
 * checks that
 *   - the record is a current-format ('TDBK', version 1) record whose side
 *     section is much smaller than the raw dictionary;
 *   - it decodes back to the input;
 *   - a legacy ('BLKB', version 2) record carrying the same column with the
 *     dictionary stored raw, as the previous format wrote it, still decodes.
 */

#include "tdc/codec.h"
#include "tdc/format.h"
#include "tdc/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *test_realloc(void *user, void *ptr, size_t new_size) {
    (void)user;
    if (new_size == 0) { free(ptr); return NULL; }
    return realloc(ptr, new_size);
}

static tdc_buffer make_buffer(void) {
    tdc_buffer b = {0};
    b.realloc_fn = test_realloc;
    return b;
}

#define N_GENERA  400
#define N_SPECIES 50
#define N_NAMES   (N_GENERA * N_SPECIES)

static const char *syll[] = {"ab", "cer", "dra", "fol", "gyn", "hel", "lor",
                             "mon", "pha", "quer", "rho", "sal", "tri", "vio"};
#define N_SYLL (sizeof(syll) / sizeof(syll[0]))

/* Sorted binomial-like names: genus then epithet, each built from a
 * deterministic LCG over a small syllable set, like a backbone's
 * canonical_name column. */
static size_t make_names(char *data, uint32_t *off) {
    uint32_t lcg = 12345u;
    size_t pos = 0;
    for (int g = 0; g < N_GENERA; ++g) {
        char genus[64];
        int gl = snprintf(genus, sizeof(genus), "G%03d", g);
        for (int k = 0; k < 3; ++k) {
            lcg = lcg * 1664525u + 1013904223u;
            gl += snprintf(genus + gl, sizeof(genus) - (size_t)gl, "%s",
                           syll[(lcg >> 16) % N_SYLL]);
        }
        for (int s = 0; s < N_SPECIES; ++s) {
            off[g * N_SPECIES + s] = (uint32_t)pos;
            memcpy(data + pos, genus, (size_t)gl); pos += (size_t)gl;
            data[pos++] = ' ';
            int sl = snprintf(data + pos, 64, "s%02d", s);
            pos += (size_t)sl;
            for (int k = 0; k < 2; ++k) {
                lcg = lcg * 1664525u + 1013904223u;
                const char *y = syll[(lcg >> 16) % N_SYLL];
                size_t yl = strlen(y);
                memcpy(data + pos, y, yl); pos += yl;
            }
        }
    }
    off[N_NAMES] = (uint32_t)pos;
    return pos;
}

static int check_dict(const char *label, const uint8_t *rec, size_t rec_size,
                      const char *data, const uint32_t *off) {
    tdc_dict_block dict = {0};
    tdc_buffer alloc = make_buffer();
    tdc_status st = tdc_decode_block_dict(rec, rec_size, &dict, &alloc);
    if (st != TDC_OK) {
        printf("  %-10s decode_dict -> %d FAIL\n", label, (int)st);
        return 1;
    }
    int rc = 0;
    if (dict.n != N_NAMES) rc = 1;
    for (int64_t i = 0; i < N_NAMES && !rc; ++i) {
        uint32_t idx = dict.indices[i];
        if (idx >= dict.dict_count) { rc = 1; break; }
        uint32_t ds = dict.dict_offsets[idx], de = dict.dict_offsets[idx + 1];
        uint32_t es = off[i], ee = off[i + 1];
        if (de - ds != ee - es || memcmp(dict.dict_data + ds, data + es, de - ds)) rc = 1;
    }
    printf("  %-10s %s\n", label, rc ? "value mismatch FAIL" : "OK");
    if (dict.indices)      test_realloc(NULL, dict.indices, 0);
    if (dict.dict_offsets) test_realloc(NULL, dict.dict_offsets, 0);
    if (dict.dict_data)    test_realloc(NULL, dict.dict_data, 0);
    return rc;
}

int main(void) {
    int rc = 0;
    char *data = (char *)malloc((size_t)N_NAMES * 64);
    uint32_t *off = (uint32_t *)malloc(sizeof(uint32_t) * (N_NAMES + 1));
    if (!data || !off) return 1;
    size_t total = make_names(data, off);

    tdc_block in = {0};
    in.data    = data;
    in.offsets = off;
    in.dtype   = TDC_DT_STRING;
    in.layout  = TDC_LAYOUT_VECTOR_1D;
    in.shape.rank      = 1;
    in.shape.dim[0]    = N_NAMES;
    in.shape.stride[0] = 1;

    tdc_codec_spec spec = {0};
    spec.model      = TDC_MODEL_DICT_1D;
    spec.entropy[0] = TDC_ENTROPY_LZ;

    tdc_buffer enc = make_buffer();
    if (tdc_encode_block(&in, &spec, &enc) != TDC_OK) {
        printf("  encode FAIL\n");
        return 1;
    }

    tdc_block_record hdr;
    memcpy(&hdr, enc.data, TDC_BLOCK_HEADER_SIZE);
    /* Every name is distinct, so the dictionary is the column itself. */
    size_t raw_side = 8u + 4u * (size_t)(N_NAMES + 1) + total;
    printf("  raw dictionary %zu B, stored side meta %u B\n",
           raw_side, hdr.side_meta_size);
    if (hdr.magic != TDC_BLOCK_MAGIC || hdr.version != TDC_BLOCK_VERSION) {
        printf("  record is not current format FAIL\n");
        rc = 1;
    }
    if ((size_t)hdr.side_meta_size * 2u > raw_side) {
        printf("  side meta not compressed FAIL\n");
        rc = 1;
    }
    rc |= check_dict("current", enc.data, enc.size, data, off);

    /* The same column as a legacy record: same payload, raw dictionary. */
    {
        size_t tail = enc.size - TDC_BLOCK_HEADER_SIZE - hdr.side_meta_size;
        size_t leg_size = TDC_BLOCK_HEADER_SIZE + raw_side + tail;
        uint8_t *leg = (uint8_t *)malloc(leg_size);
        if (!leg) return 1;
        tdc_block_record lh = hdr;
        lh.magic          = TDC_LEGACY_BLOCK_MAGIC;
        lh.version        = TDC_LEGACY_BLOCK_VERSION;
        lh.side_meta_size = (uint32_t)raw_side;
        memcpy(leg, &lh, TDC_BLOCK_HEADER_SIZE);
        uint8_t *p = leg + TDC_BLOCK_HEADER_SIZE;
        uint32_t count = N_NAMES, tot = (uint32_t)total;
        memcpy(p, &count, 4); memcpy(p + 4, &tot, 4);
        memcpy(p + 8, off, 4u * (size_t)(N_NAMES + 1));
        memcpy(p + 8 + 4u * (size_t)(N_NAMES + 1), data, total);
        memcpy(p + raw_side,
               enc.data + TDC_BLOCK_HEADER_SIZE + hdr.side_meta_size, tail);
        rc |= check_dict("legacy", leg, leg_size, data, off);
        free(leg);
    }

    test_realloc(NULL, enc.data, 0);
    free(data);
    free(off);
    printf(rc ? "FAILED\n" : "ALL OK\n");
    return rc;
}
