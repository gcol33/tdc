/*
 * src/format/stats_internal.h — column statistics for zone-map filtering
 *
 * Per-column min/max stats attached to row-group index entries.
 * Wire format: 33 bytes per column (1 has_stats + 16 min + 16 max).
 *
 * Internal header — not part of the public ABI.
 */

#ifndef TDC_FORMAT_STATS_INTERNAL_H
#define TDC_FORMAT_STATS_INTERNAL_H

#include "tdc/types.h"
#include <stdint.h>
#include <stddef.h>

#define TDC_STATS_ENTRY_SIZE 33   /* 1 + 16 + 16 */
#define TDC_STATS_VALUE_SIZE 16

/* One column's statistics */
typedef struct {
    uint8_t  has_stats;
    uint8_t  min[TDC_STATS_VALUE_SIZE];
    uint8_t  max[TDC_STATS_VALUE_SIZE];
} tdc_column_stats;

/*
 * Compute min/max statistics from a tdc_block. Writes into `out`.
 * Sets has_stats = 1 on success, 0 if dtype is unsupported for stats
 * or the block is empty (n_elems == 0).
 */
tdc_status tdc_stats_compute(const tdc_block *blk, tdc_column_stats *out);

/*
 * Serialize n_cols stats entries into buf (must be n_cols * 33 bytes).
 * Returns bytes written.
 */
size_t tdc_stats_serialize(const tdc_column_stats *stats, uint16_t n_cols,
                           uint8_t *buf);

/*
 * Parse n_cols stats entries from buf. out must point to an array of
 * at least n_cols tdc_column_stats.
 */
tdc_status tdc_stats_parse(const uint8_t *buf, size_t buf_size,
                           uint16_t n_cols, tdc_column_stats *out);

/*
 * Helper: store a typed value into a 16-byte stats buffer.
 * value points to a value of the appropriate dtype. Zero-pads remaining bytes.
 */
void tdc_stats_store_value(uint8_t dst[TDC_STATS_VALUE_SIZE],
                           const void *value, tdc_dtype dtype);

/*
 * Helper: load a typed value from a 16-byte stats buffer into dst.
 * dst must be large enough for the dtype.
 */
void tdc_stats_load_value(const uint8_t src[TDC_STATS_VALUE_SIZE],
                          void *dst, tdc_dtype dtype);

#endif /* TDC_FORMAT_STATS_INTERNAL_H */
