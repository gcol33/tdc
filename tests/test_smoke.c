/*
 * tests/test_smoke.c
 *
 * Minimal smoke test: include all public headers and call
 * tdc_codec_spec_raw() to confirm the headers compile and link.
 *
 * Real round-trip tests will land here once src/api/encode.c and
 * src/api/decode.c are implemented:
 *   - test_lz_roundtrip       (entropy stage isolated)
 *   - test_byte_shuffle        (transform stage isolated)
 *   - test_pred2d              (model stage isolated, on a synthetic raster)
 *   - test_pipeline_full       (model + transform chain + entropy)
 *   - test_block_record_format (binary on-disk byte layout)
 */

#include "tdc.h"

int main(void) {
    tdc_codec_spec spec = tdc_codec_spec_raw();
    (void)spec;
    return 0;
}
