/* docs/examples/xforms_quantize_tune.c
 *
 * QUANTIZE parameter walk. One f64 input, four (scale, target) combos.
 * The RAW model is used so the only compression source is the transform
 * chain. QUANTIZE maps float input to a signed integer target; putting
 * ZIGZAG + BYTE_SHUFFLE + LZ downstream collapses the quantized bytes.
 *
 * Build:
 *   cc -I include docs/examples/xforms_quantize_tune.c \
 *      build/libtdc.a -lm -o /tmp/xf_qtune
 */

#include "quickstart_common.h"
#include <math.h>
#include <stdlib.h>

static void run(const char *label,
                const tdc_block    *src,
                double              scale,
                tdc_dtype           target,
                size_t              raw) {
    tdc_quantize_params qp = { .scale = scale, .offset = 0.0, .target = target };

    tdc_codec_spec spec = {0};
    spec.model         = TDC_MODEL_RAW;
    spec.xform[0]      = TDC_XFORM_QUANTIZE;
    spec.xform[1]      = TDC_XFORM_ZIGZAG;
    spec.xform[2]      = TDC_XFORM_BYTE_SHUFFLE;
    spec.xform_params[0] = &qp;
    spec.entropy[0]    = TDC_ENTROPY_LZ;

    tdc_buffer enc = qs_buffer();
    if (qs_check(tdc_encode_block(src, &spec, &enc), label)) exit(1);
    double ratio = (double)raw / (double)enc.size;
    printf("  %-32s %6zu bytes  ratio=%.2fx\n", label, enc.size, ratio);
    qs_buffer_free(&enc);
}

int main(void) {
    enum { N = 4096 };
    static double data[N];
    /* A smooth sinusoid in [-100, 100]. */
    for (int i = 0; i < N; ++i) {
        data[i] = 100.0 * sin((double)i * 0.01);
    }

    tdc_block src = {0};
    src.data   = data;
    src.dtype  = TDC_DT_F64;
    src.layout = TDC_LAYOUT_VECTOR_1D;
    src.shape.rank   = 1;
    src.shape.dim[0] = N;
    tdc_shape_set_contiguous(&src.shape);

    size_t raw = N * sizeof(double);
    printf("input: %zu bytes (%d x f64), values in [-100, 100]\n\n", raw, N);

    /* scale = k -> step size = 1/k. The target range must hold scale *
     * (max_abs_value). Oversizing scale for a narrow target clamps every
     * value to the target's [min, max] and throws away the signal.
     *
     * I8 holds [-128, 127]. I16 holds [-32768, 32767]. I32 holds the
     * full 32-bit range. Values here are in [-100, 100], so:
     *
     *   scale=1      -> stored in [-100, 100], fits I8.
     *   scale=100    -> stored in [-10000, 10000], fits I16 with headroom.
     *   scale=1000   -> stored in [-100000, 100000], fits I32.
     *   scale=1e6    -> stored in [-1e8, 1e8], fits I32 (step 1e-6). */
    run("scale=1      target=i8 ",   &src, 1.0,    TDC_DT_I8,  raw);
    run("scale=100    target=i16",   &src, 100.0,  TDC_DT_I16, raw);
    run("scale=1000   target=i32",   &src, 1000.0, TDC_DT_I32, raw);
    run("scale=1e6    target=i32",   &src, 1.0e6,  TDC_DT_I32, raw);

    return 0;
}
