# Predictor math

This page derives the math behind the four core tdc predictors: `DELTA_1D`
for 1D columns, `PRED_2D / PAETH` for rasters, `PLANE_2D` for tiled
piecewise-planar rasters, and `PRED_3D / GRAD3D` for volumes. Every
formula is written to match the code under `src/model/` line for line.
Enum ids and params structs come from
[`include/tdc/codec.h`](https://github.com/gcol33/tdc/blob/main/include/tdc/codec.h),
the vtable from [`include/tdc/model.h`](https://github.com/gcol33/tdc/blob/main/include/tdc/model.h).
The [Models walkthrough](../backends/models.md) covers usage and
benchmarks; this page is about why the kernels are what they are.

The worked example `docs/examples/theory_residual_printer.c` runs every
predictor in this vignette on a small hand-designed input, prints the
expected residual stream alongside the reconstructed output, and
verifies a round trip. Running the binary is the quickest way to
cross-check the math below against a live kernel.

```
$ build/Debug/theory_residual_printer
DELTA_1D int32, input: 1000 1005 1012 1020 1019 1025 1040 1100
  expected residual (i32): seed, then src[i]-src[i-1]:
    1000 5 7 8 -1 6 15 60
  roundtrip: ok
```

## Problem statement

A predictor's job is to turn a block of correlated values into a stream
of residuals whose distribution is friendlier to the transform and
entropy stages downstream. The unit we optimize against is the sum of
absolute residuals, `sum_i |val_i - pred_i|`, because every v0 entropy
backend (LZ, Huffman, FSE, lane-split) wins harder on distributions
tightly clustered around zero. The signals tdc exploits are
along-axis correlation in 1D columns, local gradient structure in 2D
rasters, piecewise-planar structure in DEM-like rasters, and tri-affine
structure in smooth volumes. Every predictor is an exact arithmetic
inverse: residual plus reconstructed context reproduces the input
bit-for-bit, with no rounding, no quantization, and no lossy step
inside the model stage. Anything lossy lives in the transform stage
(`TDC_XFORM_QUANTIZE` or the fused `TDC_MODEL_QUANTIZE_PRED_2D` model
id) and is applied before the predictor sees the data, not after.

The four predictors on this page form a coarse dimensional ladder.
`DELTA_1D` handles the 1D case where the only available context is the
previous sample. `PRED_2D / PAETH` handles the 2D case where a pixel
has three causal neighbors (left, up, up-left) and we must decide
between them on a per-pixel basis. `PLANE_2D` handles the case where a
whole tile carries a consistent linear trend that even Paeth misses.
`PRED_3D / GRAD3D` handles the 3D case where the natural linear
predictor has seven neighbors and the coefficients follow from
inclusion-exclusion. Every kernel reads only from causal (already-seen)
positions so the decoder can invert the step without lookahead.

The block shape we have to work with comes from `tdc_block` in
[`types.h`](https://github.com/gcol33/tdc/blob/main/include/tdc/types.h):
a `data` pointer, a `dtype`, a semantic `layout` (`VECTOR_1D`,
`RASTER_2D`, `STACK_2D`, `VOLUME_3D`), and a `shape` with up to three
dimensions. Models dispatch on layout first so a single-axis predictor
never gets asked to handle a volume, and a volume predictor never gets
a flat column. The layout / dtype acceptance bitmasks in
`tdc_model_vt` enforce this before any kernel code runs. Integer
dtypes wrap modularly at their native width; float dtypes route
through a separate kernel that operates on the raw bit representation
rather than the floating-point value.

## Math

### DELTA_1D integer: first-order difference

For an integer `VECTOR_1D` block `v[0..n-1]` of width N bits, the
encoder writes

```
res[0] = v[0]
res[i] = v[i] - v[i-1]    (mod 2^N, i >= 1)
```

and the decoder inverts with a prefix sum

```
v[0] = res[0]
v[i] = v[i-1] + res[i]    (mod 2^N, i >= 1)
```

`src/model/delta1d.c` implements the subtraction in the unsigned
counterpart of the input dtype (`uint8_t`, `uint16_t`, `uint32_t`,
`uint64_t`), which is well-defined modular arithmetic in C11. A signed
`i32` input of `[1000, 1005, 1012, 1020, 1019, 1025, 1040, 1100]`
yields the residual `[1000, 5, 7, 8, -1, 6, 15, 60]` (the first slot
carries the seed value; the sign of `res[4] = -1` is preserved under
two's-complement reinterpretation). That residual, fed to zigzag, byte
shuffle, and LZ, compresses the 32-byte input past the block record
header to a 127-byte record because the delta stream has 4 bytes of
real signal and the rest are zero-valued high-order lanes.

### DELTA_1D float: XOR delta

For float dtypes, the encoder treats the raw IEEE-754 bit pattern as an
unsigned integer of the matching width and writes

```
res[0] = bits(v[0])
res[i] = bits(v[i]) XOR bits(v[i-1])
```

Decode inverts with a running XOR (XOR is self-inverse, so the same
operator runs both ways). Two adjacent f64 samples with the same sign
and the same exponent share the 12 high bits (1 sign + 11 exponent)
and usually most of the top mantissa bits as well. Their XOR has a
long run of leading zero bytes. On the Pi-neighborhood input
`[3.141592653589793, 3.1416, 3.1417, 3.1418]` the XOR residuals are:

```
[0] 400921FB54442D18
[1] 000000047A0CC5BF
[2] 000003CCB2465647
[3] 0000005B95DA2BF8
```

Every non-seed residual has 3 leading zero bytes. If we had subtracted
the floats instead of XORing their bits, we would have produced a
small double (roughly 1e-4) which, when reinterpreted as uint64, has
the opposite shape: zero mantissa noise near the top but a dense
exponent field in the middle. LZ and byte-shuffle both prefer the XOR
form, which is why `src/model/delta1d.c` picks XOR for every IEEE
float dtype. The kernel lives in the `DEFINE_DELTA1D_ENCODE_FLOAT` /
`DEFINE_DELTA1D_DECODE_FLOAT` macros, specialized for F16, F32, and
F64.

### PRED_2D PAETH: closest of three to the gradient plane

Paeth's predictor comes from the PNG filter family. For an interior
pixel `v(x, y)`, the three causal neighbors are

```
a = v(x-1, y)      (left)
b = v(x,   y-1)    (up)
c = v(x-1, y-1)    (up-left)
```

Assume the neighborhood is locally planar, so `v(x, y)` lies on a
plane passing through `a`, `b`, `c`. Writing the plane as
`v(x, y) = alpha + beta*x + gamma*y` and subtracting out the three
constraints gives

```
v(x, y) = a + b - c
```

which is the PNG "average-of-two" predictor written as a linear
extrapolation. Call this value `p`. PAETH does not use `p` as the
prediction directly; it instead picks whichever of `{a, b, c}` sits
closest to `p` on the real line:

```
p  = a + b - c
pa = |p - a| = |b - c|
pb = |p - b| = |a - c|
pc = |p - c| = |a + b - 2c| = |(b-c) + (a-c)|

pred = a if pa <= pb and pa <= pc
     = b if pb <= pc
     = c otherwise
```

The tie-break order (a beats b, b beats c) matches the PNG
specification and is what `paeth32` and `paeth64` in
`src/model/pred2d.c` use. The algebraic identities for `pa`, `pb`,
`pc` fall out by substitution: `p - a = (a+b-c) - a = b - c`, and
similarly for the others. The kernel hoists `bc = b - c` and
`ac = a - c` as shared subexpressions and uses `pc = |bc + ac|`, which
shortens the dependency chain and lets the compiler emit a flat cmov
sequence on x86_64 and aarch64.

Why closest-of-three and not `p` itself? On a perfectly planar
neighborhood `p == a + b - c` hits the true value exactly, so either
form would give a zero residual. The advantage shows up at edges.
Suppose the raster contains a vertical step: `a` is on one side of the
edge and `b` is on the other. Then `p = a + b - c` lands between them,
which is wrong for every pixel on the step. PAETH picks whichever of
`a` or `b` the target value is actually closest to and gets the right
half right; the residual is the edge crossing, not a smeared
half-step. The worked example shows this on a 3x3 raster
`v(x, y) = 10 + 3x + 5y`:

```
   input        residual
    10 13 16     10  3  3
    15 18 21      5  3  3
    20 23 26      5  3  3
```

Row 0 and column 0 run through the "only `a` in bounds" and "only `b`
in bounds" degenerate cases, which reduce to `v - a` and `v - b`
respectively. The interior has the constant residual 3 because the
plane is `beta = 3, gamma = 5, c = 10`, and `b - c` picks up the x-axis
step of 3 while `a` beats the other candidates on the tie-break.

### PLANE_2D: least-squares fit per tile

`PLANE_2D` handles rasters where the gradient is too consistent across
a region for Paeth's per-pixel decisions to matter: DEM tiles, depth
maps, synthetic aperture products. The raster is cut into
`tile_size * tile_size` tiles (default 32), and each tile gets its own
three-coefficient plane fit.

For a tile of pixels with local coordinates `(lx, ly)` and values
`v_k`, we fit

```
pred(lx, ly) = a + b * lx + c * ly
```

by minimizing `sum_k (v_k - a - b*lx_k - c*ly_k)^2`. Taking partial
derivatives with respect to `a`, `b`, `c` and setting them to zero
gives the normal equations

```
[ s_1   s_x   s_y  ] [a]   [s_v ]
[ s_x   s_xx  s_xy ] [b] = [s_vx]
[ s_y   s_xy  s_yy ] [c]   [s_vy]
```

where `s_1 = count`, `s_x = sum lx`, `s_y = sum ly`, `s_xx = sum lx^2`,
`s_xy = sum lx*ly`, `s_yy = sum ly^2`, `s_v = sum v`, `s_vx = sum v*lx`,
`s_vy = sum v*ly`. The coefficient matrix is symmetric positive
definite whenever the tile has at least three non-collinear sample
points, which always holds on a non-degenerate tile because the
local coords span a rectangular grid. `src/model/plane2d.c` solves
the 3x3 system by Cramer's rule:

```
det = s_1 * (s_xx * s_yy - s_xy^2)
    - s_x * (s_x  * s_yy - s_xy * s_y)
    + s_y * (s_x  * s_xy - s_xx * s_y)

a = ( s_v  * (s_xx * s_yy - s_xy^2)
    - s_vx * (s_x  * s_yy - s_xy * s_y)
    + s_vy * (s_x  * s_xy - s_xx * s_y) ) / det

b = ( s_1  * (s_vx * s_yy - s_vy * s_xy)
    - s_x  * (s_v  * s_yy - s_vy * s_y)
    + s_y  * (s_v  * s_xy - s_vx * s_y)  ) / det

c = ( s_1  * (s_xx * s_vy - s_xy * s_vx)
    - s_x  * (s_x  * s_vy - s_xy * s_v)
    + s_y  * (s_x  * s_vx - s_xx * s_v)  ) / det
```

The fitted coefficients are stored as int32 at a 256x fixed-point
scale (8 bits of fractional precision) in the side metadata, so the
predictor `a + b*lx + c*ly` evaluates in pure integer arithmetic at
decode time with a rounding divide by 256. The residual is
`val - round((a + b*lx + c*ly) / 256)` at the input dtype's width, stored
modularly.

The residual decomposition is the useful property here: `a` absorbs
the tile mean, `b*lx + c*ly` absorbs the large-scale linear drift, and
anything left in the residual is the high-frequency texture the plane
cannot explain. Running Paeth on top of a plane fit would double-code
the drift; running PLANE_2D on top of a Paeth residual would try to
fit a plane through noise. The two models operate on different
structural scales and the encoder uses whichever one the caller picks.

On the example tile `v(lx, ly) = 100 + 2*lx + 3*ly` covering 8x8
pixels, the normal-equation solve returns `a = 100`, `b = 2`, `c = 3`
exactly, and every residual is zero. The encoded record
(95 bytes, of which 80 is the block record header) lands on the
zero-residual fast path flagged by `TDC_BLOCK_FLAG_ZERO_RESIDUAL`.

### PRED_3D GRAD3D: tri-affine extrapolation

In 3D the causal neighborhood of voxel `v(x, y, z)` has seven entries.
Name them after the PNG convention, with one letter per missing axis:

```
a   = v(x-1, y,   z  )
b   = v(x,   y-1, z  )
c   = v(x,   y,   z-1)
ab  = v(x-1, y-1, z  )
ac  = v(x-1, y,   z-1)
bc  = v(x,   y-1, z-1)
abc = v(x-1, y-1, z-1)
```

Assume the 2x2x2 block containing the target voxel is tri-affine:
`v(x, y, z) = alpha + beta*x + gamma*y + delta*z`. We would like to
express the target as a linear combination of the seven causal
neighbors. Write each neighbor in the affine basis, for example
`a = alpha + beta*(x-1) + gamma*y + delta*z`. Inclusion-exclusion
over the three axes gives

```
pred = a + b + c - ab - ac - bc + abc
```

Substituting the affine form into the right-hand side, the constant
`alpha` shows up once per face-neighbor, minus once per edge-neighbor,
plus once for the corner, which sums to 1. The `beta*x` term shows up
with coefficient +1 as well, and similarly `gamma*y` and `delta*z`.
All the shifts (`beta*(x-1)`, etc.) cancel. So `pred = v(x, y, z)`
exactly on any tri-affine field. This is `TDC_PRED3D_GRAD3D` as
implemented in the `O8` (interior) branch of
`src/model/pred3d.c`.

The worked example on `v(x, y, z) = 7 + 2x + 3y + 5z` confirms this:
for the inner voxel at `(1, 1, 1)` the neighbors are
`(a, b, c, ab, ac, bc, abc) = (15, 14, 12, 12, 10, 9, 7)`, so
`pred = 15 + 14 + 12 - 12 - 10 - 9 + 7 = 17`. The actual value is
`7 + 2 + 3 + 5 = 17`, residual zero.

Boundary octants collapse the formula naturally. On the z=0 face
(front face), `c`, `ac`, `bc`, `abc` are all out of bounds (treated as
zero by the kernel), so `pred = a + b - 0 - 0 - 0 + 0 = a + b`. That
is not quite the 2D Paeth plane `a + b - ab`. On GRAD3D, the missing
term is absorbed by the neighbors themselves being rooted at an origin
of zero, and the face-slab code path in the `O4..O7` octants
(`src/model/pred3d.c` lines 786-823) writes the explicit 2D plane
predictor `a + b - ab` for the z=0 face, `a + c - ac` for the y=0
face, and `b + c - bc` for the x=0 face so the boundary reduces to
2D Paeth's linear predictor. Each edge degenerates to a 1D delta. The
corner is the first voxel, stored as the seed.

`PRED3D_PAETH3D` builds on top of GRAD3D: on the inner octant it
computes `p = a + b + c - ab - ac - bc + abc` and then picks whichever
of `{a, b, c}` is closest to `p`, using the same tie-break order as 2D
Paeth. On each face slab the formula reduces to `paeth2D(a, b, ab)` or
its y/z analogues. On each edge it reduces to the nearest 1D delta.
The per-octant dispatch is in `pred3d_compute` in
`src/model/pred3d.c` around lines 1328-1372.

## Correctness argument

Every predictor in tdc is bijective on its input domain: given the
residual stream and the causal reconstruction so far, the decoder
computes the same prediction as the encoder did and adds (for
subtraction-based predictors) or XORs (for the float delta) to
recover the original value.

For `DELTA_1D` integer, the reconstruction is a prefix sum. At step
`i`, `v_dec[i] = v_dec[i-1] + res[i]` where `v_dec[i-1]` was
reconstructed in the previous iteration. The equality
`v_dec[i] = v_enc[i]` follows by induction on `i`: the base case
`v_dec[0] = res[0] = v_enc[0]` is the seed, and the step
`v_dec[i] = v_dec[i-1] + (v_enc[i] - v_enc[i-1]) = v_enc[i]` uses
commutativity and the induction hypothesis. Every arithmetic step is
well-defined modular arithmetic at the element's native width, so the
round trip survives the full dynamic range of the input dtype,
including the overflow case where `v_enc[i] - v_enc[i-1]` does not
fit in signed N-bit but does wrap cleanly in unsigned N-bit.

For `DELTA_1D` XOR on floats, the reconstruction is a prefix XOR. XOR
is self-inverse on any bit width, so `v_dec[i] = v_dec[i-1] XOR res[i]`
recovers `v_enc[i]` exactly under the same induction. No float
arithmetic enters the path, so NaN bit patterns, signed zeros,
subnormals, and infinities all round-trip bit-for-bit.

For `PRED_2D`, the decoder walks the raster in the same row-major
order as the encoder. At each pixel `(x, y)`, the three causal
neighbors `a`, `b`, `c` come from already-reconstructed positions in
the output buffer, which the encoder read from the same positions in
its input. Both sides compute the same `paeth32(a, b, c)` (or
`paeth64` for 64-bit dtypes) because the function is deterministic
and the inputs match. The residual is stored in the unsigned narrow
counterpart of the dtype and added modularly at the narrow width, so
the inverse of a modular subtract is a modular add. On row 0 the kernel
uses `a = 0` for the first pixel and `c = 0, b = 0` for the seed; on
column 0 it uses `a = 0`, `c = 0`. The boundary rules match between
encode and decode by construction.

For `PLANE_2D`, the encoder fits the plane, stores the three
fixed-point coefficients in the side metadata (delta-coded varint),
and computes the residual `val - round((a + b*lx + c*ly) / 256)`. The
decoder reads the coefficients, evaluates the same integer plane at
the same local coordinates, and adds the residual. Because
`round_div256_i64` in `src/model/plane2d.c` uses round-half-away-from-zero
with a sign-guarded branch, the rounding step is
deterministic and symmetric. The encoder's fit uses doubles for the
9-moment accumulation, but only the final fixed-point coefficients
cross the encode/decode boundary; the decoder never sees or re-runs
the fit.

For `PRED_3D`, the same argument runs octant by octant. The eight
octants (corner, three edges, three faces, interior) partition the
volume by which of `x > 0`, `y > 0`, `z > 0` hold. Each octant uses a
specific prediction rule that collapses out-of-bounds neighbors to
zero, and both encode and decode dispatch on the same
`(has_a, has_b, has_c)` triple. The interior `O8` formula
`pred = a + b + c - ab - ac - bc + abc` is exact on tri-affine fields
as shown above; on general inputs it is still a deterministic function
of seven already-reconstructed neighbors, so the residual plus `pred`
recovers the input value. For `PAETH3D` the same closest-of-three rule
runs with the same tie-break order on both sides, so the kernel chooses
the same branch given the same neighbors.

Edge cases. An empty block (`n_elems = 0`) writes no residual bytes and
reconstructs trivially. A single-element block stores the seed value as
its only residual. A degenerate 1-row raster feeds PRED_2D with no `b`
or `c` neighbors anywhere, which reduces every interior pixel to the
`LEFT` rule `pred = a`. A 1x1x1 volume reduces to a bare seed. The
worked example `theory_residual_printer.c` exercises round trips on
every predictor and asserts byte equality after decode.

## Complexity and hot path

All four predictors are O(n_elems) in encode and decode, where
n_elems is the block's total element count. `PLANE_2D` adds an O(n_elems)
normal-equation accumulation pass plus an O(n_tiles) Cramer's rule
solve, and the side metadata is O(n_tiles) bytes (delta-coded varint).
There is no per-pixel divide or sqrt in any of the kernels.

Memory traffic is the bound on every kernel except the interior octant
of PRED_3D. `DELTA_1D` reads each element once and writes each residual
once, which is 2x the block size in DRAM traffic. The SSE2 and NEON
prefix-sum kernels in `delta1d.c` (`inverse_delta_{2,4,8}_sse2`,
`inverse_delta_{2,4,8}_neon`) lower the decode to a log2(lane-count)
sequence of shift-and-add passes per chunk, which saturates L1
bandwidth on warm blocks. `PRED_2D / PAETH` encode is fully
data-parallel because every neighbor lives in the read-only input
buffer; decode is serial per row because `a` is the previous output
pixel. `src/model/pred2d.c` ships a 4-lane `paeth_4x_sse2` helper for
the wavefront decode path that stages four staggered rows at once to
recover some parallelism. `UP` decode has no `a` dependency and
vectorizes cleanly.

`PRED_3D GRAD3D` interior encode is the heaviest kernel: seven
dependent loads, six subtractions, a store. The row kernels
`pred3d_grad3d_enc_row_sse2` and `pred3d_grad3d_enc_row_neon`
vectorize across the x axis at each (y, z) pair, reusing `sy`, `sz`,
`syz` pointers for the full row. Decode is scalar inside a row because
`a` again depends on the previous store, but the `O2..O7` boundary
octants decode in parallel along their free axes.

`PLANE_2D`'s fit-phase walks each tile once to accumulate the nine
moments. With the default tile size of 32, a 1024x1024 raster has 1024
tiles, each costing 1024 multiply-adds on 9 accumulators, so the total
is about 9 million ops before the Cramer's rule solve. The
`plane2d_nores_tile_avx2_*` kernels (u8/u16/u32) handle the
zero-residual fast path, which applies whenever `TDC_BLOCK_FLAG_ZERO_RESIDUAL`
fires; they evaluate the integer plane 8 lanes at a time and skip the
residual-add step entirely. Decode is dominated by these kernels on
any piecewise-planar raster.

The hot path across all four predictors is the inner `for` loop over
the last (innermost) axis. `DELTA_1D` gets a single contiguous sweep;
`PRED_2D` gets a sweep per row; `PLANE_2D` gets a sweep per row within
a tile; `PRED_3D` gets a sweep per (y, z) within a slab. None of these
loops branch on the kind after the outer `switch` dispatch because the
kind is hoisted to the top of the function. The one branch inside the
inner loop is PAETH's ternary chain, which the compiler lowers to a
cmov sequence.

## Deviations from textbook

**Paeth tie-break order.** The PNG specification ties broken in order
a, b, c (left, up, up-left). tdc follows this order in `paeth32` and
`paeth64` so the on-disk side metadata stays 1 byte regardless of
which kernel path ran. A reference implementation using `<` instead of
`<=` would produce a subtly different output on tied cases, which
would drift between MSVC and clang builds. The strict `<=` with the
strict a/b/c order is required for a reproducible record.

**Paeth algebraic simplification.** The textbook Paeth writes
`pa = |p - a|` with `p = a + b - c` computed explicitly. tdc's
`paeth32` hoists `bc = b - c` and `ac = a - c` as shared subexpressions
and never materializes `p`. The identities `pa = |b - c|`,
`pb = |a - c|`, `pc = |(b-c) + (a-c)|` are exact. The savings are one
subtraction and one register, and the `pc` dependency chain shrinks
by one step. The 64-bit helper `paeth64` keeps the textbook form
because the dependency pressure is smaller at that width.

**DELTA_1D XOR on floats.** A textbook float delta would subtract
consecutive values and store a float residual. tdc XORs the raw bits
instead. XOR produces long runs of leading zero bits when adjacent
samples share sign and exponent, which every downstream stage
(byte-shuffle, LZ, Huffman) benefits from directly. Float subtraction
produces a value whose bit pattern has the mantissa noise at the top
and the exponent in the middle: the inverse of what the zero-run-loving
entropy stages want. `src/model/delta1d.c` comment around
"Why XOR, not ordered-subtract?" cites a bench on USGS streamflow f64
showing 2.61x for ordered-subtract against 3.02x for no-model BSHUF+LZ;
the XOR path closes the gap.

**PLANE_2D coefficient storage.** A textbook plane fit would store the
three floating-point coefficients as-is. tdc stores them as int32 at a
256x fixed-point scale, so the decoder's plane evaluation is pure
integer arithmetic with a rounding divide by 256. The double-arithmetic
fit only runs at encode time; the decoder never re-runs it. This costs
8 bits of fractional precision on each coefficient, which is enough
for the predictor's sub-pixel slope without overflowing int32 for any
realistic raster magnitude. The coefficient delta stream across tiles
is varint-packed (2D-predicted zigzag-LEB128) so interior tiles that
predict exactly from their left neighbor cost about 3 bytes each
instead of a flat 12. The flat i32 layout was the measured bottleneck
on the 1024x1024 split-plane bench: residuals were exactly zero but
the side metadata alone cost 12 KB.

**PRED_3D inclusion-exclusion coefficients.** The textbook for a 3D
linear predictor would use `(a + b + c) / 3` as the "average"
predictor. tdc ships that as `AVG3` but defaults `AUTO` to GRAD3D on
smooth data because `(a + b + c) / 3` loses to the tri-affine
inclusion-exclusion formula by roughly one LSB per voxel of C integer
truncation, and the truncation error accumulates as a systematic bias
on ramps. GRAD3D's per-voxel cost is one extra add and three extra
loads, which the row-kernel amortizes across a full SIMD lane.

**Decode prefix-sum SIMD.** A naive DELTA_1D decode is a serial
scalar loop because `acc += res[i]` carries a true dependency. tdc's
SSE2 `inverse_delta_4_sse2` lowers each 4-lane chunk to a log2(4) = 2
shift-and-add prefix-sum pass, then folds a broadcast carry from the
previous chunk. NEON has the same shape. This is the same technique
used by bitonic merge kernels on the prefix-sum direction; the tdc
acceptance bar for this kernel is DELTA1D+LZ decode at or above 6 GB/s
on the i32 ramp, which puts decode above the API memcpy ceiling.

The math and the kernel are the same shape in every backend above.
When a new model id lands (a dictionary for categorical columns, an
N-point context predictor for hyperspectral volumes), the derivation
follows the same pattern: write the model as a deterministic function
of already-seen neighbors, prove it is bijective given the residual,
bound its per-element cost. Deviations from textbook are documented
inline in the source file and here, so anyone reading either side of
the fence can match up the formula to the loop.
