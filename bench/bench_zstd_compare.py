"""
bench/bench_zstd_compare.py

Counterpart to bench/bench_throughput.c. Regenerates the same synthetic
blocks in numpy, then runs libzstd (via the `zstandard` Python binding)
at multiple levels on the raw bytes, reporting compression ratio,
encode MB/s and decode MB/s on uncompressed bytes — same metric as the
C bench.

This is the "zstd alone on the raw input" comparison. It is the answer
to: 'does tdc's specialized model+transform stack actually beat a generic
high-quality compressor on the same data?'

Run:
    python bench/bench_zstd_compare.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zstandard as zstd

ITERS = 5
LEVELS = (1, 3, 9, 19)


# ----- Synthetic data generators (mirror bench_throughput.c) ----------------

def gen_raw_u8_16M() -> tuple[str, str, np.ndarray]:
    n = 16 * 1024 * 1024
    arr = (np.arange(n, dtype=np.uint64) & 0xFF).astype(np.uint8)
    return "RAW u8 (passthrough)", "vec1d u8 16M", arr


def gen_ramp_i32_4M() -> tuple[str, str, np.ndarray]:
    n = 4 * 1024 * 1024
    arr = (1000 + np.arange(n, dtype=np.int64) * 3).astype(np.int32)
    return "RAMP i32", "vec1d i32 4M (ramp)", arr


def gen_walk_i16_8M() -> tuple[str, str, np.ndarray]:
    n = 8 * 1024 * 1024
    rng = np.random.default_rng(0xDEADBEEF)
    steps = rng.integers(-16, 16, size=n, dtype=np.int16)  # -16..15
    arr = np.cumsum(steps, dtype=np.int32).astype(np.int16)  # wraps like C
    return "WALK i16", "vec1d i16 8M (walk)", arr


def gen_grad_u16_2048() -> tuple[str, str, np.ndarray]:
    ny, nx = 2048, 2048
    rng = np.random.default_rng(0xC0FFEE)
    noise = rng.integers(-3, 5, size=(ny, nx), dtype=np.int32)  # -3..4
    rows = np.arange(ny, dtype=np.int32)[:, None] * 5
    cols = np.arange(nx, dtype=np.int32)[None, :] * 3
    arr = (100 + rows + cols + noise).astype(np.uint16)
    return "GRAD u16 (smooth+noise)", "rast2d u16 2048x2048", arr


def gen_split_planes_i32_1024() -> tuple[str, str, np.ndarray]:
    ny, nx = 1024, 1024
    rows = np.arange(ny, dtype=np.int32)[:, None]
    cols = np.arange(nx, dtype=np.int32)[None, :]
    top = 200 + rows * 4 + cols * 2
    bot = 1000 - rows * 3 + cols
    mask = rows < (ny // 2)
    arr = np.where(mask, top, bot).astype(np.int32)
    return "SPLIT-PLANES i32", "rast2d i32 1024x1024", arr


CASES = [
    gen_raw_u8_16M,
    gen_ramp_i32_4M,
    gen_walk_i16_8M,
    gen_grad_u16_2048,
    gen_split_planes_i32_1024,
]


# ----- Bench driver ---------------------------------------------------------

@dataclass
class Result:
    label: str
    block: str
    level: int
    raw_bytes: int
    enc_bytes: int
    enc_secs: float
    dec_secs: float

    @property
    def ratio(self) -> float:
        return self.raw_bytes / self.enc_bytes if self.enc_bytes else 0.0

    @property
    def enc_mbps(self) -> float:
        return (self.raw_bytes / (1024.0 * 1024.0)) / self.enc_secs

    @property
    def dec_mbps(self) -> float:
        return (self.raw_bytes / (1024.0 * 1024.0)) / self.dec_secs


def bench_one(label: str, block: str, arr: np.ndarray, level: int) -> Result:
    raw = arr.tobytes()
    cctx = zstd.ZstdCompressor(level=level)
    dctx = zstd.ZstdDecompressor()

    # Encode: best of ITERS
    best_enc = float("inf")
    enc_bytes = b""
    for _ in range(ITERS):
        t0 = time.perf_counter()
        enc_bytes = cctx.compress(raw)
        t1 = time.perf_counter()
        if (t1 - t0) < best_enc:
            best_enc = t1 - t0

    # Decode: best of ITERS
    best_dec = float("inf")
    for _ in range(ITERS):
        t0 = time.perf_counter()
        dec = dctx.decompress(enc_bytes)
        t1 = time.perf_counter()
        if (t1 - t0) < best_dec:
            best_dec = t1 - t0

    if dec != raw:
        raise RuntimeError(f"FAIL [{label} L{level}]: round-trip mismatch")

    return Result(label, block, level, len(raw), len(enc_bytes), best_enc, best_dec)


_DTYPE_MAP = {
    "i8": "int8",  "u8": "uint8",
    "i16": "int16", "u16": "uint16",
    "i32": "int32", "u32": "uint32",
    "i64": "int64", "u64": "uint64",
    "f32": "float32", "f64": "float64",
}


def from_file(path: str, dtype: str, shape: str) -> tuple[str, str, np.ndarray]:
    """Load a flat little-endian binary blob as a numpy array."""
    np_dtype = np.dtype(_DTYPE_MAP.get(dtype, dtype)).newbyteorder("<")
    raw = np.fromfile(path, dtype=np_dtype)
    dims = [int(x) for x in shape.replace(",", "x").split("x")]
    expect = 1
    for d in dims:
        expect *= d
    if raw.size != expect:
        raise SystemExit(
            f"--from {path}: expected {expect} elems for dtype={dtype} "
            f"shape={shape}, got {raw.size}"
        )
    arr = raw.reshape(dims)
    label = f"REAL {Path(path).stem}"
    block = f"{dtype} {shape}"
    return label, block, arr


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.strip())
    ap.add_argument("--from", dest="from_path", default=None,
                    help="path to flat binary blob (overrides synthetic suite)")
    ap.add_argument("--dtype", default=None,
                    help="dtype name for --from (e.g. f64, i16)")
    ap.add_argument("--shape", default=None,
                    help="shape for --from, e.g. '10958' or '128x128'")
    args = ap.parse_args()

    print(f"libzstd {'.'.join(map(str, zstd.ZSTD_VERSION))}  "
          f"(via python-zstandard {zstd.__version__})")

    hdr = (f"{'pipeline':<28}  {'block':<22}  {'L':>2}  "
           f"{'raw MiB':>8}  {'enc MiB':>8}  {'ratio':>7}  "
           f"{'enc MB/s':>9}  {'dec MB/s':>9}")

    if args.from_path:
        if not args.dtype or not args.shape:
            raise SystemExit("--from requires --dtype and --shape")
        print(f"zstd bench on {args.from_path} (best of {ITERS})")
        print()
        print(hdr)
        print("-" * len(hdr))
        label, block, arr = from_file(args.from_path, args.dtype, args.shape)
        for lvl in LEVELS:
            r = bench_one(label, block, arr, lvl)
            print(f"{r.label:<28}  {r.block:<22}  {r.level:>2}  "
                  f"{r.raw_bytes/(1024*1024):>8.2f}  "
                  f"{r.enc_bytes/(1024*1024):>8.2f}  "
                  f"{r.ratio:>6.2f}x  "
                  f"{r.enc_mbps:>9.1f}  "
                  f"{r.dec_mbps:>9.1f}")
        return

    print(f"zstd-alone bench on the same synthetic blocks as bench_throughput.c")
    print(f"throughput is on uncompressed bytes, best of {ITERS}")
    print()

    print(hdr)
    print("-" * len(hdr))

    for gen in CASES:
        label, block, arr = gen()
        for lvl in LEVELS:
            r = bench_one(label, block, arr, lvl)
            print(f"{r.label:<28}  {r.block:<22}  {r.level:>2}  "
                  f"{r.raw_bytes/(1024*1024):>8.2f}  "
                  f"{r.enc_bytes/(1024*1024):>8.2f}  "
                  f"{r.ratio:>6.2f}x  "
                  f"{r.enc_mbps:>9.1f}  "
                  f"{r.dec_mbps:>9.1f}")
        print()


if __name__ == "__main__":
    main()
