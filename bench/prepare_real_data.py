"""
bench/prepare_real_data.py

Fetches a small set of real-world datasets and writes them as flat
little-endian binary blobs into bench/data/. Each blob is shipped with
a sibling .meta.json that records dtype, shape, and provenance so the
companion bench drivers can read it back without guessing.

What we fetch:
  1. USGS NWIS daily streamflow at one gauge (Mississippi at St Louis,
     ID 07010000), ~30 years of f64 daily mean discharge. Real,
     non-stationary, mildly noisy 1D time series.
  2. NASA POWER daily 2m air temperature at one grid cell, ~30 years
     of f64 daily means. Smooth seasonal periodic signal — the kind
     of thing climate users hand to a compressor.
  3. Open Topo Data SRTM30m elevation tile, ~512x512 i16 meters.
     Real geo-raster with strong 2D structure.

Run:
    python bench/prepare_real_data.py            # fetch + write all
    python bench/prepare_real_data.py --list     # show what's prepared

If a fetch fails (network down, endpoint changed), the script writes
nothing for that dataset and reports the error. The remaining datasets
are still prepared. Re-run to retry.

The .bin / .meta.json files are gitignored — they live in bench/data/
and must be regenerated locally.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
USER_AGENT = "tdc-bench-prep/0.1 (https://github.com/gcol33/tdc)"


def http_get(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def write_blob(name: str, arr: np.ndarray, meta: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    blob_path = DATA_DIR / f"{name}.bin"
    meta_path = DATA_DIR / f"{name}.meta.json"

    # Force little-endian, contiguous, on-disk type matches dtype string.
    arr_le = np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder("<"))
    arr_le.tofile(blob_path)

    full_meta = {
        "name": name,
        "dtype": str(arr.dtype.name),
        "shape": list(arr.shape),
        "n_elems": int(arr.size),
        "bytes": int(arr_le.nbytes),
        **meta,
    }
    meta_path.write_text(json.dumps(full_meta, indent=2))
    print(f"  -> {blob_path.name}  ({arr_le.nbytes / 1024:.1f} KiB, "
          f"{arr.dtype.name}, shape={list(arr.shape)})")


# ----- Sources --------------------------------------------------------------

def fetch_usgs_streamflow() -> None:
    """USGS NWIS daily mean discharge, Mississippi at St. Louis, MO."""
    name = "usgs_streamflow_mississippi_stlouis"
    print(f"[1/3] {name}")
    site = "07010000"  # Mississippi River at St. Louis, MO
    start = "1995-01-01"
    end = "2024-12-31"
    url = (
        "https://waterservices.usgs.gov/nwis/dv/"
        f"?format=json&sites={site}&startDT={start}&endDT={end}"
        "&parameterCd=00060&siteStatus=all"
    )
    try:
        body = http_get(url, timeout=60.0)
        payload = json.loads(body)
        ts_list = payload["value"]["timeSeries"]
        if not ts_list:
            raise RuntimeError("USGS returned no time series")
        values = ts_list[0]["values"][0]["value"]
        arr = np.array(
            [float(v["value"]) if v["value"] not in ("", None) else np.nan
             for v in values],
            dtype=np.float64,
        )
        # Drop NaNs (a few missing days) so the bench operates on a
        # contiguous block — we want compression behaviour, not gap handling.
        arr = arr[np.isfinite(arr)]
        meta = {
            "source": "USGS NWIS",
            "url": url,
            "site": site,
            "site_name": "Mississippi River at St. Louis, MO",
            "parameter": "discharge_cfs (00060)",
            "period": f"{start}..{end}",
            "description": "Daily mean discharge, ~30 years, real time series.",
        }
        write_blob(name, arr, meta)
    except (urllib.error.URLError, RuntimeError, KeyError) as e:
        print(f"  !! USGS fetch failed: {e}", file=sys.stderr)


def fetch_nasa_power_temperature() -> None:
    """NASA POWER daily mean 2m air temp at one cell."""
    name = "nasa_power_t2m_daily"
    print(f"[2/3] {name}")
    lat, lon = 47.07, 15.43  # Graz, AT — anywhere with strong seasonality
    start = "19950101"
    end = "20241231"
    params = {
        "parameters": "T2M",
        "community": "AG",
        "longitude": f"{lon}",
        "latitude": f"{lat}",
        "start": start,
        "end": end,
        "format": "JSON",
    }
    url = "https://power.larc.nasa.gov/api/temporal/daily/point?" + \
          urllib.parse.urlencode(params)
    try:
        body = http_get(url, timeout=120.0)
        payload = json.loads(body)
        t2m = payload["properties"]["parameter"]["T2M"]
        # Sort by date so the array is in chronological order; drop fill values.
        items = sorted(t2m.items())
        arr = np.array([v for _, v in items], dtype=np.float64)
        # NASA POWER uses -999 as missing.
        arr = arr[arr > -900.0]
        meta = {
            "source": "NASA POWER",
            "url": url,
            "lat": lat,
            "lon": lon,
            "location": "Graz, AT (47.07N 15.43E)",
            "parameter": "T2M (degC, daily mean 2m air temperature)",
            "period": f"{start}..{end}",
            "description": "Daily mean temperature, ~30 years, smooth seasonal.",
        }
        write_blob(name, arr, meta)
    except (urllib.error.URLError, RuntimeError, KeyError) as e:
        print(f"  !! NASA POWER fetch failed: {e}", file=sys.stderr)


def fetch_opentopo_srtm_tile() -> None:
    """Open Topo Data SRTM 30m elevation, ~512x512 i16 meters."""
    name = "opentopo_srtm30m_alps"
    print(f"[3/3] {name}")
    # 0.005 deg ~= 555m at 47N. 256 samples each side = ~140 km square.
    # The Open Topo Data API is rate-limited; we batch in chunks of 100
    # locations per request.
    lat0, lon0 = 47.00, 11.00  # central Alps
    n = 128                    # 128x128 = 16384 samples ~= 32 KiB i16
    step = 0.01                # ~1.1 km
    lats = lat0 + np.arange(n) * step
    lons = lon0 + np.arange(n) * step
    grid_lat = np.repeat(lats, n)
    grid_lon = np.tile(lons, n)

    elev = np.empty(n * n, dtype=np.float64)
    elev.fill(np.nan)
    chunk = 100  # API max locations per request
    base = "https://api.opentopodata.org/v1/srtm30m?locations="
    try:
        for i in range(0, n * n, chunk):
            sl = slice(i, i + chunk)
            locs = "|".join(f"{la:.6f},{lo:.6f}"
                            for la, lo in zip(grid_lat[sl], grid_lon[sl]))
            body = http_get(base + locs, timeout=60.0)
            payload = json.loads(body)
            results = payload.get("results", [])
            for j, r in enumerate(results):
                v = r.get("elevation")
                elev[i + j] = float(v) if v is not None else np.nan
            # Be polite to the public endpoint — 1 req/sec keeps us
            # well below the documented free-tier limit.
            if i + chunk < n * n:
                import time
                time.sleep(1.1)
            print(f"    {min(i + chunk, n * n)}/{n * n} samples", end="\r")
        print()
        if not np.all(np.isfinite(elev)):
            n_bad = int(np.sum(~np.isfinite(elev)))
            print(f"  !! {n_bad} samples missing — skipping", file=sys.stderr)
            return
        arr = elev.reshape(n, n).astype(np.int16)
        meta = {
            "source": "Open Topo Data (SRTM 30m)",
            "url": base + "...",
            "lat0": lat0, "lon0": lon0,
            "step_deg": step,
            "shape": [n, n],
            "description": "SRTM 30m elevation, central Alps, 256x256 i16 m.",
        }
        write_blob(name, arr, meta)
    except (urllib.error.URLError, RuntimeError, KeyError) as e:
        print(f"  !! Open Topo Data fetch failed: {e}", file=sys.stderr)


# ----- Driver ---------------------------------------------------------------

def list_prepared() -> None:
    if not DATA_DIR.exists():
        print("(no bench/data/ directory)")
        return
    metas = sorted(DATA_DIR.glob("*.meta.json"))
    if not metas:
        print("(no prepared blobs)")
        return
    for m in metas:
        info = json.loads(m.read_text())
        print(f"{info['name']}")
        print(f"  dtype={info['dtype']}  shape={info['shape']}  "
              f"bytes={info['bytes']}")
        print(f"  source: {info.get('source', '?')}")
        print(f"  desc:   {info.get('description', '')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip())
    ap.add_argument("--list", action="store_true",
                    help="list prepared datasets and exit")
    args = ap.parse_args()

    if args.list:
        list_prepared()
        return 0

    print(f"Preparing real data into {DATA_DIR}")
    fetch_usgs_streamflow()
    fetch_nasa_power_temperature()
    fetch_opentopo_srtm_tile()
    print()
    list_prepared()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
