"""Test: can NASA POWER handle 30-year regional request?"""
import urllib.request
import urllib.error
import json
import time

# 5x5 degree box, 30 years
url = (
    "https://power.larc.nasa.gov/api/temporal/daily/regional?"
    "parameters=T2M&community=AG"
    "&latitude-min=45&latitude-max=50"
    "&longitude-min=10&longitude-max=15"
    "&start=19950101&end=20241231"
    "&format=JSON"
)
print(f"Fetching 30-year regional data...")
t0 = time.time()
req = urllib.request.Request(url, headers={"User-Agent": "tdc-bench/0.1"})
try:
    resp = urllib.request.urlopen(req, timeout=600)
    body = resp.read()
    elapsed = time.time() - t0
    print(f"OK in {elapsed:.1f}s, {len(body)} bytes ({len(body)/1024/1024:.1f} MiB JSON)")
    data = json.loads(body)
    n_feat = len(data.get("features", []))
    if n_feat:
        t2m = data["features"][0]["properties"]["parameter"]["T2M"]
        print(f"  {n_feat} cells × {len(t2m)} days = {n_feat * len(t2m)} values")
        print(f"  binary size: {n_feat * len(t2m) * 8 / 1024 / 1024:.1f} MiB")
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read()[:300]}")
except Exception as e:
    elapsed = time.time() - t0
    print(f"Error after {elapsed:.1f}s: {e}")
