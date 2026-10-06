"""Download Washington's terrain from USGS 3DEP at 250 m.

The box runs from the Pacific to Idaho and from Portland to the Canadian
border (see config.py), requested in UTM 10N in tiles of at most 2,000
pixels. The ocean comes back empty and is set to sea level.
"""
from __future__ import annotations

import argparse
import io
import math
import sys
import urllib.parse
import urllib.request

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform

import config

SERVICE = ("https://elevation.nationalmap.gov/arcgis/rest/services/"
           "3DEPElevation/ImageServer/exportImage")
MAX_TILE_PX = 2000
NODATA = -9999.0


def utm_box() -> tuple[float, float, float, float]:
    """UTM box that contains the whole lon/lat box."""
    lons = [config.WEST, config.EAST, config.EAST, config.WEST, (config.WEST + config.EAST) / 2]
    lats = [config.SOUTH, config.SOUTH, config.NORTH, config.NORTH, config.NORTH]
    xs, ys = transform("EPSG:4326", f"EPSG:{config.EPSG}", lons, lats)
    px = config.DEM_PX
    return (math.floor(min(xs) / px) * px, math.floor(min(ys) / px) * px,
            math.ceil(max(xs) / px) * px, math.ceil(max(ys) / px) * px)


def fetch_tile(bbox, width: int, height: int) -> np.ndarray:
    params = {
        "bbox": ",".join(f"{v:.1f}" for v in bbox), "bboxSR": config.EPSG, "imageSR": config.EPSG,
        "size": f"{width},{height}", "format": "tiff", "pixelType": "F32", "noData": NODATA,
        "interpolation": "RSP_BilinearInterpolation", "f": "image",
    }
    with urllib.request.urlopen(SERVICE + "?" + urllib.parse.urlencode(params), timeout=300) as r:
        body = r.read()
    if body[:4] not in (b"II*\x00", b"MM\x00*"):
        raise RuntimeError("USGS did not return a GeoTIFF:\n" + body[:500].decode("utf-8", "replace"))
    with rasterio.open(io.BytesIO(body)) as src:
        return src.read(1).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if config.DEM.exists() and not args.force:
        print(f"DEM already present: {config.DEM}")
        return 0
    sibling = config.SIBLING_DATA / config.DEM.name
    if sibling.exists() and not args.force:
        import shutil
        config.DEM.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sibling, config.DEM)
        print(f"Copied the terrain from the heat-dome project: {sibling}")
        return 0
    west, south, east, north = utm_box()
    px = config.DEM_PX
    w, h = int((east - west) / px), int((north - south) / px)
    gx, gy = math.ceil(w / MAX_TILE_PX), math.ceil(h / MAX_TILE_PX)
    tw, th = math.ceil(w / gx), math.ceil(h / gy)
    w, h = tw * gx, th * gy
    print(f"Requesting Washington terrain from USGS 3DEP: {w}x{h} px at {px} m, {gx * gy} tile(s)...")
    dem = np.empty((h, w), dtype=np.float32)
    for r in range(gy):
        for c in range(gx):
            x0, y1 = west + c * tw * px, north - r * th * px
            dem[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = fetch_tile(
                (x0, y1 - th * px, x0 + tw * px, y1), tw, th)
            print(f"  tile {r * gx + c + 1}/{gx * gy}")
    dem[(dem < -500) | ~np.isfinite(dem)] = 0.0        # ocean and gaps to sea level
    config.DATA.mkdir(parents=True, exist_ok=True)
    with rasterio.open(config.DEM, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
                       crs=f"EPSG:{config.EPSG}", transform=from_origin(west, north, px, px),
                       compress="deflate", predictor=3) as dst:
        dst.write(dem, 1)
    print(f"Saved {config.DEM.name}: {w}x{h} px, elevation {dem.min():.0f}-{dem.max():.0f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
