"""Download hourly near-surface wildfire smoke over Washington from NOAA's HRRR.

HRRR (High-Resolution Rapid Refresh) runs every hour on a 3 km grid, and it
carries a smoke tracer fed by satellite fire detections ("HRRR-Smoke"). The
field used here is MASSDEN at 8 m above ground: the mass of smoke in the air
people breathe, in kg/m3, saved as micrograms per cubic metre.

The files live in NOAA's open-data bucket on AWS. Each full file is ~150 MB,
but every file comes with an .idx index, so only the byte range holding the
smoke field (~1-2 MB) is downloaded. For each hour the 1-hour forecast from
the previous run is used, the usual choice for HRRR-Smoke.

Each hour is cached in data/cache/hrrr/, so an interrupted run picks up
where it stopped.

    pixi run smoke
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

import config

BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
FIELD = ":MASSDEN:8 m above ground:"
CACHE = config.DATA / "cache" / "hrrr"
MARGIN_DEG = 0.6


def hours() -> list[dt.datetime]:
    t = dt.datetime.fromisoformat(config.START).replace(tzinfo=dt.timezone.utc)
    end = dt.datetime.fromisoformat(config.END).replace(tzinfo=dt.timezone.utc) + dt.timedelta(hours=23)
    out = []
    while t <= end:
        out.append(t)
        t += dt.timedelta(hours=config.STEP_HOURS)
    return out


def file_url(valid: dt.datetime) -> str:
    run = valid - dt.timedelta(hours=1)
    return f"{BUCKET}/hrrr.{run:%Y%m%d}/conus/hrrr.t{run:%H}z.wrfsfcf01.grib2"


def get(url: str, headers: dict | None = None, tries: int = 4) -> bytes | None:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * (k + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"Could not download {url}")


def byte_range(idx_text: str) -> tuple[int, int | None] | None:
    """Start and end byte of the smoke field from an .idx file."""
    lines = [ln for ln in idx_text.splitlines() if ln.strip()]
    for k, ln in enumerate(lines):
        if FIELD in ln:
            start = int(ln.split(":")[1])
            end = int(lines[k + 1].split(":")[1]) - 1 if k + 1 < len(lines) else None
            return start, end
    return None


def decode(message: bytes, want_grid: bool = False):
    """GRIB2 message -> values (Ny, Nx) and, if asked, latitudes/longitudes."""
    import eccodes
    gid = eccodes.codes_new_from_message(message)
    try:
        nx, ny = eccodes.codes_get(gid, "Nx"), eccodes.codes_get(gid, "Ny")
        vals = eccodes.codes_get_values(gid).reshape(ny, nx)
        if not want_grid:
            return vals, None, None
        lat = eccodes.codes_get_array(gid, "latitudes").reshape(ny, nx)
        lon = eccodes.codes_get_array(gid, "longitudes").reshape(ny, nx)
        lon = np.where(lon > 180, lon - 360, lon)
        return vals, lat, lon
    finally:
        eccodes.codes_release(gid)


def crop_box(lat: np.ndarray, lon: np.ndarray) -> tuple[slice, slice]:
    inside = ((lat >= config.SOUTH - MARGIN_DEG) & (lat <= config.NORTH + MARGIN_DEG)
              & (lon >= config.WEST - MARGIN_DEG) & (lon <= config.EAST + MARGIN_DEG))
    rows, cols = np.where(inside)
    if rows.size == 0:
        raise RuntimeError("The HRRR grid doesn't cover Washington? Check the download.")
    return slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1)


def fetch_hour(valid: dt.datetime) -> tuple[str, bytes | None]:
    """Download the smoke message for one hour (or None if that run is missing)."""
    url = file_url(valid)
    idx = get(url + ".idx")
    if idx is None:
        return "missing", None
    rng = byte_range(idx.decode("utf-8", "replace"))
    if rng is None:
        return "no-smoke-field", None
    start, end = rng
    data = get(url, {"Range": f"bytes={start}-{'' if end is None else end}"})
    return ("ok" if data else "missing"), data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild the combined file from the cache")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = config.SMOKE
    if out.exists() and not args.force:
        print(f"Smoke already present: {out}")
        return 0
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = hours()
    grid_path = CACHE / "grid.npz"

    # The grid: from the first hour that downloads, then cropped to Washington.
    if not grid_path.exists():
        for valid in todo:
            status, msg = fetch_hour(valid)
            if status == "no-smoke-field":
                print(f"{file_url(valid)}.idx has no {FIELD.strip(':')} line. HRRR may have renamed it.")
                return 1
            if msg:
                vals, lat, lon = decode(msg, want_grid=True)
                rs, cs = crop_box(lat, lon)
                np.savez(grid_path, lat=lat[rs, cs], lon=lon[rs, cs],
                         rows=np.array([rs.start, rs.stop]), cols=np.array([cs.start, cs.stop]))
                np.save(CACHE / f"{valid:%Y%m%d%H}.npy", (vals[rs, cs] * 1e9).astype(np.float16))
                break
        else:
            print("No HRRR files found for the date range.")
            return 1
    g = np.load(grid_path)
    rs, cs = slice(*g["rows"]), slice(*g["cols"])

    need = [v for v in todo if not (CACHE / f"{v:%Y%m%d%H}.npy").exists()
            and not (CACHE / f"{v:%Y%m%d%H}.missing").exists()]
    print(f"{len(todo)} hours of HRRR-Smoke ({config.START} to {config.END} UTC), "
          f"{len(todo) - len(need)} cached, {len(need)} to download...")
    t0, done = time.time(), 0

    def work(valid):
        status, msg = fetch_hour(valid)
        if msg is None:
            (CACHE / f"{valid:%Y%m%d%H}.missing").write_text(status)
            return valid, status
        vals, _, _ = decode(msg)
        np.save(CACHE / f"{valid:%Y%m%d%H}.npy", (vals[rs, cs] * 1e9).astype(np.float16))
        return valid, status

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(work, v) for v in need]):
            valid, status = fut.result()
            done += 1
            if status != "ok":
                print(f"\n  {valid:%Y-%m-%d %H}Z: {status}")
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(need)}  ~{(len(need) - done) / max(rate, 1e-6) / 60:.0f} min left", end="\r")
    print()

    # Stack, filling any missing hours from their neighbours in time.
    shape = g["lat"].shape
    cube = np.full((len(todo), *shape), np.nan, dtype=np.float32)
    for k, v in enumerate(todo):
        f = CACHE / f"{v:%Y%m%d%H}.npy"
        if f.exists():
            cube[k] = np.load(f).astype(np.float32)
    have = ~np.isnan(cube[:, 0, 0])
    if not have.any():
        print("No smoke data downloaded.")
        return 1
    if not have.all():
        idx = np.arange(len(todo))
        for j in range(shape[0]):
            for i in range(shape[1]):
                cube[~have, j, i] = np.interp(idx[~have], idx[have], cube[have, j, i])
        print(f"Filled {int((~have).sum())} missing hour(s) from neighbouring hours.")
    cube = np.clip(cube, 0, None)
    times = np.array([f"{v:%Y-%m-%dT%H:00}" for v in todo])
    np.savez_compressed(out, lat=g["lat"], lon=g["lon"], times_utc=times, smoke=cube.astype(np.float16))
    k = np.unravel_index(np.nanargmax(cube), cube.shape)
    print(f"Saved {out.name}: {len(todo)} hours on a {shape[1]}x{shape[0]} grid, peak "
          f"{cube[k]:.0f} ug/m3 at {g['lat'][k[1], k[2]]:.2f}N {-g['lon'][k[1], k[2]]:.2f}W, {times[k[0]]} UTC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
