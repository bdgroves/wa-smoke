"""Download the fires: satellite detections and official perimeters.

Detections: NASA FIRMS, VIIRS on Suomi-NPP, NOAA-20 and NOAA-21. Each
satellite passes over Washington about twice a day and flags 375 m pixels
that are burning. Accumulated night by night they show each fire growing.
FIRMS needs a free MAP_KEY: request one at
https://firms.modaps.eosdis.nasa.gov/api/map_key/ and either set the
FIRMS_MAP_KEY environment variable or save it in data/firms_key.txt.

Perimeters: the National Interagency Fire Center's WFIGS service, the
official mapped outline of each fire (latest version). No key needed.

    pixi run fires
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import config

FIRMS = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{days}/{date}"
SOURCES = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "VIIRS_SNPP_SP", "VIIRS_NOAA20_SP"]
WINDOW_DAYS = 5
WFIGS = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
         "WFIGS_Interagency_Perimeters/FeatureServer/0/query")
KEEP = ["latitude", "longitude", "acq_date", "acq_time", "frp", "confidence", "satellite"]


def firms_key(arg: str | None) -> str | None:
    if arg:
        return arg.strip()
    if os.environ.get("FIRMS_MAP_KEY"):
        return os.environ["FIRMS_MAP_KEY"].strip()
    f = config.DATA / "firms_key.txt"
    return f.read_text().strip() if f.exists() else None


def get_text(url: str, tries: int = 4) -> str:
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:                       # noqa: BLE001 - network hiccups, retry
            err = e
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"{url.split('?')[0]}: {err}")


def detections(key: str) -> int:
    bbox = f"{config.WEST},{config.SOUTH},{config.EAST},{config.NORTH}"
    start = dt.date.fromisoformat(config.START)
    end = dt.date.fromisoformat(config.END)
    rows, seen = [], set()
    for source in SOURCES:
        got = 0
        day = start
        while day <= end:
            days = min(WINDOW_DAYS, (end - day).days + 1)
            url = FIRMS.format(key=key, source=source, bbox=bbox, days=days, date=day.isoformat())
            text = get_text(url)
            header = text.splitlines()[0] if text.strip() else ""
            if "latitude" not in header:
                msg = text.strip()[:200]
                if "invalid" in msg.lower() and "key" in msg.lower():
                    print(f"FIRMS says the MAP_KEY is invalid: {msg}")
                    return -1
                # a source with no data for these dates just returns an error line; move on
                day += dt.timedelta(days=days)
                continue
            for r in csv.DictReader(io.StringIO(text)):
                k = (r["latitude"], r["longitude"], r["acq_date"], r["acq_time"], r.get("satellite", source))
                if k in seen:
                    continue
                seen.add(k)
                rows.append({f: r.get(f, "") for f in KEEP} | {"satellite": r.get("satellite", source)})
                got += 1
            day += dt.timedelta(days=days)
            time.sleep(0.3)
        print(f"  {source}: {got} detections")
    rows.sort(key=lambda r: (r["acq_date"], r["acq_time"].zfill(4)))
    with open(config.DETECTIONS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEEP)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def perimeters() -> int:
    features, offset = [], 0
    for where in (f"poly_GISAcres >= {config.FIRE_MIN_ACRES}", "1=1"):
        features, offset = [], 0
        try:
            while True:
                params = {
                    "where": where, "outFields": "*", "returnGeometry": "true", "outSR": 4326, "f": "geojson",
                    "geometry": f"{config.WEST},{config.SOUTH},{config.EAST},{config.NORTH}",
                    "geometryType": "esriGeometryEnvelope", "inSR": 4326,
                    "spatialRel": "esriSpatialRelIntersects", "resultOffset": offset, "resultRecordCount": 500,
                }
                page = json.loads(get_text(WFIGS + "?" + urllib.parse.urlencode(params)))
                if "error" in page:
                    raise RuntimeError(page["error"])
                features += page.get("features", [])
                if not page.get("properties", {}).get("exceededTransferLimit") and len(page.get("features", [])) < 500:
                    break
                offset += 500
            break
        except RuntimeError as e:
            print(f"  WFIGS query '{where}' failed ({e}); trying a simpler one...")
    start = dt.datetime.fromisoformat(config.START).replace(tzinfo=dt.timezone.utc) - dt.timedelta(days=45)
    end = dt.datetime.fromisoformat(config.END).replace(tzinfo=dt.timezone.utc) + dt.timedelta(days=1)
    keep = []
    for f in features:
        p = f.get("properties") or {}
        found = p.get("attr_FireDiscoveryDateTime") or p.get("poly_PolygonDateTime")
        acres = p.get("poly_GISAcres") or p.get("attr_IncidentSize") or 0
        if not found or not f.get("geometry"):
            continue
        when = dt.datetime.fromtimestamp(found / 1000, tz=dt.timezone.utc)
        if not (start <= when <= end) or acres < config.FIRE_MIN_ACRES:
            continue
        name = (p.get("attr_IncidentName") or p.get("poly_IncidentName") or "Unnamed").strip().title()
        contained = p.get("attr_ContainmentDateTime")
        keep.append({"type": "Feature", "geometry": f["geometry"], "properties": {
            "name": name, "acres": round(float(acres)), "discovered": when.isoformat(),
            "contained": (dt.datetime.fromtimestamp(contained / 1000, tz=dt.timezone.utc).isoformat()
                          if contained else None)}})
    keep.sort(key=lambda f: -f["properties"]["acres"])
    config.PERIMETERS.write_text(json.dumps({"type": "FeatureCollection", "features": keep}))
    for f in keep[:12]:
        p = f["properties"]
        print(f"  {p['name']:<28} {p['acres']:>9,} acres  found {p['discovered'][:10]}")
    return len(keep)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", help="FIRMS MAP_KEY (or set FIRMS_MAP_KEY, or save it in data/firms_key.txt)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    config.DATA.mkdir(parents=True, exist_ok=True)

    if config.PERIMETERS.exists() and not args.force:
        print(f"Perimeters already present: {config.PERIMETERS.name}")
    else:
        print("Fire perimeters from NIFC WFIGS...")
        print(f"Saved {perimeters()} perimeters of {config.FIRE_MIN_ACRES:,}+ acres")

    if config.DETECTIONS.exists() and not args.force:
        print(f"Detections already present: {config.DETECTIONS.name}")
        return 0
    key = firms_key(args.key)
    if not key:
        print("\nNo FIRMS MAP_KEY, so no satellite fire detections (the video still works, fires just\n"
              "won't grow night by night). Get a free key at https://firms.modaps.eosdis.nasa.gov/api/map_key/\n"
              "then:  pixi run fires --key YOURKEY   (it's saved for next time)")
        return 0
    if args.key:
        (config.DATA / "firms_key.txt").write_text(args.key.strip())
    print("Satellite fire detections from NASA FIRMS (VIIRS)...")
    n = detections(key)
    if n < 0:
        return 1
    print(f"Saved {n} detections to {config.DETECTIONS.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
