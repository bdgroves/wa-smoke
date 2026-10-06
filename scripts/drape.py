"""Build the image draped over the terrain for any moment in the animation.

Layers, bottom to top:

1. Land: a muted late-summer palette. Forest green west of the Cascades,
   dry tan to the east, grey rock and snow up high. Outside Washington it's
   faded toward charcoal so the state carries the story.
2. Burn scars: every satellite fire detection (NASA FIRMS, VIIRS, 375 m)
   up to this moment, in charcoal, so each fire grows night by night.
3. Smoke: HRRR-Smoke near the ground, µg/m³, as a haze whose opacity rises
   with concentration and whose colour goes from pale grey to brown.
4. Active fire: detections from the last 12 hours glow orange through the smoke.
5. At the end, the official fire perimeters are traced in.

Water is painted key green and swapped for blue after rendering (see finish.py).
"""
from __future__ import annotations

import csv
import re
import datetime as dt
import json

import numpy as np
import rasterio
from rasterio import features
from rasterio.warp import transform, transform_geom
from scipy import ndimage

import config

HRRR_LCC = "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +R=6371229 +units=m +no_defs"
PDT = dt.timezone(dt.timedelta(hours=-7), "PDT")   # the whole window is in daylight time

OCEAN_KEY = np.array([0, 210, 0], dtype=np.float32)
OCEAN_PREVIEW = np.array([14, 38, 76], dtype=np.float32)
OUTSIDE_DIM = 0.62
OUTSIDE_TO = np.array([44, 47, 54], dtype=np.float32)

# land: (elevation m, west-side colour, east-side colour)
LAND = [
    (0, (74, 98, 72), (142, 126, 94)),
    (600, (62, 88, 64), (124, 112, 84)),
    (1400, (78, 92, 76), (108, 104, 86)),
    (2200, (128, 126, 118), (132, 128, 118)),
    (2900, (205, 206, 204), (205, 206, 204)),
]
SCAR = np.array([38, 26, 22], dtype=np.float32)
EMBER_LOW = np.array([255, 120, 20], dtype=np.float32)
EMBER_HIGH = np.array([255, 226, 140], dtype=np.float32)
ACTIVE_HOURS = 12.0
PERIMETER = np.array([255, 176, 92], dtype=np.float32)

# smoke: (µg/m³, colour); opacity = 1 - exp(-c / SMOKE_SCALE), capped
SMOKE_RAMP = [(0, (206, 208, 210)), (35, (200, 196, 188)), (150, (178, 150, 116)),
              (400, (150, 108, 70)), (1000, (120, 78, 46))]
SMOKE_SCALE = 90.0
SMOKE_OPACITY_MAX = 0.78
SMOKE_CODE_MAX = 1500.0       # µg/m³ at the top of the smoke pass's colour code (log scale)


def ramp(v: np.ndarray, stops) -> np.ndarray:
    xs = np.array([s for s, _ in stops], dtype=np.float32)
    cs = np.array([c for _, c in stops], dtype=np.float32)
    v = np.clip(v, xs[0], xs[-1])
    return np.stack([np.interp(v, xs, cs[:, k]) for k in range(3)], axis=-1)


def smoke_alpha(c: np.ndarray) -> np.ndarray:
    return np.clip(1 - np.exp(-np.asarray(c, dtype=np.float32) / SMOKE_SCALE), 0, SMOKE_OPACITY_MAX)


def smoke_color(c: np.ndarray) -> np.ndarray:
    return ramp(np.asarray(c, dtype=np.float32), SMOKE_RAMP)


def utc(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)


class Drape:
    def __init__(self, water_key: bool = True):
        self.water = OCEAN_KEY if water_key else OCEAN_PREVIEW
        d = np.load(config.SMOKE)
        self.glat, self.glon, self.smoke = d["lat"], d["lon"], d["smoke"]
        self.times = [utc(s) for s in d["times_utc"]]
        self.t0 = self.times[0]
        self.step = (self.times[1] - self.times[0]).total_seconds() / 3600 if len(self.times) > 1 else 1.0
        with rasterio.open(config.DEM) as src:
            self.dem = src.read(1).astype(np.float32)
            self.tf, self.crs = src.transform, src.crs
        h, w = self.dem.shape
        self.shape = (h, w)
        cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        xs = (self.tf.c + cols * self.tf.a).ravel()
        ys = (self.tf.f + rows * self.tf.e).ravel()
        lons, lats = transform(self.crs, "EPSG:4326", xs, ys)
        self.lons, self.lats = np.asarray(lons).reshape(h, w), np.asarray(lats).reshape(h, w)
        self.gi, self.gj = self._grid_index()
        self.ocean = self.dem <= 0.5
        self.inside, self.border = self._washington()
        self.base = self._land()
        self._load_fires()

    # ------------------------------------------------------------ geometry

    def _grid_index(self):
        """Fractional (row, col) into the HRRR grid for every DEM pixel."""
        gx, gy = transform("EPSG:4326", HRRR_LCC, self.glon.ravel().tolist(), self.glat.ravel().tolist())
        gx, gy = np.asarray(gx).reshape(self.glat.shape), np.asarray(gy).reshape(self.glat.shape)
        x0, dx = gx[0, 0], np.mean(np.diff(gx, axis=1))
        y0, dy = gy[0, 0], np.mean(np.diff(gy, axis=0))
        px, py = transform("EPSG:4326", HRRR_LCC, self.lons.ravel().tolist(), self.lats.ravel().tolist())
        px, py = np.asarray(px).reshape(self.shape), np.asarray(py).reshape(self.shape)
        # check the grid really is regular in that projection; if not, fall back to nearest point
        cc, rr = np.meshgrid(np.arange(gx.shape[1]), np.arange(gx.shape[0]))
        err = max(np.abs(x0 + cc * dx - gx).max() / abs(dx), np.abs(y0 + rr * dy - gy).max() / abs(dy))
        if err < 0.25:
            return (py - y0) / dy, (px - x0) / dx
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([gx.ravel(), gy.ravel()]))
        _, k = tree.query(np.column_stack([px.ravel(), py.ravel()]))
        r, c = np.unravel_index(k, gx.shape)
        return r.reshape(self.shape).astype(np.float32), c.reshape(self.shape).astype(np.float32)

    def _washington(self):
        gj = json.loads(config.BOUNDARY.read_text())
        geoms = [transform_geom("EPSG:4326", self.crs, f["geometry"]) for f in gj["features"]]
        inside = features.rasterize(geoms, out_shape=self.shape, transform=self.tf).astype(bool)
        edge = inside ^ ndimage.binary_erosion(inside, iterations=1)
        edge = ndimage.binary_dilation(edge, iterations=1) & ~self.ocean
        return inside, edge

    def _land(self) -> np.ndarray:
        z = np.where(self.ocean, 0, self.dem)
        zs = np.array([s for s, _, _ in LAND], dtype=np.float32)
        west = np.stack([np.interp(z, zs, [c[k] for _, c, _ in LAND]) for k in range(3)], axis=-1)
        east = np.stack([np.interp(z, zs, [c[k] for _, _, c in LAND]) for k in range(3)], axis=-1)
        # the Cascade crest, roughly: green to the west, dry to the east
        e = 1 / (1 + np.exp(-(self.lons + 121.05) / 0.25))
        rgb = west * (1 - e[..., None]) + east * e[..., None]
        outside = ~self.inside & ~self.ocean
        rgb[outside] = rgb[outside] * (1 - OUTSIDE_DIM) + OUTSIDE_TO * OUTSIDE_DIM
        return rgb.astype(np.float32)

    # ------------------------------------------------------------ fires

    def _load_fires(self):
        h, w = self.shape
        self.det_rc = np.zeros((0, 2), dtype=np.int32)
        self.det_hour = np.zeros(0, dtype=np.float32)
        self.det_frp = np.zeros(0, dtype=np.float32)
        if config.DETECTIONS.exists():
            lat, lon, when, frp = [], [], [], []
            with open(config.DETECTIONS, newline="") as f:
                for r in csv.DictReader(f):
                    t = r["acq_time"].zfill(4)
                    moment = dt.datetime.fromisoformat(f"{r['acq_date']}T{t[:2]}:{t[2:]}").replace(tzinfo=dt.timezone.utc)
                    lat.append(float(r["latitude"]))
                    lon.append(float(r["longitude"]))
                    when.append((moment - self.t0).total_seconds() / 3600)
                    frp.append(float(r.get("frp") or 0))
            if lat:
                x, y = transform("EPSG:4326", self.crs, lon, lat)
                rr = ((np.asarray(y) - self.tf.f) / self.tf.e).astype(int)
                cc = ((np.asarray(x) - self.tf.c) / self.tf.a).astype(int)
                ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                self.det_rc = np.column_stack([rr[ok], cc[ok]]).astype(np.int32)
                self.det_hour = np.asarray(when, dtype=np.float32)[ok]
                self.det_frp = np.asarray(frp, dtype=np.float32)[ok]
        # a VIIRS pixel is 375 m; stamp each detection as a small disc on the 250 m DEM
        self.stamp = ndimage.generate_binary_structure(2, 1)

        self.fires, self.perimeter_edge = [], np.zeros(self.shape, dtype=bool)
        if not config.PERIMETERS.exists():
            return
        gj = json.loads(config.PERIMETERS.read_text())
        for f in gj["features"]:
            p = f["properties"]
            geom = transform_geom("EPSG:4326", self.crs, f["geometry"])
            mask = features.rasterize([geom], out_shape=self.shape, transform=self.tf).astype(bool)
            if not mask.any():
                continue
            edge = mask ^ ndimage.binary_erosion(mask, iterations=1)
            self.perimeter_edge |= ndimage.binary_dilation(edge, iterations=1)
            # tidy names like "0476 Hoag" or "Tucannon Mutual Aid"
            name = re.sub(r"^\d+\s+", "", p["name"]).replace(" Mutual Aid", "").strip()
            found = (utc(p["discovered"][:19]) - self.t0).total_seconds() / 3600
            burning = found >= 0              # reported during the window...
            # ...or seen burning by satellite during it: first detection inside the perimeter
            if len(self.det_hour):
                hit = mask[self.det_rc[:, 0], self.det_rc[:, 1]]
                if hit.any():
                    found = float(self.det_hour[hit].min())
                    burning = True
            rows, cols = np.nonzero(mask)
            # label at the perimeter's pixel nearest its centroid (so it sits on the burn)
            cr, cc = rows.mean(), cols.mean()
            k = int(np.argmin((rows - cr) ** 2 + (cols - cc) ** 2))
            x = self.tf.c + (cols[k] + 0.5) * self.tf.a
            y = self.tf.f + (rows[k] + 0.5) * self.tf.e
            lon, lat = transform(self.crs, "EPSG:4326", [x], [y])
            in_wa = bool(self.inside[rows[k], cols[k]])
            self.fires.append({"name": name, "acres": p["acres"], "hour": max(found, 0.0), "lat": lat[0], "lon": lon[0],
                               "label": p["acres"] >= config.FIRE_LABEL_ACRES and in_wa and burning,
                               "mask": mask if not len(self.det_hour) else None})
        # complexes: one label, on the biggest member, with the acres summed and the earliest start
        for complex_name, members in config.FIRE_COMPLEXES.items():
            parts = [f for f in self.fires if f["name"].lower() in {m.lower() for m in members}]
            if not parts:
                continue
            lead = max(parts, key=lambda f: f["acres"])
            total, start = sum(f["acres"] for f in parts), min(f["hour"] for f in parts)
            for f in parts:
                f["label"] = False
            self.fires.append(dict(lead, name=complex_name, acres=total, hour=start, label=True, mask=None))

    # ------------------------------------------------------------ per frame

    def hours(self) -> float:
        """Length of the record in hours (last sample's offset from the first)."""
        return (self.times[-1] - self.t0).total_seconds() / 3600

    def field(self, hour: float) -> np.ndarray:
        """Smoke (µg/m³) at every DEM pixel, `hour` hours after the first sample."""
        k = hour / self.step
        k0 = int(np.clip(np.floor(k), 0, len(self.times) - 1))
        k1 = min(k0 + 1, len(self.times) - 1)
        f = float(np.clip(k - k0, 0, 1))
        g = self.smoke[k0].astype(np.float32) * (1 - f) + self.smoke[k1].astype(np.float32) * f
        return np.clip(ndimage.map_coordinates(g, [self.gi, self.gj], order=1, mode="nearest"), 0, None)

    def image(self, hour: float, outline: float = 0.0, smoke: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """The drape for one moment. With smoke=False the smoke is left out of the
        image (finish.py lays it over the render instead, as an unshaded veil)."""
        rgb = self.base.copy()

        # smoke
        c = self.field(hour)
        a = smoke_alpha(c)[..., None] if smoke else np.zeros((*self.shape, 1), dtype=np.float32)
        rgb = rgb * (1 - a) + smoke_color(c) * a

        # burn scars so far: on top of the smoke (half strength) so the burned ground stays readable
        seen = self.det_hour <= hour
        scar = np.zeros(self.shape, dtype=bool)
        if seen.any():
            scar[self.det_rc[seen, 0], self.det_rc[seen, 1]] = True
            scar = ndimage.binary_dilation(scar, structure=self.stamp)
        for fire in self.fires:              # no detections file: show each burn once it's reported
            if fire["mask"] is not None and fire["hour"] <= hour:
                scar |= fire["mask"]
        if scar.any():
            k = (0.8 - 0.35 * a[..., 0])[scar][:, None]
            rgb[scar] = rgb[scar] * (1 - k) + SCAR * k

        # active fire glows through the smoke, fading over ACTIVE_HOURS
        active = seen & (self.det_hour > hour - ACTIVE_HOURS)
        if active.any():
            heat = np.zeros(self.shape, dtype=np.float32)
            age = 1 - (hour - self.det_hour[active]) / ACTIVE_HOURS
            power = np.clip(np.log1p(self.det_frp[active]) / np.log1p(200), 0.25, 1)
            np.maximum.at(heat, (self.det_rc[active, 0], self.det_rc[active, 1]), age * power)
            heat = ndimage.grey_dilation(heat, footprint=self.stamp)
            ember = EMBER_LOW + (EMBER_HIGH - EMBER_LOW) * heat[..., None]
            k = np.where(heat > 0, np.clip(0.6 + heat, 0, 1), 0)[..., None]
            rgb = rgb * (1 - k) + ember * k

        if outline > 0:
            e = self.perimeter_edge
            rgb[e] = rgb[e] * (1 - outline) + PERIMETER * outline

        rgb[self.ocean] = self.water
        line = self.border & ~self.ocean
        rgb[line] = rgb[line] * 0.3 + 235 * 0.7
        return np.clip(rgb, 0, 255).astype(np.uint8), c

    @staticmethod
    def smoke_code(c: np.ndarray) -> np.ndarray:
        """Smoke concentration encoded as colour for a second, smoke-only render.

        The viewer shades every pixel by a single brightness factor, which scales
        red and blue together, so the ratio red / (red + blue) survives the
        lighting: finish.py reads the concentration back from that ratio
        (log scale, so 5 and 500 µg/m³ both get resolution).
        """
        v = np.clip(np.log1p(c) / np.log1p(SMOKE_CODE_MAX), 0, 1)
        return np.stack([255 * v, np.zeros_like(v), 255 * (1 - v)], axis=-1).astype(np.uint8)

    def time_label(self, hour: float) -> str:
        """Local (Pacific daylight) time for the clock, ISO format."""
        return (self.t0 + dt.timedelta(hours=hour)).astimezone(PDT).strftime("%Y-%m-%dT%H:%M")
