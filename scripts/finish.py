"""Finish the smoke frames: sky, water, glow on active fires, clock and
timeline, city smoke readings, fire names, legend and credits, then the MP4.

render.py --export runs this automatically. Run it alone to restyle frames
already rendered:  pixi run finish
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.warp import transform as warp
from scipy import ndimage

import config
from drape import SMOKE_CODE_MAX, smoke_alpha, smoke_color

SKY = ((5, 9, 20), (36, 44, 64))             # zenith, horizon: a dark night-blue backdrop
WATER = ((10, 30, 62), (22, 58, 100))        # far, near: deep Pacific blue
TITLE = "Washington’s summer of smoke, 2026"
SUBTITLE = "Wildfire smoke near the ground (NOAA HRRR-Smoke) · fires seen by satellite (NASA VIIRS)"
CREDIT = ("Data: NOAA HRRR-Smoke · NASA FIRMS · NIFC fire perimeters · USGS 3DEP · Natural Earth   "
          "Rendered with forge3d · brooksgroves.com")

# EPA PM2.5 categories, for the colour of each city's reading
AQI = [(12, "Good 0\u201312", (84, 168, 92)), (35.4, "Moderate 12\u201335", (236, 196, 48)),
       (55.4, "Sensitive groups 35\u201355", (240, 128, 36)), (150.4, "Unhealthy 55\u2013150", (214, 48, 48)),
       (250.4, "Very unhealthy 150\u2013250", (143, 63, 151)), (1e9, "Hazardous 250+", (126, 0, 35))]
LEGEND_LAND = (104, 110, 92)       # the smoke legend is drawn over a typical land colour
FIRE_TEXT = (255, 200, 150, 255)


def font(name: str, px: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(config.FONTS / name), max(8, int(px)))


# ---------------------------------------------------------------- camera

def camera_matrices(cam: dict, width: int, height: int):
    """forge3d terrain viewer: eye = target + r(sinθcosφ, cosθ, sinθsinφ), y up, vertical FOV."""
    t = np.array(cam["target"], dtype=np.float64)
    phi, theta = np.radians(cam["phi"]), np.radians(cam["theta"])
    eye = t + cam["radius"] * np.array([np.sin(theta) * np.cos(phi), np.cos(theta),
                                        np.sin(theta) * np.sin(phi)])
    f = (t - eye) / np.linalg.norm(t - eye)
    s = np.cross(f, [0.0, 1.0, 0.0]); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    view = np.array([[*s, -s @ eye], [*u, -u @ eye], [*-f, f @ eye], [0, 0, 0, 1]])
    fy = 1.0 / np.tan(np.radians(cam["fov"]) / 2)
    return view, fy / (width / height), fy


def project(p, view, fx, fy, width, height):
    c = view @ np.array([*p, 1.0])
    depth = -c[2]
    if depth <= 1.0:
        return None
    return ((c[0] * fx / depth + 1) * 0.5 * width, (1 - c[1] * fy / depth) * 0.5 * height)


# ---------------------------------------------------------------- sky

def sky_image(width: int, height: int) -> np.ndarray:
    top, horizon = (np.array(c, dtype=np.float32) for c in SKY)
    v = np.linspace(0, 1, height, dtype=np.float32)[:, None] ** 0.7
    rows = top[None, :] * (1 - v) + horizon[None, :] * v
    return np.repeat(rows[:, None, :], width, axis=1)


def water_mask(rgb: np.ndarray) -> np.ndarray:
    """Soft alpha for the key-green water drape.py paints (no temperature colour is green)."""
    r, g, b = (rgb[..., k].astype(np.float32) for k in range(3))
    green = g - np.maximum(r, b)
    return np.clip((green - 18) / 40, 0, 1)


GRADE_SATURATION = 1.0
GRADE_GAMMA = 0.88       # <1 lifts the shaded flats a little on the dark background


def grade(rgb: np.ndarray) -> np.ndarray:
    """Colour grade for the terrain: richer and a touch darker, so 110 °F reads crimson, not salmon."""
    lum = rgb.mean(axis=-1, keepdims=True)
    out = np.clip(lum + (rgb - lum) * GRADE_SATURATION, 0, 255)
    return 255.0 * (out / 255.0) ** GRADE_GAMMA


def water_image(width: int, height: int) -> np.ndarray:
    far, near = (np.array(c, dtype=np.float32) for c in WATER)
    v = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    rows = far[None, :] * (1 - v) + near[None, :] * v
    return np.repeat(rows[:, None, :], width, axis=1)


def sky_mask(rgb: np.ndarray) -> np.ndarray:
    """Soft alpha for the viewer's flat background, connected to the top edge."""
    bg = np.median(rgb[:4].reshape(-1, 3), axis=0)
    near = np.all(np.abs(rgb.astype(np.int16) - bg.astype(np.int16)) <= 8, axis=2)
    lab, n = ndimage.label(near)
    # the sky (touching the top edge) plus any big patch of background showing
    # below the terrain's near edge; small near-white specks stay terrain
    sizes = ndimage.sum(near, lab, np.arange(1, n + 1))
    big = np.flatnonzero(sizes > 0.002 * near.size) + 1
    mask = np.isin(lab, np.union1d(np.unique(lab[0][lab[0] > 0]), big))
    mask = ndimage.binary_dilation(mask, iterations=1)
    return ndimage.gaussian_filter(mask.astype(np.float32), 0.7)


# ---------------------------------------------------------------- overlays

def text(d: ImageDraw.ImageDraw, xy, s: str, f, fill=(255, 255, 255, 255), anchor="la", halo=3):
    d.text(xy, s, font=f, fill=fill, anchor=anchor, stroke_width=halo, stroke_fill=(10, 14, 20, 200))


def when(label: str) -> tuple[str, str]:
    t = dt.datetime.fromisoformat(label)
    hour = t.hour % 12 or 12
    return f"{t:%A}, {t:%B} {t.day}", f"{hour} {'AM' if t.hour < 12 else 'PM'} PDT"


def draw_timeline(d, x, y, w, sc, hour, total_hours, start_label):
    start = dt.datetime.fromisoformat(start_label)
    d.rounded_rectangle([x, y, x + w, y + 6 * sc], radius=3 * sc, fill=(255, 255, 255, 70))
    d.rounded_rectangle([x, y, x + w * min(hour / total_hours, 1), y + 6 * sc], radius=3 * sc,
                        fill=(255, 255, 255, 230))
    small = font("Roboto-Regular.ttf", 15 * sc)
    midnight = start.replace(hour=0, minute=0)
    day = midnight if midnight >= start else midnight + dt.timedelta(days=1)
    while (day - start).total_seconds() / 3600 <= total_hours:
        dx = x + w * ((day - start).total_seconds() / 3600) / total_hours
        if day.weekday() == 0:                      # a tick and label every Monday: "Aug 3"
            d.line([(dx, y - 4 * sc), (dx, y + 10 * sc)], fill=(255, 255, 255, 170), width=max(1, int(sc)))
            text(d, (dx, y + 14 * sc), f"{day:%b} {day.day}", small, anchor="ma", halo=2)
        day += dt.timedelta(days=1)


def aqi_color(v: float):
    for top, _, col in AQI:
        if v <= top:
            return col
    return AQI[-1][2]


def draw_aqi_key(d, x, y, w, sc) -> float:
    """Key for the city chips: EPA Air Quality Index categories for PM2.5, wrapped to width w.
    Drawn upward from y (the bottom of the key); returns the top."""
    f = font("Roboto-Regular.ttf", 14 * sc)
    head = font("Roboto-Medium.ttf", 15 * sc)
    chip, gap = 11 * sc, 14 * sc
    items, rows, row, used = [], [], [], 0.0
    for _, name, col in AQI:
        iw = chip + 5 * sc + f.getlength(name)
        if row and (used + gap + iw > w or len(row) == 3):     # three to a row
            rows.append(row)
            row, used = [], 0.0
        row.append((name, col, iw))
        used += (gap if len(row) > 1 else 0) + iw
    rows.append(row)
    line = 19 * sc
    top = y - line * len(rows) - 22 * sc
    text(d, (x, top), "City numbers: smoke \u00b5g/m\u00b3 \u00b7 EPA Air Quality Index (AQI) category",
         head, anchor="lt", halo=2)
    for r, items in enumerate(rows):
        cx, cy = x, top + 22 * sc + r * line + line / 2
        for name, col, iw in items:
            d.rounded_rectangle([cx, cy - chip / 2, cx + chip, cy + chip / 2], radius=2 * sc,
                                fill=tuple(col) + (255,), outline=(255, 255, 255, 200), width=max(1, int(sc)))
            text(d, (cx + chip + 5 * sc, cy), name, f, anchor="lm", halo=2)
            cx += iw + gap
    return top


def draw_legend(d, x, y, w, sc):
    h = 14 * sc
    vmax = 250.0
    vals = np.linspace(0, vmax, int(w))
    a = smoke_alpha(vals)[:, None]
    cols = np.array(LEGEND_LAND, dtype=np.float32) * (1 - a) + smoke_color(vals) * a
    for i, c in enumerate(cols):
        d.line([(x + i, y), (x + i, y + h)], fill=tuple(int(v) for v in c) + (255,))
    d.rectangle([x, y, x + w, y + h], outline=(255, 255, 255, 160), width=1)
    f = font("Roboto-Regular.ttf", 15 * sc)
    for v, lab in config.SMOKE_TICKS:
        tx = x + w * v / vmax
        d.line([(tx, y + h), (tx, y + h + 5 * sc)], fill=(255, 255, 255, 220), width=max(1, int(sc)))
        text(d, (tx, y + h + 7 * sc), lab, f, anchor="ma", halo=2)
    text(d, (x, y - 6 * sc), "Smoke near the ground, µg/m³", font("Roboto-Medium.ttf", 17 * sc),
         anchor="ld", halo=2)
    # a small key for the fire colours
    fx = x + w + 26 * sc
    r = 5 * sc
    d.ellipse([fx - r, y + h / 2 - r, fx + r, y + h / 2 + r], fill=(255, 170, 60, 255))
    text(d, (fx + 10 * sc, y + h / 2), "burning now", f, anchor="lm", halo=2)
    d.rectangle([fx - r, y + h + 10 * sc, fx + r, y + h + 10 * sc + 2 * r], fill=(40, 28, 24, 255),
                outline=(255, 255, 255, 120))
    text(d, (fx + 10 * sc, y + h + 10 * sc + r), "burned", f, anchor="lm", halo=2)


def draw_city(d, p, name, temp, sc, primary):
    x, y = p
    stem = (46 if primary else 34) * sc
    col = tuple(aqi_color(temp)) + (255,)
    d.line([(x, y), (x, y - stem)], fill=(255, 255, 255, 220), width=max(1, int(2 * sc)))
    r = 4.5 * sc
    d.ellipse([x - r, y - r, x + r, y + r], fill=col, outline=(255, 255, 255, 255), width=max(1, int(1.5 * sc)))
    name_f = font("Roboto-Medium.ttf", (27 if primary else 22) * sc)
    temp_f = font("Roboto-Medium.ttf", (27 if primary else 22) * sc)
    label, reading = f"{name} ", f"{temp:.0f}"
    lw, tw = name_f.getlength(label), temp_f.getlength(reading)
    left, base = x - (lw + tw) / 2, y - stem - 6 * sc
    text(d, (left, base), label, name_f, anchor="ls")
    # temperature in a chip of its own colour, so the number and the map agree
    pad = 5 * sc
    box = [left + lw - pad * 0.4, base - temp_f.size * 0.95, left + lw + tw + pad, base + pad * 0.9]
    d.rounded_rectangle(box, radius=4 * sc, fill=col, outline=(255, 255, 255, 230), width=max(1, int(sc)))
    dark = sum(col[:3]) > 480
    d.text((left + lw + pad * 0.3, base), reading, font=temp_f, anchor="ls",
           fill=(20, 20, 20, 255) if dark else (255, 255, 255, 255))
    return (left - pad, base - temp_f.size - pad, left + lw + tw + 2 * pad, y + r)


FIRE_SIDES = ("right", "left", "below", "above", "low")   # low: further down, clear of a city pin


def draw_fire(d, p, fire, sc, with_acres: bool, side: str = "right"):
    """Fire name beside a small ember marker; acres too once the perimeters are in.
    Returns the box the label covers."""
    x, y = p
    r = 4 * sc
    d.polygon([(x, y - r * 1.4), (x + r, y), (x, y + r * 1.4), (x - r, y)], fill=(255, 150, 50, 255),
              outline=(255, 235, 200, 255))
    f = font("Roboto-Medium.ttf", 19 * sc)
    small = font("Roboto-Regular.ttf", 15 * sc)
    label = fire["name"] + (" Fire" if "Complex" not in fire["name"] and "Fire" not in fire["name"] else "")
    acres = f"{fire['acres']:,} acres"
    w = max(f.getlength(label), small.getlength(acres) if with_acres else 0)
    h = (34 if with_acres else 20) * sc
    if side == "right":
        tx, ty, anchor = x + 9 * sc, y, "lm"
        box = (x - r, y - 12 * sc, tx + w + 4 * sc, y - 12 * sc + h + 4 * sc)
    elif side == "left":
        tx, ty, anchor = x - 9 * sc, y, "rm"
        box = (tx - w - 4 * sc, y - 12 * sc, x + r, y - 12 * sc + h + 4 * sc)
    elif side in ("below", "low"):
        drop = 18 if side == "below" else 40
        tx, ty, anchor = x, y + drop * sc, "mm"
        top = y - r * 1.4 if side == "below" else ty - 11 * sc
        box = (x - w / 2 - 4 * sc, top, x + w / 2 + 4 * sc, ty - 10 * sc + h + 4 * sc)
    else:
        tx, ty, anchor = x, y - 18 * sc - (19 * sc if with_acres else 0), "mm"
        box = (x - w / 2 - 4 * sc, ty - 12 * sc, x + w / 2 + 4 * sc, y + r * 1.4)
    text(d, (tx, ty), label, f, fill=FIRE_TEXT, anchor=anchor, halo=3)
    if with_acres:
        text(d, (tx, ty + 19 * sc), acres, small, fill=(255, 230, 210, 235), anchor=anchor, halo=2)
    return box


def ember_mask(rgb: np.ndarray) -> np.ndarray:
    """Actively burning pixels: the only strongly orange things in the render."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (np.clip((r - b - 60) / 50, 0, 1) * np.clip((r - 115) / 40, 0, 1)
            * np.clip((r / np.maximum(g, 1) - 1.25) / 0.2, 0, 1))


def glow(rgb: np.ndarray, ember: np.ndarray, core: np.ndarray, sc: float) -> np.ndarray:
    """Put the fires back on top of the smoke, with a soft orange bloom around them."""
    if ember.max() <= 0:
        return rgb
    e = ember[..., None]
    hot = np.array([255, 196, 110], dtype=np.float32)
    rgb = rgb * (1 - 0.9 * e) + hot * 0.9 * e
    halo = ndimage.gaussian_filter(ember, 3 * sc) * 3.0 + ndimage.gaussian_filter(ember, 12 * sc) * 6.0
    halo = np.clip(halo, 0, 1)[..., None]
    return np.clip(rgb + halo * np.array([255, 120, 30], dtype=np.float32) * 0.8, 0, 255)


def smoke_veil(code_path: Path, width: int, height: int, sc: float):
    """Opacity and colour of the smoke at every screen pixel, from the colour-coded smoke render."""
    code = np.asarray(Image.open(code_path).convert("RGB")).astype(np.float32)
    r, b = code[..., 0], code[..., 2]
    weight = np.clip((r + b) / 120.0, 0, 1)                   # dark (deep shadow) pixels count less
    ratio = r / np.maximum(r + b, 1e-3)
    num = np.asarray(Image.fromarray(ratio * weight).resize((width, height), Image.BILINEAR))
    den = np.asarray(Image.fromarray(weight).resize((width, height), Image.BILINEAR))
    num, den = ndimage.gaussian_filter(num, 3 * sc), ndimage.gaussian_filter(den, 3 * sc)
    v = np.clip(num / np.maximum(den, 1e-3), 0, 1) * (den > 0.02)
    c = np.expm1(v * np.log1p(SMOKE_CODE_MAX))
    return smoke_alpha(c)[..., None], smoke_color(c)


def overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ---------------------------------------------------------------- main

_CTX: dict = {}


def _init_worker(ctx: dict) -> None:
    """Per-process setup: the big constant images and fonts are built once per worker."""
    W, H, sc = ctx["W"], ctx["H"], ctx["sc"]
    ctx = dict(ctx)
    ctx["sky"], ctx["water"] = sky_image(W, H), water_image(W, H)
    ctx["fonts"] = {
        "title": font("Roboto-Medium.ttf", 40 * sc), "sub": font("Roboto-Regular.ttf", 21 * sc),
        "day": font("Roboto-Medium.ttf", 34 * sc), "hour": font("Roboto-Regular.ttf", 26 * sc),
        "credit": font("Roboto-Regular.ttf", 14 * sc),
    }
    _CTX.clear()
    _CTX.update(ctx)


def _finish_frame(job) -> int:
    """All the per-pixel work and drawing for one frame, following a precomputed label plan."""
    i, path, fm, plan = job
    c = _CTX
    W, H, sc, wide, sky, water, fnt = c["W"], c["H"], c["sc"], c["wide"], c["sky"], c["water"], c["fonts"]
    path = Path(path)
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    w = water_mask(rgb)[..., None]
    rgb = grade(rgb)
    # keep a little of the rendered shading on the water, in blue
    shade = np.clip(rgb.mean(axis=-1, keepdims=True) / 170.0, 0.75, 1.15)
    rgb = rgb * (1 - w) + water * shade * w
    # smoke over the land as a veil (unshaded, like haze), then the fires glowing through it
    ember = ember_mask(rgb)
    core = rgb.copy()
    code = path.with_name(path.name.replace("frame_", "smoke_"))
    a = sky_mask(rgb.astype(np.uint8))[..., None]
    if code.exists():
        sa, scol = smoke_veil(code, W, H, sc)
        rgb = rgb * (1 - sa) + scol * sa
    rgb = glow(rgb, ember, core, sc)
    img = Image.fromarray(np.clip(rgb * (1 - a) + sky * a, 0, 255).astype(np.uint8)).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    def faded(draw_fn, alpha):
        if alpha >= 0.99:
            draw_fn(d)                                   # fully shown: draw straight on
            return
        part = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_fn(ImageDraw.Draw(part))
        part.putalpha(part.getchannel("A").point(lambda v, a=alpha: int(v * a)))
        layer.alpha_composite(part)

    for item in plan:
        if item[0] == "city":
            _, k, p, alpha = item
            city = c["cities"][k]
            faded(lambda dd: draw_city(dd, p, city["name"], fm["cities"][k], sc, k == 0), alpha)
        else:
            _, n, p, side, alpha, acres = item
            fire = c["fires"][n]
            faded(lambda dd: draw_fire(dd, p, fire, sc, acres, side), alpha)

    # title, clock, timeline, legend, credit
    m = 44 * sc
    text(d, (m, m), TITLE, fnt["title"], anchor="lt")
    text(d, (m, m + 52 * sc), SUBTITLE, fnt["sub"], anchor="lt", halo=2)
    day, clock = when(fm["time"])
    if wide:
        text(d, (W - m, m), day, fnt["day"], anchor="rt")
        text(d, (W - m, m + 44 * sc), clock, fnt["hour"], anchor="rt")
        draw_timeline(d, W - m - 420 * sc, m + 88 * sc, 420 * sc, sc, fm["hour"], c["total_hours"], c["start"])
        draw_legend(d, m, H - m - 44 * sc, 480 * sc, sc)
        draw_aqi_key(d, m, H - m - 76 * sc, 560 * sc, sc)
        text(d, (W - m, H - m + 6 * sc), CREDIT, fnt["credit"], fill=(255, 255, 255, 215), anchor="rd", halo=2)
    else:
        # stacked: title, then the clock on one row, a full-width timeline under it
        text(d, (m, m + 96 * sc), day, fnt["day"], anchor="lt")
        text(d, (W - m, m + 100 * sc), clock, fnt["hour"], anchor="rt")
        draw_timeline(d, m, m + 150 * sc, W - 2 * m, sc, fm["hour"], c["total_hours"], c["start"])
        lw = min(W - 2 * m, 560 * sc)
        draw_legend(d, (W - lw) / 2, H - m - 78 * sc, lw, sc)
        draw_aqi_key(d, (W - lw) / 2, H - m - 110 * sc, lw, sc)
        left, right = CREDIT.split("   ")
        text(d, (W / 2, H - m - 14 * sc), left, fnt["credit"], fill=(255, 255, 255, 215), anchor="md", halo=2)
        text(d, (W / 2, H - m + 6 * sc), right, fnt["credit"], fill=(255, 255, 255, 215), anchor="md", halo=2)

    img.alpha_composite(layer)
    img.convert("RGB").save(Path(c["final"]) / path.name, compress_level=1)
    return i


def default_workers() -> int:
    """Leave a couple of cores free; each worker holds a few full-frame float images (~0.3 GB)."""
    return max(1, min(8, (os.cpu_count() or 2) - 2))


def finish(frames_dir: Path, out_mp4: Path, workers: int | None = None) -> bool:
    meta = json.loads((frames_dir / "frames.json").read_text())
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        print("No frames in", frames_dir)
        return False
    W, H = Image.open(frames[0]).size
    sc = min(W, H) / 1080
    wide = W / H > 1.5          # 16x9; square and portrait stack the layout
    top_reserve, bottom_reserve = (150 * sc, 0) if wide else (215 * sc, 215 * sc)
    total_hours = meta.get("total_hours") or max(f["hour"] for f in meta["frames"])
    start = meta.get("start") or meta["frames"][0]["time"]
    z = meta["z_scale"]
    ws = meta.get("world_scale", 1.0)
    with rasterio.open(meta["dem"]) as src:
        dem = src.read(1)
        base = float(dem.min())
        cities = meta["cities"]
        xs, ys = warp("EPSG:4326", src.crs, [c["lon"] for c in cities], [c["lat"] for c in cities])
        grounds = []
        for x, y in zip(xs, ys):
            r, c = src.index(x, y)
            h = float(dem[min(max(r, 0), dem.shape[0] - 1), min(max(c, 0), dem.shape[1] - 1)])
            grounds.append(np.array([x * ws, (h - base) * z, -y * ws]))
        fires = [f for f in meta.get("fires", []) if f["label"]]
        # complexes first (they're the story near the cities), then by size
        fires.sort(key=lambda f: (f["name"] not in config.FIRE_COMPLEXES, -f["acres"]))
        fire_grounds = []
        if fires:
            fxs, fys = warp("EPSG:4326", src.crs, [f["lon"] for f in fires], [f["lat"] for f in fires])
            for x, y in zip(fxs, fys):
                r, c = src.index(x, y)
                h = float(dem[min(max(r, 0), dem.shape[0] - 1), min(max(c, 0), dem.shape[1] - 1)])
                fire_grounds.append(np.array([x * ws, (h - base) * z, -y * ws]))
    order = sorted(range(len(cities)), key=lambda k: -cities[k]["priority"])
    fade = [1.0] * len(cities)

    # Pass 1, in order (labels fade in and out across frames): where every label goes.
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    plans = []
    for i, fm in enumerate(meta["frames"][:len(frames)]):
        view, fx, fy = camera_matrices(fm["camera"], W, H)
        placed, plan = [], []
        for k in order:
            temp = fm["cities"][k]
            p = project(grounds[k], view, fx, fy, W, H)
            show = (temp is not None and p is not None and 40 * sc < p[0] < W - 40 * sc
                    and top_reserve < p[1] < H - bottom_reserve)
            if show:
                box = draw_city(probe, p, cities[k]["name"], temp, sc, k == 0)
                show = not any(overlaps(box, b) for b in placed)
                if show:
                    placed.append(box)
            fade[k] = (1.0 if show else 0.0) if i == 0 else fade[k] + ((1.0 if show else 0.0) - fade[k]) * 0.25
            if p is not None and temp is not None and fade[k] > 0.02:
                plan.append(("city", k, (float(p[0]), float(p[1])), fade[k]))
        # fire names, once satellites first see them; acres added during the final hold
        for n, (f, g) in enumerate(zip(fires, fire_grounds)):
            age = fm["hour"] - f["hour"]
            if age < 0:
                continue
            p = project(g, view, fx, fy, W, H)
            if p is None or not (40 * sc < p[0] < W - 40 * sc and top_reserve < p[1] < H - bottom_reserve):
                continue
            acres = fm.get("outline", 0) > 0
            side = next((s for s in FIRE_SIDES
                         if not any(overlaps(draw_fire(probe, p, f, sc, acres, s), b) for b in placed)), None)
            if side is None:
                continue
            placed.append(draw_fire(probe, p, f, sc, acres, side))
            plan.append(("fire", n, (float(p[0]), float(p[1])), side, min(1.0, age / 18.0), acres))
        plans.append(plan)

    final = frames_dir.parent / "final"
    if final.exists():
        shutil.rmtree(final)
    final.mkdir(parents=True)
    ctx = {"W": W, "H": H, "sc": sc, "wide": wide, "cities": cities, "fires": fires,
           "total_hours": total_hours, "start": start, "final": str(final)}
    jobs = [(i, str(path), meta["frames"][i], plans[i]) for i, path in enumerate(frames)]

    # Pass 2, in parallel: the pixel work, drawing and saving for each frame.
    workers = workers or default_workers()
    t0, done = time.time(), 0
    if workers <= 1:
        _init_worker(ctx)
        results = map(_finish_frame, jobs)
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        # "spawn" everywhere (Windows' only option), so it behaves the same on every machine
        pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(ctx,),
                                   mp_context=mp.get_context("spawn"))
        results = pool.map(_finish_frame, jobs, chunksize=4)
    print(f"Finishing {len(jobs)} frames on {workers} worker(s)...")
    for _ in results:
        done += 1
        rate = done / max(time.time() - t0, 1e-6)
        print(f"  finishing {done}/{len(jobs)}  ~{(len(jobs) - done) / rate / 60:.0f} min left", end="\r")
    if workers > 1:
        pool.shutdown()
    print()
    return encode(final, meta["fps"], out_mp4)


def encode(frames_dir: Path, fps: int, out: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg not found; frames are in", frames_dir)
        return False
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "frame_%04d.png"),
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-crf", "18", "-maxrate", "30M", "-bufsize", "60M", "-preset", "slow",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    ok = subprocess.run(cmd).returncode == 0
    print(("Video: " if ok else "ffmpeg failed; frames are in ") + str(out if ok else frames_dir))
    return ok


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Re-finish rendered frames (labels, sky, legend) and encode.")
    ap.add_argument("--format", choices=list(config.FORMATS), default="16x9")
    ap.add_argument("--workers", type=int, default=None, help="parallel processes (default: cores - 2, max 8)")
    args = ap.parse_args()
    fmt = args.format
    frames = config.out_dir(fmt) / "frames"
    if not (frames / "frames.json").exists():
        print(f"No rendered {fmt} frames. Run: pixi run python scripts/render.py --export --format {fmt}")
        return 1
    return 0 if finish(frames, config.video_path(fmt), args.workers) else 1


if __name__ == "__main__":
    sys.exit(main())
