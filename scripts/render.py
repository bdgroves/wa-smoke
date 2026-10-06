"""Render Washington's 2026 smoke season over the state's terrain with forge3d.

    pixi run preview      # a few seconds of the animation in a window
    pixi run render       # every frame, then the finishing pass and MP4

Each frame gets its own drape (drape.py): land, burn scars so far, the
smoke at that hour and the fires burning right then. The camera looks north
across the state and drifts slowly while the summer plays out; at the end it
holds while the official fire perimeters are traced in. finish.py adds the
sky, clock, city smoke readings, fire names and legend.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image

import forge3d as f3d

import config
from drape import Drape

# Show the temperature colours as they are, with the terrain shading carried on top
# (forge3d's preserve-colours mode) rather than re-lit, which washes reds to salmon.
PRESERVE_COLORS = True
TERRAIN_LOOK = {"ambient": 0.30, "shadow": 0.55, "sun_intensity": 1.05}
PBR = {
    "enabled": True, "shadow_technique": "pcss", "shadow_map_res": 4096,
    "exposure": 0.86, "msaa": 8, "ibl_intensity": 0.12, "normal_strength": 0.75,
    "height_ao": {"enabled": True, "directions": 12, "steps": 24, "max_distance": 3000.0,
                  "strength": 0.5, "resolution_scale": 0.75},
}

LITE = {"shadow_technique": "pcf", "shadow_map_res": 2048, "msaa": 4,
        "height_ao": dict(PBR["height_ao"], directions=8, steps=16, resolution_scale=0.5)}

# forge3d's viewer caps the camera distance at 50 km, which is plenty for one
# mountain but not for a whole state. So the viewer gets a copy of the DEM
# with every coordinate multiplied by WORLD_SCALE (same pixels, smaller
# numbers) and the height exaggeration scaled to match: the picture is the
# same, and the camera stays inside its limit. finish.py uses the same scale.
WORLD_WIDTH_M = 45_000


def world_scale(drape: Drape) -> float:
    return WORLD_WIDTH_M / (drape.shape[1] * drape.tf.a)


def scaled_dem(drape: Drape, s: float):
    path = config.DATA / "wa_dem_viewer.tif"
    with rasterio.open(config.DEM) as src:
        t = src.transform
        profile = src.profile
        data = src.read(1)
    # a light blur rounds off single-pixel needles; water stays flat at sea level
    from scipy.ndimage import gaussian_filter
    water = data <= 0.5
    data = np.where(water, 0.0, gaussian_filter(np.where(water, 0.0, data), config.SMOOTH_PX)).astype(data.dtype)
    profile.update(transform=Affine(t.a * s, t.b, t.c * s, t.d, t.e * s, t.f * s))
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


class _Tee:
    """Wraps the viewer's stdout so everything it prints also lands in out/viewer.log."""

    def __init__(self, stream, log):
        self._s, self._log = stream, log

    def _keep(self, line):
        if line:
            self._log.write(line)
            self._log.flush()
        return line

    def readline(self, *a):
        return self._keep(self._s.readline(*a))

    def read(self, *a):
        return self._keep(self._s.read(*a))

    def __iter__(self):
        for line in self._s:
            yield self._keep(line)

    def __getattr__(self, name):
        return getattr(self._s, name)


def open_viewer(log_path, **kw):
    """open_viewer_async, but with the viewer's own messages saved to a log file."""
    import forge3d.viewer as fv
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    real_popen = fv.subprocess.Popen

    def popen(*a, **k):
        proc = real_popen(*a, **k)
        if proc.stdout is not None:
            proc.stdout = _Tee(proc.stdout, log)
        return proc

    fv.subprocess.Popen = popen
    try:
        return f3d.open_viewer_async(timeout=120.0, **kw)
    finally:
        fv.subprocess.Popen = real_popen


def camera_at(u: float, drape: Drape, z_scale: float, s: float, fmt: str = "16x9") -> dict:
    """Camera for progress u in [0, 1]: south-south-west of the state, drifting.

    Positions are in the viewer's scaled world (real UTM metres times s).
    """
    h, w = drape.shape
    width_m = w * drape.tf.a
    cx = drape.tf.c + width_m / 2
    cy = drape.tf.f + (h * drape.tf.e) / 2
    ease = 0.5 - 0.5 * np.cos(np.pi * u)
    return {
        "phi": 96.0 + 14.0 * ease,                # 90 = due south; drift toward south-west
        "theta": config.FORMATS[fmt]["theta"],    # degrees down from vertical
        "radius": width_m * s * 1.0,              # stays under the viewer's 50 km clamp
        "fov": config.FORMATS[fmt]["fov"],
        # aim 30 km south of the box centre so the whole state sits in frame
        "target": [cx * s, 400.0 * z_scale, -(cy - config.FORMATS[fmt]["south_m"]) * s],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=False)
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--export", action="store_true")
    ap.add_argument("--frames-per-hour", type=float, default=1.0, help="1 = a day every second at 24 fps")
    ap.add_argument("--hold", type=float, default=4.0, help="seconds to hold at the end while perimeters appear")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--format", choices=list(config.FORMATS), default="16x9",
                    help="16x9 (default), 1x1 or 4x5 for social feeds")
    ap.add_argument("--exaggeration", type=float, default=config.EXAGGERATION)
    ap.add_argument("--stills", type=str, default=None,
                    help="comma-separated hours from the start (e.g. 112,136) -> just those frames, finished")
    ap.add_argument("--lite", action="store_true",
                    help="lighter shadows/antialiasing, for slower GPUs or if the viewer stalls")
    ap.add_argument("--hours", type=float, default=None, help="render only the first N hours (testing)")
    args = ap.parse_args()

    for path, task in ((config.DEM, "dem"), (config.BOUNDARY, "boundary"), (config.SMOKE, "smoke")):
        if not path.exists():
            print(f"Missing {path.name}. Run: pixi run {task}")
            return 1

    if not config.DETECTIONS.exists():
        print("(No satellite fire detections; run 'pixi run fires' with a FIRMS key to see fires grow.)")
    print("Preparing the land, fires and smoke...")
    drape = Drape(water_key=bool(args.export or args.stills))
    hours = drape.hours() if args.hours is None else min(args.hours, drape.hours())
    n = int(round(hours * args.frames_per_hour)) + 1
    s = world_scale(drape)
    z = args.exaggeration * s                 # heights are real metres, so scale them too
    viewer_dem = scaled_dem(drape, s)
    args.width, args.height = config.FORMATS[args.format]["size"]
    base_dir = config.out_dir(args.format)
    work = base_dir / "work"
    frames_dir = base_dir / "frames"
    for d in (work, frames_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # City pixels, for the temperatures finish.py prints next to each name.
    from rasterio.warp import transform as warp
    xs, ys = warp("EPSG:4326", drape.crs, [c.lon for c in config.CITIES], [c.lat for c in config.CITIES])
    city_px = [(int((y - drape.tf.f) / drape.tf.e), int((x - drape.tf.c) / drape.tf.a)) for x, y in zip(xs, ys)]

    first, _ = drape.image(0.0)
    Image.fromarray(first).save(work / "drape_a.png")
    pbr = dict(PBR, **LITE) if args.lite else PBR
    log_path = config.OUT / "viewer.log"
    with open_viewer(log_path, terrain_path=viewer_dem, width=args.width, height=args.height,
                               title="Washington smoke 2026", fov_deg=34.0) as viewer:
        viewer.send_ipc({"cmd": "set_terrain", "zscale": z, "sun_azimuth": 225.0,
                         "sun_elevation": 38.0, **TERRAIN_LOOK})
        viewer.send_ipc({"cmd": "set_terrain_pbr", **pbr})
        viewer.load_overlay("drape", work / "drape_a.png", extent=(0.0, 0.0, 1.0, 1.0),
                            opacity=1.0, preserve_colors=PRESERVE_COLORS)
        viewer.send_ipc({"cmd": "set_overlays_enabled", "enabled": True})
        viewer.send_ipc({"cmd": "set_overlay_solid", "solid": True})
        cam = camera_at(0.0, drape, z, s, args.format)
        viewer.set_orbit_camera(phi_deg=cam["phi"], theta_deg=cam["theta"], radius=cam["radius"],
                                fov_deg=cam["fov"], target=tuple(cam["target"]))
        time.sleep(2.0)
        layer_id = 0

        meta = []
        if args.stills:
            # hours from the start; "end" = the final frame with the perimeters traced in
            plan = [(hours, 1.0, 1.0) if h.strip() == "end" else (float(h), float(h) / max(hours, 1), 0.0)
                    for h in args.stills.split(",")]
        else:
            count = n if args.export else min(n, int(args.fps * 6))
            plan = [(i / args.frames_per_hour, i / max(n - 1, 1), 0.0) for i in range(count)]
            if args.export and args.hold > 0:
                held = int(args.hold * args.fps)
                plan += [(hours, 1.0, min(1.0, (k + 1) / (0.6 * held))) for k in range(held)]
        count = len(plan)
        print(f"{'Rendering' if args.export else 'Previewing'} {count} frames "
              f"({hours:.0f} hours at {args.frames_per_hour:g} frames an hour)...")
        final_pass = bool(args.export or args.stills)

        def show(path: Path):
            """Swap the overlay: load_overlay adds a layer (ids count up from 0), so drop the old one."""
            nonlocal layer_id
            viewer.load_overlay("drape", path, extent=(0.0, 0.0, 1.0, 1.0),
                                opacity=1.0, preserve_colors=PRESERVE_COLORS)
            viewer.send_ipc({"cmd": "remove_overlay", "id": layer_id})
            layer_id += 1

        def prepare(i: int):
            """Build and save frame i's drapes. Runs on a helper thread while the viewer
            renders the previous frame, so the CPU and GPU work overlap."""
            hour, _, outline = plan[i]
            # the final pass renders the smoke separately (see finish.py); the preview bakes it in
            rgb, field = drape.image(hour, outline, smoke=not final_pass)
            main = work / f"drape_{i % 4}.png"           # rotate names so a file is never reused mid-load
            Image.fromarray(rgb).save(main, compress_level=1)
            code = None
            if final_pass:                               # half-size is plenty for the smooth smoke field
                code = work / f"smoke_{i % 4}.png"
                Image.fromarray(drape.smoke_code(field[::2, ::2])).save(code, compress_level=1)
            cities_now = [float(field[r, c]) if 0 <= r < field.shape[0] and 0 <= c < field.shape[1]
                          else None for r, c in city_px]
            return main, code, cities_now

        from concurrent.futures import ThreadPoolExecutor
        helper = ThreadPoolExecutor(max_workers=1)
        upcoming = helper.submit(prepare, 0)
        for i, (hour, u, outline) in enumerate(plan):
            main, code, cities_now = upcoming.result()
            if i + 1 < count:
                upcoming = helper.submit(prepare, i + 1)
            show(main)
            cam = camera_at(u, drape, z, s, args.format)
            viewer.set_orbit_camera(phi_deg=cam["phi"], theta_deg=cam["theta"], radius=cam["radius"],
                                    fov_deg=cam["fov"], target=tuple(cam["target"]))
            if final_pass:
                viewer.snapshot(frames_dir / f"frame_{i:04d}.png", width=args.width, height=args.height)
                # second, half-size render of the smoke alone, colour-coded to survive the lighting
                show(code)
                viewer.snapshot(frames_dir / f"smoke_{i:04d}.png", width=args.width // 2, height=args.height // 2)
            else:
                time.sleep(1 / args.fps)
            meta.append({"camera": cam, "hour": hour, "outline": outline, "time": drape.time_label(hour),
                         "cities": cities_now})
            print(f"  frame {i + 1}/{count}  {meta[-1]['time']}", end="\r")
        print()
        helper.shutdown()

    if not (args.export or args.stills):
        return 0
    (frames_dir / "frames.json").write_text(json.dumps({
        "fps": args.fps, "total_hours": hours, "start": drape.time_label(0.0), "z_scale": z,
        "fires": [{"name": f["name"], "acres": int(f["acres"]), "hour": float(f["hour"]), "lat": float(f["lat"]),
                   "lon": float(f["lon"]), "label": bool(f["label"])} for f in drape.fires], "world_scale": s, "dem": str(config.DEM.resolve()),
        "cities": [c.__dict__ for c in config.CITIES], "frames": meta}))
    from finish import finish
    finish(frames_dir, config.video_path(args.format))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (TimeoutError, f3d.viewer.ViewerError) as err:
        log = config.OUT / "viewer.log"
        print(f"\nThe viewer stopped answering ({err}).")
        if log.exists():
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            print(f"Last lines from {log}:")
            print("\n".join("  " + s for s in lines[-25:]))
        print("Try again with --lite, and send the log above if it still stalls.")
        sys.exit(1)
