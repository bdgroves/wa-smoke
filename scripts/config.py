"""Shared settings for the Washington smoke animation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"
FONTS = ROOT / "assets" / "fonts"

# Washington plus a margin: Portland to the south, the Canadian border to the north.
WEST, EAST = -125.0, -116.6
SOUTH, NORTH = 45.45, 49.02
EPSG = 26910                  # NAD83 / UTM 10N, metres
DEM_PX = 250                  # metres
EXAGGERATION = 3.5
SMOOTH_PX = 1.3               # soften the 250 m DEM for the 3D view so ridges read as ridges

# The window: Little Giant (Jul 16) and Sinlahekin (Jul 26) are already
# burning in the Okanogan, the Spokane Complex starts Aug 1, and the season
# is still going into September. Dates are UTC days for the downloads; the
# clock on screen is Pacific time.
START = "2026-07-25"
END = "2026-09-10"
TIMEZONE = "America/Los_Angeles"
STEP_HOURS = 1                # smoke every hour; 2 or 3 halves/thirds the download

# Smoke colour scale, micrograms of smoke per cubic metre near the ground.
# The ticks are the EPA's PM2.5 category breakpoints, for reference.
SMOKE_TICKS = [(0, "0"), (12, "12"), (35, "35"), (55, "55"), (150, "150"), (250, "250+")]
SMOKE_MAX = 250.0

# Big fires get a label once satellites first see them burning.
FIRE_LABEL_ACRES = 50_000
FIRE_MIN_ACRES = 1_000        # smaller perimeters are skipped entirely
# Perimeters are mapped per fire; some are better known by their complex. The
# complex gets one label (on its largest fire) with the acres added up.
FIRE_COMPLEXES = {"Spokane Complex": ["Old Trails", "Autumn Lane", "Fairview"]}

DEM = DATA / "wa_dem.tif"
BOUNDARY = DATA / "washington.geojson"
SMOKE = DATA / f"smoke_hrrr_{START}_{END}_{STEP_HOURS}h.npz"
DETECTIONS = DATA / f"firms_viirs_{START}_{END}.csv"
PERIMETERS = DATA / "fire_perimeters_2026.geojson"

# The heat-dome project already has the terrain and outline; reuse them if it's next door.
SIBLING_DATA = ROOT.parent / "wa-heat-dome" / "data"


@dataclass
class City:
    name: str
    lat: float
    lon: float
    priority: int = 50


CITIES = [
    City("Seattle", 47.6062, -122.3321, 100),
    City("Spokane", 47.6588, -117.4260, 95),
    City("Portland", 45.5152, -122.6784, 80),
    City("Wenatchee", 47.4235, -120.3103, 75),
    City("Yakima", 46.6021, -120.5059, 70),
    City("Tri-Cities", 46.2112, -119.1372, 65),
    City("Bellingham", 48.7519, -122.4787, 55),
]


# Output shapes, same cameras as the heat-dome project.
FORMATS = {
    "16x9": {"size": (1920, 1080), "theta": 45.0, "fov": 38.0, "south_m": 30_000},
    "1x1": {"size": (1080, 1080), "theta": 45.0, "fov": 55.0, "south_m": 60_000},
    "4x5": {"size": (1080, 1350), "theta": 40.0, "fov": 70.0, "south_m": 0},
}


def out_dir(fmt: str) -> Path:
    return OUT if fmt == "16x9" else OUT / fmt


def video_path(fmt: str) -> Path:
    return OUT / ("wa_smoke_2026.mp4" if fmt == "16x9" else f"wa_smoke_2026_{fmt}.mp4")
