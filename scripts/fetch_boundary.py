"""Washington's outline from Natural Earth (public domain), 1:50m states."""
from __future__ import annotations

import json
import sys
import urllib.request

import config

URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
       "geojson/ne_50m_admin_1_states_provinces.geojson")


def main() -> int:
    if config.BOUNDARY.exists():
        print(f"Boundary already present: {config.BOUNDARY}")
        return 0
    sibling = config.SIBLING_DATA / config.BOUNDARY.name
    if sibling.exists():
        import shutil
        config.BOUNDARY.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sibling, config.BOUNDARY)
        print(f"Copied the outline from the heat-dome project: {sibling}")
        return 0
    print("Downloading Natural Earth state outlines...")
    with urllib.request.urlopen(URL, timeout=120) as r:
        data = json.load(r)
    wa = [f for f in data["features"]
          if f["properties"].get("name") == "Washington"
          and f["properties"].get("admin") == "United States of America"]
    if not wa:
        print("Washington not found in the Natural Earth file.")
        return 1
    config.DATA.mkdir(parents=True, exist_ok=True)
    config.BOUNDARY.write_text(json.dumps({"type": "FeatureCollection", "features": wa}))
    print(f"Saved {config.BOUNDARY.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
