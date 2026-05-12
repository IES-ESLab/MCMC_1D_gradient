#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pygmt
from matplotlib.path import Path as MplPath

# =========================
# INPUT
# =========================
PWD = os.getcwd()

STATION_FILE = Path(PWD) / "FM" / "station_cor.lst"
QC_DIR = Path(PWD) / "output_figures" / "QC_map_mcmc"

OUT_DIR = Path(PWD) / "output_figures" / "QC_polygon_select"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NAME = "Tatun_volcano"   # <<< change this

# -------------------------
# DEFINE POLYGON HERE
# need at least 3 points
# -------------------------
# Ilan
# polygon = np.array([
#     [121.85, 24.9],
#     [121.55, 24.65],
#     [121.9, 24.55],
# ], dtype=float)
# -- Tatun Volcano group --
polygon = np.array([
    [121.60, 25.30],
    [121.45, 25.15],
    [121.60, 25.00],
    [121.75, 25.15],
], dtype=float)
# =========================
# PARAMETERS
# =========================
LON_COL = "long"
LAT_COL = "lat"
NAME_COL = "name"

PROJ = "M15c"
DPI = 300

# =========================
# HELPERS
# =========================
def validate_and_close_polygon(poly):
    poly = np.asarray(poly, dtype=float)

    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("polygon must be an array of shape (n, 2)")

    # remove NaN rows
    poly = poly[np.isfinite(poly).all(axis=1)]

    # need at least 3 points
    if len(poly) < 3:
        raise ValueError("polygon must contain at least 3 valid points")

    # remove repeated last point if already same as first
    if len(poly) > 1 and np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]

    # still need at least 3 unique points
    unique_poly = np.unique(np.round(poly, decimals=10), axis=0)
    if len(unique_poly) < 3:
        raise ValueError("polygon must contain at least 3 unique points")

    # close polygon
    poly_closed = np.vstack([poly, poly[0]])
    return poly_closed


def build_polygon_path(poly_closed):
    """
    Build a closed matplotlib Path using explicit codes.
    poly_closed must already have first point repeated at the end.
    """
    n = len(poly_closed)
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (n - 2) + [MplPath.CLOSEPOLY]
    return MplPath(poly_closed, codes)


# =========================
# READ STATIONS
# =========================
df = pd.read_csv(
    STATION_FILE,
    sep=r"\s+",
    header=None,
    names=[NAME_COL, LON_COL, LAT_COL],
    engine="python",
)

df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
df = df.dropna(subset=[LON_COL, LAT_COL]).reset_index(drop=True)

# =========================
# SELECT STATIONS IN POLYGON
# =========================
polygon_closed = validate_and_close_polygon(polygon)
poly_path = build_polygon_path(polygon_closed)

points = df[[LON_COL, LAT_COL]].values

# small positive radius helps include points that fall numerically on boundary
inside_mask = poly_path.contains_points(points, radius=1e-10)
df_inside = df[inside_mask].copy()

print("Total stations:", len(df))
print("Inside polygon:", len(df_inside))

# =========================
# COPY QC MAPS
# =========================
OUT_COPY_DIR = OUT_DIR / f"{OUT_NAME}_maps"
OUT_COPY_DIR.mkdir(parents=True, exist_ok=True)

for _, row in df_inside.iterrows():
    lon = float(row[LON_COL])
    lat = float(row[LAT_COL])

    fname = f"QC_{lon:06.3f}-{lat:05.3f}.png"
    src = QC_DIR / fname
    dst = OUT_COPY_DIR / fname

    if src.exists():
        shutil.copy2(src, dst)
    else:
        print("[MISS]", fname)

print("Copied QC maps to:", OUT_COPY_DIR)

# =========================
# PLOT MAP
# =========================
region = [
    float(df[LON_COL].min()) - 0.1,
    float(df[LON_COL].max()) + 0.1,
    float(df[LAT_COL].min()) - 0.1,
    float(df[LAT_COL].max()) + 0.1,
]

grid = pygmt.datasets.load_earth_relief(resolution="15s", region=region)
shade = pygmt.grdgradient(grid=grid, azimuth=315, normalize="t0.8")

fig = pygmt.Figure()
pygmt.config(
    FONT_LABEL="15p,Times-Bold,black",
    FONT_TITLE="22p,Times-Bold,black",
    FONT_ANNOT_PRIMARY="12p,Times-Bold,black",
    FONT_ANNOT_SECONDARY="10p,Times-Bold,black",
    MAP_FRAME_TYPE="fancy",
    FORMAT_GEO_MAP="ddd.xx",
)
pygmt.makecpt(cmap="geo", series=[float(grid.min()), float(grid.max())])

fig.grdimage(
    grid=grid,
    region=region,
    projection=PROJ,
    shading=shade,
    cmap=True,
    frame=[
        f'WSne+t"Polygon selection: {OUT_NAME}"',
        "xafg+lLongitude (°)",
        "yafg+lLatitude (°)",
    ],
)

fig.coast(shorelines="0.5p,black", borders="1/0.5p,black")

# ---- ALL stations (black)
fig.plot(
    x=df[LON_COL],
    y=df[LAT_COL],
    style="c0.12c",
    fill="black",
    pen="0.2p,black",
)
fig.plot(
    data="faults/Fault_TEM_2016.gmt",
    pen="1.0p,purple",
)
# ---- POLYGON (red line)
fig.plot(
    x=polygon_closed[:, 0],
    y=polygon_closed[:, 1],
    pen="2p,red",
)

# ---- INSIDE stations (red)
if not df_inside.empty:
    fig.plot(
        x=df_inside[LON_COL],
        y=df_inside[LAT_COL],
        style="c0.18c",
        fill="red",
        pen="0.5p,black",
    )

fig.colorbar(frame='af+l"Elevation (m)"')

out_png = OUT_DIR / f"{OUT_NAME}_polygon_map.png"
fig.savefig(out_png, dpi=DPI)

print("Saved map:", out_png)