#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# MUST be set before importing pygmt AND before any pygmt call starts a session
os.environ["GMT_COMPATIBILITY"] = "6"

import multiprocessing as mp
from pathlib import Path
import shutil
import pandas as pd
import pygmt
from concurrent.futures import ProcessPoolExecutor, as_completed


# =========================
# Parameters (edit below)
# =========================
# Parallel
REQUESTED_WORKERS = 20
CPU_COUNT = (os.cpu_count() or 1) - 1
CPU_COUNT = max(1, CPU_COUNT)

if REQUESTED_WORKERS and REQUESTED_WORKERS > 0:
    N_WORKERS = min(REQUESTED_WORKERS, CPU_COUNT)
else:
    N_WORKERS = CPU_COUNT

PWD = os.getcwd()
FN = "FM"

STATION_FILE = os.path.join(PWD, FN, "station_cor.lst")
SEP = r"\s+"
HEADER = None
NAME_COL = "name"
LON_COL = "long"
LAT_COL = "lat"

OUT_DIR = Path(PWD) / "output_figures" / "station_maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGION_PAD_DEG = 0.05

RELIEF_RESOLUTION = "01s"
CPT = "geo"
DPI = 100
PROJ = "M15c"
# FRAME = ["a0.2f0.2", "+tMap of FM array nodes"] # Frame move to in_figure for station name on title

TRI_SIZE = "0.25c"
STAR_SIZE = "0.35c"
FONT_LABEL = "10p,Helvetica-Bold,black"

# ---- Name the figure to sort in order of long - lat as xxx.xxx-yy.yyy
LON_DEC = 3
LAT_DEC = 3
LON_INT_WIDTH = 3   # 000–180
LAT_INT_WIDTH = 2   # 00–90

def enforce_modern_gmt():
    """Enforce GMT modern mode before any GMT/PyGMT module runs."""
    os.environ["GMT_COMPATIBILITY"] = "6"
    pygmt.config(GMT_COMPATIBILITY="6")


def read_stations() -> pd.DataFrame:
    station_path = Path(STATION_FILE).expanduser().resolve()
    if not station_path.exists():
        raise FileNotFoundError(f"Station file not found: {station_path}")

    if HEADER is None:
        df = pd.read_csv(
            station_path,
            sep=SEP,
            header=None,
            names=[NAME_COL, LON_COL, LAT_COL],
            dtype={NAME_COL: "string"},
            engine="python",
        )
    else:
        df = pd.read_csv(
            station_path,
            sep=SEP,
            header=HEADER,
            dtype={NAME_COL: "string"},
            engine="python",
        )

    df[NAME_COL] = df[NAME_COL].astype("string").str.strip()
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
    df = df.dropna(subset=[NAME_COL, LON_COL, LAT_COL]).reset_index(drop=True)
    return df


def compute_region_all(df: pd.DataFrame):
    minlon = float(df[LON_COL].min())
    maxlon = float(df[LON_COL].max())
    minlat = float(df[LAT_COL].min())
    maxlat = float(df[LAT_COL].max())

    region_all = [
        minlon - REGION_PAD_DEG,
        maxlon + REGION_PAD_DEG,
        minlat - REGION_PAD_DEG,
        maxlat + REGION_PAD_DEG,
    ]
    return [round(v, 3) for v in region_all]


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def make_one_map(df_all: pd.DataFrame, region_all, cur_name: str, cur_lon: float, cur_lat: float) -> str:
    # CRITICAL: enforce modern mode BEFORE any pygmt call in this worker
    enforce_modern_gmt()

    region = region_all

    # Now it's safe to call any pygmt module
    grid = pygmt.datasets.load_earth_relief(resolution=RELIEF_RESOLUTION, region=region)
    shade = pygmt.grdgradient(grid=grid, azimuth=315, normalize="t0.8")

    fig = pygmt.Figure()
    
    pygmt.config(FONT_LABEL="15p,Times-Bold,black",
             FONT_TITLE="22p,Times-Bold,black",
             FONT_ANNOT_PRIMARY="12p,Times-Bold,black",
             FONT_ANNOT_SECONDARY="10p,Times-Bold,black",
             MAP_FRAME_TYPE="fancy"
            )
    pygmt.makecpt(cmap=CPT, series=[float(grid.min()), float(grid.max())])

    fig.grdimage(
        grid=grid,
        region=region,
        projection=PROJ,
        shading=shade,
        cmap=True,
        frame=["a0.2f0.2", "+tMap of FM array nodes: {}".format(cur_name)],
    )

    fig.coast(
        region=region,
        projection=PROJ,
        shorelines="0.5p,black",
        borders="1/0.5p,black",
        rivers="a/0.25p,blue",
    )

    fig.colorbar(frame='af+l"Elevation (m)"')

    fig.plot(
        x=df_all[LON_COL].to_numpy(),
        y=df_all[LAT_COL].to_numpy(),
        style=f"t{TRI_SIZE}",
        pen="0.6p,black",
        fill="black",
    )

    fig.plot(
        x=[cur_lon],
        y=[cur_lat],
        style=f"a{STAR_SIZE}",
        pen="0.8p,black",
        fill="red",
    )

    fig.text(
        x=cur_lon,
        y=cur_lat,
        text=str(cur_name),
        font=FONT_LABEL,
        justify="LM",
        offset="0.20c/0.20c",
        fill="white@60",
        pen="0.25p,black",
    )

    # out_png = OUT_DIR / f"map_{cur_name}.png" # old name - can not sort in order
    lon_tag = f"{cur_lon:0{LON_INT_WIDTH}.{LON_DEC}f}"
    lat_tag = f"{cur_lat:0{LAT_INT_WIDTH}.{LAT_DEC}f}"
    out_png = OUT_DIR / f"map_{lon_tag}-{lat_tag}.png"

    fig.savefig(str(out_png), dpi=DPI)
    return str(out_png)


def _worker(payload):
    df_all = pd.DataFrame(payload[0])
    region_all = payload[1]
    return make_one_map(df_all, region_all, payload[2], payload[3], payload[4])


def main():
    # Enforce modern mode in the main process too (before any pygmt call)
    enforce_modern_gmt()

    df = read_stations()
    region_all = compute_region_all(df)

    print("Computed region (rounded 3 decimals):", [f"{v:.3f}" for v in region_all])
    print(f"Stations: {len(df)}")
    print(f"Output  : {OUT_DIR}")
    print(f"Workers : {N_WORKERS}")

    df_dict = {c: df[c].to_list() for c in [NAME_COL, LON_COL, LAT_COL]}

    jobs = []
    for _, row in df.iterrows():
        jobs.append((df_dict, region_all, str(row[NAME_COL]), float(row[LON_COL]), float(row[LAT_COL])))

    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(_worker, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            try:
                out = fut.result()
                print(f"[{done}/{len(jobs)}] {out}")
            except Exception as e:
                print(f"[{done}/{len(jobs)}] ERROR: {e}")

    print("Finished.")


if __name__ == "__main__":
    # IMPORTANT: avoid forking a partially initialized GMT session
    mp.set_start_method("spawn", force=True)
    main()