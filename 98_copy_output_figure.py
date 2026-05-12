#!/usr/bin/env python3
"""
Copy station-specific PNG figures into organized output folders,
and RENAME them using zero-padded lon/lat tags so `ls` sorts correctly.

Station list file is read by pandas.read_csv.
- Required columns: name, long, lat
- IMPORTANT: station 'name' is already a STRING in "long-lat" format (keep as str)

Source (per station):
  {PWD}/{FN}/MonteCarlo/{name}/{name}_IterVsMisfit.png
  {PWD}/{FN}/MonteCarlo/{name}/{name}_MCMC.png

Destinations:
  {PWD}/output_figures/misfit/
  {PWD}/output_figures/1D_model/
"""

from pathlib import Path
import shutil
import pandas as pd
import os


# =======================
# Parameters (edit here)
# =======================
PWD = os.getcwd()
STATION_FILE = os.path.join(PWD, "FM", "station_cor.lst")
SEP = r"\s+"
HEADER = None
NAME_COL = "name"
LON_COL = "long"
LAT_COL = "lat"

FN = "FM"

OUT_MISFIT_DIR = os.path.join(PWD, "output_figures", "misfit")
OUT_1D_DIR = os.path.join(PWD, "output_figures", "1D_model")

OVERWRITE = True
DRY_RUN = False
VERBOSE = True

# ---- Rename/sort tags (for linux `ls` ordering) ----
# output filename will be:
#   <PREFIX>_<LON_TAG>-<LAT_TAG>.png
# Example:
#   IterVsMisfit_121.100-025.025.png
#   MCMC_121.100-025.025.png
LON_DEC = 3
LAT_DEC = 3
LON_INT_WIDTH = 3   # 000–180
LAT_INT_WIDTH = 2   # 00–90
USE_LAT_LON_ORDER = False  # False: lon-lat (requested); True: lat-lon


# =======================
# functions
# =======================
def fmt_tag(lon: float, lat: float) -> str:
    lon_tag = f"{lon:0{LON_INT_WIDTH}.{LON_DEC}f}"
    lat_tag = f"{lat:0{LAT_INT_WIDTH}.{LAT_DEC}f}"
    return f"{lat_tag}-{lon_tag}" if USE_LAT_LON_ORDER else f"{lon_tag}-{lat_tag}"


def safe_copy(src: Path, dst: Path, overwrite: bool = True, dry_run: bool = False) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not overwrite:
        if VERBOSE:
            print(f"[SKIP] exists: {dst}")
        return False

    if dry_run:
        print(f"[DRY]  {src}  ->  {dst}")
        return True

    shutil.copy2(src, dst)
    if VERBOSE:
        print(f"[OK]   {src.name} -> {dst}")
    return True


def main():
    station_path = Path(STATION_FILE).expanduser().resolve()
    base = Path(PWD).expanduser().resolve()
    fn_dir = base / FN

    if not station_path.exists():
        raise FileNotFoundError(f"Station file not found: {station_path}")
    if not fn_dir.exists():
        raise FileNotFoundError(f"FN directory not found: {fn_dir}")

    # Read stations
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

    out_misfit = Path(OUT_MISFIT_DIR)
    out_1d = Path(OUT_1D_DIR)

    n_sta = 0
    n_copied = 0
    n_missing = 0

    for _, row in df.iterrows():
        name = str(row[NAME_COL]).strip()
        if not name:
            continue

        lon = float(row[LON_COL])
        lat = float(row[LAT_COL])
        tag = fmt_tag(lon, lat)

        n_sta += 1
        src_dir = fn_dir / "MonteCarlo" / name

        f1 = src_dir / f"{name}_IterVsMisfit.png"
        f2 = src_dir / f"{name}_MCMC.png"

        # Rename in destination using tag (so ls sorts)
        dst1 = out_misfit / f"IterVsMisfit_{tag}.png"
        dst2 = out_1d / f"MCMC_{tag}.png"

        if not f1.exists():
            n_missing += 1
            if VERBOSE:
                print(f"[MISS] {f1}")
        else:
            if safe_copy(f1, dst1, overwrite=OVERWRITE, dry_run=DRY_RUN):
                n_copied += 1

        if not f2.exists():
            n_missing += 1
            if VERBOSE:
                print(f"[MISS] {f2}")
        else:
            if safe_copy(f2, dst2, overwrite=OVERWRITE, dry_run=DRY_RUN):
                n_copied += 1

    print("\n==== Summary ====")
    print(f"Stations processed : {n_sta}")
    print(f"Files copied       : {n_copied}")
    print(f"Missing files      : {n_missing}")
    print(f"Output misfit dir  : {out_misfit}")
    print(f"Output 1D dir      : {out_1d}")


if __name__ == "__main__":
    main()