#!/usr/bin/env python3
"""
Copy station-specific PNG figures into organized output folders.

New station list format:
    name   long   lat   dist

Required columns:
    name, long, lat, dist

IMPORTANT
---------
- Read station 'name' as STRING
- Source directory and source file names keep the ORIGINAL station name
    example:
        source dir/file: 121.1-25.025
- Destination output filenames can be reformatted
    example:
        output file: MCMC_121.100-25.025.png

Source (per station):
  {BASE}/{FN}/MonteCarlo/{original_name}/{original_name}_IterVsMisfit.png
  {BASE}/{FN}/MonteCarlo/{original_name}/{original_name}_MCMC.png

Destinations:
  {PWD}/output_figures/misfit_previous/
  {PWD}/output_figures/1D_model_previous/
"""

from pathlib import Path
import shutil
import pandas as pd
import os
import re


# =======================
# Parameters
# =======================
PWD = Path(os.getcwd()).resolve()

# Station file can be inside current project tree
STATION_FILE = PWD / "FM" / "station_cor.lst"
SEP = r"\s+"
HEADER = None

NAME_COL = "name"
LON_COL = "long"
LAT_COL = "lat"
DIST_COL = "dist"

# Base directory that contains the project folder FN
BASE = Path(
    "/home/longhv/Research/Projects/"
    "MCMC_FM/"
    "00_FM_6_crust_para_model_crust_non_monotonic_old_avegrage_model_style"
).resolve()

FN = "FM"

OUT_MISFIT_DIR = PWD / "output_figures" / "misfit_previous"
OUT_1D_DIR = PWD / "output_figures" / "1D_model_previous"

OVERWRITE = True
DRY_RUN = False
VERBOSE = True

# sort by profile distance
SORT_BY_DIST = False


# =======================
# functions
# =======================
def keep_original_station_name(value) -> str:
    """
    Keep the original station name exactly as string from station file.
    """
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return s
    return s


def format_output_station_name(value) -> str:
    """
    Format station name ONLY for output filename.

    Rules
    -----
    1) Purely numeric name:
        0    -> 00000
        12   -> 00012
        12.0 -> 00012

    2) lon-lat style name:
        121.1-25.025 -> 121.100-25.025
        121.12-25.1  -> 121.120-25.100

    3) Other names:
        keep as original string
    """
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return s

    # Case 1: purely numeric station ID
    if re.fullmatch(r"[+-]?\d+(\.0+)?", s):
        return str(int(float(s))).zfill(5)

    # Case 2: lon-lat style station name
    m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)-([+-]?\d+(?:\.\d+)?)", s)
    if m:
        lon = float(m.group(1))
        lat = float(m.group(2))
        return f"{lon:.3f}-{lat:.3f}"

    return s


def safe_copy(src: Path, dst: Path, overwrite: bool = True, dry_run: bool = False) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not overwrite:
        if VERBOSE:
            print(f"[SKIP] exists: {dst}")
        return False

    if dry_run:
        print(f"[DRY]  {src} -> {dst}")
        return True

    shutil.copy2(src, dst)
    if VERBOSE:
        print(f"[OK]   {src.name} -> {dst}")
    return True


def main():
    station_path = Path(STATION_FILE).expanduser().resolve()
    base = Path(BASE).expanduser().resolve()
    fn_dir = base / FN

    if not station_path.exists():
        raise FileNotFoundError(f"Station file not found: {station_path}")
    if not base.exists():
        raise FileNotFoundError(f"BASE directory not found: {base}")
    if not fn_dir.exists():
        raise FileNotFoundError(f"FN directory not found: {fn_dir}")

    # Read stations: name long lat dist
    if HEADER is None:
        df = pd.read_csv(
            station_path,
            sep=SEP,
            header=None,
            names=[NAME_COL, LON_COL, LAT_COL, DIST_COL],
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

    # Keep original source name
    df["name_original"] = df[NAME_COL].map(keep_original_station_name)

    # Format only for destination filename
    df["name_out"] = df[NAME_COL].map(format_output_station_name)

    # Numeric columns
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
    # df[DIST_COL] = pd.to_numeric(df[DIST_COL], errors="coerce")

    df = df.dropna(subset=["name_original", "name_out", LON_COL, LAT_COL]).reset_index(drop=True)

    if SORT_BY_DIST:
        df = df.sort_values(DIST_COL).reset_index(drop=True)

    out_misfit = Path(OUT_MISFIT_DIR)
    out_1d = Path(OUT_1D_DIR)
    out_misfit.mkdir(parents=True, exist_ok=True)
    out_1d.mkdir(parents=True, exist_ok=True)

    n_sta = 0
    n_copied = 0
    n_missing = 0

    for _, row in df.iterrows():
        name_original = str(row["name_original"]).strip()
        name_out = str(row["name_out"]).strip()

        if not name_original:
            continue

        n_sta += 1
        src_dir = fn_dir / "MonteCarlo" / name_original

        # SOURCE keeps original name
        f1 = src_dir / f"{name_original}_IterVsMisfit.png"
        f2 = src_dir / f"{name_original}_MCMC.png"

        # DESTINATION uses formatted output name
        dst1 = out_misfit / f"IterVsMisfit_{name_out}.png"
        dst2 = out_1d / f"MCMC_{name_out}.png"

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