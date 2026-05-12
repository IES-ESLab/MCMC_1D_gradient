#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ================= USER SETTINGS =================
pwd = os.getcwd()
maindir = os.path.join(pwd, "FM", "Vel_mod")

inmodfile = os.path.join(maindir, "NTW1d_H14_1km_final")

mdep = 30.0            # Moho depth (km)
bottom_depth = 50.0    # deepest depth to output (km)
mantle_ref_depth = 40.0  # pick mantle reference (km) for the 1-layer mantle

# depths you want in the output (Moho will be inserted automatically if missing)
base_depths = np.array([0., 6., 12., 18., 24, 30., 50.])

# output file
out_file = os.path.join(maindir, f"NTW1d_H14_intp_{int(bottom_depth)}km")
# =================================================


# -------------------- helpers --------------------
def make_branch(df: pd.DataFrame, branch: str) -> pd.DataFrame:
    """
    Resolve duplicate depths into two branches:
      - crust  : choose smaller Vs (and corresponding Vp) at the same depth
      - mantle : choose larger  Vs (and corresponding Vp) at the same depth
    This preserves a discontinuity at Moho where dep is duplicated (e.g., 30.0 twice).
    """
    df2 = df.sort_values(["dep", "vs"]).copy()
    if branch == "crust":
        out = df2.groupby("dep", as_index=False).first()
    elif branch == "mantle":
        out = df2.groupby("dep", as_index=False).last()
    else:
        raise ValueError("branch must be 'crust' or 'mantle'")
    return out.sort_values("dep").reset_index(drop=True)


# 1) read original model
indata = pd.read_csv(inmodfile, sep=r"\s+", names=["dep", "vs", "vp"])
indata = indata.sort_values("dep").reset_index(drop=True)

# keep enough depth for mantle_ref_depth and bottom_depth
max_need = max(bottom_depth, mantle_ref_depth)
indata = indata[(indata["dep"] >= 0.0) & (indata["dep"] <= max_need)].reset_index(drop=True)

if indata.empty:
    raise RuntimeError("Input model is empty after clipping. Check bottom_depth/mantle_ref_depth.")

if mantle_ref_depth <= mdep:
    raise ValueError("mantle_ref_depth must be deeper than mdep (Moho), e.g. 40 km when mdep=30 km.")

# 2) split into crust/mantle branches to handle duplicate depths safely
crust = make_branch(indata, "crust")
mantle = make_branch(indata, "mantle")

# 3) interpolators (NOTE: each branch has unique depths -> safe for interp1d)
f_vs_c = interp1d(crust["dep"], crust["vs"], kind="linear",
                  bounds_error=False, fill_value="extrapolate")
f_vp_c = interp1d(crust["dep"], crust["vp"], kind="linear",
                  bounds_error=False, fill_value="extrapolate")

f_vs_m = interp1d(mantle["dep"], mantle["vs"], kind="linear",
                  bounds_error=False, fill_value="extrapolate")
f_vp_m = interp1d(mantle["dep"], mantle["vp"], kind="linear",
                  bounds_error=False, fill_value="extrapolate")

# 4) mantle reference (for 1-layer mantle below Moho)
vs_mantle_ref = float(f_vs_m(mantle_ref_depth))
vp_mantle_ref = float(f_vp_m(mantle_ref_depth))
print(f"Mantle reference at {mantle_ref_depth:.1f} km: Vs = {vs_mantle_ref:.5f}, Vp = {vp_mantle_ref:.5f}")

# 5) output depths (ensure mdep is included; keep order)
resamp_depths = np.unique(np.append(base_depths, mdep)).astype(float)
resamp_depths = np.sort(resamp_depths)

rows = []

# 6) above Moho (pure crust, use crust branch)
for d in resamp_depths[resamp_depths < mdep]:
    rows.append({"dep": float(d), "vs": float(f_vs_c(d)), "vp": float(f_vp_c(d))})

# 7) Moho crust side at mdep  ✅ this will match the *crust* value at 30 km in the input
vs_crust_mdep = float(f_vs_c(mdep))
vp_crust_mdep = float(f_vp_c(mdep))
rows.append({"dep": float(mdep), "vs": vs_crust_mdep, "vp": vp_crust_mdep})

# 8) Moho mantle side at mdep (jump)
# If your input has mantle value at exactly mdep (e.g., 30.0 4.3 7.6927), this keeps it.
vs_mantle_mdep = float(f_vs_m(mdep))
vp_mantle_mdep = float(f_vp_m(mdep))
rows.append({"dep": float(mdep), "vs": vs_mantle_mdep, "vp": vp_mantle_mdep})

# 9) below Moho: all mantle as ONE layer (copy mantle_ref_depth values)
for d in resamp_depths[resamp_depths > mdep]:
    if d <= bottom_depth:
        rows.append({"dep": float(d), "vs": vs_mantle_ref, "vp": vp_mantle_ref})

new_model = pd.DataFrame(rows).sort_values("dep").reset_index(drop=True)
new_model = new_model.round(5)

# 10) save
new_model.to_csv(out_file, sep=" ", header=False, index=False, float_format="%.5f")

print("\nSaved resampled model to:", out_file)
print(new_model)
