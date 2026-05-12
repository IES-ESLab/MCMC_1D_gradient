#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
import numpy as np
import pandas as pd
import os, sys
import subprocess
import time
import matplotlib.pyplot as plt
#
import parameters
from src import setup_functions
#
# Suppress all warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------------------------
# Vp/Vs perturbation helper functions
# --------------------------------------------------------------------------------------------------------
def _is_on(name, default=0):
    """Read optional integer switch from parameters.py safely."""
    try:
        return int(getattr(parameters, name, default)) == 1
    except Exception:
        return False


def _fmt_float_str(v):
    """Return a clean string for values written to mod.{sta}."""
    try:
        return "%5.6f" % float(v)
    except Exception:
        return str(v)


def _expand_vpvs_values(group_name, npara, default_value):
    """
    Build initial Vp/Vs values for one model group.

    If parameters.uniform_vpvs == 1:
        use one scalar value and clone it to npara values.
    If parameters.uniform_vpvs == 0:
        use the list vpvs_{group_name}; its length must match npara.
    """
    npara = int(npara)
    uniform_vpvs = int(getattr(parameters, "uniform_vpvs", 1))
    attr = "vpvs_%s" % group_name

    if uniform_vpvs == 1:
        base = getattr(parameters, attr, default_value)
        if isinstance(base, (list, tuple, np.ndarray)):
            if len(base) == 0:
                base = default_value
            else:
                base = base[0]
        values = [base for _ in range(npara)]
    else:
        # tolerate the old typo where mantle was accidentally stored in vpvs_sed twice
        if (not hasattr(parameters, attr)) and group_name == "mantle":
            raw = getattr(parameters, "vpvs_mantle", default_value)
        else:
            raw = getattr(parameters, attr, default_value)

        if isinstance(raw, (list, tuple, np.ndarray)):
            values = list(raw)
        else:
            values = [raw]

        if len(values) != npara:
            print("ERROR: parameters.%s length = %d, but %s npara = %d" % (attr, len(values), group_name, npara))
            print("       Set uniform_vpvs = 1 to clone one value, or give exactly %d values." % npara)
            sys.exit(0)

    return [_fmt_float_str(v) for v in values]


def _vpvs_text(values):
    """Return one mod.{sta} text block for Vp/Vs values."""
    if len(values) == 0:
        return ""
    return " " + " ".join(values)


def _make_group_boundary_depths(top_depth, group_thick, npara):
    """
    Create simple group-boundary depths for plotting Vp/Vs.
    For Bspline groups, this is only for visual checking of initial Vp/Vs control values.
    """
    top_depth = float(top_depth)
    group_thick = float(group_thick)
    npara = int(npara)
    if npara <= 0:
        return []
    dz = group_thick / float(npara)
    return [top_depth + dz * ii for ii in range(0, npara + 1)]


def _detect_moho_index(depth, vs, man_flag=1, min_jump=0.20, search_min_depth=15.0, search_max_depth=None):
    """
    Detect Moho index from a positive Vs jump, but ignore shallow crustal jumps.

    The jump is between depth[i] and depth[i+1], and the returned Moho index is i+1
    (the lower-side point).  This prevents a shallow velocity increase, for example at
    6 km, from being mistaken as Moho when the expected crust/mantle boundary is
    deeper.
    """
    depth = np.asarray(depth, dtype=float)
    vs = np.asarray(vs, dtype=float)

    if len(depth) < 2:
        print("ERROR: input model must contain at least 2 depth points.")
        sys.exit(0)

    if search_max_depth is None:
        search_max_depth = float(np.nanmax(depth))

    dvs = np.diff(vs)
    lower_depth = depth[1:]

    # Prefer positive Moho-like jumps inside the search window.
    cand = np.where(
        (dvs >= float(min_jump)) &
        (lower_depth >= float(search_min_depth)) &
        (lower_depth <= float(search_max_depth))
    )[0]

    if len(cand) > 0:
        jump_i = cand[np.argmax(dvs[cand])]
        mohoindex = int(jump_i + 1)
        method = "largest positive jump >= %.3f within %.1f-%.1f km" % (
            float(min_jump), float(search_min_depth), float(search_max_depth)
        )
    else:
        # Fallback inside the depth window, then fallback globally if needed.
        win = np.where(
            (lower_depth >= float(search_min_depth)) &
            (lower_depth <= float(search_max_depth))
        )[0]
        if len(win) > 0:
            jump_i = win[np.argmax(np.abs(dvs[win]))]
            mohoindex = int(jump_i + 1)
            method = "largest absolute jump fallback within %.1f-%.1f km" % (
                float(search_min_depth), float(search_max_depth)
            )
        else:
            jump_i = int(np.argmax(np.abs(dvs)))
            mohoindex = int(jump_i + 1)
            method = "largest absolute jump global fallback"

    # If the selected Moho is the last sample, mantle would be empty.
    # Move it one point upward only for model setup safety.
    if int(man_flag) == 1 and mohoindex >= len(depth) - 1:
        print(">>> WARNING: detected Moho index is last depth sample; clamp from %d to %d so mantle is not empty." % (mohoindex, len(depth)-2))
        mohoindex = len(depth) - 2

    if mohoindex < 1:
        mohoindex = 1

    print(">>> Bspline auto Moho: index=%d depth=%.3f km by %s" % (mohoindex, depth[mohoindex], method))
    return mohoindex


def _resample_profile_for_bspline(depth, values, min_points, label):
    """
    Ensure enough profile samples for B-spline coefficient fitting.
    If the input group has fewer samples than npara, interpolate to npara samples.
    """
    depth = np.asarray(depth, dtype=float)
    values = np.asarray(values, dtype=float)
    min_points = int(min_points)

    if len(depth) == 0:
        print("ERROR: %s profile is empty. Check Moho detection and input model depth range." % label)
        sys.exit(0)

    # Remove duplicated depth points for np.interp stability.
    uniq_depth, uniq_idx = np.unique(depth, return_index=True)
    uniq_values = values[uniq_idx]
    depth, values = uniq_depth, uniq_values

    if len(depth) == 1:
        # Cannot interpolate from one point; create a tiny artificial interval for fitting.
        # The output group thickness in mod.{sta} still uses real crustthick/mantlethick.
        depth = np.linspace(depth[0], depth[0] + 1.0, max(min_points, 2))
        values = np.ones(len(depth)) * values[0]
        print(">>> WARNING: %s had only 1 sample; cloned value to %d samples for B-spline setup." % (label, len(depth)))
        return values.astype(np.float32), depth.astype(np.float32)

    if len(depth) < min_points:
        new_depth = np.linspace(depth[0], depth[-1], min_points)
        new_values = np.interp(new_depth, depth, values)
        print(">>> WARNING: %s had %d samples but npara=%d; interpolated to %d samples for B-spline setup." % (label, len(depth), min_points, min_points))
        return new_values.astype(np.float32), new_depth.astype(np.float32)

    return values.astype(np.float32), depth.astype(np.float32)


def setup(stadata,Vel_dir,src_dir,query_sub,data_sub,in_sub):
    # print(stadata)
    # --------------------------------------------------------------------------------------------------------
    # --------------------------------- Data import and preprocessing ----------------------------------------
    # --------------------------------------------------------------------------------------------------------
    # reading the model file:
    if (parameters.vel_mod_uniform_flag==1):
        modfile = os.path.join(Vel_dir,parameters.modfile); 
        # print("<< ",modfile)
    elif (parameters.vel_mod_uniform_flag==0):
        modfile = os.path.join(Vel_dir,parameters.modfile);
        modfile = os.path.join(modfile,f"{stadata['name']}.mod");
        # print("<< ",modfile)
    # Now read the model data 
    if os.path.exists(modfile):
        mod_data = pd.read_csv(modfile,delim_whitespace=True, na_values=0,names=["dep","vs",'vp'])
        mod_data = mod_data.fillna(0); 
    else:
        print("file %s does not exist!",mod_data)
        os.exit(0)
    
    # save out the Vs model file to check
    modfile_out = os.path.join(query_sub,f"{stadata['name']}_depth_vs.txt"); 
    modvs =  mod_data.iloc[:, :2]; 
    #
    modvs['dep'] = modvs['dep'].apply(lambda x: f"{x:5.3f}")
    modvs['vs'] = modvs['vs'].apply(lambda x: f"{x:5.5f}")
    #
    modvs.to_csv(modfile_out,index=False,header=True,sep=" ")
    #
    dVs=[]; dVsdepth=[]
    modstavs = modvs['vs'].to_numpy().astype(np.float32); 
    allgridmoddepth = modvs['dep'].to_numpy().astype(np.float32);
    tmp=allgridmoddepth; 
    # --------------------------------------------------------------------------------------------------------
    # ---------------------------------  in.para_{sta} file produce   ----------------------------------------
    # --------------------------------------------------------------------------------------------------------
    if (parameters.sed_flag==1):
        #5th column
        SedLayerId='0'; 
        CrustLayerId='1'; 
        locals()['inparasedthick'] = np.append(parameters.sediment_thick,[SedLayerId]); 
         
        if (parameters.man_flag==1):
            ManLayerId='2'; 
            for x in range(0,parameters.mantlenpara):
                locals()['inparamantleVs%s' % x] = np.append(parameters.manlte_vel, [ManLayerId, str(x)]); 
                # print('inparamantleVs',x, locals()['inparamantleVs%s' % x])
        #
        
        sed_value = parameters.sed_value; 
        for x in range(0,parameters.sednpara):
            locals()['inparasedVs%s' % x] = np.append(parameters.sediment_vel,[SedLayerId, str(x)])
            # print('inparasedVs',x, locals()['inparasedVs%s' % x])
    else:
        #5th column
        CrustLayerId='0'; 
        if (parameters.man_flag==1):
            ManLayerId='1'; 
            for x in range(0,parameters.mantlenpara):
                locals()['inparamantleVs%s' % x] = np.append(parameters.manlte_vel, [ManLayerId, str(x)]); 
                # print('inparamantleVs',x, locals()['inparamantleVs%s' % x])
        sed_value=modvs['vs'][0];     

    # ------------------------------------------------------------------------------------
    # Vp/Vs perturbation setup for Bspline
    # Do NOT change parameters.py values; only override local output flags/values here.
    # ------------------------------------------------------------------------------------
    use_sed_vpvs = _is_on("sublay_vpvschange") and _is_on("subsedvpvschange")
    use_crust_vpvs = _is_on("sublay_vpvschange") and _is_on("subcrustvpvschange")
    use_mantle_vpvs = _is_on("sublay_vpvschange") and _is_on("submantlevpvschange") and (parameters.man_flag == 1)

    sedPflag_out = '5' if use_sed_vpvs else parameters.sedPflag
    crustPflag_out = '5' if use_crust_vpvs else parameters.crustPflag
    mantlePflag_out = '5' if use_mantle_vpvs else parameters.mantlePflag

    sed_vpvs_values = []
    crust_vpvs_values = []
    mantle_vpvs_values = []

    if use_sed_vpvs and parameters.sed_flag == 1:
        sed_vpvs_values = _expand_vpvs_values("sed", parameters.sednpara, parameters.sedmodvpvs)
        for x in range(0, parameters.sednpara):
            locals()['inparasedVpVs%s' % x] = np.append(parameters.sediment_vpvs, [SedLayerId, str(x)])

    if use_crust_vpvs:
        crust_vpvs_values = _expand_vpvs_values("crust", parameters.crustnpara, parameters.crmodvpvs)
        for x in range(0, parameters.crustnpara):
            locals()['inparacrustVpVs%s' % x] = np.append(parameters.crust_vpvs, [CrustLayerId, str(x)])

    if use_mantle_vpvs:
        mantle_vpvs_values = _expand_vpvs_values("mantle", parameters.mantlenpara, parameters.mtmodvpvs)
        for x in range(0, parameters.mantlenpara):
            locals()['inparamantleVpVs%s' % x] = np.append(parameters.manlte_vpvs, [ManLayerId, str(x)])

    print(">>> Vp/Vs perturbation flags: sed=%s crust=%s mantle=%s" % (sedPflag_out, crustPflag_out, mantlePflag_out))
    if use_crust_vpvs:
        print(">>> Initial crust Vp/Vs values:", crust_vpvs_values)
    if use_mantle_vpvs:
        print(">>> Initial mantle Vp/Vs values:", mantle_vpvs_values)
        
    for x in range(0,parameters.crustnpara):
        locals()['inparacrustVs%s' % x] = np.append(parameters.crust_vel, [CrustLayerId, str(x)]); 
        # print('inparacrustVs',x, locals()['inparacrustVs%s' % x]) 
        
# ---------------------- Now we find the sediment depth and moho depth ---------------------------
    modstavs[0] = sed_value; 
    for kk in np.arange(1,len(allgridmoddepth)-1):
            dVs.append(abs((modstavs[kk+1]-modstavs[kk-1])/(allgridmoddepth[kk+1]-allgridmoddepth[kk-1])))
            dVsdepth.append((allgridmoddepth[kk+1]+allgridmoddepth[kk-1])*0.5)
    # print(dVsdepth,dVs)

    # Old version used hard-coded mohoindex=8; this fails when the model has <= 8 rows.
    # Detect Moho from Vs jumps instead, then keep the index inside the valid depth array.
    try:
        prominence,prdep=setup_functions.peakdet2(dVs,dVsdepth)
    except Exception:
        prominence=[]
        prdep=[]

    mohoindex = _detect_moho_index(
        allgridmoddepth,
        modstavs,
        man_flag=parameters.man_flag,
        min_jump=float(getattr(parameters, "moho_min_jump", 0.20)),
        search_min_depth=float(getattr(parameters, "moho_search_min_depth", 15.0)),
        search_max_depth=getattr(parameters, "moho_search_max_depth", None)
    )
    mohodepth=allgridmoddepth[mohoindex]

    if len(dVs) > 0 and len(dVsdepth) > 0:
        n_sed_search = min(4, len(dVs))
        dVsmax1tmp=np.where(np.asarray(dVs[0:n_sed_search])==max(dVs[0:n_sed_search]))[0][0]
        seddepth,sedindex=setup_functions.find_nearest(allgridmoddepth,dVsdepth[dVsmax1tmp])
    else:
        seddepth=0
        sedindex=0

    moho_id = mohoindex+1; 
    # Correction the sediment and moho depth based on model settup
    if (parameters.sed_flag==1):
        crustthick=mohodepth-seddepth; 
        sedthick=seddepth; 
    elif (parameters.sed_flag==0):
        sedthick=0; 
        seddepth=0; 
        crustthick=mohodepth; 
    mantlethick=max(allgridmoddepth)-mohodepth
    pertRangCrustThick=np.round((np.float32(parameters.PertRangeCrustThick)*crustthick/100),1)
    # print("pertRangCrustThick: ", pertRangCrustThick)
    if np.isnan(pertRangCrustThick):
        print('error!!!! percpert')
        sys.exit(0); 
    print(">>>seddepth, mohodepth,crustthick, sedindex,mohoindex:",seddepth, mohodepth,crustthick,sedindex,mohoindex)
    parameters.crust_thick[2]=pertRangCrustThick; 
    if (parameters.man_flag==1): # If no manlte then no crustal thickness change.
        locals()['inparacrustthick']=np.append(parameters.crust_thick,CrustLayerId); 
    # ---------------------- in.para_{sta} print out ----------------------------------------------------------------
    # ------------------------------- >>>>>>>> write in.para_{STA} here !!! <<<<<<< ----------------------------------
    fnpara=os.path.join(data_sub, f"in.para_{stadata['name']}")
    print("Write the in.para: ",fnpara)
    # # Remove the file if exist
    # if os.path.exists(fnpara):
    #     os.remove(fnpara); 
    with open(fnpara,'w') as ff0:
        # 1) Velocity perturbation rows
        if (parameters.sed_flag==1):
            for x in range(0,parameters.sednpara):
                ff0.write(' '.join(locals()['inparasedVs%s' % x])+'\n')
        for x in range(0,parameters.crustnpara):
            ff0.write(' '.join(locals()['inparacrustVs%s' % x])+'\n')
        if (parameters.man_flag==1):
            for x in range(0,parameters.mantlenpara):
                ff0.write(' '.join(locals()['inparamantleVs%s' % x])+'\n')

        # 2) Thickness perturbation rows
        #    Keep crustal thickness perturbation between velocity and Vp/Vs perturbation.
        if (parameters.sed_flag==1):
            ff0.write(' '.join(locals()['inparasedthick'])+'\n')
        if (parameters.man_flag==1):
            ff0.write(' '.join(locals()['inparacrustthick'])+'\n')

        # 3) Vp/Vs perturbation rows
        if use_sed_vpvs and (parameters.sed_flag==1):
            for x in range(0,parameters.sednpara):
                ff0.write(' '.join(locals()['inparasedVpVs%s' % x])+'\n')
        if use_crust_vpvs:
            for x in range(0,parameters.crustnpara):
                ff0.write(' '.join(locals()['inparacrustVpVs%s' % x])+'\n')
        if use_mantle_vpvs and (parameters.man_flag==1):
            for x in range(0,parameters.mantlenpara):
                ff0.write(' '.join(locals()['inparamantleVpVs%s' % x])+'\n')
    ff0.close()
    # --------------------------------------------------------------------------------------------------------
    # ---------------------------------  in.data_{sta} file produce   ----------------------------------------
    # --------------------------------------------------------------------------------------------------------   
    fndata = os.path.join(data_sub, f"in.data_{stadata['name']}")
    if (parameters.is_in_data==1):
        # input data name format
        gvfile = os.path.join(data_sub, f"{stadata['name']}.gv")
        phfile = os.path.join(data_sub, f"{stadata['name']}.ph")
        hvfile = os.path.join(data_sub, f"{stadata['name']}.HV")
        rffile = os.path.join(data_sub, f"{stadata['name']}.RF")
        #
        nindata=0
        print("Write the in.data with misfit based on true input data: ",fndata)
        with open(fndata,'w') as ff0:
            if os.path.exists(phfile):
                ff0.write("1 \n")
                nindata+=1; 
                pindata=1; 
            else:
                ff0.write("0 \n")
                pindata=0; 
            #
            if os.path.exists(gvfile):
                ff0.write("1 \n")
                nindata+=1; 
                gindata=1;  
            else:
                ff0.write("0 \n")
                gindata=0; 
            #
            if os.path.exists(hvfile):
                ff0.write("1 \n")
                nindata+=1; 
                hindata=1; 
            else:
                ff0.write("0 \n")
                hindata=0; 
            #
            if os.path.exists(rffile):
                ff0.write("1 \n")
                nindata+=1; 
                rindata=1; 
            else:
                ff0.write("0 \n")
                rindata=0; 
            print(">>>",pindata,gindata,hindata,rindata,nindata)
            if (parameters.is_equal_weight==0):
                print("Write the weighting 0")
                ff0.write("{:d} \n".format(parameters.is_equal_weight))
                #
                ff0.write("{} \n".format(parameters.phw))
                ff0.write("{} \n".format(parameters.gvw))
                ff0.write("{} \n".format(parameters.hvw))
                ff0.write("{} \n".format(parameters.rfw))
            else:
                print("Write the weighting 1")
                ff0.write("{:d} \n".format(parameters.is_equal_weight))
                #
                ff0.write("{:.6f} \n".format(pindata/nindata))
                ff0.write("{:.6f} \n".format(gindata/nindata))
                ff0.write("{:.6f} \n".format(hindata/nindata))
                ff0.write("{:.6f} \n".format(rindata/nindata))

        ff0.close()
    elif (parameters.is_in_data==0):
        print("Write the in.data with optional data misfit style: ",fndata)
        with open(fndata,'w') as ff0:
            ff0.write("{} \n".format(parameters.iph))
            ff0.write("{} \n".format(parameters.igv))
            ff0.write("{} \n".format(parameters.ihv))
            ff0.write("{} \n".format(parameters.irf))
            if (parameters.is_equal_weight==0):
                print("Write the weighting 0")
                ff0.write("{:d} \n".format(parameters.is_equal_weight))
                #
                ff0.write("{} \n".format(parameters.phw))
                ff0.write("{} \n".format(parameters.gvw))
                ff0.write("{} \n".format(parameters.hvw))
                ff0.write("{} \n".format(parameters.rfw))
            else:
                print("Write the weighting 1")
                ff0.write("{:d} \n".format(parameters.is_equal_weight))
                #
                ff0.write("{:.6f} \n".format(pindata/nindata))
                ff0.write("{:.6f} \n".format(gindata/nindata))
                ff0.write("{:.6f} \n".format(hindata/nindata))
                ff0.write("{:.6f} \n".format(rindata/nindata))
        ff0.close()
        if (parameters.igv==1):
            gvfile = os.path.join(data_sub, f"{stadata['name']}.gv")
            if not os.path.exists(gvfile):
                gvfile = os.path.join(in_sub, "in.gv")
                print("---!!!  Warning! Using fake GV data: {}".format(gvfile))
        else:
            gvfile=""; 
        if (parameters.iph==1):
            phfile = os.path.join(data_sub, f"{stadata['name']}.ph")
            if not os.path.exists(phfile):
                phfile = os.path.join(in_sub, "in.ph")
                print("---!!!  Warning! Using fake PH data: {}".format(phfile))
        else:
            phfile=""; 
        if (parameters.ihv==1):
            hvfile = os.path.join(data_sub, f"{stadata['name']}.HV")
            if not os.path.exists(hvfile):
                hvfile = os.path.join(in_sub, "in.hv")
                print("---!!!  Warning! Using fake HV data: {}".format(hvfile))
        else:
            hvfile=""; 
        if (parameters.irf==1):
            rffile = os.path.join(data_sub, f"{stadata['name']}.RF")
            if not os.path.exists(rffile):
                rffile = os.path.join(in_sub, "in.rf")
                print("---!!!  Warning! Using fake RF data: {}".format(rffile))
        else:
            rffile=""; 
    else:
        print("Wrong values of [is_in_data]. Stop!")
        sys.exit(0)
    # --------------------------------------------------------------------------------------------------------
    # -----------------------------------  {sta}.mod file produce   ------------------------------------------
    # --------------------------------------------------------------------------------------------------------

    # --------------------------- Bspline calculation for model ----------------------------------------------
    if (parameters.sed_flag==1):
        vsBspinversion=modstavs[1:moho_id]; depthBspinversion=allgridmoddepth[1:moho_id]
        # include the Moho point in the mantle fitting profile so sparse models still have a boundary value
        mvsBspinversion=modstavs[mohoindex:]; mdepthBspinversion=allgridmoddepth[mohoindex:]
    else:
        vsBspinversion=modstavs[0:moho_id]; depthBspinversion=allgridmoddepth[0:moho_id]
        # include the Moho point in the mantle fitting profile so sparse models still have a boundary value
        mvsBspinversion=modstavs[mohoindex:]; mdepthBspinversion=allgridmoddepth[mohoindex:]

    vsBspinversion, depthBspinversion = _resample_profile_for_bspline(
        depthBspinversion, vsBspinversion, parameters.crustnpara, "crust"
    )
    if (parameters.man_flag==1):
        mvsBspinversion, mdepthBspinversion = _resample_profile_for_bspline(
            mdepthBspinversion, mvsBspinversion, parameters.mantlenpara, "mantle"
        )
    # run Bspline for crustal
    cmd=src_dir+'/lfBsp_nlay 0 '+str(len(vsBspinversion))+' 2.0 '+str(parameters.crustnpara)+' 4 '+str(len(vsBspinversion))
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()
    bspfile='B_spline_'
    FCBs=[]
    # maxtrix forming for crustal
    for kk in np.arange(parameters.crustnpara):
        FCBs.append(np.loadtxt(os.path.dirname(src_dir)+'/'+bspfile+str(kk)+'.txt',usecols=(1),unpack=True))
        cmd='mv '+bspfile+str(kk)+'.txt '+bspfile+str(kk)+'_crust.txt'
        pr = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        pr.wait()
    # run Bspline for mantle
    cmd=src_dir+'/lfBsp_nlay 0 '+str(len(mvsBspinversion))+' 0.75 '+str(parameters.mantlenpara)+' 4 '+str(len(mvsBspinversion))
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()
    bspfile='B_spline_'
    mFCBs=[]
    # maxtrix forming for mantal   
    for kk in np.arange(parameters.mantlenpara):
        mFCBs.append(np.loadtxt(os.path.dirname(src_dir)+'/'+bspfile+str(kk)+'.txt',usecols=(1),unpack=True))
        cmd='mv '+bspfile+str(kk)+'.txt '+bspfile+str(kk)+'_mantle.txt'
        pr = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        pr.wait()
        
    # Inverted the coefficient to get the inverted value
    FCGmatrix=np.vstack([FCBs]).T
    mFCGmatrix=np.vstack([mFCBs]).T
    #print(FCGmatrix,mFCGmatrix)
    # upload and invert bsplines
    FCcrustcoeff=np.linalg.lstsq(FCGmatrix,vsBspinversion)[0]
    mFCcrustcoeff=np.linalg.lstsq(mFCGmatrix,mvsBspinversion)[0]
    # solve the equaltion
    #vsfromcrustcoeff=np.dot(Gmatrix,crustcoeff)
    FCvsfromcrustcoeff=np.dot(FCGmatrix,FCcrustcoeff)
    mFCvsfromcrustcoeff=np.dot(mFCGmatrix,mFCcrustcoeff)
    if (parameters.sed_flag==1):
        sedvalue=[]; seddepth=[]
        sedvalue.append(modstavs[0]); seddepth.append(allgridmoddepth[0])
        sedvalue.append(modstavs[1]); seddepth.append(allgridmoddepth[1])
    
        if sedvalue[0]==0:
            sedvalue[0]=0.5
        if sedvalue[1]==0:
            sedvalue[1]=0.
        
# # ----------------- >>>>> More calculation >>>> print to mod.{STA} ---------------------------------
    # sediment
    if (parameters.sed_flag==1):
        sthick=seddepth[len(seddepth)-1]-seddepth[0]
        locals()['modsedlayer']=SedLayerId+' '+parameters.modlinear+' %4.1f'%(sthick)+' %d'%(parameters.sednpara)+' %5.2f'%(float(parameters.dds))+\
        ' %5.6f %5.6f '%(sedvalue[0],sedvalue[1])+_vpvs_text(sed_vpvs_values)+' '+parameters.sedrhoflag+' '+parameters.sedQflag+' '+sedPflag_out+' '+\
        parameters.sedmodvpvs+' '+parameters.sedVpVs1+' '+parameters.sedDrho+' '+parameters.sedDrho1+' '+parameters.seddVs+' '+\
        parameters.seddvs1+' '+parameters.sedfdvs+' '+parameters.sedfdvs1
    # crust
    printcrusttmp=''
    for kk in np.arange(parameters.crustnpara):
        printcrusttmp=printcrusttmp+' %5.6f'%(FCcrustcoeff[kk])
    # 
    locals()['modcrustlayer']=CrustLayerId+' '+parameters.modsplines+' %4.1f %d %5.2f'%(crustthick,parameters.crustnpara,float(parameters.ddc))+printcrusttmp+_vpvs_text(crust_vpvs_values)+\
    ' '+parameters.crustrhoflag+' '+parameters.crustQflag+' '+crustPflag_out+' '+parameters.crmodvpvs+' '+parameters.crVpVs1+' '+\
    parameters.crDrho+' '+parameters.crDrho1+' '+parameters.crdVs+' '+parameters.crdvs1+' '+parameters.crfdvs+' '+parameters.crfdvs1
    # Mantle
    if (parameters.man_flag==1):
        printmantletmp=''
        for kk in np.arange(parameters.mantlenpara):
            printmantletmp=printmantletmp+' %5.6f'%(mFCcrustcoeff[kk])
        locals()['modmantlelayer']=ManLayerId+' '+parameters.modsplines+' %4.1f %d %5.2f'%(mantlethick,parameters.mantlenpara,float(parameters.ddm))+printmantletmp+_vpvs_text(mantle_vpvs_values)+\
        ' '+parameters.mantlerhoflag+' '+parameters.mantleQflag+' '+mantlePflag_out+' '+parameters.mtmodvpvs+' '+parameters.mtVpVs1+' '+\
        parameters.mtDrho+' '+parameters.mtDrho1+' '+parameters.mtdVs+' '+parameters.mtdVs1+' '+parameters.mtfdvs+' '+parameters.mtfdvs1
# ------------------------------- >>>>>>>> write mod.{STA} here !!! <<<<<<< ----------------------------------
    fnmod= os.path.join(data_sub, f"mod.{stadata['name']}")
    print("Write the model file in bspline style: ",fnmod)
    with open (fnmod, 'w') as ff2:
        if (parameters.sed_flag==1):
            ff2.write(locals()['modsedlayer']+'\n'); 
        ff2.write(locals()['modcrustlayer']+'\n'); 
        if (parameters.man_flag==1):
            ff2.write(locals()['modmantlelayer']+'\n')
    ff2.close()
# ------------------------------- >>>>>>>> write to in.connector here !!! <<<<<<< ----------------------------------

    fconnect= os.path.join(data_sub, 'in.connector')
    print("Write the connector file to: ",fconnect)
    with open(fconnect,'w') as ff3:
        ff3.write(str(parameters.MC_inversion_type)+"\n")
        ff3.write(str(parameters.RF_gaussian_width)+"\n")
        ff3.write(str(parameters.MC_number_of_jump)+"\n")
        ff3.write(str(parameters.MC_number_of_iteration)+"\n")
        ff3.write(str(parameters.MC_number_of_cores)+"\n")
        ff3.write(str(parameters.MC_percentage_post_process_select)+"\n")
        ff3.write(str(parameters.depth_step)+"\n")
        ff3.write(str(parameters.modout_plot_type)+"\n")
        # ff3.write(str(parameters.sed_mono_check)+"\n")
        # ff3.write(str(parameters.crust_mono_check)+"\n")
    ff3.close()
    # --------------------------------------------------------------------------------------------------------
    # -----------------------------------  Figure plot for initial setup  ------------------------------------
    # --------------------------------------------------------------------------------------------------------
    print("Plot the input model of station %s in Bspline type"%(stadata['name']))
    # read in Vph, H/V, and RF info from file
    if os.path.exists(gvfile):
        allgvper,mergedgv,mergedgvun=np.loadtxt(gvfile,usecols=[0,1,2],unpack=True)
    if os.path.exists(phfile):
        allphper,mergedph,mergedphun=np.loadtxt(phfile,usecols=[0,1,2],unpack=True)
    if os.path.exists(hvfile):
        hvper,hvall,hvun=np.loadtxt(hvfile,usecols=[0,1,2],unpack=True)
    if os.path.exists(rffile):
        rft,rfamp,rfunc=np.loadtxt(rffile,usecols=[0,1,2],unpack=True)
#     ##%% Plot all
# #    '''
    fig=plt.figure(0,figsize=(10,10))
    
    ax1=plt.subplot2grid((3,2), (0,0), rowspan=3) #Velocity models
    ax4=plt.subplot2grid((3,2), (0,1)) #Phase and gv
    ax2=plt.subplot2grid((3,2), (1,1)) #H/V
    ax3=plt.subplot2grid((3,2), (2,1)) #RF
# 
    ax1.set_facecolor('lightyellow')
    ax4.set_facecolor('lightyellow')
    ax2.set_facecolor('lightyellow')
    ax3.set_facecolor('lightyellow')
    
    # Vs
    if (parameters.sed_flag==1):
        ax1.plot(sedvalue,seddepth,'yo',label='Sed Picks')
    ax1.plot(FCvsfromcrustcoeff,depthBspinversion,'ko',label='Est from '+str(parameters.crustnpara)+' Bsplines Vs',alpha=0.4)
    if (parameters.man_flag==1):
        ax1.plot(mFCvsfromcrustcoeff,mdepthBspinversion,'gs',label='Est from '+str(parameters.mantlenpara)+' Bsplines Vs',alpha=0.4)
    ax1.plot(modstavs,allgridmoddepth,'r-',label='Raw Vs Model')

    # Plot initial Vp/Vs control values on a twin x-axis
    ax1_vpvs = ax1.twiny()
    ax1_vpvs.set_facecolor('none')
    vpvs_plot_depth = []
    vpvs_plot_value = []
    if use_sed_vpvs and (parameters.sed_flag == 1):
        sed_bound = _make_group_boundary_depths(0.0, sedthick, parameters.sednpara)
        if len(sed_bound) > 0:
            vpvs_plot_depth += sed_bound
            vpvs_plot_value += [float(sed_vpvs_values[0])] + [float(v) for v in sed_vpvs_values]
    if use_crust_vpvs:
        crust_bound = _make_group_boundary_depths(seddepth, crustthick, parameters.crustnpara)
        if len(crust_bound) > 0:
            vpvs_plot_depth += crust_bound
            vpvs_plot_value += [float(crust_vpvs_values[0])] + [float(v) for v in crust_vpvs_values]
    if use_mantle_vpvs and (parameters.man_flag == 1):
        mantle_bound = _make_group_boundary_depths(mohodepth, mantlethick, parameters.mantlenpara)
        if len(mantle_bound) > 0:
            vpvs_plot_depth += mantle_bound
            vpvs_plot_value += [float(mantle_vpvs_values[0])] + [float(v) for v in mantle_vpvs_values]
    if len(vpvs_plot_depth) > 0:
        ax1_vpvs.plot(vpvs_plot_value, vpvs_plot_depth, 'bD-', ms=5, lw=1.5, label='Initial Vp/Vs')
        ax1_vpvs.set_xlabel('Vp/Vs', fontdict={'family':'serif','color':'blue','size':12,'weight':'bold'})
        ax1_vpvs.tick_params(axis='x', colors='blue', labeltop=True)
        ax1_vpvs.legend(loc='upper right', fontsize=10)
        print(">>> Plot Vp/Vs depth/value pairs:")
        for dd, vv in zip(vpvs_plot_depth, vpvs_plot_value):
            print("    depth=%8.3f km   vpvs=%8.4f" % (dd, vv))
    
    ax1.set_ylim([0,allgridmoddepth[-1]])
    ax1.legend(loc='lower left',fontsize=12)
    
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.set_xlim([0.0,5.7])
    ax1.set_ylabel('Depth (km)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
    ax1.set_xlabel('Vs (km/s)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
    ax1.tick_params(labeltop=True)
    ax1.grid(color='k', axis='both',linestyle='-', linewidth=2,alpha=0.1,zorder=2)
    # PHASE
    if os.path.exists(phfile):
        ax2.errorbar(allphper,mergedph,yerr=mergedphun,fmt='.:',color='r',ecolor='r',elinewidth=1.5,capthick=1.5,label='Vph',zorder=5,alpha=0.5)
    if os.path.exists(gvfile):
        ax2.errorbar(allgvper,mergedgv,yerr=mergedgvun,fmt='.:',color='b',ecolor='b',elinewidth=1.5,capthick=1.5,label='Vgv',zorder=5,alpha=0.5)
    # ax2.errorbar(phper,mergedph[0:len(phper)],yerr=mergedphun[0:len(phper)],fmt='.:',color='k',ecolor='k',elinewidth=1.5,capthick=1.5,label='ANT Vph',zorder=5,alpha=0.5)
    #ax2.errorbar(allphper,mergedph,yerr=mergedphun,fmt='.:',color='k',ecolor='r',elinewidth=1.5,capthick=1.5,label='All Vph',zorder=5,alpha=0.5)
    ax2.set_xlabel('Period (s)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
    ax2.set_ylabel('Vph (km/s)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
    #ax2.set_xlim([0.0,105.0])
    ax2.set_ylim([0,5.0])
    ax2.set_xlim([0,30])
    ax2.legend(loc='lower right',fontsize=12)
    ax2.tick_params(labeltop=False)
    ax2.grid(color='k', axis='both',linestyle='-', linewidth=2,alpha=0.1,zorder=2)

    # H/V
    if os.path.exists(hvfile):
        ax3.errorbar(hvper,hvall,yerr=hvun,fmt='.:',color='r',ecolor='r',elinewidth=1.5,capthick=1.5,label='H/V',zorder=5,alpha=0.5)
    #ax3.errorbar(hvper,hvall,yerr=hvun,fmt='.:',color='r',ecolor='k',elinewidth=1.5,capthick=1.5,label='All H/V',zorder=5,alpha=0.5)
    ax3.set_xlim([0,40])
    ax3.set_xlabel('Period (s)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
    ax3.set_ylabel('H/V',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
    ax3.set_ylim([0,2.0])
    ax3.legend(loc='upper right',fontsize=12)
    ax3.tick_params(labeltop=False)
    ax3.grid(color='k', axis='both',linestyle='-', linewidth=2,alpha=0.1,zorder=2)

    #RF
    if os.path.exists(rffile):
        ax4.errorbar(rft,rfamp,yerr=rfunc,fmt='.:',color='k',ecolor='k',elinewidth=1.5,capthick=0.0,label='RFunc',zorder=5,alpha=0.5)
        ax4.plot(rft,rfamp,'r-',zorder=7,alpha=0.96,label='RF')
        ax4.set_xlim([min(rft)-0.2,max(rft)+0.5])
        ax4.set_ylim([-0.25,0.6])
        ax4.legend(loc='upper right',fontsize=12)
    ax4.set_xlabel('Time (s)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
    ax4.set_ylabel('RF Amp',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
    ax4.tick_params(labeltop=False)
    ax4.grid(color='k', axis='both',linestyle='-', linewidth=2,alpha=0.1,zorder=2)

    plt.suptitle("Starting model and input data for station %s"%(stadata['name']),fontdict = {'family':'serif','color':'darkred','size':25,'weight':'bold'})

    #plt.show()
    figout0 = os.path.join(query_sub,f"StartingModel_{stadata['name']}.png")
    fig.savefig(figout0,bbox_inches='tight',transparent=False,pad_inches=0.1)
    plt.close()
    

    fig, axs = plt.subplots(1,2, figsize=(8,11), sharey=True)
    axs[0].set_facecolor('lightyellow')
    axs[1].set_facecolor('lightyellow')

    axs[1].set_title('B-spline (%s, %s)' %(parameters.crustnpara, parameters.mantlenpara), fontsize=24)

    axs[0].plot(mod_data['vs'], mod_data['dep'], 'ko-',label="Input model")
    if (parameters.sed_flag==1):
        # sed value
        axs[0].plot(sedvalue, tmp[0:2], 'c.-',label="Sed Val")
        # value from initial
        axs[0].plot(vsBspinversion, depthBspinversion, 'ro-',ms=10, alpha=0.5,label="Crustal Val from input")
        if (parameters.man_flag==1):
            axs[0].plot(mvsBspinversion, mdepthBspinversion, 'rs-',ms=10, alpha=0.5,label="Mantle Val from input")
        # value from Bspline coefficient
        axs[0].plot(FCvsfromcrustcoeff, depthBspinversion, 'bo--', alpha=0.5,label="Val from crustal coeff")
        if (parameters.man_flag==1):
            axs[0].plot(mFCvsfromcrustcoeff, mdepthBspinversion, 'bs--', alpha=0.5,label="Val from mantle coeff")
        # value from initial
        axs[1].plot(vsBspinversion, depthBspinversion, 'bo--',ms=10,label="Crustal Val from input")
        if (parameters.man_flag==1):
            axs[1].plot(mvsBspinversion, mdepthBspinversion, 'bs-',ms=10,label="Mantle Val from input")
        # value from Bspline coefficient
        axs[1].plot(FCvsfromcrustcoeff, depthBspinversion, 'ro-',label="Val from crustal coeff")
        if (parameters.man_flag==1):
            axs[1].plot(mFCvsfromcrustcoeff, mdepthBspinversion, 'rs--',label="Val from mantle coeff")

    else: # no sediment
        # value from initial
        axs[0].plot(vsBspinversion, depthBspinversion, 'ro-',ms=10, alpha=0.5,label="Crustal Val from input")
        if (parameters.man_flag==1):
            axs[0].plot(mvsBspinversion, mdepthBspinversion, 'rs-',ms=10, alpha=0.5,label="Mantle Val from input")
        # value from Bspline coefficient
        axs[0].plot(FCvsfromcrustcoeff, depthBspinversion, 'bo--', alpha=0.5,label="Val from crustal coeff")
        if (parameters.man_flag==1):
            axs[0].plot(mFCvsfromcrustcoeff, mdepthBspinversion, 'bs--', alpha=0.5,label="Val from mantle coeff")
        # value from initial
        axs[1].plot(vsBspinversion, depthBspinversion, 'bo-',ms=10,label="Crustal Val from input")
        if (parameters.man_flag==1):
            axs[1].plot(mvsBspinversion, mdepthBspinversion, 'bs--',ms=10,label="Mantle Val from input")
        # value from Bspline coefficient
        axs[1].plot(FCvsfromcrustcoeff, depthBspinversion, 'ro-',label="Val from crustal coeff")
        if (parameters.man_flag==1):
            axs[1].plot(mFCvsfromcrustcoeff, mdepthBspinversion, 'rs--',label="Val from mantle coeff")

    axs[0].tick_params(labeltop=False)
    axs[0].grid(color='k', axis='both',linestyle='-', linewidth=2,alpha=0.1,zorder=2)
    axs[0].legend(loc='lower left',fontsize=12)
    axs[0].set_ylim(axs[0].get_ylim()[::-1])
    axs[0].set_ylabel('Depth (km)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
    axs[0].set_xlabel('Vs (km/s)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
    
    axs[1].tick_params(labeltop=False)
    axs[1].grid(color='k', axis='both',linestyle='-', linewidth=2,alpha=0.1,zorder=2)
    axs[1].legend(loc='lower left',fontsize=12)
    axs[1].set_ylim(axs[0].get_ylim())
    axs[1].set_xlim(axs[0].get_xlim())
    axs[1].set_ylabel('Depth (km)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
    axs[1].set_xlabel('Vs (km/s)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})

    # Adjust layout to prevent overlap
    # plt.tight_layout()
    
    plt.suptitle("Model interpolation for station: %s"%(stadata['name']),fontdict = {'family':'serif','color':'darkred','size':55,'weight':'bold'})

    figout1 = os.path.join(query_sub,f"B-spline_{str(parameters.crustnpara)}_{str(parameters.mantlenpara)}_comparisons_{stadata['name']}.png")
    fig.savefig(figout1,bbox_inches='tight',transparent=False,pad_inches=0.1)
    plt.close()
    
# if __name__ == "__setup_bspline__":
#     setup()

