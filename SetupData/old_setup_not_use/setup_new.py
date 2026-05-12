#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Imports...
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import subprocess
import time 
############################################## function to use in this script ##################################################### 
#function to linearly interpolate
def lininterpwiki(x,x0,y0,x1,y1):
    tmp=((x-x0)/(x1-x0))
    y=(y0*(1-tmp))+(y1*tmp)
    return y
# ----------------------------------------------------------------------------------
#load and find nearest
def find_nearest(array,value):
    #may be useful to determine grid value closest to station from Hongrui Results
    idx=(np.abs(array-value)).argmin();
    return array[idx],idx
#Example: lonsta,ii=find_nearest(llons,lontmp)
# ----------------------------------------------------------------------------------
##%% take derivative, find peaks, create starting model...
## Peak Detection ##
def peakdet2(v, depth):
    """
    Elizabeth Berg's version of detecting peak prominence
    Should work better (maybe?) than peakdet from Eli Billauer       
    Similar to FTAN, if a peak is 'prominent', then include   
    """
    if len(v)!=len(depth):
        #print len(v)
        #print '!='
        #print len(depth)
        print('ERROR!!!! len(v)!=len(depth)')
        sys.exit()    
    allprom=[]; depout=[]
    for ii in np.arange(len(v)-2):
        h1=depth[ii+1]-depth[ii]
        h2=depth[ii+2]-depth[ii+1]
        h3=h1+h2
        
        tmp=((v[ii]/h1)-(((1./h1)+(1./h2))*v[ii+1])+(v[ii+2]/h2))*(h3/4.)*(10.)
        allprom.append(tmp)
        depout.append(depth[ii+1])
    return allprom,depout
## Read the data_avaiable_type and give a flag
def read_data_avaiable(link):
    with open(link, 'r') as file:
        line_count=0;
        for line in file:
            line = line.strip()
            if not line.startswith('#'):
                line_count+=1
                if(line_count==1):
                    phase_flag=int(line.strip()[0])
                elif(line_count==2):
                    group_flag=int(line.strip()[0])
                elif(line_count==3):
                    ellip_flag=int(line.strip()[0])
                elif(line_count==4):
                    rf_flag=int(line.strip()[0])
                else:
                    print("Why we can reach to this line count?")
    file.close()
    total_flag=(phase_flag*1)+(group_flag*2)+(ellip_flag*4)+(rf_flag*8)
    return total_flag
####################################################################################################################################
# ##### SETTINGS - EDIT HERE ###### #
pwd = os.getcwd()
if 1==1:
    datadir=pwd+'/../CHTH3/'#+num+'_20hz'
    scriptdir=pwd
    data_avaiable_dir=datadir+'/inf/indata.inf' # which kind of data you have will regist here!
    # Setting for some case # HVLong
    # ------------------ seting for the model type  ------------------------------------------
    # mod_type_flag: 1 = layered; 2 = Bspline ; 4 = linear;  
    # Note: if mode_type_flag = 2 and sediment = 1 then sediment model type = linear
    mod_type_flag = 2
    # ---------------------------------------------------------------------------------------
    # model flag when use similar 1D velocity model or difference velocity layer for each survey points
    # (vel_mod_uniform_flag = 0: use difference models - need define model names consistent with survey point - see modfile_dir)
    # (vel_mod_uniform_flag = 1: use 1 model for all)
    vel_mod_uniform_flag = 1
    # sed_flag: = 0 no sediment setting, = 1 with sediment setting
    # if  sed_flag = 1: the 1st layer of 1D model will be replace by sed_value
    sed_flag = 1
    # if  man_flag = 0: no mantle setup for run (modify mod STA.mod and inpara.STA files)
    # If man_flag = 0 then no depth pertubation for crustal
    man_flag = 0
    # -----------------------------------------------------------------------------------
    #general settings
    sed_value=2.0 # fixed sediment value [if sed_flag =0 then no use] 
    # number of b-splines in each layer use if mod_type_flag = 2 (leave as an int)
    # number of sediment parameter - 4th col in mod.STA (required as assuming a top linear layer here)
    sednpara=2 
    crustnpara=6; 
    mantlenpara=4 
    ####################################################################################################################################
    # the station name and station coordinantes
    stafile=datadir+'/station_cor.lst'
    # ------------- velocity model filename setting here -------------------------------
    # if vel_mod_uniform_flag = 1, use the 1D model define here!
    if (vel_mod_uniform_flag==1):
        modfile=datadir+'/Vel_mod/NTW1d_H14_1km_final'
        # modfile='../Vel_mod/NTW1d_H14_1km'
    elif (vel_mod_uniform_flag==0): # set link to directory only
        modfile_dir=datadir+'/Vel_mod/vel_mods_step/'
    else:
        print("Wrong vel_mod_uniform_flag, stop!")
        sys.exit()

    ####################################################################################################################################
    # in.para_STA settings #see inputs for in.para_STA in for loop below to edit as needed
    crustpercpert=.4 #40% search of total crust thickness (will find and search according to starting model)
    # 1st column options (fix thickness = 0 | fix value = 1)
    perturbVal='0';
    perturbThick='1';
    
    # 2nd col options (percent = -1 vs absolute = 1)
    perturbPerc='-1';
    perturbAbs='1' 

    # 3rd col; perturbation range
    pertRangeSed='2.0'; 
    pertRangeCrust='25'; 
    pertRangeMantle='20'; 
    pertRangSedThick='100' 
    # 4th col #gaussian step width
    gwVsSedstep='0.05'; 
    gwVsCruststep='0.05'; 
    gwVsMantlestep='0.05'; 
    gwstepsedthick='0.1'; 
    gwstepcrustthick='1.0'
    # -------------------------- Now setting the values base on the number of value each layer ---------------------------------
    if (sed_flag==0):
        Sedlayerid='2'; Crustlayerid='0'; Mantlelayerid='1' #5th column
    elif (sed_flag==1):
        Sedlayerid='0'; Crustlayerid='1'; Mantlelayerid='2' #5th column
    else:
        print("Wrong sed_flag value stop!")
        sys.exit()
    # ---------> in.para_{STA} --->>>>>
    # if (mod_type_flag==2): # Case = 1 and 4 is calculate later
    if (sed_flag==1):
        # 
        for x in range(0,sednpara):
            locals()['inparasedVs%s' % x] = perturbVal+' '+perturbAbs+' '+pertRangeSed+' '+gwVsSedstep+' '+Sedlayerid+' %d'%(x)
            print("> ",locals()['inparasedVs%s' % x])
            time.sleep(3)
    # else: only consider crust and mantle
    for x in range(0,crustnpara):
        locals()['inparacrustVs%s' % x] = perturbVal+' '+perturbPerc+' '+pertRangeCrust+' '+gwVsCruststep+' '+Crustlayerid+' %d'%(x)
    for x in range(0,mantlenpara):
        locals()['inparamantleVs%s' % x] = perturbVal+' '+perturbPerc+' '+pertRangeMantle+' '+gwVsMantlestep+' '+Mantlelayerid+' %d'%(x)
        
    locals()['inparasedthick'] = perturbThick+' '+perturbPerc+' '+pertRangSedThick+' '+gwstepsedthick+' '+Sedlayerid
    
    # note: still have "inparacrustthick" bellow
    ####################################################################################################################################
    # mod.STA settings ...
    # model type: linear (gradient) = 4 ; Bspline = 2; layered = 1 ; water = 5; 
    modlinear='4' ; modsplines='2' ; modlay="1" #2nd col
    sedrhoflag='1'; sedQflag='2'; sedPflag='1'; sedmodvpvs='0'; sedVpVs1='0'; #cols -11, -10, -9 for sediment
    crustrhoflag='1'; crustQflag='3'; crustPflag='3'; crmodvpvs='1.70'; crVpVs1='1.73'; #cols -11, -10, -9 for crustal
    mantlerhoflag='2'; mantleQflag='4'; mantlePflag='3'; mtmodvpvs='1.75'; mtVpVs1='1.78'; #cols -11, -10, -9 for mantle
#    
    #leave these as 0's for now; untested as to if they carry through codes correctly or not
    sedDrho='0'; sedDrho1='0'; seddVs='0'; seddvs1='0'; sedfdvs='0'; sedfdvs1='0'
    crDrho='0'; crDrho1='0'; crdVs='0'; crdvs1='0'; crfdvs='0'; crfdvs1='0'
    mtDrho='0'; mtDrho1='0'; mtdVs='0'; mtdVs1='0'; mtfdvs='0'; mtfdvs1='0'
    ####################################################################################################################################
    # setting period to read the observed data 
    allper=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    phper=[8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    eqper=[ 0 ] # eqper = dispersion from earthquake data, no type of data exist!
    ####################################################################################################################################
    ########################### SHOULD NOT NEED TO EDIT BEYOND THIS POINT ##############################################################
    ####################################################################################################################################
    #%% Get a starting model! Using Shapiro & Ritzwoller diffraction model
    modxmax=127.0
    modymax=26.0
    modxmin=121.0
    modymin=21
    modstepx=0.025
    modstepy=0.025
    # 
    modixr=int((modxmax-modxmin)/modstepx+1.0)
    modiyr=int((modymax-modymin)/modstepy+1.0)
    modixs=modxmin
    modiys=modymin
    #final phasevel result info
    allgridmodvs=np.empty([100,modiyr,modixr]) #fast, filled with random tho, [lat,lon]
    allgridmodvs[:]=np.nan

    allgridmoddepth=[]
    for i in np.arange(100):
    #    allgridmoddepth.append(i*4)
        allgridmoddepth.append(i*10)
    allgridmoddepth=np.array(allgridmoddepth)
    #print(allgridmoddepth)
    print('\n Reading in model, will take a minute or two.. \n\n')
    os.chdir(scriptdir)
    # make the directory contained all check files
    if not os.path.isdir(datadir+'/query_dir'):
        os.mkdir(datadir+'/query_dir')
    # create the file for all excutable stations:
    if os.path.isfile(datadir+'/runstations.lst'):
        os.remove(datadir+'/runstations.lst')
    ####################################################################################################################################
    # ------------------------------------- loop over the staiton list ------------------------------------------------------
    # read the station information
    stalonarr,stalatarr=np.loadtxt(stafile,usecols=[1,2],unpack=True)
    staarr=np.genfromtxt(stafile,usecols=[0],unpack=True,dtype='str')
    if np.size([staarr]) < 2: staarr = [str(staarr)];
    # loop
    for zz,sta in enumerate (staarr):
        print(zz,sta)
        if len([staarr]) < 2:
            stalon = stalonarr
            stalat = stalatarr
        else:
            stalon=stalonarr[zz]
            stalat=stalatarr[zz]
        if (vel_mod_uniform_flag == 1):
            modfile = modfile;
            print("using same 1D input model for all station: ",(sta))
        else:
            modfile = modfile_dir+sta+'.mod'
            print("using different 1D input model for station: ",(modfile))
        # read depth values from the model file
        allgridmoddepth = np.genfromtxt(modfile,usecols=[0],unpack=True,dtype='float')#/data/cnliu/ >> ../Vel_mod/
        tmp=allgridmoddepth
        # read the vs values
        modvs = np.genfromtxt(modfile,usecols=[1],unpack=True,dtype='float')
        if sed_flag==0:
            sed_value=modvs[0];
        else:
            sed_value=sed_value;
        # create the output directory
        outdir=datadir+'/'+sta+'_data'
        # print("check the data dir: ",outdir)
        if not os.path.isdir(outdir):
            print(outdir," is not exist! create the directory")
            os.mkdir(outdir)
        else:
            print(outdir," is exist! write over the files")
        modstavs = modvs
        modstavs[0] = sed_value
        # write to check we read the right value or not
        print("Model check file: ",datadir+'/query_dir/'+sta+'_depth_vs.txt')
        with open ( datadir+'/query_dir/'+sta+'_depth_vs.txt', 'w') as ff:
            for ii in np.arange(len(allgridmoddepth)):
                printmf='%5.3f %5.5f\n'%(allgridmoddepth[ii],modstavs[ii])
                ff.write(printmf)
        # ---------------------- Now we find the sediment depth and moho depth ---------------------------
        dVs=[]; dVsdepth=[]
        # for kk in np.arange(1,len(allgridmoddepth)-2):
        for kk in np.arange(1,len(allgridmoddepth)-2):
            dVs.append(abs((modstavs[kk+1]-modstavs[kk-1])/(allgridmoddepth[kk+1]-allgridmoddepth[kk-1])))
            dVsdepth.append((allgridmoddepth[kk+1]+allgridmoddepth[kk-1])*0.5)
        prominence,prdep=peakdet2(dVs,dVsdepth)
        dVsmax1index=np.where(prominence[0:4]==min(prominence[0:4]))[0][0]
        dVsmax2index=(np.where(prominence[4:]==min(prominence[4:]))[0][0])+4
        dVsmax1tmp=np.where(dVs[0:4]==max(dVs[0:4]))[0][0]
        # sediment depth and sediment index here
        seddepth,sedindex=find_nearest(allgridmoddepth,dVsdepth[dVsmax1tmp]) 
        if (sed_flag==0):
            seddepth=0
        # moho depth and moho index here
        # mohodepth,mohoindex=find_nearest(allgridmoddepth,prdep[dVsmax2index+1])
        mohodepth,mohoindex=find_nearest(allgridmoddepth,prdep[dVsmax2index])
        print(">>>seddepth, mohodepth,sedindex,mohoindex:",seddepth, mohodepth,sedindex,mohoindex)
        moho_id = mohoindex+1;
        if (sed_flag==1):
            crustthick=mohodepth-seddepth;
            sedthick=seddepth;
        else:
            sedthick=0;
            crustthick=mohodepth;
        mantlethick=max(allgridmoddepth)-mohodepth
        pertRangCrustThick=crustpercpert*crustthick
        if np.isnan(pertRangCrustThick):
            print('error!!!! percpert')
            break
        # last line for in.para_STA for crustal pertubation
        locals()['inparacrustthick']=perturbThick+' '+perturbAbs+' %3.1f '%(pertRangCrustThick)+gwstepcrustthick+' '+Crustlayerid
        # ---------------------- base on the model type to make a model ---------------------------
        # > ------------------------- Casse Bspline -----------------------------------------------
        if (mod_type_flag==2):
            if (sed_flag==1):
                vsBspinversion=modstavs[1:moho_id]; depthBspinversion=allgridmoddepth[1:moho_id]
        #       mvsBspinversion=modstavs[mohoindex:]; mdepthBspinversion=allgridmoddepth[mohoindex:]
                mvsBspinversion=modstavs[moho_id:]; mdepthBspinversion=allgridmoddepth[moho_id:]
            else:
                vsBspinversion=modstavs[0:moho_id]; depthBspinversion=allgridmoddepth[0:moho_id]
                mvsBspinversion=modstavs[moho_id:]; mdepthBspinversion=allgridmoddepth[moho_id:]
            # run Bspline for crustal
            cmd=scriptdir+'/lfBsp_nlay 0 '+str(len(vsBspinversion))+' 2.0 '+str(crustnpara)+' 4 '+str(len(vsBspinversion))
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            process.wait()
            bspfile='B_spline_'
            FCBs=[]
            # maxtrix forming for crustal
            for kk in np.arange(crustnpara):
                FCBs.append(np.loadtxt(scriptdir+'/'+bspfile+str(kk)+'.txt',usecols=(1),unpack=True))
                cmd='mv '+bspfile+str(kk)+'.txt '+bspfile+str(kk)+'_crust.txt'
                pr = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                pr.wait()
            # run Bspline for mantle
            cmd=scriptdir+'/lfBsp_nlay 0 '+str(len(mvsBspinversion))+' 0.75 '+str(mantlenpara)+' 4 '+str(len(mvsBspinversion))
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            process.wait()
            bspfile='B_spline_'
            mFCBs=[]
            # maxtrix forming for mantal   
            for kk in np.arange(mantlenpara):
                mFCBs.append(np.loadtxt(scriptdir+'/'+bspfile+str(kk)+'.txt',usecols=(1),unpack=True))
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
            if (sed_flag==1):
                sedvalue=[]; seddepth=[]
                sedvalue.append(modstavs[0]); seddepth.append(allgridmoddepth[0])
                sedvalue.append(modstavs[1]); seddepth.append(allgridmoddepth[1])
            
                if sedvalue[0]==0:
                    sedvalue[0]=0.5
                if sedvalue[1]==0:
                    sedvalue[1]=0.5
            # Now we can plot the figure 
            print("Plot the input model of station %s in Bspline type"%(sta))
            # read in Vph, H/V, and RF info from file
            allphper,mergedph,mergedphun=np.loadtxt(outdir+'/'+sta+'.ph',usecols=[0,1,2],unpack=True)
            hvper,hvall,hvun=np.loadtxt(outdir+'/'+sta+'.HV',usecols=[0,1,2],unpack=True)
            rft,rfamp,rfunc=np.loadtxt(outdir+'/'+sta+'.RF',usecols=[0,1,2],unpack=True)
            ##%% Plot all
        #    '''
            fig=plt.figure(zz,figsize=(10,10))
            
            ax1=plt.subplot2grid((3,2), (0,0), rowspan=3) #Velocity models
            ax4=plt.subplot2grid((3,2), (0,1)) #Phase
            ax2=plt.subplot2grid((3,2), (1,1)) #H/V
            ax3=plt.subplot2grid((3,2), (2,1)) #RF
        # 
            ax1.set_facecolor('lightyellow')
            ax4.set_facecolor('lightyellow')
            ax2.set_facecolor('lightyellow')
            ax3.set_facecolor('lightyellow')
            
            # Vs
            if (sed_flag==1):
                ax1.plot(sedvalue,seddepth,'yo',label='Sed Picks')
            ax1.plot(FCvsfromcrustcoeff,depthBspinversion,'ko',label='Est from '+str(crustnpara)+' Bsplines Vs',alpha=0.4)
            if (man_flag==1):
                ax1.plot(mFCvsfromcrustcoeff,mdepthBspinversion,'gs',label='Est from '+str(mantlenpara)+' Bsplines Vs',alpha=0.4)
            ax1.plot(modstavs,allgridmoddepth,'r-',label='Raw Vs Model')
            
            ax1.set_ylim([0,150])
            ax1.legend(loc='lower left',fontsize=12)
            
            ax1.set_ylim(ax1.get_ylim()[::-1])
            ax1.set_xlim([0.0,5.7])
            ax1.set_ylabel('Depth (km)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
            ax1.set_xlabel('Vs (km/s)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
            ax1.tick_params(labeltop=True)
            ax1.grid(color='k', axis='both',linestyle='-', linewidth=2,alpha=0.1,zorder=2)
            # PHASE
            ax2.errorbar(allphper,mergedph,yerr=mergedphun,fmt='.:',color='r',ecolor='r',elinewidth=1.5,capthick=1.5,label='Vph',zorder=5,alpha=0.5)
            # ax2.errorbar(phper,mergedph[0:len(phper)],yerr=mergedphun[0:len(phper)],fmt='.:',color='k',ecolor='k',elinewidth=1.5,capthick=1.5,label='ANT Vph',zorder=5,alpha=0.5)
            #ax2.errorbar(allphper,mergedph,yerr=mergedphun,fmt='.:',color='k',ecolor='r',elinewidth=1.5,capthick=1.5,label='All Vph',zorder=5,alpha=0.5)
            ax2.set_xlabel('Period (s)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
            ax2.set_ylabel('Vph (km/s)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
            #ax2.set_xlim([0.0,105.0])
            # ax2.set_ylim([1.5,5.0])
            # ax2.set_xlim([0,30])
            ax2.legend(loc='lower right',fontsize=12)
            ax2.tick_params(labeltop=False)
            ax2.grid(color='k', axis='both',linestyle='-', linewidth=2,alpha=0.1,zorder=2)

            # H/VP
            #AN
            # ax3.errorbar(phper,hvall[0:len(phper)],yerr=8.0*hvun[0:len(phper)],fmt='.:',color='k',ecolor='k',elinewidth=1.5,capthick=1.5,label='ANT H/V',zorder=5,alpha=0.5)
            #EQ
            ax3.errorbar(hvper,hvall,yerr=hvun,fmt='.:',color='r',ecolor='r',elinewidth=1.5,capthick=1.5,label='H/V',zorder=5,alpha=0.5)
            #ax3.errorbar(hvper,hvall,yerr=hvun,fmt='.:',color='r',ecolor='k',elinewidth=1.5,capthick=1.5,label='All H/V',zorder=5,alpha=0.5)
            ax3.set_xlim([0,30])
            ax3.set_xlabel('Period (s)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
            ax3.set_ylabel('H/V',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
            ax3.set_ylim([0,2.0])
            ax3.legend(loc='upper right',fontsize=12)
            ax3.tick_params(labeltop=False)
            ax3.grid(color='k', axis='both',linestyle='-', linewidth=2,alpha=0.1,zorder=2)
    
            #RF
            ax4.errorbar(rft,rfamp,yerr=rfunc,fmt='.:',color='k',ecolor='k',elinewidth=1.5,capthick=0.0,label='RFunc',zorder=5,alpha=0.5)
            ax4.plot(rft,rfamp,'r-',zorder=7,alpha=0.96,label='RF')
            ax4.set_xlabel('Time (s)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
            ax4.set_ylabel('RF Amp',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
            ax4.set_xlim([min(rft)-0.2,max(rft)+0.5])
            ax4.set_ylim([-0.25,0.6])
            ax4.legend(loc='upper right',fontsize=12)
            ax4.tick_params(labeltop=False)
            ax4.grid(color='k', axis='both',linestyle='-', linewidth=2,alpha=0.1,zorder=2)
    
            plt.suptitle("Starting model and input data for station %s"%(sta),fontdict = {'family':'serif','color':'darkred','size':25,'weight':'bold'})
    
            #plt.show()
            fig.savefig(datadir+'/query_dir/StartingModel_'+sta+'.png',bbox_inches='tight',transparent=False,pad_inches=0.1)
            plt.close()
            
    
            fig, axs = plt.subplots(1,2, figsize=(8,11), sharey=True)
            axs[0].set_facecolor('lightyellow')
            axs[1].set_facecolor('lightyellow')
            # plt.figure(figsize=(3,7))
            # axs[0].set_title('NVN_1D', fontsize=24)
            axs[1].set_title('B-spline (%s, %s)' %(crustnpara, mantlenpara), fontsize=24)
            # print("tmp_here:",tmp)
            axs[0].plot(modvs, tmp*-1, 'ko-',label="Input model")
            if (sed_flag==1):
                # sed value
                axs[0].plot(sedvalue, tmp[0:2]*-1, 'c.-',label="Sed Val")
                # value from initial
                axs[0].plot(vsBspinversion, tmp[1:moho_id]*-1, 'ro-',ms=10, alpha=0.5,label="Crustal Val from input")
                if (man_flag==1):
                    axs[0].plot(mvsBspinversion, tmp[moho_id:]*-1, 'rs-',ms=10, alpha=0.5,label="Mantle Val from input")
                # value from Bspline coefficient
                axs[0].plot(FCvsfromcrustcoeff, tmp[1:moho_id]*-1, 'bo--', alpha=0.5,label="Val from crustal coeff")
                if (man_flag==1):
                    axs[0].plot(mFCvsfromcrustcoeff, tmp[moho_id:]*-1, 'bs--', alpha=0.5,label="Val from mantle coeff")
                # value from initial
                axs[1].plot(vsBspinversion, tmp[1:moho_id]*-1, 'bo--',ms=10,label="Crustal Val from input")
                if (man_flag==1):
                    axs[1].plot(mvsBspinversion, tmp[moho_id:]*-1, 'bs-',ms=10,label="Mantle Val from input")
                # value from Bspline coefficient
                axs[1].plot(FCvsfromcrustcoeff, tmp[1:moho_id]*-1, 'ro-',label="Val from crustal coeff")
                if (man_flag==1):
                    axs[1].plot(mFCvsfromcrustcoeff, tmp[moho_id:]*-1, 'rs--',label="Val from mantle coeff")
    
            else: # no sediment
                # value from initial
                axs[0].plot(vsBspinversion, tmp[0:moho_id]*-1, 'ro-',ms=10, alpha=0.5,label="Crustal Val from input")
                if (man_flag==1):
                    axs[0].plot(mvsBspinversion, tmp[moho_id:]*-1, 'rs-',ms=10, alpha=0.5,label="Mantle Val from input")
                # value from Bspline coefficient
                axs[0].plot(FCvsfromcrustcoeff, tmp[0:moho_id]*-1, 'bo--', alpha=0.5,label="Val from crustal coeff")
                if (man_flag==1):
                    axs[0].plot(mFCvsfromcrustcoeff, tmp[moho_id:]*-1, 'bs--', alpha=0.5,label="Val from mantle coeff")
                # value from initial
                axs[1].plot(vsBspinversion, tmp[0:moho_id]*-1, 'bo-',ms=10,label="Crustal Val from input")
                if (man_flag==1):
                    axs[1].plot(mvsBspinversion, tmp[moho_id:]*-1, 'bs--',ms=10,label="Mantle Val from input")
                # value from Bspline coefficient
                axs[1].plot(FCvsfromcrustcoeff, tmp[0:moho_id]*-1, 'ro-',label="Val from crustal coeff")
                if (man_flag==1):
                    axs[1].plot(mFCvsfromcrustcoeff, tmp[moho_id:]*-1, 'rs--',label="Val from mantle coeff")
    
            axs[0].tick_params(labeltop=False)
            axs[0].grid(color='k', axis='both',linestyle='-', linewidth=2,alpha=0.1,zorder=2)
            axs[0].legend(loc='lower left',fontsize=12)
            axs[1].tick_params(labeltop=False)
            axs[1].grid(color='k', axis='both',linestyle='-', linewidth=2,alpha=0.1,zorder=2)
            axs[1].legend(loc='lower left',fontsize=12)
            plt.suptitle("Model interpolation for station: %s"%(sta),fontdict = {'family':'serif','color':'darkred','size':25,'weight':'bold'})
    
            fig.savefig(datadir+'/query_dir/B-spline_'+str(crustnpara)+'-'+str(mantlenpara)+'_comparisons_'+sta+'.png',bbox_inches='tight',transparent=False,pad_inches=0.1)
            plt.close()
            print("Plot finish now write out the model and in.para file:")
            # -------------------- more calculation -------------------------------
            # ----------------- >>>>> More calculation >>>> print to mod.{STA} ---------------------------------
            if (sed_flag==1):
                sthick=seddepth[len(seddepth)-1]-seddepth[0]
                locals()['modsedlayer']=Sedlayerid+' '+modlinear+' %4.1f'%(sthick)+' %d'%(sednpara)+\
                ' %5.6f %5.6f '%(sedvalue[0],sedvalue[1])+sedrhoflag+' '+sedQflag+' '+sedPflag+' '+\
                sedmodvpvs+' '+sedVpVs1+' '+sedDrho+' '+sedDrho1+' '+seddVs+' '+seddvs1+' '+sedfdvs+' '+sedfdvs1
            # crust
            printcrusttmp=''
            for kk in np.arange(crustnpara):
                printcrusttmp=printcrusttmp+' %5.6f'%(FCcrustcoeff[kk])
            # 
            locals()['modcrustlayer']=Crustlayerid+' '+modsplines+' %4.1f %d'%(crustthick,crustnpara)+printcrusttmp+\
            ' '+crustrhoflag+' '+crustQflag+' '+crustPflag+' '+crmodvpvs+' '+crVpVs1+' '+crDrho+' '+crDrho1+' '+\
            crdVs+' '+crdvs1+' '+crfdvs+' '+crfdvs1
            # Mantle
            printmantletmp=''
            for kk in np.arange(mantlenpara):
                printmantletmp=printmantletmp+' %5.6f'%(mFCcrustcoeff[kk])
            locals()['modmantlelayer']=Mantlelayerid+' '+modsplines+' %4.1f %d '%(mantlethick,mantlenpara)+printmantletmp+\
            ' '+mantlerhoflag+' '+mantleQflag+' '+mantlePflag+' '+mtmodvpvs+' '+mtVpVs1+' '+mtDrho+' '+mtDrho1+' '+mtdVs+\
            ' '+mtdVs1+' '+mtfdvs+' '+mtfdvs1
            # prepare finish now write
            # ------------------------------- >>>>>>>> write mod.{STA} here !!! <<<<<<< ----------------------------------
            print("Write the model file in bspline style: ",outdir+'/mod.'+sta)
            with open ( outdir+'/mod.'+sta, 'w') as ff:
                if (sed_flag==1):
                    ff.write(locals()['modsedlayer']+'\n'); 
                ff.write(locals()['modcrustlayer']+'\n'); 
                if (man_flag==1):
                    ff.write(locals()['modmantlelayer']+'\n')
            ff.close()
            # ------------------------------- >>>>>>>> write in.para_{STA} here !!! <<<<<<< ----------------------------------
            # locals()['inparacrustthick']=perturbVal+' '+perturbPerc+' %3.1f '%(pertRangCrustThick)+gwstepcrustthick+' '+Crustlayerid
            locals()['inparacrustthick']=perturbThick+' '+perturbAbs+' %3.1f '%(pertRangCrustThick)+gwstepcrustthick+' '+Crustlayerid
            fnpara=datadir+'/MonteCarlo/in.para_'+sta
            print("Write the in.para: ",fnpara)
            with open(fnpara,'w') as ff2:
                if (sed_flag==1):
                    for x in range(0,sednpara):
                        ff2.write(locals()['inparasedVs%s' % x]+'\n')
                for x in range(0,crustnpara):
                    ff2.write(locals()['inparacrustVs%s' % x]+'\n')
                if (man_flag==1):
                    for x in range(0,mantlenpara):
                        ff2.write(locals()['inparamantleVs%s' % x]+'\n')
                # crustal pertubation part
                if (sed_flag==1):
                    ff2.write(locals()['inparasedthick']+'\n')
                if (man_flag==1):
                    ff2.write(locals()['inparacrustthick']+'\n')
            ff2.close()        
        # > ------------------------- End of casse Bspline -----------------------------------------------
        elif (mod_type_flag==1) | (mod_type_flag==4):
            # we calculate the sednpara, crustnpara, mantlenpara based on sedindex and mohoindex above:     
            if (sed_flag==1): # If have sediment setting
                sednpara = sedindex;
                if sednpara==0: sednpara=1;
                crustnpara = (mohoindex - sedindex);
                mantlenpara = (len(allgridmoddepth)-mohoindex)-2;
            # 
                for x in range(0,sednpara+1):
                    locals()['inparasedVs%s' % x] = perturbVal+' '+perturbAbs+' '+pertRangeSed+' '+gwVsSedstep+' '+Sedlayerid+' %d'%(x)
                for x in range(0,crustnpara):
                    locals()['inparacrustVs%s' % x] = perturbVal+' '+perturbPerc+' '+pertRangeCrust+' '+gwVsCruststep+' '+Crustlayerid+' %d'%(x)
                for x in range(0,mantlenpara):
                    locals()['inparamantleVs%s' % x] = perturbVal+' '+perturbPerc+' '+pertRangeMantle+' '+gwVsMantlestep+' '+Mantlelayerid+' %d'%(x)
                # ------------------------ calculate through the model layers -----------------------------------------------
                zero_d = allgridmoddepth[0];
                # 
                tmpsv0=''
                tmpsd0=''
                # 
                tmpcv0=''
                tmpcd0=''
                # 
                tmpmv0=''
                tmpmd0=''
                # forming layered model or linear model
                mod_vs = []
                mod_d = []
                for i,depth in enumerate(allgridmoddepth):   
                    # forming for plot and mod.STA
                    if (i==0):
                        mod_d.append(depth)
                        mod_vs.append(sed_value)
                        # 
                        # mod_d.append(depth)
                        # mod_vs.append(modvs[i])
                    else:
                        if (mod_type_flag==1): # layered
                            # 2 time depth
                            mod_d.append(depth)
                            mod_d.append(depth)
                            # 
                            mod_vs.append(modvs[i-1])
                            mod_vs.append(modvs[i])
                        else:
                            mod_d.append(depth)
                            mod_vs.append(modvs[i])
                     # the layered style of mod.STA using layer thickness. Thus start from layer 2
                        thick_now = depth - allgridmoddepth[i-1]
                        if (i==1):
                            tmpsv0=tmpsv0+' %5.6f'%(sed_value)
                            tmpsd0=tmpsd0+' %5.6f'%((thick_now)/sedthick)
                            # tmpsd0=tmpsd0+' %5.6f'%((thick_now/2)/sedthick)
                            # tmpsv0=tmpsv0+' %5.6f'%((modvs[i-1]+modvs[i])/2)
                            # tmpsd0=tmpsd0+' %5.6f'%((thick_now/2)/sedthick)
                        elif (i <= sednpara) and (i > 0):
                            tmpsv0=tmpsv0+' %5.6f'%((modvs[i-1]+modvs[i])/2)
                            tmpsd0=tmpsd0+' %5.6f'%(thick_now/sedthick)
                        elif (i <= crustnpara+sednpara) & (i > sednpara):
                            tmpcv0=tmpcv0+' %5.6f'%((modvs[i-1]+modvs[i])/2)
                            tmpcd0=tmpcd0+' %5.6f'%(thick_now/crustthick)
                        elif (i <= mantlenpara+crustnpara+2) & (i > crustnpara+sednpara+1):
                            tmpmv0=tmpmv0+' %5.6f'%((modvs[i-1]+modvs[i])/2)
                            tmpmd0=tmpmd0+' %5.6f'%(thick_now/mantlethick)
                # Set the model print line
                if (mod_type_flag==1): # layered
                    locals()['modsedlayer']=Sedlayerid+' '+modlay+' %4.1f'%(sedthick)+' %d'%(sednpara)+\
                    tmpsv0+' '+tmpsd0+' '+sedrhoflag+' '+sedQflag+' '+sedPflag+' '+\
                    sedmodvpvs+' '+sedVpVs1+' '+sedDrho+' '+sedDrho1+' '+seddVs+' '+seddvs1+' '+sedfdvs+' '+sedfdvs1+"\n"
                    # 
                    locals()['modcrustlayer']=Crustlayerid+' '+modlay+' %4.1f %d'%(crustthick,crustnpara)+tmpcv0+' '+tmpcd0+\
                    ' '+crustrhoflag+' '+crustQflag+' '+crustPflag+' '+crmodvpvs+' '+crVpVs1+' '+crDrho+' '+crDrho1+' '+\
                    crdVs+' '+crdvs1+' '+crfdvs+' '+crfdvs1+'\n'
                    # 
                    locals()['modmantlelayer']=Mantlelayerid+' '+modlay+' %4.1f %d '%(mantlethick,mantlenpara)+tmpmv0+' '+tmpmd0+\
                    ' '+mantlerhoflag+' '+mantleQflag+' '+mantlePflag+' '+mtmodvpvs+' '+mtVpVs1+' '+mtDrho+' '+mtDrho1+' '+mtdVs+\
                    ' '+mtdVs1+' '+mtfdvs+' '+mtfdvs1+'\n'
                elif (mod_type_flag==4):
                    # Set the model print line
                    locals()['modsedlayer']=Sedlayerid+' '+modlinear+' %4.1f'%(sedthick)+' %d'%(sednpara)+\
                    tmpsv0+' '+tmpsd0+' '+sedrhoflag+' '+sedQflag+' '+sedPflag+' '+\
                    sedmodvpvs+' '+sedVpVs1+' '+sedDrho+' '+sedDrho1+' '+seddVs+' '+seddvs1+' '+sedfdvs+' '+sedfdvs1+"\n"
                    # 
                    locals()['modcrustlayer']=Crustlayerid+' '+modlinear+' %4.1f %d'%(crustthick,crustnpara)+tmpcv0+' '+tmpcd0+\
                    ' '+crustrhoflag+' '+crustQflag+' '+crustPflag+' '+crmodvpvs+' '+crVpVs1+' '+crDrho+' '+crDrho1+' '+\
                    crdVs+' '+crdvs1+' '+crfdvs+' '+crfdvs1+'\n'
                    # 
                    locals()['modmantlelayer']=Mantlelayerid+' '+modlinear+' %4.1f %d '%(mantlethick,mantlenpara)+tmpmv0+' '+tmpmd0+\
                    ' '+mantlerhoflag+' '+mantleQflag+' '+mantlePflag+' '+mtmodvpvs+' '+mtVpVs1+' '+mtDrho+' '+mtDrho1+' '+mtdVs+\
                    ' '+mtdVs1+' '+mtfdvs+' '+mtfdvs1+'\n'
                    #
            else: # No sediment setting
                sednpara=0
                crustnpara = mohoindex#+1;
                mantlenpara = (len(allgridmoddepth)-mohoindex)-2;
                for x in range(0,crustnpara):
                    locals()['inparacrustVs%s' % x] = perturbVal+' '+perturbPerc+' '+pertRangeCrust+' '+gwVsCruststep+' '+Crustlayerid+' %d'%(x)
                for x in range(0,mantlenpara):
                    locals()['inparamantleVs%s' % x] = perturbVal+' '+perturbPerc+' '+pertRangeMantle+' '+gwVsMantlestep+' '+Mantlelayerid+' %d'%(x)
                # ------------------------ calculate through the model layers -----------------------------------------------
                zero_d = allgridmoddepth[0];
                # 
                tmpsv0=''
                tmpsd0=''
                # 
                tmpcv0=''
                tmpcd0=''
                # 
                tmpmv0=''
                tmpmd0=''
                # forming layered model or linear model
                mod_vs = []
                mod_d = []
                for i,depth in enumerate(allgridmoddepth):   
                    # forming for plot and mod.STA
                    if (i==0):
                        mod_d.append(depth)
                        mod_vs.append(modvs[i])
                    else:
                        if (mod_type_flag==1): # layered
                            # 2 time depth
                            mod_d.append(depth)
                            mod_d.append(depth)
                            # 
                            mod_vs.append(modvs[i-1])
                            mod_vs.append(modvs[i])
                        else:
                            mod_d.append(depth)
                            mod_vs.append(modvs[i])
                     # the layered style of mod.STA using layer thickness. Thus start from layer 2
                        thick_now = depth - allgridmoddepth[i-1]
                        if (i <= sednpara):
                            tmpsv0=tmpsv0+' %5.6f'%((modvs[i-1]+modvs[i])/2)
                            tmpsd0=tmpsd0+' %5.6f'%(thick_now/sedthick)
                        elif (i <= crustnpara+sednpara) & (i > sednpara):
                            tmpcv0=tmpcv0+' %5.6f'%((modvs[i-1]+modvs[i])/2)
                            tmpcd0=tmpcd0+' %5.6f'%(thick_now/crustthick)
                        # elif (i <= mantlenpara+crustnpara+2) & (i > crustnpara+sednpara+1):
                        elif (i > crustnpara+sednpara+1):
                            # print(i)
                            tmpmv0=tmpmv0+' %5.6f'%((modvs[i-1]+modvs[i])/2)
                            tmpmd0=tmpmd0+' %5.6f'%(thick_now/crustthick)
                # Set the model print line
                # Set the model print line
                if (mod_type_flag==1): # layered
                    # 
                    locals()['modcrustlayer']=Crustlayerid+' '+modlay+' %4.1f %d'%(crustthick,crustnpara)+tmpcv0+' '+tmpcd0+\
                    ' '+crustrhoflag+' '+crustQflag+' '+crustPflag+' '+crmodvpvs+' '+crVpVs1+' '+crDrho+' '+crDrho1+' '+\
                    crdVs+' '+crdvs1+' '+crfdvs+' '+crfdvs1+'\n'
                    # 
                    locals()['modmantlelayer']=Mantlelayerid+' '+modlay+' %4.1f %d '%(mantlethick,mantlenpara)+tmpmv0+' '+tmpmd0+\
                    ' '+mantlerhoflag+' '+mantleQflag+' '+mantlePflag+' '+mtmodvpvs+' '+mtVpVs1+' '+mtDrho+' '+mtDrho1+' '+mtdVs+\
                    ' '+mtdVs1+' '+mtfdvs+' '+mtfdvs1+'\n'
                elif (mod_type_flag==4):
                    # 
                    locals()['modcrustlayer']=Crustlayerid+' '+modlinear+' %4.1f %d'%(crustthick,crustnpara)+tmpcv0+' '+tmpcd0+\
                    ' '+crustrhoflag+' '+crustQflag+' '+crustPflag+' '+crmodvpvs+' '+crVpVs1+' '+crDrho+' '+crDrho1+' '+\
                    crdVs+' '+crdvs1+' '+crfdvs+' '+crfdvs1+'\n'
                    # 
                    locals()['modmantlelayer']=Mantlelayerid+' '+modlinear+' %4.1f %d '%(mantlethick,mantlenpara)+tmpmv0+' '+tmpmd0+\
                    ' '+mantlerhoflag+' '+mantleQflag+' '+mantlePflag+' '+mtmodvpvs+' '+mtVpVs1+' '+mtDrho+' '+mtDrho1+' '+mtdVs+\
                    ' '+mtdVs1+' '+mtfdvs+' '+mtfdvs1+'\n'
                #

            # --------------------------------------------------------------------------------------------------------------------------
            # Now we can plot the figure
            if (mod_type_flag==1):
                print("Plot the input model of station %s in Layer-cake style"%(sta))
            else:
                print("Plot the input model of station %s in gradient style"%(sta))
            # read in Vph, H/V, and RF info from file
            allphper,mergedph,mergedphun=np.loadtxt(outdir+'/'+sta+'.ph',usecols=[0,1,2],unpack=True)
            hvper,hvall,hvun=np.loadtxt(outdir+'/'+sta+'.HV',usecols=[0,1,2],unpack=True)
            rft,rfamp,rfunc=np.loadtxt(outdir+'/'+sta+'.RF',usecols=[0,1,2],unpack=True)
            ##%% Plot all
        #    '''
            fig=plt.figure(zz,figsize=(10,10))
            
            ax1=plt.subplot2grid((3,2), (0,0), rowspan=3) #Velocity models
            ax4=plt.subplot2grid((3,2), (0,1)) #Phase
            ax2=plt.subplot2grid((3,2), (1,1)) #H/V
            ax3=plt.subplot2grid((3,2), (2,1)) #RF
        # 
            ax1.set_facecolor('lightyellow')
            ax4.set_facecolor('lightyellow')
            ax2.set_facecolor('lightyellow')
            ax3.set_facecolor('lightyellow')
            
            # Vs
            ax1.plot(modstavs,allgridmoddepth,'k-',label='Raw Vs Model')
            if (sed_flag==1)&(mod_type_flag==1):
                ax1.plot(mod_vs[0:sedindex*2+1],mod_d[0:sedindex*2+1],'y*',ms=10,label='Sed Picks')
                ax1.plot(mod_vs[sedindex*2:mohoindex*2],mod_d[sedindex*2:mohoindex*2],'ko',label='crustal Picks',alpha=0.4)
                if (man_flag==1):
                    ax1.plot(mod_vs[mohoindex*2+2:-1],mod_d[mohoindex*2+2:-1],'gs',label='mantle Picks',alpha=0.4)
                ax1.plot(mod_vs,mod_d,'r--',label='Est Vs Model')
            elif (sed_flag==0)&(mod_type_flag==1):
                ax1.plot(mod_vs[0:mohoindex*2],mod_d[0:mohoindex*2],'ko',label='crustal Picks',alpha=0.4)
                if (man_flag==1):
                    ax1.plot(mod_vs[mohoindex*2+2:-1],mod_d[mohoindex*2+2:-1],'gs',label='mantle Picks',alpha=0.4)
                ax1.plot(mod_vs,mod_d,'r--',label='Est Vs Model')
            elif (sed_flag==1)&(mod_type_flag==4):
                ax1.plot(mod_vs[0:sedindex+1],mod_d[0:sedindex+1],'y*',ms=10,label='Sed Picks')
                ax1.plot(mod_vs[sedindex:mohoindex+1],mod_d[sedindex:mohoindex+1],'ko',label='crustal Picks',alpha=0.4)
                if (man_flag==1):
                    ax1.plot(mod_vs[mohoindex+1:-1],mod_d[mohoindex+1:-1],'gs',label='mantle Picks',alpha=0.4)
                ax1.plot(mod_vs,mod_d,'r--',label='Est Vs Model')
            elif (sed_flag==0)&(mod_type_flag==4):
                ax1.plot(mod_vs[0:mohoindex+1],mod_d[0:mohoindex+1],'ko',label='crustal Picks',alpha=0.4)
                if (man_flag==1):
                    ax1.plot(mod_vs[mohoindex+1:-1],mod_d[mohoindex+1:-1],'gs',label='mantle Picks',alpha=0.4)
                ax1.plot(mod_vs,mod_d,'r--',label='Est Vs Model')
            
            ax1.set_ylim([-2,150])
            ax1.legend(loc='lower left',fontsize=12)
            
            ax1.set_ylim(ax1.get_ylim()[::-1])
            ax1.set_xlim([0.0,5.7])
            ax1.set_ylabel('Depth (km)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
            ax1.set_xlabel('Vs (km/s)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
            ax1.tick_params(labeltop=True)
            ax1.grid(color='k', axis='both',linestyle='-', linewidth=2,alpha=0.1,zorder=2)
            # PHASE
            ax2.errorbar(allphper,mergedph,yerr=mergedphun,fmt='.:',color='r',ecolor='r',elinewidth=1.5,capthick=1.5,label='Vph',zorder=5,alpha=0.5)
            # ax2.errorbar(phper,mergedph[0:len(phper)],yerr=mergedphun[0:len(phper)],fmt='.:',color='k',ecolor='k',elinewidth=1.5,capthick=1.5,label='ANT Vph',zorder=5,alpha=0.5)
            #ax2.errorbar(allphper,mergedph,yerr=mergedphun,fmt='.:',color='k',ecolor='r',elinewidth=1.5,capthick=1.5,label='All Vph',zorder=5,alpha=0.5)
            ax2.set_xlabel('Period (s)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
            ax2.set_ylabel('Vph (km/s)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
            #ax2.set_xlim([0.0,105.0])
            # ax2.set_ylim([1.5,5.0])
            # ax2.set_xlim([0,30])
            ax2.legend(loc='lower right',fontsize=12)
            ax2.tick_params(labeltop=False)
            ax2.grid(color='k', axis='both',linestyle='-', linewidth=2,alpha=0.1,zorder=2)

            # H/V
            #AN
            # ax3.errorbar(phper,hvall[0:len(phper)],yerr=8.0*hvun[0:len(phper)],fmt='.:',color='k',ecolor='k',elinewidth=1.5,capthick=1.5,label='ANT H/V',zorder=5,alpha=0.5)
            #EQ
            ax3.errorbar(hvper,hvall,yerr=hvun,fmt='.:',color='r',ecolor='r',elinewidth=1.5,capthick=1.5,label='H/V',zorder=5,alpha=0.5)
            #ax3.errorbar(hvper,hvall,yerr=hvun,fmt='.:',color='r',ecolor='k',elinewidth=1.5,capthick=1.5,label='All H/V',zorder=5,alpha=0.5)
            ax3.set_xlim([0,30])
            ax3.set_xlabel('Period (s)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
            ax3.set_ylabel('H/V',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
            ax3.set_ylim([0,2.0])
            ax3.legend(loc='upper right',fontsize=12)
            ax3.tick_params(labeltop=False)
            ax3.grid(color='k', axis='both',linestyle='-', linewidth=2,alpha=0.1,zorder=2)
    
            #RF
            ax4.errorbar(rft,rfamp,yerr=rfunc,fmt='.:',color='k',ecolor='k',elinewidth=1.5,capthick=0.0,label='RFunc',zorder=5,alpha=0.5)
            ax4.plot(rft,rfamp,'r-',zorder=7,alpha=0.96,label='RF')
            ax4.set_xlabel('Time (s)',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
            ax4.set_ylabel('RF Amp',fontdict = {'family':'serif','color':'darkblue','size':12,'weight':'bold'})
            ax4.set_xlim([min(rft)-0.2,max(rft)+0.5])
            ax4.set_ylim([-0.25,0.6])
            ax4.legend(loc='upper right',fontsize=12)
            ax4.tick_params(labeltop=False)
            ax4.grid(color='k', axis='both',linestyle='-', linewidth=2,alpha=0.1,zorder=2)
    
            plt.suptitle("Starting model and input data for station %s"%(sta),fontdict = {'family':'serif','color':'darkred','size':55,'weight':'bold'})
    
            # plt.show()
            fig.savefig(datadir+'/query_dir/StartingModel_'+sta+'.png',bbox_inches='tight',transparent=False,pad_inches=0.1)
            plt.close()

            # ------------------------------- >>>>>>>> write mod.{STA} here !!! <<<<<<< ----------------------------------
            print("Write the model file: ",outdir+'/mod.'+sta)
            with open ( outdir+'/mod.'+sta, 'w') as ff:
                if (sed_flag==1):
                    ff.write(locals()['modsedlayer']); 
                ff.write(locals()['modcrustlayer']); 
                if (man_flag==1):
                    ff.write(locals()['modmantlelayer'])
            ff.close()
            # ------------------------------- >>>>>>>> write in.para_{STA} here !!! <<<<<<< ----------------------------------
            fnpara=datadir+'/MonteCarlo/in.para_'+sta
            print("Write the in.para: ",fnpara)
            with open(fnpara,'w') as ff2:
                if (sed_flag==1):
                    for x in range(0,sednpara):
                        ff2.write(locals()['inparasedVs%s' % x]+'\n')
                for x in range(0,crustnpara):
                    ff2.write(locals()['inparacrustVs%s' % x]+'\n')
                if (man_flag==1):
                    for x in range(0,mantlenpara):
                        ff2.write(locals()['inparamantleVs%s' % x]+'\n')
                # crustal pertubation part
                if (sed_flag==1):
                    ff2.write(locals()['inparasedthick']+'\n')
                if (man_flag==1):
                    ff2.write(locals()['inparacrustthick']+'\n')
            ff2.close()
            
            print("Plot finish now write out the model and in.para file:")
        # write station for every case:   
        with open(datadir+'/runstations.lst','a') as ff:
            if len([staarr]) < 2:
                ff.write(sta+' '+str(stalon[0])+' '+str(stalat[0])+'\n')
            else:
                ff.write(sta+' '+str(stalon[zz])+' '+str(stalat[zz])+'\n')
    # > ------------------------- End of casse layercake and gradient -----------------------------------------------
print("done!")
