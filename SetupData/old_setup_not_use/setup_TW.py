#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 
Find closest starting model for this point, and plot the reference model vs. how the model is interpreted via b-splines for this location

Assumes the datadir/runstations.lst exists and contains (space-delimited) columns with station names, longitude, and latitude
    
Also, this does not extrapolate the phase velocity, receiver function, or H/V data for this station, 
but assumes that this has already been pulled for this location and stored 
in datadir/STA_data/STA.ph, datadir/STA_data/STA.RF, and datadir/STA_data/STA.HV respectively
(note that this will be used as-is in the inversion, so make sure the relative weights are where they should be --
for example, I scaled the H/V and phase velocity by 1.5x)

Also, assumes that the .ph and .HV files are in ambient noise measurments, followed by earthquake-derived measurements

Edit the in.para_STA and mod.STA input settings (see JointInversion_Overview.pdf or ipynb for details)
See: Sediment / Crust / Mantle mod.STA settings!! (lines- settings: 66-74; application: 438-470)
Also, see: in.para_STA settings (lines settings: 53-60; application: 60-78, 497-502)

Note that this code is set up for python3

@author: Elizabeth M. Berg
"""


# Imports...

import numpy as np
import os, sys
import matplotlib.pyplot as plt
import subprocess
import time 


#%% # ##################################### ############################################
# ##### SETTINGS - EDIT HERE ###### #
pwd = os.getcwd()
# datadir=pwd+'/../Data/Test_grids_unfold-Bspline/'
# numlist = ['5', '10', '15', '25']
# for num in numlist:
if 1==1:
    datadir=pwd+'/../Data/'#+num+'_20hz'


    scriptdir=pwd

    # stafile=scriptdir+'/../pts-in_important.lst'
    stafile=scriptdir+'/../station_FM.lst'
    stalonarr,stalatarr=np.loadtxt(stafile,usecols=[1,2],unpack=True)
    staarr=np.genfromtxt(stafile,usecols=[0],unpack=True,dtype='str')
    # ############################################
    # in.para_STA settings #see inputs for in.para_STA in for loop below to edit as needed
    crustpercpert=.4 #20% search of total crust thickness (will find and search according to starting model)
    perturbVal='0'; perturbThick='1' #first column options (fix thicness = 0 | fix value = 1)
    perturbPerc='-1'; perturbAbs='1' #2nd col options (percent = -1 vs absolute = 1)
    pertRangeSed='2.0'; pertRangeCrust='40'; pertRangeMantle='30'; pertRangSedThick='100' #3rd col; perturbation range
    gwVsSedstep='0.05'; gwVsCruststep='0.05'; gwVsMantlestep='0.05'; gwstepsedthick='0.1'; gwstepcrustthick='1.0'#4th col #gaussian step width
    # ############################################

    Sedlayerid='0'; Crustlayerid='1'; Mantlelayerid='2' #5th column


    #######################
    #### inputs for in.para_STA -- all laid out so easier to edit/tweak as needed ####
    inparasedVs0=perturbVal+' '+perturbAbs+' '+pertRangeSed+' '+gwVsSedstep+' '+Sedlayerid+' 0\n'
    inparasedVs1=perturbVal+' '+perturbAbs+' '+pertRangeSed+' '+gwVsSedstep+' '+Sedlayerid+' 1\n'

    inparacrustVs0=perturbVal+' '+perturbPerc+' '+pertRangeCrust+' '+gwVsCruststep+' '+Crustlayerid+' 0\n'
    inparacrustVs1=perturbVal+' '+perturbPerc+' '+pertRangeCrust+' '+gwVsCruststep+' '+Crustlayerid+' 1\n'
    inparacrustVs2=perturbVal+' '+perturbPerc+' '+pertRangeCrust+' '+gwVsCruststep+' '+Crustlayerid+' 2\n'
    inparacrustVs3=perturbVal+' '+perturbPerc+' '+pertRangeCrust+' '+gwVsCruststep+' '+Crustlayerid+' 3\n'
    inparacrustVs4=perturbVal+' '+perturbPerc+' '+pertRangeCrust+' '+gwVsCruststep+' '+Crustlayerid+' 4\n'
    inparacrustVs5=perturbVal+' '+perturbPerc+' '+pertRangeCrust+' '+gwVsCruststep+' '+Crustlayerid+' 5\n'
    inparacrustVs6=perturbVal+' '+perturbPerc+' '+pertRangeCrust+' '+gwVsCruststep+' '+Crustlayerid+' 6\n'
    inparacrustVs7=perturbVal+' '+perturbPerc+' '+pertRangeCrust+' '+gwVsCruststep+' '+Crustlayerid+' 7\n'

    ##Mantle
    inparamantleVs0=perturbVal+' '+perturbPerc+' '+pertRangeMantle+' '+gwVsMantlestep+' '+Mantlelayerid+' 0\n'
    inparamantleVs1=perturbVal+' '+perturbPerc+' '+pertRangeMantle+' '+gwVsMantlestep+' '+Mantlelayerid+' 1\n'
    inparamantleVs2=perturbVal+' '+perturbPerc+' '+pertRangeMantle+' '+gwVsMantlestep+' '+Mantlelayerid+' 2\n'
    inparamantleVs3=perturbVal+' '+perturbPerc+' '+pertRangeMantle+' '+gwVsMantlestep+' '+Mantlelayerid+' 3\n'
    inparamantleVs4=perturbVal+' '+perturbPerc+' '+pertRangeMantle+' '+gwVsMantlestep+' '+Mantlelayerid+' 4\n'
    inparamantleVs5=perturbVal+' '+perturbPerc+' '+pertRangeMantle+' '+gwVsMantlestep+' '+Mantlelayerid+' 5\n'
    inparamantleVs6=perturbVal+' '+perturbPerc+' '+pertRangeMantle+' '+gwVsMantlestep+' '+Mantlelayerid+' 6\n'

    inparasedthick=perturbThick+' '+perturbPerc+' '+pertRangSedThick+' '+gwstepsedthick+' '+Sedlayerid+'\n'
    #inparacrust thick set in for loop below, with range = crustpercpert*Foundcrustalthickness
    #######################


    # ############################################
    #general settings
    crustnpara=6; mantlenpara=4 #number of b-splines in each layer (leave as an int)
    sednpara=2 #4th col in mod.STA (required as assuming a top linear layer here)
    sed_value=1.5 # 
    # Layer index of layer right above the Moho in 1D velocity model
    # moho_id = 9
    #

    #
    # modfile='../Vel_mod/NTW1d_H14_1km'
    modfile='../Vel_mod/NTW1d_H14_1km_final'
    # currently set up to use diffraction global model as a starting model
    # ############################################



    # ############################################
    # mod.STA settings ...
    modlinear='4' ; modsplines='2' #2nd col
    sedrhoflag='1'; sedQflag='2'; sedPflag='1'; sedmodvpvs='0'; sedVpVs1='0'; #cols -11, -10, -9
    crustrhoflag='1'; crustQflag='3'; crustPflag='3'; crmodvpvs='1.75'; crVpVs1='0'; #cols -11, -10, -9
    mantlerhoflag='2'; mantleQflag='4'; mantlePflag='2'; mtmodvpvs='0'; mtVpVs1='0'; #cols -11, -10, -9
#
    # modlinear='4' ; modsplines='2' #2nd col
    # sedrhoflag='1'; sedQflag='2'; sedPflag='1'; sedmodvpvs='0'; sedVpVs1='0'; #cols -11, -10, -9
    # crustrhoflag='1'; crustQflag='3'; crustPflag='1'; crmodvpvs='0'; crVpVs1='0'; #cols -11, -10, -9
    # mantlerhoflag='2'; mantleQflag='4'; mantlePflag='2'; mtmodvpvs='0'; mtVpVs1='0'; #cols -11, -10, -9
#    
    #leave these as 0's for now; untested as to if they carry through codes correctly or not
    sedDrho='0'; sedDrho1='0'; seddVs='0'; seddvs1='0'; sedfdvs='0'; sedfdvs1='0'
    crDrho='0'; crDrho1='0'; crdVs='0'; crdvs1='0'; crfdvs='0'; crfdvs1='0'
    mtDrho='0'; mtDrho1='0'; mtdVs='0'; mtdVs1='0'; mtfdvs='0'; mtfdvs1='0'
    # ############################################
    


    #this is used in looking at the STA_data in the figure created below
    #allper=[8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 24, 30, 40, 50, 60, 70, 80, 90, 100]
    allper=[2, 3, 4, 5, 6, 7, 8, 9]#[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    #allper=[3, 4 , 5, 6, 7, 8, 9, 10]
    phper=[4, 5, 6, 7, 8, 9]#[3, 4 , 5, 6, 7, 8, 9, 10]
    eqper=[ 0 ]
    #note that this does not need to be used if instead use the 'plot ALL' option for Vph and H/V (lines 370 & 381 - comment out other plt.plot() info though!)



    # ########## SHOULD NOT NEED TO EDIT BEYOND THIS POINT ###########
    # ############################################# ############################################

    #%% Get a starting model! Using Shapiro & Ritzwoller diffraction model


    modxmax=127.0
    modymax=26.0
    modxmin=121.0
    modymin=21
    modstepx=0.025
    modstepy=0.025

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
    '''
    with open(modfile,'r') as file:
        i = 0 
        ii=0
        for line in file:
            tmp=line.split()

            #get depth, vsv
            if ii>0:
                depth=np.float(tmp[0])
                vsvtmp=np.float(tmp[2])
                vsvtmp = np.float(tmp[1])
                depthidx=int(np.round(depth/4))
                hhhmodvs[i] = vsvtmp
                allgridmoddepth[i] = depth
                i += 1    
            if ii<100:
                ii=ii+1
            else:
                ii=0
    '''

    allgridmoddepth = np.genfromtxt(modfile,usecols=[0],unpack=True,dtype='float')#/data/cnliu/ >> ../Vel_mod/
    modvs = np.genfromtxt(modfile,usecols=[1],unpack=True,dtype='float')
    #allgridmoddepth = np.array(allgridmoddepth)
    tmp=allgridmoddepth
    #allgridmoddepth=tmp[:37] # Choose the deepest depth!!
    #print(allgridmoddepth)
    os.chdir(scriptdir)
    #'''
    #function to linearly interpolate
    def lininterpwiki(x,x0,y0,x1,y1):
        tmp=((x-x0)/(x1-x0))
        y=(y0*(1-tmp))+(y1*tmp)
        return y

    #load and find nearest
    def find_nearest(array,value):
        #may be useful to determine grid value closest to station from Hongrui Results
        idx=(np.abs(array-value)).argmin();
        return array[idx],idx
    #Example: lonsta,ii=find_nearest(llons,lontmp)

    if not os.path.isdir(datadir+'/query_dir'):
        os.mkdir(datadir+'/query_dir')

    if os.path.isfile(datadir+'/runstations.lst'):
        os.remove(datadir+'/runstations.lst')


    for zz in np.arange(len(staarr)):
        sta=staarr[zz]
        stalon=stalonarr[zz]
        stalat=stalatarr[zz]
        print(sta)
        #print(zz)
        
        
        outdir=datadir+'/'+sta+'_data'
        
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        ##%% Pull model for this specific location and record their velocity
    #    mlatsta,jj=find_nearest(modlatartmp,stalat)
    #    mlonsta,ii=find_nearest(modlonartmp,stalon)
    #    print(mlatsta,jj,modlatartmp,stalat) 
        #print(stalat)
        #print(stalon)
        
        modstavs = modvs
        modstavs[0] = sed_value
        print(modstavs,allgridmoddepth)
        #print to file
        with open ( datadir+'/query_dir/'+sta+'_depth_vs.txt', 'w') as ff:
            for ii in np.arange(len(allgridmoddepth)):#rft[500]=10.0s;#(len(rft)):
                printmf='%5.3f %5.5f\n'%(allgridmoddepth[ii],modstavs[ii])
                ff.write(printmf)
    #    sys.exit()
                
    #'''    
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
        
        
        
        # Get dVs/ddepth
        dVs=[]; dVsdepth=[]
        for kk in np.arange(1,len(allgridmoddepth)-2):
            dVs.append(abs((modstavs[kk+1]-modstavs[kk-1])/(allgridmoddepth[kk+1]-allgridmoddepth[kk-1])))
            dVsdepth.append((allgridmoddepth[kk+1]+allgridmoddepth[kk-1])*0.5)
        
        #dVs=dVs[1:]; dVsdepth=dVsdepth[1:] #added nov2017
        prominence,prdep=peakdet2(dVs,dVsdepth)
    #    print(prominence,prdep)
        dVsmax1index=np.where(prominence[0:4]==min(prominence[0:4]))[0][0]
        dVsmax2index=(np.where(prominence[4:]==min(prominence[4:]))[0][0])+4
        
    #    print(prdep[dVsmax2index],prdep[dVsmax1index])
        
        dVsmax1tmp=np.where(dVs[0:4]==max(dVs[0:4]))[0][0]
    #    print(dVsdepth[dVsmax1tmp])
        
        
        seddepth,sedindex=find_nearest(allgridmoddepth,dVsdepth[dVsmax1tmp])
        # mohodepth,mohoindex=find_nearest(allgridmoddepth,prdep[dVsmax2index+1])
        mohodepth,mohoindex=find_nearest(allgridmoddepth,prdep[dVsmax2index+1])
        print(">>>seddepth, mohodepth,sedindex,mohoindex:",seddepth, mohodepth,sedindex,mohoindex,dVsmax2index)
        moho_id = mohoindex;
        #print("Sediimentary thickness= ",seddepth," Moho depth= ",mohodepth)
        # seddepth=2;mohodepth=34.5
    #    print(seddepth,sedindex, mohodepth,mohoindex)
        
        crustthick=mohodepth-seddepth
        mantlethick=max(allgridmoddepth)-mohodepth
    #    print(modstavs,allgridmoddepth)
    #    vsBspinversion=modstavs[sedindex+1:mohoindex]; depthBspinversion=allgridmoddepth[sedindex+1:mohoindex]

    #    vsBspinversion=modstavs[1:9]; depthBspinversion=allgridmoddepth[1:9]

        vsBspinversion=modstavs[1:moho_id]; depthBspinversion=allgridmoddepth[1:moho_id]
    #     mvsBspinversion=modstavs[mohoindex:]; mdepthBspinversion=allgridmoddepth[mohoindex:]
        mvsBspinversion=modstavs[moho_id+1:]; mdepthBspinversion=allgridmoddepth[moho_id+1:]
        cmd=scriptdir+'/lfBsp_nlay 0 '+str(len(vsBspinversion))+' 2.0 '+str(crustnpara)+' 4 '+str(len(vsBspinversion))

    #    cmd=scriptdir+'/lfBsp_nlay 0 8 1.35 '+str(crustnpara)+' 4 '+str(len(vsBspinversion))
        #print(cmd)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        process.wait()
    #    print(process.returncode)
        bspfile='B_spline_'
        FCBs=[]
        for kk in np.arange(crustnpara):
            FCBs.append(np.loadtxt(scriptdir+'/'+bspfile+str(kk)+'.txt',usecols=(1),unpack=True))
            
    #    print(cmd)
    #    print('cmd finished')
        
        
        cmd=scriptdir+'/lfBsp_nlay 0 '+str(len(mvsBspinversion))+' 0.75 '+str(mantlenpara)+' 4 '+str(len(mvsBspinversion))
    #    cmd=scriptdir+'/lfBsp_nlay 0 20 1.35 '+str(mantlenpara)+' 4 '+str(len(mvsBspinversion))
        #print(cmd)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        process.wait()
    #    print(process.returncode)
    #    print('cmd fine')
        bspfile='B_spline_'
        mFCBs=[]
        for kk in np.arange(mantlenpara):
            mFCBs.append(np.loadtxt(scriptdir+'/'+bspfile+str(kk)+'.txt',usecols=(1),unpack=True))
            #print(np.loadtxt(scriptdir+'/'+bspfile+str(kk)+'.txt',usecols=(1),unpack=True))
        
        FCGmatrix=np.vstack([FCBs]).T
        mFCGmatrix=np.vstack([mFCBs]).T
        #print(FCGmatrix,mFCGmatrix)

        #upload and invert bsplines
        FCcrustcoeff=np.linalg.lstsq(FCGmatrix,vsBspinversion)[0]
        mFCcrustcoeff=np.linalg.lstsq(mFCGmatrix,mvsBspinversion)[0]
                    
        #vsfromcrustcoeff=np.dot(Gmatrix,crustcoeff)
        FCvsfromcrustcoeff=np.dot(FCGmatrix,FCcrustcoeff)
        mFCvsfromcrustcoeff=np.dot(mFCGmatrix,mFCcrustcoeff)
                    
        sedvalue=[]; seddepth=[]
        sedvalue.append(modstavs[0]); seddepth.append(allgridmoddepth[0])
    #    sedvalue.append(modstavs[sedindex]); seddepth.append(allgridmoddepth[sedindex])
        sedvalue.append(modstavs[1]); seddepth.append(allgridmoddepth[1])
    #    print(modstavs[0])
    #    sys.exit()
        if sedvalue[0]==0:
    #        sedvalue[0]=0.25
            sedvalue[0]=0.5
        if sedvalue[1]==0:
            sedvalue[1]=0.5

        
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
        
        # Vs
        ax1.plot(sedvalue,seddepth,'yo',label='Sed Picks')
        ax1.plot(FCvsfromcrustcoeff,depthBspinversion,'ko',label='Est from '+str(crustnpara)+' Bsplines Vs',alpha=0.4)
        ax1.plot(mFCvsfromcrustcoeff,mdepthBspinversion,'go',label='Est from '+str(mantlenpara)+' Bsplines Vs',alpha=0.4)
        ax1.plot(modstavs,allgridmoddepth,'r-',label='Raw Vs Model')
        
        ax1.set_ylim([0,120])
        ax1.legend(loc='lower left',fontsize=10)
        
        ax1.set_ylim(ax1.get_ylim()[::-1])
        ax1.set_xlim([0.0,5.7])
        ax1.set_ylabel('Depth (km)')
        ax1.set_xlabel('Vs (km/s)')
        ax1.tick_params(labeltop=True)
        ax1.grid(color='k', axis='both',linestyle='-', linewidth=2,alpha=0.1,zorder=2)
        
        print(len(phper),len(mergedph[0:len(phper)]))
        print(zz, staarr[zz])
        # PHASE
        ax2.errorbar(allphper,mergedph,yerr=mergedphun,fmt='.:',color='r',ecolor='r',elinewidth=1.5,capthick=1.5,label='Eq Vph',zorder=5,alpha=0.5)
        # ax2.errorbar(phper,mergedph[0:len(phper)],yerr=mergedphun[0:len(phper)],fmt='.:',color='k',ecolor='k',elinewidth=1.5,capthick=1.5,label='ANT Vph',zorder=5,alpha=0.5)
        #ax2.errorbar(allphper,mergedph,yerr=mergedphun,fmt='.:',color='k',ecolor='r',elinewidth=1.5,capthick=1.5,label='All Vph',zorder=5,alpha=0.5)
        ax2.set_xlabel('Period (s)')
        ax2.set_ylabel('Vph (km/s)')
        #ax2.set_xlim([0.0,105.0])
        ax2.set_ylim([1.5,5.0])
        ax2.set_xlim([0,30])
        ax2.legend(loc='lower right',fontsize=10)
        
    
        
        # H/V
        #AN
        # ax3.errorbar(phper,hvall[0:len(phper)],yerr=8.0*hvun[0:len(phper)],fmt='.:',color='k',ecolor='k',elinewidth=1.5,capthick=1.5,label='ANT H/V',zorder=5,alpha=0.5)
        #EQ
        ax3.errorbar(hvper,hvall,yerr=hvun,fmt='.:',color='r',ecolor='r',elinewidth=1.5,capthick=1.5,label='Eq H/V',zorder=5,alpha=0.5)
        #ax3.errorbar(hvper,hvall,yerr=hvun,fmt='.:',color='r',ecolor='k',elinewidth=1.5,capthick=1.5,label='All H/V',zorder=5,alpha=0.5)
        ax3.set_xlim([0,30])
        ax3.set_xlabel('Period (s)')
        ax3.set_ylabel('H/V')
        ax3.set_ylim([0,2.0])
        ax3.legend(loc='upper right',fontsize=10)
        
        #RF
        ax4.errorbar(rft,rfamp,yerr=rfunc,fmt='.:',color='k',ecolor='k',elinewidth=1.5,capthick=0.0,label='RFunc',zorder=5,alpha=0.5)
        ax4.plot(rft,rfamp,'r-',zorder=7,alpha=0.96,label='RF')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('RF Amp')
        ax4.set_xlim([min(rft)-0.2,max(rft)+0.2])
        ax4.set_ylim([-0.25,0.6])
        ax4.legend(loc='upper right',fontsize=10)
        
        #plt.show()
        fig.savefig(datadir+'/query_dir/StartingModel_'+sta+'.png',bbox_inches='tight',transparent=True,pad_inches=0.1)
        plt.close()
        time.sleep(2)

        fig, axs = plt.subplots(1,2, figsize=(8,11), sharey=True)
        # plt.figure(figsize=(3,7))
        axs[0].set_title('NVN_1D', fontsize=24)
        axs[1].set_title('B-spline (%s, %s)' %(crustnpara, mantlenpara), fontsize=24)

        axs[0].plot(modvs, tmp*-1, 'ko-')
        axs[0].plot(vsBspinversion, tmp[1:moho_id]*-1, 'ro-', alpha=0.5)
        axs[0].plot(mvsBspinversion, tmp[moho_id+1:]*-1, 'ro-', alpha=0.5)
        axs[0].plot(FCvsfromcrustcoeff, tmp[1:moho_id]*-1, 'bo--', alpha=0.5)
        axs[0].plot(mFCvsfromcrustcoeff, tmp[moho_id+1:]*-1, 'bo--', alpha=0.5)
        axs[0].plot(sedvalue, tmp[0:2]*-1, 'c.-')

        axs[1].plot(vsBspinversion, tmp[1:moho_id]*-1, 'bo--')
        axs[1].plot(FCvsfromcrustcoeff, tmp[1:moho_id]*-1, 'ro--')
        axs[1].plot(mvsBspinversion, tmp[moho_id+1:]*-1, 'bo--')
        axs[1].plot(mFCvsfromcrustcoeff, tmp[moho_id+1:]*-1, 'ro--')
        fig.savefig(datadir+'/query_dir/B-spline_'+str(crustnpara)+'-'+str(mantlenpara)+'_comparisons_'+sta+'.png',bbox_inches='tight',transparent=True,pad_inches=0.1)
        plt.close()

    #    '''
        print('writing starting model to file..')

        ###############################################
        ### output starting model ###
        ###############################################
        #pulled Vp/Vs from Moschetti et al, 2010 (http://ciei.colorado.edu/pubs/2010/2010JB007448.pdf)
        #sedvpvs=2.125
        # VP/VS, sediment layer 1.75-2.5 km/s Brocher, 2005
        #crustvpvs=1.789
        # VP/VS, crystalline crust (same in all layers) 1.70-1.8 km/s Brocher [2005]
        #mantlevpvs=1.8
        # VP/VS, mantle 1.8 km/s Shapiro and Ritzwoller [2002]
        
        
        ########################
        ## Sediment mod.STA settings ##
        sthick=seddepth[len(seddepth)-1]-seddepth[0]
        printsedlayer=Sedlayerid+' '+modlinear+' %4.1f'%(sthick)+' %d'%(sednpara)+' %5.6f %5.6f '%(sedvalue[0],sedvalue[1])+sedrhoflag+' '+sedQflag+' '+sedPflag
        #printsedlayer=Sedlayerid+' '+modsplines+' %4.1f'%(sthick)+' %d'%(sednpara)+' %5.6f %5.6f '%(sedvalue[0],sedvalue[1])+sedrhoflag+' '+sedQflag+' '+sedPflag
        printsedlayer=printsedlayer+' '+sedmodvpvs+' '+sedVpVs1+' '+sedDrho+' '+sedDrho1+' '+seddVs+' '+seddvs1+' '+sedfdvs+' '+sedfdvs1+'\n'
        #printsedlayer='0 4 %4.1f 2 %5.6f %5.6f 1 2 1 0 0 0 0 0 0 0 0\n'%(sthick,sedvalue[0],sedvalue[1])
        #layerID layertype thickness numberparams Vs0 Vs1.. VsN Rflag Qflag Pflag VpVs VpVs1 Drho Drho1 dVs dvs1 fdvs fdvs1
        # see Calmodel.C for details on declaration and use
        ########################
        
        ########################
        ## Crust mod.STA settings!! ##
        #Note crustthick & mantlethick previously defined
        printcrustlayer=Crustlayerid+' '+modsplines+' %4.1f %d'%(crustthick,crustnpara)
        printcrusttmp=''
        for kk in np.arange(crustnpara):
    #       printcrusttmp=printcrusttmp+' %5.6f'%(FCcrustcoeff[kk])
            printcrusttmp=printcrusttmp+' %5.6f'%(FCcrustcoeff[kk])
            print(FCcrustcoeff[kk])
        printcrustvpvs=' '+crustrhoflag+' '+crustQflag+' '+crustPflag+' '+crmodvpvs+' '+crVpVs1+' '+crDrho+' '+crDrho1+' '+crdVs+' '+crdvs1+' '+crfdvs+' '+crfdvs1+'\n'
        #'1 3 1 0 0 0 0 0 0 0 0\n'
        printcrustlayer=printcrustlayer+printcrusttmp+printcrustvpvs
        #######################
        ########################
        ## Mantle mod.STA settings!! ##
        #Note crustthick & mantlethick previously defined
        printmantletmp=''
        for kk in np.arange(mantlenpara):
            printmantletmp=printmantletmp+' %5.6f'%(mFCcrustcoeff[kk])
        printmantlelayer=Mantlelayerid+' '+modsplines+' %4.1f %d '%(mantlethick,mantlenpara)
        printmantletail=' '+mantlerhoflag+' '+mantleQflag+' '+mantlePflag+' '+mtmodvpvs+' '+mtVpVs1+' '+mtDrho+' '+mtDrho1+' '+mtdVs+' '+mtdVs1+' '+mtfdvs+' '+mtfdvs1+'\n'
        #' 2 4 2 0 0 0 0 0 0 0 0\n'
        printmantlelayer=printmantlelayer+printmantletmp+printmantletail
        #######################
        
        with open ( outdir+'/mod.'+sta, 'w') as ff:
            ff.write(printsedlayer); ff.write(printcrustlayer); ff.write(printmantlelayer)

        
        
        
        
        #create in.para files
        fname=datadir+'/'+sta+'_data/mod.'+sta
        if os.path.isfile(fname):
            print(sta+' exists!')
            
            
            #read in crust thickness
            with open(fname,'r') as ff:
                datatmpmfout=ff.readlines()
            
            mftmp=np.float(datatmpmfout[1].split()[2])
            
            pertRangCrustThick=crustpercpert*mftmp
            if np.isnan(pertRangCrustThick):
                print('error!!!! percpert')
                break
            
            inparacrustthick=perturbThick+' '+perturbAbs+' %3.1f '%(pertRangCrustThick)+gwstepcrustthick+' '+Crustlayerid
            
            #Velocity in mod.121.8-24.675
            rerunallstas=inparasedVs0+inparasedVs1
            rerunallstas=rerunallstas+inparacrustVs0+inparacrustVs1+inparacrustVs2+inparacrustVs3\
                                +inparacrustVs4+inparacrustVs5\
                                # +inparacrustVs6+inparacrustVs7
            #rerunallstas = inparacrustVs1+inparacrustVs2+inparacrustVs3+inparacrustVs4+inparacrustVs5+inparacrustVs6
            ##Mantle
            
            rerunallstas=rerunallstas+inparamantleVs0+inparamantleVs1+inparamantleVs2+inparamantleVs3\
                # +inparamantleVs4+inparamantleVs5\
                # +inparamantleVs6


            rerunallstas=rerunallstas+inparasedthick+inparacrustthick
            
            
            fnpara=datadir+'/MonteCarlo/in.para_'+sta
            with open(fnpara,'w') as ff2:
                ff2.write(rerunallstas)



            #time.sleep(2)
            print('done for this station!!! :')
            print(sta)
            
            with open(datadir+'/runstations.lst','a') as ff:
                ff.write(sta+' '+str(stalon)+' '+str(stalat)+'\n')
        
        
    print(datadir)
