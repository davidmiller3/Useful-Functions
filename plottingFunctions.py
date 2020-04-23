# -*- coding: utf-8 -*-
"""
Created on Wed Jul 05 10:33:09 2017

@author: David
"""
from fittingFunctions import *

def generateBGHeatmaps(gscan,bscan,axlist,plot='x',cmg='YlGn',cmb='PuBu'):
#make plots of 
    gx=gscan
    bx=bscan
    gUF=gx.data['frequency'].unique()
    bUF=bx.data['frequency'].unique()
#make a figure and add subplots
#==============================================================================
#     fig1 = plt.figure(1, figsize=(10,10))
#     ax1 = fig1.add_subplot(221, aspect='equal')
#     ax2 = fig1.add_subplot(222, aspect='equal')
#     ax3 = fig1.add_subplot(223, aspect='equal')
#     ax4 = fig1.add_subplot(224, aspect='equal')
#==============================================================================
#make a heatmap for each map
    makeXYHeatmap2(gx.data,'Green X','Green Y','$f_0$ = ' +str(gUF.min())+', '+'r',axis = axlist[0], cm='YlGn',plot='r')
    
    makeXYHeatmap2(bx.data,'Blue X','Blue Y', '$f_0$ = ' +str(bUF.min())+', '+'r',axis = axlist[1],  cm='PuBu',plot='r')

#Takes the position of the other color laser during a scan and plots it on the position scan of that color laser. This allows orientation.
#The yellow is the position of the blue while the green scans while the red is the position of the green during the blue scan. The 
#convertXYToAxis function takes the position in scan space and converts it to the position on the heatmap axis (which is by # of points)
    x,y=convertXYToAxis(gx.data['Blue X'].unique()[0],gx.data['Blue Y'].unique()[0],axlist[1])
    axlist[1].scatter(x,y,marker='*', s=300, color='yellow')
    x,y=convertXYToAxis(bx.data['Green X'].unique()[0],bx.data['Green Y'].unique()[0],axlist[0])
    axlist[0].scatter(x,y,marker='*', s=300, color='cyan')
#Repeats this but for the value plot. Could be x,y, or phase
    makeXYHeatmap2(gx.data,'Green X','Green Y','$f_0$ = ' +str(gUF.min())+', '+plot,axis = axlist[2], cm='RdBu',plot=plot)
    makeXYHeatmap2(bx.data,'Blue X','Blue Y', '$f_0$ = ' +str(bUF.min())+', '+plot,axis = axlist[3],  cm='RdBu',plot=plot)
    x,y=convertXYToAxis(gx.data['Blue X'].unique()[0],gx.data['Blue Y'].unique()[0],axlist[3])
    axlist[3].scatter(x,y,marker='*', s=300, color='yellow')
    x,y=convertXYToAxis(bx.data['Green X'].unique()[0],bx.data['Green Y'].unique()[0],axlist[2])
    axlist[2].scatter(x,y,marker='*', s=300, color='cyan')
    

    axlist[0].set_xlabel('Green X $(\mu m)$')
    axlist[0].set_ylabel('Green Y $(\mu m)$')
    axlist[1].set_xlabel('Blue X $(\mu m)$')
    axlist[1].set_ylabel('Blue Y $(\mu m)$')    
    axlist[2].set_xlabel('Green X $(\mu m)$')
    axlist[2].set_ylabel('Green Y $(\mu m)$')
    axlist[3].set_xlabel('Blue X $(\mu m)$')
    
    
    axlist[3].set_ylabel('Blue Y $(\mu m)$')     

    
    for i in axlist:
        i.set_title('')
    for tick in axlist[3].get_xticklabels():
        tick.set_rotation(90)
    plt.tight_layout()
    return axlist
    

def generateBBHeatmaps(gscan,bscan,axlist,plot='x',cmg='YlGn',cmb='PuBu'):
#make plots of 
    gx=gscan
    bx=bscan
    gUF=gx.data['frequency'].unique()
    bUF=bx.data['frequency'].unique()
#make a figure and add subplots
#==============================================================================
#     fig1 = plt.figure(1, figsize=(10,10))
#     ax1 = fig1.add_subplot(221, aspect='equal')
#     ax2 = fig1.add_subplot(222, aspect='equal')
#     ax3 = fig1.add_subplot(223, aspect='equal')
#     ax4 = fig1.add_subplot(224, aspect='equal')
#==============================================================================
#make a heatmap for each map
    makeXYHeatmap2(gx.data,'Blue X','Blue Y','$f_0$ = ' +str(gUF.min())+', '+'r',axis = axlist[0], cm='PuBu',plot='r')
    
    makeXYHeatmap2(bx.data,'Blue X','Blue Y', '$f_0$ = ' +str(bUF.min())+', '+'r',axis = axlist[1],  cm='PuBu',plot='r')

#Takes the position of the other color laser during a scan and plots it on the position scan of that color laser. This allows orientation.
#The yellow is the position of the blue while the green scans while the red is the position of the green during the blue scan. The 
#convertXYToAxis function takes the position in scan space and converts it to the position on the heatmap axis (which is by # of points)
    x,y=convertXYToAxis(gx.data['Blue X'].unique()[0],gx.data['Blue Y'].unique()[0],axlist[1])
    axlist[1].scatter(x,y,marker='*', s=300, color='yellow')
    x,y=convertXYToAxis(bx.data['Blue X'].unique()[0],bx.data['Blue Y'].unique()[0],axlist[0])
    axlist[0].scatter(x,y,marker='*', s=300, color='cyan')
#Repeats this but for the value plot. Could be x,y, or phase
    makeXYHeatmap2(gx.data,'Blue X','Blue Y','$f_0$ = ' +str(gUF.min())+', '+plot,axis = axlist[2], cm='RdBu',plot=plot)
    makeXYHeatmap2(bx.data,'Blue X','Blue Y', '$f_0$ = ' +str(bUF.min())+', '+plot,axis = axlist[3],  cm='RdBu',plot=plot)
    x,y=convertXYToAxis(gx.data['Blue X'].unique()[0],gx.data['Blue Y'].unique()[0],axlist[3])
    axlist[3].scatter(x,y,marker='*', s=300, color='yellow')
    x,y=convertXYToAxis(bx.data['Blue X'].unique()[0],bx.data['Blue Y'].unique()[0],axlist[2])
    axlist[2].scatter(x,y,marker='*', s=300, color='cyan')
    

    axlist[0].set_xlabel('Blue X $(\mu m)$')
    axlist[0].set_ylabel('Blue Y $(\mu m)$')
    axlist[1].set_xlabel('Blue X $(\mu m)$')
    axlist[1].set_ylabel('Blue Y $(\mu m)$')    
    axlist[2].set_xlabel('Blue X $(\mu m)$')
    axlist[2].set_ylabel('Blue Y $(\mu m)$')
    axlist[3].set_xlabel('Blue X $(\mu m)$')
    
    
    axlist[3].set_ylabel('Blue Y $(\mu m)$')     

    
    for i in axlist:
        i.set_title('')
    for tick in axlist[3].get_xticklabels():
        tick.set_rotation(90)
    plt.tight_layout()
    return axlist
    
    
def makeXYHeatmap3(data,X,Y, title,axis,vm=[], cm='OrRd',plot='r',*args, **kwargs):
        if len(vm)==0:
            rmin = data[plot].min()
            rmax = data[plot].max()
        else:
           rmin=vm[0]
           rmax=vm[1] 
        xmi=data[X].min()
        xmx=data[X].max()
        ymi=data[Y].min()
        ymx=data[Y].max()
        n=5
        xticks = np.round(np.linspace(xmi,xmx,n),2)
        yticks = np.round(np.linspace(ymi,ymx,n),2)
        g= data.pivot_table(columns=X, index=Y, values = plot).sort_index(axis=0, ascending=False)
        p=sns.heatmap(g,ax=axis,cmap = cm,*args, **kwargs)
        p.set_xticklabels(xticks)
        p.set_yticklabels(yticks)
        p.set_xticks(np.linspace(p.get_xlim()[0],p.get_xlim()[1],n))
        p.set_yticks(np.linspace(p.get_ylim()[0],p.get_ylim()[1],n))
        plt.xticks(rotation=0)
        axis.set_title(title)
        return p

def makeXYHeatmap2(data,X,Y, title,axis,vm=[], cm='OrRd',plot='r',*args, **kwargs):
        if len(vm)==0:
            rmin = data[plot].min()
            rmax = data[plot].max()
        else:
           rmin=vm[0]
           rmax=vm[1] 
        xmi=data[X].min()
        xmx=data[X].max()
        ymi=data[Y].min()
        ymx=data[Y].max()
        n=5
        xticks = np.round(np.linspace(xmi,xmx,n),2)
        yticks = np.round(np.linspace(ymi,ymx,n),2)
        g= data.pivot_table(columns=X, index=Y, values = plot).sort_index(axis=0, ascending=False)
        p=sns.heatmap(g,ax=axis,cmap = cm,*args, **kwargs)
        p.set_xticklabels(xticks)
        p.set_yticklabels(yticks)
        p.set_xticks(np.linspace(p.get_xlim()[0],p.get_xlim()[1],n))
        p.set_yticks(np.linspace(p.get_ylim()[0],p.get_ylim()[1],n))
        plt.xticks(rotation=0)
        axis.set_title(title)

def makeXYHeatmap(data,X,Y, title,axis, vm=[],cm='OrRd',plot='r',*args, **kwargs):

        xmi=data[X].min()
        xmx=data[X].max()
        ymi=data[Y].min()
        ymx=data[Y].max()
        n=5
        xticks = np.round(np.linspace(xmi,xmx,n),2)
        yticks = np.round(np.linspace(ymi,ymx,n),2)
        g= data.pivot_table(columns=X, index=Y, values = plot).sort_index(axis=0, ascending=False)
        p=sns.heatmap(g,ax=axis,cmap = cm,vmin=vm[0],vmax=vm[1],*args, **kwargs)
        p.set_xticklabels(xticks)
        p.set_yticklabels(yticks)
        p.set_xticks(np.linspace(p.get_xlim()[0],p.get_xlim()[1],n))
        p.set_yticks(np.linspace(p.get_ylim()[0],p.get_ylim()[1],n))
        plt.xticks(rotation=0)
        axis.set_title(title)
        return p
    
def FRLineScan(data,X,Y, title,axis, cm='RdBu',plot='r',vmin=-2*np.pi, vmax=+2*np.pi):
    rmin = data[plot].min()
    rmax = data[plot].max()
    xmi=data[X].min()
    xmx=data[X].max()
    ymi=data[Y].min()
    ymx=data[Y].max()
    n=5
    xticks = np.round(np.linspace(xmi,xmx,n),2)
    yticks = np.round(np.linspace(ymi,ymx,n),2)
    g= data.pivot_table(columns=X, index=Y, values = plot).sort_index(axis=0, ascending=False)
    if plot =='phase':
         p=sns.heatmap(g,ax=axis,cmap = cm,vmin=vmin,vmax=vmax)
    else:    
        p=sns.heatmap(g,ax=axis,cmap = cm,vmin=rmin,vmax=rmax)
    p.set_xticklabels(xticks)
    p.set_yticklabels(yticks)
    p.set_xticks(np.linspace(p.get_xlim()[0],p.get_xlim()[1],n))
    p.set_yticks(np.linspace(p.get_ylim()[0],p.get_ylim()[1],n))
    plt.xticks(rotation=0)
    axis.set_title(title)
    return p

def FRPlot(scandata, axlist, FR4Path= None, logx=False, *args, **kwargs):
    data=scandata
    if FR4Path == None:
        pass
    else:
        data4=scan(FR4Path).data
        data.phase=data.phase-data4.phase
    if logx == False:   
        axlist[0].plot(data.frequency, data.r, *args, **kwargs)
        axlist[1].plot(data.frequency, data.phase, *args, **kwargs)
    else:
        axlist[0].plot(data.frequency, data.r, *args, **kwargs)
        axlist[1].plot(data.frequency, data.phase, *args, **kwargs)
        axlist[0].set_xscale("log")
        axlist[1].set_xscale("log")
        

def fitFRLine2(data, Q=300, sweep = 'Blue Drive', soverN=4, ctr_range = 1.2,edgeTrim=10, peakFilter = 4, phaseTrim=2, fitPhase = True, plot = False):
# Pick the axis
    f0=0
    x = data  
    
    master = pd.DataFrame()
    for i in x[sweep].unique():
# Loop through unique values of the laser scan position
        sigma=1.0/(2*Q)
# Make subframe
        FRScan = x[x[sweep] == i]
        try:
            maxidx=FRScan.index[-1]
        except IndexError:
            print 'Index Error'
            continue
        minidx=FRScan.index[0]
        try:
            if f0 == 0:
                idx = returnNearestValue(FRScan.r, FRScan.r.max())
                idxm = returnNearestValue(FRScan.r, FRScan.r.max())
            else:
                idx = returnNearestValue(FRScan.frequency, f0)
                idxm = returnNearestValue(FRScan.r, FRScan.r.max())
    # Don't fit anything where the max is wit hin edgeTrim from the end of the data window
    #        if (abs(idx - minidx)<edgeTrim or abs(idx - maxidx))<edgeTrim:
    #            continue
            try:
                dF=FRScan.frequency[idx+1]-FRScan.frequency[idx]
            except KeyError:
                print 'Key Error'
                continue

            rmax =  FRScan.r[idxm]  
            fmax= FRScan.frequency[idxm]
            sigmaIndex = int(sigma*2*np.pi*fmax/dF)
    # Removes the peak up to peakFilter HWHM from the center based on entered Q
            peakRemoved =  pd.concat([FRScan.r[:idx-peakFilter*sigmaIndex],FRScan.r[idx+peakFilter*sigmaIndex:]])
            avg= peakRemoved.mean()
    # Don't fit anything where the max/soverN less than the average of the data with the peak removed
            if avg>rmax/soverN:
#                print 'SN'
                continue
    # DFit Amplitude spectrum
            out, pars = lmDDOFit(FRScan.frequency,FRScan.r,[FRScan.frequency[idx],rmax*2*sigma,sigma],ctr_range)
            if plot ==True:
                plt.figure()
                out.plot_fit()
                plt.xlim(pars)
            f0idx = returnNearestValue(FRScan.frequency, pars[0])
            sigmaIndex=int(pars[2]*2*np.pi*pars[0]/dF)
    # Based on the fitted amplitude, trim the range of the phase data to fit to just around the peak. Trim to phaseTrim*sigma
            phaseMin=f0idx-phaseTrim*sigmaIndex-FRScan.index[0]
            phaseMax = f0idx+phaseTrim*sigmaIndex-FRScan.index[0]
            idx = str(i)
    # Fit phase in degrees
            if fitPhase == True:
                out, phasepars = lmDDOPhaseFit(FRScan.frequency[phaseMin:phaseMax],FRScan.phase[phaseMin:phaseMax]*180/np.pi,pars)
            else:
                phasepars=[FRScan.phase.iloc[0],0,0,0,0,0,0]
    # Save fit to dataframe
            newDFRow = pd.DataFrame({'Green X' : FRScan['Green X'].unique()[0], 'Green Y' : FRScan['Green Y'].unique()[0], 'Blue X' : \
                FRScan['Blue X'].unique()[0], 'Blue Y' : FRScan['Blue Y'].unique()[0],\
                'DC Offset' : FRScan['DC Offset'].unique()[0], 'Blue Drive' : \
                   FRScan['Blue Drive'].unique()[0], 'Output 1 Amplitude' : FRScan['Output 1 Amplitude'].unique()[0],'Output 2 Amplitude' : FRScan['Output 2 Amplitude'].unique()[0], \
                    'Q': pars[3], 'Qp' : phasepars[3], 'A' : pars[1], \
                    'f0' : pars[0], 'f0p' : phasepars[0], 'm' : pars[5], 'mp' : phasepars[1],\
                    'b' : pars[4], 'bp' : phasepars[2], 'Keithley Voltage':FRScan['Keithley Voltage'].unique()[0], \
                    'HWP':FRScan['HWP'].unique()[0],'Qerr':(1/(2*out.params['ddo0_sigma'].value)**2)*(out.params['ddo0_sigma'].stderr),\
                    'f0err':out.params['ddo0_center'].stderr\
                    ,'timestamp':FRScan['timestamp'].unique()[0]}, index=[idx])
            master = pd.concat([master,newDFRow])

            f0 = pars[0]
        except TypeError:
            print i,'TypeError'
            pass
    return master    
def fitFRLine(data, Q=300, sweep = 'Blue Drive', soverN=8, ctr_range = 1.2,edgeTrim=10, peakFilter = 4, phaseTrim=2, fitPhase = True, plot = False, time = 5):
# Pick the axis
    x = data  
    master = pd.DataFrame()
    for i in x[sweep].unique():
# Loop through unique values of the laser scan position
        sigma=1.0/(2*Q)
# Make subframe
        FRScan = x[x[sweep] == i]
        idxm = returnNearestValue(FRScan.r, FRScan.r.max())
        rmax =  FRScan.r[idxm]  
        fmax= FRScan.frequency[idxm]
#        sigmaIndex = int(sigma*2*np.pi*fmax/dF)
# Removes the peak up to peakFilter HWHM from the center based on entered Q
#        peakRemoved =  pd.concat([FRScan.r[:idx-peakFilter*sigmaIndex],FRScan.r[idx+peakFilter*sigmaIndex:]])
#        avg= peakRemoved.mean()
# Don't fit anything where the max/soverN less than the average of the data with the peak removed
#        if avg>rmax/soverN:
#                print 'SN'
#            continue
# DFit Amplitude spectrum
        stmean = np.array(FRScan.r)[0:5].mean()
        fnmean = np.array(FRScan.r)[-5:].mean()
        if (rmax/float(soverN)>stmean)|(rmax/float(soverN)>stmean):
            try:
                with timeout(seconds=time):
                    out, pars = lmDDOFit(FRScan.frequency,FRScan.r,[FRScan.frequency[idxm],rmax*2*sigma,sigma],ctr_range)
                    ferr = out.params['ddo0_center'].stderr
                    try:
                        Qerr = (1/(2*out.params['ddo0_sigma'].value)**2)*(out.params['ddo0_sigma'].stderr)
                    except TypeError:
                        Qerr = np.nan
                if plot ==True:
                    plt.figure()
                    out.plot_fit()
                    plt.xlim(pars[0]-6*pars[0]/pars[3],pars[0]+6*pars[0]/pars[3])
            except TimeoutError:
                pars = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
                Qerr = np.nan
                ferr = np.nan
        else:
            pars = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]            
            Qerr = np.nan
            ferr = np.nan
        phasepars=[FRScan.phase.iloc[0],0,0,0,0,0,0]
# Save fit to dataframe
        idx = str(i)
        newDFRow = pd.DataFrame({'Green X' : FRScan['Green X'].unique()[0], 'Green Y' : FRScan['Green Y'].unique()[0], 'Blue X' : \
            FRScan['Blue X'].unique()[0], 'Blue Y' : FRScan['Blue Y'].unique()[0],\
            'DC Offset' : FRScan['DC Offset'].unique()[0], 'Blue Drive' : \
               FRScan['Blue Drive'].unique()[0], 'Output 1 Amplitude' : FRScan['Output 1 Amplitude'].unique()[0],'Output 2 Amplitude' : FRScan['Output 2 Amplitude'].unique()[0], \
                'Q': pars[3], 'Qp' : phasepars[3], 'A' : pars[1], \
                'f0' : pars[0], 'f0p' : phasepars[0], 'm' : pars[5], 'mp' : phasepars[1],\
                'b' : pars[4], 'bp' : phasepars[2], 'Keithley Voltage':FRScan['Keithley Voltage'].unique()[0], \
                'HWP':FRScan['HWP'].unique()[0],'Qerr':Qerr,\
                'f0err':ferr\
                ,'timestamp':FRScan['timestamp'].unique()[0]}, index=[idx])
        master = pd.concat([master,newDFRow])
    return master 
#,'Qerr':1/(2*(out.params['ddo0_sigma'].stderr))
def complex_array_to_rgb(X, theme='dark', rmax=False):
    '''Takes an array of complex number and converts it to an array of [r, g,
    b], where phase angle indicates hue and saturaton/value are given by the
    magnitude of the complex number.'''
    
    if not rmax:
        absmax = np.abs(X).max()
    else:
        absmax = rmax
    Y = np.zeros(X.shape + (3,), dtype='float')
    Y[..., 0] = np.angle(X) / (2 * np.pi) % 1
    Y[..., 0].max()
    Y[..., 0].max()
    if theme == 'light':
        Y[..., 1] = np.clip(np.abs(X) / absmax, 0, 1)
        Y[..., 2] = 1
    elif theme == 'dark':
        Y[..., 1] = 1
        Y[..., 2] = np.clip(np.abs(X) / absmax, 0, 1)
    Y = colors.hsv_to_rgb(Y)
    return Y
def makeAmpPhasePlot(data, ax, green=True, theme = 'dark'):
    if green == True:
        xaxis = 'Green X'
        yaxis = 'Green Y'
    else:
        xaxis = 'Blue X'
        yaxis = 'Blue Y'
    com = data.x + 1j * data.y
    C = complex_array_to_rgb(np.reshape(com,(len(data[xaxis].unique()),len(data[yaxis].unique()))),theme=theme)
    ax.imshow(C, origin='lower', cmap='gray', interpolation='none',extent=[data[xaxis].min(),data[xaxis].max(),\
                                                                              data[yaxis].min(),data[yaxis].max()])
#    ax.imshow(C, origin='lower', cmap='gray', interpolation='none',extent=[0,5,0,5])
#    rc('text', usetex=False)                                                                       
    ax.set_xlabel(r'$X}$ $(\mu m)}$')
    ax.set_ylabel(r'$Y$ $(\mu m)$')
    
    return ax

def addLaserPositions(d, axlist, marker, color):
#Adds the position of the lasers ontop of an given already made 2x2 amp phase XY heatmap
    x,y=convertXYToAxis(d['Blue X'].unique()[0],d['Blue Y'].unique()[0],axlist[1])
    axlist[1].scatter(x,y,marker=marker, s=300, color=color)
    x,y=convertXYToAxis(d['Green X'].unique()[0],d['Green Y'].unique()[0],axlist[0])
    axlist[0].scatter(x,y,marker=marker, s=300, color=color)
    x,y=convertXYToAxis(d['Blue X'].unique()[0],d['Blue Y'].unique()[0],axlist[3])
    axlist[3].scatter(x,y,marker=marker, s=300, color=color)
    x,y=convertXYToAxis(d['Green X'].unique()[0],d['Green Y'].unique()[0],axlist[2])
    axlist[2].scatter(x,y,marker=marker, s=300, color=color)
    return axlist
    
def convertXYToAxis(x,y,ax):
# Takes a real xy point and a heatmap and converts the XY to heatmap coordinates
    axmi=ax.get_xlim()[0]
    axmx=ax.get_xlim()[1]
    axl=axmx-axmi
    dmi = float(ax.get_xticklabels()[0].get_text())
    dmx = float(ax.get_xticklabels()[-1].get_text())
    dml=dmx-dmi
    xp=axl*((x-dmi)/dml)+axmi
    axmi=ax.get_ylim()[0]
    axmx=ax.get_ylim()[1]
    axl=axmx-axmi
    dmi = float(ax.get_yticklabels()[0].get_text())
    dmx = float(ax.get_yticklabels()[-1].get_text())
    dml=dmx-dmi
    yp=axl*((y-dmi)/dml)+axmi
    return xp,yp
    
def drawInset(ax1,ax2):
    dmxi = float(ax2.get_xticklabels()[0].get_text())
    dmxx = float(ax2.get_xticklabels()[-1].get_text())
    dmyi = float(ax2.get_yticklabels()[0].get_text())
    dmyx = float(ax2.get_yticklabels()[-1].get_text())
    xi,yi = convertXYToAxis(dmxi, dmyi, ax1)
    xf,yf = convertXYToAxis(dmxx, dmyx, ax1)
    ax1.plot((xi,xi), (yi,yf),'r')
    ax1.plot((xi,xf), (yf,yf),'r')
    ax1.plot((xf,xf), (yi,yf),'r')
    ax1.plot((xi,xf), (yi,yi),'r')
    
def fitFRMap(x, Q=300, green = True, soverN=4, edgeTrim=10, peakFilter = 4, phaseTrim=2, fitPhase = True):
# Pick the axis
    if green == True:
        print 'green'
        xaxis = 'Green X'
        yaxis = 'Green Y'
    else:
        print 'blue'
        xaxis = 'Blue X'
        yaxis = 'Blue Y'
        
    master = pd.DataFrame()
    for i in x.data[yaxis].unique():
        for ii in x.data[xaxis].unique():
            print i, ii
# Loop through unique values of the laser scan position
            sigma=1.0/(2*Q)
# Make subframe
            FRScan = x.data[x.data[yaxis] == i][x.data[xaxis]==ii]
            try:
                maxidx=FRScan.index[-1]
            except IndexError:
                continue
            minidx=FRScan.index[0]
            idx = returnNearestValue(FRScan.r, FRScan.r.max())
# Don't fit anything where the max is within edgeTrim from the end of the data window
            if (abs(idx - minidx)<edgeTrim or abs(idx - maxidx))<edgeTrim:
                continue
            dF=FRScan.frequency[idx]-FRScan.frequency[idx-1]
            rmax =  FRScan.r[idx]  
            fmax= FRScan.frequency[idx]
            sigmaIndex = int(sigma*2*np.pi*fmax/dF)
# Removes the peak up to peakFilter HWHM from the center based on entered Q
            peakRemoved =  pd.concat([FRScan.r[:idx-peakFilter*sigmaIndex],FRScan.r[idx+peakFilter*sigmaIndex:]])
            avg= peakRemoved.mean()
# Don't fit anything where the max/soverN less than the average of the data with the peak removed
            if avg>rmax/soverN:
                continue
# DFit Amplitude spectrum
            out, pars = lmDDOFit(FRScan.frequency,FRScan.r,[fmax,rmax*2*sigma,sigma])
            f0idx = returnNearestValue(FRScan.frequency, pars[0])
            sigmaIndex=int(pars[2]*2*np.pi*pars[0]/dF)
# Based on the fitted amplitude, trim the range of the phase data to fit to just around the peak. Trim to phaseTrim*sigma
            phaseMin=f0idx-phaseTrim*sigmaIndex-FRScan.index[0]
            phaseMax = f0idx+phaseTrim*sigmaIndex-FRScan.index[0]
            idx = str(i)+'_'+str(ii)
# Fit phase in degrees
            if fitPhase == True:
                try:
                    out, phasepars = lmDDOPhaseFit(FRScan.frequency[phaseMin:phaseMax],np.unwrap(FRScan.phase[phaseMin:phaseMax])*180/np.pi,pars)
                except TypeError:
                    phasepars=[0,0,0,0,0,0,0]
                    
            else:
                phasepars=[0,0,0,0,0,0,0]
    # Save fit to dataframe
            newDFRow = pd.DataFrame({'Green X' : FRScan['Green X'].unique()[0], 'Green Y' : FRScan['Green Y'].unique()[0], 'Blue X' : \
                FRScan['Blue X'].unique()[0], 'Blue Y' : FRScan['Blue Y'].unique()[0],\
                'DC Offset' : FRScan['DC Offset'].unique()[0], 'Blue Drive' : \
                   FRScan['Blue Drive'].unique()[0], 'F1 Amplitude' : FRScan['Output 1 Amplitude'].unique()[0], \
                    'Q': pars[3], 'Qp' : phasepars[3], 'A' : pars[1], \
                    'f0' : pars[0], 'f0p' : phasepars[0], 'm' : pars[5], 'mp' : phasepars[1],\
                    'b' : pars[4], 'bp' : phasepars[2],'timestamp':FRScan['timestamp'].unique()[0]}, index=[idx])
            master = pd.concat([master,newDFRow])
# Return the fitted values for the entire scan
    return master
    
    
def addCrossSectionLines(x,y,ax):
    x,y=convertXYToAxis(x,y,ax)
    x=int(round(x))
    y=int(round(y))
    ax.axhline(y, color = 'k', ls = '--')
    ax.axvline(x, color = 'k', ls = '--')
def plotFrequency(f,ax):
     ax.axvline(x=f, color='r', linestyle='--')       


def addLines(axlist, xlist, ylist,*args,**kwargs):
    for i in axlist:
        i.plot(xlist, ylist,*args,**kwargs)
        i.plot(xlist, ylist,*args,**kwargs)
        
def fitAndPlot(frequency,r,phase,ax4,Q=300,phaseTrim=3, plotRange = 3,decRound=2,textloc=[30.86, .9], phaselim = [-120,120]):
    
    frequency=frequency/10**6
    r=r/r.max()
    idx = returnNearestValue(r, r.max())
    rmax =  r[idx]  
    fmax= frequency[idx]
    sigma=1.0/(2*Q)
    out, pars = lmDDOFit(frequency,r,[fmax,rmax*2*sigma,sigma])
    
    f0=returnNearestValue(r,pars[0])
    phase=np.deg2rad(phase)
    outp,parsp,x = lmDDOPhaseFit(frequency,phase,pars,phaseTrim)
    
    ax4.plot(frequency,r,sns.xkcd_rgb["medium green"],ls='', marker='o')
    ax4.plot(frequency,out.best_fit, sns.xkcd_rgb["very dark blue"],linewidth=2)
    ax4t=ax4.twinx()
    ax4t.plot(frequency,np.rad2deg(phase),sns.xkcd_rgb["medium blue"],ls='', marker='o')
    ax4t.plot(x,np.rad2deg(outp.best_fit), sns.xkcd_rgb["very dark blue"],linewidth=2)
    
    
    dF=frequency[f0]-frequency[f0-1]
    sigmaIndex=int(pars[2]*2*np.pi*pars[0]/dF)
    Min=f0-plotRange*sigmaIndex
    Max =f0+plotRange*sigmaIndex
    
    
    ax4.set_xlim([frequency[Min],frequency[Max]])
    ax4.set_xticks([round(frequency[Min], decRound),pars[0],round(frequency[Max],decRound)])
    ax4t.set_ylim(phaselim)
    ax4t.set_yticks(np.linspace(phaselim[0],phaselim[1],3))
    ax4.set_yticks([0,.5,1])
    ax4t.set_ylabel('Phase (degrees)')
    ax4.set_ylabel('Amplitude (a.u.)')
    ax4.set_xlabel('Freqeuncy (MHz)')
    ax4.set_xlabel('Freqeuncy (MHz)')
    ax4.axvline(pars[0], color = sns.xkcd_rgb["very dark blue"], ls = '--', linewidth = 2)
    ax4.text(textloc[0],textloc[1], r'Q = ' + str(int(round(pars[3],0))), fontsize=15)


    return out,pars,parsp


def makePubQualHeatMap(data, ax, fig, dataType='r',cutoff=.1,phasevm=[-180,180], blue=[2.4,.2],bgcolor='#5E5E5E',savePath='',sbar=0, loc = 'top'):
#    labeldic={'Green X':'Probe X','Green Y':'Probe Y','Blue X':'Pump X','Blue Y':'Pump Y'}
    labeldic={'Green X':'Probe Amplitude (a.u)','Blue X':'Pump Amplitude (a.u)'}
    labeldicp={'Green X':'Probe Phase ($\degree$)','Blue X':'Pump Phase ($\degree$)'}
    bg = col.ListedColormap([bgcolor])
    blueMap = sns.cubehelix_palette(start=blue[0], rot=blue[1],as_cmap=True)
    cm = hsluv_anglemap
    if len(data['Blue X'].unique())==1:
        X = 'Green X'
        Y = 'Green Y'
        cmap = 'YlGn'
    else:
        X = 'Blue X'
        Y = 'Blue Y'
        cmap = blueMap
    
    g= data.pivot_table(columns=X, index=Y, values = 'r').sort_index(axis=0, ascending=False)
    mask=g>cutoff
    data.phase[data.r<cutoff]=0
    data.r[data.r<cutoff]=0
    data.r=data.r
    
    if dataType=='r':
        p=makeXYHeatmap(data,X,Y,'',ax,cm=bg,vm=[0,100],cbar=False ,\
                      cbar_kws={'label': 'Amplitude (a.u.)',"ticks":[0.00,1]},mask=mask,\
                        linewidths=0.0, rasterized=True)

        mask = ~mask
        
        divider = make_axes_locatable(ax)
        if loc == 'top':
            cax = divider.append_axes('top', size='5%', pad=0.05)           
            p=makeXYHeatmap(data,X,Y,'',ax,cm=cmap\
                          ,vm=[0,1],cbar_kws={'orientation':'horizontal','label': labeldic[X],"ticks":[0.00,1]},\
                            mask=mask,linewidths=0.0, rasterized=True,cbar_ax=cax)
            cax.xaxis.set_label_position('top')
            cax.xaxis.set_ticks_position('top')
        if loc == 'bottom':
            cax = divider.append_axes('bottom', size='5%', pad=0.05)           
            p=makeXYHeatmap(data,X,Y,'',ax,cm=cmap\
                          ,vm=[0,1],cbar_kws={'orientation':'horizontal','label': labeldic[X],"ticks":[0.00,1]},\
                            mask=mask,linewidths=0.0, rasterized=True,cbar_ax=cax)
            cax.xaxis.set_label_position('bottom')
            cax.xaxis.set_ticks_position('bottom')      


    if dataType == 'phase':

        p=makeXYHeatmap(data,X,Y,'',ax,mask=mask,\
                        plot='phase',cm=bg,vm=phasevm,cbar=False\
                      ,linewidths=0.0, rasterized=True)
        mask = ~mask
#        p=makeXYHeatmap(data,X,Y,'',ax,mask=mask,plot='phase',\
#                      cm='RdBu',vm=phasevm,cbar_kws={'label': 'Phase',"ticks":np.linspace(phasevm[0],phasevm[1],2)}\
#                      ,linewidths=0.0, rasterized=True)
        divider = make_axes_locatable(ax)
        if loc == 'top':
            cax = divider.append_axes('top', size='5%', pad=0.05)           
            p=makeXYHeatmap(data,X,Y,'',ax,mask=mask,plot='phase',\
                      cm=cm,vm=phasevm\
                      ,linewidths=0.0, rasterized=True,cbar_kws={'orientation':'horizontal','label': labeldicp[X],"ticks":np.linspace(.8*phasevm[0],.8*phasevm[1],3)}\
                        ,cbar=True,cbar_ax=cax)
            cax.xaxis.set_label_position('top')
            cax.xaxis.set_ticks_position('top')
            cbar = p.collections[1].colorbar
            labels = [str(int(i)) for i in np.linspace(phasevm[0],phasevm[1],3)]
            cbar.set_ticklabels(labels)
            
        if loc == 'bottom':
            cax = divider.append_axes('bottom', size='5%', pad=0.05)           
            p=makeXYHeatmap(data,X,Y,'',ax,mask=mask,plot='phase',\
                      cm=cm,vm=phasevm\
                      ,linewidths=0.0, rasterized=True,cbar_kws={'orientation':'horizontal','label': labeldicp[X],"ticks":np.linspace(.8*phasevm[0],.8*phasevm[1],3)}\
                        ,cbar=True,cbar_ax=cax)
            cax.xaxis.set_label_position('bottom')
            cax.xaxis.set_ticks_position('bottom')
            cbar = p.collections[1].colorbar
            labels = [str(int(i)) for i in np.linspace(phasevm[0],phasevm[1],3)]
            cbar.set_ticklabels(labels)
 
    sqlen = data[X].max()-data[X].min()
    axmi = ax.get_xlim()[1]-ax.get_xlim()[0]
    barlen = axmi/sqlen*sbar
    bary=.05*axmi
    barwid=.05*axmi
    barx=axmi-barlen-.05*axmi
    
    
    
    currentAxis = ax
    currentAxis.add_patch(Rectangle((barx, bary), barlen, barwid, facecolor="black"))

    ax.set_xticks([])
    ax.set_yticks([])
#    ax.set_xlabel(labeldic[X]+' $(\mu m)$')
#    ax.set_ylabel(labeldic[Y]+' $(\mu m)$')
    ax.set_xlabel('')
    ax.set_ylabel('')
    #ax.yaxis.set_label_position("right")
    plt.tight_layout()
    ax.set_aspect("equal")
    if savePath=='':
        return fig,ax
    fig.savefig(savePath, format='pdf')
    return fig,ax


def cog_Backgate_analysis(scans):
    greenMap = copy.deepcopy(scans[0])
    wideGate = copy.deepcopy(scans[1])
    fig,axlist = plt.subplots(nrows = 2, ncols =1)
    axlist = axlist.ravel()
    
    greenMap.data.r=greenMap.data.r/np.max(greenMap.data.r)
    greenMap.data.phase=np.rad2deg(greenMap.data.phase)
    
    makePubQualHeatMap(greenMap.data, axlist[0], fig, dataType='r',cutoff=.1,phasevm=[-180,180], blue=[2.4,.2],bgcolor='#5E5E5E',savePath='',sbar=1, loc = 'top')
    makePubQualHeatMap(greenMap.data, axlist[1], fig, dataType='phase',cutoff=.1,phasevm=[-180,180], blue=[2.4,.2],bgcolor='#5E5E5E',savePath='',sbar=1, loc = 'bottom')
    plt.tight_layout()
    
    fig,axlist = plt.subplots(nrows = 2, ncols =1, figsize = (5,5))
    axlist = axlist.ravel()
    
    wideGate.data.r=np.log10(wideGate.data.r)*20
    wideGate.data.phase=np.rad2deg(wideGate.data.phase)
    
    makeXYHeatmap2(wideGate.data,'frequency','Keithley Voltage', '',axlist[0],vm=[], cm='OrRd',plot='r',rasterized = True, cbar_kws ={'label':'Amplitude (dB)'})
    axlist[0].set_xlabel('Frequency (MHz)')
    axlist[0].set_ylabel('Keithley Voltage (V)')

    makeXYHeatmap2(wideGate.data,'frequency','Keithley Voltage', '',axlist[1],vm=[], cm=hsluv_anglemap,plot='phase',rasterized = True, cbar_kws ={'label':'Phase ($\degree$)'})
    axlist[1].set_xlabel('Frequency (MHz)')
    axlist[1].set_ylabel('Keithley Voltage (V)')

    plt.tight_layout()
    

def cog_fittedPlots(scans, soverN = 2, Q=1000 , plot = False, current_palette =sns.color_palette()):
    wideGate = scans
    wide = fitFRLine(wideGate.data, Q=Q, sweep = 'Keithley Voltage',ctr_range = 1.3, soverN=soverN, edgeTrim=10, peakFilter = 4, phaseTrim=2, fitPhase = False, plot=plot)

    fig, axlist = plt.subplots(nrows =3, ncols = 2, figsize = (10,10))
    axlist = axlist.ravel()
    axlist[1].plot(wide['Keithley Voltage'],wide['Q'], marker = 'o',ls='', color = current_palette[0])
    axlist[1].set_xlabel('Keithley Voltage (V)')
    axlist[1].set_ylabel('Q')
#    axlist[0].set_ylim([0,400])
    
    axlist[3].plot(wide['Keithley Voltage'],wide['f0'], marker = 'o', color = current_palette[1])
    axlist[3].set_xlabel('Keithley Voltage (V)')
    axlist[3].set_ylabel('f0 (MHz)')
    
    axlist[5].plot(wide['Keithley Voltage'],wide['A'], marker = 'o', color = current_palette[2])
    axlist[5].set_xlabel('Keithley Voltage (V)')
    axlist[5].set_ylabel('Amplitude (a.u.)')   
    
    axlist[0].errorbar(wide['Keithley Voltage'],wide['Q'],yerr = wide['Qerr'] ,marker = 'o',ls='', color = current_palette[0])
    axlist[0].set_xlabel('Keithley Voltage (V)')
    axlist[0].set_ylabel('Q')
#    axlist[0].set_xlim([wide['Keithley Voltage'][center]-5,wide['Keithley Voltage'][center]+5])
    
    axlist[2].plot(wide['Keithley Voltage'],wide['f0'], marker = 'o', color = current_palette[1])
    axlist[2].set_xlabel('Keithley Voltage (V)')
    axlist[2].set_ylabel('f0 (MHz)')
#    axlist[2].set_xlim([wide['Keithley Voltage'][center]-5,wide['Keithley Voltage'][center]+5])
    
    
    axlist[4].plot(wide['Keithley Voltage'],wide['A'], marker = 'o', color = current_palette[2])
    axlist[4].set_xlabel('Keithley Voltage (V)')
    axlist[4].set_ylabel('Amplitude (a.u.)')
#    axlist[4].set_xlim([wide['Keithley Voltage'][center]-5,wide['Keithley Voltage'][center]+5])

    plt.tight_layout()
    
    return fig,axlist, out


def fitSweep(data, sweep_param, soverN = 2, Q=1000 , plot = False, current_palette =sns.color_palette(), plotFits=True,time = 5):
    out = fitFRLine(data, Q=Q, sweep = sweep_param,ctr_range = 1.3, soverN=soverN, edgeTrim=10, peakFilter = 4, phaseTrim=2, fitPhase = False, plot=plot,time = time)

    if plotFits == True:
        fig, axlist = plt.subplots(nrows =3, ncols = 1, figsize = (5,10))
        mk = (out.Qerr/out.Q)<.2
        axlist = axlist.ravel()
        axlist[0].errorbar(out[sweep_param][mk],out['Q'][mk],yerr =out['Qerr'][mk] , marker = 'o',ls='', color = current_palette[0])
#        axlist[0].errorbar(out[sweep_param],out['Q'][~mk],yerr =out['Qerr'][~mk] , marker = 'o',ls='', color = 'k')
        axlist[0].set_xlabel(sweep_param)
        axlist[0].set_ylabel('Q')
    #    axlist[0].set_ylim([0,400])
        
        axlist[1].errorbar(out[sweep_param][mk],out['f0'][mk],yerr =out['f0err'][mk], marker = 'o', color = current_palette[1])
        axlist[1].set_xlabel(sweep_param)
        axlist[1].set_ylabel('f0 (MHz)')
        
        axlist[2].plot(out[sweep_param][mk],out['A'][mk], marker = 'o', color = current_palette[2])
        axlist[2].set_xlabel(sweep_param)
        axlist[2].set_ylabel('Amplitude (a.u.)')   
        
        plt.tight_layout()
    
    return out

def pubgenerateBGHeatmaps(gscan,bscan,axlist,plot='x',cmg='YlGn',cmb='PuBu',*args, **kwargs):
#make plots of 
    gx=gscan
    bx=bscan
    gUF=gx.data['frequency'].unique()
    bUF=bx.data['frequency'].unique()
#make a figure and add subplots
#==============================================================================
#     fig1 = plt.figure(1, figsize=(10,10))
#     ax1 = fig1.add_subplot(221, aspect='equal')
#     ax2 = fig1.add_subplot(222, aspect='equal')
#     ax3 = fig1.add_subplot(223, aspect='equal')
#     ax4 = fig1.add_subplot(224, aspect='equal')
#==============================================================================
#make a heatmap for each map
    makeXYHeatmap2(gx.data,'Green X','Green Y','$f_0$ = ' +str(gUF.min())+', '+'r',axis = axlist[0], cm='YlGn',plot='r',*args, **kwargs)
    
    makeXYHeatmap2(bx.data,'Blue X','Blue Y', '$f_0$ = ' +str(bUF.min())+', '+'r',axis = axlist[1],  cm='PuBu',plot='r',*args, **kwargs)

#Takes the position of the other color laser during a scan and plots it on the position scan of that color laser. This allows orientation.
#The yellow is the position of the blue while the green scans while the red is the position of the green during the blue scan. The 
#convertXYToAxis function takes the position in scan space and converts it to the position on the heatmap axis (which is by # of points)
    x,y=convertXYToAxis(gx.data['Blue X'].unique()[0],gx.data['Blue Y'].unique()[0],axlist[1])
    axlist[1].scatter(x,y,marker='*', s=300, color='yellow')
    x,y=convertXYToAxis(bx.data['Green X'].unique()[0],bx.data['Green Y'].unique()[0],axlist[0])
    axlist[0].scatter(x,y,marker='*', s=300, color='cyan')
#Repeats this but for the value plot. Could be x,y, or phase
    makeXYHeatmap(gx.data,'Green X','Green Y','$f_0$ = ' +str(gUF.min())+', '+plot,axis = axlist[2], cm='RdBu',plot=plot)
    makeXYHeatmap(bx.data,'Blue X','Blue Y', '$f_0$ = ' +str(bUF.min())+', '+plot,axis = axlist[3],  cm='RdBu',plot=plot)
    x,y=convertXYToAxis(gx.data['Blue X'].unique()[0],gx.data['Blue Y'].unique()[0],axlist[3])
    axlist[3].scatter(x,y,marker='*', s=300, color='yellow')
    x,y=convertXYToAxis(bx.data['Green X'].unique()[0],bx.data['Green Y'].unique()[0],axlist[2])
    axlist[2].scatter(x,y,marker='*', s=300, color='cyan')
    

    axlist[0].set_xlabel('Green X $(\mu m)$')
    axlist[0].set_ylabel('Green Y $(\mu m)$')
    axlist[1].set_xlabel('Blue X $(\mu m)$')
    axlist[1].set_ylabel('Blue Y $(\mu m)$')    
    axlist[2].set_xlabel('Green X $(\mu m)$')
    axlist[2].set_ylabel('Green Y $(\mu m)$')
    axlist[3].set_xlabel('Blue X $(\mu m)$')
    
    
    axlist[3].set_ylabel('Blue Y $(\mu m)$')     

    
    for i in axlist:
        i.set_title('')
    for tick in axlist[3].get_xticklabels():
        tick.set_rotation(90)
    plt.tight_layout()
    return axlist

def addLaserPosition(to_plot,to_plot_onto, ax, *args,**kwargs):
    if len(to_plot['Blue X'].unique())==1:
        
        X = 'Blue X'
        Y = 'Blue Y'

    else:
        X = 'Green X'
        Y = 'Green Y'
    print X,Y
#Adds the position of the lasers ontop of an given already made 2x2 amp phase XY heatmap

    xm=[to_plot_onto[X].min(),to_plot_onto[X].max()]
    ym=[to_plot_onto[Y].min(),to_plot_onto[Y].max()]
    x,y=convertXYToAxis2(to_plot[X].unique()[0],to_plot[Y].unique()[0],ax,xm,ym)
    print x , y
    ax.scatter(x,y,*args,**kwargs)
def convertXYToAxis2(x,y,ax,xm,ym):
# Takes a real xy point and a heatmap and converts the XY to heatmap coordinates
    axmi=ax.get_xlim()[0]
    axmx=ax.get_xlim()[1]
    axl=axmx-axmi
    dmi = xm[0]
    dmx = xm[1]
    dml=dmx-dmi
    xp=axl*((x-dmi)/dml)+axmi
    axmi=ax.get_ylim()[0]
    axmx=ax.get_ylim()[1]
    axl=axmx-axmi
    dmi = ym[0]
    dmx = ym[1]
    dml=dmx-dmi
    yp=axl*((y-dmi)/dml)+axmi
    return xp,yp

greencolor = rgbTupToHex(current_palette[2])
bluecolor = rgbTupToHex(current_palette[4])

sweep_parameters = {'Keithley DC':'Keithley Voltage','DC Offset':'DC Offset','timestamp':'timestamp','Drive Amplitude':'Output 1 Amplitude'}

def show_figure(fig):

    # create a dummy figure and use its
    # manager to display "fig"

    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)

def addToDic(dic,key,data):
    print key
    if key == '':
        print dic.keys()
        dic[str(len(dic.keys()))]=data
        print str(len(dic.keys()))
    else:
        print 'a'
        dic[key]=data
def saveClass(path, Class):
    with open(path, "wb") as file_:
        cpk.dump(Class, file_, -1)
def loadClass(path):
    Class = cpk.load(open(path, "rb", -1))
    return Class
def setArbAmp(data):
    data.data.r=data.data.r/np.max(data.data.r)
    return data

def setftoMHz(data):
    data.data.frequency=data.data.frequency/10**6
    return data

def setDeg(data):
    data.data.phase=np.rad2deg(data.data.phase)
    return data
def overQvG(V, alpha, Q0,V0):
    Q = (1.0/(Q0))+(alpha*(V-V0)**2)
    return Q
from lmfit import Parameters


from lmfit import Parameters
def fitQvG(Vg, Q, alpha, Q0,V0,vary = True):
    
    mod = Model(overQvG)
    params = Parameters()
    params.add('alpha', value=alpha,min = 0)
    params.add('Q0', value=Q0, min = 5,max = 30000)
    params.add('V0', value=V0,vary = vary)
    result = mod.fit(Q, params,V=Vg)
    return result
def ampLin(x,m,o,vmin):
    y = m*np.abs(x-vmin)+o
    return y
def fitAmpLin(x,y,m = .0004,o = .00025, vmin = .25):
    mod = Model(ampLin)
    params = Parameters()
    params.add('m', value=m)
    params.add('o', value=o)
    params.add('vmin', value=vmin,min = -.75,max = .75)
    res = mod.fit(y, params, x = x,nan_policy = 'omit')
    return res

def AmpQuad(V, alpha, a0,V0):
    y = (a0+alpha*(V-V0)**2)
    return y
def fitAmpQuad(Vg, Q,vary = True,vm = -10, vx = 10):
    
    mod = Model(AmpQuad)
    params = Parameters()
    params.add('alpha', value=.1,min = 0)
    params.add('a0', value=1*10**(-6), min = 0)
    params.add('V0', value=0,min = vm,max = vx, vary = vary)
    result = mod.fit(Q, params,V=Vg, nan_policy='omit')
    return result

class NEMS_device():
#NEMS device is a class meant to encapsulate all data taken on an individual device.
    def __init__(self, location, device_type, quad = None,device = None, impath = None):
        self.quad = quad
        self.loc = location
        self.device = device
        self.device_type=device_type
        self.probeMaps = {}
        self.pumpMaps = {}
        self.spectrograms = {}
        self.impath = impath
        
    def addProbeMap(self,path,normalize = True, key = ''):
        curr_scan = scan(path)
        if normalize ==True:
            curr_scan=setArbAmp(curr_scan)
            curr_scan=setDeg(curr_scan)
        addToDic(self.probeMaps,key,Map(key,curr_scan))

    def addPumpMap(self,path,normalize = True, key = ''):
        curr_scan = scan(path)
        if normalize ==True:
            curr_scan=setArbAmp(curr_scan)
            curr_scan=setDeg(curr_scan)
        addToDic(self.pumpMaps,key,Map(key,curr_scan))
                             
    def addSpectrogram(self,path,key = '',normalize = True,sweep_parameter='Keithley DC'):
        curr_scan = scan(path)
        if normalize == True:
            curr_scan=setArbAmp(curr_scan)
            curr_scan=setftoMHz(curr_scan)
        addToDic(self.spectrograms,key,Spectrogram(key,curr_scan,sweep_parameter))
    def plotPumpProbe(self, probe_name = '0', pump_name = '0'):
            figPP, axlistPP = plt.subplots(nrows=2, ncols=2, figsize = (3.5*.8,5*.8))
            axlistPP = axlistPP.ravel()
            pump = True
            probe = True
            if len(self.pumpMaps.keys())==0:
                pump = False
            if len(self.probeMaps.keys())==0: 
                probe = False
            with sns.plotting_context("paper", font_scale =.8):
                if pump:
                    bscan = self.pumpMaps[pump_name].data
                    makePubQualHeatMap(bscan.data,axlistPP[1],figPP,'r',sbar=1)
                    makePubQualHeatMap(bscan.data,axlistPP[3],figPP,'phase',sbar=1,loc = 'bottom')
                if probe:
                    gscan = self.probeMaps[probe_name].data
                    makePubQualHeatMap(bscan.data,axlistPP[0],self.figPP,'r',sbar=1)
                    makePubQualHeatMap(bscan.data,axlistPP[2],self.figPP,'phase',sbar=1,loc = 'bottom')
                if (pump & probe):
                    addLaserPosition(bscan.data,gscan.data,axlistPP[0], marker='o', color=bluecolor,s=50)
                    addLaserPosition(bscan.data,gscan.data,axlistPP[2], marker='o', color=bluecolor,s=50)                
                    addLaserPosition(gscan.data,bscan.data,axlistPP[1], marker='o', color=greencolor,s=50)
                    addLaserPosition(gscan.data,bscan.data,axlistPP[3], marker='o', color=greencolor,s=50)
            plt.tight_layout()
            self.figPP.subplots_adjust(wspace=.05, hspace=-.4)
    def clearDic(self,name):
        self.spectrograms.pop(name)

class Map():
    def __init__(self, name, scan):
        self.name = name
        self.meta = scan.meta
        self.data = scan.data
        self.date = self.meta.Date
     
class Spectrogram():
    def __init__(self, name, scan, sweep_parameter):
        self.name = name
        self.data = scan.data
        self.data['rlog']=np.log10(self.data.r)*20
        self.meta = scan.meta
        self.sweep_parameter = sweep_parameters[sweep_parameter]
        self.date = self.meta.Date
    def plotSpectrogram(self):
        fig, axlist = plt.subplots(nrows=2, ncols=1, figsize = (5,5))
        makeXYHeatmap2(self.data,'frequency',self.sweep_parameter, '',axlist[0],vm=[], cm='OrRd',plot='rlog',rasterized = True, cbar_kws ={'label':'Amplitude (dB)'})
        axlist[0].set_xlabel('Frequency (MHz)')
        axlist[0].set_ylabel(self.sweep_parameter)
        makeXYHeatmap2(self.data,'frequency',self.sweep_parameter, '',axlist[1],vm=[], cm=hsluv_anglemap,plot='phase',rasterized = True, cbar_kws ={'label':'Phase ($\degree$)'})
        axlist[1].set_xlabel('Frequency (MHz)')
        axlist[1].set_ylabel(self.sweep_parameter)
        plt.tight_layout()
    def fitSpectrogram(self,Q=1000,soverN = 8, plot = False, plotFits=True,time = 5):
        self.FRFit = fitSweep(self.data, self.sweep_parameter, soverN = soverN, Q=Q , plot = plot\
                 ,plotFits=plotFits,time=time)
    def fitFRFitToModel(self, model = 'Barton', pars=[1,1,1,0], plot = False, fitRange = []):
        Ampfit = fitAmpLin(self.FRFit['Keithley Voltage'], self.FRFit['A'])
        v0 = Ampfit.params['vmin'].value
        if len(fitRange)!=0:
            mask = (self.FRFit[self.sweep_parameter]>=fitRange[0]) & (self.FRFit[self.sweep_parameter]<=fitRange[1])
        else:
            mask = (self.FRFit[self.sweep_parameter]>=self.FRFit[self.sweep_parameter].min()) & (self.FRFit[self.sweep_parameter]<=self.FRFit[self.sweep_parameter].max())
        if model == 'Barton':
#model from the barton paper with c1,c2,c3 parameters
            self.out = fitF0(self.FRFit[mask],pars[0],pars[1],pars[2], V0i=v0,sweep_param = self.sweep_parameter)
        if model == 'linBarton':
#model from the barton paper with c1,c2,c3 parameters and an additional linear term to account to heating changes to the resonance
            self.out = fitF0lin(self.FRFit[mask],pars[0],pars[1],pars[2], V0i=v0,sweep_param = self.sweep_parameter)
        if model == 'par':
#model from the barton paper with c1,c2,c3 parameters and an additional linear term to account to heating changes to the resonance
            self.out = fitF0par(self.FRFit[mask],pars[0],pars[1],pars[2], V0i=v0,sweep_param = self.sweep_parameter)
   
        if plot == True:
            plt.figure()
            self.out.plot_fit()
        self.fitParams = self.out.params.valuesdict()
    def fitdeAlba(self,Y0,rho0,s0_0,R,ep,d,m=0,V0=0,f0=0,varyV = True, plot = False, fitRange = [],\
                  varyrho=True,varym = True,varysig = True,varyE = True,lockf = True):
        if len(fitRange)!=0:
            mask = (self.FRFit[self.sweep_parameter]>=fitRange[0]) & (self.FRFit[self.sweep_parameter]<=fitRange[1])
            
        else:
            mask = (self.FRFit[self.sweep_parameter]>=self.FRFit[self.sweep_parameter].min()) & (self.FRFit[self.sweep_parameter]<=self.FRFit[self.sweep_parameter].max())
        self.out = fitDeAlbaU01(self.FRFit[mask][self.sweep_parameter],self.FRFit[mask]['f0']*10**6,Y0,rho0,s0_0,R,ep,d,\
                                m=m,V0=V0,varyV = varyV,f0 = f0,varym = varym,varyrho = varyrho,varysig = varysig,varyE = varyE, lockf = lockf)
   
        if plot == True:
            plt.figure()
            self.out.plot_fit()
        self.fitParams = self.out.params.valuesdict()
    def fitQ2(self,alpha = .0001,Q0 = 5000, Qerrmax = .2,plot = False, vary = True, v0 = None, useA = True):
        mk = self.FRFit.Qerr/self.FRFit.Q<Qerrmax
        Vg = self.FRFit['Keithley Voltage'][mk]
        Q = self.FRFit.Q[mk]
        if v0 == None:
            v0 = self.fitParams['V0']
        else:
            v0 = V0
        self.res = fitQvG(Vg,1/Q,alpha,Q0,v0, vary = vary)
        
        if plot ==True:
            plt.figure()
            plt.errorbar(Vg, 1.0/Q,yerr = self.FRFit.Qerr[mk]/Q**2,  ls = '', marker = 'o')
            plt.plot(Vg,self.res.best_fit, ls = '-')
    def fitQ(self,alpha = .0001,Q0 = 5000, Qerrmax = .2,plot = False, vary = True,plotA = False, vm=-1.5,vx=1.5, v0 = None):
            mk = self.FRFit.Qerr/self.FRFit.Q<Qerrmax
            mkA = (self.FRFit['Keithley Voltage']>vm) & (self.FRFit['Keithley Voltage']<vx)
            Vg = self.FRFit['Keithley Voltage'][mk&mkA]
            Q = self.FRFit.Q[mk&mkA]
            v0rmin = self.data.groupby('Keithley Voltage').mean().r.idxmin()            
            if v0 == None:
                v0 = v0rmin
            else:
                v0 = v0
           

#            self.Ampfit = fitAmpQuad(self.FRFit['Keithley Voltage'][mkA], self.FRFit['A'][mkA],vm = -10,vx = 10)
#            v0 = self.Ampfit.params['V0'].value
#            self.Av0 = v0
            self.res = fitQvG(Vg,1/Q,alpha,Q0,v0, vary = vary)
            if plotA == True:
                plt.figure()
                self.Ampfit.plot_fit()
            if plot ==True:
                plt.figure()
                plt.axvline(v0rmin)
                plt.errorbar(Vg, 1.0/Q,yerr = self.FRFit.Qerr[mk&mkA]/Q**2,  ls = '', marker = 'o')
                plt.plot(Vg,self.res.best_fit, ls = '-')
    def getParsFromFRFit(self,Qerrmax = .05):
        mk = self.FRFit.Qerr/self.FRFit.Q\
        <Qerrmax
        self.Qmean = self.FRFit.Q[mk].mean()
        self.Qstd = self.FRFit.Q[mk].std()
        
        self.Qfit = self.res.params['Q0'].value
        self.Qerr = self.res.params['Q0'].stderr
        
        self.f0 = self.FRFit.f0[mk].mean()
        self.f0std = self.FRFit.f0[mk].std()
#        print FittedDevices2[i].spectrograms['0'].res.params['Q0'].init
        self.Qdiff = self.Qfit - self.Qmean
        try:
            self.Damp = np.pi*2*self.f0/self.Qfit
#        print f0, Qfit, Qerr,Qfit
        
            self.Damperr = np.pi*2*np.abs(self.f0/self.Qfit)*np.abs(self.Qerr/self.Qfit)
        except TypeError as e:
            print e

def simpleAnalyzeSaveDevice(device,Type,map1,map2,gate,savePath,base):
    try:
        dev = loadClass(savePath)
    except IOError:
        dev = NEMS_device(device,Type)
    dev.addProbeMap(scanPath(base,map1))
    dev.addPumpMap(scanPath(base,map2))
    dev.addSpectrogram(scanPath(base,gate))
    dev.spectrograms['0'].fitSpectrogram(Q=5000)
    dev.spectrograms['0'].fitFRFitToModel(pars=[314.5,0,-.15,-.3])
    saveClass(savePath,dev)
    return dev