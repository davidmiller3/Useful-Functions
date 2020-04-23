# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:55:29 2017

@author: David
"""

from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid import make_axes_locatable
import matplotlib.colors as col
import hsluv
from matplotlib.figure import Figure
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
import pandas as pd
import math, sys
from scipy import signal
import peak_Detect as peak
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
import os
from lmfit.models import GaussianModel, ExpressionModel, ExponentialModel, LorentzianModel, LinearModel, DampedOscillatorModel
from lmfit import CompositeModel, Model, Parameters
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import scipy.special as sps
import platform
import errno
import dill as cpk
import copy
sns.set_style("whitegrid", {'axes.grid' : False})
import array
from matplotlib import pyplot as plt
from matplotlib import colors
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['text.usetex'] = 'False' 

current_palette = sns.color_palette()
"""
if os.name == 'posix':
    pathToMeasurmentFile = '/Users/davidmiller/Dropbox/Photothermal Control of Graphene/Data/MeasurmentData'
if os.name == 'nt':
    pathToMeasurmentFile = 'Z:\\Group\\Projects\\Optical drive control over graphene drumheads\\MeasurmentData'
"""    
class scan():
    def __init__(self, pathToDataFolder):
        self.data = pd.read_csv(pathToDataFolder, skiprows=2)
        self.meta = pd.read_table(pathToDataFolder, nrows=1, delimiter = ',')
#        try: 
#            self.meta['Blue X']
#            pass
#        except ValueError:
#            print 'e'
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
def returnNearestValue(array, value):
    # Retruns the index of the closest value in the array to the entered value
    return np.argmin(np.abs(array-value))

    
def scanPath(basepath,scanpath, demod = 'Demod1'):
    d=demod+'.csv'
    return os.path.join(basepath,scanpath,d)
    

def readZIScan(base, folderName, addLoc = [100,100,100,100]):
    data = pd.read_csv(os.path.join(base,folderName, 'Data.csv'), header=None, names=['frequency','f1','r','phase','b','a0','a1','r1','r2'])

    new=np.empty(len(data.r))
    new.fill(addLoc[0])
    data['Green X'] = pd.Series(new, index=data.index)
    new=np.empty(len(data.r))
    new.fill(addLoc[1])
    data['Green Y'] = pd.Series(new, index=data.index)
    new=np.empty(len(data.r))
    new.fill(addLoc[2])
    data['Blue X'] = pd.Series(new, index=data.index)
    new=np.empty(len(data.r))
    new.fill(addLoc[3])
    data['Blue Y'] = pd.Series(new, index=data.index)
            
    new=np.empty(len(data.r))
    new.fill(0)
    data['DC Offset'] = pd.Series(new, index=data.index)

    new=np.empty(len(data.r))
    new.fill(0)
    data['Blue Drive'] = pd.Series(new, index=data.index)
 
    new=np.empty(len(data.r))
    new.fill(0)
    data['Output 1 Amplitude'] = pd.Series(new, index=data.index)
 

    data['x'] = data.r*np.cos(data.phase)
    data['Y'] = data.r*np.sin(data.phase)
    return data    




def writeGwyddion(baseLoc, fileName, gdata,bdata, plot1='r',plot2='phase'):
    folder = os.path.join(baseLoc,fileName)
    make_sure_path_exists(folder)
    laser = 'Blue'
    X='Blue X'
    Y='Blue Y'
    df = bdata
    cpath = os.path.join(folder,'Blue_'+plot1+'.gsf')
    write_gsf(cpath,df[plot1],len(df[X].unique()),len(df[Y].unique()),xreal =df[X].unique().max()-df[X].unique().min()\
         ,yreal =df[Y].unique().max()-df[Y].unique().min(), title = laser + plot1)
    
    cpath = os.path.join(folder,'Blue_'+plot2+'.gsf')
    write_gsf(cpath,df[plot2],len(df[X].unique()),len(df[Y].unique()),xreal =df[X].unique().max()-df[X].unique().min()\
         ,yreal =df[Y].unique().max()-df[Y].unique().min(),title = laser + plot2)
    laser = 'Green'
    X='Green X'
    Y='Green Y'
    df=gdata
    cpath = os.path.join(folder,'Green_'+plot1+'.gsf')
    write_gsf(cpath,df[plot1],len(df[X].unique()),len(df[Y].unique()),xreal =df[X].unique().max()-df[X].unique().min()\
         ,yreal =df[Y].unique().max()-df[Y].unique().min(),title = laser + plot1)    
    
    cpath = os.path.join(folder,'Green_'+plot2+'.gsf')
    write_gsf(cpath,df[plot2],len(df[X].unique()),len(df[Y].unique()),xreal =df[X].unique().max()-df[X].unique().min()\
         ,yreal =df[Y].unique().max()-df[Y].unique().min(),title = laser + plot2)    
    
def write_gsf(filename, imagedata, xres, yres,
              xreal=None, yreal=None, xyunits=None, zunits=None, title=None):
    """Write a Gwyddion GSF file.

    filename -- Name of the output file.
    imagedata -- Image data.
    xres -- Horizontal image resolution.
    yres -- Vertical image resolution.
    xreal -- Horizontal physical dimension (optional).
    yreal -- Vertical physical dimension (optional).
    xyunits -- Unit of physical dimensions (optional).
    zunits -- Unit of values (optional).
    title -- Image title (optional).

    Image data may be passed as any listable object that can be used to form
    a floating point array.array().  This includes tuples, lists, arrays,
    numpy arrays and other stuff.
    """
    data = array.array('f', imagedata)
    if len(data) != xres*yres:
        raise ValueError, "imagedata does not have xres*yres items"
    isinf = math.isinf
    isnan = math.isnan
    for z in data:
        if isinf(z) or isnan(z):
            raise ValueError, "GSF files may not contain NaNs and infinities"
    if sys.byteorder == 'big':
        data.byteswap()
    header = ['Gwyddion Simple Field 1.0']
    header.append('XRes = %u' % xres)
    header.append('YRes = %u' % yres)
    if xreal is not None:
        header.append('XReal = %.12g' % xreal)
    if yreal is not None:
        header.append('YReal = %.12g' % yreal)
    if xyunits is not None:
        header.append('XYUnits = %s' % xyunits.encode('utf-8'))
    if zunits is not None:
        header.append('ZUnits = %s' % zunits.encode('utf-8'))
    if title is not None:
        header.append('Title = %s' % title.encode('utf-8'))

    header = ''.join(x + '\n' for x in header)
    l = len(header)
    sentinel = bytearray()
    for j in range(4 - l % 4):
        sentinel.append(0)

    fh = file(filename, 'wb')
    fh.write(header)
    fh.write(sentinel)
    fh.write(data)
    fh.close()

def gwydxyz(scan, savepath):
    cols = scan.data.columns.values.tolist()
    numpts = len(scan.data.r)
    dic={'NChannels':4,'NPoints':numpts,'XYUnits':'m','ZUnits1':'V','ZUnits2':'radians','ZUnits3':'V','ZUnits4':'V'\
        ,'Title1':'r','Title2':'phase','Title3':'x','Title4':'y'}
    if len(scan.data['Blue X'].unique())==1:
        X='Green X'
        Y='Green Y'
    else:
        X='Blue X'
        Y='Blue Y'
    xdata=scan.data[X]
    ydata=scan.data[Y]
    df = pd.DataFrame({X:xdata,Y:ydata,'r':scan.data['r'],'phase':scan.data['phase'],'x':scan.data['x'],'y':scan.data['y']})
    df = df[[X,Y,'r','phase','x','y']]
    if sys.byteorder == 'big':
        data.byteswap()
    xyunits='m'
    ZUnits1='V'
    ZUnits2='radians'
    ZUnits3='V'
    ZUnits4='V'
    Title1 = 'r'
    Title2 = 'phase'
    Title3='x'
    Title4='y'
    header = ['Gwyddion XYZ Field 1.0']
    header.append('NChannels = %u' % 4)
    header.append('NPoints = %u' % numpts) 
    fh = file(savepath, 'wb')
    fh.write(header)
    fh.write(sentinel)
#    for i in xrange(len(df.as_matrix())):
    data =  df.as_matrix()
    data =  np.array(df.as_matrix(),'d')
    
    data=np.ascontiguousarray(data)
    fh.write(data)
    fh.close()


def wrapPhase(phase):
    return (( phase + np.pi) % (2 * np.pi ) - np.pi)

def subtractPhase(data, cal, wrapunwrap = True):
    p = np.interp(data.frequency, cal.frequency,cal.phase)
    phase = data.phase-p
    phase = np.deg2rad(phase)
    if wrapunwrap == True:
        return np.unwrap(wrapPhase(phase),discont=3.141592653589793*1.5)
    else:
        return phase
        
def wrapIndPoints(array):
    for i in xrange(len(array)):
        while array[i]>np.pi:
           array[i] = array[i]-np.pi
        while array[i] < -np.pi:
            array[i]=array[i]+np.pi

    return array

def cut(data,t, points, interp = 0):
#    x0, y0 = 1, 1 # These are in _pixel_ coordinates!!
#    x1, y1 = 39, 39
    x0=points[0]
    x1=points[1]
    y0=points[2]
    y1=points[3]
    if len(data['Blue X'].unique())==1:
        X='Green X'
        Y='Green Y'
    else:
        X='Blue X'
        Y='Blue Y'
    length = int(np.hypot(x1-x0, y1-y0))
    x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
    g= data.pivot_table(columns=[X], index=[Y], values = t).sort_index(axis=0, ascending=False)
    p = g.as_matrix()
    p = p.transpose()
    si = [p[int(x[n]),-int(y[n])] for n in range(len(x))]
    se = []
    for i in xrange(interp):
        print 'i'
        x0=points[0]+1
        x1=points[1]+1
        y0=points[2]+1
        y1=points[3]+1

        x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
        s = [p[int(x[n]),-int(y[n])] for n in range(len(x))]
        se.append(s)
        x0=points[0]-1
        x1=points[1]-1
        y0=points[2]-1
        y1=points[3]-1
        x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
        s = [p[int(x[n]),-int(y[n])] for n in range(len(x))]
        se.append(s)
    ltot = data[X].max()-data[X].min()
    ntot = len(data[X].unique())
    totlength = (ltot/ntot)*length
    return si,totlength,se


##### generate custom colormaps
def make_segmented_cmap(): 
    white = '#ffffff'
    black = '#000000'
    red = '#ff0000'
    blue = '#0000ff'
    anglemap = col.LinearSegmentedColormap.from_list(
        'anglemap', [black, red, white, blue, black], N=256, gamma=1)
    return anglemap

def make_anglemap( N = 256, use_hpl = True ):
    h = np.ones(N) # hue
    h[:N//2] = 11.6 # red 
    h[N//2:] = 258.6 # blue
    s = 100 # saturation
    l = np.linspace(0, 100, N//2) # luminosity
    l = np.hstack( (l,l[::-1] ) )

    colorlist = np.zeros((N,3))
    for ii in range(N):
        if use_hpl:
            colorlist[ii,:] = hsluv.hpluv_to_rgb( (h[ii], s, l[ii]) )
        else:
            colorlist[ii,:] = hsluv.hsluv_to_rgb( (h[ii], s, l[ii]) )
    colorlist[colorlist > 1] = 1 # correct numeric errors
    colorlist[colorlist < 0] = 0 
    return col.ListedColormap( colorlist )

N = 256
segmented_cmap = make_segmented_cmap()
flat_huslmap = col.ListedColormap(sns.color_palette('husl',N))
hsluv_anglemap = make_anglemap( use_hpl = False )
hpluv_anglemap = make_anglemap( use_hpl = True )


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


def fixFR(data, sweepparam,freq = 'frequency', tol = .2):
    # Takes FR data and looks for missing data.
    stop = False
    d=data.data
    while stop == False:
        d=d.sort_values(by=[sweepparam, freq])
        un = d[sweepparam].unique()
        for i in xrange(len(un)-2):
            ii=i+1
            if np.abs((un[ii+1]-un[ii]) - (un[ii]-un[ii-1]))<tol:
                pass
            else:
                np.abs((un[ii+1]-un[ii]) - (un[ii]-un[ii-1]))
                df = pd.DataFrame({sweepparam : np.full(len(d[d[sweepparam]==un[ii]][freq]),((un[ii+1]+un[ii])/2)),freq:d[d[sweepparam]==un[ii]][freq].as_matrix(),'rlog':np.full(len(d[d[sweepparam]==un[ii]][freq]),d.rlog.min())},index =[range(len(d),len(d)+len(d[d[sweepparam]==un[ii]][freq]))])

                d = d.append(df)
                break
            if (i+3) == len(un):      
                stop = True
    return d


def rgbTupToHex(rgb):
        return rgb_to_hex(rgb[0],rgb[1],rgb[2])
def rgb_to_hex(red, green, blue):
        """Return color as #rrggbb for the given color values."""
        return '#%02x%02x%02x' % (int(red*256), int(green*256), int(blue*256))
# Add to least    
from functools import wraps
import errno
import os
import signal
import time

class TimeoutError(Exception):
    pass

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)
try:        
    with timeout(seconds=3):
#        time.sleep(4)
        print 'good'
except TimeoutError:
    print 'timedout'