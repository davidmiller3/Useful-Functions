
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 05 10:28:22 2017

@author: David
"""
from helperFunctions import *

def lmDDOFit(xdata, ydata, params, ctr_range = 1.2, amp_range = 100 , sig_range= 100, weightexponential = 0):    
    
    
    x = xdata
    y = ydata
#Define a linear model and a Damped Oscillator Model    
    line_mod = LinearModel(prefix='line_')
    ddo_mod = DampedOscillatorModel(prefix='ddo_')
#Initial Pars for Linear Model
    pars =  line_mod.make_params(intercept=0, slope=0)
    pars['line_intercept'].set(0, vary=True)
    pars['line_slope'].set(0, vary=False)
#Extend param list to use multiple peaks. Currently unused.
    peaks=[]
#Add fit parameters, Center, Amplitude, and Sigma
    for i in range(0, len(params)/3):
        peaks.append(DampedOscillatorModel(prefix='ddo'+str(i)+'_'))
        pars.update(peaks[i].make_params())
        ctr=params[3*i]
        amp=params[3*i+1]
        sig=params[3*i+2]
        pars['ddo'+str(i)+'_center'].set(ctr, min=ctr/ctr_range, max=ctr*ctr_range)
        pars['ddo'+str(i)+'_amplitude'].set(amp,min=amp/amp_range, max=amp*amp_range)
        pars['ddo'+str(i)+'_sigma'].set(sig, min=sig/sig_range, max=sig*sig_range)
#Create full model. Add linear model and all peaks
    mod=line_mod
    for i in xrange(len(peaks)):
        mod=mod+peaks[i]
#Initialize fit
    init = mod.eval(pars, x=x)
#Do the fit. The weight exponential can weight the points porportional to the
#amplitude of y point. In this way, points on peak can be given more weight.     
    out=mod.fit(y, pars,x=x)
#Get the fit parameters
    fittedsigma = out.params['ddo0_sigma'].value
    fittedAmp = out.params['ddo0_amplitude'].value
    fittedCenter = out.params['ddo0_center'].value
    fittedIntercept = out.params['line_intercept'].value
    fittedSlope = out.params['line_slope'].value
    fittedQ=1/(2*fittedsigma)
#Returns the output fit as well as an array of the fit parameters
    """Returns output fit as will as list of important fitting parameters"""
    return out, [fittedCenter, fittedAmp, fittedsigma, fittedQ, fittedIntercept, fittedSlope]
    
def lmDDOPhaseFit(xdata, ydata, params, f0_range = 1.2, Q_range = 3, phaseTrim=3):    
    f0= params[0]
    Q=1/(2*params[2])

    x = xdata
    y = ydata    
    f0idx = returnNearestValue(x, f0)

    dF=x[f0idx]-x[f0idx-1]
    sigmaIndex=int(params[2]*2*np.pi*params[0]/dF)

    phaseMin=f0idx-phaseTrim*sigmaIndex
    phaseMax = f0idx+phaseTrim*sigmaIndex
#Define a linear model and a Damped Oscillator Model    
#    ddophase_mod = ExpressionModel('off + m*x- arctan(1/Q * 1/(f0/x - x/f0))-')


    ddophase_mod = Model(DDOPhase)
#Initial Pars for Linear Model
    pars =  ddophase_mod.make_params(b=0, m=0, f0=f0, Q=Q)

#Add fit parameters, Center, Amplitude, and Sigma
    pars['f0'].set(min=f0/f0_range, max=f0*f0_range)
    pars['Q'].set(min=Q/Q_range, max=Q*Q_range)
#Create full model. Add linear model and all peaks
#Initialize fit
    init = ddophase_mod.eval(pars, x=x[phaseMin:phaseMax])
#Do the fit. The weight exponential can weight the points porportional to the
#amplitude of y point. In this way, points on peak can be given more weight.     
    out=ddophase_mod.fit(y[phaseMin:phaseMax], pars,x=x[phaseMin:phaseMax])
#Get the fit parameters
    fittedf0= out.params['f0'].value
    fittedQ = out.params['Q'].value
    fittedm = out.params['m'].value
    fittedb = out.params['b'].value
#Returns the output fit as well as an array of the fit parameters
    """Returns output fit as will as list of important fitting parameters"""
    return out, [fittedf0, fittedm, fittedb, fittedQ],x[phaseMin:phaseMax]
    
def fitSinglePeak(FRScan,Q=300):
    idx = returnNearestValue(FRScan.r, FRScan.r.max())
    rmax =  FRScan.r[idx]  
    fmax= FRScan.frequency[idx]
    sigma=1.0/(2*Q)
    out, pars = lmDDOFit(FRScan.frequency,FRScan.r,[fmax,rmax*2*sigma,sigma])
    return out,pars
def fitSinglePeak2(x,y,Q=300):
    idx = returnNearestValue(y, y.max())
    rmax =  y[idx]  
    fmax= x[idx]
    sigma=1.0/(2*Q)
    out, pars = lmDDOFit(x,y,[fmax,rmax*2*sigma,sigma])
    return out,pars
def DDOPhase(x, f0, Q, m , b):
    y= (-np.arctan2(f0 * x/Q,f0**2-x**2))+b
    return y


def fvsgate(Vg,c1,c2,c3,V0):
    f =         np.sqrt(6*c3*(Vg-V0)**2 + 2**0.6666666666666666*\
          (2*c1**3 + 27*c2*(Vg-V0)**4 + 3*np.sqrt(3)*np.sqrt(np.sqrt(c2**2)*(Vg-V0)**4*(4*c1**3 + 27*np.sqrt(c2**2)*(Vg-V0)**4)))**\
           0.3333333333333333 + 2*c1*\
          (1 + c1/\
             (c1**3 + (3*(9*c2*(Vg-V0)**4 + np.sqrt(3)*np.sqrt(np.sqrt(c2**2)*(Vg-V0)**4*(4*c1**3 + 27*np.sqrt(c2**2)*(Vg-V0)**4))))/\
                 2.)**0.3333333333333333))/np.sqrt(6)
    return f


     
def fvsgatelin(Vg,c1,c2,c3,V0,m,b):
    f = m*  Vg+b+      np.sqrt(6*c3*(Vg-V0)**2 + 2**0.6666666666666666*\
          (2*c1**3 + 27*c2*(Vg-V0)**4 + 3*np.sqrt(3)*np.sqrt(np.sqrt(c2**2)*(Vg-V0)**4*(4*c1**3 + 27*np.sqrt(c2**2)*(Vg-V0)**4)))**\
           0.3333333333333333 + 2*c1*\
          (1 + c1/\
             (c1**3 + (3*(9*c2*(Vg-V0)**4 + np.sqrt(3)*np.sqrt(np.sqrt(c2**2)*(Vg-V0)**4*(4*c1**3 + 27*np.sqrt(c2**2)*(Vg-V0)**4))))/\
                 2.)**0.3333333333333333))/np.sqrt(6)
    return f


def fitF0(data,c1i,c2i,c3i,V0i=0,sweep_param='Keithley Voltage'):

    idx = data.f0.idxmin()
    f0sweep = Model(fvsgate,missing='drop', nan_policy='omit')
    f0sweep.set_param_hint('c2', min = .0001)
    f0sweep.set_param_hint('c3')
    result = f0sweep.fit(data.f0, Vg=data[sweep_param], c1=c1i,c2=c2i,c3=c3i,V0=V0i,method = 'nelder',nan_policy = 'omit')

    return result

def fitF0lin(data,c1i,c2i,c3i,V0i=0,sweep_param='Keithley Voltage'):

    idx = data.f0.idxmin()
    f0sweep = Model(fvsgatelin,missing='drop', nan_policy='omit')
    f0sweep.set_param_hint('c2', min = 0)
    f0sweep.set_param_hint('c3')
    print f0sweep.param_hints
    result = f0sweep.fit(data.f0, Vg=data[sweep_param], c1=c1i,c2=c2i,c3=c3i,V0=V0i,m=0,b=0,method = 'nelder',nan_policy = 'omit')

    return result

def fitF0par(data,c1i,c2i,c3i,V0i=0,sweep_param='Keithley Voltage'):
    freq = data.f0
    idx = freq.idxmin()
    f0sweep = Model(fvsgate,missing='drop', nan_policy='omit')
    f0sweep.set_param_hint('c1', min = .0001)
    f0sweep.set_param_hint('c2', min = .0001)
    f0sweep.set_param_hint('c3', value = 0, vary = False)
    result = f0sweep.fit(freq, Vg=data[sweep_param], c1=c1i,c2=c2i,c3=c3i,V0=V0i,method = 'nelder',nan_policy = 'omit')

    return result



def addFitParamsFromFitF0(params,df, label=''):
    newDFRow = pd.DataFrame({'c1' : params['c1'].value, 'c2' : params['c2'].value, 'c3' :params['c3'].value, \
                'V0':params['V0'].value, 'name':label}, index=[df.index[-1]+1])
    return pd.concat([df,newDFRow])






from scipy.optimize import fsolve
def fvsgateD2P(Vg,sigma0,f0,E,V0,R,ep0,d,m):

    nu = 0.16
#    rho = rho * 7.4* 10**-7
    rho = 2.4048**2 * sigma0/(R**2 * (2*np.pi*f0)**2)
    print rho/(7.4* 10**-7)
    def sigmaroot(sigma):
        return sigma - sigma0 - (E * R**2 * ep0**2 * (Vg-V0)**4)/((1-nu**2) * 128 * d**4 * sigma**2)
    
    sigma = fsolve(sigmaroot, np.full(len(Vg),sigma0))

    f = np.sqrt((2*np.pi*f0)**2\
                - ep0 * (Vg-V0)**2/(d**3 * rho)\
                + 0.1316 * E *ep0**2 * (Vg-V0)**4/((1-nu**2) * d**4 * rho * sigma**2))/(2*np.pi)+m*(Vg-V0)

    return f

def fvsgateD(Vg,sigma0,rho,E,V0,R,ep0,d,m):
    nu = 0.16
    rho = rho * 7.4* 10**-7
    def sigmaroot(sigma):
        return sigma - sigma0 - (E * R**2 * ep0**2 * (Vg-V0)**4)/((1-nu**2) * 128.0 * d**4 * sigma**2)
    try:
        sigma = fsolve(sigmaroot, np.full(len(Vg),sigma0))
    except TypeError:
        sigma = fsolve(sigmaroot, sigma0)
    f = np.sqrt(2.4048**2 * sigma0/(R**2 * rho)\
                - ep0 * (Vg-V0)**2/(d**3 * rho)\
                + 0.1316 * E *ep0**2 * (Vg-V0)**4/((1-nu**2) * d**4 * rho * sigma**2))/(2*np.pi)+m*(Vg-V0)
    return f


def fitDeAlbaU01(x,y,Y0,rho0,s0_0,R,ep,d,m= 0, V0 = 0,f0=0,varyV = True,varym = True, varyrho = True,varysig=True,varyE = True,lockf = True):
    mod = Model(fvsgateD)
    params = Parameters()
    params.add('E', value = Y0,min = 0,max = 3000, vary = varyE)
    params.add('rho', value = rho0,min = 1, max = 20, vary = varyrho)
    params.add('sigma0', value = s0_0, min = 0, max = 3, vary = varysig)
    params.add('R', value = R, vary = False)
    params.add('ep0', value = ep, vary = False)
    params.add('d', value = d, vary = False)
    params.add('m',value = m, vary = varym)
    params.add('V0', value = V0, vary =varyV)
#    params.add('f0', value = f0,vary = False)
    res = mod.fit(np.array(y),Vg=np.array(x), params = params,nan_policy = 'omit')
    return res

def fitDeAlbaU01OG(x,y,Y0,rho0,s0_0,R,ep,d,m= 0, V0 = 0,f0=0,varyV = True,varym = True, varyrho = True,varysig=True,varyE = True,lockf = True):
    mod = Model(fvsgateD)
    params = Parameters()
    params.add('E', value = Y0,min = 0,max = 800, vary = varyE)
    params.add('rho', value = rho0,min = 1, max = 20, vary = varyrho)
    params.add('sigma0', value = s0_0, min = 0, max = 3, vary = varysig)
    params.add('R', value = R, vary = False)
    params.add('ep0', value = ep, vary = False)
    params.add('d', value = d, vary = False)
    params.add('m',value = m, vary = varym)
    params.add('V0', value = V0, vary =varyV)
#    params.add('f0', value = f0,vary = False)

    params.add('f0',value = f0, expr='sqrt(2.4048**2 * sigma0/(R**2 * rho))/(2*pi)',vary = False)

    res = mod.fit(np.array(y),Vg=np.array(x), params = params,nan_policy = 'omit')
    return res