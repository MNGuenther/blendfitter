# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:46:15 2017

@author:
Maximilian N. Guenther
Battcock Centre for Experimental Astrophysics,
Cavendish Laboratory,
JJ Thomson Avenue
Cambridge CB3 0HE
Email: mg719@cam.ac.uk
"""

from __future__ import print_function, division, absolute_import

#::: modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize, fmin
#from scipy.stats import norm
#from scipy import integrate
#import imageio
from matplotlib import animation
from scipy.interpolate import CubicSpline
 
#::: blendfitter modules
from . import ccf_models




###########################################################################
#::: plot settings
########################################################################### 
try:
    import seaborn as sns
    sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    sns.set_context(rc={'lines.markeredgewidth': 1})
except ImportError:
    pass
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'medium',
         'axes.titlesize':'medium',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium',
         'font.size': 18.}
pylab.rcParams.update(params)
########################################################################### 



def inverse_gauss(x,offset,amp,mu,std):
    return 1. - offset - amp*np.exp(-(x-mu)**2/(2.*std**2))

 
    
def inverse_gauss_no_offset(x,amp,mu,std):
    return 1. - amp*np.exp(-0.5*((x-mu)/std)**2)




#::: fit an inverse gaussian to determine amp, mu and fwhm
def fit_inverse_gauss(x,y,guess=None):
    '''
    popt = [offset,amp,mu,std]
    
    initial values have to be close to the real values
    use weighted arithmetic mean and std
    from strpeter's answer in 
    https://stackoverflow.com/questions/19206332/gaussian-fit-for-python
    (don't look at the other answers, they're wrong!)
    '''
    if guess is None:
        offset = 0.
        yy = 1.-y
        n = np.sum(yy)
        amp = np.max(yy)
        mu = 1.*np.sum(x*yy)/n
        std = np.sqrt(1.*np.sum(yy*(x-mu)**2)/n)
        if np.isnan(std): std = 7.
    else:
        [offset,amp,mu,std] = guess
#    print 'guess', mu, std, amp, offset
    popt,pcov = curve_fit(inverse_gauss,x,y,p0=[offset,amp,mu,std], bounds=([-1, 0, -100, -100], [1, 1, 100, 100]))  
    perr = np.sqrt(np.diag(pcov))
    return popt, perr
 
    

def fit_inverse_gauss_no_offset(x,y,guess=None,bounds=([0,-100,-100],[1,100,100])):
    '''
    popt = [amp,mu,std]
    
    initial values have to be close to the real values
    use weighted arithmetic mean and std
    from strpeter's answer in 
    https://stackoverflow.com/questions/19206332/gaussian-fit-for-python
    (don't look at the other answers, they're wrong!)
    '''
    if guess is None:
        yy = 1.-y
        n = np.sum(yy)
        amp = np.max(yy)
        mu = 1.*np.sum(x*yy)/n
        std = np.sqrt(1.*np.sum(yy*(x-mu)**2)/n)
    else:
        [amp,mu,std] = guess
#    print 'guess', mu, std, amp, offset
    popt,pcov = curve_fit(inverse_gauss_no_offset,x,y,p0=[amp,mu,std], bounds=bounds)  
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


    
#::: fit an inverse gaussian to determine amp, mu and fwhm
def fit_inverse_gauss_fast(x,y,guess,bounds=([0,0,0],[1,20,20])):
    '''
    popt = [amp,mu,std]
    initial values have to be close to the real values
    '''
    popt,_ = curve_fit(inverse_gauss_no_offset,x,y,p0=guess, bounds=bounds, method='dogbox', xtol=1e-1, ftol=1e-3)  
    return popt

    
    
#::: convert the CCF to an empirical probability density function
#::: i.e. normalize to mu=0, amp=1, fwhm=1 and range=[0,1]    
def convert_CCF_to_PDF(x,y,plot=False):
   
    #::: fit an inverse gaussian to determine amp, mu and fwhm
    popt, perr = fit_inverse_gauss(x,y)
    
    #::: convert and normalize the CCF to PDF
    amp = np.min(y) #instead of popt[0] to account for non-gaussian curves
    mu = popt[1]
    fwhm = 2.35482004503*popt[2]
    PDF_x = (x-mu)/fwhm
    PDF_y = (1.-y)/(1.-amp)
    
    #::: plot, if desired
    if plot:
        fig, axes = plt.subplots(3,1,figsize=(6,8))
        axes[0].plot(x,y,'ko-',label='data')
        axes[0].plot(x,inverse_gauss(x,*popt),'r-',lw=3,label='fit')
        axes[0].set(ylabel='CCF')
        axes[1].plot(x,inverse_gauss(x,*popt)-y,'ro',label='res')
        axes[1].set(ylabel='Res.')
        axes[2].plot(PDF_x,PDF_y,'ko',label='res')
        axes[2].set(ylabel='PDF')
    
    return PDF_x, PDF_y, amp, mu, fwhm
    


def get_RV_from_ccf(x, y_total):
    #RV = x[ np.argmin(y_total) ]
    popt, perr = fit_inverse_gauss(x,y_total)
    RV = popt[2]
    return RV


    
############################
def compute_RV_and_BIS(x, ytot, short_output=False):
############################
    '''
    Numerical least squares curve fit (slow) for real data
    Fits for offset, amp, mu, std

    WARNING: 
    --------
    - this is slow (~0.1 seconds)
    
    Input:
    ------
    x : array
        the x-coordinates of the bimodal CCF (i.e. the RV values)
    ytot : array
        the y-coordinates of the bimodal CCF (i.e. the CCF values)
    short_output : bool
        what to return
    '''
    
    popt, perr = fit_inverse_gauss(x,ytot)
    offset,amp,v0,sigma = popt #[offset,amp,mu,std]
    offset_err,amp_err,v0_err,sigma_err = perr
    
    span, span_err, bisector_x, bisector_y, Contrast, FWHM =\
        subcompute_HARPS_improved(x, ytot, v0, sigma, amp, offset)

    if not short_output:
        return v0, v0_err, span, span_err, bisector_x, bisector_y, offset, Contrast, FWHM
    else:
        return v0, span, Contrast, FWHM
    
    
    
############################
def compute_RV_and_BIS_from_model(x, ytot, CCF_modelpar_i, method, short_output=False):
############################
    if method == 'numerical':
        return compute_RV_and_BIS_from_model_numerical_least_squares_fit(x, ytot, CCF_modelpar_i, short_output=short_output)
    
    elif method == 'stat':
        return compute_RV_and_BIS_from_model_stat_fit(x, ytot, CCF_modelpar_i, short_output=short_output)
    
    elif method == 'algebraic':
        return compute_RV_and_BIS_from_model_algebraic_least_squares_fit(x, ytot, CCF_modelpar_i, short_output=short_output)
    
    elif method == 'fast_numerical':
        return compute_RV_and_BIS_from_model_fast_numerical_least_squares_fit(x, ytot, CCF_modelpar_i, short_output=short_output)
    
    else:
        raise ValueError("'compute_RV_and_BIS_from_model' expects 'method' to be 'numerical', 'stat', 'algebraic', or 'fast_numerical'.")
    
    
############################
def compute_RV_and_BIS_from_model_numerical_least_squares_fit(x, ytot, CCF_modelpar_i, short_output=False):
############################
    '''
    Numerical least squares curve fit (slow) for a given offset

    WARNING: 
    --------
    - this is slow (~0.1 seconds)
    - this does take the MCMC offset as given, and only fits for amp, mu, std of the Gaussian
        
    Input:
    ------
    x : array
        the x-coordinates of the bimodal CCF (i.e. the RV values)
    ytot : array
        the y-coordinates of the bimodal CCF (i.e. the CCF values)
    CCF_modelpar_i : numpy structured array
        with keys 'x','y','y0','y1','mu0','mu1','sigma0','sigma1','A0','A1','weight0','weight1','offset'
        here: contains only one row (advantage of this format over dict: one can pick a specific row but keep all the column names)
    short_output : bool
        what to return
    '''
    
    offset = CCF_modelpar_i['offset']
    
    popt, perr = fit_inverse_gauss_no_offset(x,(ytot+offset))
    amp,v0,sigma = popt #[offset,amp,mu,std]
    amp_err,v0_err,sigma_err = perr
   
    span, span_err, bisector_x, bisector_y, Contrast, FWHM =\
        subcompute_HARPS_improved(x, ytot, v0, sigma, amp, offset)
    
    if not short_output:
        return v0, v0_err, span, span_err, bisector_x, bisector_y, offset, Contrast, FWHM
    else:
        return v0, span, Contrast, FWHM



############################
def compute_RV_and_BIS_from_model_stat_fit(x, ytot, CCF_modelpar_i, short_output=False):
############################
    '''
    Statistical fit for a given offset (ultra fast, but systematically different from a numerical curve fit!)
    
    WARNING:
    -------
    - this uses a PDF model, while the HARPS fit is usually done with a curve_fit model
      i.e. there are systematic differences in the methods and their results (on the <<1% level)
    - this does take the MCMC offset as given, and only fits for amp, mu, std of the Gaussian
    
    Input:
    ------
    x : array
        the x-coordinates of the bimodal CCF (i.e. the RV values)
    ytot : array
        the y-coordinates of the bimodal CCF (i.e. the CCF values)
    CCF_modelpar_i : numpy structured array
        with keys 'x','y','y0','y1','mu0','mu1','sigma0','sigma1','A0','A1','weight0','weight1','offset'
        here: contains only one row (advantage of this format over dict: one can pick a specific row but keep all the column names)
    short_output : bool
        what to return
    '''
        
    offset = CCF_modelpar_i['offset']
#    print 'weight0', CCF_modelpar_i['weight0'], 'mu0', CCF_modelpar_i['mu0']
#    print 'weight1', CCF_modelpar_i['weight1'], 'mu1', CCF_modelpar_i['mu1']
    v0 = CCF_modelpar_i['weight0']*CCF_modelpar_i['mu0'] + CCF_modelpar_i['weight1']*CCF_modelpar_i['mu1']
    v0_err = 0.
    buf0 = CCF_modelpar_i['weight0'] * CCF_modelpar_i['sigma0']**2
    buf1 = CCF_modelpar_i['weight1'] * CCF_modelpar_i['sigma1']**2
    buf2 = CCF_modelpar_i['weight0'] * CCF_modelpar_i['weight1'] * (CCF_modelpar_i['mu0'] - CCF_modelpar_i['mu1'])**2
    sigma = np.sqrt( buf0 + buf1 + buf2 )
    #sigma formula from https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians
    y,_,_ = ccf_models.bimodal_ccf(v0,\
                                 CCF_modelpar_i['A0'],CCF_modelpar_i['mu0'],CCF_modelpar_i['fwhm0'],\
                                 CCF_modelpar_i['A1'],CCF_modelpar_i['mu1'],CCF_modelpar_i['fwhm1'],\
                                 CCF_modelpar_i['offset'],'normal',data=None)
    amp = 1.-offset-y   
#    print '-----'
#    print v0, sigma, amp, offset
   
    span, span_err, bisector_x, bisector_y, Contrast, FWHM =\
        subcompute_HARPS_improved(x, ytot, v0, sigma, amp, offset)
    
    if not short_output:
        return v0, v0_err, span, span_err, bisector_x, bisector_y, offset, Contrast, FWHM
    else:
        return v0, span, Contrast, FWHM




############################
def compute_RV_and_BIS_from_model_algebraic_least_squares_fit(x, ytot, CCF_modelpar_i, short_output=False):
############################
    '''
    Algebraic least squares curve fit for a given offset (ultra fast and mimicking the numerical curve fit)

    WARNING: 
    --------
    - use only for noise-free models, this is sensitive to outliers on the wings;
    - this does take the MCMC offset as given, and only fits for amp, mu, std of the Gaussian
        
    Input:
    ------
    x : array
        the x-coordinates of the bimodal CCF (i.e. the RV values)
    ytot : array
        the y-coordinates of the bimodal CCF (i.e. the CCF values)
    CCF_modelpar_i : numpy structured array
        with keys 'x','y','y0','y1','mu0','mu1','sigma0','sigma1','A0','A1','weight0','weight1','offset'
        here: contains only one row (advantage of this format over dict: one can pick a specific row but keep all the column names)
    short_output : bool
        what to return
    '''
    
    offset = CCF_modelpar_i['offset']
    
    y = np.log( 1.-offset-ytot )
    
#    w = np.ones_like(x)
    w = (1.-offset-ytot)**2 
    w /= np.max( w )
    
    n = np.sum(w)
#    n = len(x)
    sum_x = np.sum(w*x)
    sum_x2 = np.sum(w*x**2)
    sum_x3 = np.sum(w*x**3)
    sum_x4 = np.sum(w*x**4)
    sum_y = np.sum(w*y)
    sum_yx = np.sum(w*y*x)
    sum_yx2 = np.sum(w*y*x**2)
    
    A = np.array( [[n, sum_x, sum_x2],\
                   [sum_x, sum_x2, sum_x3],\
                   [sum_x2, sum_x3, sum_x4]] )
    
    Y = np.array( [sum_y, sum_yx, sum_yx2] )
    Y.shape = (3,1) #column vector
    
    X = np.dot( np.dot( np.linalg.inv( np.dot( A.T, A ) ), A.T ), Y )
    
    a = X[0][0]
    b = X[1][0]
    c = X[2][0]
    
    amp = np.exp( a - b**2/(4.*c) )
    mu = - b/(2.*c)
    std = np.sqrt( - 1./(2.*c) )
    
    v0 = mu
    v0_err = 0.
    sigma = std
   
    span, span_err, bisector_x, bisector_y, Contrast, FWHM =\
        subcompute_HARPS_improved(x, ytot, v0, sigma, amp, offset)
    
    if not short_output:
        return v0, v0_err, span, span_err, bisector_x, bisector_y, offset, Contrast, FWHM
    else:
        return v0, span, Contrast, FWHM
    
    
    

############################
def compute_RV_and_BIS_from_model_fast_numerical_least_squares_fit(x, ytot, CCF_modelpar_i, short_output=False):
############################
    '''
    Fast numerical least sqaures curve fit for a given offset
    using an initial guess derived from an algebraic least squares curve fit

    WARNING: 
    --------
    - this uses lower precision / higher convergence tolerance than the usual setting
    - this does take the MCMC offset as given, and only fits for amp, mu, std of the Gaussian
        
    Input:
    ------
    x : array
        the x-coordinates of the bimodal CCF (i.e. the RV values)
    ytot : array
        the y-coordinates of the bimodal CCF (i.e. the CCF values)
    CCF_modelpar_i : numpy structured array
        with keys 'x','y','y0','y1','mu0','mu1','sigma0','sigma1','A0','A1','weight0','weight1','offset'
        here: contains only one row (advantage of this format over dict: one can pick a specific row but keep all the column names)
    short_output : bool
        what to return
    '''
    
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #::: get analytical solution
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    offset = CCF_modelpar_i['offset']
    
    y = np.log( 1.-offset-ytot )
    
#    w = np.ones_like(x)
    w = (1.-offset-ytot)**2 
    w /= np.max( w )
    
    n = np.sum(w)
#    n = len(x)
    sum_x = np.sum(w*x)
    sum_x2 = np.sum(w*x**2)
    sum_x3 = np.sum(w*x**3)
    sum_x4 = np.sum(w*x**4)
    sum_y = np.sum(w*y)
    sum_yx = np.sum(w*y*x)
    sum_yx2 = np.sum(w*y*x**2)
    
    A = np.array( [[n, sum_x, sum_x2],\
                   [sum_x, sum_x2, sum_x3],\
                   [sum_x2, sum_x3, sum_x4]] )
    
    Y = np.array( [sum_y, sum_yx, sum_yx2] )
    Y.shape = (3,1) #column vector
    
    X = np.dot( np.dot( np.linalg.inv( np.dot( A.T, A ) ), A.T ), Y )
    
    a = X[0][0]
    b = X[1][0]
    c = X[2][0]
    
    amp = np.exp( a - b**2/(4.*c) )
    mu = - b/(2.*c)
    std = np.sqrt( - 1./(2.*c) )
    
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #::: fit a single Gaussian using the analytical soluton as initial guess
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #::: 0.025 s
    popt = fit_inverse_gauss_fast(x,(ytot+offset),guess=[amp,mu,std])
    amp,v0,sigma = popt #[offset,amp,mu,std]
    v0_err = 0.

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #::: do the rest
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    span, span_err, bisector_x, bisector_y, Contrast, FWHM =\
        subcompute_HARPS_improved(x, ytot, v0, sigma, amp, offset)
    
    if not short_output:
        return v0, v0_err, span, span_err, bisector_x, bisector_y, offset, Contrast, FWHM
    else:
        return v0, span, Contrast, FWHM
    
    
    

def subcompute_HARPS_improved(RV, CCF, v0, sigma, amp, offset):
    '''
    improved the HARPS DRS codes by removing all their for & while loops
    instead use numpy arrays, broadcasting and list comprehension
    '''
    norm_CCF = (1.-offset)/amp*(1.-CCF/(1.-offset))
    nstep = 100
    margin = 5
    depth = 1.*np.arange(nstep-2*margin+1)/nstep +1.*margin/nstep
   
    p = np.zeros([len(CCF),3])
#    bis_b = np.zeros(len(depth))
#    bis_r = np.zeros(len(depth))

    ind = np.where( (np.max((norm_CCF[:-1],norm_CCF[1:]), axis=0) >= depth[0]) & (np.min((norm_CCF[:-1],norm_CCF[1:]), axis=0) <= depth[-1]) )[0]
    diff_v = np.diff(RV) #= RV[i+1]-RV[i]
    v = (2*RV[:-1]+diff_v)/2. #= (RV[i+1]+RV[i]) / 2
    diff_norm_CCF = np.diff(norm_CCF) #=norm_CCF[i+1]-norm_CCF[i]
    mean_norm_CCF = (2*norm_CCF[:-1]+diff_norm_CCF)/2. #= (norm_CCF[i+1]+norm_CCF[i]) / 2
    dCCFdRV = -(v-v0)/sigma**2*np.exp(-(v-v0)**2/2/sigma**2)
    d2CCFdRV2 = ((v-v0)**2/sigma**2-1)/sigma**2*np.exp(-(v-v0)**2/2/sigma**2)
    d2RVdCCF2 = -d2CCFdRV2/dCCFdRV**3
    p[ind,2] = d2RVdCCF2[ind]/2
    p[ind,1] = (diff_v[ind]-p[ind,2]*(2*mean_norm_CCF[ind])*diff_norm_CCF[ind]) / diff_norm_CCF[ind]
    p[ind,0] = RV[ind]-p[ind,1]*norm_CCF[ind]-p[ind,2]*norm_CCF[ind]**2
      
    i_peak = np.argmax( norm_CCF ) 
    
    x = norm_CCF[:i_peak+1]
    xargsort = np.argsort(x)
    xsort = x[xargsort]
    i_b = xargsort[np.searchsorted(xsort, depth)-1]
    
    x = norm_CCF[i_peak:]
    xargsort = np.argsort(x)
    xsort = x[xargsort]
    i_r = xargsort[np.searchsorted(xsort, depth)-1]+i_peak-1
    
#    i_b = [ np.where( (norm_CCF[:i_peak+1]-d)<0 )[0][-1] for d in depth ]
#    i_r = [ np.where( (norm_CCF[i_peak:]-d)<0 )[0][0]+i_peak-1 for d in depth ]
    
    bis_b = p[i_b,0]+p[i_b,1]*depth+p[i_b,2]*depth**2
    bis_r = p[i_r,0]+p[i_r,1]*depth+p[i_r,2]*depth**2
    
    bis = (bis_b+bis_r)/2.
   
    qq = np.greater_equal(depth,0.1)*np.less_equal(depth,0.4)
    RV_top = np.mean(np.compress(qq,bis))
    RV_top_err = np.std(np.compress(qq,bis))
    qq = np.greater_equal(depth,0.6)*np.less_equal(depth,0.9)
    RV_bottom = np.mean(np.compress(qq,bis))
    RV_bottom_err = np.std(np.compress(qq,bis))
    span = RV_top-RV_bottom
    span_err = np.sqrt(RV_top_err**2 + RV_bottom_err**2)
   
    bisector_x = bis
    bisector_y = (1.-offset)*(1.-depth*amp/(1.-offset))
   
    FWHM = 2.35482004503*sigma 
    Contrast = 100.*amp
    
    return span, span_err, bisector_x, bisector_y, Contrast, FWHM




def subcompute_HARPS_DRS3(RV, CCF, v0, v0_err, sigma, amp, offset):
    norm_CCF = (1.-offset)/amp*(1.-CCF/(1.-offset))
    nstep = 100
    margin = 5
    depth = 1.*np.arange(nstep-2*margin+1)/nstep +1.*margin/nstep
   
    p = np.zeros([len(CCF),3],'d')
    bis_b = np.zeros(len(depth),'d')
    bis_r = np.zeros(len(depth),'d')
   
    for i in range(len(CCF)-1):
       if (np.max((norm_CCF[i],norm_CCF[i+1])) >= depth[0]) & (np.min((norm_CCF[i],norm_CCF[i+1])) <= depth[-1]):
          v = (RV[i]+RV[i+1])/2.
          dCCFdRV = -(v-v0)/sigma**2*np.exp(-(v-v0)**2/2/sigma**2)
          d2CCFdRV2 = ((v-v0)**2/sigma**2-1)/sigma**2*np.exp(-(v-v0)**2/2/sigma**2)
          d2RVdCCF2 = -d2CCFdRV2/dCCFdRV**3
          p[i,2] = d2RVdCCF2/2
          p[i,1] = (RV[i+1]-RV[i]-p[i,2]*(norm_CCF[i+1]**2-norm_CCF[i]**2))/(norm_CCF[i+1]-norm_CCF[i])
          p[i,0] = RV[i]-p[i,1]*norm_CCF[i]-p[i,2]*norm_CCF[i]**2
   
    for j in range(len(depth)):
       i_b = norm_CCF.argmax()
       while (norm_CCF[i_b] > depth[j]) & (i_b > 1): i_b = i_b-1
       i_r = norm_CCF.argmax()
       while (norm_CCF[i_r+1] > depth[j]) & (i_r < len(CCF)-2): i_r = i_r+1
       bis_b[j] = p[i_b,0]+p[i_b,1]*depth[j]+p[i_b,2]*depth[j]**2
       bis_r[j] = p[i_r,0]+p[i_r,1]*depth[j]+p[i_r,2]*depth[j]**2
   
    bis = (bis_b+bis_r)/2.
   
    for i in range(len(bis)):
       if not np.isfinite(bis[i]): bis = np.zeros(len(depth),'d')
   
    qq = np.greater_equal(depth,0.1)*np.less_equal(depth,0.4)
    RV_top = np.mean(np.compress(qq,bis))
    RV_top_err = np.std(np.compress(qq,bis))
    qq = np.greater_equal(depth,0.6)*np.less_equal(depth,0.9)
    RV_bottom = np.mean(np.compress(qq,bis))
    RV_bottom_err = np.std(np.compress(qq,bis))
    span = RV_top-RV_bottom
    span_err = np.sqrt(RV_top_err**2 + RV_bottom_err**2)
   
    bisector_x = bis
    bisector_y = (1.-offset)*(1.-depth*amp/(1.-offset))
   
    FWHM = 2.35482004503*sigma 
    Contrast = 100.*amp
    
    return v0, v0_err, span, span_err, bisector_x, bisector_y, offset, Contrast, FWHM



   
#def bisector(x, y_total, RV, nsep=101):
#    '''
#    Parameters
#    ----------
#
#    Notes
#    -----
#    From Queloz+2001:
#    v_t (top) := 10-40%, measured from top (CCF intensity==1)
#    v_b (bottom) := 55-90%
#    '''
#    ccf_min = np.nanmin(y_total)
#    slices = np.linspace(ccf_min, 1., nsep)
#    yrange = 1. - ccf_min
#    ind_t = np.where( (slices<=(1.-0.1*yrange)) & (slices>=(1.-0.4*yrange)) )[0] #[11,12,13,14,15,16,17,18]
#    ind_b = np.where( (slices<=(1.-0.55*yrange)) & (slices>=(1.-0.9*yrange)) )[0] #[2,3,4,5,6,7,8]
#    bisector_x, bisector_y = [], []
#    
#    y_left = y_total[ x<RV ][::-1] #to be in increasing order for np.interp
#    y_right = y_total[ x>RV ]
#    x_left = x[ x<RV ] #to be in increasing order for np.interp
#    x_right = x[ x>RV ]
#
#    x_left_interp = np.interp(slices, y_left, x_left)
#    x_right_interp = np.interp(slices, y_right, x_right)
#   
#    for i in range(len(slices)):
#        bisector_y.append( slices[i] )
#        bisector_x.append( np.mean( [x_left_interp[i], x_right_interp[i]] ) )
#    
#    bisector_x = np.array(bisector_x)
#    bisector_y = np.array(bisector_y)
#    
#    v_t = np.mean(bisector_x[ind_t])
#    v_b = np.mean(bisector_x[ind_b])
#    BIS = v_t - v_b
#    
#    return bisector_x, bisector_y, BIS




def bisector(x, y_total, RV, nsep=201, interp='2cubic', limits=[0.1,0.4,0.55,0.9]):
    '''
    Parameters
    ----------
    interp : str
        '1cubic': use measured points on the left side and interpolate on the right side with cubic spline
        '1linear': use measured points on the left side and interpolate on the right side linearly
        '2cubic': use slices/markers and interpolate on the left and right side at these slices with cubic spline
        '2linear': use slices/markers and interpolate on the left and right side at these slices linearly
        
    Notes
    -----
    From Queloz+2001:
    v_t (top) := 10-40%, measured from top (CCF intensity==1)
    v_b (bottom) := 55-90%
    '''
    #::: initialize bisector lists
    bisector_x, bisector_y = [], []
    
    #::: mark left and right sides; remove data at <10% and >90%
    ccf_min = np.nanmin(y_total)
    yrange = 1. - ccf_min
    ind_left = np.where( (x<RV) & (y_total<=(1.-0.1*yrange)) & (y_total>=(1.-0.9*yrange)) )[0]
    ind_right = np.where( (x>RV) & (y_total<=(1.-0.1*yrange)) & (y_total>=(1.-0.9*yrange)) )[0]
    y_left = y_total[ ind_left ][::-1] #to be in increasing order for np.interp
    y_right = y_total[ ind_right ]
    x_left = x[ ind_left ][::-1] #to be in increasing order for np.interp
    x_right = x[ ind_right ]
    
    if interp=='2cubic':
        slices = np.linspace(ccf_min, 1., nsep)
        cs = CubicSpline(y_left, x_left)
        x_left_interp = cs(slices)
        cs = CubicSpline(y_right, x_right)
        x_right_interp = cs(slices) 
        for i in range(len(slices)):
            bisector_y.append( slices[i] )
            bisector_x.append( np.mean( [x_left_interp[i], x_right_interp[i]] ) )
    elif interp=='2linear':
        slices = np.linspace(ccf_min, 1., nsep)
        x_left_interp = np.interp(slices, y_left, x_left)
        x_right_interp = np.interp(slices, y_right, x_right)
        for i in range(len(slices)):
            bisector_y.append( slices[i] )
            bisector_x.append( np.mean( [x_left_interp[i], x_right_interp[i]] ) )
    elif interp=='1cubic':
        cs = CubicSpline(y_right,x_right)
        x_right_interp = cs(y_left)
        for i,_ in enumerate(y_left):
            bisector_y.append( y_left[i] )
            bisector_x.append( np.mean( [x_left[i], x_right_interp[i]] ) )
    elif interp=='1linear':
        x_right_interp = np.interp(y_left, y_right, x_right)
        for i,_ in enumerate(y_left):
            bisector_y.append( y_left[i] )
            bisector_x.append( np.mean( [x_left[i], x_right_interp[i]] ) )
   
    bisector_x = np.array(bisector_x)
    bisector_y = np.array(bisector_y)
    ind_out_of_range = np.where( (bisector_y>(1.-limits[0]*yrange)) | (bisector_y<(1.-limits[3]*yrange)) )[0]
    bisector_x[ind_out_of_range] = np.nan
    bisector_y[ind_out_of_range] = np.nan
#    bisector_x[ 0 ] = np.nan
#    bisector_x[ -1 ] = np.nan
#    bisector_y[ 0 ] = np.nan
#    bisector_y[ -1 ] = np.nan

    ind_t = np.where( (bisector_y<=(1.-limits[0]*yrange)) & (bisector_y>=(1.-limits[1]*yrange)) )[0] #[11,12,13,14,15,16,17,18]
    ind_b = np.where( (bisector_y<=(1.-limits[2]*yrange)) & (bisector_y>=(1.-limits[3]*yrange)) )[0] #[2,3,4,5,6,7,8]    
    v_t = np.mean(bisector_x[ind_t])
    v_b = np.mean(bisector_x[ind_b])
    BIS = v_t - v_b
    BIS_err = np.nanstd(bisector_x)
    
    return bisector_x, bisector_y, BIS, BIS_err
   
   


#
#def bisector_from_left(x, y_total, RV, interp='linear'):
#    '''
#    Parameters
#    ----------
#    interp : str
#        'cubic', 'linear'
#        
#    Notes
#    -----
#    From Queloz+2001:
#    v_t (top) := 10-40%, measured from top (CCF intensity==1)
#    v_b (bottom) := 55-90%
#    '''
#    #::: initialize bisector lists
#    bisector_x, bisector_y = [], []
#    
#    #::: mark left and right sides; remove data at <10% and >90%
#    ccf_min = np.nanmin(y_total)
#    yrange = 1. - ccf_min
#    ind_left = np.where( (x<RV) & (y_total<=(1.-0.1*yrange)) & (y_total>=(1.-0.9*yrange)) )[0]
#    ind_right = np.where( (x>RV) & (y_total<=(1.-0.1*yrange)) & (y_total>=(1.-0.9*yrange)) )[0]
#    y_left = y_total[ ind_left ][::-1] #to be in increasing order for np.interp
#    y_right = y_total[ ind_right ]
#    x_left = x[ ind_left ][::-1] #to be in increasing order for np.interp
#    x_right = x[ ind_right ]
#    
#    #::: interpolate on right side
#    if interp=='cubic':
#        cs = CubicSpline(y_right,x_right)
#        x_right_interp = cs(y_left)
#    elif interp=='linear':
#        x_right_interp = np.interp(y_left, y_right, x_right)
#        
#    for i,_ in enumerate(y_left):
#        bisector_y.append( y_left[i] )
#        bisector_x.append( np.mean( [x_left[i], x_right_interp[i]] ) )
#    
#    bisector_x = np.array(bisector_x)
#    bisector_y = np.array(bisector_y)
#    
#    ind_t = np.where( (y_left<=(1.-0.1*yrange)) & (y_left>=(1.-0.4*yrange)) )[0] #[11,12,13,14,15,16,17,18]
#    ind_b = np.where( (y_left<=(1.-0.55*yrange)) & (y_left>=(1.-0.9*yrange)) )[0] #[2,3,4,5,6,7,8]
#    v_t = np.mean(bisector_x[ind_t])
#    v_b = np.mean(bisector_x[ind_b])
#    BIS = v_t - v_b
#    
#    fig, axes = plt.subplots(1,2,sharey=True)
#    axes[0].plot(x, y_total, 'b.')
#    axes[0].plot(x_right_interp, y_left, 'r-')
#    axes[0].plot(bisector_x, bisector_y, 'k.-')
#    axes[1].plot(bisector_x, bisector_y, 'k.-')
#    axes[1].plot(bisector_x[ind_t], bisector_y[ind_t], 'y.-')
#    axes[1].plot(bisector_x[ind_b], bisector_y[ind_b], 'g.-')
#    
#    return bisector_x, bisector_y, BIS
    
    
        

class model():
    
    def __init__(self, phases, DIL, CCF_amp,
                 RVsys_0, RVK_0, fwhm_0, 
                 RVsys_1, RVK_1, fwhm_1, 
                 typ, data=None, 
                 outfnames={'gif':'test.gif'}):
        
        #::: the bright target is always Obj0
        #::: this does not mean it's the eclipsing one
        #::: hence, invert DIL in the case of a blend eclipse, to scale the amplitudes correctly            
        if DIL > 0.5: 
            DIL = 1.-DIL
              
        self.phases = phases
        self.DIL = DIL
        self.A0 = (1.-DIL) * CCF_amp
        self.RVsys_0 = RVsys_0
        self.RVK_0 = RVK_0
        self.fwhm_0 = fwhm_0
        self.A1 = CCF_amp - self.A0
        self.RVsys_1 = RVsys_1
        self.RVK_1 = RVK_1
        self.fwhm_1 = fwhm_1
        self.typ = typ
        self.data = data
        self.outfnames = outfnames
        
        self.y_total_list = []
        self.y0_list = []
        self.y1_list = []
        self.RV_list = []
        self.bisector_x_list = []
        self.bisector_y_list = []
        self.BIS_list = []
        
        self.x = np.concatenate((np.linspace(RVsys_0-3*self.fwhm_0,RVsys_0-self.fwhm_0/2.,100)[:-1], \
                                 np.linspace(RVsys_0-self.fwhm_0/2.,RVsys_0+self.fwhm_0/2.,10000+1), \
                                 np.linspace(RVsys_0+self.fwhm_0/2.,RVsys_0+3*self.fwhm_0,100)[1:]))
                        
                        
                
#    def amp_ratio_from_dil_and_A0(self,DIL,A0):
#        '''
#        with Obj0 == eclipsing object and Obj1 == blend:
#        A0 / A1 = Flux0 / Flux1
#        DIL = 1 - Flux0 / (Flux0 + Flux1) 
#        DIL = 1 - 1 / (1 + Flux1/Flux0)
#        1 / (1 - DIL) = 1 + Flux1/Flux0
#        Flux1/Flux0 = 1 / (1 - DIL) - 1
#        A1/A0 = 1 / (1 - DIL) - 1
#        
#        if system flux is given:
#        A0 / A1 = Flux0 / Flux1
#        DIL = 1 - Flux0 / (Flux0 + Flux1) 
#        Flux0 = (1 - DIL) * Flux_sys 
#        Flux1 = Flux_sys - Flux0
#        '''
#        return (1. / (1. - DIL) - 1.) * A0
        
       
       
    
#    def model_RV_and_BIS(self, RVsys_0, RVK_0, fwhm_0, 
#                         RVsys_1, RVK_1, fwhm_1):
#        pass
        
        
        
        
    def bimodal_ccf(self,x,A0,mu0,fwhm0,A1,mu1,fwhm1,typ):
        '''
        note: if fwhm gets passed instead of std, they have to be transformed via
        std = fwhm/2.35482004503
        
        typ: 'normal' or 'cauchy'
        '''
        if typ=='normal':
            std0 = fwhm0/2.35482004503
            std1 = fwhm1/2.35482004503
            y0 = 1.-A0*np.exp(-(x-mu0)**2/(2.*std0**2)) #norm.pdf(x,loc=mu0,scale=fwhm_0) 
            y1 = 1.-A1*np.exp(-(x-mu1)**2/(2.*std1**2)) #norm.pdf(x,loc=mu1,scale=fwhm_1)
            y_total = ( y0+y1-1. )
        elif typ=='cauchy':
            gamma0 = fwhm0/2.
            gamma1 = fwhm1/2.
            y0 = 1. - A0*1./( np.pi*gamma0 * (1 + ((x-mu0)/gamma0)**2 ))
            y1 = 1. - A1*1./( np.pi*gamma1 * (1 + ((x-mu1)/gamma1)**2 ))
            y_total = ( y0+y1-1. )
        elif typ=='custom':
            y0 = self.custom_ccf(x,A0,mu0,fwhm0,0)
            y1 = self.custom_ccf(x,A1,mu1,fwhm1,1)
            y_total = ( y0+y1-1. )
        return y_total,y0,y1
        
        
        
    def custom_ccf(self,x,A,mu,fwhm,obj):
        '''
        note: the passed ccf template (CCF_pdf) has to be
        converted to an empirical probability density function
        i.e. normalized to mu=0, amp=1, fwhm=1 and range=[0,1]  
        '''
        xbuf = self.data['CCF_pdf_x_'+str(obj)] * fwhm + mu
        ybuf = 1. - A * self.data['CCF_pdf_y_'+str(obj)]
        y = np.interp(x,xbuf,ybuf)
        return y
        
        
    

        
        
        
#    def get_maxima(self):
#        '''
#        for plot boundaries and BIS slope calculation;
#        mu0 at 0, mu1 at maximum or vice versa
#        '''
#        mu0_max = np.abs(self.RVsys_0)+np.abs(self.RVK_0)
#        mu1_max = np.abs(self.RVsys_1)+np.abs(self.RVK_1)
#        y_total,y0,y1 = self.bimodal_ccf(self.x, 
#                                         self.A0, mu0_max, self.std_0, 
#                                         self.A1, mu1_max, self.std_1)
#        RV_max = self.get_RV_fromccf(0.25, self.x, y_total)
#        BIS_max = self.bisector(self.x, y_total, RV_max)[-1]
#        return RV_max, BIS_max
        
        
        
#    def get_BIS_slope(self,RV_max,BIS_max):
#        BIS_slope = 1.*BIS_max / RV_max
#        return BIS_slope
        
        
            
    def get_plotlim(self):
        self.RV_center = np.median(self.RV_list)
        self.RV_ll = np.min(self.RV_list)
        self.RV_ul = np.max(self.RV_list)
        self.RV_range = self.RV_ul - self.RV_ll
        self.BIS_center = 0.
        self.BIS_ll = np.min(self.BIS_list)
        self.BIS_ul = np.max(self.BIS_list)
        self.BIS_range = self.BIS_ul - self.BIS_ll
        
        self.RV_plot_ll = np.min( [self.RV_ll, np.min(self.data['RV'])] ) - 0.1*self.RV_range
        self.RV_plot_ul = np.max( [self.RV_ul, np.max(self.data['RV'])] ) + 0.1*self.RV_range
        
        
#        return RV_center, RV_ll, RV_ul
#        pass
#        if x is None: return x
#        else: 
#            if datax is None:
#                return [scale*x,scale*x]
#            else:
#                return [scale*np.min(datax), scale*np.max(datax)]
#        print self.RV_list
#        ll = np.min( np.min(self.RV_list), self.data['RV'] )
#        ul = np.max( np.max(self.RV_list), self.data['RV'] )
#        return [ ll, ul ]
        
        
    def plot_CCF(self,x,y_total,y0,y1,bisector_x,bisector_y,ax=None):
        if ax is None: 
            fig, ax = plt.subplots()
        ax.plot(x, y_total, 'k-', label='Total')
        ax.plot(x, y0, 'r-', label='Obj 0')
        ax.plot(x, y1, 'b-', label='Obj 1')
        ax.plot(bisector_x[1:-1], bisector_y[1:-1], 'k-', label='Bisector')
        ax.plot(bisector_x[1:-1][10::10], bisector_y[1:-1][10::10], 'ko')
        ax.axvline(x[np.argmin(y0)],color='r',linestyle='--')
        ax.axvline(x[np.argmin(y1)],color='b',linestyle='--')
        ax.set(xlabel='RV (km/s)',ylabel='CCF Intensity',ylim=[0.99*np.min(bisector_y),1.01*np.max(bisector_y)])
#        ax.legend(loc='lower left')
        
    
    
    def plot_bisector(self,bisector_x,bisector_y,data=None,RV_max=None,ax=None):
        if ax is None: 
            fig, ax = plt.subplots()
        ax.plot(1e3*bisector_x[1:-1], bisector_y[1:-1], 'k-', label='Bisector')
        ax.plot(1e3*bisector_x[1:-1][10::10], bisector_y[1:-1][10::10], 'ko')        
        ax.set(xlabel='RV (m/s)',ylabel='CCF Intensity',xlim=[1e3*self.RV_plot_ll, 1e3*self.RV_plot_ul],ylim=[0.99*bisector_y[0],1.01*bisector_y[-1]])
        
     
     
    def plot_BIS(self,RV,BIS,RV_max=None,BIS_max=None,data=None,ax=None):
        if ax is None: 
            fig, ax = plt.subplots()
        ax.plot(1e3*RV,1e3*BIS,'ko')
#        ax.errorbar(1e3*RV,1e3*BIS,yerr=0.1,xerr=0.1,color='k',marker='o')
        ax.axhline(0,color='grey',linestyle='--')
        ax.axvline(0,color='grey',linestyle='--')
        #::: overplot measured data if given
        if all(key in data for key in ['RV', 'RV_err', 'BIS', 'BIS_err']):
            ax.errorbar(1e3*data['RV'], 1e3*data['BIS'], xerr=1e3*data['RV_err'], yerr=1e3*data['BIS_err'], color='k', linestyle='none')        
        elif all(key in data for key in ['RV', 'BIS']):
            ax.plot(1e3*data['RV'], 1e3*data['BIS'], 'k+', ms=12)
        else:
            pass
        #::: overplot RV-BIS-curve
        ax.plot(1e3*np.array(self.RV_list),1e3*np.array(self.BIS_list),'r-')
#        if (RV_max is not None) and (BIS_max is not None):
#            ax.plot([-10e3*RV_max,10e3*RV_max], [-10e3*BIS_max,10e3*BIS_max], 'r-')
#            
#            phase_grid = np.linspace(-1,1,1000)
#            xx = 1e3*RV_max*phase_grid
#            yy = 1e3*BIS_max/np.sqrt((2.*np.pi))*np.sin(phase_grid*np.pi)
#            
#            yy = xx*BIS_max/RV_max - yy
#            ax.plot(xx,yy,'g-')
        ax.set(xlabel='RV (m/s)', ylabel='BIS (m/s)', xlim=[1e3*self.RV_plot_ll, 1e3*self.RV_plot_ul], ylim=[-70,70])                
        
        
        
    def plot_RV(self,phase,RV,RV_max,data=None,ax=None):
        if ax is None: 
            fig, ax = plt.subplots()
#        RV_curve = y_total * np.sin(phase*2.*np.pi)
#        phase_grid = np.linspace(0,1,1000)
#        ax.plot(phase_grid, 1e3*self.RV_max*np.sin(phase_grid*2.*np.pi), color='grey', linestyle='-')
        ax.plot(self.phases, 1e3*np.array(self.RV_list), 'r-') #1e3*self.RV_max*np.sin(-phase_grid*2.*np.pi), 'r-')
        ax.plot(phase, 1e3*RV, 'k.')
        if all(key in data for key in ['Phase', 'RV', 'RV_err']):
            ax.errorbar(data['Phase'], 1e3*data['RV'], yerr=1e3*data['RV_err'], color='k', linestyle='none')  
        elif all(key in data for key in ['Phase', 'RV']):
            ax.plot(data['Phase'], 1e3*data['RV'], 'k+', linestyle='none')       
        ax.axhline(0,color='grey',linestyle='--')
        ax.set(xlabel='Phase', ylabel='RV (m/s)', xlim=[0,1], ylim=[1e3*self.RV_plot_ll, 1e3*self.RV_plot_ul])
        
        
        
    def plot(self,phase,RV,BIS,x,y_total,y0,y1,bisector_x,bisector_y,
             RV_max=None,BIS_max=None,data=None,axes=None):
        if axes is None:
            fig, axes = plt.subplots(1,4,figsize=(13,3))
        self.plot_CCF(x,y_total,y0,y1,bisector_x,bisector_y,ax=axes[0])
        self.plot_bisector(bisector_x,bisector_y,data=data,RV_max=RV_max,ax=axes[1])
        self.plot_BIS(RV,BIS,RV_max,BIS_max,data=data,ax=axes[2])
        self.plot_RV(phase,RV,RV_max,data=data,ax=axes[3])
        plt.tight_layout()
        
        
#    def overplot_BIS_curve(self,ax):
#        ax.plot(self)
        
        
    def animate(self,nframe):
        print(nframe, '/', self.N_frames)
            
        N_phases = len(self.phases)
        ind = int(1.*nframe/self.N_frames * N_phases)
        phase = self.phases[ind]
        RV = self.RV_list[ind]
        BIS = self.BIS_list[ind]
        y_total = self.y_total_list[ind]
        y0 = self.y0_list[ind]
        y1 = self.y1_list[ind]
        bisector_x = self.bisector_x_list[ind]
        bisector_y = self.bisector_y_list[ind]
        
        for i in [0,1,2,3]: self.axes[i].clear()
        self.plot(phase,RV,BIS,self.x,y_total,y0,y1,bisector_x,bisector_y,
                  RV_max=None,BIS_max=None,
                  data=self.data,axes=self.axes)
        self.fig.suptitle('Phase %.2f'%(phase))
        self.fig.subplots_adjust(top=0.90)
        


    def movie(self,N_frames):        
#        self.RV_max, self.BIS_max = self.get_maxima()
#        self.BIS_slope = self.get_BIS_slope(self.RV_max, self.BIS_max)
        self.calculate_all()
        self.get_plotlim()
        self.N_frames = N_frames
        
        self.fig, self.axes = plt.subplots(1,4,figsize=(13,3))
        anim = animation.FuncAnimation(self.fig, self.animate, frames=self.N_frames)
        anim.save(self.outfnames['gif'], writer='imagemagick', fps=4);
   
   
 
    def calculate_all(self):
        for phase in self.phases:
            mu0_temp = self.RVsys_0 + self.RVK_0 * np.sin(phase*2.*np.pi)
            mu1_temp = self.RVsys_1 + self.RVK_1 * np.sin(phase*2.*np.pi)
                
            y_total,y0,y1 = self.bimodal_ccf(self.x,self.A0,mu0_temp,self.fwhm_0,self.A1,mu1_temp,self.fwhm_1,self.typ)
            RV = get_RV_from_ccf(self.x, y_total)
            bisector_x, bisector_y, BIS, BIS_err = bisector(self.x, y_total, RV)
        
            self.y_total_list.append( y_total )
            self.y0_list.append( y0 )
            self.y1_list.append( y1 )
            self.RV_list.append( RV )
            self.bisector_x_list.append( bisector_x )
            self.bisector_y_list.append( bisector_y )
            self.BIS_list.append( BIS )    
        
#        self.RV_max = np.max(self.RV_list)
   