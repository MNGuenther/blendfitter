#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:35:27 2018

@author:
Maximilian N. Günther
MIT Kavli Institute for Astrophysics and Space Research, 
Massachusetts Institute of Technology,
77 Massachusetts Avenue,
Cambridge, MA 02109, 
USA
Email: maxgue@mit.edu
Web: www.mnguenther.com
"""

from __future__ import print_function, division, absolute_import

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

#::: modules
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
#import warnings
#from tqdm import tqdm

#::: blendfitter modules
from . import bisector, fit_CCF




class RV_inst():
    
    def __init__(self, \
                 time = None, \
                 CCFs=None, \
                 RV=None, RV_err=None, \
                 Contrast=None, Contrast_err=None, \
                 FWHM=None, FWHM_err=None, \
                 offset=None, offset_err=None, \
                 BIS=None, BIS_err=None, \
                 bisectors_x=None, bisectors_y=None \
                 ):
        '''
        Inputs:
        -------
        time : array-like 
            time stamps (e.g. HJD or BJD)
        CCFs : list 
            list of CCF instances
        RV : array-like
            RV values from the instrument's standard pipeline
            
        ...
        
        Computed parameters:
        --------------------
        RV2 : array-like
            RV values as computed directly from the CCFs by this code
        RV3 : array-like
            RV values as computed directly from the GP-detrended CCFs by this code
        '''
        self.time = time
        self.CCFs = CCFs
        self.RV = RV
        self.RV_err = RV_err
        self.Contrast = Contrast
        self.Contrast_err = Contrast_err
        self.FWHM = FWHM
        self.FWHM_err = FWHM_err
        self.offset = offset
        self.offset_err = offset_err
        self.BIS = BIS
        self.BIS_err = BIS_err
        self.bisectors_x = bisectors_x
        self.bisectors_y = bisectors_y
        
        
    def compute_values_from_CCFs(self):
        self.RV2, self.RV_err2, \
           self.Contrast2, self.Contrast_err2, \
           self.FWHM2, self.FWHM_err2, \
           self.offset2, self.offset_err2, \
           self.BIS2, self.BIS_err2, \
           self.bisector_x2, self.bisector_y2 = zip( *[fit_CCF.extract_values(ccf.x, ccf.y) for ccf in self.CCFs] )          
        
        
    def compute_values_from_GPdetrended_CCFs(self):
        self.RV3, self.RV_err3, \
           self.Contrast3, self.Contrast_err3, \
           self.FWHM3, self.FWHM_err3, \
           self.offset3, self.offset_err3, \
           self.BIS3, self.BIS_err3, \
           self.bisector_x3, self.bisector_y3 = zip( *[fit_CCF.extract_values(ccf.x, ccf.y_detr) for ccf in self.CCFs] )          
                
        
        

class CCF():
    
    def __init__(self, x=None, y=None, yerr=None):
        self.x = x
        self.y = y
        self.yerr = yerr
        
#        if yerr is not None: 
#            self.yerr = yerr
#        else: 
#            self.estimate_yerr()
            
            
    def load_HARPS_CCF(self, fname):
        
        hdulist = fits.open(fname)
        header = hdulist[0].header
        table = hdulist[0].data
        
        self.x = np.arange( header['CRVAL1'], header['CRVAL1']+table.shape[1]*header['CDELT1']-1e-12, header['CDELT1'] )
        self.y = table[-1]  
        ind_max = np.argpartition(self.y, -10)[-10:] #indices of 10 maximum values
        self.y /= np.mean(self.y[ind_max])
            
                
    def estimate_yerr(self):
        [offset, amp, mu, std], _ = bisector.fit_inverse_gauss(self.x,self.y)
        self.yerr = np.nanstd( self.y - bisector.inverse_gauss(self.x,offset,amp,mu,std) )
        
    
    def plot(self):
        fig, ax = plt.subplots()
        plt.plot(self.x, self.y, 'k.')
        return fig, ax
    
    
    ###########################################################################
    #::: Model: Gauss with constant baseline
    ###########################################################################
#    def fit_inverse_gauss(self):
#        [self.offset, self.amp, self.mu, self.std], _ = bisector.fit_inverse_gauss(self.x,self.y)
#        
#    
#    def plot_inverse_gauss(self,outdir,outname):
#        pass
    
    
#    def extract_values(self, return_=False):
#        if return_:
#            return fit_CCF.extract_values(self.x, self.y)
#        
#        else:
#            self.RV, self.RV_err, \
#               self.Contrast, self.Contrast_err, \
#               self.FWHM, self.FWHM_err, \
#               self.offset, self.offset_err, \
#               self.BIS, self.BIS_err, \
#               self.bisector_x, self.bisector_y = \
#               fit_CCF.extract_values(self.x, self.y)      
    ###########################################################################
     
     
    ###########################################################################
    #::: Model: Gauss with GP baseline
    ###########################################################################
    def fit_inverse_gauss_with_GP(self,outdir,outname,**kwargs):
        if self.x is None:
            raise ValueError('x is None')
        if self.y is None:
            raise ValueError('y is None')
        if self.yerr is None:
            raise ValueError('yerr is None')
        if (self.x is not None) & (self.y is not None) & (self.yerr is not None):
            fit_CCF.fit_inverse_gauss_with_GP_rdx(self.x,self.y,self.yerr,outdir,outname,**kwargs)
    
    
    
    def replot_inverse_gauss_with_GP(self,outdir,outname):
        fit_CCF.replot_inverse_gauss_with_GP(self.x,self.y,self.yerr,outdir,outname)
        
    
    def remove_GP_baseline(self,outdir,outname):
        self.y_detr = fit_CCF.remove_GP_baseline(self.x,self.y,self.yerr,outdir,outname)
    ###########################################################################
        
        
        
        


    

    