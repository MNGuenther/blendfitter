#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:20:46 2018

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

    
    
def read_HARPS_CCF(fname):
    hdulist = fits.open(fname)
    header = hdulist[0].header
    table = hdulist[0].data
    
    BJD = np.float64(header['HIERARCH ESO DRS BJD'])
    RV_grid = np.arange( header['CRVAL1'], header['CRVAL1']+table.shape[1]*header['CDELT1']-1e-12, header['CDELT1'] )
    CCF = table[-1]  
    ind_max = np.argpartition(CCF, -10)[-10:] #indices of 10 maximum values
    CCF /= np.mean(CCF[ind_max])
    
    return BJD, RV_grid, CCF
 
    


def extract_HARPS_data_from_fits(fname):
    hdulist = fits.open(fname)
    header = hdulist[0].header
    
    BJD = np.float64(header['HIERARCH ESO DRS BJD'])
    RV = np.float64(header['HIERARCH ESO DRS RVC'])
    RV_err = np.sqrt( np.float64(header['HIERARCH ESO DRS CCF NOISE'])**2 + np.float64(header['HIERARCH ESO DRS DRIFT NOISE'])**2 )
    Contrast = np.float64(header['HIERARCH ESO DRS CONTRAST'])
    FWHM = np.float64(header['HIERARCH ESO DRS FWHM'])
    
    return BJD, RV, RV_err, Contrast, FWHM