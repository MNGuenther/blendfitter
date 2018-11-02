#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:10:55 2018

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
import os, warnings
from glob import glob
from tqdm import tqdm 

#::: blendfitter modules
from . import ccf_io
from . import ccf_fitting



def analyse_CCFs(indir, outdir='ccfs_detrended', inst_rv='HARPS', **kwargs):
    '''
    Inputs:
    -------
    indir : str
        the directory that contains all the *_ccf_*.fits files you want to re-analyse
    outdir : str
        the directory where the output should be saved
        
    Outputs:
    --------
    plots and csv files
    '''
    if not os.path.exists(outdir): os.makedirs(outdir)
    
    fname_summary = os.path.join(outdir,'summary.csv')
    with open( fname_summary, 'ab' ) as f:
        f.write('#File,BJD,RV,RV_ll,RV_ul,Contrast,Contrast_ll,Contrast_ul,FWHM,FWHM_ll,FWHM_ul,BIS,BIS_ll,BIS_ul\n') 
        if inst_rv=='HARPS':
            f.write('#,,km/s,km/s,km/s,,,,km/s,km/s,km/s,km/s,km/s,km/s\n') 
        
    fnames = np.sort( glob( os.path.join(indir,'*_ccf_*.fits') ) )
    
    #::: for all files in that folder
    for i, fname in tqdm(enumerate(fnames), total=len(fnames)):
        
        try:
            #::: read the file
            if inst_rv=='HARPS':
                bjd,x,y = ccf_io.read_HARPS_CCF(fname)
            else:
                raise ValueError('inst_rv hast to be "HARPS".')
            
            #::: do the fit
            outname = os.path.basename(os.path.normpath(fname))[0:-5]
            ccf_fitting.run(x,y,outdir=outdir,fname=outname,
                            fname_summary=fname_summary, bjd=bjd,\
                            quiet=True,\
                            **kwargs)
        
        except:
            with open( fname_summary, 'ab' ) as f:
                f.write('#'+fname+' crashed\n') 