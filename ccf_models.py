# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 16:37:53 2017

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




def bimodal_ccf(x,A0,mu0,fwhm0,A1,mu1,fwhm1,offset,typ,data=None):
    '''
    note: offset refers to the total offset, i.e. each curve is offset by only 1/2 of that value
        
    note: fwhm gets passed instead of std, i.e. std have to be transformed via
    std = fwhm/2.35482004503
    
    typ: 'normal' or 'cauchy'
    '''
    offset0 = 0.5*offset
    offset1 = 0.5*offset
    
    if typ=='normal':
        std0 = fwhm0/2.35482004503
        std1 = fwhm1/2.35482004503
        y0 = 1.-offset0-A0*np.exp(-(x-mu0)**2/(2.*std0**2)) #norm.pdf(x,loc=mu0,scale=fwhm_0) 
        y1 = 1.-offset1-A1*np.exp(-(x-mu1)**2/(2.*std1**2)) #norm.pdf(x,loc=mu1,scale=fwhm_1)
        y_total = ( y0+y1-1. )
    elif typ=='cauchy':
        gamma0 = fwhm0/2.
        gamma1 = fwhm1/2.
        y0 = 1. - offset0 - A0*1./( np.pi*gamma0 * (1 + ((x-mu0)/gamma0)**2 ))
        y1 = 1. - offset1 - A1*1./( np.pi*gamma1 * (1 + ((x-mu1)/gamma1)**2 ))
        y_total = ( y0+y1-1. )
    elif typ=='custom':
        y0 = custom_ccf(x,A0,mu0,fwhm0,offset0,data,0)
        y1 = custom_ccf(x,A1,mu1,fwhm1,offset1,data,1)
        y_total = ( y0+y1-1. )
    return y_total,y0,y1
    
    
    
def custom_ccf(x,A,mu,fwhm,offset,data,obj):
    '''
    note: the passed ccf template (CCF_template_pdf)
    has to be an empirical probability density function
    i.e. normalized to mu=0, amp=1, fwhm=1 and range=[0,1]  
    '''
    xbuf = data['CCF_template_pdf_x_'+str(obj)] * fwhm + mu
    ybuf = 1.-offset - A * data['CCF_template_pdf_y_'+str(obj)]
    y = np.interp(x,xbuf,ybuf)
    return y