# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:33:11 2017

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
import os, sys
from collections import OrderedDict 
import warnings
import emcee
try:
    import celerite
    from celerite import terms
except:
    warnings.warn('Module "celerite" could not be imported. Some functionality might not be available.')
try:
    import george
    from george import kernels
except:
    warnings.warn('Module "george" could not be imported. Some functionality might not be available.')
import corner
from multiprocessing import Pool, cpu_count
from contextlib import closing
from tqdm import tqdm
from datetime import datetime

from . import bisector



np.random.seed(42)



###############################################################################
#::: gp mean models
############################################################################### 
class Model_george(george.modeling.Model):
    parameter_names = ("Contrast", "RV", "FWHM")
    def get_value(self, x):
        amp = self.Contrast/100.
        mu = self.RV
        std = self.FWHM/2.35482004503  
        return 1. - amp * np.exp( -0.5*((x.flatten()-mu)/std)**2 )
    
class Model_celerite(celerite.modeling.Model):
    parameter_names = ("Contrast", "RV", "FWHM")
    def get_value(self, x):
        amp = self.Contrast/100.
        mu = self.RV
        std = self.FWHM/2.35482004503  
        return 1. - amp * np.exp( -0.5*((x.flatten()-mu)/std)**2 )


    
###############################################################################
#::: call the gp
###############################################################################     
def call_gp(params):
    log_sigma, log_rho, log_error_scale, contrast, rv, fwhm = params
    if GP_CODE=='celerite':
        mean_model = Model_celerite(**dict(**dict(Contrast=contrast, RV=rv, FWHM=fwhm)))
        kernel = terms.Matern32Term(log_sigma=log_sigma, log_rho=log_rho)
        gp = celerite.GP(kernel, mean=mean_model) 
        gp.compute(xx, yerr=yyerr/err_norm*np.exp(log_error_scale))
        return gp
    elif GP_CODE=='george':
        mean_model = Model_george(**dict(**dict(Contrast=contrast, RV=rv, FWHM=fwhm)))
        kernel = np.exp(log_sigma) * kernels.Matern32Kernel(log_rho)
        gp = george.GP(kernel, mean=mean_model)
        gp.compute(xx, yerr=yyerr/err_norm*np.exp(log_error_scale))
        return gp
    else:
        raise ValueError('gp_code must be "celerite" or "george".')



###############################################################################
#::: priors
###############################################################################  
def external_log_prior(params):
    log_sigma, log_rho, log_error_scale, contrast, rv, fwhm = params
    
    lp = 0
    if not (-23 < log_sigma < 23):
        lp = -np.inf
    if not (-23 < log_rho < 23):
        lp = -np.inf
    if not (-23 < log_error_scale < 23):
        lp = -np.inf
    if not (0 < contrast < 100):
        lp = -np.inf
    if not (-1000 < rv < 1000):
        lp = -np.inf
    if not (-100 < fwhm < 100):
        lp = -np.inf
    
    return lp
    

###############################################################################
#::: set up MCMC log probability function
#::: (has to be top-level for pickle)
###############################################################################
def log_probability(params):
    '''
    works on xx, yy
    '''
#    log_sigma, log_rho, log_error_scale, contrast, rv, fwhm = params
    
    try:
        gp = call_gp(params)
        ll = gp.log_likelihood(yy)
        lp = gp.log_prior() + external_log_prior(params)
    except:
        return -np.inf
    if not np.isfinite(lp):
        return -np.inf
    return ll + lp
    


###############################################################################
#::: run
###############################################################################    
def run(x,y,
        yerr=None,
        systematics_rv_grid_scale=10.,
        nwalkers=50, thin_by=50, burn_steps=2500, total_steps=5000,
        gp_code='celerite',
        method='median_posterior', chunk_size=2000, Nsamples_detr=10, Nsamples_plot=10, 
        xlabel='RV (km/s)', ylabel='CCF', ydetr_label='ydetr',
        outdir='ccf_gp_decor', fname=None, 
        fname_summary=None, bjd=None,
        multiprocess=False, multiprocess_cores=None,
        quiet=False):
    
    '''
    Required Input:
    ---------------
    x : array of float
        RV-grid values of the data set
    y : array of float
        CCF values of the data set
        
    Optional Input:
    ---------------
    yerr : array of float / float
        errorbars on y-values of the data set;
        if None, these are estimated as std(y);
        this is only needed to set an initial guess for the GP-fit;
        white noise is fitted as a jitter term
    systematics_rvrgrid_scale : float (defaut None)
        the rv grid scale of the systeamtics 
        must be in the same units as x
        if None, set to 1. (assuming usually x is in days, 1. day is reasonable)
    nwalkers : int
        number of MCMC walkers
    thin_by : int
        thinning the MCMC chain by how much
    burn_steps : int
        how many steps to burn in the MCMC
    total_steps : int
        total MCMC steps (including burn_steps)
    gp_code : str (default 'celerite')
        'celerite' or 'george'
        which GP code to use
    method : str (default 'median_posterior')
        how to calculate the GP curve that's used for detrending
            'mean_curve' : take Nsamples_detr and calculate many curves, detrend by the mean of all of them
            'median_posterior' : take the median of the posterior and predict a single curve
    chunk_size : int (default 5000)
        calculate gp.predict in chunks of the entire light curve (to not crash memory)
    Nsamples_detr : float (default 10)
        only used if method=='mean_curve'
        how many samples used for detrending
    Nsampels_plot : float (default 10)
        only used if method=='mean_curve'
        how many samples used for plotting
    xlabel : str
        x axis label (for plots)
    ylabel : str
        y axis label (for plots)       
    ydetr_label : str
        y_detr axis label (for plots)    
    outdir : str
        name of the output directory
    fname : str
        prefix of the output files (e.g. a planet name)
    multiprocess : bool (default True)
        run MCMC on many cores        
    '''

    if (gp_code=='celerite') & ('celerite' not in sys.modules):
        raise ValueError('You are trying to use "celerite", but it is not installed.')
    elif (gp_code=='george') & ('george' not in sys.modules):
        raise ValueError('You are trying to use "george", but it is not installed.')

    if multiprocess_cores is None:
        multiprocess_cores = cpu_count()-1
        
    
    #::: this is ugly, I know;
    #::: blame the multiprocessing and pickling issues, 
    #::: which demand global variables for efficiency    
    global xx
    global yy
    global yyerr
    global err_norm
    global GP_CODE
    xx = x
    yy = y
    GP_CODE = gp_code
    

    #::: outdir
    if not os.path.exists(outdir): os.makedirs(outdir)
    
    
    #::: print function that prints into console and logfile at the same time 
    now = datetime.now().isoformat()
    def logprint(*text):
        if not quiet: print(*text)
        original = sys.stdout
        with open( os.path.join(outdir,fname+'logfile_'+now+'.log'), 'a' ) as f:
            sys.stdout = f
            print(*text)
        sys.stdout = original

    
    #::: fname
    if fname is not None:
        fname += '_gp_decor_'
    else:
        fname = 'gp_decor_'
    
    
    
    #::: save settings
    if not os.path.exists(outdir): os.makedirs(outdir)
    header = 'gp_code,nwalkers,thin_by,burn_steps,total_steps'
    X = np.column_stack(( gp_code, nwalkers, thin_by, burn_steps, total_steps ))
    np.savetxt( os.path.join(outdir,fname+'settings.csv'), X, header=header, delimiter=',', fmt='%s')
    
    
    
    #::: plot the data
    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=yerr, fmt='b.', capsize=0)
    ax.set( xlabel=xlabel, ylabel=ylabel, title='Original data' )
    fig.savefig( os.path.join(outdir,fname+'data.jpg'), dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    
    #::: MCMC plot settings
    names = [r'gp: $\log{\sigma}$', r'gp: $\log{\rho}$', r'$\log{(y_\mathrm{err})}$', "Contrast", "RV", "FWHM"]
    discard = int(1.*burn_steps/thin_by)
     
  
    #::: plot grid
    t = np.linspace(np.min(x), np.max(x), 2000)
    
    
    logprint('\nStarting...')
    
        
    
    #::: set up mean model
    #::: get the mean model's initial guess from inverse gauss with constant offset
    [offset, amp, mu, std], _ = bisector.fit_inverse_gauss(x,y)
    contrast_init = 100.*amp
    rv_init = mu
    fwhm_init = 2.35482004503*std
    model_init = 1. - offset - amp * np.exp( -0.5*((x-mu)/std)**2 )
    
    
    
    #::: plot the fit of an inverse gauss with constant offset
    model_curve = 1. - offset - amp * np.exp( -0.5*((t-mu)/std)**2 )
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b.')
    ax.plot(t, model_curve, 'r-', lw=2)
    ax.set( xlabel=xlabel, ylabel=ylabel, title='Simple Gaussian fit' )
    fig.savefig( os.path.join(outdir,fname+'simple_gaussian_fit.jpg'), dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    
    #::: guess yerr if not given
    if yerr is None:
        yerr = np.nanstd(y-model_init) * np.ones_like(y)
    yyerr = yerr
    
    
    
    #::: get the gp params' initial guesses
    #::: log(sigma)
    log_sigma_init = np.log(np.nanstd(y-model_init))
    
    #::: log(rho)
    log_rho_init = np.log(systematics_rv_grid_scale)
    
    #::: log(yerr)
    err_norm = np.nanmean(yerr)
    err_scale = np.nanmean(yerr)
    log_err_scale_init = np.log(err_scale)
    
    
    #::: all initial guesses 
    initial = np.array([log_sigma_init, log_rho_init, log_err_scale_init, contrast_init, rv_init, fwhm_init])
    ndim = 6
    
    

    ###########################################################################
    #::: MCMC fit
    ###########################################################################
    logprint('\nRunning MCMC fit...')
    if multiprocess: logprint('\tRunning on', multiprocess_cores, 'CPUs.')   

    
    #::: set up MCMC
    ndim = len(initial)
    backend = emcee.backends.HDFBackend(os.path.join(outdir,fname+'mcmc_save.h5')) # Set up a new backend
    backend.reset(nwalkers, ndim)


    #::: run MCMC
    def run_mcmc(sampler):
        p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
        sampler.run_mcmc(p0, total_steps/thin_by, thin_by=thin_by, progress=not quiet);
    
    if multiprocess:    
        with closing(Pool(processes=(cpu_count()-1))) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, backend=backend)
            run_mcmc(sampler)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend)
        run_mcmc(sampler)
    
    logprint('\nAcceptance fractions:')
    logprint(sampler.acceptance_fraction)

    tau = sampler.get_autocorr_time(discard=discard, c=5, tol=10, quiet=True)*thin_by
    logprint('\nAutocorrelation times:')
    logprint('\t', '{0: <30}'.format('parameter'), '{0: <20}'.format('tau (in steps)'), '{0: <20}'.format('Chain length (in multiples of tau)'))
    for i, key in enumerate(names):
        logprint('\t', '{0: <30}'.format(key), '{0: <20}'.format(tau[i]), '{0: <20}'.format((total_steps-burn_steps) / tau[i]))
    
        
        
        
        
        
    def gp_predict_in_chunks(y, x, quiet=True):
        #::: predict in chunks of 1000 data points to not crash memory
        mu = []
        var = []
        for i in tqdm(range( int(1.*len(x)/chunk_size)+1 ), disable=quiet):
            m, v = gp.predict(y, x[i*chunk_size:(i+1)*chunk_size], return_var=True)
            mu += list(m)
            var += list(v)
        return np.array(mu), np.array(var)
        
    
        
        
    #::: get the samples, 
    #::: the posterior-median yerr, 
    #::: and calculate the mean GP curve / posterior-median GP curve
    samples = sampler.get_chain(flat=True, discard=discard)
    err_scale = np.exp(np.median(samples[:,2]))
    yyerr = yyerr/err_norm*err_scale
    yerr = yerr/err_norm*err_scale #TODO: check this... scale the yerr the same way the OOE / binned data was rescaled


#    logprint '\nPlot 1'
    if method=='mean_curve':
        mu_all_samples = []
        std_all_samples = []
        for s in tqdm(samples[np.random.randint(len(samples), size=Nsamples_plot)], disable=quiet):
            gp = call_gp(s)
#            mu, var = gp.predict(yy, t, return_var=True)
            mu, var = gp_predict_in_chunks(yy, t, quiet=True)
            std = np.sqrt(var)
            mu_all_samples.append( mu )
            std_all_samples.append( std )
        mu_GP_curve = np.mean(mu_all_samples, axis=0)
        std_GP_curve = np.mean(std_all_samples, axis=0)
    
    elif method=='median_posterior':      
        params = [ np.median( samples[:,i] ) for i in range(ndim) ]
        gp = call_gp(params)
#        mu, var = gp.predict(yy, t, return_var=True)
        mu, var = gp_predict_in_chunks(yy, t, quiet=True)
        mu_GP_curve = mu
        std_GP_curve = np.sqrt(var)
    
    
    #::: Plot the data and individual posterior samples
#    fig, ax = plt.subplots()
#    ax.errorbar(x, y, yerr=yerr, fmt=".b", capsize=0)
#    ax.errorbar(x[ind_in], y[ind_in], yerr=yerr, fmt=".", color='skyblue', capsize=0)
#    for mu, std in zip(mu_all_samples, std_all_samples):
#        ax.plot(t, mu, color='r', alpha=0.1, zorder=11)    
#    ax.set( xlabel=xlabel, ylabel=ylabel, title="MCMC posterior samples", ylim=[1-0.002, 1.002] )
#    fig.savefig( os.path.join(outdir,fname+'MCMC_fit_samples.jpg'), dpi=100, bbox_inches='tight')
    
    
    #::: plot the data and "mean"+"std" GP curve
    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=yerr, fmt='b.', capsize=0)
    ax.plot(t, mu_GP_curve, color='r', zorder=11)
    ax.fill_between(t, mu_GP_curve+std_GP_curve, mu_GP_curve-std_GP_curve, color='r', alpha=0.3, edgecolor="none", zorder=10)
    ax.set( xlabel=xlabel, ylabel=ylabel, title="MCMC posterior predictions" )
    fig.savefig( os.path.join(outdir,fname+'mcmc_fit.jpg'), dpi=100, bbox_inches='tight')
    plt.close(fig)

    #::: plot chains; format of chain = (nwalkers, nsteps, nparameters)
#    logprint('Plot chains')
    fig, axes = plt.subplots(ndim+1, 1, figsize=(6,4*(ndim+1)) )
    steps = np.arange(0,total_steps,thin_by)
    
    
    #::: plot the lnprob_values (nwalkers, nsteps)
    for j in range(nwalkers):
        axes[0].plot(steps, sampler.get_log_prob()[:,j], '-')
    axes[0].set( ylabel='lnprob', xlabel='steps' )
    
    
    #:::plot all chains of parameters
    for i in range(ndim):
        ax = axes[i+1]
        ax.set( ylabel=names[i], xlabel='steps')
        for j in range(nwalkers):
            ax.plot(steps, sampler.chain[j,:,i], '-')
        ax.axvline( burn_steps, color='k', linestyle='--' )
    
    plt.tight_layout()
    fig.savefig( os.path.join(outdir,fname+'mcmc_chains.jpg'), dpi=100, bbox_inches='tight')
    plt.close(fig)
        
    
    #::: plot corner
    fig = corner.corner(samples,
                        labels=names,
                        show_titles=True, title_kwargs={"fontsize": 12});
    fig.savefig( os.path.join(outdir,fname+'mcmc_corner.jpg'), dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    
    #::: Calculate the detrended data
    logprint('\nRetrieve samples for detrending...')
    sys.stdout.flush()
    if method=='mean_curve':
        mu_all_samples = []
        std_all_samples = []
        for s in tqdm(samples[np.random.randint(len(samples), size=Nsamples_detr)], disable=quiet):
            gp = call_gp(s)
#            mu, var = gp.predict(yy, x, return_var=True)
            mu, var = gp_predict_in_chunks(yy, x)
            std = np.sqrt(var)
            mu_all_samples.append( mu )
            std_all_samples.append( std )
        mu_GP_curve = np.mean(mu_all_samples, axis=0)
        std_GP_curve = np.mean(std_all_samples, axis=0)
        
    elif method=='median_posterior':      
        params = [ np.median( samples[:,i] ) for i in range(ndim) ]
        gp = call_gp(params)
#        mu, var = gp.predict(yy, x, return_var=True)
        mu, var = gp_predict_in_chunks(yy, x)
        mu_GP_curve = mu
        std_GP_curve = np.sqrt(var)
        mu_gauss_curve = 1. - params[3]/100. * np.exp( -0.5*((x.flatten()-params[4])/(params[5]/2.35482004503))**2 )
    
    
    logprint('\nCreating output...')
    
    
    #::: remove the GP baseline, but keep the Gaussian; 
    #::: so that the bisector can be extracted afterwards
    ydetr = y - mu_GP_curve + mu_gauss_curve
    ydetr_err = yerr
    
    
    #::: Save the detrended data as .txt
#    logprint 'Output results.csv'
    header = xlabel+','+ydetr_label+','+ydetr_label+'_err'
    X = np.column_stack(( x, ydetr, ydetr_err ))
    np.savetxt( os.path.join(outdir,fname+'mcmc_ydetr.csv'), X, header=header, delimiter=',')


    #::: Save the GP curve as .txt
#    logprint 'Output results_gp.csv'
    header = xlabel+',gp_mu,gp_std'
    X = np.column_stack(( x, mu_GP_curve, std_GP_curve ))
    np.savetxt( os.path.join(outdir,fname+'mcmc_gp.csv'), X, header=header, delimiter=',')

    

    #::: Plot the detrended data
#    logprint 'Plot 1'
    fig, ax = plt.subplots()
    ax.errorbar(x, ydetr, yerr=ydetr_err, fmt='b.', capsize=0)
    ax.set( xlabel=xlabel, ylabel=ylabel, title="Detrended data" )
    fig.savefig( os.path.join(outdir,fname+'mcmc_ydetr.jpg'), dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    
    #::: get the resulting params dictionaries
    params, params_ll, params_ul = get_params_from_samples(samples, names)
    
    
    #::: derive bisector posteriors
    BIS = []
    for s in samples:
        BIS.append( bisector.subcompute_HARPS_improved(x, ydetr, s[4], s[5]/2.35482004503, s[3]/100., 0.)[0] )
    v = np.percentile(np.array(BIS), [16, 50, 84], axis=0)
    params['BIS'] = v[1]
    params_ul['BIS'] = v[2]-v[1]
    params_ll['BIS'] = v[1]-v[0]
    
    
    #::: Save the resulting parameters in a table
    with open( os.path.join(outdir,fname+'table.csv'), 'wb' ) as f:
        f.write('name,median,ll,ul\n')
        for i, key in enumerate(names+['BIS']):
            f.write(key + ',' + str(params[key]) + ',' + str(params_ll[key]) + ',' + str(params_ul[key]) + '\n' )
                
    
    #::: if requested, append a row into the summary file, too
    if fname_summary is not None:
        with open( fname_summary, 'ab' ) as f:
            f.write(fname + ',')
            if bjd is not None:
                f.write(str(bjd) + ',')
            for i, key in enumerate(['RV','Contrast','FWHM','BIS']):
                f.write(str(params[key]) + ',' + str(params_ll[key]) + ',' + str(params_ul[key]) )
                if key is not 'BIS': f.write(',')
                else: f.write('\n')
            
            
    logprint('\nDone. All output files are in '+outdir)


    
###############################################################################
#::: update params with MCMC/NS results
###############################################################################
def get_params_from_samples(samples, names):
    '''
    read MCMC results and update params
    '''

    buf = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))                                         
    theta_median = [ item[0] for item in buf ]
    theta_ul = [ item[1] for item in buf ]
    theta_ll = [ item[2] for item in buf ]
    params_median = { n:t for n,t in zip(names,theta_median) }
    params_ul = { n:t for n,t in zip(names,theta_ul) }
    params_ll = { n:t for n,t in zip(names,theta_ll) }
    
    return params_median, params_ll, params_ul

    
    
                
                
                
if __name__ == '__main__':
    pass
    
