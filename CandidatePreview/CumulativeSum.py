import inspect
try:
    from eve_smlm.Utils import utilsHelper
    from eve_smlm.TemporalFitting import timeFitting
except ImportError:
    from Utils import utilsHelper
    from TemporalFitting import timeFitting
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import fit
import logging
import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("error", OptimizeWarning)

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "CumulativeSum": {
            "required_kwargs": [
                {"name": "show_fits","display_text":"show fits", "description": "Show lognormal-cdf fits.", "default":"True"},
                {"name": "show_first", "display_text":"show first events", "description": "Plot the first events per pixel","default":"False"},
            ],
            "optional_kwargs": [
                {"name": "weigh_first", "display_text":"weigh first events", "description": "Weigh first events per pixel with number of events/pixel","default":"False"},
                {"name": "use_background_parameters", "display_text":"use background parameters", "description": "Use background parameters for fitting","default":"True"}
            ],
            "help_string": "Draws cumulative sum (and lognormal cdf fit) of all events of the candidate cluster.",
            "display_name": "Cumulative sum of candidate cluster (Lognormal cdf fit)"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def CumulativeSum(findingResult, fittingResult, previewEvents, figure, settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    show_fits = utilsHelper.strtobool(kwargs['show_fits'])
    show_first = utilsHelper.strtobool(kwargs['show_first'])
    weigh_first = utilsHelper.strtobool(kwargs['weigh_first'])
    use_background_parameters = utilsHelper.strtobool(kwargs['use_background_parameters'])

    ax = figure.add_subplot(111)
    figure.tight_layout()
    figure.subplots_adjust(top=0.95,bottom=0.190,left=0.080)
    t_min = None
    text = ''
    # Plot the cumulative sums and fits
    # all events
    if len(findingResult[findingResult['p'] == 0]) != 0 and len(findingResult[findingResult['p'] == 1]) != 0:
        lognorm_cdf = timeFitting.LognormCDFAllEvents()
        t, del_t, fit_info, opt = lognorm_cdf(findingResult, fittingResult)
        ax.errorbar(lognorm_cdf.fit.times, lognorm_cdf.fit.cumsum, color='olive', marker='.', markersize=5, mfc='darkolivegreen', label='Cumulative sum (mix)', zorder=-2)
        if show_fits== True and np.isnan(opt[0]):
            # Fit failed, add info to text
            text += f'Fit of all events failed! {fit_info}'
        if show_fits==True and not np.isnan(opt[0]):
            if not t_min==None:
                t_min = np.min([t, t_min])
            else:
                t_min = t
            times = np.linspace(np.min([lognorm_cdf.fit.times[0], t]), lognorm_cdf.fit.times[-1], 100)
            ax.plot(times, lognorm_cdf.fit.func(times, *opt), color='black', label='Lognormal CDF fit (mix)')
            ax.axvline(x=t, color='maroon', label='Fitted time (mix)')
            # ax.axvspan(t-del_t, t+del_t, alpha=0.1, color='maroon')

    # positive events
    elif len(findingResult[findingResult['p'] == 1]) != 0 and len(findingResult[findingResult['p'] == 0]) == 0:
        lognorm_cdf = timeFitting.LognormCDFAllEvents()
        t, del_t, fit_info, opt = lognorm_cdf(findingResult[findingResult['p'] == 1], fittingResult)
        ax.errorbar(lognorm_cdf.fit.times, lognorm_cdf.fit.cumsum, color='C0', marker='.', markersize=5, mfc='mediumblue', label='Cumulative sum (pos)',zorder=-2)
        if show_fits== True and np.isnan(opt[0]):
            # Fit failed, add info to text
            text += f'Fit of pos. events failed! {fit_info}'
        if show_fits==True and not np.isnan(opt[0]):
            if not t_min==None:
                t_min = np.min([t, t_min])
            else:
                t_min = t
            times = np.linspace(np.min([lognorm_cdf.fit.times[0], t]), lognorm_cdf.fit.times[-1], 100)
            ax.plot(times, lognorm_cdf.fit.func(times, *opt), color='black', label='Lognormal CDF fit (pos)')
            ax.axvline(x=t, color='maroon', label='Fitted time (pos)')
            # ax.axvspan(t-del_t, t+del_t, alpha=0.1, color='maroon')

    # negative events
    else: #  len(findingResult[findingResult['p'] == 0]) == 0:
        lognorm_cdf = timeFitting.LognormCDFAllEvents()
        t, del_t, fit_info, opt = lognorm_cdf(findingResult[findingResult['p'] == 0], fittingResult)
        ax.errorbar(lognorm_cdf.fit.times, lognorm_cdf.fit.cumsum, color='C1', marker='.', markersize=5, mfc='peru', label='Cumulative sum (neg)', zorder=-2)
        if show_fits== True and np.isnan(opt[0]):
            # Fit failed, add info to text
            text += f'Fit of neg. events failed! {fit_info}'
        if show_fits==True  and not np.isnan(opt[0]):
            if not t_min==None:
                t_min = np.min([t, t_min])
            else:
                t_min = t
            times = np.linspace(np.min([lognorm_cdf.fit.times[0], t]), lognorm_cdf.fit.times[-1], 100)
            ax.plot(times, lognorm_cdf.fit.func(times, *opt), color='black', label='Lognormal CDF fit (neg)')
            ax.axvline(x=t, color='maroon', label='Fitted time (neg)')
            # ax.axvspan(t-del_t, t+del_t, alpha=0.1, color='maroon')
    
    # first events
    if show_first==True and use_background_parameters==True:
        if weigh_first!=True:
            lognorm_cdf = timeFitting.LognormCDFFirstEvents()
            t, del_t, fit_info, opt = lognorm_cdf(findingResult, fittingResult)
            ax.errorbar(lognorm_cdf.fit.times, lognorm_cdf.fit.cumsum, color='C2', marker='.', markersize=5, mfc='darkgreen', label='Cumulative sum (first)', zorder=-1)
            if show_fits== True and np.isnan(opt[0]):
                # Fit failed, add info to text
                if text != '':
                    text += '\n'
                text += f'Fit of first events failed! {fit_info}'
            if show_fits==True and not np.isnan(opt[0]):
                if not t_min==None:
                    t_min = np.min([t, t_min])
                else:
                    t_min = t
                times = np.linspace(np.min([lognorm_cdf.fit.times[0], t]), lognorm_cdf.fit.times[-1], 100)
                ax.plot(times, lognorm_cdf.fit.func(times, *opt), color='black', ls='--', label='Lognormal CDF fit (first)')
                ax.axvline(x=t, color='maroon', ls='--', label='Fitted time (first)')
                # ax.axvspan(t-del_t, t+del_t, alpha=0.1, color='maroon')
        else:
            lognorm_cdf = timeFitting.LognormCDFFirstEvents_weighted()
            t, del_t, fit_info, opt = lognorm_cdf(findingResult, fittingResult)
            
            ax.errorbar(lognorm_cdf.fit.times, lognorm_cdf.fit.cumsum, yerr=lognorm_cdf.sigma, color='C2', marker='.', markersize=5, mfc='darkgreen', label='Cumulative sum (first, weighted)', zorder=-1)
            if show_fits== True and np.isnan(opt[0]):
                # Fit failed, add info to text
                if text != '':
                    text += '\n'
                text += f'Fit of first events failed! {fit_info}'
            if show_fits==True and not np.isnan(opt[0]):
                if not t_min==None:
                    t_min = np.min([t, t_min])
                else:
                    t_min = t
                times = np.linspace(np.min([lognorm_cdf.fit.times[0], t]), lognorm_cdf.fit.times[-1], 100)
                ax.plot(times, lognorm_cdf.fit.func(times, *opt), color='black', ls='--', label='Lognormal CDF fit (first, weighted)')
                ax.axvline(x=t, color='maroon', ls='--', label='Fitted time (first, weighted)')
                # ax.axvspan(t-del_t, t+del_t, alpha=0.1, color='maroon')
    elif show_first==True and use_background_parameters==False:
        if weigh_first != True:
            lognorm_cdf = timeFitting.LognormCDFFirstEvents_NoBackground()
            t, del_t, fit_info, opt = lognorm_cdf(findingResult, fittingResult)
            ax.errorbar(lognorm_cdf.fit.times, lognorm_cdf.fit.cumsum, color='C2', marker='.', markersize=5, mfc='darkgreen', label='Cumulative sum (first)', zorder=-1)
            if show_fits== True and np.isnan(opt[0]):
                # Fit failed, add info to text
                if text != '':
                    text += '\n'
                text += f'Fit of first events failed! {fit_info}'
            if show_fits==True and not np.isnan(opt[0]):
                if not t_min==None:
                    t_min = np.min([t, t_min])
                else:
                    t_min = t
                times = np.linspace(np.min([lognorm_cdf.fit.times[0], t]), lognorm_cdf.fit.times[-1], 100)
                ax.plot(times, lognorm_cdf.fit.func(times, *opt), color='black', ls='--', label='Lognormal CDF fit (first, no background)')
                ax.axvline(x=t, color='maroon', ls='--', label='Fitted time (first)')
                # ax.axvspan(t-del_t, t+del_t, alpha=0.1, color='maroon')
        else:
            lognorm_cdf = timeFitting.LognormCDFFirstEvents_weighted()
            t, del_t, fit_info, opt = lognorm_cdf(findingResult, fittingResult)
            
            ax.errorbar(lognorm_cdf.fit.times, lognorm_cdf.fit.cumsum, yerr=lognorm_cdf.sigma, color='C2', marker='.', markersize=5, mfc='darkgreen', label='Cumulative sum (first, weighted)', zorder=-1)
            if show_fits== True and np.isnan(opt[0]):
                # Fit failed, add info to text
                if text != '':
                    text += '\n'
                text += f'Fit of first events failed! {fit_info}'
            if show_fits==True and not np.isnan(opt[0]):
                if not t_min==None:
                    t_min = np.min([t, t_min])
                else:
                    t_min = t
                times = np.linspace(np.min([lognorm_cdf.fit.times[0], t]), lognorm_cdf.fit.times[-1], 100)
                ax.plot(times, lognorm_cdf.fit.func(times, *opt), color='black', ls='--', label='Lognormal CDF fit (first, weighted, no background)')
                ax.axvline(x=t, color='maroon', ls='--', label='Fitted time (first, weighted)')
                # ax.axvspan(t-del_t, t+del_t, alpha=0.1, color='maroon')
    elif show_first==False and use_background_parameters==False:
        lognorm_cdf = timeFitting.LognormCDFAllEvents_NoBackground()
        t, del_t, fit_info, opt = lognorm_cdf(findingResult, fittingResult)
        ax.errorbar(lognorm_cdf.fit.times, lognorm_cdf.fit.cumsum, color='C2', marker='.', markersize=5, mfc='darkgreen', label='Cumulative sum (first)', zorder=-1)
        if show_fits== True and np.isnan(opt[0]):
            # Fit failed, add info to text
            if text != '':
                text += '\n'
            text += f'Fit of first events failed! {fit_info}'
        if show_fits==True and not np.isnan(opt[0]):
            if not t_min==None:
                t_min = np.min([t, t_min])
            else:
                t_min = t
            times = np.linspace(np.min([lognorm_cdf.fit.times[0], t]), lognorm_cdf.fit.times[-1], 100)
            ax.plot(times, lognorm_cdf.fit.func(times, *opt), color='black', ls='--', label='Lognormal CDF fit (no background)')
            ax.axvline(x=t, color='maroon', ls='--', label='Fitted time (first)')
            # ax.axvspan(t-del_t, t+del_t, alpha=0.1, color='maroon')

    for index, row in fittingResult[:-1].iterrows():
        ax.axvline(x=row['t'], color='red')
    ax.axvline(x=fittingResult.iloc[-1]['t'], color='red', label='Localization(s)')

    if text != '':
        props = dict(boxstyle='round', facecolor='white', edgecolor='None', alpha=0.8)
        figure.text(0.05, 0.05, s=text, fontsize=10, color='red', verticalalignment='bottom', horizontalalignment='left', bbox=props, wrap=True)

    # Add and set labels
    if not t_min==None:
        t_min = np.min([t_min, fittingResult.iloc[0]['t']])
    else:
        t_min = fittingResult.iloc[0]['t']
    if t_min-2 < findingResult['t'].iloc[0]*1e-3:
        ax.set_xlim(t_min-2, findingResult['t'].iloc[-1]*1e-3) # in ms
    else:
        ax.set_xlim(findingResult['t'].iloc[0]*1e-3, findingResult['t'].iloc[-1]*1e-3) # in ms
    ax.set_xlabel('t [ms]')
    ax.set_ylabel('cumulative number of events')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    figure.tight_layout()
    
    # required output none
    return 1
