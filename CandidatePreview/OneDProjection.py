import inspect
from Utils import utilsHelper
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import fit
import logging
import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("error", OptimizeWarning)
#Obtain eventdistribution functions
from EventDistributions import eventDistributions

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "OneDProjection": {
            "required_kwargs": [
                {"name": "t_bin_width","display_text":"temporal bin width", "description": "Temporal bin width (in ms)", "default":"1"},
                {"name": "show_first", "display_text":"show first events", "description": "Plot the first events per pixel","default":"False"},
                {"name": "weigh_first", "display_text":"weigh first events", "description": "Weigh first events per pixel with number of events/pixel","default":"False"}
            ],
            "optional_kwargs": [
            ],
            "help_string": "Draws temporal trace of the candidate cluster.",
            "display_name": "Temporal trace of candidate cluster"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

# Define the modified shifted Rayleigh function with an amplitude parameter
def rayleigh_distribution(x, shift, sigma, amplitude, offset):
    result = amplitude * (x - shift)/(sigma**2) * np.exp(-(x - shift)**2 / (2 * sigma**2))
    result[x < shift] = 0  # Set the function to zero for x less than the shift
    result += offset # Add an additional offset for x greater than the shift
    return result

# likelihood function of rayleigh distribution
def likelihood_rayleigh(params, data):
    shift, sigma, amplitude, offset = params
    res=rayleigh_distribution(data, shift, sigma, amplitude, offset)
    #print(params)
    res[res<=0] = 1e-6
    try:
        log_likelihood = np.sum(np.log(res))
    except RuntimeWarning:
        log_likelihood = -np.inf
    return -log_likelihood

class hist_fit:

    def __init__(self, times, **kwargs):
        self.hist, self.hist_edges = np.histogram(times, **kwargs)
        bin_width = self.hist_edges[1] - self.hist_edges[0]
        logging.info("Set bin width for time fit to: {:.2f} ms".format(bin_width))
        self.bincentres = (self.hist_edges[1:]-self.hist_edges[:-1])/2.0 + self.hist_edges[:-1]
        self.fit_info = ''
    
    def __call__(self, func, **kwargs):
        try:
            popt, pcov = curve_fit(func, self.bincentres, self.hist, nan_policy='omit', **kwargs)
            perr = np.sqrt(np.diag(pcov))
        except RuntimeError as warning:
            self.fit_info += 'RuntimeError: ' + str(warning)
            popt = np.array([np.nan])
            perr = np.array([np.nan])
        except ValueError as warning:
            self.fit_info += 'ValueError: ' + str(warning)
            popt = np.array([np.nan])
            perr = np.array([np.nan])
        except OptimizeWarning as warning:
            self.fit_info += 'OptimizeWarning: ' + str(warning)
            popt = np.array([np.nan])
            perr = np.array([np.nan])
        return popt, perr

# rayleigh fit
class rayleigh(hist_fit):

    def __init__(self, times, **kwargs):
        super().__init__(times, **kwargs)
        self.bounds = self.bounds()
        self.p0 = self.p0(times)

    def bounds(self):
        bounds = ([-np.inf, np.finfo(np.float64).tiny, 0.5*np.sqrt(np.finfo(np.float64).tiny), np.finfo(np.float64).tiny], [np.inf, np.inf, np.inf, np.inf])
        return bounds
    
    def p0(self, times):
        shift = np.percentile(times, 5)
        sigma = np.max([np.mean(times) - shift, 0])
        amplitude = np.sqrt(np.exp(1))*np.amax(self.hist)*sigma
        p0 = [shift, sigma, amplitude, np.finfo(np.float64).tiny]
        return p0
    
    def func(self, t, shift, sigma, amplitude, offset):
        result = amplitude * (t - shift)/(sigma**2) * np.exp(-(t - shift)**2 / (2 * sigma**2))
        result[t < shift] = 0  # Set the function to zero for x less than the shift
        result += offset # Add an additional offset for x greater than the shift
        return result
    
    def __call__(self, times, localizations, **kwargs):
        opt, err = super().__call__(self.func, bounds=self.bounds, p0=self.p0, **kwargs)
        if self.fit_info != '':
            if pd.notna(localizations['x']).any():
                logging.info("Rayleigh fit failed, using mean time as temporal localization estimate.")
                t = np.mean(times)
            else:
                t = np.nan
        else: 
            times_std = np.std(times)
            if err[0] > times_std:
                if pd.notna(localizations['x']).any():
                    logging.info("Fitting uncertainties exceed the tolerance. Using mean time instead of {:.2f} ms".format(opt[0]))
                    t = np.mean(times)
                else:
                    t = np.nan
            else:
                t = opt[0]
        return opt, t, self.fit_info

#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------
def OneDProjection(findingResult, fittingResult, previewEvents, figure, settings,**kwargs):
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    if str(kwargs['t_bin_width']) == 'auto':
        bins = 'auto'
        t_bin_width = None
    if str(kwargs['t_bin_width']) != 'auto':
        t_bin_width = float(kwargs['t_bin_width'])
    show_first = utilsHelper.strtobool(kwargs['show_first'])
    weigh_first = utilsHelper.strtobool(kwargs['weigh_first'])

    first_events = pd.DataFrame()
    if show_first==True:
        first_events = eventDistributions.FirstTimestamp(findingResult).get_smallest_t(findingResult)

    ax = figure.add_subplot(111)
    figure.tight_layout()
    figure.subplots_adjust(top=0.95,bottom=0.190,left=0.080)
    # print('Min, mean, 5th percentile, max:\n')
    # print(np.min(findingResult['t'].diff())*1e-3, np.mean(findingResult['t'].diff())*1e-3, np.percentile(findingResult['t'].diff(), 10)*1e-3, np.max(findingResult['t'].diff())*1e-3)
    # hist_t = eventDistributions.Hist1d_t(findingResult)
    # hist_t.set_t_bin_width(t_bin_width, findingResult)
    if t_bin_width is not None:
        hist_t = eventDistributions.Hist1d_t(findingResult, t_bin_width=t_bin_width)
        hist_t_dist, hist_t_edges = hist_t.dist1D, hist_t.t_edges
    else: 
        hist_t_dist, hist_t_edges = np.histogram(findingResult['t']*1e-3, bins=bins)
        t_bin_width = hist_t_edges[1]-hist_t_edges[0]
    
    t_all_events = hist_t_edges[0]
    t_first_events = hist_t_edges[0]

    bincentres = (hist_t_edges[1:]-hist_t_edges[:-1])/2.0 + hist_t_edges[:-1]
    
    # try a Rayleigh fit
    rayleigh_fit_all = rayleigh(findingResult['t']*1e-3, bins=hist_t_edges)
    all_events_fit = rayleigh_fit_all(findingResult['t']*1e-3, fittingResult)
    if not np.isnan(all_events_fit[0]).any():
        t = np.linspace(np.min([np.min(findingResult['t']*1e-3), all_events_fit[1]]), np.max(findingResult['t']*1e-3), 100)
        ax.plot(t, rayleigh_distribution(t, *all_events_fit[0]), label='Rayleigh fit', color='black')
        ax.axvline(x=all_events_fit[1], color='red')
        t_all_events = all_events_fit[1]

    # try a MLE Rayleigh fit
    # np.save("/home/laura/PhD/Event_Based_Sensor_Project/GUI_tests/MLE_fit/data.npy", findingResult['t'])
    # method = 'l-bfgs-b'
    # offset_bound = 0.1
    # mle_initial = [p0[0], p0[1], 1., 0.05/(p0[1]*np.sqrt(np.exp(1)))]
    # print(p0, f"max offset is: {offset_bound/(p0[1]*np.sqrt(np.exp(1)))}")
    # bounds = [(-np.inf, np.inf), (0.5*np.sqrt(np.finfo(np.float64).tiny), np.inf), (np.finfo(np.float64).tiny, 1.0), (np.finfo(np.float64).tiny, offset_bound/(p0[1]*np.sqrt(np.exp(1))))]
    # mle_rayleigh = minimize(likelihood_rayleigh, mle_initial, args=(findingResult['t']*1e-3,), bounds=bounds, method=method)
    # print(f"MLE results: {mle_rayleigh.x}")
    # ax.plot(t, len(findingResult['t'])*t_bin_width*rayleigh_distribution(t,*mle_rayleigh.x), label='MLE Rayleigh fit', color='red')
  
    # Plot the 2D histograms
    if not len(findingResult[findingResult['p'] == 1]) == 0:
        hist_t_pos = np.histogram(findingResult[findingResult['p'] == 1]['t']*1e-3, bins=hist_t_edges)[0]
        ax.bar(hist_t_edges[:-1], hist_t_pos, width=t_bin_width,  label='Positive events', color='C0', alpha=0.5, align='edge')
        cumsum = np.cumsum(findingResult[findingResult['p'] == 1]['t']*1e-3)
        normed_cumsum = cumsum/cumsum.iloc[-1]
        # plot derivative of cumsum
        derivative = normed_cumsum.diff()/(findingResult[findingResult['p'] == 1]['t'].diff()*1e-3)
        derivative = derivative/np.max(derivative)*np.max(hist_t_pos)
        ax.plot(findingResult[findingResult['p'] == 1]['t']*1e-3, derivative, color='C0', linestyle='dashed', label='PDF positive events')
        ax.plot(findingResult[findingResult['p'] == 1]['t']*1e-3, normed_cumsum*np.max(hist_t_pos), color='C0', label='Cumulative positive events')
    if not len(findingResult[findingResult['p'] == 0]) == 0:
        hist_t_neg = np.histogram(findingResult[findingResult['p'] == 0]['t']*1e-3, bins=hist_t_edges)[0]
        cumsum = np.cumsum(findingResult[findingResult['p'] == 0]['t']*1e-3)
        normed_cumsum = cumsum/cumsum.iloc[-1]
        ax.plot(findingResult[findingResult['p'] == 0]['t']*1e-3, normed_cumsum*np.max(hist_t_neg), color='C1', label='Cumulative negative events')
        # plot derivative of cumsum
        derivative = normed_cumsum.diff()/(findingResult[findingResult['p'] == 0]['t'].diff()*1e-3)
        derivative = derivative/np.max(derivative)*np.max(hist_t_neg)
        ax.plot(findingResult[findingResult['p'] == 0]['t']*1e-3, derivative, color='C1', linestyle='dashed', label='PDF negative events')
        ax.bar(hist_t_edges[:-1], hist_t_neg, width=t_bin_width,  label='Negative events', color='C1', alpha=0.5, align='edge')
    if not len(first_events) == 0:
        if weigh_first:
            weights = first_events['weight']
            label_data = 'First events (weighted)'
            label_fit = 'Rayleigh fit first events (weighted)'
        else: 
            weights = None
            label_data = 'First events'
            label_fit = 'Rayleigh fit first events'
        hist_t_first = np.histogram(first_events['t']*1e-3, weights=weights, bins=hist_t_edges)[0]
        first_events = first_events.sort_values(by=['t'])
        cumsum = np.cumsum(first_events['t']*1e-3)
        normed_cumsum = cumsum/cumsum.iloc[-1]
        ax.plot(first_events['t']*1e-3, normed_cumsum*np.max(hist_t_first), color='darkred', label=label_data)
        rayleigh_fit_first = rayleigh(first_events['t']*1e-3, weights=weights, bins=hist_t_edges)
        first_events_fit = rayleigh_fit_first(first_events['t']*1e-3, fittingResult)
        if not np.isnan(first_events_fit[0]).any():
            t = np.linspace(np.min([np.min(findingResult['t']*1e-3), first_events_fit[1]]), np.max(findingResult['t']*1e-3), 100)
            ax.plot(t, rayleigh_distribution(t, *first_events_fit[0]), label=label_fit, color='darkgreen')
            ax.axvline(x=first_events_fit[1], color='darkred', linestyle='dashed')
            t_first_events = first_events_fit[1]

        # Add a zero bin to the beginning and end of hist_t_first for plot with step function
        hist_t_first = np.insert(hist_t_first, 0, 0)  # Add a zero bin at the beginning
        hist_t_first = np.append(hist_t_first, 0)  # Add a zero bin at the end
        t_edges = np.insert(hist_t_edges, 0, hist_t_edges[0]-(hist_t_edges[1]-hist_t_edges[0]))  # Add a zero bin at the beginning
        t_edges = np.append(t_edges, hist_t_edges[-1]+(hist_t_edges[1]-hist_t_edges[0]))  # Add a zero bin at the end

        bincentres = (t_edges[1:]-t_edges[:-1])/2.+t_edges[:-1] 
        ax.step(bincentres,hist_t_first,where='mid',color='C2', label=label_data,linewidth=1.5)

    # plot cumulative distribution
    # ax.plot(hist_t_edges[:-1], np.cumsum(hist_t_neg), color='C1', label='Cumulative negative events')

    # Add and set labels
    ax.set_xlim(np.nanmin([hist_t_edges[0], t_first_events-1, t_all_events-1]), hist_t_edges[-1])
    ax.set_xlabel('t [ms]')
    ax.set_ylabel('number of events')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    figure.tight_layout()
    
    # required output none
    return 1
